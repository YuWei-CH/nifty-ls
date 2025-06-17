from __future__ import annotations

__all__ = ['lombscargle', 'FFTW_MEASURE', 'FFTW_ESTIMATE']

from timeit import default_timer as timer

import finufft
import numpy as np

from . import cpu_helpers, chi2_helpers

from itertools import chain

FFTW_MEASURE = 0
FFTW_ESTIMATE = 64


def lombscargle(
    t,
    y,
    fmin,
    df,
    Nf,
    dy=None,
    nthreads=None,
    center_data=True,
    fit_mean=True,
    normalization='standard',
    _no_cpp_helpers=False,
    verbose=False,
    finufft_kwargs=None,
    nterms=1,
):
    """
    Compute the Lomb-Scargle periodogram using the FINUFFT backend.

    Performance Tuning
    ------------------
    The performance of this backend depends almost entirely on the performance of finufft,
    which can vary significantly depending on tuning parameters like the number of threads.
    Order-of-magnitude speedups or slowdowns are possible. Unfortunately, it's very difficult
    to predict what will work best for a given problem on a given platform, so some experimentation
    may be necessary. For nthreads, start with 1 and increase until performance stops improving.
    Some other tuning parameters are listed under "finufft_kwargs" below.

    See the finufft documentation for the full list of tuning parameters:
    https://finufft.readthedocs.io/en/latest/opts.html


    Parameters
    ----------
    t : array-like
        The time values, shape (N_t,)
    y : array-like
        The data values, shape (N_t,) or (N_y, N_t)
    fmin : float
        The minimum frequency of the periodogram.
    df : float
        The frequency bin width.
    Nf : int
        The number of frequency bins.
    dy : array-like, optional
        The uncertainties of the data values, broadcastable to `y`
    nthreads : int, optional
        The number of threads to use. The default behavior is to use (N_t / 4) * (Nf / 2^15) threads,
        capped to the maximum number of OpenMP threads. This is a heuristic that may not work well in all cases.
    center_data : bool, optional
        Whether to center the data before computing the periodogram. Default is True.
    fit_mean : bool, optional
        Whether to fit a mean value to the data before computing the periodogram. Default is True.
    normalization : str, optional
        The normalization method to use. One of ['standard', 'model', 'log', 'psd']. Default is 'standard'.
    _no_cpp_helpers : bool, optional
        Whether to use the pure Python implementation of the finufft pre- and post-processing.
        Default is False.
    verbose : bool, optional
        Whether to print additional information about the finufft computation.
    finufft_kwargs : dict, optional
        Additional keyword arguments to pass to the `finufft.Plan()` constructor.
        Particular finufft parameters of interest may be:
        - `eps`: the requested precision [1e-9 for double precision and 1e-5 for single precision]
        - `upsampfac`: the upsampling factor [1.25]
        - `fftw`: the FFTW planner flags [FFTW_ESTIMATE]
    nterms : int, optional
        Number of Fourier terms in the fit
    """
    
    if nterms == 0 and not fit_mean:
        raise ValueError("Cannot have nterms = 0 without fitting bias")

    default_finufft_kwargs = dict(
        eps='default',
        upsampfac=1.25, # Default upsampling factor
        fftw=FFTW_ESTIMATE,
        debug=int(verbose),
    )

    finufft_kwargs = {**default_finufft_kwargs, **(finufft_kwargs or {})}

    dtype = t.dtype

    if finufft_kwargs['eps'] == 'default':
        if dtype == np.float32:
            finufft_kwargs['eps'] = 1e-5
        else:
            finufft_kwargs['eps'] = 1e-9
    if 'backend' in finufft_kwargs:
        raise ValueError('backend should not be passed as a keyword argument')

    cdtype = np.complex128 if dtype == np.float64 else np.complex64

    if dy is None:
        dy = dtype.type(1.0)
    
    # treat 1D arrays as a batch of size 1
    squeeze_output = (y.ndim == 1)
    y = np.atleast_2d(y)
    dy = np.atleast_2d(dy)

    # If fit_mean, we need to transform (t,yw) and (t,w),
    # so we stack yw and w into a single array to allow a batched transform.
    # Regardless, we need to do a separate (t2,w2) transform.
    Nbatch, N = y.shape

    # Broadcast dy to match the shape of t and y
    if dy.ndim == 1:
        dy_broadcasted = np.broadcast_to(dy, (Nbatch, N))
    elif dy.shape[1] == 1:
        dy_broadcasted = np.broadcast_to(dy, (Nbatch, N))
    else:
        dy_broadcasted = dy

    if nthreads is None:
        # This heuristic feels fragile, it would be much better if finufft could do this upstream!
        nthreads = max(1, Nbatch // 4) * max(1, Nf // (1 << 15))
        # Using get_finufft_max_threads() is safe because finufft never calls omp_set_num_threads()
        nthreads = min(nthreads, get_finufft_max_threads())
        # finufft (and cpu_helpers) will warn if the user exceeds omp_get_max_threads()

    if verbose:
        print(
            f'[nifty-ls finufft] Using {nthreads} {"thread" if nthreads == 1 else "threads"}'
        )

    # Could probably be more than finufft nthreads in many cases,
    # but it's conceptually cleaner to keep them the same, and it
    # will almost never matter in practice.
    # Technically, this is also suboptimal in the rare case of a finufft
    # library without OpenMP and a nifty-ls with OpenMP
    nthreads_helpers = nthreads

    # Force fit them into 2 arrays to use the batched finufft transform for yw and w
    yw_w_shape = (2 * Nbatch, N)
    yw_w = np.empty(yw_w_shape, dtype=cdtype)

    yw = yw_w[:Nbatch]
    w = yw_w[Nbatch:]
    
    # Pre‐allocate memory for trig matrix in dtype, since it not store complex numbers
    # 2*nterms + 1 terms for w, nterms + 1 terms for yw. Fetching the Plan
    nSW = 2*nterms + 1
    nSY = nterms + 1
    Sw  = np.empty((nSW, Nbatch, Nf), dtype=dtype) # Shape(nSW, Nbatch, Nf)
    Cw  = np.empty((nSW, Nbatch, Nf), dtype=dtype)
    Syw = np.empty((nSY, Nbatch, Nf), dtype=dtype)
    Cyw = np.empty((nSY, Nbatch, Nf), dtype=dtype)

    t_helpers = -timer()
    if not _no_cpp_helpers: # Use C++ helpers
        t1 = np.empty_like(t)
        w2s = np.empty(Nbatch, dtype=dtype)
        
        norm = np.empty((Nbatch, 1), dtype=dtype)
        yws = np.empty(Nbatch, dtype=dtype)
        
        chi2_helpers.process_chi2_inputs(
            t1,  # output
            yw,  # output
            w,  # output
            w2s,  # output
            norm,  # output
            yws, # output
            Sw, Cw, Syw, Cyw, # output, use for initial trig matrix
            t,  # input
            y,  # input
            dy_broadcasted,  # input
            df,
            center_data,
            fit_mean,
            nthreads_helpers,
        )
        
    else:
        t1 = 2 * np.pi * df * t
        t1 = t1.astype(dtype, copy=False)
        y = y.astype(dtype, copy=False)
        dy = dy.astype(dtype, copy=False)

        # w2_base equivalent to w2 in fastfinufft
        w2_base = (dy_broadcasted ** -2.0).astype(dtype)
        w2s = np.sum(w2_base.real, axis=-1, keepdims=True)  # sum over N
        w2 = w2_base.astype(cdtype, copy=True) # convert to complex dtype
       
        if center_data or fit_mean:
            y = y - ((w2.real * y).sum(axis=-1, keepdims=True) / w2s)
        # np.dot(yw, yw)
        norm = (w2.real * y**2).sum(axis=-1, keepdims=True)

        yw[:] = y * w2.real
        w[:]  = w2
        
        yws = (y * w2.real).sum(axis=-1)
        
        # SCw = [(np.zeros(Nf), ws * np.ones(Nf))]
        # SCyw = [(np.zeros(Nf), yws * np.ones(Nf))]
        Sw[0,:,:] = 0
        Cw[0,:,:] = w2s  # broadcasting your w2s over Nf
        Syw[0,:,:] = 0
        Cyw[0,:,:] = yws[:,None]
        
    # This function applies a time shift to the reference time t1 and computes
    # the corresponding phase shifts. It then creates a new batch of weights
    # by multiplying the input weights with the phase shifts.
    def compute_t(time_shift, yw_w):
        tn = time_shift * t1
        tn = tn.astype(dtype, copy=False)
        phase_shiftn = np.exp(1j * ((Nf // 2) + fmin / df) * tn)  # shape = (N,)

        # Build a brand-new "batch" of phased weights for this i:
        yw_w_s = (yw_w * phase_shiftn[None, :]).astype(cdtype)
        return tn, yw_w_s
    t_helpers += timer()

    t_finufft = -timer()
    
    # Loop over harmonics from i = 0 to nterms (inclusive)
    # For each frequency term pass yw_w as input:
    #   - Weighted data: y_i × w  at time points t
    #   - Pure weights: w  at time points t
    # Both share the same time coordinates (t), so we can batch them together
    # Use a single NUFFT plan to transform both y_i × w and w simultaneously and efficiently.
    plan_yw = finufft.Plan(
        nufft_type=1,
        n_modes_or_dim=(Nf,),
        n_trans= 2 * Nbatch, # paired processing of y * w and w
        dtype=cdtype,
        nthreads=nthreads,
        **finufft_kwargs,
    )
    for j in range(1, nterms + 1):
        if not _no_cpp_helpers:
            tj = np.empty_like(t1)
            yw_w_j = np.empty_like(yw_w)
            chi2_helpers.compute_t(t1, yw_w, j, fmin, df, Nf, tj, yw_w_j)
        else:
            tj, yw_w_j = compute_t(j, yw_w)
        plan_yw.setpts(tj)
        f1_fw = plan_yw.execute(yw_w_j)
        # TODO: use out parameter in finufft.Plan.execute() to 
        # write directly to Sw/Cw/Syw/Cyw arrays and avoid copying
        Sw[j,:,:] = f1_fw[Nbatch:].imag# yw
        Cw[j,:,:] = f1_fw[Nbatch:].real
        Syw[j,:,:] = f1_fw[:Nbatch].imag
        Cyw[j,:,:] = f1_fw[:Nbatch].real
    
    # Since in fastchi2, the freq_factor of w includes terms 
    # from 1 to 2*nterms, we need one more loop to handle 
    # the result of the transform for indices nterms + 1 to 2*nterms(inclusive).
    plan_w = finufft.Plan(
        nufft_type=1,
        n_modes_or_dim=(Nf,),
        n_trans=Nbatch,
        dtype=cdtype,
        nthreads=nthreads,
        **finufft_kwargs,
    )
    for i in range(nterms + 1, 2 * nterms + 1):
        if not _no_cpp_helpers:
            ti = np.empty_like(t1)
            yw_w_i = np.empty_like(yw_w)
            chi2_helpers.compute_t(t1, yw_w, i, fmin, df, Nf, ti, yw_w_i)
        else:
            ti, yw_w_i = compute_t(i, yw_w)
        plan_w.setpts(ti)     
        f2_all = plan_w.execute(yw_w_i[Nbatch:]) # w only
        Sw[i,:,:] = f2_all.imag.copy()
        Cw[i,:,:] = f2_all.real.copy()

    t_finufft += timer()

    t_helpers -= timer()
    if not _no_cpp_helpers: # Using C++ helpers
        norm_enum = dict(
            standard=cpu_helpers.NormKind.Standard,
            model=cpu_helpers.NormKind.Model,
            log=cpu_helpers.NormKind.Log,
            psd=cpu_helpers.NormKind.PSD,
        )[normalization.lower()]

        power = np.zeros((Nbatch, Nf), dtype=dtype)
        # Build the "order" list once (same for all batches):
        order = [("C", 0)] if fit_mean else []
        order.extend(chain(*([("S", i), ("C", i)] for i in range(1, nterms + 1))))
        
        order_types = [chi2_helpers.TermType.Cosine if t == "C" else chi2_helpers.TermType.Sine for t, _ in order]
        order_indices = [item[1] for item in order]
        
        chi2_helpers.process_chi2_outputs(
            power,
            Sw, Cw, Syw, Cyw, # 3D arrays
            norm,
            order_types,
            order_indices,
            norm_enum,
            nthreads_helpers,
        )
        
    else:
        # Build-up matrices at each frequency
        power = np.zeros((Nbatch, Nf), dtype=dtype)
        
        # Prepare the "order" list as a template (reuse for all batches)
        order = [("C", 0)] if fit_mean else []
        order.extend(chain(*([("S", i), ("C", i)] for i in range(1, nterms + 1))))
        
        # Returns a dictionary of lambda functions that provide access to precomputed 
        # sine and cosine basis terms and their weighted versions.
        def create_funcs(Sw_b, Cw_b, Syw_b, Cyw_b):
            return {
            'S': lambda m, i: Syw_b[m][i],
            'C': lambda m, i: Cyw_b[m][i],
            'SS': lambda m, n, i: 0.5 * (Cw_b[abs(m - n)][i] - Cw_b[m + n][i]),
            'CC': lambda m, n, i: 0.5 * (Cw_b[abs(m - n)][i] + Cw_b[m + n][i]),
            'SC': lambda m, n, i: 0.5 * (np.sign(m - n) * Sw_b[abs(m - n)][i] + Sw_b[m + n][i]),
            'CS': lambda m, n, i: 0.5 * (np.sign(n - m) * Sw_b[abs(n - m)][i] + Sw_b[n + m][i]),
            }
        
        def compute_power_single(funcs, order, i):
            # Build XTX and XTy for the current frequency i
            XTX = np.array(
            [[ funcs[A[0] + B[0]](A[1], B[1], i) for A in order] for B in order]
            )
            XTy = np.array([ funcs[A[0]](A[1], i) for A in order])
            return np.dot(XTy.T, np.linalg.solve(XTX, XTy))

        # Batch processing
        for batch_idx in range(Nbatch):
            # For this batch, pull out the precomputed arrays:
            Sw_b  = [ Sw[m][batch_idx]  for m in range(2 * nterms + 1) ]  # list of (Nf,) arrays
            Cw_b  = [ Cw[m][batch_idx]  for m in range(2 * nterms + 1) ]
            Syw_b = [ Syw[m][batch_idx] for m in range(nterms + 1) ]
            Cyw_b = [ Cyw[m][batch_idx] for m in range(nterms + 1) ]

            # Create functions outside the inner loop
            batch_funcs = create_funcs(Sw_b, Cw_b, Syw_b, Cyw_b)

            # Compute power for all frequencies for this batch:
            for i in range(Nf):
                power[batch_idx, i] = compute_power_single(batch_funcs, order, i)
            
        # Apply normalization once for all batches
        if normalization == 'standard':
            power = power / norm
        elif normalization == 'model':
            power = power / (norm - power)
        elif normalization == 'log':
            power = -np.log(1 - power / norm)
        elif normalization == 'psd':
            power = power * 0.5
        else:
            raise ValueError(f'Unknown normalization: {normalization}')

    # If only one batch and squeeze requested, drop batch axis
    if squeeze_output:
        power = power.squeeze()
    
    t_helpers += timer()

    if verbose:
        print(
            f'[nifty-ls finufft] FINUFFT took {t_finufft:.4g} sec, pre-/post-processing took {t_helpers:.4g} sec'
        )

    return power


def get_finufft_max_threads():
    try:
        return finufft._finufft.lib.omp_get_max_threads()
    except AttributeError:
        return 1
