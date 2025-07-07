from __future__ import annotations

__all__ = ['lombscargle', 'FFTW_MEASURE', 'FFTW_ESTIMATE']

from timeit import default_timer as timer

import finufft
import numpy as np

import threadpoolctl

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
    nthreads_finufft=None,
    nthreads_blas=None,
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

    Threading Model
    ---------------
    This function uses a three-level threading approach:
    1. nthreads: Batch-level parallelization (cannot exceed Nbatch)
    2. nthreads_finufft: FINUFFT computation threads (can exceed Nbatch for large frequency grids)  
    3. nthreads_blas: BLAS/LAPACK threads for linear algebra, applied only during matrix operations
       (automatically reduced for small matrices)

    BLAS threading is controlled via threadpoolctl context managers applied only around 
    the specific np.dot/np.linalg.solve operations, ensuring FINUFFT can use its own 
    threading settings independently.

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
    nthreads_batch : int, optional
        The number of threads to use for batch-level parallelization in pre/post-processing. 
        Default is min(Nbatch, nthreads_finufft). Setting nthreads > Nbatch wastes resources
        since you cannot parallelize more batches than available.
        If None, it will be set to 1 for. Only used if `_no_cpp_helpers` is False.
    nthreads_finufft : int, optional
        The number of threads to use for FINUFFT computation. Default is computed from problem 
        size using: max(1, Nbatch//4) * max(1, Nf//32768), capped by system limits. FINUFFT 
        can benefit from multiple threads even on single batches for large frequency grids.
    nthreads_blas : int, optional
        The number of threads to use for BLAS/LAPACK linear algebra operations. Default is 
        nthreads_finufft // nthreads. Use -1 to disable threadpoolctl and use system default 
        BLAS threading. For small matrices (nterms ≤ 2), automatically uses 1 thread for efficiency.
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
        fftw=FFTW_ESTIMATE, # FFTW_ESTIMATE
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

    # Multithreading heuristics
    # nthreads * nthreads_blas = omp_max_threads()
    if nthreads_finufft is None:
        # Compute optimal FINUFFT threads based on problem size
        nthreads_finufft = max(1, Nbatch // 4) * max(1, Nf // (1 << 15))
        # Allocate threads more than system limits may bring performance down
        nthreads_finufft = min(nthreads_finufft, get_finufft_max_threads())
    
    if not _no_cpp_helpers:
        if nthreads is None:
            #TODO: Remove it, and now we didn't pass it to the function
            # Scale batch-level threads based on frequency grid size and batch count
            # For small Nf, reduce threading even with multiple batches
            # nf_factor = max(1, Nf // (1 << 15))  # Reduce threads for small frequency grids
            # nthreads = min(Nbatch, nthreads_finufft, nf_factor)
            nthreads = get_finufft_max_threads()

        if nthreads_blas is None:
            # Default to 1 thread for BLAS, since the matrices size are n * n where n is nterms + 1. 
            # This size is very small, so using multiple threads is not efficient. 
            nthreads_blas = 1
        
        # If nthreads_blas is -1, we disable threadpoolctl and use system defaults. If not, cap 
        # the threads for BLAS, which is used for linear algebra operations in C++.
        if nthreads_blas != -1:
            controller = threadpoolctl.threadpool_limits(limits=nthreads_blas, user_api='blas')
            controller.__enter__()
    
    if verbose:
        if not _no_cpp_helpers:
            if nthreads_blas == -1:
                blas_msg = "default BLAS threading"
            else:
                blas_msg = f"{nthreads_blas} BLAS {'thread' if nthreads_blas == 1 else 'threads'}"
            nthreads_msg = f"{nthreads} {'thread' if nthreads == 1 else 'threads'} for batch"
        else:
            blas_msg = "default BLAS threading"
            nthreads_msg = f"single thread for batch"
        print(
            f'[nifty-ls finufft] Using {nthreads_msg}, {nthreads_finufft} for FINUFFT, {blas_msg}'
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
    Sw  = np.zeros((Nbatch, nSW, Nf), dtype=dtype) # Shape(Nbatch, nSW, Nf) and Initlize to 0
    Cw  = np.zeros((Nbatch, nSW, Nf), dtype=dtype)
    Syw = np.zeros((Nbatch, nSY, Nf), dtype=dtype)
    Cyw = np.zeros((Nbatch, nSY, Nf), dtype=dtype)

    t_helpers = -timer()
    if not _no_cpp_helpers: # Use C++ helpers
        t1 = np.empty_like(t)
        w2s = np.empty((Nbatch, 1), dtype=dtype)  # Changed to 2D to match keepdims=True
        
        norm = np.empty((Nbatch, 1), dtype=dtype)
        yws = np.empty((Nbatch, 1), dtype=dtype)  # Changed to 2D to match keepdims=True
        
        chi2_helpers.process_chi2_inputs(
            t1,  # output
            yw,  # output
            w,  # output, refer to yw_w[Nbatch:]
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

        # w_base equivalent to w2 in fastfinufft
        w_base = (dy_broadcasted ** -2.0).astype(dtype)
        w2s = np.sum(w_base.real, axis=-1, keepdims=True)  # sum over N, shape: (Nbatch, 1)
       
        if center_data or fit_mean:
            y = y - ((w_base * y).sum(axis=-1, keepdims=True) / w2s)
            yws = np.zeros((Nbatch, 1), dtype=dtype)  # shape: (Nbatch, 1)
        else:
            yws = (y * w_base).sum(axis=-1, keepdims=True)  # shape: (Nbatch, 1) to match w2s

        norm = (w_base * y**2).sum(axis=-1, keepdims=True)

        yw[:] = y * w_base
        w[:]  = w_base
        
        # SCw = [(np.zeros(Nf), ws * np.ones(Nf))]
        # SCyw = [(np.zeros(Nf), yws * np.ones(Nf))]
        Sw[:,0,:] = 0
        Cw[:,0,:] = w2s  # broadcasting w2s from (Nbatch, 1) to (Nbatch, Nf)
        Syw[:,0,:] = 0
        Cyw[:,0,:] = yws  # broadcasting yws from (Nbatch, 1) to (Nbatch, Nf)

    # This function applies a time shift to the reference time t1 and computes
    # the corresponding phase shifts. It then creates a new batch of weights
    # by multiplying the input weights with the phase shifts.
    def compute_t(time_shift, yw_w):
        tn = time_shift * t1
        tn = tn.astype(dtype, copy=False)
        phase_shiftn = np.exp(1j * ((Nf // 2) + fmin / df) * tn)  # shape = (N,)

        # Build a brand-new "batch" of phased weights for this i:
        yw_w_s = (yw_w * phase_shiftn).astype(cdtype) # broadcasting explicit
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
        nthreads=nthreads_finufft,
        **finufft_kwargs,
    )
    
    # Pre-allocate arrays for compute_t to avoid repeated allocations
    tj = np.empty_like(t1)
    yw_w_j = np.empty_like(yw_w)
    
    for j in range(1, nterms + 1):
        if not _no_cpp_helpers: # Using C++ helpers
            chi2_helpers.compute_t(t1, yw_w, j, fmin, df, Nf, tj, yw_w_j, nthreads_helpers)
        else:
            tj, yw_w_j = compute_t(j, yw_w)
        plan_yw.setpts(tj)
        f1_fw = plan_yw.execute(yw_w_j)
        # TODO: use out parameter in finufft.Plan.execute() to 
        # write directly to Sw/Cw/Syw/Cyw arrays and avoid copying
        Sw[:,j,:] = f1_fw[Nbatch:].imag# yw
        Cw[:,j,:] = f1_fw[Nbatch:].real
        Syw[:,j,:] = f1_fw[:Nbatch].imag
        Cyw[:,j,:] = f1_fw[:Nbatch].real

    # Since in fastchi2, the freq_factor of w includes terms 
    # from 1 to 2*nterms, we need one more loop to handle 
    # the result of the transform for indices nterms + 1 to 2*nterms(inclusive).
    plan_w = finufft.Plan(
        nufft_type=1,
        n_modes_or_dim=(Nf,),
        n_trans=Nbatch,
        dtype=cdtype,
        nthreads=nthreads_finufft,
        **finufft_kwargs,
    )
    
    # Pre-allocate arrays for the second loop
    ti = np.empty_like(t1)
    yw_w_i = np.empty_like(yw_w)
    
    for i in range(nterms + 1, 2 * nterms + 1):
        if not _no_cpp_helpers:
            chi2_helpers.compute_t(t1, yw_w, i, fmin, df, Nf, ti, yw_w_i, nthreads_helpers)
        else:
            ti, yw_w_i = compute_t(i, yw_w)
        plan_w.setpts(ti)     
        f2_all = plan_w.execute(yw_w_i[Nbatch:]) # w only
        Sw[:,i,:] = f2_all.imag
        Cw[:,i,:] = f2_all.real
    t_finufft += timer()

    t_helpers -= timer()
    # Build the "order" list once (same for all batches):
    order = [("C", 0)] if fit_mean else []
    order.extend(chain(*([("S", i), ("C", i)] for i in range(1, nterms + 1))))
    if not _no_cpp_helpers: # Using C++ helpers
        norm_enum = dict(
            standard=cpu_helpers.NormKind.Standard,
            model=cpu_helpers.NormKind.Model,
            log=cpu_helpers.NormKind.Log,
            psd=cpu_helpers.NormKind.PSD,
        )[normalization.lower()]

        power = np.zeros((Nbatch, Nf), dtype=dtype)
        
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
        
        # Returns a dictionary of lambda functions that provide access to precomputed 
        # sine and cosine basis terms and their weighted versions.
        def create_funcs(Sw_b, Cw_b, Syw_b, Cyw_b):
            return {
            'S': lambda m, i: Syw_b[m, i],
            'C': lambda m, i: Cyw_b[m, i],
            'SS': lambda m, n, i: 0.5 * (Cw_b[abs(m - n), i] - Cw_b[m + n, i]),
            'CC': lambda m, n, i: 0.5 * (Cw_b[abs(m - n), i] + Cw_b[m + n, i]),
            'SC': lambda m, n, i: 0.5 * (np.sign(m - n) * Sw_b[abs(m - n), i] + Sw_b[m + n, i]),
            'CS': lambda m, n, i: 0.5 * (np.sign(n - m) * Sw_b[abs(n - m), i] + Sw_b[n + m, i]),
            }
        
        def compute_power_single(funcs, order, i, norm_value, normalization):
            # Build XTX and XTy for the current frequency i
            XTX = np.array(
            [[ funcs[A[0] + B[0]](A[1], B[1], i) for A in order] for B in order]
            )
            XTy = np.array([ funcs[A[0]](A[1], i) for A in order])
            
            raw_power = np.dot(XTy.T, np.linalg.solve(XTX, XTy))
            
            # Apply normalization per batch
            if normalization == 'standard':
                return raw_power / norm_value
            elif normalization == 'model':
                return raw_power / (norm_value - raw_power)
            elif normalization == 'log':
                return -np.log(1 - raw_power / norm_value)
            elif normalization == 'psd':
                return raw_power * 0.5
            else:
                raise ValueError(f'Unknown normalization: {normalization}')

        # Parallel batch processing using ThreadPoolExecutor
        def process_batch(batch_idx):
            
            # # Limit BLAS threads for this batch
            # with threadpoolctl.threadpool_limits(limits=nthreads_blas, user_api='blas'):
            # For this batch, directly use the array views for better performance
            # These slices give us direct views into the arrays without creating copies
            Sw_b = Sw[batch_idx, :, :]  # shape: (nSW, Nf)
            Cw_b = Cw[batch_idx, :, :]
            Syw_b = Syw[batch_idx, :, :]  # shape: (nSY, Nf)
            Cyw_b = Cyw[batch_idx, :, :]
            # Create functions for this batch
            batch_funcs = create_funcs(Sw_b, Cw_b, Syw_b, Cyw_b)

            # Get the normalization value for this batch
            norm_value = norm[batch_idx, 0]
            
            # Compute power and normalization for all frequencies for this batch:
            batch_power = np.zeros(Nf, dtype=dtype)
            for i in range(Nf):
                batch_power[i] = compute_power_single(batch_funcs, order, i, norm_value, normalization)
            return batch_idx, batch_power
        
        for batch_idx in range(Nbatch):
            # Collect results back into the power array
            _, batch_power = process_batch(batch_idx)
            power[batch_idx] = batch_power

    # If only one batch and squeeze requested, drop batch axis
    if squeeze_output:
        power = power.squeeze()
    
    t_helpers += timer()

    if verbose:
        print(
            f'[nifty-ls finufft] FINUFFT took {t_finufft:.4g} sec, pre-/post-processing took {t_helpers:.4g} sec'
        )

    # # Clean up threadpoolctl context manager if used

    if not _no_cpp_helpers and nthreads_blas != -1:
        controller.__exit__(None, None, None)

    return power


def get_finufft_max_threads():
    try:
        return finufft._finufft.lib.omp_get_max_threads()
    except AttributeError:
        return 1
