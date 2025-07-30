from __future__ import annotations

from dataclasses import dataclass
import importlib
from typing import Optional, Literal

import numpy as np
import numpy.typing as npt

from typing import List

from . import utils
from .backends import available_backends, BACKEND_TYPE

__all__ = [
    'lombscargle',
    'NiftyResult',
    'NORMALIZATION_TYPE',
    'AVAILABLE_BACKENDS',
]


AVAILABLE_BACKENDS = available_backends()
NORMALIZATION_TYPE = Literal['standard', 'model', 'log', 'psd']


def lombscargle(
    t: npt.NDArray[np.floating],
    y: npt.NDArray[np.floating],
    dy: Optional[npt.NDArray[np.floating]] = None,
    fmin: Optional[float] = None,
    fmax: Optional[float] = None,
    Nf: Optional[int] = None,
    center_data: bool = True,
    fit_mean: bool = True,
    normalization: NORMALIZATION_TYPE = 'standard',
    assume_sorted_t: bool = True,
    samples_per_peak: int = 5,
    nyquist_factor: int = 5,
    backend: BACKEND_TYPE = 'auto',
    nterms: int = 1,
    **backend_kwargs: Optional[dict],
) -> NiftyResult:
    """
    Compute a Lomb-Scargle periodogram, or a batch of periodograms if `y` and `dy` are 2D arrays.

    This function can dispatch to multiple backends, including 'finufft' and 'cufinufft'. The latter
    uses CUDA and requires that nifty-ls was installed with the 'cuda' extra.

    The result is a `NiftyResult` dataclass containing the computed periodogram(s), frequency grid parameters,
    and other metadata. The actual frequency grid can be obtained by calling `freq()` on the result.

    The meanings of these parameters conform to the Lomb-Scargle implementation in Astropy:
    https://docs.astropy.org/en/stable/timeseries/lombscargle.html

    Parameters
    ----------
    t : array-like
        The time values, shape (N_t,)
    y : array-like
        The data values, shape (N_t,) or (N_y, N_t)
    dy : array-like, optional
        The uncertainties of the data values, broadcastable to `y`
    fmin : float, optional
        The minimum frequency of the periodogram. If not provided, it will be chosen automatically.
    fmax : float, optional
        The maximum frequency of the periodogram. If not provided, it will be chosen automatically.
    Nf : int, optional
        The number of frequency bins. If not provided, it will be chosen automatically.
    center_data : bool, optional
        Whether to center the data before computing the periodogram. Default is True.
    fit_mean : bool, optional
        Whether to fit a mean value to the data before computing the periodogram. Default is True.
    normalization : str, optional
        The normalization method to use. One of ['standard', 'model', 'log', 'psd']. Default is 'standard'.
    assume_sorted_t : bool, optional
        Whether to assume that the time values are sorted in ascending order, allowing for a performance
        optimization in determining the frequency grid.  Default is True.
    samples_per_peak : int, optional
        The number of samples per peak to use when determining the frequency grid. Default is 5.
    nyquist_factor : int, optional
        The factor by which to multiply the Nyquist frequency when determining the frequency grid. Default is 5.
    backend : str, optional
        The backend to use for the computation. Default is 'auto' which selects the best available backend.
    nterms : int, optional
        The number of terms to use in the Lomb-Scargle computation. Must be at least 1.
        If greater than 1, the 'cufinufft_chi2' or 'finufft_chi2' should be used for backend.
    backend_kwargs : dict, optional
        Additional keyword arguments to pass to the backend.

    Returns
    -------
    nifty_result : NiftyResult
        A dataclass containing the computed periodogram(s), frequency grid parameters, and other.
        The fields are 'power', 'fmin', 'df', 'Nf', and 'fmax'.
        `nifty_result.power` will be an ndarray of shape (Nf,) or (N_y, Nf) if `y` is 2D.
    """
    fmin, df, Nf = utils.validate_frequency_grid(
        fmin,
        fmax,
        Nf,
        t,
        assume_sorted_t=assume_sorted_t,
        samples_per_peak=samples_per_peak,
        nyquist_factor=nyquist_factor,
    )
    # Nterm verification
    if nterms is None:
        nterms = 1
    if nterms < 1:
        raise ValueError(f'nterms must be at least 1, got {nterms}.')

    # Backend selection
    if backend == 'auto':
        if nterms > 1:
            if 'cufinufft_chi2' in AVAILABLE_BACKENDS:
                backend = 'cufinufft_chi2'
            elif 'finufft_chi2' in AVAILABLE_BACKENDS:
                backend = 'finufft_chi2'
            else:
                raise ValueError(
                    'Please install and select the "cufinufft_chi2" or "finufft_chi2" backend when nterms > 1.'
                )
        elif 'cufinufft' in AVAILABLE_BACKENDS:
            backend = 'cufinufft'
        elif 'finufft' in AVAILABLE_BACKENDS:
            backend = 'finufft'
        else:
            raise ValueError(
                f'No valid backends available. AVAILABLE_BACKENDS = {AVAILABLE_BACKENDS}'
            )
    if backend not in AVAILABLE_BACKENDS:
        raise ValueError(
            f'Unknown or unavailable backend: {backend}. Available backends are: {AVAILABLE_BACKENDS}'
        )
    if backend in ('finufft', 'cufinufft') and nterms > 1:
        raise ValueError(
            f'Backend "{backend}" only supports nterms == 1. '
            'Use "cufinufft_chi2" or "finufft_chi2" for nterms > 1.'
        )
    if backend == 'cufinufft_chi2' or backend == 'finufft_chi2':
        # Add nterms to backend_kwargs and pass it to the backend
        backend_kwargs.setdefault('nterms', nterms)

    backend_module = importlib.import_module(f'.{backend}', __package__)

    power = backend_module.lombscargle(
        t=t,
        y=y,
        dy=dy,
        fmin=fmin,
        df=df,
        Nf=Nf,
        center_data=center_data,
        fit_mean=fit_mean,
        normalization=normalization,
        **backend_kwargs,
    )

    fmax = fmin + df * (Nf - 1)
    nifty_result = NiftyResult(
        power=power,
        fmin=fmin,
        df=df,
        Nf=Nf,
        fmax=fmax,
        center_data=center_data,
        fit_mean=fit_mean,
        normalization=normalization,
        backend=backend,
        backend_kwargs=backend_kwargs,
    )

    return nifty_result


def lombscargle_heterobatch(
    t_lst: List[npt.NDArray[np.floating]],
    y_lst: List[npt.NDArray[np.floating]],
    dy_lst: Optional[List[npt.NDArray[np.floating]]] = None,
    fmin_lst:Optional[List[float]] = None,
    fmax_lst: Optional[List[float]] = None,
    Nf_lst: Optional[List[float]] = None,
    center_data: bool = True,
    fit_mean: bool = True,
    normalization: NORMALIZATION_TYPE = 'standard',
    assume_sorted_t: bool = True,
    samples_per_peak: int = 5,
    nyquist_factor: int = 5,
    backend: BACKEND_TYPE = 'auto',
    nterms: int = 1,
    **backend_kwargs: Optional[dict],
) -> NiftyResult:
    """
    TODO: Need update
    Compute a Lomb-Scargle periodogram, or a batch of periodograms if `y` and `dy` are 2D arrays.

    This function can dispatch to multiple backends, including 'finufft' and 'cufinufft'. The latter
    uses CUDA and requires that nifty-ls was installed with the 'cuda' extra.

    The result is a `NiftyResult` dataclass containing the computed periodogram(s), frequency grid parameters,
    and other metadata. The actual frequency grid can be obtained by calling `freq()` on the result.

    The meanings of these parameters conform to the Lomb-Scargle implementation in Astropy:
    https://docs.astropy.org/en/stable/timeseries/lombscargle.html

    Parameters
    ----------
    t : List of array-like
        The time values, shape (N_series, N_d_i) for i in [0..N_series-1]
    y : List of array-like
        The data values, shape (N_series, N_t_i) or (N_series, N_y, N_t_i)
        for i in [0..N_series-1]. 
    dy : List of array-like, optional
        List of the uncertainties of the data values, broadcastable to `y`
    fmin : List of float, optional
        The minimum frequency of the periodogram. If not provided, it will be chosen automatically.
    fmax : List of float, optional
        The maximum frequency of the periodogram. If not provided, it will be chosen automatically.
    Nf : List of int, optional
        The number of frequency bins. If not provided, it will be chosen automatically.
    center_data : bool, optional
        Whether to center the data before computing the periodogram. Default is True.
    fit_mean : bool, optional
        Whether to fit a mean value to the data before computing the periodogram. Default is True.
    normalization : str, optional
        The normalization method to use. One of ['standard', 'model', 'log', 'psd']. Default is 'standard'.
    assume_sorted_t : bool, optional
        Whether to assume that the time values are sorted in ascending order, allowing for a performance
        optimization in determining the frequency grid.  Default is True.
    samples_per_peak : int, optional
        The number of samples per peak to use when determining the frequency grid. Default is 5.
    nyquist_factor : int, optional
        The factor by which to multiply the Nyquist frequency when determining the frequency grid. Default is 5.
    backend : str, optional
        The backend to use for the computation. Default is 'auto' which selects the best available backend.
    nterms : int, optional
        The number of terms to use in the Lomb-Scargle computation. Must be at least 1.
        If greater than 1, the 'cufinufft_chi2' or 'finufft_chi2' should be used for backend.
    backend_kwargs : dict, optional
        Additional keyword arguments to pass to the backend.

    Returns
    -------
    nifty_result : NiftyResult
        A dataclass containing the computed periodogram(s), frequency grid parameters, and other.
        The fields are 'power', 'fmin', 'df', 'Nf', and 'fmax'.
        `nifty_result.power` will be an ndarray of shape (Nf,) or (N_y, Nf) if `y` is 2D.
    """
    fmin_lst, df_lst, Nf_lst = utils.validate_frequency_grid_mp(
        fmin_lst,
        fmax_lst,
        Nf_lst,
        t_lst,
        assume_sorted_t=assume_sorted_t,
        samples_per_peak=samples_per_peak,
        nyquist_factor=nyquist_factor,
    )
    # Nterm verification
    if nterms is None:
        nterms = 1
    if nterms < 1:
        raise ValueError(f'nterms must be at least 1, got {nterms}.')

    # Backend selection
    if backend == 'auto':
        if nterms > 1:
            if 'cufinufft_chi2' in AVAILABLE_BACKENDS:
                backend = 'cufinufft_chi2'
            elif 'finufft_chi2' in AVAILABLE_BACKENDS:
                backend = 'finufft_chi2'
            else:
                raise ValueError(
                    'Please install and select the "cufinufft_chi2" or "finufft_chi2" backend when nterms > 1.'
                )
        elif 'cufinufft' in AVAILABLE_BACKENDS:
            backend = 'cufinufft'
        elif 'finufft_heterobatch' in AVAILABLE_BACKENDS:
            backend = 'finufft_heterobatch'
        else:
            raise ValueError(
                f'No valid backends available. AVAILABLE_BACKENDS = {AVAILABLE_BACKENDS}'
            )
    if backend not in AVAILABLE_BACKENDS:
        raise ValueError(
            f'Unknown or unavailable backend: {backend}. Available backends are: {AVAILABLE_BACKENDS}'
        )
    if backend in ('finufft_heterobatch', 'cufinufft') and nterms > 1:
        raise ValueError(
            f'Backend "{backend}" only supports nterms == 1. '
            'Use "cufinufft_chi2" or "finufft_chi2" for nterms > 1.'
        )
    if backend == 'cufinufft_chi2' or backend == 'finufft_chi2':
        # Add nterms to backend_kwargs and pass it to the backend
        backend_kwargs.setdefault('nterms', nterms)

    backend_module = importlib.import_module(f'.{backend}', __package__)

    powers = backend_module.lombscargle_heterobatch(
        t_lst=t_lst,
        y_lst=y_lst,
        dy_lst=dy_lst,
        fmin_lst=fmin_lst,
        df_lst=df_lst,
        Nf_lst=Nf_lst,
        center_data=center_data,
        fit_mean=fit_mean,
        normalization=normalization,
        **backend_kwargs,
    )

    fmax_lst = [fmin_lst[i] + df_lst[i] * (Nf_lst[i] - 1) for i in range(len(fmin_lst))]
    nifty_results = NiftyHeteroBatchResult(
        powes=powers,
        fmin_lst=fmin_lst,
        df_lst=df_lst,
        Nf_lst=Nf_lst,
        fmax_lst=fmax_lst,
        center_data=center_data,
        fit_mean=fit_mean,
        normalization=normalization,
        backend=backend,
        backend_kwargs=backend_kwargs,
    )

    return nifty_results


@dataclass
class NiftyResult:
    power: npt.NDArray[np.floating]
    fmin: float
    df: float
    Nf: int
    fmax: float
    center_data: bool
    fit_mean: bool
    normalization: NORMALIZATION_TYPE
    backend: BACKEND_TYPE
    backend_kwargs: Optional[dict]

    def freq(self) -> npt.NDArray[np.floating]:
        return self.fmin + self.df * np.arange(self.Nf)

@dataclass
class NiftyHeteroBatchResult:
    powers: List[npt.NDArray[np.floating]]
    fmin_lst: List[float]
    df_lst: List[float]
    Nf_lst: List[float]
    fmax_lst: List[float]
    center_data: bool
    fit_mean: bool
    normalization: NORMALIZATION_TYPE
    backend: BACKEND_TYPE
    backend_kwargs: Optional[dict]

    def freqs(self) -> List[npt.NDArray[np.floating]]:
        return [fmin_i + df_i * np.arange(Nf_i) for fmin_i, df_i, Nf_i 
                in zip(self.fmin_lst, self.df_lst, self.Nf_lst)]