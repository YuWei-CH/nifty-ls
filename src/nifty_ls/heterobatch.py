from __future__ import annotations

__all__ = ['lombscargle', 'FFTW_MEASURE', 'FFTW_ESTIMATE']

import numpy as np

from nifty_ls.heterobatch_helpers import process_hetero_batch


def lombscargle_heterobatch(
        t_list,
        y_list,
        fmin_list,
        df_list,
        Nf_list,
        dy_list = None,
        nthreads = None,
        center_data = True,
        fit_mean = True,
        normalization='standard',
        verbose=False,
        finufft_kwargs=None, # TODO: Use it
):
    # TODO: Add function intro

    dtype = t_list[0].dtype

    # Create array with same shape as y, filled with ones
    if dy_list is None:
        dy_list = []
        for i in range(len(y_list)):
            dy_list.append(np.ones_like(y_list[i]))
    # Check if dy is a scalar (number) rather than an array
    for i, dy in enumerate(dy_list):
        if np.isscalar(dy):
            dy_list[i] = np.atleast_2d(dy)
    
    powers = []
    for i in range(len(t_list)):
        power = np.zeros((y_list[i].shape[0], Nf_list[i]), dtype=np.float64)
        powers.append(power)

    process_hetero_batch(
            t_list,
            y_list,
            dy_list,
            fmin_list,
            df_list,
            Nf_list,
            powers,
            normalization,
            nthreads,
            center_data,
            fit_mean,
            verbose
        )
    
    return power