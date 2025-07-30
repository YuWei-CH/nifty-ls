"""
Test module for heterobatch implementation of Lomb-Scargle periodogram.
Compares heterobatch results against single-batch calculations.
"""

from __future__ import annotations

from functools import partial

import numpy as np
import pytest

import nifty_ls
import nifty_ls.backends
from nifty_ls.test_helpers.utils import gen_data_mp

def rtol(dtype, Nf):
    """Use a relative tolerance that accounts for the condition number of the problem"""
    if dtype == np.float32:
        # NB we don't test float32
        return max(1e-3, 1e-7 * Nf)
    elif dtype == np.float64:
        return max(1e-5, 1e-9 * Nf)
    else:
        raise ValueError(f'Unknown dtype {dtype}')

@pytest.fixture(scope='module')
def data(N):
    """Generate test data with N data points per series"""
    return gen_data_mp(N_series=5, N_d=N, N_batch=10)

@pytest.fixture(scope='module')
def nifty_backend(request):
    """Set up the backend function based on the parameterized backend name"""
    avail_backends = nifty_ls.core.AVAILABLE_BACKENDS

    if request.param in avail_backends:
        if "heterobatch" in request.param:
            fn = partial(nifty_ls.lombscargle_heterobatch, backend=request.param)
        else:
            fn = partial(nifty_ls.lombscargle, backend=request.param)
        return fn, request.param
    else:
        pytest.skip(f'Backend {request.param} is not available')


@pytest.mark.parametrize('N', [100, 1_000])
@pytest.mark.parametrize('Nf', [100, 500])
@pytest.mark.parametrize(
    'nifty_backend',
    [
        'finufft_heterobatch',
        'finufft_heterobatch_chi2',
    ],
    indirect=['nifty_backend'],
)
def test_lombscargle(data, N, Nf, nifty_backend):
    """Check that heterobatch implementation agrees with single series results"""
    
    backend_fn, backend_name = nifty_backend
    
    # Extract data into appropriate format for heterobatch
    t_list = data['t']
    y_list = data['y']
    dy_list = data['dy']
    fmin_list = data['fmin']
    df_list = data['df']
    Nf_list = [Nf] * len(t_list)  # Use same Nf for all series
    
    # Run heterobatch version
    heterobatch_results = backend_fn(
        t_list=t_list,
        y_list=y_list,
        dy_list=dy_list,
        fmin_list=fmin_list,
        df_list=df_list,
        Nf_list=Nf_list
    )
    
    # Compare with individual series calculations
    for i in range(len(t_list)):
        # Run single-series calculation with standard backend
        standard_result = nifty_ls.lombscargle(
            t=t_list[i],
            y=y_list[i],
            dy=dy_list[i],
            fmin=fmin_list[i],
            df=df_list[i],
            Nf=Nf
        )
        
        # Compare results for this series
        np.testing.assert_allclose(
            heterobatch_results.powers[i],
            standard_result.power,
            rtol=rtol(t_list[i].dtype, Nf),
            err_msg=f"Series {i} failed comparison"
        )

    # If we get here, all comparisons passed
    assert True