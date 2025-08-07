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


@pytest.fixture
def data(request):
    """Generate test data with specified N data points and N_batch series"""
    # Check if this is a parametrized call
    if hasattr(request, 'param'):
        N_series = request.param.get('N_series', 10_000)
        N_d = request.param.get('N', 100)
        N_batch = request.param.get('N_batch', 10)
    else:
        # Default values when not parametrized
        N_series = 100
        N_d = 100
        N_batch = 1

    return gen_data_mp(N_series=N_series, N_d=N_d, N_batch=N_batch)


@pytest.fixture(scope='module')
def nifty_backend(request):
    """Set up the backend function based on the parameterized backend name"""
    available_heterobatch_backends = nifty_ls.backends.available_backends(
        backend_names=nifty_ls.backends.HETEROBATCH_BACKEND_NAMES
    )

    if request.param in available_heterobatch_backends:
        return (
            request.param,
            partial(nifty_ls.lombscargle_heterobatch, backend=request.param),
        )
    else:
        pytest.skip(f'Backend {request.param} is not available')


@pytest.mark.parametrize(
    'data',
    [
        {'N_series': 1000, 'N': 100, 'N_batch': 5},
        {'N_series': 10_000, 'N': 100, 'N_batch': 1},
    ],
    indirect=['data'],
)
@pytest.mark.parametrize(
    'nifty_backend,nterms',
    [
        ('finufft_heterobatch', 1),
        ('finufft_chi2_heterobatch', 2),
    ],
    indirect=['nifty_backend'],
)
def test_lombscargle(data, nifty_backend, nterms, Nf=1000):
    """Check that heterobatch implementation agrees with single series results"""

    t_list = data['t']
    y_list = data['y']
    dy_list = data['dy']
    fmin_list = data['fmin']
    Nf_list = [Nf] * len(t_list)

    # heterobatch
    heterobatch_results = nifty_backend[1](
        t_list=t_list,
        y_list=y_list,
        dy_list=dy_list,
        fmin_list=fmin_list,
        Nf_list=Nf_list,
        nterms=nterms,
    )

    # Single series
    standard_result_powers = []
    for i in range(len(t_list)):
        standard_result = nifty_ls.lombscargle(
            t=t_list[i],
            y=y_list[i],
            dy=dy_list[i],
            fmin=fmin_list[i],
            Nf=Nf,
            nterms=nterms,
            backend='finufft'
            if nifty_backend[0] == 'finufft_heterobatch'
            else 'finufft_chi2',
        )
        standard_result_powers.append(standard_result.power)

    np.testing.assert_allclose(heterobatch_results.powers, standard_result_powers)


@pytest.mark.parametrize(
    'nifty_backend,nterms',
    [
        ('finufft_heterobatch', 1),
        ('finufft_chi2_heterobatch', 2),
    ],
    indirect=['nifty_backend'],
)
def test_normalization(data, nifty_backend, nterms, Nf=1000):
    """Check that the normalization modes work as expected"""

    t_list = data['t']
    y_list = data['y']
    dy_list = data['dy']
    fmin_list = data['fmin']
    Nf_list = [Nf] * len(t_list)

    for norm in ['standard', 'model', 'log', 'psd']:
        # heterobatch
        heterobatch_results = nifty_backend[1](
            t_list=t_list,
            y_list=y_list,
            dy_list=dy_list,
            fmin_list=fmin_list,
            Nf_list=Nf_list,
            normalization=norm,
            nterms=nterms,
        )

        # Single series
        standard_result_powers = []
        for i in range(len(t_list)):
            standard_result = nifty_ls.lombscargle(
                t=t_list[i],
                y=y_list[i],
                dy=dy_list[i],
                fmin=fmin_list[i],
                Nf=Nf,
                normalization=norm,
                nterms=nterms,
                backend='finufft'
                if nifty_backend[0] == 'finufft_heterobatch'
                else 'finufft_chi2',
            )
            standard_result_powers.append(standard_result.power)

        np.testing.assert_allclose(heterobatch_results.powers, standard_result_powers)


@pytest.mark.parametrize('center_data', [True, False])
@pytest.mark.parametrize(
    'nifty_backend,nterms',
    [
        ('finufft_heterobatch', 1),
        ('finufft_chi2_heterobatch', 2),
    ],
    indirect=['nifty_backend'],
)
def test_center_data(data, center_data, nifty_backend, nterms, Nf=1000):
    """Check that the center_data parameter work as expected"""

    t_list = data['t']
    y_list = data['y']
    dy_list = data['dy']
    fmin_list = data['fmin']
    Nf_list = [Nf] * len(t_list)

    # heterobatch
    heterobatch_results = nifty_backend[1](
        t_list=t_list,
        y_list=y_list,
        dy_list=dy_list,
        fmin_list=fmin_list,
        Nf_list=Nf_list,
        center_data=center_data,
        nterms=nterms,
    )

    # Single series
    standard_result_powers = []
    for i in range(len(t_list)):
        standard_result = nifty_ls.lombscargle(
            t=t_list[i],
            y=y_list[i],
            dy=dy_list[i],
            fmin=fmin_list[i],
            Nf=Nf,
            center_data=center_data,
            nterms=nterms,
            backend='finufft'
            if nifty_backend[0] == 'finufft_heterobatch'
            else 'finufft_chi2',
        )
        standard_result_powers.append(standard_result.power)

    np.testing.assert_allclose(heterobatch_results.powers, standard_result_powers)


@pytest.mark.parametrize('fit_mean', [True, False])
@pytest.mark.parametrize(
    'nifty_backend,nterms',
    [
        ('finufft_heterobatch', 1),
        ('finufft_chi2_heterobatch', 2),
    ],
    indirect=['nifty_backend'],
)
def test_fit_mean(data, fit_mean, nifty_backend, nterms, Nf=1000):
    """Check that the fit_mean parameter work as expected"""

    t_list = data['t']
    y_list = data['y']
    dy_list = data['dy']
    fmin_list = data['fmin']
    Nf_list = [Nf] * len(t_list)

    # heterobatch
    heterobatch_results = nifty_backend[1](
        t_list=t_list,
        y_list=y_list,
        dy_list=dy_list,
        fmin_list=fmin_list,
        Nf_list=Nf_list,
        fit_mean=fit_mean,
        nterms=nterms,
    )

    # Single series
    standard_result_powers = []
    for i in range(len(t_list)):
        standard_result = nifty_ls.lombscargle(
            t=t_list[i],
            y=y_list[i],
            dy=dy_list[i],
            fmin=fmin_list[i],
            Nf=Nf,
            fit_mean=fit_mean,
            nterms=nterms,
            backend='finufft'
            if nifty_backend[0] == 'finufft_heterobatch'
            else 'finufft_chi2',
        )
        standard_result_powers.append(standard_result.power)

    np.testing.assert_allclose(heterobatch_results.powers, standard_result_powers)


@pytest.mark.parametrize(
    'data',
    [
        {'N_series': 100, 'N': 100, 'N_batch': 1},
        {'N_series': 100, 'N': 100, 'N_batch': 5},
    ],
    indirect=['data'],
)
@pytest.mark.parametrize(
    'nifty_backend,nterms',
    [
        ('finufft_heterobatch', 1),
        ('finufft_chi2_heterobatch', 2),
    ],
    indirect=['nifty_backend'],
)
def test_dy_none(data, nifty_backend, nterms, Nf=1000):
    """Test that `dy = None` works properly"""

    t_list = data['t']
    y_list = data['y']
    fmin_list = data['fmin']
    Nf_list = [Nf] * len(t_list)

    # heterobatch
    heterobatch_results = nifty_backend[1](
        t_list=t_list,
        y_list=y_list,
        dy_list=None,
        fmin_list=fmin_list,
        Nf_list=Nf_list,
        nterms=nterms,
    )

    # Single series
    standard_result_powers = []
    for i in range(len(t_list)):
        standard_result = nifty_ls.lombscargle(
            t=t_list[i],
            y=y_list[i],
            dy=None,
            fmin=fmin_list[i],
            Nf=Nf,
            nterms=nterms,
            backend='finufft'
            if nifty_backend[0] == 'finufft_heterobatch'
            else 'finufft_chi2',
        )
        standard_result_powers.append(standard_result.power)

    np.testing.assert_allclose(heterobatch_results.powers, standard_result_powers)


@pytest.mark.parametrize(
    'data',
    [
        {'N_series': 100, 'N': 100, 'N_batch': 1},
        {'N_series': 100, 'N': 100, 'N_batch': 5},
    ],
    indirect=['data'],
)
@pytest.mark.parametrize(
    'nifty_backend,nterms',
    [
        ('finufft_heterobatch', 1),
        ('finufft_chi2_heterobatch', 2),
    ],
    indirect=['nifty_backend'],
)
def test_dy_scalar(data, nifty_backend, nterms, Nf=1000):
    """Test that dy_list can be a scalar value"""

    t_list = data['t']
    y_list = data['y']
    fmin_list = data['fmin']
    Nf_list = [Nf] * len(t_list)

    scalar_dy = 0.5

    heterobatch_results = nifty_backend[1](
        t_list=t_list,
        y_list=y_list,
        dy_list=scalar_dy,
        fmin_list=fmin_list,
        Nf_list=Nf_list,
        nterms=nterms,
    )

    standard_result_powers = []
    for i in range(len(t_list)):
        standard_result = nifty_ls.lombscargle(
            t=t_list[i],
            y=y_list[i],
            dy=scalar_dy,
            fmin=fmin_list[i],
            Nf=Nf,
            nterms=nterms,
            backend='finufft'
            if nifty_backend[0] == 'finufft_heterobatch'
            else 'finufft_chi2',
        )
        standard_result_powers.append(standard_result.power)

    np.testing.assert_allclose(heterobatch_results.powers, standard_result_powers)


@pytest.mark.parametrize(
    'data',
    [
        {'N_series': 100, 'N': 100, 'N_batch': 1},
        {'N_series': 100, 'N': 100, 'N_batch': 5},
    ],
    indirect=['data'],
)
@pytest.mark.parametrize(
    'nifty_backend,nterms',
    [
        ('finufft_heterobatch', 1),
        ('finufft_chi2_heterobatch', 2),
    ],
    indirect=['nifty_backend'],
)
def test_dy_scalar_list(data, nifty_backend, nterms, Nf=1000):
    """Test that dy_list can be a list of scalar values"""

    t_list = data['t']
    y_list = data['y']
    fmin_list = data['fmin']
    Nf_list = [Nf] * len(t_list)

    scalar_dy_list = [0.5 * (i + 1) for i in range(len(t_list))]

    heterobatch_results = nifty_backend[1](
        t_list=t_list,
        y_list=y_list,
        dy_list=scalar_dy_list,
        fmin_list=fmin_list,
        Nf_list=Nf_list,
        nterms=nterms,
    )

    standard_result_powers = []
    for i in range(len(t_list)):
        standard_result = nifty_ls.lombscargle(
            t=t_list[i],
            y=y_list[i],
            dy=scalar_dy_list[i],
            fmin=fmin_list[i],
            Nf=Nf,
            nterms=nterms,
            backend='finufft'
            if nifty_backend[0] == 'finufft_heterobatch'
            else 'finufft_chi2',
        )
        standard_result_powers.append(standard_result.power)

    np.testing.assert_allclose(heterobatch_results.powers, standard_result_powers)


@pytest.mark.parametrize(
    'nifty_backend,nterms',
    [
        ('finufft_heterobatch', 1),
        ('finufft_chi2_heterobatch', 2),
    ],
    indirect=['nifty_backend'],
)
def test_mixed_dtypes(data, nifty_backend, nterms):
    """Test that mixed dtypes raise an appropriate error"""
    data_mixed = {}
    data_mixed['t_list'] = [data['t'][0].astype(np.float32)]
    data_mixed['y_list'] = [data['y'][-1].astype(np.float64)]
    data_mixed['dy_list'] = [data['dy'][len(data['dy']) // 2].astype(np.float64)]

    with pytest.raises(ValueError, match='dtype'):
        nifty_backend[1](**data_mixed, nterms=nterms)
