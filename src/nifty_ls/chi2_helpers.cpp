/* This module contains C++ helper routines for processing finufft_chi2 method
 * inputs and outputs. Its main purpose is to enable "kernel fusion",
 * i.e. do as much array processing as possible element-wise, instead
 * of array-wise as occurs in Numpy.
 */
#include <iostream>

#include <complex>
#include <algorithm>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/complex.h>
#include <nanobind/stl/vector.h>
#include <lapacke.h>
#include <cblas.h>

#ifdef _OPENMP
#include <omp.h>

#include "cpu_helpers.hpp"
using cpu_helpers::NormKind;
using cpu_helpers::TermType;

// Declare a reduction for std::vector<double> using std::transform
#pragma omp declare reduction(                                                                                                             \
        vsum : std::vector<double> : std::transform(omp_out.begin(), omp_out.end(), omp_in.begin(), omp_out.begin(), std::plus<double>())) \
    initializer(omp_priv = decltype(omp_orig)(omp_orig.size()))

#endif

namespace nb = nanobind;
using namespace nb::literals;

const double PI = 3.14159265358979323846;

template <typename Scalar>
using nifty_arr_1d = nb::ndarray<Scalar, nb::ndim<1>, nb::device::cpu>;

template <typename Scalar>
using nifty_arr_2d = nb::ndarray<Scalar, nb::ndim<2>, nb::device::cpu>;

template <typename Scalar>
using nifty_arr_3d = nb::ndarray<Scalar, nb::ndim<3>, nb::device::cpu>;

template <typename Scalar>
using Complex = std::complex<Scalar>;

template <typename Scalar>
void process_chi2_inputs(
    nifty_arr_1d<Scalar> t1_,
    nifty_arr_2d<Complex<Scalar>> yw_,
    nifty_arr_2d<Complex<Scalar>> w_,
    nifty_arr_1d<Scalar> w2s_,
    nifty_arr_2d<Scalar> norm_,
    nifty_arr_1d<Scalar> yws_,
    nifty_arr_3d<Scalar> Sw_,
    nifty_arr_3d<Scalar> Cw_,
    nifty_arr_3d<Scalar> Syw_,
    nifty_arr_3d<Scalar> Cyw_,
    nifty_arr_1d<const Scalar> t_,
    nifty_arr_2d<const Scalar> y_,
    nifty_arr_2d<const Scalar> dy_,
    const Scalar df,
    const bool center_data,
    const bool fit_mean,
    int nthreads)
{
    auto t1 = t1_.view();
    auto yw = yw_.view();
    auto w = w_.view();
    auto w2s = w2s_.view();
    auto norm = norm_.view();
    auto yws = yws_.view();
    auto Sw = Sw_.view(); // shape (nSW(factor*nterms), Nbatch, Nf)
    auto Cw = Cw_.view();
    auto Syw = Syw_.view();
    auto Cyw = Cyw_.view();
    auto t = t_.view();   // read-only
    auto y = y_.view();   // read-only
    auto dy = dy_.view(); // read-only
    size_t Nf = Sw.shape(2);

    size_t Nbatch = y.shape(0);
    size_t N = y.shape(1);

    const Scalar TWO_PI = 2 * static_cast<Scalar>(PI);

#ifdef _OPENMP
    if (nthreads < 1)
    {
        nthreads = omp_get_max_threads();
    }
    if (nthreads > omp_get_max_threads())
    {
        fprintf(stderr,
                "[nifty-ls finufft] Warning: nthreads (%d) > omp_get_max_threads() (%d). Performance may be suboptimal.\n",
                nthreads, omp_get_max_threads());
    }
#else
    (void)nthreads; // suppress unused variable warning
#endif

#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(nthreads)
#endif
    // Compute t1
    for (size_t j = 0; j < N; ++j)
    {
        t1(j) = TWO_PI * df * t(j);
    }
    // TODO: Can I do parallel for here?
    // #ifdef _OPENMP
    // #pragma omp parallel for schedule(static) num_threads(nthreads) collapse(2) reduction(vsum : wsum) reduction(vsum : yoff)
    // #endif
    // Batch loop : compute w2s, norm, yws, and fill yw, w
    for (size_t i = 0; i < Nbatch; ++i)
    {
        Scalar sum_w = Scalar(0); // Type is <Scalar>, value is 0
        Scalar yoff = Scalar(0);

        // First pass: sum weights and weighted y
        // w2_base = (dy ** -2.0).astype(dtype)
        // w2s = np.sum(w2_base.real, axis=-1)
        // (w2.real * y).sum(axis=-1, keepdims=True)
        for (size_t j = 0; j < N; ++j)
        {
            Scalar wt = Scalar(1) / (dy(i, j) * dy(i, j));
            sum_w += wt;
            yoff += wt * y(i, j);
        }

        // Store sum of weights
        w2s(i) = sum_w;

        if (center_data || fit_mean)
        {
            // ((w2.real * y).sum(axis=-1, keepdims=True) / w2s)
            yoff /= sum_w; // Normalize yoff by the sum of weights
        }
        else
        {
            yoff = Scalar(0); // y(i, j) - yoff = y(i, j)
        }

        Scalar sum_norm = Scalar(0);
        Scalar sum_yw2 = Scalar(0);

        // Second pass: compute norm, yws, and fill arrays
        for (size_t j = 0; j < N; ++j)
        {
            Scalar wt = Scalar(1) / (dy(i, j) * dy(i, j));
            Scalar ym = y(i, j) - yoff;
            sum_norm += wt * (ym * ym);
            sum_yw2 += ym * wt;

            // Construct a complex number for yw and w
            yw(i, j) = std::complex<Scalar>(ym * wt, Scalar(0));
            w(i, j) = std::complex<Scalar>(wt, Scalar(0));
        }
        // norm = (w2.real * y**2).sum(axis=-1, keepdims=True)
        norm(i, 0) = sum_norm;
        // yws = (y * w2.real).sum(axis=-1)
        yws(i) = sum_yw2;

        // Fill trig matrix
        for (size_t f = 0; f < Nf; ++f)
        {
            Sw(0, i, f) = Scalar(0);
            Syw(0, i, f) = Scalar(0);
            Cw(0, i, f) = w2s(i);
            Cyw(0, i, f) = yws(i);
        }
    }
}

template <typename Scalar>
void compute_t(
    nifty_arr_1d<const Scalar> &t1_,
    nifty_arr_2d<const Complex<Scalar>> &yw_w_,
    const int time_shift,
    const Scalar fmin,
    const Scalar df,
    const int Nf,
    nifty_arr_1d<Scalar> &tn_out,
    nifty_arr_2d<Complex<Scalar>> &yw_w_s_out)
{
    auto t1 = t1_.view();          // input
    auto yw_w = yw_w_.view();      // input with 2*Nbatch × N size
    auto tn = tn_out.view();       // output length-N array
    auto yw_s = yw_w_s_out.view(); // output same shape as yw_w_

    const size_t N = t1.shape(0);
    const size_t nTrans = yw_w.shape(0);

    // TODO: Do parallel for here?
    // tn = time_shift * t1
    for (size_t j = 0; j < N; ++j)
    {
        tn(j) = Scalar(time_shift) * t1(j);
    }

    // shift factor = (Nf/2) + fmin/df
    Scalar factor = Scalar(Nf / 2) + fmin / df;

    // TODO: Do parallel for here? And can I use OpenMP reduction?
    // do phase shift: phase_shiftn = np.exp(1j * ((Nf // 2) + fmin / df) * tn)
    // size_t chunk_size = std::max(8, N / nthreads);
    // #pragma omp parallel for schedule(static, chunk_size)
    for (size_t j = 0; j < N; ++j)
    {
        // complex phase = exp(i * factor * tn[j])
        Complex<Scalar> phase = std::exp(
            Complex<Scalar>(0, factor * tn(j)));

        // multiply every transform in the batch by this phase
        for (size_t b = 0; b < nTrans; ++b)
        {
            yw_s(b, j) = yw_w(b, j) * phase;
        }
    }
}

// TODO: Using namespace std
// TODO: check pass by reference or by value
template <typename Scalar>
void process_chi2_outputs(
    nifty_arr_2d<Scalar> power_,
    nifty_arr_3d<const Scalar> Sw_,
    nifty_arr_3d<const Scalar> Cw_,
    nifty_arr_3d<const Scalar> Syw_,
    nifty_arr_3d<const Scalar> Cyw_,
    nifty_arr_2d<const Scalar> norm_,
    const std::vector<TermType> &order_types,
    const std::vector<size_t> &order_indices,
    const NormKind norm_kind,
    int nthreads)
{
    auto power = power_.view();                   // output
    const size_t order_size = order_types.size(); // input
    auto Sw = Sw_.view();                         // input (n_terms, Nbatch, Nf)
    auto Cw = Cw_.view();                         // input
    auto Syw = Syw_.view();                       // input
    auto Cyw = Cyw_.view();                       // input
    auto norm = norm_.view();                     // input

    // const size_t n_terms = Sw.shape(0);
    const size_t Nbatch = Sw.shape(1);
    const size_t Nf = Sw.shape(2);

#ifdef _OPENMP
    if (nthreads < 1)
    {
        nthreads = omp_get_max_threads();
    }
#else
    (void)nthreads; // suppress unused variable warning
#endif

    // Process each batch and frequency
    // TODO: Can I use OpenMP here? OpenMP reduction?
    // #ifdef _OPENMP
    // #pragma omp parallel for schedule(static) num_threads(nthreads) collapse(2)
    // #endif
    for (size_t b = 0; b < Nbatch; ++b) // b: batches index
    {
        for (size_t f = 0; f < Nf; ++f) // f: frequencies index
        {
            // Build XTX matrix and XTy vector for this batch and frequency
            // Bottle Neck!
            std::vector<std::vector<Scalar>> XTX(order_size, std::vector<Scalar>(order_size));
            std::vector<Scalar> XTy(order_size, 0.0);

            // Fill XTy
            for (size_t i = 0; i < order_size; ++i)
            {
                TermType t = order_types[i]; // Sine or Cosine
                size_t m = order_indices[i];
                XTy[i] = (t == TermType::Sine) ? Syw(m, b, f) : Cyw(m, b, f);
            }

            // Fill XTX
            for (size_t i = 0; i < order_size; ++i)
            {
                TermType ti = order_types[i];
                size_t m = order_indices[i];
                for (size_t j = 0; j < order_size; ++j)
                {
                    TermType tj = order_types[j];
                    size_t n = order_indices[j];

                    size_t d = (m > n) ? (m - n) : (n - m);
                    size_t s = m + n;

                    if (ti == TermType::Sine && tj == TermType::Sine)
                    {
                        XTX[i][j] = Scalar(0.5) * (Cw(d, b, f) - Cw(s, b, f));
                    }
                    else if (ti == TermType::Cosine && tj == TermType::Cosine)
                    {
                        XTX[i][j] = Scalar(0.5) * (Cw(d, b, f) + Cw(s, b, f));
                    }
                    else if (ti == TermType::Sine && tj == TermType::Cosine)
                    {
                        int sign = (m > n ? 1 : (m < n ? -1 : 0));
                        XTX[i][j] = Scalar(0.5) * (sign * Sw(d, b, f) + Sw(s, b, f));
                    }
                    else
                    { // Cosine, Sine
                        int sign = (n > m ? 1 : (n < m ? -1 : 0));
                        XTX[i][j] = Scalar(0.5) * (sign * Sw(d, b, f) + Sw(s, b, f));
                    }
                }
            }

            // 2) Solve XTX * beta = XTy using Cholesky
            // Flatten into column-major for LAPACK
            std::vector<Scalar> A(order_size * order_size);
            for (size_t i = 0; i < order_size; ++i)
                for (size_t j = 0; j < order_size; ++j)
                    A[j * order_size + i] = XTX[i][j];
            std::vector<Scalar> bvec = XTy;
            int n = int(order_size), nrhs = 1, info;
            if constexpr (std::is_same_v<Scalar, double>)
            {
                info = LAPACKE_dposv(LAPACK_COL_MAJOR, 'U', n, nrhs,
                                     A.data(), n,
                                     bvec.data(), n);
            }
            else
            {
                info = LAPACKE_sposv(LAPACK_COL_MAJOR, 'U', n, nrhs,
                                     A.data(), n,
                                     bvec.data(), n);
            }
            if (info != 0)
                throw std::runtime_error("Cholesky solve failed");

            // 3) Compute dot(XTy, beta)
            Scalar pw;
            if constexpr (std::is_same_v<Scalar, double>)
            {
                pw = cblas_ddot(n, bvec.data(), 1, XTy.data(), 1);
            }
            else
            {
                pw = cblas_sdot(n, bvec.data(), 1, XTy.data(), 1);
            }
            power(b, f) = pw;

            // TODO: Make it parallel
            // Apply normalization
            switch (norm_kind)
            {
            case NormKind::Standard:
                power(b, f) /= norm(b, 0);
                break;
            case NormKind::Model:
                power(b, f) /= (norm(b, 0) - power(b, f));
                break;
            case NormKind::Log:
                power(b, f) = -std::log(1 - power(b, f) / norm(b, 0));
                break;
            case NormKind::PSD:
                power(b, f) *= 0.5;
                break;
            }
        }
    }
}

template <typename Scalar>
Scalar _normalization(nifty_arr_1d<const Scalar> y_, std::vector<Scalar> &w, Scalar wsum, Scalar yoff)
{
    // only used for winding

    const auto y = y_.view(); // read-only

    Scalar invnorm = Scalar(0);
    const size_t N = y.shape(0);
    for (size_t n = 0; n < N; ++n)
    {
        invnorm += (w[n] / wsum) * (y(n) - yoff) * (y(n) - yoff);
    }
    return Scalar(1) / invnorm;
}

NB_MODULE(chi2_helpers, m)
{
    // We're using noconvert() here to ensure the input arrays are not copied

    m.def("process_chi2_inputs", &process_chi2_inputs<double>,
          "t1"_a.noconvert(),
          "yw"_a.noconvert(),
          "w"_a.noconvert(),
          "w2s"_a.noconvert(),
          "norm"_a.noconvert(),
          "yws"_a.noconvert(),
          "Sw"_a.noconvert(),
          "Cw"_a.noconvert(),
          "Syw"_a.noconvert(),
          "Cyw"_a.noconvert(),
          "t"_a.noconvert(),
          "y"_a.noconvert(),
          "dy"_a.noconvert(),
          "df"_a,
          "center_data"_a,
          "fit_mean"_a,
          "nthreads"_a);

    m.def("process_chi2_inputs", &process_chi2_inputs<float>,
          "t1"_a.noconvert(),
          "yw"_a.noconvert(),
          "w"_a.noconvert(),
          "w2s"_a.noconvert(),
          "norm"_a.noconvert(),
          "yws"_a.noconvert(),
          "Sw"_a.noconvert(),
          "Cw"_a.noconvert(),
          "Syw"_a.noconvert(),
          "Cyw"_a.noconvert(),
          "t"_a.noconvert(),
          "y"_a.noconvert(),
          "dy"_a.noconvert(),
          "df"_a,
          "center_data"_a,
          "fit_mean"_a,
          "nthreads"_a);

    m.def("compute_t", &compute_t<double>,
          "t1"_a.noconvert(),
          "yw_w"_a.noconvert(),
          "time_shift"_a,
          "fmin"_a,
          "df"_a,
          "Nf"_a,
          "tn_out"_a.noconvert(),
          "yw_w_s_out"_a.noconvert());

    m.def("compute_t", &compute_t<double>,
          "t1"_a.noconvert(),
          "yw_w"_a.noconvert(),
          "time_shift"_a,
          "fmin"_a,
          "df"_a,
          "Nf"_a,
          "tn_out"_a.noconvert(),
          "yw_w_s_out"_a.noconvert());

    m.def("process_chi2_outputs", &process_chi2_outputs<double>,
          "power"_a,
          "Sw"_a,
          "Cw"_a,
          "Syw"_a,
          "Cyw"_a,
          "norm"_a,
          "order_types"_a,
          "order_indices"_a,
          "norm_kind"_a,
          "nthreads"_a);

    m.def("process_chi2_outputs", &process_chi2_outputs<float>,
          "power"_a,
          "Sw"_a,
          "Cw"_a,
          "Syw"_a,
          "Cyw"_a,
          "norm"_a,
          "order_types"_a,
          "order_indices"_a,
          "norm_kind"_a,
          "nthreads"_a);

    m.def("compute_winding", &compute_winding<double>,
          "power"_a.noconvert(),
          "t"_a.noconvert(),
          "y"_a.noconvert(),
          "w"_a.noconvert(),
          "fmin"_a,
          "df"_a,
          "center_data"_a,
          "fit_mean"_a);

    nb::enum_<TermType>(m, "TermType")
        .value("Sine", TermType::Sine)
        .value("Cosine", TermType::Cosine);
}
