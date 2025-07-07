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

#include "cpu_helpers.hpp"
using cpu_helpers::NormKind;
using cpu_helpers::TermType;

#ifdef _OPENMP
#include <omp.h>

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
    nifty_arr_2d<Scalar> w2s_, // Changed from 1D to 2D to match Python keepdims=True
    nifty_arr_2d<Scalar> norm_,
    nifty_arr_2d<Scalar> yws_, // Changed from 1D to 2D to match Python keepdims=True
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
    auto Sw = Sw_.view(); // shape (Nbatch, nSW, Nf) - changed from (nSW, Nbatch, Nf)
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

// Initialize and fill value of yw, w, w2s, norm, yws, Sw, Cw, Syw, Cyw
#ifdef _OPENMP
#pragma omp parallel num_threads(nthreads)
#endif
    {
#ifdef _OPENMP
// Use single thread to pack the task, and rest threads will execute the task
#pragma omp single nowait
#pragma omp taskloop
#endif
        // Batch loop : compute w2s, norm, yws, and fill yw, w
        for (size_t i = 0; i < Nbatch; ++i)
        {
            Scalar sum_w = Scalar(0); // Type is <Scalar>, value is 0
            Scalar yoff = Scalar(0);

// First pass: sum weights and weighted y
// w2_base = (dy ** -2.0).astype(dtype)
// w2s = np.sum(w2_base.real, axis=-1)
// (w2.real * y).sum(axis=-1, keepdims=True)
#ifdef _OPENMP
#pragma omp taskloop reduction(+ : sum_w) reduction(+ : yoff)
#endif
            for (size_t j = 0; j < N; ++j)
            {
                Scalar wt = Scalar(1) / (dy(i, j) * dy(i, j));
                sum_w += wt;
                yoff += wt * y(i, j);
            }

            // Store sum of weights with keepdims=True shape (Nbatch, 1)
            w2s(i, 0) = sum_w;

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

#ifdef _OPENMP
#pragma omp taskloop reduction(+ : sum_norm) reduction(+ : sum_yw2)
#endif
            // Second pass: compute norm, yws, and fill arrays
            for (size_t m = 0; m < N; ++m)
            {
                Scalar wt = Scalar(1) / (dy(i, m) * dy(i, m));
                Scalar ym = y(i, m) - yoff;
                sum_norm += wt * (ym * ym);
                sum_yw2 += ym * wt;

                // Construct a complex number for yw and w
                yw(i, m) = std::complex<Scalar>(ym * wt, Scalar(0));
                w(i, m) = std::complex<Scalar>(wt, Scalar(0));
            }
            // norm = (w2.real * y**2).sum(axis=-1, keepdims=True)
            norm(i, 0) = sum_norm;
            // yws = (y * w2.real).sum(axis=-1, keepdims=True)
            if (center_data || fit_mean)
            {
                yws(i, 0) = Scalar(0); // Mathematically, fit_mean or center_data will set yws to 0
            }
            else
            {
                yws(i, 0) = sum_yw2; // Use the sum of weighted y directly
            }

#ifdef _OPENMP
#pragma omp taskloop
#endif
            // Initialize trig matrix
            for (size_t f = 0; f < Nf; ++f)
            {
                Sw(i, 0, f) = Scalar(0);
                Syw(i, 0, f) = Scalar(0);
                Cw(i, 0, f) = w2s(i, 0); // Use 2D indexing to match Python keepdims=True
                Cyw(i, 0, f) = yws(i, 0);
            }
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
    nifty_arr_2d<Complex<Scalar>> &yw_w_s_out,
    int nthreads)
{
    auto t1 = t1_.view();          // input
    auto yw_w = yw_w_.view();      // input with 2*Nbatch × N size
    auto tn = tn_out.view();       // output length-N array
    auto yw_s = yw_w_s_out.view(); // output same shape as yw_w_

    const size_t N = t1.shape(0);
    const size_t nTrans = yw_w.shape(0);

#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(nthreads)
#endif
    // tn = time_shift * t1
    for (size_t j = 0; j < N; ++j)
    {
        tn(j) = Scalar(time_shift) * t1(j);
    }

    // shift factor = (Nf/2) + fmin/df
    Scalar factor = Scalar(Nf / 2) + fmin / df;

    // do phase shift: phase_shiftn = np.exp(1j * ((Nf // 2) + fmin / df) * tn)
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static) num_threads(nthreads)
#endif
    for (size_t b = 0; b < nTrans; ++b)
    {
        for (size_t j = 0; j < N; ++j)
        {
            Complex<Scalar> phase = std::exp(Complex<Scalar>(0, factor * tn(j)));
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
    auto Sw = Sw_.view();                         // input (Nbatch, nSW, Nf)
    auto Cw = Cw_.view();                         // input
    auto Syw = Syw_.view();                       // input
    auto Cyw = Cyw_.view();                       // input
    auto norm = norm_.view();                     // input

    const size_t Nbatch = Sw.shape(0); // Updated index for new array layout
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
#ifdef _OPENMP
#pragma omp parallel num_threads(nthreads)
#endif
    {
        // Pre-allocate arrays to avoid repeated allocations
        std::vector<std::vector<Scalar>> XTX(order_size, std::vector<Scalar>(order_size));
        std::vector<Scalar> XTy(order_size);
        std::vector<Scalar> sw_local(Sw.shape(1));
        std::vector<Scalar> cw_local(Cw.shape(1));
        std::vector<Scalar> A(order_size * order_size);
        std::vector<Scalar> bvec(order_size);
        std::vector<lapack_int> ipiv(order_size);

#ifdef _OPENMP
#pragma omp single
#pragma omp taskloop
#endif
        for (size_t b = 0; b < Nbatch; ++b)
        {
// Parallelize batch processing with taskloop to achieve dynamic load balancing,
// since the cost of LU decomposition varies across batches.
#ifdef _OPENMP
#pragma omp taskloop
#endif
            for (size_t f = 0; f < Nf; ++f)
            {
                // Prefetch data into local arrays
                for (size_t i = 0; i < Sw.shape(1); ++i)
                {
                    sw_local[i] = Sw(b, i, f);
                    cw_local[i] = Cw(b, i, f);
                }

                // Fill XTy with proper indexing for new array layout
                for (size_t i = 0; i < order_size; ++i)
                {
                    TermType t = order_types[i]; // Sine or Cosine
                    size_t m = order_indices[i]; // Nterms
                    XTy[i] = (t == TermType::Sine) ? Syw(b, m, f) : Cyw(b, m, f);
                }

                // Fill XTX efficiently using local arrays
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
                            XTX[i][j] = Scalar(0.5) * (cw_local[d] - cw_local[s]);
                        }
                        else if (ti == TermType::Cosine && tj == TermType::Cosine)
                        {
                            XTX[i][j] = Scalar(0.5) * (cw_local[d] + cw_local[s]);
                        }
                        else if (ti == TermType::Sine && tj == TermType::Cosine)
                        {
                            int sign = (m > n ? 1 : (m < n ? -1 : 0));
                            XTX[i][j] = Scalar(0.5) * (sign * sw_local[d] + sw_local[s]);
                        }
                        else
                        { // Cosine, Sine
                            int sign = (n > m ? 1 : (n < m ? -1 : 0));
                            XTX[i][j] = Scalar(0.5) * (sign * sw_local[d] + sw_local[s]);
                        }
                    }
                }

                // 2) Solve XTX * beta = XTy using general LU decomposition
                // Flatten into column-major for LAPACK
                std::vector<Scalar> A(order_size * order_size);
                for (size_t i = 0; i < order_size; ++i)
                {
                    for (size_t j = 0; j < order_size; ++j)
                    {
                        A[j * order_size + i] = XTX[i][j];
                    }
                }

                // Make a copy of XTy to preserve it for dot product later
                std::vector<Scalar> bvec = XTy;

                int n = int(order_size), nrhs = 1, info;
                std::vector<lapack_int> ipiv(n);

                // Solve the system with proper error handling
                if constexpr (std::is_same_v<Scalar, double>)
                {
                    info = LAPACKE_dgesv(LAPACK_COL_MAJOR, n, nrhs,
                                         A.data(), n,
                                         ipiv.data(),
                                         bvec.data(), n);
                }
                else
                {
                    info = LAPACKE_sgesv(LAPACK_COL_MAJOR, n, nrhs,
                                         A.data(), n,
                                         ipiv.data(),
                                         bvec.data(), n);
                }
                if (info != 0)
                    throw std::runtime_error("LU solve failed (LAPACKE_dgesv/sgesv returned non-zero)");

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
          "yw_w_s_out"_a.noconvert(),
          "nthreads"_a);

    m.def("compute_t", &compute_t<float>,
          "t1"_a.noconvert(),
          "yw_w"_a.noconvert(),
          "time_shift"_a,
          "fmin"_a,
          "df"_a,
          "Nf"_a,
          "tn_out"_a.noconvert(),
          "yw_w_s_out"_a.noconvert(),
          "nthreads"_a);

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

    nb::enum_<TermType>(m, "TermType")
        .value("Sine", TermType::Sine)
        .value("Cosine", TermType::Cosine);
}
