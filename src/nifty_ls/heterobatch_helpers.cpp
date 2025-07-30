/* Heterobatch implementation for multi-series processing
 */

#include <algorithm>
#include <complex>
#include <finufft.h>
#include <iostream>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <string>
#include <unordered_map>
#include <vector>

#include "cpu_helpers.hpp"
#include "utils_helpers.hpp"

namespace nb = nanobind;
using namespace nb::literals;

template <typename Scalar>
using nifty_arr_1d = nb::ndarray<Scalar, nb::ndim<1>, nb::device::cpu>;

template <typename Scalar>
using nifty_arr_2d = nb::ndarray<Scalar, nb::ndim<2>, nb::device::cpu>;

using utils_helpers::NormKind;

template <typename Scalar>
using Complex = std::complex<Scalar>;

template <typename Scalar>
void process_single_series(
   const Scalar *t,   // (N_d)
   const Scalar *y,   // (N_batch, N_d)
   const Scalar *dy,  // (N_batch, N_d) / (N_batch)
   const bool center_data,
   const bool fit_mean,
   const Scalar fmin,
   const Scalar df,
   const size_t Nf,
   const std::string normalization,
   int nthreads,
   Scalar *power,  // (N_batch, Nf)
   const size_t N_batch,
   const size_t N_d
) {
    std::vector<Scalar> t1_(N_d), t2_(N_d), norm_(N_batch);
    std::vector<Complex<Scalar>> yw_(N_batch * N_d), w_(N_batch * N_d),
       w2_(N_batch * N_d);
    const bool psd_norm = (normalization == "psd");

    bool broadcast_dy = (N_d == 1);

    process_finufft_inputs_raw(
       t1_.data(),
       t2_.data(),
       yw_.data(),
       w_.data(),
       w2_.data(),
       norm_.data(),
       t,   // input
       y,   // input
       dy,  // input
       broadcast_dy,
       fmin,
       df,
       Nf,
       center_data,
       fit_mean,
       psd_norm,
       nthreads,
       N_batch,
       N_d
    );
    std::cout << "Completed process_finufft_inputs_raw" << std::endl;

    // Finufft
    // Create yw_w
    std::vector<Complex<Scalar>> yw_w_(2 * N_batch * N_d);
    std::copy(yw_.begin(), yw_.end(), yw_w_.begin());
    std::copy(w_.begin(), w_.end(), yw_w_.begin() + yw_.size());

    // Set up options
    finufft_opts opts;
    finufft_default_opts(&opts);
    opts.nthreads = 1;  // Single thread for small matrix
    opts.debug    = 0;
    opts.modeord  = 1;  // Use FFT-style mode ordering (0,1,...,N/2-1,-N/2,...,-1)
    int type      = 1;  // Type-1 NUFFT
    int dim       = 1;  // 1D

    std::vector<Complex<Scalar>> f1_(N_batch * Nf), fw_(N_batch * Nf),
       f2_(N_batch * Nf);

    double eps = 1e-9;

    // Plan solo
    finufft_plan solo_plan;
    int64_t nmodes[] = {static_cast<int64_t>(Nf)};
    int ntrans       = N_batch;
    int solo_ier =
       finufft_makeplan(type, dim, nmodes, +1, ntrans, eps, &solo_plan, &opts);
    if (solo_ier != 0) {
        throw std::runtime_error(
           "finufft_makeplan(solo) failed with error code " + std::to_string(solo_ier)
        );
    }

    if (fit_mean) {
        // Plan pair
        finufft_plan pair_plan;
        int ntrans = 2 * N_batch;
        int pair_ier =
           finufft_makeplan(type, dim, nmodes, +1, ntrans, eps, &pair_plan, &opts);
        if (pair_ier != 0) {
            throw std::runtime_error(
               "finufft_makeplan(pair) failed with error code "
               + std::to_string(pair_ier)
            );
        }

        // setpts (pair)
        int setpts_ier = finufft_setpts(
           pair_plan, N_d, t1_.data(), nullptr, nullptr, 0, nullptr, nullptr, nullptr
        );
        if (setpts_ier != 0) {
            throw std::runtime_error(
               "finufft_setpts(pair) failed with error code "
               + std::to_string(setpts_ier)
            );
        }

        // execute (pair)
        std::vector<Complex<Scalar>> f1_fw(2 * N_batch * Nf);
        int exec_ier = finufft_execute(pair_plan, yw_w_.data(), f1_fw.data());
        if (exec_ier != 0) {
            throw std::runtime_error(
               "finufft_execute(pair) failed with error code "
               + std::to_string(exec_ier)
            );
        }

        // Save and clean
        std::copy(f1_fw.begin(), f1_fw.begin() + N_batch * Nf, f1_.begin());
        std::copy(f1_fw.begin() + N_batch * Nf, f1_fw.end(), fw_.begin());
        finufft_destroy(pair_plan);

    } else {
        // setpts (solo)
        int setpts_ier = finufft_setpts(
           solo_plan, N_d, t1_.data(), nullptr, nullptr, 0, nullptr, nullptr, nullptr
        );
        if (setpts_ier != 0) {
            throw std::runtime_error(
               "finufft_setpts(solo) failed with error code "
               + std::to_string(setpts_ier)
            );
        }

        // execute (solo)
        std::vector<Complex<Scalar>> f1_fw(N_batch * Nf);
        int exec_ier = finufft_execute(solo_plan, yw_w_.data(), f1_fw.data());
        if (exec_ier != 0) {
            throw std::runtime_error(
               "finufft_execute(solo) failed with error code "
               + std::to_string(exec_ier)
            );
        }

        // Save
        std::copy(f1_fw.begin(), f1_fw.begin() + N_batch * Nf, f1_.begin());
        std::copy(f1_fw.begin() + N_batch * Nf, f1_fw.end(), fw_.begin());
    }

    // second transform
    int setpts_ier = finufft_setpts(
       solo_plan, N_d, t2_.data(), nullptr, nullptr, 0, nullptr, nullptr, nullptr
    );
    if (setpts_ier != 0) {
        throw std::runtime_error(
           "finufft_setpts(solo) failed with error code " + std::to_string(setpts_ier)
        );
    }
    int exec_ier = finufft_execute(solo_plan, w2_.data(), f2_.data());
    if (exec_ier != 0) {
        throw std::runtime_error(
           "finufft_execute(second solo) failed with error code "
           + std::to_string(exec_ier)
        );
    }

    // Clean
    finufft_destroy(solo_plan);

    static const std::unordered_map<std::string, NormKind> norm_map = {
       {"standard", NormKind::Standard},
       {"model", NormKind::Model},
       {"log", NormKind::Log},
       {"psd", NormKind::PSD}
    };

    std::cout << "Start Transform" << std::endl;
    std::string norm_lower = normalization;
    std::transform(
       norm_lower.begin(), norm_lower.end(), norm_lower.begin(), [](unsigned char c) {
           return std::tolower(c);
       }
    );

    std::cout << "End Transform" << std::endl;

    NormKind norm_kind;
    try {
        norm_kind = norm_map.at(norm_lower);
    } catch (const std::out_of_range &e) {
        throw std::invalid_argument("Unknown normalization type: " + norm_lower);
    }

    std::cout << "f1_[0] is " << f1_[0] << std::endl;
    std::cout << "fw_[0] is " << fw_[0] << std::endl;

    std::cout << "Start Postprocess" << std::endl;
    process_finufft_outputs_raw(
       power,
       f1_.data(),
       fw_.data(),
       f2_.data(),
       norm_.data(),
       norm_kind,
       fit_mean,
       nthreads,
       N_batch,
       Nf
    );
    std::cout << "Completed process_finufft_outputs_raw" << std::endl;
}

template <typename Scalar>
void process_hetero_batch(
   const std::vector<nifty_arr_1d<const Scalar>> &t_lst,
   const std::vector<nifty_arr_2d<const Scalar>> &y_lst,
   const std::vector<nifty_arr_2d<const Scalar>> &dy_lst,
   const std::vector<Scalar> &fmin_lst,
   const std::vector<Scalar> &df_lst,
   const std::vector<size_t> &Nf_lst,
   std::vector<nifty_arr_2d<Scalar>> &all_powers,  // output
   const std::string &normalization,
   int nthreads,
   const bool center_data,
   const bool fit_mean,
   const bool verbose
) {
    nthreads = 1;

    const size_t N_series = t_lst.size();
    //  all_powers.reserve(N_series);

    // Check input sizes
    if (y_lst.size() != N_series || dy_lst.size() != N_series
        || fmin_lst.size() != N_series || df_lst.size() != N_series
        || Nf_lst.size() != N_series) {
        throw std::runtime_error("All input lists must have same length");
    }

    // Start OMP
    for (size_t i = 0; i < N_series; ++i) {

        // Data for single series
        const auto &t_i  = t_lst[i];
        const auto &y_i  = y_lst[i];
        const auto &dy_i = dy_lst[i];
        auto &power      = all_powers[i];

        size_t N_d     = t_i.shape(0);
        size_t N_batch = y_i.shape(0);

        std::cout << "Before single process " << i << std::endl;

        process_single_series(
           t_i.data(),
           y_i.data(),
           dy_i.data(),
           center_data,
           fit_mean,
           fmin_lst[i],
           df_lst[i],
           Nf_lst[i],
           normalization,
           nthreads,  // Set Nthread to 1
           power.data(),
           N_batch,
           N_d
        );
        std::cout << "Completed process_single_series for series " << i << std::endl;

        if (verbose) {
            std::cout << "Processed series " << i << " (" << N_batch << "," << N_d
                      << ")\n";
        }
    }
}

// TDOO: Add float precision
NB_MODULE(heterobatch_helpers, m) {
    m.def(
       "process_hetero_batch",
       &process_hetero_batch<double>,
       "t_lst"_a,
       "y_lst"_a,
       "dy_lst"_a,
       "fmin_lst"_a,
       "df_lst"_a,
       "Nf_lst"_a,
       "all_powers"_a,
       "normalization"_a,
       "nthreads"_a,
       "center_data"_a,
       "fit_mean"_a,
       "verbose"_a
    );
}