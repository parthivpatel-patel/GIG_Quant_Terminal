// C++20 categorical OLS projection, matching the Python sector-dummy contract.
// A constant plus mutually exclusive sector dummies spans group means even if
// masking removes the baseline group. No singular normal equations are needed.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace py = pybind11;

py::array_t<double> neutralize_cs(
    py::array_t<double, py::array::c_style | py::array::forcecast> values,
    py::array_t<double, py::array::c_style | py::array::forcecast> dummy,
    py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast> mask_all,
    bool demean
) {
    if (values.ndim() != 2 || dummy.ndim() != 2 || mask_all.ndim() != 2) {
        throw std::invalid_argument("neutralize_cs: all inputs must be matrices");
    }
    auto v = values.unchecked<2>();
    auto d = dummy.unchecked<2>();
    auto m = mask_all.unchecked<2>();
    const py::ssize_t t = v.shape(0), n = v.shape(1), n_sec = d.shape(1);
    if (d.shape(0) != n || m.shape(0) != t || m.shape(1) != n) {
        throw std::invalid_argument("neutralize_cs: shape mismatch");
    }
    if (values.size() > 20'000'000 || dummy.size() > 20'000'000 || n_sec > 4096) {
        throw std::invalid_argument("neutralize_cs: input exceeds declared work bounds");
    }
    std::vector<py::ssize_t> groups(static_cast<size_t>(n), 0);
    for (py::ssize_t j = 0; j < n; ++j) {
        for (py::ssize_t s = 0; s < n_sec; ++s) {
            if (!std::isfinite(d(j, s)) || (d(j, s) != 0.0 && d(j, s) != 1.0)) {
                throw std::invalid_argument("neutralize_cs: categorical zero/one dummy design required");
            }
            if (d(j, s) == 1.0) {
                if (groups[static_cast<size_t>(j)] != 0) {
                    throw std::invalid_argument("neutralize_cs: sector groups must be mutually exclusive");
                }
                groups[static_cast<size_t>(j)] = s + 1;
            }
        }
    }
    for (py::ssize_t i = 0; i < t; ++i) {
        for (py::ssize_t j = 0; j < n; ++j) {
            if (m(i, j) > 1) {
                throw std::invalid_argument("neutralize_cs: mask must contain only zero or one");
            }
        }
    }
    auto out = py::array_t<double>({t, n});
    auto o = out.mutable_unchecked<2>();
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const auto k = static_cast<size_t>(n_sec + 1);
    std::vector<py::ssize_t> count(k);
    std::vector<long double> anchor(k), deviation_sum(k), means(k);
    py::gil_scoped_release release;
    for (py::ssize_t i = 0; i < t; ++i) {
        std::fill(count.begin(), count.end(), 0);
        std::fill(deviation_sum.begin(), deviation_sum.end(), 0.0L);
        py::ssize_t nobs = 0;
        for (py::ssize_t j = 0; j < n; ++j) {
            o(i, j) = nan;
            if (!m(i, j) || !std::isfinite(v(i, j))) continue;
            const auto group = static_cast<size_t>(groups[static_cast<size_t>(j)]);
            if (count[group] == 0) anchor[group] = static_cast<long double>(v(i, j));
            deviation_sum[group] += static_cast<long double>(v(i, j)) - anchor[group];
            ++count[group];
            ++nobs;
        }
        // n_sec excludes the intercept: exactly the Python admission threshold.
        if (nobs < n_sec + 8) continue;
        for (size_t group = 0; group < k; ++group) {
            if (count[group]) means[group] = anchor[group] + deviation_sum[group] / count[group];
        }
        long double residual_sum = 0;
        for (py::ssize_t j = 0; j < n; ++j) {
            if (!m(i, j) || !std::isfinite(v(i, j))) continue;
            const auto group = static_cast<size_t>(groups[static_cast<size_t>(j)]);
            const double residual = static_cast<double>(static_cast<long double>(v(i, j)) - means[group]);
            if (!std::isfinite(residual)) throw std::overflow_error("neutralize_cs: residual arithmetic overflow");
            o(i, j) = residual;
            residual_sum += static_cast<long double>(residual);
        }
        if (demean) {
            const long double average = residual_sum / nobs;
            for (py::ssize_t j = 0; j < n; ++j) {
                if (!m(i, j) || !std::isfinite(v(i, j))) continue;
                const double residual = static_cast<double>(static_cast<long double>(o(i, j)) - average);
                if (!std::isfinite(residual)) throw std::overflow_error("neutralize_cs: demeaning arithmetic overflow");
                o(i, j) = residual;
            }
        }
    }
    return out;
}

PYBIND11_MODULE(_speed, mod) {
    mod.doc() = "GIG C++20 categorical sector projection; not a general dense-design OLS kernel";
    mod.def("neutralize_cs", &neutralize_cs, py::arg("values"), py::arg("dummy"),
            py::arg("mask"), py::arg("demean") = true);
}
