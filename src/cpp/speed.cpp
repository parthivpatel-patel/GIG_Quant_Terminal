// GIG Trading Algorithm — C++ hot path (sector neutralization).
// Built as gig._speed via pybind11. Python fallback if this does not compile.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace {

bool solve_normal(std::vector<double>& a, std::vector<double>& b, int k) {
    // Gaussian elimination on k x k system stored row-major in a, rhs b.
    const double eps = 1e-12;
    for (int col = 0; col < k; ++col) {
        int pivot = col;
        double best = std::fabs(a[col * k + col]);
        for (int row = col + 1; row < k; ++row) {
            double v = std::fabs(a[row * k + col]);
            if (v > best) {
                best = v;
                pivot = row;
            }
        }
        if (best < eps) {
            return false;
        }
        if (pivot != col) {
            for (int j = 0; j < k; ++j) {
                std::swap(a[col * k + j], a[pivot * k + j]);
            }
            std::swap(b[col], b[pivot]);
        }
        double diag = a[col * k + col];
        for (int j = col; j < k; ++j) {
            a[col * k + j] /= diag;
        }
        b[col] /= diag;
        for (int row = 0; row < k; ++row) {
            if (row == col) {
                continue;
            }
            double f = a[row * k + col];
            if (f == 0.0) {
                continue;
            }
            for (int j = col; j < k; ++j) {
                a[row * k + j] -= f * a[col * k + j];
            }
            b[row] -= f * b[col];
        }
    }
    return true;
}

}  // namespace

py::array_t<double> neutralize_cs(
    py::array_t<double, py::array::c_style | py::array::forcecast> values,
    py::array_t<double, py::array::c_style | py::array::forcecast> dummy,
    py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast> mask_all,
    bool demean
) {
    auto v = values.unchecked<2>();
    auto d = dummy.unchecked<2>();
    auto m = mask_all.unchecked<2>();
    const py::ssize_t t = v.shape(0);
    const py::ssize_t n = v.shape(1);
    const py::ssize_t n_sec = d.shape(1);
    if (d.shape(0) != n || m.shape(0) != t || m.shape(1) != n) {
        throw std::runtime_error("neutralize_cs: shape mismatch");
    }

    auto out = py::array_t<double>({t, n});
    auto o = out.mutable_unchecked<2>();
    const double nan = std::nan("");
    for (py::ssize_t i = 0; i < t; ++i) {
        for (py::ssize_t j = 0; j < n; ++j) {
            o(i, j) = nan;
        }
    }

    const int k = static_cast<int>(n_sec) + 1;
    std::vector<py::ssize_t> idx;
    idx.reserve(static_cast<size_t>(n));
    std::vector<double> xtx(static_cast<size_t>(k * k));
    std::vector<double> xty(static_cast<size_t>(k));

    for (py::ssize_t i = 0; i < t; ++i) {
        idx.clear();
        for (py::ssize_t j = 0; j < n; ++j) {
            if (m(i, j) && !std::isnan(v(i, j))) {
                idx.push_back(j);
            }
        }
        const int nobs = static_cast<int>(idx.size());
        if (nobs < k + 8) {
            continue;
        }
        std::vector<double> X(static_cast<size_t>(nobs) * static_cast<size_t>(k), 0.0);
        std::vector<double> yv(static_cast<size_t>(nobs));
        for (int r = 0; r < nobs; ++r) {
            const py::ssize_t j = idx[static_cast<size_t>(r)];
            yv[static_cast<size_t>(r)] = v(i, j);
            X[static_cast<size_t>(r) * static_cast<size_t>(k)] = 1.0;
            for (int s = 0; s < n_sec; ++s) {
                X[static_cast<size_t>(r) * static_cast<size_t>(k) + static_cast<size_t>(s + 1)] = d(j, s);
            }
        }
        std::fill(xtx.begin(), xtx.end(), 0.0);
        std::fill(xty.begin(), xty.end(), 0.0);
        for (int r = 0; r < nobs; ++r) {
            for (int a = 0; a < k; ++a) {
                const double xa = X[static_cast<size_t>(r) * static_cast<size_t>(k) + static_cast<size_t>(a)];
                xty[static_cast<size_t>(a)] += xa * yv[static_cast<size_t>(r)];
                for (int b = 0; b < k; ++b) {
                    xtx[static_cast<size_t>(a) * static_cast<size_t>(k) + static_cast<size_t>(b)] +=
                        xa * X[static_cast<size_t>(r) * static_cast<size_t>(k) + static_cast<size_t>(b)];
                }
            }
        }
        if (!solve_normal(xtx, xty, k)) {
            continue;
        }
        double mean_resid = 0.0;
        std::vector<double> resid(static_cast<size_t>(nobs));
        for (int r = 0; r < nobs; ++r) {
            const py::ssize_t j = idx[static_cast<size_t>(r)];
            double pred = xty[0];
            for (int s = 0; s < n_sec; ++s) {
                pred += xty[s + 1] * d(j, s);
            }
            resid[static_cast<size_t>(r)] = v(i, j) - pred;
            mean_resid += resid[static_cast<size_t>(r)];
        }
        mean_resid /= static_cast<double>(nobs);
        for (int r = 0; r < nobs; ++r) {
            double val = resid[static_cast<size_t>(r)];
            if (demean) {
                val -= mean_resid;
            }
            o(i, idx[static_cast<size_t>(r)]) = val;
        }
    }
    return out;
}

PYBIND11_MODULE(_speed, mod) {
    mod.doc() = "GIG Trading Algorithm C++ kernels";
    mod.def(
        "neutralize_cs",
        &neutralize_cs,
        py::arg("values"),
        py::arg("dummy"),
        py::arg("mask"),
        py::arg("demean") = true
    );
}
