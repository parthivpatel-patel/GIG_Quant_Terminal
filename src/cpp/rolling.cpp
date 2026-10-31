// Optional standalone C++20 kernel. C ABI permits use without pybind11.
// All work happens in caller-owned buffers: no per-call heap allocation.
#include <cmath>
#include <cstddef>

extern "C" int gig_rolling_stats(const double* input, std::size_t n,
                                 std::size_t window, std::size_t ddof,
                                 double* means, double* deviations) noexcept {
    if (!input || !means || !deviations || !window || window > n || ddof >= window)
        return 1;
    for (std::size_t i = 0; i < n; ++i)
        if (!std::isfinite(input[i]) || std::abs(input[i]) > 1e150) return 2;
    long double mean = 0, m2 = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const long double x = input[i];
        if (i >= window) {
            if (window > 1) {
                const long double old = input[i-window];
                const long double next = mean + (mean-old) / (window-1);
                m2 -= (old-mean) * (old-next);
                mean = next;
            } else { mean = 0; m2 = 0; }
        }
        const std::size_t count = i < window ? i+1 : window;
        const long double delta = x-mean;
        mean += delta/count;
        m2 += delta*(x-mean);
        // Bound accumulated deletion error with a periodic stable rebase.
        if (i >= window-1 && (i % 1024 == 0 || m2 < 0)) {
            mean = 0; m2 = 0;
            for (std::size_t j = 0; j < window; ++j) {
                const long double d = static_cast<long double>(input[i-window+1+j])-mean;
                mean += d/(j+1);
                m2 += d*(static_cast<long double>(input[i-window+1+j])-mean);
            }
        }
        if (i < window-1) { means[i] = NAN; deviations[i] = NAN; }
        else {
            means[i] = static_cast<double>(mean);
            deviations[i] = std::sqrt(static_cast<double>(m2/(window-ddof)));
            if (!std::isfinite(means[i]) || !std::isfinite(deviations[i])) return 3;
        }
    }
    return 0;
}
