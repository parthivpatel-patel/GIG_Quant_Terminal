// C++20 ARM64 executable reference; run by explicit QEMU user mode, never binfmt.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

#ifndef __aarch64__
#error "This fixture must be compiled for aarch64, not the host architecture"
#endif

extern "C" int gig_rolling_stats(const double*, std::size_t, std::size_t,
                                 std::size_t, double*, double*) noexcept;

int main() {
    std::mt19937_64 generator(20261101);
    std::uniform_real_distribution<double> distribution(-10.0, 10.0);
    std::vector<double> input(512), means(512), deviations(512);
    for (double& value : input) value = distribution(generator);
    double max_mean_error = 0.0, max_std_error = 0.0;
    std::size_t comparisons = 0;
    for (const std::size_t window : {1, 2, 7, 64, 252}) {
        for (std::size_t ddof = 0; ddof <= 1 && ddof < window; ++ddof) {
            if (gig_rolling_stats(input.data(), input.size(), window, ddof,
                                  means.data(), deviations.data()) != 0) return 10;
            for (std::size_t i = 0; i < input.size(); ++i) {
                if (i + 1 < window) {
                    if (!std::isnan(means[i]) || !std::isnan(deviations[i])) return 11;
                    continue;
                }
                // Independent two-pass scalar reference, not the kernel's
                // running add/delete update or periodic-rebase implementation.
                long double mean = 0.0L;
                for (std::size_t j = i + 1 - window; j <= i; ++j)
                    mean += static_cast<long double>(input[j]) / window;
                long double squares = 0.0L;
                for (std::size_t j = i + 1 - window; j <= i; ++j) {
                    const long double difference = input[j] - mean;
                    squares += difference * difference;
                }
                const double sd = std::sqrt(static_cast<double>(squares / (window - ddof)));
                const double mean_error = std::abs(means[i] - static_cast<double>(mean));
                const double std_error = std::abs(deviations[i] - sd);
                max_mean_error = std::max(max_mean_error, mean_error);
                max_std_error = std::max(max_std_error, std_error);
                if (mean_error > 2e-11 || std_error > 2e-10) return 12;
                ++comparisons;
            }
        }
    }
    if (gig_rolling_stats(input.data(), 512, 0, 0, means.data(), deviations.data()) != 1)
        return 13;
    input[0] = std::numeric_limits<double>::quiet_NaN();
    if (gig_rolling_stats(input.data(), 512, 7, 1, means.data(), deviations.data()) != 2)
        return 14;
    input[0] = std::numeric_limits<double>::infinity();
    if (gig_rolling_stats(input.data(), 512, 7, 1, means.data(), deviations.data()) != 2)
        return 15;
    input[0] = 1e151;
    if (gig_rolling_stats(input.data(), 512, 7, 1, means.data(), deviations.data()) != 2)
        return 16;
    std::cout.precision(17);
    std::cout << "{\"architecture\":\"aarch64\",\"result\":\"PASS\","
              << "\"comparisons\":" << comparisons << ","
              << "\"max_mean_abs_error\":" << max_mean_error << ","
              << "\"max_std_abs_error\":" << max_std_error << ","
              << "\"input_admission_cases\":4}" << std::endl;
}
