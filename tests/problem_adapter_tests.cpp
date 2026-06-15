#include "test_harness.hpp"

#include "hpoea/core/problem.hpp"
#include "hpoea/wrappers/problems/benchmark_problems.hpp"
#include "problem_adapter.hpp"

#include <atomic>
#include <limits>
#include <memory>
#include <pagmo/population.hpp>
#include <pagmo/problem.hpp>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// one problem whose bounds are whatever the case supplies
// covers size/dimension/order defects
class BadBoundsProblem final : public hpoea::core::IProblem {
public:
    BadBoundsProblem(std::size_t dimension, std::vector<double> lower, std::vector<double> upper)
        : dimension_(dimension), lower_(std::move(lower)), upper_(std::move(upper)) {
        metadata_.id = "bad_bounds";
        metadata_.family = "tests";
    }

    [[nodiscard]] const hpoea::core::ProblemMetadata &metadata() const noexcept override { return metadata_; }
    [[nodiscard]] std::size_t dimension() const override { return dimension_; }
    [[nodiscard]] std::vector<double> lower_bounds() const override { return lower_; }
    [[nodiscard]] std::vector<double> upper_bounds() const override { return upper_; }
    [[nodiscard]] double evaluate(const std::vector<double> &) const override { return 0.0; }

private:
    std::size_t dimension_;
    std::vector<double> lower_;
    std::vector<double> upper_;
    hpoea::core::ProblemMetadata metadata_{};
};

// one problem that returns a fixed value or throws on evaluate
// the fixed value may be non-finite
class ConstantProblem final : public hpoea::core::IProblem {
public:
    ConstantProblem(double value, bool throws) : value_(value), throws_(throws) {
        metadata_.id = "constant";
        metadata_.family = "tests";
    }

    [[nodiscard]] const hpoea::core::ProblemMetadata &metadata() const noexcept override { return metadata_; }
    [[nodiscard]] std::size_t dimension() const override { return 1; }
    [[nodiscard]] std::vector<double> lower_bounds() const override { return {0.0}; }
    [[nodiscard]] std::vector<double> upper_bounds() const override { return {1.0}; }
    [[nodiscard]] double evaluate(const std::vector<double> &) const override {
        if (throws_) {
            throw std::runtime_error("boom");
        }
        return value_;
    }

private:
    double value_;
    bool throws_;
    hpoea::core::ProblemMetadata metadata_{};
};

}

int main() {
    hpoea::tests_v2::TestRunner runner;

    {
        hpoea::wrappers::problems::SphereProblem sphere(2);
        hpoea::pagmo_wrappers::ProblemAdapter adapter(sphere);
        auto bounds = adapter.get_bounds();
        HPOEA_V2_CHECK(runner, bounds.first.size() == 2 && bounds.second.size() == 2,
                       "adapter get_bounds matches problem dimension");
        auto fitness = adapter.fitness({0.0, 0.0});
        HPOEA_V2_CHECK(runner, fitness.size() == 1 && fitness[0] == 0.0,
                       "adapter fitness returns objective value");
        HPOEA_V2_CHECK(runner, adapter.get_name() == "sphere",
                       "adapter name matches problem id");
    }

    {
        struct BadBoundsCase {
            std::size_t dimension;
            std::vector<double> lower;
            std::vector<double> upper;
            const char *label;
        };
        const std::vector<BadBoundsCase> bad_bounds_cases = {
            {2, {0.0}, {1.0, 2.0}, "adapter rejects lower/upper bounds size mismatch"},
            {3, {0.0, 0.0}, {1.0, 1.0}, "adapter rejects bounds dimension mismatch"},
            {2, {0.0, 2.0}, {1.0, 1.0}, "adapter rejects inverted bounds"},
        };
        for (const auto &c : bad_bounds_cases) {
            BadBoundsProblem problem(c.dimension, c.lower, c.upper);
            hpoea::pagmo_wrappers::ProblemAdapter adapter(problem);
            bool threw = false;
            try {
                (void)adapter.get_bounds();
            } catch (const std::invalid_argument &) {
                threw = true;
            }
            HPOEA_V2_CHECK(runner, threw, c.label);
        }
    }

    {
        struct FailCase { double value; bool throws; const char *label; };
        const std::vector<FailCase> fail_cases = {
            {std::numeric_limits<double>::quiet_NaN(), false, "adapter converts NaN to EvaluationFailure"},
            {std::numeric_limits<double>::infinity(), false, "adapter converts inf to EvaluationFailure"},
            {-std::numeric_limits<double>::infinity(), false, "adapter converts -inf to EvaluationFailure"},
            {0.0, true, "adapter converts exceptions to EvaluationFailure"},
        };
        for (const auto &c : fail_cases) {
            ConstantProblem problem(c.value, c.throws);
            hpoea::pagmo_wrappers::ProblemAdapter adapter(problem);
            bool threw = false;
            try {
                (void)adapter.fitness({0.5});
            } catch (const hpoea::core::EvaluationFailure &) {
                threw = true;
            }
            HPOEA_V2_CHECK(runner, threw, c.label);
        }
    }

    {
        auto counter = std::make_shared<std::atomic<std::size_t>>(0);
        hpoea::wrappers::problems::SphereProblem sphere(2);
        hpoea::pagmo_wrappers::ProblemAdapter adapter(sphere, counter);
        auto copy = adapter;
        (void)copy.fitness({0.1, 0.2});
        (void)copy.fitness({0.3, 0.4});
        (void)adapter.fitness({0.5, 0.6});
        HPOEA_V2_CHECK(runner, counter->load() == 3u,
                       "ProblemAdapter copies share one evaluation counter");
    }

    {
        auto counter = std::make_shared<std::atomic<std::size_t>>(0);
        hpoea::wrappers::problems::SphereProblem sphere(2);
        pagmo::problem pg{hpoea::pagmo_wrappers::ProblemAdapter{sphere, counter}};
        pagmo::population pop{pg, 7u, 42u};
        HPOEA_V2_CHECK(runner, counter->load() == 7u,
                       "ProblemAdapter counts every initial-population evaluation");
        HPOEA_V2_CHECK(runner, counter->load() == static_cast<std::size_t>(pop.get_problem().get_fevals()),
                       "ProblemAdapter counter equals pagmo get_fevals on the success path");
    }

    return runner.summarize("problem_adapter_tests");
}
