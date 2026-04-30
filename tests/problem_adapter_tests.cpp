#include "test_harness.hpp"

#include "hpoea/core/problem.hpp"
#include "hpoea/wrappers/problems/benchmark_problems.hpp"
#include "problem_adapter.hpp"

#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

class NaNProblem final : public hpoea::core::IProblem {
public:
    NaNProblem() {
        metadata_.id = "nan";
        metadata_.family = "tests";
        metadata_.description = "returns NaN";
    }

    [[nodiscard]] const hpoea::core::ProblemMetadata &metadata() const noexcept override { return metadata_; }
    [[nodiscard]] std::size_t dimension() const override { return 1; }
    [[nodiscard]] std::vector<double> lower_bounds() const override { return {0.0}; }
    [[nodiscard]] std::vector<double> upper_bounds() const override { return {1.0}; }
    [[nodiscard]] double evaluate(const std::vector<double> &) const override {
        return std::numeric_limits<double>::quiet_NaN();
    }

private:
    hpoea::core::ProblemMetadata metadata_{};
};

class MismatchedBoundsProblem final : public hpoea::core::IProblem {
public:
    MismatchedBoundsProblem() {
        metadata_.id = "mismatched_bounds";
        metadata_.family = "tests";
        metadata_.description = "returns mismatched bounds";
    }

    [[nodiscard]] const hpoea::core::ProblemMetadata &metadata() const noexcept override { return metadata_; }
    [[nodiscard]] std::size_t dimension() const override { return 2; }
    [[nodiscard]] std::vector<double> lower_bounds() const override { return {0.0}; }
    [[nodiscard]] std::vector<double> upper_bounds() const override { return {1.0, 2.0}; }
    [[nodiscard]] double evaluate(const std::vector<double> &) const override { return 0.0; }

private:
    hpoea::core::ProblemMetadata metadata_{};
};

class WrongDimensionBoundsProblem final : public hpoea::core::IProblem {
public:
    WrongDimensionBoundsProblem() {
        metadata_.id = "wrong_dimension_bounds";
        metadata_.family = "tests";
        metadata_.description = "returns same-sized bounds with wrong dimension";
    }

    [[nodiscard]] const hpoea::core::ProblemMetadata &metadata() const noexcept override { return metadata_; }
    [[nodiscard]] std::size_t dimension() const override { return 3; }
    [[nodiscard]] std::vector<double> lower_bounds() const override { return {0.0, 0.0}; }
    [[nodiscard]] std::vector<double> upper_bounds() const override { return {1.0, 1.0}; }
    [[nodiscard]] double evaluate(const std::vector<double> &) const override { return 0.0; }

private:
    hpoea::core::ProblemMetadata metadata_{};
};

class InvertedBoundsProblem final : public hpoea::core::IProblem {
public:
    InvertedBoundsProblem() {
        metadata_.id = "inverted_bounds";
        metadata_.family = "tests";
        metadata_.description = "returns lower bound greater than upper bound";
    }

    [[nodiscard]] const hpoea::core::ProblemMetadata &metadata() const noexcept override { return metadata_; }
    [[nodiscard]] std::size_t dimension() const override { return 2; }
    [[nodiscard]] std::vector<double> lower_bounds() const override { return {0.0, 2.0}; }
    [[nodiscard]] std::vector<double> upper_bounds() const override { return {1.0, 1.0}; }
    [[nodiscard]] double evaluate(const std::vector<double> &) const override { return 0.0; }

private:
    hpoea::core::ProblemMetadata metadata_{};
};

class ThrowingProblem final : public hpoea::core::IProblem {
public:
    ThrowingProblem() {
        metadata_.id = "throwing";
        metadata_.family = "tests";
        metadata_.description = "throws";
    }

    [[nodiscard]] const hpoea::core::ProblemMetadata &metadata() const noexcept override { return metadata_; }
    [[nodiscard]] std::size_t dimension() const override { return 1; }
    [[nodiscard]] std::vector<double> lower_bounds() const override { return {0.0}; }
    [[nodiscard]] std::vector<double> upper_bounds() const override { return {1.0}; }
    [[nodiscard]] double evaluate(const std::vector<double> &) const override {
        throw std::runtime_error("boom");
    }

private:
    hpoea::core::ProblemMetadata metadata_{};
};

}

int main() {
    hpoea::tests_v2::TestRunner runner;

    {
        hpoea::pagmo_wrappers::ProblemAdapter adapter;
        bool threw = false;
        try {
            (void)adapter.get_name();
        } catch (const std::exception &) {
            threw = true;
        }
        HPOEA_V2_CHECK(runner, threw, "default adapter throws without problem");
    }

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
        MismatchedBoundsProblem problem;
        hpoea::pagmo_wrappers::ProblemAdapter adapter(problem);
        bool threw = false;
        try {
            (void)adapter.get_bounds();
        } catch (const std::invalid_argument &) {
            threw = true;
        }
        HPOEA_V2_CHECK(runner, threw, "adapter rejects lower/upper bounds size mismatch");
    }

    {
        WrongDimensionBoundsProblem problem;
        hpoea::pagmo_wrappers::ProblemAdapter adapter(problem);
        bool threw = false;
        try {
            (void)adapter.get_bounds();
        } catch (const std::invalid_argument &) {
            threw = true;
        }
        HPOEA_V2_CHECK(runner, threw, "adapter rejects bounds dimension mismatch");
    }

    {
        InvertedBoundsProblem problem;
        hpoea::pagmo_wrappers::ProblemAdapter adapter(problem);
        bool threw = false;
        try {
            (void)adapter.get_bounds();
        } catch (const std::invalid_argument &) {
            threw = true;
        }
        HPOEA_V2_CHECK(runner, threw, "adapter rejects inverted bounds");
    }

    {
        NaNProblem nan_problem;
        hpoea::pagmo_wrappers::ProblemAdapter adapter(nan_problem);
        bool threw = false;
        try {
            (void)adapter.fitness({0.5});
        } catch (const hpoea::core::EvaluationFailure &) {
            threw = true;
        }
        HPOEA_V2_CHECK(runner, threw, "adapter converts NaN to EvaluationFailure");
    }

    {
        class InfProblem final : public hpoea::core::IProblem {
        public:
            InfProblem() {
                metadata_.id = "inf";
                metadata_.family = "tests";
                metadata_.description = "returns inf";
            }

            [[nodiscard]] const hpoea::core::ProblemMetadata &metadata() const noexcept override { return metadata_; }
            [[nodiscard]] std::size_t dimension() const override { return 1; }
            [[nodiscard]] std::vector<double> lower_bounds() const override { return {0.0}; }
            [[nodiscard]] std::vector<double> upper_bounds() const override { return {1.0}; }
            [[nodiscard]] double evaluate(const std::vector<double> &) const override {
                return std::numeric_limits<double>::infinity();
            }

        private:
            hpoea::core::ProblemMetadata metadata_{};
        } inf_problem;

        hpoea::pagmo_wrappers::ProblemAdapter adapter(inf_problem);
        bool threw = false;
        try {
            (void)adapter.fitness({0.5});
        } catch (const hpoea::core::EvaluationFailure &) {
            threw = true;
        }
        HPOEA_V2_CHECK(runner, threw, "adapter converts inf to EvaluationFailure");
    }

    {
        class NegInfProblem final : public hpoea::core::IProblem {
        public:
            NegInfProblem() {
                metadata_.id = "neginf";
                metadata_.family = "tests";
                metadata_.description = "returns -inf";
            }

            [[nodiscard]] const hpoea::core::ProblemMetadata &metadata() const noexcept override { return metadata_; }
            [[nodiscard]] std::size_t dimension() const override { return 1; }
            [[nodiscard]] std::vector<double> lower_bounds() const override { return {0.0}; }
            [[nodiscard]] std::vector<double> upper_bounds() const override { return {1.0}; }
            [[nodiscard]] double evaluate(const std::vector<double> &) const override {
                return -std::numeric_limits<double>::infinity();
            }

        private:
            hpoea::core::ProblemMetadata metadata_{};
        } neginf_problem;

        hpoea::pagmo_wrappers::ProblemAdapter adapter(neginf_problem);
        bool threw = false;
        try {
            (void)adapter.fitness({0.5});
        } catch (const hpoea::core::EvaluationFailure &) {
            threw = true;
        }
        HPOEA_V2_CHECK(runner, threw, "adapter converts -inf to EvaluationFailure");
    }

    {
        ThrowingProblem throwing_problem;
        hpoea::pagmo_wrappers::ProblemAdapter adapter(throwing_problem);
        bool threw = false;
        try {
            (void)adapter.fitness({0.5});
        } catch (const hpoea::core::EvaluationFailure &) {
            threw = true;
        }
        HPOEA_V2_CHECK(runner, threw, "adapter converts exceptions to EvaluationFailure");
    }

    return runner.summarize("problem_adapter_tests");
}
