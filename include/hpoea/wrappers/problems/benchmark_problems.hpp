#pragma once

#include "hpoea/config/config_types.hpp"
#include "hpoea/core/problem.hpp"

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace hpoea::wrappers::problems {

// shared base holding metadata, dimension, and bounds for benchmark problems
class BenchmarkProblemBase : public core::IProblem {
public:
    [[nodiscard]] const core::ProblemMetadata &metadata() const noexcept override { return metadata_; }

    [[nodiscard]] std::size_t dimension() const override { return dimension_; }

    [[nodiscard]] std::vector<double> lower_bounds() const override { return lower_bounds_; }

    [[nodiscard]] std::vector<double> upper_bounds() const override { return upper_bounds_; }

protected:
    BenchmarkProblemBase(core::ProblemMetadata metadata,
                         std::size_t dimension,
                         std::vector<double> lower_bounds,
                         std::vector<double> upper_bounds)
        : metadata_(std::move(metadata)),
          dimension_(dimension),
          lower_bounds_(std::move(lower_bounds)),
          upper_bounds_(std::move(upper_bounds)) {}

    core::ProblemMetadata metadata_{};
    std::size_t dimension_{0};
    std::vector<double> lower_bounds_{};
    std::vector<double> upper_bounds_{};
};

class SphereProblem final : public BenchmarkProblemBase {
public:
    explicit SphereProblem(std::size_t dimension, double lower_bound = -5.0, double upper_bound = 5.0);

    [[nodiscard]] double evaluate(const std::vector<double> &decision_vector) const override;
};

class RosenbrockProblem final : public BenchmarkProblemBase {
public:
    explicit RosenbrockProblem(std::size_t dimension, double lower_bound = -5.0, double upper_bound = 10.0);

    [[nodiscard]] double evaluate(const std::vector<double> &decision_vector) const override;
};

class RastriginProblem final : public BenchmarkProblemBase {
public:
    explicit RastriginProblem(std::size_t dimension, double lower_bound = -5.12, double upper_bound = 5.12);

    [[nodiscard]] double evaluate(const std::vector<double> &decision_vector) const override;
};

class AckleyProblem final : public BenchmarkProblemBase {
public:
    explicit AckleyProblem(std::size_t dimension, double lower_bound = -32.768, double upper_bound = 32.768);

    [[nodiscard]] double evaluate(const std::vector<double> &decision_vector) const override;
};

// griewank function, many local minima
class GriewankProblem final : public BenchmarkProblemBase {
public:
    explicit GriewankProblem(std::size_t dimension, double lower_bound = -600.0, double upper_bound = 600.0);

    [[nodiscard]] double evaluate(const std::vector<double> &decision_vector) const override;
};

// schwefel function, many local minima
class SchwefelProblem final : public BenchmarkProblemBase {
public:
    explicit SchwefelProblem(std::size_t dimension, double lower_bound = -500.0, double upper_bound = 500.0);

    [[nodiscard]] double evaluate(const std::vector<double> &decision_vector) const override;
};

// zakharov function with plate-shaped landscape
class ZakharovProblem final : public BenchmarkProblemBase {
public:
    explicit ZakharovProblem(std::size_t dimension, double lower_bound = -5.0, double upper_bound = 10.0);

    [[nodiscard]] double evaluate(const std::vector<double> &decision_vector) const override;
};

// styblinski-tang function
class StyblinskiTangProblem final : public BenchmarkProblemBase {
public:
    explicit StyblinskiTangProblem(std::size_t dimension, double lower_bound = -5.0, double upper_bound = 5.0);

    [[nodiscard]] double evaluate(const std::vector<double> &decision_vector) const override;
};

// 0-1 knapsack problem with continuous encoding
class KnapsackProblem final : public BenchmarkProblemBase {
public:
    KnapsackProblem(const std::vector<double> &values, const std::vector<double> &weights, double capacity);

    [[nodiscard]] double evaluate(const std::vector<double> &decision_vector) const override;

private:
    std::vector<double> values_{};
    std::vector<double> weights_{};
    double capacity_{0.0};
    double total_value_{0.0};
};

// build a benchmark problem from a config map
// unknown keys throw invalid_argument
// box problems take dimension/lower_bound/upper_bound
// knapsack takes values/weights/capacity
std::unique_ptr<core::IProblem> make_benchmark_problem(
    const std::string &problem_type,
    const config::ProblemParameterSet &parameters);

} // namespace hpoea::wrappers::problems
