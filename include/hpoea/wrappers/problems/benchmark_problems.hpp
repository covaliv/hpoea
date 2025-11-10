#pragma once

#include "hpoea/core/problem.hpp"

#include <cmath>
#include <vector>

namespace hpoea::wrappers::problems {

class SphereProblem final : public core::IProblem {
public:
    explicit SphereProblem(std::size_t dimension, double lower_bound = -5.0, double upper_bound = 5.0);

    [[nodiscard]] const core::ProblemMetadata &metadata() const noexcept override { return metadata_; }

    [[nodiscard]] std::size_t dimension() const override { return dimension_; }

    [[nodiscard]] std::vector<double> lower_bounds() const override { return lower_bounds_; }

    [[nodiscard]] std::vector<double> upper_bounds() const override { return upper_bounds_; }

    [[nodiscard]] double evaluate(const std::vector<double> &decision_vector) const override;

private:
    core::ProblemMetadata metadata_{};
    std::size_t dimension_{0};
    std::vector<double> lower_bounds_{};
    std::vector<double> upper_bounds_{};
};

class RosenbrockProblem final : public core::IProblem {
public:
    explicit RosenbrockProblem(std::size_t dimension, double lower_bound = -5.0, double upper_bound = 10.0);

    [[nodiscard]] const core::ProblemMetadata &metadata() const noexcept override { return metadata_; }

    [[nodiscard]] std::size_t dimension() const override { return dimension_; }

    [[nodiscard]] std::vector<double> lower_bounds() const override { return lower_bounds_; }

    [[nodiscard]] std::vector<double> upper_bounds() const override { return upper_bounds_; }

    [[nodiscard]] double evaluate(const std::vector<double> &decision_vector) const override;

private:
    core::ProblemMetadata metadata_{};
    std::size_t dimension_{0};
    std::vector<double> lower_bounds_{};
    std::vector<double> upper_bounds_{};
};

class RastriginProblem final : public core::IProblem {
public:
    explicit RastriginProblem(std::size_t dimension, double lower_bound = -5.12, double upper_bound = 5.12);

    [[nodiscard]] const core::ProblemMetadata &metadata() const noexcept override { return metadata_; }

    [[nodiscard]] std::size_t dimension() const override { return dimension_; }

    [[nodiscard]] std::vector<double> lower_bounds() const override { return lower_bounds_; }

    [[nodiscard]] std::vector<double> upper_bounds() const override { return upper_bounds_; }

    [[nodiscard]] double evaluate(const std::vector<double> &decision_vector) const override;

private:
    core::ProblemMetadata metadata_{};
    std::size_t dimension_{0};
    std::vector<double> lower_bounds_{};
    std::vector<double> upper_bounds_{};
};

class AckleyProblem final : public core::IProblem {
public:
    explicit AckleyProblem(std::size_t dimension, double lower_bound = -32.768, double upper_bound = 32.768);

    [[nodiscard]] const core::ProblemMetadata &metadata() const noexcept override { return metadata_; }

    [[nodiscard]] std::size_t dimension() const override { return dimension_; }

    [[nodiscard]] std::vector<double> lower_bounds() const override { return lower_bounds_; }

    [[nodiscard]] std::vector<double> upper_bounds() const override { return upper_bounds_; }

    [[nodiscard]] double evaluate(const std::vector<double> &decision_vector) const override;

private:
    core::ProblemMetadata metadata_{};
    std::size_t dimension_{0};
    std::vector<double> lower_bounds_{};
    std::vector<double> upper_bounds_{};
};

// griewank function with many local minima
class GriewankProblem final : public core::IProblem {
public:
    explicit GriewankProblem(std::size_t dimension, double lower_bound = -600.0, double upper_bound = 600.0);

    [[nodiscard]] const core::ProblemMetadata &metadata() const noexcept override { return metadata_; }

    [[nodiscard]] std::size_t dimension() const override { return dimension_; }

    [[nodiscard]] std::vector<double> lower_bounds() const override { return lower_bounds_; }

    [[nodiscard]] std::vector<double> upper_bounds() const override { return upper_bounds_; }

    [[nodiscard]] double evaluate(const std::vector<double> &decision_vector) const override;

private:
    core::ProblemMetadata metadata_{};
    std::size_t dimension_{0};
    std::vector<double> lower_bounds_{};
    std::vector<double> upper_bounds_{};
};

// schwefel function with many local minima
class SchwefelProblem final : public core::IProblem {
public:
    explicit SchwefelProblem(std::size_t dimension, double lower_bound = -500.0, double upper_bound = 500.0);

    [[nodiscard]] const core::ProblemMetadata &metadata() const noexcept override { return metadata_; }

    [[nodiscard]] std::size_t dimension() const override { return dimension_; }

    [[nodiscard]] std::vector<double> lower_bounds() const override { return lower_bounds_; }

    [[nodiscard]] std::vector<double> upper_bounds() const override { return upper_bounds_; }

    [[nodiscard]] double evaluate(const std::vector<double> &decision_vector) const override;

private:
    core::ProblemMetadata metadata_{};
    std::size_t dimension_{0};
    std::vector<double> lower_bounds_{};
    std::vector<double> upper_bounds_{};
};

// zakharov function with plate-shaped landscape
class ZakharovProblem final : public core::IProblem {
public:
    explicit ZakharovProblem(std::size_t dimension, double lower_bound = -5.0, double upper_bound = 10.0);

    [[nodiscard]] const core::ProblemMetadata &metadata() const noexcept override { return metadata_; }

    [[nodiscard]] std::size_t dimension() const override { return dimension_; }

    [[nodiscard]] std::vector<double> lower_bounds() const override { return lower_bounds_; }

    [[nodiscard]] std::vector<double> upper_bounds() const override { return upper_bounds_; }

    [[nodiscard]] double evaluate(const std::vector<double> &decision_vector) const override;

private:
    core::ProblemMetadata metadata_{};
    std::size_t dimension_{0};
    std::vector<double> lower_bounds_{};
    std::vector<double> upper_bounds_{};
};

// styblinski-tang function
class StyblinskiTangProblem final : public core::IProblem {
public:
    explicit StyblinskiTangProblem(std::size_t dimension, double lower_bound = -5.0, double upper_bound = 5.0);

    [[nodiscard]] const core::ProblemMetadata &metadata() const noexcept override { return metadata_; }

    [[nodiscard]] std::size_t dimension() const override { return dimension_; }

    [[nodiscard]] std::vector<double> lower_bounds() const override { return lower_bounds_; }

    [[nodiscard]] std::vector<double> upper_bounds() const override { return upper_bounds_; }

    [[nodiscard]] double evaluate(const std::vector<double> &decision_vector) const override;

private:
    core::ProblemMetadata metadata_{};
    std::size_t dimension_{0};
    std::vector<double> lower_bounds_{};
    std::vector<double> upper_bounds_{};
};

// 0-1 knapsack problem with continuous encoding
class KnapsackProblem final : public core::IProblem {
public:
    KnapsackProblem(const std::vector<double> &values, const std::vector<double> &weights, double capacity);

    [[nodiscard]] const core::ProblemMetadata &metadata() const noexcept override { return metadata_; }

    [[nodiscard]] std::size_t dimension() const override { return dimension_; }

    [[nodiscard]] std::vector<double> lower_bounds() const override { return lower_bounds_; }

    [[nodiscard]] std::vector<double> upper_bounds() const override { return upper_bounds_; }

    [[nodiscard]] double evaluate(const std::vector<double> &decision_vector) const override;

private:
    core::ProblemMetadata metadata_{};
    std::size_t dimension_{0};
    std::vector<double> values_{};
    std::vector<double> weights_{};
    double capacity_{0.0};
    std::vector<double> lower_bounds_{};
    std::vector<double> upper_bounds_{};
};

} // namespace hpoea::wrappers::problems

