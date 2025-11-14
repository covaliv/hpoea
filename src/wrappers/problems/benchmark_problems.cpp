#include "hpoea/wrappers/problems/benchmark_problems.hpp"

#include "hpoea/core/problem.hpp"

#include <cmath>
#include <numbers>
#include <stdexcept>

namespace hpoea::wrappers::problems {

SphereProblem::SphereProblem(std::size_t dimension, double lower_bound, double upper_bound) {
    metadata_.id = "sphere";
    metadata_.family = "benchmark";
    metadata_.description = "Sphere function (unimodal, separable)";
    dimension_ = dimension;
    lower_bounds_.assign(dimension_, lower_bound);
    upper_bounds_.assign(dimension_, upper_bound);
}

double SphereProblem::evaluate(const std::vector<double> &decision_vector) const {
    if (decision_vector.size() != dimension_) {
        throw std::runtime_error("Decision vector dimension mismatch");
    }
    double sum = 0.0;
    for (const auto value : decision_vector) {
        sum += value * value;
    }
    return sum;
}

RosenbrockProblem::RosenbrockProblem(std::size_t dimension, double lower_bound, double upper_bound) {
    metadata_.id = "rosenbrock";
    metadata_.family = "benchmark";
    metadata_.description = "Rosenbrock function (unimodal, non-separable)";
    dimension_ = dimension;
    lower_bounds_.assign(dimension_, lower_bound);
    upper_bounds_.assign(dimension_, upper_bound);
}

double RosenbrockProblem::evaluate(const std::vector<double> &decision_vector) const {
    if (decision_vector.size() != dimension_) {
        throw std::runtime_error("Decision vector dimension mismatch");
    }
    double sum = 0.0;
    for (std::size_t i = 0; i + 1 < decision_vector.size(); ++i) {
        const double xi = decision_vector[i];
        const double xnext = decision_vector[i + 1];
        const double term1 = 100.0 * std::pow(xnext - xi * xi, 2);
        const double term2 = std::pow(1.0 - xi, 2);
        sum += term1 + term2;
    }
    return sum;
}

RastriginProblem::RastriginProblem(std::size_t dimension, double lower_bound, double upper_bound) {
    metadata_.id = "rastrigin";
    metadata_.family = "benchmark";
    metadata_.description = "Rastrigin function (multimodal, separable)";
    dimension_ = dimension;
    lower_bounds_.assign(dimension_, lower_bound);
    upper_bounds_.assign(dimension_, upper_bound);
}

double RastriginProblem::evaluate(const std::vector<double> &decision_vector) const {
    if (decision_vector.size() != dimension_) {
        throw std::runtime_error("Decision vector dimension mismatch");
    }
    constexpr double A = 10.0;
    double sum = A * static_cast<double>(dimension_);
    for (const auto value : decision_vector) {
        sum += value * value - A * std::cos(2.0 * std::numbers::pi * value);
    }
    return sum;
}

AckleyProblem::AckleyProblem(std::size_t dimension, double lower_bound, double upper_bound) {
    metadata_.id = "ackley";
    metadata_.family = "benchmark";
    metadata_.description = "Ackley function (multimodal, non-separable)";
    dimension_ = dimension;
    lower_bounds_.assign(dimension_, lower_bound);
    upper_bounds_.assign(dimension_, upper_bound);
}

double AckleyProblem::evaluate(const std::vector<double> &decision_vector) const {
    if (decision_vector.size() != dimension_) {
        throw std::runtime_error("Decision vector dimension mismatch");
    }
    constexpr double a = 20.0;
    constexpr double b = 0.2;
    constexpr double c = 2.0 * std::numbers::pi;

    double sum1 = 0.0;
    double sum2 = 0.0;
    for (const auto value : decision_vector) {
        sum1 += value * value;
        sum2 += std::cos(c * value);
    }

    const double n = static_cast<double>(dimension_);
    const double term1 = -a * std::exp(-b * std::sqrt(sum1 / n));
    const double term2 = -std::exp(sum2 / n);

    return term1 + term2 + a + std::numbers::e;
}

GriewankProblem::GriewankProblem(std::size_t dimension, double lower_bound, double upper_bound) {
    metadata_.id = "griewank";
    metadata_.family = "benchmark";
    metadata_.description = "Griewank function (multimodal, many local minima)";
    dimension_ = dimension;
    lower_bounds_.assign(dimension_, lower_bound);
    upper_bounds_.assign(dimension_, upper_bound);
}

double GriewankProblem::evaluate(const std::vector<double> &decision_vector) const {
    if (decision_vector.size() != dimension_) {
        throw std::runtime_error("Decision vector dimension mismatch");
    }
    double sum = 0.0;
    double product = 1.0;
    
    for (std::size_t i = 0; i < decision_vector.size(); ++i) {
        const double xi = decision_vector[i];
        sum += xi * xi / 4000.0;
        product *= std::cos(xi / std::sqrt(static_cast<double>(i + 1)));
    }
    
    return sum - product + 1.0;
}

SchwefelProblem::SchwefelProblem(std::size_t dimension, double lower_bound, double upper_bound) {
    metadata_.id = "schwefel";
    metadata_.family = "benchmark";
    metadata_.description = "Schwefel function (multimodal, deceptive landscape)";
    dimension_ = dimension;
    lower_bounds_.assign(dimension_, lower_bound);
    upper_bounds_.assign(dimension_, upper_bound);
}

double SchwefelProblem::evaluate(const std::vector<double> &decision_vector) const {
    if (decision_vector.size() != dimension_) {
        throw std::runtime_error("Decision vector dimension mismatch");
    }
    constexpr double alpha = 418.9828872724339; // constant for global minimum calculation
    double sum = 0.0;
    
    for (const auto value : decision_vector) {
        sum += -value * std::sin(std::sqrt(std::abs(value)));
    }
    
    return alpha * static_cast<double>(dimension_) + sum;
}

ZakharovProblem::ZakharovProblem(std::size_t dimension, double lower_bound, double upper_bound) {
    metadata_.id = "zakharov";
    metadata_.family = "benchmark";
    metadata_.description = "Zakharov function (unimodal, plate-shaped)";
    dimension_ = dimension;
    lower_bounds_.assign(dimension_, lower_bound);
    upper_bounds_.assign(dimension_, upper_bound);
}

double ZakharovProblem::evaluate(const std::vector<double> &decision_vector) const {
    if (decision_vector.size() != dimension_) {
        throw std::runtime_error("Decision vector dimension mismatch");
    }
    double sum1 = 0.0;
    double sum2 = 0.0;
    
    for (std::size_t i = 0; i < decision_vector.size(); ++i) {
        const double xi = decision_vector[i];
        sum1 += xi * xi;
        sum2 += 0.5 * static_cast<double>(i + 1) * xi;
    }
    
    return sum1 + sum2 * sum2 + std::pow(sum2, 4);
}

StyblinskiTangProblem::StyblinskiTangProblem(std::size_t dimension, double lower_bound, double upper_bound) {
    metadata_.id = "styblinski_tang";
    metadata_.family = "benchmark";
    metadata_.description = "Styblinski-Tang function (multimodal)";
    dimension_ = dimension;
    lower_bounds_.assign(dimension_, lower_bound);
    upper_bounds_.assign(dimension_, upper_bound);
}

double StyblinskiTangProblem::evaluate(const std::vector<double> &decision_vector) const {
    if (decision_vector.size() != dimension_) {
        throw std::runtime_error("Decision vector dimension mismatch");
    }
    double sum = 0.0;
    
    for (const auto value : decision_vector) {
        const double x4 = value * value * value * value;
        const double x2 = value * value;
        sum += (x4 - 16.0 * x2 + 5.0 * value) / 2.0;
    }
    
    return sum;
}

KnapsackProblem::KnapsackProblem(const std::vector<double> &values, const std::vector<double> &weights, double capacity) {
    if (values.size() != weights.size()) {
        throw std::runtime_error("values and weights vectors must have same size");
    }
    if (values.empty()) {
        throw std::runtime_error("knapsack problem must have at least one item");
    }
    if (capacity <= 0.0) {
        throw std::runtime_error("knapsack capacity must be positive");
    }
    
    metadata_.id = "knapsack";
    metadata_.family = "combinatorial";
    metadata_.description = "0-1 knapsack problem (continuous encoding)";
    dimension_ = values.size();
    values_ = values;
    weights_ = weights;
    capacity_ = capacity;
    lower_bounds_.assign(dimension_, 0.0);
    upper_bounds_.assign(dimension_, 1.0);
}

double KnapsackProblem::evaluate(const std::vector<double> &decision_vector) const {
    if (decision_vector.size() != dimension_) {
        throw std::runtime_error("decision vector dimension mismatch");
    }
    
    double total_value = 0.0;
    double total_weight = 0.0;
    
    for (std::size_t i = 0; i < decision_vector.size(); ++i) {
        const bool selected = decision_vector[i] >= 0.5;
        if (selected) {
            total_value += values_[i];
            total_weight += weights_[i];
        }
    }
    
    constexpr double penalty_factor = 1000.0;
    const double capacity_violation = std::max(0.0, total_weight - capacity_);
    const double penalty = penalty_factor * capacity_violation;
    
    return -(total_value - penalty);
}

} // namespace hpoea::wrappers::problems

