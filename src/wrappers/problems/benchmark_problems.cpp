#include "hpoea/wrappers/problems/benchmark_problems.hpp"

#include "hpoea/core/problem.hpp"

#include <cmath>
#include <memory>
#include <numbers>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>

namespace {

void validate_bounds(double lower, double upper, const char *problem_name) {
    if (lower >= upper) {
        throw std::invalid_argument(
            std::string(problem_name) + ": lower bound must be less than upper bound");
    }
}

void validate_dimension(std::size_t dim, const char *problem_name) {
    if (dim == 0) {
        throw std::invalid_argument(
            std::string(problem_name) + ": dimension must be at least 1");
    }
}

hpoea::core::ProblemMetadata make_metadata(const char *id, const char *family, const char *description) {
    hpoea::core::ProblemMetadata metadata;
    metadata.id = id;
    metadata.family = family;
    metadata.description = description;
    return metadata;
}

} // namespace

namespace hpoea::wrappers::problems {

SphereProblem::SphereProblem(std::size_t dimension, double lower_bound, double upper_bound)
    : BenchmarkProblemBase(
          make_metadata("sphere", "benchmark", "Sphere function (unimodal, separable)"),
          dimension,
          std::vector<double>(dimension, lower_bound),
          std::vector<double>(dimension, upper_bound)) {
    validate_dimension(dimension, "sphere");
    validate_bounds(lower_bound, upper_bound, "sphere");
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

RosenbrockProblem::RosenbrockProblem(std::size_t dimension, double lower_bound, double upper_bound)
    : BenchmarkProblemBase(
          make_metadata("rosenbrock", "benchmark", "Rosenbrock function (unimodal, non-separable)"),
          dimension,
          std::vector<double>(dimension, lower_bound),
          std::vector<double>(dimension, upper_bound)) {
    if (dimension < 2) {
        throw std::invalid_argument("rosenbrock: dimension must be at least 2");
    }
    validate_bounds(lower_bound, upper_bound, "rosenbrock");
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

RastriginProblem::RastriginProblem(std::size_t dimension, double lower_bound, double upper_bound)
    : BenchmarkProblemBase(
          make_metadata("rastrigin", "benchmark", "Rastrigin function (multimodal, separable)"),
          dimension,
          std::vector<double>(dimension, lower_bound),
          std::vector<double>(dimension, upper_bound)) {
    validate_dimension(dimension, "rastrigin");
    validate_bounds(lower_bound, upper_bound, "rastrigin");
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

AckleyProblem::AckleyProblem(std::size_t dimension, double lower_bound, double upper_bound)
    : BenchmarkProblemBase(
          make_metadata("ackley", "benchmark", "Ackley function (multimodal, non-separable)"),
          dimension,
          std::vector<double>(dimension, lower_bound),
          std::vector<double>(dimension, upper_bound)) {
    validate_dimension(dimension, "ackley");
    validate_bounds(lower_bound, upper_bound, "ackley");
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

GriewankProblem::GriewankProblem(std::size_t dimension, double lower_bound, double upper_bound)
    : BenchmarkProblemBase(
          make_metadata("griewank", "benchmark", "Griewank function (multimodal, many local minima)"),
          dimension,
          std::vector<double>(dimension, lower_bound),
          std::vector<double>(dimension, upper_bound)) {
    validate_dimension(dimension, "griewank");
    validate_bounds(lower_bound, upper_bound, "griewank");
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

SchwefelProblem::SchwefelProblem(std::size_t dimension, double lower_bound, double upper_bound)
    : BenchmarkProblemBase(
          make_metadata("schwefel", "benchmark", "Schwefel function (multimodal, deceptive landscape)"),
          dimension,
          std::vector<double>(dimension, lower_bound),
          std::vector<double>(dimension, upper_bound)) {
    validate_dimension(dimension, "schwefel");
    validate_bounds(lower_bound, upper_bound, "schwefel");
}

double SchwefelProblem::evaluate(const std::vector<double> &decision_vector) const {
    if (decision_vector.size() != dimension_) {
        throw std::runtime_error("Decision vector dimension mismatch");
    }
    constexpr double alpha = 418.9828872724339; // constant for global minimum
    double sum = 0.0;
    
    for (const auto value : decision_vector) {
        sum += -value * std::sin(std::sqrt(std::abs(value)));
    }
    
    return alpha * static_cast<double>(dimension_) + sum;
}

ZakharovProblem::ZakharovProblem(std::size_t dimension, double lower_bound, double upper_bound)
    : BenchmarkProblemBase(
          make_metadata("zakharov", "benchmark", "Zakharov function (unimodal, plate-shaped)"),
          dimension,
          std::vector<double>(dimension, lower_bound),
          std::vector<double>(dimension, upper_bound)) {
    validate_dimension(dimension, "zakharov");
    validate_bounds(lower_bound, upper_bound, "zakharov");
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

StyblinskiTangProblem::StyblinskiTangProblem(std::size_t dimension, double lower_bound, double upper_bound)
    : BenchmarkProblemBase(
          make_metadata("styblinski_tang", "benchmark", "Styblinski-Tang function (multimodal)"),
          dimension,
          std::vector<double>(dimension, lower_bound),
          std::vector<double>(dimension, upper_bound)) {
    validate_dimension(dimension, "styblinski_tang");
    validate_bounds(lower_bound, upper_bound, "styblinski_tang");
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

KnapsackProblem::KnapsackProblem(const std::vector<double> &values, const std::vector<double> &weights, double capacity)
    : BenchmarkProblemBase(
          make_metadata("knapsack", "combinatorial", "0-1 knapsack problem (continuous encoding)"),
          values.size(),
          std::vector<double>(values.size(), 0.0),
          std::vector<double>(values.size(), 1.0)),
      values_(values),
      weights_(weights),
      capacity_(capacity),
      total_value_(std::accumulate(values_.begin(), values_.end(), 0.0)) {
    if (values.size() != weights.size()) {
        throw std::runtime_error("values and weights vectors must have same size");
    }
    if (values.empty()) {
        throw std::runtime_error("knapsack problem must have at least one item");
    }
    if (capacity <= 0.0) {
        throw std::runtime_error("knapsack capacity must be positive");
    }
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
    
    const double capacity_violation = std::max(0.0, total_weight - capacity_);

    if (capacity_violation > 0.0) {
        // feasible objectives are <= 0
        // this stays worse than all of them
        return total_value_ + capacity_violation;
    }

    return -total_value;
}

namespace {

const config::ConfigValue *find_config_value(const config::ProblemParameterSet &parameters,
                                             const std::string &name) {
    auto it = parameters.find(name);
    return it == parameters.end() ? nullptr : &it->second;
}

std::optional<std::int64_t> read_config_int(const config::ProblemParameterSet &parameters,
                                            const std::string &name) {
    const auto *value = find_config_value(parameters, name);
    if (!value) {
        return std::nullopt;
    }
    if (const auto *integer = std::get_if<std::int64_t>(value)) {
        return *integer;
    }
    throw std::invalid_argument("problem parameter '" + name + "' must be an integer");
}

std::optional<double> read_config_number(const config::ProblemParameterSet &parameters,
                                         const std::string &name) {
    const auto *value = find_config_value(parameters, name);
    if (!value) {
        return std::nullopt;
    }
    if (const auto *floating = std::get_if<double>(value)) {
        return *floating;
    }
    if (const auto *integer = std::get_if<std::int64_t>(value)) {
        return static_cast<double>(*integer);
    }
    throw std::invalid_argument("problem parameter '" + name + "' must be numeric");
}

std::vector<double> read_config_number_vector(const config::ProblemParameterSet &parameters,
                                              const std::string &name) {
    const auto *value = find_config_value(parameters, name);
    if (!value) {
        return {};
    }
    if (const auto *doubles = std::get_if<std::vector<double>>(value)) {
        return *doubles;
    }
    if (const auto *ints = std::get_if<std::vector<std::int64_t>>(value)) {
        return std::vector<double>(ints->begin(), ints->end());
    }
    throw std::invalid_argument("problem parameter '" + name + "' must be a numeric array");
}

void reject_unknown_keys(const std::string &problem_type,
                         const config::ProblemParameterSet &parameters,
                         const std::vector<std::string> &allowed) {
    for (const auto &[key, value] : parameters) {
        (void)value;
        bool recognized = false;
        for (const auto &name : allowed) {
            if (key == name) {
                recognized = true;
                break;
            }
        }
        if (!recognized) {
            throw std::invalid_argument(
                "unknown problem parameter '" + key + "' for problem type '" + problem_type + "'");
        }
    }
}

// omit bounds to keep each problem's canonical default domain
// pass both to override it
template <typename Problem>
std::unique_ptr<core::IProblem> make_box_problem(
    std::size_t dimension, const std::optional<double> &lower, const std::optional<double> &upper) {
    if (lower.has_value()) {
        return std::make_unique<Problem>(dimension, *lower, *upper);
    }
    return std::make_unique<Problem>(dimension);
}

} // namespace

std::unique_ptr<core::IProblem> make_benchmark_problem(
    const std::string &problem_type,
    const config::ProblemParameterSet &parameters) {

    static const std::vector<std::string> box_keys = {"dimension", "lower_bound", "upper_bound"};

    if (problem_type == "knapsack") {
        reject_unknown_keys(problem_type, parameters, {"values", "weights", "capacity"});
        auto values = read_config_number_vector(parameters, "values");
        auto weights = read_config_number_vector(parameters, "weights");
        if (values.empty()) {
            throw std::invalid_argument("knapsack problem parameter 'values' is required");
        }
        if (weights.empty()) {
            throw std::invalid_argument("knapsack problem parameter 'weights' is required");
        }
        const auto capacity = read_config_number(parameters, "capacity");
        if (!capacity.has_value()) {
            throw std::invalid_argument("knapsack problem parameter 'capacity' is required");
        }
        return std::make_unique<KnapsackProblem>(values, weights, *capacity);
    }

    const bool is_box = problem_type == "sphere" || problem_type == "rosenbrock" ||
                        problem_type == "rastrigin" || problem_type == "ackley" ||
                        problem_type == "griewank" || problem_type == "schwefel" ||
                        problem_type == "zakharov" || problem_type == "styblinski_tang";
    if (!is_box) {
        throw std::invalid_argument("unknown problem type '" + problem_type + "'");
    }

    reject_unknown_keys(problem_type, parameters, box_keys);

    const auto dimension = read_config_int(parameters, "dimension");
    if (!dimension.has_value()) {
        throw std::invalid_argument(
            "problem parameter 'dimension' is required for problem type '" + problem_type + "'");
    }
    if (*dimension < 1) {
        throw std::invalid_argument(
            "problem parameter 'dimension' must be at least 1 for problem type '" + problem_type + "'");
    }
    const auto lb = read_config_number(parameters, "lower_bound");
    const auto ub = read_config_number(parameters, "upper_bound");
    if (lb.has_value() != ub.has_value()) {
        // one bound alone would silently pair with the other's canonical default
        throw std::invalid_argument(
            "problem parameters 'lower_bound' and 'upper_bound' must be provided together");
    }
    const auto dim = static_cast<std::size_t>(*dimension);

    if (problem_type == "sphere") return make_box_problem<SphereProblem>(dim, lb, ub);
    if (problem_type == "rosenbrock") return make_box_problem<RosenbrockProblem>(dim, lb, ub);
    if (problem_type == "rastrigin") return make_box_problem<RastriginProblem>(dim, lb, ub);
    if (problem_type == "ackley") return make_box_problem<AckleyProblem>(dim, lb, ub);
    if (problem_type == "griewank") return make_box_problem<GriewankProblem>(dim, lb, ub);
    if (problem_type == "schwefel") return make_box_problem<SchwefelProblem>(dim, lb, ub);
    if (problem_type == "zakharov") return make_box_problem<ZakharovProblem>(dim, lb, ub);
    return make_box_problem<StyblinskiTangProblem>(dim, lb, ub);
}

} // namespace hpoea::wrappers::problems
