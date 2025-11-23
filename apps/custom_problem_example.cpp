#include "hpoea/core/problem.hpp"
#include "hpoea/core/types.hpp"
#include "hpoea/wrappers/pagmo/de_algorithm.hpp"

#include <cmath>
#include <vector>
#include <iostream>
#include <iomanip>

// custom problem: shifted sphere function
class ShiftedSphereProblem final : public hpoea::core::IProblem {
public:
    explicit ShiftedSphereProblem(std::size_t dimension, const std::vector<double> &shift)
        : dimension_(dimension), shift_(shift) {
        if (shift.size() != dimension) {
            throw std::invalid_argument("Shift vector size must match dimension");
        }
        metadata_.id = "shifted_sphere";
        metadata_.family = "custom";
        metadata_.description = "Shifted Sphere Function";
    }
    
    [[nodiscard]] const hpoea::core::ProblemMetadata &metadata() const noexcept override { 
        return metadata_; 
    }
    
    [[nodiscard]] std::size_t dimension() const override { return dimension_; }
    
    [[nodiscard]] double evaluate(const std::vector<double> &x) const override {
        if (x.size() != dimension_) {
            throw std::invalid_argument("Input size must match dimension");
        }
        
        double sum = 0.0;
        for (std::size_t i = 0; i < dimension_; ++i) {
            const double shifted = x[i] - shift_[i];
            sum += shifted * shifted;
        }
        return sum;
    }
    
    [[nodiscard]] std::vector<double> lower_bounds() const override {
        return std::vector<double>(dimension_, -10.0);
    }
    
    [[nodiscard]] std::vector<double> upper_bounds() const override {
        return std::vector<double>(dimension_, 10.0);
    }
    
    [[nodiscard]] bool is_stochastic() const noexcept override { return false; }
    
private:
    hpoea::core::ProblemMetadata metadata_{};
    std::size_t dimension_;
    std::vector<double> shift_;
};

int main() {
    using namespace hpoea;
    
    std::vector<double> shift = {2.5, -1.3, 0.7, -0.5, 1.1, -2.0, 0.3, 1.5};
    ShiftedSphereProblem problem(8, shift);
    
    pagmo_wrappers::PagmoDifferentialEvolutionFactory factory;
    auto algorithm = factory.create();
    
    core::ParameterSet params;
    params.emplace("population_size", static_cast<std::int64_t>(60));
    params.emplace("generations", static_cast<std::int64_t>(150));
    params.emplace("scaling_factor", 0.7);
    params.emplace("crossover_rate", 0.9);
    algorithm->configure(params);
    
    core::Budget budget;
    budget.generations = 150;
    
    auto result = algorithm->run(problem, budget, 42UL);
    
    if (result.status == core::RunStatus::Success) {
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "best_fitness: " << result.best_fitness << "\n";
        
        double distance = 0.0;
        for (std::size_t i = 0; i < result.best_solution.size(); ++i) {
            const double diff = result.best_solution[i] - shift[i];
            distance += diff * diff;
        }
        distance = std::sqrt(distance);
        std::cout << "distance_to_optimum: " << distance << "\n";
        std::cout << "function_evaluations: " << result.budget_usage.function_evaluations << "\n";
    } else {
        std::cerr << "error: " << result.message << "\n";
        return 1;
    }
    
    return 0;
}

