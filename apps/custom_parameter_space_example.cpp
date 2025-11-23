#include "hpoea/core/parameters.hpp"
#include "hpoea/core/problem.hpp"
#include "hpoea/core/types.hpp"
#include "hpoea/wrappers/pagmo/de_algorithm.hpp"
#include "hpoea/wrappers/problems/benchmark_problems.hpp"

#include <iostream>
#include <iomanip>
#include <vector>
#include <random>

int main() {
    using namespace hpoea;
    
    core::ParameterSpace custom_space;
    
    core::ParameterDescriptor pop_size;
    pop_size.name = "population_size";
    pop_size.type = core::ParameterType::Integer;
    pop_size.integer_range = core::IntegerRange{20, 200};
    pop_size.default_value = static_cast<std::int64_t>(50);
    custom_space.add_descriptor(pop_size);
    
    core::ParameterDescriptor gens;
    gens.name = "generations";
    gens.type = core::ParameterType::Integer;
    gens.integer_range = core::IntegerRange{10, 500};
    gens.default_value = static_cast<std::int64_t>(100);
    custom_space.add_descriptor(gens);
    
    core::ParameterDescriptor scaling;
    scaling.name = "scaling_factor";
    scaling.type = core::ParameterType::Continuous;
    scaling.continuous_range = core::ContinuousRange{0.1, 1.0};
    scaling.default_value = 0.8;
    custom_space.add_descriptor(scaling);
    
    core::ParameterDescriptor crossover;
    crossover.name = "crossover_rate";
    crossover.type = core::ParameterType::Continuous;
    crossover.continuous_range = core::ContinuousRange{0.0, 1.0};
    crossover.default_value = 0.9;
    custom_space.add_descriptor(crossover);
    
    core::ParameterDescriptor variant;
    variant.name = "variant";
    variant.type = core::ParameterType::Integer;
    variant.integer_range = core::IntegerRange{1, 10};
    variant.default_value = static_cast<std::int64_t>(2);
    custom_space.add_descriptor(variant);
    
    std::mt19937 rng(42);
    std::vector<core::ParameterSet> test_configs;
    
    for (int i = 0; i < 5; ++i) {
        core::ParameterSet config;
        
        for (const auto &desc : custom_space.descriptors()) {
            if (desc.type == core::ParameterType::Integer && desc.integer_range.has_value()) {
                std::uniform_int_distribution<std::int64_t> dist(
                    desc.integer_range->lower, desc.integer_range->upper);
                config.emplace(desc.name, dist(rng));
            } else if (desc.type == core::ParameterType::Continuous && desc.continuous_range.has_value()) {
                std::uniform_real_distribution<double> dist(
                    desc.continuous_range->lower, desc.continuous_range->upper);
                config.emplace(desc.name, dist(rng));
            }
        }
        
        try {
            custom_space.validate(config);
            test_configs.push_back(config);
            
            std::cout << "config_" << (i + 1) << ": ";
            for (const auto &[name, value] : config) {
                std::cout << name << "=";
                std::visit([](auto v) { std::cout << v; }, value);
                std::cout << " ";
            }
            std::cout << "\n";
        } catch (const core::ParameterValidationError &e) {
            std::cerr << "config_" << (i + 1) << ": validation_error: " << e.what() << "\n";
        }
    }
    
    if (!test_configs.empty()) {
        wrappers::problems::SphereProblem problem(8);
        pagmo_wrappers::PagmoDifferentialEvolutionFactory factory;
        auto algorithm = factory.create();
        
        algorithm->configure(test_configs[0]);
        
        core::Budget budget;
        budget.generations = std::get<std::int64_t>(test_configs[0].at("generations"));
        
        auto result = algorithm->run(problem, budget, 42UL);
        
        if (result.status == core::RunStatus::Success) {
            std::cout << std::fixed << std::setprecision(6);
            std::cout << "best_fitness: " << result.best_fitness << "\n";
            std::cout << "function_evaluations: " << result.budget_usage.function_evaluations << "\n";
        }
    }
    
    return 0;
}

