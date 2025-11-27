#include "hpoea/wrappers/pagmo/de_algorithm.hpp"
#include "hpoea/wrappers/pagmo/pso_algorithm.hpp"
#include "hpoea/wrappers/pagmo/sade_algorithm.hpp"
#include "hpoea/wrappers/pagmo/cmaes_hyper.hpp"
#include "hpoea/wrappers/problems/benchmark_problems.hpp"
#include "hpoea/core/types.hpp"

#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <memory>

using namespace hpoea;

void print_vec(const std::vector<double> &v) { // just prints [x, y, z, ...]
    std::cout << "[";
    for (size_t i = 0; i < v.size(); ++i) {
        std::cout << v[i];
        if (i < v.size() - 1) std::cout << ", ";
    }
    std::cout << "]";
}

int main() {
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "hpoea framework examples\n\n";
    
    // basic DE on sphere
    std::cout << "1. sphere (5d) with de\n";
    {
        wrappers::problems::SphereProblem problem(5);
        pagmo_wrappers::PagmoDifferentialEvolutionFactory factory;
        auto algo = factory.create();
        
        core::ParameterSet params;
        params.emplace("population_size", static_cast<std::int64_t>(30));
        params.emplace("generations", static_cast<std::int64_t>(50));
        params.emplace("variant", static_cast<std::int64_t>(2));
        params.emplace("scaling_factor", 0.8);
        params.emplace("crossover_rate", 0.9);
        algo->configure(params);
        
        core::Budget budget;
        budget.generations = 50;
        
        auto r = algo->run(problem, budget, 42);
        
        std::cout << "   fitness: " << r.best_fitness << "\n";
        std::cout << "   solution: "; print_vec(r.best_solution); std::cout << "\n";
        std::cout << "   evals: " << r.budget_usage.function_evaluations << "\n\n";
    }
    
    // see which algo does best
    std::cout << "2. algorithm comparison on sphere (10d)\n";
    {
        wrappers::problems::SphereProblem problem(10);
        
        struct AlgoTest {
            std::string name;
            std::unique_ptr<core::IEvolutionaryAlgorithmFactory> factory;
        };
        std::vector<AlgoTest> algos;
        algos.push_back({"de", std::make_unique<pagmo_wrappers::PagmoDifferentialEvolutionFactory>()});
        algos.push_back({"pso", std::make_unique<pagmo_wrappers::PagmoParticleSwarmOptimizationFactory>()});
        algos.push_back({"sade", std::make_unique<pagmo_wrappers::PagmoSelfAdaptiveDEFactory>()});
        
        core::Budget budget;
        budget.generations = 100;
        
        std::vector<std::pair<std::string, double>> results;
        
        for (const auto &a : algos) {
            auto algo = a.factory->create();
            core::ParameterSet params;
            params.emplace("population_size", static_cast<std::int64_t>(50));
            params.emplace("generations", static_cast<std::int64_t>(100));
            if (a.name == "pso") {
                params.emplace("omega", 0.7298);
                params.emplace("eta1", 2.05);
                params.emplace("eta2", 2.05);
                params.emplace("max_velocity", 0.5);
                params.emplace("variant", static_cast<std::int64_t>(5));
            }
            algo->configure(params);
            
            auto r = algo->run(problem, budget, 42);
            if (r.status == core::RunStatus::Success) {
                results.push_back({a.name, r.best_fitness});
                std::cout << "   " << a.name << ": " << r.best_fitness << "\n";
            }
        }
        
        if (!results.empty()) {
            auto best = std::min_element(results.begin(), results.end(),
                [](auto &a, auto &b) { return a.second < b.second; });
            std::cout << "   best: " << best->first << "\n\n";
        }
    }
    
    // let cmaes find good DE params
    std::cout << "3. cma-es tuning de hyperparameters\n";
    {
        wrappers::problems::SphereProblem problem(5);
        pagmo_wrappers::PagmoDifferentialEvolutionFactory ea_factory;
        pagmo_wrappers::PagmoCmaesHyperOptimizer hpo;
        
        core::ParameterSet hpo_params;
        hpo_params.emplace("generations", static_cast<std::int64_t>(15));
        hpo_params.emplace("sigma0", 0.5);
        hpo.configure(hpo_params);
        
        core::Budget budget;
        budget.generations = 15;
        budget.function_evaluations = 5000;
        
        auto r = hpo.optimize(ea_factory, problem, budget, 42);
        
        std::cout << "   objective: " << r.best_objective << "\n";
        std::cout << "   trials: " << r.trials.size() << "\n";
        std::cout << "   params: ";
        for (const auto &[k, v] : r.best_parameters) {
            std::cout << k << "=";
            std::visit([](auto x) { std::cout << x; }, v);
            std::cout << " ";
        }
        std::cout << "\n\n";
    }
    
    // try a few different test functions
    std::cout << "4. de on multiple benchmarks\n";
    {
        struct Problem {
            std::string name;
            std::unique_ptr<core::IProblem> prob;
        };
        std::vector<Problem> problems;
        problems.push_back({"sphere", std::make_unique<wrappers::problems::SphereProblem>(5)});
        problems.push_back({"rosenbrock", std::make_unique<wrappers::problems::RosenbrockProblem>(6)});
        problems.push_back({"rastrigin", std::make_unique<wrappers::problems::RastriginProblem>(8)});
        problems.push_back({"ackley", std::make_unique<wrappers::problems::AckleyProblem>(5)});
        
        pagmo_wrappers::PagmoDifferentialEvolutionFactory factory;
        core::Budget budget;
        budget.generations = 100;
        
        for (const auto &p : problems) {
            auto algo = factory.create();
            core::ParameterSet params;
            params.emplace("population_size", static_cast<std::int64_t>(50));
            params.emplace("generations", static_cast<std::int64_t>(100));
            algo->configure(params);
            
            auto r = algo->run(*p.prob, budget, 42);
            std::cout << "   " << p.name << ": " << r.best_fitness << "\n";
        }
    }
    
    std::cout << "\ndone\n";
    return 0;
}

