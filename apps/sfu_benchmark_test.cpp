// tests based on sfu.ca/~ssurjano/optimization.html

#include "hpoea/wrappers/pagmo/de_algorithm.hpp"
#include "hpoea/wrappers/pagmo/pso_algorithm.hpp"
#include "hpoea/wrappers/problems/benchmark_problems.hpp"
#include "hpoea/core/types.hpp"

#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <string>
#include <memory>

using namespace hpoea;

bool in_bounds(const std::vector<double> &x, const std::vector<double> &lo, const std::vector<double> &hi) {
    if (x.size() != lo.size()) return false;
    for (size_t i = 0; i < x.size(); ++i)
        if (x[i] < lo[i] || x[i] > hi[i]) return false;
    return true;
}

int main() {
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "sfu benchmark functions test\n\n";
    
    int passed = 0, failed = 0;
    
    struct Problem {
        std::string name;
        std::unique_ptr<core::IProblem> prob;
        double optimum;
    };
    
    std::vector<Problem> problems;
    problems.push_back({"griewank", std::make_unique<wrappers::problems::GriewankProblem>(5, -600.0, 600.0), 0.0});
    problems.push_back({"schwefel", std::make_unique<wrappers::problems::SchwefelProblem>(5, -500.0, 500.0), 0.0});
    problems.push_back({"zakharov", std::make_unique<wrappers::problems::ZakharovProblem>(5, -5.0, 10.0), 0.0});
    problems.push_back({"styblinski-tang", std::make_unique<wrappers::problems::StyblinskiTangProblem>(5, -5.0, 5.0), -39.16599 * 5.0});
    problems.push_back({"sphere", std::make_unique<wrappers::problems::SphereProblem>(5), 0.0});
    problems.push_back({"rastrigin", std::make_unique<wrappers::problems::RastriginProblem>(5), 0.0});
    problems.push_back({"ackley", std::make_unique<wrappers::problems::AckleyProblem>(5), 0.0});
    
    pagmo_wrappers::PagmoDifferentialEvolutionFactory factory;
    core::Budget budget;
    budget.generations = 200;
    budget.function_evaluations = 20000;
    
    for (const auto &p : problems) {
        auto algo = factory.create();
        core::ParameterSet params;
        params.emplace("population_size", static_cast<std::int64_t>(100));
        params.emplace("generations", static_cast<std::int64_t>(200));
        params.emplace("scaling_factor", 0.5);
        params.emplace("crossover_rate", 0.9);
        params.emplace("variant", static_cast<std::int64_t>(2));
        algo->configure(params);
        
        auto r = algo->run(*p.prob, budget, 42);
        double dist = std::abs(r.best_fitness - p.optimum);
        
        bool ok = r.status == core::RunStatus::Success &&
                  r.best_solution.size() == p.prob->dimension() &&
                  in_bounds(r.best_solution, p.prob->lower_bounds(), p.prob->upper_bounds()) &&
                  std::isfinite(r.best_fitness);
        
        std::cout << p.name << ": " << (ok ? "ok" : "fail")
                  << " (fitness=" << r.best_fitness << " dist=" << dist << ")\n";
        ok ? ++passed : ++failed;
    }
    
    std::cout << "\nde vs pso on griewank:\n"; // quick comparison
    {
        wrappers::problems::GriewankProblem problem(5, -600.0, 600.0);
        
        pagmo_wrappers::PagmoDifferentialEvolutionFactory de_factory;
        pagmo_wrappers::PagmoParticleSwarmOptimizationFactory pso_factory;
        
        auto de = de_factory.create();
        auto pso = pso_factory.create();
        
        core::ParameterSet de_params;
        de_params.emplace("population_size", static_cast<std::int64_t>(100));
        de_params.emplace("generations", static_cast<std::int64_t>(200));
        de_params.emplace("scaling_factor", 0.5);
        de_params.emplace("crossover_rate", 0.9);
        de_params.emplace("variant", static_cast<std::int64_t>(2));
        de->configure(de_params);
        
        core::ParameterSet pso_params;
        pso_params.emplace("population_size", static_cast<std::int64_t>(100));
        pso_params.emplace("generations", static_cast<std::int64_t>(200));
        pso_params.emplace("omega", 0.7298);
        pso_params.emplace("eta1", 2.05);
        pso_params.emplace("eta2", 2.05);
        pso_params.emplace("max_velocity", 0.5);
        pso_params.emplace("variant", static_cast<std::int64_t>(5));
        pso->configure(pso_params);
        
        auto de_r = de->run(problem, budget, 42);
        auto pso_r = pso->run(problem, budget, 42);
        
        bool ok = de_r.status == core::RunStatus::Success && pso_r.status == core::RunStatus::Success;
        std::cout << "  de=" << de_r.best_fitness << " pso=" << pso_r.best_fitness
                  << " better=" << (de_r.best_fitness < pso_r.best_fitness ? "de" : "pso") << "\n";
        ok ? ++passed : ++failed;
    }
    
    std::cout << "\nsummary: " << passed << " passed, " << failed << " failed\n";
    return failed == 0 ? 0 : 1;
}

