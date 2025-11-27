#include "hpoea/wrappers/pagmo/de_algorithm.hpp"
#include "hpoea/wrappers/pagmo/cmaes_hyper.hpp"
#include "hpoea/wrappers/problems/benchmark_problems.hpp"
#include "hpoea/core/types.hpp"

#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <string>

using namespace hpoea;

struct TestResult { // keeps track of pass/fail
    int passed = 0;
    int failed = 0;
    
    void check(const std::string &name, bool condition, const std::string &detail = "") {
        std::cout << name << ": " << (condition ? "PASSED" : "FAILED");
        if (!detail.empty()) std::cout << " (" << detail << ")";
        std::cout << "\n";
        condition ? ++passed : ++failed;
    }
};

bool in_bounds(const std::vector<double> &x, const std::vector<double> &lo, const std::vector<double> &hi) {
    if (x.size() != lo.size()) return false;
    for (size_t i = 0; i < x.size(); ++i)
        if (x[i] < lo[i] || x[i] > hi[i]) return false;
    return true;
}

core::OptimizationResult run_de(int dim, int pop, int gens, unsigned long seed) { // helper
    wrappers::problems::SphereProblem problem(dim);
    pagmo_wrappers::PagmoDifferentialEvolutionFactory factory;
    auto algo = factory.create();
    
    core::ParameterSet params;
    params.emplace("population_size", static_cast<std::int64_t>(pop));
    params.emplace("generations", static_cast<std::int64_t>(gens));
    algo->configure(params);
    
    core::Budget budget;
    budget.generations = gens;
    return algo->run(problem, budget, seed);
}

int main() {
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "hpoea correctness tests\n\n";
    
    TestResult t;
    
    { // basic opt should work
        auto r = run_de(5, 30, 50, 42);
        wrappers::problems::SphereProblem p(5);
        bool ok = r.status == core::RunStatus::Success &&
                  r.best_fitness >= 0 &&
                  r.best_solution.size() == 5 &&
                  in_bounds(r.best_solution, p.lower_bounds(), p.upper_bounds()) &&
                  r.budget_usage.generations <= 50 &&
                  r.budget_usage.function_evaluations > 0;
        t.check("basic optimization", ok, "fitness=" + std::to_string(r.best_fitness));
    }
    
    { // same seed = same result, right?
        auto r1 = run_de(5, 20, 30, 999);
        auto r2 = run_de(5, 20, 30, 999);
        double diff = std::abs(r1.best_fitness - r2.best_fitness);
        bool ok = r1.status == core::RunStatus::Success &&
                  r2.status == core::RunStatus::Success &&
                  diff < 1e-10;
        t.check("reproducibility", ok, "diff=" + std::to_string(diff));
    }
    
    { // shouldn't exceed budget even if we ask for more
        wrappers::problems::SphereProblem problem(5);
        pagmo_wrappers::PagmoDifferentialEvolutionFactory factory;
        auto algo = factory.create();
        
        core::ParameterSet params;
        params.emplace("population_size", static_cast<std::int64_t>(20));
        params.emplace("generations", static_cast<std::int64_t>(1000)); // way more than we'll allow
        algo->configure(params);
        
        core::Budget budget;
        budget.generations = 50; // hard cap
        auto r = algo->run(problem, budget, 42);
        
        bool ok = (r.status == core::RunStatus::Success || r.status == core::RunStatus::BudgetExceeded) &&
                  r.budget_usage.generations <= 50;
        t.check("budget enforcement", ok, "used=" + std::to_string(r.budget_usage.generations));
    }
    
    { // more gens should give better results
        auto r20 = run_de(5, 30, 20, 42);
        auto r100 = run_de(5, 30, 100, 42);
        bool ok = r20.status == core::RunStatus::Success &&
                  r100.status == core::RunStatus::Success &&
                  r100.best_fitness <= r20.best_fitness;
        t.check("quality improvement", ok, 
                "20g=" + std::to_string(r20.best_fitness) + " 100g=" + std::to_string(r100.best_fitness));
    }
    
    { // HPO should find something decent
        wrappers::problems::SphereProblem problem(5);
        pagmo_wrappers::PagmoDifferentialEvolutionFactory ea_factory;
        pagmo_wrappers::PagmoCmaesHyperOptimizer hpo;
        
        core::ParameterSet hpo_params;
        hpo_params.emplace("generations", static_cast<std::int64_t>(10));
        hpo_params.emplace("sigma0", 0.5);
        hpo.configure(hpo_params);
        
        core::Budget budget;
        budget.generations = 10;
        budget.function_evaluations = 3000;
        
        auto r = hpo.optimize(ea_factory, problem, budget, 42);
        bool ok = r.status == core::RunStatus::Success &&
                  !r.trials.empty() &&
                  std::isfinite(r.best_objective) &&
                  r.best_objective >= 0 &&
                  !r.best_parameters.empty() &&
                  r.budget_usage.function_evaluations > 0;
        t.check("hyperparameter optimization", ok, 
                "obj=" + std::to_string(r.best_objective) + " trials=" + std::to_string(r.trials.size()));
    }
    
    { // try a bunch of different problems
        pagmo_wrappers::PagmoDifferentialEvolutionFactory factory;
        core::Budget budget;
        budget.generations = 100;
        
        struct ProblemCase {
            std::string name;
            std::unique_ptr<core::IProblem> prob;
            double max_fitness;
        };
        std::vector<ProblemCase> cases;
        cases.push_back({"sphere", std::make_unique<wrappers::problems::SphereProblem>(5), 100.0});
        cases.push_back({"rosenbrock", std::make_unique<wrappers::problems::RosenbrockProblem>(6), 1000.0});
        cases.push_back({"rastrigin", std::make_unique<wrappers::problems::RastriginProblem>(8), 200.0});
        cases.push_back({"ackley", std::make_unique<wrappers::problems::AckleyProblem>(5), 50.0});
        
        bool all_ok = true;
        for (const auto &c : cases) {
            auto algo = factory.create();
            core::ParameterSet params;
            params.emplace("population_size", static_cast<std::int64_t>(50));
            params.emplace("generations", static_cast<std::int64_t>(100));
            algo->configure(params);
            
            auto r = algo->run(*c.prob, budget, 42);
            bool ok = r.status == core::RunStatus::Success &&
                      r.best_fitness >= 0 && r.best_fitness <= c.max_fitness &&
                      r.best_solution.size() == c.prob->dimension() &&
                      in_bounds(r.best_solution, c.prob->lower_bounds(), c.prob->upper_bounds());
            
            std::cout << "  " << c.name << ": " << (ok ? "ok" : "fail") 
                      << " (fitness=" << r.best_fitness << ")\n";
            all_ok = all_ok && ok;
        }
        t.check("multiple problems", all_ok);
    }
    
    std::cout << "\nsummary: " << t.passed << " passed, " << t.failed << " failed\n";
    return t.failed == 0 ? 0 : 1;
}

