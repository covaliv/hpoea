#include "hpoea/wrappers/pagmo/nm_hyper.hpp"

#include "budget_util.hpp"
#include "hyper_util.hpp"

#include <algorithm>
#include <limits>
#include <pagmo/algorithms/nlopt.hpp>

namespace {

using hpoea::core::AlgorithmIdentity;
using hpoea::core::ParameterDescriptor;
using hpoea::core::ParameterSpace;
using hpoea::core::ParameterType;

ParameterSpace make_parameter_space() {
    ParameterSpace space;

    ParameterDescriptor d;
    d.name = "max_fevals";
    d.type = ParameterType::Integer;
    d.integer_range = hpoea::core::IntegerRange{1, 100000};
    d.default_value = std::int64_t{1000};
    space.add_descriptor(d);

    d = {};
    d.name = "xtol_rel";
    d.type = ParameterType::Continuous;
    d.continuous_range = hpoea::core::ContinuousRange{1e-15, 1e-1};
    d.default_value = 1e-8;
    space.add_descriptor(d);

    d = {};
    d.name = "ftol_rel";
    d.type = ParameterType::Continuous;
    d.continuous_range = hpoea::core::ContinuousRange{1e-15, 1e-1};
    d.default_value = 1e-8;
    space.add_descriptor(d);

    return space;
}

AlgorithmIdentity make_identity() {
    return {"NelderMead", "nlopt::neldermead", "2.x"};
}

} // namespace

namespace hpoea::pagmo_wrappers {

PagmoNelderMeadHyperOptimizer::PagmoNelderMeadHyperOptimizer()
    : PagmoHyperOptimizerBase(make_parameter_space(), make_identity()) {}

core::HyperparameterOptimizerPtr PagmoNelderMeadHyperOptimizer::clone() const {
    return std::make_unique<PagmoNelderMeadHyperOptimizer>(*this);
}

core::HyperparameterOptimizationResult PagmoNelderMeadHyperOptimizer::optimize(
    const core::IEvolutionaryAlgorithmFactory &algorithm_factory,
    const core::IProblem &problem,
    const core::Budget &optimizer_budget,
    const core::Budget &algorithm_budget,
    unsigned long seed) {

    return run_hyper_optimization(
        algorithm_factory, problem, optimizer_budget, algorithm_budget,
        seed, configured_parameters_, search_space_,
        [&](pagmo::problem &tuning_problem,
            const auto &bounds,
            const core::Budget &budget,
            std::chrono::steady_clock::time_point start,
            HyperparameterTuningProblem::Context &) -> std::pair<pagmo::population, std::size_t> {

            const auto dim = bounds.first.size();
            const auto pop_size = static_cast<pagmo::population::size_type>(dim + 1);

            auto max_fevals = get_param<std::int64_t>(configured_parameters_, "max_fevals");
            if (budget.function_evaluations) {
                auto available = *budget.function_evaluations > static_cast<std::size_t>(pop_size)
                    ? *budget.function_evaluations - static_cast<std::size_t>(pop_size)
                    : std::size_t{0};
                max_fevals = std::min(max_fevals, available);
            }

            // saturate to int for nlopt's set_maxeval
            constexpr auto int_max = static_cast<std::size_t>(std::numeric_limits<int>::max());
            const auto max_fevals_int = static_cast<int>(std::min(max_fevals, int_max));

            const auto xtol_rel = get_param<double>(configured_parameters_, "xtol_rel");
            const auto ftol_rel = get_param<double>(configured_parameters_, "ftol_rel");

            pagmo::nlopt nm_alg("neldermead");
            nm_alg.set_maxeval(max_fevals_int);
            nm_alg.set_xtol_rel(xtol_rel);
            nm_alg.set_ftol_rel(ftol_rel);
            pagmo::algorithm algorithm{nm_alg};
            pagmo::population population{tuning_problem, pop_size, derive_seed(seed, 1)};

            const bool skip_evolve = budget.generations.has_value() && *budget.generations == 0;
            if (!skip_evolve) {
                population = algorithm.evolve(population);
            }

            const auto actual_generations = skip_evolve ? std::size_t{0} : std::size_t{1};
            return {std::move(population), actual_generations};
        });
}

} // namespace hpoea::pagmo_wrappers
