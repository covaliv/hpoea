#include "hpoea/wrappers/pagmo/nm_hyper.hpp"

#include "budget_util.hpp"
#include "hyper_util.hpp"

#include <algorithm>
#include <limits>
#include <string>
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
        seed, search_space_,
        [&](pagmo::problem &tuning_problem,
            const auto &bounds,
            const core::Budget &budget,
            std::chrono::steady_clock::time_point,
            HyperparameterTuningProblem::Context &) -> HyperEvolveOutcome {

            const auto dim = bounds.first.size();
            const auto pop_size = static_cast<pagmo::population::size_type>(dim + 1);

            // nlopt neldermead has no per-generation stepping
            // so wall time can't bound it mid-run
            const auto configured_max_fevals =
                get_param<std::int64_t>(configured_parameters_, "max_fevals");
            auto max_fevals = configured_max_fevals;
            if (budget.function_evaluations) {
                // reserve initial simplex and nlopt's one final re-evaluation of the best point
                const auto reserved = static_cast<std::size_t>(pop_size) + 1;
                auto available = *budget.function_evaluations > reserved
                    ? *budget.function_evaluations - reserved
                    : std::size_t{0};
                max_fevals = std::min(max_fevals, available);
            }

            const auto xtol_rel = get_param<double>(configured_parameters_, "xtol_rel");
            const auto ftol_rel = get_param<double>(configured_parameters_, "ftol_rel");

            // nlopt reads set_maxeval(0) as "no cap"
            // so too small a budget must skip evolve
            const bool budget_starved = budget.function_evaluations.has_value() &&
                max_fevals == 0 && configured_max_fevals > 0;
            const bool skip_evolve = budget_starved ||
                (budget.generations.has_value() && *budget.generations == 0);

            pagmo::population population{tuning_problem, pop_size, derive_seed32(seed, 0)};
            if (!skip_evolve) {
                constexpr auto int_max = static_cast<std::size_t>(std::numeric_limits<int>::max());
                const auto max_fevals_int = static_cast<int>(std::min(max_fevals, int_max));
                pagmo::nlopt nm_alg("neldermead");
                nm_alg.set_maxeval(max_fevals_int);
                nm_alg.set_xtol_rel(xtol_rel);
                nm_alg.set_ftol_rel(ftol_rel);
                pagmo::algorithm algorithm{nm_alg};
                population = algorithm.evolve(population);
            }

            auto effective_parameters = configured_parameters_;
            effective_parameters.insert_or_assign("max_fevals", static_cast<std::int64_t>(max_fevals));

            HyperEvolveOutcome outcome;
            outcome.iterations = skip_evolve ? std::size_t{0} : std::size_t{1};
            outcome.effective_parameters = std::move(effective_parameters);
            if (budget_starved) {
                const auto simplex = static_cast<std::size_t>(pop_size);
                outcome.starved_message =
                    "budget insufficient for any nelder-mead step; minimum " +
                    std::to_string(simplex + 2) + " evaluations required (" +
                    std::to_string(simplex) + " initial simplex + 1 step + 1 final re-evaluation)";
            }
            return outcome;
        });
}

} // namespace hpoea::pagmo_wrappers
