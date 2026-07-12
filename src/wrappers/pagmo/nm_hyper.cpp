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
            HyperparameterTuningProblem::Context &ctx) -> HyperEvolveOutcome {

            const auto dim = bounds.first.size();
            const auto pop_size = static_cast<pagmo::population::size_type>(dim + 1);

            // nlopt neldermead has no per-generation stepping
            // so wall time can't bound it mid-run
            const auto configured_max_fevals =
                get_param<std::int64_t>(configured_parameters_, "max_fevals");
            const auto xtol_rel = get_param<double>(configured_parameters_, "xtol_rel");
            const auto ftol_rel = get_param<double>(configured_parameters_, "ftol_rel");

            const auto simplex = static_cast<std::size_t>(pop_size);

            // nlopt neldermead stops on its own tolerances before maxeval
            // one capped solve underspends run budget so restart simplex from fresh seed until budget spent
            // every solve reserves simplex plus nlopt's final re evaluation
            // remainder below one solve stays unspent
            auto solve_once = [&](std::size_t max_fevals_this, unsigned long restart) {
                constexpr auto int_max = static_cast<std::size_t>(std::numeric_limits<int>::max());
                const auto max_fevals_int = static_cast<int>(std::min(max_fevals_this, int_max));
                pagmo::population population{tuning_problem, pop_size, derive_seed32(seed, restart)};
                pagmo::nlopt nm_alg("neldermead");
                nm_alg.set_maxeval(max_fevals_int);
                nm_alg.set_xtol_rel(xtol_rel);
                nm_alg.set_ftol_rel(ftol_rel);
                pagmo::algorithm algorithm{nm_alg};
                algorithm.evolve(population);
            };

            std::size_t restarts = 0;
            std::size_t first_max_fevals = 0;

            if (!budget.function_evaluations) {
                // one solve capped by configured max_fevals
                first_max_fevals = static_cast<std::size_t>(configured_max_fevals);
                solve_once(first_max_fevals, 0);
                restarts = 1;
            } else {
                const auto target = *budget.function_evaluations;
                const auto reserve = simplex + 1;  // construction + nlopt final re evaluation
                while (true) {
                    const auto spent = ctx.get_evaluations();
                    if (spent >= target) break;
                    const auto remaining = target - spent;
                    // room for simplex, one internal step and final re eval
                    // nlopt reads set_maxeval(0) as "no cap"
                    if (remaining < reserve + 1) break;
                    const auto max_fevals_this = std::min<std::size_t>(
                        static_cast<std::size_t>(configured_max_fevals), remaining - reserve);
                    if (restarts == 0) first_max_fevals = max_fevals_this;
                    solve_once(max_fevals_this, static_cast<unsigned long>(restarts));
                    ++restarts;
                }
            }

            auto effective_parameters = configured_parameters_;
            effective_parameters.insert_or_assign(
                "max_fevals", static_cast<std::int64_t>(first_max_fevals));

            HyperEvolveOutcome outcome;
            outcome.iterations = restarts;
            outcome.effective_parameters = std::move(effective_parameters);
            if (budget.function_evaluations && restarts == 0) {
                outcome.starved_message =
                    "budget insufficient for any nelder-mead step; minimum " +
                    std::to_string(simplex + 2) + " evaluations required (" +
                    std::to_string(simplex) + " initial simplex + 1 step + 1 final re-evaluation)";
            }
            return outcome;
        });
}

} // namespace hpoea::pagmo_wrappers
