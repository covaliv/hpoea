#include "hpoea/wrappers/pagmo/cmaes_hyper.hpp"

#include "budget_util.hpp"
#include "hyper_util.hpp"

#include <algorithm>
#include <limits>
#include <pagmo/algorithms/cmaes.hpp>

namespace {

using hpoea::core::AlgorithmIdentity;
using hpoea::core::ParameterDescriptor;
using hpoea::core::ParameterSpace;
using hpoea::core::ParameterType;

ParameterSpace make_parameter_space() {
    ParameterSpace space;

    ParameterDescriptor d;
    d.name = "generations";
    d.type = ParameterType::Integer;
    d.integer_range = hpoea::core::IntegerRange{1, 1000};
    d.default_value = std::int64_t{100};
    space.add_descriptor(d);

    d = {};
    d.name = "sigma0";
    d.type = ParameterType::Continuous;
    d.continuous_range = hpoea::core::ContinuousRange{1e-6, 10.0};
    d.default_value = 0.5;
    space.add_descriptor(d);

    d = {};
    d.name = "cc";
    d.type = ParameterType::Continuous;
    d.continuous_range = hpoea::core::ContinuousRange{0.0, 1.0};
    d.default_value = 0.4;
    space.add_descriptor(d);

    d = {};
    d.name = "cs";
    d.type = ParameterType::Continuous;
    d.continuous_range = hpoea::core::ContinuousRange{0.0, 1.0};
    d.default_value = 0.3;
    space.add_descriptor(d);

    d = {};
    d.name = "c1";
    d.type = ParameterType::Continuous;
    d.continuous_range = hpoea::core::ContinuousRange{0.0, 1.0};
    d.default_value = 0.05;
    space.add_descriptor(d);

    d = {};
    d.name = "cmu";
    d.type = ParameterType::Continuous;
    d.continuous_range = hpoea::core::ContinuousRange{0.0, 1.0};
    d.default_value = 0.1;
    space.add_descriptor(d);

    d = {};
    d.name = "ftol";
    d.type = ParameterType::Continuous;
    d.continuous_range = hpoea::core::ContinuousRange{0.0, 1.0};
    d.default_value = 1e-6;
    space.add_descriptor(d);

    d = {};
    d.name = "xtol";
    d.type = ParameterType::Continuous;
    d.continuous_range = hpoea::core::ContinuousRange{0.0, 1.0};
    d.default_value = 1e-6;
    space.add_descriptor(d);

    d = {};
    d.name = "force_bounds";
    d.type = ParameterType::Boolean;
    d.default_value = false;
    space.add_descriptor(d);

    return space;
}

AlgorithmIdentity make_identity() {
    return {"CMAESHyperOptimizer", "pagmo::cmaes", "2.x"};
}

} // namespace

namespace hpoea::pagmo_wrappers {

PagmoCmaesHyperOptimizer::PagmoCmaesHyperOptimizer()
    : PagmoHyperOptimizerBase(make_parameter_space(), make_identity()) {}

core::HyperparameterOptimizerPtr PagmoCmaesHyperOptimizer::clone() const {
    return std::make_unique<PagmoCmaesHyperOptimizer>(*this);
}

core::HyperparameterOptimizationResult PagmoCmaesHyperOptimizer::optimize(
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
            std::chrono::steady_clock::time_point start,
            HyperparameterTuningProblem::Context &ctx) -> HyperEvolveOutcome {

            const auto seed32 = to_seed32(seed);
            const auto dim = bounds.first.size();
            // pagmo::cmaes requires at least 5 individuals
            const auto pop_size = static_cast<pagmo::population::size_type>(
                std::max(dim * 4, std::size_t{5}));

            const auto configured_generations =
                get_param<std::int64_t>(configured_parameters_, "generations");
            const auto generations = clamp_hyper_generations(
                configured_generations, budget, static_cast<std::size_t>(pop_size));

            // 1 gen/evolve with memory=true equals one N-gen call
            // so budgets can bound work mid-run
            pagmo::algorithm algorithm{pagmo::cmaes{
                1u,
                get_param<double>(configured_parameters_, "cc"),
                get_param<double>(configured_parameters_, "cs"),
                get_param<double>(configured_parameters_, "c1"),
                get_param<double>(configured_parameters_, "cmu"),
                get_param<double>(configured_parameters_, "sigma0"),
                get_param<double>(configured_parameters_, "ftol"),
                get_param<double>(configured_parameters_, "xtol"),
                true,
                get_param<bool>(configured_parameters_, "force_bounds"),
                seed32}};

            pagmo::population population{tuning_problem, pop_size, derive_seed32(seed, 0)};

            std::size_t actual_iterations = 0;
            for (std::size_t g = 0; g < generations; ++g) {
                const auto before = ctx.get_evaluations();
                population = algorithm.evolve(population);
                if (ctx.get_evaluations() == before) {
                    // no new evaluations means ftol/xtol converged
                    // don't count a no-op generation
                    break;
                }
                ++actual_iterations;
                if (budget.wall_time) {
                    const auto now = std::chrono::steady_clock::now();
                    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start);
                    if (elapsed > *budget.wall_time)
                        break;
                }
                if (budget.function_evaluations &&
                    ctx.get_evaluations() >= *budget.function_evaluations)
                    break;
            }

            auto effective_parameters = configured_parameters_;
            effective_parameters.insert_or_assign("generations", static_cast<std::int64_t>(generations));

            HyperEvolveOutcome outcome;
            outcome.iterations = actual_iterations;
            outcome.effective_parameters = std::move(effective_parameters);
            if (generations == 0 && configured_generations > 0) {
                outcome.starved_message =
                    "budget insufficient for any generations; only initial population evaluated";
            }
            return outcome;
        });
}

} // namespace hpoea::pagmo_wrappers
