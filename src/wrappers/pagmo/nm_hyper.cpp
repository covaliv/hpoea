#include "hpoea/wrappers/pagmo/nm_hyper.hpp"

#include "budget_util.hpp"
#include "hyper_tuning_udp.hpp"
#include "hyper_util.hpp"

#include <algorithm>
#include <chrono>
#include <pagmo/algorithm.hpp>
#include <pagmo/algorithms/nlopt.hpp>
#include <pagmo/population.hpp>
#include <pagmo/problem.hpp>

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
    : parameter_space_(make_parameter_space()),
      configured_parameters_(parameter_space_.apply_defaults({})),
      identity_(make_identity()) {}

core::HyperparameterOptimizerPtr PagmoNelderMeadHyperOptimizer::clone() const {
    return std::make_unique<PagmoNelderMeadHyperOptimizer>(*this);
}

void PagmoNelderMeadHyperOptimizer::configure(const core::ParameterSet &parameters) {
    configured_parameters_ = parameter_space_.apply_defaults(parameters);
    parameter_space_.validate(configured_parameters_);
}

void PagmoNelderMeadHyperOptimizer::set_search_space(
    std::shared_ptr<core::SearchSpace> search_space) {
    search_space_ = std::move(search_space);
}

core::HyperparameterOptimizationResult PagmoNelderMeadHyperOptimizer::optimize(
    const core::IEvolutionaryAlgorithmFactory &algorithm_factory,
    const core::IProblem &problem,
    const core::Budget &optimizer_budget,
    const core::Budget &algorithm_budget,
    unsigned long seed) {

    core::HyperparameterOptimizationResult result;
    result.status = core::RunStatus::InternalError;
    result.seed = seed;

    std::shared_ptr<HyperparameterTuningProblem::Context> ctx;
    const auto start_time = std::chrono::steady_clock::now();

    try {
        if (search_space_) {
            search_space_->validate(algorithm_factory.parameter_space());
        }

        ctx = make_hyper_context(algorithm_factory, problem, algorithm_budget, seed, search_space_);
        HyperparameterTuningProblem udp{ctx};

        const auto bounds = udp.get_bounds();
        pagmo::problem tuning_problem{udp};

        auto max_fevals = static_cast<unsigned>(
            std::get<std::int64_t>(configured_parameters_.at("max_fevals")));
        if (optimizer_budget.function_evaluations)
            max_fevals = std::min<unsigned>(max_fevals, static_cast<unsigned>(*optimizer_budget.function_evaluations));

        const auto xtol_rel = std::get<double>(configured_parameters_.at("xtol_rel"));
        const auto ftol_rel = std::get<double>(configured_parameters_.at("ftol_rel"));

        pagmo::nlopt nm_alg("neldermead");
        nm_alg.set_maxeval(static_cast<int>(max_fevals));
        nm_alg.set_xtol_rel(xtol_rel);
        nm_alg.set_ftol_rel(ftol_rel);
        pagmo::algorithm algorithm{nm_alg};

        const auto dim = bounds.first.size();
        const auto pop_size = static_cast<pagmo::population::size_type>(dim + 1);
        const auto t0 = start_time;
        pagmo::population population{tuning_problem, pop_size, derive_seed(seed, 1)};

        const bool skip_evolve = optimizer_budget.generations.has_value() && *optimizer_budget.generations == 0;
        if (!skip_evolve) {
            population = algorithm.evolve(population);
        }
        const auto t1 = std::chrono::steady_clock::now();

        const auto actual_generations = skip_evolve ? 0u : 1u;
        fill_hyper_result(result, *ctx, population, actual_generations, t0, t1, configured_parameters_);
        apply_budget_status(
            optimizer_budget,
            result.budget_usage,
            result.status,
            result.message);
    } catch (const core::ParameterValidationError &ex) {
        const auto end_time = std::chrono::steady_clock::now();
        result.budget_usage.wall_time =
            std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        result.status = core::RunStatus::InvalidConfiguration;
        result.message = ex.what();
    } catch (const std::invalid_argument &ex) {
        const auto end_time = std::chrono::steady_clock::now();
        result.budget_usage.wall_time =
            std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        result.status = core::RunStatus::InvalidConfiguration;
        result.message = ex.what();
    } catch (const std::exception &ex) {
        const auto end_time = std::chrono::steady_clock::now();
        result.budget_usage.wall_time =
            std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        result.status = core::RunStatus::InternalError;
        result.message = ex.what();
    }

    if (ctx && ctx->trials && !ctx->trials->empty() && result.trials.empty()) {
        result.trials = std::move(*ctx->trials);
        auto it = std::min_element(result.trials.begin(), result.trials.end(),
            [](const auto &a, const auto &b) {
                return a.optimization_result.best_fitness < b.optimization_result.best_fitness;
            });
        result.best_parameters = it->parameters;
        result.best_objective = it->optimization_result.best_fitness;
    }

    return result;
}

} // namespace hpoea::pagmo_wrappers
