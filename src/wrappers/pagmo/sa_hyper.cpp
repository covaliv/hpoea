#include "hpoea/wrappers/pagmo/sa_hyper.hpp"

#include "budget_util.hpp"
#include "hyper_tuning_udp.hpp"
#include "hyper_util.hpp"

#include <algorithm>
#include <chrono>
#include <limits>
#include <pagmo/algorithm.hpp>
#include <pagmo/algorithms/simulated_annealing.hpp>
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
    d.name = "iterations";
    d.type = ParameterType::Integer;
    d.integer_range = hpoea::core::IntegerRange{1, 100000};
    d.default_value = std::int64_t{1000};
    space.add_descriptor(d);

    d = {};
    d.name = "ts";
    d.type = ParameterType::Continuous;
    d.continuous_range = hpoea::core::ContinuousRange{1e-6, 100.0};
    d.default_value = 10.0;
    space.add_descriptor(d);

    d = {};
    d.name = "tf";
    d.type = ParameterType::Continuous;
    d.continuous_range = hpoea::core::ContinuousRange{1e-6, 100.0};
    d.default_value = 0.1;
    space.add_descriptor(d);

    d = {};
    d.name = "n_T_adj";
    d.type = ParameterType::Integer;
    d.integer_range = hpoea::core::IntegerRange{1, 10000};
    d.default_value = std::int64_t{10};
    space.add_descriptor(d);

    d = {};
    d.name = "n_range_adj";
    d.type = ParameterType::Integer;
    d.integer_range = hpoea::core::IntegerRange{1, 10000};
    d.default_value = std::int64_t{1};
    space.add_descriptor(d);

    d = {};
    d.name = "bin_size";
    d.type = ParameterType::Integer;
    d.integer_range = hpoea::core::IntegerRange{1, 1000};
    d.default_value = std::int64_t{10};
    space.add_descriptor(d);

    d = {};
    d.name = "start_range";
    d.type = ParameterType::Continuous;
    d.continuous_range = hpoea::core::ContinuousRange{0.0, 1.0};
    d.default_value = 1.0;
    space.add_descriptor(d);

    return space;
}

AlgorithmIdentity make_identity() {
    return {"SimulatedAnnealing", "pagmo::simulated_annealing", "2.x"};
}

} // namespace

namespace hpoea::pagmo_wrappers {

PagmoSimulatedAnnealingHyperOptimizer::PagmoSimulatedAnnealingHyperOptimizer()
    : parameter_space_(make_parameter_space()),
      configured_parameters_(parameter_space_.apply_defaults({})),
      identity_(make_identity()) {}

core::HyperparameterOptimizerPtr PagmoSimulatedAnnealingHyperOptimizer::clone() const {
    return std::make_unique<PagmoSimulatedAnnealingHyperOptimizer>(*this);
}

void PagmoSimulatedAnnealingHyperOptimizer::configure(const core::ParameterSet &parameters) {
    configured_parameters_ = parameter_space_.apply_defaults(parameters);
    parameter_space_.validate(configured_parameters_);
}

void PagmoSimulatedAnnealingHyperOptimizer::set_search_space(
    std::shared_ptr<core::SearchSpace> search_space) {
    search_space_ = std::move(search_space);
}

core::HyperparameterOptimizationResult
PagmoSimulatedAnnealingHyperOptimizer::optimize(
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

        const auto ts = std::get<double>(configured_parameters_.at("ts"));
        const auto tf = std::get<double>(configured_parameters_.at("tf"));
        const auto n_T_adj = static_cast<unsigned>(
            std::get<std::int64_t>(configured_parameters_.at("n_T_adj")));
        const auto n_range_adj = static_cast<unsigned>(
            std::get<std::int64_t>(configured_parameters_.at("n_range_adj")));
        const auto bin_size = static_cast<unsigned>(
            std::get<std::int64_t>(configured_parameters_.at("bin_size")));
        const auto start_range = std::get<double>(configured_parameters_.at("start_range"));

        const auto seed32 = to_seed32(seed);

        pagmo::simulated_annealing sa_alg(ts, tf, n_T_adj, n_range_adj, bin_size, start_range, seed32);
        pagmo::algorithm algorithm{sa_alg};

        const auto t0 = start_time;
        pagmo::population population{tuning_problem, 1, derive_seed(seed, 1)};

        auto iterations = static_cast<unsigned>(
            std::get<std::int64_t>(configured_parameters_.at("iterations")));
        if (optimizer_budget.generations) {
            iterations = std::min<unsigned>(iterations, static_cast<unsigned>(*optimizer_budget.generations));
        }
        std::size_t evals_per_evolve = 1;
        const auto max_sz = std::numeric_limits<std::size_t>::max();
        const auto mul_sat = [max_sz](std::size_t a, std::size_t b) -> std::size_t {
            if (a == 0 || b == 0) {
                return 0;
            }
            if (a > max_sz / b) {
                return max_sz;
            }
            return a * b;
        };
        evals_per_evolve = mul_sat(evals_per_evolve, static_cast<std::size_t>(n_T_adj));
        evals_per_evolve = mul_sat(evals_per_evolve, static_cast<std::size_t>(n_range_adj));
        evals_per_evolve = mul_sat(evals_per_evolve, static_cast<std::size_t>(bin_size));

        if (optimizer_budget.function_evaluations) {
            auto max_evolves = static_cast<unsigned>(
                *optimizer_budget.function_evaluations / std::max<std::size_t>(evals_per_evolve, 1));
            iterations = std::min(iterations, max_evolves);
        }

        unsigned actual_iterations = 0;
        for (unsigned i = 0; i < iterations; ++i) {
            population = algorithm.evolve(population);
            ++actual_iterations;
            if (optimizer_budget.wall_time) {
                const auto now = std::chrono::steady_clock::now();
                const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - t0);
                if (elapsed > *optimizer_budget.wall_time)
                    break;
            }
            if (optimizer_budget.function_evaluations &&
                ctx->get_evaluations() >= *optimizer_budget.function_evaluations)
                break;
        }
        const auto t1 = std::chrono::steady_clock::now();

        fill_hyper_result(result, *ctx, population, actual_iterations, t0, t1, configured_parameters_);
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
