#include "hpoea/wrappers/pagmo/sa_hyper.hpp"

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

void PagmoSimulatedAnnealingHyperOptimizer::configure(const core::ParameterSet &parameters) {
    configured_parameters_ = parameter_space_.apply_defaults(parameters);
}

void PagmoSimulatedAnnealingHyperOptimizer::set_search_space(
    std::shared_ptr<core::SearchSpace> search_space) {
  search_space_ = std::move(search_space);
}

core::HyperparameterOptimizationResult
PagmoSimulatedAnnealingHyperOptimizer::optimize(
    const core::IEvolutionaryAlgorithmFactory &algorithm_factory,
    const core::IProblem &problem, const core::Budget &budget, unsigned long seed) {

    core::HyperparameterOptimizationResult result;
    result.status = core::RunStatus::InternalError;
    result.seed = seed;

    try {
        auto ctx = make_hyper_context(algorithm_factory, problem, budget, seed, search_space_);
        HyperTuningUdp udp{ctx};

        const auto [lower, upper] = udp.get_bounds();
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

        const auto seed32 = static_cast<unsigned>(seed & std::numeric_limits<unsigned>::max());

        pagmo::simulated_annealing sa_alg(ts, tf, n_T_adj, n_range_adj, bin_size, start_range, seed32);
        pagmo::algorithm algorithm{sa_alg};
        pagmo::population population{tuning_problem, 1, seed32};

        auto iterations = static_cast<unsigned>(
            std::get<std::int64_t>(configured_parameters_.at("iterations")));
        const auto dim = lower.size();
        const auto evals_per_evolve = n_T_adj * n_range_adj * bin_size * dim;

        if (budget.function_evaluations) {
            auto max_evolves = static_cast<unsigned>(
                *budget.function_evaluations / std::max<std::size_t>(evals_per_evolve, 1));
            iterations = std::min(iterations, max_evolves);
        }

        const auto t0 = std::chrono::steady_clock::now();
        unsigned actual_iterations = 0;
        for (unsigned i = 0; i < iterations; ++i) {
            population = algorithm.evolve(population);
            ++actual_iterations;
            if (budget.function_evaluations && ctx->get_evaluations() >= *budget.function_evaluations)
                break;
        }
        const auto t1 = std::chrono::steady_clock::now();

        fill_hyper_result(result, *ctx, population, actual_iterations, t0, t1, configured_parameters_);
    } catch (const std::exception &ex) {
        result.status = core::RunStatus::InternalError;
        result.message = ex.what();
    }

    return result;
}

} // namespace hpoea::pagmo_wrappers
