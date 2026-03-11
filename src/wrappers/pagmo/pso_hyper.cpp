#include "hpoea/wrappers/pagmo/pso_hyper.hpp"

#include "budget_util.hpp"
#include "hyper_util.hpp"

#include <algorithm>
#include <limits>
#include <pagmo/algorithms/pso.hpp>

namespace {

using hpoea::core::AlgorithmIdentity;
using hpoea::core::ParameterDescriptor;
using hpoea::core::ParameterSpace;
using hpoea::core::ParameterType;

ParameterSpace make_parameter_space() {
    ParameterSpace space;

    ParameterDescriptor d;
    d.name = "variant";
    d.type = ParameterType::Integer;
    d.integer_range = hpoea::core::IntegerRange{1, 6};
    d.default_value = std::int64_t{5};
    space.add_descriptor(d);

    d = {};
    d.name = "generations";
    d.type = ParameterType::Integer;
    d.integer_range = hpoea::core::IntegerRange{1, 1000};
    d.default_value = std::int64_t{100};
    space.add_descriptor(d);

    d = {};
    d.name = "omega";
    d.type = ParameterType::Continuous;
    d.continuous_range = hpoea::core::ContinuousRange{0.0, 1.0};
    d.default_value = 0.7298;
    space.add_descriptor(d);

    d = {};
    d.name = "eta1";
    d.type = ParameterType::Continuous;
    d.continuous_range = hpoea::core::ContinuousRange{1.0, 3.0};
    d.default_value = 2.05;
    space.add_descriptor(d);

    d = {};
    d.name = "eta2";
    d.type = ParameterType::Continuous;
    d.continuous_range = hpoea::core::ContinuousRange{1.0, 3.0};
    d.default_value = 2.05;
    space.add_descriptor(d);

    d = {};
    d.name = "max_velocity";
    d.type = ParameterType::Continuous;
    d.continuous_range = hpoea::core::ContinuousRange{0.0, 100.0};
    d.default_value = 0.5;
    space.add_descriptor(d);

    return space;
}

AlgorithmIdentity make_identity() {
    return {"PSOHyperOptimizer", "pagmo::pso", "2.x"};
}

} // namespace

namespace hpoea::pagmo_wrappers {

PagmoPsoHyperOptimizer::PagmoPsoHyperOptimizer()
    : PagmoHyperOptimizerBase(make_parameter_space(), make_identity()) {}

core::HyperparameterOptimizerPtr PagmoPsoHyperOptimizer::clone() const {
    return std::make_unique<PagmoPsoHyperOptimizer>(*this);
}

core::HyperparameterOptimizationResult PagmoPsoHyperOptimizer::optimize(
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

            const auto omega = std::get<double>(configured_parameters_.at("omega"));
            const auto eta1 = std::get<double>(configured_parameters_.at("eta1"));
            const auto eta2 = std::get<double>(configured_parameters_.at("eta2"));
            const auto max_vel = std::get<double>(configured_parameters_.at("max_velocity"));

            constexpr auto uint_max = static_cast<std::size_t>(std::numeric_limits<unsigned>::max());

            const auto variant_i64 = std::get<std::int64_t>(configured_parameters_.at("variant"));
            const auto variant = static_cast<unsigned>(
                std::min(static_cast<std::size_t>(variant_i64), uint_max));

            const auto seed32 = to_seed32(seed);
            const auto dim = bounds.first.size();
            const auto pop_size = static_cast<pagmo::population::size_type>(
                std::max(dim * 4, dim + 1));

            // compute generations from configured value, clamped by budget
            auto generations = static_cast<std::size_t>(
                std::get<std::int64_t>(configured_parameters_.at("generations")));
            if (budget.generations) {
                generations = std::min(generations, *budget.generations);
            }
            if (budget.function_evaluations) {
                const auto ps = std::max<std::size_t>(static_cast<std::size_t>(pop_size), 1);
                auto available = *budget.function_evaluations > ps
                    ? *budget.function_evaluations - ps : std::size_t{0};
                generations = std::min(generations, available / ps);
            }

            // saturate to unsigned
            const auto gen_u = static_cast<unsigned>(std::min(generations, uint_max));

            pagmo::algorithm algorithm{
                pagmo::pso(gen_u, omega, eta1, eta2, max_vel, variant, 2u, 4u, false, seed32)};

            pagmo::population population{tuning_problem, pop_size, derive_seed(seed, 1)};
            if (gen_u > 0) {
                population = algorithm.evolve(population);
            }

            return {std::move(population), generations};
        });
}

} // namespace hpoea::pagmo_wrappers
