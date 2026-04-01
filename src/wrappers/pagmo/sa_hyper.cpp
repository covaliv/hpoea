#include "hpoea/wrappers/pagmo/sa_hyper.hpp"

#include "budget_util.hpp"
#include "hyper_util.hpp"

#include <algorithm>
#include <limits>
#include <pagmo/algorithms/simulated_annealing.hpp>

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

// saturating multiply for size_t
std::size_t mul_sat(std::size_t a, std::size_t b) {
    if (a == 0 || b == 0) return 0;
    constexpr auto max_sz = std::numeric_limits<std::size_t>::max();
    if (a > max_sz / b) return max_sz;
    return a * b;
}

} // namespace

namespace hpoea::pagmo_wrappers {

PagmoSimulatedAnnealingHyperOptimizer::PagmoSimulatedAnnealingHyperOptimizer()
    : PagmoHyperOptimizerBase(make_parameter_space(), make_identity()) {}

core::HyperparameterOptimizerPtr PagmoSimulatedAnnealingHyperOptimizer::clone() const {
    return std::make_unique<PagmoSimulatedAnnealingHyperOptimizer>(*this);
}

core::HyperparameterOptimizationResult
PagmoSimulatedAnnealingHyperOptimizer::optimize(
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
            HyperparameterTuningProblem::Context &ctx) -> std::pair<pagmo::population, std::size_t> {

            const auto ts = get_param<double>(configured_parameters_, "ts");
            const auto tf = get_param<double>(configured_parameters_, "tf");
            const auto start_range = get_param<double>(configured_parameters_, "start_range");

            constexpr auto uint_max = static_cast<std::size_t>(std::numeric_limits<unsigned>::max());

            const auto n_T_adj = static_cast<unsigned>(
                std::min(get_param<std::int64_t>(configured_parameters_, "n_T_adj"), uint_max));

            const auto n_range_adj = static_cast<unsigned>(
                std::min(get_param<std::int64_t>(configured_parameters_, "n_range_adj"), uint_max));

            const auto bin_size = static_cast<unsigned>(
                std::min(get_param<std::int64_t>(configured_parameters_, "bin_size"), uint_max));

            const auto seed32 = to_seed32(seed);

            pagmo::simulated_annealing sa_alg(ts, tf, n_T_adj, n_range_adj, bin_size, start_range, seed32);
            pagmo::algorithm algorithm{sa_alg};

            pagmo::population population{tuning_problem, 1, derive_seed(seed, 1)};

            auto iterations = static_cast<unsigned>(
                std::min(get_param<std::int64_t>(configured_parameters_, "iterations"), uint_max));

            if (budget.generations) {
                iterations = std::min(iterations,
                    static_cast<unsigned>(std::min(*budget.generations, uint_max)));
            }

            // pagmo SA evaluates n_T_adj * n_range_adj * bin_size * dimension per evolve
            const auto dim = bounds.first.size();
            std::size_t evals_per_evolve = mul_sat(
                mul_sat(mul_sat(static_cast<std::size_t>(n_T_adj),
                                static_cast<std::size_t>(n_range_adj)),
                        static_cast<std::size_t>(bin_size)),
                dim);

            if (budget.function_evaluations) {
                auto max_evolves = static_cast<unsigned>(std::min(
                    *budget.function_evaluations / std::max<std::size_t>(evals_per_evolve, 1),
                    uint_max));
                iterations = std::min(iterations, max_evolves);
            }

            unsigned actual_iterations = 0;
            for (unsigned i = 0; i < iterations; ++i) {
                population = algorithm.evolve(population);
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

            return {std::move(population), static_cast<std::size_t>(actual_iterations)};
        });
}

} // namespace hpoea::pagmo_wrappers
