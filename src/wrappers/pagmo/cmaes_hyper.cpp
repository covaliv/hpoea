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
    d.name = "memory";
    d.type = ParameterType::Boolean;
    d.default_value = false;
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
        seed, configured_parameters_, search_space_,
        [&](pagmo::problem &tuning_problem,
            const auto &bounds,
            const core::Budget &budget,
            std::chrono::steady_clock::time_point start,
            HyperparameterTuningProblem::Context &) -> std::pair<pagmo::population, std::size_t> {

            const auto seed32 = to_seed32(seed);
            const auto dim = bounds.first.size();
            const auto pop_size = static_cast<pagmo::population::size_type>(
                std::max(dim * 4, dim + 1));

            auto generations = clamp_hyper_generations(
                get_param<std::int64_t>(configured_parameters_, "generations"),
                budget, static_cast<std::size_t>(pop_size));
            constexpr auto uint_max = static_cast<std::size_t>(std::numeric_limits<unsigned>::max());
            const auto gen_u = static_cast<unsigned>(std::min(generations, uint_max));

            pagmo::algorithm algorithm{pagmo::cmaes{
                gen_u,
                get_param<double>(configured_parameters_, "cc"),
                get_param<double>(configured_parameters_, "cs"),
                get_param<double>(configured_parameters_, "c1"),
                get_param<double>(configured_parameters_, "cmu"),
                get_param<double>(configured_parameters_, "sigma0"),
                get_param<double>(configured_parameters_, "ftol"),
                get_param<double>(configured_parameters_, "xtol"),
                get_param<bool>(configured_parameters_, "memory"),
                get_param<bool>(configured_parameters_, "force_bounds"),
                seed32}};

            pagmo::population population{tuning_problem, pop_size, derive_seed(seed, 1)};
            if (gen_u > 0) {
                population = algorithm.evolve(population);
            }

            return {std::move(population), generations};
        });
}

} // namespace hpoea::pagmo_wrappers
