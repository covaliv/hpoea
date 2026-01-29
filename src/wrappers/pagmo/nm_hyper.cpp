#include "hpoea/wrappers/pagmo/nm_hyper.hpp"

#include "hyper_tuning_udp.hpp"
#include "hyper_util.hpp"

#include <algorithm>
#include <chrono>
#include <limits>
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

void PagmoNelderMeadHyperOptimizer::configure(const core::ParameterSet &parameters) {
    configured_parameters_ = parameter_space_.apply_defaults(parameters);
}

void PagmoNelderMeadHyperOptimizer::set_search_space(
    std::shared_ptr<core::SearchSpace> search_space) {
  search_space_ = std::move(search_space);
}

core::HyperparameterOptimizationResult PagmoNelderMeadHyperOptimizer::optimize(
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

        auto max_fevals = static_cast<unsigned>(
            std::get<std::int64_t>(configured_parameters_.at("max_fevals")));
        if (budget.function_evaluations)
            max_fevals = std::min<unsigned>(max_fevals, static_cast<unsigned>(*budget.function_evaluations));

        const auto xtol_rel = std::get<double>(configured_parameters_.at("xtol_rel"));
        const auto ftol_rel = std::get<double>(configured_parameters_.at("ftol_rel"));
        const auto seed32 = static_cast<unsigned>(seed & std::numeric_limits<unsigned>::max());

        pagmo::nlopt nm_alg("neldermead");
        nm_alg.set_maxeval(static_cast<int>(max_fevals));
        nm_alg.set_xtol_rel(xtol_rel);
        nm_alg.set_ftol_rel(ftol_rel);
        pagmo::algorithm algorithm{nm_alg};

        const auto dim = lower.size();
        const auto pop_size = static_cast<pagmo::population::size_type>(dim + 1);
        pagmo::population population{tuning_problem, pop_size, seed32};

        const auto t0 = std::chrono::steady_clock::now();
        population = algorithm.evolve(population);
        const auto t1 = std::chrono::steady_clock::now();

        fill_hyper_result(result, *ctx, population, 1, t0, t1, configured_parameters_);
    } catch (const std::exception &ex) {
        result.status = core::RunStatus::InternalError;
        result.message = ex.what();
    }

    return result;
}

} // namespace hpoea::pagmo_wrappers
