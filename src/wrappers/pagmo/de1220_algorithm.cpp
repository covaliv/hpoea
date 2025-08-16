#include "hpoea/wrappers/pagmo/de1220_algorithm.hpp"

#include "hpoea/core/problem.hpp"

#include <algorithm>
#include <chrono>
#include <limits>
#include <pagmo/algorithm.hpp>
#include <pagmo/algorithms/de1220.hpp>
#include <pagmo/population.hpp>
#include <pagmo/problem.hpp>
#include <vector>

#include "problem_adapter.hpp"

namespace {

using hpoea::core::AlgorithmIdentity;
using hpoea::core::Budget;
using hpoea::core::OptimizationResult;
using hpoea::core::ParameterDescriptor;
using hpoea::core::ParameterSet;
using hpoea::core::ParameterSpace;
using hpoea::core::ParameterType;

constexpr std::string_view kFamily = "DE1220";
constexpr std::string_view kImplementation = "pagmo::de1220";
constexpr std::string_view kVersion = "2.x";

// Note: pagmo::de1220 (also called pDE) is a self-adaptive DE variant
// that is different from jDE. jDE (Brest et al.) is implemented in
// pagmo::sade with variant_adptv=1.

ParameterSpace make_parameter_space() {
    ParameterSpace space;

    ParameterDescriptor population_size;
    population_size.name = "population_size";
    population_size.type = ParameterType::Integer;
    population_size.integer_range = hpoea::core::IntegerRange{5, 5000};
    population_size.default_value = static_cast<std::int64_t>(50);
    population_size.required = true;
    space.add_descriptor(population_size);

    ParameterDescriptor generations;
    generations.name = "generations";
    generations.type = ParameterType::Integer;
    generations.integer_range = hpoea::core::IntegerRange{1, 100000};
    generations.default_value = static_cast<std::int64_t>(200);
    space.add_descriptor(generations);

    ParameterDescriptor ftol;
    ftol.name = "ftol";
    ftol.type = ParameterType::Continuous;
    ftol.continuous_range = hpoea::core::ContinuousRange{0.0, 1.0};
    ftol.default_value = 1e-6;
    space.add_descriptor(ftol);

    ParameterDescriptor xtol;
    xtol.name = "xtol";
    xtol.type = ParameterType::Continuous;
    xtol.continuous_range = hpoea::core::ContinuousRange{0.0, 1.0};
    xtol.default_value = 1e-6;
    space.add_descriptor(xtol);

    ParameterDescriptor variant_adaptation;
    variant_adaptation.name = "variant_adaptation";
    variant_adaptation.type = ParameterType::Integer;
    variant_adaptation.integer_range = hpoea::core::IntegerRange{1, 2};
    variant_adaptation.default_value = static_cast<std::int64_t>(1);
    space.add_descriptor(variant_adaptation);

    ParameterDescriptor memory;
    memory.name = "memory";
    memory.type = ParameterType::Boolean;
    memory.default_value = false;
    space.add_descriptor(memory);

    return space;
}

AlgorithmIdentity make_identity() {
    AlgorithmIdentity id;
    id.family = std::string{kFamily};
    id.implementation = std::string{kImplementation};
    id.version = std::string{kVersion};
    return id;
}

std::size_t determine_generations(const ParameterSet &parameters, const Budget &budget, std::size_t population) {
    std::size_t generations = static_cast<std::size_t>(std::get<std::int64_t>(parameters.at("generations")));

    if (budget.generations.has_value()) {
        generations = std::min(generations, budget.generations.value());
    }

    if (budget.function_evaluations.has_value()) {
        const std::size_t max_generations = budget.function_evaluations.value() / std::max<std::size_t>(population, 1);
        if (max_generations > 0) {
            generations = std::min(generations, max_generations);
        }
    }

    return std::max<std::size_t>(generations, 1);
}

} // namespace

namespace hpoea::pagmo_wrappers {

PagmoDe1220::PagmoDe1220()
    : parameter_space_(make_parameter_space()),
      configured_parameters_(parameter_space_.apply_defaults({})),
      identity_(make_identity()) {}

void PagmoDe1220::configure(const ParameterSet &parameters) {
    configured_parameters_ = parameter_space_.apply_defaults(parameters);
}

core::OptimizationResult PagmoDe1220::run(const core::IProblem &problem,
                                       const core::Budget &budget,
                                       unsigned long seed) {
    OptimizationResult result;
    result.status = core::RunStatus::InternalError;
    result.seed = seed;

    try {
        const auto population_size = static_cast<std::size_t>(std::get<std::int64_t>(configured_parameters_.at("population_size")));
        const auto ftol = std::get<double>(configured_parameters_.at("ftol"));
        const auto xtol = std::get<double>(configured_parameters_.at("xtol"));
        const auto variant_adaptation = static_cast<unsigned>(std::get<std::int64_t>(configured_parameters_.at("variant_adaptation")));
        const auto memory = std::get<bool>(configured_parameters_.at("memory"));

        auto effective_parameters = configured_parameters_;
        const auto generations = determine_generations(configured_parameters_, budget, population_size);
        effective_parameters.insert_or_assign("generations", static_cast<std::int64_t>(generations));

        std::vector<unsigned> allowed_variants = pagmo::de1220_statics<void>::allowed_variants;
        pagmo::algorithm algorithm{pagmo::de1220(static_cast<unsigned>(generations), allowed_variants, variant_adaptation, ftol, xtol, memory)};
        pagmo::problem pg_problem{ProblemAdapter{problem}};

        const auto population_seed = static_cast<unsigned int>(seed & std::numeric_limits<unsigned int>::max());
        pagmo::population population{pg_problem, population_size, population_seed};

        const auto start_time = std::chrono::steady_clock::now();
        population = algorithm.evolve(population);
        const auto end_time = std::chrono::steady_clock::now();

        result.status = core::RunStatus::Success;
        result.best_fitness = population.champion_f()[0];
        result.best_solution = population.champion_x();
        result.budget_usage.function_evaluations = population_size * generations;
        result.budget_usage.generations = generations;
        result.budget_usage.wall_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        result.effective_parameters = std::move(effective_parameters);
        result.message = "Optimization completed.";
    } catch (const std::exception &ex) {
        result.status = core::RunStatus::InternalError;
        result.message = ex.what();
    }

    return result;
}

std::unique_ptr<core::IEvolutionaryAlgorithm> PagmoDe1220::clone() const {
    return std::make_unique<PagmoDe1220>(*this);
}

PagmoDe1220Factory::PagmoDe1220Factory()
    : parameter_space_(make_parameter_space()),
      identity_(make_identity()) {}

core::EvolutionaryAlgorithmPtr PagmoDe1220Factory::create() const {
    return std::make_unique<PagmoDe1220>();
}

} // namespace hpoea::pagmo_wrappers


