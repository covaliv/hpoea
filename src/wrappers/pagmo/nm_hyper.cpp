#include "hpoea/wrappers/pagmo/nm_hyper.hpp"

#include "hpoea/core/evolution_algorithm.hpp"
#include "hpoea/core/parameters.hpp"
#include "hpoea/core/problem.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <pagmo/algorithm.hpp>
#include <pagmo/algorithms/nlopt.hpp>
#include <pagmo/population.hpp>
#include <pagmo/problem.hpp>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <variant>

namespace {

using hpoea::core::AlgorithmIdentity;
using hpoea::core::Budget;
using hpoea::core::BudgetUsage;
using hpoea::core::HyperparameterOptimizationResult;
using hpoea::core::HyperparameterTrialRecord;
using hpoea::core::IEvolutionaryAlgorithmFactory;
using hpoea::core::IProblem;
using hpoea::core::ParameterDescriptor;
using hpoea::core::ParameterSet;
using hpoea::core::ParameterSpace;
using hpoea::core::ParameterType;
using hpoea::core::ParameterValidationError;
using hpoea::core::RunStatus;

constexpr std::string_view kFamily = "NelderMead";
constexpr std::string_view kImplementation = "nlopt::neldermead";
constexpr std::string_view kVersion = "2.x";

ParameterSpace make_parameter_space() {
    ParameterSpace space;

    ParameterDescriptor max_fevals;
    max_fevals.name = "max_fevals";
    max_fevals.type = ParameterType::Integer;
    max_fevals.integer_range = hpoea::core::IntegerRange{1, 100000};
    max_fevals.default_value = static_cast<std::int64_t>(1000);
    space.add_descriptor(max_fevals);

    ParameterDescriptor xtol_rel;
    xtol_rel.name = "xtol_rel";
    xtol_rel.type = ParameterType::Continuous;
    xtol_rel.continuous_range = hpoea::core::ContinuousRange{1e-15, 1e-1};
    xtol_rel.default_value = 1e-8;
    space.add_descriptor(xtol_rel);

    ParameterDescriptor ftol_rel;
    ftol_rel.name = "ftol_rel";
    ftol_rel.type = ParameterType::Continuous;
    ftol_rel.continuous_range = hpoea::core::ContinuousRange{1e-15, 1e-1};
    ftol_rel.default_value = 1e-8;
    space.add_descriptor(ftol_rel);

    return space;
}

AlgorithmIdentity make_identity() {
    AlgorithmIdentity id;
    id.family = std::string{kFamily};
    id.implementation = std::string{kImplementation};
    id.version = std::string{kVersion};
    return id;
}

struct HyperTuningUdp {
    struct Context {
        const IEvolutionaryAlgorithmFactory *factory{nullptr};
        const IProblem *problem{nullptr};
        Budget algorithm_budget;
        unsigned long base_seed{0};
        std::shared_ptr<std::vector<HyperparameterTrialRecord>> trials;
        mutable std::optional<HyperparameterTrialRecord> best_trial;
        mutable std::size_t evaluations{0};
        mutable std::mutex mutex;
    };

    HyperTuningUdp() = default;

    explicit HyperTuningUdp(std::shared_ptr<Context> context) : context_(std::move(context)) {}

    [[nodiscard]] std::pair<pagmo::vector_double, pagmo::vector_double> get_bounds() const {
        const auto &ctx = ensure_context();
        const auto &space = ctx.factory->parameter_space();
        const auto &descriptors = space.descriptors();
        if (descriptors.empty()) {
            throw ParameterValidationError("Algorithm parameter space is empty. Nelder-Mead requires at least one parameter.");
        }

        pagmo::vector_double lower;
        pagmo::vector_double upper;
        lower.reserve(descriptors.size());
        upper.reserve(descriptors.size());

        for (const auto &descriptor : descriptors) {
            switch (descriptor.type) {
            case ParameterType::Continuous: {
                const auto range = descriptor.continuous_range.value_or(hpoea::core::ContinuousRange{-1.0, 1.0});
                lower.push_back(range.lower);
                upper.push_back(range.upper);
                break;
            }
            case ParameterType::Integer: {
                const auto range = descriptor.integer_range.value_or(hpoea::core::IntegerRange{-100, 100});
                lower.push_back(static_cast<double>(range.lower));
                upper.push_back(static_cast<double>(range.upper));
                break;
            }
            case ParameterType::Boolean: {
                lower.push_back(0.0);
                upper.push_back(1.0);
                break;
            }
            case ParameterType::Categorical: {
                lower.push_back(0.0);
                upper.push_back(static_cast<double>(descriptor.categorical_choices.size() - 1));
                break;
            }
            }
        }

        return {lower, upper};
    }

    [[nodiscard]] pagmo::vector_double fitness(const pagmo::vector_double &candidate) const {
        const auto &ctx = ensure_context();
        const auto &space = ctx.factory->parameter_space();
        const auto &descriptors = space.descriptors();

        ParameterSet parameters;
        parameters.reserve(descriptors.size());

        for (std::size_t index = 0; index < descriptors.size(); ++index) {
            const auto &descriptor = descriptors[index];
            const auto value = candidate[index];

            switch (descriptor.type) {
            case ParameterType::Continuous: {
                double numeric = value;
                if (descriptor.continuous_range.has_value()) {
                    numeric = std::clamp(numeric, descriptor.continuous_range->lower, descriptor.continuous_range->upper);
                }
                parameters.emplace(descriptor.name, numeric);
                break;
            }
            case ParameterType::Integer: {
                auto rounded = static_cast<std::int64_t>(std::llround(value));
                if (descriptor.integer_range.has_value()) {
                    rounded = std::clamp(rounded, descriptor.integer_range->lower, descriptor.integer_range->upper);
                }
                parameters.emplace(descriptor.name, rounded);
                break;
            }
            case ParameterType::Boolean: {
                parameters.emplace(descriptor.name, value > 0.5);
                break;
            }
            case ParameterType::Categorical: {
                const auto &choices = descriptor.categorical_choices;
                if (choices.empty()) {
                    throw ParameterValidationError("Categorical descriptor without choices: " + descriptor.name);
                }
                auto index_value = static_cast<std::int64_t>(std::llround(value));
                index_value = std::clamp<std::int64_t>(index_value, 0, static_cast<std::int64_t>(choices.size() - 1));
                parameters.emplace(descriptor.name, choices[static_cast<std::size_t>(index_value)]);
                break;
            }
            }
        }

        parameters = space.apply_defaults(parameters);

        auto algorithm = ctx.factory->create();
        algorithm->configure(parameters);
        unsigned long eval_seed;
        {
            std::scoped_lock lock(ctx.mutex);
            eval_seed = ctx.base_seed + static_cast<unsigned long>(ctx.evaluations++);
        }

        const auto result = algorithm->run(*ctx.problem, ctx.algorithm_budget, eval_seed);

        HyperparameterTrialRecord record;
        record.parameters = parameters;
        record.optimization_result = result;

        {
            std::scoped_lock lock(ctx.mutex);
            if (ctx.trials) {
                ctx.trials->push_back(record);
            }
            const bool should_update_best = !ctx.best_trial
                                            || record.optimization_result.best_fitness
                                                   < ctx.best_trial->optimization_result.best_fitness;
            if (should_update_best) {
                ctx.best_trial = record;
            }
        }

        return pagmo::vector_double{record.optimization_result.best_fitness};
    }

    [[nodiscard]] bool has_gradient() const { return false; }

    [[nodiscard]] bool has_hessians() const { return false; }

    [[nodiscard]] std::string get_name() const { return "HyperTuningUdp"; }

private:
    [[nodiscard]] const Context &ensure_context() const {
        if (!context_) {
            throw std::runtime_error("HyperTuningUdp used without associated context");
        }
        return *context_;
    }

    std::shared_ptr<Context> context_;
};

} // namespace

namespace hpoea::pagmo_wrappers {

PagmoNelderMeadHyperOptimizer::PagmoNelderMeadHyperOptimizer()
    : parameter_space_(make_parameter_space()),
      configured_parameters_(parameter_space_.apply_defaults({})),
      identity_(make_identity()) {}

void PagmoNelderMeadHyperOptimizer::configure(const ParameterSet &parameters) {
    configured_parameters_ = parameter_space_.apply_defaults(parameters);
}

core::HyperparameterOptimizationResult PagmoNelderMeadHyperOptimizer::optimize(
    const core::IEvolutionaryAlgorithmFactory &algorithm_factory,
    const core::IProblem &problem,
    const Budget &budget,
    unsigned long seed) {

    HyperparameterOptimizationResult result;
    result.status = RunStatus::InternalError;
    result.seed = seed;

    try {
        auto context = std::make_shared<HyperTuningUdp::Context>();
        context->factory = &algorithm_factory;
        context->problem = &problem;
        context->algorithm_budget = budget;
        context->base_seed = seed;
        context->trials = std::make_shared<std::vector<HyperparameterTrialRecord>>();

        HyperTuningUdp udp{context};

        const auto [lower, upper] = udp.get_bounds();
        pagmo::problem tuning_problem{udp};

        auto max_fevals = static_cast<unsigned>(std::get<std::int64_t>(configured_parameters_.at("max_fevals")));
        if (budget.function_evaluations.has_value()) {
            max_fevals = std::min<unsigned>(max_fevals, static_cast<unsigned>(budget.function_evaluations.value()));
        }

        const auto xtol_rel = std::get<double>(configured_parameters_.at("xtol_rel"));
        const auto ftol_rel = std::get<double>(configured_parameters_.at("ftol_rel"));

        const auto seed32 = static_cast<unsigned>(seed & std::numeric_limits<unsigned>::max());

        pagmo::nlopt nm_alg("neldermead");
        nm_alg.set_maxeval(static_cast<int>(max_fevals));
        nm_alg.set_xtol_rel(xtol_rel);
        nm_alg.set_ftol_rel(ftol_rel);
        pagmo::algorithm algorithm{nm_alg};

        const auto dimension = lower.size();
        const auto population_size = static_cast<pagmo::population::size_type>(dimension + 1);

        pagmo::population population{tuning_problem, population_size, seed32};

        const auto start_time = std::chrono::steady_clock::now();
        population = algorithm.evolve(population);
        const auto end_time = std::chrono::steady_clock::now();

        result.status = RunStatus::Success;
        result.trials = context->trials ? std::move(*context->trials)
                                        : std::vector<HyperparameterTrialRecord>{};

        if (context->best_trial.has_value()) {
            result.best_parameters = context->best_trial->parameters;
            result.best_objective = context->best_trial->optimization_result.best_fitness;
        } else {
            result.best_objective = population.champion_f()[0];
        }

        result.budget_usage.wall_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        result.budget_usage.generations = 1;
        result.budget_usage.function_evaluations = context->evaluations;
        result.effective_optimizer_parameters = configured_parameters_;
        result.message = "Hyperparameter optimization completed.";
    } catch (const std::exception &ex) {
        result.status = RunStatus::InternalError;
        result.message = ex.what();
    }

    return result;
}

} // namespace hpoea::pagmo_wrappers

