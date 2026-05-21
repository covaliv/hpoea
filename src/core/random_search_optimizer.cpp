#include "hpoea/core/random_search_optimizer.hpp"

#include "hpoea/core/budget_checks.hpp"
#include "hpoea/core/error_classification.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>

namespace {

constexpr const char *SAMPLE_COUNT = "sample_count";

hpoea::core::ParameterSpace make_parameter_space() {
    hpoea::core::ParameterSpace space;

    hpoea::core::ParameterDescriptor d;
    d.name = SAMPLE_COUNT;
    d.type = hpoea::core::ParameterType::Integer;
    d.integer_range = hpoea::core::IntegerRange{1, 100000};
    d.default_value = std::int64_t{100};
    space.add_descriptor(d);

    return space;
}

std::size_t get_sample_count(const hpoea::core::ParameterSet &parameters) {
    const auto it = parameters.find(SAMPLE_COUNT);
    if (it == parameters.end()) {
        throw std::invalid_argument("missing parameter: sample_count");
    }
    if (!std::holds_alternative<std::int64_t>(it->second)) {
        throw std::invalid_argument("parameter 'sample_count' type mismatch");
    }
    const auto value = std::get<std::int64_t>(it->second);
    if (value < 0) {
        throw std::invalid_argument("parameter 'sample_count' cannot be negative");
    }
    return static_cast<std::size_t>(value);
}

unsigned long splitmix64(unsigned long x) {
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9UL;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebUL;
    x ^= x >> 31;
    return x;
}

unsigned long derive_trial_seed(unsigned long seed, std::size_t trial_index) {
    const auto salt = static_cast<unsigned long>(trial_index + 1u);
    return splitmix64(seed ^ (salt * 0x9e3779b97f4a7c15UL));
}

const hpoea::core::ParameterConfig *find_config(const hpoea::core::SearchSpace *search_space, const std::string &name) {
    if (!search_space) {
        return nullptr;
    }
    return search_space->get(name);
}

bool is_excluded(const hpoea::core::ParameterConfig *config) {
    return config && config->mode == hpoea::core::SearchMode::exclude;
}

bool is_tunable(const hpoea::core::ParameterConfig *config) {
    return !config || config->mode == hpoea::core::SearchMode::optimize;
}

bool has_tunable_dimension(const hpoea::core::ParameterSpace &space, const hpoea::core::SearchSpace *search_space) {
    for (const auto &descriptor : space.descriptors()) {
        if (is_tunable(find_config(search_space, descriptor.name))) {
            return true;
        }
    }
    return false;
}

hpoea::core::ContinuousRange resolve_continuous_range(const hpoea::core::ParameterDescriptor &descriptor,
                                                      const hpoea::core::ParameterConfig *config) {
    if (!descriptor.continuous_range.has_value()) {
        throw std::logic_error("continuous parameter missing range: " + descriptor.name);
    }
    if (config && config->continuous_bounds.has_value()) {
        return config->continuous_bounds.value();
    }
    return descriptor.continuous_range.value();
}

hpoea::core::IntegerRange resolve_integer_range(const hpoea::core::ParameterDescriptor &descriptor,
                                                const hpoea::core::ParameterConfig *config) {
    if (!descriptor.integer_range.has_value()) {
        throw std::logic_error("integer parameter missing range: " + descriptor.name);
    }
    if (config && config->integer_bounds.has_value()) {
        return config->integer_bounds.value();
    }
    return descriptor.integer_range.value();
}

template <typename Rng>
hpoea::core::ParameterValue sample_choice(const std::vector<hpoea::core::ParameterValue> &choices, Rng &rng) {
    if (choices.empty()) {
        throw hpoea::core::ParameterValidationError("discrete choices cannot be empty");
    }
    std::uniform_int_distribution<std::size_t> dist{0, choices.size() - 1};
    return choices[dist(rng)];
}

template <typename Rng>
hpoea::core::ParameterValue sample_continuous(const hpoea::core::ParameterDescriptor &descriptor,
                                              const hpoea::core::ParameterConfig *config, Rng &rng) {
    const auto range = resolve_continuous_range(descriptor, config);
    const auto transform = config ? config->transform : hpoea::core::Transform::none;
    const auto transformed = hpoea::core::transform_bounds(range, transform);
    std::uniform_real_distribution<double> dist{transformed.lower, transformed.upper};
    const auto transformed_value = dist(rng);
    const auto numeric = hpoea::core::inverse_transform(transformed_value, transform);
    return std::clamp(numeric, range.lower, range.upper);
}

template <typename Rng>
hpoea::core::ParameterValue sample_integer(const hpoea::core::ParameterDescriptor &descriptor,
                                           const hpoea::core::ParameterConfig *config, Rng &rng) {
    if (config && !config->discrete_choices.empty()) {
        return sample_choice(config->discrete_choices, rng);
    }
    const auto range = resolve_integer_range(descriptor, config);
    std::uniform_int_distribution<std::int64_t> dist{range.lower, range.upper};
    return dist(rng);
}

template <typename Rng>
hpoea::core::ParameterValue sample_categorical(const hpoea::core::ParameterDescriptor &descriptor,
                                               const hpoea::core::ParameterConfig *config, Rng &rng) {
    if (config && !config->discrete_choices.empty()) {
        return sample_choice(config->discrete_choices, rng);
    }
    if (descriptor.categorical_choices.empty()) {
        throw hpoea::core::ParameterValidationError("Categorical descriptor without choices: " + descriptor.name);
    }
    std::uniform_int_distribution<std::size_t> dist{0, descriptor.categorical_choices.size() - 1};
    return std::string{descriptor.categorical_choices[dist(rng)]};
}

template <typename Rng>
hpoea::core::ParameterValue sample_value(const hpoea::core::ParameterDescriptor &descriptor,
                                         const hpoea::core::ParameterConfig *config, Rng &rng) {
    switch (descriptor.type) {
    case hpoea::core::ParameterType::Continuous:
        return sample_continuous(descriptor, config, rng);
    case hpoea::core::ParameterType::Integer:
        return sample_integer(descriptor, config, rng);
    case hpoea::core::ParameterType::Boolean: {
        std::bernoulli_distribution dist{0.5};
        return dist(rng);
    }
    case hpoea::core::ParameterType::Categorical:
        return sample_categorical(descriptor, config, rng);
    }
    throw std::logic_error("unhandled ParameterType value");
}

template <typename Rng>
hpoea::core::ParameterSet sample_parameters(const hpoea::core::ParameterSpace &space,
                                            const hpoea::core::SearchSpace *search_space, Rng &rng) {
    hpoea::core::ParameterSet parameters;
    parameters.reserve(space.size());

    for (const auto &descriptor : space.descriptors()) {
        const auto *config = find_config(search_space, descriptor.name);
        if (config && config->mode == hpoea::core::SearchMode::fixed) {
            if (config->fixed_value.has_value()) {
                parameters.emplace(descriptor.name, config->fixed_value.value());
            }
            continue;
        }
        if (is_excluded(config)) {
            continue;
        }
        parameters.emplace(descriptor.name, sample_value(descriptor, config, rng));
    }

    for (const auto &descriptor : space.descriptors()) {
        if (parameters.contains(descriptor.name)) {
            continue;
        }
        const auto *config = find_config(search_space, descriptor.name);
        if (is_excluded(config)) {
            continue;
        }
        if (descriptor.default_value.has_value()) {
            parameters.emplace(descriptor.name, descriptor.default_value.value());
        }
    }

    for (const auto &[name, value] : parameters) {
        space.validate_value(space.descriptor(name), value);
    }
    for (const auto &descriptor : space.descriptors()) {
        if (descriptor.required && !parameters.contains(descriptor.name) &&
            !is_excluded(find_config(search_space, descriptor.name))) {
            throw hpoea::core::ParameterValidationError("missing required parameter: " + descriptor.name);
        }
    }

    return parameters;
}

void mark_non_finite_success(hpoea::core::OptimizationResult &result) {
    if (result.status != hpoea::core::RunStatus::Success || std::isfinite(result.best_fitness)) {
        return;
    }
    result.status = hpoea::core::RunStatus::InternalError;
    result.error_info =
        hpoea::core::ErrorInfo{"internal_error", "non_finite_objective", "algorithm returned non-finite objective"};
    if (result.message.empty()) {
        result.message = "algorithm returned non-finite objective";
    }
}

bool is_selectable_trial(const hpoea::core::HyperparameterTrialRecord &trial) {
    const auto status = trial.optimization_result.status;
    return (status == hpoea::core::RunStatus::Success || status == hpoea::core::RunStatus::BudgetExceeded) &&
           std::isfinite(trial.optimization_result.best_fitness);
}

void fill_success_result(hpoea::core::HyperparameterOptimizationResult &result) {
    auto best = result.trials.end();
    for (auto it = result.trials.begin(); it != result.trials.end(); ++it) {
        if (!is_selectable_trial(*it)) {
            continue;
        }
        if (best == result.trials.end() ||
            it->optimization_result.best_fitness < best->optimization_result.best_fitness) {
            best = it;
        }
    }

    if (best != result.trials.end()) {
        result.status = hpoea::core::RunStatus::Success;
        result.best_parameters = best->parameters;
        result.best_objective = best->optimization_result.best_fitness;
        result.error_info = std::nullopt;
        result.message = "random search completed";
        return;
    }

    if (!result.trials.empty()) {
        const auto &first = result.trials.front().optimization_result;
        result.status = first.status;
        result.best_objective = std::numeric_limits<double>::infinity();
        result.error_info = first.error_info;
        result.message = first.message.empty() ? "random search produced no successful finite trial" : first.message;
        return;
    }

    result.status = hpoea::core::RunStatus::InternalError;
    result.best_objective = std::numeric_limits<double>::infinity();
    result.error_info = hpoea::core::ErrorInfo{"internal_error", "no_valid_trial", "random search produced no trial"};
    result.message = "random search produced no trial";
}

} // namespace

namespace hpoea::core {

RandomSearchOptimizer::RandomSearchOptimizer()
    : parameter_space_(make_parameter_space()), configured_parameters_(parameter_space_.apply_defaults({})),
      identity_{"RandomSearch", "uniform_random", "1.0"} {}

RandomSearchOptimizer::RandomSearchOptimizer(const RandomSearchOptimizer &other)
    : parameter_space_(other.parameter_space_), configured_parameters_(other.configured_parameters_),
      identity_(other.identity_),
      search_space_(other.search_space_ ? std::make_shared<SearchSpace>(*other.search_space_) : nullptr) {}

void RandomSearchOptimizer::configure(const ParameterSet &parameters) {
    configured_parameters_ = parameter_space_.apply_defaults(parameters);
    parameter_space_.validate(configured_parameters_);
}

void RandomSearchOptimizer::set_search_space(std::shared_ptr<SearchSpace> search_space) {
    search_space_ = std::move(search_space);
}

HyperparameterOptimizationResult RandomSearchOptimizer::optimize(const IEvolutionaryAlgorithmFactory &algorithm_factory,
                                                                 const IProblem &problem,
                                                                 const Budget &optimizer_budget,
                                                                 const Budget &algorithm_budget, unsigned long seed) {

    HyperparameterOptimizationResult result;
    result.status = RunStatus::InternalError;
    result.seed = seed;
    result.effective_optimizer_parameters = configured_parameters_;

    const auto start_time = std::chrono::steady_clock::now();
    std::size_t objective_calls = 0;

    try {
        const auto &algorithm_space = algorithm_factory.parameter_space();
        if (algorithm_space.empty()) {
            throw std::invalid_argument("algorithm has no tunable parameters");
        }
        if (search_space_) {
            search_space_->validate(algorithm_space);
        }
        if (!has_tunable_dimension(algorithm_space, search_space_.get())) {
            throw ParameterValidationError(
                "all parameters are fixed or excluded; use BaselineOptimizer for fixed/default runs");
        }

        const auto configured_samples = get_sample_count(configured_parameters_);
        auto planned_samples = configured_samples;
        if (optimizer_budget.function_evaluations.has_value()) {
            planned_samples = std::min(planned_samples, *optimizer_budget.function_evaluations);
        }
        if (optimizer_budget.generations.has_value()) {
            planned_samples = std::min(planned_samples, *optimizer_budget.generations);
        }

        if (planned_samples == 0u) {
            const auto end_time = std::chrono::steady_clock::now();
            result.status = RunStatus::BudgetExceeded;
            result.message = "optimizer budget allows zero random search samples";
            result.optimizer_usage.wall_time =
                std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
            return result;
        }

        std::mt19937_64 rng{splitmix64(seed)};
        result.trials.reserve(planned_samples);

        for (std::size_t trial_index = 0; trial_index < planned_samples; ++trial_index) {
            const auto trial_start = std::chrono::steady_clock::now();
            const auto trial_seed = derive_trial_seed(seed, trial_index);
            HyperparameterTrialRecord trial;
            trial.trial_index = trial_index;

            try {
                trial.parameters = sample_parameters(algorithm_space, search_space_.get(), rng);
                auto algorithm = algorithm_factory.create();
                algorithm->configure(trial.parameters);
                ++objective_calls;
                trial.optimization_result = algorithm->run(problem, algorithm_budget, trial_seed);
                trial.optimization_result.seed = trial_seed;
                mark_non_finite_success(trial.optimization_result);
            } catch (const std::exception &ex) {
                const auto trial_end = std::chrono::steady_clock::now();
                const auto classified = classify_exception(ex);
                trial.optimization_result.status = classified.status;
                trial.optimization_result.error_info = classified.error_info;
                trial.optimization_result.message = ex.what();
                trial.optimization_result.seed = trial_seed;
                trial.optimization_result.algorithm_usage.wall_time =
                    std::chrono::duration_cast<std::chrono::milliseconds>(trial_end - trial_start);
            }

            result.trials.push_back(std::move(trial));
        }

        const auto end_time = std::chrono::steady_clock::now();
        result.optimizer_usage.objective_calls = objective_calls;
        result.optimizer_usage.iterations = result.trials.size();
        result.optimizer_usage.wall_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        fill_success_result(result);
        apply_optimizer_budget_status(optimizer_budget, result.optimizer_usage, result.status, result.message);
    } catch (const std::exception &ex) {
        const auto end_time = std::chrono::steady_clock::now();
        const auto classified = classify_exception(ex);
        result.status = classified.status;
        result.error_info = classified.error_info;
        result.message = ex.what();
        result.optimizer_usage.wall_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    }

    return result;
}

} // namespace hpoea::core
