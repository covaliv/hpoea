#include "hpoea/config/suite_expander.hpp"

#include <iomanip>
#include <sstream>

namespace {

std::string format_repetition_index(std::size_t value) {
    std::ostringstream stream;
    stream << std::setw(3) << std::setfill('0') << value;
    return stream.str();
}

} // namespace

namespace hpoea::config {

bool ExpansionResult::has_errors() const noexcept {
    for (const auto &diagnostic : diagnostics) {
        if (diagnostic.severity == ExpansionDiagnosticSeverity::Error) {
            return true;
        }
    }
    return false;
}

bool ExpansionResult::ok() const noexcept {
    return !has_errors();
}

ExpansionResult expand_suite_config(const SuiteConfig &config) {
    ExpansionResult result;
    for (const auto &experiment : config.experiments) {
        const auto repetitions = experiment.repetitions.value_or(config.repetitions);
        for (std::size_t i = 0; i < repetitions; ++i) {
            ResolvedRunSpec run;
            run.experiment_id = experiment.id;
            run.problem_id = experiment.problem;
            run.algorithm_id = experiment.algorithm;
            run.optimizer_id = experiment.optimizer;
            run.repetition_index = i;
            run.seed = experiment.seed.value_or(config.suite_seed.value_or(0)) + i;
            run.output_name = experiment.output_name.value_or(experiment.id);
            run.run_id = experiment.id + "__rep" + format_repetition_index(i);
            run.planned_output_path = config.output_dir / run.output_name / ("run-" + format_repetition_index(i) + ".jsonl");
            run.algorithm_budget = experiment.algorithm_budget.value_or(BudgetConfig{});
            run.optimizer_budget = experiment.optimizer_budget.value_or(BudgetConfig{});
            result.runs.push_back(std::move(run));
        }
    }
    return result;
}

} // namespace hpoea::config
