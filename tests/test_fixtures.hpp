#pragma once

#include "hpoea/core/experiment.hpp"
#include "hpoea/core/hyperparameter_optimizer.hpp"
#include "hpoea/core/logging.hpp"
#include "hpoea/core/problem.hpp"
#include "hpoea/core/types.hpp"

#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace hpoea::tests_v2 {

class CapturingLogger final : public core::ILogger {
public:
    void log(const core::RunRecord &record) override {
        std::scoped_lock lock(mutex_);
        records.push_back(record);
    }

    void flush() override { flushed = true; }

    [[nodiscard]] std::size_t records_written() const noexcept override {
        std::scoped_lock lock(mutex_);
        return records.size();
    }

    [[nodiscard]] bool good() const noexcept override { return !simulate_failure; }

    std::vector<core::RunRecord> records;
    bool flushed{false};
    bool simulate_failure{false};

private:
    mutable std::mutex mutex_;
};

class DummyProblem final : public core::IProblem {
public:
    explicit DummyProblem(std::size_t dim = 2) : dim_(dim) {
        metadata_.id = "dummy";
        metadata_.family = "tests";
        metadata_.description = "dummy problem";
        lower_.assign(dim_, -1.0);
        upper_.assign(dim_, 1.0);
    }

    [[nodiscard]] const core::ProblemMetadata &metadata() const noexcept override { return metadata_; }

    [[nodiscard]] std::size_t dimension() const override { return dim_; }

    [[nodiscard]] std::vector<double> lower_bounds() const override { return lower_; }

    [[nodiscard]] std::vector<double> upper_bounds() const override { return upper_; }

    [[nodiscard]] double evaluate(const std::vector<double> &decision_vector) const override {
        if (decision_vector.size() != dim_) {
            throw std::runtime_error("dimension mismatch");
        }
        double sum = 0.0;
        for (double v : decision_vector) {
            sum += v * v;
        }
        return sum;
    }

private:
    core::ProblemMetadata metadata_{};
    std::size_t dim_{0};
    std::vector<double> lower_{};
    std::vector<double> upper_{};
};

class ThrowingProblem final : public core::IProblem {
public:
    explicit ThrowingProblem(std::size_t dim = 2) : dim_(dim) {
        metadata_.id = "throwing";
        metadata_.family = "tests";
        metadata_.description = "always throws on evaluate";
        lower_.assign(dim_, -1.0);
        upper_.assign(dim_, 1.0);
    }

    [[nodiscard]] const core::ProblemMetadata &metadata() const noexcept override { return metadata_; }
    [[nodiscard]] std::size_t dimension() const override { return dim_; }
    [[nodiscard]] std::vector<double> lower_bounds() const override { return lower_; }
    [[nodiscard]] std::vector<double> upper_bounds() const override { return upper_; }

    [[nodiscard]] double evaluate(const std::vector<double> &) const override {
        throw std::runtime_error("intentional test failure");
    }

private:
    core::ProblemMetadata metadata_{};
    std::size_t dim_{0};
    std::vector<double> lower_{};
    std::vector<double> upper_{};
};

class StubHyperOptimizer final : public core::IHyperparameterOptimizer {
public:
    using OptimizeFn = std::function<core::HyperparameterOptimizationResult(
        const core::IEvolutionaryAlgorithmFactory &,
        const core::IProblem &,
        const core::Budget &,
        const core::Budget &,
        unsigned long)>;

    explicit StubHyperOptimizer(OptimizeFn fn)
        : optimize_fn_(std::move(fn)) {
        identity_ = {"StubHyper", "tests", "1.0"};
    }

    [[nodiscard]] const core::AlgorithmIdentity &identity() const noexcept override { return identity_; }

    [[nodiscard]] const core::ParameterSpace &parameter_space() const noexcept override { return parameter_space_; }

    [[nodiscard]] core::HyperparameterOptimizerPtr clone() const override {
        return std::make_unique<StubHyperOptimizer>(*this);
    }

    void configure(const core::ParameterSet &parameters) override {
        configured_parameters_ = parameter_space_.apply_defaults(parameters);
    }

    [[nodiscard]] core::HyperparameterOptimizationResult optimize(
        const core::IEvolutionaryAlgorithmFactory &factory,
        const core::IProblem &problem,
        const core::Budget &optimizer_budget,
        const core::Budget &algorithm_budget,
        unsigned long seed) override {
        if (optimize_fn_) {
            return optimize_fn_(factory, problem, optimizer_budget, algorithm_budget, seed);
        }
        return {};
    }

    core::ParameterSpace parameter_space_{};
    core::ParameterSet configured_parameters_{};

private:
    OptimizeFn optimize_fn_{};
    core::AlgorithmIdentity identity_{};
};

}
