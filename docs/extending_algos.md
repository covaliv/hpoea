# Extending HPOEA

These notes describe the project shape for a new evolutionary algorithm, hyperparameter optimizer, or problem.

Each extension follows the same project pattern:

1. public API
2. implementation
3. CMake wiring
4. focused test
5. small runnable example

## Nearby examples

Closest existing implementations:

- Evolutionary algorithm: `include/hpoea/wrappers/pagmo/de_algorithm.hpp` and `src/wrappers/pagmo/de_algorithm.cpp`
- Hyperparameter optimizer: `include/hpoea/wrappers/pagmo/cmaes_hyper.hpp` and `src/wrappers/pagmo/cmaes_hyper.cpp`
- Problem: `include/hpoea/wrappers/problems/benchmark_problems.hpp` and `src/wrappers/problems/benchmark_problems.cpp`
- Example: `apps/basic_ea_example.cpp` or `apps/basic_hpo_example.cpp`
- Tests: files under `tests/` using `tests/test_harness.hpp`

New dependencies are outside the usual extension pattern unless the project requires one.

## Evolutionary algorithm extension shape

- Public header under `include/hpoea/wrappers/...`.
- `core::IEvolutionaryAlgorithm` implementation.
- Factory implementing `core::IEvolutionaryAlgorithmFactory`.
- Tunable parameters exposed through `core::ParameterSpace`.
- Defaults and parameter validation in `configure()`.
- `core::Budget` converted into the wrapped solver's limits.
- `core::OptimizationResult` filled with status, best fitness, best solution, budgets, usage, seed, effective parameters, and message.
- `clone()` creates independent copies for experiment managers.
- Source file listed in the relevant `CMakeLists.txt`.
- Focused test and small app example.

Current Pagmo-backed algorithm implementations live in `src/wrappers/pagmo/` and build into `hpoea_pagmo`.

## Hyperparameter optimizer extension shape

- Public header under `include/hpoea/wrappers/...`.
- `core::IHyperparameterOptimizer` implementation.
- `identity()`, `parameter_space()`, `configure()`, `optimize()`, and `clone()` definitions.
- Algorithm instances created through the factory inside `optimize()`.
- Each trial tracked in `core::HyperparameterTrialRecord`.
- Separate optimizer and inner algorithm budgets.
- `core::HyperparameterOptimizationResult` filled with status, best parameters, best objective, trials, usage, seed, effective optimizer parameters, and message.
- Configured state preserved in `clone()`.
- Source file listed in CMake.
- Focused tests and one example in `apps/`.

## Benchmark or custom problem shape

- `core::IProblem` implementation.
- `core::ProblemMetadata` with a stable `id`, `family`, and short description.
- Dimension and bounds returned by the problem.
- Decision vector size validation in `evaluate()`.
- Objective value returned with the same minimization convention as the existing problems.
- `is_stochastic()` override for stochastic problems.
- Tests for bounds, known values, and invalid input.
- Example for a new problem pattern.

Built-in shared problems belong in `include/hpoea/wrappers/problems/` and `src/wrappers/problems/`. One-off examples can define a local problem class inside an app, as `apps/custom_problem_example.cpp` does.

## Extension test shape

Tests use the local test harness in `tests/test_harness.hpp`. The project does not use a separate test framework.

Behavior covered by extension tests:

- invalid parameters are rejected
- defaults are applied
- budgets are respected
- returned solutions match the problem dimension and bounds
- seeds are deterministic where expected
- clone preserves configured state

New tests are registered with `hpoea_add_test()` in `tests/CMakeLists.txt` and labeled with `hpoea-core` or `hpoea-pagmo`.

## User-visible documentation shape

Extension documentation includes, when applicable:

- new algorithm, optimizer, or problem listed in `README.md`
- parameters, defaults, and ranges in `docs/reference.md`
- new config type ids, fields, or custom dispatch rules in `docs/reference.md`
- runnable example mentioned in `apps/README.md`
- reproducibility notes for changes to seed, budget, search-space, or logging behavior
- diagnostic notes for config parsing, validation, or expansion failures
- troubleshooting notes for common setup or runtime failures

## Final shape summary

- Header lives under `include/hpoea/...`.
- Implementation lives under `src/...`.
- CMake builds the new source.
- Parameters are described by `core::ParameterSpace`.
- `identity.family`, `identity.implementation`, and `identity.version` are set.
- Budgets and usage counters are reported clearly.
- Tests are registered with the right label.
- A small example shows the feature.
