# HPOEA

HPOEA is a C++20 framework for repeatable optimization experiments with evolutionary algorithms and hyperparameter tuning.

A typical experiment defines a problem, selects an algorithm, sets budgets and seeds, then records the result.

Hyperparameter optimization adds an outer search over algorithm parameters. Experiment runs can write JSON Lines logs for later analysis.

The core library builds without Pagmo2. `HPOEA_WITH_PAGMO=ON` adds Pagmo2 algorithms, optimizers, examples, and Pagmo tests.

## What is included

### Core library

- Public interfaces for problems, evolutionary algorithms, and hyperparameter optimizers.
- Parameter spaces and search-space restrictions.
- Sequential and parallel experiment managers.
- Budget, seed, status, and error result types.
- JSON Lines logging for experiment records.
- Config parsing and validation support.
- Baseline optimizer for default or fixed-parameter comparisons.
- Random Search optimizer for baseline hyperparameter tuning.

### Pagmo2 wrappers

When `HPOEA_WITH_PAGMO=ON`, the project builds wrappers for:

- Differential Evolution (`pagmo::de`)
- Particle Swarm Optimization (`pagmo::pso`)
- Self-Adaptive Differential Evolution (`pagmo::sade`)
- DE1220 / pDE (`pagmo::de1220`)
- Simple Genetic Algorithm (`pagmo::sga`)
- CMA-ES as an evolutionary algorithm (`pagmo::cmaes`)

Pagmo-backed hyperparameter optimizers include:

- CMA-ES
- Simulated Annealing
- PSO-based tuning
- NLopt Nelder-Mead

Built-in problems include:

- Sphere
- Rosenbrock
- Rastrigin
- Ackley
- Griewank
- Schwefel
- Zakharov
- Styblinski-Tang
- 0-1 Knapsack with continuous encoding

## Requirements

- C++20 compiler
- CMake 3.20 or newer
- Optional: Pagmo2 for the wrapper library, examples, and Pagmo tests
- For `HPOEA_WITH_PAGMO=ON`: Pagmo2 built with Eigen3 support for CMA-ES and NLopt support for Nelder-Mead

The build uses `tomlplusplus`. If CMake cannot find an installed package, it fetches `tomlplusplus` through CMake `FetchContent`. Dependency details are in [docs/reference.md](docs/reference.md#build-and-dependencies).

## Quick start

Core library build and test:

```bash
cmake -S . -B build/hpoea-core -DHPOEA_BUILD_TESTS=ON
cmake --build build/hpoea-core
ctest --test-dir build/hpoea-core -L hpoea-core --output-on-failure
```

The build also creates the `hpoea` command-line tool:

```bash
./build/hpoea-core/apps/hpoea --help
./build/hpoea-core/apps/hpoea validate tests/fixtures/configs/custom_ids_valid.toml
./build/hpoea-core/apps/hpoea plan examples/configs/basic_experiment.toml
```

`validate` checks a config for the current build. In a core-only build,
`examples/configs/basic_experiment.toml` does not validate because it uses
Pagmo-backed `de` and `cmaes` type ids. `plan` can still show the expanded
runs and mark the Pagmo parts as unavailable.

Pagmo2 wrapper and example build:

```bash
cmake -S . -B build/hpoea-pagmo \
  -DHPOEA_BUILD_TESTS=ON \
  -DHPOEA_WITH_PAGMO=ON \
  -DPagmo_DIR=/path/to/pagmo/lib/cmake/pagmo
cmake --build build/hpoea-pagmo
```

Small example executables:

```bash
./build/hpoea-pagmo/apps/basic_ea_example
./build/hpoea-pagmo/apps/basic_hpo_example
```

In a Pagmo-enabled build, the CLI can run the supported built-in path:

```bash
./build/hpoea-pagmo/apps/hpoea run examples/configs/basic_experiment.toml
```

For now, `run` supports configs that use `sphere`, `de`, and either `cmaes` or `random_search`.

The helper script provides the same checks:

```bash
./run_tests.sh --core-only
./run_tests.sh --with-pagmo --pagmo-dir /path/to/pagmo/lib/cmake/pagmo
```

CMake integration, config, parameters, logging, and troubleshooting details are in [docs/reference.md](docs/reference.md).

## Minimal example

```cpp
#include "hpoea/core/types.hpp"
#include "hpoea/wrappers/pagmo/de_algorithm.hpp"
#include "hpoea/wrappers/problems/benchmark_problems.hpp"

#include <cstdint>
#include <iostream>

int main() {
    using namespace hpoea;

    wrappers::problems::SphereProblem problem(10);
    pagmo_wrappers::PagmoDifferentialEvolutionFactory factory;
    auto algorithm = factory.create();

    core::ParameterSet parameters;
    parameters.emplace("population_size", std::int64_t{50});
    parameters.emplace("generations", std::int64_t{100});
    parameters.emplace("scaling_factor", 0.8);
    parameters.emplace("crossover_rate", 0.9);
    algorithm->configure(parameters);

    core::Budget budget;
    budget.generations = 100;

    const auto result = algorithm->run(problem, budget, 42UL);
    if (result.status != core::RunStatus::Success) {
        std::cerr << "error: " << result.message << "\n";
        return 1;
    }

    std::cout << "best_fitness: " << result.best_fitness << "\n";
    std::cout << "function_evaluations: "
              << result.algorithm_usage.function_evaluations << "\n";
}
```

## Main concepts

HPOEA has a small flow:

1. A `core::IProblem` describes the objective, dimension, and bounds.
2. A `core::IEvolutionaryAlgorithmFactory` creates algorithm instances.
3. A `core::IHyperparameterOptimizer` searches algorithm parameters.
4. A `core::IExperimentManager` repeats runs, assigns seeds, and logs records.
5. A `core::JsonlLogger` writes one JSON object per inner algorithm trial.

The full mental model is in [docs/reference.md#core-concepts](docs/reference.md#core-concepts).

## Documentation map

- [README.md](README.md): project overview, requirements, quick start, and minimal example.
- [apps/README.md](apps/README.md): CLI commands, example programs, and executable names.
- [docs/reference.md](docs/reference.md): build, CLI, CMake integration, concepts, config, parameters, logging, reproducibility, benchmarking, and troubleshooting.
- [docs/extending_algos.md](docs/extending_algos.md): project shape for new algorithms, optimizers, and problems.

## Experiment records

Repeatable experiment records include the source snapshot, compiler, CMake version, dependency versions, seed, budgets, command line, and JSONL output path.

Inner algorithm budgets and outer optimizer budgets are separate. Fixed seeds make repeated comparisons stable.

JSONL logs append to the target file. A clean run starts with a new or empty log file.

More reproducibility details are in [docs/reference.md#reproducibility-and-benchmarking](docs/reference.md#reproducibility-and-benchmarking).
