# HPOEA reference

This page is the detailed HPOEA reference for build modes, the CLI, CMake
integration, config files, parameters, logging, reproducibility, and
troubleshooting.

The root `README.md` stays focused on orientation. `apps/README.md` is for the
CLI and runnable examples. `docs/extending_algos.md` is for extension work.

## Build and dependencies

Core requirements:

- C++20 compiler
- CMake 3.20 or newer
- `tomlplusplus` for TOML parsing. CMake first tries `find_package(tomlplusplus CONFIG QUIET)` and otherwise fetches v3.4.0 with `FetchContent`.

The core build checks the library without Pagmo2.

Measurement and benchmark runs must use a `Release` build. The top-level CMakeLists defaults `CMAKE_BUILD_TYPE` to `Release` when it is unset (and no multi-config generator is in use), but pass it explicitly for reproducible measurements, or `-DCMAKE_BUILD_TYPE=Debug` for debugging.

```bash
cmake -S . -B build/hpoea-core -DCMAKE_BUILD_TYPE=Release -DHPOEA_BUILD_TESTS=ON
cmake --build build/hpoea-core
ctest --test-dir build/hpoea-core -L hpoea-core --output-on-failure
```

The Pagmo build includes wrappers, examples, and Pagmo-labeled tests.

Pagmo-enabled requirements:

- Pagmo2 discoverable by CMake as `Pagmo::pagmo`
- `PagmoConfig.cmake` or `pagmo-config.cmake`
- Pagmo built with the pieces used by HPOEA wrappers, especially CMA-ES/Eigen3 and NLopt support for Nelder-Mead tuning

Pagmo-enabled configure, build, and test:

```bash
cmake -S . -B build/hpoea-pagmo \
  -DCMAKE_BUILD_TYPE=Release \
  -DHPOEA_BUILD_TESTS=ON \
  -DHPOEA_WITH_PAGMO=ON \
  -DPagmo_DIR=/path/to/pagmo/lib/cmake/pagmo
cmake --build build/hpoea-pagmo
ctest --test-dir build/hpoea-pagmo -L hpoea-pagmo --output-on-failure
```

`Pagmo_DIR` points to the directory that contains `PagmoConfig.cmake`.
`CMAKE_PREFIX_PATH` adds install prefixes to CMake's search paths.
The helper script runs both the core and Pagmo-enabled flows by default; use
`--core-only` to skip Pagmo:

```bash
./run_tests.sh
./run_tests.sh --with-pagmo --pagmo-dir /path/to/pagmo/lib/cmake/pagmo
```

## Command-line interface

The build creates `hpoea` in the `apps` build directory. It is available in
core-only and Pagmo-enabled builds.

Core-only command examples:

```bash
./build/hpoea-core/apps/hpoea --help
./build/hpoea-core/apps/hpoea validate tests/fixtures/configs/custom_ids_valid.toml
./build/hpoea-core/apps/hpoea plan examples/configs/basic_experiment.toml
```

Commands:

- `validate <config.toml>` parses TOML and validates it for the current build.
- `plan <config.toml>` parses and expands the suite, prints run ids, seeds,
  planned JSONL paths, backend status, and dispatch status, and does not create
  output files or directories.
- `run <config.toml>` validates, expands, checks that every run is supported,
  creates output directories, and runs them with `core::SequentialExperimentManager`.

`validate` is strict about the current build. A core-only build rejects Pagmo
type ids, so `examples/configs/basic_experiment.toml` only validates in a
Pagmo-enabled build. `plan` can still preview that config in a core-only build
when the only validation issue is the missing Pagmo backend. It marks those
runs as not runnable.

Pagmo-enabled run example:

```bash
./build/hpoea-pagmo/apps/hpoea run examples/configs/basic_experiment.toml
```

For now, `run` supports configs that use:

- problem type `sphere`
- algorithm type `de`
- optimizer type `cmaes` or `random_search`

`random_search` is a core optimizer, but the built-in `de` algorithm dispatch is
still Pagmo-backed, so full CLI runs using `de` require a Pagmo-enabled build.
Other problem, algorithm, or optimizer type ids return an unsupported dispatch
error for now. The parser and validator can still accept custom type ids where
their current rules allow them.

## Using HPOEA from CMake

The repository defines build-tree targets. It does not currently install an exported `find_package(hpoea)` package.

Core-only integration:

```cmake
cmake_minimum_required(VERSION 3.20)
project(my_hpoea_run LANGUAGES CXX)

set(HPOEA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(HPOEA_WITH_PAGMO OFF CACHE BOOL "" FORCE)
add_subdirectory(/path/to/hpoea hpoea-build)

add_executable(my_run main.cpp)
target_link_libraries(my_run PRIVATE hpoea_core)
target_compile_features(my_run PRIVATE cxx_std_20)
```

Pagmo-backed integration:

```cmake
cmake_minimum_required(VERSION 3.20)
project(my_hpoea_pagmo_run LANGUAGES CXX)

set(HPOEA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(HPOEA_WITH_PAGMO ON CACHE BOOL "" FORCE)
set(Pagmo_DIR "/path/to/pagmo/lib/cmake/pagmo" CACHE PATH "" FORCE)
add_subdirectory(/path/to/hpoea hpoea-build)

add_executable(my_run main.cpp)
target_link_libraries(my_run PRIVATE hpoea_pagmo)
target_compile_features(my_run PRIVATE cxx_std_20)
```

`hpoea_pagmo` links `hpoea_core` publicly, so linking `hpoea_pagmo` is enough for normal Pagmo-backed programs.

## Core concepts

A typical run has these parts:

1. `core::IProblem` for the objective, dimension, and bounds.
2. `core::IEvolutionaryAlgorithmFactory` for algorithm instances.
3. `core::IHyperparameterOptimizer` for parameter tuning.
4. Inner algorithm budget and outer optimizer budget.
5. Direct runs, or experiment-manager runs with repeated trials and log records.

Key types:

- `core::IProblem`: objective metadata, dimension, bounds, `evaluate()`, and optional stochastic marker.
- `core::IEvolutionaryAlgorithm`: configurable optimizer that returns one `core::OptimizationResult`.
- `core::IEvolutionaryAlgorithmFactory`: creates fresh algorithm instances and exposes their parameter space.
- `core::IHyperparameterOptimizer`: searches algorithm parameters and returns one `core::HyperparameterOptimizationResult`.
- `core::SequentialExperimentManager` / `core::ParallelExperimentManager`: repeat optimizer trials and log inner algorithm trials. The parallel manager distributes independent trials over a std::thread worker pool (`max_parallel_trials` caps the workers); trial seeds are fixed per trial index in advance, so per-trial results are identical to the sequential manager.
- `core::JsonlLogger`: appends JSON Lines records.

Budget and usage names are intentionally different for the inner and outer loops:

| Context | Budget type | Usage fields |
|---|---|---|
| Inner evolutionary algorithm run | `core::Budget` | `function_evaluations`, `generations`, `wall_time` |
| Outer hyperparameter optimizer run | `core::Budget` | `objective_calls`, `iterations`, `wall_time` |

TOML config budgets support `generations` and `function_evaluations`; `wall_time` is available through the C++ API.

Budget currency for comparisons: `optimizer_budget.function_evaluations` counts completed inner-EA runs and is the only cross-optimizer-comparable unit, so optimizer comparisons should budget by `function_evaluations`. `optimizer_budget.generations` is optimizer-specific (random search rejects it) and is not comparable across optimizers.

## TOML config

A config file describes a group of runs.

HPOEA reads the TOML into `hpoea::config::SuiteConfig`, checks references and shapes, then expands the suite into concrete run plans.
It does not execute algorithms or create result files.

The config API is experimental and currently supports examples and project-specific tooling. Names and fields may change before the library API is stabilized.

Related config headers:

```cpp
#include "hpoea/config/config_parser.hpp"
#include "hpoea/config/config_validator.hpp"
#include "hpoea/config/suite_expander.hpp"
```

The repository example is `examples/configs/basic_experiment.toml`. It uses the Pagmo-backed `de` and `cmaes` type ids, so it parses in a core-only build but full validation requires `HPOEA_WITH_PAGMO=ON`.

Supported top-level shape:

```toml
schema_version = 1

[suite]
name = "basic_experiment"
output_dir = "results/basic_experiment"
suite_seed = 123456
repetitions = 2

[problems.sphere10]
type = "sphere"
dimension = 10
lower_bound = -5.0
upper_bound = 5.0

[algorithms.de_default]
type = "de"
fixed = { population_size = 40, variant = 2, scaling_factor = 0.8, crossover_rate = 0.9 }

[algorithms.de_default.search.ftol]
mode = "exclude"

[optimizers.cmaes_fast]
type = "cmaes"
parameters = { generations = 12, sigma0 = 0.3, ftol = 1e-6, xtol = 1e-6 }

[[experiments]]
id = "sphere_de_cmaes"
problem = "sphere10"
algorithm = "de_default"
optimizer = "cmaes_fast"
repetitions = 3
seed = 9001
output_name = "sphere_de_cmaes"

[experiments.algorithm_budget]
generations = 25

[experiments.optimizer_budget]
generations = 8
function_evaluations = 800
```

Config notes:

- `schema_version`: value `1`.
- `[suite].name` and `[suite].output_dir`: required non-empty strings.
- `[suite].repetitions`: defaults to `1`; valid range starts at `1`.
- Problem parameter values may be integer, floating-point, boolean, string, or numeric arrays.
- Box-problem `lower_bound` and `upper_bound` are optional but must be given together; omit both to keep each problem's canonical domain (e.g. Schwefel `[-500, 500]`, Ackley `[-32.768, 32.768]`).
- Nested problem parameter tables, mixed non-numeric arrays, `[suite.defaults]`, and `[[matrices]]` are rejected.
- `[[experiments]].seed` seeds the experiment; each repetition derives its own seed by hashing the explicit seed and the repetition index (FNV-1a), so nearby explicit seeds do not share repetition seeds.
- If an experiment seed is missing, suite expansion derives a deterministic seed from the suite and experiment fields.
- Expanded output paths look like `output_dir/experiments/<output_name>/run-000.jsonl`.

Diagnostics:

- Parse, validation, and expansion results carry diagnostics.
- Diagnostics include a severity and a path. Parse diagnostics also include the source name.
- `Error` diagnostics are blocking. Warnings are for information when a future check adds them.

Known config type ids:

| Kind | Type ids |
|---|---|
| Core hyperparameter optimizers | `random_search` |
| Pagmo-backed algorithms | `de`, `pso`, `sade`, `sga`, `de1220`, `cmaes` |
| Pagmo-backed hyperparameter optimizers | `cmaes`, `pso`, `simulated_annealing`, `nelder_mead` |

Type ids are open strings. The parser can read custom problem, algorithm, and optimizer ids, but it does not create those objects.
Application dispatch code maps each id to the right problem, factory, or optimizer.

Search rules live under `[algorithms.<id>.search.<parameter_name>]`:

| Mode | Required fields | Meaning |
|---|---|---|
| `range` | `min`, `max` | Floating-point range; `min < max`. |
| `integer_range` | `min`, `max` | Integer range; `min < max`. |
| `choice` | `values` | Explicit list of scalar values. |
| `exclude` | none | Leave the parameter out of the optimizer search space so the algorithm default can apply. |

A parameter may not appear in both `fixed` and `search` for the same algorithm.

### Search-space transforms

The TOML config supports the search modes above. It does not currently expose transform syntax.

The C++ `core::SearchSpace` API can also search continuous parameters in a transformed scale:

| Transform | Use for | Constraint |
|---|---|---|
| `none` | normal linear search | no extra constraint |
| `log` | powers of 10 | positive lower bound |
| `log2` | powers of 2 | positive lower bound |
| `sqrt` | square-root scale | zero or positive lower bound |

Transforms affect the values seen by the hyperparameter optimizer. Decoded values are converted back before the algorithm is configured.

## Parameter reference

`core::ParameterSet` integer values use `std::int64_t`. Continuous values use `double`, booleans use `bool`, and categorical values use `std::string`. Concrete `ParameterSpace` validation treats bounds as inclusive.

### Pagmo evolutionary algorithms

| Algorithm | Config id | Identity | Parameters |
|---|---|---|---|
| Differential Evolution | `de` | `DifferentialEvolution` / `pagmo::de` | `population_size` integer default `50` range `5..2000`; `crossover_rate` double default `0.9` range `0..1`; `scaling_factor` double default `0.8` range `0..1`; `variant` integer default `2` range `1..10`; `generations` integer default `100` range `1..1000`; `ftol` double default `1e-6` range `0..1`; `xtol` double default `1e-6` range `0..1` |
| Particle Swarm Optimization | `pso` | `ParticleSwarmOptimization` / `pagmo::pso` | `population_size` integer default `50` range `5..2000`; `omega` double default `0.7298` range `0..1`; `eta1` double default `2.05` range `1..3`; `eta2` double default `2.05` range `1..3`; `max_velocity` double default `0.5` range `0..100`; `variant` integer default `5` range `1..6`; `generations` integer default `100` range `1..1000` |
| Self-Adaptive Differential Evolution | `sade` | `SelfAdaptiveDE` / `pagmo::sade` | `population_size` integer default `50` range `5..2000`; `generations` integer default `100` range `1..1000`; `variant` integer default `2` range `1..18`; `variant_adptv` integer default `1` range `1..2`; `ftol` double default `1e-6` range `0..1`; `xtol` double default `1e-6` range `0..1`; `memory` boolean default `false` |
| DE1220 / pDE | `de1220` | `DE1220` / `pagmo::de1220` | `population_size` integer default `50` range `5..5000`; `generations` integer default `200` range `1..1000`; `ftol` double default `1e-6` range `0..1`; `xtol` double default `1e-6` range `0..1`; `variant_adaptation` integer default `1` range `1..2`; `memory` boolean default `false` |
| Simple Genetic Algorithm | `sga` | `SGA` / `pagmo::sga` | `population_size` integer default `50` range `5..5000`; `generations` integer default `200` range `1..1000`; `crossover_probability` double default `0.9` range `0..1`; `mutation_probability` double default `0.02` range `0..1` |
| CMA-ES | `cmaes` | `CMAES` / `pagmo::cmaes` | `population_size` integer default `50` range `5..5000`; `generations` integer default `100` range `1..1000`; `sigma0` double default `0.5` range `1e-6..5`; `ftol` double default `1e-6` range `0..1`; `xtol` double default `1e-6` range `0..1` |

### Core hyperparameter optimizers

| Optimizer | Config id | Identity | Parameters |
|---|---|---|---|
| Random Search | `random_search` | `RandomSearch` / `uniform_random` | `sample_count` integer default `0` range `0..100000`; `0` means budget-driven (requires `optimizer_budget.function_evaluations`) |

### Pagmo hyperparameter optimizers

| Optimizer | Config id | Identity | Parameters |
|---|---|---|---|
| CMA-ES | `cmaes` | `CMAESHyperOptimizer` / `pagmo::cmaes` | `generations` integer default `100` range `1..1000`; `sigma0` double default `0.5` range `1e-6..10`; `cc` double default `0.4` range `0..1`; `cs` double default `0.3` range `0..1`; `c1` double default `0.05` range `0..1`; `cmu` double default `0.1` range `0..1`; `ftol` double default `1e-6` range `0..1`; `xtol` double default `1e-6` range `0..1`; `force_bounds` boolean default `false` |
| PSO | `pso` | `PSOHyperOptimizer` / `pagmo::pso` | `variant` integer default `5` range `1..6`; `generations` integer default `100` range `1..1000`; `omega` double default `0.7298` range `0..1`; `eta1` double default `2.05` range `1..3`; `eta2` double default `2.05` range `1..3`; `max_velocity` double default `0.5` range `0..100` |
| Simulated Annealing | `simulated_annealing` | `SimulatedAnnealing` / `pagmo::simulated_annealing` | `iterations` integer default `1000` range `1..100000`; `ts` double default `10.0` range `1e-6..100`; `tf` double default `0.1` range `1e-6..100`; `n_T_adj` integer default `10` range `1..10000`; `n_range_adj` integer default `1` range `1..10000`; `bin_size` integer default `10` range `1..1000`; `start_range` double default `1.0` range `0..1` |
| NLopt Nelder-Mead | `nelder_mead` | `NelderMead` / `nlopt::neldermead` | `max_fevals` integer default `1000` range `1..100000`; `xtol_rel` double default `1e-8` range `1e-15..1e-1`; `ftol_rel` double default `1e-8` range `1e-15..1e-1` |

## Logging schema

`core::JsonlLogger` writes one JSON object per line. Each row is one logged inner algorithm trial.

Current log schema version: `3`.

Logger behavior:

- Existing files are appended to.
- Missing parent directories are not created.
- Records are flushed after each write by default.
- Parameter keys are written in sorted order.
- Non-finite floating-point values are written as `null`.
- Open, reopen, and write failures throw exceptions.

Fields:

- `schema_version`
- `experiment_id`
- `problem_id`
- `evolutionary_algorithm`
- `hyper_optimizer`
- `algorithm_parameters`
- `optimizer_parameters`
- `status`
- `objective_value`
- `requested_budget`
- `effective_budget`
- `algorithm_usage`
- `error_info`
- `algorithm_seed`
- `optimizer_seed`
- `message`

Status values are `success`, `budget_exceeded`, `failed_evaluation`, `invalid_configuration`, and `internal_error`.
Missing budget values are written as `null`. `error_info` is either `null` or an object with `category`, `code`, and `detail`.

Example shape, formatted for readability:

```json
{
  "schema_version": 3,
  "experiment_id": "example",
  "problem_id": "sphere",
  "evolutionary_algorithm": {
    "family": "DifferentialEvolution",
    "implementation": "pagmo::de",
    "version": "2.x"
  },
  "hyper_optimizer": null,
  "algorithm_parameters": {
    "generations": 50,
    "population_size": 50
  },
  "optimizer_parameters": {},
  "status": "success",
  "objective_value": 0.001,
  "requested_budget": {
    "function_evaluations": null,
    "generations": 50,
    "wall_time_ms": null
  },
  "effective_budget": {
    "function_evaluations": null,
    "generations": 50,
    "wall_time_ms": null
  },
  "algorithm_usage": {
    "function_evaluations": 2550,
    "generations": 50,
    "wall_time_ms": 12
  },
  "error_info": null,
  "algorithm_seed": 12345,
  "optimizer_seed": null,
  "message": "ok"
}
```

Exact identity strings, defaults, parameters, and evaluation counts depend on the algorithm and optimizer used. For Differential Evolution, the initial population also consumes evaluations, so function evaluations can be larger than `population_size * generations`.

## Reproducibility and benchmarking

Repeatable experiment records include the source snapshot, compiler, CMake version, dependency versions, build type, configure/build command line, executable, problem, algorithm, optimizer, budgets, seeds, and JSONL output path.

Fair comparison records include:

- problem set, dimensions, and bounds
- number of repetitions
- seed policy and all top-level/generated seeds
- inner algorithm budget
- outer optimizer budget
- comparison unit: wall time, objective calls, generations, or another measure
- search-space bounds and fixed parameters
- failure handling policy
- summary statistics and preserved JSONL logs

Seed behavior:

| Place | Behavior |
|---|---|
| `ExperimentConfig.random_seed` set | experiment managers use that seed to create optimizer trial seeds |
| `ExperimentConfig.random_seed` missing | experiment managers create an `actual_seed` with `std::random_device` |
| `[[experiments]].seed` set | each repetition derives its seed by hashing the explicit seed and the repetition index (FNV-1a) |
| experiment seed missing in TOML expansion | suite expansion derives a deterministic seed from suite and experiment fields |
| parallel experiments | preserve run seeds and results; log line order alone is not a stable source of run identity |

The built-in benchmark problems and `apps/benchmark_suite.cpp` are small benchmark examples, not a complete comparison protocol by themselves.

The benchmark apps (`apps/benchmark_suite.cpp`, `apps/optimizer_comparison_example.cpp`) set a fixed `ExperimentConfig.random_seed` and print `ExperimentResult.actual_seed`, so a recorded run names the seed that produced its numbers and is reproducible. Their optimizer budgets use `function_evaluations` so the compared optimizers receive the same number of inner-EA runs.

## Troubleshooting

- **CMake cannot find Pagmo2:** `-DPagmo_DIR=/path/to/pagmo/lib/cmake/pagmo` points CMake at the directory that contains `PagmoConfig.cmake` or `pagmo-config.cmake`.
- **CMA-ES or Nelder-Mead paths fail:** Pagmo2 needs Eigen3 support for CMA-ES and NLopt support for `pagmo::nlopt`.
- **Examples are missing:** Pagmo-backed example executables are built only with
  `-DHPOEA_WITH_PAGMO=ON`; the CLI still builds in a core-only build.
- **Tests are missing:** test executables are built only with `-DHPOEA_BUILD_TESTS=ON`; CTest reads tests from the matching build tree.
- **JSONL logs contain old records:** `JsonlLogger` appends; a clean rerun starts from a new or empty log file.
- **Config validation rejects a suite:** `examples/configs/basic_experiment.toml` shows the supported top-level shape. Unknown fields, unsupported search modes, and known Pagmo-backed type ids without a Pagmo-enabled build can all produce diagnostics.
- **Pagmo tests cannot run locally:** the core suite remains available when Pagmo is not discoverable.
