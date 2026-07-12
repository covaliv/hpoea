# CLI and example programs

This directory contains the `hpoea` CLI and runnable examples for the
Pagmo-backed part of HPOEA.

The CLI is built in core-only and Pagmo-enabled builds.
The example programs are built only when CMake is configured with
`HPOEA_WITH_PAGMO=ON`.

Build commands, dependency details, and parameter/config references are in
[docs/reference.md](../docs/reference.md).

## Build

Core-only CLI build:

```bash
cmake -S . -B build/hpoea-core -DHPOEA_BUILD_TESTS=ON
cmake --build build/hpoea-core --target cli
```

Pagmo-enabled CLI and example build:

```bash
cmake -S . -B build/hpoea-pagmo \
  -DHPOEA_WITH_PAGMO=ON \
  -DPagmo_DIR=/path/to/pagmo/lib/cmake/pagmo
cmake --build build/hpoea-pagmo
```

The same build tree includes tests when configured with `-DHPOEA_BUILD_TESTS=ON`.

## CLI

Basic commands:

```bash
./build/hpoea-core/apps/hpoea --help
./build/hpoea-core/apps/hpoea validate tests/fixtures/configs/custom_ids_valid.toml
./build/hpoea-core/apps/hpoea plan examples/configs/basic_experiment.toml
```

`validate` checks a config for the current build. `plan` expands the suite and
prints the planned runs without creating output directories or log files.

In a Pagmo-enabled build, `run` can execute supported configs:

```bash
./build/hpoea-pagmo/apps/hpoea run examples/configs/basic_experiment.toml
```

`run` supports the built-in benchmark problems (`sphere`, `rosenbrock`, `rastrigin`, `ackley`, `griewank`, `schwefel`, `zakharov`, `styblinski_tang`, `knapsack`), the algorithms `de`, `sade`, `pso`, `sga`, and `de1220`, and the optimizers `random_search`, `baseline`, `cmaes`, `pso`, `simulated_annealing`, and `nelder_mead`.

## Introductory examples

Small introductory executables:

```bash
./build/hpoea-pagmo/apps/basic_ea_example
./build/hpoea-pagmo/apps/basic_hpo_example
./build/hpoea-pagmo/apps/experiment_management_example
```

Together they cover:

1. run one evolutionary algorithm
2. tune algorithm parameters with a hyperparameter optimizer
3. repeat runs and write JSONL records

## Examples by area

### Single evolutionary algorithm runs

- `basic_ea_example.cpp`: Differential Evolution on a 10-dimensional Sphere problem.
- `cmaes_optimization_example.cpp`: CMA-ES used as the evolutionary algorithm.
- `sga_optimization_example.cpp`: Simple Genetic Algorithm on Rastrigin.
- `de1220_optimization_example.cpp`: DE1220 / pDE on Ackley.
- `knapsack_optimization_example.cpp`: Differential Evolution on a 0-1 knapsack problem with continuous encoding.

### Tune hyperparameters

- Random Search is available as a core hyperparameter optimizer for baseline comparisons.
- `basic_hpo_example.cpp`: CMA-ES tunes Differential Evolution on Rosenbrock.
- `knapsack_hpo_example.cpp`: CMA-ES tunes Differential Evolution on knapsack.
- `knapsack_pso_sa_example.cpp`: Simulated Annealing tunes Particle Swarm Optimization on knapsack.
- `optimizer_comparison_example.cpp`: compares CMA-ES, Simulated Annealing, and PSO-based hyperparameter optimization.

### Experiments and logs

- `experiment_management_example.cpp`: runs repeated optimizer trials and writes `experiment_results.jsonl`.
- `benchmark_suite.cpp`: runs a small benchmark suite. `HPOEA_BENCHMARK_FULL=1` enables a longer run.

The benchmark suite executable is named:

```bash
./build/hpoea-pagmo/apps/hpoea_benchmark_suite
```

### Custom inputs

- `custom_problem_example.cpp`: implements a local `IProblem` in the example file.
- `custom_parameter_space_example.cpp`: builds and validates a custom `ParameterSpace`.
- `search_space_example.cpp`: fixes, narrows, and excludes parameters during hyperparameter optimization.
  C++ transform notes are in [search-space transforms](../docs/reference.md#search-space-transforms).

### Extra check programs

These are extra check programs rather than the main CTest suite:

- `simple_example.cpp`: a compact tour of several algorithms.
- `correctness_test.cpp`: checks basic optimization behavior and reproducibility.
- `sfu_benchmark_test.cpp`: checks several benchmark functions.

Their executable names are:

```bash
./build/hpoea-pagmo/apps/hpoea_simple_example
./build/hpoea-pagmo/apps/hpoea_app_correctness_test
./build/hpoea-pagmo/apps/hpoea_sfu_benchmark_test
```

## Output format

Most examples print plain `key: value` lines. Most optimization examples write errors to stderr with an `error:` prefix and return a non-zero exit code.

`custom_parameter_space_example.cpp` also prints `validation_error:` for expected validation failures and still exits successfully after demonstrating them. The check programs print validation lines as they run and return non-zero when a check fails.

Examples that use `core::JsonlLogger` write JSON Lines files. Each line is one logged inner algorithm trial.

Existing files are appended to. A clean rerun starts from a new or empty log file. Exact fields are in the [logging schema](../docs/reference.md#logging-schema).

## Full executable list

CLI executable:

```bash
./build/hpoea-core/apps/hpoea
./build/hpoea-pagmo/apps/hpoea
```

Pagmo-enabled examples and check programs:

```bash
./build/hpoea-pagmo/apps/basic_ea_example
./build/hpoea-pagmo/apps/basic_hpo_example
./build/hpoea-pagmo/apps/experiment_management_example
./build/hpoea-pagmo/apps/custom_problem_example
./build/hpoea-pagmo/apps/optimizer_comparison_example
./build/hpoea-pagmo/apps/custom_parameter_space_example
./build/hpoea-pagmo/apps/cmaes_optimization_example
./build/hpoea-pagmo/apps/sga_optimization_example
./build/hpoea-pagmo/apps/de1220_optimization_example
./build/hpoea-pagmo/apps/knapsack_optimization_example
./build/hpoea-pagmo/apps/knapsack_hpo_example
./build/hpoea-pagmo/apps/knapsack_pso_sa_example
./build/hpoea-pagmo/apps/search_space_example
./build/hpoea-pagmo/apps/hpoea_simple_example
./build/hpoea-pagmo/apps/hpoea_app_correctness_test
./build/hpoea-pagmo/apps/hpoea_sfu_benchmark_test
./build/hpoea-pagmo/apps/hpoea_benchmark_suite
```
