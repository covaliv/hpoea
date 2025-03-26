# HPOEA - Hyperparameter Optimization Framework for Evolutionary Algorithms

Modular C++ framework for hyperparameter optimization of evolutionary algorithms. Built on Pagmo2 library.

## Components

### Evolutionary Algorithms
- Differential Evolution (DE)
- Particle Swarm Optimization (PSO)
- Self-Adaptive Differential Evolution (SADE)

### Hyperparameter Optimizers
- CMA-ES
- Simulated Annealing
- PSO-based tuning
- Nelder-Mead (compass search)

### Benchmark Problems
- Sphere
- Rosenbrock
- Rastrigin
- Ackley
- Griewank
- Schwefel
- Zakharov
- Styblinski-Tang

### Features
- Parallel execution via thread-based experiment management
- Structured logging in JSON Lines format
- Reproducibility through seed management and parameter tracking

## Requirements

- C++20 compiler
- CMake 3.20 or later
- Pagmo2 library

## Building

Standard build:

```bash
mkdir build && cd build
cmake .. -DHPOEA_WITH_PAGMO=ON
cmake --build .
```

Build with tests:

```bash
cmake .. -DHPOEA_WITH_PAGMO=ON -DHPOEA_BUILD_TESTS=ON
cmake --build .
```

## Usage Example

```cpp
#include "hpoea/core/experiment.hpp"
#include "hpoea/wrappers/pagmo/de_algorithm.hpp"
#include "hpoea/wrappers/pagmo/cmaes_hyper.hpp"
#include "hpoea/wrappers/problems/benchmark_problems.hpp"
#include "hpoea/core/logging.hpp"

using namespace hpoea;

wrappers::problems::SphereProblem problem(10);
pagmo_wrappers::PagmoDifferentialEvolutionFactory ea_factory;
pagmo_wrappers::PagmoCmaesHyperOptimizer optimizer;

core::ExperimentConfig config;
config.experiment_id = "example";
config.trials_per_optimizer = 5;
config.islands = 4;
config.algorithm_budget.generations = 100;
config.optimizer_budget.generations = 50;
config.log_file_path = "results.jsonl";

core::JsonlLogger logger(config.log_file_path);
core::ParallelExperimentManager manager(4);
auto result = manager.run_experiment(config, optimizer, ea_factory, problem, logger);
```

## Architecture

Layered architecture with four components:

1. Core interfaces: abstract base classes for problems, algorithms, and optimizers
2. Wrappers: Pagmo2 adapters implementing core interfaces
3. Experiment management: orchestration layer for running experiments
4. Logging: structured logging system for reproducibility

## Example Programs

Example programs are located in `apps/` directory:

- `01_basic_ea_optimization.cpp`: basic evolutionary algorithm usage
- `02_basic_hyperparameter_optimization.cpp`: hyperparameter optimization example
- `03_experiment_management.cpp`: experiment management and logging
- `04_custom_problem.cpp`: custom problem implementation
- `05_multi_optimizer_comparison.cpp`: comparing multiple hyperparameter optimizers
- `06_custom_parameter_space.cpp`: custom parameter space definition

See `apps/README.md` for detailed documentation.
