# HPOEA - Hyperparameter Optimization Framework for Evolutionary Algorithms

Modular C++ framework for hyperparameter optimization of evolutionary algorithms. Built on Pagmo2 library.

## Components

### Evolutionary Algorithms
- Differential Evolution (DE) - `pagmo::de`
- Particle Swarm Optimization (PSO) - `pagmo::pso`
- Self-Adaptive Differential Evolution (SADE) - `pagmo::sade`
  - Can be configured as jDE (variant_adptv=1, Brest et al.) or iDE (variant_adptv=2, Elsayed et al.)
  - Defaults to jDE (variant_adptv=1)
- DE1220 (pDE) - `pagmo::de1220` - Alternative self-adaptive DE variant
- Simple Genetic Algorithm (SGA) - `pagmo::sga`
- CMA-ES (also available as an evolutionary algorithm) - `pagmo::cmaes`

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
- Knapsack (0-1 knapsack with continuous encoding and penalty-based constraint handling)

### Features
- Parallel execution via thread-based experiment management
- Structured logging in JSON Lines format
- Reproducibility through seed management and parameter tracking

## Requirements

- c++20 compiler
- cmake 3.20 or later
- pagmo2 library with eigen3 support

## Installing Pagmo2

install eigen3:
```bash
sudo pacman -S eigen
```

build and install pagmo2 with eigen3 support:
```bash
git clone https://github.com/esa/pagmo2.git
cd pagmo2
mkdir build && cd build
cmake .. -DPAGMO_WITH_EIGEN3=ON -DEigen3_DIR=/usr/lib/cmake/eigen3
cmake --build .
cmake --install . --prefix ~/.local
```

verify installation:
```bash
ls ~/.local/lib/cmake/pagmo/PagmoConfig.cmake
ls ~/.local/include/pagmo/pagmo.hpp
ls ~/.local/lib/libpagmo.so*
```

if cmake cannot find pagmo2, specify the installation directory:
```bash
cmake -S . -B build -DHPOEA_WITH_PAGMO=ON -DPagmo_DIR=~/.local/lib/cmake/pagmo
```

if you get errors about missing eigen3 support, rebuild pagmo2 with eigen3 enabled:
```bash
cd pagmo2/build
cmake .. -DPAGMO_WITH_EIGEN3=ON -DEigen3_DIR=/usr/lib/cmake/eigen3
cmake --build .
cmake --install . --prefix ~/.local
```

## Building

standard build:
```bash
cmake -S . -B build -DHPOEA_WITH_PAGMO=ON -DPagmo_DIR=~/.local/lib/cmake/pagmo
cmake --build build
```

build with tests:
```bash
cmake -S . -B build -DHPOEA_WITH_PAGMO=ON -DPagmo_DIR=~/.local/lib/cmake/pagmo -DHPOEA_BUILD_TESTS=ON
cmake --build build
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

## experiment complexity guide

- lines of code: adapting the examples in `apps/10_knapsack_optimization.cpp` or `apps/11_knapsack_hyperparameter_optimization.cpp` requires roughly 40-60 lines, covering includes, problem setup, optimizer configuration, budgets, and the run loop.
- required concepts: understand `core::IProblem` (problem definition), `pagmo_wrappers::IEvolutionaryAlgorithmFactory` (algorithm choice), `core::IHyperparameterOptimizer` (meta-optimizer), `core::Budget` (limits), and `core::Logger` (jsonl output). these abstractions mirror the headers included in every example.
- time estimate: once dependencies are installed and the build tree is configured with `HPOEA_WITH_PAGMO=ON`, customizing an existing example to a new problem typically takes 10–15 minutes; implementing a brand new experiment (new `apps/*.cpp`) usually fits in a 30–45 minute window including compilation.
- common pitfalls: forgetting to build pagmo with eigen3 support (causes `pagmo::cmaes` compile errors), omitting `-DPagmo_DIR` so cmake cannot find the local install, mismatching problem dimension vs. decision vector size (throws at runtime), and running optimizers without setting budgets which keeps defaults at zero trials.

## Example Programs

Example programs are located in `apps/` directory:

- `01_basic_ea_optimization.cpp`: basic evolutionary algorithm usage
- `02_basic_hyperparameter_optimization.cpp`: hyperparameter optimization example
- `03_experiment_management.cpp`: experiment management and logging
- `04_custom_problem.cpp`: custom problem implementation
- `05_multi_optimizer_comparison.cpp`: comparing multiple hyperparameter optimizers
- `06_custom_parameter_space.cpp`: custom parameter space definition
- `07_cmaes_optimization.cpp`: CMA-ES as an evolutionary algorithm
- `08_sga_optimization.cpp`: Simple Genetic Algorithm usage
- `09_de1220_optimization.cpp`: DE1220 (pDE) usage - alternative self-adaptive DE variant
- `10_knapsack_optimization.cpp`: knapsack problem optimization using differential evolution
- `11_knapsack_hyperparameter_optimization.cpp`: hyperparameter optimization for knapsack problem using cma-es to tune de parameters
- `13_knapsack_pso_sa_optimization.cpp`: pso with simulated annealing hyperparameter optimization for knapsack problem

See `apps/README.md` for detailed documentation.
