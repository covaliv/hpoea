# Example Programs

Example programs demonstrating HPOEA framework usage.

## Building Examples

Examples are built automatically when configuring with `HPOEA_WITH_PAGMO=ON`:

```bash
cd build
cmake .. -DHPOEA_WITH_PAGMO=ON
cmake --build .
```

## Running Examples

```bash
./build/apps/basic_ea_example
./build/apps/basic_hpo_example
./build/apps/experiment_management_example
./build/apps/custom_problem_example
./build/apps/optimizer_comparison_example
./build/apps/custom_parameter_space_example
./build/apps/cmaes_optimization_example
./build/apps/sga_optimization_example
./build/apps/de1220_optimization_example
./build/apps/knapsack_optimization_example
./build/apps/knapsack_hpo_example
./build/apps/knapsack_pso_sa_example
```

## Example Descriptions

### basic_ea_example.cpp

Demonstrates basic evolutionary algorithm usage. Configures Differential Evolution algorithm with specified parameters and optimizes a 10-dimensional Sphere problem. Outputs best fitness, function evaluations, generations, and wall time.

### basic_hpo_example.cpp

Demonstrates hyperparameter optimization using CMA-ES to tune DE parameters. Configures CMA-ES optimizer and optimizes DE hyperparameters on an 8-dimensional Rosenbrock problem. Outputs best objective value, number of trials, best hyperparameters, function evaluations, and wall time.

### experiment_management_example.cpp

Demonstrates experiment management with structured logging. Uses SequentialExperimentManager to run multiple trials with logging to JSON Lines format. Outputs experiment ID, number of optimizer runs, best objective, trials count, function evaluations, and best hyperparameters.

### custom_problem_example.cpp

Demonstrates custom problem implementation. Implements a Shifted Sphere problem by extending IProblem interface. Shows how to define custom problem metadata, bounds, and evaluation function. Outputs best fitness, distance to optimum, and function evaluations.

### optimizer_comparison_example.cpp

Compares multiple hyperparameter optimizers (CMA-ES, Simulated Annealing, PSO-based) on the same problem. Runs experiments for each optimizer and compares results. Outputs optimizer name, best objective, trials count, function evaluations, and ranked comparison results.

### custom_parameter_space_example.cpp

Demonstrates custom parameter space definition and validation. Creates a ParameterSpace with custom descriptors, generates random valid configurations, validates them, and tests a sampled configuration with an algorithm. Outputs generated configurations and optimization results.

### cmaes_optimization_example.cpp

Demonstrates CMA-ES used as an evolutionary algorithm (not as a hyperparameter optimizer). Configures CMA-ES algorithm with specified parameters and optimizes a 10-dimensional Sphere problem. Outputs best fitness, function evaluations, generations, and wall time.

### sga_optimization_example.cpp

Demonstrates Simple Genetic Algorithm (SGA) usage. Configures SGA with population size, generations, crossover probability, and mutation probability. Optimizes a 10-dimensional Rastrigin problem. Outputs best fitness, function evaluations, generations, and wall time.

### de1220_optimization_example.cpp

Demonstrates DE1220 (pDE) usage - an alternative self-adaptive Differential Evolution variant implemented via `pagmo::de1220`. Note: This is different from jDE (Brest et al.), which is available via SADE with `variant_adptv=1`. Configures DE1220 with population size, generations, tolerance parameters, variant adaptation, and memory settings. Optimizes a 10-dimensional Ackley problem. Outputs best fitness, function evaluations, generations, and wall time.

### knapsack_optimization_example.cpp

Demonstrates knapsack problem optimization using Differential Evolution. Creates a 0-1 knapsack problem with item values and weights, then optimizes it to find the best item selection. Uses continuous encoding where values in [0,1] are thresholded at 0.5 to determine item selection. Outputs best fitness, selected items, total value, total weight, capacity, function evaluations, generations, and wall time.

### knapsack_hpo_example.cpp

Demonstrates hyperparameter optimization for the knapsack problem. Uses CMA-ES to tune Differential Evolution parameters while optimizing a knapsack problem instance. Shows how to apply hyperparameter optimization to combinatorial problems. Outputs best objective, number of trials, best hyperparameters, function evaluations, and wall time.

### knapsack_pso_sa_example.cpp

Demonstrates pso with simulated annealing hyperparameter optimization for knapsack problem. Uses simulated annealing to tune particle swarm optimization parameters. Shows alternative hyperparameter optimizer and evolutionary algorithm combination. Outputs best objective, number of trials, best hyperparameters, function evaluations, and wall time.

## Output Format

All examples output results in key:value format. Error messages are written to stderr with "error:" prefix. Successful execution returns exit code 0.
