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
./build/apps/01_basic_ea_optimization
./build/apps/02_basic_hyperparameter_optimization
./build/apps/03_experiment_management
./build/apps/04_custom_problem
./build/apps/05_multi_optimizer_comparison
./build/apps/06_custom_parameter_space
```

## Example Descriptions

### 01_basic_ea_optimization.cpp

Demonstrates basic evolutionary algorithm usage. Configures Differential Evolution algorithm with specified parameters and optimizes a 10-dimensional Sphere problem. Outputs best fitness, function evaluations, generations, and wall time.

### 02_basic_hyperparameter_optimization.cpp

Demonstrates hyperparameter optimization using CMA-ES to tune DE parameters. Configures CMA-ES optimizer and optimizes DE hyperparameters on an 8-dimensional Rosenbrock problem. Outputs best objective value, number of trials, best hyperparameters, function evaluations, and wall time.

### 03_experiment_management.cpp

Demonstrates experiment management with structured logging. Uses SequentialExperimentManager to run multiple trials with logging to JSON Lines format. Outputs experiment ID, number of optimizer runs, best objective, trials count, function evaluations, and best hyperparameters.

### 04_custom_problem.cpp

Demonstrates custom problem implementation. Implements a Shifted Sphere problem by extending IProblem interface. Shows how to define custom problem metadata, bounds, and evaluation function. Outputs best fitness, distance to optimum, and function evaluations.

### 05_multi_optimizer_comparison.cpp

Compares multiple hyperparameter optimizers (CMA-ES, Simulated Annealing, PSO-based) on the same problem. Runs experiments for each optimizer and compares results. Outputs optimizer name, best objective, trials count, function evaluations, and ranked comparison results.

### 06_custom_parameter_space.cpp

Demonstrates custom parameter space definition and validation. Creates a ParameterSpace with custom descriptors, generates random valid configurations, validates them, and tests a sampled configuration with an algorithm. Outputs generated configurations and optimization results.

## Output Format

All examples output results in key:value format. Error messages are written to stderr with "error:" prefix. Successful execution returns exit code 0.
