## goal

these notes explain how to add an evolutionary algorithm or optimizer by following the current structure.

## adding an evolutionary algorithm

1. create a header under `include/hpoea/wrappers/...`. look at `include/hpoea/wrappers/pagmo/de_algorithm.hpp` to see the required functions. inherit from `core::IEvolutionaryAlgorithm` and add a factory class that derives from `core::IEvolutionaryAlgorithmFactory`. the factory lets the hyper optimizers create copies.

2. write the implementation in `src/wrappers/...`. use pagmo or another solver to run the algorithm. convert the `core::Budget` into your solver's limits and fill `core::OptimizationResult` with the best value, solution, and how much budget you used. expose all tunable options through `core::ParameterSpace`.

3. add the new `.cpp` file to `src/wrappers/pagmo/CMakeLists.txt` so it builds into `hpoea_pagmo`. rerun cmake with `-DHPOEA_WITH_PAGMO=ON` to refresh the build files.

4. show the new algorithm in one of the apps, like `apps/07_cmaes_optimization.cpp`, or add a new simple program. that gives others a runnable example and doubles as documentation.

## adding a hyperparameter optimizer

1. add a header under `include/hpoea/wrappers/...` similar to `include/hpoea/wrappers/pagmo/cmaes_hyper.hpp`. inherit from `core::IHyperparameterOptimizer` and define `identity()`, `parameter_space()`, `configure()`, and `optimize()`.

2. implement `optimize()` in `src/wrappers/...`. it should take the factory, create an algorithm, set the parameters, and track trials inside `core::HyperparameterOptimizationResult`. also honor the `core::Budget` so trials stop when the limits are reached.

3. add the new source file to `HPOEA_PAGMO_SOURCES` so cmake builds it with the rest of the pagmo wrappers.

4. give the optimizer a simple example in `apps/`, copying the style of `apps/11_knapsack_hyperparameter_optimization.cpp`.

## checklist

- place headers in `include/hpoea/...` and implementations in `src/...`
- describe parameters with `core::ParameterSpace` and set defaults
- set clear `identity.family` and `identity.implementation` strings
- list sources in `src/wrappers/pagmo/CMakeLists.txt`
- add a runnable `apps/*` example so others can build and run it

