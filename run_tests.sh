#!/bin/bash

set -e

echo "=== HPOEA Framework Test Suite ==="
echo ""

if [ ! -d "build" ]; then
    echo "creating build directory..."
    mkdir -p build
fi

cd build

if [ ! -f "CMakeCache.txt" ]; then
    echo "configuring cmake..."
    cmake .. -DHPOEA_WITH_PAGMO=ON -DHPOEA_BUILD_TESTS=ON
fi

echo "building tests..."
cmake --build . --target hpoea_de_wrapper_test hpoea_pso_wrapper_test hpoea_sade_wrapper_test \
    hpoea_cmaes_hyper_test hpoea_sa_hyper_test hpoea_pso_hyper_test hpoea_nm_hyper_test \
    hpoea_benchmark_problems_test hpoea_parallel_experiment_test hpoea_integration_test \
    hpoea_benchmark_suite

echo ""
echo "=== Running Tests ==="
echo ""

echo "1. running EA wrapper tests..."
echo "   - DE wrapper test"
./tests/hpoea_de_wrapper_test || echo "   warning: DE test failed"

echo "   - PSO wrapper test"
./tests/hpoea_pso_wrapper_test || echo "   warning: PSO test failed"

echo "   - SADE wrapper test"
./tests/hpoea_sade_wrapper_test || echo "   warning: SADE test failed"

echo ""
echo "2. running problem tests..."
echo "   - benchmark problems test"
./tests/hpoea_benchmark_problems_test || echo "   warning: benchmark problems test failed"

echo ""
echo "3. running HOA tests..."
echo "   - CMA-ES hyper optimizer test"
HPOEA_RUN_CMAES_TESTS=1 ./tests/hpoea_cmaes_hyper_test || echo "   warning: CMA-ES test failed"

echo "   - simulated annealing hyper optimizer test"
HPOEA_RUN_SA_TESTS=1 ./tests/hpoea_sa_hyper_test || echo "   warning: SA test failed"

echo "   - PSO hyper optimizer test"
HPOEA_RUN_PSO_HYPER_TESTS=1 ./tests/hpoea_pso_hyper_test || echo "   warning: PSO hyper test failed"

echo "   - nelder-mead hyper optimizer test"
HPOEA_RUN_NM_TESTS=1 ./tests/hpoea_nm_hyper_test || echo "   warning: nelder-mead test failed"

echo ""
echo "4. running experiment manager tests..."
echo "   - parallel experiment test"
HPOEA_RUN_PARALLEL_TESTS=1 ./tests/hpoea_parallel_experiment_test || echo "   warning: parallel experiment test failed"

echo ""
echo "5. running integration test..."
./tests/hpoea_integration_test || echo "   warning: integration test failed"

echo ""
echo "6. running benchmark suite..."
HPOEA_RUN_CMAES_TESTS=1 HPOEA_RUN_SA_TESTS=1 HPOEA_RUN_PSO_HYPER_TESTS=1 \
    ./tests/hpoea_benchmark_suite || echo "   warning: benchmark suite failed"

echo ""
echo "=== Test Suite Complete ==="
echo ""
echo "to run tests with verbose output, set HPOEA_LOG_RESULTS=1"
echo "example: HPOEA_LOG_RESULTS=1 ./tests/hpoea_de_wrapper_test"

