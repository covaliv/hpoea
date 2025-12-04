#!/bin/bash

set -e

echo "-Test Suite-"
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
cmake --build . --target hpoea_wrapper_test hpoea_hyper_test \
    hpoea_benchmark_problems_test hpoea_parallel_experiment_test \
    hpoea_integration_test hpoea_benchmark_suite hpoea_correctness_test

echo ""
echo "=== Running Tests ==="

echo ""
echo "1. wrapper tests (de, pso, sade)"
./tests/hpoea_wrapper_test || exit 1

echo ""
echo "2. hyper optimizer tests (cmaes, sa, pso, nm)"
./tests/hpoea_hyper_test || exit 1

echo ""
echo "3. benchmark problems"
./tests/hpoea_benchmark_problems_test || exit 1

echo ""
echo "4. parallel experiments"
HPOEA_RUN_PARALLEL_TESTS=1 ./tests/hpoea_parallel_experiment_test || exit 1

echo ""
echo "5. integration test"
./tests/hpoea_integration_test || exit 1

echo ""
echo "6. correctness test"
./tests/hpoea_correctness_test || exit 1

echo ""
echo "7. benchmark suite"
./tests/hpoea_benchmark_suite || exit 1

echo ""
echo "=== All Tests Passed ==="
echo ""
echo "verbose output: HPOEA_LOG_RESULTS=1 ./tests/hpoea_wrapper_test"

