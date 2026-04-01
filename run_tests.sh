#!/bin/bash

set -e

echo "-Test Suite-"
echo ""

if [ ! -d "build" ]; then
    echo "creating build directory..."
    mkdir -p build
fi

cd build

echo "configuring cmake..."
cmake .. -DHPOEA_WITH_PAGMO=ON -DHPOEA_BUILD_TESTS=ON

echo "building tests..."
cmake --build .

echo ""
echo "=== Running Tests ==="

echo ""
echo "1. types/budget tests"
./tests/hpoea_types_budget_tests || exit 1

echo ""
echo "2. parameter space tests"
./tests/hpoea_parameter_space_tests || exit 1

echo ""
echo "3. search space tests"
./tests/hpoea_search_space_tests || exit 1

echo ""
echo "4. logging tests"
./tests/hpoea_logging_tests || exit 1

echo ""
echo "5. benchmark problem tests"
./tests/hpoea_benchmark_problem_tests || exit 1

echo ""
echo "6. problem adapter tests"
./tests/hpoea_problem_adapter_tests || exit 1

echo ""
echo "7. budget util tests"
./tests/hpoea_budget_util_tests || exit 1

echo ""
echo "8. hyper tuning udp tests"
./tests/hpoea_hyper_tuning_udp_tests || exit 1

echo ""
echo "9. hyper util tests"
./tests/hpoea_hyper_util_tests || exit 1

echo ""
echo "10. evolutionary algorithm tests"
./tests/hpoea_evolutionary_algorithms_tests || exit 1

echo ""
echo "11. hyper optimizer tests"
./tests/hpoea_hyper_optimizer_tests || exit 1

echo ""
echo "12. experiment manager tests"
./tests/hpoea_experiment_manager_tests || exit 1

echo ""
echo "13. parallel error handling tests"
./tests/hpoea_parallel_error_handling_test || exit 1

echo ""
echo "14. baseline optimizer tests"
./tests/hpoea_baseline_optimizer_tests || exit 1

echo ""
echo "15. clone tests"
./tests/hpoea_clone_tests || exit 1

echo ""
echo "16. factory contract tests"
./tests/hpoea_factory_contract_tests || exit 1

echo ""
echo "17. transform bounds tests"
./tests/hpoea_transform_bounds_tests || exit 1

echo ""
echo "18. hyper clone tests"
./tests/hpoea_hyper_clone_tests || exit 1

echo ""
echo "19. fix regression tests"
./tests/hpoea_fix_regression_tests || exit 1

echo ""
echo "20. coverage gap tests"
./tests/hpoea_coverage_gap_tests || exit 1

echo ""
echo "=== All Tests Passed ==="
echo ""
