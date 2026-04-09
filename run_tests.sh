#!/bin/bash

set -euo pipefail

CORE_BUILD_DIR=".omc/thesis-core"
PAGMO_BUILD_DIR=".omc/thesis-pagmo"

printf '%s\n\n' '-Test Suite-'

echo '=== Core thesis suite ==='
cmake -S . -B "$CORE_BUILD_DIR" -DHPOEA_BUILD_TESTS=ON
cmake --build "$CORE_BUILD_DIR"
ctest --test-dir "$CORE_BUILD_DIR" --show-only -L thesis-core
ctest --test-dir "$CORE_BUILD_DIR" -L thesis-core --output-on-failure

echo
echo '=== Pagmo thesis suite ==='
cmake -S . -B "$PAGMO_BUILD_DIR" -DHPOEA_BUILD_TESTS=ON -DHPOEA_WITH_PAGMO=ON
cmake --build "$PAGMO_BUILD_DIR"
ctest --test-dir "$PAGMO_BUILD_DIR" --show-only -L thesis-pagmo
ctest --test-dir "$PAGMO_BUILD_DIR" -L thesis-pagmo --output-on-failure

echo
echo '=== All thesis suites passed ==='
