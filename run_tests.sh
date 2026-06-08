#!/usr/bin/env bash

set -euo pipefail

readonly CORE_BUILD_DIR="build/hpoea-core"
readonly PAGMO_BUILD_DIR="build/hpoea-pagmo"

show_help() {
    cat <<'EOF'
Usage: ./run_tests.sh [--core-only] [--no-pagmo] [--pagmo-dir PATH] [--help]

Options:
  --core-only         Run only the core configure/build/test flow.
  --no-pagmo          Skip the Pagmo-enabled flow (alias for --core-only).
  --pagmo-dir PATH    Pass PATH as -DPagmo_DIR=... for the Pagmo-enabled flow.
  --help              Show this help message.

Examples:
  ./run_tests.sh
  ./run_tests.sh --core-only
  ./run_tests.sh --no-pagmo
  ./run_tests.sh --pagmo-dir ~/.local/lib/cmake/pagmo

By default, the script runs both the core and the Pagmo-enabled build/test flows.
Use --core-only (or --no-pagmo) to skip the Pagmo-enabled flow.
EOF
}

run_core_suite() {
    printf '%s\n' '=== Core test suite ==='
    cmake -S . -B "$CORE_BUILD_DIR" -DCMAKE_BUILD_TYPE=Release -DHPOEA_BUILD_TESTS=ON
    cmake --build "$CORE_BUILD_DIR"
    ctest --test-dir "$CORE_BUILD_DIR" --show-only -L hpoea-core
    ctest --test-dir "$CORE_BUILD_DIR" -L hpoea-core --output-on-failure
}

run_pagmo_suite() {
    local -a cmake_args=(
        -S .
        -B "$PAGMO_BUILD_DIR"
        -DCMAKE_BUILD_TYPE=Release
        -DHPOEA_BUILD_TESTS=ON
        -DHPOEA_WITH_PAGMO=ON
    )

    if [[ -n "${pagmo_dir}" ]]; then
        cmake_args+=("-DPagmo_DIR=${pagmo_dir}")
    fi

    printf '%s\n' '=== Pagmo test suite ==='
    if ! cmake "${cmake_args[@]}"; then
        cat >&2 <<EOF
error: failed to configure the Pagmo-enabled build.

Make sure Pagmo is installed and discoverable by CMake.
If it is installed in a non-standard location, rerun with:
  ./run_tests.sh --pagmo-dir /path/to/pagmo
EOF
        exit 1
    fi

    cmake --build "$PAGMO_BUILD_DIR"
    ctest --test-dir "$PAGMO_BUILD_DIR" --show-only -L hpoea-pagmo
    ctest --test-dir "$PAGMO_BUILD_DIR" -L hpoea-pagmo --output-on-failure
}

run_pagmo=true
pagmo_dir="${Pagmo_DIR:-}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --core-only|--no-pagmo)
            run_pagmo=false
            shift
            ;;
        --with-pagmo)
            run_pagmo=true
            shift
            ;;
        --pagmo-dir)
            if [[ $# -lt 2 ]]; then
                echo 'error: --pagmo-dir requires a path argument' >&2
                exit 1
            fi
            pagmo_dir="$2"
            shift 2
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        *)
            echo "error: unknown option: $1" >&2
            echo 'Run ./run_tests.sh --help for usage.' >&2
            exit 1
            ;;
    esac
done

printf '%s\n\n' '-Test Suite-'

run_core_suite

if [[ "$run_pagmo" == true ]]; then
    printf '\n'
    run_pagmo_suite
fi

printf '\n%s\n' '=== Requested test suites passed ==='
