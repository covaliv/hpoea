function(hpoea_add_test target source)
    cmake_parse_arguments(HPOEA_TEST "" "LABEL" "LIBS;INCLUDE_DIRS" ${ARGN})

    add_executable(${target} ${source})
    target_link_libraries(${target} PRIVATE ${HPOEA_TEST_LIBS})
    target_compile_features(${target} PRIVATE cxx_std_20)
    if (HPOEA_TEST_INCLUDE_DIRS)
        target_include_directories(${target} PRIVATE ${HPOEA_TEST_INCLUDE_DIRS})
    endif ()

    add_test(NAME ${target} COMMAND ${target})
    if (HPOEA_TEST_LABEL)
        set_tests_properties(${target} PROPERTIES LABELS ${HPOEA_TEST_LABEL})
    endif ()
endfunction()

set(HPOEA_PAGMO_TEST_PRIVATE_INCLUDES
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/src/wrappers/pagmo
)

hpoea_add_test(hpoea_types_budget_tests types_budget_tests.cpp
    LABEL hpoea-core
    LIBS hpoea_core)

hpoea_add_test(hpoea_parameter_space_tests parameter_space_tests.cpp
    LABEL hpoea-core
    LIBS hpoea_core)

hpoea_add_test(hpoea_search_space_tests search_space_tests.cpp
    LABEL hpoea-core
    LIBS hpoea_core)

hpoea_add_test(hpoea_logging_tests logging_tests.cpp
    LABEL hpoea-core
    LIBS hpoea_core)

hpoea_add_test(hpoea_benchmark_problem_tests benchmark_problem_tests.cpp
    LABEL hpoea-core
    LIBS hpoea_core)

hpoea_add_test(hpoea_transform_bounds_tests transform_bounds_tests.cpp
    LABEL hpoea-core
    LIBS hpoea_core)

hpoea_add_test(hpoea_config_parser_tests config_parser_tests.cpp
    LABEL hpoea-core
    LIBS hpoea_core)
target_compile_definitions(hpoea_config_parser_tests
    PRIVATE
        HPOEA_PROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
)

hpoea_add_test(hpoea_config_validator_tests config_validator_tests.cpp
    LABEL hpoea-core
    LIBS hpoea_core)
target_compile_definitions(hpoea_config_validator_tests
    PRIVATE
        HPOEA_PROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
)

if (HPOEA_WITH_PAGMO)
    set_tests_properties(hpoea_config_validator_tests
        PROPERTIES LABELS "hpoea-core;hpoea-pagmo")
endif ()

hpoea_add_test(hpoea_suite_expander_tests suite_expander_tests.cpp
    LABEL hpoea-core
    LIBS hpoea_core)
target_compile_definitions(hpoea_suite_expander_tests
    PRIVATE
        HPOEA_PROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
)

if (HPOEA_WITH_PAGMO)
    hpoea_add_test(hpoea_problem_adapter_tests problem_adapter_tests.cpp
        LABEL hpoea-pagmo
        LIBS hpoea_pagmo hpoea_core
        INCLUDE_DIRS ${HPOEA_PAGMO_TEST_PRIVATE_INCLUDES})

    hpoea_add_test(hpoea_budget_util_tests budget_util_tests.cpp
        LABEL hpoea-pagmo
        LIBS hpoea_pagmo hpoea_core
        INCLUDE_DIRS ${HPOEA_PAGMO_TEST_PRIVATE_INCLUDES})

    hpoea_add_test(hpoea_evolutionary_algorithms_tests evolutionary_algorithms_tests.cpp
        LABEL hpoea-pagmo
        LIBS hpoea_pagmo hpoea_core)

    hpoea_add_test(hpoea_hyper_optimizer_tests hyper_optimizer_tests.cpp
        LABEL hpoea-pagmo
        LIBS hpoea_pagmo hpoea_core)

    hpoea_add_test(hpoea_experiment_manager_tests experiment_manager_tests.cpp
        LABEL hpoea-pagmo
        LIBS hpoea_pagmo hpoea_core)

    hpoea_add_test(hpoea_baseline_optimizer_tests baseline_optimizer_tests.cpp
        LABEL hpoea-pagmo
        LIBS hpoea_pagmo hpoea_core)

    hpoea_add_test(hpoea_hyper_tuning_udp_tests hyper_tuning_udp_tests.cpp
        LABEL hpoea-pagmo
        LIBS hpoea_pagmo hpoea_core
        INCLUDE_DIRS ${HPOEA_PAGMO_TEST_PRIVATE_INCLUDES})
endif ()
