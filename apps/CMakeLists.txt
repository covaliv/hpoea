if (HPOEA_WITH_PAGMO)
    add_executable(hpoea_simple_example simple_example.cpp)
    target_link_libraries(hpoea_simple_example PRIVATE hpoea_pagmo hpoea_core)
    target_compile_features(hpoea_simple_example PRIVATE cxx_std_20)

    add_executable(hpoea_app_correctness_test correctness_test.cpp)
    target_link_libraries(hpoea_app_correctness_test PRIVATE hpoea_pagmo hpoea_core)
    target_compile_features(hpoea_app_correctness_test PRIVATE cxx_std_20)

    add_executable(hpoea_sfu_benchmark_test sfu_benchmark_test.cpp)
    target_link_libraries(hpoea_sfu_benchmark_test PRIVATE hpoea_pagmo hpoea_core)
    target_compile_features(hpoea_sfu_benchmark_test PRIVATE cxx_std_20)

    add_executable(basic_ea_example basic_ea_example.cpp)
    target_link_libraries(basic_ea_example PRIVATE hpoea_pagmo hpoea_core)
    target_compile_features(basic_ea_example PRIVATE cxx_std_20)

    add_executable(basic_hpo_example basic_hpo_example.cpp)
    target_link_libraries(basic_hpo_example PRIVATE hpoea_pagmo hpoea_core)
    target_compile_features(basic_hpo_example PRIVATE cxx_std_20)

    add_executable(experiment_management_example experiment_management_example.cpp)
    target_link_libraries(experiment_management_example PRIVATE hpoea_pagmo hpoea_core)
    target_compile_features(experiment_management_example PRIVATE cxx_std_20)

    add_executable(custom_problem_example custom_problem_example.cpp)
    target_link_libraries(custom_problem_example PRIVATE hpoea_pagmo hpoea_core)
    target_compile_features(custom_problem_example PRIVATE cxx_std_20)

    add_executable(optimizer_comparison_example optimizer_comparison_example.cpp)
    target_link_libraries(optimizer_comparison_example PRIVATE hpoea_pagmo hpoea_core)
    target_compile_features(optimizer_comparison_example PRIVATE cxx_std_20)

    add_executable(custom_parameter_space_example custom_parameter_space_example.cpp)
    target_link_libraries(custom_parameter_space_example PRIVATE hpoea_pagmo hpoea_core)
    target_compile_features(custom_parameter_space_example PRIVATE cxx_std_20)

    add_executable(cmaes_optimization_example cmaes_optimization_example.cpp)
    target_link_libraries(cmaes_optimization_example PRIVATE hpoea_pagmo hpoea_core)
    target_compile_features(cmaes_optimization_example PRIVATE cxx_std_20)

    add_executable(sga_optimization_example sga_optimization_example.cpp)
    target_link_libraries(sga_optimization_example PRIVATE hpoea_pagmo hpoea_core)
    target_compile_features(sga_optimization_example PRIVATE cxx_std_20)

    add_executable(de1220_optimization_example de1220_optimization_example.cpp)
    target_link_libraries(de1220_optimization_example PRIVATE hpoea_pagmo hpoea_core)
    target_compile_features(de1220_optimization_example PRIVATE cxx_std_20)

    add_executable(knapsack_optimization_example knapsack_optimization_example.cpp)
    target_link_libraries(knapsack_optimization_example PRIVATE hpoea_pagmo hpoea_core)
    target_compile_features(knapsack_optimization_example PRIVATE cxx_std_20)

    add_executable(knapsack_hpo_example knapsack_hpo_example.cpp)
    target_link_libraries(knapsack_hpo_example PRIVATE hpoea_pagmo hpoea_core)
    target_compile_features(knapsack_hpo_example PRIVATE cxx_std_20)

    add_executable(knapsack_pso_sa_example knapsack_pso_sa_example.cpp)
    target_link_libraries(knapsack_pso_sa_example PRIVATE hpoea_pagmo hpoea_core)
    target_compile_features(knapsack_pso_sa_example PRIVATE cxx_std_20)
endif ()

