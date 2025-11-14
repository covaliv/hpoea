add_executable(hpoea_cli hpoea_cli.cpp)
target_link_libraries(hpoea_cli PRIVATE hpoea_core)
target_compile_features(hpoea_cli PRIVATE cxx_std_20)

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

    add_executable(01_basic_ea_optimization 01_basic_ea_optimization.cpp)
    target_link_libraries(01_basic_ea_optimization PRIVATE hpoea_pagmo hpoea_core)
    target_compile_features(01_basic_ea_optimization PRIVATE cxx_std_20)

    add_executable(02_basic_hyperparameter_optimization 02_basic_hyperparameter_optimization.cpp)
    target_link_libraries(02_basic_hyperparameter_optimization PRIVATE hpoea_pagmo hpoea_core)
    target_compile_features(02_basic_hyperparameter_optimization PRIVATE cxx_std_20)

    add_executable(03_experiment_management 03_experiment_management.cpp)
    target_link_libraries(03_experiment_management PRIVATE hpoea_pagmo hpoea_core)
    target_compile_features(03_experiment_management PRIVATE cxx_std_20)

    add_executable(04_custom_problem 04_custom_problem.cpp)
    target_link_libraries(04_custom_problem PRIVATE hpoea_pagmo hpoea_core)
    target_compile_features(04_custom_problem PRIVATE cxx_std_20)

    add_executable(05_multi_optimizer_comparison 05_multi_optimizer_comparison.cpp)
    target_link_libraries(05_multi_optimizer_comparison PRIVATE hpoea_pagmo hpoea_core)
    target_compile_features(05_multi_optimizer_comparison PRIVATE cxx_std_20)

    add_executable(06_custom_parameter_space 06_custom_parameter_space.cpp)
    target_link_libraries(06_custom_parameter_space PRIVATE hpoea_pagmo hpoea_core)
    target_compile_features(06_custom_parameter_space PRIVATE cxx_std_20)

    add_executable(07_cmaes_optimization 07_cmaes_optimization.cpp)
    target_link_libraries(07_cmaes_optimization PRIVATE hpoea_pagmo hpoea_core)
    target_compile_features(07_cmaes_optimization PRIVATE cxx_std_20)

    add_executable(08_sga_optimization 08_sga_optimization.cpp)
    target_link_libraries(08_sga_optimization PRIVATE hpoea_pagmo hpoea_core)
    target_compile_features(08_sga_optimization PRIVATE cxx_std_20)

    add_executable(09_de1220_optimization 09_de1220_optimization.cpp)
    target_link_libraries(09_de1220_optimization PRIVATE hpoea_pagmo hpoea_core)
    target_compile_features(09_de1220_optimization PRIVATE cxx_std_20)

    add_executable(10_knapsack_optimization 10_knapsack_optimization.cpp)
    target_link_libraries(10_knapsack_optimization PRIVATE hpoea_pagmo hpoea_core)
    target_compile_features(10_knapsack_optimization PRIVATE cxx_std_20)

    add_executable(11_knapsack_hyperparameter_optimization 11_knapsack_hyperparameter_optimization.cpp)
    target_link_libraries(11_knapsack_hyperparameter_optimization PRIVATE hpoea_pagmo hpoea_core)
    target_compile_features(11_knapsack_hyperparameter_optimization PRIVATE cxx_std_20)

    add_executable(13_knapsack_pso_sa_optimization 13_knapsack_pso_sa_optimization.cpp)
    target_link_libraries(13_knapsack_pso_sa_optimization PRIVATE hpoea_pagmo hpoea_core)
    target_compile_features(13_knapsack_pso_sa_optimization PRIVATE cxx_std_20)
endif ()

