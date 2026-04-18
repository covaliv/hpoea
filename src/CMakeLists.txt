set(HPOEA_CORE_SOURCES
    config/config_parser.cpp
    config/config_descriptors.cpp
    config/config_validator.cpp
    config/suite_expander.cpp
    core/baseline_optimizer.cpp
    core/error_classification.cpp
    core/experiment.cpp
    core/logging.cpp
    core/parameters.cpp
    core/search_space.cpp
    wrappers/problems/benchmark_problems.cpp
)

add_library(hpoea_core ${HPOEA_CORE_SOURCES})

target_include_directories(hpoea_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
        $<INSTALL_INTERFACE:include>
)

target_compile_features(hpoea_core PUBLIC cxx_std_20)

target_compile_definitions(hpoea_core
    PUBLIC
        $<$<BOOL:${HPOEA_WITH_PAGMO}>:HPOEA_CONFIG_HAS_PAGMO=1>
)

target_link_libraries(hpoea_core
    PRIVATE
        tomlplusplus::tomlplusplus
)

if (HPOEA_WITH_PAGMO)
    if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/wrappers/pagmo/CMakeLists.txt)
        add_subdirectory(wrappers/pagmo)
    endif ()
endif ()
