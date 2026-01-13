set(HPOEA_CORE_SOURCES
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
        $<$<BOOL:${HPOEA_WITH_PAGMO}>:HPOEA_WITH_PAGMO>)

if (HPOEA_WITH_PAGMO)
    if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/wrappers/pagmo/CMakeLists.txt)
        add_subdirectory(wrappers/pagmo)
    endif ()
endif ()

