find_package(Pagmo REQUIRED)

set(HPOEA_PAGMO_SOURCES
    cmaes_hyper.cpp
    de_algorithm.cpp
    cmaes_algorithm.cpp
    sga_algorithm.cpp
    de1220_algorithm.cpp
    pso_algorithm.cpp
    sade_algorithm.cpp
    sa_hyper.cpp
    pso_hyper.cpp
    nm_hyper.cpp
)

add_library(hpoea_pagmo ${HPOEA_PAGMO_SOURCES})

target_include_directories(hpoea_pagmo
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../../include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(hpoea_pagmo
    PUBLIC
        hpoea_core
        Pagmo::pagmo
)

target_compile_features(hpoea_pagmo PUBLIC cxx_std_20)

