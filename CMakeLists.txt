cmake_minimum_required(VERSION 3.20)

project(hpoea
        VERSION 0.1.0
        DESCRIPTION "Hyperparameter optimization framework for evolutionary algorithms"
        LANGUAGES CXX)

option(HPOEA_WITH_PAGMO "Enable Pagmo2-based wrappers" OFF)
option(HPOEA_BUILD_TESTS "Build the test suite" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    add_compile_options(-Wall -Wextra -Wpedantic)
elseif (MSVC)
    add_compile_options(/W4 /permissive-)
endif ()

include(FetchContent)
find_package(tomlplusplus CONFIG QUIET)
if (NOT tomlplusplus_FOUND)
    FetchContent_Declare(
        tomlplusplus
        GIT_REPOSITORY https://github.com/marzer/tomlplusplus.git
        GIT_TAG v3.4.0
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(tomlplusplus)
endif ()

add_subdirectory(src)
add_subdirectory(apps)

if (HPOEA_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif ()
