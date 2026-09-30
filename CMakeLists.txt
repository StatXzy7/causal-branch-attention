cmake_minimum_required(VERSION 3.16)
project(causal_branch_attention LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

add_library(tri_attn
    src/tri_attn.cpp
    src/naive.cpp
)
target_include_directories(tri_attn PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(tri_attn PRIVATE -Wall -Wextra -Wpedantic)
endif()

enable_testing()

add_executable(unit_tests
    tests/test_main.cpp
    tests/test_selection.cpp
    tests/test_gating.cpp
    tests/test_safety.cpp
)
target_link_libraries(unit_tests PRIVATE tri_attn)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(unit_tests PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_test(NAME unit_tests COMMAND unit_tests)

# List the registered cases; a broken build that registers nothing fails here.
add_test(NAME test_cases_registered
         COMMAND ${CMAKE_COMMAND}
                 -DTEST_BIN=$<TARGET_FILE:unit_tests>
                 -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/check_test_count.cmake)

# The runner must exit non-zero when zero tests are registered
# (simulated through TA_SIMULATE_EMPTY_REGISTRY); this CTest case expects that
# failure via WILL_FAIL.
add_test(NAME zero_tests_are_a_failure
         COMMAND ${CMAKE_COMMAND}
                 -DTEST_BIN=$<TARGET_FILE:unit_tests>
                 -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/check_empty_registry.cmake)

add_executable(demo examples/demo.cpp)
target_link_libraries(demo PRIVATE tri_attn)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(demo PRIVATE -Wall -Wextra -Wpedantic)
endif()
