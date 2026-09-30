cmake_minimum_required(VERSION 3.16)
project(causal_branch_attention LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

add_library(causal3
    src/engine.cpp
    src/reference.cpp
)
target_include_directories(causal3 PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_compile_options(causal3 PRIVATE -Wall -Wextra -Wpedantic)

enable_testing()

add_executable(test_engine tests/test_engine.cpp)
target_link_libraries(test_engine PRIVATE causal3)
target_compile_options(test_engine PRIVATE -Wall -Wextra -Wpedantic)

# Enumerate test cases straight from REGISTER_TEST() declarations in the test
# source, so CTest cannot silently run zero tests when cases are renamed or
# registration breaks.
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_engine.cpp" test_source)
# Drop preprocessor lines so the macro definition is not mistaken for a case.
string(REGEX REPLACE "[^\n]*#define[^\n]*\\n" "" test_body
       "${test_source}")
string(REGEX MATCHALL "REGISTER_TEST\\([A-Za-z0-9_]+\\)" test_declarations
       "${test_body}")

set(registered_count 0)
foreach(decl IN LISTS test_declarations)
    string(REGEX REPLACE "REGISTER_TEST\\(([A-Za-z0-9_]+)\\)" "\\1"
           case_name "${decl}")
    add_test(NAME ${case_name} COMMAND test_engine ${case_name})
    math(EXPR registered_count "${registered_count} + 1")
endforeach()

# Guard: configuration itself fails if the runner advertises no cases.
if(registered_count EQUAL 0)
    message(FATAL_ERROR "No test cases were registered; CTest must never run zero tests")
endif()

# Guard test: the compiled binary must advertise the same set of names.
add_test(NAME __test_case_list_guard
         COMMAND ${CMAKE_COMMAND}
                 -DTEST_BINARY=$<TARGET_FILE:test_engine>
                 -DEXPECTED_COUNT=${registered_count}
                 -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/test_case_list_guard.cmake)

# Runtime guard invoked via CTest: fail if `ctest -N` discovers fewer tests
# than the cases parsed at configure time.
add_test(NAME __ctest_nonempty_guard
         COMMAND ${CMAKE_COMMAND}
                 -DBINARY_DIR=${CMAKE_BINARY_DIR}
                 -DEXPECTED_MIN=${registered_count}
                 -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/ctest_nonempty_guard.cmake)
message(STATUS "Registered ${registered_count} causal3 test case(s) with CTest")
