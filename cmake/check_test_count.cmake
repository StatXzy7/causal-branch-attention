# Runs `unit_tests --list` and fails if zero cases are registered.
execute_process(
    COMMAND ${TEST_BIN} --list
    OUTPUT_VARIABLE out
    RESULT_VARIABLE rc)
if(NOT rc EQUAL 0)
    message(FATAL_ERROR "test binary --list failed with code ${rc}")
endif()
string(STRIP "${out}" out)
string(REGEX MATCH "^([0-9]+)" m "${out}")
set(n "${CMAKE_MATCH_1}")
if(n STREQUAL "" OR n LESS 1)
    message(FATAL_ERROR "zero test cases registered; output: ${out}")
endif()
message(STATUS "${n} test cases registered")
