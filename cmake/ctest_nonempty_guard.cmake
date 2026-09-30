# Fails when `ctest -N` discovers fewer than EXPECTED_MIN tests.
execute_process(
    COMMAND ${CMAKE_CTEST_COMMAND} -N
    WORKING_DIRECTORY "${BINARY_DIR}"
    OUTPUT_VARIABLE ctest_listing
    RESULT_VARIABLE ctest_result)

if(NOT ctest_result EQUAL 0)
    message(FATAL_ERROR "ctest -N failed")
endif()

string(REGEX MATCH "Total Tests: ([0-9]+)" _match "${ctest_listing}")
if(NOT _match)
    message(FATAL_ERROR "Could not parse CTest test count from:\n${ctest_listing}")
endif()
set(discovered ${CMAKE_MATCH_1})
if(discovered LESS EXPECTED_MIN)
    message(FATAL_ERROR
        "CTest discovered only ${discovered} test(s), expected at least "
        "${EXPECTED_MIN}. Zero/partial test runs are not allowed.")
endif()
message(STATUS "CTest non-empty guard: ${discovered} tests discovered")
