# The compiled test binary must advertise exactly EXPECTED_COUNT cases.
execute_process(
    COMMAND "${TEST_BINARY}" --list
    OUTPUT_VARIABLE listing
    RESULT_VARIABLE result)
if(NOT result EQUAL 0)
    message(FATAL_ERROR "test_engine --list failed")
endif()
string(REGEX MATCHALL "[A-Za-z0-9_]+" names "${listing}")
list(LENGTH names count)
if(count LESS EXPECTED_COUNT)
    message(FATAL_ERROR
        "test_engine advertises ${count} case(s), expected at least "
        "${EXPECTED_COUNT}")
endif()
message(STATUS "Test case list guard: ${count} case(s) advertised")
