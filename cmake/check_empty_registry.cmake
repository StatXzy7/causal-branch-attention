# With TA_SIMULATE_EMPTY_REGISTRY=1 the runner must report zero tests and
# exit non-zero. This wrapper FAILS if that simulated run succeeds, so wire it
# as a WILL_FAIL test.
set(ENV{TA_SIMULATE_EMPTY_REGISTRY} "1")
execute_process(
    COMMAND ${TEST_BIN}
    RESULT_VARIABLE rc)
if(rc EQUAL 0)
    message(FATAL_ERROR "runner unexpectedly succeeded with zero tests")
endif()
message(STATUS "runner correctly exited non-zero (${rc}) for empty registry")
