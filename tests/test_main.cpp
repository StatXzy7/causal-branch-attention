// SPDX-License-Identifier: MIT
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <string>

#include "test_framework.h"

int main(int argc, char** argv) {
    bool list_only = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--list") == 0) list_only = true;
    }

    auto& tests = ::tri_attn_test::registry();
    const bool simulate_empty =
        std::getenv("TA_SIMULATE_EMPTY_REGISTRY") != nullptr;
    if (simulate_empty) {
        std::cerr << "FATAL: zero test cases registered\n";
        return 2;
    }
    if (list_only) {
        std::cout << tests.size() << " test case(s) registered\n";
        for (const auto& t : tests) std::cout << "  " << t.name << "\n";
        return 0;
    }

    // Zero registered tests is a hard failure: a build/link problem that
    // dropped the suite must not look like a green CTest run.
    if (tests.empty()) {
        std::cerr << "FATAL: zero test cases registered\n";
        return 2;
    }

    int failed = 0;
    for (const auto& t : tests) {
        try {
            t.fn();
            std::cout << "[ PASS ] " << t.name << "\n";
        } catch (const std::exception& ex) {
            ++failed;
            std::cout << "[ FAIL ] " << t.name << ": " << ex.what() << "\n";
        } catch (...) {
            ++failed;
            std::cout << "[ FAIL ] " << t.name << ": unknown exception\n";
        }
    }
    std::cout << (tests.size() - failed) << "/" << tests.size()
              << " test case(s) passed\n";
    return failed == 0 ? 0 : 1;
}
