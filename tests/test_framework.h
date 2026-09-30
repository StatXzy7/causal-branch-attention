// SPDX-License-Identifier: MIT
/// @file test_framework.h
/// @brief Minimal zero-dependency unit-test framework. Tests register
/// themselves in a global registry; the runner fails (non-zero exit) when the
/// registry contains zero tests, so CTest cannot silently pass an empty suite.
#pragma once

#include <cmath>
#include <cstring>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace tri_attn_test {

struct TestCase {
    std::string name;
    std::function<void()> fn;
};

inline std::vector<TestCase>& registry() {
    static std::vector<TestCase> r;
    return r;
}

struct Registrar {
    Registrar(const char* name, std::function<void()> fn) {
        registry().push_back({name, std::move(fn)});
    }
};

struct AssertionError : std::runtime_error {
    explicit AssertionError(const std::string& msg) : std::runtime_error(msg) {}
};

template <typename T>
std::string to_string_v(const T& value) {
    std::ostringstream os;
    os << value;
    return os.str();
}

template <typename T>
std::string to_string_v(const std::vector<T>& value) {
    std::ostringstream os;
    os << "[";
    for (std::size_t i = 0; i < value.size(); ++i) {
        if (i) os << ", ";
        os << value[i];
    }
    os << "]";
    return os.str();
}

inline bool close(float a, float b,
                  float atol = 1e-5f, float rtol = 1e-5f) {
    const float d = std::fabs(a - b);
    return d <= atol + rtol * std::max(std::fabs(a), std::fabs(b));
}

// Bit-for-bit equality for float vectors (same code path must produce the
// same bytes).
inline bool same_bits(const std::vector<float>& a,
                      const std::vector<float>& b) {
    return a.size() == b.size() &&
           std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

}  // namespace tri_attn_test

#define TA_CONCAT_INNER(a, b) a##b
#define TA_CONCAT(a, b) TA_CONCAT_INNER(a, b)

#define TEST_CASE(NAME)                                                        \
    static void TA_CONCAT(ta_test_fn_, __LINE__)();                            \
    static ::tri_attn_test::Registrar TA_CONCAT(ta_test_reg_, __LINE__)(       \
        NAME, &TA_CONCAT(ta_test_fn_, __LINE__));                              \
    static void TA_CONCAT(ta_test_fn_, __LINE__)()

#define TA_FAIL(msg)                                                           \
    throw ::tri_attn_test::AssertionError(                                     \
        std::string(msg) + " (" + __FILE__ + ":" + std::to_string(__LINE__) +  \
        ")")

#define REQUIRE(cond)                                                          \
    do {                                                                       \
        if (!(cond)) TA_FAIL(std::string("REQUIRE failed: ") + #cond);         \
    } while (0)

#define REQUIRE_EQ(actual, expected)                                          \
    do {                                                                       \
        auto ta_a = (actual);                                                  \
        auto ta_e = (expected);                                                \
        if (!(ta_a == ta_e))                                                   \
            TA_FAIL(std::string("REQUIRE_EQ failed: ") + #actual + " == " +    \
                    #expected + " (got " +                                     \
                    ::tri_attn_test::to_string_v(ta_a) + ", want " +           \
                    ::tri_attn_test::to_string_v(ta_e) + ")");                 \
    } while (0)

#define REQUIRE_CLOSE(actual, expected)                                        \
    do {                                                                       \
        float ta_a = static_cast<float>(actual);                               \
        float ta_e = static_cast<float>(expected);                             \
        if (!::tri_attn_test::close(ta_a, ta_e))                               \
            TA_FAIL(std::string("REQUIRE_CLOSE failed: ") + #actual + " ~= " + \
                    #expected + " (got " +                                     \
                    ::tri_attn_test::to_string_v(ta_a) + ", want " +           \
                    ::tri_attn_test::to_string_v(ta_e) + ")");                 \
    } while (0)

#define REQUIRE_THROWS(expr, ex_type)                                         \
    do {                                                                       \
        bool ta_threw = false;                                                 \
        try {                                                                  \
            expr;                                                              \
        } catch (const ex_type&) {                                             \
            ta_threw = true;                                                   \
        } catch (...) {                                                        \
            TA_FAIL(std::string("REQUIRE_THROWS: wrong exception type: ") +    \
                    #expr);                                                    \
        }                                                                      \
        if (!ta_threw)                                                         \
            TA_FAIL(std::string("REQUIRE_THROWS: nothing thrown: ") + #expr);  \
    } while (0)
