// Self-contained test runner for the three-branch causal attention engine.
// Registered cases are also enumerated by CTest (one ctest entry each).
#include <cmath>
#include <cstdint>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <random>
#include <string>
#include <vector>

#include "causal3/engine.h"
#include "causal3/reference.h"

namespace {

int g_failures = 0;
int g_checks = 0;

#define CHECK(cond)                                                       \
    do {                                                                  \
        ++g_checks;                                                       \
        if (!(cond)) {                                                    \
            ++g_failures;                                                 \
            std::cerr << "FAIL [" << __FILE__ << ":" << __LINE__          \
                      << "] " << #cond << "\n";                           \
        }                                                                 \
    } while (0)

float inf() { return std::numeric_limits<float>::infinity(); }
float quiet_nan() { return std::numeric_limits<float>::quiet_NaN(); }

bool close(float a, float b, float atol = 1e-4f, float rtol = 1e-3f) {
    if (a == b) return true;
    const float diff = std::fabs(a - b);
    if (diff <= atol) return true;
    return diff <= rtol * std::max(std::fabs(a), std::fabs(b));
}

bool close_vec(const std::vector<float>& a, const std::vector<float>& b,
               float atol = 1e-4f, float rtol = 1e-3f) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i)
        if (!close(a[i], b[i], atol, rtol)) {
            std::cerr << "  mismatch at [" << i << "]: " << a[i] << " vs "
                      << b[i] << "\n";
            return false;
        }
    return true;
}

struct Rng {
    explicit Rng(uint32_t seed) : mt(seed) {}
    std::mt19937 mt;
    float f(float lo = -1.5f, float hi = 1.5f) {
        std::uniform_real_distribution<float> d(lo, hi);
        return d(mt);
    }
    int integer(int lo, int hi) {
        std::uniform_int_distribution<int> d(lo, hi);
        return d(mt);
    }
};

using TestFn = std::function<void()>;
std::map<std::string, TestFn>& registry() {
    static std::map<std::string, TestFn> r;
    return r;
}

struct Registrar {
    Registrar(const std::string& name, TestFn fn) { registry()[name] = fn; }
};

#define REGISTER_TEST(name)                                  \
    static void name();                                      \
    static Registrar reg_##name(#name, name);                \
    static void name()

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

std::vector<float> make_data(int T, int dim, uint32_t seed) {
    Rng rng(seed);
    std::vector<float> data(static_cast<std::size_t>(T) * dim);
    for (float& x : data) x = rng.f();
    return data;
}

void expect_step_vs_ref(const causal3::StepResult& a,
                        const causal3::StepResult& b) {
    CHECK(close_vec(a.compressed, b.compressed));
    CHECK(close_vec(a.selected, b.selected));
    CHECK(close_vec(a.windowed, b.windowed));
    CHECK(close_vec(a.output, b.output));
    CHECK(close(a.gate_compressed, b.gate_compressed, 1e-5f, 1e-5f));
    CHECK(close(a.gate_selected, b.gate_selected, 1e-5f, 1e-5f));
    CHECK(close(a.gate_windowed, b.gate_windowed, 1e-5f, 1e-5f));
    CHECK(a.selected_blocks == b.selected_blocks);
    CHECK(a.comp_probs.size() == b.comp_probs.size());
    for (std::size_t i = 0; i < a.comp_probs.size(); ++i)
        CHECK(close(a.comp_probs[i], b.comp_probs[i], 1e-5f, 1e-5f));
}

}  // namespace

// ---------------------------------------------------------------------------
// Test cases
// ---------------------------------------------------------------------------

// Single token: every branch sees exactly one identical row.
REGISTER_TEST(single_token) {
    const int dim = 2;
    const float k0[2] = {0.3f, -0.4f};
    const float v0[2] = {0.5f, 0.7f};
    causal3::Params p{dim, /*block_len=*/4, /*select_count=*/2, /*window=*/3};
    auto engine = causal3::Engine::create(p);
    CHECK(engine != nullptr);
    std::string err;
    CHECK(engine->append(k0, k0, v0, dim, &err));

    const causal3::StepResult s = engine->last();
    CHECK(s.selected_blocks == std::vector<int>{0});
    CHECK(s.comp_probs.size() == 1u);
    CHECK(close(s.comp_probs[0], 1.0f, 1e-6f));
    // Each branch attends only to v0, so branch outputs equal v0.
    CHECK(close_vec(s.compressed, {0.5f, 0.7f}, 1e-6f));
    CHECK(close_vec(s.selected, {0.5f, 0.7f}, 1e-6f));
    CHECK(close_vec(s.windowed, {0.5f, 0.7f}, 1e-6f));
    const float logit =
        (0.3f * 0.3f + -0.4f * -0.4f) / std::sqrt(static_cast<float>(dim));
    const float expected_gate = 1.0f / (1.0f + std::exp(-logit));
    CHECK(close(s.gate_compressed, expected_gate, 1e-6f));
    CHECK(close(s.gate_selected, expected_gate, 1e-6f));
    CHECK(close(s.gate_windowed, expected_gate, 1e-6f));
    // Fused output: three equal gates added without renormalization.
    for (int d = 0; d < dim; ++d)
        CHECK(close(s.output[d], 3.0f * expected_gate * v0[d], 1e-5f));
}

// Hand-checked gate values for two tokens with orthogonal unit keys.
REGISTER_TEST(gate_numeric_two_tokens) {
    const int dim = 2;
    // q0 = k0 = (1,0); k1 = (0,1). Query at t=0 sees only itself.
    causal3::Params p{dim, /*block_len=*/1, /*select_count=*/1, /*window=*/1};
    auto engine = causal3::Engine::create(p);
    const float a[2] = {1.0f, 0.0f};
    const float z[2] = {0.0f, 1.0f};
    const float va[2] = {2.0f, -2.0f};
    const float vz[2] = {-3.0f, 3.0f};
    std::string err;
    CHECK(engine->append(a, a, va, dim, &err));
    causal3::StepResult s0 = engine->last();
    CHECK(close(s0.gate_compressed, 1.0f / (1.0f + std::exp(-0.70710678f)),
                1e-6f));
    CHECK(engine->append(z, z, vz, dim, &err));
    // Query (0,1) against k (1,0) and (0,1), but window=1 so the window branch
    // sees only the current token: logit = 1/sqrt(2).
    causal3::StepResult s1 = engine->last();
    const float logit_self = 0.70710678f;
    CHECK(close(s1.gate_windowed, 1.0f / (1.0f + std::exp(-logit_self)),
                1e-6f));
    // Selection over both blocks: probs for [logit 0, logit 1/sqrt2] -> block 1.
    CHECK(s1.selected_blocks == std::vector<int>{1});
    CHECK(close_vec(s1.selected, {-3.0f, 3.0f}, 1e-6f));
}

// Equal block scores must resolve by ascending original block index; indices
// are returned ascending even after selection.
REGISTER_TEST(selection_tie_break) {
    const int dim = 1;
    const int T = 4;
    const int b = 1;
    std::vector<float> q(T, 0.0f), k(T, 0.0f), v(T);
    for (int t = 0; t < T; ++t) v[t] = static_cast<float>(t);
    causal3::Params p{dim, b, /*select_count=*/2, /*window=*/1};
    causal3::SequenceResult sr;
    std::string err;
    CHECK(causal3::forward(p, q.data(), k.data(), v.data(), T, dim, &sr, &err));
    // All block scores identical: take blocks 0..take-1, ascending.
    CHECK(sr.steps[T - 1].selected_blocks ==
          (std::vector<int>{0, 1}));
    // select_count larger than visible blocks truncates to visible count.
    causal3::Params p2{dim, b, /*select_count=*/99, /*window=*/1};
    causal3::SequenceResult sr2;
    CHECK(causal3::forward(p2, q.data(), k.data(), v.data(), 2, dim, &sr2,
                           &err));
    CHECK(sr2.steps[1].selected_blocks == (std::vector<int>{0, 1}));
    // select_count == 0 yields no candidates and a zero selection branch.
    causal3::Params p3{dim, b, 0, 1};
    causal3::SequenceResult sr3;
    CHECK(causal3::forward(p3, q.data(), k.data(), v.data(), T, dim, &sr3,
                           &err));
    for (const causal3::StepResult& s : sr3.steps) {
        CHECK(s.selected_blocks.empty());
        CHECK(close_vec(s.selected, std::vector<float>(dim, 0.0f), 1e-7f));
        CHECK(s.gate_selected == 0.0f);
    }
}

// Compression must average the visible part of the incomplete trailing block
// instead of dividing by the full block length.
REGISTER_TEST(compression_partial_tail) {
    const int dim = 1;
    const int T = 5;
    const int b = 2;
    std::vector<float> q(T, 1.0f);  // all queries equal
    std::vector<float> k(T, 0.0f);
    std::vector<float> v{1.0f, 3.0f, 5.0f, 7.0f, 9.0f};
    causal3::Params p{dim, b, 0, 1};
    causal3::SequenceResult sr;
    std::string err;
    CHECK(causal3::forward(p, q.data(), k.data(), v.data(), T, dim, &sr,
                           &err));
    // t = 0: only partial block with a single visible row, mean v = 1.
    CHECK(close(sr.steps[0].compressed[0], 1.0f, 1e-6f));
    // t = 1: block 0 closed, mean v = 2.
    CHECK(close(sr.steps[1].compressed[0], 2.0f, 1e-6f));
    // t = 4: block means 2, 6 and partial trailing mean 9; with all logits
    // equal the branch averages them: (2 + 6 + 9) / 3.
    CHECK(sr.steps[4].comp_probs.size() == 3u);
    CHECK(close(sr.steps[4].compressed[0], 17.0f / 3.0f, 1e-5f));
}

// Window branch only uses the most recent min(w, t+1) positions; direct check.
REGISTER_TEST(window_branch_scope) {
    const int dim = 1;
    const int T = 6;
    const int w = 3;
    std::vector<float> q(T, 0.0f), k(T, 0.0f), v(T);
    for (int t = 0; t < T; ++t) v[t] = static_cast<float>(10 + t);
    causal3::Params p{dim, 4, 0, w};
    causal3::SequenceResult sr;
    std::string err;
    CHECK(causal3::forward(p, q.data(), k.data(), v.data(), T, dim, &sr,
                           &err));
    // Equal logits -> uniform attention over the window rows.
    CHECK(close(sr.steps[0].windowed[0], 10.0f, 1e-6f));
    CHECK(close(sr.steps[1].windowed[0], 10.5f, 1e-6f));
    CHECK(close(sr.steps[2].windowed[0], 11.0f, 1e-6f));
    CHECK(close(sr.steps[3].windowed[0], 12.0f, 1e-6f));
    CHECK(close(sr.steps[5].windowed[0], 14.0f, 1e-6f));  // v3,v4,v5
}

// Batch forward, streaming appends and the naive reference must agree on many
// random shapes; batch and streaming paths must match bit-for-bit.
REGISTER_TEST(batch_streaming_reference_agree) {
    Rng rng(20261001u);
    for (int trial = 0; trial < 40; ++trial) {
        const int dim = rng.integer(1, 5);
        const int b = rng.integer(1, 6);
        const int w = rng.integer(1, 8);
        const int nsel = rng.integer(0, 5);
        const int T = rng.integer(1, 18);
        causal3::Params p{dim, b, nsel, w};
        std::vector<float> q = make_data(T, dim, rng.integer(0, 1 << 30));
        std::vector<float> k = make_data(T, dim, rng.integer(0, 1 << 30));
        std::vector<float> v = make_data(T, dim, rng.integer(0, 1 << 30));

        causal3::SequenceResult batch;
        causal3::SequenceResult ref;
        std::string err;
        CHECK(causal3::forward(p, q.data(), k.data(), v.data(), T, dim, &batch,
                               &err));
        CHECK(causal3::reference_forward(p, q.data(), k.data(), v.data(), T,
                                         dim, &ref, &err));
        CHECK(batch.steps.size() == static_cast<std::size_t>(T));
        for (int t = 0; t < T; ++t)
            expect_step_vs_ref(batch.steps[t], ref.steps[t]);

        auto engine = causal3::Engine::create(p);
        for (int t = 0; t < T; ++t) {
            std::string aerr;
            CHECK(engine->append(&q[t * dim], &k[t * dim], &v[t * dim], dim,
                                 &aerr));
            const causal3::StepResult s = engine->last();
            const causal3::StepResult& bs = batch.steps[t];
            // Same code path, so results must be identical bit-for-bit.
            CHECK(s.compressed == bs.compressed);
            CHECK(s.selected == bs.selected);
            CHECK(s.windowed == bs.windowed);
            CHECK(s.output == bs.output);
            CHECK(s.selected_blocks == bs.selected_blocks);
            CHECK(s.comp_probs == bs.comp_probs);
            CHECK(s.gate_compressed == bs.gate_compressed);
            CHECK(s.gate_selected == bs.gate_selected);
            CHECK(s.gate_windowed == bs.gate_windowed);
        }
        CHECK(engine->length() == T);
    }
}

// Outputs computed before future tokens exist must not change when future
// keys/values are appended (causality), even if future rows are adversarial.
REGISTER_TEST(causality_future_cannot_change_past) {
    const int dim = 3;
    const int T1 = 6;
    const int T2 = 12;
    causal3::Params p{dim, /*block_len=*/2, /*select_count=*/2,
                      /*window=*/3};
    std::vector<float> q1 = make_data(T1, dim, 11u);
    std::vector<float> k1 = make_data(T1, dim, 22u);
    std::vector<float> v1 = make_data(T1, dim, 33u);

    causal3::SequenceResult early;
    std::string err;
    CHECK(causal3::forward(p, q1.data(), k1.data(), v1.data(), T1, dim, &early,
                           &err));

    std::vector<float> q2 = make_data(T2, dim, 44u);
    std::vector<float> k2 = make_data(T2, dim, 55u);
    std::vector<float> v2 = make_data(T2, dim, 66u);
    std::copy(q1.begin(), q1.end(), q2.begin());
    std::copy(k1.begin(), k1.end(), k2.begin());
    std::copy(v1.begin(), v1.end(), v2.begin());
    // Corrupt all future rows with huge values.
    for (int t = T1; t < T2; ++t)
        for (int d = 0; d < dim; ++d) {
            k2[t * dim + d] = (d == 0 ? 1e6f : -1e6f);
            v2[t * dim + d] = (d == 1 ? 1e6f : -1e6f);
        }
    causal3::SequenceResult later;
    CHECK(causal3::forward(p, q2.data(), k2.data(), v2.data(), T2, dim, &later,
                           &err));
    for (int t = 0; t < T1; ++t) {
        CHECK(later.steps[t].compressed == early.steps[t].compressed);
        CHECK(later.steps[t].selected == early.steps[t].selected);
        CHECK(later.steps[t].windowed == early.steps[t].windowed);
        CHECK(later.steps[t].output == early.steps[t].output);
        CHECK(later.steps[t].selected_blocks ==
              early.steps[t].selected_blocks);
    }
}

// Invalid parameters and inputs are rejected; failed appends leave state
// untouched.
REGISTER_TEST(validation_and_failed_append_state) {
    std::string err;
    CHECK(causal3::Engine::create({2, 0, 1, 2}, &err) == nullptr);
    CHECK(!err.empty());
    CHECK(causal3::Engine::create({2, 2, 1, 0}, &err) == nullptr);
    CHECK(causal3::Engine::create({0, 2, 1, 2}, &err) == nullptr);
    CHECK(causal3::Engine::create({2, 2, -1, 2}, &err) == nullptr);

    causal3::Params p{2, 2, 1, 2};
    auto engine = causal3::Engine::create(p);
    const float q[2] = {0.1f, 0.2f};
    const float k[2] = {0.3f, 0.4f};
    const float v[2] = {0.5f, 0.6f};
    CHECK(engine->append(q, k, v, 2, &err));
    const causal3::StepResult good = engine->last();
    const int len_before = engine->length();

    float bad[2] = {0.0f, quiet_nan()};
    std::string err1;
    CHECK(!engine->append(bad, k, v, 2, &err1));
    CHECK(!err1.empty());
    bad[1] = inf();
    CHECK(!engine->append(q, bad, v, 2, &err1));
    CHECK(!engine->append(q, k, bad, 2, &err1));
    // Shape mismatch and null pointers.
    CHECK(!engine->append(q, k, v, 3, &err1));
    CHECK(!engine->append(nullptr, k, v, 2, &err1));
    CHECK(!engine->append(q, nullptr, v, 2, &err1));
    CHECK(!engine->append(q, k, nullptr, 2, &err1));
    // Nothing changed.
    CHECK(engine->length() == len_before);
    const causal3::StepResult after = engine->last();
    CHECK(after.compressed == good.compressed);
    CHECK(after.selected == good.selected);
    CHECK(after.windowed == good.windowed);
    CHECK(after.output == good.output);
    CHECK(after.selected_blocks == good.selected_blocks);

    // A valid append after failures still works normally.
    CHECK(engine->append(q, k, v, 2, &err1));
    CHECK(engine->length() == len_before + 1);
}

// Batch forward rejects bad input and leaves the output untouched.
REGISTER_TEST(forward_validation) {
    const int dim = 2;
    const int T = 3;
    std::vector<float> q = make_data(T, dim, 1u);
    std::vector<float> k = make_data(T, dim, 2u);
    std::vector<float> v = make_data(T, dim, 3u);
    causal3::Params p{dim, 2, 1, 2};
    causal3::SequenceResult sentinel;
    sentinel.output.assign(7, 12345.0f);
    causal3::SequenceResult out = sentinel;
    std::string err;

    CHECK(!causal3::forward(p, q.data(), k.data(), v.data(), 0, dim, &out,
                            &err));
    CHECK(out.output == sentinel.output);  // untouched
    CHECK(!causal3::forward(p, q.data(), k.data(), v.data(), T, 4, &out,
                            &err));
    CHECK(out.output == sentinel.output);
    std::vector<float> badq = q;
    badq[2] = quiet_nan();
    CHECK(!causal3::forward(p, badq.data(), k.data(), v.data(), T, dim, &out,
                            &err));
    CHECK(out.output == sentinel.output);
    badq[2] = -inf();
    CHECK(!causal3::forward(p, badq.data(), k.data(), v.data(), T, dim, &out,
                            &err));
    CHECK(out.output == sentinel.output);
    CHECK(!causal3::forward(p, nullptr, k.data(), v.data(), T, dim, &out,
                            &err));
    CHECK(!causal3::forward(p, q.data(), k.data(), v.data(), T, dim, nullptr,
                            &err));

    // The same valid inputs still succeed and populate the result.
    CHECK(causal3::forward(p, q.data(), k.data(), v.data(), T, dim, &out,
                           &err));
    CHECK(out.steps.size() == static_cast<std::size_t>(T));
    CHECK(out.output.size() == static_cast<std::size_t>(T * dim));
    CHECK(out.compressed.size() == out.output.size());
    CHECK(out.windowed.size() == out.output.size());
    CHECK(out.selected.size() == out.output.size());
}

int main(int argc, char** argv) {
    if (argc == 2 && std::string(argv[1]) == "--list") {
        for (const auto& kv : registry()) std::cout << kv.first << "\n";
        return 0;
    }
    if (argc != 2) {
        std::cerr << "usage: " << argv[0] << " --list | <case-name>\n";
        return 2;
    }
    const std::string name = argv[1];
    auto it = registry().find(name);
    if (it == registry().end()) {
        std::cerr << "unknown test case: " << name << "\n";
        return 2;
    }
    it->second();
    std::cout << (g_failures == 0 ? "PASS" : "FAIL") << " " << name << " ("
              << g_checks << " checks)\n";
    return g_failures == 0 ? 0 : 1;
}
