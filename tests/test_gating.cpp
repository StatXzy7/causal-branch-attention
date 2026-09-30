// SPDX-License-Identifier: MIT
#include "test_framework.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

#include "tri_attn/naive.h"
#include "tri_attn/tri_attn.h"

using tri_attn::Config;
using tri_attn::Engine;
using tri_attn::RowResult;
using tri_attn::SequenceResult;
using tri_attn::naive::reference;
using tri_attn::run_full;

namespace {

void require_row_close(const RowResult& a, const RowResult& b) {
    REQUIRE_EQ(a.compression.size(), b.compression.size());
    REQUIRE_EQ(a.selection.size(), b.selection.size());
    REQUIRE_EQ(a.window.size(), b.window.size());
    REQUIRE_EQ(a.fused.size(), b.fused.size());
    REQUIRE_EQ(a.selected_blocks, b.selected_blocks);
    const auto cmp_vec = [](const std::vector<float>& x,
                            const std::vector<float>& y) {
        REQUIRE_EQ(x.size(), y.size());
        for (std::size_t i = 0; i < x.size(); ++i) REQUIRE_CLOSE(x[i], y[i]);
    };
    cmp_vec(a.compression, b.compression);
    cmp_vec(a.selection, b.selection);
    cmp_vec(a.window, b.window);
    cmp_vec(a.fused, b.fused);
    REQUIRE_CLOSE(a.gate_compression, b.gate_compression);
    REQUIRE_CLOSE(a.gate_selection, b.gate_selection);
    REQUIRE_CLOSE(a.gate_window, b.gate_window);
}

double ref_sigmoid(double x) {
    return x >= 0.0 ? 1.0 / (1.0 + std::exp(-x))
                    : std::exp(x) / (1.0 + std::exp(x));
}

}  // namespace

TEST_CASE("gates_are_three_independent_sigmoids_not_renormalised") {
    Config cfg;
    cfg.block_len = 1;
    cfg.select_k = 1;
    cfg.window_len = 1;
    cfg.gate_compression = -1.5f;
    cfg.gate_selection = 0.8f;
    cfg.gate_window = 2.2f;
    const float q[1] = {0.3f};
    const float k[1] = {0.1f};
    const float v[1] = {6.0f};
    auto res = run_full(cfg, q, k, v, 1, 1);
    const auto& r = res.rows[0];
    const double gc = ref_sigmoid(-1.5);
    const double gs = ref_sigmoid(0.8);
    const double gw = ref_sigmoid(2.2);
    REQUIRE_CLOSE(r.gate_compression, gc);
    REQUIRE_CLOSE(r.gate_selection, gs);
    REQUIRE_CLOSE(r.gate_window, gw);
    // Each branch sees the sole position with weight 1, so every branch
    // output equals v[0]; fused must be the plain sum of the three gates.
    REQUIRE_CLOSE(r.compression[0], 6.0f);
    REQUIRE_CLOSE(r.selection[0], 6.0f);
    REQUIRE_CLOSE(r.window[0], 6.0f);
    REQUIRE_CLOSE(r.fused[0], (gc + gs + gw) * 6.0);
    // Explicitly: gates are NOT re-normalised (their sum exceeds 1).
    REQUIRE(gc + gs + gw > 1.0);
}

TEST_CASE("fused_numeric_matches_branch_outputs_times_gates_multiple_rows") {
    Config cfg;
    cfg.block_len = 2;
    cfg.select_k = 1;
    cfg.window_len = 3;
    cfg.gate_compression = -0.4f;
    cfg.gate_selection = 1.3f;
    cfg.gate_window = 0.2f;
    const std::size_t T = 9, D = 3;
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    std::vector<float> q(T * D), k(T * D), v(T * D);
    for (float* p : {q.data(), k.data(), v.data()}) {
        for (std::size_t i = 0; i < T * D; ++i) p[i] = dist(rng);
    }
    auto res = run_full(cfg, q.data(), k.data(), v.data(), T, D);
    for (std::size_t t = 0; t < T; ++t) {
        const auto& r = res.rows[t];
        for (std::size_t j = 0; j < D; ++j) {
            const double expect = r.gate_compression * r.compression[j] +
                                  r.gate_selection * r.selection[j] +
                                  r.gate_window * r.window[j];
            REQUIRE_CLOSE(r.fused[j], expect);
        }
    }
}

TEST_CASE("batch_and_append_modes_match_bit_for_bit") {
    const std::size_t T = 12, D = 4;
    std::mt19937 rng(777);
    std::uniform_real_distribution<float> dist(-1.5f, 1.5f);
    std::vector<float> q(T * D), k(T * D), v(T * D);
    for (auto* p : {q.data(), k.data(), v.data()}) {
        for (std::size_t i = 0; i < T * D; ++i) p[i] = dist(rng);
    }
    const Config configs[] = {
        [] { Config c; c.block_len = 1; c.select_k = 1; c.window_len = 1;
             return c; }(),
        [] { Config c; c.block_len = 3; c.select_k = 2; c.window_len = 4;
             c.gate_compression = -1.0f; c.gate_selection = 0.5f;
             c.gate_window = 1.7f; return c; }(),
        [] { Config c; c.block_len = 5; c.select_k = 0; c.window_len = 2;
             return c; }(),
        [] { Config c; c.block_len = T; c.select_k = 9; c.window_len = 100;
             return c; }(),
    };
    for (const Config& cfg : configs) {
        SequenceResult batch = run_full(cfg, q.data(), k.data(), v.data(), T, D);
        Engine engine(cfg);
        for (std::size_t t = 0; t < T; ++t) {
            RowResult streamed =
                engine.append(q.data() + t * D, k.data() + t * D,
                              v.data() + t * D, D);
            const RowResult& want = batch.rows[t];
            // Shared kernel => identical bytes, not just close values.
            REQUIRE(::tri_attn_test::same_bits(streamed.compression, want.compression));
            REQUIRE(::tri_attn_test::same_bits(streamed.selection, want.selection));
            REQUIRE(::tri_attn_test::same_bits(streamed.window, want.window));
            REQUIRE(::tri_attn_test::same_bits(streamed.fused, want.fused));
            REQUIRE_EQ(streamed.selected_blocks, want.selected_blocks);
            REQUIRE_EQ(streamed.gate_compression, want.gate_compression);
            REQUIRE_EQ(streamed.gate_selection, want.gate_selection);
            REQUIRE_EQ(streamed.gate_window, want.gate_window);
        }
        REQUIRE_EQ(engine.size(), T);
    }
}

TEST_CASE("engine_matches_naive_reference_on_random_sequences") {
    std::mt19937 rng(424242);
    std::uniform_real_distribution<float> dist(-3.0f, 3.0f);
    for (int trial = 0; trial < 40; ++trial) {
        const std::size_t T = 1 + rng() % 20;
        const std::size_t D = 1 + rng() % 5;
        std::vector<float> q(T * D), k(T * D), v(T * D);
        for (auto* p : {q.data(), k.data(), v.data()}) {
            for (std::size_t i = 0; i < T * D; ++i) p[i] = dist(rng);
        }
        Config cfg;
        cfg.block_len = 1 + rng() % 6;
        cfg.select_k = static_cast<int>(rng() % 5);
        cfg.window_len = 1 + rng() % 8;
        cfg.gate_compression = dist(rng) * 2.0f;
        cfg.gate_selection = dist(rng) * 2.0f;
        cfg.gate_window = dist(rng) * 2.0f;

        SequenceResult got = run_full(cfg, q.data(), k.data(), v.data(), T, D);
        SequenceResult ref = reference(cfg, q.data(), k.data(), v.data(), T, D);

        Engine engine(cfg);
        REQUIRE_EQ(got.rows.size(), T);
        REQUIRE_EQ(ref.rows.size(), T);
        for (std::size_t t = 0; t < T; ++t) {
            require_row_close(got.rows[t], ref.rows[t]);
            RowResult streamed =
                engine.append(q.data() + t * D, k.data() + t * D,
                              v.data() + t * D, D);
            require_row_close(streamed, ref.rows[t]);
        }
    }
}

TEST_CASE("window_uses_only_most_recent_visible_positions") {
    Config cfg;
    cfg.block_len = 1;
    cfg.select_k = 0;
    cfg.window_len = 2;
    cfg.gate_window = 1.0f;
    cfg.gate_compression = -100.0f;
    const std::size_t T = 4, D = 1;
    std::vector<float> q(T * D, 0.0f);  // uniform attention weights
    std::vector<float> k(T * D, 0.0f);
    std::vector<float> v{10.0f, 20.0f, 30.0f, 40.0f};
    auto res = run_full(cfg, q.data(), k.data(), v.data(), T, D);
    REQUIRE_CLOSE(res.rows[0].window[0], 10.0f);
    REQUIRE_CLOSE(res.rows[1].window[0], 15.0f);   // (10+20)/2
    REQUIRE_CLOSE(res.rows[2].window[0], 25.0f);   // (20+30)/2
    REQUIRE_CLOSE(res.rows[3].window[0], 35.0f);   // (30+40)/2
}

TEST_CASE("empty_sequence_returns_empty_result_and_streaming_works") {
    Config cfg;
    cfg.block_len = 2;
    cfg.select_k = 1;
    cfg.window_len = 3;
    auto res = run_full(cfg, nullptr, nullptr, nullptr, 0, 0);
    REQUIRE(res.rows.empty());
    Engine engine(cfg);
    REQUIRE_EQ(engine.size(), 0u);
    const float q[2] = {0.1f, 0.2f};
    const float k[2] = {0.3f, 0.4f};
    const float v[2] = {0.5f, 0.6f};
    RowResult r = engine.append(q, k, v, 2);
    REQUIRE_EQ(engine.size(), 1u);
    REQUIRE_EQ(r.compression.size(), 2u);
    // After streaming the single row, engine history equals full history.
    SequenceResult full = run_full(cfg, q, k, v, 1, 2);
    require_row_close(r, full.rows[0]);
}
