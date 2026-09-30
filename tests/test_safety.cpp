// SPDX-License-Identifier: MIT
#include "test_framework.h"

#include <cmath>
#include <limits>
#include <vector>

#include "tri_attn/tri_attn.h"

using tri_attn::Config;
using tri_attn::Engine;
using tri_attn::RowResult;
using tri_attn::run_full;

namespace {
constexpr float kInf = std::numeric_limits<float>::infinity();
constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
}  // namespace

TEST_CASE("rejects_non_positive_block_len_and_window_len") {
    const float q[1] = {1.0f}, k[1] = {1.0f}, v[1] = {1.0f};
    { Config cfg; cfg.block_len = 0;
      REQUIRE_THROWS(Engine engine(cfg), std::invalid_argument); }
    { Config cfg; cfg.block_len = -3;
      REQUIRE_THROWS(run_full(cfg, q, k, v, 1, 1), std::invalid_argument); }
    { Config cfg; cfg.window_len = 0;
      REQUIRE_THROWS(Engine engine(cfg), std::invalid_argument); }
    { Config cfg; cfg.window_len = -1;
      REQUIRE_THROWS(run_full(cfg, q, k, v, 1, 1), std::invalid_argument); }
}

TEST_CASE("rejects_negative_select_k_and_non_finite_gates") {
    const float q[1] = {1.0f}, k[1] = {1.0f}, v[1] = {1.0f};
    { Config cfg; cfg.select_k = -1;
      REQUIRE_THROWS(Engine engine(cfg), std::invalid_argument); }
    { Config cfg; cfg.gate_window = kInf;
      REQUIRE_THROWS(run_full(cfg, q, k, v, 1, 1), std::invalid_argument); }
    { Config cfg; cfg.gate_selection = kNaN;
      REQUIRE_THROWS(Engine engine(cfg), std::invalid_argument); }
}

TEST_CASE("rejects_non_finite_qkv_and_shape_conflicts") {
    Config cfg;
    cfg.block_len = 2;
    cfg.select_k = 1;
    cfg.window_len = 2;
    const std::size_t T = 3, D = 2;
    std::vector<float> q(T * D, 0.1f), k(T * D, 0.2f), v(T * D, 0.3f);

    auto bad_q = q;
    bad_q[4] = kNaN;
    REQUIRE_THROWS(run_full(cfg, bad_q.data(), k.data(), v.data(), T, D),
                   std::invalid_argument);

    auto bad_k = k;
    bad_k[0] = kInf;
    REQUIRE_THROWS(run_full(cfg, q.data(), bad_k.data(), v.data(), T, D),
                   std::invalid_argument);

    auto bad_v = v;
    bad_v[5] = -kInf;
    REQUIRE_THROWS(run_full(cfg, q.data(), k.data(), bad_v.data(), T, D),
                   std::invalid_argument);

    // Zero d_model with a non-empty sequence.
    REQUIRE_THROWS(run_full(cfg, q.data(), k.data(), v.data(), T, 0),
                   std::invalid_argument);
    // Null buffers with a non-empty sequence.
    REQUIRE_THROWS(run_full(cfg, nullptr, k.data(), v.data(), T, D),
                   std::invalid_argument);
}

TEST_CASE("append_rejects_bad_rows_without_touching_state") {
    Config cfg;
    cfg.block_len = 2;
    cfg.select_k = 1;
    cfg.window_len = 2;
    Engine engine(cfg);
    const std::size_t D = 3;
    std::vector<float> q(D, 0.5f), k(D, 0.2f), v(D, 1.0f);
    engine.append(q.data(), k.data(), v.data(), D);
    engine.append(q.data(), k.data(), v.data(), D);

    // Snapshot state after the two successful appends.
    const std::size_t saved_size = engine.size();
    std::vector<float> saved_q = engine.queries();
    std::vector<float> saved_k = engine.keys();
    std::vector<float> saved_v = engine.values();
    std::vector<RowResult> saved_results = engine.results();

    std::vector<float> nan_row(D, kNaN);
    std::vector<float> inf_row(D, kInf);
    std::vector<float> narrow(D - 1, 0.0f);

    REQUIRE_THROWS(engine.append(nan_row.data(), k.data(), v.data(), D),
                   std::invalid_argument);
    REQUIRE_THROWS(engine.append(q.data(), inf_row.data(), v.data(), D),
                   std::invalid_argument);
    REQUIRE_THROWS(engine.append(q.data(), k.data(), nan_row.data(), D),
                   std::invalid_argument);
    REQUIRE_THROWS(engine.append(nullptr, k.data(), v.data(), D),
                   std::invalid_argument);
    REQUIRE_THROWS(engine.append(q.data(), k.data(), v.data(), 0),
                   std::invalid_argument);
    REQUIRE_THROWS(engine.append(narrow.data(), narrow.data(), narrow.data(),
                                 D - 1),
                   std::invalid_argument);

    // State must be byte-for-byte unchanged after every rejected append.
    REQUIRE_EQ(engine.size(), saved_size);
    REQUIRE(engine.queries() == saved_q);
    REQUIRE(engine.keys() == saved_k);
    REQUIRE(engine.values() == saved_v);
    REQUIRE(engine.results().size() == saved_results.size());
    for (std::size_t i = 0; i < saved_results.size(); ++i) {
        REQUIRE(engine.results()[i].fused == saved_results[i].fused);
        REQUIRE(engine.results()[i].selected_blocks ==
                saved_results[i].selected_blocks);
    }

    // A valid append still works after the failures.
    RowResult next = engine.append(q.data(), k.data(), v.data(), D);
    REQUIRE_EQ(engine.size(), saved_size + 1);
    REQUIRE_EQ(next.fused.size(), D);
}

TEST_CASE("future_keys_values_cannot_change_previous_outputs") {
    Config cfg;
    cfg.block_len = 2;
    cfg.select_k = 2;
    cfg.window_len = 3;
    const std::size_t T = 8, D = 3;
    std::vector<float> q(T * D, 0.1f);
    std::vector<float> k(T * D, 0.2f);
    std::vector<float> v(T * D, 0.3f);

    // 1) Streaming: append first 5 positions, snapshot, then append mutated
    //    future positions; earlier emitted results stay identical.
    Engine engine(cfg);
    for (std::size_t t = 0; t < 5; ++t) {
        engine.append(q.data() + t * D, k.data() + t * D, v.data() + t * D, D);
    }
    std::vector<RowResult> frozen = engine.results();
    for (std::size_t t = 5; t < T; ++t) {
        std::vector<float> fq(D, 99.0f), fk(D, -99.0f), fv(D, 1e6f);
        engine.append(fq.data(), fk.data(), fv.data(), D);
    }
    REQUIRE_EQ(engine.size(), T);
    for (std::size_t t = 0; t < 5; ++t) {
        REQUIRE(engine.results()[t].fused == frozen[t].fused);
        REQUIRE(engine.results()[t].selection == frozen[t].selection);
        REQUIRE(engine.results()[t].selected_blocks ==
                frozen[t].selected_blocks);
    }

    // 2) Batch: recomputing after mutating future rows must leave prefix rows
    //    untouched (each row reads only positions <= t).
    auto baseline = run_full(cfg, q.data(), k.data(), v.data(), T, D);
    auto mutated_q = q, mutated_k = k, mutated_v = v;
    for (std::size_t i = 4 * D; i < T * D; ++i) {
        mutated_q[i] = -77.0f;
        mutated_k[i] = 77.0f;
        mutated_v[i] = -1e5f;
    }
    auto after = run_full(cfg, mutated_q.data(), mutated_k.data(),
                          mutated_v.data(), T, D);
    for (std::size_t t = 0; t < 4; ++t) {
        REQUIRE(after.rows[t].fused == baseline.rows[t].fused);
        REQUIRE(after.rows[t].compression == baseline.rows[t].compression);
        REQUIRE(after.rows[t].window == baseline.rows[t].window);
        REQUIRE(after.rows[t].selected_blocks ==
                baseline.rows[t].selected_blocks);
    }
}

TEST_CASE("all_outputs_finite_under_extreme_but_finite_inputs") {
    Config cfg;
    cfg.block_len = 3;
    cfg.select_k = 2;
    cfg.window_len = 4;
    cfg.gate_compression = 60.0f;
    cfg.gate_selection = -60.0f;
    cfg.gate_window = 30.0f;
    const std::size_t T = 7, D = 2;
    std::vector<float> q(T * D), k(T * D), v(T * D);
    for (std::size_t i = 0; i < T * D; ++i) {
        q[i] = (i % 2 ? 1.0f : -1.0f) * 40.0f;
        k[i] = (i % 3 ? 1.0f : -1.0f) * 40.0f;
        v[i] = (i % 2 ? -1.0f : 1.0f) * 50.0f;
    }
    auto res = run_full(cfg, q.data(), k.data(), v.data(), T, D);
    for (const auto& r : res.rows) {
        for (const auto* vec : {&r.compression, &r.selection, &r.window,
                                &r.fused}) {
            for (float x : *vec) REQUIRE(std::isfinite(x));
        }
    }
}
