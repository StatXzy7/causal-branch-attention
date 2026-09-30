// SPDX-License-Identifier: MIT
#include "test_framework.h"

#include <cmath>
#include <vector>

#include "tri_attn/tri_attn.h"

using tri_attn::Config;
using tri_attn::Engine;
using tri_attn::run_full;

TEST_CASE("single_row_single_position_all_branches_identical") {
    Config cfg;
    cfg.block_len = 1;
    cfg.select_k = 1;
    cfg.window_len = 1;
    cfg.gate_compression = 0.0f;
    cfg.gate_selection = 0.0f;
    cfg.gate_window = 0.0f;
    const float q[2] = {0.7f, -1.2f};
    const float k[2] = {0.3f, 0.9f};
    const float v[2] = {4.0f, -5.0f};
    auto res = run_full(cfg, q, k, v, 1, 2);
    REQUIRE_EQ(res.rows.size(), 1u);
    const auto& r = res.rows[0];
    for (std::size_t j = 0; j < 2; ++j) {
        REQUIRE_CLOSE(r.compression[j], v[j]);
        REQUIRE_CLOSE(r.selection[j], v[j]);
        REQUIRE_CLOSE(r.window[j], v[j]);
        REQUIRE_CLOSE(r.fused[j], 1.5f * v[j]);  // 0.5 + 0.5 + 0.5
    }
    REQUIRE_EQ(r.selected_blocks.size(), 1u);
    REQUIRE_EQ(r.selected_blocks[0], 0);
    REQUIRE_CLOSE(r.gate_compression, 0.5f);
    REQUIRE_CLOSE(r.gate_selection, 0.5f);
    REQUIRE_CLOSE(r.gate_window, 0.5f);
}

TEST_CASE("selection_disabled_select_k_zero_contributes_zero") {
    Config cfg;
    cfg.block_len = 2;
    cfg.select_k = 0;
    cfg.window_len = 1;
    cfg.gate_selection = 2.0f;  // even a large gate multiplies zero
    std::vector<float> q{1.0f, 0.0f, 0.0f, 1.0f};
    std::vector<float> k{1.0f, 0.0f, 0.0f, 1.0f};
    std::vector<float> v{7.0f, 8.0f, 9.0f, 10.0f};
    auto res = run_full(cfg, q.data(), k.data(), v.data(), 2, 2);
    const auto& r = res.rows[1];
    REQUIRE(r.selected_blocks.empty());
    for (float x : r.selection) REQUIRE_CLOSE(x, 0.0f);
    // Fused output still contains compression + window contributions.
    for (std::size_t j = 0; j < 2; ++j) {
        const double expect = r.gate_compression * r.compression[j] +
                              r.gate_window * r.window[j];
        REQUIRE_CLOSE(r.fused[j], expect);
    }
}

TEST_CASE("selection_tie_breaks_by_ascending_block_index") {
    // q = 0 makes every dot product 0 -> all compression scores tie.
    Config cfg;
    cfg.block_len = 2;
    cfg.select_k = 2;
    cfg.window_len = 2;
    // 6 positions => full blocks [0,1],[2,3],[4,5].
    const std::size_t T = 6, D = 2;
    std::vector<float> q(T * D, 0.0f);
    std::vector<float> k(T * D, 1.0f);
    std::vector<float> v(T * D, 1.0f);
    auto res = run_full(cfg, q.data(), k.data(), v.data(), T, D);
    const auto& blocks = res.rows[T - 1].selected_blocks;
    REQUIRE_EQ(blocks.size(), 2u);
    REQUIRE_EQ(blocks[0], 0);
    REQUIRE_EQ(blocks[1], 1);

    // Partial tail case: 5 positions => blocks 0,1 full, block 2 partial.
    auto res2 = run_full(cfg, q.data(), k.data(), v.data(), 5, D);
    const auto& blocks2 = res2.rows[4].selected_blocks;
    REQUIRE_EQ(blocks2.size(), 2u);
    REQUIRE_EQ(blocks2[0], 0);
    REQUIRE_EQ(blocks2[1], 1);
}

TEST_CASE("selection_picks_highest_scoring_blocks_descending") {
    // Blocks engineered so their mean keys dot the final query with distinct
    // magnitudes: block 1 highest, block 0 middle, block 2 lowest.
    Config cfg;
    cfg.block_len = 2;
    cfg.select_k = 2;
    cfg.window_len = 2;
    const std::size_t T = 6, D = 1;
    std::vector<float> q(T * D, 0.0f);
    q[T - 1] = 1.0f;
    // mean keys: block0 -> -2, block1 -> 3, block2 -> 1.
    std::vector<float> k{-2.0f, -2.0f, 3.0f, 3.0f, 1.0f, 1.0f};
    std::vector<float> v(T * D, 0.0f);
    auto res = run_full(cfg, q.data(), k.data(), v.data(), T, D);
    const auto& blocks = res.rows[T - 1].selected_blocks;
    REQUIRE_EQ(blocks.size(), 2u);
    REQUIRE_EQ(blocks[0], 1);
    REQUIRE_EQ(blocks[1], 2);
}

TEST_CASE("selection_truncated_when_select_k_exceeds_visible_blocks") {
    Config cfg;
    cfg.block_len = 4;
    cfg.select_k = 100;  // far more than the 2 visible blocks at t=6
    cfg.window_len = 2;
    const std::size_t T = 7, D = 2;
    std::vector<float> q(T * D, 0.0f);
    std::vector<float> k(T * D, 0.5f);
    std::vector<float> v(T * D, 1.0f);
    auto res = run_full(cfg, q.data(), k.data(), v.data(), T, D);
    // t=6: positions 0..6 -> full block [0..3], partial [4..6].
    const auto& blocks = res.rows[6].selected_blocks;
    REQUIRE_EQ(blocks.size(), 2u);
    REQUIRE_EQ(blocks[0], 0);
    REQUIRE_EQ(blocks[1], 1);

    // t=0: a single partial block [0], truncated to one.
    REQUIRE_EQ(res.rows[0].selected_blocks.size(), 1u);
    REQUIRE_EQ(res.rows[0].selected_blocks[0], 0);
}

TEST_CASE("selection_partial_trailing_block_raw_values_attended") {
    // select_k = 1 and trailing partial block forced as the winner; the
    // selection output must read its RAW (not averaged) member values.
    Config cfg;
    cfg.block_len = 4;
    cfg.select_k = 1;
    cfg.window_len = 1;
    const std::size_t T = 6, D = 1;
    std::vector<float> q(T * D, 0.0f);
    q[T - 1] = 1.0f;
    std::vector<float> k(T * D, 0.0f);
    // Full block 0 mean key = 0; partial block 1 (positions 4,5) mean = 5.
    k[4] = 4.0f;
    k[5] = 6.0f;
    std::vector<float> v(T * D, 0.0f);
    v[4] = 3.0f;
    v[5] = 5.0f;
    auto res = run_full(cfg, q.data(), k.data(), v.data(), T, D);
    const auto& r = res.rows[T - 1];
    REQUIRE_EQ(r.selected_blocks.size(), 1u);
    REQUIRE_EQ(r.selected_blocks[0], 1);
    // Raw softmax over positions 4,5 with keys 4 and 6 (scale = 1).
    const double w4 = std::exp(4.0) / (std::exp(4.0) + std::exp(6.0));
    const double w5 = std::exp(6.0) / (std::exp(4.0) + std::exp(6.0));
    REQUIRE_CLOSE(r.selection[0], w4 * 3.0f + w5 * 5.0f);
}

TEST_CASE("compression_partial_block_mean_uses_only_its_members") {
    // select_k = 0 isolates the compression branch; at t=1 with block_len=4
    // the single partial block mean equals the arithmetic mean of positions.
    Config cfg;
    cfg.block_len = 4;
    cfg.select_k = 0;
    cfg.window_len = 1;
    cfg.gate_compression = 1.0f;
    cfg.gate_window = -100.0f;  // effectively closes the window gate
    const std::size_t D = 1;
    std::vector<float> q{1.0f, 1.0f};
    std::vector<float> k{0.0f, 0.0f};
    std::vector<float> v{2.0f, 4.0f};
    auto res = run_full(cfg, q.data(), k.data(), v.data(), 2, D);
    REQUIRE_CLOSE(res.rows[1].compression[0], 3.0f);
    REQUIRE_CLOSE(res.rows[1].fused[0],
                  res.rows[1].gate_compression * 3.0f);
}
