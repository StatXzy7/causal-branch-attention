// SPDX-License-Identifier: MIT
/// @file tri_attn.h
/// @brief Three-branch causal attention engine (compression / selection / window).
///
/// For every query position t only positions 0..t are visible (strict causal
/// mask, current position included). Three branches run in parallel:
///
///   * Compression: keys/values are averaged over contiguous blocks of length
///     block_len. A trailing run shorter than block_len is treated as one extra
///     partial block whose mean is taken over its (fewer) members. Attention is
///     computed over those block mean keys/values.
///   * Selection: scores of the compression keys are used to rank the blocks.
///     The select_k highest scoring blocks (ties broken by ascending original
///     block index) contribute, and attention is recomputed over the *raw*
///     keys/values contained in the selected blocks.
///   * Window: attention over the at most window_len most recent visible
///     positions.
///
/// Each branch applies its own numerically stable scaled-dot-product softmax.
/// The fused output is
///
///   out = sigmoid(g_c) * out_c + sigmoid(g_s) * out_s + sigmoid(g_w) * out_w
///
/// i.e. the three sigmoid gates are summed without renormalisation. A branch
/// with no visible candidates contributes the zero vector.
#ifndef TRI_ATTN_TRI_ATTN_H
#define TRI_ATTN_TRI_ATTN_H

#include <cstddef>
#include <vector>

namespace tri_attn {

/// Engine configuration.
struct Config {
    int block_len = 1;    ///< Compression block length, must be > 0.
    int select_k = 1;     ///< Number of blocks selected, 0 disables selection.
    int window_len = 1;   ///< Local window width, must be > 0.
    float gate_compression = 0.0f; ///< Sigmoid input for the compression gate.
    float gate_selection = 0.0f;   ///< Sigmoid input for the selection gate.
    float gate_window = 0.0f;      ///< Sigmoid input for the window gate.
};

/// Per-position result.
struct RowResult {
    std::vector<float> compression;        ///< [d_model] compression output.
    std::vector<float> selection;          ///< [d_model] selection output.
    std::vector<float> window;             ///< [d_model] window output.
    std::vector<float> fused;              ///< [d_model] gated sum.
    std::vector<int> selected_blocks;      ///< Selected original block indices,
                                           ///  ordered by descending score then
                                           ///  ascending block index.
    float gate_compression = 0.0f;         ///< sigmoid of config gate.
    float gate_selection = 0.0f;
    float gate_window = 0.0f;
};

/// Result for a whole sequence, one RowResult per query position.
struct SequenceResult {
    std::vector<RowResult> rows;
};

/// Process a complete sequence at once.
///
/// @param q T x d_model query matrix, row major.
/// @param k T x d_model key matrix, row major.
/// @param v T x d_model value matrix, row major.
/// @param seq_len T, number of positions, may be 0.
/// @param d_model feature width, must be > 0 when T > 0.
/// @throws std::invalid_argument on non-finite entries, shape conflicts,
///         non-positive block_len / window_len, negative select_k,
///         non-finite gate logits or bad d_model.
SequenceResult run_full(const Config& config,
                        const float* q, const float* k, const float* v,
                        std::size_t seq_len, std::size_t d_model);

/// Streaming engine. Append positions one at a time; each appended row only
/// ever observes current and earlier data, so mutating future keys/values can
/// never change previously emitted outputs.
class Engine {
public:
    /// @throws std::invalid_argument for invalid configurations (see Config).
    explicit Engine(Config config);

    /// Append one position and return its causal attention result.
    ///
    /// On invalid input the call throws std::invalid_argument and the engine
    /// state is left untouched (strong exception guarantee).
    ///
    /// @param q_row,k_row,v_row pointers to d_model finite floats each.
    /// @throws std::invalid_argument on non-finite entries or null pointers.
    RowResult append(const float* q_row, const float* k_row, const float* v_row,
                     std::size_t d_model);

    std::size_t size() const noexcept { return seq_len_; }
    const std::vector<float>& queries() const noexcept { return q_; }
    const std::vector<float>& keys() const noexcept { return k_; }
    const std::vector<float>& values() const noexcept { return v_; }
    const std::vector<RowResult>& results() const noexcept { return results_; }
    const Config& config() const noexcept { return config_; }

private:
    Config config_;
    std::size_t d_model_ = 0;
    std::size_t seq_len_ = 0;
    std::vector<float> q_;
    std::vector<float> k_;
    std::vector<float> v_;
    std::vector<RowResult> results_;
};

}  // namespace tri_attn

#endif  // TRI_ATTN_TRI_ATTN_H
