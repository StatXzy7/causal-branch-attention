// SPDX-License-Identifier: MIT
#include "tri_attn/tri_attn.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <stdexcept>

namespace tri_attn {
namespace {

constexpr float kPi = 3.14159265358979323846f;

bool finite_all(const float* data, std::size_t count) {
    if (data == nullptr) return false;
    for (std::size_t i = 0; i < count; ++i) {
        if (!std::isfinite(data[i])) return false;
    }
    return true;
}

void validate_config(const Config& config) {
    if (config.block_len <= 0) {
        throw std::invalid_argument("block_len must be positive");
    }
    if (config.select_k < 0) {
        throw std::invalid_argument("select_k must be non-negative");
    }
    if (config.window_len <= 0) {
        throw std::invalid_argument("window_len must be positive");
    }
    if (!std::isfinite(config.gate_compression) ||
        !std::isfinite(config.gate_selection) ||
        !std::isfinite(config.gate_window)) {
        throw std::invalid_argument("gate logits must be finite");
    }
}

double dot_product(const float* a, const float* b, std::size_t d) {
    double acc = 0.0;
    for (std::size_t i = 0; i < d; ++i) {
        acc += static_cast<double>(a[i]) * static_cast<double>(b[i]);
    }
    return acc;
}

// Numerically stable softmax in place. All inputs are finite.
void softmax_stable(std::vector<double>& logits) {
    if (logits.empty()) return;
    double max_logit = logits[0];
    for (double x : logits) max_logit = std::max(max_logit, x);
    double sum = 0.0;
    for (double& x : logits) {
        x = std::exp(x - max_logit);
        sum += x;
    }
    for (double& x : logits) x /= sum;
}

float sigmoid_stable(float x) {
    // expf(-|x|) avoids overflow for large |x|.
    const float z = std::exp(-std::fabs(x));
    return x >= 0.0f ? 1.0f / (1.0f + z) : z / (1.0f + z);
}

struct BlockSpan {
    int index;
    std::size_t begin;
    std::size_t length;
};

// Attention weighted sum: out[j] = sum_i weights[i] * v[pos[i]][j].
void weighted_sum(const std::vector<double>& weights,
                  const std::vector<std::size_t>& positions,
                  const float* v, std::size_t d,
                  std::vector<float>& out) {
    std::fill(out.begin(), out.end(), 0.0f);
    for (std::size_t i = 0; i < positions.size(); ++i) {
        const float w = static_cast<float>(weights[i]);
        const float* vrow = v + positions[i] * d;
        for (std::size_t j = 0; j < d; ++j) out[j] += w * vrow[j];
    }
}

// Core kernel shared by the batch path and the streaming path, so the two are
// bit-identical by construction. q/k/v are T x d row-major arrays containing
// the full prefix; only positions 0..t are ever read.
RowResult compute_row(const Config& config,
                      const float* q, const float* k, const float* v,
                      std::size_t t, std::size_t d) {
    const int block_len = config.block_len;
    const std::size_t visible = t + 1;
    const std::size_t num_full = visible / static_cast<std::size_t>(block_len);
    const std::size_t remainder = visible % static_cast<std::size_t>(block_len);
    const std::size_t num_blocks = num_full + (remainder != 0 ? 1 : 0);
    const double inv_sqrt_d = 1.0 / std::sqrt(static_cast<double>(d));
    const float* qrow = q + t * d;

    // ---- Block mean keys/values ------------------------------------------
    std::vector<BlockSpan> spans;
    spans.reserve(num_blocks);
    for (std::size_t b = 0; b < num_full; ++b) {
        spans.push_back({static_cast<int>(b), b * block_len,
                         static_cast<std::size_t>(block_len)});
    }
    if (remainder != 0) {
        spans.push_back({static_cast<int>(num_full), num_full * block_len,
                         remainder});
    }

    std::vector<std::vector<float>> mean_k(num_blocks,
                                           std::vector<float>(d, 0.0f));
    std::vector<std::vector<float>> mean_v(num_blocks,
                                           std::vector<float>(d, 0.0f));
    for (std::size_t b = 0; b < num_blocks; ++b) {
        const double inv_len = 1.0 / static_cast<double>(spans[b].length);
        for (std::size_t p = spans[b].begin;
             p < spans[b].begin + spans[b].length; ++p) {
            for (std::size_t j = 0; j < d; ++j) {
                mean_k[b][j] += k[p * d + j];
                mean_v[b][j] += v[p * d + j];
            }
        }
        for (std::size_t j = 0; j < d; ++j) {
            mean_k[b][j] = static_cast<float>(mean_k[b][j] * inv_len);
            mean_v[b][j] = static_cast<float>(mean_v[b][j] * inv_len);
        }
    }

    RowResult row;
    row.compression.assign(d, 0.0f);
    row.selection.assign(d, 0.0f);
    row.window.assign(d, 0.0f);
    row.fused.assign(d, 0.0f);

    // ---- Compression branch ----------------------------------------------
    std::vector<double> comp_logits(num_blocks);
    for (std::size_t b = 0; b < num_blocks; ++b) {
        comp_logits[b] = dot_product(qrow, mean_k[b].data(), d) * inv_sqrt_d;
    }
    softmax_stable(comp_logits);
    for (std::size_t b = 0; b < num_blocks; ++b) {
        const float w = static_cast<float>(comp_logits[b]);
        for (std::size_t j = 0; j < d; ++j) {
            row.compression[j] += w * mean_v[b][j];
        }
    }

    // ---- Selection branch ------------------------------------------------
    // Rank by descending compression score; ties keep ascending block index.
    std::vector<int> block_order(num_blocks);
    std::iota(block_order.begin(), block_order.end(), 0);
    std::sort(block_order.begin(), block_order.end(),
              [&](int a, int b) {
                  if (comp_logits[a] != comp_logits[b])
                      return comp_logits[a] > comp_logits[b];
                  return a < b;
              });
    const std::size_t k_select = std::min<std::size_t>(
        static_cast<std::size_t>(config.select_k), num_blocks);
    row.selected_blocks.reserve(k_select);
    for (std::size_t i = 0; i < k_select; ++i) {
        row.selected_blocks.push_back(spans[block_order[i]].index);
    }

    if (k_select > 0) {
        std::vector<std::size_t> selected_positions;
        for (std::size_t i = 0; i < k_select; ++i) {
            const BlockSpan& span = spans[block_order[i]];
            for (std::size_t p = span.begin; p < span.begin + span.length; ++p) {
                selected_positions.push_back(p);
            }
        }
        // Positions are gathered from blocks in score order; sort them so the
        // softmax semantics are position-order independent anyway (softmax is
        // permutation invariant), but keep deterministic storage order.
        std::sort(selected_positions.begin(), selected_positions.end());

        std::vector<double> sel_logits(selected_positions.size());
        for (std::size_t i = 0; i < selected_positions.size(); ++i) {
            const std::size_t p = selected_positions[i];
            sel_logits[i] = dot_product(qrow, k + p * d, d) * inv_sqrt_d;
        }
        softmax_stable(sel_logits);
        weighted_sum(sel_logits, selected_positions, v, d, row.selection);
    }

    // ---- Window branch ---------------------------------------------------
    std::size_t lo = 0;
    if (visible > static_cast<std::size_t>(config.window_len)) {
        lo = visible - static_cast<std::size_t>(config.window_len);
    }
    std::vector<std::size_t> window_positions;
    for (std::size_t p = lo; p < visible; ++p) window_positions.push_back(p);

    std::vector<double> win_logits(window_positions.size());
    for (std::size_t i = 0; i < window_positions.size(); ++i) {
        const std::size_t p = window_positions[i];
        win_logits[i] = dot_product(qrow, k + p * d, d) * inv_sqrt_d;
    }
    softmax_stable(win_logits);
    weighted_sum(win_logits, window_positions, v, d, row.window);

    // ---- Independent sigmoid gates, summed without renormalisation -------
    row.gate_compression = sigmoid_stable(config.gate_compression);
    row.gate_selection = sigmoid_stable(config.gate_selection);
    row.gate_window = sigmoid_stable(config.gate_window);
    for (std::size_t j = 0; j < d; ++j) {
        row.fused[j] = row.gate_compression * row.compression[j] +
                       row.gate_selection * row.selection[j] +
                       row.gate_window * row.window[j];
    }
    return row;
}

}  // namespace

SequenceResult run_full(const Config& config,
                        const float* q, const float* k, const float* v,
                        std::size_t seq_len, std::size_t d_model) {
    validate_config(config);
    if (seq_len > 0) {
        if (d_model == 0) {
            throw std::invalid_argument("d_model must be positive");
        }
        if (q == nullptr || k == nullptr || v == nullptr) {
            throw std::invalid_argument("q/k/v must not be null");
        }
        const std::size_t n = seq_len * d_model;
        if (!finite_all(q, n) || !finite_all(k, n) || !finite_all(v, n)) {
            throw std::invalid_argument("q/k/v must contain only finite values");
        }
    }

    SequenceResult result;
    result.rows.reserve(seq_len);
    for (std::size_t t = 0; t < seq_len; ++t) {
        result.rows.push_back(compute_row(config, q, k, v, t, d_model));
    }
    return result;
}

Engine::Engine(Config config) : config_(std::move(config)) {
    validate_config(config_);
}

RowResult Engine::append(const float* q_row, const float* k_row,
                         const float* v_row, std::size_t d_model) {
    if (d_model == 0) {
        throw std::invalid_argument("d_model must be positive");
    }
    if (q_row == nullptr || k_row == nullptr || v_row == nullptr) {
        throw std::invalid_argument("q_row/k_row/v_row must not be null");
    }
    if (!finite_all(q_row, d_model) || !finite_all(k_row, d_model) ||
        !finite_all(v_row, d_model)) {
        throw std::invalid_argument(
            "q_row/k_row/v_row must contain only finite values");
    }
    if (seq_len_ > 0 && d_model != d_model_) {
        throw std::invalid_argument(
            "d_model must match the width used by previous appends");
    }

    // Build the extended prefix in locals; commit only after the row has been
    // computed, so a throwing allocation leaves the engine state untouched.
    const std::size_t new_len = seq_len_ + 1;
    std::vector<float> qn;
    std::vector<float> kn;
    std::vector<float> vn;
    qn.reserve(new_len * d_model);
    kn.reserve(new_len * d_model);
    vn.reserve(new_len * d_model);
    qn = q_;
    kn = k_;
    vn = v_;
    qn.insert(qn.end(), q_row, q_row + d_model);
    kn.insert(kn.end(), k_row, k_row + d_model);
    vn.insert(vn.end(), v_row, v_row + d_model);

    RowResult row = compute_row(config_, qn.data(), kn.data(), vn.data(),
                                seq_len_, d_model);

    q_ = std::move(qn);
    k_ = std::move(kn);
    v_ = std::move(vn);
    results_.push_back(std::move(row));
    d_model_ = d_model;
    ++seq_len_;
    return results_.back();
}

}  // namespace tri_attn
