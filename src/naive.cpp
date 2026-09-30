// SPDX-License-Identifier: MIT
#include "tri_attn/naive.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace tri_attn {
namespace naive {
namespace {

double ref_dot(const std::vector<float>& a, const float* b, std::size_t d) {
    double s = 0.0;
    for (std::size_t i = 0; i < d; ++i) {
        s += static_cast<double>(a[i]) * static_cast<double>(b[i]);
    }
    return s;
}

std::vector<double> ref_softmax(const std::vector<double>& z) {
    std::vector<double> p(z.size(), 0.0);
    if (z.empty()) return p;
    double m = z[0];
    for (double x : z) m = std::max(m, x);
    double denom = 0.0;
    for (std::size_t i = 0; i < z.size(); ++i) {
        p[i] = std::exp(z[i] - m);
        denom += p[i];
    }
    for (double& x : p) x /= denom;
    return p;
}

double ref_sigmoid(float x) {
    if (x >= 0.0f) return 1.0 / (1.0 + std::exp(-static_cast<double>(x)));
    const double e = std::exp(static_cast<double>(x));
    return e / (1.0 + e);
}

}  // namespace

SequenceResult reference(const Config& config,
                         const float* q, const float* k, const float* v,
                         std::size_t seq_len, std::size_t d_model) {
    SequenceResult out;
    const int B = config.block_len;
    const int W = config.window_len;
    const double scale = 1.0 / std::sqrt(static_cast<double>(d_model));

    for (std::size_t t = 0; t < seq_len; ++t) {
        const std::size_t visible = t + 1;
        const std::size_t nfull = visible / static_cast<std::size_t>(B);
        const std::size_t rem = visible % static_cast<std::size_t>(B);
        const std::size_t nblock = nfull + (rem > 0 ? 1 : 0);

        // Block means.
        std::vector<std::vector<float>> bk(nblock,
                                           std::vector<float>(d_model, 0.0f));
        std::vector<std::vector<float>> bv(nblock,
                                           std::vector<float>(d_model, 0.0f));
        for (std::size_t b = 0; b < nblock; ++b) {
            const std::size_t start = b * static_cast<std::size_t>(B);
            const std::size_t stop =
                std::min(visible, start + static_cast<std::size_t>(B));
            for (std::size_t p = start; p < stop; ++p) {
                for (std::size_t j = 0; j < d_model; ++j) {
                    bk[b][j] += k[p * d_model + j];
                    bv[b][j] += v[p * d_model + j];
                }
            }
            const double inv = 1.0 / static_cast<double>(stop - start);
            for (std::size_t j = 0; j < d_model; ++j) {
                bk[b][j] = static_cast<float>(bk[b][j] * inv);
                bv[b][j] = static_cast<float>(bv[b][j] * inv);
            }
        }

        RowResult row;
        row.compression.assign(d_model, 0.0f);
        row.selection.assign(d_model, 0.0f);
        row.window.assign(d_model, 0.0f);
        row.fused.assign(d_model, 0.0f);

        const float* qt = q + t * d_model;

        // Compression scores (normalised relevance) and weighted mean values.
        std::vector<double> scores(nblock, 0.0);
        for (std::size_t b = 0; b < nblock; ++b) {
            scores[b] = ref_dot(bk[b], qt, d_model) * scale;
        }
        const std::vector<double> cw = ref_softmax(scores);
        for (std::size_t b = 0; b < nblock; ++b) {
            for (std::size_t j = 0; j < d_model; ++j) {
                row.compression[j] +=
                    static_cast<float>(cw[b]) * bv[b][j];
            }
        }

        // Ranking: descending score, ascending block index on ties.
        std::vector<int> order(nblock);
        for (std::size_t b = 0; b < nblock; ++b) order[b] = static_cast<int>(b);
        std::sort(order.begin(), order.end(), [&](int x, int y) {
            if (scores[x] != scores[y]) return scores[x] > scores[y];
            return x < y;
        });
        std::size_t take = static_cast<std::size_t>(config.select_k);
        if (take > nblock) take = nblock;
        for (std::size_t i = 0; i < take; ++i) {
            row.selected_blocks.push_back(order[i]);
        }

        // Attention over raw keys/values of selected blocks.
        if (take > 0) {
            std::vector<std::size_t> pos;
            for (std::size_t i = 0; i < take; ++i) {
                const std::size_t start =
                    static_cast<std::size_t>(order[i]) *
                    static_cast<std::size_t>(B);
                const std::size_t stop =
                    std::min(visible, start + static_cast<std::size_t>(B));
                for (std::size_t p = start; p < stop; ++p) pos.push_back(p);
            }
            std::sort(pos.begin(), pos.end());
            std::vector<double> z(pos.size(), 0.0);
            for (std::size_t i = 0; i < pos.size(); ++i) {
                double s = 0.0;
                for (std::size_t j = 0; j < d_model; ++j) {
                    s += static_cast<double>(qt[j]) *
                         static_cast<double>(k[pos[i] * d_model + j]);
                }
                z[i] = s * scale;
            }
            const std::vector<double> sw = ref_softmax(z);
            for (std::size_t i = 0; i < pos.size(); ++i) {
                for (std::size_t j = 0; j < d_model; ++j) {
                    row.selection[j] += static_cast<float>(sw[i]) *
                                        v[pos[i] * d_model + j];
                }
            }
        }

        // Local window over the most recent visible positions.
        std::size_t lo = 0;
        if (visible > static_cast<std::size_t>(W)) {
            lo = visible - static_cast<std::size_t>(W);
        }
        std::vector<double> z;
        std::vector<std::size_t> wp;
        for (std::size_t p = lo; p < visible; ++p) {
            double s = 0.0;
            for (std::size_t j = 0; j < d_model; ++j) {
                s += static_cast<double>(qt[j]) *
                     static_cast<double>(k[p * d_model + j]);
            }
            z.push_back(s * scale);
            wp.push_back(p);
        }
        const std::vector<double> ww = ref_softmax(z);
        for (std::size_t i = 0; i < wp.size(); ++i) {
            for (std::size_t j = 0; j < d_model; ++j) {
                row.window[j] += static_cast<float>(ww[i]) *
                                 v[wp[i] * d_model + j];
            }
        }

        row.gate_compression = static_cast<float>(ref_sigmoid(
            config.gate_compression));
        row.gate_selection = static_cast<float>(ref_sigmoid(
            config.gate_selection));
        row.gate_window = static_cast<float>(ref_sigmoid(config.gate_window));
        for (std::size_t j = 0; j < d_model; ++j) {
            row.fused[j] = row.gate_compression * row.compression[j] +
                           row.gate_selection * row.selection[j] +
                           row.gate_window * row.window[j];
        }
        out.rows.push_back(std::move(row));
    }
    return out;
}

}  // namespace naive
}  // namespace tri_attn
