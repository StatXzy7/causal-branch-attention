// SPDX-License-Identifier: MIT
/// @file demo.cpp
/// @brief Tiny float demo: global compression + important-block selection +
/// local window on a single short sequence, printed for manual inspection.
#include <iomanip>
#include <iostream>
#include <vector>

#include "tri_attn/tri_attn.h"

int main() {
    using tri_attn::Config;
    using tri_attn::run_full;

    const std::size_t T = 7;
    const std::size_t D = 2;
    // Small float query/key/value arrays.
    const std::vector<float> q = {
        0.2f, 0.1f,  -0.3f, 0.4f,  0.5f, -0.2f,
        0.1f, 0.6f,  -0.4f, 0.3f,  0.7f, -0.1f,
        0.0f, 0.2f,
    };
    const std::vector<float> k = {
        0.1f, 0.0f,   0.2f, -0.1f, -0.3f, 0.5f,
        0.4f, 0.2f,  -0.2f, 0.1f,  0.6f, -0.4f,
        0.1f, 0.3f,
    };
    const std::vector<float> v = {
        1.0f, 0.0f,   0.0f, 2.0f,   3.0f, 1.0f,
        1.0f, 4.0f,   2.0f, 0.0f,   0.0f, 5.0f,
        4.0f, 2.0f,
    };

    Config cfg;
    cfg.block_len = 2;       // mean over contiguous pairs; tail handled alone
    cfg.select_k = 2;        // pick the two most relevant blocks
    cfg.window_len = 3;      // most recent three visible positions
    cfg.gate_compression = 0.5f;
    cfg.gate_selection = 1.0f;
    cfg.gate_window = 0.0f;

    const auto result = run_full(cfg, q.data(), k.data(), v.data(), T, D);

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "block_len=" << cfg.block_len
              << " select_k=" << cfg.select_k
              << " window_len=" << cfg.window_len << "\n\n";
    for (std::size_t t = 0; t < T; ++t) {
        const auto& r = result.rows[t];
        std::cout << "t=" << t << "  selected blocks=[";
        for (std::size_t i = 0; i < r.selected_blocks.size(); ++i) {
            if (i) std::cout << ",";
            std::cout << r.selected_blocks[i];
        }
        std::cout << "]\n";
        std::cout << "  compression = [" << r.compression[0] << ", "
                  << r.compression[1] << "]  gate=" << r.gate_compression
                  << "\n";
        std::cout << "  selection   = [" << r.selection[0] << ", "
                  << r.selection[1] << "]  gate=" << r.gate_selection
                  << "\n";
        std::cout << "  window      = [" << r.window[0] << ", "
                  << r.window[1] << "]  gate=" << r.gate_window << "\n";
        std::cout << "  fused       = [" << r.fused[0] << ", "
                  << r.fused[1] << "]\n\n";
    }
    return 0;
}
