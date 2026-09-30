// SPDX-License-Identifier: MIT
/// @file naive.h
/// @brief Straightforward reference implementation of the three-branch causal
/// attention. It is written independently of tri_attn.cpp (plain nested loops,
/// its own softmax/sigmoid) and is intended for cross-checking the engine.
#ifndef TRI_ATTN_NAIVE_H
#define TRI_ATTN_NAIVE_H

#include <cstddef>
#include "tri_attn/tri_attn.h"

namespace tri_attn {
namespace naive {

/// Reference batch computation. Assumes valid, finite inputs; the production
/// engine is responsible for validation.
SequenceResult reference(const Config& config,
                         const float* q, const float* k, const float* v,
                         std::size_t seq_len, std::size_t d_model);

}  // namespace naive
}  // namespace tri_attn

#endif  // TRI_ATTN_NAIVE_H
