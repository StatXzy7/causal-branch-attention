// Naive reference implementation of the three-branch causal attention.
// Deliberately simple and independent from engine.cpp; it performs the same
// math with explicit per-position, per-block loops. It also performs the same
// input validation and reports the same failure conditions.
#ifndef CAUSAL3_REFERENCE_H
#define CAUSAL3_REFERENCE_H

#include <string>

#include "causal3/engine.h"

namespace causal3 {

bool reference_forward(const Params& p, const float* q, const float* k,
                       const float* v, int T, int dim, SequenceResult* out,
                       std::string* err = nullptr);

}  // namespace causal3

#endif  // CAUSAL3_REFERENCE_H
