// Three-branch causal attention engine (compression + selection + local window).
//
// Pure C++17, Linux CPU, no third-party dependencies. Row-major flat arrays:
// element at row r, column c lives at data[r * dim + c].
#ifndef CAUSAL3_ENGINE_H
#define CAUSAL3_ENGINE_H

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace causal3 {

struct Params {
    int dim = 0;           // feature width of q/k/v
    int block_len = 0;     // b: compression block length (must be > 0)
    int select_count = 0;  // n: number of selected blocks (>= 0; truncated)
    int window = 0;        // w: local window width (must be > 0)
};

// Output for one query position.
struct StepResult {
    std::vector<float> compressed;        // compression branch output, dim floats
    std::vector<float> selected;          // selection branch output, dim floats
    std::vector<float> windowed;          // local window branch output, dim floats
    std::vector<float> output;            // final fused output, dim floats
    std::vector<int> selected_blocks;     // chosen block indices, ascending
    std::vector<float> comp_probs;        // normalized block relevance, size = visible blocks
    float gate_compressed = 0.0f;
    float gate_selected = 0.0f;
    float gate_windowed = 0.0f;
};

// Outputs for a full sequence; fields at index t correspond to query position t.
struct SequenceResult {
    std::vector<StepResult> steps;        // size = T
    std::vector<float> compressed;        // T * dim
    std::vector<float> selected;          // T * dim
    std::vector<float> windowed;          // T * dim
    std::vector<float> output;            // T * dim
};

// Streaming engine: append one (q, k, v) triple at a time, read the result for
// that position immediately. A failed append never mutates stored state.
class Engine {
public:
    virtual ~Engine();

    // Append the (q, k, v) triple at the next position and run attention for
    // that query over positions [0, t]. Returns false and fills err on
    // validation failure; on failure stored state is unchanged.
    virtual bool append(const float* q, const float* k, const float* v,
                        int dim, std::string* err = nullptr) = 0;

    // Take the output for the most recently appended position.
    virtual StepResult last() const = 0;

    virtual const Params& params() const = 0;
    virtual int length() const = 0;
    virtual int dim() const = 0;

    // Create an engine with validated parameters.
    static std::unique_ptr<Engine> create(const Params& p,
                                          std::string* err = nullptr);
};

// Whole-sequence computation. Arrays are T*dim row-major. A failed call leaves
// *out completely untouched.
bool forward(const Params& p, const float* q, const float* k, const float* v,
             int T, int dim, SequenceResult* out, std::string* err = nullptr);

}  // namespace causal3

#endif  // CAUSAL3_ENGINE_H
