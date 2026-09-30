#include "causal3/engine.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>

namespace causal3 {
namespace {

constexpr float kNegInf = -std::numeric_limits<float>::infinity();

void set_err(std::string* err, std::string msg) {
    if (err) *err = std::move(msg);
}

bool valid_params(const Params& p, std::string* err) {
    if (p.dim <= 0) {
        set_err(err, "dim must be positive");
        return false;
    }
    if (p.block_len <= 0) {
        set_err(err, "block_len must be positive");
        return false;
    }
    if (p.window <= 0) {
        set_err(err, "window must be positive");
        return false;
    }
    if (p.select_count < 0) {
        set_err(err, "select_count must be non-negative");
        return false;
    }
    return true;
}

bool all_finite(const float* data, std::size_t count) {
    for (std::size_t i = 0; i < count; ++i) {
        if (!std::isfinite(data[i])) return false;
    }
    return true;
}

float dot(const float* a, const float* b, int dim) {
    float acc = 0.0f;
    for (int d = 0; d < dim; ++d) acc += a[d] * b[d];
    return acc;
}

// Numerically stable sigmoid.
float sigmoid(float x) {
    if (x >= 0.0f) {
        const float e = std::exp(-x);
        return 1.0f / (1.0f + e);
    }
    const float e = std::exp(x);
    return e / (1.0f + e);
}

// Numerically stable softmax over vals[0..n). n must be at least 1 and every
// value must be finite; writes probabilities into probs.
void softmax(const float* vals, int n, float* probs) {
    float m = vals[0];
    for (int i = 1; i < n; ++i) m = std::max(m, vals[i]);
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        probs[i] = std::exp(vals[i] - m);
        sum += probs[i];
    }
    const float inv = 1.0f / sum;
    for (int i = 0; i < n; ++i) probs[i] *= inv;
}

struct EngineImpl final : public Engine {
    EngineImpl(Params p, int dim)
        : p_(p),
          open_ksum_(dim, 0.0f),
          open_vsum_(dim, 0.0f) {}

    const Params& params() const override { return p_; }
    int length() const override { return t_ + 1; }
    int dim() const override { return p_.dim; }

    bool append(const float* q, const float* k, const float* v, int dim,
                std::string* err) override {
        if (q == nullptr || k == nullptr || v == nullptr) {
            set_err(err, "q, k and v must not be null");
            return false;
        }
        if (dim != p_.dim) {
            set_err(err, "dim does not match engine configuration");
            return false;
        }
        if (!all_finite(q, dim) || !all_finite(k, dim) ||
            !all_finite(v, dim)) {
            set_err(err, "q, k and v must contain only finite values");
            return false;
        }
        // All validation passed: from here on no failure can occur, so mutating
        // state is safe.
        ++t_;
        q_.insert(q_.end(), q, q + dim);
        k_.insert(k_.end(), k, k + dim);
        v_.insert(v_.end(), v, v + dim);
        for (int d = 0; d < dim; ++d) {
            open_ksum_[d] += k[d];
            open_vsum_[d] += v[d];
        }
        if (t_ % p_.block_len == p_.block_len - 1) {
            closed_ksum_.insert(closed_ksum_.end(), open_ksum_.begin(),
                                open_ksum_.end());
            closed_vsum_.insert(closed_vsum_.end(), open_vsum_.begin(),
                                open_vsum_.end());
            std::fill(open_ksum_.begin(), open_ksum_.end(), 0.0f);
            std::fill(open_vsum_.begin(), open_vsum_.end(), 0.0f);
        }
        last_ = compute();
        return true;
    }

    StepResult last() const override { return last_; }

private:
    // Scaled dot-product attention over an explicit list of key/value rows,
    // with the query already on hand. Returns branch output in `out` and the
    // (possibly kNegInf) maximum scaled logit.
    float raw_attention(const float* query, const std::vector<int>& rows,
                        std::vector<float>* out, std::vector<float>* probs) {
        const int n = static_cast<int>(rows.size());
        probs->assign(n, 0.0f);
        out->assign(p_.dim, 0.0f);
        if (n == 0) return kNegInf;
        std::vector<float> logits(n);
        float max_logit = kNegInf;
        for (int i = 0; i < n; ++i) {
            logits[i] = dot(query, &k_[rows[i] * p_.dim], p_.dim) /
                        std::sqrt(static_cast<float>(p_.dim));
            max_logit = std::max(max_logit, logits[i]);
        }
        softmax(logits.data(), n, probs->data());
        for (int i = 0; i < n; ++i) {
            const float w = (*probs)[i];
            const float* vr = &v_[rows[i] * p_.dim];
            for (int d = 0; d < p_.dim; ++d) (*out)[d] += w * vr[d];
        }
        return max_logit;
    }

    StepResult compute() {
        const int n = t_ + 1;
        const int b = p_.block_len;
        const int dim = p_.dim;
        const float* query = &q_[t_ * dim];
        const int nb_closed = static_cast<int>(closed_ksum_.size()) / dim;
        const int open_len = n - nb_closed * b;  // 1..b while a step exists
        const int nb = nb_closed + (open_len > 0 ? 1 : 0);

        StepResult r;
        r.compressed.assign(dim, 0.0f);
        r.selected.assign(dim, 0.0f);
        r.windowed.assign(dim, 0.0f);
        r.output.assign(dim, 0.0f);

        // ---- Compression branch: block-mean keys/values. ----
        std::vector<float> logits(nb);
        std::vector<float> kmean(dim), vmean(dim);
        for (int j = 0; j < nb; ++j) {
            int lo, hi;
            const float* ksum;
            const float* vsum;
            if (j < nb_closed) {
                lo = j * b;
                hi = lo + b;
                ksum = &closed_ksum_[j * dim];
                vsum = &closed_vsum_[j * dim];
            } else {
                lo = nb_closed * b;
                hi = n;  // incomplete trailing block: only visible entries
                ksum = open_ksum_.data();
                vsum = open_vsum_.data();
            }
            const float inv = 1.0f / static_cast<float>(hi - lo);
            float lg = 0.0f;
            for (int d = 0; d < dim; ++d) {
                kmean[d] = ksum[d] * inv;
                vmean[d] = vsum[d] * inv;
                lg += query[d] * kmean[d];
            }
            logits[j] = lg / std::sqrt(static_cast<float>(dim));
        }
        r.comp_probs.resize(nb);
        softmax(logits.data(), nb, r.comp_probs.data());
        float comp_max_logit = kNegInf;
        for (int j = 0; j < nb; ++j) {
            comp_max_logit = std::max(comp_max_logit, logits[j]);
            int lo, hi;
            const float* vsum;
            if (j < nb_closed) {
                lo = j * b;
                hi = lo + b;
                vsum = &closed_vsum_[j * dim];
            } else {
                lo = nb_closed * b;
                hi = n;
                vsum = open_vsum_.data();
            }
            const float inv = 1.0f / static_cast<float>(hi - lo);
            const float w = r.comp_probs[j];
            for (int d = 0; d < dim; ++d) r.compressed[d] += w * vsum[d] * inv;
        }
        r.gate_compressed = sigmoid(comp_max_logit);

        // ---- Selection branch: top blocks by normalized relevance. ----
        const int take = std::min(p_.select_count, nb);
        std::vector<int> order(nb);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](int a, int idx) {
            if (r.comp_probs[a] != r.comp_probs[idx])
                return r.comp_probs[a] > r.comp_probs[idx];
            return a < idx;  // equal scores: smaller original block index first
        });
        std::vector<int> picked(order.begin(), order.begin() + take);
        std::sort(picked.begin(), picked.end());
        r.selected_blocks = picked;

        std::vector<int> sel_rows;
        sel_rows.reserve(static_cast<std::size_t>(take) * b);
        for (int j : picked) {
            int lo = j * b;
            int hi = std::min((j + 1) * b, n);
            for (int pos = lo; pos < hi; ++pos) sel_rows.push_back(pos);
        }
        std::vector<float> sel_probs;
        const float sel_max_logit =
            raw_attention(query, sel_rows, &r.selected, &sel_probs);
        r.gate_selected = take == 0 ? 0.0f : sigmoid(sel_max_logit);

        // ---- Local window branch: most recent visible positions. ----
        const int lo = std::max(0, n - p_.window);
        std::vector<int> win_rows;
        win_rows.reserve(n - lo);
        for (int pos = lo; pos < n; ++pos) win_rows.push_back(pos);
        std::vector<float> win_probs;
        const float win_max_logit =
            raw_attention(query, win_rows, &r.windowed, &win_probs);
        r.gate_windowed = sigmoid(win_max_logit);

        // ---- Fuse: three independent sigmoid gates, no renormalization. ----
        for (int d = 0; d < dim; ++d) {
            r.output[d] = r.gate_compressed * r.compressed[d] +
                          r.gate_selected * r.selected[d] +
                          r.gate_windowed * r.windowed[d];
        }
        return r;
    }

    Params p_;
    int t_ = -1;
    std::vector<float> q_, k_, v_;
    std::vector<float> closed_ksum_, closed_vsum_;
    std::vector<float> open_ksum_, open_vsum_;
    StepResult last_;
};

}  // namespace

Engine::~Engine() = default;

std::unique_ptr<Engine> Engine::create(const Params& p, std::string* err) {
    if (!valid_params(p, err)) return nullptr;
    return std::make_unique<EngineImpl>(p, p.dim);
}

bool forward(const Params& p, const float* q, const float* k, const float* v,
             int T, int dim, SequenceResult* out, std::string* err) {
    if (out == nullptr) {
        set_err(err, "output pointer must not be null");
        return false;
    }
    if (T <= 0) {
        set_err(err, "T must be positive");
        return false;
    }
    if (q == nullptr || k == nullptr || v == nullptr) {
        set_err(err, "q, k and v must not be null");
        return false;
    }
    if (dim != p.dim) {
        set_err(err, "dim does not match params.dim");
        return false;
    }
    const Params checked = p;
    if (!valid_params(p, err)) return false;
    const std::size_t count = static_cast<std::size_t>(T) * dim;
    if (!all_finite(q, count) || !all_finite(k, count) ||
        !all_finite(v, count)) {
        set_err(err, "q, k and v must contain only finite values");
        return false;
    }

    auto engine = Engine::create(checked, err);
    if (!engine) return false;
    SequenceResult local;
    local.steps.reserve(T);
    local.compressed.resize(count, 0.0f);
    local.selected.resize(count, 0.0f);
    local.windowed.resize(count, 0.0f);
    local.output.resize(count, 0.0f);
    for (int t = 0; t < T; ++t) {
        std::string append_err;
        if (!engine->append(q + t * dim, k + t * dim, v + t * dim, dim,
                            &append_err)) {
            set_err(err, append_err);
            return false;
        }
        StepResult step = engine->last();
        std::copy(step.compressed.begin(), step.compressed.end(),
                  &local.compressed[t * dim]);
        std::copy(step.selected.begin(), step.selected.end(),
                  &local.selected[t * dim]);
        std::copy(step.windowed.begin(), step.windowed.end(),
                  &local.windowed[t * dim]);
        std::copy(step.output.begin(), step.output.end(),
                  &local.output[t * dim]);
        local.steps.push_back(std::move(step));
    }
    *out = std::move(local);
    return true;
}

}  // namespace causal3
