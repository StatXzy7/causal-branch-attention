// Naive reference: explicit loops, double precision accumulators, no block-sum
// state. One pass per query position over blocks and rows. The operation order
// mirrors the definition of the three branches.
#include "causal3/reference.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace causal3 {
namespace ref_detail {

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

std::vector<double> softmax_d(const std::vector<double>& logits) {
    const int n = static_cast<int>(logits.size());
    double m = logits[0];
    for (double x : logits) m = std::max(m, x);
    std::vector<double> probs(n);
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        probs[i] = std::exp(logits[i] - m);
        sum += probs[i];
    }
    for (double& x : probs) x /= sum;
    return probs;
}

double sigmoid_d(double x) {
    if (x >= 0.0) {
        const double e = std::exp(-x);
        return 1.0 / (1.0 + e);
    }
    const double e = std::exp(x);
    return e / (1.0 + e);
}

}  // namespace ref_detail

bool reference_forward(const Params& p, const float* qf, const float* kf,
                       const float* vf, int T, int dim, SequenceResult* out,
                       std::string* err) {
    using namespace ref_detail;
    if (out == nullptr) {
        set_err(err, "output pointer must not be null");
        return false;
    }
    if (T <= 0) {
        set_err(err, "T must be positive");
        return false;
    }
    if (qf == nullptr || kf == nullptr || vf == nullptr) {
        set_err(err, "q, k and v must not be null");
        return false;
    }
    if (dim != p.dim) {
        set_err(err, "dim does not match params.dim");
        return false;
    }
    const Params checked = p;
    if (!valid_params(checked, err)) return false;
    const std::size_t count = static_cast<std::size_t>(T) * dim;
    for (std::size_t i = 0; i < count; ++i) {
        if (!std::isfinite(qf[i]) || !std::isfinite(kf[i]) ||
            !std::isfinite(vf[i])) {
            set_err(err, "q, k and v must contain only finite values");
            return false;
        }
    }

    std::vector<double> q(count), k(count), v(count);
    for (std::size_t i = 0; i < count; ++i) {
        q[i] = qf[i];
        k[i] = kf[i];
        v[i] = vf[i];
    }
    const int b = checked.block_len;
    const int w = checked.window;
    const int nsel = checked.select_count;
    const double scale = 1.0 / std::sqrt(static_cast<double>(dim));

    SequenceResult result;
    result.steps.resize(T);
    result.compressed.assign(count, 0.0f);
    result.selected.assign(count, 0.0f);
    result.windowed.assign(count, 0.0f);
    result.output.assign(count, 0.0f);

    for (int t = 0; t < T; ++t) {
        const double* query = &q[t * dim];
        const int n = t + 1;
        const int nb_full = n / b;
        const int rem = n - nb_full * b;  // visible length of trailing block
        const int nb = nb_full + (rem > 0 ? 1 : 0);

        auto block_mean = [&](int j, int which) {
            const int lo = j * b;
            const int hi = std::min((j + 1) * b, n);
            std::vector<double> mean(dim, 0.0);
            for (int pos = lo; pos < hi; ++pos) {
                const double* src = which == 0 ? &k[pos * dim] : &v[pos * dim];
                for (int d = 0; d < dim; ++d) mean[d] += src[d];
            }
            const double inv = 1.0 / static_cast<double>(hi - lo);
            for (int d = 0; d < dim; ++d) mean[d] *= inv;
            return mean;
        };

        StepResult& step = result.steps[t];
        step.compressed.assign(dim, 0.0f);
        step.selected.assign(dim, 0.0f);
        step.windowed.assign(dim, 0.0f);
        step.output.assign(dim, 0.0f);

        // Compression branch.
        std::vector<double> comp_logits(nb);
        std::vector<std::vector<double>> vmeans(nb);
        for (int j = 0; j < nb; ++j) {
            std::vector<double> km = block_mean(j, 0);
            vmeans[j] = block_mean(j, 1);
            double s = 0.0;
            for (int d = 0; d < dim; ++d) s += query[d] * km[d];
            comp_logits[j] = s * scale;
        }
        std::vector<double> comp_probs = softmax_d(comp_logits);
        std::vector<double> comp_out(dim, 0.0);
        for (int j = 0; j < nb; ++j)
            for (int d = 0; d < dim; ++d)
                comp_out[d] += comp_probs[j] * vmeans[j][d];
        double comp_max = comp_logits[0];
        for (double x : comp_logits) comp_max = std::max(comp_max, x);
        const double g_comp = sigmoid_d(comp_max);

        // Selection branch.
        const int take = std::min(nsel, nb);
        std::vector<int> order(nb);
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(),
                         [&](int a, int c) {
                             if (comp_probs[a] != comp_probs[c])
                                 return comp_probs[a] > comp_probs[c];
                             return a < c;
                         });
        std::vector<int> picked(order.begin(), order.begin() + take);
        std::sort(picked.begin(), picked.end());
        step.selected_blocks.assign(picked.begin(), picked.end());
        for (double pr : comp_probs) step.comp_probs.push_back(static_cast<float>(pr));

        std::vector<int> sel_rows;
        for (int j : picked) {
            const int lo = j * b;
            const int hi = std::min((j + 1) * b, n);
            for (int pos = lo; pos < hi; ++pos) sel_rows.push_back(pos);
        }
        std::vector<double> sel_out(dim, 0.0);
        double g_sel = 0.0;
        if (!sel_rows.empty()) {
            std::vector<double> logits(sel_rows.size());
            for (std::size_t i = 0; i < sel_rows.size(); ++i) {
                double s = 0.0;
                const double* kr = &k[sel_rows[i] * dim];
                for (int d = 0; d < dim; ++d) s += query[d] * kr[d];
                logits[i] = s * scale;
            }
            std::vector<double> probs = softmax_d(logits);
            double mx = logits[0];
            for (double x : logits) mx = std::max(mx, x);
            g_sel = sigmoid_d(mx);
            for (std::size_t i = 0; i < sel_rows.size(); ++i) {
                const double* vr = &v[sel_rows[i] * dim];
                for (int d = 0; d < dim; ++d) sel_out[d] += probs[i] * vr[d];
            }
        }

        // Window branch.
        const int lo = std::max(0, n - w);
        std::vector<int> win_rows;
        for (int pos = lo; pos < n; ++pos) win_rows.push_back(pos);
        std::vector<double> win_logits(win_rows.size());
        for (std::size_t i = 0; i < win_rows.size(); ++i) {
            double s = 0.0;
            const double* kr = &k[win_rows[i] * dim];
            for (int d = 0; d < dim; ++d) s += query[d] * kr[d];
            win_logits[i] = s * scale;
        }
        std::vector<double> win_probs = softmax_d(win_logits);
        double win_max = win_logits[0];
        for (double x : win_logits) win_max = std::max(win_max, x);
        const double g_win = sigmoid_d(win_max);
        std::vector<double> win_out(dim, 0.0);
        for (std::size_t i = 0; i < win_rows.size(); ++i) {
            const double* vr = &v[win_rows[i] * dim];
            for (int d = 0; d < dim; ++d) win_out[d] += win_probs[i] * vr[d];
        }

        step.gate_compressed = static_cast<float>(g_comp);
        step.gate_selected = static_cast<float>(g_sel);
        step.gate_windowed = static_cast<float>(g_win);
        for (int d = 0; d < dim; ++d) {
            step.compressed[d] = static_cast<float>(comp_out[d]);
            step.selected[d] = static_cast<float>(sel_out[d]);
            step.windowed[d] = static_cast<float>(win_out[d]);
            step.output[d] = static_cast<float>(g_comp * comp_out[d] +
                                                g_sel * sel_out[d] +
                                                g_win * win_out[d]);
            result.compressed[t * dim + d] = step.compressed[d];
            result.selected[t * dim + d] = step.selected[d];
            result.windowed[t * dim + d] = step.windowed[d];
            result.output[t * dim + d] = step.output[d];
        }
    }

    *out = std::move(result);
    return true;
}

}  // namespace causal3
