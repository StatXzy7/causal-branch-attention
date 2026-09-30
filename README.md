# causal-branch-attention

A small Linux/CPU three-branch causal attention engine in C++17:

1. **Compression (global)** — keys/values are averaged over contiguous blocks
   of `block_len`. The trailing run shorter than `block_len` forms one extra
   partial block whose mean is taken over its own members. Stable scaled
   dot-product softmax is computed over the block means.
2. **Selection (important regions)** — blocks are ranked by the compression
   branch's normalised relevance (descending score, ties broken by ascending
   original block index); the top `select_k` blocks are kept (truncated to the
   number of visible blocks). Attention is then recomputed over the **raw**
   keys/values inside the chosen blocks.
3. **Window (local)** — attention over the at most `window_len` most recent
   visible positions.

Every query at position `t` reads positions `0..t` only. Each branch has its
own numerically stable softmax, and the outputs are fused with three
**independent sigmoid gates that are summed without re-normalisation**:

```
out[t] = sigmoid(g_compression) * out_compression
       + sigmoid(g_selection)   * out_selection
       + sigmoid(g_window)      * out_window
```

A branch with no candidates contributes the zero vector (`select_k = 0`, or a
single-token input, still work).

Both modes are supported and share one row-computation kernel, so they are
bit-for-bit identical:

- `tri_attn::run_full(...)` — whole sequence at once.
- `tri_attn::Engine::append(...)` — stream one position at a time; previously
  emitted outputs can never change when future keys/values are appended or
  mutated. A rejected append leaves engine state untouched (strong exception
  guarantee).

An independently written plain-loop reference lives in
`tri_attn::naive::reference(...)`.

## Validation rules

The API rejects with `std::invalid_argument`:

- non-finite (`NaN`/`±Inf`) entries in queries, keys, values or gate logits;
- shape conflicts (null buffers for a non-empty run, zero `d_model`, width
  changes mid-stream);
- non-positive `block_len` / `window_len`; negative `select_k`.

## Layout

```
include/tri_attn/tri_attn.h   public API (Config, RowResult, run_full, Engine)
include/tri_attn/naive.h      naive reference declaration
src/tri_attn.cpp              engine + shared causal row kernel
src/naive.cpp                 independent naive reference
examples/demo.cpp             tiny float walk-through
tests/                        zero-dependency test framework + 19 cases
cmake/                        CTest helpers (incl. empty-suite failure)
```

Third-party dependencies: none (STL only).

## Build and test

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/demo
```

CTest runs three entries: the unit suite, a check that at least one test case
is registered, and a guard proving the runner exits non-zero when zero cases
are registered (so an accidentally empty suite fails CTest instead of passing).

The 19 unit cases cover: single-token inputs, disabled/empty branches,
block-selection order and tie-breaking, `select_k` truncation, partial
trailing blocks (mean and raw-value attention), three-gate numerics,
window-only visibility, batch-versus-append bit equality, naive-reference
agreement over random sequences, input validation, state preservation after
failed appends, causal invariance to future mutation, and finiteness under
extreme-but-finite inputs.
