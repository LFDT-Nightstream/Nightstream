# neo-reductions

The folding engines: Π_CCS as a sum-check protocol, plus the RLC/DEC operations, in
optimized and paper-exact variants. This crate owns the protocol *math*;
`neo-fold-clean`'s paper layer owns the protocol *sequencing*.

## Owns

- **`api`** — the public entry points: `FoldingMode` (`Optimized`; `PaperExact` and
  `OptimizedWithCrosscheck` behind the `paper-exact` feature), `pi_ccs_prove` /
  `pi_ccs_verify`, `PiCcsProof`, `Challenges`, `CcsOracle`.
- **`engines`** —
  - `optimized_engine` — production: sparse-aware sum-check with structure caching
    (`OptimizedStructureCache`), instance digests, accumulator-handle threading, and
    perf instrumentation.
  - `paper_exact_engine` — O(2^ℓ) brute-force reference implementation, used only to
    cross-check the optimized engine.
  - `crosscheck_engine` — runs both and compares.
  - `pi_rlc_dec` — the RLC and DEC operations re-exported under a stable path.
- **`sumcheck`** — the SumCheck(T; Q) protocol (SuperNeo Def. 6) over `K = F_{q²}`.
- **`superneo_eval`** — transformed-matrix evaluators for the SuperNeo bar-lifted
  forms.
- **`common`** — strong-set challenge material: `RotRho` / `RotRing`,
  `sample_rot_rhos_n*`, `split_b_matrix_k*`.

## Policy

Tests always use `FoldingMode::Optimized`; `PaperExact` requires explicit approval per
[CLAUDE.md](../../CLAUDE.md) — it is a correctness oracle, not a usable engine
(exponential in the number of sum-check rounds).

## Specs

`specs/{PiCCS, PiRLC, PiDEC, SumCheck, Engines, SuperNeoEval}.spec.md`. Engine parity
and digest discipline are tested in this crate's `tests/` (e.g. `matrix_digest.rs`,
`nc_digit_table_parity.rs`).
