# neo-math

Field, ring, and embedding primitives. `#![forbid(unsafe_code)]`.

## Owns

- **`field`** — `Fq` (Goldilocks, `q = 2⁶⁴ − 2³² + 1`; workspace-wide alias
  `neo_math::F`) and the degree-2 extension `K = F_{q²}` used for sum-check
  soundness (`KExtensions`, `from_complex`).
- **`ring`** — `Rq = F_q[X]/Φ_81(X)` with `Φ_81 = X⁵⁴ + X²⁷ + 1` (`D = 54`,
  `ETA = 81`); the coefficient maps `cf` / `cf_inv` and constant-term functional `ct`;
  the SuperNeo §5 lifted transform `superneo_bar_block` / `superneo_bar_vec` /
  `superneo_bar_matrix` realizing `Mz = ct(bar(M)·z)`.
- **`balanced`** — centered (balanced) representation, the ℓ∞ norm, and `split_b`
  (Def. 3 balanced b-ary decomposition).
- **`s_action`** — `SAction`: the ring acting as `d×d` rotation matrices on
  coefficient vectors; the algebra behind commitment-homomorphic challenge mixing.

## Consumers and proofs

`neo-ajtai` builds commitments from ring arithmetic. `neo-ccs` defines
relations over the base and extension fields. `neo-reductions` and
`nightstream` use these operations in folding and terminal verification.

The maintained algebra and implementation proofs live in
`formal/nightstream-fprime`. Rust checks live in `crates/neo-math/tests`.
