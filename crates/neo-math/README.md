# neo-math

Algebraic foundations used everywhere.

## Scope
- Base field **Fq = Goldilocks** (q = 2^64 − 2^32 + 1).
- Extension field **K = F_{q^s}** with **s supplied by `neo-params`** (GL‑128 preset: s=2).
- Cyclotomic ring **R_q = F_q[X]/(Φ_η)** with **η provided by `neo-params`**; let **d = φ(η)**.
- Coefficient maps `cf: R_q → F_q^d` and `cf^{-1}`.
- Rotation matrices `rot(a)` realizing **S = { rot(a) : a ∈ R_q } ⊂ F_q^{d×d}** and the isomorphism **R_q ≅ S**.
- Fixed‑size vectors/matrices over Fq; left action by S.

## Requirements
- Constant‑time field ops; batch inversion helper.
- Prove `rot(a)·cf(b) == cf(a·b)` on random samples.
- (Optional) NTT hooks for common `d` (kept behind internal config, not features).

## Tests
- Field laws on Fq and K (for any `s`).
- Ring isomorphism checks R_q↔S via `cf`, `rot`.
- Rotation identities; scalar matrices ⊂ S.