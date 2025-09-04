# neo-ajtai

Ajtai matrix commitment with pay‑per‑bit embedding.

## Surface
- `Setup`: sample **M ∈ R_q^{κ×m}** (κ, m come from `neo-params`).
- `Commit(pp, Z)`: **L(Z) = cf(M · cf^{-1}(Z))**.
- **S‑homomorphism:** for ρ ∈ S, `ρ·L(Z) = L(ρ·Z)`; linear in Z.
- Embedding: `decomp_b(z) → Z`, `split_b(Z) → [Z_i]`, `||Z_i||_∞ < b`.
- **Verified openings (v1)**: range/decomposition recomposition only:
  - `verify_open(pp, c, Z)`
  - `verify_split_open(pp, c, b, c_i, Z_i)`
  - ⚠️ **Linear openings (`y = Z·v`) are intentionally NOT PROVIDED**.
    Use `neo_fold::verify_linear` (Π_RLC) for linear relation verification instead.

## Properties
- (d,m,B)‑binding and (d,m,B,C)‑relaxed binding under MSIS presets from `neo-params`.
- **Pay‑per‑bit ring multiply** so commit cost scales with Hamming weight.

## Tests
- S‑linearity; decomp/split recomposition; negative cases.
- Opening‑verification tests; microbench showing pay‑per‑bit scaling.
- Security test confirms `verify_linear()` is not provided (use `neo_fold::verify_linear` instead).
