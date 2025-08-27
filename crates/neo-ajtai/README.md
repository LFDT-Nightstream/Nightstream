# neo-ajtai

Ajtai matrix commitment with pay‑per‑bit embedding.

## Surface
- `Setup`: sample **M ∈ R_q^{κ×m}** (κ, m come from `neo-params`).
- `Commit(pp, Z)`: **L(Z) = cf(M · cf^{-1}(Z))**.
- **S‑homomorphism:** for ρ ∈ S, `ρ·L(Z) = L(ρ·Z)`; linear in Z.
- Embedding: `decomp_b(z) → Z`, `split_b(Z) → [Z_i]`, `||Z_i||_∞ < b`.
- **Verified openings**: APIs to check openings used by range/decomp and eval recomposition.

## Properties
- (d,m,B)‑binding and (d,m,B,C)‑relaxed binding under MSIS presets from `neo-params`.
- **Pay‑per‑bit ring multiply** so commit cost scales with Hamming weight.

## Tests
- S‑linearity; decomp/split recomposition; negative cases.
- Opening‑verification tests; microbench showing pay‑per‑bit scaling.
