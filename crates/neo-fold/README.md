# neo-fold

Owns the **only** transcript and the **single sum‑check over K = F_{q^s}**.
Implements Π_CCS, Π_RLC, Π_DEC, and their composition.

## Surface
- **Transcript**: public‑coin, domain‑separated labels for Σ‑check and each reduction.
- **Π_CCS**: build a batched Q that combines CCS constraints (F), base‑b **range** constraints (vanishing over digit alphabet), and **evaluation ties**; run a single sum‑check verifying `Σ_{x∈{0,1}^ℓ} Q(x)=0`, then open at one random point α ∈ K^ℓ to re‑randomize.
- **Π_RLC**: sample ρ_i ∈ C (from `neo-challenge`), combine k+1 ME(b,L) → one ME(B,L); track `||(Σ ρ_i Z_i)||_∞ ≤ (k+1)·T·(b−1) < B`.
- **Π_DEC**: split base‑B back to base‑b: verify `c = Σ b^{i-1} c_i` and `y_j = Σ b^{i-1} y_{i,j}`; return k ME(b,L).
- **Param guard**: assert `(k+1)·T·(b−1) < B` at fold start.

## Correctness & soundness
- SZ error ≤ **deg(Q)/|K|** (compute from actual construction).
- **Verified openings**: Ajtai openings and recomposition checks enforced here.

## Tests
- Sum‑check property tests vs. |K|.
- Π_DEC recomposition + negative tests.
- Π_RLC norm‑bound checks for the GL‑128 preset.
- Transcript determinism (same inputs → same challenges).