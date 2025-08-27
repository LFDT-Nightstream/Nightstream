# neo-challenge

Strong sampling set C ⊂ S for random‑linear combination (RLC).

## Surface
- Transcript‑seeded sampler over small‑coeff ring elements a ∈ R_q (distribution from `neo-params`).
- Produce ρ = rot(a) and track **expansion T** for the chosen (η, coeff set).
- **Invertibility**: for ρ ≠ ρ′, verify (ρ − ρ′) is invertible in S by checking **gcd(a − a′, Φ_η) = 1** (or extended‑Euclid inverse in R_q).

## Guarantees
- With overwhelming probability, pairwise differences are invertible (failure rate measured & exported).
- Export **T** to consumers; Π_RLC uses it to enforce `(k+1)·T·(b−1) < B`.

## Tests
- Empirical expansion vs. bound T.
- Invertibility failure‑rate measurements.
- Deterministic seeding via domain‑separated labels.
