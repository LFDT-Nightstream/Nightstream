# neo-params

Typed, validated parameter sets, plus the canonical Poseidon2 configuration. No
workspace dependencies — this is the root crate.

## Owns

- **`NeoParams`** — the full symbol set with paper names: `q, η, d, κ, m, b, k_rho,
  B, T, s, λ`. Constructors validate; invalid bundles do not exist as values.
  - Π_RLC bound: `(k+1)·T·(b−1) < B` where `B = b^k`.
  - Extension policy v1: `s_min = ceil((λ + log₂(soundness_factor)) / log₂(q))`;
    only `s = 2` supported; `s_min > 2` is a configuration error; slack recorded.
- **`goldilocks_paper_b2`** — the SuperNeo Appendix B.2 preset constants (single
  source of truth; values in [Parameters](../protocol/parameters.md)), including
  `PHI_COEFFS`, `CHALLENGE_ALPHABET = {−2…2}`, `MAX_FRESH_K = 61`.
- **Shape-aware constructors** — `goldilocks_auto_r1cs_ccs_with` /
  `goldilocks_auto_ccs_with`: keep the B.2 core, derive the effective λ a concrete
  shape can support under `s = 2` (these back `Params::for_r1cs_shape*` in
  `neo-fold-clean`).
- **`poseidon2_goldilocks`** — the production Poseidon2 instance every transcript and
  digest in the workspace must use.

## Does not own

Per-instance sum-check dimensions (ℓ, d_sc) — those are checked at the folding layer
against the preset's `q, s, λ`.
