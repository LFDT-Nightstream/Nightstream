# neo-params

Typed, validated parameter sets, plus the canonical Poseidon2 configuration. No
workspace dependencies — this is the root crate.

## Owns

- **`NeoParams`** — the full symbol set with paper names: `q, η, d, κ, m, b, k_rho,
  B, T, s, λ`. Constructors validate; invalid bundles do not exist as values.
  - Π_RLC bound: `(k+1)·T·(b−1) < B` where `B = b^k`.
  - `λ` is the statistical bits that a gate certified. It is absent until
    `with_certified_lambda`, and the census check refuses parameters without it.
- **`nightstream_p64_k14`** — the constants of the one profile, "Nightstream P64,
  k_rho = 14" (single source of truth; values in [SPEC.md](../../SPEC.md),
  section 3), including `CHALLENGE_ALPHABET = {−2…2}`.
  `NeoParams::nightstream_p64_k14()` builds the validated parameter set.
- **`poseidon2`** — the sizes of the production Poseidon2 instance that every
  transcript and digest in the workspace must use.

## Does not own

Per-instance sum-check dimensions (ℓ, d_sc) — those are checked at the folding layer
against the preset's `q, s, λ`.
