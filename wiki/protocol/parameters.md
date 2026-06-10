# Parameters (SuperNeo Appendix B.2, Goldilocks)

Parameter selection and validation live in `crates/neo-params`; the paper layer wraps
the validated bundle as `neo-fold-clean` `paper::params::Params` (Definition 14
globals), and `crates/neo-fold-clean/src/config.rs` re-exports the audited constants
under the profile name `superneo-appendix-b2-goldilocks-b2`.

## The production preset

`Params::production()` = `NeoParams::goldilocks_paper_b2()` — the SuperNeo
Appendix B.2 `b = 2` row:

| Symbol | Value | Meaning |
|---|---|---|
| q | 2⁶⁴ − 2³² + 1 | Goldilocks base field modulus |
| η | 81 | Cyclotomic index; `Φ_81(X) = X⁵⁴ + X²⁷ + 1` |
| d | 54 | `φ(81)` — ring degree, S-action dimension |
| κ | 18 | Ajtai module-SIS rank (`M ∈ R_q^{κ×m}`) |
| m | 2³⁰ | Max Ajtai message length (columns) |
| b | 2 | Decomposition base / low-norm bound |
| k_rho | 14 | Folding exponent; Π_DEC child count |
| B | 2¹⁴ | Post-RLC norm bound, `B = b^k_rho` |
| T | 216 | Strong-set expansion factor (Thm. 9) |
| s | 2 | Sum-check extension degree, `K = F_{q²}` |
| λ | 125 | Target soundness bits |

Supporting constants (`neo_params::goldilocks_paper_b2`): challenge alphabet
`{−2, −1, 0, 1, 2}` for the strong sampling set, `MAX_FRESH_K = 61` (cap on fresh
instances per fold), `B_INV_FLOOR`.

## Validity checks enforced in code

`neo-params` refuses to construct invalid bundles:

1. **Π_RLC norm bound** — `(k+1)·T·(b−1) < B` at preset level; the paper layer
   additionally checks the fold-width form `(K+k)·T·(b−1) < B` for a concrete fold via
   `paper/sampling.rs::check_rlc_bound`. This is what guarantees Π_RLC's output stays
   inside the Ajtai binding norm.
2. **Extension policy v1** — `s_min = ceil((λ + log₂(soundness_factor)) / log₂(q))`;
   only `s = 2` is supported. If a shape's sum-check soundness budget needs `s_min > 2`
   the configuration errors out rather than silently degrading; remaining slack is
   recorded as `slack_bits`.

## Shape-specific effective λ

The `s = 2` engine cannot give all shapes the full λ = 125: the SuperNeo D.4
Schwartz–Zippel term grows with the sum-check size. For R1CS-derived CCS,
`Params::for_r1cs_shape(n_rows)` keeps every Appendix B.2 core value and lowers *only*
the effective λ, with a floor of 96 bits and a safety margin of 2
(`for_r1cs_shape_with` exposes both knobs; `for_ccs_shape_with` is the general-CCS
variant taking SuperNeo's `t` and `u`). This is the profile named
`superneo-appendix-b2-goldilocks-b2-r1cs-effective-lambda` in `config.rs`.

`crates/neo-fold-clean/tests/system/production_params.rs` pins the production profile;
treat any test that wants different parameters as a red flag.

## Poseidon2

`neo-params` also owns the canonical Poseidon2-over-Goldilocks configuration
(`neo_params::poseidon2_goldilocks`) — the single source of truth for every transcript
and digest in the workspace. No other hash family is permitted in protocol-binding
paths (see [Transcript & digests](transcript-and-digests.md)).
