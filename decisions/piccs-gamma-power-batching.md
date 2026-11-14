# Π_CCS γ-power batching

**Status:** Accepted

## Problem

The fold gate takes its bits from the census Π_CCS test error `N/q²`
(`Lifecycle.BoundedDepth.RandomOracle.FoldGate.production_bits`). At
`k_rho = 14`, `N = 4859`, and the γ term `J − 1 = 4719` of the
`J = 4720` joint coefficients is most of it
(`Spec.Folding.PiCCS.PaperJoint.RoundByRound.gammaBad_probability_le`).
A batching with `eq(γ⃗, j)` over `⌈log₂ J⌉ = 13` challenges replaces
`J − 1` with 13. Then `N = 153`, about 5.0 more bits (computed).

## SuperNeo

Appendix B.2, equation (14), batches the `J` claims with the powers of one
challenge `γ`. The paper does not use another batching.

## Decision

Keep the γ-power batching of SuperNeo v1.2. Do not use the `eq(γ⃗, j)`
batching. Owner, 2026-11-14.
