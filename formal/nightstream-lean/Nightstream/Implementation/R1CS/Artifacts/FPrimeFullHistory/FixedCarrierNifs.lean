import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFixedCarrierNifsArtifact

/-!
Stable facade for the fixed F-prime carrier NIFS/F-prime execution artifact.

Owns: the public import boundary for exact optimized NIFS executions and one
recursive F-prime circuit execution under the current `1 x 257` all-zero
carrier fixture, preprocessing seed 41, and canonical fixed-k zero accumulator.

Does not own: independent NC semantics, a general Pi_CCS/NIFS refinement
theorem, F-prime acceptance, a production relation choice, R1CS constraints,
constraint removal, or counts.

Emits constraints: no.

Authority boundary: generated values are implementation evidence only. The
handwritten correspondence layer must identify the exported witness with an
independently specified semantic counterexample.

| Child path | Mathematical obligation | Emits constraints? | Owner |
|---|---|---|---|
| `Generated.FPrimeFixedCarrierNifsArtifact` | exact structure profile, completed-carrier witness pairs, canonical running count, optimized Pi_CCS/fixed-NIFS acceptance, and recursive F-prime satisfaction | no | Rust drift artifact |
-/
