import Nightstream.Implementation.R1CS.Artifacts.PiCcsNc.Generated.PiCcsNcCarrierArtifact

/-!
Stable facade for the exact production SplitNc carrier-boundary export.

Owns: the public import boundary for one generated packed-witness
counterexample snapshot, including exact optimized `Pi_CCS` acceptance
booleans.

Does not own: independent NC semantics, general `Pi_CCS` refinement, any
NIFS/F-prime acceptance claim, Rust conformance beyond the drift-tested
snapshot, R1CS constraints, row removal, or constraint counts.

Emits constraints: no.

Authority boundary: generated values are implementation evidence only. A
handwritten correspondence module must interpret them against independent
full-carrier semantics.

| Child path | Mathematical obligation | Emits constraints? | Owner |
|---|---|---|---|
| `Generated.PiCcsNcCarrierArtifact` | exact logical/full decoding, `y_zcol`, and optimized `Pi_CCS` acceptance for one packed pair | no | Rust drift artifact |
-/
