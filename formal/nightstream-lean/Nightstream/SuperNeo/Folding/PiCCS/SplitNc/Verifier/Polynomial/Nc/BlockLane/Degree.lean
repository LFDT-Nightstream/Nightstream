import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane.Degree.SumCheck

/-!
Degree contract for the canonical Split-NC block×lane polynomial.

Assurance tier: model-level.

Owns: the parent ownership boundary for source-slice and complete-polynomial
degree proofs.

Does not own: recursive SumCheck suffixes or messages, transcript scheduling,
Rust, R1CS, costs, or row removal.

Emits constraints: no.

Authority boundary: this facade exposes only coefficient representations
constructed from source-derived tables and verifier coins. It accepts no
prover-supplied degree claim.

| Child | Mathematical obligation | Emits constraints? | Lean owner |
|---|---|---:|---|
| `Degree.Source` | source MLE affine slices and strict cubic source mix | no | `BlockLane.Degree.Source` |
| `Degree.Polynomial` | equality-gated quartic slices and five-slot projection | no | `BlockLane.Degree.Polynomial` |
| `Degree.SumCheck` | flattened-coordinate, Boolean-suffix, and honest-round degree preservation | no | `BlockLane.Degree.SumCheck` |
-/
