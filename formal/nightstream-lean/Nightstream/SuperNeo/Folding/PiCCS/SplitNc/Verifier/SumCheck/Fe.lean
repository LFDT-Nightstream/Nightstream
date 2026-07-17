import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.Fe.Semantics

/-!
Public facade for mixed-width Split-NC FE SumCheck.

Owns: the stable import surface and child ownership only.

Does not own: declarations, transcript replay, Rust, R1CS, rows, removals, or
costs.

Emits constraints: no.

| Child module | Mathematical obligation | Emits constraints? | Lean owner |
|---|---|---|---|
| `Fe.Interface` | physical messages, serialization, and claimed-chain checking | no | `Verifier.SumCheck.Fe` |
| `Fe.Semantics` | honest-certificate existence and fixed-challenge completeness | no | `Verifier.SumCheck.Fe` |
-/
