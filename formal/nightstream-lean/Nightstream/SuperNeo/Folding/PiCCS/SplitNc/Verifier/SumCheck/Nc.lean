import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.Nc.Semantics

/-!
Public facade for exact-width Split-NC NC SumCheck.

Owns: the stable import surface and child ownership only.

Does not own: declarations, transcript replay, output authority, Rust, R1CS,
rows, removals, or costs.

Emits constraints: no.

| Child module | Mathematical obligation | Emits constraints? | Lean owner |
|---|---|---|---|
| `Nc.Interface` | exact-width codec and claimed-chain checking | no | `Verifier.SumCheck.Nc` |
| `Nc.Semantics` | terminal binding, completeness, and named bad events | no | `Verifier.SumCheck.Nc` |
-/
