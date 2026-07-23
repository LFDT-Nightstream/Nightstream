import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.Accumulator.PendingFamilyCodec

/-!
Fixed-point accumulator correspondence facade.

Owns: the exact typed codec contract for the direct pending-family
accumulator carrier used by the constraint-reduction track.

Does not own: Rust serialization, SIS/Poseidon2 lowering, native delayed-state
integration, generated row ownership, measured savings, or row deletion.

Emits constraints: no.

| Child path | Mathematical obligation | Excluded boundary |
|---|---|---|
| `Accumulator.PendingFamilyCodec` | injectively retain the shared points, ordered child payloads, and delayed packed parent | concrete Rust/R1CS refinement and physical cost |
-/
