import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiRlcProjection
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.Accumulator

/-!
Correspondence root for selected fixed-point F-prime source-arm evidence.

Owns: the protocol → phase → constraint-family refinement hierarchy for the
bounded source fixture and the fixed-profile production `Pi_CCS`/NC delayed
projection boundary.

Does not own: selective-lowering refinement, native production-trace
construction, commitment binding, bad-event probability bounds, final costs,
or row removal.

Emits constraints: no.

| Phase | Checked source-arm slice | Still open |
|---|---|---|
| PiRLC projection | exact shared + `y_zcol` row/consumer consequence | selective lowering, source/transcript authority, bad-root bound |
| PiCCS/NC delayed projection | exact 25-round block×lane model, artifact-checked packed-`Z` geometry, and base/recursive/terminal active trace closure | native raw-witness handoff, concrete combined-NC/state/terminal rows, commitment/key alignment |
| accumulator pending-family codec | injective shared-point plus 14-child plus delayed-state carrier; exact κ=4/κ=18 field counts | Rust serializer, row ownership, and deletion authority |
-/
