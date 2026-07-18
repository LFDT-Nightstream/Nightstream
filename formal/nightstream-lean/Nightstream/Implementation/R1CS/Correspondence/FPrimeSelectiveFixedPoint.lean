import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiRlcProjection

/-!
Correspondence root for selected fixed-point F-prime source-arm evidence.

Owns: the protocol → phase → constraint-family refinement hierarchy for the
bounded source fixture.

Does not own: selective-lowering refinement, complete F-prime/NIFS soundness,
source/transcript authority, bad-event bounds, final costs, or row removal.

Emits constraints: no.

| Phase | Checked source-arm slice | Still open |
|---|---|---|
| PiRLC projection | exact shared + `y_zcol` row/consumer consequence | selective lowering, source/transcript authority, bad-root bound |
-/
