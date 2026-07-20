import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.HonestAssignment.MaterializedExecution.DerivedSlotEvidence

/-!
Facade for source-value and derived-slot materialization evidence in the
bounded selective fixed-point `y_zcol` rewrite program.

Owns: the stable import boundary joining source-value and derived-slot
materialization evidence.

Does not own: either child proof, full-program recurrence composition,
retained checks, selected-row completeness, or producer authority.

Emits constraints: no.

| Child leaf | Mathematical obligation | Authority class |
|---|---|---|
| `SourceValues` | source linear forms and factor sums preserve abstract values | derived |
| `DerivedSlotEvidence` | centered-word decoding preserves each derived field | derived |
-/
