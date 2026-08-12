import Nightstream.Implementation.NebulaV2.Memory.Segment.ContinuationRows
import Nightstream.Implementation.NebulaV2.FPrime.Manifest.RecursiveNifsCall
import tests.Axioms.Support

open Nightstream.Implementation.NebulaV2

/-- info: 'Nightstream.Implementation.NebulaV2.SelectorGatedRows.rows_sound_selected' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms SelectorGatedRows.rows_sound_selected

/-- info: 'Nightstream.Implementation.NebulaV2.SelectorGatedRows.rows_complete_selected' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms SelectorGatedRows.rows_complete_selected

/-- info: 'Nightstream.Implementation.NebulaV2.SelectorGatedRows.rows_complete_unselected' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms SelectorGatedRows.rows_complete_unselected

/-- info: 'Nightstream.Implementation.NebulaV2.MemorySegmentContinuationRows.sound' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms MemorySegmentContinuationRows.sound

/-- info: 'Nightstream.Implementation.NebulaV2.MemorySegmentContinuationRows.rows_complete_active' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms MemorySegmentContinuationRows.rows_complete_active

/-- info: 'Nightstream.Implementation.NebulaV2.MemorySegmentContinuationRows.rows_complete_closed' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms MemorySegmentContinuationRows.rows_complete_closed

/-- info: 'Nightstream.Implementation.NebulaV2.RecursiveManifestNifsCall.Call.continuesExactIntermediateCarry' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms RecursiveManifestNifsCall.Call.continuesExactIntermediateCarry

/-- info: 'Nightstream.Implementation.NebulaV2.RecursiveManifestNifsCall.Call.priorStateLinkedConsumesAndContinues' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms RecursiveManifestNifsCall.Call.priorStateLinkedConsumesAndContinues
