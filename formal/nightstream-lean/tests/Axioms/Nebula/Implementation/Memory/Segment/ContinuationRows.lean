import Nightstream.Implementation.Nebula.Memory.Segment.ContinuationRows
import Nightstream.Implementation.Nebula.FPrime.Manifest.RecursiveNifsCall
import tests.Axioms.Support

open Nightstream.Implementation.Nebula

/-- info: 'Nightstream.Implementation.Nebula.SelectorGatedRows.rows_sound_selected' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms SelectorGatedRows.rows_sound_selected

/-- info: 'Nightstream.Implementation.Nebula.SelectorGatedRows.rows_complete_selected' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms SelectorGatedRows.rows_complete_selected

/-- info: 'Nightstream.Implementation.Nebula.SelectorGatedRows.rows_complete_unselected' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms SelectorGatedRows.rows_complete_unselected

/-- info: 'Nightstream.Implementation.Nebula.MemorySegmentContinuationRows.sound' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms MemorySegmentContinuationRows.sound

/-- info: 'Nightstream.Implementation.Nebula.MemorySegmentContinuationRows.rows_complete_active' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms MemorySegmentContinuationRows.rows_complete_active

/-- info: 'Nightstream.Implementation.Nebula.MemorySegmentContinuationRows.rows_complete_closed' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms MemorySegmentContinuationRows.rows_complete_closed

/-- info: 'Nightstream.Implementation.Nebula.RecursiveManifestNifsCall.Call.continuesExactIntermediateCarry' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms RecursiveManifestNifsCall.Call.continuesExactIntermediateCarry

/-- info: 'Nightstream.Implementation.Nebula.RecursiveManifestNifsCall.Call.priorStateLinkedConsumesAndContinues' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms RecursiveManifestNifsCall.Call.priorStateLinkedConsumesAndContinues
