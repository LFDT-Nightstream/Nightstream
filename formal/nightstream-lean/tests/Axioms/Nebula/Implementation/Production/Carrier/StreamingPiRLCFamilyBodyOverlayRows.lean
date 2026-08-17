import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiRLCFamilyBodyOverlayRows
import tests.Axioms.Support

/-! Dependency audit for the split production PiRLC family rows. -/

open Nightstream.Implementation.Nebula.ProductionStreamingPiRlcFamilyBodyOverlayRows

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiRlcFamilyBodyOverlayRows.bodyRowsForParity_length' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms bodyRowsForParity_length

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiRlcFamilyBodyOverlayRows.sourceColumnsExact_of_bodyRows' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms sourceColumnsExact_of_bodyRows

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiRlcFamilyBodyOverlayRows.bodyRows_sound_of_output_exact' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms bodyRows_sound_of_output_exact

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiRlcFamilyBodyOverlayRows.rows_sound' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms rows_sound
