import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiRLCFamilyCompleteRows
import tests.Axioms.Support

/-! Dependency audit for the complete production PiRLC family rows. -/

open Nightstream.Implementation.Nebula.ProductionStreamingPiRlcFamilyCompleteRows

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiRlcFamilyCompleteRows.exact_layout' depends on axioms: [Lean.trustCompiler] -/
#guard_msgs in
#audit_axioms exact_layout

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiRlcFamilyCompleteRows.rows_length' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms rows_length

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiRlcFamilyCompleteRows.rows_sound' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms rows_sound
