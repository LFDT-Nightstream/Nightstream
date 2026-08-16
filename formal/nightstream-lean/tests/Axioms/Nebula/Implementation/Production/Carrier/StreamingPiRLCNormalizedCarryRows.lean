import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiRLCNormalizedCarryRows
import tests.Axioms.Support

/-! Dependency audit for the normalized production PiRLC carry rows. -/

open Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedCarryRows.Normalized

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedCarryRows.Normalized.evaluate_combinationImage' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms evaluate_combinationImage

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedCarryRows.Normalized.equalityImage_accepted_iff' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms equalityImage_accepted_iff

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedCarryRows.Normalized.productionAccepted_implies_source_rows' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms productionAccepted_implies_source_rows

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedCarryRows.Normalized.productionAccepted_implies_range' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms productionAccepted_implies_range

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedCarryRows.Normalized.productionAccepted_implies_exact' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms productionAccepted_implies_exact

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedCarryRows.Normalized.productionAccepted_implies_exact_of_strong_set' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms productionAccepted_implies_exact_of_strong_set

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedCarryRows.Normalized.receipt_geometry_exact' depends on axioms: [Lean.trustCompiler] -/
#guard_msgs in
#audit_axioms receipt_geometry_exact
