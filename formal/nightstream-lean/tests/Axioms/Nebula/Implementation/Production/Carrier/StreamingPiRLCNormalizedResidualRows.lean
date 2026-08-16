import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiRLCNormalizedResidualRows
import tests.Axioms.Support

/-! Dependency audit for the normalized production PiRLC residual rows. -/

open Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedResidualRows.Normalized

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedResidualRows.Normalized.productionAccepted_implies_source_rows' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms productionAccepted_implies_source_rows

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedResidualRows.Normalized.productionAccepted_implies_transition' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms productionAccepted_implies_transition

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedResidualRows.Normalized.receipt_geometry_exact' depends on axioms: [Lean.trustCompiler] -/
#guard_msgs in
#audit_axioms receipt_geometry_exact
