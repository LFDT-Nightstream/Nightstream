import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiRLCNormalizedOverlayRows
import tests.Axioms.Support

/-! Dependency audit for normalized production PiRLC family-overlay rows. -/

open Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedOverlayRows.Normalized

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedOverlayRows.Normalized.accepted_implies_phaseBindingPlaced' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms accepted_implies_phaseBindingPlaced

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedOverlayRows.Normalized.receipt_geometry_exact' depends on axioms: [Lean.trustCompiler] -/
#guard_msgs in
#audit_axioms receipt_geometry_exact
