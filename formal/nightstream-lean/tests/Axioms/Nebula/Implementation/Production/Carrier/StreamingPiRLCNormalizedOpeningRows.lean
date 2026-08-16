import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiRLCNormalizedOpeningRows
import tests.Axioms.Support

/-! Dependency audit for exact normalized production PiRLC opening rows. -/

open Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedOpeningRows.Normalized

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedOpeningRows.Normalized.openingChunkSchedule' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms openingChunkSchedule

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedOpeningRows.Normalized.accepted_implies_canonicalOpening' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms accepted_implies_canonicalOpening

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedOpeningRows.Normalized.accepted_implies_bodySourceColumnsExact' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms accepted_implies_bodySourceColumnsExact

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedOpeningRows.Normalized.accepted_implies_bodyPhaseBindingPlaced' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms accepted_implies_bodyPhaseBindingPlaced
