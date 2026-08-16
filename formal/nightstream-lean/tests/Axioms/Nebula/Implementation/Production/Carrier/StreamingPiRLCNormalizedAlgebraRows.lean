import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiRLCNormalizedAlgebraRows
import tests.Axioms.Support

/-! Dependency audit for the normalized production PiRLC algebra rows. -/

/-- info: 'Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.Rows.evaluate_productPoint_one_eq_zero_iff' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.Rows.evaluate_productPoint_one_eq_zero_iff

/-- info: 'Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.Artifact.SourceImage.evaluate_sourceSlotForm' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.Artifact.SourceImage.evaluate_sourceSlotForm

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedAlgebraRows.Normalized.evaluate_combinationImage' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedAlgebraRows.Normalized.evaluate_combinationImage

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedAlgebraRows.Normalized.satisfies_implies_source_rows' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedAlgebraRows.Normalized.satisfies_implies_source_rows

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedAlgebraRows.Normalized.productionAccepted_implies_concrete_phase' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedAlgebraRows.Normalized.productionAccepted_implies_concrete_phase
