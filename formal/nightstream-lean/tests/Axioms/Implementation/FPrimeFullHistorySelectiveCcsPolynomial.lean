import Nightstream.Implementation.R1CS.Correspondence.SelectiveCcs.Polynomial
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.CenteredDomainPackingArtifact
import tests.Axioms.Support

/-!
Fail-closed dependency gate for the model-level selective-polynomial component
decomposition and omission witnesses.
-/

/-- info: 'Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.Ports.Role.ofIndex_index' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.Ports.Role.ofIndex_index

/-- info: 'Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.Ports.Role.index_ofIndex' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.Ports.Role.index_ofIndex

/-- info: 'Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.Components.evaluate_eq_combinedResidual' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.Components.evaluate_eq_combinedResidual

/-- info: 'Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.Components.canonicalResidual_zero_of_generalSelector_zero' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.Components.canonicalResidual_zero_of_generalSelector_zero

/-- info: 'Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.Components.canonicalResidual_zero_of_classPorts_zero' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.Components.canonicalResidual_zero_of_classPorts_zero

/-- info: 'Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.Necessity.boolean_necessary' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.Necessity.boolean_necessary

/-- info: 'Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.Necessity.product_necessary' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.Necessity.product_necessary

/-- info: 'Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.Necessity.sbox_necessary' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.Necessity.sbox_necessary

/-- info: 'Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.Necessity.centered_necessary' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.Necessity.centered_necessary

/-- info: 'Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.Necessity.evaluation_necessary' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.Necessity.evaluation_necessary

/-- info: 'Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.Necessity.canonical_necessary' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.Necessity.canonical_necessary

/-- info: 'Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.Rows.evaluate_booleanPoint' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.Rows.evaluate_booleanPoint

/-- info: 'Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.Rows.evaluate_productPoint' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.Rows.evaluate_productPoint

/-- info: 'Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.Rows.evaluate_sboxPoint' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.Rows.evaluate_sboxPoint

/-- info: 'Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.Rows.evaluate_centeredPoint' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.Rows.evaluate_centeredPoint

/-- info: 'Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.Rows.evaluate_evaluationPoint' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.Rows.evaluate_evaluationPoint

/-- info: 'Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.Rows.evaluate_canonicalPoint' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.Rows.evaluate_canonicalPoint

/-- info: 'Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.Rows.evaluate_combinedBooleanPoint' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.Rows.evaluate_combinedBooleanPoint

/-- info: 'Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.Rows.evaluate_centeredPairPoint' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.Rows.evaluate_centeredPairPoint

/-- info: 'Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.PackedRows.centeredResidualAt_one_one' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.PackedRows.centeredResidualAt_one_one

/-- info: 'Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.PackedRows.evaluate_centeredPairPoint_one_zero_iff' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.PackedRows.evaluate_centeredPairPoint_one_zero_iff

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.CenteredDomainPacking.production_centeredPair_zero_iff' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.CenteredDomainPacking.production_centeredPair_zero_iff

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.CenteredDomainPacking.production_centeredTail_zero_iff' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.CenteredDomainPacking.production_centeredTail_zero_iff

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Row.CenteredDomain.rowPoint_eq_centeredPairPoint' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Row.CenteredDomain.rowPoint_eq_centeredPairPoint

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.CenteredDomainPackingArtifact.generated_pair_shape' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.CenteredDomainPackingArtifact.generated_pair_shape

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.CenteredDomainPackingArtifact.generated_pair_zero_iff' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.CenteredDomainPackingArtifact.generated_pair_zero_iff

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.CenteredDomainPackingArtifact.generated_tail_shape' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.CenteredDomainPackingArtifact.generated_tail_shape

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.CenteredDomainPackingArtifact.generated_tail_zero_iff' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.CenteredDomainPackingArtifact.generated_tail_zero_iff
