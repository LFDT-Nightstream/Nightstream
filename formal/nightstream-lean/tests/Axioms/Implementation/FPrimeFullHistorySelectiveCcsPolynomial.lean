import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.Polynomial
import tests.Axioms.Support

/-!
Fail-closed dependency gate for the model-level selective-polynomial component
decomposition and omission witnesses.
-/

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Polynomial.Ports.Role.ofIndex_index' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Polynomial.Ports.Role.ofIndex_index

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Polynomial.Ports.Role.index_ofIndex' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Polynomial.Ports.Role.index_ofIndex

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Polynomial.Components.evaluate_eq_combinedResidual' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Polynomial.Components.evaluate_eq_combinedResidual

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Polynomial.Components.canonicalResidual_zero_of_generalSelector_zero' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Polynomial.Components.canonicalResidual_zero_of_generalSelector_zero

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Polynomial.Components.canonicalResidual_zero_of_classPorts_zero' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Polynomial.Components.canonicalResidual_zero_of_classPorts_zero

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Polynomial.Necessity.boolean_necessary' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Polynomial.Necessity.boolean_necessary

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Polynomial.Necessity.product_necessary' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Polynomial.Necessity.product_necessary

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Polynomial.Necessity.sbox_necessary' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Polynomial.Necessity.sbox_necessary

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Polynomial.Necessity.centered_necessary' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Polynomial.Necessity.centered_necessary

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Polynomial.Necessity.evaluation_necessary' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Polynomial.Necessity.evaluation_necessary

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Polynomial.Necessity.canonical_necessary' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Polynomial.Necessity.canonical_necessary

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Polynomial.Rows.evaluate_booleanPoint' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Polynomial.Rows.evaluate_booleanPoint

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Polynomial.Rows.evaluate_productPoint' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Polynomial.Rows.evaluate_productPoint

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Polynomial.Rows.evaluate_sboxPoint' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Polynomial.Rows.evaluate_sboxPoint

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Polynomial.Rows.evaluate_centeredPoint' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Polynomial.Rows.evaluate_centeredPoint

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Polynomial.Rows.evaluate_evaluationPoint' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Polynomial.Rows.evaluate_evaluationPoint

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Polynomial.Rows.evaluate_canonicalPoint' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Polynomial.Rows.evaluate_canonicalPoint
