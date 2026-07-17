import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.Artifact.Row.Gating
import tests.Axioms.Support

/-!
Fail-closed dependency gate for the model-level selective-CCS row-action
bridge. These theorems still require an exact concrete matrix-image refinement.
-/

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.RowAction.matrixImageAt_eq_paddedMatrixVectorAt' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.RowAction.matrixImageAt_eq_paddedMatrixVectorAt

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.RowAction.residualAt_eq_evaluate' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.RowAction.residualAt_eq_evaluate

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.RowAction.residualAt_booleanPoint' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.RowAction.residualAt_booleanPoint

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.RowAction.residualAt_productPoint' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.RowAction.residualAt_productPoint

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.RowAction.residualAt_sboxPoint' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.RowAction.residualAt_sboxPoint

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.RowAction.residualAt_centeredPoint' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.RowAction.residualAt_centeredPoint

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.RowAction.residualAt_evaluationPoint' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.RowAction.residualAt_evaluationPoint

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.RowAction.residualAt_canonicalPoint' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.RowAction.residualAt_canonicalPoint

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Row.Gating.residual_eq_selector_mul_ungated' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Row.Gating.residual_eq_selector_mul_ungated

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Row.Gating.ExactRowAction.residualAt_eq_selector_mul_ungated' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Row.Gating.ExactRowAction.residualAt_eq_selector_mul_ungated
