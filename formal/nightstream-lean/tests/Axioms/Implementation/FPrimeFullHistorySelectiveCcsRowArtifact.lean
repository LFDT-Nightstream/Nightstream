import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.Artifact.Row
import tests.Axioms.Support

/-!
Fail-closed dependency gate for coefficient-based classification of one
materialized selective-CCS row.
-/

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Row.Boolean.rowPoint_eq_booleanPoint' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Row.Boolean.rowPoint_eq_booleanPoint

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Row.Boolean.residual_eq_booleanResidual' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Row.Boolean.residual_eq_booleanResidual
