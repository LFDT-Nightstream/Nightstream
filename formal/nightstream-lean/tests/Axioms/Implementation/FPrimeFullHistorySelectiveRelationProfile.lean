import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.RelationProfile
import tests.Axioms.Support

/-!
Fail-closed dependency guards for the active selective relation profile.

| Guard | Audited boundary |
|---|---|
| Matrix-count mismatch | independent thirteen-port semantics cannot be relabeled as three rows |
| Matrix padding | finite matrices embed into the typed Boolean-row structure |
| Polynomial authority | the typed structure uses the independent selective polynomial |
-/

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.RelationProfile.Profile.shape_matrixCount_ne_three' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.RelationProfile.Profile.shape_matrixCount_ne_three

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.RelationProfile.FiniteRelation.toStructure_matrix' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.RelationProfile.FiniteRelation.toStructure_matrix

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.RelationProfile.FiniteRelation.toStructure_roleMatrix' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.RelationProfile.FiniteRelation.toStructure_roleMatrix

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.RelationProfile.FiniteRelation.toStructure_constraintPolynomial' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.RelationProfile.FiniteRelation.toStructure_constraintPolynomial
