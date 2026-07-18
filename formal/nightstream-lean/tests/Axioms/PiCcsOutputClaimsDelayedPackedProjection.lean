import Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism.Authority.DelayedPackedProjection
import Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism.Authority.DelayedPackedProjection.LimbDecomposition
import tests.Axioms.Support

/-! Fail-closed dependency gate for delayed packed-sidecar authority. -/

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism.Authority.DelayedPackedProjection.LimbDecomposition.projectedValue_eq_limbEvaluations' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism.Authority.DelayedPackedProjection.LimbDecomposition.projectedValue_eq_limbEvaluations

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism.Authority.DelayedPackedProjection.pairAccepted_of_scalar_matches' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism.Authority.DelayedPackedProjection.pairAccepted_of_scalar_matches

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism.Authority.DelayedPackedProjection.pairAccepted_implies_exact_or_badRoot' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism.Authority.DelayedPackedProjection.pairAccepted_implies_exact_or_badRoot
