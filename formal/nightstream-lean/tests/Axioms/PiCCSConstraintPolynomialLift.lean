import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConstraintPolynomialLift.Evaluation
import tests.Axioms.Support

/-!
Fail-closed trusted-dependency gate for sparse CCS carrier lifting.

Owns: dependency expectations for syntax-degree preservation and evaluation
commutation under the minimal zero/one/add/mul lift contract.

Does not own: a concrete carrier embedding, norm semantics, SplitNC,
SumCheck, Rust, R1CS, or constraint counts.

| Audited theorem | Model-level guarantee |
|---|---|
| `liftMonomial_totalDegree` | coefficient lifting cannot alter sparse degree |
| `evaluateMonomial_lift` | lifted monomial evaluation equals lifted base evaluation |
| `evaluatePolynomial_lift` | lifted sparse polynomial evaluation equals lifted base evaluation |
-/

open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConstraintPolynomialLift
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConstraintPolynomialLift.Evaluation

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConstraintPolynomialLift.liftMonomial_totalDegree' does not depend on any axioms -/
#guard_msgs in
#audit_axioms liftMonomial_totalDegree

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConstraintPolynomialLift.Evaluation.evaluateMonomial_lift' does not depend on any axioms -/
#guard_msgs in
#audit_axioms evaluateMonomial_lift

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConstraintPolynomialLift.Evaluation.evaluatePolynomial_lift' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms evaluatePolynomial_lift
