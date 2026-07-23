import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ProtocolPolynomialHonestProver
import tests.Axioms.Support

/-! Fail-closed dependency gate for interactive paper `Pi_CCS` completeness. -/

open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ProtocolPolynomialHonestProver

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ProtocolPolynomialHonestProver.sequentialRun_eq_runRaw' does not depend on any axioms -/
#guard_msgs in
#audit_axioms sequentialRun_eq_runRaw

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ProtocolPolynomialHonestProver.exists_honest_accepted' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms exists_honest_accepted
