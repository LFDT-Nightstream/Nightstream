import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ProtocolPolynomialDegree
import tests.Axioms.Support

/-!
Fail-closed dependency gate for the paper-level `Pi_CCS` protocol-polynomial
degree theorem. These are symbolic, artifact-independent degree proofs.
-/

open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ProtocolPolynomialDegree

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ProtocolPolynomialDegree.sequentialRoundRepresentable' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms sequentialRoundRepresentable

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ProtocolPolynomialDegree.expectedRoundsRepresentable' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms expectedRoundsRepresentable
