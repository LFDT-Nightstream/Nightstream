import Nightstream.SuperNeo.Folding.PiCCS.PaperRectangular
import tests.Axioms.Support

/-!
Fail-closed dependency probes for the exact rectangular decomposition of the
paper's one-joint PiCCS polynomial.
-/

open Nightstream.SuperNeo.Folding.PiCCS.PaperRectangular

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperRectangular.Square.joint_qAt_eq_fe_add_nc' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Square.joint_qAt_eq_fe_add_nc

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperRectangular.Square.joint_summedQ_eq_summedFe_add_summedNc' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Square.joint_summedQ_eq_summedFe_add_summedNc

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperRectangular.Square.feInitial_eq_joint_target' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Square.feInitial_eq_joint_target

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperRectangular.Square.holds_implies_joint_claim' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Square.holds_implies_joint_claim
