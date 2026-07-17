import Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism.PackedBlockAction
import tests.Axioms.Support

/-! Fail-closed dependency gate for packed block-projection action compatibility. -/

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism.PackedBlockAction.blockRows_act' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism.PackedBlockAction.blockRows_act

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism.PackedBlockAction.packedYZcol_ringAction' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism.PackedBlockAction.packedYZcol_ringAction
