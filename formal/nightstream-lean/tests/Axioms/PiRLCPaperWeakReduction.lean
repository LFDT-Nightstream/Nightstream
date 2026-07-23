import Nightstream.SuperNeo.Folding.PiRLC.PaperWeakReduction
import tests.Axioms.Support

/-!
Fail-closed trusted-dependency gate for the operational paper `Pi_RLC` weak
reduction and its execution-dependent collision receipt.
-/

/-- info: 'Nightstream.SuperNeo.Folding.PiRLC.PaperForkCollision.coordinate_differingExtractions_imply_collisionReceipt' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiRLC.PaperForkCollision.coordinate_differingExtractions_imply_collisionReceipt

/-- info: 'Nightstream.SuperNeo.Folding.PiRLC.PaperWeakReduction.pairedDisagreement_implies_collisionReceipt' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiRLC.PaperWeakReduction.pairedDisagreement_implies_collisionReceipt

/-- info: 'Nightstream.SuperNeo.Folding.PiRLC.PaperWeakReduction.paperWeak' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiRLC.PaperWeakReduction.paperWeak
