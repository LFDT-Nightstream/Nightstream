import Nightstream.SuperNeo.Folding.PiRLC.PaperForkCollision
import tests.Axioms.Support

/-!
Fail-closed trusted-dependency gate for the direct paper `Pi_RLC`
relaxed-binding collision reduction.
-/

/-- info: 'Nightstream.SuperNeo.Folding.PiRLC.PaperForkCollision.samePhi_differingExtractions_imply_relaxedBindingCollision' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiRLC.PaperForkCollision.samePhi_differingExtractions_imply_relaxedBindingCollision
