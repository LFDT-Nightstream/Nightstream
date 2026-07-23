import Nightstream.SuperNeo.Folding.PiRLC.PaperForkExtraction
import tests.Axioms.Support

/-!
Fail-closed trusted-dependency gate for operational paper `Pi_RLC`
coordinate-fork extraction.
-/

/-- info: 'Nightstream.SuperNeo.Folding.PiRLC.PaperForkExtraction.completeFork_implies_correctedAmbientHolds' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiRLC.PaperForkExtraction.completeFork_implies_correctedAmbientHolds
