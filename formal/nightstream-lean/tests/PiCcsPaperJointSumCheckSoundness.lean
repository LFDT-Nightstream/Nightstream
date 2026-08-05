import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.SumCheckSoundness

/-!
Focused theorem-surface checks for concrete paper-joint SumCheck soundness.

The implementation theorem derives the repository security contract from the
exact paper degree width, finite root counting, causal challenge sampling, and
the literal operational experiment. Fiat--Shamir remains outside this module.
-/

set_option autoImplicit false

namespace tests.PiCcsPaperJointSumCheckSoundness

open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.SumCheckSoundness

#check FiniteRootCounting.roots_count_le_degree
#check FiniteRootCounting.collisions_count_le_degree
#check CausalSumCheckBound.probability_detects_le_ratio
#check sumCheckFailure_implies_detects
#check verifierDetects_probability_le
#check sumCheckBadChallenge_probability_le
#check sumCheckSoundnessContract_of_rootCounting
#check extraction_after_first_success_of_rootCounting
#check extraction_after_success_gate_of_rootCounting

end tests.PiCcsPaperJointSumCheckSoundness
