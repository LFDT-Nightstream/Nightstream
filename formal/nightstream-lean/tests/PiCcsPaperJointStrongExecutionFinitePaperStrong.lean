import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.FinitePaperStrong

/-!
Focused interface regression for the finite operational `Pi_CCS` strong game.
-/

namespace tests.PiCcsPaperJointStrongExecutionFinitePaperStrong

open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.FinitePaperStrong

#check PerfectComplete
#check perfectComplete
#check PublicCoin
#check publicCoin
#check SuccessGatedWorkBound
#check successGatedWorkBound
#check Extractor.successGated
#check successGatedSourceExtractionProbability
#check successGatedSourceExtractionProbability_eq_of_nonempty
#check outputPhiMismatchProbability_eq_zero
#check successGatedFiniteStrongGame
#check NamedSecurityContracts
#check finitePaperStrong
#check @finitePaperStrong

-- Legacy floor-based comparison theorem. It is not the paper-facing result.
#check legacyRejectionAdjustedFinitePaperStrong

end tests.PiCcsPaperJointStrongExecutionFinitePaperStrong
