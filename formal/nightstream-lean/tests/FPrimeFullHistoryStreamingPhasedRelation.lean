import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingPhasedRelation

/-! Focused surface for the exact 400-arm phased F-prime model. -/

namespace Tests.FPrimeFullHistoryStreamingPhasedRelation

open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPhasedRelation

#check workArm_count
#check lifecycleGroup_base
#check lifecycleGroup_bootstrap
#check lifecycleGroup_steady
#check lifecycleCircuit_base
#check lifecycleCircuit_recursive
#check phaseKind
#check exists_phaseAtArm_iff_step
#check exactRefinement
#check exists_linkedAccepts_iff_armSemantics
#check linkedAccepts_implies_step
#check terminal_complete_steps_exact

end Tests.FPrimeFullHistoryStreamingPhasedRelation
