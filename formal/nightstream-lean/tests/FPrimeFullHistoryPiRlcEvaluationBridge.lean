import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.NifsPaper.PiRlc.EvaluationBridge

/-!
Focused surface checks for typed production `Pi_RLC` evaluations.

Assurance tier: model-level compile-time surface checks.
-/

namespace tests.FPrimeFullHistoryPiRlcEvaluationBridge

open Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.EvaluationBridge

#check pairRingF
#check pairRings
#check decodeYRingRings
#check pairRingF_action
#check pairRings_phi81Combine
#check decodeYRingRings_size
#check decodeYRingRings_getD
#check decodeYRingRings_phi81Combine
#check typedEvaluationEquation_of_refinement

end tests.FPrimeFullHistoryPiRlcEvaluationBridge
