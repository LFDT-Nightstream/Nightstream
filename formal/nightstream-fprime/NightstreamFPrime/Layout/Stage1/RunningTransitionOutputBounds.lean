import NightstreamFPrime.Layout.Stage1.RunningTransitionData

/-! Owns bounds for the pilot output-preimage running words. -/

namespace NightstreamFPrime.Layout.Stage1.RunningTransitionInputs

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.Stage1

theorem outputWordBelowOutputDigestStart (index : RunningTransition.WordIndex) :
    (outputWord index).VarsBelow PilotProduction.outputDigestStart := by
  have indexBound : index.val < 27794 := index.isLt
  simp only [outputWord, outputBase, Expr.VarsBelow]
  norm_num [PilotProduction.outputDigestStart, PilotProduction.outputPreimageStart,
    PilotProduction.priorPublicInputStart, PilotProduction.priorPreimageStart,
    PilotProduction.stateHashWords_eq, PriorStateHash.publicWidth_eq,
    PiCCSInputs.priorRunningStart]
  omega

theorem outputWordBelow (index : RunningTransition.WordIndex) :
    (outputWord index).VarsBelow phaseOffset := by
  apply Expr.VarsBelow.mono _ (outputWordBelowOutputDigestStart index)
  apply Nat.le_trans (m := PiDECInputs.phaseOffset) ?_ piDecPhaseOffset_le
  norm_num [PilotProduction.outputDigestStart,
    PilotProduction.outputPreimageStart,
    PilotProduction.priorPublicInputStart,
    PilotProduction.priorPreimageStart, PilotProduction.stateHashWords_eq,
    PriorStateHash.publicWidth_eq, PiDECInputs.phaseOffset,
    PiDECInputs.proofInputStart, PiDECInputs.proofInputColumnCount_eq,
    PiRLCStarts.finalBoundaries_eq.2]

end NightstreamFPrime.Layout.Stage1.RunningTransitionInputs
