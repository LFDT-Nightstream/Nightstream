import NightstreamFPrime.Export.Stage1.PiCCSInvocations

/-!
Owns the final schedule composition theorem for the four PiCCS transcript
packets. Executable traces and the four child schedule proofs remain in
`PiCCSInvocations`.
-/

namespace NightstreamFPrime.Export.Stage1.PiCCSInvocations

open NightstreamFPrime.Spec
open NightstreamFPrime.Gadgets.Poseidon2
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Export.Stage1.Invocations

/-- The one production PiCCS invocation list is the ordered composition of
the four transcript packets. The proof uses only child schedule and footprint
theorems; it does not inspect any child invocation. -/
theorem invocations_scheduleWithin (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth)
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    ScheduleWithin
        (NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan
          statementWitnessStart)
        invocationCeiling (invocations logicalWidth publicFits) ∧
      InvocationsBefore invocationCeiling
        (invocations logicalWidth publicFits) := by
  have statement := statementTrace_scheduleWithin logicalWidth publicFits
    relation
  have challenge := challengeTrace_scheduleWithin logicalWidth publicFits
    relation
  have rounds := roundTrace_scheduleWithin logicalWidth publicFits relation
  have output := outputTrace_scheduleWithin logicalWidth publicFits relation
  have statementLocal :
      NightstreamFPrime.Layout.Stage1.Spartan.piCcsPhaseOffset ≤
        statementWitnessStart := by
    unfold statementWitnessStart
    rw [NightstreamFPrime.Layout.Stage1.PiCCSStarts.statementWitnessStart_eq]
    norm_num [NightstreamFPrime.Layout.Stage1.Spartan.piCcsPhaseOffset]
  have challengeLocal :
      NightstreamFPrime.Layout.Stage1.Spartan.piCcsPhaseOffset ≤
        challengeWitnessStart := by
    unfold challengeWitnessStart
    rw [NightstreamFPrime.Layout.Stage1.PiCCSStarts.challengeWitnessStart_eq]
    norm_num [NightstreamFPrime.Layout.Stage1.Spartan.piCcsPhaseOffset]
  have roundLocal :
      NightstreamFPrime.Layout.Stage1.Spartan.piCcsPhaseOffset ≤
        roundWitnessStart := by
    unfold roundWitnessStart
    rw [NightstreamFPrime.Layout.Stage1.PiCCSStarts.roundTranscriptWitnessStart_eq]
    norm_num [NightstreamFPrime.Layout.Stage1.Spartan.piCcsPhaseOffset]
  have outputLocal :
      NightstreamFPrime.Layout.Stage1.Spartan.piCcsPhaseOffset ≤
        outputWitnessStart := by
    unfold outputWitnessStart
    rw [NightstreamFPrime.Layout.Stage1.PiCCSStarts.outputBindingWitnessStart_eq]
    norm_num [NightstreamFPrime.Layout.Stage1.Spartan.piCcsPhaseOffset]
  have statementStrict : statementWitnessStart < challengeWitnessStart := by
    calc
      statementWitnessStart < statementWitnessStart +
          invocationCount (statementActions logicalWidth publicFits) * 592 := by
        rw [statementInvocationCount_eq]
        omega
      _ = challengeWitnessStart :=
        statementEnd_eq_challengeStart logicalWidth publicFits
  have challengeStrict : challengeWitnessStart < roundWitnessStart := by
    calc
      challengeWitnessStart < challengeWitnessStart +
          invocationCount (challengeActions logicalWidth publicFits) * 592 := by
        rw [challengeInvocationCount_eq]
        omega
      _ = roundWitnessStart :=
        challengeEnd_eq_roundStart logicalWidth publicFits
  have roundStrict : roundWitnessStart < outputWitnessStart := by
    calc
      roundWitnessStart < roundWitnessStart +
          invocationCount (roundActions logicalWidth publicFits) * 592 := by
        rw [roundInvocationCount_eq]
        omega
      _ < outputWitnessStart :=
        roundEnd_lt_outputStart logicalWidth publicFits
  have outputStrict : outputWitnessStart <
      NightstreamFPrime.Layout.Stage1.PiCCSStarts.logicalFreshBase := by
    calc
      outputWitnessStart < outputWitnessStart +
          invocationCount (outputActions logicalWidth publicFits) * 592 := by
        rw [outputInvocationCount_eq]
        omega
      _ = NightstreamFPrime.Layout.Stage1.PiCCSStarts.logicalFreshBase :=
        outputEnd_eq_logicalFreshBase logicalWidth publicFits
  have statementToChallenge :
      NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan
          statementWitnessStart ≤
        NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan
          challengeWitnessStart :=
    (NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan_lt_of_piCcsLocal
      statementWitnessStart challengeWitnessStart statementLocal
      statementStrict).le
  have challengeToRound :
      NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan
          challengeWitnessStart ≤
        NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan
          roundWitnessStart :=
    (NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan_lt_of_piCcsLocal
      challengeWitnessStart roundWitnessStart challengeLocal
      challengeStrict).le
  have roundToOutput :
      NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan
          roundWitnessStart ≤
        NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan
          outputWitnessStart :=
    (NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan_lt_of_piCcsLocal
      roundWitnessStart outputWitnessStart roundLocal roundStrict).le
  have outputToCeiling :
      NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan
          outputWitnessStart ≤ invocationCeiling := by
    exact (NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan_lt_of_piCcsLocal
      outputWitnessStart
      NightstreamFPrime.Layout.Stage1.PiCCSStarts.logicalFreshBase outputLocal
      outputStrict).le
  have statementChallengeSchedule := ScheduleWithin.append statement.1
    statement.2 statementToChallenge challenge.1
  have statementBeforeRound := InvocationsBefore.mono statement.2
    challengeToRound
  have statementChallengeBeforeRound := InvocationsBefore.append
    statementBeforeRound challenge.2
  have statementToRound := Nat.le_trans statementToChallenge challengeToRound
  have prefixRoundSchedule := ScheduleWithin.append
    statementChallengeSchedule statementChallengeBeforeRound statementToRound
    rounds.1
  have statementChallengeBeforeOutput := InvocationsBefore.mono
    statementChallengeBeforeRound roundToOutput
  have roundsBeforeOutput := InvocationsBefore.mono rounds.2
    ((NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan_lt_of_piCcsLocal
      _ outputWitnessStart (by omega)
      (roundEnd_lt_outputStart logicalWidth publicFits)).le)
  have prefixRoundBeforeOutput := InvocationsBefore.append
    statementChallengeBeforeOutput roundsBeforeOutput
  have statementToOutput := Nat.le_trans statementToRound roundToOutput
  have prefixOutputSchedule := ScheduleWithin.append prefixRoundSchedule
    prefixRoundBeforeOutput statementToOutput output.1
  have prefixRoundBeforeCeiling := InvocationsBefore.mono
    prefixRoundBeforeOutput outputToCeiling
  have allBefore := InvocationsBefore.append prefixRoundBeforeCeiling output.2
  constructor
  · simpa [invocations, List.append_assoc] using prefixOutputSchedule
  · simpa [invocations, List.append_assoc] using allBefore

end NightstreamFPrime.Export.Stage1.PiCCSInvocations
