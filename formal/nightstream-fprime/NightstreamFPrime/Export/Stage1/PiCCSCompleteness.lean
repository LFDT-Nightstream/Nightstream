import NightstreamFPrime.Export.Stage1.Package
import NightstreamFPrime.Lifecycle.PiCCS.v1_1.Completeness

/-!
Owns constructive completeness of the canonical Stage 1 PiCCS package rows.
It composes the existing compact transcript invocations and eight ordinary
packets. It adds no relation, circuit operation, or alternate package path.
-/

namespace NightstreamFPrime.Export.Stage1.PiCCSCompleteness

open NightstreamFPrime.Circuit
open NightstreamFPrime.Export.Package
open NightstreamFPrime.Layout
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.PiCCS.v1_1
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

def preOutputInvocations
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :
    List PermutationInvocation :=
  (PiCCSInvocations.statementTrace logicalWidth publicFits).invocations ++
    (PiCCSInvocations.challengeTrace logicalWidth publicFits).invocations ++
      (PiCCSInvocations.roundTrace logicalWidth publicFits).invocations

theorem roundEnd_eq_initialClaimLogicalStart
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :
    PiCCSInvocations.roundWitnessStart +
        Invocations.invocationCount
          (PiCCSInvocations.roundActions logicalWidth publicFits) * 592 =
      PiCCSArithmetic.initialClaimLogicalStart := by
  rw [PiCCSInvocations.roundInvocationCount_eq]
  unfold PiCCSInvocations.roundWitnessStart
    PiCCSArithmetic.initialClaimLogicalStart
  rw [NightstreamFPrime.Layout.Stage1.PiCCSStarts.roundTranscriptWitnessStart_eq]
  rfl

private theorem preOutput_schedule
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth)
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    Invocations.ScheduleWithin
      (NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan
        PiCCSInvocations.statementWitnessStart)
      PiCCSInvocations.invocationCeiling
      (preOutputInvocations logicalWidth publicFits) := by
  have statement := PiCCSInvocations.statementTrace_scheduleWithin
    logicalWidth publicFits relation
  have challenge := PiCCSInvocations.challengeTrace_scheduleWithin
    logicalWidth publicFits relation
  have rounds := PiCCSInvocations.roundTrace_scheduleWithin
    logicalWidth publicFits relation
  have statementLocal :
      NightstreamFPrime.Layout.Stage1.Spartan.piCcsPhaseOffset ≤
        PiCCSInvocations.statementWitnessStart := by
    unfold PiCCSInvocations.statementWitnessStart
    rw [NightstreamFPrime.Layout.Stage1.PiCCSStarts.statementWitnessStart_eq]
    norm_num [NightstreamFPrime.Layout.Stage1.Spartan.piCcsPhaseOffset]
  have challengeLocal :
      NightstreamFPrime.Layout.Stage1.Spartan.piCcsPhaseOffset ≤
        PiCCSInvocations.challengeWitnessStart := by
    unfold PiCCSInvocations.challengeWitnessStart
    rw [NightstreamFPrime.Layout.Stage1.PiCCSStarts.challengeWitnessStart_eq]
    norm_num [NightstreamFPrime.Layout.Stage1.Spartan.piCcsPhaseOffset]
  have statementStrict : PiCCSInvocations.statementWitnessStart <
      PiCCSInvocations.challengeWitnessStart := by
    calc
      PiCCSInvocations.statementWitnessStart <
          PiCCSInvocations.statementWitnessStart +
            Invocations.invocationCount
              (PiCCSInvocations.statementActions logicalWidth publicFits) *
                592 := by
          rw [PiCCSInvocations.statementInvocationCount_eq]
          omega
      _ = PiCCSInvocations.challengeWitnessStart :=
        PiCCSInvocations.statementEnd_eq_challengeStart logicalWidth publicFits
  have challengeStrict : PiCCSInvocations.challengeWitnessStart <
      PiCCSInvocations.roundWitnessStart := by
    calc
      PiCCSInvocations.challengeWitnessStart <
          PiCCSInvocations.challengeWitnessStart +
            Invocations.invocationCount
              (PiCCSInvocations.challengeActions logicalWidth publicFits) *
                592 := by
          rw [PiCCSInvocations.challengeInvocationCount_eq]
          omega
      _ = PiCCSInvocations.roundWitnessStart :=
        PiCCSInvocations.challengeEnd_eq_roundStart logicalWidth publicFits
  have statementToChallenge :=
    (NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan_lt_of_piCcsLocal
      PiCCSInvocations.statementWitnessStart
      PiCCSInvocations.challengeWitnessStart statementLocal
      statementStrict).le
  have challengeToRound :=
    (NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan_lt_of_piCcsLocal
      PiCCSInvocations.challengeWitnessStart
      PiCCSInvocations.roundWitnessStart challengeLocal challengeStrict).le
  have statementChallenge := Invocations.ScheduleWithin.append statement.1
    statement.2 statementToChallenge challenge.1
  have statementBeforeRound := Invocations.InvocationsBefore.mono statement.2
    challengeToRound
  have prefixBeforeRound := Invocations.InvocationsBefore.append
    statementBeforeRound challenge.2
  have statementToRound := Nat.le_trans statementToChallenge challengeToRound
  simpa [preOutputInvocations, List.append_assoc] using
    Invocations.ScheduleWithin.append statementChallenge prefixBeforeRound
      statementToRound rounds.1

theorem preOutput_invocationsBeforeArithmetic
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth)
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    Invocations.InvocationsBefore
      (NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan
        PiCCSArithmetic.initialClaimLogicalStart)
      (preOutputInvocations logicalWidth publicFits) := by
  have statement := PiCCSInvocations.statementTrace_scheduleWithin
    logicalWidth publicFits relation
  have challenge := PiCCSInvocations.challengeTrace_scheduleWithin
    logicalWidth publicFits relation
  have rounds := PiCCSInvocations.roundTrace_scheduleWithin
    logicalWidth publicFits relation
  have challengeLocal :
      NightstreamFPrime.Layout.Stage1.Spartan.piCcsPhaseOffset ≤
        PiCCSInvocations.challengeWitnessStart := by
    unfold PiCCSInvocations.challengeWitnessStart
    rw [NightstreamFPrime.Layout.Stage1.PiCCSStarts.challengeWitnessStart_eq]
    norm_num [NightstreamFPrime.Layout.Stage1.Spartan.piCcsPhaseOffset]
  have roundLocal :
      NightstreamFPrime.Layout.Stage1.Spartan.piCcsPhaseOffset ≤
        PiCCSInvocations.roundWitnessStart := by
    unfold PiCCSInvocations.roundWitnessStart
    rw [NightstreamFPrime.Layout.Stage1.PiCCSStarts.roundTranscriptWitnessStart_eq]
    norm_num [NightstreamFPrime.Layout.Stage1.Spartan.piCcsPhaseOffset]
  have challengeStrict : PiCCSInvocations.challengeWitnessStart <
      PiCCSInvocations.roundWitnessStart := by
    calc
      PiCCSInvocations.challengeWitnessStart <
          PiCCSInvocations.challengeWitnessStart +
            Invocations.invocationCount
              (PiCCSInvocations.challengeActions logicalWidth publicFits) *
                592 := by
          rw [PiCCSInvocations.challengeInvocationCount_eq]
          omega
      _ = PiCCSInvocations.roundWitnessStart :=
        PiCCSInvocations.challengeEnd_eq_roundStart logicalWidth publicFits
  have roundStrict : PiCCSInvocations.roundWitnessStart <
      PiCCSArithmetic.initialClaimLogicalStart := by
    rw [← roundEnd_eq_initialClaimLogicalStart logicalWidth publicFits]
    rw [PiCCSInvocations.roundInvocationCount_eq]
    omega
  have challengeToRound :=
    (NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan_lt_of_piCcsLocal
      PiCCSInvocations.challengeWitnessStart
      PiCCSInvocations.roundWitnessStart challengeLocal challengeStrict).le
  have roundToArithmetic :=
    (NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan_lt_of_piCcsLocal
      PiCCSInvocations.roundWitnessStart
      PiCCSArithmetic.initialClaimLogicalStart roundLocal roundStrict).le
  have statementBefore := Invocations.InvocationsBefore.mono statement.2
    (Nat.le_trans challengeToRound roundToArithmetic)
  have challengeBefore := Invocations.InvocationsBefore.mono challenge.2
    roundToArithmetic
  have roundsBefore : Invocations.InvocationsBefore
      (NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan
        PiCCSArithmetic.initialClaimLogicalStart)
      (PiCCSInvocations.roundTrace logicalWidth publicFits).invocations := by
    rw [← roundEnd_eq_initialClaimLogicalStart logicalWidth publicFits]
    exact rounds.2
  exact Invocations.InvocationsBefore.append
    (Invocations.InvocationsBefore.append statementBefore challengeBefore)
      roundsBefore

theorem complete_preOutputInvocations
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth)
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (env : Env) :
    ∃ completed,
      AgreesOutside env completed
          (NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan
            PiCCSInvocations.statementWitnessStart)
          (PiCCSInvocations.invocationCeiling -
            NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan
              PiCCSInvocations.statementWitnessStart) ∧
        Invocations.AgreesOutsideInvocations env completed
          (preOutputInvocations logicalWidth publicFits) ∧
        ∀ invocation ∈ preOutputInvocations logicalWidth publicFits,
          PermutationInvocationHolds (PilotData.circuitPackage ()) invocation
            completed := by
  exact Invocations.completeInvocations env
    (NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan
      PiCCSInvocations.statementWitnessStart)
    PiCCSInvocations.invocationCeiling
    (preOutputInvocations logicalWidth publicFits)
    (preOutput_schedule logicalWidth publicFits relation)

structure PreOutputTranscriptSpecs
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth)
    (env : Env) : Prop where
  statementAbsorption : StatementAbsorption.SpecHolds
    (PiCCSInvocations.statementInterface logicalWidth publicFits)
    PiCCSInvocations.statementWitnessStart
    (NightstreamFPrime.Layout.Stage1.Spartan.pullback env)
  challenge : ChallengeDerivation.SpecHolds
    (PiCCSInvocations.challengeInterface logicalWidth publicFits)
    PiCCSInvocations.challengeWitnessStart
    (NightstreamFPrime.Layout.Stage1.Spartan.pullback env)
  rounds : RoundTranscript.SpecHolds
    (PiCCSInvocations.roundInterface logicalWidth publicFits)
    PiCCSInvocations.roundWitnessStart
    (NightstreamFPrime.Layout.Stage1.Spartan.pullback env)

theorem preOutputInvocations_imply_specs
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth)
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (env : Env)
    (holds : ∀ invocation ∈ preOutputInvocations logicalWidth publicFits,
      PermutationInvocationHolds (PilotData.circuitPackage ()) invocation env) :
    PreOutputTranscriptSpecs logicalWidth publicFits env := by
  constructor
  · apply PiCCSInvocations.statementTrace_implies_spec logicalWidth publicFits
      relation env
    intro invocation member
    exact holds invocation (by
      simp [preOutputInvocations, member])
  · apply PiCCSInvocations.challengeTrace_implies_spec logicalWidth publicFits
      relation env
    intro invocation member
    exact holds invocation (by
      simp [preOutputInvocations, member])
  · apply PiCCSInvocations.roundTrace_implies_spec logicalWidth publicFits
      relation env
    intro invocation member
    exact holds invocation (by
      simp [preOutputInvocations, member])

theorem PreOutputTranscriptSpecs.statement_parent
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    {env : Env}
    (specs : PreOutputTranscriptSpecs logicalWidth publicFits env) :
    StatementAbsorption.SpecHolds
      (Formal.statementAbsorptionInterface
        (Formal.atOffset
          (PiCCSInvocations.parentInterface logicalWidth publicFits)
          NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset))
      (Formal.statementAbsorptionOffset
        (PiCCSInvocations.parentInterface logicalWidth publicFits)
        NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset)
      (NightstreamFPrime.Layout.Stage1.Spartan.pullback env) := by
  change StatementAbsorption.SpecHolds
    (PiCCSInvocations.statementInterface logicalWidth publicFits)
    (Formal.statementAbsorptionOffset
      (PiCCSInvocations.parentInterface logicalWidth publicFits)
      NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset)
    (NightstreamFPrime.Layout.Stage1.Spartan.pullback env)
  rw [← PiCCSInvocations.statementWitnessStart_matches]
  exact specs.statementAbsorption

theorem PreOutputTranscriptSpecs.challenge_parent
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    {env : Env}
    (specs : PreOutputTranscriptSpecs logicalWidth publicFits env) :
    ChallengeDerivation.SpecHolds
      (Formal.challengeInterface
        (PiCCSInvocations.parentInterface logicalWidth publicFits)
        NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset)
      (Formal.challengeOffset
        (PiCCSInvocations.parentInterface logicalWidth publicFits)
        NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset)
      (NightstreamFPrime.Layout.Stage1.Spartan.pullback env) := by
  change ChallengeDerivation.SpecHolds
    (PiCCSInvocations.challengeInterface logicalWidth publicFits)
    (Formal.challengeOffset
      (PiCCSInvocations.parentInterface logicalWidth publicFits)
      NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset)
    (NightstreamFPrime.Layout.Stage1.Spartan.pullback env)
  rw [← PiCCSInvocations.challengeWitnessStart_matches]
  exact specs.challenge

theorem PreOutputTranscriptSpecs.rounds_parent
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    {env : Env}
    (specs : PreOutputTranscriptSpecs logicalWidth publicFits env) :
    RoundTranscript.SpecHolds
      (Formal.roundTranscriptInterface
        (Formal.atOffset
          (PiCCSInvocations.parentInterface logicalWidth publicFits)
          NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset))
      (Formal.roundTranscriptOffset
        (PiCCSInvocations.parentInterface logicalWidth publicFits)
        NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset)
      (NightstreamFPrime.Layout.Stage1.Spartan.pullback env) := by
  change RoundTranscript.SpecHolds
    (PiCCSInvocations.roundInterface logicalWidth publicFits)
    (Formal.roundTranscriptOffset
      (PiCCSInvocations.parentInterface logicalWidth publicFits)
      NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset)
    (NightstreamFPrime.Layout.Stage1.Spartan.pullback env)
  rw [← PiCCSInvocations.roundWitnessStart_matches]
  exact specs.rounds

def arithmeticLogicalOps
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    List Op :=
  Formal.CompletenessSupport.evaluationPrefixOps
      (PiCCSInvocations.parentInterface logicalWidth publicFits)
      NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset ++
    Formal.CompletenessSupport.preOutputOps relation
      (PiCCSInvocations.parentInterface logicalWidth publicFits)
      NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset

theorem complete_arithmeticLogical
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (template : Proof (ProductionKey.degreeBound relation))
    (env : Env)
    (accepted : NightstreamFPrime.Spec.Folding.PiCCS.Accepted
      (ProductionKey.key relation ajtai)
      (Formal.evalRunning
        (PiCCSInvocations.parentInterface logicalWidth publicFits)
        NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset
        (NightstreamFPrime.Layout.Stage1.Spartan.pullback env))
      (Formal.evalFresh
        (PiCCSInvocations.parentInterface logicalWidth publicFits)
        NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset
        (NightstreamFPrime.Layout.Stage1.Spartan.pullback env))
      (Formal.evalProof relation
        (PiCCSInvocations.parentInterface logicalWidth publicFits)
        NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset
        (NightstreamFPrime.Layout.Stage1.Spartan.pullback env) template))
    (transcript : PreOutputTranscriptSpecs logicalWidth publicFits env) :
    ∃ completed : Sequence.Prefix
        (NightstreamFPrime.Layout.Stage1.Spartan.pullback env)
        PiCCSArithmetic.initialClaimLogicalStart,
      completed.operations = arithmeticLogicalOps relation ∧
        PiCCSArithmetic.initialClaimLogicalStart +
            localLength completed.operations =
          PiCCSInvocations.outputWitnessStart := by
  let parent := PiCCSInvocations.parentInterface logicalWidth publicFits
  let source := NightstreamFPrime.Layout.Stage1.Spartan.pullback env
  let assumptions := NightstreamFPrime.Layout.PiCCS.v1_1.Assumptions.production
    relation parent NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset
    (NightstreamFPrime.Layout.Stage1.PiCCSInputs.externalInputsLinear
      logicalWidth publicFits) source
  let p0 := Sequence.empty source PiCCSArithmetic.initialClaimLogicalStart
  have offsetLeBase :
      NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset ≤
        PiCCSArithmetic.initialClaimLogicalStart := by
    unfold PiCCSArithmetic.initialClaimLogicalStart
      NightstreamFPrime.Layout.Stage1.PiCCSStarts.initialClaimLogicalStart
      NightstreamFPrime.Layout.Stage1.PiCCSStarts.roundTranscriptWitnessStart
      NightstreamFPrime.Layout.Stage1.PiCCSStarts.challengeWitnessStart
      NightstreamFPrime.Layout.Stage1.PiCCSStarts.statementWitnessStart
    omega
  have initialStart : PiCCSArithmetic.initialClaimLogicalStart +
      localLength p0.operations =
        Formal.initialClaimOffset parent
          NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset := by
    change PiCCSArithmetic.initialClaimLogicalStart = _
    exact PiCCSArithmetic.initialClaimLogicalStart_matches logicalWidth
      publicFits
  rcases Formal.CompletenessSupport.completeEvaluationPrefix relation ajtai
      parent source NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset
      template assumptions accepted p0 offsetLeBase
      transcript.statement_parent transcript.challenge_parent
      transcript.rounds_parent initialStart with
    ⟨p8, operations8, end8, _p0to8, evidence8⟩
  rcases Formal.CompletenessSupport.completePreOutputPrefix relation ajtai
      parent source NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset
      template assumptions p8 offsetLeBase evidence8 end8 with
    ⟨p11, operations11, end11, _p8to11⟩
  refine ⟨p11, ?_, ?_⟩
  · rw [operations11, operations8]
    simp [p0, Sequence.empty, arithmeticLogicalOps, parent]
    rfl
  · rw [PiCCSInvocations.outputWitnessStart_matches logicalWidth publicFits
      relation]
    exact end11

def arithmeticLogicalLength : Nat :=
  PiCCSInvocations.outputWitnessStart -
    PiCCSArithmetic.initialClaimLogicalStart

theorem arithmeticLogicalEnd_eq :
    PiCCSArithmetic.initialClaimLogicalStart + arithmeticLogicalLength =
      PiCCSInvocations.outputWitnessStart := by
  unfold arithmeticLogicalLength PiCCSInvocations.outputWitnessStart
    PiCCSArithmetic.initialClaimLogicalStart
    NightstreamFPrime.Layout.Stage1.PiCCSStarts.outputBindingWitnessStart
    NightstreamFPrime.Layout.Stage1.PiCCSStarts.finalIdentityLogicalStart
    NightstreamFPrime.Layout.Stage1.PiCCSStarts.normLogicalStart
    NightstreamFPrime.Layout.Stage1.PiCCSStarts.ccsLogicalStart
    NightstreamFPrime.Layout.Stage1.PiCCSStarts.evalALogicalStart
    NightstreamFPrime.Layout.Stage1.PiCCSStarts.evalKLogicalStart
    NightstreamFPrime.Layout.Stage1.PiCCSStarts.sumcheckLogicalStart
  omega

private theorem arithmeticStartLocal :
    NightstreamFPrime.Layout.Stage1.Spartan.piCcsPhaseOffset ≤
      PiCCSArithmetic.initialClaimLogicalStart := by
  unfold PiCCSArithmetic.initialClaimLogicalStart
    NightstreamFPrime.Layout.Stage1.PiCCSStarts.initialClaimLogicalStart
    NightstreamFPrime.Layout.Stage1.PiCCSStarts.roundTranscriptWitnessStart
    NightstreamFPrime.Layout.Stage1.PiCCSStarts.challengeWitnessStart
    NightstreamFPrime.Layout.Stage1.PiCCSStarts.statementWitnessStart
  rw [NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset_eq]
  norm_num [NightstreamFPrime.Layout.Stage1.Spartan.piCcsPhaseOffset]

theorem arithmeticMappedEnd_eq :
    NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan
          PiCCSArithmetic.initialClaimLogicalStart + arithmeticLogicalLength =
      NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan
        PiCCSInvocations.outputWitnessStart := by
  have mapped :=
    NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan_add_of_piCcsLocal
      PiCCSArithmetic.initialClaimLogicalStart arithmeticLogicalLength
      arithmeticStartLocal
  rw [arithmeticLogicalEnd_eq] at mapped
  exact mapped.symm

theorem arithmeticMappedEnd_le_invocationCeiling :
    NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan
          PiCCSArithmetic.initialClaimLogicalStart + arithmeticLogicalLength ≤
      PiCCSInvocations.invocationCeiling := by
  rw [arithmeticMappedEnd_eq]
  exact
    (NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan_lt_of_piCcsLocal
      PiCCSInvocations.outputWitnessStart
      NightstreamFPrime.Layout.Stage1.PiCCSStarts.logicalFreshBase (by
        unfold PiCCSInvocations.outputWitnessStart
        rw [NightstreamFPrime.Layout.Stage1.PiCCSStarts.outputBindingWitnessStart_eq]
        norm_num [NightstreamFPrime.Layout.Stage1.Spartan.piCcsPhaseOffset]) (by
          unfold PiCCSInvocations.outputWitnessStart
          rw [NightstreamFPrime.Layout.Stage1.PiCCSStarts.outputBindingWitnessStart_eq]
          norm_num [NightstreamFPrime.Layout.Stage1.PiCCSStarts.logicalFreshBase,
            NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset_eq])).le

private theorem schedule_inputsOutside
    {bound ceiling : Nat} {invocations : List PermutationInvocation}
    (schedule : Invocations.ScheduleWithin bound ceiling invocations) :
    ∀ invocation ∈ invocations,
      Invocations.InvocationInputsOutside ceiling invocation := by
  induction invocations generalizing bound with
  | nil => simp
  | cons head rest inductionHypothesis =>
      intro invocation member
      rcases schedule with
        ⟨_starts, _ends, inputs, _stableInputs, restSchedule⟩
      rcases List.mem_cons.mp member with rfl | member
      · exact inputs
      · exact inductionHypothesis restSchedule invocation member

theorem schedule_stableInputs
    {bound ceiling : Nat} {invocations : List PermutationInvocation}
    (schedule : Invocations.ScheduleWithin bound ceiling invocations) :
    ∀ invocation ∈ invocations,
      Invocations.InvocationInputsOutside
        NightstreamFPrime.Layout.Stage1.Spartan.privateColumnCount
        invocation := by
  induction invocations generalizing bound with
  | nil => simp
  | cons head rest inductionHypothesis =>
      intro invocation member
      rcases schedule with
        ⟨_starts, _ends, _inputs, stableInputs, restSchedule⟩
      rcases List.mem_cons.mp member with rfl | member
      · exact stableInputs
      · exact inductionHypothesis restSchedule invocation member

theorem copyArithmetic_pullback_eq
    (base : Env)
    (completed : Sequence.Prefix
      (NightstreamFPrime.Layout.Stage1.Spartan.pullback base)
      PiCCSArithmetic.initialClaimLogicalStart)
    (endEq : PiCCSArithmetic.initialClaimLogicalStart +
        localLength completed.operations = PiCCSInvocations.outputWitnessStart) :
    NightstreamFPrime.Layout.Stage1.Spartan.pullback
        (NightstreamFPrime.Layout.Stage1.Spartan.copyMappedInterval base
          completed.current PiCCSArithmetic.initialClaimLogicalStart
          arithmeticLogicalLength) =
      completed.current := by
  apply NightstreamFPrime.Layout.Stage1.Spartan.pullback_copyMappedInterval_eq
  · exact arithmeticStartLocal
  · exact Nat.le_trans arithmeticMappedEnd_le_invocationCeiling
      PiCCSInvocations.invocationCeiling_le_private
  · have agrees := completed.agrees
    have lengthEq : localLength completed.operations =
        arithmeticLogicalLength := by
      have canonicalEnd := arithmeticLogicalEnd_eq
      omega
    rw [lengthEq] at agrees
    exact agrees

theorem preOutputHolds_after_copyArithmetic
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (before : Env)
    (source : Env)
    (holds : ∀ invocation ∈ preOutputInvocations logicalWidth publicFits,
      PermutationInvocationHolds (PilotData.circuitPackage ()) invocation
        before) :
    ∀ invocation ∈ preOutputInvocations logicalWidth publicFits,
      PermutationInvocationHolds (PilotData.circuitPackage ()) invocation
        (NightstreamFPrime.Layout.Stage1.Spartan.copyMappedInterval before
          source PiCCSArithmetic.initialClaimLogicalStart
          arithmeticLogicalLength) := by
  intro invocation member
  have beforeArithmetic := preOutput_invocationsBeforeArithmetic logicalWidth
    publicFits relation invocation member
  have inputs := schedule_inputsOutside
    (preOutput_schedule logicalWidth publicFits relation) invocation member
  apply NightstreamFPrime.Export.Pilot.permutationInvocationHolds_of_agreesOutside
    invocation before
      (NightstreamFPrime.Layout.Stage1.Spartan.copyMappedInterval before source
        PiCCSArithmetic.initialClaimLogicalStart arithmeticLogicalLength)
      (NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan
        PiCCSArithmetic.initialClaimLogicalStart)
      arithmeticLogicalLength
  · intro lane term termMember
    rcases inputs lane term termMember with inputBefore | inputAfter
    · exact Or.inl (lt_of_lt_of_le inputBefore (by omega))
    · exact Or.inr
        (Nat.le_trans arithmeticMappedEnd_le_invocationCeiling inputAfter)
  · intro index below
    exact Or.inl (lt_of_lt_of_le (by omega) beforeArithmetic)
  · exact
      NightstreamFPrime.Layout.Stage1.Spartan.copyMappedInterval_agreesOutside
        before source PiCCSArithmetic.initialClaimLogicalStart
        arithmeticLogicalLength
  · exact holds invocation member

theorem complete_outputInvocations
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (env : Env) :
    ∃ completed,
      AgreesOutside env completed
          (NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan
            PiCCSInvocations.outputWitnessStart)
          (PiCCSInvocations.invocationCeiling -
            NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan
              PiCCSInvocations.outputWitnessStart) ∧
        Invocations.AgreesOutsideInvocations env completed
          (PiCCSInvocations.outputTrace logicalWidth publicFits).invocations ∧
        ∀ invocation ∈
            (PiCCSInvocations.outputTrace logicalWidth publicFits).invocations,
          PermutationInvocationHolds (PilotData.circuitPackage ()) invocation
            completed := by
  exact Invocations.completeInvocations env
    (NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan
      PiCCSInvocations.outputWitnessStart)
    PiCCSInvocations.invocationCeiling
    (PiCCSInvocations.outputTrace logicalWidth publicFits).invocations
    (PiCCSInvocations.outputTrace_scheduleWithin logicalWidth publicFits
      relation).1

theorem outputIntervalEnd_eq :
    NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan
          PiCCSInvocations.outputWitnessStart +
        (PiCCSInvocations.invocationCeiling -
          NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan
            PiCCSInvocations.outputWitnessStart) =
      PiCCSInvocations.invocationCeiling := by
  have outputStartLe :
      NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan
          PiCCSInvocations.outputWitnessStart ≤
        PiCCSInvocations.invocationCeiling := by
    rw [← arithmeticMappedEnd_eq]
    exact arithmeticMappedEnd_le_invocationCeiling
  omega

theorem pullback_after_output_agreesBelow
    (before after : Env)
    (agrees : AgreesOutside before after
      (NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan
        PiCCSInvocations.outputWitnessStart)
      (PiCCSInvocations.invocationCeiling -
        NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan
          PiCCSInvocations.outputWitnessStart)) :
    ∀ column, column < PiCCSInvocations.outputWitnessStart →
      NightstreamFPrime.Layout.Stage1.Spartan.pullback after column =
        NightstreamFPrime.Layout.Stage1.Spartan.pullback before column := by
  intro column below
  unfold NightstreamFPrime.Layout.Stage1.Spartan.pullback
  apply agrees
  rcases
      NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan_before_piCcsLocal
        column PiCCSInvocations.outputWitnessStart (by
          unfold PiCCSInvocations.outputWitnessStart
          rw [NightstreamFPrime.Layout.Stage1.PiCCSStarts.outputBindingWitnessStart_eq]
          norm_num [NightstreamFPrime.Layout.Stage1.Spartan.piCcsPhaseOffset])
        below with mappedBefore | mappedPublic
  · exact Or.inl mappedBefore
  · exact Or.inr (by
      rw [outputIntervalEnd_eq]
      exact Nat.le_trans PiCCSInvocations.invocationCeiling_le_private
        mappedPublic.le)

theorem preOutputHolds_after_output
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (before after : Env)
    (agrees : AgreesOutside before after
      (NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan
        PiCCSInvocations.outputWitnessStart)
      (PiCCSInvocations.invocationCeiling -
        NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan
          PiCCSInvocations.outputWitnessStart))
    (holds : ∀ invocation ∈ preOutputInvocations logicalWidth publicFits,
      PermutationInvocationHolds (PilotData.circuitPackage ()) invocation
        before) :
    ∀ invocation ∈ preOutputInvocations logicalWidth publicFits,
      PermutationInvocationHolds (PilotData.circuitPackage ()) invocation
        after := by
  intro invocation member
  have beforeArithmetic := preOutput_invocationsBeforeArithmetic logicalWidth
    publicFits relation invocation member
  have inputs := schedule_inputsOutside
    (preOutput_schedule logicalWidth publicFits relation) invocation member
  apply NightstreamFPrime.Export.Pilot.permutationInvocationHolds_of_agreesOutside
    invocation before after
      (NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan
        PiCCSInvocations.outputWitnessStart)
      (PiCCSInvocations.invocationCeiling -
        NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan
          PiCCSInvocations.outputWitnessStart)
  · intro lane term termMember
    rcases inputs lane term termMember with inputBefore | inputAfter
    · exact Or.inl (lt_of_lt_of_le inputBefore (by
        calc
          invocation.witnessStart ≤ invocation.witnessStart + 592 := by omega
          _ ≤ NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan
              PiCCSArithmetic.initialClaimLogicalStart := beforeArithmetic
          _ ≤ NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan
              PiCCSInvocations.outputWitnessStart := by
            rw [← arithmeticMappedEnd_eq]
            omega))
    · exact Or.inr (by rw [outputIntervalEnd_eq]; exact inputAfter)
  · intro index below
    apply Or.inl
    calc
      invocation.witnessStart + index < invocation.witnessStart + 592 := by
        omega
      _ ≤ NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan
          PiCCSInvocations.outputWitnessStart :=
        Nat.le_trans beforeArithmetic (by
          rw [← arithmeticMappedEnd_eq]
          omega)
  · exact agrees
  · exact holds invocation member

def packetConstraints
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :
    List Expr :=
  PiCCSArithmetic.initialClaimConstraints logicalWidth publicFits ++
    (PiCCSArithmetic.sumcheckConstraints logicalWidth publicFits ++
      (PiCCSArithmetic.evalKConstraints logicalWidth publicFits ++
        (PiCCSArithmetic.evalAConstraints logicalWidth publicFits ++
          (PiCCSArithmetic.ccsConstraints logicalWidth publicFits ++
            (PiCCSArithmetic.normConstraints logicalWidth publicFits ++
              PiCCSArithmetic.finalIdentityConstraints logicalWidth
                publicFits)))))

/-- Exact ordinary-row constraint order in the emitted package. -/
def emittedConstraints
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :
    List Expr :=
  PiCCSArithmetic.statementBindingConstraints logicalWidth publicFits ++
    packetConstraints logicalWidth publicFits

theorem statementBindingConstraints_hold
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (env : Env)
    (specification : StateBinding.SpecHolds
      (Formal.statementBindingInterface
        (PiCCSArithmetic.sharedInterface logicalWidth publicFits)).state
      NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset env) :
    ConstraintsHold env
      (PiCCSArithmetic.statementBindingConstraints logicalWidth
        publicFits) := by
  unfold PiCCSArithmetic.statementBindingConstraints
    PiCCSArithmetic.statementBindingLogicalStart
    NightstreamFPrime.Layout.PiCCS.v1_1.childConstraints
    Formal.statementBindingCircuit
  rw [FormalCircuit.withConstantFootprint_main]
  exact StatementBinding.constraintsHold_of_spec
    (Formal.statementBindingInterface
      (PiCCSArithmetic.sharedInterface logicalWidth publicFits))
    env NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset
    ⟨specification, fun _ => rfl, fun _ => rfl, fun _ => rfl⟩

theorem statementBindingConstraints_varsBelow
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (env : Env)
    (assumptions : StateBinding.Assumptions
      (Formal.statementBindingInterface
        (PiCCSArithmetic.sharedInterface logicalWidth publicFits)).state
      NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset env) :
    ∀ expression ∈
      PiCCSArithmetic.statementBindingConstraints logicalWidth publicFits,
      expression.VarsBelow
        NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset := by
  intro expression member
  have scope := StatementBinding.flatConstraints_varsBelow
    (Formal.statementBindingInterface
      (PiCCSArithmetic.sharedInterface logicalWidth publicFits))
    NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset env assumptions
  have below := scope expression (by
    simpa [PiCCSArithmetic.statementBindingConstraints,
      PiCCSArithmetic.statementBindingLogicalStart,
      NightstreamFPrime.Layout.PiCCS.v1_1.childConstraints,
      Formal.statementBindingCircuit] using member)
  rw [StatementBinding.localLength_eq, Nat.add_zero] at below
  exact below

private theorem childOp_flatConstraints (name : String)
    (child : FormalCircuit) (offset : Nat) :
    (Formal.childOp name child offset).flatConstraints =
      NightstreamFPrime.Layout.PiCCS.v1_1.childConstraints child offset := by
  rfl

theorem initialClaimConstraints_eq
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :
    NightstreamFPrime.Layout.PiCCS.v1_1.childConstraints
        (Formal.initialClaimCircuit
          (Formal.atOffset
            (PiCCSInvocations.parentInterface logicalWidth publicFits)
            NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset))
        (Formal.initialClaimOffset
          (PiCCSInvocations.parentInterface logicalWidth publicFits)
          NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset) =
      PiCCSArithmetic.initialClaimConstraints logicalWidth publicFits := by
  unfold PiCCSArithmetic.initialClaimConstraints
  rw [PiCCSArithmetic.initialClaimLogicalStart_matches logicalWidth publicFits]
  unfold PiCCSArithmetic.sharedInterface PiCCSArithmetic.parentInterface
  rfl

theorem sumcheckConstraints_eq
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :
    NightstreamFPrime.Layout.PiCCS.v1_1.childConstraints
        (Formal.sumcheckCircuit
          (Formal.atOffset
            (PiCCSInvocations.parentInterface logicalWidth publicFits)
            NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset))
        (Formal.sumcheckOffset
          (PiCCSInvocations.parentInterface logicalWidth publicFits)
          NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset) =
      PiCCSArithmetic.sumcheckConstraints logicalWidth publicFits := by
  unfold PiCCSArithmetic.sumcheckConstraints
  rw [PiCCSArithmetic.sumcheckLogicalStart_matches logicalWidth publicFits]
  unfold PiCCSArithmetic.sharedInterface PiCCSArithmetic.parentInterface
  rfl

theorem evalKConstraints_eq
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :
    NightstreamFPrime.Layout.PiCCS.v1_1.childConstraints
        (Formal.evalKCircuit
          (Formal.atOffset
            (PiCCSInvocations.parentInterface logicalWidth publicFits)
            NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset))
        (Formal.evalKOffset
          (PiCCSInvocations.parentInterface logicalWidth publicFits)
          NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset) =
      PiCCSArithmetic.evalKConstraints logicalWidth publicFits := by
  unfold PiCCSArithmetic.evalKConstraints
  rw [PiCCSArithmetic.evalKLogicalStart_matches logicalWidth publicFits]
  unfold PiCCSArithmetic.sharedInterface PiCCSArithmetic.parentInterface
  rfl

theorem evalAConstraints_eq
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :
    NightstreamFPrime.Layout.PiCCS.v1_1.childConstraints
        (Formal.evalACircuit
          (Formal.atOffset
            (PiCCSInvocations.parentInterface logicalWidth publicFits)
            NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset))
        (Formal.evalAOffset
          (PiCCSInvocations.parentInterface logicalWidth publicFits)
          NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset) =
      PiCCSArithmetic.evalAConstraints logicalWidth publicFits := by
  unfold PiCCSArithmetic.evalAConstraints
  rw [PiCCSArithmetic.evalALogicalStart_matches logicalWidth publicFits]
  unfold PiCCSArithmetic.sharedInterface PiCCSArithmetic.parentInterface
  rfl

theorem ccsConstraints_eq
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    NightstreamFPrime.Layout.PiCCS.v1_1.childConstraints
        (Formal.ccsCircuit relation
          (Formal.atOffset
            (PiCCSInvocations.parentInterface logicalWidth publicFits)
            NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset))
        (Formal.ccsOffset
          (PiCCSInvocations.parentInterface logicalWidth publicFits)
          NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset) =
      PiCCSArithmetic.ccsConstraints logicalWidth publicFits := by
  unfold PiCCSArithmetic.ccsConstraints PiCCSArithmetic.mainConstraints
  rw [PiCCSArithmetic.ccsLogicalStart_matches logicalWidth publicFits]
  rw [← Formal.ccsCircuit_main_eq_rowMain relation
    (PiCCSArithmetic.sharedInterface logicalWidth publicFits)]
  unfold NightstreamFPrime.Layout.PiCCS.v1_1.childConstraints
    PiCCSArithmetic.sharedInterface PiCCSArithmetic.parentInterface
  rfl

theorem normConstraints_eq
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    NightstreamFPrime.Layout.PiCCS.v1_1.childConstraints
        (Formal.normCircuit relation
          (Formal.atOffset
            (PiCCSInvocations.parentInterface logicalWidth publicFits)
            NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset))
        (Formal.normOffset relation
          (PiCCSInvocations.parentInterface logicalWidth publicFits)
          NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset) =
      PiCCSArithmetic.normConstraints logicalWidth publicFits := by
  unfold PiCCSArithmetic.normConstraints PiCCSArithmetic.mainConstraints
  rw [PiCCSArithmetic.normLogicalStart_matches logicalWidth publicFits]
  unfold PiCCSArithmetic.sharedInterface PiCCSArithmetic.parentInterface
  rw [← Formal.normOffset_eq_normRowOffset relation
      (PiCCSInvocations.parentInterface logicalWidth publicFits)
      NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset]
  rw [← Formal.normCircuit_main_eq_rowMain relation
    (PiCCSInvocations.sharedInterface logicalWidth publicFits)]
  unfold NightstreamFPrime.Layout.PiCCS.v1_1.childConstraints
    PiCCSInvocations.sharedInterface
  rfl

theorem finalIdentityConstraints_eq
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    NightstreamFPrime.Layout.PiCCS.v1_1.childConstraints
        (Formal.finalIdentityCircuit relation
          (Formal.atOffset
            (PiCCSInvocations.parentInterface logicalWidth publicFits)
            NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset))
        (Formal.finalIdentityOffset relation
          (PiCCSInvocations.parentInterface logicalWidth publicFits)
          NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset) =
      PiCCSArithmetic.finalIdentityConstraints logicalWidth publicFits := by
  unfold PiCCSArithmetic.finalIdentityConstraints
    PiCCSArithmetic.mainConstraints
  rw [PiCCSArithmetic.finalIdentityLogicalStart_matches logicalWidth publicFits]
  unfold PiCCSArithmetic.sharedInterface PiCCSArithmetic.parentInterface
  rw [← Formal.finalIdentityOffset_eq_finalIdentityRowOffset relation
      (PiCCSInvocations.parentInterface logicalWidth publicFits)
      NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset]
  rw [← Formal.finalIdentityCircuit_main_eq_rowMain relation
    (PiCCSInvocations.sharedInterface logicalWidth publicFits)]
  unfold NightstreamFPrime.Layout.PiCCS.v1_1.childConstraints
    PiCCSInvocations.sharedInterface
  rfl

theorem arithmeticLogicalOps_constraints
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    flatConstraints (arithmeticLogicalOps relation) =
      packetConstraints logicalWidth publicFits := by
  unfold arithmeticLogicalOps
    Formal.CompletenessSupport.evaluationPrefixOps
    Formal.CompletenessSupport.preOutputOps packetConstraints
  rw [flatConstraints_append]
  simp only [flatConstraints, List.flatMap_cons, List.flatMap_nil,
    List.append_nil, childOp_flatConstraints]
  rw [initialClaimConstraints_eq logicalWidth publicFits,
    sumcheckConstraints_eq logicalWidth publicFits,
    evalKConstraints_eq logicalWidth publicFits,
    evalAConstraints_eq logicalWidth publicFits,
    ccsConstraints_eq relation, normConstraints_eq relation,
    finalIdentityConstraints_eq relation]
  simp only [List.append_assoc]

theorem arithmeticLogicalHolds_after_output
    {logicalWidth : Nat} {initial : Env}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (logicalBase after : Env)
    (completed : Sequence.Prefix initial PiCCSArithmetic.initialClaimLogicalStart)
    (operations : completed.operations = arithmeticLogicalOps relation)
    (endEq : PiCCSArithmetic.initialClaimLogicalStart +
        localLength completed.operations = PiCCSInvocations.outputWitnessStart)
    (pullbackEq : NightstreamFPrime.Layout.Stage1.Spartan.pullback logicalBase =
      completed.current)
    (outputAgrees : AgreesOutside logicalBase after
      (NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan
        PiCCSInvocations.outputWitnessStart)
      (PiCCSInvocations.invocationCeiling -
        NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan
          PiCCSInvocations.outputWitnessStart)) :
    ConstraintsHold (NightstreamFPrime.Layout.Stage1.Spartan.pullback after)
      (packetConstraints logicalWidth publicFits) := by
  have logicalHolds : ConstraintsHold completed.current
      (packetConstraints logicalWidth publicFits) := by
    have rows := completed.rows
    change ConstraintsHold completed.current
      (flatConstraints completed.operations) at rows
    rw [operations, arithmeticLogicalOps_constraints relation] at rows
    exact rows
  have scope : ∀ expression ∈ packetConstraints logicalWidth publicFits,
      expression.VarsBelow PiCCSInvocations.outputWitnessStart := by
    intro expression member
    have operationMember : expression ∈
        flatConstraints completed.operations := by
      rw [operations, arithmeticLogicalOps_constraints relation]
      exact member
    have below := completed.scope expression operationMember
    rw [endEq] at below
    exact below
  apply constraintsHold_of_agree_below completed.current
    (NightstreamFPrime.Layout.Stage1.Spartan.pullback after)
    (packetConstraints logicalWidth publicFits)
    PiCCSInvocations.outputWitnessStart scope
  · intro index below
    calc
      NightstreamFPrime.Layout.Stage1.Spartan.pullback after index =
          NightstreamFPrime.Layout.Stage1.Spartan.pullback logicalBase index :=
        pullback_after_output_agreesBelow logicalBase after outputAgrees index
          below
      _ = completed.current index := congrFun pullbackEq index
  · exact logicalHolds

theorem initialClaimFreshCount_eq
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    R1CS.totalFreshCount
      (PiCCSArithmetic.initialClaimConstraints logicalWidth publicFits) =
        90713 := by
  unfold PiCCSArithmetic.initialClaimConstraints
  rw [PiCCSArithmetic.initialClaimLogicalStart_matches logicalWidth publicFits]
  exact
    NightstreamFPrime.Layout.PiCCS.v1_1.Leaves.InitialClaim.freshColumnCount_eq
      (PiCCSArithmetic.sharedInterface logicalWidth publicFits)
      (PiCCSArithmetic.inputShapes logicalWidth publicFits relation).initialClaim
      (Formal.initialClaimOffset
        (PiCCSArithmetic.parentInterface logicalWidth publicFits)
        NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset)

theorem statementBindingFreshCount_eq
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    R1CS.totalFreshCount
      (PiCCSArithmetic.statementBindingConstraints logicalWidth publicFits) =
        0 := by
  unfold PiCCSArithmetic.statementBindingConstraints
  exact
    NightstreamFPrime.Layout.PiCCS.v1_1.Leaves.StatementBinding.freshColumnCount_eq
      (PiCCSArithmetic.sharedInterface logicalWidth publicFits)
      (fun childOffset => by
        simpa [PiCCSArithmetic.sharedInterface,
          PiCCSInvocations.sharedInterface,
          PiCCSArithmetic.parentInterface,
          PiCCSInvocations.parentInterface] using
          (PiCCSArithmetic.inputShapes logicalWidth publicFits relation
            ).statementBinding childOffset)
      PiCCSArithmetic.statementBindingLogicalStart

theorem sumcheckFreshCount_eq
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    R1CS.totalFreshCount
      (PiCCSArithmetic.sumcheckConstraints logicalWidth publicFits) =
        424601 := by
  unfold PiCCSArithmetic.sumcheckConstraints
  rw [PiCCSArithmetic.sumcheckLogicalStart_matches logicalWidth publicFits]
  exact
    NightstreamFPrime.Layout.PiCCS.v1_1.Leaves.SumcheckChain.freshColumnCount_eq
      (PiCCSArithmetic.sharedInterface logicalWidth publicFits)
      (PiCCSArithmetic.inputShapes logicalWidth publicFits relation).sumcheck
      (Formal.sumcheckOffset
        (PiCCSArithmetic.parentInterface logicalWidth publicFits)
        NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset)

theorem evalKFreshCount_eq
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    R1CS.totalFreshCount
      (PiCCSArithmetic.evalKConstraints logicalWidth publicFits) = 6706 := by
  unfold PiCCSArithmetic.evalKConstraints
  rw [PiCCSArithmetic.evalKLogicalStart_matches logicalWidth publicFits]
  exact
    NightstreamFPrime.Layout.PiCCS.v1_1.Leaves.EvalKTerminal.freshColumnCount_eq
      (PiCCSArithmetic.sharedInterface logicalWidth publicFits)
      (PiCCSArithmetic.inputShapes logicalWidth publicFits relation).eval_K
      (Formal.evalKOffset
        (PiCCSArithmetic.parentInterface logicalWidth publicFits)
        NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset)

theorem evalAFreshCount_eq
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    R1CS.totalFreshCount
      (PiCCSArithmetic.evalAConstraints logicalWidth publicFits) = 85330 := by
  unfold PiCCSArithmetic.evalAConstraints
  rw [PiCCSArithmetic.evalALogicalStart_matches logicalWidth publicFits]
  exact
    NightstreamFPrime.Layout.PiCCS.v1_1.Leaves.EvalATerminal.freshColumnCount_eq
      (PiCCSArithmetic.sharedInterface logicalWidth publicFits)
      (PiCCSArithmetic.inputShapes logicalWidth publicFits relation).eval_A
      (Formal.evalAOffset
        (PiCCSArithmetic.parentInterface logicalWidth publicFits)
        NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset)

theorem ccsFreshCount_eq
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    R1CS.totalFreshCount
      (PiCCSArithmetic.ccsConstraints logicalWidth publicFits) = 20792 := by
  unfold PiCCSArithmetic.ccsConstraints PiCCSArithmetic.mainConstraints
  rw [PiCCSArithmetic.ccsLogicalStart_matches logicalWidth publicFits]
  rw [← Formal.ccsCircuit_main_eq_rowMain relation
    (PiCCSArithmetic.sharedInterface logicalWidth publicFits)]
  exact NightstreamFPrime.Layout.PiCCS.v1_1.Leaves.CcsTerminal.freshColumnCount_eq
    relation (PiCCSArithmetic.sharedInterface logicalWidth publicFits)
    (PiCCSArithmetic.inputShapes logicalWidth publicFits relation).ccs
    (Formal.ccsOffset
      (PiCCSArithmetic.parentInterface logicalWidth publicFits)
      NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset)

theorem normFreshCount_eq
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    R1CS.totalFreshCount
      (PiCCSArithmetic.normConstraints logicalWidth publicFits) = 720 := by
  unfold PiCCSArithmetic.normConstraints PiCCSArithmetic.mainConstraints
  rw [PiCCSArithmetic.normLogicalStart_matches logicalWidth publicFits,
    ← Formal.normOffset_eq_normRowOffset relation
      (PiCCSArithmetic.parentInterface logicalWidth publicFits)
      NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset]
  rw [← Formal.normCircuit_main_eq_rowMain relation
    (PiCCSArithmetic.sharedInterface logicalWidth publicFits)]
  exact NightstreamFPrime.Layout.PiCCS.v1_1.Leaves.NormTerminal.freshColumnCount_eq
    relation (PiCCSArithmetic.sharedInterface logicalWidth publicFits)
    (PiCCSArithmetic.inputShapes logicalWidth publicFits relation).norm
    (Formal.normOffset relation
      (PiCCSArithmetic.parentInterface logicalWidth publicFits)
      NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset)

theorem finalIdentityFreshCount_eq
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    R1CS.totalFreshCount
      (PiCCSArithmetic.finalIdentityConstraints logicalWidth publicFits) =
        102743 := by
  unfold PiCCSArithmetic.finalIdentityConstraints
    PiCCSArithmetic.mainConstraints
  rw [PiCCSArithmetic.finalIdentityLogicalStart_matches logicalWidth publicFits,
    ← Formal.finalIdentityOffset_eq_finalIdentityRowOffset relation
      (PiCCSArithmetic.parentInterface logicalWidth publicFits)
      NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset]
  rw [← Formal.finalIdentityCircuit_main_eq_rowMain relation
    (PiCCSArithmetic.sharedInterface logicalWidth publicFits)]
  rw [NightstreamFPrime.Layout.PiCCS.v1_1.Leaves.FinalIdentity.freshColumnCount_eq
    relation (PiCCSArithmetic.sharedInterface logicalWidth publicFits)
    (PiCCSArithmetic.inputShapes logicalWidth publicFits relation).finalIdentity
    (Formal.finalIdentityOffset relation
      (PiCCSArithmetic.parentInterface logicalWidth publicFits)
      NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset)]
  have terminal := NightstreamFPrime.Layout.PiCCS.v1_1.terminalFreshCost_eq
    relation (PiCCSArithmetic.parentInterface logicalWidth publicFits)
    NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset
    (PiCCSArithmetic.inputShapes logicalWidth publicFits relation)
  unfold NightstreamFPrime.Layout.PiCCS.v1_1.terminalFreshCost at terminal
  unfold PiCCSArithmetic.parentInterface at terminal
  unfold PiCCSArithmetic.sharedInterface PiCCSArithmetic.parentInterface
    PiCCSInvocations.sharedInterface
  rw [terminal]

theorem packetConstraints_totalFreshCount
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    R1CS.totalFreshCount (packetConstraints logicalWidth publicFits) =
      731605 := by
  unfold packetConstraints
  rw [R1CS.totalFreshCount_append, R1CS.totalFreshCount_append,
    R1CS.totalFreshCount_append, R1CS.totalFreshCount_append,
    R1CS.totalFreshCount_append, R1CS.totalFreshCount_append,
    initialClaimFreshCount_eq relation, sumcheckFreshCount_eq relation,
    evalKFreshCount_eq relation, evalAFreshCount_eq relation,
    ccsFreshCount_eq relation, normFreshCount_eq relation,
    finalIdentityFreshCount_eq relation]

theorem emittedConstraints_totalFreshCount
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    R1CS.totalFreshCount (emittedConstraints logicalWidth publicFits) =
      731605 := by
  rw [emittedConstraints, R1CS.totalFreshCount_append,
    statementBindingFreshCount_eq relation,
    packetConstraints_totalFreshCount relation]

private theorem initialNext_eq_sumcheckFresh
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    PiCCSArithmetic.initialClaimFreshStart + R1CS.totalFreshCount
        (PiCCSArithmetic.initialClaimConstraints logicalWidth publicFits) =
      PiCCSArithmetic.sumcheckFreshStart := by
  rw [initialClaimFreshCount_eq relation]
  rfl

private theorem sumcheckNext_eq_evalKFresh
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    PiCCSArithmetic.sumcheckFreshStart + R1CS.totalFreshCount
        (PiCCSArithmetic.sumcheckConstraints logicalWidth publicFits) =
      PiCCSArithmetic.evalKFreshStart := by
  rw [sumcheckFreshCount_eq relation]
  rfl

private theorem evalKNext_eq_evalAFresh
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    PiCCSArithmetic.evalKFreshStart + R1CS.totalFreshCount
        (PiCCSArithmetic.evalKConstraints logicalWidth publicFits) =
      PiCCSArithmetic.evalAFreshStart := by
  rw [evalKFreshCount_eq relation]
  rfl

private theorem evalANext_eq_ccsFresh
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    PiCCSArithmetic.evalAFreshStart + R1CS.totalFreshCount
        (PiCCSArithmetic.evalAConstraints logicalWidth publicFits) =
      PiCCSArithmetic.ccsFreshStart := by
  rw [evalAFreshCount_eq relation]
  rfl

private theorem ccsNext_eq_normFresh
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    PiCCSArithmetic.ccsFreshStart + R1CS.totalFreshCount
        (PiCCSArithmetic.ccsConstraints logicalWidth publicFits) =
      PiCCSArithmetic.normFreshStart := by
  rw [ccsFreshCount_eq relation]
  rfl

private theorem normNext_eq_finalIdentityFresh
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    PiCCSArithmetic.normFreshStart + R1CS.totalFreshCount
        (PiCCSArithmetic.normConstraints logicalWidth publicFits) =
      PiCCSArithmetic.finalIdentityFreshStart := by
  rw [normFreshCount_eq relation]
  rfl

theorem lowerPacketConstraints_rows
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    (R1CS.lowerConstraints (packetConstraints logicalWidth publicFits)
      PiCCSArithmetic.initialClaimFreshStart).rows =
      (R1CS.lowerConstraints
        (PiCCSArithmetic.initialClaimConstraints logicalWidth publicFits)
        PiCCSArithmetic.initialClaimFreshStart).rows ++
      (R1CS.lowerConstraints
        (PiCCSArithmetic.sumcheckConstraints logicalWidth publicFits)
        PiCCSArithmetic.sumcheckFreshStart).rows ++
      (R1CS.lowerConstraints
        (PiCCSArithmetic.evalKConstraints logicalWidth publicFits)
        PiCCSArithmetic.evalKFreshStart).rows ++
      (R1CS.lowerConstraints
        (PiCCSArithmetic.evalAConstraints logicalWidth publicFits)
        PiCCSArithmetic.evalAFreshStart).rows ++
      (R1CS.lowerConstraints
        (PiCCSArithmetic.ccsConstraints logicalWidth publicFits)
        PiCCSArithmetic.ccsFreshStart).rows ++
      (R1CS.lowerConstraints
        (PiCCSArithmetic.normConstraints logicalWidth publicFits)
        PiCCSArithmetic.normFreshStart).rows ++
      (R1CS.lowerConstraints
        (PiCCSArithmetic.finalIdentityConstraints logicalWidth publicFits)
        PiCCSArithmetic.finalIdentityFreshStart).rows := by
  unfold packetConstraints
  rw [R1CS.lowerConstraints_append_rows,
    initialNext_eq_sumcheckFresh relation,
    R1CS.lowerConstraints_append_rows,
    sumcheckNext_eq_evalKFresh relation,
    R1CS.lowerConstraints_append_rows,
    evalKNext_eq_evalAFresh relation,
    R1CS.lowerConstraints_append_rows,
    evalANext_eq_ccsFresh relation,
    R1CS.lowerConstraints_append_rows,
    ccsNext_eq_normFresh relation,
    R1CS.lowerConstraints_append_rows,
    normNext_eq_finalIdentityFresh relation]
  simp only [List.append_assoc]

theorem lowerEmittedConstraints_rows
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    (R1CS.lowerConstraints (emittedConstraints logicalWidth publicFits)
      PiCCSArithmetic.initialClaimFreshStart).rows =
      (R1CS.lowerConstraints
        (PiCCSArithmetic.statementBindingConstraints logicalWidth publicFits)
        PiCCSArithmetic.initialClaimFreshStart).rows ++
      (R1CS.lowerConstraints (packetConstraints logicalWidth publicFits)
        PiCCSArithmetic.initialClaimFreshStart).rows := by
  unfold emittedConstraints
  rw [R1CS.lowerConstraints_append_rows,
    statementBindingFreshCount_eq relation]
  simp only [Nat.add_zero]

theorem arithmeticRows_toR1CS_eq
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    (PiCCSArithmetic.arithmeticRows logicalWidth publicFits).map
        Rows.CompiledRow.toR1CS =
      NightstreamFPrime.Layout.Stage1.Spartan.remapRows
        (R1CS.lowerConstraints (emittedConstraints logicalWidth publicFits)
          PiCCSArithmetic.initialClaimFreshStart).rows := by
  have statementFreshEq : PiCCSArithmetic.statementBindingFreshStart =
      PiCCSArithmetic.initialClaimFreshStart := by
    rfl
  unfold PiCCSArithmetic.arithmeticRows PiCCSArithmetic.statementBindingRows
    PiCCSArithmetic.initialClaimRows
    PiCCSArithmetic.sumcheckRows PiCCSArithmetic.evalKRows
    PiCCSArithmetic.evalARows PiCCSArithmetic.ccsRows
    PiCCSArithmetic.normRows PiCCSArithmetic.finalIdentityRows
  simp only [List.map_append, PiCCSArithmetic.compilePacket_toR1CS]
  rw [lowerEmittedConstraints_rows relation,
    lowerPacketConstraints_rows relation]
  rw [statementFreshEq]
  simp only [NightstreamFPrime.Layout.Stage1.Spartan.remapRows,
    List.map_append, List.append_assoc]

theorem complete_arithmeticRows
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (env : Env)
    (scope : ∀ expression ∈ emittedConstraints logicalWidth publicFits,
      expression.VarsBelow PiCCSArithmetic.initialClaimFreshStart)
    (logical : ConstraintsHold
      (NightstreamFPrime.Layout.Stage1.Spartan.pullback env)
      (emittedConstraints logicalWidth publicFits)) :
    ∃ completed,
      AgreesOutside env completed PiCCSInvocations.invocationCeiling 731605 ∧
        R1CS.RowsHold completed
          ((PiCCSArithmetic.arithmeticRows logicalWidth publicFits).map
            Rows.CompiledRow.toR1CS) := by
  rcases R1CS.lowerConstraints_complete
      (NightstreamFPrime.Layout.Stage1.Spartan.pullback env)
      (emittedConstraints logicalWidth publicFits)
      PiCCSArithmetic.initialClaimFreshStart scope logical with
    ⟨source, sourceAgrees, sourceRows⟩
  have totalFresh := emittedConstraints_totalFreshCount relation
  have sourceAgreesFixed : AgreesOutside
      (NightstreamFPrime.Layout.Stage1.Spartan.pullback env) source
      PiCCSArithmetic.initialClaimFreshStart 731605 := by
    rw [totalFresh] at sourceAgrees
    exact sourceAgrees
  have mappedStart :
      NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan
          PiCCSArithmetic.initialClaimFreshStart =
        PiCCSInvocations.invocationCeiling := by
    rfl
  have startLocal :
      NightstreamFPrime.Layout.Stage1.Spartan.piCcsPhaseOffset ≤
        PiCCSArithmetic.initialClaimFreshStart := by
    unfold PiCCSArithmetic.initialClaimFreshStart
      NightstreamFPrime.Layout.Stage1.PiCCSStarts.initialClaimFreshStart
      NightstreamFPrime.Layout.Stage1.PiCCSStarts.roundTranscriptFreshStart
      NightstreamFPrime.Layout.Stage1.PiCCSStarts.challengeFreshStart
      NightstreamFPrime.Layout.Stage1.PiCCSStarts.statementAbsorptionFreshStart
      NightstreamFPrime.Layout.Stage1.PiCCSStarts.statementBindingFreshStart
      NightstreamFPrime.Layout.Stage1.PiCCSStarts.logicalFreshBase
    rw [NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset_eq]
    norm_num [NightstreamFPrime.Layout.Stage1.Spartan.piCcsPhaseOffset]
  have targetEndPrivate :
      NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan
          PiCCSArithmetic.initialClaimFreshStart + 731605 ≤
        NightstreamFPrime.Layout.Stage1.Spartan.privateColumnCount := by
    rw [mappedStart, PiCCSInvocations.invocationCeiling_eq,
      NightstreamFPrime.Layout.Stage1.Spartan.privateColumnCount_eq]
    norm_num
  let completed :=
    NightstreamFPrime.Layout.Stage1.Spartan.copyMappedInterval env source
      PiCCSArithmetic.initialClaimFreshStart 731605
  refine ⟨completed, ?_, ?_⟩
  · rw [← mappedStart]
    exact
      NightstreamFPrime.Layout.Stage1.Spartan.copyMappedInterval_agreesOutside
        env source PiCCSArithmetic.initialClaimFreshStart 731605
  · rw [arithmeticRows_toR1CS_eq relation]
    exact
      NightstreamFPrime.Layout.Stage1.Spartan.remapRows_hold_copyMappedInterval
        (R1CS.lowerConstraints (emittedConstraints logicalWidth publicFits)
          PiCCSArithmetic.initialClaimFreshStart).rows env source
        PiCCSArithmetic.initialClaimFreshStart 731605 startLocal
        targetEndPrivate sourceAgreesFixed sourceRows

end NightstreamFPrime.Export.Stage1.PiCCSCompleteness
