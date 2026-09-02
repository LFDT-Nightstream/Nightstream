import NightstreamFPrime.Export.Stage1.PiCCSOrdinaryDirectPlan
import NightstreamFPrime.Export.Stage1.PiCCSTranscriptDirectSemantics
import NightstreamFPrime.Export.Stage1.InvocationLastOutput
import NightstreamFPrime.Layout.ProductionRelation.PinFamilyPlan

/-!
Owns 32 zero-pin rows that bind the four direct PiCCS transcript endpoint
states to the exact lifecycle compiler variables. The first three endpoints
reuse the proof-logical retained block. The output endpoint uses its dedicated
eight-field retained block.

This module does not assemble the other PiCCS leaves.
-/

namespace NightstreamFPrime.Export.Stage1.PiCCSTranscriptEndpointPlan

open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Circuit
open NightstreamFPrime.Gadgets.Poseidon2
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.PiCCS.v1_1
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra

def familyCount : Nat := 4
def laneCount : Nat := Spec.Poseidon2.width
def rowCount : Nat := familyCount * laneCount

def statementFamily : Fin familyCount := ⟨0, by norm_num [familyCount]⟩
def challengeFamily : Fin familyCount := ⟨1, by norm_num [familyCount]⟩
def roundFamily : Fin familyCount := ⟨2, by norm_num [familyCount]⟩
def outputFamily : Fin familyCount := ⟨3, by norm_num [familyCount]⟩

@[simp] theorem rowCount_eq : rowCount = 32 := by
  rfl

def descriptor (row : Fin rowCount) : Fin familyCount × Fin laneCount :=
  Fin.decodeProd row

def row (family : Fin familyCount) (lane : Fin laneCount) : Fin rowCount :=
  Fin.encodeProd (family, lane)

@[simp] theorem descriptor_row (family : Fin familyCount)
    (lane : Fin laneCount) : descriptor (row family lane) = (family, lane) := by
  exact Fin.decodeProd_encodeProd (family, lane)

private theorem absorb_positive_of_nonempty
    (input : List NightstreamFPrime.Circuit.Expr) (nonempty : input ≠ []) :
    0 < Invocations.Action.invocationCount (.absorb input) := by
  change 0 < (NightstreamFPrime.Gadgets.Poseidon2.Hash.inputChunks input).length
  unfold NightstreamFPrime.Gadgets.Poseidon2.Hash.inputChunks
  simp only [List.length_map, List.length_range]
  apply Nat.div_pos
  · have lengthPositive : 0 < input.length := by
      cases input with
      | nil => exact False.elim (nonempty rfl)
      | cons head tail => simp
    norm_num [Spec.Poseidon2.rate]
    omega
  · norm_num [Spec.Poseidon2.rate]

private theorem framedAbsorb_positive
    (words : List NightstreamFPrime.Circuit.Expr) :
    0 < Invocations.Action.invocationCount
      (StatementAbsorption.absorbBlock words) := by
  apply absorb_positive_of_nonempty
  simp [StatementAbsorption.blockExpr]

private theorem constantAbsorb_positive (words : List F)
    (nonempty : words ≠ []) :
    0 < Invocations.Action.invocationCount
      (.absorb (words.map NightstreamFPrime.Circuit.Expr.const)) := by
  apply absorb_positive_of_nonempty
  simpa using nonempty

private theorem squeeze_positive
    (expected : NightstreamFPrime.Circuit.Quadratic.KExpr) :
    0 < Invocations.Action.invocationCount (.squeezeK expected) := by
  simp [Invocations.Action.invocationCount]

private theorem actionsPositive_append
    {left right : List NightstreamFPrime.Gadgets.Poseidon2.Duplex.Formal.Action}
    (leftPositive : InvocationLastOutput.ActionsPositive left)
    (rightPositive : InvocationLastOutput.ActionsPositive right) :
    InvocationLastOutput.ActionsPositive (left ++ right) := by
  intro action member
  rw [List.mem_append] at member
  exact member.elim (leftPositive action) (rightPositive action)

private theorem actionsPositive_flatMap {Index : Type}
    (indices : List Index)
    (group : Index →
      List NightstreamFPrime.Gadgets.Poseidon2.Duplex.Formal.Action)
    (each : ∀ index ∈ indices,
      InvocationLastOutput.ActionsPositive (group index)) :
    InvocationLastOutput.ActionsPositive (indices.flatMap group) := by
  intro action member
  rw [List.mem_flatMap] at member
  rcases member with ⟨index, indexMember, actionMember⟩
  exact each index indexMember action actionMember

private theorem statementActions_positive :
    InvocationLastOutput.ActionsPositive
      PiCCSActionPayloadBlock.statementActions := by
  intro action member
  unfold PiCCSActionPayloadBlock.statementActions
    PiCCSInvocations.statementActions StatementAbsorption.actions
    StatementAbsorption.publicInputActions at member
  simp only [List.mem_append, List.mem_singleton, List.mem_map] at member
  rcases member with rfl | ⟨words, _, rfl⟩
  · simpa [StatementAbsorption.constantWords] using
      constantAbsorb_positive
        NightstreamFPrime.Lifecycle.Transcript.piCcsDigestDomainTag (by
          intro empty
          have lengths := congrArg List.length empty
          simp at lengths)
  · exact framedAbsorb_positive words

private theorem labelWord_nonempty
    (label : FiatShamir.ChallengeLabel
      NightstreamFPrime.Lifecycle.productionShape) :
    NightstreamFPrime.Lifecycle.Transcript.labelWord label ≠ [] := by
  cases label <;>
    simp [NightstreamFPrime.Lifecycle.Transcript.labelWord]

private theorem challengeLabelActions_positive
    (label : FiatShamir.ChallengeLabel
      NightstreamFPrime.Lifecycle.productionShape)
    (expected : NightstreamFPrime.Circuit.Quadratic.KExpr) :
    InvocationLastOutput.ActionsPositive
      (ChallengeDerivation.labelActions label expected) := by
  intro action member
  simp only [ChallengeDerivation.labelActions, List.mem_cons,
    List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl
  · simpa [ChallengeDerivation.constantWords] using
      constantAbsorb_positive
        (NightstreamFPrime.Lifecycle.Transcript.labelWord label)
        (labelWord_nonempty label)
  · exact squeeze_positive expected

private theorem labelledActions_positive :
    ∀ labels samples,
      InvocationLastOutput.ActionsPositive
        (ChallengeDerivation.labelledActions labels samples)
  | [], samples => by
      intro action member
      simp [ChallengeDerivation.labelledActions] at member
  | label :: labels, [] => by
      intro action member
      simp [ChallengeDerivation.labelledActions] at member
  | label :: labels, sample :: samples => by
      rw [ChallengeDerivation.labelledActions]
      exact actionsPositive_append
        (challengeLabelActions_positive label sample)
        (labelledActions_positive labels samples)

private theorem challengeActions_positive :
    InvocationLastOutput.ActionsPositive
      PiCCSActionPayloadBlock.challengeActions := by
  unfold PiCCSActionPayloadBlock.challengeActions
  rw [ChallengeDerivation.actions_eq_labelled]
  exact labelledActions_positive _ _

private theorem roundActionsWithExpected_positive
    (interface : RoundTranscript.Interface 9) (offset : Nat)
    (roundIndex : Fin
      NightstreamFPrime.Lifecycle.productionShape.cubeVariables)
    (expected : NightstreamFPrime.Circuit.Quadratic.KExpr) :
    InvocationLastOutput.ActionsPositive
      (RoundTranscript.roundActionsWithExpected interface offset roundIndex
        expected) := by
  intro action member
  simp only [RoundTranscript.roundActionsWithExpected, List.mem_cons,
    List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl | rfl
  · apply absorb_positive_of_nonempty
    simp [RoundTranscript.blockExpr]
  · simpa [RoundTranscript.constantWords] using
      constantAbsorb_positive
        (NightstreamFPrime.Lifecycle.Transcript.labelWord
          (.sumcheck roundIndex)) (labelWord_nonempty (.sumcheck roundIndex))
  · exact squeeze_positive expected

private theorem roundActions_positive :
    InvocationLastOutput.ActionsPositive
      PiCCSActionPayloadBlock.roundActions := by
  unfold PiCCSActionPayloadBlock.roundActions RoundTranscript.actions
  apply actionsPositive_flatMap
  intro roundIndex member
  unfold RoundTranscript.roundActions
  exact roundActionsWithExpected_positive _ _ roundIndex _

private theorem outputActions_positive :
    InvocationLastOutput.ActionsPositive
      PiCCSActionPayloadBlock.outputActions := by
  intro action member
  unfold PiCCSActionPayloadBlock.outputActions PiCCSInvocations.outputActions
    OutputBinding.actions at member
  simp only [List.mem_singleton] at member
  subst action
  exact framedAbsorb_positive _

private theorem statementActions_nonempty :
    PiCCSActionPayloadBlock.statementActions ≠ [] := by
  intro empty
  have count := PiCCSInvocations.statementInvocationCount_eq
    Data.logicalWidth Data.publicFits
  unfold PiCCSActionPayloadBlock.statementActions at empty
  rw [empty] at count
  simp [Invocations.invocationCount] at count

private theorem challengeActions_nonempty :
    PiCCSActionPayloadBlock.challengeActions ≠ [] := by
  intro empty
  have count := PiCCSActionPayloadBlock.challengeInvocationCount_eq
  rw [empty] at count
  simp [Invocations.invocationCount] at count

private theorem roundActions_nonempty :
    PiCCSActionPayloadBlock.roundActions ≠ [] := by
  intro empty
  have count := PiCCSActionPayloadBlock.roundInvocationCount_eq
  rw [empty] at count
  simp [Invocations.invocationCount] at count

private theorem outputActions_nonempty :
    PiCCSActionPayloadBlock.outputActions ≠ [] := by
  intro empty
  have count := PiCCSInvocations.outputInvocationCount_eq
    Data.logicalWidth Data.publicFits
  unfold PiCCSActionPayloadBlock.outputActions at empty
  rw [empty] at count
  simp [Invocations.invocationCount] at count

def endpointInvocation (family : Fin familyCount) :
    Fin PiCCSActionPayloadBlock.invocationCount :=
  if family.val = 0 then
    PiCCSTranscriptDirectSemantics.statementLast
  else if family.val = 1 then
    PiCCSTranscriptDirectSemantics.challengeLast
  else if family.val = 2 then
    PiCCSTranscriptDirectSemantics.roundLast
  else
    PiCCSTranscriptDirectSemantics.outputLast

def endpointStart (family : Fin familyCount) : Nat :=
  if family.val = 0 then
    PiCCSInvocations.challengeWitnessStart - 8
  else if family.val = 1 then
    PiCCSInvocations.roundWitnessStart - 8
  else if family.val = 2 then
    PiCCSInvocations.roundWitnessStart +
      PiCCSTranscriptDirectSemantics.roundCount * 592 - 8
  else
    PiCCSStarts.logicalFreshBase - 8

def endpointColumn (family : Fin familyCount) (lane : Fin laneCount) : Nat :=
  endpointStart family + lane.val

private theorem statementTrace_state_endpoint (lane : Fin laneCount) :
    (PiCCSInvocations.statementTrace Data.logicalWidth Data.publicFits).state
        lane =
      Expr.var (endpointColumn statementFamily lane) := by
  have compiled := InvocationLastOutput.compileActions_state_scheduleOutput
    PiCCSInvocations.statementPhase PiCCSInvocations.statementRowStart
    PiCCSInvocations.statementWitnessStart Hash.zeroE
    PiCCSActionPayloadBlock.statementActions statementActions_nonempty
    statementActions_positive
  change
    (PiCCSInvocations.statementTrace Data.logicalWidth Data.publicFits).state =
      Permutation.scheduleOutput
        (PiCCSInvocations.statementWitnessStart +
          (Invocations.invocationCount
            PiCCSActionPayloadBlock.statementActions - 1) * 592) at compiled
  have count : Invocations.invocationCount
      PiCCSActionPayloadBlock.statementActions = 379 := by
    exact PiCCSInvocations.statementInvocationCount_eq
      Data.logicalWidth Data.publicFits
  have endEq := PiCCSInvocations.statementEnd_eq_challengeStart
    Data.logicalWidth Data.publicFits
  rw [count] at compiled
  rw [PiCCSInvocations.statementInvocationCount_eq] at endEq
  have startEq :
      PiCCSInvocations.statementWitnessStart + (379 - 1) * 592 + 584 =
        PiCCSInvocations.challengeWitnessStart - 8 := by
    rw [← endEq]
    generalize PiCCSInvocations.statementWitnessStart = start
    omega
  rw [congrFun compiled lane]
  unfold Permutation.scheduleOutput Permutation.freshState endpointColumn
  rw [show endpointStart statementFamily =
      PiCCSInvocations.challengeWitnessStart - 8 by rfl]
  rw [startEq]

private theorem challengeTrace_state_endpoint (lane : Fin laneCount) :
    (PiCCSInvocations.challengeTrace Data.logicalWidth Data.publicFits).state
        lane =
      Expr.var (endpointColumn challengeFamily lane) := by
  rw [PiCCSInvocations.challengeTrace_eq_semantic]
  have compiled := InvocationLastOutput.compileActions_state_scheduleOutput
    PiCCSInvocations.challengePhase PiCCSInvocations.challengeRowStart
    PiCCSInvocations.challengeWitnessStart
    ((PiCCSInvocations.challengeInterface Data.logicalWidth
      Data.publicFits).initialState PiCCSInvocations.challengeWitnessStart)
    PiCCSActionPayloadBlock.challengeActions challengeActions_nonempty
    challengeActions_positive
  change
    (PiCCSInvocations.challengeSemanticTrace Data.logicalWidth
        Data.publicFits).state =
      Permutation.scheduleOutput
        (PiCCSInvocations.challengeWitnessStart +
          (Invocations.invocationCount
            PiCCSActionPayloadBlock.challengeActions - 1) * 592) at compiled
  have endEq := PiCCSInvocations.challengeEnd_eq_roundStart
    Data.logicalWidth Data.publicFits
  rw [PiCCSActionPayloadBlock.challengeInvocationCount_eq] at compiled
  rw [PiCCSInvocations.challengeInvocationCount_eq] at endEq
  have startEq :
      PiCCSInvocations.challengeWitnessStart + (87 - 1) * 592 + 584 =
        PiCCSInvocations.roundWitnessStart - 8 := by
    rw [← endEq]
    generalize PiCCSInvocations.challengeWitnessStart = start
    omega
  rw [congrFun compiled lane]
  unfold Permutation.scheduleOutput Permutation.freshState endpointColumn
  rw [show endpointStart challengeFamily =
      PiCCSInvocations.roundWitnessStart - 8 by rfl]
  rw [startEq]

private theorem roundTrace_state_endpoint (lane : Fin laneCount) :
    (PiCCSInvocations.roundTrace Data.logicalWidth Data.publicFits).state lane =
      Expr.var (endpointColumn roundFamily lane) := by
  rw [PiCCSInvocations.roundTrace_eq_semantic]
  have compiled := InvocationLastOutput.compileActions_state_scheduleOutput
    PiCCSInvocations.roundPhase PiCCSInvocations.roundRowStart
    PiCCSInvocations.roundWitnessStart
    ((PiCCSInvocations.roundInterface Data.logicalWidth
      Data.publicFits).initialState PiCCSInvocations.roundWitnessStart)
    PiCCSActionPayloadBlock.roundActions roundActions_nonempty
    roundActions_positive
  change
    (PiCCSInvocations.roundSemanticTrace Data.logicalWidth
        Data.publicFits).state =
      Permutation.scheduleOutput
        (PiCCSInvocations.roundWitnessStart +
          (Invocations.invocationCount PiCCSActionPayloadBlock.roundActions - 1) *
            592) at compiled
  rw [PiCCSActionPayloadBlock.roundInvocationCount_eq] at compiled
  have startEq :
      PiCCSInvocations.roundWitnessStart + (252 - 1) * 592 + 584 =
        PiCCSInvocations.roundWitnessStart + 252 * 592 - 8 := by
    generalize PiCCSInvocations.roundWitnessStart = start
    omega
  rw [congrFun compiled lane]
  unfold Permutation.scheduleOutput Permutation.freshState endpointColumn
  rw [show endpointStart roundFamily =
      PiCCSInvocations.roundWitnessStart +
        PiCCSTranscriptDirectSemantics.roundCount * 592 - 8 by rfl]
  rw [show PiCCSTranscriptDirectSemantics.roundCount = 252 by rfl]
  rw [startEq]

private theorem outputTrace_state_endpoint (lane : Fin laneCount) :
    (PiCCSInvocations.outputTrace Data.logicalWidth Data.publicFits).state lane =
      Expr.var (endpointColumn outputFamily lane) := by
  have compiled := InvocationLastOutput.compileActions_state_scheduleOutput
    PiCCSInvocations.outputPhase PiCCSInvocations.outputRowStart
    PiCCSInvocations.outputWitnessStart
    (PiCCSInvocations.roundTrace Data.logicalWidth Data.publicFits).state
    PiCCSActionPayloadBlock.outputActions outputActions_nonempty
    outputActions_positive
  change
    (PiCCSInvocations.outputTrace Data.logicalWidth Data.publicFits).state =
      Permutation.scheduleOutput
        (PiCCSInvocations.outputWitnessStart +
          (Invocations.invocationCount PiCCSActionPayloadBlock.outputActions - 1) *
            592) at compiled
  have count : Invocations.invocationCount
      PiCCSActionPayloadBlock.outputActions = 6886 := by
    exact PiCCSInvocations.outputInvocationCount_eq
      Data.logicalWidth Data.publicFits
  have endEq := PiCCSInvocations.outputEnd_eq_logicalFreshBase
    Data.logicalWidth Data.publicFits
  rw [count] at compiled
  rw [PiCCSInvocations.outputInvocationCount_eq] at endEq
  have startEq :
      PiCCSInvocations.outputWitnessStart + (6886 - 1) * 592 + 584 =
        PiCCSStarts.logicalFreshBase - 8 := by
    rw [← endEq]
    generalize PiCCSInvocations.outputWitnessStart = start
    omega
  rw [congrFun compiled lane]
  unfold Permutation.scheduleOutput Permutation.freshState endpointColumn
  rw [show endpointStart outputFamily = PiCCSStarts.logicalFreshBase - 8 by rfl]
  rw [startEq]

theorem endpointColumn_lt_source (family : Fin familyCount)
    (lane : Fin laneCount) : endpointColumn family lane < Spartan.SourceColumnCount := by
  have familyBound := family.isLt
  have laneBound := lane.isLt
  change lane.val < 8 at laneBound
  unfold endpointColumn endpointStart familyCount laneCount
  rw [Spartan.sourceColumnCount_eq]
  split <;> try split <;> try split
  all_goals
    try simp only [PiCCSInvocations.challengeWitnessStart,
      PiCCSInvocations.roundWitnessStart]
  all_goals
    try simp only [PiCCSStarts.challengeWitnessStart_eq,
      PiCCSStarts.roundTranscriptWitnessStart_eq]
  all_goals
    norm_num [PiCCSTranscriptDirectSemantics.roundCount,
      PiCCSStarts.logicalFreshBase, PiCCSInputs.phaseOffset_eq] at *
  all_goals omega

def endpointTranscriptInvocation (family : Fin familyCount) :
    Fin PiCCSOrdinaryRetainedBlocks.transcriptInvocationCount :=
  if family.val = 0 then
    ⟨378, by rw [PiCCSOrdinaryRetainedBlocks.transcriptInvocationCount_eq]; omega⟩
  else if family.val = 1 then
    ⟨465, by rw [PiCCSOrdinaryRetainedBlocks.transcriptInvocationCount_eq]; omega⟩
  else
    ⟨717, by rw [PiCCSOrdinaryRetainedBlocks.transcriptInvocationCount_eq]; omega⟩

def endpointTranscriptIndex (family : Fin familyCount) (lane : Fin laneCount) :
    Fin PiCCSOrdinaryRetainedBlocks.transcriptOutputCount :=
  Fin.encodeProd (endpointTranscriptInvocation family, lane)

def proofLogicalIndex (family : Fin familyCount) (_notOutput : family.val ≠ 3)
    (lane : Fin laneCount) :
    Fin PiCCSOrdinaryRetainedBlocks.proofLogicalCount :=
  PiCCSOrdinaryRetainedBlocks.transcriptOutputSlot
    (endpointTranscriptIndex family lane)

@[simp] theorem proofLogicalIndex_source (family : Fin familyCount)
    (notOutput : family.val ≠ 3) (lane : Fin laneCount) :
    PiCCSOrdinaryRetainedBlocks.proofLogicalSource
        (proofLogicalIndex family notOutput lane) =
      endpointColumn family lane := by
  have familyBound := family.isLt
  have laneBound := lane.isLt
  change lane.val < 8 at laneBound
  unfold proofLogicalIndex
  rw [PiCCSOrdinaryRetainedBlocks.proofLogicalSource_transcriptOutput]
  unfold endpointTranscriptIndex
  rw [PiCCSOrdinaryRetainedBlocks.transcriptOutputSource_encodeProd]
  unfold endpointTranscriptInvocation endpointColumn endpointStart familyCount
    laneCount
  split <;> try split <;> try split
  all_goals
    try simp only [PiCCSInvocations.challengeWitnessStart,
      PiCCSInvocations.roundWitnessStart]
  all_goals
    try simp only [PiCCSStarts.challengeWitnessStart_eq,
      PiCCSStarts.roundTranscriptWitnessStart_eq]
  all_goals
    norm_num [PiCCSTranscriptDirectSemantics.roundCount,
      PiCCSStarts.logicalFreshBase, PiCCSInputs.phaseOffset_eq] at *
  all_goals omega

private theorem packageSourceColumn_congr
    (program : Lifecycle.Stage1.Application.Program)
    (left right : Nat) (leftBound : left < Spartan.SourceColumnCount)
    (rightBound : right < Spartan.SourceColumnCount) (same : left = right) :
    RunningTransitionRetainedBlocks.packageSourceColumn program left leftBound =
      RunningTransitionRetainedBlocks.packageSourceColumn program right
        rightBound := by
  subst right
  rfl

def directForm {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (geometry : PiCCSPoseidonPlan.Geometry program logicalWidth)
    (family : Fin familyCount) (lane : Fin laneCount) :
    SparseForm logicalWidth :=
  PiCCSPoseidonPlan.outputState geometry (endpointInvocation family) lane

def sourceForm {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (geometry : PiCCSOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (family : Fin familyCount) (lane : Fin laneCount) :
    SparseForm logicalWidth :=
  if output : family.val = 3 then
    (PiCCSOrdinaryRetainedBlocks.outputEndpointBlock program).form
      (PiCCSOrdinaryRetainedGeometry.outputEndpointStart program)
      (PiCCSOrdinaryRetainedGeometry.outputEndpointFits geometry) lane
  else
    (PiCCSOrdinaryRetainedBlocks.proofLogicalBlock program).form
      (PiCCSOrdinaryRetainedGeometry.proofLogicalStart program)
      (PiCCSOrdinaryRetainedGeometry.proofLogicalFits geometry)
      (proofLogicalIndex family output lane)

theorem sourceForm_eval
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiCCSOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 33 → F)
    (products : Fin PiRLCFirst54DirectSchedule.candidateCount → F)
    (encoding : PiCCSOrdinaryRetainedGeometry.Encodes geometry assignment
      (PiRLCRetainedPreservation.sourceAssignment
        program base groupValue products))
    (family : Fin familyCount) (lane : Fin laneCount) :
    (sourceForm geometry family lane).eval assignment =
      PiCCSActionPayloadBlock.packageEnv program
        (PiRLCRetainedPreservation.sourceAssignment
          program base groupValue products) (endpointColumn family lane) := by
  rw [PiCCSPoseidonPreservation.packageEnv_sourceAssignment program base
    groupValue products (endpointColumn family lane)
    (endpointColumn_lt_source family lane)]
  by_cases output : family.val = 3
  · have familyEq : family = outputFamily := by
      apply Fin.ext
      exact output
    subst family
    rw [sourceForm, dif_pos (by rfl)]
    rw [LowNormBlock.Block.form_eval _ _ _ assignment _
      encoding.outputEndpoint]
    have sourceEq :
        (PiCCSOrdinaryRetainedBlocks.outputEndpointBlock program).source lane =
          RunningTransitionRetainedBlocks.packageSourceColumn program
            (endpointColumn outputFamily lane)
            (endpointColumn_lt_source outputFamily lane) := by
      apply Fin.ext
      rfl
    rw [sourceEq]
    unfold RunningTransitionRetainedBlocks.packageSourceColumn
    rw [PiRLCRetainedPreservation.sourceAssignment_base]
    apply congrArg base
    apply Fin.ext
    rfl
  · rw [sourceForm, dif_neg output]
    rw [LowNormBlock.Block.form_eval _ _ _ assignment _
      encoding.proofLogical]
    have sourceEq :
        (PiCCSOrdinaryRetainedBlocks.proofLogicalBlock program).source
            (proofLogicalIndex family output lane) =
          RunningTransitionRetainedBlocks.packageSourceColumn program
            (endpointColumn family lane)
            (endpointColumn_lt_source family lane) := by
      calc
        _ = RunningTransitionRetainedBlocks.packageSourceColumn program
              (PiCCSOrdinaryRetainedBlocks.proofLogicalSource
                (proofLogicalIndex family output lane))
              (PiCCSOrdinaryRetainedBlocks.proofLogicalSource_lt _) := rfl
        _ = _ := packageSourceColumn_congr program _ _ _ _
          (proofLogicalIndex_source family output lane)
    rw [sourceEq]
    unfold RunningTransitionRetainedBlocks.packageSourceColumn
    rw [PiRLCRetainedPreservation.sourceAssignment_base]
    apply congrArg base
    apply Fin.ext
    rfl

def bindingForm {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (poseidonGeometry : PiCCSPoseidonPlan.Geometry program logicalWidth)
    (ordinaryGeometry :
      PiCCSOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (rowIndex : Fin rowCount) : SparseForm logicalWidth :=
  let decoded := descriptor rowIndex
  SparseForm.add (directForm poseidonGeometry decoded.1 decoded.2)
    (SparseForm.scale (-1)
      (sourceForm ordinaryGeometry decoded.1 decoded.2))

def interface {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (poseidonGeometry : PiCCSPoseidonPlan.Geometry program logicalWidth)
    (ordinaryGeometry :
      PiCCSOrdinaryRetainedGeometry.Geometry program logicalWidth) :
    PinFamilyPlan.Interface logicalWidth rowCount where
  oneColumn := PiCCSOrdinaryRetainedGeometry.oneColumn ordinaryGeometry
  value := bindingForm poseidonGeometry ordinaryGeometry

def rowCount_le : rowCount ≤ 2 ^ NightstreamFPrime.Lifecycle.cubeVariables := by
  rw [rowCount_eq]
  norm_num [NightstreamFPrime.Lifecycle.cubeVariables]

def plan {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (poseidonGeometry : PiCCSPoseidonPlan.Geometry program logicalWidth)
    (ordinaryGeometry :
      PiCCSOrdinaryRetainedGeometry.Geometry program logicalWidth) :
    ProductionRelation.Plan logicalWidth :=
  PinFamilyPlan.plan (interface poseidonGeometry ordinaryGeometry) rowCount_le

@[simp] theorem plan_rowCount
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (poseidonGeometry : PiCCSPoseidonPlan.Geometry program logicalWidth)
    (ordinaryGeometry :
      PiCCSOrdinaryRetainedGeometry.Geometry program logicalWidth) :
    (plan poseidonGeometry ordinaryGeometry).rowCount = 32 := by
  rfl

theorem rowsZero_iff
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (poseidonGeometry : PiCCSPoseidonPlan.Geometry program logicalWidth)
    (ordinaryGeometry :
      PiCCSOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (one : assignment
      (PiCCSOrdinaryRetainedGeometry.oneColumn ordinaryGeometry) = 1) :
    (plan poseidonGeometry ordinaryGeometry).RowsZero assignment ↔
      ∀ rowIndex,
        (bindingForm poseidonGeometry ordinaryGeometry rowIndex).eval
          assignment = 0 := by
  exact PinFamilyPlan.planRowsZero_iff
    (interface poseidonGeometry ordinaryGeometry) rowCount_le assignment one

theorem rowsZero_implies_endpointValue
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (poseidonGeometry : PiCCSPoseidonPlan.Geometry program logicalWidth)
    (ordinaryGeometry :
      PiCCSOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 33 → F)
    (products : Fin PiRLCFirst54DirectSchedule.candidateCount → F)
    (one : assignment
      (PiCCSOrdinaryRetainedGeometry.oneColumn ordinaryGeometry) = 1)
    (encoding : PiCCSOrdinaryRetainedGeometry.Encodes ordinaryGeometry
      assignment (PiRLCRetainedPreservation.sourceAssignment
        program base groupValue products))
    (rowsZero : (plan poseidonGeometry ordinaryGeometry).RowsZero assignment)
    (family : Fin familyCount) (lane : Fin laneCount) :
    PiCCSPoseidonPreservation.outputValue poseidonGeometry assignment
        (endpointInvocation family) lane =
      PiCCSActionPayloadBlock.packageEnv program
        (PiRLCRetainedPreservation.sourceAssignment
          program base groupValue products) (endpointColumn family lane) := by
  have rowZero := (rowsZero_iff poseidonGeometry ordinaryGeometry assignment
    one).mp rowsZero (row family lane)
  rw [bindingForm, descriptor_row, SparseForm.add_eval,
    SparseForm.scale_eval] at rowZero
  have formsEq :
      (directForm poseidonGeometry family lane).eval assignment =
        (sourceForm ordinaryGeometry family lane).eval assignment := by
    apply Lean.Grind.AddCommGroup.sub_eq_zero_iff.mp
    simpa [sub_eq_add_neg] using rowZero
  calc
    PiCCSPoseidonPreservation.outputValue poseidonGeometry assignment
        (endpointInvocation family) lane =
        (directForm poseidonGeometry family lane).eval assignment := by
      rfl
    _ = (sourceForm ordinaryGeometry family lane).eval assignment := formsEq
    _ = _ := sourceForm_eval ordinaryGeometry assignment base groupValue
      products encoding family lane

private theorem statementFinalState_endpoint (lane : Fin laneCount) :
    StatementAbsorption.finalState
        (PiCCSInvocations.statementInterface Data.logicalWidth Data.publicFits)
        PiCCSInvocations.statementWitnessStart lane =
      Expr.var (endpointColumn statementFamily lane) := by
  calc
    _ = (PiCCSInvocations.statementTrace Data.logicalWidth
          Data.publicFits).state lane :=
      congrFun (PiCCSInvocations.statementTrace_state_matches
        Data.logicalWidth Data.publicFits).symm lane
    _ = _ := statementTrace_state_endpoint lane

private theorem challengeFinalState_endpoint (lane : Fin laneCount) :
    ChallengeDerivation.finalState
        (PiCCSInvocations.challengeInterface Data.logicalWidth Data.publicFits)
        PiCCSInvocations.challengeWitnessStart lane =
      Expr.var (endpointColumn challengeFamily lane) := by
  calc
    _ = (PiCCSInvocations.challengeTrace Data.logicalWidth
          Data.publicFits).state lane :=
      congrFun (PiCCSInvocations.challengeTrace_state_matches
        Data.logicalWidth Data.publicFits).symm lane
    _ = _ := challengeTrace_state_endpoint lane

private theorem roundFinalState_endpoint (lane : Fin laneCount) :
    RoundTranscript.finalState
        (PiCCSInvocations.roundInterface Data.logicalWidth Data.publicFits)
        PiCCSInvocations.roundWitnessStart lane =
      Expr.var (endpointColumn roundFamily lane) := by
  calc
    _ = (PiCCSInvocations.roundTrace Data.logicalWidth
          Data.publicFits).state lane :=
      congrFun (PiCCSInvocations.roundTrace_state_matches
        Data.logicalWidth Data.publicFits).symm lane
    _ = _ := roundTrace_state_endpoint lane

private theorem roundInitialState_eq_challengeFinalState :
    (PiCCSInvocations.roundInterface Data.logicalWidth
      Data.publicFits).initialState PiCCSInvocations.roundWitnessStart =
      ChallengeDerivation.finalState
        (PiCCSInvocations.challengeInterface Data.logicalWidth Data.publicFits)
        PiCCSInvocations.challengeWitnessStart := by
  have initialEq :
      (PiCCSInvocations.challengeTrace Data.logicalWidth Data.publicFits).state =
        (PiCCSInvocations.roundInterface Data.logicalWidth
          Data.publicFits).initialState PiCCSInvocations.roundWitnessStart := by
    simpa [PiCCSInvocations.roundInterface,
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.Formal.roundTranscriptInterface,
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.Formal.challengeFinalState,
      PiCCSInvocations.sharedInterface] using
      PiCCSInvocations.challengeTrace_state_matches
        Data.logicalWidth Data.publicFits
  exact initialEq.symm.trans
    (PiCCSInvocations.challengeTrace_state_matches
      Data.logicalWidth Data.publicFits)

private theorem outputTrace_state_matches :
    (PiCCSInvocations.outputTrace Data.logicalWidth Data.publicFits).state =
      OutputBinding.finalState
        (PiCCSInvocations.outputInterface Data.logicalWidth Data.publicFits)
        PiCCSInvocations.outputWitnessStart := by
  calc
    _ = (PiCCSInvocations.outputSemanticTrace Data.logicalWidth
          Data.publicFits).state :=
      congrArg Invocations.Trace.state
        (PiCCSInvocations.outputTrace_eq_semantic
          Data.logicalWidth Data.publicFits)
    _ = _ := PiCCSInvocations.outputSemanticTrace_state_matches
      Data.logicalWidth Data.publicFits

private theorem outputInitialState_eq_roundFinalState :
    (PiCCSInvocations.outputInterface Data.logicalWidth
      Data.publicFits).initialState PiCCSInvocations.outputWitnessStart =
      RoundTranscript.finalState
        (PiCCSInvocations.roundInterface Data.logicalWidth Data.publicFits)
        PiCCSInvocations.roundWitnessStart := by
  have initialEq :
      (PiCCSInvocations.roundTrace Data.logicalWidth Data.publicFits).state =
        (PiCCSInvocations.outputInterface Data.logicalWidth
          Data.publicFits).initialState PiCCSInvocations.outputWitnessStart := by
    simpa [PiCCSInvocations.outputInterface,
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.Formal.outputBindingInterface,
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.Formal.roundTranscriptFinalState,
      PiCCSInvocations.sharedInterface] using
      PiCCSInvocations.roundTrace_state_matches Data.logicalWidth
        Data.publicFits
  exact initialEq.symm.trans
    (PiCCSInvocations.roundTrace_state_matches
      Data.logicalWidth Data.publicFits)

private theorem outputFinalState_endpoint (lane : Fin laneCount) :
    OutputBinding.finalState
        (PiCCSInvocations.outputInterface Data.logicalWidth Data.publicFits)
        PiCCSInvocations.outputWitnessStart lane =
      Expr.var (endpointColumn outputFamily lane) := by
  calc
    _ = (PiCCSInvocations.outputTrace Data.logicalWidth
          Data.publicFits).state lane := congrFun outputTrace_state_matches.symm lane
    _ = _ := outputTrace_state_endpoint lane

theorem rowsZero_implies_endpointState
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (poseidonGeometry : PiCCSPoseidonPlan.Geometry program logicalWidth)
    (ordinaryGeometry :
      PiCCSOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 33 → F)
    (products : Fin PiRLCFirst54DirectSchedule.candidateCount → F)
    (one : assignment
      (PiCCSOrdinaryRetainedGeometry.oneColumn ordinaryGeometry) = 1)
    (encoding : PiCCSOrdinaryRetainedGeometry.Encodes ordinaryGeometry
      assignment (PiRLCRetainedPreservation.sourceAssignment
        program base groupValue products))
    (rowsZero : (plan poseidonGeometry ordinaryGeometry).RowsZero assignment)
    (family : Fin familyCount) :
    PiCCSPoseidonPreservation.valueState poseidonGeometry assignment
        (endpointInvocation family) =
      List.ofFn fun lane : Fin laneCount =>
        PiCCSActionPayloadBlock.packageEnv program
          (PiRLCRetainedPreservation.sourceAssignment
            program base groupValue products) (endpointColumn family lane) := by
  unfold PiCCSPoseidonPreservation.valueState
  apply congrArg List.ofFn
  funext lane
  exact rowsZero_implies_endpointValue poseidonGeometry ordinaryGeometry
    assignment base groupValue products one encoding rowsZero family lane

private theorem endpointState_eq_finalEval
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (poseidonGeometry : PiCCSPoseidonPlan.Geometry program logicalWidth)
    (ordinaryGeometry :
      PiCCSOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 33 → F)
    (products : Fin PiRLCFirst54DirectSchedule.candidateCount → F)
    (one : assignment
      (PiCCSOrdinaryRetainedGeometry.oneColumn ordinaryGeometry) = 1)
    (encoding : PiCCSOrdinaryRetainedGeometry.Encodes ordinaryGeometry
      assignment (PiRLCRetainedPreservation.sourceAssignment
        program base groupValue products))
    (rowsZero : (plan poseidonGeometry ordinaryGeometry).RowsZero assignment)
    (family : Fin familyCount) (finalState : Layer.EState)
    (finalEndpoint : ∀ lane : Fin laneCount,
      finalState lane = Expr.var (endpointColumn family lane)) :
    PiCCSPoseidonPreservation.valueState poseidonGeometry assignment
        (endpointInvocation family) =
      List.ofFn (Layer.evalState
        (PiCCSActionPayloadBlock.packageEnv program
          (PiRLCRetainedPreservation.sourceAssignment
            program base groupValue products)) finalState) := by
  rw [rowsZero_implies_endpointState poseidonGeometry ordinaryGeometry
    assignment base groupValue products one encoding rowsZero family]
  apply congrArg List.ofFn
  funext lane
  unfold Layer.evalState
  rw [finalEndpoint lane]
  rfl

private theorem statementEndpoint_eq_finalEval
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (poseidonGeometry : PiCCSPoseidonPlan.Geometry program logicalWidth)
    (ordinaryGeometry :
      PiCCSOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 33 → F)
    (products : Fin PiRLCFirst54DirectSchedule.candidateCount → F)
    (one : assignment
      (PiCCSOrdinaryRetainedGeometry.oneColumn ordinaryGeometry) = 1)
    (encoding : PiCCSOrdinaryRetainedGeometry.Encodes ordinaryGeometry
      assignment (PiRLCRetainedPreservation.sourceAssignment
        program base groupValue products))
    (rowsZero : (plan poseidonGeometry ordinaryGeometry).RowsZero assignment) :
    PiCCSPoseidonPreservation.valueState poseidonGeometry assignment
        PiCCSTranscriptDirectSemantics.statementLast =
      List.ofFn (Layer.evalState
        (PiCCSActionPayloadBlock.packageEnv program
          (PiRLCRetainedPreservation.sourceAssignment
            program base groupValue products))
        (StatementAbsorption.finalState
          (PiCCSInvocations.statementInterface Data.logicalWidth
            Data.publicFits) PiCCSInvocations.statementWitnessStart)) := by
  simpa [endpointInvocation, statementFamily] using
    endpointState_eq_finalEval poseidonGeometry ordinaryGeometry assignment
      base groupValue products one encoding rowsZero statementFamily
      (StatementAbsorption.finalState
        (PiCCSInvocations.statementInterface Data.logicalWidth Data.publicFits)
        PiCCSInvocations.statementWitnessStart) statementFinalState_endpoint

private theorem challengeEndpoint_eq_finalEval
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (poseidonGeometry : PiCCSPoseidonPlan.Geometry program logicalWidth)
    (ordinaryGeometry :
      PiCCSOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 33 → F)
    (products : Fin PiRLCFirst54DirectSchedule.candidateCount → F)
    (one : assignment
      (PiCCSOrdinaryRetainedGeometry.oneColumn ordinaryGeometry) = 1)
    (encoding : PiCCSOrdinaryRetainedGeometry.Encodes ordinaryGeometry
      assignment (PiRLCRetainedPreservation.sourceAssignment
        program base groupValue products))
    (rowsZero : (plan poseidonGeometry ordinaryGeometry).RowsZero assignment) :
    PiCCSPoseidonPreservation.valueState poseidonGeometry assignment
        PiCCSTranscriptDirectSemantics.challengeLast =
      List.ofFn (Layer.evalState
        (PiCCSActionPayloadBlock.packageEnv program
          (PiRLCRetainedPreservation.sourceAssignment
            program base groupValue products))
        (ChallengeDerivation.finalState
          (PiCCSInvocations.challengeInterface Data.logicalWidth
            Data.publicFits) PiCCSInvocations.challengeWitnessStart)) := by
  simpa [endpointInvocation, challengeFamily] using
    endpointState_eq_finalEval poseidonGeometry ordinaryGeometry assignment
      base groupValue products one encoding rowsZero challengeFamily
      (ChallengeDerivation.finalState
        (PiCCSInvocations.challengeInterface Data.logicalWidth Data.publicFits)
        PiCCSInvocations.challengeWitnessStart) challengeFinalState_endpoint

private theorem roundEndpoint_eq_finalEval
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (poseidonGeometry : PiCCSPoseidonPlan.Geometry program logicalWidth)
    (ordinaryGeometry :
      PiCCSOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 33 → F)
    (products : Fin PiRLCFirst54DirectSchedule.candidateCount → F)
    (one : assignment
      (PiCCSOrdinaryRetainedGeometry.oneColumn ordinaryGeometry) = 1)
    (encoding : PiCCSOrdinaryRetainedGeometry.Encodes ordinaryGeometry
      assignment (PiRLCRetainedPreservation.sourceAssignment
        program base groupValue products))
    (rowsZero : (plan poseidonGeometry ordinaryGeometry).RowsZero assignment) :
    PiCCSPoseidonPreservation.valueState poseidonGeometry assignment
        PiCCSTranscriptDirectSemantics.roundLast =
      List.ofFn (Layer.evalState
        (PiCCSActionPayloadBlock.packageEnv program
          (PiRLCRetainedPreservation.sourceAssignment
            program base groupValue products))
        (RoundTranscript.finalState
          (PiCCSInvocations.roundInterface Data.logicalWidth Data.publicFits)
          PiCCSInvocations.roundWitnessStart)) := by
  simpa [endpointInvocation, roundFamily] using
    endpointState_eq_finalEval poseidonGeometry ordinaryGeometry assignment
      base groupValue products one encoding rowsZero roundFamily
      (RoundTranscript.finalState
        (PiCCSInvocations.roundInterface Data.logicalWidth Data.publicFits)
        PiCCSInvocations.roundWitnessStart) roundFinalState_endpoint

/-- The retained final PiCCS Poseidon2 state is the exact semantic output
binding state selected by the endpoint rows. -/
theorem outputEndpoint_eq_finalEval
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (poseidonGeometry : PiCCSPoseidonPlan.Geometry program logicalWidth)
    (ordinaryGeometry :
      PiCCSOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 33 → F)
    (products : Fin PiRLCFirst54DirectSchedule.candidateCount → F)
    (one : assignment
      (PiCCSOrdinaryRetainedGeometry.oneColumn ordinaryGeometry) = 1)
    (encoding : PiCCSOrdinaryRetainedGeometry.Encodes ordinaryGeometry
      assignment (PiRLCRetainedPreservation.sourceAssignment
        program base groupValue products))
    (rowsZero : (plan poseidonGeometry ordinaryGeometry).RowsZero assignment) :
    PiCCSPoseidonPreservation.valueState poseidonGeometry assignment
        PiCCSTranscriptDirectSemantics.outputLast =
      List.ofFn (Layer.evalState
        (PiCCSActionPayloadBlock.packageEnv program
          (PiRLCRetainedPreservation.sourceAssignment
            program base groupValue products))
        (OutputBinding.finalState
          (PiCCSInvocations.outputInterface Data.logicalWidth Data.publicFits)
          PiCCSInvocations.outputWitnessStart)) := by
  simpa [endpointInvocation, outputFamily] using
    endpointState_eq_finalEval poseidonGeometry ordinaryGeometry assignment
      base groupValue products one encoding rowsZero outputFamily
      (OutputBinding.finalState
        (PiCCSInvocations.outputInterface Data.logicalWidth Data.publicFits)
        PiCCSInvocations.outputWitnessStart) outputFinalState_endpoint

def transcriptEnv (program : Lifecycle.Stage1.Application.Program)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 33 → F)
    (products : Fin PiRLCFirst54DirectSchedule.candidateCount → F) : Env :=
  PerApplicationPackage.baseEnv program <| SourceCompiler.sourceEnv <|
    PiRLCRetainedPreservation.sourceAssignment
      program base groupValue products

theorem transcriptEnv_eq_transitionEnv_of_lt
    (program : Lifecycle.Stage1.Application.Program)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 33 → F)
    (products : Fin PiRLCFirst54DirectSchedule.candidateCount → F)
    (column : Nat) (bound : column < Spartan.spartanColumnCount) :
    transcriptEnv program base groupValue products column =
      RunningTransitionDirectPlan.transitionEnv program base column := by
  have packageTotal :
      PiRLCProductPlan.basePackage.layout.totalColumnCount = 29336725 := by
    change PerApplicationPackage.basePackage.layout.totalColumnCount = 29336725
    exact Package.circuitPackage_layout_values.2.2.2.2
  have packageBound :
      column < PiRLCProductPlan.basePackage.layout.totalColumnCount := by
    rw [packageTotal]
    simpa [Spartan.spartanColumnCount_eq] using bound
  have shiftedPrefix :
      PerApplicationPackage.shiftColumn program column <
        PiCCSActionPayloadBlock.prefixSourceWidth program := by
    unfold PiCCSActionPayloadBlock.prefixSourceWidth
      PiRLCRetainedGeometry.sourceWidth
      PiRLCFirst54DirectPlan.sourceWidth
      PiRLCFirst54DirectPlan.prefixSourceWidth
      PiRLCProductPlan.sourceWidth ProductRetainedBlock.sourceWidth
      FieldSuffixBlock.sourceWidth
    have shiftedBase := PiRLCProductPlan.shiftColumn_lt_baseSourceWidth
      program column packageBound
    omega
  have shiftedRetained :
      PerApplicationPackage.shiftColumn program column <
        PiRLCRetainedGeometry.sourceWidth program := shiftedPrefix
  unfold transcriptEnv RunningTransitionDirectPlan.transitionEnv
    PerApplicationPackage.baseEnv SourceCompiler.sourceEnv
  rw [dif_pos shiftedRetained, dif_pos packageBound]
  rw [show
      (⟨PerApplicationPackage.shiftColumn program column, shiftedRetained⟩ :
        Fin (PiRLCRetainedGeometry.sourceWidth program)) =
        PiRLCRetainedPreservation.baseSourceColumn program
          (PiRLCProductPlan.shiftedPackageColumn program column packageBound) by
    apply Fin.ext
    rfl]
  rw [PiRLCRetainedPreservation.sourceAssignment_base]

private theorem zeroState_eq_evalZero (env : Env) :
    Spec.Poseidon2.zeroState =
      List.ofFn (Layer.evalState env Hash.zeroE) := by
  unfold Spec.Poseidon2.zeroState Layer.evalState Hash.zeroE
  change List.replicate 8 0 = List.ofFn (fun _ : Fin 8 => (0 : F))
  norm_num [List.ofFn_succ]
  rfl

private theorem statementActions_eq_of_shape
    (relationLogicalWidth : Nat)
    (relationPublicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth relationLogicalWidth) :
    PiCCSInvocations.statementActions relationLogicalWidth
        relationPublicFits =
      PiCCSActionPayloadBlock.statementActions := by
  rfl

private theorem statementTrace_state_endpoint_of_shape
    (relationLogicalWidth : Nat)
    (relationPublicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth relationLogicalWidth)
    (lane : Fin laneCount) :
    (PiCCSInvocations.statementTrace relationLogicalWidth
      relationPublicFits).state lane =
      Expr.var (endpointColumn statementFamily lane) := by
  have nonempty : PiCCSInvocations.statementActions relationLogicalWidth
      relationPublicFits ≠ [] := by
    rw [statementActions_eq_of_shape]
    exact statementActions_nonempty
  have positive : InvocationLastOutput.ActionsPositive
      (PiCCSInvocations.statementActions relationLogicalWidth
        relationPublicFits) := by
    rw [statementActions_eq_of_shape]
    exact statementActions_positive
  have compiled := InvocationLastOutput.compileActions_state_scheduleOutput
    PiCCSInvocations.statementPhase PiCCSInvocations.statementRowStart
    PiCCSInvocations.statementWitnessStart Hash.zeroE
    (PiCCSInvocations.statementActions relationLogicalWidth
      relationPublicFits) nonempty positive
  change
    (PiCCSInvocations.statementTrace relationLogicalWidth
        relationPublicFits).state =
      Permutation.scheduleOutput
        (PiCCSInvocations.statementWitnessStart +
          (Invocations.invocationCount
            (PiCCSInvocations.statementActions relationLogicalWidth
              relationPublicFits) - 1) * 592) at compiled
  have count := PiCCSInvocations.statementInvocationCount_eq
    relationLogicalWidth relationPublicFits
  have endEq := PiCCSInvocations.statementEnd_eq_challengeStart
    relationLogicalWidth relationPublicFits
  rw [count] at compiled endEq
  have startEq :
      PiCCSInvocations.statementWitnessStart + (379 - 1) * 592 + 584 =
        PiCCSInvocations.challengeWitnessStart - 8 := by
    rw [← endEq]
    generalize PiCCSInvocations.statementWitnessStart = start
    omega
  rw [congrFun compiled lane]
  unfold Permutation.scheduleOutput Permutation.freshState endpointColumn
  rw [show endpointStart statementFamily =
      PiCCSInvocations.challengeWitnessStart - 8 by rfl]
  rw [startEq]

private theorem statementFinalState_endpoint_of_shape
    (relationLogicalWidth : Nat)
    (relationPublicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth relationLogicalWidth)
    (lane : Fin laneCount) :
    StatementAbsorption.finalState
        (PiCCSInvocations.statementInterface relationLogicalWidth
          relationPublicFits)
        PiCCSInvocations.statementWitnessStart lane =
      Expr.var (endpointColumn statementFamily lane) := by
  calc
    _ = (PiCCSInvocations.statementTrace relationLogicalWidth
          relationPublicFits).state lane :=
      congrFun (PiCCSInvocations.statementTrace_state_matches
        relationLogicalWidth relationPublicFits).symm lane
    _ = _ := statementTrace_state_endpoint_of_shape relationLogicalWidth
      relationPublicFits lane

private theorem challengeActions_eq_of_shape
    (relationLogicalWidth : Nat)
    (relationPublicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth relationLogicalWidth) :
    ChallengeDerivation.actions
        (PiCCSInvocations.challengeInterface relationLogicalWidth
          relationPublicFits)
        PiCCSInvocations.challengeWitnessStart =
      PiCCSActionPayloadBlock.challengeActions := by
  rfl

private theorem challengeTrace_state_endpoint_of_shape
    (relationLogicalWidth : Nat)
    (relationPublicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth relationLogicalWidth)
    (lane : Fin laneCount) :
    (PiCCSInvocations.challengeTrace relationLogicalWidth
      relationPublicFits).state lane =
      Expr.var (endpointColumn challengeFamily lane) := by
  rw [PiCCSInvocations.challengeTrace_eq_semantic]
  have nonempty : ChallengeDerivation.actions
      (PiCCSInvocations.challengeInterface relationLogicalWidth
        relationPublicFits) PiCCSInvocations.challengeWitnessStart ≠ [] := by
    rw [challengeActions_eq_of_shape]
    exact challengeActions_nonempty
  have positive : InvocationLastOutput.ActionsPositive
      (ChallengeDerivation.actions
        (PiCCSInvocations.challengeInterface relationLogicalWidth
          relationPublicFits) PiCCSInvocations.challengeWitnessStart) := by
    rw [challengeActions_eq_of_shape]
    exact challengeActions_positive
  have compiled := InvocationLastOutput.compileActions_state_scheduleOutput
    PiCCSInvocations.challengePhase PiCCSInvocations.challengeRowStart
    PiCCSInvocations.challengeWitnessStart
    ((PiCCSInvocations.challengeInterface relationLogicalWidth
      relationPublicFits).initialState PiCCSInvocations.challengeWitnessStart)
    (ChallengeDerivation.actions
      (PiCCSInvocations.challengeInterface relationLogicalWidth
        relationPublicFits) PiCCSInvocations.challengeWitnessStart)
    nonempty positive
  change
    (PiCCSInvocations.challengeSemanticTrace relationLogicalWidth
        relationPublicFits).state =
      Permutation.scheduleOutput
        (PiCCSInvocations.challengeWitnessStart +
          (Invocations.invocationCount
            (ChallengeDerivation.actions
              (PiCCSInvocations.challengeInterface relationLogicalWidth
                relationPublicFits)
              PiCCSInvocations.challengeWitnessStart) - 1) * 592) at compiled
  have count : Invocations.invocationCount
      (ChallengeDerivation.actions
        (PiCCSInvocations.challengeInterface relationLogicalWidth
          relationPublicFits) PiCCSInvocations.challengeWitnessStart) = 87 := by
    rw [challengeActions_eq_of_shape]
    exact PiCCSActionPayloadBlock.challengeInvocationCount_eq
  have endEq := PiCCSInvocations.challengeEnd_eq_roundStart
    relationLogicalWidth relationPublicFits
  rw [count] at compiled
  rw [PiCCSInvocations.challengeInvocationCount_eq] at endEq
  have startEq :
      PiCCSInvocations.challengeWitnessStart + (87 - 1) * 592 + 584 =
        PiCCSInvocations.roundWitnessStart - 8 := by
    rw [← endEq]
    generalize PiCCSInvocations.challengeWitnessStart = start
    omega
  rw [congrFun compiled lane]
  unfold Permutation.scheduleOutput Permutation.freshState endpointColumn
  rw [show endpointStart challengeFamily =
      PiCCSInvocations.roundWitnessStart - 8 by rfl]
  rw [startEq]

private theorem challengeFinalState_endpoint_of_shape
    (relationLogicalWidth : Nat)
    (relationPublicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth relationLogicalWidth)
    (lane : Fin laneCount) :
    ChallengeDerivation.finalState
        (PiCCSInvocations.challengeInterface relationLogicalWidth
          relationPublicFits)
        PiCCSInvocations.challengeWitnessStart lane =
      Expr.var (endpointColumn challengeFamily lane) := by
  calc
    _ = (PiCCSInvocations.challengeTrace relationLogicalWidth
          relationPublicFits).state lane :=
      congrFun (PiCCSInvocations.challengeTrace_state_matches
        relationLogicalWidth relationPublicFits).symm lane
    _ = _ := challengeTrace_state_endpoint_of_shape relationLogicalWidth
      relationPublicFits lane

private theorem roundActions_eq_of_shape
    (relationLogicalWidth : Nat)
    (relationPublicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth relationLogicalWidth) :
    RoundTranscript.actions
        (PiCCSInvocations.roundInterface relationLogicalWidth
          relationPublicFits)
        PiCCSInvocations.roundWitnessStart =
      PiCCSActionPayloadBlock.roundActions := by
  rfl

private theorem roundTrace_state_endpoint_of_shape
    (relationLogicalWidth : Nat)
    (relationPublicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth relationLogicalWidth)
    (lane : Fin laneCount) :
    (PiCCSInvocations.roundTrace relationLogicalWidth
      relationPublicFits).state lane =
      Expr.var (endpointColumn roundFamily lane) := by
  rw [PiCCSInvocations.roundTrace_eq_semantic]
  have nonempty : RoundTranscript.actions
      (PiCCSInvocations.roundInterface relationLogicalWidth
        relationPublicFits) PiCCSInvocations.roundWitnessStart ≠ [] := by
    rw [roundActions_eq_of_shape]
    exact roundActions_nonempty
  have positive : InvocationLastOutput.ActionsPositive
      (RoundTranscript.actions
        (PiCCSInvocations.roundInterface relationLogicalWidth
          relationPublicFits) PiCCSInvocations.roundWitnessStart) := by
    rw [roundActions_eq_of_shape]
    exact roundActions_positive
  have compiled := InvocationLastOutput.compileActions_state_scheduleOutput
    PiCCSInvocations.roundPhase PiCCSInvocations.roundRowStart
    PiCCSInvocations.roundWitnessStart
    ((PiCCSInvocations.roundInterface relationLogicalWidth
      relationPublicFits).initialState PiCCSInvocations.roundWitnessStart)
    (RoundTranscript.actions
      (PiCCSInvocations.roundInterface relationLogicalWidth
        relationPublicFits) PiCCSInvocations.roundWitnessStart)
    nonempty positive
  change
    (PiCCSInvocations.roundSemanticTrace relationLogicalWidth
        relationPublicFits).state =
      Permutation.scheduleOutput
        (PiCCSInvocations.roundWitnessStart +
          (Invocations.invocationCount
            (RoundTranscript.actions
              (PiCCSInvocations.roundInterface relationLogicalWidth
                relationPublicFits)
              PiCCSInvocations.roundWitnessStart) - 1) * 592) at compiled
  have count : Invocations.invocationCount
      (RoundTranscript.actions
        (PiCCSInvocations.roundInterface relationLogicalWidth
          relationPublicFits) PiCCSInvocations.roundWitnessStart) = 252 := by
    rw [roundActions_eq_of_shape]
    exact PiCCSActionPayloadBlock.roundInvocationCount_eq
  rw [count] at compiled
  have startEq :
      PiCCSInvocations.roundWitnessStart + (252 - 1) * 592 + 584 =
        PiCCSInvocations.roundWitnessStart + 252 * 592 - 8 := by
    generalize PiCCSInvocations.roundWitnessStart = start
    omega
  rw [congrFun compiled lane]
  unfold Permutation.scheduleOutput Permutation.freshState endpointColumn
  rw [show endpointStart roundFamily =
      PiCCSInvocations.roundWitnessStart +
        PiCCSTranscriptDirectSemantics.roundCount * 592 - 8 by rfl]
  rw [show PiCCSTranscriptDirectSemantics.roundCount = 252 by rfl]
  rw [startEq]

private theorem roundFinalState_endpoint_of_shape
    (relationLogicalWidth : Nat)
    (relationPublicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth relationLogicalWidth)
    (lane : Fin laneCount) :
    RoundTranscript.finalState
        (PiCCSInvocations.roundInterface relationLogicalWidth
          relationPublicFits)
        PiCCSInvocations.roundWitnessStart lane =
      Expr.var (endpointColumn roundFamily lane) := by
  calc
    _ = (PiCCSInvocations.roundTrace relationLogicalWidth
          relationPublicFits).state lane :=
      congrFun (PiCCSInvocations.roundTrace_state_matches
        relationLogicalWidth relationPublicFits).symm lane
    _ = _ := roundTrace_state_endpoint_of_shape relationLogicalWidth
      relationPublicFits lane

private theorem outputActions_eq_of_shape
    (relationLogicalWidth : Nat)
    (relationPublicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth relationLogicalWidth) :
    PiCCSInvocations.outputActions relationLogicalWidth relationPublicFits =
      PiCCSActionPayloadBlock.outputActions := by
  rfl

private theorem outputTrace_state_endpoint_of_shape
    (relationLogicalWidth : Nat)
    (relationPublicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth relationLogicalWidth)
    (lane : Fin laneCount) :
    (PiCCSInvocations.outputTrace relationLogicalWidth
      relationPublicFits).state lane =
      Expr.var (endpointColumn outputFamily lane) := by
  have nonempty : PiCCSInvocations.outputActions relationLogicalWidth
      relationPublicFits ≠ [] := by
    rw [outputActions_eq_of_shape]
    exact outputActions_nonempty
  have positive : InvocationLastOutput.ActionsPositive
      (PiCCSInvocations.outputActions relationLogicalWidth
        relationPublicFits) := by
    rw [outputActions_eq_of_shape]
    exact outputActions_positive
  have compiled := InvocationLastOutput.compileActions_state_scheduleOutput
    PiCCSInvocations.outputPhase PiCCSInvocations.outputRowStart
    PiCCSInvocations.outputWitnessStart
    (PiCCSInvocations.roundTrace relationLogicalWidth relationPublicFits).state
    (PiCCSInvocations.outputActions relationLogicalWidth relationPublicFits)
    nonempty positive
  change
    (PiCCSInvocations.outputTrace relationLogicalWidth
        relationPublicFits).state =
      Permutation.scheduleOutput
        (PiCCSInvocations.outputWitnessStart +
          (Invocations.invocationCount
            (PiCCSInvocations.outputActions relationLogicalWidth
              relationPublicFits) - 1) * 592) at compiled
  have count := PiCCSInvocations.outputInvocationCount_eq
    relationLogicalWidth relationPublicFits
  have endEq := PiCCSInvocations.outputEnd_eq_logicalFreshBase
    relationLogicalWidth relationPublicFits
  rw [count] at compiled endEq
  have startEq :
      PiCCSInvocations.outputWitnessStart + (6886 - 1) * 592 + 584 =
        PiCCSStarts.logicalFreshBase - 8 := by
    rw [← endEq]
    generalize PiCCSInvocations.outputWitnessStart = start
    omega
  rw [congrFun compiled lane]
  unfold Permutation.scheduleOutput Permutation.freshState endpointColumn
  rw [show endpointStart outputFamily = PiCCSStarts.logicalFreshBase - 8 by rfl]
  rw [startEq]

private theorem outputTrace_state_matches_of_shape
    (relationLogicalWidth : Nat)
    (relationPublicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth relationLogicalWidth) :
    (PiCCSInvocations.outputTrace relationLogicalWidth
      relationPublicFits).state =
      OutputBinding.finalState
        (PiCCSInvocations.outputInterface relationLogicalWidth
          relationPublicFits) PiCCSInvocations.outputWitnessStart := by
  calc
    _ = (PiCCSInvocations.outputSemanticTrace relationLogicalWidth
          relationPublicFits).state :=
      congrArg Invocations.Trace.state
        (PiCCSInvocations.outputTrace_eq_semantic relationLogicalWidth
          relationPublicFits)
    _ = _ := PiCCSInvocations.outputSemanticTrace_state_matches
      relationLogicalWidth relationPublicFits

private theorem outputFinalState_endpoint_of_shape
    (relationLogicalWidth : Nat)
    (relationPublicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth relationLogicalWidth)
    (lane : Fin laneCount) :
    OutputBinding.finalState
        (PiCCSInvocations.outputInterface relationLogicalWidth
          relationPublicFits)
        PiCCSInvocations.outputWitnessStart lane =
      Expr.var (endpointColumn outputFamily lane) := by
  calc
    _ = (PiCCSInvocations.outputTrace relationLogicalWidth
          relationPublicFits).state lane :=
      congrFun (outputTrace_state_matches_of_shape relationLogicalWidth
        relationPublicFits).symm lane
    _ = _ := outputTrace_state_endpoint_of_shape relationLogicalWidth
      relationPublicFits lane

private theorem roundInitialState_eq_challengeFinalState_of_shape
    (relationLogicalWidth : Nat)
    (relationPublicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth relationLogicalWidth) :
    (PiCCSInvocations.roundInterface relationLogicalWidth
      relationPublicFits).initialState PiCCSInvocations.roundWitnessStart =
      ChallengeDerivation.finalState
        (PiCCSInvocations.challengeInterface relationLogicalWidth
          relationPublicFits) PiCCSInvocations.challengeWitnessStart := by
  have initialEq :
      (PiCCSInvocations.challengeTrace relationLogicalWidth
        relationPublicFits).state =
        (PiCCSInvocations.roundInterface relationLogicalWidth
          relationPublicFits).initialState
            PiCCSInvocations.roundWitnessStart := by
    simpa [PiCCSInvocations.roundInterface,
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.Formal.roundTranscriptInterface,
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.Formal.challengeFinalState,
      PiCCSInvocations.sharedInterface] using
      PiCCSInvocations.challengeTrace_state_matches relationLogicalWidth
        relationPublicFits
  exact initialEq.symm.trans
    (PiCCSInvocations.challengeTrace_state_matches relationLogicalWidth
      relationPublicFits)

private theorem outputInitialState_eq_roundFinalState_of_shape
    (relationLogicalWidth : Nat)
    (relationPublicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth relationLogicalWidth) :
    (PiCCSInvocations.outputInterface relationLogicalWidth
      relationPublicFits).initialState PiCCSInvocations.outputWitnessStart =
      RoundTranscript.finalState
        (PiCCSInvocations.roundInterface relationLogicalWidth
          relationPublicFits) PiCCSInvocations.roundWitnessStart := by
  have initialEq :
      (PiCCSInvocations.roundTrace relationLogicalWidth
        relationPublicFits).state =
        (PiCCSInvocations.outputInterface relationLogicalWidth
          relationPublicFits).initialState
            PiCCSInvocations.outputWitnessStart := by
    simpa [PiCCSInvocations.outputInterface,
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.Formal.outputBindingInterface,
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.Formal.roundTranscriptFinalState,
      PiCCSInvocations.sharedInterface] using
      PiCCSInvocations.roundTrace_state_matches relationLogicalWidth
        relationPublicFits
  exact initialEq.symm.trans
    (PiCCSInvocations.roundTrace_state_matches relationLogicalWidth
      relationPublicFits)

theorem traces_and_rowsZero_imply_transcriptSpecs
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    {relationLogicalWidth : Nat}
    {relationPublicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth relationLogicalWidth}
    (poseidonGeometry : PiCCSPoseidonPlan.Geometry program logicalWidth)
    (ordinaryGeometry :
      PiCCSOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 33 → F)
    (products : Fin PiRLCFirst54DirectSchedule.candidateCount → F)
    (one : assignment
      (PiCCSOrdinaryRetainedGeometry.oneColumn ordinaryGeometry) = 1)
    (encoding : PiCCSOrdinaryRetainedGeometry.Encodes ordinaryGeometry
      assignment (PiRLCRetainedPreservation.sourceAssignment
        program base groupValue products))
    (traces : PiCCSTranscriptDirectSemantics.Traces poseidonGeometry assignment
      (PiRLCRetainedPreservation.sourceAssignment
        program base groupValue products))
    (rowsZero : (plan poseidonGeometry ordinaryGeometry).RowsZero assignment) :
    PiCCSInvocations.TranscriptSpecs relationLogicalWidth relationPublicFits
      (transcriptEnv program base groupValue products) := by
  have statementEndpoint := endpointState_eq_finalEval poseidonGeometry
    ordinaryGeometry assignment base groupValue products one encoding rowsZero
    statementFamily
    (StatementAbsorption.finalState
      (PiCCSInvocations.statementInterface relationLogicalWidth
        relationPublicFits) PiCCSInvocations.statementWitnessStart)
    (statementFinalState_endpoint_of_shape relationLogicalWidth
      relationPublicFits)
  have challengeEndpoint := endpointState_eq_finalEval poseidonGeometry
    ordinaryGeometry assignment base groupValue products one encoding rowsZero
    challengeFamily
    (ChallengeDerivation.finalState
      (PiCCSInvocations.challengeInterface relationLogicalWidth
        relationPublicFits) PiCCSInvocations.challengeWitnessStart)
    (challengeFinalState_endpoint_of_shape relationLogicalWidth
      relationPublicFits)
  have roundEndpoint := endpointState_eq_finalEval poseidonGeometry
    ordinaryGeometry assignment base groupValue products one encoding rowsZero
    roundFamily
    (RoundTranscript.finalState
      (PiCCSInvocations.roundInterface relationLogicalWidth
        relationPublicFits) PiCCSInvocations.roundWitnessStart)
    (roundFinalState_endpoint_of_shape relationLogicalWidth
      relationPublicFits)
  have outputEndpoint := endpointState_eq_finalEval poseidonGeometry
    ordinaryGeometry assignment base groupValue products one encoding rowsZero
    outputFamily
    (OutputBinding.finalState
      (PiCCSInvocations.outputInterface relationLogicalWidth
        relationPublicFits) PiCCSInvocations.outputWitnessStart)
    (outputFinalState_endpoint_of_shape relationLogicalWidth
      relationPublicFits)
  simp [endpointInvocation, statementFamily] at statementEndpoint
  simp [endpointInvocation, challengeFamily] at challengeEndpoint
  simp [endpointInvocation, roundFamily] at roundEndpoint
  simp [endpointInvocation, outputFamily] at outputEndpoint
  refine {
    statementAbsorption := ?_
    challengeDerivation := ?_
    roundTranscript := ?_
    outputBinding := ?_ }
  · unfold StatementAbsorption.SpecHolds
    have trace := traces.statement
    rw [statementEndpoint] at trace
    rw [zeroState_eq_evalZero
      (PiCCSActionPayloadBlock.packageEnv program
        (PiRLCRetainedPreservation.sourceAssignment
          program base groupValue products))] at trace
    simpa [transcriptEnv, PiCCSActionPayloadBlock.packageEnv,
      PiCCSActionPayloadBlock.statementActions,
      PiCCSInvocations.statementActions] using trace
  · apply ChallengeDerivation.trace_implies_specHolds
    have trace := traces.challenge
    rw [statementEndpoint, challengeEndpoint] at trace
    rw [← PiCCSInvocations.challengeInitialState_eq_statementFinalState
      relationLogicalWidth relationPublicFits] at trace
    simpa [transcriptEnv, PiCCSActionPayloadBlock.packageEnv,
      PiCCSActionPayloadBlock.challengeActions] using trace
  · apply (RoundTranscript.trace_iff_specHolds
      (PiCCSInvocations.roundInterface relationLogicalWidth relationPublicFits)
      PiCCSInvocations.roundWitnessStart
      (Spartan.pullback (transcriptEnv program base groupValue products))).mp
    have trace := traces.rounds
    rw [challengeEndpoint, roundEndpoint] at trace
    rw [← roundInitialState_eq_challengeFinalState_of_shape
      relationLogicalWidth relationPublicFits] at trace
    simpa [transcriptEnv, PiCCSActionPayloadBlock.packageEnv,
      PiCCSActionPayloadBlock.roundActions] using trace
  · apply OutputBinding.trace_implies_specHolds
    have trace := traces.output
    rw [roundEndpoint, outputEndpoint] at trace
    rw [← outputInitialState_eq_roundFinalState_of_shape
      relationLogicalWidth relationPublicFits] at trace
    simpa [transcriptEnv, PiCCSActionPayloadBlock.packageEnv,
      PiCCSActionPayloadBlock.outputActions,
      PiCCSInvocations.outputActions] using trace

end NightstreamFPrime.Export.Stage1.PiCCSTranscriptEndpointPlan
