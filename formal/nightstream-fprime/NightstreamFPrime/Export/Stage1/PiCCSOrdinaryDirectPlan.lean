import NightstreamFPrime.Export.Stage1.PiCCSOrdinaryDirectSupport
import NightstreamFPrime.Export.Stage1.PiCCSOrdinaryRetainedGeometry
import NightstreamFPrime.Export.Stage1.RunningTransitionDirectPlan

/-!
Owns the executable source resolver and direct 14-matrix plan for the
canonical PiCCS ordinary rows.

The resolver decodes each Spartan column back to its Lean source column and
selects one of the six retained source families. This module does not append
the plan to the Stage 1 package or close PiCCS conformance.
-/

namespace NightstreamFPrime.Export.Stage1.PiCCSOrdinaryDirectPlan

open NightstreamFPrime.Circuit
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra
open PiCCSOrdinaryRetainedBlocks
open PiCCSOrdinaryRetainedGeometry

/-- One exact retained PiCCS source before the Spartan permutation. -/
inductive Location where
  | priorInput (index : Fin PilotProduction.stateHashWords)
  | freshPublicInput (index : Fin 270)
  | outputInput (index : Fin PilotProduction.stateHashWords)
  | expectedContext (index : Fin PiCCSInputs.expectedContextWords)
  | proofLogical (index : Fin proofLogicalCount)
  | fresh (index : Fin freshCount)

namespace Location

def sourceColumn : Location → Nat
  | .priorInput index => PilotProduction.priorPreimageStart + index.val
  | .freshPublicInput index => PilotProduction.priorPublicInputStart + index.val
  | .outputInput index => PilotProduction.outputPreimageStart + index.val
  | .expectedContext index => PiCCSInputs.expectedContextStart + index.val
  | .proofLogical index => proofLogicalSource index
  | .fresh index => PiCCSArithmetic.initialClaimFreshStart + index.val

theorem sourceSupport (location : Location) :
    PiCCSOrdinarySourceSupport.Source location.sourceColumn := by
  cases location with
  | priorInput index =>
      apply PiCCSOrdinarySourceSupport.external_source
      apply PiCCSOrdinarySourceSupport.external_prior
      constructor
      · change PilotProduction.priorPreimageStart ≤
          PilotProduction.priorPreimageStart + index.val
        omega
      · have bound := index.isLt
        change PilotProduction.priorPreimageStart + index.val <
          PilotProduction.priorPreimageStart + PilotProduction.stateHashWords
        omega
  | freshPublicInput index =>
      apply PiCCSOrdinarySourceSupport.external_source
      apply PiCCSOrdinarySourceSupport.external_public
      constructor
      · change PilotProduction.priorPublicInputStart ≤
          PilotProduction.priorPublicInputStart + index.val
        omega
      · have bound := index.isLt
        change PilotProduction.priorPublicInputStart + index.val <
          PilotProduction.priorPublicInputStart + 270
        omega
  | outputInput index =>
      apply PiCCSOrdinarySourceSupport.external_source
      apply PiCCSOrdinarySourceSupport.external_output
      constructor
      · change PilotProduction.outputPreimageStart ≤
          PilotProduction.outputPreimageStart + index.val
        omega
      · have bound := index.isLt
        change PilotProduction.outputPreimageStart + index.val <
          PilotProduction.outputPreimageStart + PilotProduction.stateHashWords
        omega
  | expectedContext index =>
      apply PiCCSOrdinarySourceSupport.external_source
      apply PiCCSOrdinarySourceSupport.external_context
      constructor
      · change PiCCSInputs.expectedContextStart ≤
          PiCCSInputs.expectedContextStart + index.val
        omega
      · have bound := index.isLt
        change PiCCSInputs.expectedContextStart + index.val <
          PiCCSInputs.expectedContextStart + PiCCSInputs.expectedContextWords
        omega
  | proofLogical index =>
      exact proofLogicalSource_support index
  | fresh index =>
      have bound := index.isLt
      change index.val < 731605 at bound
      apply PiCCSOrdinarySourceSupport.fresh_source
      · change PiCCSStarts.initialClaimFreshStart ≤
          PiCCSArithmetic.initialClaimFreshStart + index.val
        unfold PiCCSArithmetic.initialClaimFreshStart
        omega
      · change PiCCSArithmetic.initialClaimFreshStart + index.val <
          PiRLCInputs.phaseOffset
        unfold PiCCSArithmetic.initialClaimFreshStart
        unfold PiCCSStarts.initialClaimFreshStart
          PiCCSStarts.roundTranscriptFreshStart PiCCSStarts.challengeFreshStart
          PiCCSStarts.statementAbsorptionFreshStart
          PiCCSStarts.statementBindingFreshStart PiCCSStarts.logicalFreshBase
        rw [PiCCSInputs.phaseOffset_eq]
        norm_num [PiRLCInputs.phaseOffset]
        omega

theorem sourceColumn_lt (location : Location) :
    location.sourceColumn < Spartan.SourceColumnCount :=
  PiCCSOrdinarySourceSupport.source_lt_sourceColumnCount location.sourceSupport

def form {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (geometry : Geometry program logicalWidth) :
    Location → SparseForm logicalWidth
  | .priorInput index => (priorInputBlock program).form
      (priorInputStart program) (priorInputFits geometry) index
  | .freshPublicInput index => (freshPublicInputBlock program).form
      (freshPublicInputStart program) (freshPublicInputFits geometry) index
  | .outputInput index => (outputInputBlock program).form
      (outputInputStart program) (outputInputFits geometry) index
  | .expectedContext index => (expectedContextBlock program).form
      (expectedContextStart program) (expectedContextFits geometry) index
  | .proofLogical index => (proofLogicalBlock program).form
      (proofLogicalStart program) (proofLogicalFits geometry) index
  | .fresh index => (freshBlock program).form
      (freshStart program) (freshFits geometry) index

/-- Every selected form reconstructs the nested source coordinate for the
exact pre-Spartan column owned by the location. -/
theorem form_eval {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (geometry : Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (source : Fin (sourceWidth program) → F)
    (encodes : Encodes geometry assignment source) (location : Location) :
    (location.form geometry).eval assignment =
      source (RunningTransitionRetainedBlocks.packageSourceColumn program
        location.sourceColumn location.sourceColumn_lt) := by
  cases location with
  | priorInput index =>
      rw [form, LowNormBlock.Block.form_eval _ _ _ assignment source
        encodes.priorInput]
      apply congrArg source
      apply Fin.ext
      rfl
  | freshPublicInput index =>
      rw [form, LowNormBlock.Block.form_eval _ _ _ assignment source
        encodes.freshPublicInput]
      apply congrArg source
      apply Fin.ext
      rfl
  | outputInput index =>
      rw [form, LowNormBlock.Block.form_eval _ _ _ assignment source
        encodes.outputInput]
      apply congrArg source
      apply Fin.ext
      rfl
  | expectedContext index =>
      rw [form, LowNormBlock.Block.form_eval _ _ _ assignment source
        encodes.expectedContext]
      apply congrArg source
      apply Fin.ext
      rfl
  | proofLogical index =>
      rw [form, LowNormBlock.Block.form_eval _ _ _ assignment source
        encodes.proofLogical]
      apply congrArg source
      apply Fin.ext
      simp [proofLogicalBlock, Location.sourceColumn]
  | fresh index =>
      rw [form, LowNormBlock.Block.form_eval _ _ _ assignment source
        encodes.fresh]
      apply congrArg source
      apply Fin.ext
      rfl

end Location

def rangeIndex {start count column : Nat}
    (inside : PiCCSOrdinarySourceSupport.InRange start count column) :
    Fin count :=
  ⟨column - start, by
    unfold PiCCSOrdinarySourceSupport.InRange at inside
    omega⟩

@[simp] theorem rangeIndex_source {start count column : Nat}
    (inside : PiCCSOrdinarySourceSupport.InRange start count column) :
    start + (rangeIndex inside).val = column := by
  change start + (column - start) = column
  unfold PiCCSOrdinarySourceSupport.InRange at inside
  omega

local instance (start count column : Nat) : Decidable
    (PiCCSOrdinarySourceSupport.InRange start count column) := by
  unfold PiCCSOrdinarySourceSupport.InRange
  infer_instance

structure Located (column : Nat) where
  location : Location
  owns : location.sourceColumn = column

def proofInputLocated {column : Nat}
    (inside : PiCCSOrdinarySourceSupport.InRange PiCCSInputs.proofInputStart
      proofInputCount column) : Located column :=
  ⟨.proofLogical (proofInputSlot (rangeIndex inside)), by
    rw [Location.sourceColumn, proofLogicalSource_proofInput,
      rangeIndex_source inside]⟩

def transcriptColumnStart : Nat := PiCCSInputs.phaseOffset + 584

def transcriptOffset (column : Nat) : Nat := column - transcriptColumnStart

def transcriptInvocationIndex (column : Nat)
    (bounded : transcriptOffset column / 592 < transcriptInvocationCount) :
    Fin transcriptInvocationCount :=
  ⟨transcriptOffset column / 592, bounded⟩

def transcriptLaneIndex (column : Nat)
    (bounded : transcriptOffset column % 592 < Spec.Poseidon2.width) :
    Fin Spec.Poseidon2.width :=
  ⟨transcriptOffset column % 592, bounded⟩

private theorem transcriptDecoded_source_eq (column : Nat)
    (lower : transcriptColumnStart ≤ column)
    (invocationBound : transcriptOffset column / 592 < transcriptInvocationCount)
    (laneBound : transcriptOffset column % 592 < Spec.Poseidon2.width) :
    transcriptOutputSource
        (Fin.encodeProd (transcriptInvocationIndex column invocationBound,
          transcriptLaneIndex column laneBound)) =
      column := by
  rw [transcriptOutputSource_encodeProd]
  dsimp only [transcriptInvocationIndex, transcriptLaneIndex]
  have divmod := Nat.mod_add_div (transcriptOffset column) 592
  unfold transcriptColumnStart at lower
  unfold transcriptOffset transcriptColumnStart at divmod ⊢
  omega

structure TranscriptDecoded (column : Nat) where
  index : Fin transcriptOutputCount
  source_eq : transcriptOutputSource index = column

def decodeTranscript (column : Nat) : Option (TranscriptDecoded column) :=
  if lower : transcriptColumnStart ≤ column then
    if invocationBound :
        transcriptOffset column / 592 < transcriptInvocationCount then
      if laneBound : transcriptOffset column % 592 < Spec.Poseidon2.width then
        some ⟨Fin.encodeProd
            (transcriptInvocationIndex column invocationBound,
              transcriptLaneIndex column laneBound),
          transcriptDecoded_source_eq column lower invocationBound laneBound⟩
      else
        none
    else
      none
  else
    none

theorem decodeTranscript_complete {column : Nat}
    (inside : PiCCSOrdinarySourceSupport.TranscriptOutput column) :
    (decodeTranscript column).isSome := by
  rcases inside with ⟨invocation, lane, equals⟩
  have invocationBound := invocation.isLt
  have laneBound := lane.isLt
  have regroup :
      PiCCSInputs.phaseOffset + invocation.val * 592 + 584 + lane.val =
        (PiCCSInputs.phaseOffset + 584) +
          (invocation.val * 592 + lane.val) := by
    omega
  have lower : transcriptColumnStart ≤ column := by
    unfold transcriptColumnStart
    rw [equals, regroup]
    exact Nat.le_add_right _ _
  have offsetEq : transcriptOffset column = invocation.val * 592 + lane.val := by
    unfold transcriptOffset transcriptColumnStart
    rw [equals, regroup, Nat.add_sub_cancel_left]
  have laneBound592 : lane.val < 592 := by
    have laneBound8 := laneBound
    change lane.val < 8 at laneBound8
    omega
  have quotientEq : transcriptOffset column / 592 = invocation.val := by
    rw [offsetEq]
    omega
  have remainderEq : transcriptOffset column % 592 = lane.val := by
    rw [offsetEq]
    omega
  have decodedInvocationBound :
      transcriptOffset column / 592 < transcriptInvocationCount := by
    rw [quotientEq]
    exact invocationBound
  have decodedLaneBound :
      transcriptOffset column % 592 < Spec.Poseidon2.width := by
    rw [remainderEq]
    exact laneBound
  unfold decodeTranscript
  rw [dif_pos lower, dif_pos decodedInvocationBound, dif_pos decodedLaneBound]
  rfl

def ordinaryLogicalLocated {column : Nat}
    (inside : PiCCSOrdinarySourceSupport.InRange
      PiCCSStarts.initialClaimLogicalStart ordinaryLogicalCount column) :
    Located column :=
  ⟨.proofLogical (ordinaryLogicalSlot (rangeIndex inside)), by
    rw [Location.sourceColumn, proofLogicalSource_ordinaryLogical]
    change PiCCSStarts.initialClaimLogicalStart +
      (column - PiCCSStarts.initialClaimLogicalStart) = column
    unfold PiCCSOrdinarySourceSupport.InRange at inside
    omega⟩

def classifySource (column : Nat) : Option (Located column) :=
  if prior : PiCCSOrdinarySourceSupport.InRange
      PilotProduction.priorPreimageStart PilotProduction.stateHashWords column then
    some ⟨.priorInput (rangeIndex prior), by
      rw [Location.sourceColumn, rangeIndex_source prior]⟩
  else if freshPublic : PiCCSOrdinarySourceSupport.InRange
      PilotProduction.priorPublicInputStart 270 column then
    some ⟨.freshPublicInput (rangeIndex freshPublic), by
      rw [Location.sourceColumn, rangeIndex_source freshPublic]⟩
  else if output : PiCCSOrdinarySourceSupport.InRange
      PilotProduction.outputPreimageStart PilotProduction.stateHashWords column then
    some ⟨.outputInput (rangeIndex output), by
      rw [Location.sourceColumn, rangeIndex_source output]⟩
  else if context : PiCCSOrdinarySourceSupport.InRange
      PiCCSInputs.expectedContextStart PiCCSInputs.expectedContextWords column then
    some ⟨.expectedContext (rangeIndex context), by
      rw [Location.sourceColumn, rangeIndex_source context]⟩
  else if proofInput : PiCCSOrdinarySourceSupport.InRange
      PiCCSInputs.proofInputStart proofInputCount column then
    some (proofInputLocated proofInput)
  else match decodeTranscript column with
    | some decoded =>
        some ⟨.proofLogical (transcriptOutputSlot decoded.index), by
          rw [Location.sourceColumn, proofLogicalSource_transcriptOutput,
            decoded.source_eq]⟩
    | none =>
        if ordinary : PiCCSOrdinarySourceSupport.InRange
            PiCCSStarts.initialClaimLogicalStart ordinaryLogicalCount column then
          some (ordinaryLogicalLocated ordinary)
        else if fresh : PiCCSOrdinarySourceSupport.InRange
            PiCCSArithmetic.initialClaimFreshStart
              PiCCSOrdinaryRetainedBlocks.freshCount column then
          some ⟨.fresh (rangeIndex fresh), by
            rw [Location.sourceColumn, rangeIndex_source fresh]⟩
        else
          none

private theorem externalProof_in_proofInput {column : Nat}
    (inside : PiCCSOrdinarySourceSupport.InRange PiCCSInputs.proofInputStart
      (PiCCSInputs.phaseOffset - PiCCSInputs.proofInputStart) column) :
    PiCCSOrdinarySourceSupport.InRange PiCCSInputs.proofInputStart
      proofInputCount column := by
  exact inside

private theorem sourceFresh_inRange {column : Nat}
    (inside : PiCCSStarts.initialClaimFreshStart ≤ column ∧
      column < PiRLCInputs.phaseOffset) :
    PiCCSOrdinarySourceSupport.InRange PiCCSArithmetic.initialClaimFreshStart
      PiCCSOrdinaryRetainedBlocks.freshCount column := by
  unfold PiCCSOrdinarySourceSupport.InRange
    PiCCSOrdinaryRetainedBlocks.freshCount
    PiCCSArithmetic.initialClaimFreshStart at *
  unfold PiCCSStarts.initialClaimFreshStart PiCCSStarts.roundTranscriptFreshStart
    PiCCSStarts.challengeFreshStart PiCCSStarts.statementAbsorptionFreshStart
    PiCCSStarts.statementBindingFreshStart PiCCSStarts.logicalFreshBase at *
  rw [PiCCSInputs.phaseOffset_eq] at *
  norm_num [PiRLCInputs.phaseOffset] at *
  exact inside

theorem classifySource_complete {column : Nat}
    (support : PiCCSOrdinarySourceSupport.Source column) :
    (classifySource column).isSome := by
  by_cases prior : PiCCSOrdinarySourceSupport.InRange
      PilotProduction.priorPreimageStart PilotProduction.stateHashWords column
  · unfold classifySource
    rw [dif_pos prior]
    rfl
  by_cases freshPublic : PiCCSOrdinarySourceSupport.InRange
      PilotProduction.priorPublicInputStart 270 column
  · unfold classifySource
    rw [dif_neg prior, dif_pos freshPublic]
    rfl
  by_cases output : PiCCSOrdinarySourceSupport.InRange
      PilotProduction.outputPreimageStart PilotProduction.stateHashWords column
  · unfold classifySource
    rw [dif_neg prior, dif_neg freshPublic, dif_pos output]
    rfl
  by_cases context : PiCCSOrdinarySourceSupport.InRange
      PiCCSInputs.expectedContextStart PiCCSInputs.expectedContextWords column
  · unfold classifySource
    rw [dif_neg prior, dif_neg freshPublic, dif_neg output, dif_pos context]
    rfl
  by_cases proofInput : PiCCSOrdinarySourceSupport.InRange
      PiCCSInputs.proofInputStart proofInputCount column
  · unfold classifySource
    rw [dif_neg prior, dif_neg freshPublic, dif_neg output, dif_neg context,
      dif_pos proofInput]
    rfl
  cases decodedEq : decodeTranscript column with
  | some decoded =>
      unfold classifySource
      rw [dif_neg prior, dif_neg freshPublic, dif_neg output, dif_neg context,
        dif_neg proofInput, decodedEq]
      rfl
  | none =>
    by_cases ordinary : PiCCSOrdinarySourceSupport.InRange
        PiCCSStarts.initialClaimLogicalStart ordinaryLogicalCount column
    · unfold classifySource
      rw [dif_neg prior, dif_neg freshPublic, dif_neg output, dif_neg context,
        dif_neg proofInput, decodedEq, dif_pos ordinary]
      rfl
    by_cases fresh : PiCCSOrdinarySourceSupport.InRange
        PiCCSArithmetic.initialClaimFreshStart
          PiCCSOrdinaryRetainedBlocks.freshCount column
    · unfold classifySource
      rw [dif_neg prior, dif_neg freshPublic, dif_neg output, dif_neg context,
        dif_neg proofInput, decodedEq, dif_neg ordinary, dif_pos fresh]
      rfl
    · exfalso
      rcases support with (external | transcriptOrOrdinary) | freshSupport
      · rcases external with priorSupport | publicSupport | outputSupport |
          contextSupport | proofSupport
        · exact prior priorSupport
        · exact freshPublic publicSupport
        · exact output outputSupport
        · exact context contextSupport
        · exact proofInput (externalProof_in_proofInput proofSupport)
      · rcases transcriptOrOrdinary with transcriptSupport | ordinarySupport
        · have complete := decodeTranscript_complete transcriptSupport
          simp [decodedEq] at complete
        · exact ordinary ordinarySupport
      · exact fresh (sourceFresh_inRange freshSupport)

structure Decoded where
  source : Nat
  location : Location
  owns : location.sourceColumn = source

def classifyTarget (column : Nat) : Option Decoded :=
  match Spartan.spartanToSource column with
  | none => none
  | some source =>
      match classifySource source with
      | none => none
      | some located => some ⟨source, located.location, located.owns⟩

theorem classifyTarget_complete {column : Nat}
    (support : PiCCSOrdinarySourceSupport.Target column) :
    ∃ decoded, classifyTarget column = some decoded ∧
      Spartan.sourceToSpartan decoded.source = column := by
  rcases support with ⟨source, sourceSupport, rfl⟩
  have inverse := Spartan.spartanToSource_sourceToSpartan source
    (PiCCSOrdinarySourceSupport.source_lt_sourceColumnCount sourceSupport)
  have complete := classifySource_complete sourceSupport
  cases found : classifySource source with
  | none => simp [found] at complete
  | some located =>
      refine ⟨⟨source, located.location, located.owns⟩, ?_, rfl⟩
      simp [classifyTarget, inverse, found]

def sourceMap {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (geometry : Geometry program logicalWidth) :
    SourceCompiler.SourceMap Spartan.spartanColumnCount logicalWidth where
  form := fun column =>
    match classifyTarget column.val with
    | none => .empty
    | some decoded => decoded.location.form geometry

theorem sourceMap_form_eval_of_target
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 33 → F)
    (products : Fin PiRLCFirst54DirectSchedule.candidateCount → F)
    (encodes : Encodes geometry assignment
      (PiRLCRetainedPreservation.sourceAssignment program base groupValue products))
    (column : Fin Spartan.spartanColumnCount)
    (support : PiCCSOrdinarySourceSupport.Target column.val) :
    ((sourceMap geometry).form column).eval assignment =
      RunningTransitionDirectPlan.transitionEnv program base column.val := by
  rcases classifyTarget_complete support with ⟨decoded, found, mapped⟩
  change (match classifyTarget column.val with
    | none => SparseForm.empty
    | some value => value.location.form geometry).eval assignment = _
  rw [found]
  rw [Location.form_eval geometry assignment _ encodes decoded.location]
  rw [RunningTransitionDirectPlan.sourceAssignment_packageSource program base
    groupValue products decoded.location.sourceColumn
    decoded.location.sourceColumn_lt]
  have mappedLocation :
      Spartan.sourceToSpartan decoded.location.sourceColumn = column.val := by
    rw [decoded.owns, mapped]
  rw [mappedLocation]

private theorem preservesCombination
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 33 → F)
    (products : Fin PiRLCFirst54DirectSchedule.candidateCount → F)
    (encodes : Encodes geometry assignment
      (PiRLCRetainedPreservation.sourceAssignment program base groupValue products))
    (combination : R1CS.LinearCombination)
    (bounded : SourceCompiler.CombinationBounded Spartan.spartanColumnCount
      combination)
    (scope : combination.VarsSatisfy PiCCSOrdinarySourceSupport.Target) :
    OrdinarySourcePlan.SourceMap.PreservesCombination (sourceMap geometry)
      assignment (RunningTransitionDirectPlan.transitionEnv program base)
      combination bounded := by
  intro term member
  exact sourceMap_form_eval_of_target geometry assignment base groupValue products
    encodes ⟨term.1, bounded term member⟩ (scope term member)

private theorem programRow_support
    {relationLogicalWidth : Nat}
    {relationPublicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth relationLogicalWidth}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (index : Fin 811669) :
    (PiCCSOrdinaryDirectSource.programRow relation index).VarsSatisfy
      PiCCSOrdinarySourceSupport.Target := by
  exact PiCCSOrdinaryDirectSupport.sourceRows_varsSatisfy relation _
    (List.get_mem _
      (PiCCSOrdinaryDirectSource.sourceListIndex relation index))

def inputs
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    {relationLogicalWidth : Nat}
    {relationPublicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth relationLogicalWidth}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (geometry : Geometry program logicalWidth) :
    (PiCCSOrdinaryDirectSource.program relation).Inputs logicalWidth where
  oneColumn := oneColumn geometry
  sourceMap := fun _ => sourceMap geometry

theorem inputs_preserve
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    {relationLogicalWidth : Nat}
    {relationPublicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth relationLogicalWidth}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (geometry : Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 33 → F)
    (products : Fin PiRLCFirst54DirectSchedule.candidateCount → F)
    (encodes : Encodes geometry assignment
      (PiRLCRetainedPreservation.sourceAssignment program base groupValue products)) :
    ∀ index, OrdinarySourcePlan.SourceMap.PreservesRow
      ((inputs relation geometry).sourceMap index) assignment
      (RunningTransitionDirectPlan.transitionEnv program base)
      ((PiCCSOrdinaryDirectSource.program relation).row index)
      ((PiCCSOrdinaryDirectSource.program relation).bounded index) := by
  intro index
  have directScope := programRow_support relation index
  have scope :
      ((PiCCSOrdinaryDirectSource.program relation).row index).VarsSatisfy
        PiCCSOrdinarySourceSupport.Target := by
    simpa only [PiCCSOrdinaryDirectSource.program,
      PiCCSOrdinaryDirectSource.SupportedProgram.toProgram,
      PiCCSOrdinaryDirectSource.supportedProgram] using directScope
  exact ⟨
    preservesCombination geometry assignment base groupValue products encodes
      _ _ scope.1,
    preservesCombination geometry assignment base groupValue products encodes
      _ _ scope.2.1,
    preservesCombination geometry assignment base groupValue products encodes
      _ _ scope.2.2⟩

/-- Exact row-local preservation for the explicit canonical PiCCS row. -/
theorem programRow_preserve
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    {relationLogicalWidth : Nat}
    {relationPublicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth relationLogicalWidth}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (geometry : Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 33 → F)
    (products : Fin PiRLCFirst54DirectSchedule.candidateCount → F)
    (encodes : Encodes geometry assignment
      (PiRLCRetainedPreservation.sourceAssignment program base groupValue products))
    (index : Fin 811669) :
    OrdinarySourcePlan.SourceMap.PreservesRow (sourceMap geometry) assignment
      (RunningTransitionDirectPlan.transitionEnv program base)
      (PiCCSOrdinaryDirectSource.programRow relation index)
      (PiCCSOrdinaryDirectSource.programRow_bounded relation index) := by
  have scope := programRow_support relation index
  exact ⟨
    preservesCombination geometry assignment base groupValue products encodes
      _ _ scope.1,
    preservesCombination geometry assignment base groupValue products encodes
      _ _ scope.2.1,
    preservesCombination geometry assignment base groupValue products encodes
      _ _ scope.2.2⟩

/-- Exact sparse forms for one canonical Lean-lowered PiCCS ordinary row. -/
def rowForms
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    {relationLogicalWidth : Nat}
    {relationPublicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth relationLogicalWidth}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (geometry : Geometry program logicalWidth) (index : Fin 811669) :
    OrdinaryRow.Forms logicalWidth :=
  SourceCompiler.compileRow (sourceMap geometry) (oneColumn geometry)
    (PiCCSOrdinaryDirectSource.programRow relation index)
    (PiCCSOrdinaryDirectSource.programRow_bounded relation index)

theorem rowForms_eq_of_same_shape
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    {relationLogicalWidth : Nat}
    {relationPublicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth relationLogicalWidth}
    (left right : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (geometry : Geometry program logicalWidth) :
    rowForms left geometry = rowForms right geometry := by
  funext index
  exact SourceCompiler.compileRow_eq_of_row (sourceMap geometry)
    (oneColumn geometry)
    (congrFun
      (PiCCSOrdinaryDirectSource.programRow_eq_of_same_shape left right) index)
    (PiCCSOrdinaryDirectSource.programRow_bounded left index)
    (PiCCSOrdinaryDirectSource.programRow_bounded right index)

/-- Canonical direct 14-matrix rows for all PiCCS ordinary constraints. -/
def plan
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    {relationLogicalWidth : Nat}
    {relationPublicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth relationLogicalWidth}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (geometry : Geometry program logicalWidth) :
    ProductionRelation.Plan logicalWidth :=
  OrdinaryRow.planOfForms (by norm_num [Lifecycle.cubeVariables])
    (rowForms relation geometry)

@[simp] theorem plan_rowCount
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    {relationLogicalWidth : Nat}
    {relationPublicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth relationLogicalWidth}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (geometry : Geometry program logicalWidth) :
    (plan relation geometry).rowCount = 811669 := by
  rfl

/-- The compiled matrix plan depends only on the relation shape. Matrix
entries do not participate in source-row generation or retained placement. -/
theorem plan_eq_of_same_shape
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    {relationLogicalWidth : Nat}
    {relationPublicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth relationLogicalWidth}
    (left right : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (geometry : Geometry program logicalWidth) :
    plan left geometry = plan right geometry := by
  unfold plan
  rw [rowForms_eq_of_same_shape left right]

/-- Direct matrix acceptance is exactly the canonical Lean-lowered PiCCS
ordinary row relation. -/
theorem rowsZero_iff_rowsHold
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    {relationLogicalWidth : Nat}
    {relationPublicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth relationLogicalWidth}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (geometry : Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 33 → F)
    (products : Fin PiRLCFirst54DirectSchedule.candidateCount → F)
    (one : assignment (oneColumn geometry) = 1)
    (encodes : Encodes geometry assignment
      (PiRLCRetainedPreservation.sourceAssignment program base groupValue products)) :
    (plan relation geometry).RowsZero assignment ↔
      R1CS.RowsHold (RunningTransitionDirectPlan.transitionEnv program base)
        (PiCCSOrdinaryDirectSource.sourceRows relationLogicalWidth
          relationPublicFits) := by
  rw [← PiCCSOrdinaryDirectSource.programRows_hold_iff_rowsHold relation
    (RunningTransitionDirectPlan.transitionEnv program base)]
  constructor
  · intro rows index
    have preserves := OrdinarySourcePlan.compileRow_preserves_local
      (sourceMap geometry) (oneColumn geometry)
      (PiCCSOrdinaryDirectSource.programRow relation index)
      (PiCCSOrdinaryDirectSource.programRow_bounded relation index)
      assignment (RunningTransitionDirectPlan.transitionEnv program base) one
      (programRow_preserve relation geometry assignment base groupValue
        products encodes index)
    exact (OrdinaryRow.planOfForms_residual_zero_iff
      (by norm_num [Lifecycle.cubeVariables]) (rowForms relation geometry)
      assignment (RunningTransitionDirectPlan.transitionEnv program base)
      index (PiCCSOrdinaryDirectSource.programRow relation index)
      preserves).mp (rows index)
  · intro rows index
    have preserves := OrdinarySourcePlan.compileRow_preserves_local
      (sourceMap geometry) (oneColumn geometry)
      (PiCCSOrdinaryDirectSource.programRow relation index)
      (PiCCSOrdinaryDirectSource.programRow_bounded relation index)
      assignment (RunningTransitionDirectPlan.transitionEnv program base) one
      (programRow_preserve relation geometry assignment base groupValue
        products encodes index)
    exact (OrdinaryRow.planOfForms_residual_zero_iff
      (by norm_num [Lifecycle.cubeVariables]) (rowForms relation geometry)
      assignment (RunningTransitionDirectPlan.transitionEnv program base)
      index (PiCCSOrdinaryDirectSource.programRow relation index)
      preserves).mpr (rows index)

end NightstreamFPrime.Export.Stage1.PiCCSOrdinaryDirectPlan
