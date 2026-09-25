import NightstreamFPrime.Export.RowSemantics
import NightstreamFPrime.Export.Stage1.PiRLCCombinationTemplates
import NightstreamFPrime.Export.Stage1.PiRLCFirst54Templates
import NightstreamFPrime.Layout.Stage1.PiRLCStarts
import NightstreamFPrime.Layout.Stage1.Spartan

/-!
Owns the exact compact invocation schedule for all 17 PiRLC `First54`
selectors.

Each source contains 64 candidate rounds. A round emits 55 position recipes
followed by 54 value recipes. Row and R1CS-fresh starts follow the proved
`First54` physical costs. Input ranges contain only final Spartan columns.
-/

namespace NightstreamFPrime.Export.Stage1.PiRLCFirst54Invocations

open NightstreamFPrime.Export.Package
open NightstreamFPrime.Gadgets.Range
open NightstreamFPrime.Gadgets.Sampling
open NightstreamFPrime.Layout.Stage1

def phase : Nat := 7
def sourceCount : Nat := 17
def roundCount : Nat := First54.candidateCount

def templateBase : Nat :=
  PiRLCCombinationTemplates.templates.length

def packageTemplates : List CompactRowTemplate :=
  PiRLCCombinationTemplates.templates ++ PiRLCFirst54Templates.templates

def roundFreshPrefix : Nat → Nat
  | 0 => 0
  | previous + 1 => 216 + previous * 537

def roundRowPrefix : Nat → Nat
  | 0 => 0
  | previous + 1 => 325 + previous * 646

def positionFreshPrefix (round slot : Nat) : Nat :=
  if round = 0 then 0 else if slot = 0 then 0 else 6 * (slot - 1)

def positionRowPrefix (round slot : Nat) : Nat :=
  if round = 0 then slot
  else if slot = 0 then 0 else 1 + 7 * (slot - 1)

def positionFreshCount (round : Nat) : Nat :=
  if round = 0 then 0 else 321

def positionRowCount (round : Nat) : Nat :=
  if round = 0 then 55 else 376

def valueFreshPrefix (round slot : Nat) : Nat :=
  positionFreshCount round + 4 * slot

def valueRowPrefix (round slot : Nat) : Nat :=
  positionRowCount round + 5 * slot

def candidateDigestRound (candidate : Nat) : Nat := candidate / 8
def candidateLane (candidate : Nat) : Nat := candidate % 8 / 2
def candidatePart (candidate : Nat) : Nat := candidate % 2

def decoderLogicalStart (source candidate : Nat) : Nat :=
  PiRLCStarts.digestLaneLogicalStart source
      (candidateDigestRound candidate) (candidateLane candidate) +
    CanonicalU64.auxiliaryCount +
      candidatePart candidate * Candidate16Five.auxiliaryCount

def rejectSourceColumn (source candidate : Nat) : Nat :=
  decoderLogicalStart source candidate + 16

def remainderSourceColumn (source candidate : Nat) : Nat :=
  decoderLogicalStart source candidate + 1

def positionSourceStart (source round : Nat) : Nat :=
  First54.positionOffset (PiRLCStarts.selectorLogicalStart source) round

def valueSourceStart (source round : Nat) : Nat :=
  First54.valueOffset (PiRLCStarts.selectorLogicalStart source) round

def previousPositionSourceStart (source round : Nat) : Nat :=
  positionSourceStart source (round - 1)

def previousValueSourceStart (source round : Nat) : Nat :=
  valueSourceStart source (round - 1)

def finalColumn (sourceColumn : Nat) : Nat :=
  Spartan.sourceToSpartan sourceColumn

def firstPositionInputRanges (source round slot : Nat) :
    List CompactInputRange :=
  [⟨0, 1, finalColumn (rejectSourceColumn source round), 1⟩,
   ⟨1, 1, finalColumn (positionSourceStart source round + slot), 1⟩]

def laterPositionInputRanges (source round slot : Nat) :
    List CompactInputRange :=
  [⟨0, 1, finalColumn (rejectSourceColumn source round), 1⟩,
   ⟨1, First54Step.slotCount,
      finalColumn (previousPositionSourceStart source round), 1⟩,
   ⟨56, 1, finalColumn (positionSourceStart source round + slot), 1⟩]

def firstValueInputRanges (source round slot : Nat) :
    List CompactInputRange :=
  [⟨0, 1, finalColumn (rejectSourceColumn source round), 1⟩,
   ⟨1, 1, finalColumn (remainderSourceColumn source round), 1⟩,
   ⟨2, 1, finalColumn (valueSourceStart source round + slot), 1⟩]

def laterValueInputRanges (source round slot : Nat) :
    List CompactInputRange :=
  [⟨0, 1, finalColumn (rejectSourceColumn source round), 1⟩,
   ⟨1, 1, finalColumn (remainderSourceColumn source round), 1⟩,
   ⟨2, First54Step.slotCount,
      finalColumn (previousPositionSourceStart source round), 1⟩,
   ⟨57, First54ValueStep.outputCount,
      finalColumn (previousValueSourceStart source round), 1⟩,
   ⟨111, 1, finalColumn (valueSourceStart source round + slot), 1⟩]

def positionTemplateIndex (round slot : Nat) : Nat :=
  templateBase +
    if round = 0 then
      PiRLCFirst54Templates.firstPositionTemplateIndex slot
    else
      PiRLCFirst54Templates.laterPositionTemplateIndex slot

def valueTemplateIndex (round slot : Nat) : Nat :=
  templateBase +
    if round = 0 then
      PiRLCFirst54Templates.firstValueTemplateIndex slot
    else
      PiRLCFirst54Templates.laterValueTemplateIndex slot

def positionInvocation (source round slot : Nat) : CompactRowInvocation where
  phase := phase
  templateIndex := positionTemplateIndex round slot
  rowStart := PiRLCStarts.selectorRowStart source + roundRowPrefix round +
    positionRowPrefix round slot
  localStart := finalColumn (PiRLCStarts.selectorFreshStart source +
    roundFreshPrefix round + positionFreshPrefix round slot)
  inputRanges := if round = 0 then
    firstPositionInputRanges source round slot
  else laterPositionInputRanges source round slot

def valueInvocation (source round slot : Nat) : CompactRowInvocation where
  phase := phase
  templateIndex := valueTemplateIndex round slot
  rowStart := PiRLCStarts.selectorRowStart source + roundRowPrefix round +
    valueRowPrefix round slot
  localStart := finalColumn (PiRLCStarts.selectorFreshStart source +
    roundFreshPrefix round + valueFreshPrefix round slot)
  inputRanges := if round = 0 then
    firstValueInputRanges source round slot
  else laterValueInputRanges source round slot

@[simp] theorem positionInvocation_localStart (source round slot : Nat) :
    (positionInvocation source round slot).localStart =
      finalColumn (PiRLCStarts.selectorFreshStart source +
        roundFreshPrefix round + positionFreshPrefix round slot) := by
  rfl

@[simp] theorem valueInvocation_localStart (source round slot : Nat) :
    (valueInvocation source round slot).localStart =
      finalColumn (PiRLCStarts.selectorFreshStart source +
        roundFreshPrefix round + valueFreshPrefix round slot) := by
  rfl

@[simp] theorem positionInvocation_zero_inputRanges (source slot : Nat) :
    (positionInvocation source 0 slot).inputRanges =
      firstPositionInputRanges source 0 slot := by
  rfl

@[simp] theorem positionInvocation_succ_inputRanges
    (source round slot : Nat) :
    (positionInvocation source (round + 1) slot).inputRanges =
      laterPositionInputRanges source (round + 1) slot := by
  simp [positionInvocation]

@[simp] theorem valueInvocation_zero_inputRanges (source slot : Nat) :
    (valueInvocation source 0 slot).inputRanges =
      firstValueInputRanges source 0 slot := by
  rfl

@[simp] theorem valueInvocation_succ_inputRanges
    (source round slot : Nat) :
    (valueInvocation source (round + 1) slot).inputRanges =
      laterValueInputRanges source (round + 1) slot := by
  simp [valueInvocation]

def positionInvocations (source round : Nat) : List CompactRowInvocation :=
  (List.finRange First54Step.slotCount).map fun slot =>
    positionInvocation source round slot.val

def valueInvocations (source round : Nat) : List CompactRowInvocation :=
  (List.finRange First54ValueStep.outputCount).map fun slot =>
    valueInvocation source round slot.val

def roundInvocations (source round : Nat) : List CompactRowInvocation :=
  positionInvocations source round ++ valueInvocations source round

def sourceInvocations (source : Nat) : List CompactRowInvocation :=
  (List.range roundCount).flatMap (roundInvocations source)

def invocations : List CompactRowInvocation :=
  (List.range sourceCount).flatMap sourceInvocations

theorem positionInvocation_mem (source round : Nat)
    (slot : Fin First54Step.slotCount)
    (sourceLt : source < sourceCount) (roundLt : round < roundCount) :
    positionInvocation source round slot.val ∈ invocations := by
  unfold invocations
  apply List.mem_flatMap.mpr
  refine ⟨source, List.mem_range.mpr sourceLt, ?_⟩
  unfold sourceInvocations
  apply List.mem_flatMap.mpr
  refine ⟨round, List.mem_range.mpr roundLt, ?_⟩
  unfold roundInvocations
  apply List.mem_append_left
  unfold positionInvocations
  apply List.mem_map.mpr
  exact ⟨slot, by simp, rfl⟩

theorem valueInvocation_mem (source round : Nat)
    (slot : Fin First54ValueStep.outputCount)
    (sourceLt : source < sourceCount) (roundLt : round < roundCount) :
    valueInvocation source round slot.val ∈ invocations := by
  unfold invocations
  apply List.mem_flatMap.mpr
  refine ⟨source, List.mem_range.mpr sourceLt, ?_⟩
  unfold sourceInvocations
  apply List.mem_flatMap.mpr
  refine ⟨round, List.mem_range.mpr roundLt, ?_⟩
  unfold roundInvocations
  apply List.mem_append_right
  unfold valueInvocations
  apply List.mem_map.mpr
  exact ⟨slot, by simp, rfl⟩

private theorem shiftedTemplate_getElem? {index : Nat}
    {template : CompactRowTemplate}
    (selected : PiRLCFirst54Templates.templates[index]? = some template) :
    packageTemplates[templateBase + index]? =
      some template := by
  unfold packageTemplates templateBase
  rw [List.getElem?_append_right (by omega), Nat.add_sub_cancel_left]
  exact selected

theorem positionInvocation_zero_template (source : Nat)
    (slot : Fin First54Step.slotCount) :
    packageTemplates[(positionInvocation source 0 slot.val).templateIndex]? =
      some (PiRLCFirst54Templates.firstPositionTemplate slot) := by
  simpa [positionInvocation, positionTemplateIndex] using
    shiftedTemplate_getElem?
      (PiRLCFirst54Templates.firstPositionTemplate_getElem?
        slot.val slot.isLt)

theorem positionInvocation_succ_template (source round : Nat)
    (slot : Fin First54Step.slotCount) :
    packageTemplates[
        (positionInvocation source (round + 1) slot.val).templateIndex]? =
      some (PiRLCFirst54Templates.laterPositionTemplate slot) := by
  simpa [positionInvocation, positionTemplateIndex] using
    shiftedTemplate_getElem?
      (PiRLCFirst54Templates.laterPositionTemplate_getElem?
        slot.val slot.isLt)

theorem valueInvocation_zero_template (source : Nat)
    (slot : Fin First54ValueStep.outputCount) :
    packageTemplates[(valueInvocation source 0 slot.val).templateIndex]? =
      some (PiRLCFirst54Templates.firstValueTemplate slot) := by
  simpa [valueInvocation, valueTemplateIndex] using
    shiftedTemplate_getElem?
      (PiRLCFirst54Templates.firstValueTemplate_getElem?
        slot.val slot.isLt)

theorem valueInvocation_succ_template (source round : Nat)
    (slot : Fin First54ValueStep.outputCount) :
    packageTemplates[
        (valueInvocation source (round + 1) slot.val).templateIndex]? =
      some (PiRLCFirst54Templates.laterValueTemplate slot) := by
  simpa [valueInvocation, valueTemplateIndex] using
    shiftedTemplate_getElem?
      (PiRLCFirst54Templates.laterValueTemplate_getElem?
        slot.val slot.isLt)

theorem positionInvocation_zero_rowCount (source : Nat)
    (slot : Fin First54Step.slotCount) :
    compactInvocationRowCountFor
        packageTemplates
        (positionInvocation source 0 slot.val) = 1 := by
  unfold compactInvocationRowCountFor
  rw [positionInvocation_zero_template source slot]
  exact PiRLCFirst54Templates.firstPositionTemplate_rows_length slot

theorem positionInvocation_succ_rowCount (source round : Nat)
    (slot : Fin First54Step.slotCount) :
    compactInvocationRowCountFor
        packageTemplates
        (positionInvocation source (round + 1) slot.val) =
      (if slot.val = 0 then 1 else if slot.val = 54 then 4 else 7) := by
  unfold compactInvocationRowCountFor
  rw [positionInvocation_succ_template source round slot]
  exact PiRLCFirst54Templates.laterPositionTemplate_rows_length slot

theorem valueInvocation_zero_rowCount (source : Nat)
    (slot : Fin First54ValueStep.outputCount) :
    compactInvocationRowCountFor
        packageTemplates
        (valueInvocation source 0 slot.val) = 5 := by
  unfold compactInvocationRowCountFor
  rw [valueInvocation_zero_template source slot]
  exact PiRLCFirst54Templates.firstValueTemplate_rows_length slot

theorem valueInvocation_succ_rowCount (source round : Nat)
    (slot : Fin First54ValueStep.outputCount) :
    compactInvocationRowCountFor
        packageTemplates
        (valueInvocation source (round + 1) slot.val) = 5 := by
  unfold compactInvocationRowCountFor
  rw [valueInvocation_succ_template source round slot]
  exact PiRLCFirst54Templates.laterValueTemplate_rows_length slot

private theorem compactRowCountFor_flatMap {Index : Type}
    (templates : List CompactRowTemplate) (indices : List Index)
    (entries : Index → List CompactRowInvocation) :
    compactRowCountFor templates (indices.flatMap entries) =
      (indices.map fun index =>
        compactRowCountFor templates (entries index)).sum := by
  induction indices with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      simp [compactRowCountFor_append, inductionHypothesis]

theorem positionInvocations_zero_rowCount (source : Nat) :
    compactRowCountFor packageTemplates (positionInvocations source 0) =
      55 := by
  unfold compactRowCountFor positionInvocations
  rw [List.map_map]
  calc
    (List.map
        (fun slot => compactInvocationRowCountFor packageTemplates
          (positionInvocation source 0 slot.val))
        (List.finRange First54Step.slotCount)).sum =
      (List.map (fun _ => 1) (List.finRange First54Step.slotCount)).sum := by
        apply congrArg List.sum
        apply List.map_congr_left
        intro slot _member
        exact positionInvocation_zero_rowCount source slot
    _ = 55 := by simp [First54Step.slotCount]

theorem positionInvocations_succ_rowCount (source round : Nat) :
    compactRowCountFor packageTemplates
        (positionInvocations source (round + 1)) = 376 := by
  unfold compactRowCountFor positionInvocations
  rw [List.map_map]
  calc
    (List.map
        (fun slot => compactInvocationRowCountFor packageTemplates
          (positionInvocation source (round + 1) slot.val))
        (List.finRange First54Step.slotCount)).sum =
      (List.map
        (fun slot =>
          if slot.val = 0 then 1 else if slot.val = 54 then 4 else 7)
        (List.finRange First54Step.slotCount)).sum := by
        apply congrArg List.sum
        apply List.map_congr_left
        intro slot _member
        exact positionInvocation_succ_rowCount source round slot
    _ = 376 := by rfl

theorem valueInvocations_zero_rowCount (source : Nat) :
    compactRowCountFor packageTemplates (valueInvocations source 0) = 270 := by
  unfold compactRowCountFor valueInvocations
  rw [List.map_map]
  calc
    (List.map
        (fun slot => compactInvocationRowCountFor packageTemplates
          (valueInvocation source 0 slot.val))
        (List.finRange First54ValueStep.outputCount)).sum =
      (List.map (fun _ => 5)
        (List.finRange First54ValueStep.outputCount)).sum := by
        apply congrArg List.sum
        apply List.map_congr_left
        intro slot _member
        exact valueInvocation_zero_rowCount source slot
    _ = 270 := by simp [First54ValueStep.outputCount]

theorem valueInvocations_succ_rowCount (source round : Nat) :
    compactRowCountFor packageTemplates
        (valueInvocations source (round + 1)) = 270 := by
  unfold compactRowCountFor valueInvocations
  rw [List.map_map]
  calc
    (List.map
        (fun slot => compactInvocationRowCountFor packageTemplates
          (valueInvocation source (round + 1) slot.val))
        (List.finRange First54ValueStep.outputCount)).sum =
      (List.map (fun _ => 5)
        (List.finRange First54ValueStep.outputCount)).sum := by
        apply congrArg List.sum
        apply List.map_congr_left
        intro slot _member
        exact valueInvocation_succ_rowCount source round slot
    _ = 270 := by simp [First54ValueStep.outputCount]

def roundRowCount : Nat → Nat
  | 0 => 325
  | _ + 1 => 646

theorem roundInvocations_rowCount (source round : Nat) :
    compactRowCountFor packageTemplates (roundInvocations source round) =
      roundRowCount round := by
  cases round with
  | zero =>
      rw [roundInvocations, compactRowCountFor_append,
        positionInvocations_zero_rowCount, valueInvocations_zero_rowCount]
      rfl
  | succ round =>
      rw [roundInvocations, compactRowCountFor_append,
        positionInvocations_succ_rowCount, valueInvocations_succ_rowCount]
      rfl

theorem sourceInvocations_rowCount (source : Nat) :
    compactRowCountFor packageTemplates (sourceInvocations source) = 41023 := by
  rw [sourceInvocations, compactRowCountFor_flatMap]
  calc
    ((List.range roundCount).map fun round =>
        compactRowCountFor packageTemplates
          (roundInvocations source round)).sum =
      ((List.range roundCount).map roundRowCount).sum := by
        apply congrArg List.sum
        apply List.map_congr_left
        intro round _member
        exact roundInvocations_rowCount source round
    _ = 41023 := by rfl

theorem compactRowCount :
    compactRowCountFor packageTemplates invocations = 697391 := by
  rw [invocations, compactRowCountFor_flatMap]
  calc
    ((List.range sourceCount).map fun source =>
        compactRowCountFor packageTemplates
          (sourceInvocations source)).sum =
      ((List.range sourceCount).map fun _ => 41023).sum := by
        apply congrArg List.sum
        apply List.map_congr_left
        intro source _member
        exact sourceInvocations_rowCount source
    _ = 697391 := by rfl

@[simp] theorem positionInvocations_length (source round : Nat) :
    (positionInvocations source round).length = 55 := by
  simp [positionInvocations, First54Step.slotCount]

@[simp] theorem valueInvocations_length (source round : Nat) :
    (valueInvocations source round).length = 54 := by
  simp [valueInvocations, First54ValueStep.outputCount]

@[simp] theorem roundInvocations_length (source round : Nat) :
    (roundInvocations source round).length = 109 := by
  simp [roundInvocations]

@[simp] theorem sourceInvocations_length (source : Nat) :
    (sourceInvocations source).length = 6976 := by
  simp [sourceInvocations, roundCount, First54.candidateCount]

@[simp] theorem invocations_length : invocations.length = 118592 := by
  simp [invocations, sourceCount]

theorem finalAssertionRowStart (source : Nat) :
    PiRLCStarts.selectorRowStart source + roundRowPrefix roundCount =
      PiRLCStarts.selectorRowStart source + 41023 := by
  norm_num [roundRowPrefix, roundCount, First54.candidateCount]

theorem finalFreshStart (source : Nat) :
    PiRLCStarts.selectorFreshStart source + roundFreshPrefix roundCount =
      PiRLCStarts.selectorFreshStart source + 34047 := by
  norm_num [roundFreshPrefix, roundCount, First54.candidateCount]

end NightstreamFPrime.Export.Stage1.PiRLCFirst54Invocations
