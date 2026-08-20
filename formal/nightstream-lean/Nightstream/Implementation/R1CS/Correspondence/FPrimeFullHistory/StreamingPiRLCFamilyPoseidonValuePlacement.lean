import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiRLCNormalizedFamilyRows
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingPiRLCFamilyPoseidonCallFamily
import Nightstream.Implementation.R1CS.Correspondence.SelectiveCcs.Rewrite.Artifact.SourceImage

/-!
Contract: exact value placement for the normalized production PiRLC Poseidon2
replay words.

Assurance tier: artifact-checked same-assignment value authority.

Owns: equality between the compact leaf's 41-coordinate radix-three action
and the normalized algebra decoder, plus the exact input and output algebra
column maps.

Does not own: replay state placement, selector activation, emitted-row
satisfaction, complete duplex execution, lifecycle semantics, or permission
to remove constraints.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonValuePlacement

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonLeafModel
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonLeafModel.Wire
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonLeafReconstruction
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonCallProjection
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonCallFamily
open Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.Artifact.Decoder
open Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.Artifact.SourceImage
open Nightstream.Implementation.Nebula.ProductionStreamingPiRlcAuthority

private abbrev algebraLocalColumns :=
  Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedAlgebraRows.Normalized.localColumns

private abbrev algebraLocalSlot :=
  Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedAlgebraRows.Normalized.localSlot

private abbrev generatedLayout :=
  Nightstream.Implementation.Nebula.ProductionStreamingPiRlcArtifact.Generated.layout

private abbrev Source :=
  Nightstream.Implementation.Nebula.ProductPiRlcRingCombinationRows.Source

private abbrev Lane := Fin ringDegree

/-- A normalized 41-coordinate radix-three word at one final-column start. -/
def wordSlot (start : Nat)
    (fits : start + 41 <= productionFinalColumns) :
    DecodedSourceSlot 1 productionFinalColumns where
  column := 0
  start := start
  width := 41
  widthPositive := by decide
  columnsFit := fits

def wordValue (assignment : Fin productionFinalColumns -> F)
    (start : Nat) (fits : start + 41 <= productionFinalColumns) : F :=
  sourceSlotValue (wordSlot start fits) assignment

private theorem geometricCoefficient_one_three (index : Nat) :
    geometricCoefficient 1 3 index = (3 : F) ^ index := by
  induction index with
  | zero => rfl
  | succ index inductionHypothesis =>
      simp only [geometricCoefficient]
      rw [inductionHypothesis]
      exact (pow_succ (3 : F) index).symm

private theorem sum_map_eq_foldr
    (indices : List (Fin 41)) (term : Fin 41 -> F) :
    sum (indices.map term) =
      indices.foldr (fun index tail => term index + tail) 0 := by
  induction indices with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      simp only [List.map_cons, sum, List.foldr_cons, inductionHypothesis]

private theorem foldr_weighted_coordinate_congr
    (indices : List (Fin 41)) (left right : Fin 41 -> F)
    (coordinate : forall index, left index = right index) :
    indices.foldr
        (fun index tail => (3 : F) ^ index.val * left index + tail) 0 =
      indices.foldr
        (fun index tail => (3 : F) ^ index.val * right index + tail) 0 := by
  induction indices with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      simp only [List.foldr_cons]
      rw [coordinate head, inductionHypothesis]

/-- The compact replay leaf and normalized algebra use the same structural
radix-three evaluator. This proof is independent of generated artifact size. -/
theorem slotValue_eq_wordValue_of_digits
    (final : FinalAssignment) (slot : Slot)
    (assignment : Fin productionFinalColumns -> F)
    (start : Nat) (fits : start + 41 <= productionFinalColumns)
    (digits : forall digit : Fin 41,
      final.digit slot digit =
        assignment
          ⟨start + digit.val,
            Nat.lt_of_lt_of_le (Nat.add_lt_add_left digit.isLt start) fits⟩) :
    slotValue final slot = wordValue assignment start fits := by
  unfold slotValue geometricAction
  have compactFold :
      sum (List.ofFn fun index : Fin 41 =>
        geometricCoefficient 1 3 index.val * final.digit slot index) =
        (List.ofFn id).foldr
          (fun index tail =>
            geometricCoefficient 1 3 index.val * final.digit slot index +
              tail)
          0 := by
    simpa only [List.map_ofFn, Function.comp_apply] using
      sum_map_eq_foldr (List.ofFn id)
        (fun index : Fin 41 =>
          geometricCoefficient 1 3 index.val * final.digit slot index)
  rw [compactFold]
  unfold wordValue
  rw [sourceSlotValue_eq_foldr]
  unfold wordSlot
  simp only [canonicalFinIndices, Function.comp_apply]
  congr 1
  funext digit tail
  rw [geometricCoefficient_one_three, digits]
  rfl

def inputColumn (source : Source) (lane : Lane) : Fin algebraLocalColumns :=
  ⟨919 + source.val * ringDegree + lane.val, by
    have sourceLt := source.isLt
    have laneLt := lane.isLt
    change source.val < 17 at sourceLt
    change lane.val < 54 at laneLt
    change 919 + source.val * 54 + lane.val < 51463
    omega⟩

def outputColumn (lane : Lane) : Fin algebraLocalColumns :=
  ⟨1837 + lane.val, by
    have laneLt := lane.isLt
    change lane.val < 54 at laneLt
    change 1837 + lane.val < 51463
    omega⟩

theorem inputColumn_eq_layout (source : Source) (lane : Lane) :
    (inputColumn source lane).val = generatedLayout.input source lane := by
  rw [Nightstream.Implementation.Nebula.ProductionStreamingPiRlcArtifact.Generated.layout_input]
  change 919 + source.val * 54 + lane.val =
    919 + source.val * 54 + lane.val
  rfl

theorem outputColumn_eq_layout (lane : Lane) :
    (outputColumn lane).val = generatedLayout.output lane := by
  simp [outputColumn]

private theorem inputColumn_nonzero (source : Source) (lane : Lane) :
    (inputColumn source lane).val ≠ 0 := by
  simp [inputColumn]

private theorem outputColumn_nonzero (lane : Lane) :
    (outputColumn lane).val ≠ 0 := by
  simp [outputColumn]

theorem inputLocalSlot_start (source : Source) (lane : Lane) :
    (algebraLocalSlot (inputColumn source lane)
      (inputColumn_nonzero source lane)).start =
        38340 + (source.val * ringDegree + lane.val) * 41 := by
  unfold algebraLocalSlot
  unfold Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedAlgebraRows.Normalized.localSlot
  rw [dif_neg (by
    have sourceLt := source.isLt
    have laneLt := lane.isLt
    change source.val < 17 at sourceLt
    change lane.val < 54 at laneLt
    simp only [inputColumn]
    omega)]
  rw [dif_pos (by
    have sourceLt := source.isLt
    have laneLt := lane.isLt
    change source.val < 17 at sourceLt
    change lane.val < 54 at laneLt
    simp only [inputColumn]
    change 919 + source.val * 54 + lane.val < 1837
    omega)]
  simp only [inputColumn, ringDegree]
  omega

theorem inputLocalSlot_width (source : Source) (lane : Lane) :
    (algebraLocalSlot (inputColumn source lane)
      (inputColumn_nonzero source lane)).width = 41 := by
  unfold algebraLocalSlot
  unfold Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedAlgebraRows.Normalized.localSlot
  rw [dif_neg (by
    have sourceLt := source.isLt
    have laneLt := lane.isLt
    change source.val < 17 at sourceLt
    change lane.val < 54 at laneLt
    simp only [inputColumn]
    omega)]
  rw [dif_pos (by
    have sourceLt := source.isLt
    have laneLt := lane.isLt
    change source.val < 17 at sourceLt
    change lane.val < 54 at laneLt
    simp only [inputColumn]
    change 919 + source.val * 54 + lane.val < 1837
    omega)]

theorem outputLocalSlot_start (lane : Lane) :
    (algebraLocalSlot (outputColumn lane)
      (outputColumn_nonzero lane)).start = 75978 + lane.val * 41 := by
  unfold algebraLocalSlot
  unfold Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedAlgebraRows.Normalized.localSlot
  rw [dif_neg (by
    simp only [outputColumn]
    omega)]
  rw [dif_neg (by
    have laneLt := lane.isLt
    change lane.val < 54 at laneLt
    simp only [outputColumn]
    omega)]
  rw [dif_pos (by
    have laneLt := lane.isLt
    change lane.val < 54 at laneLt
    simp only [outputColumn]
    omega)]
  simp [outputColumn]

theorem outputLocalSlot_width (lane : Lane) :
    (algebraLocalSlot (outputColumn lane)
      (outputColumn_nonzero lane)).width = 41 := by
  unfold algebraLocalSlot
  unfold Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedAlgebraRows.Normalized.localSlot
  rw [dif_neg (by
    simp only [outputColumn]
    omega)]
  rw [dif_neg (by
    have laneLt := lane.isLt
    change lane.val < 54 at laneLt
    simp only [outputColumn]
    omega)]
  rw [dif_pos (by
    have laneLt := lane.isLt
    change lane.val < 54 at laneLt
    simp only [outputColumn]
    omega)]

private theorem inputColumn_below (source : Source) (lane : Lane) :
    generatedLayout.input source lane < algebraLocalColumns := by
  rw [
    Nightstream.Implementation.Nebula.ProductionStreamingPiRlcArtifact.Generated.layout_input]
  have sourceLt := source.isLt
  have laneLt := lane.isLt
  change source.val < 17 at sourceLt
  change lane.val < 54 at laneLt
  change 919 + source.val * 54 + lane.val < 51463
  omega

private theorem outputColumn_below (lane : Lane) :
    generatedLayout.output lane < algebraLocalColumns := by
  rw [
    Nightstream.Implementation.Nebula.ProductionStreamingPiRlcArtifact.Generated.layout_output]
  have laneLt := lane.isLt
  change lane.val < 54 at laneLt
  change 1837 + lane.val < 51463
  omega

/-- The typed input coefficient is exactly the normalized local algebra value
at its verifier-owned source-major column. -/
theorem algebraInput_eq_decodedLocalAssignment
    (assignment : Fin productionFinalColumns -> F)
    (source : Source) (lane : Lane) :
    Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedFamilyRows.Normalized.algebraInputs
        assignment source lane =
      Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedAlgebraRows.Normalized.decodedLocalAssignment
        assignment (inputColumn source lane) := by
  unfold
    Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedFamilyRows.Normalized.algebraInputs
    Nightstream.Implementation.Nebula.ProductionStreamingPiRlcAuthority.decodedInputs
    Nightstream.Implementation.Nebula.ProductPiRlcRingCombinationSound.inputRing
    Nightstream.Implementation.Nebula.ProductPiRlcRingCombinationSound.wireField
    Nightstream.Implementation.Nebula.ProductPiDecLinearCombination.fieldAt
    Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedFamilyRows.Normalized.algebraAssignment
    Nightstream.Implementation.Nebula.ProductionStreamingPiRlcArtifact.Generated.numericAssignment
  have inputBelow :
      generatedLayout.input source lane <
        Nightstream.Implementation.Nebula.ProductionStreamingPiRlcArtifact.Generated.columns := by
    simpa [algebraLocalColumns] using inputColumn_below source lane
  have columnExact :
      (⟨generatedLayout.input source lane,
        inputBelow⟩ : Fin algebraLocalColumns) =
        inputColumn source lane := by
    apply Fin.ext
    exact (inputColumn_eq_layout source lane).symm
  apply Fin.ext
  simp only [dif_pos inputBelow]
  exact congrArg Fin.val
    (congrArg
      (Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedAlgebraRows.Normalized.decodedLocalAssignment
        assignment)
      columnExact)

/-- The typed output coefficient is exactly the normalized local algebra value
at its verifier-owned output column. -/
theorem algebraOutput_eq_decodedLocalAssignment
    (assignment : Fin productionFinalColumns -> F) (lane : Lane) :
    Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedFamilyRows.Normalized.algebraOutput
        assignment lane =
      Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedAlgebraRows.Normalized.decodedLocalAssignment
        assignment (outputColumn lane) := by
  unfold
    Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedFamilyRows.Normalized.algebraOutput
    Nightstream.Implementation.Nebula.ProductPiRlcRingCombinationSound.outputRing
    Nightstream.Implementation.Nebula.ProductPiRlcRingCombinationSound.wireField
    Nightstream.Implementation.Nebula.ProductPiDecLinearCombination.fieldAt
    Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedFamilyRows.Normalized.algebraAssignment
    Nightstream.Implementation.Nebula.ProductionStreamingPiRlcArtifact.Generated.numericAssignment
  have outputBelow :
      generatedLayout.output lane <
        Nightstream.Implementation.Nebula.ProductionStreamingPiRlcArtifact.Generated.columns := by
    simpa [algebraLocalColumns] using outputColumn_below lane
  have columnExact :
      (⟨generatedLayout.output lane,
        outputBelow⟩ : Fin algebraLocalColumns) =
        outputColumn lane := by
    apply Fin.ext
    exact (outputColumn_eq_layout lane).symm
  apply Fin.ext
  simp only [dif_pos outputBelow]
  exact congrArg Fin.val
    (congrArg
      (Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedAlgebraRows.Normalized.decodedLocalAssignment
        assignment)
      columnExact)

private theorem inputWord_fits (source : Source) (lane : Lane) :
    38340 + (source.val * ringDegree + lane.val) * 41 + 41 <=
      productionFinalColumns := by
  have sourceLt := source.isLt
  have laneLt := lane.isLt
  change source.val < 17 at sourceLt
  change lane.val < 54 at laneLt
  change 38340 + (source.val * 54 + lane.val) * 41 + 41 <= 8858862
  omega

private theorem outputWord_fits (lane : Lane) :
    75978 + lane.val * 41 + 41 <= productionFinalColumns := by
  have laneLt := lane.isLt
  change lane.val < 54 at laneLt
  change 75978 + lane.val * 41 + 41 <= 8858862
  omega

private theorem inputWord_eq_localValue
    (assignment : Fin productionFinalColumns -> F)
    (source : Source) (lane : Lane) :
    wordValue assignment
        (38340 + (source.val * ringDegree + lane.val) * 41)
        (inputWord_fits source lane) =
      Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedAlgebraRows.Normalized.decodedLocalAssignment
        assignment (inputColumn source lane) := by
  unfold
    Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedAlgebraRows.Normalized.decodedLocalAssignment
    Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedAlgebraRows.Normalized.localColumnValue
  rw [dif_neg (inputColumn_nonzero source lane)]
  unfold Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedAlgebraRows.Normalized.localSlot
  rw [dif_neg (by
    have sourceLt := source.isLt
    have laneLt := lane.isLt
    change source.val < 17 at sourceLt
    change lane.val < 54 at laneLt
    simp only [inputColumn]
    omega)]
  rw [dif_pos (by
    have sourceLt := source.isLt
    have laneLt := lane.isLt
    change source.val < 17 at sourceLt
    change lane.val < 54 at laneLt
    simp only [inputColumn]
    change 919 + source.val * 54 + lane.val < 1837
    omega)]
  unfold wordValue
  rw [sourceSlotValue_eq_foldr, sourceSlotValue_eq_foldr]
  simp only [wordSlot, slotRadix, if_pos, canonicalFinIndices, inputColumn,
    ringDegree, Nat.add_sub_cancel_left]
  apply foldr_weighted_coordinate_congr
  intro index
  apply congrArg assignment
  apply Fin.ext
  change
    38340 + (source.val * 54 + lane.val) * 41 + index.val =
      38340 + (919 + source.val * 54 + lane.val - 919) * 41 + index.val
  have offset :
      919 + source.val * 54 + lane.val - 919 =
        source.val * 54 + lane.val := by
    omega
  rw [offset]

private theorem outputWord_eq_localValue
    (assignment : Fin productionFinalColumns -> F) (lane : Lane) :
    wordValue assignment (75978 + lane.val * 41)
        (outputWord_fits lane) =
      Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedAlgebraRows.Normalized.decodedLocalAssignment
        assignment (outputColumn lane) := by
  unfold
    Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedAlgebraRows.Normalized.decodedLocalAssignment
    Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedAlgebraRows.Normalized.localColumnValue
  rw [dif_neg (outputColumn_nonzero lane)]
  unfold Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedAlgebraRows.Normalized.localSlot
  rw [dif_neg (by
    simp only [outputColumn]
    omega)]
  rw [dif_neg (by
    have laneLt := lane.isLt
    change lane.val < 54 at laneLt
    simp only [outputColumn]
    omega)]
  rw [dif_pos (by
    have laneLt := lane.isLt
    change lane.val < 54 at laneLt
    simp only [outputColumn]
    omega)]
  unfold wordValue
  rw [sourceSlotValue_eq_foldr, sourceSlotValue_eq_foldr]
  simp only [wordSlot, slotRadix, if_pos, canonicalFinIndices, outputColumn,
    Nat.add_sub_cancel_left]
  apply foldr_weighted_coordinate_congr
  intro index
  apply congrArg assignment
  apply Fin.ext
  rfl

/-- Exact semantic authority for every one of the 918 source-major replay
input words. -/
theorem inputWordValue_eq_algebraInput
    (assignment : Fin productionFinalColumns -> F)
    (source : Source) (lane : Lane) :
    wordValue assignment
        (38340 + (source.val * ringDegree + lane.val) * 41)
        (inputWord_fits source lane) =
      Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedFamilyRows.Normalized.algebraInputs
        assignment source lane := by
  rw [inputWord_eq_localValue, algebraInput_eq_decodedLocalAssignment]

/-- Exact semantic authority for every one of the 54 output replay words. -/
theorem outputWordValue_eq_algebraOutput
    (assignment : Fin productionFinalColumns -> F) (lane : Lane) :
    wordValue assignment (75978 + lane.val * 41)
        (outputWord_fits lane) =
      Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedFamilyRows.Normalized.algebraOutput
        assignment lane := by
  rw [outputWord_eq_localValue, algebraOutput_eq_decodedLocalAssignment]

/-- Exact source-major input words read by the normalized Poseidon2 replay.
The list is built structurally as seventeen blocks of fifty-four words. -/
def inputReplayWords
    (assignment : Fin productionFinalColumns -> F) : List Nat :=
  (List.ofFn fun source : Source =>
    List.ofFn fun lane : Lane =>
      (wordValue assignment
        (38340 + (source.val * ringDegree + lane.val) * 41)
        (inputWord_fits source lane)).val).flatten

/-- Exact lane-major output words read by the normalized Poseidon2 replay. -/
def outputReplayWords
    (assignment : Fin productionFinalColumns -> F) : List Nat :=
  List.ofFn fun lane : Lane =>
    (wordValue assignment (75978 + lane.val * 41)
      (outputWord_fits lane)).val

/-- The 918 exact final-assignment input words are the typed source-major
PiRLC phase frame. -/
theorem inputReplayWords_eq_phaseFields
    (assignment : Fin productionFinalColumns -> F) :
    inputReplayWords assignment =
      phaseFields
        (Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedFamilyRows.Normalized.algebraInputs
          assignment) := by
  unfold inputReplayWords phaseFields sourceBlocks
  apply congrArg List.flatten
  apply congrArg List.ofFn
  funext source
  unfold ringFields
  apply congrArg List.ofFn
  funext lane
  exact congrArg Fin.val
    (inputWordValue_eq_algebraInput assignment source lane)

/-- The 54 exact final-assignment output words are the typed ring frame. -/
theorem outputReplayWords_eq_ringFields
    (assignment : Fin productionFinalColumns -> F) :
    outputReplayWords assignment =
      ringFields
        (Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedFamilyRows.Normalized.algebraOutput
          assignment) := by
  unfold outputReplayWords ringFields
  apply congrArg List.ofFn
  funext lane
  exact congrArg Fin.val
    (outputWordValue_eq_algebraOutput assignment lane)

@[simp] theorem inputReplayWords_length
    (assignment : Fin productionFinalColumns -> F) :
    (inputReplayWords assignment).length = 918 := by
  rw [inputReplayWords_eq_phaseFields]
  exact phaseFields_length _

@[simp] theorem outputReplayWords_length
    (assignment : Fin productionFinalColumns -> F) :
    (outputReplayWords assignment).length = 54 := by
  rw [outputReplayWords_eq_ringFields]
  exact ringFields_length _

/-! ## Exact fresh-word ownership -/

private def inputSourceAtOrdinal (ordinal : Fin 918) : Source :=
  ⟨ordinal.val / ringDegree, by
    have ordinalLt := ordinal.isLt
    change ordinal.val / 54 < 17
    omega⟩

private def inputLaneAtOrdinal (ordinal : Fin 918) : Lane :=
  ⟨ordinal.val % ringDegree, Nat.mod_lt _ (by simp [ringDegree])⟩

private theorem inputOrdinal_split (ordinal : Fin 918) :
    (inputSourceAtOrdinal ordinal).val * ringDegree +
        (inputLaneAtOrdinal ordinal).val = ordinal.val := by
  have split := Nat.mod_add_div ordinal.val ringDegree
  simpa only [inputSourceAtOrdinal, inputLaneAtOrdinal, Nat.add_comm,
    Nat.mul_comm] using split

private theorem inputOrdinal_fits (ordinal : Fin 918) :
    38340 + ordinal.val * 41 + 41 <= productionFinalColumns := by
  have fits := inputWord_fits
    (inputSourceAtOrdinal ordinal) (inputLaneAtOrdinal ordinal)
  rw [inputOrdinal_split] at fits
  exact fits

/-- Every bounded input ordinal decodes to its exact typed source and lane.
This is the pointwise form used by the generated call schedule. -/
theorem inputWordValue_eq_algebraInput_at_ordinal
    (assignment : Fin productionFinalColumns -> F) (ordinal : Fin 918) :
    wordValue assignment (38340 + ordinal.val * 41)
        (inputOrdinal_fits ordinal) =
      Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedFamilyRows.Normalized.algebraInputs
        assignment (inputSourceAtOrdinal ordinal)
          (inputLaneAtOrdinal ordinal) := by
  simpa only [inputOrdinal_split] using
    (inputWordValue_eq_algebraInput assignment
      (inputSourceAtOrdinal ordinal) (inputLaneAtOrdinal ordinal))

private theorem evenInput_freshOrdinal_lt
    (index : Fin evenInputRun.raw.callCount) (lane : Fin 4) (ordinal : Nat)
    (fresh : (evenInputRun.callSiteAt index.val).freshOrdinal lane =
      some ordinal) :
    ordinal < 918 := by
  by_cases first : index.val = 0
  · simp [Run.callSiteAt, evenInputRun, Run.leafClassAt,
      CallSite.freshOrdinal, first,
      Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyPoseidonCallLayout.expectedEvenInput] at fresh
    omega
  · obtain ⟨prior, exact⟩ := Nat.exists_eq_succ_of_ne_zero first
    simp [Run.callSiteAt, evenInputRun, Run.leafClassAt,
      CallSite.freshOrdinal, exact,
      Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyPoseidonCallLayout.expectedEvenInput] at fresh
    have indexLt := index.isLt
    change index.val < 229 at indexLt
    rw [exact] at indexLt
    omega

private theorem oddInput_freshOrdinal_lt
    (index : Fin oddInputRun.raw.callCount) (lane : Fin 4) (ordinal : Nat)
    (fresh : (oddInputRun.callSiteAt index.val).freshOrdinal lane =
      some ordinal) :
    ordinal < 918 := by
  by_cases first : index.val = 0
  · simp [Run.callSiteAt, oddInputRun, Run.leafClassAt,
      CallSite.freshOrdinal, first,
      Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyPoseidonCallLayout.expectedOddInput] at fresh
    rcases fresh with ⟨laneLower, fresh⟩
    omega
  · obtain ⟨prior, exact⟩ := Nat.exists_eq_succ_of_ne_zero first
    simp [Run.callSiteAt, oddInputRun, Run.leafClassAt,
      CallSite.freshOrdinal, exact,
      Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyPoseidonCallLayout.expectedOddInput] at fresh
    have indexLt := index.isLt
    change index.val < 230 at indexLt
    rw [exact] at indexLt
    omega

private theorem evenOutput_freshOrdinal_lt
    (index : Fin evenOutputRun.raw.callCount) (lane : Fin 4) (ordinal : Nat)
    (fresh : (evenOutputRun.callSiteAt index.val).freshOrdinal lane =
      some ordinal) :
    ordinal < 54 := by
  by_cases first : index.val = 0
  · simp [Run.callSiteAt, evenOutputRun, Run.leafClassAt,
      CallSite.freshOrdinal, first,
      Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyPoseidonCallLayout.expectedEvenOutput] at fresh
    omega
  · obtain ⟨prior, exact⟩ := Nat.exists_eq_succ_of_ne_zero first
    simp [Run.callSiteAt, evenOutputRun, Run.leafClassAt,
      CallSite.freshOrdinal, exact,
      Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyPoseidonCallLayout.expectedEvenOutput] at fresh
    have indexLt := index.isLt
    change index.val < 13 at indexLt
    rw [exact] at indexLt
    omega

private theorem oddOutput_freshOrdinal_lt
    (index : Fin oddOutputRun.raw.callCount) (lane : Fin 4) (ordinal : Nat)
    (fresh : (oddOutputRun.callSiteAt index.val).freshOrdinal lane =
      some ordinal) :
    ordinal < 54 := by
  by_cases first : index.val = 0
  · simp [Run.callSiteAt, oddOutputRun, Run.leafClassAt,
      CallSite.freshOrdinal, first,
      Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyPoseidonCallLayout.expectedOddOutput] at fresh
    rcases fresh with ⟨laneLower, fresh⟩
    omega
  · obtain ⟨prior, exact⟩ := Nat.exists_eq_succ_of_ne_zero first
    simp [Run.callSiteAt, oddOutputRun, Run.leafClassAt,
      CallSite.freshOrdinal, exact,
      Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyPoseidonCallLayout.expectedOddOutput] at fresh
    have indexLt := index.isLt
    change index.val < 14 at indexLt
    rw [exact] at indexLt
    omega

/-- Both exact input runs point every fresh lane at the corresponding word
of the same 918-word normalized frame. -/
theorem inputRun_freshSlot_exact
    (run : Run) (selected : run = evenInputRun ∨ run = oddInputRun)
    (index : Fin run.raw.callCount) (lane : Fin 4) (ordinal : Nat)
    (fresh : (run.callSiteAt index.val).freshOrdinal lane = some ordinal) :
    ∃ bounded : Fin 918,
      bounded.val = ordinal ∧
        (run.callSiteAt index.val).externalASlotStart lane =
          some (38340 + bounded.val * 41) := by
  rcases selected with rfl | rfl
  · let bounded : Fin 918 :=
      ⟨ordinal, evenInput_freshOrdinal_lt index lane ordinal fresh⟩
    refine ⟨bounded, rfl, ?_⟩
    change (evenInputRun.callSiteAt index.val).externalASlotStart lane =
      some (38340 + ordinal * 41)
    rw [CallSite.externalASlotStart, fresh]
    rfl
  · let bounded : Fin 918 :=
      ⟨ordinal, oddInput_freshOrdinal_lt index lane ordinal fresh⟩
    refine ⟨bounded, rfl, ?_⟩
    change (oddInputRun.callSiteAt index.val).externalASlotStart lane =
      some (38340 + ordinal * 41)
    rw [CallSite.externalASlotStart, fresh]
    rfl

/-- Both exact output runs point every fresh lane at the corresponding word
of the same 54-word normalized output frame. -/
theorem outputRun_freshSlot_exact
    (run : Run) (selected : run = evenOutputRun ∨ run = oddOutputRun)
    (index : Fin run.raw.callCount) (lane : Fin 4) (ordinal : Nat)
    (fresh : (run.callSiteAt index.val).freshOrdinal lane = some ordinal) :
    ∃ bounded : Fin 54,
      bounded.val = ordinal ∧
        (run.callSiteAt index.val).externalASlotStart lane =
          some (75978 + bounded.val * 41) := by
  rcases selected with rfl | rfl
  · let bounded : Fin 54 :=
      ⟨ordinal, evenOutput_freshOrdinal_lt index lane ordinal fresh⟩
    refine ⟨bounded, rfl, ?_⟩
    change (evenOutputRun.callSiteAt index.val).externalASlotStart lane =
      some (75978 + ordinal * 41)
    rw [CallSite.externalASlotStart, fresh]
    rfl
  · let bounded : Fin 54 :=
      ⟨ordinal, oddOutput_freshOrdinal_lt index lane ordinal fresh⟩
    refine ⟨bounded, rfl, ?_⟩
    change (oddOutputRun.callSiteAt index.val).externalASlotStart lane =
      some (75978 + ordinal * 41)
    rw [CallSite.externalASlotStart, fresh]
    rfl

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonValuePlacement
