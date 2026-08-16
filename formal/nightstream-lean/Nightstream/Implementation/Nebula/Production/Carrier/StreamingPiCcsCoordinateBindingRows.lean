import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiCcsCoordinateBindingSetup
import Nightstream.Implementation.R1CS.Core.SeededPhi81RingRefinement
import Nightstream.Implementation.R1CS.Core.SeededPhi81SamplerRefinement

/-!
Contract: exact compact-row selector layout for one production PiCCS
variable-coordinate commitment phase.

Assurance tier: generated-row geometry and input-authority bridge.

Owns the active-field list, the shared constrained-zero word, the exact
21,220-entry `wordStarts` vector, the row-major 41-coordinate selector map,
the 28-coordinate zero tail, the rank-two seeded Phi81 block geometry, and
the equality of its coefficient tensor with the verifier-owned setup.

Does not own canonical-opening rows, proof that Rust `rand_chacha` implements
the pure ChaCha8 stream, Phi81 output-row soundness, public-state placement,
phase scheduling, or recursive lifecycle integration.

Emits constraints: no. A following module composes this selector certificate
with the canonical-opening and compact seeded-row soundness theorems.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBindingRows

open Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBinding
open Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBindingSetup
open Nightstream.Implementation.R1CS
open Nightstream.Protocol.Nebula.CompactCommit
open Nightstream.Protocol.Nebula.ShiftedTernary41V1
open Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism

/-- Relative columns and active global field positions used by one Rust
`enforce_commit_coordinate_fields` call. -/
structure Layout where
  activeFields : List (Fin fieldCount)
  activeFieldsNodup : activeFields.Nodup
  fieldColumn : Fin fieldCount → Nat
  digitStart : Fin fieldCount → Nat
  zeroDigitStart : Nat
  dColumn : Nat
  kappaColumn : Nat
  outputColumn : Fin (verifierRows * ringDegree) → Nat
  seededRowStart : Nat

def Layout.selected (layout : Layout) (field : Fin fieldCount) : Bool :=
  decide (field ∈ layout.activeFields)

/-- The exact start column used by Rust for one global field word. -/
def Layout.wordStart (layout : Layout) (field : Fin fieldCount) : Nat :=
  if field ∈ layout.activeFields then
    layout.digitStart field
  else
    layout.zeroDigitStart

/-- Complete global selector vector passed to `SeededPhi81LinearBlock`. -/
def Layout.wordStarts (layout : Layout) : List Nat :=
  List.ofFn layout.wordStart

theorem Layout.wordStarts_length (layout : Layout) :
    layout.wordStarts.length = fieldCount := by
  simp [Layout.wordStarts]

theorem Layout.wordStarts_getD
    (layout : Layout) (field : Fin fieldCount) :
    layout.wordStarts.getD field.val 0 = layout.wordStart field := by
  simp [Layout.wordStarts, List.getD_eq_getElem?_getD]

theorem Layout.wordStart_active
    (layout : Layout) (field : Fin fieldCount)
    (active : field ∈ layout.activeFields) :
    layout.wordStart field = layout.digitStart field := by
  simp [Layout.wordStart, active]

theorem Layout.wordStart_inactive
    (layout : Layout) (field : Fin fieldCount)
    (inactive : field ∉ layout.activeFields) :
    layout.wordStart field = layout.zeroDigitStart := by
  simp [Layout.wordStart, inactive]

private def coordinateBlockFromWords
    (production : ProductionSetup) (layout : Layout)
    (wordStarts : List Nat) :
    SeededPhi81.Block where
  rowStart := layout.seededRowStart
  wordStarts := wordStarts
  wordWidth := digitCount
  kappa := verifierRows
  messageCols := messageColumnCount
  outputColumns := List.ofFn layout.outputColumn
  superneoTransformedColumns := false
  schedule := SeededAjtai.schedule production.setup.seed.bytes verifierRows
    messageColumnCount production.setup.rejectionFuel

private theorem coordinateBlockFromWords_wordStarts
    (production : ProductionSetup) (layout : Layout)
    (wordStarts : List Nat) :
    (coordinateBlockFromWords production layout wordStarts).wordStarts =
      wordStarts := rfl

/-- Exact compact seeded block emitted after the zero word, active canonical
openings, commitment shape constants, and commitment-output allocation. -/
def coordinateBlock (production : ProductionSetup) (layout : Layout) :
    SeededPhi81.Block :=
  coordinateBlockFromWords production layout layout.wordStarts

theorem coordinateBlock_rowStart
    (production : ProductionSetup) (layout : Layout) :
    (coordinateBlock production layout).rowStart = layout.seededRowStart := rfl

theorem coordinateBlock_wordStarts
    (production : ProductionSetup) (layout : Layout) :
    (coordinateBlock production layout).wordStarts = layout.wordStarts := by
  unfold coordinateBlock
  exact coordinateBlockFromWords_wordStarts production layout layout.wordStarts

theorem coordinateBlock_wordWidth
    (production : ProductionSetup) (layout : Layout) :
    (coordinateBlock production layout).wordWidth = digitCount := rfl

theorem coordinateBlock_kappa
    (production : ProductionSetup) (layout : Layout) :
    (coordinateBlock production layout).kappa = verifierRows := rfl

theorem coordinateBlock_messageCols
    (production : ProductionSetup) (layout : Layout) :
    (coordinateBlock production layout).messageCols = messageColumnCount := rfl

theorem coordinateBlock_outputColumns
    (production : ProductionSetup) (layout : Layout) :
    (coordinateBlock production layout).outputColumns =
      List.ofFn layout.outputColumn := rfl

theorem coordinateBlock_transformed
    (production : ProductionSetup) (layout : Layout) :
    (coordinateBlock production layout).superneoTransformedColumns = false := rfl

theorem coordinateBlock_chunkSize
    (production : ProductionSetup) (layout : Layout) :
    (coordinateBlock production layout).schedule.chunkSize =
      SeededAjtai.chunkSize messageColumnCount := rfl

theorem coordinateBlock_exact_geometry
    (production : ProductionSetup) (layout : Layout) :
    (coordinateBlock production layout).rowStart = layout.seededRowStart /\
      (coordinateBlock production layout).wordStarts.length = fieldCount /\
      (coordinateBlock production layout).wordWidth = 41 /\
      (coordinateBlock production layout).kappa = 2 /\
      (coordinateBlock production layout).messageCols = 16112 /\
      (coordinateBlock production layout).outputColumns.length = 108 /\
      (coordinateBlock production layout).superneoTransformedColumns = false /\
      (coordinateBlock production layout).schedule.chunkSize = 16112 := by
  constructor
  · exact coordinateBlock_rowStart production layout
  constructor
  · rw [coordinateBlock_wordStarts]
    exact layout.wordStarts_length
  constructor
  · rw [coordinateBlock_wordWidth]
    decide
  constructor
  · rw [coordinateBlock_kappa]
    decide
  constructor
  · rw [coordinateBlock_messageCols]
    decide
  constructor
  · rw [coordinateBlock_outputColumns, List.length_ofFn]
    decide
  constructor
  · exact coordinateBlock_transformed production layout
  · rw [coordinateBlock_chunkSize]
    exact exact_chunk_geometry.1

/-- The compact block and the Module-SIS map read the same complete sampled
coefficient tensor. Fast Lean ChaCha8 is replaced by its pure model here. -/
theorem coordinateBlock_baseRotations
    (production : ProductionSetup) (layout : Layout) :
    (coordinateBlock production layout).baseRotations =
      production.setup.outputs := by
  rw [SeededPhi81SamplerRefinement.blockBaseRotations_eq_pure]
  rfl

private theorem ringOfList_baseRotation
    (production : ProductionSetup)
    (output : Fin verifierRows) (messageCol : Fin messageColumnCount) :
    SeededPhi81RingRefinement.ringOfList
        ((production.setup.outputs.getD output.val []).getD
          messageCol.val []) =
      production.setup.verifierKey output messageCol := by
  funext lane
  apply Fin.ext
  rfl

/-- Every compact coefficient is exactly the corresponding coefficient of
the verifier-owned executable Phi81 matrix action. -/
theorem coordinateBlock_coefficient_residue
    (production : ProductionSetup) (layout : Layout)
    (output : Fin verifierRows) (messageCol : Fin messageColumnCount)
    (messageRow coordinate : Fin ringDegree) :
    SeededPhi81RingRefinement.residueNat
        ((coordinateBlock production layout).coefficient
          output.val messageCol.val messageRow.val coordinate.val) =
      CarrierAction.rightCoefficient
        (production.setup.verifierKey output messageCol)
        coordinate messageRow := by
  let base :=
    ((production.setup.outputs.getD output.val []).getD messageCol.val [])
  have rotated := SeededPhi81RingRefinement.ringOfList_rotatePow
    messageRow.val messageRow.isLt base
  rw [ringOfList_baseRotation production output messageCol] at rotated
  have atCoordinate := congrFun rotated coordinate
  unfold SeededPhi81.Block.coefficient
  rw [coordinateBlock_baseRotations]
  change SeededPhi81RingRefinement.residueNat
      ((SeededPhi81.rotatePow messageRow.val base).getD coordinate.val 0) =
    CarrierAction.rightCoefficient
      (production.setup.verifierKey output messageCol) coordinate messageRow
  simpa only [SeededPhi81RingRefinement.ringOfList,
    CarrierAction.rightCoefficient] using atCoordinate

theorem coordinateBlock_coefficient_mod
    (production : ProductionSetup) (layout : Layout)
    (output : Fin verifierRows) (messageCol : Fin messageColumnCount)
    (messageRow coordinate : Fin ringDegree) :
    ((coordinateBlock production layout).coefficient
        output.val messageCol.val messageRow.val coordinate.val) %
        goldilocksP =
      (CarrierAction.rightCoefficient
        (production.setup.verifierKey output messageCol)
        coordinate messageRow).val := by
  exact congrArg Fin.val
    (coordinateBlock_coefficient_residue production layout output
      messageCol messageRow coordinate)

private theorem wordIndex_quotient
    (field : Fin fieldCount) (digit : Fin digitCount) :
    wordIndex field digit / digitCount = field.val := by
  unfold wordIndex
  rw [Nat.mul_comm]
  rw [Nat.mul_add_div (by decide : 0 < digitCount),
    Nat.div_eq_of_lt digit.isLt, Nat.add_zero]

private theorem wordIndex_remainder
    (field : Fin fieldCount) (digit : Fin digitCount) :
    wordIndex field digit % digitCount = digit.val := by
  unfold wordIndex
  exact Nat.mul_add_mod_of_lt digit.isLt

/-- Every real coordinate selects the named global word and digit. This is
the exact `bit_index / 41`, `bit_index % 41` rule used by Rust. -/
theorem coordinateBlock_bitColumn
    (production : ProductionSetup) (layout : Layout)
    (field : Fin fieldCount) (digit : Fin digitCount) :
    (coordinateBlock production layout).bitColumn (wordIndex field digit) =
      some (layout.wordStart field + digit.val) := by
  have bound := wordIndex_lt field digit
  have selectorBound :
      wordIndex field digit < layout.wordStarts.length * digitCount := by
    rw [layout.wordStarts_length]
    exact bound
  unfold SeededPhi81.Block.bitColumn
  rw [coordinateBlock_wordWidth, coordinateBlock_wordStarts]
  rw [if_neg (by decide : digitCount ≠ 0)]
  rw [if_pos selectorBound]
  rw [wordIndex_quotient, wordIndex_remainder]
  rw [layout.wordStarts_getD]

/-- The final 28 matrix coordinates have no input column. They therefore
contribute zero, instead of reading unconstrained padding advice. -/
theorem coordinateBlock_tail_bitColumn_none
    (production : ProductionSetup) (layout : Layout) (bitIndex : Nat)
    (tail : fieldCount * digitCount ≤ bitIndex) :
    (coordinateBlock production layout).bitColumn bitIndex = none := by
  have outside : ¬ bitIndex < layout.wordStarts.length * digitCount := by
    rw [layout.wordStarts_length]
    exact Nat.not_lt.mpr tail
  unfold SeededPhi81.Block.bitColumn
  rw [coordinateBlock_wordWidth, coordinateBlock_wordStarts]
  rw [if_neg (by decide : digitCount ≠ 0)]
  rw [if_neg outside]

/-- Exact authority statement supplied by the zero rows and active
canonical-opening rows. It is independent of the claimed commitment output. -/
def SourceColumnsExact
    (layout : Layout) (assignment : Nat → Nat) (fields : Fields) : Prop :=
  (∀ field ∈ layout.activeFields, ∀ digit : Fin digitCount,
    assignment (layout.digitStart field + digit.val) =
      (integerResidue (signedDigit (fields field) digit)).val) /\
  (∀ digit : Fin digitCount,
    assignment (layout.zeroDigitStart + digit.val) =
      (integerResidue 0).val)

/-- Active words read their canonical field digits. Inactive words read the
same constrained-zero word. -/
theorem selected_word_exact
    {layout : Layout} {assignment : Nat → Nat} {fields : Fields}
    (exact : SourceColumnsExact layout assignment fields)
    (field : Fin fieldCount) (digit : Fin digitCount) :
    assignment (layout.wordStart field + digit.val) =
      (integerResidue
        (if layout.selected field then signedDigit (fields field) digit
          else 0)).val := by
  by_cases active : field ∈ layout.activeFields
  · rw [layout.wordStart_active field active]
    simp only [Layout.selected, decide_eq_true active, if_true]
    exact exact.1 field active digit
  · rw [layout.wordStart_inactive field active]
    simp only [Layout.selected, decide_eq_false active]
    exact exact.2 digit

/-- The selector column read by the compact block equals the corresponding
coefficient of the phase-masked Module-SIS witness. -/
theorem selected_coordinate_exact
    {production : ProductionSetup} {layout : Layout}
    {assignment : Nat → Nat} {fields : Fields}
    (exact : SourceColumnsExact layout assignment fields)
    (field : Fin fieldCount) (digit : Fin digitCount) :
    assignment
        (((coordinateBlock production layout).bitColumn
          (wordIndex field digit)).getD 0) =
      (integerResidue
        (maskedWitness fields layout.selected
          (messagePosition field digit).1
          (messagePosition field digit).2)).val := by
  rw [coordinateBlock_bitColumn]
  simp only [Option.getD_some]
  rw [selected_word_exact exact field digit]
  rw [maskedWitness_at]

set_option maxRecDepth 10000 in
/-- Every dense compact-block input is the exact coefficient of the masked
Module-SIS witness. The final 28 matrix coordinates reduce to zero on both
sides. -/
theorem coordinateBlock_inputValue_exact
    {production : ProductionSetup} {layout : Layout}
    {assignment : Nat → Nat} {fields : Fields}
    (exact : SourceColumnsExact layout assignment fields)
    (messageCol : Fin messageColumnCount) (messageRow : Fin ringDegree) :
    SeededPhi81RingRefinement.residueNat
        ((coordinateBlock production layout).inputValue assignment
          messageCol.val messageRow.val) =
      integerResidue
        (maskedWitness fields layout.selected messageCol messageRow) := by
  let index := flatIndex messageCol messageRow
  by_cases valid : index < fieldCount * digitCount
  · let field : Fin fieldCount :=
      ⟨index / digitCount, by
        unfold fieldCount digitCount at valid ⊢
        omega⟩
    let digit : Fin digitCount :=
      ⟨index % digitCount, Nat.mod_lt _ (by decide)⟩
    have wordEqual : wordIndex field digit = index := by
      unfold wordIndex field digit
      simpa only [Nat.mul_comm] using Nat.div_add_mod index digitCount
    have bitColumn :
        (coordinateBlock production layout).bitColumn index =
          some (layout.wordStart field + digit.val) := by
      rw [← wordEqual]
      exact coordinateBlock_bitColumn production layout field digit
    have selected := selected_word_exact exact field digit
    have masked :
        maskedWitness fields layout.selected messageCol messageRow =
          if layout.selected field then signedDigit (fields field) digit
          else 0 := by
      unfold maskedWitness
      change (if valid' : index < fieldCount * digitCount then
          if layout.selected
              ⟨index / digitCount, by
                unfold fieldCount digitCount at valid' ⊢
                omega⟩ then
            signedDigit
              (fields ⟨index / digitCount, by
                unfold fieldCount digitCount at valid' ⊢
                omega⟩)
              ⟨index % digitCount, Nat.mod_lt _ (by decide)⟩
          else 0
        else 0) = _
      rw [dif_pos valid]
    have nativeIndex :
        messageRow.val *
              (coordinateBlock production layout).messageCols +
            messageCol.val = index := by
      rw [coordinateBlock_messageCols]
      rfl
    have nativeBitColumn :
        (coordinateBlock production layout).bitColumn
            (messageRow.val *
                (coordinateBlock production layout).messageCols +
              messageCol.val) =
          some (layout.wordStart field + digit.val) := by
      rw [nativeIndex]
      exact bitColumn
    rw [SeededPhi81.Block.inputValue_eq_of_bitColumn_some nativeBitColumn]
    calc
      SeededPhi81RingRefinement.residueNat
          (assignment (layout.wordStart field + digit.val)) =
          SeededPhi81RingRefinement.residueNat
            (integerResidue
              (if layout.selected field then
                signedDigit (fields field) digit else 0)).val :=
        congrArg SeededPhi81RingRefinement.residueNat selected
      _ = integerResidue
          (if layout.selected field then
            signedDigit (fields field) digit else 0) :=
        SeededPhi81RingRefinement.residueNat_fin_val _
      _ = integerResidue
          (maskedWitness fields layout.selected messageCol messageRow) :=
        congrArg integerResidue masked.symm
  · have tail : fieldCount * digitCount ≤ index := Nat.le_of_not_gt valid
    have noColumn :
        (coordinateBlock production layout).bitColumn index = none :=
      coordinateBlock_tail_bitColumn_none production layout index tail
    have maskedZero :
        maskedWitness fields layout.selected messageCol messageRow = 0 := by
      unfold maskedWitness
      change (if valid' : index < fieldCount * digitCount then
          (if layout.selected
              ⟨index / digitCount, by
                unfold fieldCount digitCount at valid' ⊢
                omega⟩ then
            signedDigit
              (fields ⟨index / digitCount, by
                unfold fieldCount digitCount at valid' ⊢
                omega⟩)
              ⟨index % digitCount, Nat.mod_lt _ (by decide)⟩
          else 0)
        else 0) = 0
      rw [dif_neg valid]
    have nativeIndex :
        messageRow.val *
              (coordinateBlock production layout).messageCols +
            messageCol.val = index := by
      rw [coordinateBlock_messageCols]
      rfl
    have nativeNoColumn :
        (coordinateBlock production layout).bitColumn
            (messageRow.val *
                (coordinateBlock production layout).messageCols +
              messageCol.val) = none := by
      rw [nativeIndex]
      exact noColumn
    rw [SeededPhi81.Block.inputValue_eq_zero_of_bitColumn_none nativeNoColumn]
    rw [maskedZero]
    rfl

theorem exact_tail_width :
    messageColumnCount * ringDegree - fieldCount * digitCount = 28 := by
  exact exact_geometry.2.2.1

end Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBindingRows
