import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.NifsPaper.PiDec
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.NifsPaper.PiDec.Weights
import Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.Assignment
import Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.PublicInput

/-!
Typed 270-coordinate public-input refinement for production `PiDEC`.

Protocol: SuperNeo `Pi_DEC` inside the fixed F' NIFS.
Phase: production packed-`X` decoding and public-input recomposition.
Constraint family: the 270 active `X` coordinates only; this file emits no
rows.

Owns: the exact logical-to-production transpose; a kernel-checked inverse for
all 270 coordinates; decoding into the independent typed Phi81 public carrier;
and refinement of strict packed recomposition to the semantic `PiDEC`
public-input equation.

Does not own: strict R1CS-row soundness, the thirteen zero pins, private CE
openings, commitments, evaluations, Ajtai binding, NIFS composition, costs,
or row removal.

Emits constraints: no. It interprets existing production columns.

Authority boundary: production stores `X` in lane-major order while the typed
relation uses logical block-major order. `packedSlot_exact` proves that the
decoder is a permutation, so no coordinate is dropped, duplicated, or supplied
through a default read for a production claim.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_dec.verify.public_input.layout` | logical `block*54+lane` maps to production `lane*5+block` | checked | `packedSlot` |
| `nifs.pi_dec.verify.public_input.coverage` | the transpose is a bijection on all 270 coordinates | derived | `packedSlot_exact` |
| `nifs.pi_dec.verify.public_input.decode` | each typed coordinate reads its unique active `X` column | checked | `decodedPublicInput_apply` |
| `nifs.pi_dec.verify.public_input.decode.injective` | full-width typed decoding loses no packed coordinate | derived | `decode_injective_of_length` |
| `nifs.pi_dec.verify.public_input.recompose` | strict packed recomposition implies typed semantic recomposition | derived | `strictAccepted_typedPublicInputEquation` |
-/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiDec.PublicInputBridge

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270
open Nightstream.Implementation.R1CS.PiDecStrictCompiler
open Nightstream.Implementation.R1CS.FPrimeFullHistoryPiDec
open Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper
open Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiDec.Weights

/-- Production slot for one logical public coordinate. Production lays out
five ring blocks inside each of the 54 coefficient lanes. -/
def packedSlot (column : Fin alignedPublicWidth) : Fin alignedPublicWidth :=
  ⟨(column.val % ringDegree) * publicRingColumns +
      column.val / ringDegree, by
    have columnLt := column.isLt
    have laneLt := Nat.mod_lt column.val (by decide : 0 < ringDegree)
    have blockLt : column.val / 54 < 5 := by
      simp only [alignedPublicWidth, ringDegree, publicRingColumns] at columnLt
      omega
    simp only [alignedPublicWidth, ringDegree, publicRingColumns] at laneLt ⊢
    omega⟩

/-- Inverse transpose from a production slot back to logical block-major
order. -/
def logicalSlot (slot : Fin alignedPublicWidth) : Fin alignedPublicWidth :=
  ⟨(slot.val % publicRingColumns) * ringDegree +
      slot.val / publicRingColumns, by
    have slotLt := slot.isLt
    have blockLt := Nat.mod_lt slot.val (by decide : 0 < publicRingColumns)
    have laneLt : slot.val / 5 < 54 := by
      simp only [alignedPublicWidth, ringDegree, publicRingColumns] at slotLt
      omega
    simp only [alignedPublicWidth, ringDegree, publicRingColumns] at blockLt ⊢
    omega⟩

theorem logicalSlot_packedSlot (column : Fin alignedPublicWidth) :
    logicalSlot (packedSlot column) = column := by
  apply Fin.ext
  have columnLt := column.isLt
  simp only [logicalSlot, packedSlot, alignedPublicWidth, ringDegree,
    publicRingColumns] at columnLt ⊢
  omega

theorem packedSlot_logicalSlot (slot : Fin alignedPublicWidth) :
    packedSlot (logicalSlot slot) = slot := by
  apply Fin.ext
  have slotLt := slot.isLt
  simp only [logicalSlot, packedSlot, alignedPublicWidth, ringDegree,
    publicRingColumns] at slotLt ⊢
  omega

/-- The layout conversion covers the complete public carrier exactly once. -/
theorem packedSlot_exact :
    Function.Injective packedSlot /\
      forall slot, exists column, packedSlot column = slot := by
  constructor
  · intro left right equal
    have inverseEqual := congrArg logicalSlot equal
    simpa only [logicalSlot_packedSlot] using inverseEqual
  · intro slot
    exact ⟨logicalSlot slot, packedSlot_logicalSlot slot⟩

/-- Forget the dimension witness after proving that every fixed F' typed
public carrier has the same exact 270-coordinate domain. -/
def alignedColumn {dimensions : Dimensions}
    (column : Fin dimensions.shape.publicWidth) : Fin alignedPublicWidth :=
  ⟨column.val, by
    simpa [alignedPublicWidth] using column.isLt⟩

/-- Restore the dimension witness for one fixed-profile logical coordinate. -/
def typedColumn {dimensions : Dimensions}
    (column : Fin alignedPublicWidth) : Fin dimensions.shape.publicWidth :=
  ⟨column.val, by
    have widthEq : dimensions.shape.publicWidth = alignedPublicWidth := by
      simp [alignedPublicWidth, ringDegree, publicRingColumns]
    rw [widthEq]
    exact column.isLt⟩

@[simp] theorem alignedColumn_typedColumn {dimensions : Dimensions}
    (column : Fin alignedPublicWidth) :
    alignedColumn (typedColumn (dimensions := dimensions) column) = column := by
  apply Fin.ext
  rfl

/-- Decode one production packed public input into logical typed order. -/
def decode {dimensions : Dimensions}
    (packed : PackedPublicInput) : Phi81Relation.PublicInput dimensions.shape :=
  fun column => packed.data.getD (packedSlot (alignedColumn column)).val 0

@[simp] theorem decode_apply {dimensions : Dimensions}
    (packed : PackedPublicInput)
    (column : Fin dimensions.shape.publicWidth) :
    decode (dimensions := dimensions) packed column =
      packed.data.getD (packedSlot (alignedColumn column)).val 0 := by
  rfl

/-- Full-width packed decoding is injective. Together with
`packedSlot_exact`, the length premises ensure that every packed coordinate is
observed by exactly one typed coordinate, so a decoder equality cannot hide a
mutation or truncation. -/
theorem decode_injective_of_length {dimensions : Dimensions}
    {left right : PackedPublicInput}
    (leftLength : left.data.length = alignedPublicWidth)
    (rightLength : right.data.length = alignedPublicWidth)
    (decoded : decode (dimensions := dimensions) left =
      decode (dimensions := dimensions) right) :
    left = right := by
  apply PackedPublicInput.eq_of_data_eq
  apply List.ext_get
  · exact leftLength.trans rightLength.symm
  · intro index leftLt rightLt
    have slotLt : index < alignedPublicWidth := by
      rw [← leftLength]
      exact leftLt
    let slot : Fin alignedPublicWidth := ⟨index, slotLt⟩
    have decodedAt := congrFun decoded
      (typedColumn (dimensions := dimensions) (logicalSlot slot))
    change left.data.getD
        (packedSlot (alignedColumn
          (typedColumn (dimensions := dimensions) (logicalSlot slot)))).val 0 =
      right.data.getD
        (packedSlot (alignedColumn
          (typedColumn (dimensions := dimensions) (logicalSlot slot)))).val 0
      at decodedAt
    rw [alignedColumn_typedColumn, packedSlot_logicalSlot] at decodedAt
    calc
      left.data[index] = left.data.getD index 0 := by
        rw [← List.getElem_eq_getD]
      _ = right.data.getD index 0 := decodedAt
      _ = right.data[index] := by
        rw [List.getElem_eq_getD]

/-- Direct decoder used by the production claim bridge. -/
def decodedPublicInput (dimensions : Dimensions)
    (assignment : Nat -> Nat) (claim : ClaimLayout) :
    Phi81Relation.PublicInput dimensions.shape :=
  decode (dimensions := dimensions) (decodedPackedInput assignment claim)

@[simp] theorem decodedPublicInput_apply
    (dimensions : Dimensions) (assignment : Nat -> Nat)
    (claim : ClaimLayout) (column : Fin dimensions.shape.publicWidth)
    (slotLt : (packedSlot (alignedColumn column)).val <
      claim.xActiveCols.length) :
    decodedPublicInput dimensions assignment claim column =
      residue (assignment (claim.xActiveCols.getD
        (packedSlot (alignedColumn column)).val 0)) := by
  unfold decodedPublicInput decode decodedPackedInput values
  rw [List.getD_eq_getElem?_getD, List.getElem?_map,
    List.getElem?_eq_getElem slotLt]
  simp only [Option.map_some, Option.getD_some]
  rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem slotLt]
  simp

private theorem childLayout_mem
    (index : Fin productionGlobalParams.k) :
    childLayout index ∈ layout.children := by
  unfold childLayout
  exact List.get_mem layout.children (Fin.cast production_child_count index)

theorem decodedParent_length (assignment : Nat -> Nat) :
    (decodedPackedInput assignment layout.parent).data.length =
      alignedPublicWidth := by
  simp [decodedPackedInput, values, alignedPublicWidth,
    ringDegree, publicRingColumns, FPrimeFullHistoryPiDec.layout]

theorem decodedChild_length (assignment : Nat -> Nat)
    (index : Fin productionGlobalParams.k) :
    (decodedPackedInput assignment (childLayout index)).data.length =
      alignedPublicWidth := by
  calc
    (decodedPackedInput assignment (childLayout index)).data.length =
        (childLayout index).xActiveCols.length := by
      simp [decodedPackedInput, values]
    _ = layout.parent.xActiveCols.length :=
      production_public_shape.xLengths
        (childLayout index) (childLayout_mem index)
    _ = alignedPublicWidth := by
      have parent := decodedParent_length assignment
      simpa [decodedPackedInput, values] using parent

private theorem combinePublicInputs_apply
    {shape : Phi81Relation.Shape} {count : Nat}
    (weights : Fin count -> F)
    (items : Fin count -> Phi81Relation.PublicInput shape)
    (column : Fin shape.publicWidth) :
    PiDECAlgebra.PublicInput.combinePublicInputs weights items column =
      combineScalars weights (fun index => items index column) := by
  induction count with
  | zero => rfl
  | succ count inductionHypothesis =>
      simp only [PiDECAlgebra.PublicInput.combinePublicInputs,
        PiDECAlgebra.PublicInput.publicInputScale,
        PiRLCAlgebra.PublicInput.publicAdd, combineScalars]
      rw [inductionHypothesis
        (fun index => weights index.succ)
        (fun index => items index.succ)]

theorem semanticRecompose_apply
    {shape : Phi81Relation.Shape}
    (items : Fin productionGlobalParams.k ->
      Phi81Relation.PublicInput shape)
    (column : Fin shape.publicWidth) :
    PiDECAlgebra.PublicInput.recomposePublicInput items column =
      combineScalars
        Phi81Relation.EvaluationHomomorphism.PiDEC.radixWeight
        (fun index => items index column) := by
  exact combinePublicInputs_apply
    Phi81Relation.EvaluationHomomorphism.PiDEC.radixWeight items column

private theorem combined_length
    (items : Fin productionGlobalParams.k -> PackedPublicInput)
    (firstLength : (items firstIndex).data.length = alignedPublicWidth) :
    (combinePackedPublicInput items).data.length = alignedPublicWidth := by
  simp [combinePackedPublicInput, combineList, firstLength]

/-- Decoding commutes with public-input recomposition. The length premise
ensures the production output list contains every slot; individual child reads
remain the same total reads used by the production recomposition operation. -/
theorem decode_combine
    {dimensions : Dimensions}
    (items : Fin productionGlobalParams.k -> PackedPublicInput)
    (firstLength : (items firstIndex).data.length = alignedPublicWidth) :
    decode (dimensions := dimensions) (combinePackedPublicInput items) =
      PiDECAlgebra.PublicInput.recomposePublicInput
        (fun index => decode (dimensions := dimensions) (items index)) := by
  funext column
  let slot := packedSlot (alignedColumn column)
  have outputLength := combined_length items firstLength
  have slotLt : slot.val < (combinePackedPublicInput items).data.length := by
    rw [outputLength]
    exact slot.isLt
  calc
    decode (dimensions := dimensions) (combinePackedPublicInput items) column =
        combineScalar fun index => (items index).data.getD slot.val 0 := by
      change (combinePackedPublicInput items).data.getD slot.val 0 = _
      rw [List.getD_eq_getElem?_getD,
        List.getElem?_eq_getElem slotLt]
      simp only [Option.getD_some, combinePackedPublicInput, combineList,
        List.getElem_map, List.getElem_range]
    _ = combineScalars
        Phi81Relation.EvaluationHomomorphism.PiDEC.radixWeight
        (fun index => (items index).data.getD slot.val 0) :=
      combineScalar_eq _
    _ = PiDECAlgebra.PublicInput.recomposePublicInput
        (fun index => decode (dimensions := dimensions) (items index)) column := by
      symm
      exact semanticRecompose_apply
        (fun index => decode (dimensions := dimensions) (items index)) column

/-- Strict semantic `PiDEC` acceptance implies the exact typed public-input
equation after the proved 270-coordinate permutation. This theorem still has
strict acceptance as a premise; the generated-row-to-acceptance edge remains
separate and open. -/
theorem strictAccepted_typedPublicInputEquation
    (dimensions : Dimensions)
    (assignment : Nat -> Nat)
    (accepted : PiDecStrictCompiler.Accepted layout assignment) :
    decodedPublicInput dimensions assignment layout.parent =
      PiDECAlgebra.PublicInput.recomposePublicInput fun index =>
        decodedPublicInput dimensions assignment (childLayout index) := by
  unfold decodedPublicInput
  rw [Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiDec.strictAccepted_packedPublicInputEquation
    assignment accepted]
  apply decode_combine
  exact decodedChild_length assignment firstIndex

end Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiDec.PublicInputBridge
