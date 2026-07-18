import Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.Assignment

/-!
Typed matrix-column ownership for the five-ring F' public carrier.

Protocol: SuperNeo CCS/CE relation specialized to the F' public interface.
Phase: legacy CCS matrix columns to the aligned 270-public-coordinate source.
Constraint family: semantic column relocation only; this file emits no rows.

Owns: the partial inverse of the old-to-aligned column map; exact recognition
of the thirteen verifier-fixed public coordinates; relocation of every legacy
matrix coefficient; and zero completion to the full Phi81 carrier.

Does not own: numeric row order, the F' constraint polynomial, CCS
satisfaction, coefficient embedding, commitments, Ajtai setup, PiCCS, NIFS,
Rust sparse storage, R1CS lowering, row removal, or constraint counts.

Emits constraints: no.

Authority boundary: an aligned logical matrix coordinate has exactly one of
two sources: one legacy verifier-owned coefficient or the additive identity in
the fixed public-padding interval. A caller cannot supply a coefficient for a
padding coordinate. Carrier completion beyond the aligned logical width is
also definitionally zero.

| Protocol | Phase | Family | Mathematical obligation | Lean owner |
|---|---|---|---|---|
| F' / CCS | column decoding | legacy owner | every mapped old column decodes to itself | `legacyIndex?_alignedIndex` |
| F' / CCS | column decoding | fixed padding | `none` is exactly the interval `[257, 270)` | `legacyIndex?_eq_none_iff` |
| F' / CCS | matrix relocation | old coefficient | mapped columns preserve the exact source value | `alignedMatrix_at_alignedIndex` |
| F' / CCS | matrix relocation | fixed padding | all thirteen new matrix coefficients are zero | `alignedMatrix_padding_zero` |
| coefficient carrier | matrix completion | logical prefix | aligned matrix values survive completion | `carrierMatrix_at_alignedCarrierIndex` |
| coefficient carrier | matrix completion | total tail | all post-logical coordinates are zero | `carrierMatrix_completion_zero` |
-/

namespace Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.ColumnMap

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.PaperLinearAlgebra
open Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270

/-- Decode an aligned logical column to its unique legacy owner. `none`
denotes one of the thirteen verifier-fixed public coordinates. -/
def legacyIndex? (dimensions : Dimensions)
    (column : Fin dimensions.alignedLogicalWidth) :
    Option (Fin dimensions.legacyLogicalWidth) :=
  if isPublic : column.val < legacyPublicWidth then
    some ⟨column.val,
      Nat.lt_of_lt_of_le isPublic dimensions.legacyPublicFits⟩
  else if isPadding : column.val < alignedPublicWidth then
    none
  else
    some ⟨column.val - fixedPaddingWidth, by
      have columnBound :
          column.val < dimensions.legacyLogicalWidth + fixedPaddingWidth := by
        simpa only [Dimensions.alignedLogicalWidth] using column.isLt
      have legacyPositive : 0 < dimensions.legacyLogicalWidth := by
        exact Nat.lt_of_lt_of_le (by decide : 0 < legacyPublicWidth)
          dimensions.legacyPublicFits
      simp only [fixedPaddingWidth] at columnBound ⊢
      omega⟩

/-- Every legacy column round-trips through the aligned map and its typed
partial inverse. -/
theorem legacyIndex?_alignedIndex (dimensions : Dimensions)
    (column : Fin dimensions.legacyLogicalWidth) :
    legacyIndex? dimensions (alignedIndex dimensions column) = some column := by
  by_cases isPublic : column.val < legacyPublicWidth
  · have mappedValue := alignedIndex_public dimensions column isPublic
    unfold legacyIndex?
    rw [dif_pos (by simpa [mappedValue] using isPublic)]
    apply congrArg some
    apply Fin.ext
    exact mappedValue
  · have isPrivate : legacyPublicWidth ≤ column.val := Nat.not_lt.mp isPublic
    have mappedValue := alignedIndex_private dimensions column isPrivate
    have notPublic :
        ¬ (alignedIndex dimensions column).val < legacyPublicWidth := by
      rw [mappedValue]
      simp only [legacyPublicWidth, fixedPaddingWidth] at isPrivate ⊢
      omega
    have notPadding :
        ¬ (alignedIndex dimensions column).val < alignedPublicWidth := by
      rw [mappedValue]
      simp only [legacyPublicWidth, fixedPaddingWidth, alignedPublicWidth,
        publicRingColumns, ringDegree] at isPrivate ⊢
      omega
    unfold legacyIndex?
    rw [dif_neg notPublic, dif_neg notPadding]
    apply congrArg some
    apply Fin.ext
    change (alignedIndex dimensions column).val - fixedPaddingWidth = column.val
    simp only [fixedPaddingWidth] at mappedValue ⊢
    omega

/-- Exactly the thirteen inserted public coordinates have no legacy owner. -/
theorem legacyIndex?_eq_none_iff (dimensions : Dimensions)
    (column : Fin dimensions.alignedLogicalWidth) :
    legacyIndex? dimensions column = none ↔
      legacyPublicWidth ≤ column.val ∧ column.val < alignedPublicWidth := by
  unfold legacyIndex?
  split
  next isPublic =>
    simp only [Option.some_ne_none, false_iff]
    omega
  next isNotPublic =>
    split
    next isPadding =>
      simp only [true_iff]
      exact ⟨Nat.not_lt.mp isNotPublic, isPadding⟩
    next isNotPadding =>
      simp only [Option.some_ne_none, false_iff]
      omega

/-- Relocate a legacy Boolean-row matrix into the aligned logical columns.
The fixed public-padding interval is constructed as zero. -/
def alignedMatrix (dimensions : Dimensions)
    (legacy : BooleanMatrix F dimensions.rowVariables
      dimensions.legacyLogicalWidth) :
    BooleanMatrix F dimensions.rowVariables dimensions.alignedLogicalWidth :=
  fun vertex column =>
    match legacyIndex? dimensions column with
    | some oldColumn => legacy vertex oldColumn
    | none => 0

/-- Every legacy matrix coefficient is preserved at its unique aligned
column. -/
theorem alignedMatrix_at_alignedIndex (dimensions : Dimensions)
    (legacy : BooleanMatrix F dimensions.rowVariables
      dimensions.legacyLogicalWidth)
    (vertex : BooleanVertex dimensions.rowVariables)
    (column : Fin dimensions.legacyLogicalWidth) :
    alignedMatrix dimensions legacy vertex (alignedIndex dimensions column) =
      legacy vertex column := by
  simp [alignedMatrix, legacyIndex?_alignedIndex]

/-- Every inserted public-padding matrix coordinate is definitionally zero. -/
theorem alignedMatrix_padding_zero (dimensions : Dimensions)
    (legacy : BooleanMatrix F dimensions.rowVariables
      dimensions.legacyLogicalWidth)
    (vertex : BooleanVertex dimensions.rowVariables)
    (offset : Fin fixedPaddingWidth) :
    alignedMatrix dimensions legacy vertex
        (paddingLogicalColumn dimensions offset) = 0 := by
  have offsetBound := offset.isLt
  simp only [fixedPaddingWidth] at offsetBound
  have padding :
      legacyPublicWidth ≤ (paddingLogicalColumn dimensions offset).val ∧
        (paddingLogicalColumn dimensions offset).val < alignedPublicWidth := by
    simp only [paddingLogicalColumn_val, legacyPublicWidth,
      alignedPublicWidth, publicRingColumns, ringDegree]
    omega
  have decoded := (legacyIndex?_eq_none_iff dimensions
    (paddingLogicalColumn dimensions offset)).2 padding
  simp [alignedMatrix, decoded]

/-- Every aligned matrix coordinate exposes exactly one authorized source:
one legacy coefficient or a fixed zero in the padding interval. -/
theorem alignedMatrix_source_cases (dimensions : Dimensions)
    (legacy : BooleanMatrix F dimensions.rowVariables
      dimensions.legacyLogicalWidth)
    (vertex : BooleanVertex dimensions.rowVariables)
    (column : Fin dimensions.alignedLogicalWidth) :
    (∃ oldColumn,
        legacyIndex? dimensions column = some oldColumn ∧
          alignedMatrix dimensions legacy vertex column =
            legacy vertex oldColumn) ∨
      (legacyPublicWidth ≤ column.val ∧
        column.val < alignedPublicWidth ∧
          alignedMatrix dimensions legacy vertex column = 0) := by
  cases decoded : legacyIndex? dimensions column with
  | none =>
      right
      have padding := (legacyIndex?_eq_none_iff dimensions column).1 decoded
      exact ⟨padding.1, padding.2, by simp [alignedMatrix, decoded]⟩
  | some oldColumn =>
      left
      refine ⟨oldColumn, ?_, ?_⟩
      · rfl
      · simp [alignedMatrix, decoded]

/-- Complete carrier matrix derived from the aligned logical source. -/
def carrierMatrix (dimensions : Dimensions)
    (legacy : BooleanMatrix F dimensions.rowVariables
      dimensions.legacyLogicalWidth) :
    BooleanMatrix F dimensions.rowVariables dimensions.shape.carrierWidth :=
  Phi81CarrierLayout.extendMatrix 0 (alignedMatrix dimensions legacy)

/-- Legacy coefficients survive both alignment and total-carrier completion. -/
theorem carrierMatrix_at_alignedCarrierIndex (dimensions : Dimensions)
    (legacy : BooleanMatrix F dimensions.rowVariables
      dimensions.legacyLogicalWidth)
    (vertex : BooleanVertex dimensions.rowVariables)
    (column : Fin dimensions.legacyLogicalWidth) :
    carrierMatrix dimensions legacy vertex
        (alignedCarrierIndex dimensions column) = legacy vertex column := by
  unfold carrierMatrix alignedCarrierIndex
  rw [Phi81CarrierLayout.extendMatrix_embedLogical]
  exact alignedMatrix_at_alignedIndex dimensions legacy vertex column

/-- Inserted public coordinates remain zero after total-carrier completion. -/
theorem carrierMatrix_padding_zero (dimensions : Dimensions)
    (legacy : BooleanMatrix F dimensions.rowVariables
      dimensions.legacyLogicalWidth)
    (vertex : BooleanVertex dimensions.rowVariables)
    (offset : Fin fixedPaddingWidth) :
    carrierMatrix dimensions legacy vertex
        (paddingCarrierColumn dimensions offset) = 0 := by
  unfold carrierMatrix paddingCarrierColumn
  rw [Phi81CarrierLayout.extendMatrix_embedLogical]
  exact alignedMatrix_padding_zero dimensions legacy vertex offset

/-- Coordinates after the aligned logical matrix are canonical zero in the
fresh complete carrier. -/
theorem carrierMatrix_completion_zero (dimensions : Dimensions)
    (legacy : BooleanMatrix F dimensions.rowVariables
      dimensions.legacyLogicalWidth)
    (vertex : BooleanVertex dimensions.rowVariables)
    (column : Fin dimensions.shape.carrierWidth)
    (tail : dimensions.alignedLogicalWidth ≤ column.val) :
    carrierMatrix dimensions legacy vertex column = 0 := by
  exact Phi81CarrierLayout.extendMatrix_tail_zero 0
    (alignedMatrix dimensions legacy) vertex column tail

end Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.ColumnMap
