import Nightstream.Implementation.NebulaV2.ProductionFreshLowNormEncoding
import Nightstream.Implementation.R1CS.Correspondence.FieldEncoding.LinearCompiler

/-!
Contract: exact source-column substitution for one V2 fresh claim.

The first 14,364 source columns are direct low-norm bits: 540 public bits and
the complete three-lane memory block. Each of the `privateWidth` later source
fields is replaced by its weighted 41-coordinate centered-trit word. Decoding
the compiled assignment recovers every source coordinate.

Owns the source and encoded column maps, linear substitution, and exact
decoding. Does not own generated source rows, commitments, NIFS, or Rust.

Assurance tier: implementation model.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 100000

namespace Nightstream.Implementation.NebulaV2.ProductionFreshLinearSubstitution

open Nightstream.Implementation.NebulaV2
open Nightstream.Implementation.NebulaV2.ProductionFreshLowNormEncoding
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.CenteredTernaryField
open Nightstream.Implementation.R1CS.CenteredTernaryLinearCompiler
open Nightstream.Implementation.R1CS.LinearSubstitution
open Nightstream.Protocol.NebulaV2.ShiftedTernary41V1

def encodedPrivateColumn (field index : Nat) : Nat :=
  directWidth + field * digitCount + index

def expansion : ColumnExpansion := fun sourceColumn =>
  if sourceColumn < directWidth then
    [(sourceColumn, 1)]
  else
    (List.range digitCount).map fun index =>
      (encodedPrivateColumn (sourceColumn - directWidth) index,
        3 ^ index % goldilocksP)

def layout (privateWidth : Nat) :
    CenteredTernaryLinearCompiler.Layout privateWidth where
  sourceColumn := fun field => directWidth + field.val
  encodedColumn := fun field index => encodedPrivateColumn field.val index
  expansion := expansion
  privateExpansion := by
    intro field
    have notDirect : ¬ directWidth + field.val < directWidth := by omega
    simp only [expansion, notDirect, ↓reduceIte]
    apply List.map_congr_left
    intro index member
    simp only [Prod.mk.injEq, and_true]
    simp [encodedPrivateColumn]

@[simp] theorem layout_sourceColumn
    (privateWidth : Nat) (field : Fin privateWidth) :
    (layout privateWidth).sourceColumn field = directWidth + field.val := rfl

@[simp] theorem layout_encodedColumn
    (privateWidth : Nat) (field : Fin privateWidth) (index : Nat) :
    (layout privateWidth).encodedColumn field index =
      encodedPrivateColumn field.val index := rfl

theorem expansion_direct {column : Nat} (bounded : column < directWidth) :
    expansion column = [(column, 1)] := by
  simp [expansion, bounded]

theorem expansion_private {column : Nat}
    (privateColumn : directWidth <= column) :
    expansion column =
      (List.range digitCount).map fun index =>
        (encodedPrivateColumn (column - directWidth) index,
          3 ^ index % goldilocksP) := by
  simp [expansion, Nat.not_lt.mpr privateColumn]

def encodedNat {privateWidth : Nat}
    (source : SourceAssignment privateWidth) : Nat -> Nat :=
  fun column =>
    if within : column < logicalWidth privateWidth then
      (encodeLogical source ⟨column, within⟩).val
    else 0

def sourceNat {privateWidth : Nat}
    (source : SourceAssignment privateWidth) : Nat -> Nat :=
  fun column =>
    if within : column < sourceWidth privateWidth then
      (source ⟨column, within⟩).val
    else 0

@[simp] theorem sourceNat_sourceColumn {privateWidth : Nat}
    (source : SourceAssignment privateWidth)
    (column : Fin (sourceWidth privateWidth)) :
    sourceNat source column.val = (source column).val := by
  unfold sourceNat
  simp only [dif_pos column.isLt]

theorem encodedPrivateColumn_lt_logicalWidth {privateWidth : Nat}
    (field : Fin privateWidth) (index : Fin digitCount) :
    encodedPrivateColumn field.val index.val < logicalWidth privateWidth := by
  apply Nat.lt_of_lt_of_le _ (payloadWidth_le_logicalWidth privateWidth)
  simp only [encodedPrivateColumn, payloadWidth, directWidth, digitCount] at *
  omega

theorem encodedNat_direct {privateWidth : Nat}
    (source : SourceAssignment privateWidth) (column : Fin directWidth) :
    encodedNat source column.val = (source (directSourceColumn column)).val := by
  have within : column.val < logicalWidth privateWidth :=
    Nat.lt_of_lt_of_le column.isLt
      (Nat.le_trans (directWidth_le_payloadWidth privateWidth)
        (payloadWidth_le_logicalWidth privateWidth))
  unfold encodedNat
  simp only [dif_pos within]
  have columnEq :
      (⟨column.val, within⟩ : Fin (logicalWidth privateWidth)) =
        payloadColumn (finSumFinEquiv (Sum.inl column :
          Fin directWidth ⊕ Fin (privateWidth * digitCount))) := by
    apply Fin.ext
    rfl
  rw [columnEq, encodeLogical_direct]

theorem encodedNat_public {privateWidth : Nat}
    (source : SourceAssignment privateWidth) (column : Fin publicWidth) :
    encodedNat source column.val = (source (publicSourceColumn column)).val := by
  simpa [publicSourceColumn] using encodedNat_direct source
    (Fin.castLE publicWidth_le_directWidth column)

theorem encodedNat_private {privateWidth : Nat}
    (source : SourceAssignment privateWidth) (field : Fin privateWidth)
    (index : Fin digitCount) :
    encodedNat source (encodedPrivateColumn field.val index.val) =
      finiteEncode (source (privateSourceColumn field)).val index := by
  have within := encodedPrivateColumn_lt_logicalWidth field index
  unfold encodedNat
  simp only [dif_pos within]
  have columnEq :
      (⟨encodedPrivateColumn field.val index.val, within⟩ :
        Fin (logicalWidth privateWidth)) =
        payloadColumn (finSumFinEquiv (Sum.inr
          (finProdFinEquiv (field, index)) :
          Fin directWidth ⊕ Fin (privateWidth * digitCount))) := by
    apply Fin.ext
    change directWidth + field.val * digitCount + index.val =
      directWidth + (index.val + digitCount * field.val)
    simp only [digitCount]
    omega
  rw [columnEq, encodeLogical_private]
  exact centeredFieldDigit_eq_compilerDigit
    (canonical (source (privateSourceColumn field))) index

theorem encodedWord_eq_finiteEncode {privateWidth : Nat}
    (source : SourceAssignment privateWidth) (field : Fin privateWidth) :
    encodedWord (layout privateWidth) (encodedNat source) field =
      finiteEncode (source (privateSourceColumn field)).val := by
  funext index
  exact encodedNat_private source field index

theorem decoded_private {privateWidth : Nat}
    (source : SourceAssignment privateWidth) (field : Fin privateWidth) :
    decodedAssignment (layout privateWidth) (encodedNat source)
        (directWidth + field.val) =
      (source (privateSourceColumn field)).val := by
  have decoded := decodedPrivateColumn
    (layout privateWidth) (encodedNat source) field
  rw [encodedWord_eq_finiteEncode source field,
    decodeFiniteWord_finiteEncode] at decoded
  · exact decoded
  · exact (source (privateSourceColumn field)).isLt

theorem decoded_direct {privateWidth : Nat}
    (source : SourceAssignment privateWidth) (column : Fin directWidth) :
    decodedAssignment (layout privateWidth) (encodedNat source) column.val =
      (source (directSourceColumn column)).val := by
  unfold decodedAssignment LinearSubstitution.assignment
  rw [show (layout privateWidth).expansion column.val = [(column.val, 1)] by
    exact expansion_direct column.isLt]
  have canonicalValue :
      (source (directSourceColumn column)).val < goldilocksP := by
    change (source (directSourceColumn column)).val < 18446744069414584321
    exact (source (directSourceColumn column)).isLt
  simp only [lcEval, List.foldl, Nat.zero_add, Nat.one_mul]
  rw [encodedNat_direct source column, Nat.mod_eq_of_lt canonicalValue]

theorem decoded_public {privateWidth : Nat}
    (source : SourceAssignment privateWidth) (column : Fin publicWidth) :
    decodedAssignment (layout privateWidth) (encodedNat source) column.val =
      (source (publicSourceColumn column)).val := by
  simpa [publicSourceColumn] using decoded_direct source
    (Fin.castLE publicWidth_le_directWidth column)

theorem decoded_source_column {privateWidth : Nat}
    (source : SourceAssignment privateWidth)
    (column : Fin (sourceWidth privateWidth)) :
    decodedAssignment (layout privateWidth) (encodedNat source) column.val =
      (source column).val := by
  by_cases directColumn : column.val < directWidth
  · let selected : Fin directWidth := ⟨column.val, directColumn⟩
    have sourceEq : directSourceColumn selected = column := by
      apply Fin.ext
      rfl
    simpa [selected, sourceEq] using decoded_direct source selected
  · have privateBase : directWidth <= column.val := Nat.le_of_not_gt directColumn
    let selected : Fin privateWidth :=
      ⟨column.val - directWidth, by
        have bounded := column.isLt
        simp only [sourceWidth] at bounded
        omega⟩
    have sourceEq : privateSourceColumn selected = column := by
      apply Fin.ext
      simp [privateSourceColumn, selected]
      omega
    have selectedEq : directWidth + selected.val = column.val := by
      simp [selected]
      omega
    simpa [selectedEq, sourceEq] using decoded_private source selected

end Nightstream.Implementation.NebulaV2.ProductionFreshLinearSubstitution
