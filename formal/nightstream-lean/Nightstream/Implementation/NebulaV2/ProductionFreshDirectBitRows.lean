import Nightstream.Implementation.NebulaV2.ProductionFreshLinearSubstitution
import Nightstream.Implementation.R1CS.Canonical.GoldilocksField

/-!
Contract: exact bit rows for the complete V2 direct fresh prefix.

The direct prefix has 540 public bits followed by all 13,824 memory-lane
bits. This module emits one ordinary R1CS bit row for every one of those
14,364 source columns. Row satisfaction derives the low-norm compiler's
`DirectBinary` fact. That fact is not a prover premise.

This file does not own public-word contents, lane record semantics, padding,
private-field ternary encoding, commitments, NIFS, or Rust refinement.

Assurance tier: implementation model.

Emits constraints: exactly 14,364 bit rows.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.NebulaV2.ProductionFreshDirectBitRows

open Nightstream.Implementation.NebulaV2.ProductionFreshLinearSubstitution
open Nightstream.Implementation.NebulaV2.ProductionFreshLowNormEncoding
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.GoldilocksField
open Nightstream.SuperNeo.Concrete

/-- One exact bit row for every direct source column. -/
def rows : List Row :=
  (List.range directWidth).map bitRow

@[simp] theorem rows_length_exact : rows.length = 14364 := by
  simp [rows, directWidth]

private theorem sourceNat_canonical
    {privateWidth : Nat} (source : SourceAssignment privateWidth) :
    forall column, sourceNat source column < goldilocksP := by
  intro column
  by_cases within : column < sourceWidth privateWidth
  · simp only [sourceNat, dif_pos within]
    simpa only [goldilocksP,
      Nightstream.SuperNeo.Concrete.goldilocksModulus] using
      (source ⟨column, within⟩).isLt
  · simp [sourceNat, within, goldilocksP]

private theorem direct_value
    {privateWidth : Nat} (source : SourceAssignment privateWidth)
    (column : Fin directWidth) :
    sourceNat source column.val =
      (source (directSourceColumn column)).val := by
  simpa [directSourceColumn] using
    sourceNat_sourceColumn source (directSourceColumn column)

/-- Satisfying the direct bit rows derives the exact low-norm bit premise. -/
theorem sound
    {privateWidth : Nat} {source : SourceAssignment privateWidth}
    (sourceOne : sourceNat source 0 = 1)
    (holds : Satisfies rows (sourceNat source)) :
    DirectBinary source := by
  intro column
  have member : bitRow column.val ∈ rows := by
    exact List.mem_map.mpr
      ⟨column.val, List.mem_range.mpr column.isLt, rfl⟩
  have bounded : sourceNat source column.val ≤ 1 :=
    bitRow_le_one goldilocks_euclidPrime
      (sourceNat_canonical source column.val) sourceOne
      (holds _ member)
  have valueBound : (source (directSourceColumn column)).val ≤ 1 := by
    simpa only [direct_value source column] using bounded
  have alternatives :
      (source (directSourceColumn column)).val = 0 ∨
        (source (directSourceColumn column)).val = 1 := by
    omega
  rcases alternatives with zero | one
  · exact Or.inl (Fin.ext zero)
  · exact Or.inr (Fin.ext one)

/-- Every binary direct prefix satisfies the exact generated row block. -/
theorem complete
    {privateWidth : Nat} {source : SourceAssignment privateWidth}
    (sourceOne : sourceNat source 0 = 1)
    (binary : DirectBinary source) :
    Satisfies rows (sourceNat source) := by
  intro row member
  rcases List.mem_map.mp member with ⟨column, columnMember, rfl⟩
  have columnBound : column < directWidth := List.mem_range.mp columnMember
  let selected : Fin directWidth := ⟨column, columnBound⟩
  have value := binary selected
  rcases value with zero | one
  · have valueNat : sourceNat source column = 0 := by
      rw [direct_value source selected]
      exact congrArg Fin.val zero
    simp [RowHolds, bitRow, lcEval, sourceOne, valueNat, goldilocksP]
  · have valueNat : sourceNat source column = 1 := by
      rw [direct_value source selected]
      exact congrArg Fin.val one
    simp [RowHolds, bitRow, lcEval, sourceOne, valueNat, goldilocksP]

end Nightstream.Implementation.NebulaV2.ProductionFreshDirectBitRows
