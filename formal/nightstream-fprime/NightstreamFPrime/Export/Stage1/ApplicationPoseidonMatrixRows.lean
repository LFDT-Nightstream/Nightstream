import NightstreamFPrime.Export.Stage1.ApplicationPoseidonMatrixSemantics

/-! Row equality between the application matrix program and its proved compact plan. -/

namespace NightstreamFPrime.Export.Stage1.ApplicationPoseidonMatrixProgram

open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Layout.MatrixProgram
open NightstreamFPrime.Lifecycle
open ApplicationPoseidonRetainedBlock ApplicationPoseidonRetainedGeometry

variable {application : Stage1.Application.Program} {certificate : Certificate application}
  {columns : Nat}

theorem inputProgram_state? (geometry : Geometry application certificate columns)
    (invocation : Fin 3) :
    (inputProgram application certificate).state? columns (oneColumn geometry).val invocation.val =
      some (Stage1.Poseidon2HashChainCompact.input (interface geometry) invocation) := by
  apply PoseidonInput.Program.state?_eq_some
  · exact inputProgram_form? geometry invocation 0
  · exact inputProgram_form? geometry invocation 1
  · exact inputProgram_form? geometry invocation 2
  · exact inputProgram_form? geometry invocation 3
  · exact inputProgram_form? geometry invocation 4
  · exact inputProgram_form? geometry invocation 5
  · exact inputProgram_form? geometry invocation 6
  · exact inputProgram_form? geometry invocation 7

theorem poseidonBlock_row? (geometry : Geometry application certificate columns)
    (row : Fin 258) :
    (poseidonBlock geometry).row? columns row.val =
      let decoded : Fin 3 × Fin 86 := Fin.decodeProd row
      some (PoseidonSboxFamilyPlan.rowForms
        (Stage1.Poseidon2HashChainCompact.family (interface geometry)) decoded.1 decoded.2) := by
  exact Poseidon.Block.row?_ofSemantic (schedule application certificate) rfl
    (localStart application) (oneColumn geometry) (inputProgram application certificate)
    (localFits geometry) (Stage1.Poseidon2HashChainCompact.input (interface geometry))
    (inputProgram_state? geometry) row

theorem matrixProgram_row? (geometry : Geometry application certificate columns)
    (sourceRow : Nat → Option R1CS.Row)
    (row : Fin (plan geometry).rowCount) :
    (matrixProgram geometry).row? columns sourceRow row.val = some ((plan geometry).forms row) := by
  by_cases first : row.val < 258
  · have selected := MatrixProgram.Program.two_first_row?
      (.poseidon (poseidonBlock geometry)) (.pin (bindingBlock geometry))
      columns sourceRow row.val first
    change (matrixProgram geometry).row? columns sourceRow row.val = _ at selected
    rw [selected]
    change (poseidonBlock geometry).row? columns row.val = _
    rw [poseidonBlock_row? geometry ⟨row.val, first⟩]
    apply congrArg some
    unfold plan Stage1.Poseidon2HashChainCompact.plan Plan.append
    simp only [PoseidonSboxFamilyPlan.plan_rowCount, show (3 : Nat) * 86 = 258 from rfl,
      Plan.splitIndex, first, ↓reduceDIte]
    rfl
  · have bound : row.val - 258 < 4 := by have := row.isLt; change row.val < 262 at this; omega
    have selected := MatrixProgram.Program.two_second_row?
      (.poseidon (poseidonBlock geometry)) (.pin (bindingBlock geometry))
      columns sourceRow row.val (Nat.le_of_not_gt first) bound
    change (matrixProgram geometry).row? columns sourceRow row.val = _ at selected
    rw [selected]
    change (do
      let forms ← (bindingBlock geometry).row? columns (row.val - 258)
      pure forms.meaningfulForm) = _
    rw [bindingBlock_row? geometry ⟨row.val - 258, bound⟩]
    apply congrArg some
    unfold plan Stage1.Poseidon2HashChainCompact.plan Plan.append
    simp only [PoseidonSboxFamilyPlan.plan_rowCount, show (3 : Nat) * 86 = 258 from rfl,
      Plan.splitIndex, first, ↓reduceDIte]
    rfl

end NightstreamFPrime.Export.Stage1.ApplicationPoseidonMatrixProgram
