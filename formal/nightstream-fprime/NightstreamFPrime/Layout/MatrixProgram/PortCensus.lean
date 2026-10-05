import NightstreamFPrime.Layout.MatrixProgram.Program

/-!
Owns the port census of the compact matrix program. No row family fills the
centered-unit port or the five canonical-class ports (meaningful ports 6 and
8–12), so their matrices are zero for every program. Slot 13 is zero through
`ProductionRelation.meaningfulPort?`.

The census is one lemma per row family. Its proof cost does not depend on the
number of rows. This module changes no row.
-/

namespace NightstreamFPrime.Layout.MatrixProgram

open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation

/-- Meaningful ports that no row family fills. -/
def DeadPort (port : Fin Spec.ProductionRelation.meaningfulPortCount) : Prop :=
  port.val = 6 ∨ 8 ≤ port.val

/-- Row forms whose dead ports are all empty. -/
def DeadPortsEmpty {logicalWidth : Nat} (forms : RowForms logicalWidth) : Prop :=
  ∀ port, DeadPort port → forms port = .empty

private theorem ordinary_deadPortsEmpty {logicalWidth : Nat}
    (forms : OrdinaryRow.Forms logicalWidth) :
    DeadPortsEmpty forms.meaningfulForm := by
  intro port dead
  fin_cases port <;> simp_all [DeadPort, OrdinaryRow.Forms.meaningfulForm]

private theorem pin_deadPortsEmpty {logicalWidth : Nat}
    (forms : PinRow.Forms logicalWidth) :
    DeadPortsEmpty forms.meaningfulForm := by
  intro port dead
  fin_cases port <;> simp_all [DeadPort, PinRow.Forms.meaningfulForm]

private theorem sbox_deadPortsEmpty {logicalWidth : Nat}
    (forms : SboxRow.Forms logicalWidth) :
    DeadPortsEmpty forms.meaningfulForm := by
  intro port dead
  fin_cases port <;> simp_all [DeadPort, SboxRow.Forms.meaningfulForm]

/-- A Φ81 product row fills lane 0 only; lanes 2–4 own the dead ports. -/
private theorem phi81Row_deadPortsEmpty {logicalWidth : Nat}
    (interface : Phi81ProductPlan.Interface logicalWidth) (row : Fin 108) :
    DeadPortsEmpty (Phi81ProductPlan.rowAt interface row).meaningfulForm := by
  intro port dead
  fin_cases port <;> simp_all [DeadPort, Phi81ProductPlan.rowAt,
    Phi81ProductPlan.productRow, ProductSumPlan.Row.meaningfulForm,
    ProductSumRow.Forms.meaningfulForm]

private theorem phi81_deadPortsEmpty (block : Phi81Product.Block)
    (logicalWidth ordinal : Nat) {forms : RowForms logicalWidth}
    (decoded : block.row? logicalWidth ordinal = some forms) :
    DeadPortsEmpty forms := by
  unfold Phi81Product.Block.row? at decoded
  split at decoded
  · simp only [Option.bind_eq_bind, Option.bind_eq_some_iff,
      Option.pure_def, Option.some.injEq] at decoded
    rcases decoded with ⟨_, _, interface, _, row, selected, rfl⟩
    have member := List.mem_of_getElem? selected
    rw [Phi81ProductPlan.rows, List.mem_ofFn'] at member
    rcases member with ⟨index, rfl⟩
    exact phi81Row_deadPortsEmpty interface index
  · cases decoded

/-- Retained Poseidon2 rows are S-box rows only. -/
private theorem poseidonRow_deadPortsEmpty {logicalWidth : Nat}
    (interface : PoseidonSboxPlan.Interface logicalWidth)
    (row : PoseidonSboxPlan.Row logicalWidth)
    (member : row ∈ PoseidonRetainedRows.rows interface) :
    DeadPortsEmpty row.meaningfulForm := by
  rw [PoseidonRetainedRows.rows, List.mem_map] at member
  rcases member with ⟨forms, _, rfl⟩
  exact sbox_deadPortsEmpty forms

private theorem poseidon_deadPortsEmpty (block : Poseidon.Block)
    (logicalWidth ordinal : Nat) {forms : RowForms logicalWidth}
    (decoded : block.row? logicalWidth ordinal = some forms) :
    DeadPortsEmpty forms := by
  unfold Poseidon.Block.row? Poseidon.Block.rowWithInput? at decoded
  split at decoded
  · split at decoded
    · split at decoded
      · split at decoded
        · split at decoded
          · simp only [Option.bind_eq_bind, Option.bind_eq_some_iff,
              Option.pure_def, Option.some.injEq] at decoded
            rcases decoded with ⟨_, _, rfl⟩
            exact poseidonRow_deadPortsEmpty _ _ (List.get_mem _ _)
          · cases decoded
        · cases decoded
      · cases decoded
    · cases decoded
  · cases decoded

/-- Every row that any block decodes leaves the dead ports empty. -/
theorem Block.row?_deadPortsEmpty (block : Block) (logicalWidth : Nat)
    (sourceRow : Nat → Option R1CS.Row) (ordinal : Nat)
    {forms : RowForms logicalWidth}
    (decoded : block.row? logicalWidth sourceRow ordinal = some forms) :
    DeadPortsEmpty forms := by
  cases block with
  | ordinary ordinaryBlock =>
      simp only [Block.row?, Option.bind_eq_bind, Option.bind_eq_some_iff,
        Option.pure_def, Option.some.injEq] at decoded
      rcases decoded with ⟨rowForms, _, rfl⟩
      exact ordinary_deadPortsEmpty rowForms
  | multiplicationGrid multiplicationBlock =>
      simp only [Block.row?, Option.bind_eq_bind, Option.bind_eq_some_iff,
        Option.pure_def, Option.some.injEq] at decoded
      rcases decoded with ⟨rowForms, _, rfl⟩
      exact ordinary_deadPortsEmpty rowForms
  | phi81Product productBlock =>
      exact phi81_deadPortsEmpty productBlock logicalWidth ordinal decoded
  | pin pinBlock =>
      simp only [Block.row?, Option.bind_eq_bind, Option.bind_eq_some_iff,
        Option.pure_def, Option.some.injEq] at decoded
      rcases decoded with ⟨rowForms, _, rfl⟩
      exact pin_deadPortsEmpty rowForms
  | poseidon poseidonBlock =>
      exact poseidon_deadPortsEmpty poseidonBlock logicalWidth ordinal decoded

private theorem select_deadPortsEmpty (logicalWidth : Nat)
    (sourceRow : Nat → Option R1CS.Row) (blocks : List Block) (ordinal : Nat)
    {forms : RowForms logicalWidth}
    (decoded : Program.row?.select logicalWidth sourceRow blocks ordinal = some forms) :
    DeadPortsEmpty forms := by
  induction blocks generalizing ordinal with
  | nil => simp [Program.row?.select] at decoded
  | cons block rest ih =>
      rw [Program.row?.select] at decoded
      split at decoded
      · exact Block.row?_deadPortsEmpty block logicalWidth sourceRow ordinal decoded
      · exact ih (ordinal - block.rowCount) decoded

/-- Every row that any program decodes leaves the dead ports empty. -/
theorem Program.row?_deadPortsEmpty (program : Program) (logicalWidth : Nat)
    (sourceRow : Nat → Option R1CS.Row) (ordinal : Nat)
    {forms : RowForms logicalWidth}
    (decoded : program.row? logicalWidth sourceRow ordinal = some forms) :
    DeadPortsEmpty forms :=
  select_deadPortsEmpty logicalWidth sourceRow program.blocks ordinal decoded

end NightstreamFPrime.Layout.MatrixProgram
