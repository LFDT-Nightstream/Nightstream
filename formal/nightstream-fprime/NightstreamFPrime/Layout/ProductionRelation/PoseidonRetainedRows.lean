import NightstreamFPrime.Layout.ProductionRelation.PoseidonSboxPlan

/-! The 86 nonlinear rows of a retained Poseidon trace. Final outputs are
derived from that trace; independent caller output pins remain in SboxPlan. -/

namespace NightstreamFPrime.Layout.ProductionRelation.PoseidonRetainedRows

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra

def rows {logicalWidth : Nat} (interface : PoseidonSboxPlan.Interface logicalWidth) :
    List (PoseidonSboxPlan.Row logicalWidth) :=
  (PoseidonSboxPlan.trace interface).rows.map PoseidonSboxPlan.Row.sbox

@[simp] theorem rows_length {logicalWidth : Nat}
    (interface : PoseidonSboxPlan.Interface logicalWidth) :
    (rows interface).length = 86 := by
  simp [rows]

theorem rowsZero_iff {logicalWidth : Nat}
    (interface : PoseidonSboxPlan.Interface logicalWidth)
    (assignment : Assignment F logicalWidth)
    (output : interface.output = (PoseidonSboxPlan.trace interface).state) :
    PoseidonSboxPlan.RowsZero interface assignment ↔
      ∀ row ∈ rows interface, row.residual assignment = 0 := by
  constructor
  · intro complete row member
    exact complete row (List.mem_append_left _ member)
  · intro retained row member
    simp only [PoseidonSboxPlan.rows, List.mem_append] at member
    rcases member with sbox | pin
    · exact retained row sbox
    · rcases List.mem_map.mp pin with ⟨forms, member, rfl⟩
      unfold PoseidonSboxPlan.outputRows at member
      rcases List.mem_ofFn.mp member with ⟨lane, rfl⟩
      change (PinRow.Forms.mk _ _).residual assignment = 0
      rw [PinRow.Forms.residual_eq]
      simp [PoseidonSboxPlan.outputDifference, output, SparseForm.add_eval,
        SparseForm.scale_eval]

end NightstreamFPrime.Layout.ProductionRelation.PoseidonRetainedRows
