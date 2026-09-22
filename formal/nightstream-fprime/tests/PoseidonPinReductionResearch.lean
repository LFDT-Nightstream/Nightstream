import NightstreamFPrime.Export.Stage1.DirectPoseidonFootprint
import NightstreamFPrime.Layout.ProductionRelation.PoseidonRetainedFamily
import tests.AxiomAudit

/-!
Research only: the eight final pins of a retained Poseidon invocation add no
condition. The reduced rows use the same assignment, so witness construction
and extraction are the identity. No production plan or artifact is changed.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Tests.PoseidonPinReductionResearch

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra
open NightstreamFPrime.Layout.ProductionRelation

theorem retained_rows_iff_sbox_rows
    {sourceWidth invocationCount logicalWidth : Nat}
    (schedule : PoseidonRetainedFamily.Schedule sourceWidth invocationCount)
    (start : Nat)
    (fits : start + schedule.block.coordinateCount ≤ logicalWidth)
    (oneColumn : Fin logicalWidth)
    (input : Fin invocationCount → PoseidonSboxPlan.State logicalWidth)
    (assignment : Assignment F logicalWidth)
    (one : assignment oneColumn = 1)
    (invocation : Fin invocationCount) :
    let interface := PoseidonRetainedFamily.invocationInterface
      schedule start fits oneColumn input invocation
    PoseidonSboxPlan.RowsZero interface assignment ↔
      PoseidonSboxPlan.SboxRowsZero assignment
        (PoseidonSboxPlan.trace interface).rows := by
  dsimp only
  constructor
  · exact PoseidonSboxPlan.rowsZero_implies_sboxRowsZero _ assignment
  · intro sboxes row member
    have pins := PoseidonSboxPlan.pinRowsZero_of_equations
      (PoseidonRetainedFamily.invocationInterface
        schedule start fits oneColumn input invocation)
      assignment one (PoseidonRetainedFamily.outputEquations
        schedule start fits oneColumn input assignment invocation)
    simp only [PoseidonSboxPlan.rows, List.mem_append, List.mem_map] at member
    rcases member with ⟨forms, formsMember, rfl⟩ |
      ⟨forms, formsMember, rfl⟩
    · exact sboxes forms formsMember
    · exact pins forms formsMember

theorem reduced_rows_per_invocation {logicalWidth : Nat}
    (interface : PoseidonSboxPlan.Interface logicalWidth) :
    (PoseidonSboxPlan.trace interface).rows.length = 86 :=
  PoseidonSboxPlan.trace_rows_length interface

theorem selected_pin_rows_saved :
    NightstreamFPrime.Export.Stage1.DirectPoseidonFootprint.totalPermutationCount *
      (94 - 86) = 259656 := by
  rw [NightstreamFPrime.Export.Stage1.DirectPoseidonFootprint.totalPermutationCount_eq]

theorem selected_logical_rows_after : 4703127 - 259656 = 4443471 := by decide

#audit_axioms retained_rows_iff_sbox_rows
#audit_axioms reduced_rows_per_invocation
#audit_axioms selected_pin_rows_saved
#audit_axioms selected_logical_rows_after

end NightstreamFPrime.Tests.PoseidonPinReductionResearch
