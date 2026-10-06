import NightstreamFPrime.Layout.PiRLC.v1_1.Sampler
import NightstreamFPrime.Layout.R1CS.Segments

/-! Structural projection of the four opaque scalar children. The checked
range owns all 144 lowering columns; the other three children add none. -/

namespace NightstreamFPrime.Layout.PiRLC.v1_1.Sampler

open NightstreamFPrime.Circuit NightstreamFPrime.Layout

def childConstraintLists (interface : Logical.Interface) (coordinate offset : Nat) : List (List Expr) :=
  [(Lifecycle.PiRLC.v1_1.Sampler.entryOp interface coordinate offset).flatConstraints,
   (Lifecycle.PiRLC.v1_1.Sampler.rangeOp interface coordinate offset).flatConstraints,
   (Lifecycle.PiRLC.v1_1.Sampler.advanceOp interface coordinate offset).flatConstraints,
   (Lifecycle.PiRLC.v1_1.Sampler.wordsOp offset).flatConstraints]

theorem logicalConstraints_eq_ordered (interface : Logical.Interface) (coordinate offset : Nat) :
    logicalConstraints interface coordinate offset = (childConstraintLists interface coordinate offset).flatten := by
  simp only [logicalConstraints, Lifecycle.PiRLC.v1_1.Sampler.opsAt,
    flatConstraints, childConstraintLists, List.flatMap_cons, List.flatMap_nil,
    List.flatten_cons, List.flatten_nil]

structure ChildRows (interface : Logical.Interface) (coordinate offset : Nat) (env : Env) (start : Nat) : Prop where
  entry : R1CS.RowsHold env (R1CS.lowerConstraints
    (Lifecycle.PiRLC.v1_1.Sampler.entryOp interface coordinate offset).flatConstraints start).rows
  range : R1CS.RowsHold env (R1CS.lowerConstraints
    (Lifecycle.PiRLC.v1_1.Sampler.rangeOp interface coordinate offset).flatConstraints start).rows
  advance : R1CS.RowsHold env (R1CS.lowerConstraints
    (Lifecycle.PiRLC.v1_1.Sampler.advanceOp interface coordinate offset).flatConstraints (start + 144)).rows
  words : R1CS.RowsHold env (R1CS.lowerConstraints
    (Lifecycle.PiRLC.v1_1.Sampler.wordsOp offset).flatConstraints (start + 144)).rows

theorem rowsHold_implies_childRows (interface : Logical.Interface) (coordinate offset : Nat)
    (env : Env) (start : Nat) (inputs : ∀ current, InputsAffine interface current)
    (rows : R1CS.RowsHold env (R1CS.lowerConstraints (logicalConstraints interface coordinate offset) start).rows) :
    ChildRows interface coordinate offset env start := by
  rw [logicalConstraints_eq_ordered, R1CS.rowsHold_flatten_iff] at rows
  simp only [childConstraintLists, R1CS.SegmentsHold,
    entry_fresh interface coordinate offset (fun current => (inputs current).initialState),
    range_fresh, advance_fresh, words_fresh, Nat.add_zero] at rows
  exact ⟨rows.1, rows.2.1, rows.2.2.1, rows.2.2.2.1⟩

end NightstreamFPrime.Layout.PiRLC.v1_1.Sampler
