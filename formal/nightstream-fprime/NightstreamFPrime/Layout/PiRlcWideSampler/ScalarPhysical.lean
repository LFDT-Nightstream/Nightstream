import NightstreamFPrime.Layout.PiRlcWideSampler.RangePhysical
import NightstreamFPrime.Layout.PiRLC.v1_1.Leaves.TranscriptAbsorption
import NightstreamFPrime.Layout.Poseidon2.PermutationOwned
import NightstreamFPrime.Lifecycle.PiRLC.Wide.Scalar

/-! Exact physical cost of one scalar. Parent proofs use the certified child
costs; they do not expand the permutation or checked-range row lists. -/

namespace NightstreamFPrime.Layout.PiRlcWideSampler.ScalarPhysical

open NightstreamFPrime.Spec NightstreamFPrime.Circuit
open NightstreamFPrime.Gadgets.Sampling
open NightstreamFPrime.Lifecycle.PiRLC.Wide
open NightstreamFPrime.Layout.Poseidon2

private theorem entered_fresh (interface : Scalar.Interface) (coordinate offset : Nat) :
    Duplex.StateFresh (Scalar.enteredState interface coordinate offset) := by
  unfold Scalar.enteredState
    Lifecycle.PiRLC.v1_1.TranscriptAbsorption.output Lifecycle.PiRLC.v1_1.TranscriptAbsorption.ownedInterface
    Gadgets.Poseidon2.Duplex.Formal.Owned.output Gadgets.Poseidon2.Duplex.Formal.Owned.program
  apply Duplex.compile_output_fresh_of_head_absorb
  intro empty
  have lengths := congrArg List.length empty
  simp [Gadgets.Poseidon2.Hash.inputChunks, Lifecycle.PiRLC.v1_1.TranscriptAbsorption.constantWords,
    Lifecycle.PiRLC.v1_1.TranscriptAbsorption.frameWords, Spec.Poseidon2.rate] at lengths

theorem entered_affine (interface : Scalar.Interface) (coordinate offset : Nat) :
    StateAffine (Scalar.enteredState interface coordinate offset) := (entered_fresh interface coordinate offset).affine

theorem output_affine (interface : Scalar.Interface) (coordinate offset : Nat) :
    StateAffine (Scalar.outputState interface coordinate offset) := by
  intro lane
  exact R1CS.isAffine_var _

private theorem child_constraints (name : String) (child : FormalCircuit) (offset : Nat) :
    (Sequence.childOp name child offset).flatConstraints = flatConstraints (Circuit.ops child.main offset) := rfl

private theorem entry_fresh (interface : Scalar.Interface) (coordinate offset : Nat)
    (inputs : ∀ current, StateAffine (interface.initialState current)) :
    R1CS.totalFreshCount ((Scalar.entryOp interface coordinate offset).flatConstraints) = 0 := by
  rw [Scalar.entryOp, child_constraints, Scalar.entry, FormalCircuit.withConstantFootprint_main]
  exact PiRLC.v1_1.Leaves.TranscriptAbsorption.freshColumnCount_eq interface coordinate (fun current => ⟨inputs current⟩) offset

private theorem range_fresh (interface : Scalar.Interface) (coordinate offset : Nat) :
    R1CS.totalFreshCount ((Scalar.rangeOp interface coordinate offset).flatConstraints) = 1548 := by
  rw [Scalar.rangeOp, child_constraints]
  change R1CS.totalFreshCount (flatConstraints (WideReduction.Program.operations
    (Scalar.rangeInterface interface coordinate offset) (Scalar.rangeOffset offset))) = _
  rw [WideReduction.Program.constraints_eq]
  exact (RangePhysical.counts _ _ _ (fun lane => entered_affine interface coordinate offset (Scalar.rateLane lane))).1

private theorem advance_fresh (interface : Scalar.Interface) (coordinate offset : Nat) :
    R1CS.totalFreshCount ((Scalar.advanceOp interface coordinate offset).flatConstraints) = 0 := by
  rw [Scalar.advanceOp, child_constraints]
  exact PermutationOwned.totalFreshCount_eq (Scalar.advanceInterface interface coordinate offset)
    (Scalar.advanceOffset offset) ⟨entered_affine interface coordinate offset⟩

theorem counts (interface : Scalar.Interface) (coordinate offset : Nat)
    (inputs : ∀ current, StateAffine (interface.initialState current)) :
    R1CS.totalFreshCount (flatConstraints (Scalar.operations interface coordinate offset)) = 1548 ∧
      R1CS.totalRowCount (flatConstraints (Scalar.operations interface coordinate offset)) = 3413 := by
  have fresh : R1CS.totalFreshCount (flatConstraints (Scalar.operations interface coordinate offset)) = 1548 := by
    simp only [Scalar.operations, flatConstraints, List.flatMap_cons, List.flatMap_nil, List.append_nil,
      R1CS.totalFreshCount_append, entry_fresh interface coordinate offset inputs,
      range_fresh, advance_fresh]
  refine ⟨fresh, ?_⟩
  rw [R1CS.totalRowCount_eq_fresh_add_length, fresh, Scalar.rowCount_eq, Scalar.counts.2]

end NightstreamFPrime.Layout.PiRlcWideSampler.ScalarPhysical
