import NightstreamFPrime.Layout.PiRlcWideSampler.ScalarPhysical
import NightstreamFPrime.Layout.PiRlcWideSampler.DigitProjection

/-! Exact physical sampler ledger, including the temporary digit words.
Costs are composed structurally from one scalar and one affine projection. -/

namespace NightstreamFPrime.Layout.PiRlcWideSampler.BatchPhysical

open NightstreamFPrime.Spec NightstreamFPrime.Circuit
open NightstreamFPrime.Lifecycle.PiRLC.Wide
open NightstreamFPrime.Layout.Poseidon2

private theorem child_affine (interface : Batch.Interface) (offset : Nat)
    (inputs : StateAffine (interface.initialState offset)) (source current : Nat) :
    StateAffine ((Batch.childInterface interface offset source).initialState current) := by
  cases source with
  | zero => exact inputs
  | succ source =>
      exact ScalarPhysical.output_affine (Batch.childInterface interface offset source)
        source (Batch.sourceOffset offset source)

private theorem child_fresh (interface : Batch.Interface) (offset : Nat)
    (inputs : StateAffine (interface.initialState offset)) (source : Nat) :
    R1CS.totalFreshCount ((Batch.childOp interface offset source).flatConstraints) = 1548 := by
  change R1CS.totalFreshCount (flatConstraints (Scalar.operations (Batch.childInterface interface offset source)
    source (Batch.sourceOffset offset source))) = _
  exact (ScalarPhysical.counts _ _ _ (child_affine interface offset inputs source)).1

theorem batch_counts (interface : Batch.Interface) (offset : Nat)
    (inputs : StateAffine (interface.initialState offset)) :
    R1CS.totalFreshCount (flatConstraints (Batch.operations interface offset)) = 26316 ∧
      R1CS.totalRowCount (flatConstraints (Batch.operations interface offset)) = 58021 := by
  have all : ∀ sources : List Nat,
      R1CS.totalFreshCount (flatConstraints (sources.map (Batch.childOp interface offset))) = sources.length * 1548 := by
    intro sources
    induction sources with
    | nil => rfl
    | cons source rest ih =>
        simp only [List.map_cons, flatConstraints, List.flatMap_cons, R1CS.totalFreshCount_append,
          child_fresh interface offset inputs, List.length_cons]
        change 1548 + R1CS.totalFreshCount (flatConstraints _) = _
        rw [ih]
        omega
  have fresh : R1CS.totalFreshCount (flatConstraints (Batch.operations interface offset)) = 26316 := by
    change R1CS.totalFreshCount (flatConstraints ((List.range Batch.sourceCount).map _)) = _
    rw [all, List.length_range, Batch.sourceCount_eq]
  refine ⟨fresh, ?_⟩
  rw [R1CS.totalRowCount_eq_fresh_add_length, fresh, Batch.rowCount_eq, Batch.counts.2]

private theorem child_constraints (name : String) (child : FormalCircuit) (offset : Nat) :
    (Sequence.childOp name child offset).flatConstraints = flatConstraints (Circuit.ops child.main offset) := rfl

theorem counts (interface : ProjectedBatch.Interface) (offset : Nat)
    (inputs : StateAffine (interface.initialState offset)) :
    R1CS.totalFreshCount (flatConstraints (ProjectedBatch.operations interface offset)) = 26316 ∧
      R1CS.totalRowCount (flatConstraints (ProjectedBatch.operations interface offset)) = 58939 := by
  have fresh : R1CS.totalFreshCount (flatConstraints (ProjectedBatch.operations interface offset)) = 26316 := by
    simp only [ProjectedBatch.operations, flatConstraints, List.flatMap_cons, List.flatMap_nil, List.append_nil,
      R1CS.totalFreshCount_append]
    rw [ProjectedBatch.sampleOp, ProjectedBatch.wordsOp, child_constraints, child_constraints,
      Batch.circuit_ops, DigitWords.circuit_ops, (batch_counts interface offset inputs).1,
      (DigitProjection.r1cs_counts offset (ProjectedBatch.wordsOffset offset)).1, Nat.add_zero]
  refine ⟨fresh, ?_⟩
  rw [R1CS.totalRowCount_eq_fresh_add_length, fresh, ProjectedBatch.rowCount_eq, ProjectedBatch.counts.2]

theorem physicalPrivateCount (interface : ProjectedBatch.Interface) (offset : Nat)
    (inputs : StateAffine (interface.initialState offset)) :
    localLength (ProjectedBatch.operations interface offset) +
      R1CS.totalFreshCount (flatConstraints (ProjectedBatch.operations interface offset)) = 81719 := by
  rw [ProjectedBatch.localLength_eq, ProjectedBatch.counts.1, (counts interface offset inputs).1]

end NightstreamFPrime.Layout.PiRlcWideSampler.BatchPhysical
