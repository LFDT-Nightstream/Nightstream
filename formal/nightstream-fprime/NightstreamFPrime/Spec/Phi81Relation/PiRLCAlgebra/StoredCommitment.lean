import NightstreamFPrime.Spec.AjtaiSetupV1.Work
import NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.Commitment
import NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism.StoredRingArithmetic

/-!
Counted dense Ajtai rows over stored complete-carrier assignments. Each
iteration expands one 54-lane key block, reads the matching witness block,
and materializes the ring product and updated accumulator. The direct index
loop retains no full key, source list, or accumulated accessor closures.
Function-valued keys and assignments occur only in the value refinement.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.StoredCommitment

open _root_.NightstreamFPrime.Spec
open _root_.NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open _root_.NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism
open _root_.NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism.StoredRingArithmetic (StoredRing)
open _root_.NightstreamFPrime.Spec.Folding.Nifs.StoredAssignmentArithmetic (build build_value build_work_le)
open _root_.NightstreamFPrime.Spec.Folding.PiRLC.PaperForkExtractionWork (Result)

private def zero (_ : Unit) : Result StoredRing :=
  let result := build (fun _ : Fin ringDegree => (⟨0, 1⟩ : Result F))
  ⟨result.value, result.work + 2⟩

private def zeroWork : Nat := ringDegree + 1 + ringDegree * (1 + 2) + 1 + 2

private theorem zero_value : (zero ()).value.get = ringFZero := by
  change _root_.NightstreamFPrime.Spec.Folding.Nifs.StoredAssignmentArithmetic.view (build _).value = _
  rw [build_value]
  rfl

private theorem zero_work_le : (zero ()).work ≤ zeroWork :=
  Nat.add_le_add_right (build_work_le _ 1 (fun _ => Nat.le_refl 1)) 2

private def keyBlock {verifierRows messageColumns : Nat}
    (setup : AjtaiSetupV1.Setup verifierRows messageColumns)
    (row : Fin verifierRows) (block : Fin messageColumns) : Result StoredRing :=
  let result := build (AjtaiSetupV1.Work.coefficient setup row block)
  ⟨result.value, result.work + 2⟩

private def keyBlockWork : Nat :=
  ringDegree + 1 + ringDegree * (AjtaiSetupV1.Work.coefficientWork + 2) + 1 + 2

private theorem keyBlock_value {verifierRows messageColumns : Nat}
    (setup : AjtaiSetupV1.Setup verifierRows messageColumns)
    (row : Fin verifierRows) (block : Fin messageColumns) :
    (keyBlock setup row block).value.get = setup.verifierKey row block := by
  change _root_.NightstreamFPrime.Spec.Folding.Nifs.StoredAssignmentArithmetic.view (build _).value = _
  rw [build_value]
  funext lane
  exact AjtaiSetupV1.Work.coefficient_value setup row block lane

private theorem keyBlock_work_le {verifierRows messageColumns : Nat}
    (setup : AjtaiSetupV1.Setup verifierRows messageColumns)
    (row : Fin verifierRows) (block : Fin messageColumns)
    (rowRange : row.val < 2 ^ 32) (blockRange : block.val < 2 ^ 64) :
    (keyBlock setup row block).work ≤ keyBlockWork :=
  Nat.add_le_add_right (build_work_le _ _
    (fun lane => AjtaiSetupV1.Work.coefficient_work_le setup row block lane rowRange blockRange)) 2

/-- The existing shape completes its logical width to whole Phi81 blocks.
Two index reads, multiply/add, the Fin constructor, vector projection,
array lookup, and return are charged at each copied coordinate. -/
private def witnessBlock {shape : Phi81Relation.Shape}
    (stored : Vector F shape.carrierWidth)
    (block : Fin (Phi81ColumnLayout.blockCount shape.carrierWidth)) : Result StoredRing :=
  let result := build (fun lane =>
    let column := Phi81CarrierLayout.carrierColumn (logicalWidth := shape.logicalWidth) block lane
    (⟨stored.get column, 8⟩ : Result F))
  ⟨result.value, result.work + 2⟩

private def witnessBlockWork : Nat := ringDegree + 1 + ringDegree * (8 + 2) + 1 + 2

private theorem witnessBlock_value {shape : Phi81Relation.Shape}
    (stored : Vector F shape.carrierWidth)
    (block : Fin (Phi81ColumnLayout.blockCount shape.carrierWidth)) :
    (witnessBlock (shape := shape) stored block).value.get =
      CarrierAction.assignmentBlock (logicalWidth := shape.logicalWidth) stored.get block := by
  change _root_.NightstreamFPrime.Spec.Folding.Nifs.StoredAssignmentArithmetic.view (build _).value = _
  rw [build_value]
  rfl

private theorem witnessBlock_work_le {shape : Phi81Relation.Shape}
    (stored : Vector F shape.carrierWidth)
    (block : Fin (Phi81ColumnLayout.blockCount shape.carrierWidth)) :
    (witnessBlock (shape := shape) stored block).work ≤ witnessBlockWork :=
  Nat.add_le_add_right (build_work_le _ 8 (fun _ => Nat.le_refl 8)) 2

private def step {shape : Phi81Relation.Shape} {verifierRows : Nat}
    (setup : AjtaiSetupV1.Setup verifierRows (Phi81ColumnLayout.blockCount shape.carrierWidth))
    (stored : Vector F shape.carrierWidth) (row : Fin verifierRows)
    (accumulated : StoredRing) (block : Fin (Phi81ColumnLayout.blockCount shape.carrierWidth)) :
    Result StoredRing :=
  let key := keyBlock setup row block
  let witness := witnessBlock (shape := shape) stored block
  let product := StoredRingArithmetic.multiply key.value witness.value
  let sum := StoredRingArithmetic.add accumulated product.value
  ⟨sum.value, key.work + witness.work + product.work + sum.work + 1⟩

private def stepWork : Nat :=
  keyBlockWork + witnessBlockWork + StoredRingArithmetic.multiplyWork + StoredRingArithmetic.addWork + 1

private theorem step_value {shape : Phi81Relation.Shape} {verifierRows : Nat}
    (setup : AjtaiSetupV1.Setup verifierRows (Phi81ColumnLayout.blockCount shape.carrierWidth))
    (stored : Vector F shape.carrierWidth) (row : Fin verifierRows)
    (accumulated : StoredRing) (block : Fin (Phi81ColumnLayout.blockCount shape.carrierWidth)) :
    (step (shape := shape) setup stored row accumulated block).value.get = ringFAdd accumulated.get
      (ringFMul (setup.verifierKey row block)
        (CarrierAction.assignmentBlock (logicalWidth := shape.logicalWidth) stored.get block)) := by
  simp only [step, StoredRingArithmetic.add_value, StoredRingArithmetic.multiply_value,
    keyBlock_value, witnessBlock_value]

private theorem step_work_le {shape : Phi81Relation.Shape} {verifierRows : Nat}
    (setup : AjtaiSetupV1.Setup verifierRows (Phi81ColumnLayout.blockCount shape.carrierWidth))
    (stored : Vector F shape.carrierWidth) (row : Fin verifierRows)
    (accumulated : StoredRing) (block : Fin (Phi81ColumnLayout.blockCount shape.carrierWidth))
    (rowRange : row.val < 2 ^ 32) (blockRange : block.val < 2 ^ 64) :
    (step (shape := shape) setup stored row accumulated block).work ≤ stepWork := by
  have key := keyBlock_work_le setup row block rowRange blockRange
  have witness := witnessBlock_work_le (shape := shape) stored block
  have product := StoredRingArithmetic.multiply_work_le
    (keyBlock setup row block).value (witnessBlock (shape := shape) stored block).value
  have sum := StoredRingArithmetic.add_work_le accumulated
    (StoredRingArithmetic.multiply (keyBlock setup row block).value
      (witnessBlock (shape := shape) stored block).value).value
  dsimp only [step]
  unfold stepWork
  omega

private def advance {count : Nat} (next : StoredRing → Fin count → Result StoredRing)
    (accumulated : Result StoredRing) (index : Fin count) : Result StoredRing :=
  let value := next accumulated.value index
  ⟨value.value, accumulated.work + value.work + 3⟩

private theorem ringAdd_assoc (left middle right : RingF) :
    ringFAdd (ringFAdd left middle) right = ringFAdd left (ringFAdd middle right) := by
  funext lane
  exact ConcreteCarrier.baseLaws.add_assoc _ _ _

private theorem ringAdd_zero (value : RingF) : ringFAdd value ringFZero = value := by
  funext lane
  exact ConcreteCarrier.baseLaws.add_zero _

private theorem zero_ringAdd (value : RingF) : ringFAdd ringFZero value = value := by
  funext lane
  exact ConcreteCarrier.baseLaws.zero_add _

/-- Reindexing occurs only in this proof. The executed fold always uses the
original direct Fin index and a stored accumulator. -/
private theorem fold_value : ∀ {count : Nat}
    (next : StoredRing → Fin count → Result StoredRing) (terms : Fin count → RingF)
    (correct : ∀ accumulated index, (next accumulated index).value.get = ringFAdd accumulated.get (terms index))
    (initial : Result StoredRing),
    (Fin.foldl count (advance next) initial).value.get =
      ringFAdd initial.value.get (Commitment.ringFSum terms)
  | 0, _, _, _, initial => by
      simp only [Fin.foldl_zero, Commitment.ringFSum]
      exact (ringAdd_zero initial.value.get).symm
  | count + 1, next, terms, correct, initial => by
      rw [Fin.foldl_succ]
      have tail := fold_value (fun value index => next value index.succ)
        (fun index => terms index.succ) (fun value index => correct value index.succ) (advance next initial 0)
      calc
        _ = ringFAdd (advance next initial 0).value.get
              (Commitment.ringFSum (fun index => terms index.succ)) := tail
        _ = ringFAdd (ringFAdd initial.value.get (terms 0))
              (Commitment.ringFSum (fun index => terms index.succ)) := by
                change ringFAdd (next initial.value 0).value.get _ = _
                rw [correct]
        _ = ringFAdd initial.value.get (Commitment.ringFSum terms) := by
              exact ringAdd_assoc _ _ _

private theorem fold_work_le (bound : Nat) : ∀ {count : Nat}
    (next : StoredRing → Fin count → Result StoredRing) (initial : Result StoredRing),
    (∀ accumulated index, (next accumulated index).work ≤ bound) →
    (Fin.foldl count (advance next) initial).work ≤ initial.work + count * (bound + 3)
  | 0, _, _, _ => by simp
  | count + 1, next, initial, bounded => by
      rw [Fin.foldl_succ]
      have tail := fold_work_le bound (fun value index => next value index.succ)
        (advance next initial 0) (fun value index => bounded value index.succ)
      have head := bounded initial.value 0
      exact Nat.le_trans tail (by simp only [advance, Nat.add_mul, Nat.one_mul]; omega)

/-- Compute every complete block of one dense row. Only the current key
block, witness block, product and accumulator are materialized. -/
def row {shape : Phi81Relation.Shape} {verifierRows : Nat}
    (setup : AjtaiSetupV1.Setup verifierRows (Phi81ColumnLayout.blockCount shape.carrierWidth))
    (stored : Vector F shape.carrierWidth) (rowIndex : Fin verifierRows) : Result StoredRing :=
  let result := Fin.foldl (Phi81ColumnLayout.blockCount shape.carrierWidth)
    (advance (step (shape := shape) setup stored rowIndex)) (zero ())
  ⟨result.value, result.work + 3⟩

theorem row_value {shape : Phi81Relation.Shape} {verifierRows : Nat}
    (setup : AjtaiSetupV1.Setup verifierRows (Phi81ColumnLayout.blockCount shape.carrierWidth))
    (stored : Vector F shape.carrierWidth) (rowIndex : Fin verifierRows) :
    (row (shape := shape) setup stored rowIndex).value.get =
      Commitment.ajtaiRow (shape := shape) setup.verifierKey stored.get rowIndex := by
  have value := fold_value (step (shape := shape) setup stored rowIndex)
    (fun block => ringFMul (setup.verifierKey rowIndex block)
      (CarrierAction.assignmentBlock (logicalWidth := shape.logicalWidth) stored.get block))
    (fun accumulated block => step_value (shape := shape) setup stored rowIndex accumulated block) (zero ())
  simpa only [row, zero_value, zero_ringAdd, Commitment.ajtaiRow, Commitment.blockSum] using value

/-- Dense block count times the executed key, witness, multiplication and
addition clocks, plus stored initialization and direct-loop controls. -/
def rowWork (shape : Phi81Relation.Shape) : Nat :=
  zeroWork + Phi81ColumnLayout.blockCount shape.carrierWidth * (stepWork + 3) + 3

theorem row_work_le {shape : Phi81Relation.Shape} {verifierRows : Nat}
    (setup : AjtaiSetupV1.Setup verifierRows (Phi81ColumnLayout.blockCount shape.carrierWidth))
    (stored : Vector F shape.carrierWidth) (rowIndex : Fin verifierRows)
    (rowRange : rowIndex.val < 2 ^ 32)
    (blockRange : Phi81ColumnLayout.blockCount shape.carrierWidth ≤ 2 ^ 64) :
    (row (shape := shape) setup stored rowIndex).work ≤ rowWork shape := by
  have folded := fold_work_le stepWork (step (shape := shape) setup stored rowIndex) (zero ()) (by
    intro accumulated block
    exact step_work_le (shape := shape) setup stored rowIndex accumulated block rowRange
      (Nat.lt_of_lt_of_le block.isLt blockRange))
  have initialized := zero_work_le
  dsimp only [row]
  unfold rowWork
  omega

end NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.StoredCommitment
