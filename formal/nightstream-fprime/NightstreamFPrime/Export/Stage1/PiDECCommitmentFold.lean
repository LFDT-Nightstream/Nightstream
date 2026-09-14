import NightstreamFPrime.Export.Stage1.PiDECCommitmentBlock
import NightstreamFPrime.Export.NativePoseidon2RoundCore
import NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.Commitment

/-!
Stored addition and finite accumulation of the existing PiDEC block products.
The value proofs identify the finite sum and complete Ajtai row. No work
counter, key representation, IO, or range size is introduced.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiDECCommitmentFold

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism
open NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism.StoredRingArithmetic
  (StoredRing)
open NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.Commitment
  (ringFSum ajtaiRow blockSum)
open NightstreamFPrime.Export.NativePoseidon2
  (add64 add64_canonical add64_denote)

/-- The stored additive identity. -/
def zero : StoredRing := Vector.replicate ringDegree 0

theorem zero_value : zero.get = ringFZero := by
  funext lane
  change (Vector.replicate ringDegree (0 : F))[lane.val] = 0
  rw [Vector.getElem_replicate]

@[inline] private def addCoefficient (left right : F) : F :=
  let a := UInt64.ofNatLT left.val (Nat.lt_trans left.isLt (by decide))
  let b := UInt64.ofNatLT right.val (Nat.lt_trans right.isLt (by decide))
  ⟨(add64 a b).toNat, add64_canonical a b
    (by simpa only [a, UInt64.toNat_ofNatLT] using left.isLt)
    (by simpa only [b, UInt64.toNat_ofNatLT] using right.isLt)⟩

private theorem canonicalWord_denote (word : UInt64)
    (canonical : word.toNat < goldilocksModulus) :
    (⟨word.toNat, canonical⟩ : F) = word.denote := by
  apply Fin.ext
  change word.toNat = word.toNat % goldilocksModulus
  exact (Nat.mod_eq_of_lt canonical).symm

private theorem addCoefficient_value (left right : F) :
    addCoefficient left right = left + right := by
  let a := UInt64.ofNatLT left.val (Nat.lt_trans left.isLt (by decide))
  let b := UInt64.ofNatLT right.val (Nat.lt_trans right.isLt (by decide))
  have ha : a.toNat < goldilocksModulus := by
    simpa only [a, UInt64.toNat_ofNatLT] using left.isLt
  have hb : b.toNat < goldilocksModulus := by
    simpa only [b, UInt64.toNat_ofNatLT] using right.isLt
  have aValue : a.denote = left := by
    apply Fin.ext
    change a.toNat % goldilocksModulus = left.val
    simp only [a, UInt64.toNat_ofNatLT, Nat.mod_eq_of_lt left.isLt]
  have bValue : b.denote = right := by
    apply Fin.ext
    change b.toNat % goldilocksModulus = right.val
    simp only [b, UInt64.toNat_ofNatLT, Nat.mod_eq_of_lt right.isLt]
  calc
    addCoefficient left right = (add64 a b).denote :=
      canonicalWord_denote (add64 a b) (add64_canonical a b ha hb)
    _ = left + right := by rw [add64_denote a b ha hb, aValue, bValue]

/-- Materialize every coefficient with native-word field addition. -/
def add (left right : StoredRing) : StoredRing :=
  Vector.ofFn fun lane => addCoefficient (left.get lane) (right.get lane)

theorem add_value (left right : StoredRing) :
    (add left right).get = ringFAdd left.get right.get := by
  funext lane
  change (Vector.ofFn (fun index : Fin ringDegree =>
    addCoefficient (left.get index) (right.get index)))[lane.val] = _
  rw [Vector.getElem_ofFn, addCoefficient_value]
  rfl

private theorem ringAdd_assoc (left middle right : RingF) :
    ringFAdd (ringFAdd left middle) right =
      ringFAdd left (ringFAdd middle right) := by
  funext lane
  exact ConcreteCarrier.baseLaws.add_assoc _ _ _

private theorem ringAdd_zero (value : RingF) : ringFAdd value ringFZero = value := by
  funext lane
  exact ConcreteCarrier.baseLaws.add_zero _

private theorem zero_ringAdd (value : RingF) : ringFAdd ringFZero value = value := by
  funext lane
  exact ConcreteCarrier.baseLaws.zero_add _

/-- Visit the exact finite index domain with a stored accumulator. -/
def fold {count : Nat} (terms : Fin count → StoredRing)
    (initial : StoredRing) : StoredRing :=
  Fin.foldl count (fun accumulated index => add accumulated (terms index)) initial

/-- The executed left fold equals the existing head-first ring sum.
Reindexing is confined to the proof. -/
theorem fold_value : ∀ {count : Nat} (terms : Fin count → StoredRing)
    (initial : StoredRing),
    (fold terms initial).get =
      ringFAdd initial.get (ringFSum fun index => (terms index).get)
  | 0, _, initial => by
      simp only [fold, Fin.foldl_zero, ringFSum]
      exact (ringAdd_zero initial.get).symm
  | count + 1, terms, initial => by
      rw [fold, Fin.foldl_succ]
      have tail := fold_value (fun index : Fin count => terms index.succ)
        (add initial (terms 0))
      calc
        _ = ringFAdd (add initial (terms 0)).get
              (ringFSum fun index : Fin count => (terms index.succ).get) := tail
        _ = ringFAdd (ringFAdd initial.get (terms 0).get)
              (ringFSum fun index : Fin count => (terms index.succ).get) := by
                rw [add_value]
        _ = ringFAdd initial.get
              (ringFSum fun index => (terms index).get) :=
                ringAdd_assoc _ _ _

/-- A complete finite sum starts at the stored zero. -/
def sum {count : Nat} (terms : Fin count → StoredRing) : StoredRing :=
  fold terms zero

theorem sum_value {count : Nat} (terms : Fin count → StoredRing) :
    (sum terms).get = ringFSum fun index => (terms index).get := by
  rw [sum, fold_value, zero_value, zero_ringAdd]

/-- Summing the existing computed contributions preserves each key/block/child
coordinate. The block family may be consumed directly by a streaming caller. -/
theorem sum_contributions_value {verifierRows messageColumns : Nat}
    (setup : AjtaiSetupV1.Setup verifierRows messageColumns)
    (row : Fin verifierRows)
    (children : Fin messageColumns → Vector StoredRing productionGlobalParams.k)
    (child : Fin productionGlobalParams.k) :
    (sum fun block =>
      (PiDECCommitmentBlock.contributions setup row block (children block)).get child).get =
      ringFSum fun block =>
        ringFMul (setup.verifierKey row block) ((children block).get child).get := by
  rw [sum_value]
  apply congrArg ringFSum
  funext block
  exact PiDECCommitmentBlock.contributions_value setup row block (children block) child

/-- Read the same complete carrier block from each stored child assignment. -/
def childBlocks {shape : Phi81Relation.Shape}
    (assignments : Vector (Vector F shape.carrierWidth) productionGlobalParams.k)
    (block : Fin (Phi81ColumnLayout.blockCount shape.carrierWidth)) :
    Vector StoredRing productionGlobalParams.k :=
  Vector.ofFn fun child =>
    Vector.ofFn (CarrierAction.assignmentBlock
      (logicalWidth := shape.logicalWidth) (assignments.get child).get block)

theorem childBlocks_value {shape : Phi81Relation.Shape}
    (assignments : Vector (Vector F shape.carrierWidth) productionGlobalParams.k)
    (block : Fin (Phi81ColumnLayout.blockCount shape.carrierWidth))
    (child : Fin productionGlobalParams.k) :
    ((childBlocks (shape := shape) assignments block).get child).get =
      CarrierAction.assignmentBlock
        (logicalWidth := shape.logicalWidth) (assignments.get child).get block := by
  funext lane
  change ((Vector.ofFn (fun selected : Fin productionGlobalParams.k =>
    Vector.ofFn (CarrierAction.assignmentBlock
      (logicalWidth := shape.logicalWidth) (assignments.get selected).get block)))[child.val])[lane.val] = _
  rw [Vector.getElem_ofFn, Vector.getElem_ofFn]

/-- Folding every complete carrier block gives the exact existing commitment
row of the selected stored child assignment. No expected row is supplied. -/
theorem sum_contributions_eq_ajtaiRow
    {shape : Phi81Relation.Shape} {verifierRows : Nat}
    (setup : AjtaiSetupV1.Setup verifierRows
      (Phi81ColumnLayout.blockCount shape.carrierWidth))
    (assignments : Vector (Vector F shape.carrierWidth) productionGlobalParams.k)
    (row : Fin verifierRows) (child : Fin productionGlobalParams.k) :
    (sum fun block =>
      (PiDECCommitmentBlock.contributions setup row block
        (childBlocks (shape := shape) assignments block)).get child).get =
      ajtaiRow (shape := shape) setup.verifierKey
        (assignments.get child).get row := by
  rw [sum_contributions_value]
  unfold ajtaiRow blockSum
  apply congrArg ringFSum
  funext block
  rw [childBlocks_value]

/-- Continuing with a second finite part preserves both ordered partial sums. -/
theorem fold_parts_value {leftCount rightCount : Nat}
    (left : Fin leftCount → StoredRing) (right : Fin rightCount → StoredRing)
    (initial : StoredRing) :
    (fold right (fold left initial)).get =
      ringFAdd initial.get
        (ringFAdd (ringFSum fun index => (left index).get)
          (ringFSum fun index => (right index).get)) := by
  rw [fold_value, fold_value]
  exact ringAdd_assoc _ _ _

/-- Adding two separately computed partial sums gives the same value as
continuing the second fold from the first partial sum. -/
theorem combine_partialSums_value {leftCount rightCount : Nat}
    (left : Fin leftCount → StoredRing) (right : Fin rightCount → StoredRing) :
    (add (sum left) (sum right)).get = (fold right (sum left)).get := by
  rw [add_value, fold_value, sum_value right]

end NightstreamFPrime.Export.Stage1.PiDECCommitmentFold
