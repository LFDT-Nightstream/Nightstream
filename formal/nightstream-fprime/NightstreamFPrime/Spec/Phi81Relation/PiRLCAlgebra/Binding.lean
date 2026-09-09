import Mathlib.Data.ZMod.ValMinAbs
import NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.Commitment
import NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.Norm.Centered

/-! Deterministic reduction from bounded Ajtai opening collisions to short
integer kernel vectors of the same key. Hardness is a separate premise. -/

namespace NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.Binding

open EvaluationHomomorphism

def difference {shape : Shape} (left right : Assignment shape) : Assignment shape :=
  fun column => left column - right column

/-- A Module-SIS solution written in the same block/lane order as the key.
The integer coordinates are reduced modulo `q` for the exact Ajtai action. -/
structure ShortKernelVector {shape : Shape} {rows : Nat}
    (key : Commitment.Key shape rows) (bound : Nat) where
  vector : Fin shape.carrierWidth → Int
  nonzero : vector ≠ fun _ => 0
  bounded : ∀ column, (vector column).natAbs < bound
  kernel : Commitment.commit key (fun column => (vector column : ZMod goldilocksModulus)) =
    Commitment.commitmentZero

theorem difference_nonzero {shape : Shape} (left right : Assignment shape)
    (different : left ≠ right) : difference left right ≠ BaseLinear.assignmentZero := by
  intro zero
  apply different
  funext column
  have atColumn : left column - right column = (0 : F) := congrFun zero column
  exact sub_eq_zero.mp (show (left column : ZMod goldilocksModulus) - right column = 0 from atColumn)

theorem difference_bounded {shape : Shape} (left right : Assignment shape)
    {leftBound rightBound : Nat}
    (leftNorm : assignmentNormBounded leftBound left)
    (rightNorm : assignmentNormBounded rightBound right) :
    assignmentNormBounded (leftBound + rightBound) (difference left right) := by
  intro column
  exact (Norm.Centered.centeredMagnitude_sub_le _ _).trans_lt
    (Nat.add_lt_add (leftNorm column) (rightNorm column))

theorem equal_commitments_difference_kernel {shape : Shape} {rows : Nat}
    (key : Commitment.Key shape rows) (left right : Assignment shape)
    (same : Commitment.commit key left = Commitment.commit key right) :
    Commitment.commit key (difference left right) = Commitment.commitmentZero := by
  have reconstruct : BaseLinear.assignmentAdd (difference left right) right = left := by
    funext column
    exact sub_add_cancel (left column : ZMod goldilocksModulus) (right column)
  have equation := Commitment.commit_add key (difference left right) right
  rw [reconstruct, same] at equation
  funext row lane
  have atLane := congrFun (congrFun equation row) lane
  change Commitment.commit key right row lane =
    Commitment.commit key (difference left right) row lane +
      Commitment.commit key right row lane at atLane
  change Commitment.commit key (difference left right) row lane = (0 : ZMod goldilocksModulus)
  exact add_right_cancel (atLane.symm.trans (zero_add _).symm)

/-- Centered lifting preserves the exact modular kernel equation and strict
bound. It converts a residue witness into the paper's integer witness. -/
def liftKernel {shape : Shape} {rows bound : Nat}
    (key : Commitment.Key shape rows) (assignment : Assignment shape)
    (nonzero : assignment ≠ BaseLinear.assignmentZero)
    (bounded : assignmentNormBounded bound assignment)
    (kernel : Commitment.commit key assignment = Commitment.commitmentZero) :
    ShortKernelVector key bound where
  vector column := ZMod.valMinAbs (n := goldilocksModulus) (assignment column)
  nonzero := by
    intro zero
    apply nonzero
    funext column
    exact (ZMod.valMinAbs_eq_zero (n := goldilocksModulus) (assignment column)).mp
      (congrFun zero column)
  bounded column := by
    simpa only [ZMod.valMinAbs_natAbs_eq_min] using bounded column
  kernel := by
    have recovered : (fun column =>
        (ZMod.valMinAbs (n := goldilocksModulus) (assignment column) :
          ZMod goldilocksModulus)) = assignment :=
      funext fun column => ZMod.coe_valMinAbs (n := goldilocksModulus) (assignment column)
    rw [recovered]
    exact kernel

/-- A collision at strict norm `B` gives a nonzero integer kernel vector of
strict norm `2B`, without changing or resampling the matrix. -/
def bindingCollision_to_shortKernel {shape : Shape} {rows bound : Nat}
    (key : Commitment.Key shape rows) (commitment : Commitment.Value rows)
    (collision : Opening.BindingCollision (relationSemantics (Commitment.commit key))
      bound commitment) : ShortKernelVector key (2 * bound) :=
  liftKernel key (difference collision.leftOpening collision.rightOpening)
    (difference_nonzero _ _ collision.different)
    (by simpa only [two_mul] using difference_bounded _ _ collision.leftNorm collision.rightNorm)
    (equal_commitments_difference_kernel key _ _
      (collision.leftCommits.trans collision.rightCommits.symm))

end NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.Binding
