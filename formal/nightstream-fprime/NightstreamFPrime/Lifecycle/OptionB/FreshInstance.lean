import NightstreamFPrime.Lifecycle.Relation
import NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.Norm.Centered

/-! The fresh instance that the option-B prover builds for a step (Rust `Relation::instance`,
`crates/nightstream/src/lifecycle/option_b/relation.rs`). Rust stores the fresh witness as two
masks, one of the coordinates that hold 1 and one of the coordinates that hold -1 (`Masks`), and
commits to the assignment that the masks give (`built`). Such an assignment is below the fresh norm
bound `b = 2` (`Masks.assignment_norm`), and each assignment below that bound is the assignment of
its masks (`Masks.assignment_ofAssignment`), so the masks lose nothing of a fresh witness.

The built instance holds exactly when the rows of the relation hold (`built_holds_iff`): its
commitment, its public input and its norm bound hold by construction. So the prover's check of the
rows alone (Rust `Relation::rows_hold`) is the check `CCS.Holds` of the terminal relation's fresh
instance (`Terminal.relations`), and it reads no commitment. -/

set_option autoImplicit false

namespace NightstreamFPrime.Lifecycle.OptionB.FreshInstance

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.ProductionKey
open NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.Norm.Centered

/-- The coordinates that hold 1 and the coordinates that hold -1. Rust keeps coordinate `c` as bit
`c % 54` of block `c / 54` in each mask. -/
structure Masks (width : Nat) where
  positive : Fin width → Bool
  negative : Fin width → Bool
  disjoint : ∀ column, ¬ (positive column = true ∧ negative column = true)

namespace Masks

variable {width : Nat}

/-- The assignment of the masks: 1, -1 or 0 at each coordinate. -/
def assignment (masks : Masks width) (column : Fin width) : F :=
  if masks.positive column then 1 else if masks.negative column then -1 else 0

private theorem magnitude_one : centeredMagnitude (1 : F) = 1 := by decide

private theorem one_ne_negOne : (1 : F) ≠ -1 := by decide

/-- A value below the fresh norm bound is 0, 1 or -1. -/
theorem unit_of_norm {value : F} (norm : centeredMagnitude value < 2) :
    value = 0 ∨ value = 1 ∨ value = -1 := by
  have negOne : ((-1 : F) : Nat) = goldilocksModulus - 1 := by decide
  have one : ((1 : F) : Nat) = 1 := by decide
  have zero : ((0 : F) : Nat) = 0 := rfl
  have bound := value.isLt
  have cases : value.val = 0 ∨ value.val = 1 ∨ value.val = goldilocksModulus - 1 := by
    unfold centeredMagnitude at norm
    rw [Nat.min_def] at norm
    split at norm <;> omega
  rcases cases with isZero | isOne | isNegOne
  · exact Or.inl (Fin.ext (isZero.trans zero.symm))
  · exact Or.inr (Or.inl (Fin.ext (isOne.trans one.symm)))
  · exact Or.inr (Or.inr (Fin.ext (isNegOne.trans negOne.symm)))

/-- The masks of an assignment. -/
def ofAssignment (assignment : Fin width → F) : Masks width where
  positive := fun column => decide (assignment column = 1)
  negative := fun column => decide (assignment column = -1)
  disjoint := by
    intro column both
    have one := of_decide_eq_true both.1
    have negative := of_decide_eq_true both.2
    exact one_ne_negOne (one.symm.trans negative)

/-- Each coordinate of the masks' assignment is below the fresh norm bound `b = 2`. -/
theorem assignment_norm (masks : Masks width) (column : Fin width) :
    centeredMagnitude (masks.assignment column) < 2 := by
  unfold assignment
  split
  · rw [magnitude_one]; decide
  · split
    · rw [centeredMagnitude_neg, magnitude_one]; decide
    · rw [centeredMagnitude_zero]; decide

/-- An assignment below the fresh norm bound is the assignment of its masks. -/
theorem assignment_ofAssignment (values : Fin width → F)
    (norm : ∀ column, centeredMagnitude (values column) < 2) :
    (ofAssignment values).assignment = values := by
  funext column
  have bounded := norm column
  show (if decide (values column = 1) = true then (1 : F)
    else if decide (values column = -1) = true then -1 else 0) = values column
  generalize values column = value at bounded ⊢
  rcases unit_of_norm bounded with rfl | rfl | rfl <;> decide

end Masks

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth}

/-- Rust `Relation::instance`: the commitment to the assignment and its public input. -/
noncomputable def built
    (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (assignment : PaperAlgebra.Assignment (logicalWidth := logicalWidth)
      (publicFits := publicFits)) :
    Fresh (logicalWidth := logicalWidth) (publicFits := publicFits) where
  commitments := fun _ => (semantics ajtai).commit assignment
  publicInputs := fun _ => (semantics ajtai).projectPublicInput assignment

/-- The built instance of a mask witness holds exactly when the rows of the relation hold on the
masks' assignment. -/
theorem built_holds_iff (relation : LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (masks : Masks (Phi81Relation.Shape.carrierWidth (FullShape logicalWidth publicFits))) :
    CCS.Holds (semantics ajtai) productionGlobalParams
        (freshStatement relation (built ajtai masks.assignment)) masks.assignment ↔
      (semantics ajtai).ccsSatisfied
        (freshStatement relation (built ajtai masks.assignment)).constraintSystem
        masks.assignment :=
  ⟨fun holds => holds.2, fun rows => ⟨⟨rfl, rfl, masks.assignment_norm⟩, rows⟩⟩

end NightstreamFPrime.Lifecycle.OptionB.FreshInstance
