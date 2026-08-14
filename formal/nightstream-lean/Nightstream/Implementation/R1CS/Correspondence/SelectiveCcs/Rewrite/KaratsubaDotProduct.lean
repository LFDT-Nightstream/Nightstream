import Mathlib.Tactic.Ring
import Mathlib.Data.ZMod.Basic
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConcreteCarrier.Algebra

/-!
Contract: model-level equivalence between six direct product-sum equations and
the thirty-three-row Karatsuba dot-product schedule used by the selective
compiler fixture.

Assurance tier: model-level.

Owns: the retained boundary values, the canonical thirty temporary values,
the three direct sum equations, and soundness and completeness of temporary
reconstruction.

Does not own: generated Rust rows, low-norm encoding, selector authority,
production-family coverage, or permission to remove a production row.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.KaratsubaDotProduct

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConcreteCarrier

local instance : Std.Associative (fun (left right : F) => left + right) :=
  ⟨baseLaws.add_assoc⟩

local instance : Std.Commutative (fun (left right : F) => left + right) :=
  ⟨baseLaws.add_comm⟩

local instance : Std.Commutative (fun (left right : F) => left * right) :=
  ⟨baseLaws.mul_comm⟩

/-- The six extension-field products share four retained input vectors and
three retained output sums. -/
structure Boundary where
  lowLeft : Fin 6 → F
  highLeft : Fin 6 → F
  lowRight : Fin 6 → F
  highRight : Fin 6 → F
  highHighSum : F
  lowOutput : F
  highOutput : F

/-- The source schedule stores five temporary values for each product. -/
structure Witness where
  lowLow : Fin 6 → F
  highHigh : Fin 6 → F
  cross : Fin 6 → F
  lowTerm : Fin 6 → F
  highTerm : Fin 6 → F

/-- The literal six-term sum used by the fixture. -/
def sumSix (term : Fin 6 → F) : F :=
  term 0 + term 1 + term 2 + term 3 + term 4 + term 5

def lowLowValue (boundary : Boundary) (index : Fin 6) : F :=
  boundary.lowLeft index * boundary.lowRight index

def highHighValue (boundary : Boundary) (index : Fin 6) : F :=
  boundary.highLeft index * boundary.highRight index

def crossValue (boundary : Boundary) (index : Fin 6) : F :=
  (boundary.lowLeft index + boundary.highLeft index) *
    (boundary.lowRight index + boundary.highRight index)

/-- Semantic normal form of the three compact product sums. The emitted rows
use equivalent difference forms so each output remains one linear port. -/
def ReducedHolds (boundary : Boundary) : Prop :=
  boundary.highHighSum = sumSix (highHighValue boundary) ∧
    boundary.lowOutput =
      sumSix (lowLowValue boundary) +
        7 * sumSix (highHighValue boundary) ∧
    boundary.highOutput =
      sumSix (crossValue boundary) -
        sumSix (lowLowValue boundary) -
        sumSix (highHighValue boundary)

/-- The three difference equations emitted by the compact compiler. -/
def EmittedHolds (boundary : Boundary) : Prop :=
  boundary.highHighSum = sumSix (highHighValue boundary) ∧
    boundary.lowOutput - 7 * boundary.highHighSum =
      sumSix (lowLowValue boundary) ∧
    boundary.highOutput + boundary.lowOutput -
        6 * boundary.highHighSum =
      sumSix (crossValue boundary)

private theorem mappedSix :
    (ZMod.finEquiv goldilocksModulus) (6 : F) =
      (6 : ZMod goldilocksModulus) := by
  decide

private theorem mappedSeven :
    (ZMod.finEquiv goldilocksModulus) (7 : F) =
      (7 : ZMod goldilocksModulus) := by
  decide

private theorem restoreLowDifference (output highHigh : F) :
    output = (output - 7 * highHigh) + 7 * highHigh := by
  apply (ZMod.finEquiv goldilocksModulus).injective
  simp only [map_add, map_sub, map_mul, mappedSeven]
  ring

private theorem removeLowDifference (lowLow highHigh : F) :
    (lowLow + 7 * highHigh) - 7 * highHigh = lowLow := by
  apply (ZMod.finEquiv goldilocksModulus).injective
  simp only [map_add, map_sub, map_mul, mappedSeven]
  ring

private theorem restoreHighDifference (output lowLow highHigh : F) :
    output =
      (output + (lowLow + 7 * highHigh) - 6 * highHigh) -
        lowLow - highHigh := by
  apply (ZMod.finEquiv goldilocksModulus).injective
  simp only [map_add, map_sub, map_mul, mappedSix, mappedSeven]
  ring

private theorem removeHighDifference (cross lowLow highHigh : F) :
    (cross - lowLow - highHigh) + (lowLow + 7 * highHigh) -
        6 * highHigh = cross := by
  apply (ZMod.finEquiv goldilocksModulus).injective
  simp only [map_add, map_sub, map_mul, mappedSix, mappedSeven]
  ring

/-- The emitted difference equations are the semantic normal form. -/
theorem emitted_iff_reduced (boundary : Boundary) :
    EmittedHolds boundary ↔ ReducedHolds boundary := by
  constructor
  · rintro ⟨highHighSum, lowDifference, highDifference⟩
    refine ⟨highHighSum, ?_, ?_⟩
    · rw [highHighSum] at lowDifference
      rw [← lowDifference]
      exact restoreLowDifference _ _
    · rw [highHighSum] at lowDifference highDifference
      have lowOutput :
          boundary.lowOutput =
            sumSix (lowLowValue boundary) +
              7 * sumSix (highHighValue boundary) := by
        rw [← lowDifference]
        exact restoreLowDifference _ _
      rw [lowOutput] at highDifference
      rw [← highDifference]
      exact restoreHighDifference _ _ _
  · rintro ⟨highHighSum, lowOutput, highOutput⟩
    refine ⟨highHighSum, ?_, ?_⟩
    · rw [highHighSum, lowOutput]
      exact removeLowDifference _ _
    · rw [highHighSum, lowOutput, highOutput]
      exact removeHighDifference _ _ _

/-- The original source schedule checks three products and two output terms
for each of the six lanes, then checks the three retained sums. -/
def SourceHolds (boundary : Boundary) (witness : Witness) : Prop :=
  (∀ index,
      witness.lowLow index = lowLowValue boundary index ∧
      witness.highHigh index = highHighValue boundary index ∧
      witness.cross index = crossValue boundary index ∧
      witness.lowTerm index =
        witness.lowLow index + 7 * witness.highHigh index ∧
      witness.highTerm index =
        witness.cross index - witness.lowLow index - witness.highHigh index) ∧
    boundary.highHighSum = sumSix witness.highHigh ∧
    boundary.lowOutput = sumSix witness.lowTerm ∧
    boundary.highOutput = sumSix witness.highTerm

/-- Canonical reconstruction of every source-only temporary value. -/
def materialize (boundary : Boundary) : Witness where
  lowLow := lowLowValue boundary
  highHigh := highHighValue boundary
  cross := crossValue boundary
  lowTerm := fun index =>
    lowLowValue boundary index + 7 * highHighValue boundary index
  highTerm := fun index =>
    crossValue boundary index - lowLowValue boundary index -
      highHighValue boundary index

private theorem sumSix_lowTerm (boundary : Boundary) :
    sumSix (materialize boundary).lowTerm =
      sumSix (lowLowValue boundary) +
        7 * sumSix (highHighValue boundary) := by
  simp only [sumSix, materialize]
  repeat rw [Lean.Grind.Fin.left_distrib]
  ac_rfl

private theorem sumSix_highTerm (boundary : Boundary) :
    sumSix (materialize boundary).highTerm =
      sumSix (crossValue boundary) -
        sumSix (lowLowValue boundary) -
        sumSix (highHighValue boundary) := by
  simp only [sumSix, materialize]
  simp only [Fin.sub_eq_add_neg, Lean.Grind.AddCommGroup.neg_add]
  ac_rfl

/-- Every compact witness reconstructs a witness for all source equations. -/
theorem sourceHolds_materialize_of_reduced
    (boundary : Boundary) (reduced : ReducedHolds boundary) :
    SourceHolds boundary (materialize boundary) := by
  rcases reduced with ⟨highHighSum, lowOutput, highOutput⟩
  refine ⟨?_, highHighSum, ?_, ?_⟩
  · intro index
    exact ⟨rfl, rfl, rfl, rfl, rfl⟩
  · rw [sumSix_lowTerm]
    exact lowOutput
  · rw [sumSix_highTerm]
    exact highOutput

/-- The original source schedule implies the three compact equations. -/
theorem reduced_of_sourceHolds
    (boundary : Boundary) (witness : Witness)
    (source : SourceHolds boundary witness) :
    ReducedHolds boundary := by
  rcases source with ⟨lanes, highHighSum, lowOutput, highOutput⟩
  have witnessEq : witness = materialize boundary := by
    cases witness with
    | mk lowLow highHigh cross lowTerm highTerm =>
        simp only at lanes
        have lowLowEq : lowLow = lowLowValue boundary :=
          funext fun index => (lanes index).1
        have highHighEq : highHigh = highHighValue boundary :=
          funext fun index => (lanes index).2.1
        have crossEq : cross = crossValue boundary :=
          funext fun index => (lanes index).2.2.1
        subst lowLow
        subst highHigh
        subst cross
        have lowTermEq :
            lowTerm = (materialize boundary).lowTerm :=
          funext fun index => (lanes index).2.2.2.1
        have highTermEq :
            highTerm = (materialize boundary).highTerm :=
          funext fun index => (lanes index).2.2.2.2
        subst lowTerm
        subst highTerm
        rfl
  subst witness
  refine ⟨highHighSum, ?_, ?_⟩
  · rw [lowOutput, sumSix_lowTerm]
  · rw [highOutput, sumSix_highTerm]

/-- Exact existential equivalence. The compact relation removes only
source-only temporary values. -/
theorem reduced_iff_exists_source (boundary : Boundary) :
    ReducedHolds boundary ↔ ∃ witness, SourceHolds boundary witness := by
  constructor
  · intro reduced
    exact ⟨materialize boundary,
      sourceHolds_materialize_of_reduced boundary reduced⟩
  · rintro ⟨witness, source⟩
    exact reduced_of_sourceHolds boundary witness source

end Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.KaratsubaDotProduct
