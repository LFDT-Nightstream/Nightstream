import Nightstream.Implementation.Nebula.NIFS.PiRLC.FirstAcceptedBatchRows
import Nightstream.Implementation.R1CS.Canonical.LinCombNormal

/-!
Contract: exact sparse rows for one V2 PiRLC ring-valued output.

One occurrence combines 17 source rings with the 17 transcript-derived
challenge rings. It computes each distinct schoolbook product once and uses
linear rows for the exact reduction by `X^54 + X^27 + 1` and source sum.

The challenge input is the selected symbol in `0..4`. Each multiplication
uses the centered field value `symbol - 2` directly as a linear combination.
Thus this program does not allocate an unconstrained challenge copy.

This file owns the row schedule, auxiliary-column ownership, exact row count,
and independent equations derived from row satisfaction. It does not own the
typed Phi81 algebra bridge, placement in the complete NIFS relation, or the
transcript and selector proofs that fix the challenge symbols.
-/

set_option autoImplicit false
set_option maxRecDepth 30000

namespace Nightstream.Implementation.Nebula.ProductPiRlcRingCombinationRows

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal

def sourceCount : Nat := ProductPiRlcFirstAcceptedBatchRows.sourceCount
def laneCount : Nat := 54

theorem sourceCount_eq : sourceCount = 17 := by decide
theorem laneCount_eq : laneCount = 54 := rfl

abbrev Source := Fin sourceCount
abbrev Lane := Fin laneCount

/-- The caller owns challenge-symbol, source-ring, and output-ring columns.
This occurrence owns every column at and above `base`. -/
structure Layout where
  base : Nat
  challengeSymbol : Source -> Lane -> Nat
  input : Source -> Lane -> Nat
  output : Lane -> Nat

def productCount : Nat := sourceCount * laneCount * laneCount
def auxiliaryCount : Nat := productCount

theorem productCount_eq : productCount = 49572 := by decide

def productOffset (source : Source) (left right : Lane) : Nat :=
  (source.val * laneCount + left.val) * laneCount + right.val

def productColumn (layout : Layout)
    (source : Source) (left right : Lane) : Nat :=
  layout.base + productOffset source left right

theorem productOffset_lt (source : Source) (left right : Lane) :
    productOffset source left right < productCount := by
  have sourceLt := source.isLt
  have leftLt := left.isLt
  have rightLt := right.isLt
  change source.val < 17 at sourceLt
  change left.val < 54 at leftLt
  change right.val < 54 at rightLt
  change (source.val * 54 + left.val) * 54 + right.val < 17 * 54 * 54
  omega

/-- Symbol `2` is centered zero. This linear form is `symbol - 2` in the
Goldilocks field. -/
def centeredChallenge
    (layout : Layout) (source : Source) (lane : Lane) : LinComb :=
  [(layout.challengeSymbol source lane, 1), (0, goldilocksP - 2)]

def productRow (layout : Layout)
    (source : Source) (left right : Lane) : Row :=
  ⟨centeredChallenge layout source left,
    [(layout.input source right, 1)],
    [(productColumn layout source left right, 1)]⟩

def indices (count : Nat) : List (Fin count) := List.finRange count

theorem index_mem {count : Nat} (index : Fin count) :
    index ∈ indices count := by
  simp [indices]

@[simp] theorem indices_length (count : Nat) :
    (indices count).length = count := by
  simp [indices]

private theorem length_flatMap_uniform
    {Alpha Beta : Type} (items : List Alpha) (values : Alpha -> List Beta)
    (count : Nat) (uniform : forall item, (values item).length = count) :
    (items.flatMap values).length = items.length * count := by
  induction items with
  | nil => simp
  | cons head tail inductionHypothesis =>
      simp [uniform, inductionHypothesis, Nat.add_mul, Nat.add_comm]

def sourceProductRows (layout : Layout) (source : Source) : List Row :=
  (indices laneCount).flatMap fun left =>
    (indices laneCount).map fun right => productRow layout source left right

def productRows (layout : Layout) : List Row :=
  (indices sourceCount).flatMap (sourceProductRows layout)

theorem sourceProductRows_length (layout : Layout) (source : Source) :
    (sourceProductRows layout source).length = laneCount * laneCount := by
  unfold sourceProductRows
  rw [length_flatMap_uniform _ _ laneCount]
  · simp
  · intro left
    simp

theorem productRows_length (layout : Layout) :
    (productRows layout).length = productCount := by
  unfold productRows productCount
  rw [length_flatMap_uniform _ _ (laneCount * laneCount)
    (sourceProductRows_length layout)]
  simp [Nat.mul_assoc]

/-- Active schoolbook terms for one unreduced coefficient. The column is
shared across all output rows; only the public linear coefficient changes. -/
def rawTerms (layout : Layout) (source : Source)
    (degree coefficient : Nat) : LinComb :=
  (indices laneCount).filterMap fun left =>
    if active : left.val <= degree ∧ degree - left.val < laneCount then
      some (productColumn layout source left
        ⟨degree - left.val, active.2⟩, coefficient)
    else none

/-- Degree used by the negative Phi81 reduction term. -/
def foldedDegree (output : Lane) : Nat :=
  if output.val < 27 then output.val + 54 else output.val + 27

def twiceEnabled (output : Lane) : Bool := decide (output.val + 81 <= 106)

def sourceOutputTerms
    (layout : Layout) (source : Source) (output : Lane) : LinComb :=
  rawTerms layout source output.val 1 ++
    rawTerms layout source (foldedDegree output) (goldilocksP - 1) ++
    if twiceEnabled output then
      rawTerms layout source (output.val + 81) 1
    else []

def outputTerms (layout : Layout) (output : Lane) : LinComb :=
  (indices sourceCount).flatMap fun source =>
    sourceOutputTerms layout source output

def outputRow (layout : Layout) (output : Lane) : Row :=
  ⟨[(0, 1)], outputTerms layout output,
    [(layout.output output, 1)]⟩

def outputRows (layout : Layout) : List Row :=
  (indices laneCount).map (outputRow layout)

theorem outputRows_length (layout : Layout) :
    (outputRows layout).length = laneCount := by
  simp [outputRows]

def rows (layout : Layout) : List Row :=
  productRows layout ++ outputRows layout

theorem rows_length (layout : Layout) :
    (rows layout).length = 49626 := by
  rw [rows, List.length_append, productRows_length, outputRows_length]
  decide

def allocation (layout : Layout) : List Nat :=
  (List.range auxiliaryCount).map fun offset => layout.base + offset

theorem allocation_length (layout : Layout) :
    (allocation layout).length = auxiliaryCount := by
  simp [allocation]

theorem allocation_mem_iff (layout : Layout) (column : Nat) :
    column ∈ allocation layout ↔
      layout.base <= column ∧ column < layout.base + auxiliaryCount := by
  unfold allocation
  constructor
  · intro member
    rcases List.mem_map.mp member with ⟨offset, inRange, rfl⟩
    have offsetLt := List.mem_range.mp inRange
    omega
  · rintro ⟨lower, upper⟩
    exact List.mem_map.mpr
      ⟨column - layout.base, List.mem_range.mpr (by omega), by omega⟩

theorem productRow_mem (layout : Layout)
    (source : Source) (left right : Lane) :
    productRow layout source left right ∈ productRows layout := by
  apply List.mem_flatMap.mpr
  refine ⟨source, index_mem source, ?_⟩
  apply List.mem_flatMap.mpr
  refine ⟨left, index_mem left, ?_⟩
  exact List.mem_map.mpr ⟨right, index_mem right, rfl⟩

theorem outputRow_mem (layout : Layout) (output : Lane) :
    outputRow layout output ∈ outputRows layout := by
  exact List.mem_map.mpr ⟨output, index_mem output, rfl⟩

/-- Independent decoded meaning of the exact row family. -/
structure Accepted (layout : Layout) (assignment : Nat -> Nat) : Prop where
  product : forall source left right,
    lcEval assignment (centeredChallenge layout source left) *
        assignment (layout.input source right) % goldilocksP =
      assignment (productColumn layout source left right)
  output : forall lane,
    lcEval assignment (outputTerms layout lane) =
      assignment (layout.output lane)

/-- Exact sparse rows imply every independent product and reduction equation. -/
theorem rows_sound
    {layout : Layout} {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : Satisfies (rows layout) assignment) :
    Accepted layout assignment := by
  constructor
  · intro source left right
    have holds := satisfied (productRow layout source left right)
      (List.mem_append_left _ (productRow_mem layout source left right))
    simpa [RowHolds, productRow, lcEval, one,
      Nat.mod_eq_of_lt (canonical (productColumn layout source left right))]
      using holds
  · intro lane
    have holds := satisfied (outputRow layout lane)
      (List.mem_append_right _ (outputRow_mem layout lane))
    simpa [RowHolds, outputRow, lcEval, one,
      Nat.mod_eq_of_lt (canonical (layout.output lane))] using holds

end Nightstream.Implementation.Nebula.ProductPiRlcRingCombinationRows
