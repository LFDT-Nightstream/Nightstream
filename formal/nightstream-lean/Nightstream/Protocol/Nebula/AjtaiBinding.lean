import Mathlib.Algebra.BigOperators.Field

/-!
Contract: finite Ajtai matrix algebra and deterministic collision extraction.

Assurance tier: model-level.

Owns exact ring-coordinate matrix shapes, centered integer witnesses, strict
infinity-norm bounds, map refinement, and extraction of a nonzero bounded
kernel vector from two different openings with the same image.

Does not own a matrix distribution, Module-SIS hardness, protocol-specific
packing, seeded setup, generated rows, or Rust refinement.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Protocol.Nebula.AjtaiBinding

open scoped BigOperators

/-- One finite Ajtai matrix shape in ring coordinates. -/
structure Shape where
  rows : Nat
  columns : Nat
  degree : Nat
deriving DecidableEq, Repr

abbrev Matrix (RingType : Type) (shape : Shape) :=
  Fin shape.rows → Fin shape.columns → RingType

abbrev CoefficientVector (shape : Shape) := Fin shape.degree → Int

/-- A Module-SIS witness has one bounded coefficient vector for each ring
column. -/
abbrev Witness (shape : Shape) :=
  Fin shape.columns → CoefficientVector shape

abbrev Commitment (RingType : Type) (shape : Shape) :=
  Fin shape.rows → RingType

/-- Strict centered infinity-norm bound. -/
def Bounded (bound : Nat) {shape : Shape}
    (witness : Witness shape) : Prop :=
  ∀ column coefficient,
    (witness column coefficient).natAbs < bound

/-- Exact finite Ajtai matrix action. -/
def commit
    {RingType : Type} [CommRing RingType] {shape : Shape}
    (matrix : Matrix RingType shape)
    (coefficientMap : CoefficientVector shape →+ RingType)
    (witness : Witness shape) :
    Commitment RingType shape :=
  fun row => ∑ column,
    coefficientMap (witness column) * matrix row column

/-- The exact bounded kernel event. Computational hardness is separate. -/
structure KernelWitness
    {RingType : Type} [CommRing RingType] {shape : Shape}
    (matrix : Matrix RingType shape)
    (coefficientMap : CoefficientVector shape →+ RingType)
    (bound : Nat) where
  vector : Witness shape
  nonzero : vector ≠ 0
  bounded : Bounded bound vector
  kernel : commit matrix coefficientMap vector = 0

theorem difference_nonzero
    {shape : Shape} {left right : Witness shape}
    (different : left ≠ right) :
    (fun column coefficient =>
      left column coefficient - right column coefficient) ≠ 0 := by
  intro zeroDifference
  apply different
  funext column coefficient
  have atColumn := congrFun zeroDifference column
  have atCoefficient := congrFun atColumn coefficient
  exact sub_eq_zero.mp atCoefficient

theorem difference_bounded
    {shape : Shape} {left right : Witness shape} {bound : Nat}
    (leftBounded : Bounded bound left)
    (rightBounded : Bounded bound right) :
    Bounded (2 * bound) (fun column coefficient =>
      left column coefficient - right column coefficient) := by
  intro column coefficient
  have triangle := Int.natAbs_sub_le
    (left column coefficient) (right column coefficient)
  have sumBound :
      (left column coefficient).natAbs +
          (right column coefficient).natAbs <
        bound + bound :=
    Nat.add_lt_add (leftBounded column coefficient)
      (rightBounded column coefficient)
  exact triangle.trans_lt (by simpa [two_mul] using sumBound)

theorem signed_unit_difference_bounded
    {shape : Shape} {left right : Witness shape}
    (leftUnit : ∀ column coefficient,
      (left column coefficient).natAbs ≤ 1)
    (rightUnit : ∀ column coefficient,
      (right column coefficient).natAbs ≤ 1) :
    Bounded 3 (fun column coefficient =>
      left column coefficient - right column coefficient) := by
  intro column coefficient
  have triangle := Int.natAbs_sub_le
    (left column coefficient) (right column coefficient)
  have sumBound :
      (left column coefficient).natAbs +
          (right column coefficient).natAbs ≤ 2 := by
    simpa using Nat.add_le_add (leftUnit column coefficient)
      (rightUnit column coefficient)
  exact triangle.trans_lt (sumBound.trans_lt (by decide))

theorem commit_difference_eq_zero
    {RingType : Type} [CommRing RingType] {shape : Shape}
    (matrix : Matrix RingType shape)
    (coefficientMap : CoefficientVector shape →+ RingType)
    {left right : Witness shape}
    (equalCommitment :
      commit matrix coefficientMap left =
        commit matrix coefficientMap right) :
    commit matrix coefficientMap (fun column coefficient =>
      left column coefficient - right column coefficient) = 0 := by
  funext row
  have atRow := congrFun equalCommitment row
  change
    (∑ column, coefficientMap (left column) * matrix row column) =
      ∑ column, coefficientMap (right column) * matrix row column at atRow
  change
    (∑ column,
      coefficientMap (fun coefficient =>
        left column coefficient - right column coefficient) *
          matrix row column) = (0 : RingType)
  have pointwise (column : Fin shape.columns) :
      (fun coefficient =>
        left column coefficient - right column coefficient) =
        left column - right column :=
    rfl
  simp_rw [pointwise, map_sub, sub_mul, Finset.sum_sub_distrib]
  exact sub_eq_zero.mpr atRow

/-- A map refinement contains exact representation equations, not security. -/
structure MapRefinement
    (Input Output RingType : Type) [CommRing RingType]
    (shape : Shape) (map : Input → Output) where
  matrix : Matrix RingType shape
  coefficientMap : CoefficientVector shape →+ RingType
  witness : Input → Witness shape
  witnessInjective : Function.Injective witness
  outputEquiv : Commitment RingType shape ≃ Output
  correct : ∀ input,
    map input = outputEquiv
      (commit matrix coefficientMap (witness input))

/-- Exact collision at a refined map boundary. -/
def MapCollision
    {Input Output : Type} (map : Input → Output) : Prop :=
  ∃ left right, left ≠ right ∧ map left = map right

theorem collision_to_kernel
    {Input Output RingType : Type} [CommRing RingType]
    {shape : Shape} {map : Input → Output}
    (refinement : MapRefinement Input Output RingType shape map)
    {left right : Input}
    (different : left ≠ right)
    (equalMap : map left = map right)
    {bound : Nat}
    (leftBounded : Bounded bound (refinement.witness left))
    (rightBounded : Bounded bound (refinement.witness right)) :
    Nonempty (KernelWitness refinement.matrix refinement.coefficientMap
      (2 * bound)) := by
  refine ⟨{
    vector := fun column coefficient =>
      refinement.witness left column coefficient -
        refinement.witness right column coefficient
    nonzero := difference_nonzero
      (fun equal => different (refinement.witnessInjective equal))
    bounded := difference_bounded leftBounded rightBounded
    kernel := ?_
  }⟩
  apply commit_difference_eq_zero refinement.matrix refinement.coefficientMap
  apply refinement.outputEquiv.injective
  rw [← refinement.correct left, ← refinement.correct right]
  exact equalMap

theorem signed_unit_collision_to_kernel
    {Input Output RingType : Type} [CommRing RingType]
    {shape : Shape} {map : Input → Output}
    (refinement : MapRefinement Input Output RingType shape map)
    {left right : Input}
    (different : left ≠ right)
    (equalMap : map left = map right)
    (unitBound : ∀ input column coefficient,
      (refinement.witness input column coefficient).natAbs ≤ 1) :
    Nonempty (KernelWitness refinement.matrix refinement.coefficientMap 3) := by
  refine ⟨{
    vector := fun column coefficient =>
      refinement.witness left column coefficient -
        refinement.witness right column coefficient
    nonzero := difference_nonzero
      (fun equal => different (refinement.witnessInjective equal))
    bounded := signed_unit_difference_bounded
      (unitBound left) (unitBound right)
    kernel := ?_
  }⟩
  apply commit_difference_eq_zero refinement.matrix refinement.coefficientMap
  apply refinement.outputEquiv.injective
  rw [← refinement.correct left, ← refinement.correct right]
  exact equalMap

end Nightstream.Protocol.Nebula.AjtaiBinding
