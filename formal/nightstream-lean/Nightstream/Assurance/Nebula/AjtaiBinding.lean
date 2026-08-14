import Mathlib.Algebra.BigOperators.Field
import Nightstream.Protocol.Nebula.CompactCommit
import Nightstream.Protocol.Nebula.CommitmentBundle

/-!
Contract: deterministic Ajtai collision-to-Module-SIS reduction for Nebula V2.

Assurance tier: security-reduced boundary.

Owns an exact finite matrix equation, centered integer witnesses, strict
infinity-norm bounds, the nonzero kernel vector extracted from two different
openings, the exact compact-token ring shapes, and bridges from the V2 primary,
short, and full-bundle collision events.

Does not prove Module-SIS hardness, concrete matrix distribution, the
ChaCha8 PRG assumption, cyclotomic-ring security estimates, or Rust/R1CS
refinement to the matrix equation.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Assurance.Nebula.AjtaiBinding

open scoped BigOperators
open Nightstream.Protocol.Nebula
open Nightstream.Protocol.Nebula.CommitmentBundle
open Nightstream.Protocol.Nebula.CompactCommit

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
column. Treating a ring column as one integer would omit 53 coefficients in
the selected degree-54 ring. -/
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

/-- The exact Module-SIS event exposed by the reduction. Computational
hardness is a separate assumption about the selected matrix distribution. -/
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

/-- A map refinement contains only exact representation equations. It does
not contain a binding or security conclusion. -/
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

/-! ## Exact V2 compact-map shapes -/

def primaryShape : Shape where
  rows := primaryRank
  columns := primaryMessageRingColumns
  degree := ringDegree

def shortShape : Shape where
  rows := shortRank
  columns := shortMessageRingColumns
  degree := ringDegree

theorem exact_compact_shapes :
    primaryShape.rows = 2 ∧ primaryShape.columns = 738 ∧
      primaryShape.degree = 54 ∧
      shortShape.rows = 1 ∧ shortShape.columns = 82 ∧
      shortShape.degree = 54 := by
  decide

/-- Exact matrix refinement for the rank-two V2 primary map. Its witness is
the normative `ShiftedTernary41V1` ring packing, not a caller-selected
injective encoding. -/
structure PrimaryMapRefinement
    {Plan Seed RingType : Type} [CommRing RingType]
    (key : Key Plan Seed) (role : Role) where
  matrix : Matrix RingType primaryShape
  coefficientMap : CoefficientVector primaryShape →+ RingType
  outputEquiv : Commitment RingType primaryShape ≃ PrimaryOutput
  correct : ∀ commitment,
    key.primary role (packFields primaryPacking commitment) =
      outputEquiv
        (commit matrix coefficientMap
          (packFields primaryPacking commitment))

def PrimaryMapRefinement.toMapRefinement
    {Plan Seed RingType : Type} [CommRing RingType]
    {key : Key Plan Seed} {role : Role}
    (refinement : PrimaryMapRefinement (RingType := RingType) key role) :
    MapRefinement CommitmentEncoding PrimaryOutput RingType primaryShape
      (fun commitment =>
        key.primary role (packFields primaryPacking commitment)) where
  matrix := refinement.matrix
  coefficientMap := refinement.coefficientMap
  witness := packFields primaryPacking
  witnessInjective := packFields_injective primaryPacking
  outputEquiv := refinement.outputEquiv
  correct := refinement.correct

/-- Exact matrix refinement for the independent rank-one short map. -/
structure ShortMapRefinement
    {Plan Seed RingType : Type} [CommRing RingType]
    (key : Key Plan Seed) (role : Role) where
  matrix : Matrix RingType shortShape
  coefficientMap : CoefficientVector shortShape →+ RingType
  outputEquiv : Commitment RingType shortShape ≃ Token
  correct : ∀ primaryOutput,
    key.short role (packFields shortPacking primaryOutput) =
      outputEquiv
        (commit matrix coefficientMap
          (packFields shortPacking primaryOutput))

def ShortMapRefinement.toMapRefinement
    {Plan Seed RingType : Type} [CommRing RingType]
    {key : Key Plan Seed} {role : Role}
    (refinement : ShortMapRefinement (RingType := RingType) key role) :
    MapRefinement PrimaryOutput Token RingType shortShape
      (fun output => key.short role (packFields shortPacking output)) where
  matrix := refinement.matrix
  coefficientMap := refinement.coefficientMap
  witness := packFields shortPacking
  witnessInjective := packFields_injective shortPacking
  outputEquiv := refinement.outputEquiv
  correct := refinement.correct

/-- A V2 primary failure becomes an exact norm-3 Module-SIS kernel witness
after the concrete signed-unit matrix refinement is supplied. -/
theorem primary_failure_to_kernel
    {Plan Seed RingType : Type} [CommRing RingType]
    (key : Key Plan Seed) (role : Role)
    (refinement :
      MapRefinement CommitmentEncoding PrimaryOutput RingType primaryShape
        (fun commitment =>
          key.primary role (packFields primaryPacking commitment)))
    (unitBound : ∀ input column coefficient,
      (refinement.witness input column coefficient).natAbs ≤ 1)
    (failure : PrimaryBindingFailure key role) :
    Nonempty (KernelWitness refinement.matrix refinement.coefficientMap 3) := by
  rcases failure with ⟨left, right, different, equalMap⟩
  exact signed_unit_collision_to_kernel refinement different equalMap unitBound

/-- The exact V2 packing supplies injectivity and the norm-one coefficient
bound. Only matrix correctness remains in the refinement input. -/
theorem exact_primary_failure_to_kernel
    {Plan Seed RingType : Type} [CommRing RingType]
    (key : Key Plan Seed) (role : Role)
    (refinement : PrimaryMapRefinement (RingType := RingType) key role)
    (failure : PrimaryBindingFailure key role) :
    Nonempty (KernelWitness refinement.matrix refinement.coefficientMap 3) := by
  rcases failure with ⟨left, right, different, equalMap⟩
  exact signed_unit_collision_to_kernel refinement.toMapRefinement
    different equalMap (fun input => packFields_unit_bound primaryPacking input)

/-- The independent short map has the exact same deterministic reduction with
its separate 1-by-82 ring matrix. -/
theorem short_failure_to_kernel
    {Plan Seed RingType : Type} [CommRing RingType]
    (key : Key Plan Seed) (role : Role)
    (refinement :
      MapRefinement PrimaryOutput Token RingType shortShape
        (fun output => key.short role (packFields shortPacking output)))
    (unitBound : ∀ input column coefficient,
      (refinement.witness input column coefficient).natAbs ≤ 1)
    (failure : ShortBindingFailure key role) :
    Nonempty (KernelWitness refinement.matrix refinement.coefficientMap 3) := by
  rcases failure with ⟨left, right, different, equalMap⟩
  exact signed_unit_collision_to_kernel refinement different equalMap unitBound

theorem exact_short_failure_to_kernel
    {Plan Seed RingType : Type} [CommRing RingType]
    (key : Key Plan Seed) (role : Role)
    (refinement : ShortMapRefinement (RingType := RingType) key role)
    (failure : ShortBindingFailure key role) :
    Nonempty (KernelWitness refinement.matrix refinement.coefficientMap 3) := by
  rcases failure with ⟨left, right, different, equalMap⟩
  exact signed_unit_collision_to_kernel refinement.toMapRefinement
    different equalMap (fun input => packFields_unit_bound shortPacking input)

/-! ## Full product-bundle opening -/

def fullShape (assignmentRingColumns : Nat) : Shape where
  rows := commitmentRank
  columns := assignmentRingColumns
  degree := ringDegree

theorem full_shape_rows (assignmentRingColumns : Nat) :
    (fullShape assignmentRingColumns).rows = 18 := by
  rfl

/-- Because the full component binds the complete assignment, one collision
in the atomic four-component bundle reduces to the exact full-map Module-SIS
event. Lane components remain necessary for sequence authority and efficient
replay, but they cannot authorize a different full witness. -/
theorem bundle_failure_to_full_kernel
    {Assignment BundleCommitment RingType : Type} [CommRing RingType]
    {assignmentRingColumns bound : Nat}
    (bounded : Assignment → Prop)
    (bundle : Assignment → Bundle BundleCommitment)
    (refinement :
      MapRefinement Assignment BundleCommitment RingType
        (fullShape assignmentRingColumns)
        (fun assignment => bundle assignment .full))
    (refinesBound : ∀ assignment,
      bounded assignment → Bounded bound (refinement.witness assignment))
    (failure : BindingFailure bounded bundle) :
    Nonempty (KernelWitness refinement.matrix refinement.coefficientMap
      (2 * bound)) := by
  rcases failure with
    ⟨left, right, leftBounded, rightBounded, different, equalBundle⟩
  exact collision_to_kernel refinement different
    (congrFun equalBundle .full)
    (refinesBound left leftBounded) (refinesBound right rightBounded)

end Nightstream.Assurance.Nebula.AjtaiBinding
