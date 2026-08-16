import Nightstream.Protocol.Nebula.AjtaiBinding
import Nightstream.Protocol.Nebula.CompactCommit
import Nightstream.Protocol.Nebula.CommitmentBundle

/-!
Contract: Nebula V2 Ajtai collision-to-Module-SIS bridges.

Assurance tier: security-reduced boundary.

Owns the exact compact-token ring shapes and bridges from the V2 primary,
short, and full-bundle collision events to the model-level bounded kernel
event.

Does not own the finite matrix algebra, concrete matrix distribution, Module-
SIS hardness, the ChaCha8 PRG assumption, cyclotomic-ring security estimates,
or Rust/R1CS refinement to the matrix equation.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Assurance.Nebula.AjtaiBinding

open Nightstream.Protocol.Nebula
open Nightstream.Protocol.Nebula.AjtaiBinding
open Nightstream.Protocol.Nebula.CommitmentBundle
open Nightstream.Protocol.Nebula.CompactCommit

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

/-- The independent short map has the same deterministic reduction with its
separate 1-by-82 ring matrix. -/
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
