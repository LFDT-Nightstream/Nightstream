import Nightstream.Protocol.Nebula.CommitmentBundle

/-!
Contract: exact whole-ring assignment-lane projections for Nebula V2.

Assurance tier: model-level.

Owns bounded, pairwise-disjoint, 54-coordinate-aligned lane ranges; concrete
coordinate projections from one full assignment; and the exact four-component
map that uses independent full and operations maps plus one shared snapshot
map.

Does not own final compiled widths, Ajtai arithmetic, seeded matrices,
bounded-opening extraction, or a Module-SIS reduction.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Protocol.Nebula.LaneLayout

open Nightstream.Protocol.Nebula.CommitmentBundle

def ringDegree : Nat := 54

def Aligned (coordinateCount : Nat) : Prop :=
  coordinateCount % ringDegree = 0

def DisjointRanges
    (leftStart leftWidth rightStart rightWidth : Nat) : Prop :=
  leftStart + leftWidth ≤ rightStart ∨
    rightStart + rightWidth ≤ leftStart

/-- Widths are type parameters because the commitment matrices are fixed by
the verifier key. Starts are checked values within the complete assignment. -/
structure Layout
    (assignmentWidth operationsWidth snapshotWidth : Nat) where
  operationsStart : Nat
  initialSnapshotStart : Nat
  finalSnapshotStart : Nat
  assignmentAligned : Aligned assignmentWidth
  operationsStartAligned : Aligned operationsStart
  operationsWidthAligned : Aligned operationsWidth
  initialSnapshotStartAligned : Aligned initialSnapshotStart
  snapshotWidthAligned : Aligned snapshotWidth
  finalSnapshotStartAligned : Aligned finalSnapshotStart
  operationsWithin : operationsStart + operationsWidth ≤ assignmentWidth
  initialSnapshotWithin :
    initialSnapshotStart + snapshotWidth ≤ assignmentWidth
  finalSnapshotWithin : finalSnapshotStart + snapshotWidth ≤ assignmentWidth
  operationsInitialDisjoint :
    DisjointRanges operationsStart operationsWidth
      initialSnapshotStart snapshotWidth
  operationsFinalDisjoint :
    DisjointRanges operationsStart operationsWidth
      finalSnapshotStart snapshotWidth
  snapshotsDisjoint :
    DisjointRanges initialSnapshotStart snapshotWidth
      finalSnapshotStart snapshotWidth

namespace Layout

variable {assignmentWidth operationsWidth snapshotWidth : Nat}

def operationsIndex
    (layout : Layout assignmentWidth operationsWidth snapshotWidth)
    (index : Fin operationsWidth) : Fin assignmentWidth :=
  ⟨layout.operationsStart + index.val, by
    have indexBound := index.isLt
    have within := layout.operationsWithin
    omega⟩

def initialSnapshotIndex
    (layout : Layout assignmentWidth operationsWidth snapshotWidth)
    (index : Fin snapshotWidth) : Fin assignmentWidth :=
  ⟨layout.initialSnapshotStart + index.val, by
    have indexBound := index.isLt
    have within := layout.initialSnapshotWithin
    omega⟩

def finalSnapshotIndex
    (layout : Layout assignmentWidth operationsWidth snapshotWidth)
    (index : Fin snapshotWidth) : Fin assignmentWidth :=
  ⟨layout.finalSnapshotStart + index.val, by
    have indexBound := index.isLt
    have within := layout.finalSnapshotWithin
    omega⟩

section Projection

variable {Scalar : Type} [Semiring Scalar]

def operationsProjection
    (layout : Layout assignmentWidth operationsWidth snapshotWidth) :
    (Fin assignmentWidth → Scalar) →ₗ[Scalar]
      (Fin operationsWidth → Scalar) where
  toFun assignment index := assignment (layout.operationsIndex index)
  map_add' := by intro left right; funext index; rfl
  map_smul' := by intro scalar assignment; funext index; rfl

def initialSnapshotProjection
    (layout : Layout assignmentWidth operationsWidth snapshotWidth) :
    (Fin assignmentWidth → Scalar) →ₗ[Scalar]
      (Fin snapshotWidth → Scalar) where
  toFun assignment index := assignment (layout.initialSnapshotIndex index)
  map_add' := by intro left right; funext index; rfl
  map_smul' := by intro scalar assignment; funext index; rfl

def finalSnapshotProjection
    (layout : Layout assignmentWidth operationsWidth snapshotWidth) :
    (Fin assignmentWidth → Scalar) →ₗ[Scalar]
      (Fin snapshotWidth → Scalar) where
  toFun assignment index := assignment (layout.finalSnapshotIndex index)
  map_add' := by intro left right; funext index; rfl
  map_smul' := by intro scalar assignment; funext index; rfl

@[simp]
theorem operationsProjection_apply
    (layout : Layout assignmentWidth operationsWidth snapshotWidth)
    (assignment : Fin assignmentWidth → Scalar)
    (index : Fin operationsWidth) :
    layout.operationsProjection assignment index =
      assignment (layout.operationsIndex index) :=
  rfl

@[simp]
theorem initialSnapshotProjection_apply
    (layout : Layout assignmentWidth operationsWidth snapshotWidth)
    (assignment : Fin assignmentWidth → Scalar)
    (index : Fin snapshotWidth) :
    layout.initialSnapshotProjection assignment index =
      assignment (layout.initialSnapshotIndex index) :=
  rfl

@[simp]
theorem finalSnapshotProjection_apply
    (layout : Layout assignmentWidth operationsWidth snapshotWidth)
    (assignment : Fin assignmentWidth → Scalar)
    (index : Fin snapshotWidth) :
    layout.finalSnapshotProjection assignment index =
      assignment (layout.finalSnapshotIndex index) :=
  rfl

end Projection

end Layout

section BundleMap

variable {Scalar Commitment : Type}
variable [Semiring Scalar]
variable [AddCommMonoid Commitment] [Module Scalar Commitment]
variable {assignmentWidth operationsWidth snapshotWidth : Nat}

/-- The two snapshot roles share the same map by type and construction. The
operations map is a separate field. -/
structure CommitmentMaps where
  full : (Fin assignmentWidth → Scalar) →ₗ[Scalar] Commitment
  operations : (Fin operationsWidth → Scalar) →ₗ[Scalar] Commitment
  snapshot : (Fin snapshotWidth → Scalar) →ₗ[Scalar] Commitment

def componentMaps
    (layout : Layout assignmentWidth operationsWidth snapshotWidth)
    (maps : CommitmentMaps (Scalar := Scalar) (Commitment := Commitment)
      (assignmentWidth := assignmentWidth)
      (operationsWidth := operationsWidth)
      (snapshotWidth := snapshotWidth)) :
    Component → (Fin assignmentWidth → Scalar) →ₗ[Scalar] Commitment
  | .full => maps.full
  | .operations => maps.operations.comp layout.operationsProjection
  | .initialSnapshot => maps.snapshot.comp layout.initialSnapshotProjection
  | .finalSnapshot => maps.snapshot.comp layout.finalSnapshotProjection

def bundleMap
    (layout : Layout assignmentWidth operationsWidth snapshotWidth)
    (maps : CommitmentMaps (Scalar := Scalar) (Commitment := Commitment)
      (assignmentWidth := assignmentWidth)
      (operationsWidth := operationsWidth)
      (snapshotWidth := snapshotWidth)) :
    (Fin assignmentWidth → Scalar) →ₗ[Scalar] Bundle Commitment :=
  productMap (componentMaps layout maps)

theorem initial_uses_shared_snapshot_map
    (layout : Layout assignmentWidth operationsWidth snapshotWidth)
    (maps : CommitmentMaps (Scalar := Scalar) (Commitment := Commitment)
      (assignmentWidth := assignmentWidth)
      (operationsWidth := operationsWidth)
      (snapshotWidth := snapshotWidth))
    (assignment : Fin assignmentWidth → Scalar) :
    bundleMap layout maps assignment .initialSnapshot =
      maps.snapshot (layout.initialSnapshotProjection assignment) :=
  rfl

theorem final_uses_shared_snapshot_map
    (layout : Layout assignmentWidth operationsWidth snapshotWidth)
    (maps : CommitmentMaps (Scalar := Scalar) (Commitment := Commitment)
      (assignmentWidth := assignmentWidth)
      (operationsWidth := operationsWidth)
      (snapshotWidth := snapshotWidth))
    (assignment : Fin assignmentWidth → Scalar) :
    bundleMap layout maps assignment .finalSnapshot =
      maps.snapshot (layout.finalSnapshotProjection assignment) :=
  rfl

end BundleMap

end Nightstream.Protocol.Nebula.LaneLayout
