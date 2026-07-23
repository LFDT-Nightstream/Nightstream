import Nightstream.Implementation.R1CS.Correspondence.Gadgets.TerminalCeCompiler

/-!
Exact compiler contract for the terminal delayed-`y_zcol` recomposition rows.

The Rust emitter writes two linear equality rows per active extension-field
lane.  The parent limb equals the radix-weighted sum of the same limb from
every ordered terminal child.  This module owns that row vocabulary and its
kernel soundness/completeness theorem; it does not identify any column with a
terminal child, a pending state, or a semantic packed projection.

Generated artifacts must provide only column layouts and exact row identity.
In particular, no generated datum may assert `Accepted`.
-/

namespace Nightstream.Implementation.R1CS.TerminalPendingProjectionCompiler

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.ProjectionProgram

/-- One active lane of the terminal recomposition. Child columns are in exact
PiDEC order. -/
structure LaneLayout where
  parent : KColumns
  children : List KColumns
deriving DecidableEq, Repr

/-- Fixed row layout. `childCount` is explicit so malformed generated child
lists cannot be silently truncated by `List.zip`. -/
structure Layout where
  radix : Nat
  childCount : Nat
  lanes : List LaneLayout
deriving DecidableEq, Repr

def coefficient (layout : Layout) (child : Nat) : Nat :=
  layout.radix ^ child % goldilocksP

def limbTerms (layout : Layout) (limb : KColumns → Nat)
    (lane : LaneLayout) : List (Nat × Nat) :=
  (List.range layout.childCount).map fun child =>
    (limb (lane.children.getD child default), coefficient layout child)

def c0Terms (layout : Layout) (lane : LaneLayout) : List (Nat × Nat) :=
  limbTerms layout KColumns.c0 lane

def c1Terms (layout : Layout) (lane : LaneLayout) : List (Nat × Nat) :=
  limbTerms layout KColumns.c1 lane

/-- Byte-for-byte row shape of two calls to `R1csBuilder::enforce_eq`. -/
def laneRows (layout : Layout) (lane : LaneLayout) : List Row :=
  [builderLinearRow lane.parent.c0 (c0Terms layout lane),
   builderLinearRow lane.parent.c1 (c1Terms layout lane)]

def rows (layout : Layout) : List Row :=
  layout.lanes.flatMap (laneRows layout)

theorem rows_length (layout : Layout) :
    (rows layout).length = 2 * layout.lanes.length := by
  unfold rows
  induction layout.lanes with
  | nil => rfl
  | cons lane lanes inductionHypothesis =>
      simp [laneRows, inductionHypothesis, Nat.mul_succ]

/-- Shape and coefficient facts required to interpret every emitted row.
These are proof-free layout facts; they carry no assignment or acceptance. -/
structure ShapeValid (layout : Layout) : Prop where
  childWidths : ∀ lane, lane ∈ layout.lanes →
    lane.children.length = layout.childCount
  coefficientsCanonical : ∀ child, child < layout.childCount →
    0 < coefficient layout child ∧ coefficient layout child < goldilocksP

theorem limbTerms_canonical
    {layout : Layout} (valid : ShapeValid layout)
    (limb : KColumns → Nat) (lane : LaneLayout)
    (laneMember : lane ∈ layout.lanes) :
    CanonicalTerms (limbTerms layout limb lane) := by
  intro term termMember
  rcases List.mem_map.mp termMember with ⟨child, childMember, rfl⟩
  exact valid.coefficientsCanonical child (by simpa using childMember)

theorem c0Terms_canonical
    {layout : Layout} (valid : ShapeValid layout)
    (lane : LaneLayout) (laneMember : lane ∈ layout.lanes) :
    CanonicalTerms (c0Terms layout lane) :=
  limbTerms_canonical valid KColumns.c0 lane laneMember

theorem c1Terms_canonical
    {layout : Layout} (valid : ShapeValid layout)
    (lane : LaneLayout) (laneMember : lane ∈ layout.lanes) :
    CanonicalTerms (c1Terms layout lane) :=
  limbTerms_canonical valid KColumns.c1 lane laneMember

/-- Assignment-level meaning of the 108-row family. No child value is a
premise: both sides are decoded from columns owned by the row layout. -/
def Accepted (layout : Layout) (assignment : Nat → Nat) : Prop :=
  ∀ lane, lane ∈ layout.lanes →
    assignment lane.parent.c0 = lcEval assignment (c0Terms layout lane) ∧
    assignment lane.parent.c1 = lcEval assignment (c1Terms layout lane)

/-- Extension-field value of the exact two limb linear combinations. -/
def recomposedValue (layout : Layout) (assignment : Nat → Nat)
    (lane : LaneLayout) : K :=
  { c0 := residue (lcEval assignment (c0Terms layout lane))
    c1 := residue (lcEval assignment (c1Terms layout lane)) }

theorem Accepted.parentValue_eq_recomposedValue
    {layout : Layout} {assignment : Nat → Nat}
    (accepted : Accepted layout assignment)
    (lane : LaneLayout) (laneMember : lane ∈ layout.lanes) :
    lane.parent.value assignment = recomposedValue layout assignment lane := by
  have equations := accepted lane laneMember
  simp only [KColumns.value, baseAt, recomposedValue, K.mk.injEq]
  exact ⟨congrArg residue equations.1, congrArg residue equations.2⟩

/-- Exact row soundness for terminal delayed projection recomposition. -/
theorem rows_sound
    {layout : Layout} (valid : ShapeValid layout)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies (rows layout) assignment) :
    Accepted layout assignment := by
  intro lane laneMember
  constructor
  · apply builderLinearRow_sound canonical one
      lane.parent.c0 (c0Terms layout lane)
      (c0Terms_canonical valid lane laneMember)
    apply satisfies
    exact List.mem_flatMap.mpr
      ⟨lane, laneMember, by simp [laneRows]⟩
  · apply builderLinearRow_sound canonical one
      lane.parent.c1 (c1Terms layout lane)
      (c1Terms_canonical valid lane laneMember)
    apply satisfies
    exact List.mem_flatMap.mpr
      ⟨lane, laneMember, by simp [laneRows]⟩

/-- Honest assignments satisfying the exact limb equations satisfy every
emitted row. This is the local completeness half used by artifact fixtures. -/
theorem rows_complete
    {layout : Layout} (valid : ShapeValid layout)
    {assignment : Nat → Nat}
    (one : assignment 0 = 1)
    (accepted : Accepted layout assignment) :
    Satisfies (rows layout) assignment := by
  intro row rowMember
  rcases List.mem_flatMap.mp rowMember with
    ⟨lane, laneMember, rowMember⟩
  have equations := accepted lane laneMember
  simp [laneRows] at rowMember
  rcases rowMember with rfl | rfl
  · exact builderLinearRow_complete one lane.parent.c0
      (c0Terms layout lane) (c0Terms_canonical valid lane laneMember)
      equations.1
  · exact builderLinearRow_complete one lane.parent.c1
      (c1Terms layout lane) (c1Terms_canonical valid lane laneMember)
      equations.2

/-- Artifact-facing contract. `exactRows` is row identity, not semantic
authority. The remaining fields pin only the radix/child/lane cardinalities;
they do not identify a concrete relation profile or witness width. -/
structure ArtifactContract (layout : Layout) (artifactRows : List Row) : Prop where
  exactRows : artifactRows = rows layout
  radixTwo : layout.radix = 2
  childCountFourteen : layout.childCount = 14
  activeLaneCount : layout.lanes.length = 54
  shape : ShapeValid layout

theorem artifact_rows_sound
    {layout : Layout} {artifactRows : List Row}
    (contract : ArtifactContract layout artifactRows)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies artifactRows assignment) :
    Accepted layout assignment := by
  rw [contract.exactRows] at satisfies
  exact rows_sound contract.shape canonical one satisfies

theorem ArtifactContract.rowCount
    {layout : Layout} {artifactRows : List Row}
    (contract : ArtifactContract layout artifactRows) :
    artifactRows.length = 108 := by
  rw [contract.exactRows, rows_length, contract.activeLaneCount]

end Nightstream.Implementation.R1CS.TerminalPendingProjectionCompiler
