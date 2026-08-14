import Mathlib.Data.List.OfFn
import Nightstream.Implementation.R1CS.Core.ConstantPins
import Nightstream.Implementation.R1CS.Core.EqualityPins
import Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.StateXOutProgramRefinement

/-!
Contract: exact 32-field stateful-with-Nebula V2 state-output frame and carry
digest links.

Assurance tier: implementation model.

Owns the fixed state-output domain, mandatory Nebula marker, exact coordinate
order, four equality rows from the recomputed carry digest to the Nebula
frame lanes, exclusion of all three non-V2 optional-lane shapes, and honest
row completeness.

Does not own the carry-digest computation, outer Poseidon2 trace, placement
of the other recursive-state fields, absolute generated columns, or Rust
execution conformance.

Emits constraints: yes.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Nebula.StateOutputFrameRows

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.Rust.CanonicalConformance.NativeStep
open Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.StateXOutProgram

def domainTag : Nat := 0x4e460002
def nebulaMarker : Nat := 0x4e424c41

structure Layout where
  domainColumn : Nat
  vkFsDigestColumn : Fin 4 → Nat
  piCcsHeaderColumn : Fin 4 → Nat
  chunkCountHalfColumn : Fin 2 → Nat
  stepCountHalfColumn : Fin 2 → Nat
  pcHalfColumn : Fin 2 → Nat
  currentBoundaryColumn : Fin 4 → Nat
  semanticStateColumn : Fin 4 → Nat
  accumulatorDigestColumn : Fin 4 → Nat
  nebulaMarkerColumn : Nat
  nebulaDigestColumn : Fin 4 → Nat
  carryDigestOutputColumn : Fin 4 → Nat

def payloadColumns (layout : Layout) : List Nat :=
  List.ofFn layout.vkFsDigestColumn ++
  List.ofFn layout.piCcsHeaderColumn ++
  List.ofFn layout.chunkCountHalfColumn ++
  List.ofFn layout.stepCountHalfColumn ++
  List.ofFn layout.pcHalfColumn ++
  List.ofFn layout.currentBoundaryColumn ++
  List.ofFn layout.semanticStateColumn ++
  List.ofFn layout.accumulatorDigestColumn

theorem payloadColumns_length (layout : Layout) :
    (payloadColumns layout).length = 26 := by
  simp [payloadColumns]

def inputColumns (layout : Layout) : List Nat :=
  [layout.domainColumn] ++ payloadColumns layout ++
    [layout.nebulaMarkerColumn] ++ List.ofFn layout.nebulaDigestColumn

theorem inputColumns_length (layout : Layout) :
    (inputColumns layout).length = 32 := by
  simp [inputColumns, payloadColumns_length]

def constantPins (layout : Layout) : List (Nat × Nat) :=
  [(layout.domainColumn, domainTag),
   (layout.nebulaMarkerColumn, nebulaMarker)]

def digestLinks (layout : Layout) : List (Nat × Nat) :=
  List.ofFn fun lane : Fin 4 =>
    (layout.nebulaDigestColumn lane, layout.carryDigestOutputColumn lane)

def rows (layout : Layout) : List Row :=
  ConstantPins.rows (constantPins layout) ++
    EqualityPins.rows (digestLinks layout)

theorem rows_length_exact (layout : Layout) :
    (rows layout).length = 6 := by
  simp [rows, ConstantPins.rows, EqualityPins.rows, constantPins,
    digestLinks]

theorem constantPins_canonical (layout : Layout) :
    ConstantPins.ValuesCanonical (constantPins layout) := by
  intro pin member
  simp [constantPins] at member
  rcases member with rfl | rfl
  · change domainTag < goldilocksP
    decide
  · change nebulaMarker < goldilocksP
    decide

private theorem constant_rows_included (layout : Layout) :
    rowsIncluded (ConstantPins.rows (constantPins layout))
      (rows layout) = true := by
  unfold rowsIncluded
  apply List.all_eq_true.mpr
  intro row member
  exact decide_eq_true (by simp [rows, member])

private theorem digest_rows_hold
    {layout : Layout} {assignment : Nat → Nat}
    (holds : Satisfies (rows layout) assignment) :
    Satisfies (EqualityPins.rows (digestLinks layout)) assignment := by
  intro row member
  exact holds row (by simp [rows, member])

theorem domain_column_eq
    {layout : Layout} {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (holds : Satisfies (rows layout) assignment) :
    assignment layout.domainColumn = domainTag := by
  exact ConstantPins.sound (constantPins_canonical layout)
    (constant_rows_included layout) canonical one holds
    (layout.domainColumn, domainTag) (by simp [constantPins])

theorem marker_column_eq
    {layout : Layout} {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (holds : Satisfies (rows layout) assignment) :
    assignment layout.nebulaMarkerColumn = nebulaMarker := by
  exact ConstantPins.sound (constantPins_canonical layout)
    (constant_rows_included layout) canonical one holds
    (layout.nebulaMarkerColumn, nebulaMarker) (by simp [constantPins])

theorem digest_column_eq
    {layout : Layout} {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (holds : Satisfies (rows layout) assignment)
    (lane : Fin 4) :
    assignment (layout.nebulaDigestColumn lane) =
      assignment (layout.carryDigestOutputColumn lane) := by
  exact EqualityPins.rows_sound canonical one (digest_rows_hold holds)
    (layout.nebulaDigestColumn lane, layout.carryDigestOutputColumn lane)
    (List.mem_ofFn.mpr ⟨lane, rfl⟩)

def sourceFrame (layout : Layout) (assignment : Nat → Nat)
    (carryDigest : Fin 4 → Nat) : List Nat :=
  [domainTag] ++ (payloadColumns layout).map assignment ++
    [nebulaMarker] ++ List.ofFn carryDigest

theorem sourceFrame_length (layout : Layout) (assignment : Nat → Nat)
    (carryDigest : Fin 4 → Nat) :
    (sourceFrame layout assignment carryDigest).length = 32 := by
  simp [sourceFrame, payloadColumns_length]

theorem sourceFrame_canonical
    (layout : Layout) (assignment : Nat → Nat) (carryDigest : Fin 4 → Nat)
    (canonicalAssignment : ∀ column, assignment column < goldilocksP)
    (canonicalCarry : ∀ lane, carryDigest lane < goldilocksP) :
    ∀ value ∈ sourceFrame layout assignment carryDigest,
      value < goldilocksP := by
  intro value member
  simp only [sourceFrame, List.mem_append, List.mem_singleton] at member
  rcases member with (((domain | payload) | marker) | digest)
  · subst value
    decide
  · rcases List.mem_map.mp payload with ⟨column, columnMember, rfl⟩
    exact canonicalAssignment column
  · subst value
    decide
  · rcases List.mem_ofFn.mp digest with ⟨lane, rfl⟩
    exact canonicalCarry lane

/-- Equality of complete source frames recovers all four carry-digest lanes.
This is a packing fact, not a collision-resistance assumption. -/
theorem carryDigest_eq_of_sourceFrame_eq
    {layout : Layout} {leftAssignment rightAssignment : Nat → Nat}
    {leftCarry rightCarry : Fin 4 → Nat}
    (equal : sourceFrame layout leftAssignment leftCarry =
      sourceFrame layout rightAssignment rightCarry) :
    leftCarry = rightCarry := by
  have tails := congrArg (List.drop 28) equal
  have encoded : List.ofFn leftCarry = List.ofFn rightCarry := by
    simpa [sourceFrame, payloadColumns] using tails
  exact List.ofFn_injective encoded

/-- Equality of complete source frames also recovers all 26 non-memory
payload values in their exact order. -/
theorem payload_values_eq_of_sourceFrame_eq
    {layout : Layout} {leftAssignment rightAssignment : Nat → Nat}
    {leftCarry rightCarry : Fin 4 → Nat}
    (equal : sourceFrame layout leftAssignment leftCarry =
      sourceFrame layout rightAssignment rightCarry) :
    (payloadColumns layout).map leftAssignment =
      (payloadColumns layout).map rightAssignment := by
  have middle := congrArg (fun values => (values.drop 1).take 26) equal
  simpa [sourceFrame, payloadColumns] using middle

/-- The exact 32 input columns contain the stateful frame and the supplied
recomputed carry digest. -/
theorem input_column_values
    {layout : Layout} {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (holds : Satisfies (rows layout) assignment)
    (carryDigest : Fin 4 → Nat)
    (carryOutputs : ∀ lane,
      assignment (layout.carryDigestOutputColumn lane) = carryDigest lane) :
    (inputColumns layout).map assignment =
      sourceFrame layout assignment carryDigest := by
  have digestValues :
      (List.ofFn layout.nebulaDigestColumn).map assignment =
        List.ofFn carryDigest := by
    apply List.ext_getElem
    · simp
    · intro index leftBound rightBound
      simp only [List.getElem_map, List.getElem_ofFn]
      exact (digest_column_eq canonical one holds ⟨index, by simpa using leftBound⟩).trans
        (carryOutputs ⟨index, by simpa using leftBound⟩)
  simp only [inputColumns, sourceFrame, List.map_append, List.map_cons,
    List.map_nil]
  rw [domain_column_eq canonical one holds,
    marker_column_eq canonical one holds, digestValues]

/-- V2 has only the stateful-with-Nebula source-program shape. -/
theorem canonical_shape_eq_v2_iff
    (semanticPresent nebulaPresent : Bool) :
    canonical semanticPresent nebulaPresent = canonical true true ↔
      semanticPresent = true ∧ nebulaPresent = true := by
  cases semanticPresent <;> cases nebulaPresent <;> decide

theorem v2_source_program_cost : cost (canonical true true) = 32 :=
  statefulNebula_cost

structure Honest (layout : Layout) (assignment : Nat → Nat) : Prop where
  domainPlaced : assignment layout.domainColumn = domainTag
  markerPlaced : assignment layout.nebulaMarkerColumn = nebulaMarker
  digestLinked : ∀ lane,
    assignment (layout.nebulaDigestColumn lane) =
      assignment (layout.carryDigestOutputColumn lane)

theorem rows_complete
    {layout : Layout} {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (honest : Honest layout assignment) :
    Satisfies (rows layout) assignment := by
  intro row member
  rw [rows, List.mem_append] at member
  rcases member with constantMember | linkMember
  · apply ConstantPins.complete (constantPins_canonical layout) one _
      row constantMember
    intro pin pinMember
    simp [constantPins] at pinMember
    rcases pinMember with rfl | rfl
    · exact honest.domainPlaced
    · exact honest.markerPlaced
  · apply EqualityPins.rows_complete canonical one _ row linkMember
    intro pair pairMember
    rcases List.mem_ofFn.mp pairMember with ⟨lane, rfl⟩
    exact honest.digestLinked lane

end Nightstream.Implementation.Nebula.StateOutputFrameRows
