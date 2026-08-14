import Nightstream.Implementation.Nebula.Commitment.Compact.ChainHashFrameRows
import Nightstream.Implementation.Nebula.Memory.Transcript.HashFrame

/-!
Contract: exact R1CS input frame for the V2 memory challenge transcript.

Assurance tier: implementation model.

Owns the six fixed prefix columns, three fixed geometry columns, ordered
authority/counter/root columns, row soundness to the exact 53-field frame,
and local honest completeness.

Does not own the source of variable-column authority, Poseidon2 rows,
challenge output linkage, absolute generated columns, or Rust conformance.

Emits constraints: yes.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Nebula.MemoryTranscriptHashFrameRows

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.Nebula.MemoryTranscriptHashFrame
open Nightstream.Implementation.Nebula.CompactChainHashFrameRows
open Nightstream.Protocol.Nebula
open Nightstream.Protocol.Nebula.Lifecycle

structure Layout where
  prefixColumn : Fin 6 → Nat
  authorityColumn : Fin 28 → Nat
  counterColumn : Fin 4 → Nat
  geometryColumn : Fin 3 → Nat
  rootColumn : Fin 12 → Nat
deriving DecidableEq, Repr

def Layout.prefixColumns (layout : Layout) : List Nat :=
  List.ofFn layout.prefixColumn

def Layout.authorityColumns (layout : Layout) : List Nat :=
  List.ofFn layout.authorityColumn

def Layout.counterColumns (layout : Layout) : List Nat :=
  List.ofFn layout.counterColumn

def Layout.geometryColumns (layout : Layout) : List Nat :=
  List.ofFn layout.geometryColumn

def Layout.rootColumns (layout : Layout) : List Nat :=
  List.ofFn layout.rootColumn

def Layout.inputColumns (layout : Layout) : List Nat :=
  layout.prefixColumns ++ layout.authorityColumns ++
    layout.counterColumns ++ layout.geometryColumns ++ layout.rootColumns

def geometryValues : List Nat := [claimsPerSegment, 63, 64]

theorem geometryValues_exact : geometryValues = [1088, 63, 64] := by
  rfl

def rows (layout : Layout) : List Row :=
  FixedFrame.rows layout.prefixColumns fixedPrefix ++
    FixedFrame.rows layout.geometryColumns geometryValues

theorem rows_length_exact (layout : Layout) :
    (rows layout).length = 9 := by
  simp [rows, FixedFrame.rows, ConstantPins.rows, FixedFrame.pins,
    Layout.prefixColumns, Layout.geometryColumns, fixedPrefix_length,
    geometryValues]

private theorem prefix_values_canonical :
    ∀ value ∈ fixedPrefix, value < goldilocksP := by
  intro value member
  simp only [fixedPrefix, profileFields, List.mem_append,
    List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with (rfl | rfl) | rfl | rfl | rfl | rfl <;> decide

private theorem geometry_values_canonical :
    ∀ value ∈ geometryValues, value < goldilocksP := by
  intro value member
  simp only [geometryValues, List.mem_cons, List.not_mem_nil, or_false]
    at member
  rcases member with rfl | rfl | rfl <;> decide

private theorem prefix_rows_hold
    {layout : Layout} {assignment : Nat → Nat}
    (holds : Satisfies (rows layout) assignment) :
    Satisfies (FixedFrame.rows layout.prefixColumns fixedPrefix) assignment := by
  intro row member
  exact holds row (by simp [rows, member])

private theorem geometry_rows_hold
    {layout : Layout} {assignment : Nat → Nat}
    (holds : Satisfies (rows layout) assignment) :
    Satisfies
      (FixedFrame.rows layout.geometryColumns geometryValues) assignment := by
  intro row member
  exact holds row (by simp [rows, member])

theorem prefix_exact
    {layout : Layout} {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (holds : Satisfies (rows layout) assignment) :
    layout.prefixColumns.map assignment = fixedPrefix := by
  exact FixedFrame.sound
    (by simp [Layout.prefixColumns, fixedPrefix_length])
    prefix_values_canonical canonical one (prefix_rows_hold holds)

theorem geometry_exact
    {layout : Layout} {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (holds : Satisfies (rows layout) assignment) :
    layout.geometryColumns.map assignment = geometryValues := by
  exact FixedFrame.sound
    (by simp [Layout.geometryColumns, geometryValues])
    geometry_values_canonical canonical one (geometry_rows_hold holds)

/-- Variable data placement is deliberately separate from row satisfaction.
The enclosing open-segment relation must prove this structure from
verifier-owned statement data, prior-state outputs, bounded counters, and the
three precommit roots. -/
structure VariablePlaced (layout : Layout) (assignment : Nat → Nat)
    (input : Input) : Prop where
  authority :
    layout.authorityColumns.map assignment = authorityDigestFields input
  counters : layout.counterColumns.map assignment =
    [ input.segmentIndex
    , input.segmentStartTimestamp
    , input.activeAccessCount
    , input.segmentEndTimestamp
    ]
  roots : layout.rootColumns.map assignment = rootFields input

/-- The ordered 53 assigned columns are the exact normative frame. -/
theorem input_exact
    {layout : Layout} {assignment : Nat → Nat} {input : Input}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (placed : VariablePlaced layout assignment input)
    (holds : Satisfies (rows layout) assignment) :
    layout.inputColumns.map assignment = encode input := by
  rw [Layout.inputColumns]
  simp only [List.map_append]
  rw [prefix_exact canonical one holds, placed.authority, placed.counters,
    geometry_exact canonical one holds, placed.roots]
  simp [encode, counterFields, geometryValues]

structure Honest (layout : Layout) (assignment : Nat → Nat)
    (input : Input) : Prop where
  variablePlaced : VariablePlaced layout assignment input
  prefixPlaced : layout.prefixColumns.map assignment = fixedPrefix
  geometryPlaced : layout.geometryColumns.map assignment = geometryValues

theorem rows_complete
    {layout : Layout} {assignment : Nat → Nat} {input : Input}
    (one : assignment 0 = 1)
    (honest : Honest layout assignment input) :
    Satisfies (rows layout) assignment := by
  intro row member
  rw [rows, List.mem_append] at member
  rcases member with prefixMember | geometryMember
  · exact FixedFrame.complete prefix_values_canonical honest.prefixPlaced one
      row prefixMember
  · exact FixedFrame.complete geometry_values_canonical honest.geometryPlaced one
      row geometryMember

/-! ## Profile-indexed successor frame -/

namespace ProfileIndexed

def geometryValues (profile : Profile.Identity) : List Nat :=
  [checkedStepCountFor profile, 63, 64]

def rows (profile : Profile.Identity) (layout : Layout) : List Row :=
  FixedFrame.rows layout.prefixColumns (fixedPrefixFor profile) ++
    FixedFrame.rows layout.geometryColumns (geometryValues profile)

theorem rows_length_exact (profile : Profile.Identity) (layout : Layout) :
    (rows profile layout).length = 9 := by
  simp [rows, FixedFrame.rows, ConstantPins.rows, FixedFrame.pins,
    Layout.prefixColumns, Layout.geometryColumns, fixedPrefixFor_length,
    geometryValues]

private theorem geometry_values_canonical
    {profile : Profile.Identity} (valid : ProfileCanonical profile) :
    ∀ value ∈ geometryValues profile, value < goldilocksP := by
  intro value member
  simp only [geometryValues, List.mem_cons, List.not_mem_nil, or_false]
    at member
  rcases member with rfl | rfl | rfl
  · exact checkedStepCountFor_lt valid
  all_goals decide

private theorem prefix_rows_hold
    {profile : Profile.Identity} {layout : Layout} {assignment : Nat → Nat}
    (holds : Satisfies (rows profile layout) assignment) :
    Satisfies
      (FixedFrame.rows layout.prefixColumns (fixedPrefixFor profile))
      assignment := by
  intro row member
  exact holds row (by simp [rows, member])

private theorem geometry_rows_hold
    {profile : Profile.Identity} {layout : Layout} {assignment : Nat → Nat}
    (holds : Satisfies (rows profile layout) assignment) :
    Satisfies
      (FixedFrame.rows layout.geometryColumns (geometryValues profile))
      assignment := by
  intro row member
  exact holds row (by simp [rows, member])

theorem prefix_exact
    {profile : Profile.Identity} {layout : Layout} {assignment : Nat → Nat}
    (profileCanonical : ProfileCanonical profile)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (holds : Satisfies (rows profile layout) assignment) :
    layout.prefixColumns.map assignment = fixedPrefixFor profile := by
  exact FixedFrame.sound
    (by simp [Layout.prefixColumns, fixedPrefixFor_length])
    (fixedPrefixFor_fields_canonical profileCanonical) canonical one
    (prefix_rows_hold holds)

theorem geometry_exact
    {profile : Profile.Identity} {layout : Layout} {assignment : Nat → Nat}
    (profileCanonical : ProfileCanonical profile)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (holds : Satisfies (rows profile layout) assignment) :
    layout.geometryColumns.map assignment = geometryValues profile := by
  exact FixedFrame.sound
    (by simp [Layout.geometryColumns, geometryValues])
    (geometry_values_canonical profileCanonical) canonical one
    (geometry_rows_hold holds)

/-- The same variable-column placement refines to the exact selected-profile
frame. Profile constants come only from the fixed rows. -/
theorem input_exact
    {profile : Profile.Identity} {layout : Layout} {assignment : Nat → Nat}
    {input : Input}
    (profileCanonical : ProfileCanonical profile)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (placed : VariablePlaced layout assignment input)
    (holds : Satisfies (rows profile layout) assignment) :
    layout.inputColumns.map assignment = encodeFor profile input := by
  rw [Layout.inputColumns]
  simp only [List.map_append]
  rw [prefix_exact profileCanonical canonical one holds, placed.authority,
    placed.counters, geometry_exact profileCanonical canonical one holds,
    placed.roots]
  simp [encodeFor, counterFieldsFor, geometryValues]

structure Honest (profile : Profile.Identity) (layout : Layout)
    (assignment : Nat → Nat) (input : Input) : Prop where
  variablePlaced : VariablePlaced layout assignment input
  prefixPlaced :
    layout.prefixColumns.map assignment = fixedPrefixFor profile
  geometryPlaced :
    layout.geometryColumns.map assignment = geometryValues profile

theorem rows_complete
    {profile : Profile.Identity} {layout : Layout} {assignment : Nat → Nat}
    {input : Input}
    (profileCanonical : ProfileCanonical profile)
    (one : assignment 0 = 1)
    (honest : Honest profile layout assignment input) :
    Satisfies (rows profile layout) assignment := by
  intro row member
  rw [rows, List.mem_append] at member
  rcases member with prefixMember | geometryMember
  · exact FixedFrame.complete
      (fixedPrefixFor_fields_canonical profileCanonical)
      honest.prefixPlaced one row prefixMember
  · exact FixedFrame.complete (geometry_values_canonical profileCanonical)
      honest.geometryPlaced one row geometryMember

end ProfileIndexed

end Nightstream.Implementation.Nebula.MemoryTranscriptHashFrameRows
