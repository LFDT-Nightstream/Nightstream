import Nightstream.Implementation.Nebula.Commitment.Compact.ChainHashFrame
import Nightstream.Implementation.Nebula.FPrime.State.SeedSchedule
import Nightstream.Implementation.R1CS.Core.ConstantPins

/-!
Contract: exact R1CS input frames for V2 compact-chain Poseidon2 calls.

Assurance tier: implementation model.

Owns constant pins for the header, leaf, and link prefixes; exact reuse of
the 54 token-output columns and four-lane digest columns; ordered input-column
lists; row soundness to `CompactChainHashFrame.encode`; and local honest
completeness.

Does not own token computation, Poseidon2 traces, chain-state transitions,
absolute generated columns, or Rust conformance.

Emits constraints: yes.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Nebula.CompactChainHashFrameRows

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.Nebula.CompactChainHashFrame
open Nightstream.Protocol.Nebula
open Nightstream.Protocol.Nebula.CompactCommit
open Nightstream.Protocol.Nebula.Lifecycle

namespace FixedFrame

def pins (columns values : List Nat) : List (Nat × Nat) :=
  columns.zip values

def rows (columns values : List Nat) : List Row :=
  ConstantPins.rows (pins columns values)

theorem pins_values_canonical
    {columns values : List Nat}
    (canonical : ∀ value ∈ values, value < goldilocksP) :
    ConstantPins.ValuesCanonical (pins columns values) := by
  intro pin member
  exact canonical pin.2 (List.of_mem_zip member).2

private theorem map_eq_of_zip_facts
    {assignment : Nat → Nat} :
    ∀ {columns values : List Nat},
      columns.length = values.length →
      (∀ pin ∈ columns.zip values, assignment pin.1 = pin.2) →
      columns.map assignment = values := by
  intro columns
  induction columns with
  | nil =>
      intro values lengths _facts
      cases values with
      | nil => rfl
      | cons _ _ => simp at lengths
  | cons column rest inductionHypothesis =>
      intro values lengths facts
      cases values with
      | nil => simp at lengths
      | cons value tail =>
          simp only [List.length_cons, Nat.succ.injEq] at lengths
          simp only [List.map_cons, List.cons.injEq]
          constructor
          · exact facts (column, value) (by simp)
          · apply inductionHypothesis lengths
            intro pin member
            exact facts pin (by simp [member])

theorem sound
    {columns values : List Nat} {assignment : Nat → Nat}
    (lengths : columns.length = values.length)
    (valuesCanonical : ∀ value ∈ values, value < goldilocksP)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (holds : Satisfies (rows columns values) assignment) :
    columns.map assignment = values := by
  apply map_eq_of_zip_facts lengths
  exact ConstantPins.sound (pins_values_canonical valuesCanonical)
    (by unfold rowsIncluded; simp [rows, pins]) canonical one holds

private theorem zip_facts_of_map_eq
    {assignment : Nat → Nat} :
    ∀ {columns values : List Nat},
      columns.map assignment = values →
      ∀ pin ∈ columns.zip values, assignment pin.1 = pin.2 := by
  intro columns
  induction columns with
  | nil =>
      intro values equal pin member
      simp only [List.map_nil] at equal
      subst values
      simp at member
  | cons column rest inductionHypothesis =>
      intro values equal pin member
      cases values with
      | nil => simp at equal
      | cons value tail =>
          simp only [List.map_cons, List.cons.injEq] at equal
          simp only [List.zip_cons_cons, List.mem_cons] at member
          rcases member with rfl | tailMember
          · exact equal.1
          · exact inductionHypothesis equal.2 pin tailMember

theorem complete
    {columns values : List Nat} {assignment : Nat → Nat}
    (valuesCanonical : ∀ value ∈ values, value < goldilocksP)
    (placed : columns.map assignment = values)
    (one : assignment 0 = 1) :
    Satisfies (rows columns values) assignment := by
  apply ConstantPins.complete (pins_values_canonical valuesCanonical) one
  exact zip_facts_of_map_eq placed

end FixedFrame

def headerValues (manifest : SeedSchedule.Manifest) (role : Role) : List Nat :=
  encode (.header role manifest.profile manifest.plan)

def leafPrefixValues
    (manifest : SeedSchedule.Manifest) (role : Role) : List Nat :=
  [leafTag role, frameVersion] ++
    (profileFields manifest.profile ++ digestFields manifest.plan)

def linkPrefixValues (role : Role) : List Nat :=
  [linkTag role, frameVersion]

theorem headerValues_length
    (manifest : SeedSchedule.Manifest) (role : Role) :
    (headerValues manifest role).length = 11 :=
  header_length role manifest.profile manifest.plan

theorem leafPrefixValues_length
    (manifest : SeedSchedule.Manifest) (role : Role) :
    (leafPrefixValues manifest role).length = 10 := by
  simp [leafPrefixValues, profileFields, digestFields_length]

theorem linkPrefixValues_length (role : Role) :
    (linkPrefixValues role).length = 2 := by
  simp [linkPrefixValues]

theorem headerValues_canonical
    (manifest : SeedSchedule.Manifest) (role : Role) :
    ∀ value ∈ headerValues manifest role, value < goldilocksP :=
  encode_fields_canonical (.header role manifest.profile manifest.plan)
    manifest.profileSupported

theorem leafPrefixValues_canonical
    (manifest : SeedSchedule.Manifest) (role : Role) :
    ∀ value ∈ leafPrefixValues manifest role, value < goldilocksP := by
  intro value member
  simp only [leafPrefixValues, List.mem_append] at member
  rcases member with tagOrVersion | rest
  · simp only [List.mem_cons, List.not_mem_nil, or_false] at tagOrVersion
    rcases tagOrVersion with rfl | rfl
    · cases role <;> decide
    · decide
  · rcases rest with profile | digest
    · exact profileFields_canonical manifest.profileSupported value profile
    · rcases List.mem_ofFn.mp digest with ⟨lane, equal⟩
      rw [← equal]
      exact (manifest.plan.lanes lane).property

theorem linkPrefixValues_canonical (role : Role) :
    ∀ value ∈ linkPrefixValues role, value < goldilocksP := by
  intro value member
  simp only [linkPrefixValues, List.mem_cons, List.not_mem_nil,
    or_false] at member
  rcases member with rfl | rfl
  · cases role <;> decide
  · decide

structure HeaderLayout where
  inputColumn : Fin 11 → Nat

def HeaderLayout.inputColumns (layout : HeaderLayout) : List Nat :=
  List.ofFn layout.inputColumn

def headerRows (manifest : SeedSchedule.Manifest) (role : Role)
    (layout : HeaderLayout) : List Row :=
  FixedFrame.rows layout.inputColumns (headerValues manifest role)

theorem headerRows_length
    (manifest : SeedSchedule.Manifest) (role : Role)
    (layout : HeaderLayout) :
    (headerRows manifest role layout).length = 11 := by
  rw [headerRows, FixedFrame.rows, ConstantPins.rows]
  simp [FixedFrame.pins, HeaderLayout.inputColumns, headerValues_length]

theorem header_input_exact
    {manifest : SeedSchedule.Manifest} {role : Role}
    {layout : HeaderLayout} {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (holds : Satisfies (headerRows manifest role layout) assignment) :
    layout.inputColumns.map assignment =
      encode (.header role manifest.profile manifest.plan) := by
  exact FixedFrame.sound (by
      rw [HeaderLayout.inputColumns, List.length_ofFn]
      exact (header_length role manifest.profile manifest.plan).symm)
    (headerValues_canonical manifest role) canonical one holds

structure LeafLayout where
  prefixColumn : Fin 10 → Nat
  tokenColumn : Fin tokenFieldCount → Nat

def LeafLayout.prefixColumns (layout : LeafLayout) : List Nat :=
  List.ofFn layout.prefixColumn

def LeafLayout.tokenColumns (layout : LeafLayout) : List Nat :=
  List.ofFn layout.tokenColumn

def LeafLayout.inputColumns (layout : LeafLayout) : List Nat :=
  layout.prefixColumns ++ layout.tokenColumns

def leafRows (manifest : SeedSchedule.Manifest) (role : Role)
    (layout : LeafLayout) : List Row :=
  FixedFrame.rows layout.prefixColumns (leafPrefixValues manifest role)

theorem leafRows_length
    (manifest : SeedSchedule.Manifest) (role : Role)
    (layout : LeafLayout) :
    (leafRows manifest role layout).length = 10 := by
  rw [leafRows, FixedFrame.rows, ConstantPins.rows]
  simp [FixedFrame.pins, LeafLayout.prefixColumns,
    leafPrefixValues_length]

def TokenPlaced (layout : LeafLayout) (assignment : Nat → Nat)
    (token : Token) : Prop :=
  ∀ coordinate,
    assignment (layout.tokenColumn coordinate) = (token coordinate).val

theorem leaf_input_exact
    {manifest : SeedSchedule.Manifest} {role : Role}
    {layout : LeafLayout} {assignment : Nat → Nat} {token : Token}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (placed : TokenPlaced layout assignment token)
    (holds : Satisfies (leafRows manifest role layout) assignment) :
    layout.inputColumns.map assignment =
      encode (.leaf role manifest.profile manifest.plan token) := by
  have prefixExact := FixedFrame.sound
    (columns := layout.prefixColumns)
    (values := leafPrefixValues manifest role)
    (by simp [LeafLayout.prefixColumns, leafPrefixValues_length])
    (leafPrefixValues_canonical manifest role) canonical one holds
  have tokenExact : layout.tokenColumns.map assignment = tokenFields token := by
    simp only [LeafLayout.tokenColumns, tokenFields, List.map_ofFn]
    apply List.ext_getElem
    · simp [tokenFieldCount, shortRank, ringDegree]
    · intro coordinate leftBound rightBound
      simp only [List.getElem_ofFn]
      exact placed ⟨coordinate, by simpa using leftBound⟩
  rw [LeafLayout.inputColumns, List.map_append, prefixExact, tokenExact]
  rfl

structure LinkLayout where
  prefixColumn : Fin 2 → Nat
  indexColumn : Nat
  priorDigestColumn : Fin 4 → Nat
  leafDigestColumn : Fin 4 → Nat

def LinkLayout.prefixColumns (layout : LinkLayout) : List Nat :=
  List.ofFn layout.prefixColumn

def LinkLayout.priorDigestColumns (layout : LinkLayout) : List Nat :=
  List.ofFn layout.priorDigestColumn

def LinkLayout.leafDigestColumns (layout : LinkLayout) : List Nat :=
  List.ofFn layout.leafDigestColumn

def LinkLayout.inputColumns (layout : LinkLayout) : List Nat :=
  layout.prefixColumns ++ [layout.indexColumn] ++
    layout.priorDigestColumns ++ layout.leafDigestColumns

def linkRows (role : Role) (layout : LinkLayout) : List Row :=
  FixedFrame.rows layout.prefixColumns (linkPrefixValues role)

theorem linkRows_length
    (role : Role) (layout : LinkLayout) :
    (linkRows role layout).length = 2 := by
  rw [linkRows, FixedFrame.rows, ConstantPins.rows]
  simp [FixedFrame.pins, LinkLayout.prefixColumns,
    linkPrefixValues_length]

def DigestPlaced (columns : Fin 4 → Nat) (assignment : Nat → Nat)
    (digest : Digest.Value) : Prop :=
  ∀ lane, assignment (columns lane) = (digest.lanes lane).val

theorem link_input_exact
    {role : Role} {index : Fin claimsPerSegment}
    {layout : LinkLayout} {assignment : Nat → Nat}
    {prior leaf : Digest.Value}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (indexPlaced : assignment layout.indexColumn = index.val)
    (priorPlaced : DigestPlaced layout.priorDigestColumn assignment prior)
    (leafPlaced : DigestPlaced layout.leafDigestColumn assignment leaf)
    (holds : Satisfies (linkRows role layout) assignment) :
    layout.inputColumns.map assignment =
      encode (.link role index prior leaf) := by
  have prefixExact := FixedFrame.sound
    (columns := layout.prefixColumns) (values := linkPrefixValues role)
    (by simp [LinkLayout.prefixColumns, linkPrefixValues_length])
    (linkPrefixValues_canonical role) canonical one holds
  have priorExact :
      layout.priorDigestColumns.map assignment = digestFields prior := by
    simp only [LinkLayout.priorDigestColumns, digestFields, List.map_ofFn]
    apply List.ext_getElem
    · simp [Digest.laneCount]
    · intro lane leftBound rightBound
      simp only [List.getElem_ofFn]
      exact priorPlaced ⟨lane, by simpa using leftBound⟩
  have leafExact :
      layout.leafDigestColumns.map assignment = digestFields leaf := by
    simp only [LinkLayout.leafDigestColumns, digestFields, List.map_ofFn]
    apply List.ext_getElem
    · simp [Digest.laneCount]
    · intro lane leftBound rightBound
      simp only [List.getElem_ofFn]
      exact leafPlaced ⟨lane, by simpa using leftBound⟩
  rw [LinkLayout.inputColumns, List.map_append, List.map_append,
    List.map_append, prefixExact, priorExact, leafExact]
  simp [indexPlaced, linkPrefixValues, encode]

structure HeaderHonest
    (manifest : SeedSchedule.Manifest) (role : Role)
    (layout : HeaderLayout) (assignment : Nat → Nat) : Prop where
  placed : layout.inputColumns.map assignment = headerValues manifest role

theorem headerRows_complete
    {manifest : SeedSchedule.Manifest} {role : Role}
    {layout : HeaderLayout} {assignment : Nat → Nat}
    (one : assignment 0 = 1)
    (honest : HeaderHonest manifest role layout assignment) :
    Satisfies (headerRows manifest role layout) assignment :=
  FixedFrame.complete (headerValues_canonical manifest role)
    honest.placed one

structure LeafHonest
    (manifest : SeedSchedule.Manifest) (role : Role)
    (layout : LeafLayout) (assignment : Nat → Nat) : Prop where
  prefixPlaced :
    layout.prefixColumns.map assignment = leafPrefixValues manifest role

theorem leafRows_complete
    {manifest : SeedSchedule.Manifest} {role : Role}
    {layout : LeafLayout} {assignment : Nat → Nat}
    (one : assignment 0 = 1)
    (honest : LeafHonest manifest role layout assignment) :
    Satisfies (leafRows manifest role layout) assignment :=
  FixedFrame.complete (leafPrefixValues_canonical manifest role)
    honest.prefixPlaced one

structure LinkHonest
    (role : Role)
    (layout : LinkLayout) (assignment : Nat → Nat) : Prop where
  prefixPlaced :
    layout.prefixColumns.map assignment = linkPrefixValues role

theorem linkRows_complete
    {role : Role}
    {layout : LinkLayout} {assignment : Nat → Nat}
    (one : assignment 0 = 1)
    (honest : LinkHonest role layout assignment) :
    Satisfies (linkRows role layout) assignment :=
  FixedFrame.complete (linkPrefixValues_canonical role)
    honest.prefixPlaced one

end Nightstream.Implementation.Nebula.CompactChainHashFrameRows
