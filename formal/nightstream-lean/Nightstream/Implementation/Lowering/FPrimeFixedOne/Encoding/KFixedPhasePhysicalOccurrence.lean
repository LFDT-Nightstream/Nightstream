import Nightstream.Implementation.Lowering.Goldilocks.NumericRowBridge
import Nightstream.Implementation.R1CS.Canonical.KFixedPhaseSumCheckSupport

/-!
Contract: give one Lean-owned fixed-phase SumCheck chain an exact typed
physical placement.

Owns:
- a stable translation of pre-existing numeric source columns;
- one contiguous auxiliary interval for every Horner frame;
- contiguous, duplicate-free row identities;
- exact row and auxiliary receipts;
- complete support and source/auxiliary separation;
- typed/numeric satisfaction equivalence; and
- a typed honest witness constructed from the frozen fixed-phase chain.

Does not own transcript generation, the protocol-specific initial or terminal
expressions, the enclosing `nifsVerify` call, or any generated/Rust layout.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.KFixedPhasePhysicalOccurrence

open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.Goldilocks.NumericRowBridge
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal
open Nightstream.Implementation.R1CS.Canonical.KMul
open Nightstream.Implementation.R1CS.Canonical.KFixedPhaseSumCheck
open Nightstream.Implementation.R1CS.Canonical.KFixedPhaseSumCheckHonest
open Nightstream.Implementation.R1CS.Canonical.KFixedPhaseSumCheckSupport
open Nightstream.SuperNeo.SumCheck.Finite

private def constantId : ColumnId where
  owner := .prelude
  bundleIndex := 0
  coordinateIndex := 0

/-- One pre-existing numeric source coordinate. -/
def sourceId (owner : PhysicalOwner) (source : Nat) : ColumnId where
  owner := owner
  bundleIndex := 0
  coordinateIndex := source

/-- One owner-local auxiliary coordinate. -/
def auxiliaryId (owner : PhysicalOwner) (offset : Nat) : ColumnId where
  owner := owner
  bundleIndex := 1
  coordinateIndex := offset

/-- Stable numeric-to-typed placement.  Numeric zero remains the global
constant; the selected interval is fresh; all other columns are reads. -/
def columnMap
    (sourceOwner occurrenceOwner : PhysicalOwner)
    (base width source : Nat) : ColumnId :=
  if source = 0 then
    constantId
  else if base ≤ source ∧ source < base + width then
    auxiliaryId occurrenceOwner (source - base)
  else
    sourceId sourceOwner source

def auxiliaryColumns
    (owner : PhysicalOwner) (width : Nat) : List OwnedColumn :=
  (List.range width).map fun offset =>
    { id := auxiliaryId owner offset
      ownership := .auxiliaryColumn }

private def numericColumns (row : Numeric.Row) : List Nat :=
  (row.a ++ row.b ++ row.c).map Prod.fst

private def numericSupport (rows : List Numeric.Row) : List Nat :=
  rows.flatMap numericColumns

private def visibleSourceColumns
    (base width : Nat) (rows : List Numeric.Row) : List Nat :=
  (numericSupport rows).filter fun source =>
    decide (source ≠ 0 ∧ ¬ (base ≤ source ∧ source < base + width))

private def visibleIds
    (sourceOwner : PhysicalOwner) (base width : Nat)
    (rows : List Numeric.Row) : List ColumnId :=
  (visibleSourceColumns base width rows).map (sourceId sourceOwner)

/-- The complete physical data for one fixed-phase chain.  Every source
combination is proved to precede `base`; the compiler alone owns the following
Horner interval. -/
structure PhysicalOccurrence
    {degree : Nat}
    (current : Carried)
    (rounds : List (Round degree))
    (challenges : List Carried)
    (terminal : Carried)
    (base : Nat) where
  sourceOwner : PhysicalOwner
  owner : PhysicalOwner
  firstOrdinal : Nat
  basePositive : 0 < base
  currentBelow : CarriedBelow current base
  roundsBelow : ∀ round ∈ rounds, RoundBelow round base
  challengesBelow :
    ∀ challenge ∈ challenges, CarriedBelow challenge base
  terminalBelow : CarriedBelow terminal base
  sameLength : rounds.length = challenges.length

def PhysicalOccurrence.numericRows
    {degree current rounds challenges terminal base}
    (_physical :
      PhysicalOccurrence (degree := degree)
        current rounds challenges terminal base) :
    List Numeric.Row :=
  chainRows current rounds challenges terminal base

def PhysicalOccurrence.auxWidth
    {degree current rounds challenges terminal base}
    (_physical :
      PhysicalOccurrence (degree := degree)
        current rounds challenges terminal base) : Nat :=
  rounds.length * (3 * degree)

def PhysicalOccurrence.map
    {degree current rounds challenges terminal base}
    (physical :
      PhysicalOccurrence (degree := degree)
        current rounds challenges terminal base) : Nat → ColumnId :=
  columnMap physical.sourceOwner physical.owner base physical.auxWidth

def PhysicalOccurrence.rows
    {degree current rounds challenges terminal base}
    (physical :
      PhysicalOccurrence (degree := degree)
        current rounds challenges terminal base) : List OwnedRow :=
  ownedRowsFrom physical.owner physical.firstOrdinal physical.map
    physical.numericRows

def PhysicalOccurrence.auxiliaries
    {degree current rounds challenges terminal base}
    (physical :
      PhysicalOccurrence (degree := degree)
        current rounds challenges terminal base) : List OwnedColumn :=
  auxiliaryColumns physical.owner physical.auxWidth

def PhysicalOccurrence.visible
    {degree current rounds challenges terminal base}
    (physical :
      PhysicalOccurrence (degree := degree)
        current rounds challenges terminal base) : List ColumnId :=
  constantId ::
    visibleIds physical.sourceOwner base physical.auxWidth physical.numericRows

@[simp] theorem auxiliaryColumns_length
    (owner : PhysicalOwner) (width : Nat) :
    (auxiliaryColumns owner width).length = width := by
  simp [auxiliaryColumns]

theorem auxiliaryColumns_ids_nodup
    (owner : PhysicalOwner) (width : Nat) :
    ((auxiliaryColumns owner width).map (fun column => column.id)).Nodup := by
  simp only [auxiliaryColumns, List.map_map]
  exact List.nodup_range.map
    (fun offset => auxiliaryId owner offset) (by
      intro left right different equal
      apply different
      exact congrArg ColumnId.coordinateIndex equal)

theorem rows_length
    {degree current rounds challenges terminal base}
    (physical :
      PhysicalOccurrence (degree := degree)
        current rounds challenges terminal base) :
    physical.rows.length =
      rounds.length * (3 * degree + 2) + 2 := by
  rw [PhysicalOccurrence.rows, ownedRowsFrom_length]
  exact chainRows_length current rounds challenges terminal base
    physical.sameLength

theorem rows_cost
    {degree current rounds challenges terminal base}
    (physical :
      PhysicalOccurrence (degree := degree)
        current rounds challenges terminal base) :
    physical.rows.length =
      (chainCost degree rounds.length).recurringRows :=
  rows_length physical

theorem auxiliaries_length
    {degree current rounds challenges terminal base}
    (physical :
      PhysicalOccurrence (degree := degree)
        current rounds challenges terminal base) :
    physical.auxiliaries.length =
      (chainCost degree rounds.length).auxiliaryColumns := by
  simp [PhysicalOccurrence.auxiliaries, PhysicalOccurrence.auxWidth,
    chainCost]

theorem rows_owned
    {degree current rounds challenges terminal base}
    (physical :
      PhysicalOccurrence (degree := degree)
        current rounds challenges terminal base)
    (row : OwnedRow) (member : row ∈ physical.rows) :
    row.id.owner = physical.owner :=
  ownedRowsFrom_owned physical.owner physical.firstOrdinal physical.map
    physical.numericRows row member

theorem row_ids_nodup
    {degree current rounds challenges terminal base}
    (physical :
      PhysicalOccurrence (degree := degree)
        current rounds challenges terminal base) :
    (physical.rows.map fun row => row.id).Nodup :=
  ownedRowsFrom_ids_nodup physical.owner physical.firstOrdinal physical.map
    physical.numericRows

theorem auxiliary_owned
    {degree current rounds challenges terminal base}
    (physical :
      PhysicalOccurrence (degree := degree)
        current rounds challenges terminal base)
    (column : OwnedColumn) (member : column ∈ physical.auxiliaries) :
    column.id.owner = physical.owner := by
  rcases List.mem_map.mp member with ⟨offset, _, rfl⟩
  rfl

/-- The compiler never reaches beyond its declared source-and-auxiliary
boundary.  This prevents a misplaced compiler column from being silently
reclassified as a visible read. -/
theorem numeric_support_below_end
    {degree current rounds challenges terminal base}
    (physical :
      PhysicalOccurrence (degree := degree)
        current rounds challenges terminal base)
    (row : Numeric.Row) (rowMember : row ∈ physical.numericRows)
    (column : Nat) (columnMember : column ∈ numericColumns row) :
    column < base + physical.auxWidth := by
  unfold numericColumns at columnMember
  rcases List.mem_map.mp columnMember with ⟨term, termMember, rfl⟩
  apply chainRows_columns_below_end current rounds challenges terminal base
    physical.basePositive physical.currentBelow physical.roundsBelow
    physical.challengesBelow physical.terminalBelow physical.sameLength
    row rowMember term.1
  simp only [List.mem_append] at termMember
  rcases termMember with (inA | inB) | inC
  · exact Or.inl (List.mem_map.mpr ⟨term, inA, rfl⟩)
  · exact Or.inr (Or.inl (List.mem_map.mpr ⟨term, inB, rfl⟩))
  · exact Or.inr (Or.inr (List.mem_map.mpr ⟨term, inC, rfl⟩))

/-- Every translated dependency is either the constant wire, an explicit
pre-existing read, or an occurrence-owned auxiliary. -/
theorem rows_supported
    {degree current rounds challenges terminal base}
    (physical :
      PhysicalOccurrence (degree := degree)
        current rounds challenges terminal base)
    (owned : OwnedRow) (ownedMember : owned ∈ physical.rows)
    (column : ColumnId) (columnMember : column ∈ owned.columnIds) :
    column ∈ physical.visible ++
      physical.auxiliaries.map (fun allocated => allocated.id) := by
  unfold PhysicalOccurrence.rows at ownedMember
  have mappedMember :
      owned.row ∈
        (ownedRowsFrom physical.owner physical.firstOrdinal physical.map
          physical.numericRows).map (fun row => row.row) :=
    List.mem_map.mpr ⟨owned, ownedMember, rfl⟩
  rw [ownedRowsFrom_rows] at mappedMember
  rcases List.mem_map.mp mappedMember with
    ⟨sourceRow, sourceRowMember, rowEqual⟩
  change column ∈ owned.row.columnIds at columnMember
  rw [← rowEqual, NumericRowBridge.row_columnIds] at columnMember
  rcases List.mem_map.mp columnMember with ⟨source, sourceMember, rfl⟩
  have inSupport : source.1 ∈ numericSupport physical.numericRows := by
    unfold numericSupport
    apply List.mem_flatMap.mpr
    exact ⟨sourceRow, sourceRowMember, by
      unfold numericColumns
      exact List.mem_map.mpr ⟨source, sourceMember, rfl⟩⟩
  by_cases zero : source.1 = 0
  · rw [PhysicalOccurrence.map, columnMap, if_pos zero]
    exact List.mem_append_left _ (List.mem_cons_self)
  by_cases allocated :
      base ≤ source.1 ∧ source.1 < base + physical.auxWidth
  · rw [PhysicalOccurrence.map, columnMap, if_neg zero, if_pos allocated]
    apply List.mem_append_right
    unfold PhysicalOccurrence.auxiliaries auxiliaryColumns
    simp only [List.map_map]
    apply List.mem_map.mpr
    refine ⟨source.1 - base, List.mem_range.mpr ?_, rfl⟩
    omega
  · rw [PhysicalOccurrence.map, columnMap, if_neg zero, if_neg allocated]
    apply List.mem_append_left
    simp only [PhysicalOccurrence.visible, List.mem_cons]
    right
    unfold visibleIds visibleSourceColumns
    apply List.mem_map.mpr
    refine ⟨source.1, ?_, rfl⟩
    apply List.mem_filter.mpr
    exact ⟨inSupport, by simp [zero, allocated]⟩

/-- Typed satisfaction is exactly numeric satisfaction on canonical field
representatives. -/
theorem satisfies_iff
    {degree current rounds challenges terminal base}
    (physical :
      PhysicalOccurrence (degree := degree)
        current rounds challenges terminal base)
    (assignment : ColumnId → Nightstream.SuperNeo.Concrete.F) :
    Satisfies physical.rows assignment ↔
      Nightstream.Implementation.R1CS.Satisfies physical.numericRows
        (numericAssignment physical.map assignment) :=
  ownedRowsFrom_satisfies_iff physical.owner physical.firstOrdinal
    physical.map physical.numericRows assignment

theorem PhysicalOccurrence.rows_sound
    {degree current rounds challenges terminal base}
    (physical :
      PhysicalOccurrence (degree := degree)
        current rounds challenges terminal base)
    (assignment : ColumnId → Nightstream.SuperNeo.Concrete.F)
    (constantWire : assignment constantId = 1)
    (satisfied : Satisfies physical.rows assignment) :
    FixedPhase.Chain sumCheckOps
      (decodeCarried (numericAssignment physical.map assignment) current)
      (rounds.map fun round =>
        round.polynomial (numericAssignment physical.map assignment))
      (challenges.map
        (decodeCarried (numericAssignment physical.map assignment)))
      (decodeCarried (numericAssignment physical.map assignment) terminal) := by
  apply chainRows_sound (numericAssignment physical.map assignment)
  · change (assignment (physical.map 0)).val = 1
    rw [PhysicalOccurrence.map, columnMap, if_pos rfl, constantWire]
    rfl
  · exact (satisfies_iff physical assignment).1 satisfied

/-- Lift a numeric witness into the exact typed placement. -/
def PhysicalOccurrence.liftAssignment
    {degree current rounds challenges terminal base}
    (physical :
      PhysicalOccurrence (degree := degree)
        current rounds challenges terminal base)
    (witness : Nat → Nat) :
    ColumnId → Nightstream.SuperNeo.Concrete.F :=
  fun column =>
    if column = constantId then
      residue (witness 0)
    else if
        column.owner = physical.owner ∧
          column.bundleIndex = 1 ∧
          column.coordinateIndex < physical.auxWidth then
      residue (witness (base + column.coordinateIndex))
    else if
        column.owner = physical.sourceOwner ∧
          column.bundleIndex = 0 then
      residue (witness column.coordinateIndex)
    else
      0

theorem PhysicalOccurrence.liftAssignment_map
    {degree current rounds challenges terminal base}
    (physical :
      PhysicalOccurrence (degree := degree)
        current rounds challenges terminal base)
    (witness : Nat → Nat) (source : Nat) :
    physical.liftAssignment witness (physical.map source) =
      residue (witness source) := by
  by_cases zero : source = 0
  · subst source
    simp [PhysicalOccurrence.liftAssignment, PhysicalOccurrence.map,
      columnMap, constantId]
  by_cases allocated :
      base ≤ source ∧ source < base + physical.auxWidth
  · have offsetLow : source - base < physical.auxWidth := by omega
    have recover : base + (source - base) = source := by omega
    simp [PhysicalOccurrence.liftAssignment, PhysicalOccurrence.map,
      columnMap, zero, allocated, constantId, auxiliaryId, offsetLow,
      recover]
  · simp [PhysicalOccurrence.liftAssignment, PhysicalOccurrence.map,
      columnMap, zero, allocated, constantId, sourceId]

theorem PhysicalOccurrence.numericAssignment_liftAssignment
    {degree current rounds challenges terminal base}
    (physical :
      PhysicalOccurrence (degree := degree)
        current rounds challenges terminal base)
    (witness : Nat → Nat) :
    numericAssignment physical.map (physical.liftAssignment witness) =
      canonicalAssignment witness := by
  funext source
  unfold numericAssignment canonicalAssignment
  rw [physical.liftAssignment_map]
  rfl

/-- Honest completeness of the physically placed chain.  The semantic input
is exactly the frozen `FixedPhase.Chain`; row equations are constructed. -/
theorem PhysicalOccurrence.rows_honest
    {degree current rounds challenges terminal base}
    (physical :
      PhysicalOccurrence (degree := degree)
        current rounds challenges terminal base)
    (source : Nat → Nat)
    (constantWire : source 0 = 1)
    (chain :
      FixedPhase.Chain sumCheckOps
        (decodeCarried source current)
        (rounds.map fun round => round.polynomial source)
        (challenges.map (decodeCarried source))
        (decodeCarried source terminal)) :
    ∃ assignment : ColumnId → Nightstream.SuperNeo.Concrete.F,
      Satisfies physical.rows assignment := by
  let witness := chainWitness source rounds challenges base
  have numericSatisfied :
      Nightstream.Implementation.R1CS.Satisfies
        physical.numericRows witness := by
    exact chainWitness_satisfies source constantWire current rounds challenges
      terminal base physical.basePositive physical.currentBelow
      physical.roundsBelow physical.challengesBelow physical.terminalBelow
      chain
  have canonicalSatisfied :
      Nightstream.Implementation.R1CS.Satisfies
        physical.numericRows (canonicalAssignment witness) :=
    satisfies_canonical physical.numericRows witness numericSatisfied
  refine ⟨physical.liftAssignment witness, ?_⟩
  apply (satisfies_iff physical (physical.liftAssignment witness)).2
  rw [physical.numericAssignment_liftAssignment]
  exact canonicalSatisfied

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.KFixedPhasePhysicalOccurrence
