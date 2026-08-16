import Mathlib.Tactic
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyPublicSchema

/-!
Contract: structural support for bounded PiRLC public-family artifact
certificates.

Owns generic list splitting, interval expansion, and glue-row geometry
soundness. It owns no generated data or protocol semantics.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPublicCertificateSupport

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplay.Artifact
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPublic.Artifact

theorem valid_of_take_drop
    {α : Type} {property : α → Prop} {items : List α} {count : Nat}
    (head : ∀ item ∈ items.take count, property item)
    (tail : ∀ item ∈ items.drop count, property item) :
    ∀ item ∈ items, property item := by
  intro item member
  rw [← List.take_append_drop count items] at member
  rcases List.mem_append.mp member with member | member
  · exact head item member
  · exact tail item member

theorem length_of_take_drop
    {α : Type} {items : List α} {count headLength tailLength : Nat}
    (head : (items.take count).length = headLength)
    (tail : (items.drop count).length = tailLength) :
    items.length = headLength + tailLength := by
  have split := congrArg List.length (List.take_append_drop count items)
  simpa only [List.length_append, head, tail] using split.symm

structure Segment where
  start : Nat
  length : Nat
deriving DecidableEq, Repr

namespace Segment

def columns (segment : Segment) : List Nat :=
  List.range' segment.start segment.length

def Disjoint (left right : Segment) : Prop :=
  left.start + left.length ≤ right.start ∨
    right.start + right.length ≤ left.start

theorem columns_nodup (segment : Segment) : segment.columns.Nodup :=
  List.nodup_range'

theorem member_bounds
    {segment : Segment} {column : Nat}
    (member : column ∈ segment.columns) :
    segment.start ≤ column ∧ column < segment.start + segment.length := by
  simpa [columns] using List.mem_range'_1.mp member

theorem columns_disjoint
    {left right : Segment} (disjoint : left.Disjoint right) :
    List.Disjoint left.columns right.columns := by
  rw [List.disjoint_iff_ne]
  intro leftColumn leftMember rightColumn rightMember equal
  have leftBounds := member_bounds leftMember
  have rightBounds := member_bounds rightMember
  rcases disjoint with disjoint | disjoint <;> omega

end Segment

def expandSegments : List Segment → List Nat
  | [] => []
  | segment :: rest => segment.columns ++ expandSegments rest

theorem mem_expandSegments {column : Nat} {segments : List Segment} :
    column ∈ expandSegments segments ↔
      ∃ segment ∈ segments, column ∈ segment.columns := by
  induction segments with
  | nil => simp [expandSegments]
  | cons segment rest inductionHypothesis =>
      simp [expandSegments, inductionHypothesis]

def SegmentsValid (columnCount : Nat) (segments : List Segment) : Prop :=
  (∀ segment ∈ segments,
      segment.start + segment.length ≤ columnCount) ∧
    segments.Pairwise Segment.Disjoint

private theorem head_disjoint_expand
    {head : Segment} {rest : List Segment}
    (disjoint : ∀ segment ∈ rest, head.Disjoint segment) :
    List.Disjoint head.columns (expandSegments rest) := by
  rw [List.disjoint_iff_ne]
  intro left leftMember right rightMember equal
  rcases mem_expandSegments.mp rightMember with
    ⟨segment, segmentMember, rightInSegment⟩
  have leftBounds := Segment.member_bounds leftMember
  have rightBounds := Segment.member_bounds rightInSegment
  rcases disjoint segment segmentMember with before | after <;> omega

theorem expandSegments_nodup
    {columnCount : Nat} {segments : List Segment}
    (valid : SegmentsValid columnCount segments) :
    (expandSegments segments).Nodup := by
  induction segments with
  | nil => simp [expandSegments]
  | cons head rest inductionHypothesis =>
      have pairwise := List.pairwise_cons.mp valid.2
      rw [expandSegments]
      apply List.Nodup.append
      · exact head.columns_nodup
      · apply inductionHypothesis
        exact ⟨by
          intro segment member
          exact valid.1 segment (List.mem_cons_of_mem head member),
          pairwise.2⟩
      · exact head_disjoint_expand pairwise.1

theorem expandSegments_bound
    {columnCount : Nat} {segments : List Segment}
    (valid : SegmentsValid columnCount segments)
    {column : Nat} (member : column ∈ expandSegments segments) :
    column < columnCount := by
  rcases mem_expandSegments.mp member with
    ⟨segment, segmentMember, columnMember⟩
  have bounds := Segment.member_bounds columnMember
  have segmentBound := valid.1 segment segmentMember
  omega

theorem expandSegments_length (segments : List Segment) :
    (expandSegments segments).length =
      (segments.map Segment.length).sum := by
  induction segments with
  | nil => rfl
  | cons segment rest inductionHypothesis =>
      simp [expandSegments, Segment.columns, inductionHypothesis]

theorem columnsValid_of_segments
    {columnCount expectedLength : Nat} {columns : List Nat}
    {segments : List Segment}
    (exact : columns = expandSegments segments)
    (length : (segments.map Segment.length).sum = expectedLength)
    (valid : SegmentsValid columnCount segments) :
    columnsValid columnCount expectedLength columns := by
  rw [exact]
  exact ⟨by rw [expandSegments_length, length],
    expandSegments_nodup valid, by
      intro column member
      exact expandSegments_bound valid member⟩

def termsBelowCheck (columnCount : Nat) (terms : List (Nat × Nat)) : Bool :=
  terms.all fun term => decide (term.1 < columnCount)

theorem termsBelowCheck_sound
    {columnCount : Nat} {terms : List (Nat × Nat)}
    (checked : termsBelowCheck columnCount terms = true) :
    ∀ term ∈ terms, term.1 < columnCount := by
  intro term member
  exact of_decide_eq_true ((List.all_eq_true.mp checked) term member)

def rowColumnsBelowCheck (columnCount : Nat) (row : Row) : Bool :=
  termsBelowCheck columnCount row.a &&
    (termsBelowCheck columnCount row.b && termsBelowCheck columnCount row.c)

theorem rowColumnsBelowCheck_sound
    {columnCount : Nat} {row : Row}
    (checked : rowColumnsBelowCheck columnCount row = true) :
    rowColumnsBelow columnCount row := by
  simp only [rowColumnsBelowCheck, Bool.and_eq_true] at checked
  exact ⟨termsBelowCheck_sound checked.1,
    termsBelowCheck_sound checked.2.1,
    termsBelowCheck_sound checked.2.2⟩

def glueRowGeometryCheck
    (sourceRowCount rowCount columnCount : Nat)
    (indexed : IndexedRow) : Bool :=
  decide (sourceRowCount ≤ indexed.index) &&
    (decide (indexed.index < rowCount) &&
      rowColumnsBelowCheck columnCount indexed.row)

def glueRowsGeometryCheck
    (sourceRowCount rowCount columnCount : Nat)
    (rows : List IndexedRow) : Bool :=
  rows.all (glueRowGeometryCheck sourceRowCount rowCount columnCount)

theorem glueRowsGeometryCheck_sound
    {sourceRowCount rowCount columnCount : Nat} {rows : List IndexedRow}
    (checked :
      glueRowsGeometryCheck sourceRowCount rowCount columnCount rows = true) :
    ∀ indexed ∈ rows,
      sourceRowCount ≤ indexed.index ∧ indexed.index < rowCount ∧
        rowColumnsBelow columnCount indexed.row := by
  intro indexed member
  have rowChecked := (List.all_eq_true.mp checked) indexed member
  simp only [glueRowGeometryCheck, Bool.and_eq_true] at rowChecked
  exact ⟨of_decide_eq_true rowChecked.1,
    of_decide_eq_true rowChecked.2.1,
    rowColumnsBelowCheck_sound rowChecked.2.2⟩

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPublicCertificateSupport
