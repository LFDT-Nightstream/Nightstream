import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ActivatedRawProgram
import Nightstream.Implementation.Lowering.Goldilocks.NativeCcsPhi81

/-!
Contract: exact terminal R1CS lowering of the Lean-owned native four-matrix
CCS relation.

Assurance tier: model-level.

Owns: relocation of each logical native-CCS column into one complete-carrier
witness, two R1CS rows and one residual column per selected CCS row, exact
support, ownership, soundness, honest completeness, and cost.

Does not own: commitment, public projection, norm, running CE checks,
terminal composition, a deployment frame, Rust, Spartan, WHIR, or artifacts.

Emits constraints: two rows and one auxiliary residual column per native CCS
row.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.TerminalR1cs.FreshCcs

open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.Goldilocks.NativeCcsProgram
open Nightstream.Implementation.Lowering.Goldilocks.NativeCcsSelector
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

private abbrev RelationShape
    (program : NativeCcsProgram.Program)
    (domain : NativeCcsCompiler.RowDomain program)
    (publicRingColumns : Nat)
    (publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length) :=
  NativeCcsPhi81.shape program domain publicRingColumns publicFits

/-- Physical placement for one fresh complete assignment and its CCS
residuals. The assignment columns are caller-owned terminal inputs. -/
structure Frame
    (program : NativeCcsProgram.Program)
    (domain : NativeCcsCompiler.RowDomain program)
    (publicRingColumns : Nat)
    (publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length) where
  owner : PhysicalOwner
  firstOrdinal : Nat
  witness : Fin (RelationShape program domain publicRingColumns publicFits).carrierWidth →
    ColumnId
  residual : Fin program.rows.length → ColumnId

/-- Physical complete-carrier column that represents one logical program
column. -/
def physicalColumn
    {program : NativeCcsProgram.Program}
    {domain : NativeCcsCompiler.RowDomain program}
    {publicRingColumns : Nat}
    {publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length}
    (valid : NativeCcsCompiler.Valid program)
    (frame : Frame program domain publicRingColumns publicFits)
    (column : ColumnId) : ColumnId :=
  frame.witness
    (Phi81CarrierLayout.embedLogical
      (NativeCcsCompiler.ColumnIndex.index program valid column))

/-- Relocate one sparse combination into the fresh complete assignment. -/
def mapCombination
    {program : NativeCcsProgram.Program}
    {domain : NativeCcsCompiler.RowDomain program}
    {publicRingColumns : Nat}
    {publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length}
    (valid : NativeCcsCompiler.Valid program)
    (frame : Frame program domain publicRingColumns publicFits)
    (combination : LinearCombination) : LinearCombination :=
  combination.map fun term =>
    ⟨physicalColumn valid frame term.column, term.coefficient⟩

/-- Relocate one selected native CCS row without changing its polynomial. -/
def mappedRow
    {program : NativeCcsProgram.Program}
    {domain : NativeCcsCompiler.RowDomain program}
    {publicRingColumns : Nat}
    {publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length}
    (valid : NativeCcsCompiler.Valid program)
    (frame : Frame program domain publicRingColumns publicFits)
    (selected : SelectedRow) : SelectedRow where
  source := {
    id := selected.source.id
    row := {
      a := mapCombination valid frame selected.source.row.a
      b := mapCombination valid frame selected.source.row.b
      c := mapCombination valid frame selected.source.row.c
    }
  }
  selector := physicalColumn valid frame selected.selector

private theorem mapCombination_eval
    {program : NativeCcsProgram.Program}
    {domain : NativeCcsCompiler.RowDomain program}
    {publicRingColumns : Nat}
    {publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length}
    (valid : NativeCcsCompiler.Valid program)
    (frame : Frame program domain publicRingColumns publicFits)
    (combination : LinearCombination)
    (assignment : ColumnId → F) :
    (mapCombination valid frame combination).eval assignment =
      combination.eval
        (fun column => assignment (physicalColumn valid frame column)) := by
  induction combination with
  | nil =>
      rfl
  | cons term tail inductionHypothesis =>
      change
        term.coefficient *
              assignment (physicalColumn valid frame term.column) +
            (mapCombination valid frame tail).eval assignment =
          term.coefficient *
              assignment (physicalColumn valid frame term.column) +
            LinearCombination.eval
              (fun column =>
                assignment (physicalColumn valid frame column))
              tail
      rw [inductionHypothesis]

theorem mappedRow_holds_iff
    {program : NativeCcsProgram.Program}
    {domain : NativeCcsCompiler.RowDomain program}
    {publicRingColumns : Nat}
    {publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length}
    (valid : NativeCcsCompiler.Valid program)
    (frame : Frame program domain publicRingColumns publicFits)
    (selected : SelectedRow)
    (assignment : ColumnId → F) :
    (mappedRow valid frame selected).Holds assignment ↔
      selected.Holds
        (fun column => assignment (physicalColumn valid frame column)) := by
  unfold SelectedRow.Holds mappedRow
  simp only [mapCombination_eval]

/-- The selected source row at one two-row terminal position. -/
def sourceAt
    {program : NativeCcsProgram.Program}
    (position : Fin (2 * program.rows.length)) :
    Fin program.rows.length :=
  ⟨position.val / 2, by
    have below := position.isLt
    omega⟩

/-- One exact terminal row. Even positions compute the residual; odd
positions apply the native CCS selector to it. -/
def rowAt
    {program : NativeCcsProgram.Program}
    {domain : NativeCcsCompiler.RowDomain program}
    {publicRingColumns : Nat}
    {publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length}
    (valid : NativeCcsCompiler.Valid program)
    (frame : Frame program domain publicRingColumns publicFits)
    (position : Fin (2 * program.rows.length)) : OwnedRow :=
  let source := sourceAt position
  let selected := mappedRow valid frame (program.rows.get source)
  if position.val % 2 = 0 then
    ⟨⟨frame.owner, frame.firstOrdinal + position.val⟩,
      ActivatedRawProgram.liftedRow selected.source.row
        (frame.residual source)⟩
  else
    ⟨⟨frame.owner, frame.firstOrdinal + position.val⟩,
      ActivatedRawProgram.gateRow selected.selector
        (frame.residual source)⟩

def rows
    {program : NativeCcsProgram.Program}
    {domain : NativeCcsCompiler.RowDomain program}
    {publicRingColumns : Nat}
    {publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length}
    (valid : NativeCcsCompiler.Valid program)
    (frame : Frame program domain publicRingColumns publicFits) :
    List OwnedRow :=
  List.ofFn (rowAt valid frame)

def columns
    {program : NativeCcsProgram.Program}
    {domain : NativeCcsCompiler.RowDomain program}
    {publicRingColumns : Nat}
    {publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length}
    (frame : Frame program domain publicRingColumns publicFits) :
    List OwnedColumn :=
  List.ofFn fun source =>
    ⟨frame.residual source, .auxiliaryColumn⟩

@[simp] theorem rows_length
    {program : NativeCcsProgram.Program}
    {domain : NativeCcsCompiler.RowDomain program}
    {publicRingColumns : Nat}
    {publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length}
    (valid : NativeCcsCompiler.Valid program)
    (frame : Frame program domain publicRingColumns publicFits) :
    (rows valid frame).length = 2 * program.rows.length := by
  simp [rows]

@[simp] theorem columns_length
    {program : NativeCcsProgram.Program}
    {domain : NativeCcsCompiler.RowDomain program}
    {publicRingColumns : Nat}
    {publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length}
    (frame : Frame program domain publicRingColumns publicFits) :
    (columns frame).length = program.rows.length := by
  simp [columns]

structure Distinct
    {program : NativeCcsProgram.Program}
    {domain : NativeCcsCompiler.RowDomain program}
    {publicRingColumns : Nat}
    {publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length}
    (frame : Frame program domain publicRingColumns publicFits) : Prop where
  residualInjective : Function.Injective frame.residual
  witnessNeResidual :
    ∀ witnessCoordinate source,
      frame.witness witnessCoordinate ≠ frame.residual source

private theorem nodup_ofFn_of_injective
    {alpha : Type} :
    ∀ {count : Nat}
      (function : Fin count → alpha),
      Function.Injective function →
      (List.ofFn function).Nodup
  | 0, function, injective => by
      simp
  | _ + 1, function, injective => by
      rw [List.ofFn_succ, List.nodup_cons]
      constructor
      · intro member
        rcases List.mem_ofFn.mp member with ⟨index, equal⟩
        exact Fin.succ_ne_zero index (injective equal)
      · exact nodup_ofFn_of_injective
          (fun index => function index.succ)
          (fun first second equal =>
            Fin.succ_inj.mp (injective equal))

theorem columnIds_nodup
    {program : NativeCcsProgram.Program}
    {domain : NativeCcsCompiler.RowDomain program}
    {publicRingColumns : Nat}
    {publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length}
    (frame : Frame program domain publicRingColumns publicFits)
    (distinct : Distinct frame) :
    ((columns frame).map fun column => column.id).Nodup := by
  rw [columns, List.map_ofFn]
  exact nodup_ofFn_of_injective frame.residual
    distinct.residualInjective

theorem rowIds_nodup
    {program : NativeCcsProgram.Program}
    {domain : NativeCcsCompiler.RowDomain program}
    {publicRingColumns : Nat}
    {publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length}
    (valid : NativeCcsCompiler.Valid program)
    (frame : Frame program domain publicRingColumns publicFits) :
    ((rows valid frame).map fun owned => owned.id).Nodup := by
  rw [rows, List.map_ofFn]
  apply nodup_ofFn_of_injective
  intro first second equal
  apply Fin.ext
  have ordinalEqual :=
    congrArg (fun id : RowId => id.ordinal) equal
  simp only [Function.comp_apply, rowAt] at ordinalEqual
  split at ordinalEqual <;> split at ordinalEqual <;>
    exact Nat.add_left_cancel ordinalEqual

theorem rows_owned
    {program : NativeCcsProgram.Program}
    {domain : NativeCcsCompiler.RowDomain program}
    {publicRingColumns : Nat}
    {publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length}
    (valid : NativeCcsCompiler.Valid program)
    (frame : Frame program domain publicRingColumns publicFits)
    (owned : OwnedRow)
    (member : owned ∈ rows valid frame) :
    owned.id.owner = frame.owner := by
  rcases List.mem_ofFn.mp member with ⟨position, rfl⟩
  unfold rowAt
  split <;> rfl

theorem rowAt_supported
    {program : NativeCcsProgram.Program}
    {domain : NativeCcsCompiler.RowDomain program}
    {publicRingColumns : Nat}
    {publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length}
    (valid : NativeCcsCompiler.Valid program)
    (frame : Frame program domain publicRingColumns publicFits)
    (position : Fin (2 * program.rows.length))
    (column : ColumnId)
    (mentioned : column ∈ (rowAt valid frame position).columnIds) :
    column ∈
        (mappedRow valid frame
          (program.rows.get (sourceAt position))).columnIds ∨
      column = frame.residual (sourceAt position) := by
  by_cases even : position.val % 2 = 0
  · simp only [rowAt, even, if_pos, OwnedRow.columnIds] at mentioned
    rw [ActivatedRawProgram.liftedRow_columnIds] at mentioned
    rcases List.mem_append.1 mentioned with sourceMember | residualMember
    · exact Or.inl
        (List.mem_cons_of_mem
          (mappedRow valid frame
            (program.rows.get (sourceAt position))).selector
          sourceMember)
    · exact Or.inr (by simpa using residualMember)
  · unfold rowAt at mentioned
    rw [if_neg even] at mentioned
    change
      column ∈
        (ActivatedRawProgram.gateRow
          (mappedRow valid frame
            (program.rows.get (sourceAt position))).selector
          (frame.residual (sourceAt position))).columnIds at mentioned
    rw [ActivatedRawProgram.gateRow_columnIds] at mentioned
    simp only [List.mem_cons, List.not_mem_nil, or_false] at mentioned
    rcases mentioned with selector | residual
    · exact Or.inl (by
        rw [SelectedRow.columnIds]
        exact selector ▸ List.mem_cons_self)
    · exact Or.inr residual

private theorem mapCombination_supported
    {program : NativeCcsProgram.Program}
    {domain : NativeCcsCompiler.RowDomain program}
    {publicRingColumns : Nat}
    {publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length}
    (valid : NativeCcsCompiler.Valid program)
    (frame : Frame program domain publicRingColumns publicFits)
    (combination : LinearCombination)
    (column : ColumnId)
    (mentioned :
      column ∈
        (mapCombination valid frame combination).map
          fun term => term.column) :
    ∃ sourceColumn,
      column = physicalColumn valid frame sourceColumn := by
  rcases List.mem_map.mp mentioned with
    ⟨term, termMember, rfl⟩
  rcases List.mem_map.mp termMember with
    ⟨sourceTerm, _sourceMember, rfl⟩
  exact ⟨sourceTerm.column, rfl⟩

theorem mappedRow_supported
    {program : NativeCcsProgram.Program}
    {domain : NativeCcsCompiler.RowDomain program}
    {publicRingColumns : Nat}
    {publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length}
    (valid : NativeCcsCompiler.Valid program)
    (frame : Frame program domain publicRingColumns publicFits)
    (selected : SelectedRow)
    (column : ColumnId)
    (mentioned : column ∈ (mappedRow valid frame selected).columnIds) :
    ∃ coordinate, column = frame.witness coordinate := by
  simp only [mappedRow, SelectedRow.columnIds, List.mem_cons] at mentioned
  rcases mentioned with selector | source
  · exact
      ⟨Phi81CarrierLayout.embedLogical
          (NativeCcsCompiler.ColumnIndex.index
            program valid selected.selector),
        selector⟩
  · simp only [OwnedRow.columnIds, Row.columnIds,
      List.map_append, List.mem_append] at source
    rcases source with (inA | inB) | inC
    · rcases mapCombination_supported valid frame selected.source.row.a
          column inA with ⟨sourceColumn, equal⟩
      exact
        ⟨Phi81CarrierLayout.embedLogical
            (NativeCcsCompiler.ColumnIndex.index
              program valid sourceColumn),
          equal⟩
    · rcases mapCombination_supported valid frame selected.source.row.b
          column inB with ⟨sourceColumn, equal⟩
      exact
        ⟨Phi81CarrierLayout.embedLogical
            (NativeCcsCompiler.ColumnIndex.index
              program valid sourceColumn),
          equal⟩
    · rcases mapCombination_supported valid frame selected.source.row.c
          column inC with ⟨sourceColumn, equal⟩
      exact
        ⟨Phi81CarrierLayout.embedLogical
            (NativeCcsCompiler.ColumnIndex.index
              program valid sourceColumn),
          equal⟩

/-- A fresh native-CCS row mentions only a complete-assignment coordinate
or the residual allocated for its selected source row. -/
theorem rowAt_supported_by_frame
    {program : NativeCcsProgram.Program}
    {domain : NativeCcsCompiler.RowDomain program}
    {publicRingColumns : Nat}
    {publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length}
    (valid : NativeCcsCompiler.Valid program)
    (frame : Frame program domain publicRingColumns publicFits)
    (position : Fin (2 * program.rows.length))
    (column : ColumnId)
    (mentioned : column ∈ (rowAt valid frame position).columnIds) :
    (∃ coordinate, column = frame.witness coordinate) ∨
      column = frame.residual (sourceAt position) := by
  rcases rowAt_supported valid frame position column mentioned with
    source | residual
  · exact Or.inl
      (mappedRow_supported valid frame
        (program.rows.get (sourceAt position)) column source)
  · exact Or.inr residual

private theorem satisfies_ofFn_iff :
    ∀ {count : Nat}
      (function : Fin count → OwnedRow)
      (assignment : ColumnId → F),
      Satisfies (List.ofFn function) assignment ↔
        ∀ position, (function position).row.Holds assignment
  | 0, function, assignment => by
      simp
  | _ + 1, function, assignment => by
      rw [List.ofFn_succ, Goldilocks.satisfies_cons,
        satisfies_ofFn_iff (fun index => function index.succ) assignment]
      constructor
      · rintro ⟨head, tail⟩ position
        exact Fin.cases head tail position
      · intro every
        exact ⟨every 0, fun index => every index.succ⟩

private def evenPosition
    {program : NativeCcsProgram.Program}
    (source : Fin program.rows.length) :
    Fin (2 * program.rows.length) :=
  ⟨2 * source.val, by
    have below := source.isLt
    omega⟩

private def oddPosition
    {program : NativeCcsProgram.Program}
    (source : Fin program.rows.length) :
    Fin (2 * program.rows.length) :=
  ⟨2 * source.val + 1, by
    have below := source.isLt
    omega⟩

@[simp] private theorem sourceAt_even
    {program : NativeCcsProgram.Program}
    (source : Fin program.rows.length) :
    sourceAt (evenPosition source) = source := by
  apply Fin.ext
  simp [sourceAt, evenPosition]

@[simp] private theorem sourceAt_odd
    {program : NativeCcsProgram.Program}
    (source : Fin program.rows.length) :
    sourceAt (oddPosition source) = source := by
  apply Fin.ext
  change (2 * source.val + 1) / 2 = source.val
  omega

@[simp] private theorem rowAt_even
    {program : NativeCcsProgram.Program}
    {domain : NativeCcsCompiler.RowDomain program}
    {publicRingColumns : Nat}
    {publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length}
    (valid : NativeCcsCompiler.Valid program)
    (frame : Frame program domain publicRingColumns publicFits)
    (source : Fin program.rows.length) :
    (rowAt valid frame (evenPosition source)).row =
      ActivatedRawProgram.liftedRow
        (mappedRow valid frame (program.rows.get source)).source.row
        (frame.residual source) := by
  unfold rowAt
  rw [if_pos (by simp [evenPosition])]
  rw [sourceAt_even]

@[simp] private theorem rowAt_odd
    {program : NativeCcsProgram.Program}
    {domain : NativeCcsCompiler.RowDomain program}
    {publicRingColumns : Nat}
    {publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length}
    (valid : NativeCcsCompiler.Valid program)
    (frame : Frame program domain publicRingColumns publicFits)
    (source : Fin program.rows.length) :
    (rowAt valid frame (oddPosition source)).row =
      ActivatedRawProgram.gateRow
        (mappedRow valid frame (program.rows.get source)).selector
        (frame.residual source) := by
  unfold rowAt
  rw [if_neg (by simp [oddPosition])]
  rw [sourceAt_odd]

private theorem add_sub_same_right (left right : F) :
    (left + right) - left = right := by
  have cancel : left + -left = 0 := by
    rw [Lean.Grind.Fin.add_comm, Lean.Grind.Fin.neg_add_cancel]
  rw [Fin.sub_eq_add_neg]
  calc
    (left + right) + -left =
        (right + left) + -left := by
      rw [Lean.Grind.Fin.add_comm left right]
    _ = right + (left + -left) :=
      Lean.Grind.Fin.add_assoc _ _ _
    _ = right + 0 := by rw [cancel]
    _ = right := Fin.add_zero _

private theorem add_sub_same_left (left right : F) :
    left + (right - left) = right := by
  have cancel : left + -left = 0 := by
    rw [Lean.Grind.Fin.add_comm, Lean.Grind.Fin.neg_add_cancel]
  rw [Fin.sub_eq_add_neg]
  calc
    left + (right + -left) =
        (left + right) + -left :=
      (Lean.Grind.Fin.add_assoc _ _ _).symm
    _ = (right + left) + -left := by
      rw [Lean.Grind.Fin.add_comm left right]
    _ = right + (left + -left) :=
      Lean.Grind.Fin.add_assoc _ _ _
    _ = right + 0 := by rw [cancel]
    _ = right := Fin.add_zero _

private theorem selected_holds_of_pair
    (selected : SelectedRow)
    (residual : ColumnId)
    (assignment : ColumnId → F)
    (lifted :
      (ActivatedRawProgram.liftedRow
        selected.source.row residual).Holds assignment)
    (gated :
      (ActivatedRawProgram.gateRow
        selected.selector residual).Holds assignment) :
    selected.Holds assignment := by
  have residualValue :
      assignment residual =
        ActivatedRawProgram.residualValue
          selected.source.row assignment := by
    simp only [ActivatedRawProgram.liftedRow, Row.Holds,
      ActivatedRawProgram.linearCombination_eval_append,
      Goldilocks.singleton, LinearCombination.eval_cons,
      LinearCombination.eval_nil, Fin.one_mul, Fin.add_zero] at lifted
    unfold ActivatedRawProgram.residualValue
    calc
      assignment residual =
          (selected.source.row.c.eval assignment +
              assignment residual) -
            selected.source.row.c.eval assignment :=
        (add_sub_same_right
          (selected.source.row.c.eval assignment)
          (assignment residual)).symm
      _ =
          selected.source.row.a.eval assignment *
              selected.source.row.b.eval assignment -
            selected.source.row.c.eval assignment := by
        rw [← lifted]
  unfold SelectedRow.Holds NativeCcsSelector.polynomial
  change
    assignment selected.selector *
      ActivatedRawProgram.residualValue
        selected.source.row assignment = 0
  rw [← residualValue]
  simpa [ActivatedRawProgram.gateRow, Row.Holds,
    Goldilocks.singleton, LinearCombination.eval,
    Fin.one_mul, Fin.add_zero] using gated

private theorem pair_holds_of_selected
    (selected : SelectedRow)
    (residual : ColumnId)
    (assignment : ColumnId → F)
    (residualValue :
      assignment residual =
        ActivatedRawProgram.residualValue
          selected.source.row assignment)
    (holds : selected.Holds assignment) :
    (ActivatedRawProgram.liftedRow
        selected.source.row residual).Holds assignment ∧
      (ActivatedRawProgram.gateRow
        selected.selector residual).Holds assignment := by
  constructor
  · simp only [ActivatedRawProgram.liftedRow, Row.Holds,
      ActivatedRawProgram.linearCombination_eval_append,
      Goldilocks.singleton, LinearCombination.eval_cons,
      LinearCombination.eval_nil, Fin.one_mul, Fin.add_zero]
    rw [residualValue]
    unfold ActivatedRawProgram.residualValue
    exact (add_sub_same_left
      (selected.source.row.c.eval assignment)
      (selected.source.row.a.eval assignment *
        selected.source.row.b.eval assignment)).symm
  · unfold SelectedRow.Holds NativeCcsSelector.polynomial at holds
    simp only [ActivatedRawProgram.gateRow, Row.Holds,
      Goldilocks.singleton, LinearCombination.eval_cons,
      LinearCombination.eval_nil, Fin.one_mul, Fin.add_zero]
    change
      assignment selected.selector * assignment residual = 0
    rw [residualValue]
    exact holds

private theorem selected_satisfies_iff_forall
    (selectedRows : List SelectedRow)
    (assignment : ColumnId → F) :
    NativeCcsSelector.Satisfies selectedRows assignment ↔
      ∀ source : Fin selectedRows.length,
        (selectedRows.get source).Holds assignment := by
  induction selectedRows with
  | nil =>
      constructor
      · intro _ source
        exact Fin.elim0 source
      · intro _
        trivial
  | cons head tail inductionHypothesis =>
      constructor
      · intro satisfied source
        refine Fin.cases satisfied.1 (fun tailSource => ?_) source
        exact inductionHypothesis.mp satisfied.2 tailSource
      · intro every
        exact ⟨
          every ⟨0, by simp⟩,
          inductionHypothesis.mpr
            (fun source => by simpa using every (Fin.succ source))
        ⟩

/-- Terminal R1CS satisfaction implies the exact native CCS source program. -/
theorem rows_sound
    {program : NativeCcsProgram.Program}
    {domain : NativeCcsCompiler.RowDomain program}
    {publicRingColumns : Nat}
    {publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length}
    (valid : NativeCcsCompiler.Valid program)
    (frame : Frame program domain publicRingColumns publicFits)
    (assignment : ColumnId → F)
    (satisfied : Satisfies (rows valid frame) assignment) :
    NativeCcsSelector.Satisfies program.rows
      (fun column => assignment (physicalColumn valid frame column)) := by
  have every :=
    (satisfies_ofFn_iff (rowAt valid frame) assignment).mp satisfied
  apply (selected_satisfies_iff_forall program.rows _).mpr
  intro source
  have lifted := every (evenPosition source)
  have gated := every (oddPosition source)
  rw [rowAt_even] at lifted
  rw [rowAt_odd] at gated
  apply (mappedRow_holds_iff valid frame
    (program.rows.get source) assignment).mp
  exact selected_holds_of_pair
    (mappedRow valid frame (program.rows.get source))
    (frame.residual source) assignment lifted gated

/-- The lowered rows reach the exact paper fresh-CCS residual relation over
the complete assignment, including arbitrary completion-suffix values. -/
theorem rows_ccsSound
    {program : NativeCcsProgram.Program}
    {domain : NativeCcsCompiler.RowDomain program}
    {publicRingColumns : Nat}
    {publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length}
    (valid : NativeCcsCompiler.Valid program)
    (frame : Frame program domain publicRingColumns publicFits)
    (assignment : ColumnId → F)
    (satisfied : Satisfies (rows valid frame) assignment) :
    Phi81Relation.ccsSatisfied
      (NativeCcsPhi81.relation program valid domain
        publicRingColumns publicFits)
      (fun coordinate => assignment (frame.witness coordinate)) := by
  apply
    (NativeCcsPhi81.ccsSatisfied_arbitrary_iff
      program valid domain publicRingColumns publicFits
      (fun coordinate => assignment (frame.witness coordinate))).mpr
  simpa [physicalColumn, NativeCcsCompiler.pulledAssignment,
    Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.LeanCompiler.StableRows.pulledAssignment]
    using rows_sound valid frame assignment satisfied

/-- Honest native CCS satisfaction and prefilled residuals satisfy every
terminal lowering row. -/
theorem rows_honest
    {program : NativeCcsProgram.Program}
    {domain : NativeCcsCompiler.RowDomain program}
    {publicRingColumns : Nat}
    {publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length}
    (valid : NativeCcsCompiler.Valid program)
    (frame : Frame program domain publicRingColumns publicFits)
    (assignment : ColumnId → F)
    (sourceSatisfied :
      NativeCcsSelector.Satisfies program.rows
        (fun column => assignment (physicalColumn valid frame column)))
    (residuals :
      ∀ source,
        assignment (frame.residual source) =
          ActivatedRawProgram.residualValue
            (mappedRow valid frame
              (program.rows.get source)).source.row assignment) :
    Satisfies (rows valid frame) assignment := by
  apply (satisfies_ofFn_iff (rowAt valid frame) assignment).mpr
  have sourcePointwise :
      ∀ source : Fin program.rows.length,
        (program.rows.get source).Holds
          (fun column =>
            assignment (physicalColumn valid frame column)) :=
    (selected_satisfies_iff_forall program.rows _).mp sourceSatisfied
  intro position
  let source := sourceAt position
  have mappedHolds :
      (mappedRow valid frame
        (program.rows.get source)).Holds assignment :=
    (mappedRow_holds_iff valid frame
      (program.rows.get source) assignment).mpr
      (sourcePointwise source)
  have pair :=
    pair_holds_of_selected
      (mappedRow valid frame (program.rows.get source))
      (frame.residual source) assignment
      (residuals source) mappedHolds
  by_cases even : position.val % 2 = 0
  · simpa [rowAt, even, source] using pair.1
  · simpa [rowAt, even, source] using pair.2

def cost (program : NativeCcsProgram.Program) : Cost :=
  ⟨2 * program.rows.length, 0, 0, program.rows.length⟩

@[simp] theorem cost_rows (program : NativeCcsProgram.Program) :
    (cost program).recurringRows = 2 * program.rows.length :=
  rfl

@[simp] theorem cost_auxiliary (program : NativeCcsProgram.Program) :
    (cost program).auxiliaryColumns = program.rows.length :=
  rfl

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.TerminalR1cs.FreshCcs
