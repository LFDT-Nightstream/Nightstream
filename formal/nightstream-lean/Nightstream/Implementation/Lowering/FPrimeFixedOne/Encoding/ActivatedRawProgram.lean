import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.DirectCalls.Common
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.RawAllocationCoverage

/-!
Contract: activation lowering for an already constructed sparse R1CS program.

Each raw equation `A * B = C` receives one fresh residual `r` and emits

* `A * B = C + r`;
* `active * r = 0`.

Thus `active = 1` recovers the raw program, while `active = 0` admits every
visible assignment by writing the exact equation residual.  Residuals are
explicit temporary columns; the construction owns no protocol semantics,
Rust layout, generated artifact, or caller-supplied acceptance fact.

Emits constraints: two rows and one auxiliary column per raw row.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ActivatedRawProgram

open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.DirectCalls
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.RawAllocationCoverage

/-- The exact value needed to relax one raw equation. -/
def residualValue
    (row : Row) (assignment : ColumnId → Field) : Field :=
  row.a.eval assignment * row.b.eval assignment - row.c.eval assignment

/-- The original equation with one explicit additive residual. -/
def liftedRow (row : Row) (residual : ColumnId) : Row where
  a := row.a
  b := row.b
  c := row.c ++ singleton residual 1

/-- Force the residual to zero exactly when the enclosing call is active. -/
def gateRow (active residual : ColumnId) : Row where
  a := singleton active 1
  b := singleton residual 1
  c := []

/-- Pair each raw equation with its fresh residual in source order. -/
def rawRows (active : ColumnId) :
    List Row → List ColumnId → List Row
  | row :: rows, residual :: residuals =>
      liftedRow row residual ::
        gateRow active residual ::
          rawRows active rows residuals
  | _, _ => []

/-- Stable positional ownership of the complete activated program. -/
def rows
    (owner : PhysicalOwner)
    (active : ColumnId)
    (source : List Row)
    (residuals : List ColumnId) : List OwnedRow :=
  ownRows owner (rawRows active source residuals)

/-- Fill every residual from the original visible assignment. -/
def complete
    (assignment : ColumnId → Field)
    (source : List Row)
    (residuals : List ColumnId) : ColumnId → Field :=
  writeColumns assignment residuals
    (source.map fun row => residualValue row assignment)

/-- Activation adds one row and one auxiliary coordinate per raw row. -/
def overheadCost (sourceRows : Nat) : Cost where
  recurringRows := sourceRows
  committedColumns := 0
  publicColumns := 0
  auxiliaryColumns := sourceRows

/-- Complete cost of activating an intrinsic raw program. -/
def cost (intrinsic : Cost) : Cost :=
  intrinsic + overheadCost intrinsic.recurringRows

theorem linearCombination_eval_append
    (left right : LinearCombination)
    (assignment : ColumnId → Field) :
    (left ++ right).eval assignment =
      left.eval assignment + right.eval assignment := by
  induction left with
  | nil => simp
  | cons term tail inductionHypothesis =>
      simp only [List.cons_append, LinearCombination.eval,
        inductionHypothesis, Lean.Grind.Fin.add_assoc]

private theorem rawSatisfies_of_forall
    (source : List Row)
    (assignment : ColumnId → Field)
    (holds : ∀ row ∈ source, row.Holds assignment) :
    RawSatisfies source assignment := by
  induction source with
  | nil => trivial
  | cons row source inductionHypothesis =>
      exact ⟨holds row List.mem_cons_self,
        inductionHypothesis (fun tail tailMember =>
          holds tail (List.mem_cons_of_mem row tailMember))⟩

private theorem holds_of_rawSatisfies
    (source : List Row)
    (assignment : ColumnId → Field)
    (satisfied : RawSatisfies source assignment) :
    ∀ row ∈ source, row.Holds assignment := by
  induction source with
  | nil =>
      intro row member
      simp at member
  | cons head tail inductionHypothesis =>
      intro row member
      rcases List.mem_cons.1 member with rfl | tailMember
      · exact satisfied.1
      · exact inductionHypothesis satisfied.2 row tailMember

private theorem linearCombination_eval_complete
    (assignment : ColumnId → Field)
    (source : List Row)
    (residuals : List ColumnId)
    (combination : LinearCombination)
    (fresh :
      IdsDisjoint residuals
        (source.flatMap fun row => row.columnIds))
    (supported :
      ∀ term ∈ combination,
        term.column ∈ source.flatMap fun row => row.columnIds) :
    combination.eval (complete assignment source residuals) =
      combination.eval assignment := by
  induction combination with
  | nil => rfl
  | cons term tail inductionHypothesis =>
      have notWritten : term.column ∉ residuals := by
        intro written
        exact fresh term.column written
          (supported term List.mem_cons_self)
      have tailSupported :
          ∀ item ∈ tail,
            item.column ∈ source.flatMap fun row => row.columnIds := by
        intro item member
        exact supported item (List.mem_cons_of_mem term member)
      have preserved :
          complete assignment source residuals term.column =
            assignment term.column := by
        exact writeColumns_of_not_mem assignment residuals
          (source.map fun row => residualValue row assignment)
          term.column notWritten
      simp only [LinearCombination.eval, preserved,
        inductionHypothesis tailSupported]

private theorem row_eval_complete
    (assignment : ColumnId → Field)
    (source : List Row)
    (residuals : List ColumnId)
    (fresh :
      IdsDisjoint residuals
        (source.flatMap fun row => row.columnIds))
    (row : Row)
    (member : row ∈ source) :
    row.a.eval (complete assignment source residuals) =
        row.a.eval assignment ∧
      row.b.eval (complete assignment source residuals) =
        row.b.eval assignment ∧
      row.c.eval (complete assignment source residuals) =
        row.c.eval assignment := by
  have inSupport :
      ∀ term ∈ row.a ++ row.b ++ row.c,
        term.column ∈ source.flatMap fun item => item.columnIds := by
    intro term termMember
    apply List.mem_flatMap.2
    refine ⟨row, member, ?_⟩
    unfold Row.columnIds
    exact List.mem_map.2 ⟨term, termMember, rfl⟩
  constructor
  · apply linearCombination_eval_complete assignment source residuals
      row.a fresh
    intro term termMember
    exact inSupport term
      (List.mem_append_left row.c
        (List.mem_append_left row.b termMember))
  constructor
  · apply linearCombination_eval_complete assignment source residuals
      row.b fresh
    intro term termMember
    exact inSupport term
      (List.mem_append_left row.c
        (List.mem_append_right row.a termMember))
  · apply linearCombination_eval_complete assignment source residuals
      row.c fresh
    intro term termMember
    exact inSupport term
      (List.mem_append_right (row.a ++ row.b) termMember)

private theorem complete_at_pair
    (assignment : ColumnId → Field)
    (source : List Row)
    (residuals : List ColumnId)
    (lengthEqual : source.length = residuals.length)
    (nodup : residuals.Nodup)
    (pair : Row × ColumnId)
    (member : pair ∈ List.zip source residuals) :
    complete assignment source residuals pair.2 =
      residualValue pair.1 assignment := by
  induction source generalizing residuals pair with
  | nil =>
      simp at member
  | cons row source inductionHypothesis =>
      cases residuals with
      | nil =>
          simp at lengthEqual
      | cons residual residuals =>
          simp only [List.length_cons, Nat.succ.injEq] at lengthEqual
          have split :
              residual ∉ residuals ∧ residuals.Nodup := by
            simpa only [List.nodup_cons] using nodup
          change pair ∈ (row, residual) :: List.zip source residuals at member
          rw [List.mem_cons] at member
          rcases member with headEqual | tailMember
          · cases headEqual
            exact writeColumns_head assignment residual residuals
              (residualValue row assignment)
              (source.map fun item => residualValue item assignment)
          · have pairResidualMember : pair.2 ∈ residuals :=
              (List.of_mem_zip tailMember).2
            have different : pair.2 ≠ residual := by
              intro equal
              rw [equal] at pairResidualMember
              exact split.1 pairResidualMember
            rw [complete, List.map_cons,
              writeColumns_tail assignment residual pair.2 residuals
                (residualValue row assignment)
                (source.map fun item => residualValue item assignment)
                different]
            exact inductionHypothesis residuals lengthEqual split.2
              pair tailMember

private theorem emitted_row
    (active : ColumnId)
    (source : List Row)
    (residuals : List ColumnId)
    (emitted : Row)
    (member : emitted ∈ rawRows active source residuals) :
    ∃ pair ∈ List.zip source residuals,
      emitted = liftedRow pair.1 pair.2 ∨
        emitted = gateRow active pair.2 := by
  induction source generalizing residuals with
  | nil =>
      simp [rawRows] at member
  | cons row source inductionHypothesis =>
      cases residuals with
      | nil =>
          simp [rawRows] at member
      | cons residual residuals =>
          simp only [rawRows, List.mem_cons] at member
          rcases member with rfl | member
          · exact ⟨(row, residual), List.mem_cons_self, Or.inl rfl⟩
          · rcases member with rfl | tailMember
            · exact ⟨(row, residual), List.mem_cons_self, Or.inr rfl⟩
            · rcases inductionHypothesis residuals tailMember with
                ⟨pair, pairMember, kind⟩
              exact ⟨pair, List.mem_cons_of_mem _ pairMember, kind⟩

private theorem lifted_complete
    (assignment : ColumnId → Field)
    (source : List Row)
    (residuals : List ColumnId)
    (lengthEqual : source.length = residuals.length)
    (nodup : residuals.Nodup)
    (fresh :
      IdsDisjoint residuals
        (source.flatMap fun row => row.columnIds))
    (pair : Row × ColumnId)
    (member : pair ∈ List.zip source residuals) :
    (liftedRow pair.1 pair.2).Holds
      (complete assignment source residuals) := by
  have sourceMember : pair.1 ∈ source :=
    (List.of_mem_zip member).1
  have values :=
    row_eval_complete assignment source residuals fresh
      pair.1 sourceMember
  have residual :=
    complete_at_pair assignment source residuals lengthEqual nodup
      pair member
  simp only [liftedRow, Row.Holds, linearCombination_eval_append,
    Goldilocks.singleton, LinearCombination.eval_cons,
    LinearCombination.eval_nil, Fin.one_mul, Fin.add_zero,
    values.1, values.2.1, values.2.2, residual, residualValue]
  rw [Fin.sub_eq_add_neg,
    Lean.Grind.Fin.add_comm (pair.1.c.eval assignment),
    Lean.Grind.Fin.add_assoc,
    Lean.Grind.Fin.neg_add_cancel, Fin.add_zero]

private theorem gate_complete_inactive
    (assignment : ColumnId → Field)
    (source : List Row)
    (residuals : List ColumnId)
    (active : ColumnId)
    (activeZero : assignment active = 0)
    (activeFresh : active ∉ residuals)
    (residual : ColumnId) :
    (gateRow active residual).Holds
      (complete assignment source residuals) := by
  have activePreserved :
      complete assignment source residuals active = 0 := by
    rw [complete,
      writeColumns_of_not_mem assignment residuals
        (source.map fun row => residualValue row assignment)
        active activeFresh,
      activeZero]
  simp only [gateRow, Row.Holds, Goldilocks.singleton,
    LinearCombination.eval_cons, LinearCombination.eval_nil,
    activePreserved, Fin.one_mul, Fin.add_zero, Fin.zero_mul]

theorem rawRows_length
    (active : ColumnId)
    (source : List Row)
    (residuals : List ColumnId)
    (lengthEqual : source.length = residuals.length) :
    (rawRows active source residuals).length = 2 * source.length := by
  induction source generalizing residuals with
  | nil =>
      simp [rawRows]
  | cons row source inductionHypothesis =>
      cases residuals with
      | nil =>
          simp at lengthEqual
      | cons residual residuals =>
          simp only [List.length_cons, Nat.succ.injEq] at lengthEqual
          simp [rawRows, inductionHypothesis residuals lengthEqual]
          omega

theorem rows_length
    (owner : PhysicalOwner)
    (active : ColumnId)
    (source : List Row)
    (residuals : List ColumnId)
    (lengthEqual : source.length = residuals.length) :
    (rows owner active source residuals).length = 2 * source.length := by
  simpa [rows] using rawRows_length active source residuals lengthEqual

theorem rows_owned
    (owner : PhysicalOwner)
    (active : ColumnId)
    (source : List Row)
    (residuals : List ColumnId)
    (row : OwnedRow)
    (member : row ∈ rows owner active source residuals) :
    row.id.owner = owner :=
  ownRows_owner owner _ row member

theorem rowIds_nodup
    (owner : PhysicalOwner)
    (active : ColumnId)
    (source : List Row)
    (residuals : List ColumnId) :
    ((rows owner active source residuals).map fun row => row.id).Nodup :=
  ownRows_ids_nodup owner _

theorem liftedRow_columnIds
    (row : Row) (residual : ColumnId) :
    (liftedRow row residual).columnIds =
      row.columnIds ++ [residual] := by
  simp [liftedRow, Row.columnIds, Goldilocks.singleton,
    List.map_append, List.append_assoc]

theorem gateRow_columnIds
    (active residual : ColumnId) :
    (gateRow active residual).columnIds = [active, residual] := by
  simp [gateRow, Row.columnIds, Goldilocks.singleton]

/-- Activation introduces no dependency except the active wire and the
explicit residual list. -/
theorem rawRows_supported
    (active : ColumnId)
    (source : List Row)
    (residuals allowed : List ColumnId)
    (sourceSupported : RawRowsSupportedBy allowed source) :
    RawRowsSupportedBy
      (active :: allowed ++ residuals)
      (rawRows active source residuals) := by
  intro emitted emittedMember column columnMember
  rcases emitted_row active source residuals emitted emittedMember with
    ⟨pair, pairMember, rfl | rfl⟩
  · rw [liftedRow_columnIds] at columnMember
    rcases List.mem_append.1 columnMember with sourceColumn | residualColumn
    · exact List.mem_cons_of_mem active
        (List.mem_append_left residuals
          (sourceSupported pair.1 (List.of_mem_zip pairMember).1
            column sourceColumn))
    · have equal : column = pair.2 := by
        simpa only [List.mem_singleton] using residualColumn
      subst column
      exact List.mem_cons_of_mem active
        (List.mem_append_right allowed
          (List.of_mem_zip pairMember).2)
  · rw [gateRow_columnIds] at columnMember
    simp only [List.mem_cons, List.not_mem_nil, or_false] at columnMember
    rcases columnMember with rfl | rfl
    · exact List.mem_cons_self
    · exact List.mem_cons_of_mem active
        (List.mem_append_right allowed
          (List.of_mem_zip pairMember).2)

/-- Owned activated rows inherit the raw support theorem without reconstructing
their equations from row identifiers. -/
theorem rows_supported
    (owner : PhysicalOwner)
    (active : ColumnId)
    (source : List Row)
    (residuals allowed : List ColumnId)
    (sourceSupported : RawRowsSupportedBy allowed source)
    (row : OwnedRow)
    (member : row ∈ rows owner active source residuals)
    (column : ColumnId)
    (columnMember : column ∈ row.columnIds) :
    column ∈ active :: allowed ++ residuals := by
  apply ownRows_supported owner
    (rawRows active source residuals)
    (active :: allowed ++ residuals)
    (rawRows_supported active source residuals allowed sourceSupported)
    row member column columnMember

private theorem lifted_mem_of_source_mem
    (active : ColumnId)
    (source : List Row)
    (residuals : List ColumnId)
    (lengthEqual : source.length = residuals.length)
    (row : Row)
    (member : row ∈ source) :
    ∃ residual,
      residual ∈ residuals ∧
        liftedRow row residual ∈ rawRows active source residuals := by
  induction source generalizing residuals with
  | nil =>
      simp at member
  | cons head tail inductionHypothesis =>
      cases residuals with
      | nil =>
          simp at lengthEqual
      | cons residual residuals =>
          simp only [List.length_cons, Nat.succ.injEq] at lengthEqual
          rcases List.mem_cons.1 member with rfl | tailMember
          · exact ⟨residual, List.mem_cons_self, List.mem_cons_self⟩
          · rcases inductionHypothesis residuals lengthEqual tailMember with
              ⟨tailResidual, residualMember, rowMember⟩
            exact ⟨tailResidual,
              List.mem_cons_of_mem residual residualMember,
              List.mem_cons_of_mem _
                (List.mem_cons_of_mem _ rowMember)⟩

private theorem gate_mem_of_residual_mem
    (active : ColumnId)
    (source : List Row)
    (residuals : List ColumnId)
    (lengthEqual : source.length = residuals.length)
    (residual : ColumnId)
    (member : residual ∈ residuals) :
    gateRow active residual ∈ rawRows active source residuals := by
  induction source generalizing residuals with
  | nil =>
      have empty : residuals = [] := by
        exact List.eq_nil_of_length_eq_zero lengthEqual.symm
      simp [empty] at member
  | cons row source inductionHypothesis =>
      cases residuals with
      | nil =>
          simp at member
      | cons head residuals =>
          simp only [List.length_cons, Nat.succ.injEq] at lengthEqual
          rcases List.mem_cons.1 member with rfl | tailMember
          · exact List.mem_cons_of_mem _ List.mem_cons_self
          · exact List.mem_cons_of_mem _
              (List.mem_cons_of_mem _
                (inductionHypothesis residuals lengthEqual tailMember))

/-- Every dependency of an intrinsic source row remains present in one
owned activated row. This is the occurrence bridge used when removing the
activation wrapper. -/
theorem source_column_emitted
    (owner : PhysicalOwner)
    (active : ColumnId)
    (source : List Row)
    (residuals : List ColumnId)
    (lengthEqual : source.length = residuals.length)
    (row : Row)
    (rowMember : row ∈ source)
    (column : ColumnId)
    (columnMember : column ∈ row.columnIds) :
    ∃ emitted,
      emitted ∈ rows owner active source residuals ∧
        column ∈ emitted.columnIds := by
  rcases lifted_mem_of_source_mem active source residuals lengthEqual
      row rowMember with
    ⟨residual, _, liftedMember⟩
  rcases ownRows_row_complete owner (rawRows active source residuals)
      (liftedRow row residual) liftedMember with
    ⟨emitted, emittedMember, emittedExact⟩
  refine ⟨emitted, emittedMember, ?_⟩
  rw [OwnedRow.columnIds, emittedExact, liftedRow_columnIds]
  exact List.mem_append_left _ columnMember

/-- The selector coordinate is present in the gate row paired with every
intrinsic source row. -/
theorem selector_emitted_of_source_mem
    (owner : PhysicalOwner)
    (active : ColumnId)
    (source : List Row)
    (residuals : List ColumnId)
    (lengthEqual : source.length = residuals.length)
    (row : Row)
    (rowMember : row ∈ source) :
    ∃ emitted,
      emitted ∈ rows owner active source residuals ∧
        active ∈ emitted.columnIds := by
  rcases lifted_mem_of_source_mem active source residuals lengthEqual
      row rowMember with
    ⟨residual, residualMember, _⟩
  have gateMember :=
    gate_mem_of_residual_mem active source residuals lengthEqual
      residual residualMember
  rcases ownRows_row_complete owner (rawRows active source residuals)
      (gateRow active residual) gateMember with
    ⟨emitted, emittedMember, emittedExact⟩
  refine ⟨emitted, emittedMember, ?_⟩
  rw [OwnedRow.columnIds, emittedExact, gateRow_columnIds]
  exact List.mem_cons_self

/-- Every source allocation remains mentioned by its lifted equation. -/
theorem source_coverage
    (active : ColumnId)
    (source : List Row)
    (residuals allocation : List ColumnId)
    (lengthEqual : source.length = residuals.length)
    (covered : TypedRowsCover source allocation) :
    TypedRowsCover (rawRows active source residuals) allocation := by
  intro column columnMember
  rcases covered column columnMember with
    ⟨sourceRow, sourceMember, mentioned⟩
  rcases lifted_mem_of_source_mem active source residuals lengthEqual
      sourceRow sourceMember with
    ⟨residual, _, liftedMember⟩
  exact ⟨liftedRow sourceRow residual, liftedMember,
    by
      rw [liftedRow_columnIds]
      exact List.mem_append_left _ mentioned⟩

/-- Every fresh residual is mentioned by its own gate row. -/
theorem residual_coverage
    (active : ColumnId)
    (source : List Row)
    (residuals : List ColumnId)
    (lengthEqual : source.length = residuals.length) :
    TypedRowsCover (rawRows active source residuals) residuals := by
  intro residual member
  refine ⟨gateRow active residual,
    gate_mem_of_residual_mem active source residuals lengthEqual
      residual member,
    ?_⟩
  rw [gateRow_columnIds]
  exact List.mem_cons_of_mem active List.mem_cons_self

/-- The activated program covers both the intrinsic raw allocation and every
new residual. -/
theorem allocation_coverage
    (active : ColumnId)
    (source : List Row)
    (residuals allocation : List ColumnId)
    (lengthEqual : source.length = residuals.length)
    (covered : TypedRowsCover source allocation) :
    TypedRowsCover
      (rawRows active source residuals)
      (allocation ++ residuals) := by
  intro column member
  rcases List.mem_append.1 member with inSource | inResidual
  · exact source_coverage active source residuals allocation
      lengthEqual covered column inSource
  · exact residual_coverage active source residuals lengthEqual
      column inResidual

theorem active_sound
    (active : ColumnId)
    (source : List Row)
    (residuals : List ColumnId)
    (assignment : ColumnId → Field)
    (lengthEqual : source.length = residuals.length)
    (activeOne : assignment active = 1)
    (satisfied : RawSatisfies (rawRows active source residuals) assignment) :
    RawSatisfies source assignment := by
  induction source generalizing residuals with
  | nil => trivial
  | cons row source inductionHypothesis =>
      cases residuals with
      | nil =>
          simp at lengthEqual
      | cons residual residuals =>
          simp only [List.length_cons, Nat.succ.injEq] at lengthEqual
          have lifted := satisfied.1
          have gated := satisfied.2.1
          have residualZero : assignment residual = 0 := by
            simpa only [gateRow, Row.Holds, Goldilocks.singleton,
              LinearCombination.eval_cons, LinearCombination.eval_nil,
              activeOne, Fin.one_mul, Fin.add_zero] using gated
          have rawHolds : row.Holds assignment := by
            simpa only [liftedRow, Row.Holds,
              linearCombination_eval_append, Goldilocks.singleton,
              LinearCombination.eval_cons, LinearCombination.eval_nil,
              residualZero, Fin.one_mul, Fin.add_zero]
              using lifted
          exact ⟨rawHolds,
            inductionHypothesis residuals lengthEqual satisfied.2.2⟩

theorem inactive_complete
    (active : ColumnId)
    (source : List Row)
    (residuals : List ColumnId)
    (assignment : ColumnId → Field)
    (lengthEqual : source.length = residuals.length)
    (nodup : residuals.Nodup)
    (fresh :
      IdsDisjoint residuals
        (source.flatMap fun row => row.columnIds))
    (activeZero : assignment active = 0)
    (activeFresh : active ∉ residuals) :
    RawSatisfies (rawRows active source residuals)
      (complete assignment source residuals) := by
  apply rawSatisfies_of_forall
  intro emitted emittedMember
  rcases emitted_row active source residuals emitted emittedMember with
    ⟨pair, pairMember, rfl | rfl⟩
  · exact lifted_complete assignment source residuals lengthEqual nodup
      fresh pair pairMember
  · exact gate_complete_inactive assignment source residuals active
      activeZero activeFresh pair.2

theorem active_complete
    (active : ColumnId)
    (source : List Row)
    (residuals : List ColumnId)
    (assignment : ColumnId → Field)
    (lengthEqual : source.length = residuals.length)
    (nodup : residuals.Nodup)
    (fresh :
      IdsDisjoint residuals
        (source.flatMap fun row => row.columnIds))
    (satisfied : RawSatisfies source assignment) :
    RawSatisfies (rawRows active source residuals)
      (complete assignment source residuals) := by
  apply rawSatisfies_of_forall
  intro emitted emittedMember
  rcases emitted_row active source residuals emitted emittedMember with
    ⟨pair, pairMember, rfl | rfl⟩
  · exact lifted_complete assignment source residuals lengthEqual nodup
      fresh pair pairMember
  · have rawHolds :=
      holds_of_rawSatisfies source assignment satisfied pair.1
        (List.of_mem_zip pairMember).1
    have residual :=
      complete_at_pair assignment source residuals lengthEqual nodup
        pair pairMember
    have residualZero :
        complete assignment source residuals pair.2 = 0 := by
      rw [residual, residualValue, rawHolds]
      exact Lean.Grind.AddCommGroup.sub_self _
    simp only [gateRow, Row.Holds, Goldilocks.singleton,
      LinearCombination.eval_cons, LinearCombination.eval_nil,
      residualZero, Fin.one_mul, Fin.add_zero, Fin.mul_zero]

theorem complete_changesOnly
    (assignment : ColumnId → Field)
    (source : List Row)
    (residuals : List ColumnId) :
    ChangesOnly residuals assignment
      (complete assignment source residuals) :=
  writeColumns_changesOnly assignment residuals _

theorem complete_agreesOn
    (assignment : ColumnId → Field)
    (source : List Row)
    (residuals visible : List ColumnId)
    (disjoint : IdsDisjoint residuals visible) :
    AgreesOn visible assignment
      (complete assignment source residuals) :=
  writeColumns_agreesOn assignment residuals visible _ disjoint

theorem overheadCost_rows (sourceRows : Nat) :
    (overheadCost sourceRows).recurringRows = sourceRows :=
  rfl

theorem overheadCost_auxiliary (sourceRows : Nat) :
    (overheadCost sourceRows).auxiliaryColumns = sourceRows :=
  rfl

theorem cost_rows (intrinsic : Cost) :
    (cost intrinsic).recurringRows = 2 * intrinsic.recurringRows := by
  simp [cost, overheadCost]
  omega

theorem cost_auxiliary (intrinsic : Cost) :
    (cost intrinsic).auxiliaryColumns =
      intrinsic.auxiliaryColumns + intrinsic.recurringRows := by
  rfl

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ActivatedRawProgram
