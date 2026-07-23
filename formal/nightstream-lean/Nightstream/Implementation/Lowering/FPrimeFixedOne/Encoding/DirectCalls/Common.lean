import Nightstream.Implementation.Lowering.Goldilocks.PrimitiveRecipes

/-!
Contract: finite assignment completion and exact bundle access used by the
direct fixed-one call recipes.

Owns:
- ordered writes to fresh physical columns;
- preservation outside the written identities;
- exact coordinate access through a bundle's proved width.

Does not own: protocol-call semantics, row selection, Rust columns, or
generated artifacts.

Emits constraints: no.
-/

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.DirectCalls

open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.Typed

universe u

/-! ## Fixed-arity bundle views -/

def unaryOperand
    {types : TypeSystem.{u}}
    {context : Schema types}
    {kind : types.Kind}
    {reference : Ref types context kind}
    (bundles : RefBundles (Refs.cons reference .nil)) :
    ColumnBundle reference.port.layout :=
  match bundles with
  | .cons head .nil => head

def firstBinaryOperand
    {types : TypeSystem.{u}}
    {context : Schema types}
    {leftKind rightKind : types.Kind}
    {left : Ref types context leftKind}
    {right : Ref types context rightKind}
    (bundles : RefBundles (Refs.cons left (Refs.cons right .nil))) :
    ColumnBundle left.port.layout :=
  match bundles with
  | .cons leftBundle (.cons _ .nil) => leftBundle

def secondBinaryOperand
    {types : TypeSystem.{u}}
    {context : Schema types}
    {leftKind rightKind : types.Kind}
    {left : Ref types context leftKind}
    {right : Ref types context rightKind}
    (bundles : RefBundles (Refs.cons left (Refs.cons right .nil))) :
    ColumnBundle right.port.layout :=
  match bundles with
  | .cons _ (.cons rightBundle .nil) => rightBundle

def unaryOutput
    {types : TypeSystem.{u}}
    {port : Port types}
    (bundles : SchemaBundles [port]) :
    ColumnBundle port.layout :=
  match bundles with
  | .cons output .nil => output

/-! ## Stable row ownership -/

/-- Assign consecutive local ordinals to an exact row list. -/
def ownRowsFrom
    (owner : PhysicalOwner) : Nat -> List Row -> List OwnedRow
  | _, [] => []
  | ordinal, row :: rows =>
      { id := { owner := owner, ordinal := ordinal }, row := row } ::
        ownRowsFrom owner (ordinal + 1) rows

def ownRows (owner : PhysicalOwner) (rows : List Row) : List OwnedRow :=
  ownRowsFrom owner 0 rows

@[simp] theorem ownRowsFrom_length
    (owner : PhysicalOwner)
    (ordinal : Nat)
    (rows : List Row) :
    (ownRowsFrom owner ordinal rows).length = rows.length := by
  induction rows generalizing ordinal with
  | nil => rfl
  | cons row rows inductionHypothesis =>
      simp [ownRowsFrom, inductionHypothesis]

@[simp] theorem ownRows_length
    (owner : PhysicalOwner)
    (rows : List Row) :
    (ownRows owner rows).length = rows.length :=
  ownRowsFrom_length owner 0 rows

theorem ownRowsFrom_owner
    (owner : PhysicalOwner)
    (ordinal : Nat)
    (rows : List Row)
    (owned : OwnedRow)
    (member : owned ∈ ownRowsFrom owner ordinal rows) :
    owned.id.owner = owner := by
  induction rows generalizing ordinal with
  | nil =>
      simp [ownRowsFrom] at member
  | cons row rows inductionHypothesis =>
      simp only [ownRowsFrom, List.mem_cons] at member
      rcases member with equal | tail
      · subst owned
        rfl
      · exact inductionHypothesis (ordinal + 1) tail

theorem ownRows_owner
    (owner : PhysicalOwner)
    (rows : List Row)
    (owned : OwnedRow)
    (member : owned ∈ ownRows owner rows) :
    owned.id.owner = owner :=
  ownRowsFrom_owner owner 0 rows owned member

theorem ownRowsFrom_row_mem
    (owner : PhysicalOwner)
    (ordinal : Nat)
    (rows : List Row)
    (owned : OwnedRow)
    (member : owned ∈ ownRowsFrom owner ordinal rows) :
    owned.row ∈ rows := by
  induction rows generalizing ordinal with
  | nil =>
      simp [ownRowsFrom] at member
  | cons row rows inductionHypothesis =>
      simp only [ownRowsFrom, List.mem_cons] at member
      rcases member with equal | tail
      · subst owned
        exact List.mem_cons_self
      · exact List.mem_cons_of_mem row
          (inductionHypothesis (ordinal + 1) tail)

theorem ownRows_row_mem
    (owner : PhysicalOwner)
    (rows : List Row)
    (owned : OwnedRow)
    (member : owned ∈ ownRows owner rows) :
    owned.row ∈ rows :=
  ownRowsFrom_row_mem owner 0 rows owned member

private theorem ownRowsFrom_ordinal_lower_bound
    (owner : PhysicalOwner)
    (ordinal : Nat)
    (rows : List Row)
    (owned : OwnedRow)
    (member : owned ∈ ownRowsFrom owner ordinal rows) :
    ordinal ≤ owned.id.ordinal := by
  induction rows generalizing ordinal with
  | nil =>
      simp [ownRowsFrom] at member
  | cons row rows inductionHypothesis =>
      simp only [ownRowsFrom, List.mem_cons] at member
      rcases member with equal | tail
      · subst owned
        exact Nat.le_refl ordinal
      · have lower :=
          inductionHypothesis (ordinal + 1) tail
        omega

theorem ownRowsFrom_ids_nodup
    (owner : PhysicalOwner)
    (ordinal : Nat)
    (rows : List Row) :
    ((ownRowsFrom owner ordinal rows).map fun row => row.id).Nodup := by
  induction rows generalizing ordinal with
  | nil =>
      simp [ownRowsFrom]
  | cons row rows inductionHypothesis =>
      simp only [ownRowsFrom, List.map_cons, List.nodup_cons]
      constructor
      · intro member
        have mapped :
            ∃ owned ∈ ownRowsFrom owner (ordinal + 1) rows,
              owned.id = { owner := owner, ordinal := ordinal } := by
          simpa only [List.mem_map] using member
        rcases mapped with ⟨owned, ownedMember, equal⟩
        have lower :=
          ownRowsFrom_ordinal_lower_bound
            owner (ordinal + 1) rows owned ownedMember
        have ordinalEqual :
            owned.id.ordinal = ordinal :=
          congrArg RowId.ordinal equal
        omega
      · exact inductionHypothesis (ordinal + 1)

theorem ownRows_ids_nodup
    (owner : PhysicalOwner)
    (rows : List Row) :
    ((ownRows owner rows).map fun row => row.id).Nodup :=
  ownRowsFrom_ids_nodup owner 0 rows

/-- Satisfaction of a raw equation list before stable occurrence identities
are attached. -/
def RawSatisfies : List Row -> (ColumnId -> Field) -> Prop
  | [], _ => True
  | row :: rows, assignment =>
      row.Holds assignment ∧ RawSatisfies rows assignment

/-- Every sparse dependency of every raw row belongs to an explicit allowed
identity list. -/
def RawRowsSupportedBy
    (allowed : List ColumnId)
    (rows : List Row) : Prop :=
  ∀ row, row ∈ rows ->
    ∀ column, column ∈ row.columnIds -> column ∈ allowed

@[simp] theorem rawSatisfies_nil
    (assignment : ColumnId -> Field) :
    RawSatisfies [] assignment :=
  True.intro

@[simp] theorem rawSatisfies_cons
    (row : Row)
    (rows : List Row)
    (assignment : ColumnId -> Field) :
    RawSatisfies (row :: rows) assignment ↔
      row.Holds assignment ∧ RawSatisfies rows assignment :=
  Iff.rfl

theorem rawSatisfies_append_iff
    (left right : List Row)
    (assignment : ColumnId -> Field) :
    RawSatisfies (left ++ right) assignment ↔
      RawSatisfies left assignment ∧ RawSatisfies right assignment := by
  induction left with
  | nil =>
      simp
  | cons row rows inductionHypothesis =>
      simp only [List.cons_append, rawSatisfies_cons,
        inductionHypothesis]
      constructor
      · rintro ⟨head, tail, right⟩
        exact ⟨⟨head, tail⟩, right⟩
      · rintro ⟨⟨head, tail⟩, right⟩
        exact ⟨head, tail, right⟩

theorem satisfies_ownRowsFrom_iff
    (owner : PhysicalOwner)
    (ordinal : Nat)
    (rows : List Row)
    (assignment : ColumnId -> Field) :
    Satisfies (ownRowsFrom owner ordinal rows) assignment ↔
      RawSatisfies rows assignment := by
  induction rows generalizing ordinal with
  | nil =>
      rfl
  | cons row rows inductionHypothesis =>
      simp only [ownRowsFrom, satisfies_cons, rawSatisfies_cons,
        inductionHypothesis (ordinal + 1)]

theorem satisfies_ownRows_iff
    (owner : PhysicalOwner)
    (rows : List Row)
    (assignment : ColumnId -> Field) :
    Satisfies (ownRows owner rows) assignment ↔
      RawSatisfies rows assignment :=
  satisfies_ownRowsFrom_iff owner 0 rows assignment

theorem ownRows_supported
    (owner : PhysicalOwner)
    (rows : List Row)
    (allowed : List ColumnId)
    (supported : RawRowsSupportedBy allowed rows)
    (owned : OwnedRow)
    (member : owned ∈ ownRows owner rows)
    (column : ColumnId)
    (columnMember : column ∈ owned.columnIds) :
    column ∈ allowed :=
  supported owned.row (ownRows_row_mem owner rows owned member)
    column columnMember

/-- Write an ordered list of values to an equally ordered list of columns.
Earlier entries take precedence, although all compiler-produced lists are
required to be duplicate-free. -/
def writeColumns
    (assignment : ColumnId -> Field) :
    List ColumnId -> List Field -> ColumnId -> Field
  | [], _, id => assignment id
  | _, [], id => assignment id
  | column :: columns, value :: values, id =>
      if id = column then value
      else writeColumns assignment columns values id

theorem writeColumns_of_not_mem
    (assignment : ColumnId -> Field)
    (columns : List ColumnId)
    (values : List Field)
    (id : ColumnId)
    (notMember : id ∉ columns) :
    writeColumns assignment columns values id = assignment id := by
  induction columns generalizing values with
  | nil =>
      simp [writeColumns]
  | cons column columns inductionHypothesis =>
      cases values with
      | nil =>
          simp [writeColumns]
      | cons value values =>
          have different : id ≠ column := by
            intro equal
            apply notMember
            simp [equal]
          have tailNotMember : id ∉ columns := by
            intro member
            exact notMember (List.mem_cons_of_mem column member)
          simp [writeColumns, different,
            inductionHypothesis values tailNotMember]

theorem writeColumns_head
    (assignment : ColumnId -> Field)
    (column : ColumnId)
    (columns : List ColumnId)
    (value : Field)
    (values : List Field) :
    writeColumns assignment (column :: columns) (value :: values) column =
      value := by
  simp [writeColumns]

theorem writeColumns_tail
    (assignment : ColumnId -> Field)
    (column target : ColumnId)
    (columns : List ColumnId)
    (value : Field)
    (values : List Field)
    (different : target ≠ column) :
    writeColumns assignment (column :: columns) (value :: values) target =
      writeColumns assignment columns values target := by
  simp [writeColumns, different]

/-- Pointwise recovery of an equally long, duplicate-free write list. -/
theorem writeColumns_map_eq
    (assignment : ColumnId -> Field)
    (columns : List ColumnId)
    (values : List Field)
    (lengthEqual : columns.length = values.length)
    (nodup : columns.Nodup) :
    columns.map (writeColumns assignment columns values) = values := by
  induction columns generalizing values with
  | nil =>
      cases values with
      | nil => rfl
      | cons value values =>
          simp at lengthEqual
  | cons column columns inductionHypothesis =>
      cases values with
      | nil =>
          simp at lengthEqual
      | cons value values =>
          simp only [List.length_cons, Nat.succ.injEq] at lengthEqual
          have split :
              column ∉ columns ∧ columns.Nodup := by
            simpa only [List.nodup_cons] using nodup
          have tailNodup : columns.Nodup := split.2
          have headNotMem : column ∉ columns := split.1
          have head :
              writeColumns assignment (column :: columns)
                  (value :: values) column =
                value :=
            writeColumns_head assignment column columns value values
          have tail :
              columns.map
                  (writeColumns assignment (column :: columns)
                    (value :: values)) =
                values := by
            have pointwise :
                ∀ target ∈ columns,
                  writeColumns assignment (column :: columns)
                      (value :: values) target =
                    writeColumns assignment columns values target := by
              intro target member
              apply writeColumns_tail
              intro equal
              subst target
              exact headNotMem member
            rw [List.map_congr_left pointwise]
            exact inductionHypothesis values lengthEqual tailNodup
          simpa only [List.map_cons, List.cons.injEq] using
            And.intro head tail

/-- Ordered bundle values after writing the exact ordered bundle identities. -/
theorem bundle_values_writeColumns
    {layout : Layout}
    (bundle : ColumnBundle layout)
    (assignment : ColumnId -> Field)
    (values : List Field)
    (lengthEqual : values.length = layout.owners.length)
    (nodup : bundle.ids.Nodup) :
    bundle.values
        (writeColumns assignment bundle.ids values) =
      values := by
  have recovered :=
    writeColumns_map_eq assignment bundle.ids values
      (by
        rw [ColumnBundle.ids, List.length_map, bundle.length_eq,
          ← lengthEqual])
      nodup
  simpa only [ColumnBundle.values, ColumnBundle.ids, List.map_map,
    Function.comp_apply] using recovered

/-- A write completion changes no identity outside its explicit write list. -/
theorem writeColumns_changesOnly
    (assignment : ColumnId -> Field)
    (columns : List ColumnId)
    (values : List Field) :
    ChangesOnly columns assignment
      (writeColumns assignment columns values) := by
  intro id notMember
  exact writeColumns_of_not_mem assignment columns values id notMember

/-- Disjoint visible identities are preserved by a write completion. -/
theorem writeColumns_agreesOn
    (assignment : ColumnId -> Field)
    (columns visible : List ColumnId)
    (values : List Field)
    (disjoint : IdsDisjoint columns visible) :
    AgreesOn visible assignment
      (writeColumns assignment columns values) := by
  intro id member
  apply writeColumns_of_not_mem
  intro written
  exact disjoint id written member

/-- Read one exact physical coordinate from a bundle using the layout's
coordinate index, never a fallback column. -/
def bundleColumn
    {layout : Layout}
    (bundle : ColumnBundle layout)
    (coordinate : Fin layout.owners.length) : OwnedColumn :=
  bundle.columns.get
    ⟨coordinate.val, by
      rw [bundle.length_eq]
      exact coordinate.isLt⟩

@[simp] theorem bundleColumn_id_mem
    {layout : Layout}
    (bundle : ColumnBundle layout)
    (coordinate : Fin layout.owners.length) :
    (bundleColumn bundle coordinate).id ∈ bundle.ids := by
  apply List.mem_map.mpr
  refine ⟨bundleColumn bundle coordinate, ?_, rfl⟩
  unfold bundleColumn
  exact List.get_mem bundle.columns _

/-- The exact coordinate read through `bundleColumn` agrees with the ordered
field string read through `ColumnBundle.values`. -/
theorem bundle_values_get
    {layout : Layout}
    (bundle : ColumnBundle layout)
    (assignment : ColumnId -> Field)
    (coordinate : Fin layout.owners.length) :
    (bundle.values assignment).get
        ⟨coordinate.val, by
          rw [ColumnBundle.values_length]
          exact coordinate.isLt⟩ =
      assignment (bundleColumn bundle coordinate).id := by
  simp only [ColumnBundle.values, bundleColumn, List.get_eq_getElem,
    List.getElem_map]

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.DirectCalls
