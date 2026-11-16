import NightstreamFPrime.Circuit.VariableSupport

/-! Rows cut into marked permutations and free rows (SPEC.md section 4.4). A circuit marks each
Poseidon2 permutation that it places (`Op.mark`). The exporter packs a marked permutation into a
block whose 1,080 inner locals have no column, so no other row may read them. `Segmented live
window marks rows` states this shape: `rows` are, in order, free rows that read only `live`
columns and the rows `window mark` of each mark. A step circuit's rows have this shape with `live`
the columns that are no inner local of a mark (`Export.Generic.MarkedBlocks`). Each gadget states
it for its own rows, for every `live` that its inputs read and that holds on its locals outside
its marks' inner locals (`Clean`). -/

set_option autoImplicit false

namespace NightstreamFPrime.Circuit

/-- An expression that is a linear combination of columns and a constant: the shape of a lane of
a packed block. -/
inductive Expr.Affine : Expr → Prop
  | var (column : Nat) : Expr.Affine (.var column)
  | const (value : Spec.F) : Expr.Affine (.const value)
  | add {left right : Expr} : Expr.Affine left → Expr.Affine right → Expr.Affine (.add left right)
  | scaleLeft (factor : Spec.F) {expression : Expr} :
      Expr.Affine expression → Expr.Affine (.mul (.const factor) expression)
  | scaleRight (factor : Spec.F) {expression : Expr} :
      Expr.Affine expression → Expr.Affine (.mul expression (.const factor))

theorem Expr.Affine.sub {left right : Expr} (leftAffine : left.Affine) (rightAffine : right.Affine) :
    (left - right).Affine :=
  .add leftAffine (.scaleLeft _ rightAffine)

/-- The 1,080 inner locals of a marked permutation: all its locals but the 16 outputs. -/
def PermutationMark.Inner (mark : PermutationMark) (column : Nat) : Prop :=
  mark.start ≤ column ∧ column < mark.start + 1080

/-- A column that is an inner local of one of `marks`. -/
def InnerOf (marks : List PermutationMark) (column : Nat) : Prop :=
  ∃ mark ∈ marks, mark.Inner column

/-- What a packed block needs of a mark: 16 affine lanes that read only `live` columns before the
mark's first local. -/
structure PermutationMark.Lanes (live : Nat → Prop) (mark : PermutationMark) : Prop where
  length : mark.lanes.length = 16
  affine : ∀ lane ∈ mark.lanes, lane.Affine
  reads : ∀ lane ∈ mark.lanes, lane.VarsSatisfy fun column => column < mark.start ∧ live column

theorem PermutationMark.Lanes.mono {live larger : Nat → Prop} {mark : PermutationMark}
    (lanes : mark.Lanes live) (includes : ∀ column, live column → larger column) :
    mark.Lanes larger :=
  ⟨lanes.length, lanes.affine, fun lane member =>
    (lanes.reads lane member).mono lane fun _ inside => ⟨inside.1, includes _ inside.2⟩⟩

/-- `rows` are, in order, free rows that read only `live` columns and the rows `window mark` of
each of `marks`. -/
inductive Segmented (live : Nat → Prop) (window : PermutationMark → List Expr) :
    List PermutationMark → List Expr → Prop
  | nil : Segmented live window [] []
  | free {marks : List PermutationMark} {rows : List Expr} (row : Expr)
      (reads : row.VarsSatisfy live) (rest : Segmented live window marks rows) :
      Segmented live window marks (row :: rows)
  | block {marks : List PermutationMark} {rows : List Expr} (mark : PermutationMark)
      (rest : Segmented live window marks rows) :
      Segmented live window (mark :: marks) (window mark ++ rows)

namespace Segmented

variable {live : Nat → Prop} {window : PermutationMark → List Expr}

/-- Free rows that read only `live` columns. -/
theorem ofFree {rows : List Expr} (reads : ∀ row ∈ rows, row.VarsSatisfy live) :
    Segmented live window [] rows := by
  induction rows with
  | nil => exact .nil
  | cons row rows ih =>
      exact .free row (reads row List.mem_cons_self)
        (ih fun other member => reads other (List.mem_cons_of_mem _ member))

/-- Free rows before a segmented tail. -/
theorem freeAppend {marks : List PermutationMark} {rest rows : List Expr}
    (reads : ∀ row ∈ rows, row.VarsSatisfy live) (tail : Segmented live window marks rest) :
    Segmented live window marks (rows ++ rest) := by
  induction rows with
  | nil => exact tail
  | cons row rows ih =>
      exact .free row (reads row List.mem_cons_self)
        (ih fun other member => reads other (List.mem_cons_of_mem _ member))

theorem append {firstMarks secondMarks : List PermutationMark} {firstRows secondRows : List Expr}
    (first : Segmented live window firstMarks firstRows)
    (second : Segmented live window secondMarks secondRows) :
    Segmented live window (firstMarks ++ secondMarks) (firstRows ++ secondRows) := by
  induction first with
  | nil => exact second
  | free row reads _ ih =>
      rw [List.cons_append]
      exact .free row reads ih
  | block mark _ ih =>
      rw [List.cons_append, List.append_assoc]
      exact .block mark ih

/-- The rows of marked permutations alone. -/
theorem ofMarks : ∀ (marks : List PermutationMark),
    Segmented live window marks (marks.flatMap window)
  | [] => .nil
  | mark :: marks => by
      rw [List.flatMap_cons]
      exact .block mark (ofMarks marks)

theorem mono {larger : Nat → Prop} (includes : ∀ column, live column → larger column)
    {marks : List PermutationMark} {rows : List Expr} (segmented : Segmented live window marks rows) :
    Segmented larger window marks rows := by
  induction segmented with
  | nil => exact .nil
  | free row reads _ ih => exact .free row (reads.mono row includes) ih
  | block mark _ ih => exact .block mark ih

/-- The same rows with another window that agrees on the marks. -/
theorem window_congr {other : PermutationMark → List Expr} {marks : List PermutationMark}
    {rows : List Expr} (segmented : Segmented live window marks rows)
    (same : ∀ mark ∈ marks, window mark = other mark) : Segmented live other marks rows := by
  induction segmented with
  | nil => exact .nil
  | free row reads _ ih => exact .free row reads (ih same)
  | block mark _ ih =>
      rw [same mark List.mem_cons_self]
      exact .block mark (ih fun other member => same other (List.mem_cons_of_mem _ member))

/-- With no marks, every row is free. -/
theorem free_of_nil {rows : List Expr} (segmented : Segmented live window [] rows) :
    ∀ row ∈ rows, row.VarsSatisfy live := by
  generalize empty : ([] : List PermutationMark) = marks at segmented
  induction segmented with
  | nil => simp
  | free row reads _ ih =>
      intro other member
      rcases List.mem_cons.mp member with rfl | member
      · exact reads
      · exact ih empty other member
  | block mark _ _ => simp at empty

/-- The rows before the first mark's rows are free. -/
theorem split_head {mark : PermutationMark} {marks : List PermutationMark} {rows : List Expr}
    (segmented : Segmented live window (mark :: marks) rows) :
    ∃ frees rest, rows = frees ++ window mark ++ rest ∧ (∀ row ∈ frees, row.VarsSatisfy live) ∧
      Segmented live window marks rest := by
  generalize head : mark :: marks = all at segmented
  induction segmented with
  | nil => simp at head
  | free row reads _ ih =>
      obtain ⟨frees, rest, rfl, freeReads, tail⟩ := ih head
      refine ⟨row :: frees, rest, rfl, fun other member => ?_, tail⟩
      rcases List.mem_cons.mp member with rfl | member
      · exact reads
      · exact freeReads other member
  | block other tail _ =>
      obtain ⟨rfl, rfl⟩ := List.cons.inj head
      exact ⟨[], _, rfl, by simp, tail⟩

end Segmented

end NightstreamFPrime.Circuit
