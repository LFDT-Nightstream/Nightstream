import CompressionLean.Ports.Shrink.Ccs
import Mathlib.Data.Nat.Log

namespace CompressionLean.Shrink

inductive Wire where
  | glue (cell : ℕ)
  | block (b cell : ℕ)
  | pub (index : ℕ)

structure GlueRow (F : Type) where
  side : Fin 7 → List (Wire × F)

def GlueRow.ofCall {F : Type} (products : List (List (Wire × F) × List (Wire × F))) (linear : List (Wire × F)) :
    GlueRow F :=
  ⟨fun side =>
    if (side : ℕ) = 6 then linear
    else
      match products[(side : ℕ) / 2]? with
      | some pair => if (side : ℕ) % 2 = 0 then pair.1 else pair.2
      | none => []⟩

structure Form (F : Type) where
  terms : List (Wire × F)
  constant : F

structure R1csRow (F : Type) where
  a : List (ℕ × F)
  b : List (ℕ × F)
  c : List (ℕ × F)

open Classical in
/-- A column past `columns` reads the zero form; Rust asserts that none is (`r1cs.rs::R1cs::push`, `record.rs::Recorder::r1cs`). -/
noncomputable def expand {F : Type} [Field F] (side : List (ℕ × F)) (columns : List (Form F)) :
    List (Wire × F) :=
  side.flatMap fun t =>
    (columns.getD t.1 ⟨[], 0⟩).terms.map (fun u => (u.1, u.2 * t.2)) ++
      if (columns.getD t.1 ⟨[], 0⟩).constant ≠ 0 then [(.pub 0, (columns.getD t.1 ⟨[], 0⟩).constant * t.2)]
      else []

inductive Call (F : Type) where
  | row (products : List (List (Wire × F) × List (Wire × F))) (linear : List (Wire × F))
      (three : products.length ≤ 3)
  | r1cs (rows : List (R1csRow F)) (columns : List (Form F))

noncomputable def Call.rows {F : Type} [Field F] : Call F → List (GlueRow F)
  | .row products linear _ => [GlueRow.ofCall products linear]
  | .r1cs rows columns =>
      rows.map fun r => GlueRow.ofCall [(expand r.a columns, expand r.b columns)] (expand r.c columns)

structure TemplateEntry (F : Type) where
  matrix : Fin 8
  row : ℕ
  cell : Option ℕ
  coef : F

structure Template (F : Type) where
  rows : ℕ
  cells : ℕ
  entries : List (TemplateEntry F)

structure Counts where
  blocks : ℕ
  glueRows : ℕ
  glueCells : ℕ
  publics : ℕ

structure Run (F : Type) where
  calls : List (Call F)
  cells : ℕ
  blocks : ℕ
  publics : List F

noncomputable def Run.rows {F : Type} [Field F] (run : Run F) : List (GlueRow F) :=
  run.calls.flatMap Call.rows

noncomputable def Run.counts {F : Type} [Field F] (run : Run F) : Counts :=
  ⟨run.blocks, run.rows.length, run.cells, run.publics.length⟩

def wide (size : ℕ) : ℕ := 2 ^ Nat.log 2 size

def narrow (size : ℕ) : ℕ :=
  if size - wide size = 0 then 0 else 2 ^ Nat.clog 2 (size - wide size)

def slot (size count b k : ℕ) : ℕ :=
  if k < wide size then wide size * b + k else wide size * count + narrow size * b + (k - wide size)

def partsEnd (size count : ℕ) : ℕ := wide size * count + narrow size * count

def variablesFor (count : ℕ) : ℕ := max 1 (Nat.clog 2 count)

namespace Layout

variable {F : Type} (T : Template F) (s : Counts)

def rowVars : ℕ := variablesFor (partsEnd T.rows s.blocks + s.glueRows)

def cellVars : ℕ := variablesFor (max (partsEnd T.cells s.blocks + s.glueCells) (s.publics + 1))

def blockRow (b k : ℕ) : ℕ := slot T.rows s.blocks b k

def blockCell (b k : ℕ) : ℕ := slot T.cells s.blocks b k

def glueRow (g : ℕ) : ℕ := partsEnd T.rows s.blocks + g

def one : ℕ := 2 ^ cellVars T s

def column : Wire → ℕ
  | .glue c => partsEnd T.cells s.blocks + c
  | .block b k => blockCell T s b k
  | .pub i => one T s + i

def entryColumn (b : ℕ) : Option ℕ → ℕ
  | none => one T s
  | some k => blockCell T s b k

def blockEntries : List (Entry F) :=
  (List.range s.blocks).flatMap fun b =>
    T.entries.map fun e => ⟨e.matrix, blockRow T s b e.row, entryColumn T s b e.cell, e.coef⟩

def rowEntries (g : ℕ) (row : GlueRow F) : List (Entry F) :=
  (List.finRange 7).flatMap fun side =>
    (row.side side).map fun t => ⟨side.succ, glueRow T s g, column T s t.1, t.2⟩

def glueEntries : ℕ → List (GlueRow F) → List (Entry F)
  | _, [] => []
  | g, row :: rest => rowEntries T s g row ++ glueEntries (g + 1) rest

end Layout

noncomputable def Run.ccs {F : Type} [Field F] (T : Template F) (s : Counts) (run : Run F) : Ccs F :=
  ⟨Layout.rowVars T s, Layout.cellVars T s,
    Layout.blockEntries T s ++ Layout.glueEntries T s 0 run.rows, run.publics⟩

deriving instance DecidableEq, Repr for Wire

section Computed

variable {F : Type} [Field F] [DecidableEq F]

namespace Form

def const (c : F) : Form F := ⟨[], c⟩

def wire (w : Wire) : Form F := ⟨[(w, 1)], 0⟩

def isConstant (f : Form F) : Bool := f.terms.isEmpty

def rowTerms (f : Form F) : List (Wire × F) := f.terms ++ if f.constant = 0 then [] else [(.pub 0, f.constant)]

end Form

def expandArray (side : List (ℕ × F)) (columns : Array (Form F)) : List (Wire × F) :=
  side.flatMap fun t =>
    (columns.getD t.1 ⟨[], 0⟩).terms.map (fun u => (u.1, u.2 * t.2)) ++
      if (columns.getD t.1 ⟨[], 0⟩).constant ≠ 0 then [(.pub 0, (columns.getD t.1 ⟨[], 0⟩).constant * t.2)]
      else []

def Call.computedRows : Call F → List (GlueRow F)
  | .row products linear _ => [GlueRow.ofCall products linear]
  | .r1cs rows columns =>
      let columns := columns.toArray
      rows.map fun r =>
        GlueRow.ofCall [(expandArray r.a columns, expandArray r.b columns)] (expandArray r.c columns)

end Computed

end CompressionLean.Shrink
