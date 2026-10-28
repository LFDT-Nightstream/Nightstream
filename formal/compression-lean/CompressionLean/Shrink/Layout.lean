import CompressionLean.Shrink.Ccs
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

structure Template.WellFormed {F : Type} (T : Template F) : Prop where
  row_lt : ∀ e ∈ T.entries, e.row < T.rows
  cell_lt : ∀ e ∈ T.entries, ∀ k, e.cell = some k → k < T.cells

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

def Wire.Valid {F : Type} (T : Template F) (run : Run F) : Wire → Prop
  | .glue c => c < run.cells
  | .block b k => b < run.blocks ∧ k < T.cells
  | .pub i => i ≤ run.publics.length

def Run.Valid {F : Type} [Field F] (T : Template F) (run : Run F) : Prop :=
  ∀ row ∈ run.rows, ∀ side, ∀ t ∈ row.side side, t.1.Valid T run

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

section Fits

theorem rest_le_narrow (size : ℕ) : size - wide size ≤ narrow size := by
  unfold narrow
  split_ifs with empty
  · omega
  · exact Nat.le_pow_clog (by norm_num) _

theorem slot_lt {size count b k : ℕ} (hb : b < count) (hk : k < size) :
    slot size count b k < partsEnd size count := by
  have rest := rest_le_narrow size
  have wideStep : wide size * (b + 1) ≤ wide size * count := Nat.mul_le_mul_left _ hb
  have narrowStep : narrow size * (b + 1) ≤ narrow size * count := Nat.mul_le_mul_left _ hb
  rw [Nat.mul_succ] at wideStep narrowStep
  unfold slot partsEnd
  split_ifs with inWide <;> omega

theorem le_pow_variablesFor (count : ℕ) : count ≤ 2 ^ variablesFor count :=
  (Nat.le_pow_clog (by norm_num) count).trans
    (Nat.pow_le_pow_right (by norm_num) (le_max_right _ _))

variable {F : Type} (T : Template F) (s : Counts)

theorem mem_glueEntries {e : Entry F} :
    ∀ (g : ℕ) (rows : List (GlueRow F)), e ∈ Layout.glueEntries T s g rows →
      ∃ k, k < rows.length ∧ e.row = Layout.glueRow T s (g + k) ∧
        ∃ row ∈ rows, ∃ side, ∃ t ∈ row.side side, e.col = Layout.column T s t.1
  | g, [] => by simp [Layout.glueEntries]
  | g, row :: rest => by
      simp only [Layout.glueEntries, List.mem_append]
      rintro (here | there)
      · simp only [Layout.rowEntries, List.mem_flatMap, List.mem_map] at here
        obtain ⟨side, -, t, member, rfl⟩ := here
        exact ⟨0, by simp, by simp, row, List.mem_cons_self, side, t, member, rfl⟩
      · obtain ⟨k, lt, rowEq, r, member, side, t, tm, colEq⟩ := mem_glueEntries (g + 1) rest there
        refine ⟨k + 1, by simp; omega, by rw [rowEq]; congr 1; omega, r, List.mem_cons_of_mem _ member,
          side, t, tm, colEq⟩

theorem ccs_fits [Field F] (run : Run F) (wf : T.WellFormed) (valid : run.Valid T) (counts : run.counts = s) :
    (run.ccs T s).Fits := by
  have blocks : run.blocks = s.blocks := congrArg Counts.blocks counts
  have glueRows : run.rows.length = s.glueRows := congrArg Counts.glueRows counts
  have cells : run.cells = s.glueCells := congrArg Counts.glueCells counts
  have publics : run.publics.length = s.publics := congrArg Counts.publics counts
  have rowsEnd : partsEnd T.rows s.blocks + s.glueRows ≤ 2 ^ Layout.rowVars T s := le_pow_variablesFor _
  have cellsEnd : max (partsEnd T.cells s.blocks + s.glueCells) (s.publics + 1) ≤ 2 ^ Layout.cellVars T s :=
    le_pow_variablesFor _
  have cellsLe := le_max_left (partsEnd T.cells s.blocks + s.glueCells) (s.publics + 1)
  have publicLe := le_max_right (partsEnd T.cells s.blocks + s.glueCells) (s.publics + 1)
  have double : 2 ^ (Layout.cellVars T s + 1) = 2 ^ Layout.cellVars T s * 2 := pow_succ _ _
  have positive : 0 < 2 ^ Layout.cellVars T s := pow_pos (by norm_num) _
  refine ⟨fun e member => ?_, fun e member => ?_, ?_⟩
  · show e.row < 2 ^ Layout.rowVars T s
    rcases List.mem_append.mp member with block | glue
    · simp only [Layout.blockEntries, List.mem_flatMap, List.mem_range, List.mem_map] at block
      obtain ⟨b, hb, te, tm, rfl⟩ := block
      have := slot_lt (size := T.rows) hb (wf.row_lt te tm)
      simp only [Layout.blockRow]
      omega
    · obtain ⟨k, lt, rowEq, -⟩ := mem_glueEntries T s 0 run.rows glue
      rw [rowEq, Layout.glueRow]
      omega
  · show e.col < 2 ^ (Layout.cellVars T s + 1)
    rcases List.mem_append.mp member with block | glue
    · simp only [Layout.blockEntries, List.mem_flatMap, List.mem_range, List.mem_map] at block
      obtain ⟨b, hb, te, tm, rfl⟩ := block
      cases cell : te.cell with
      | none =>
          simp only [Layout.entryColumn, Layout.one]
          omega
      | some c =>
          have := slot_lt (size := T.cells) hb (wf.cell_lt te tm c cell)
          simp only [Layout.entryColumn, Layout.blockCell]
          omega
    · obtain ⟨-, -, -, row, rm, side, t, tm, colEq⟩ := mem_glueEntries T s 0 run.rows glue
      have wire := valid row rm side t tm
      rw [colEq]
      cases name : t.1 with
      | glue c =>
          rw [name] at wire
          simp only [Wire.Valid] at wire
          simp only [Layout.column]
          omega
      | block b c =>
          rw [name] at wire
          obtain ⟨hb, hc⟩ := wire
          have := slot_lt (size := T.cells) (count := s.blocks) (blocks ▸ hb) hc
          simp only [Layout.column, Layout.blockCell]
          omega
      | pub i =>
          rw [name] at wire
          simp only [Wire.Valid] at wire
          simp only [Layout.column, Layout.one]
          omega
  · show run.publics.length + 1 ≤ 2 ^ Layout.cellVars T s
    omega

end Fits

end CompressionLean.Shrink
