import CompressionLean.Shrink.Ccs
import Mathlib.Data.Nat.Log

/-!
Owns where the rows and the cells of a shape run sit in the shrink CCS
(`crates/neo-spartan/src/shrink/layout.rs`), and so the CCS of a run
(`Run.ccs`).

- One block kind, the Poseidon2 permutation (`KINDS`,
  `circuit/block.rs:26`): a template of `rows` rows over `cells` cells (166
  and 182, `circuit/poseidon2.rs:20-24`).
- A range of `size` template indices has a wide part of `2^⌊log2 size⌋`
  slots per block and a narrow part for the rest, rounded up to a power of
  two (`split`, `layout.rs:56-61`). The wide part starts at `0` and the
  narrow part after all wide slots (`place`, `layout.rs:64-81`; with one
  kind, the stable sort keeps the wide part first). Index `k` of block `b`
  sits at `start + b·stride + k'` in its part (`layout.rs:116-120`).
- Glue rows and glue cells follow the block parts (`layout.rs:149-165`).
- The constant one sits at column `2^cellVars` and public word `i` at
  `2^cellVars + i` (`layout.rs:154-164`).
- A run records its sink calls that emit glue rows, in order (`Call`,
  `circuit/record.rs:136-152`). Its glue rows are the ones that `Trace`
  stores for them (`Run.rows`).

The entries of a run's CCS are the block entries, then the glue entries.
Their two sums are the verifier's `block_part` and its shape run's
`total` (`mod.rs:275`).
-/

namespace CompressionLean.Shrink

/-- A witness cell or a public word (`circuit/record.rs:24-33`); `pub 0` is
the constant one. -/
inductive Wire where
  | glue (cell : ℕ)
  | block (b cell : ℕ)
  | pub (index : ℕ)

/-- One glue row (`record.rs:7`; `Trace::row`, `record.rs:491-507`): the
terms of its sides `A_0, B_0, A_1, B_1, A_2, B_2, C`. Side `s` is matrix
`s + 1` (`mod.rs:379`, `mod.rs:432`). The row holds when
`Σ_i (A_i·z)·(B_i·z) - C·z = 0`. -/
structure GlueRow (F : Type) where
  side : Fin 7 → List (Wire × F)

/-- The row that `Trace::row` stores for a row call (`record.rs:492-507`):
product `i` gives sides `2i` and `2i + 1`, the linear terms side `6`, and a
missing product leaves its sides empty. -/
def GlueRow.ofCall {F : Type} (products : List (List (Wire × F) × List (Wire × F))) (linear : List (Wire × F)) :
    GlueRow F :=
  ⟨fun side =>
    if (side : ℕ) = 6 then linear
    else
      match products[(side : ℕ) / 2]? with
      | some pair => if (side : ℕ) % 2 = 0 then pair.1 else pair.2
      | none => []⟩

/-- `Σ coefficient·wire + constant` (`Form`, `record.rs:39-46`). -/
structure Form (F : Type) where
  terms : List (Wire × F)
  constant : F

/-- One row `(Σ a)·(Σ b) = Σ c` of a fixed R1CS system, each side as
`(column, coefficient)` terms (`R1cs::sides`, `circuit/r1cs.rs:59-65`). -/
structure R1csRow (F : Type) where
  a : List (ℕ × F)
  b : List (ℕ × F)
  c : List (ℕ × F)

open Classical in
/-- `expand` (`record.rs:118-134`): each column's form times its
coefficient, the form's constant on the one column when it is nonzero. A
column past `columns` reads the zero form; Rust asserts that none is
(`r1cs.rs:42`, `record.rs:451`). -/
noncomputable def expand {F : Type} [Field F] (side : List (ℕ × F)) (columns : List (Form F)) :
    List (Wire × F) :=
  side.flatMap fun t =>
    (columns.getD t.1 ⟨[], 0⟩).terms.map (fun u => (u.1, u.2 * t.2)) ++
      if (columns.getD t.1 ⟨[], 0⟩).constant ≠ 0 then [(.pub 0, (columns.getD t.1 ⟨[], 0⟩).constant * t.2)]
      else []

/-- A sink call of the shape run that emits glue rows (`Sink`,
`record.rs:136-152`): one row of at most three products and a linear side
(`Trace::row` asserts the bound, `record.rs:493`), or the rows of a fixed
R1CS system over form columns. -/
inductive Call (F : Type) where
  | row (products : List (List (Wire × F) × List (Wire × F))) (linear : List (Wire × F))
      (three : products.length ≤ 3)
  | r1cs (rows : List (R1csRow F)) (columns : List (Form F))

/-- The glue rows of a call as `Trace` stores them: the default `Sink::r1cs`
emits each R1CS row as a row call of one product (`record.rs:147-152`), and
`Trace::r1cs` builds the same rows (`record.rs:521-572`). -/
noncomputable def Call.rows {F : Type} [Field F] : Call F → List (GlueRow F)
  | .row products linear _ => [GlueRow.ofCall products linear]
  | .r1cs rows columns =>
      rows.map fun r => GlueRow.ofCall [(expand r.a columns, expand r.b columns)] (expand r.c columns)

/-- One template entry (`circuit/block.rs:28-36`): matrix, template row,
block cell (`none` is the constant one), coefficient. -/
structure TemplateEntry (F : Type) where
  matrix : Fin 8
  row : ℕ
  cell : Option ℕ
  coef : F

/-- The block template (`circuit/block.rs:38-70`). -/
structure Template (F : Type) where
  rows : ℕ
  cells : ℕ
  entries : List (TemplateEntry F)

/-- The template's entries name its rows and cells. -/
structure Template.WellFormed {F : Type} (T : Template F) : Prop where
  row_lt : ∀ e ∈ T.entries, e.row < T.rows
  cell_lt : ∀ e ∈ T.entries, ∀ k, e.cell = some k → k < T.cells

/-- The counts of a recording (`Shape`, `layout.rs:27-35`). -/
structure Counts where
  blocks : ℕ
  glueRows : ℕ
  glueCells : ℕ
  publics : ℕ

/-- What the verifier's shape run of one program records
(`mod.rs:263-271`, `evaluate.rs`): its calls that emit glue rows in order,
its glue cells, its blocks and its public words. -/
structure Run (F : Type) where
  calls : List (Call F)
  cells : ℕ
  blocks : ℕ
  publics : List F

/-- The run's glue rows, in order. -/
noncomputable def Run.rows {F : Type} [Field F] (run : Run F) : List (GlueRow F) :=
  run.calls.flatMap Call.rows

/-- The counts that the verifier compares with the shape (`mod.rs:266-271`). -/
noncomputable def Run.counts {F : Type} [Field F] (run : Run F) : Counts :=
  ⟨run.blocks, run.rows.length, run.cells, run.publics.length⟩

/-- A wire names a cell of the run. The recorder creates no other wire
(`record.rs:189-303`). -/
def Wire.Valid {F : Type} (T : Template F) (run : Run F) : Wire → Prop
  | .glue c => c < run.cells
  | .block b k => b < run.blocks ∧ k < T.cells
  | .pub i => i ≤ run.publics.length

/-- Every term of every glue row names a cell of the run. -/
def Run.Valid {F : Type} [Field F] (T : Template F) (run : Run F) : Prop :=
  ∀ row ∈ run.rows, ∀ side, ∀ t ∈ row.side side, t.1.Valid T run

/-- The wide part of `size` indices: `2^⌊log2 size⌋` (`layout.rs:58`). -/
def wide (size : ℕ) : ℕ := 2 ^ Nat.log 2 size

/-- The narrow part: the rest, rounded up to a power of two; none if the
rest is empty (`layout.rs:59-60`). -/
def narrow (size : ℕ) : ℕ :=
  if size - wide size = 0 then 0 else 2 ^ Nat.clog 2 (size - wide size)

/-- Index `k` of block `b` among `count` blocks (`layout.rs:116-120`). -/
def slot (size count b k : ℕ) : ℕ :=
  if k < wide size then wide size * b + k else wide size * count + narrow size * b + (k - wide size)

/-- The end of both parts (`place`, `layout.rs:80`). -/
def partsEnd (size count : ℕ) : ℕ := wide size * count + narrow size * count

/-- `max(1, ⌈log2 count⌉)` (`variables`, `layout.rs:83-85`). -/
def variablesFor (count : ℕ) : ℕ := max 1 (Nat.clog 2 count)

namespace Layout

variable {F : Type} (T : Template F) (s : Counts)

/-- The row variables (`layout.rs:111`). -/
def rowVars : ℕ := variablesFor (partsEnd T.rows s.blocks + s.glueRows)

/-- The variables of `w`; `z` has one more (`layout.rs:112`). -/
def cellVars : ℕ := variablesFor (max (partsEnd T.cells s.blocks + s.glueCells) (s.publics + 1))

/-- The row of template row `k` of block `b` (`layout.rs:140-142`). -/
def blockRow (b k : ℕ) : ℕ := slot T.rows s.blocks b k

/-- The column of template cell `k` of block `b` (`layout.rs:145-147`). -/
def blockCell (b k : ℕ) : ℕ := slot T.cells s.blocks b k

/-- The row of glue row `g` (`layout.rs:149-151`). -/
def glueRow (g : ℕ) : ℕ := partsEnd T.rows s.blocks + g

/-- The column of the constant one (`layout.rs:154-156`). -/
def one : ℕ := 2 ^ cellVars T s

/-- The column of a wire (`layout.rs:159-165`). -/
def column : Wire → ℕ
  | .glue c => partsEnd T.cells s.blocks + c
  | .block b k => blockCell T s b k
  | .pub i => one T s + i

/-- The column of a template entry's cell in block `b` (`layout.rs:168-170`). -/
def entryColumn (b : ℕ) : Option ℕ → ℕ
  | none => one T s
  | some k => blockCell T s b k

/-- Every block's template entries, placed (`block_part`, `layout.rs:174-227`). -/
def blockEntries : List (Entry F) :=
  (List.range s.blocks).flatMap fun b =>
    T.entries.map fun e => ⟨e.matrix, blockRow T s b e.row, entryColumn T s b e.cell, e.coef⟩

/-- The terms of glue row `g`, placed (`Evaluate::row`, `evaluate.rs:73-80`). -/
def rowEntries (g : ℕ) (row : GlueRow F) : List (Entry F) :=
  (List.finRange 7).flatMap fun side =>
    (row.side side).map fun t => ⟨side.succ, glueRow T s g, column T s t.1, t.2⟩

/-- The glue rows from row number `g` on. -/
def glueEntries : ℕ → List (GlueRow F) → List (Entry F)
  | _, [] => []
  | g, row :: rest => rowEntries T s g row ++ glueEntries (g + 1) rest

end Layout

/-- The CCS of a run under the layout of the shape `s`: the block entries,
then the glue entries (`mod.rs:275`). -/
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

/-- A glue entry from row number `g` on sits in some later glue row and names
the column of one of that row's wires. -/
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

/-- On a run with the shape's counts, every entry sits inside the cubes. -/
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
