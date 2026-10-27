import CompressionLean.Ports.Shrink.Layout
import CompressionLean.Shrink.Ccs
import Mathlib.Data.Nat.Log

/-!
Owns what a shape run must meet for its CCS (`Ports/Shrink/Layout.lean`) to
fit the cubes: a well-formed template (`Template.WellFormed`), wires inside
the run (`Run.Valid`), and the proof (`ccs_fits`). The lowering proves that
a recorded run meets them (`Lowering/`).
-/

namespace CompressionLean.Shrink

/-- The template's entries name its rows and cells. -/
structure Template.WellFormed {F : Type} (T : Template F) : Prop where
  row_lt : ∀ e ∈ T.entries, e.row < T.rows
  cell_lt : ∀ e ∈ T.entries, ∀ k, e.cell = some k → k < T.cells

/-- A wire names a cell of the run. The recorder creates no other wire
(`record.rs::Recorder`). -/
def Wire.Valid {F : Type} (T : Template F) (run : Run F) : Wire → Prop
  | .glue c => c < run.cells
  | .block b k => b < run.blocks ∧ k < T.cells
  | .pub i => i ≤ run.publics.length

/-- Every term of every glue row names a cell of the run. -/
def Run.Valid {F : Type} [Field F] (T : Template F) (run : Run F) : Prop :=
  ∀ row ∈ run.rows, ∀ side, ∀ t ∈ row.side side, t.1.Valid T run

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
