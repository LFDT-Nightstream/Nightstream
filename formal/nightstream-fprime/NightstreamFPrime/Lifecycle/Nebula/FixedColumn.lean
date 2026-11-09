import NightstreamFPrime.Lifecycle.Nebula.PortApp
import NightstreamFPrime.Lifecycle.Nebula.RowExprs

/-! Owns the rewrite of an application's rows that reads a pinned local variable as its constant
(SPEC.md section 7.6). When the row `var column - const value` is in the rows, every other row can
read `value` in place of variable `column`: the rewritten rows hold exactly when the rows hold
(`holds_fix_iff`), so the rewritten application keeps the model, its port coverage and its
completeness (`PortApp.fixColumn`). Rust `nebula::R1csApplication` reads its constant-one column
so. A row `x · (x − one)` then reads only `x` and constants, which the slot rule needs for a bit
slot (`Export.Generic.SlotLayout.forcedColumn?`). -/

namespace NightstreamFPrime.Lifecycle.Nebula

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Nebula
open NightstreamFPrime.Circuit

variable {p : Geometry}

/-- The wires that read variable `column` as `value` and every other variable as itself. -/
def fixWires (column : ℕ) (value : F) (k : ℕ) : Circuit.Expr :=
  if k = column then .const value else .var k

/-- The row that pins variable `column` to `value`. -/
def pinRow (column : ℕ) (value : F) : Circuit.Expr :=
  .var column - .const value

/-- `rows` with variable `column` read as `value`. -/
def fixRows (column : ℕ) (value : F) (rows : List Circuit.Expr) : List Circuit.Expr :=
  rows.map (Expr.subst (fixWires column value))

theorem pinRow_eval (column : ℕ) (value : F) (env : Env) :
    (pinRow column value).eval env = 0 ↔ env column = value := by
  simp [pinRow, sub_eq_zero]

/-- Where variable `column` holds `value`, a rewritten row has the value of the row. -/
theorem eval_fix {env : Env} {column : ℕ} {value : F} (pinned : env column = value)
    (e : Circuit.Expr) : (Expr.subst (fixWires column value) e).eval env = e.eval env := by
  rw [Expr.eval_subst]
  congr 1
  funext k
  by_cases same : k = column
  · subst same
    simp [fixWires, pinned]
  · simp [fixWires, same]

/-- With the pin row in place, the rewritten rows hold exactly when the rows hold. -/
theorem holds_fix_iff (column : ℕ) (value : F) (before after : List Circuit.Expr) (env : Env) :
    ConstraintsHold env
        (fixRows column value before ++ pinRow column value :: fixRows column value after) ↔
      ConstraintsHold env (before ++ pinRow column value :: after) := by
  have pin : ∀ rows : List Circuit.Expr, ConstraintsHold env rows →
      pinRow column value ∈ rows → env column = value :=
    fun _ holds member => (pinRow_eval column value env).mp (holds _ member)
  constructor
  · intro holds
    have pinned := pin _ holds (by simp)
    intro e member
    rcases List.mem_append.mp member with early | late
    · rw [← eval_fix pinned]
      exact holds _ (List.mem_append_left _ (List.mem_map_of_mem early))
    · rcases List.mem_cons.mp late with rfl | late
      · exact (pinRow_eval column value env).mpr pinned
      · rw [← eval_fix pinned]
        exact holds _ (List.mem_append_right _ (List.mem_cons_of_mem _ (List.mem_map_of_mem late)))
  · intro holds
    have pinned := pin _ holds (by simp)
    intro e member
    rcases List.mem_append.mp member with early | late
    · obtain ⟨row, inside, rfl⟩ := List.mem_map.mp early
      rw [eval_fix pinned]
      exact holds _ (List.mem_append_left _ inside)
    · rcases List.mem_cons.mp late with rfl | late
      · exact (pinRow_eval column value env).mpr pinned
      · obtain ⟨row, inside, rfl⟩ := List.mem_map.mp late
        rw [eval_fix pinned]
        exact holds _ (List.mem_append_right _ (List.mem_cons_of_mem _ inside))

/-- `app` with local variable `column` read as `value` in each row other than the pin row, which
`app.rows` holds between `before` and `after`. The model is `app`'s, and the port coverage and the
completeness of `app` carry over through `holds_fix_iff`. -/
def PortApp.fixColumn (app : PortApp p) (column : ℕ) (value : F)
    (before after : List Circuit.Expr) (split : app.rows = before ++ pinRow column value :: after) :
    PortApp p :=
  { app with
    rows := fixRows column value before ++ pinRow column value :: fixRows column value after
    sound := fun valid v wellFormed holds =>
      app.sound valid v wellFormed (split ▸ (holds_fix_iff column value before after _).mp holds)
    complete := fun valid {_ _ _} ops step wellFormed ports =>
      (app.complete valid ops step wellFormed ports).imp fun _ holds =>
        (holds_fix_iff column value before after _).mpr (split ▸ holds) }

theorem PortApp.fixColumn_model (app : PortApp p) (column : ℕ) (value : F)
    (before after : List Circuit.Expr) (split : app.rows = before ++ pinRow column value :: after) :
    (app.fixColumn column value before after split).model = app.model := rfl

end NightstreamFPrime.Lifecycle.Nebula
