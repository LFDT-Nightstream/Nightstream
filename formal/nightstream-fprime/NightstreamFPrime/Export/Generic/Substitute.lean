import NightstreamFPrime.Layout.R1CS

/-! Substitution of columns by linear forms in R1CS rows. `refill forms env` gives each column
that has a form the value of its form in `env`. A substituted combination has in `env` the value
of the combination in the refilled environment (`substitute_eval`), so substituted rows hold in
`env` exactly when the rows hold in the refilled environment (`rowsHold_substitute`). -/

namespace NightstreamFPrime.Export.Generic.Substitute

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Layout

/-- The environment with each column that has a form at the form's value in `env`. -/
def refill (forms : Nat → Option R1CS.LinearCombination) (env : Env) : Env := fun column =>
  match forms column with
  | some form => form.eval env
  | none => env column

/-- One term with its column replaced by the column's form. -/
def term (forms : Nat → Option R1CS.LinearCombination) (entry : Nat × F) :
    R1CS.LinearCombination :=
  match forms entry.1 with
  | some form => form.scale entry.2
  | none => ⟨0, [entry]⟩

/-- The combination with each column that has a form replaced by the form. -/
def substitute (forms : Nat → Option R1CS.LinearCombination)
    (combination : R1CS.LinearCombination) : R1CS.LinearCombination :=
  combination.terms.foldr (fun entry result => (term forms entry).add result)
    ⟨combination.constant, []⟩

theorem term_eval (forms : Nat → Option R1CS.LinearCombination) (env : Env) (entry : Nat × F) :
    (term forms entry).eval env = entry.2 * refill forms env entry.1 := by
  unfold term refill
  cases forms entry.1 with
  | some form => simp
  | none => simp [R1CS.LinearCombination.eval]

theorem substitute_eval (forms : Nat → Option R1CS.LinearCombination) (env : Env)
    (combination : R1CS.LinearCombination) :
    (substitute forms combination).eval env = combination.eval (refill forms env) := by
  rcases combination with ⟨constant, terms⟩
  unfold substitute
  induction terms with
  | nil => simp [R1CS.LinearCombination.eval]
  | cons entry rest hypothesis =>
      simp only [List.foldr_cons, R1CS.LinearCombination.eval_add, term_eval]
      rw [hypothesis]
      simp only [R1CS.LinearCombination.eval, List.map_cons, List.sum_cons]
      exact add_left_comm _ _ _

/-- The row with each side substituted. -/
def row (forms : Nat → Option R1CS.LinearCombination) (source : R1CS.Row) : R1CS.Row :=
  ⟨substitute forms source.a, substitute forms source.b, substitute forms source.c⟩

theorem row_holds (forms : Nat → Option R1CS.LinearCombination) (env : Env) (source : R1CS.Row) :
    (row forms source).Holds env ↔ source.Holds (refill forms env) := by
  simp only [R1CS.Row.Holds, row, substitute_eval]

theorem rowsHold_substitute (forms : Nat → Option R1CS.LinearCombination) (env : Env)
    (rows : List R1CS.Row) :
    R1CS.RowsHold env (rows.map (row forms)) ↔ R1CS.RowsHold (refill forms env) rows := by
  constructor
  · intro holds source member
    exact (row_holds forms env source).mp (holds _ (List.mem_map_of_mem member))
  · intro holds substituted member
    obtain ⟨source, sourceMember, rfl⟩ := List.mem_map.mp member
    exact (row_holds forms env source).mpr (holds source sourceMember)

/-- The columns of a substituted combination: a column of the combination without a form, or a
column of the form of one of its columns. -/
theorem substitute_terms (forms : Nat → Option R1CS.LinearCombination)
    (combination : R1CS.LinearCombination) {entry : Nat × F}
    (member : entry ∈ (substitute forms combination).terms) :
    (∃ source ∈ combination.terms, forms source.1 = none ∧ entry = source) ∨
      ∃ source ∈ combination.terms, ∃ form, forms source.1 = some form ∧
        entry.1 ∈ form.terms.map Prod.fst := by
  rcases combination with ⟨constant, terms⟩
  unfold substitute at member
  induction terms with
  | nil => simp at member
  | cons head rest hypothesis =>
      simp only [List.foldr_cons, R1CS.LinearCombination.add, List.mem_append] at member
      rcases member with here | later
      · unfold term at here
        cases formEq : forms head.1 with
        | some form =>
            rw [formEq] at here
            simp only [R1CS.LinearCombination.scale, List.mem_map] at here
            obtain ⟨original, originalMember, rfl⟩ := here
            exact Or.inr ⟨head, List.mem_cons_self, form, formEq,
              List.mem_map.mpr ⟨original, originalMember, rfl⟩⟩
        | none =>
            rw [formEq] at here
            simp only [List.mem_singleton] at here
            exact Or.inl ⟨head, List.mem_cons_self, formEq, here⟩
      · rcases hypothesis later with ⟨source, sourceMember, noForm, equal⟩ |
          ⟨source, sourceMember, form, formEq, inForm⟩
        · exact Or.inl ⟨source, List.mem_cons_of_mem _ sourceMember, noForm, equal⟩
        · exact Or.inr ⟨source, List.mem_cons_of_mem _ sourceMember, form, formEq, inForm⟩

/-! ### A table of forms -/

/-- `writes` on `start`: a later write to a column replaces an earlier one. -/
def fill (start : Array (Option R1CS.LinearCombination))
    (writes : List (Nat × R1CS.LinearCombination)) : Array (Option R1CS.LinearCombination) :=
  writes.foldl (fun table write => table.setIfInBounds write.1 (some write.2)) start

/-- The table of `writes` over `size` columns. -/
def table (size : Nat) (writes : List (Nat × R1CS.LinearCombination)) :
    Array (Option R1CS.LinearCombination) :=
  fill (Array.replicate size none) writes

/-- The form of a column in a table, or none past the table. -/
def lookup (forms : Array (Option R1CS.LinearCombination)) (column : Nat) :
    Option R1CS.LinearCombination :=
  forms.getD column none

theorem fill_size (start : Array (Option R1CS.LinearCombination))
    (writes : List (Nat × R1CS.LinearCombination)) : (fill start writes).size = start.size := by
  induction writes generalizing start with
  | nil => rfl
  | cons head rest hypothesis =>
      show (fill (start.setIfInBounds head.1 (some head.2)) rest).size = start.size
      rw [hypothesis]
      simp

theorem fill_absent (start : Array (Option R1CS.LinearCombination))
    (writes : List (Nat × R1CS.LinearCombination)) (column : Nat)
    (absent : ∀ write ∈ writes, write.1 ≠ column) :
    lookup (fill start writes) column = lookup start column := by
  induction writes generalizing start with
  | nil => rfl
  | cons head rest hypothesis =>
      show lookup (fill (start.setIfInBounds head.1 (some head.2)) rest) column = _
      rw [hypothesis _ fun write member => absent write (List.mem_cons_of_mem _ member)]
      have other : head.1 ≠ column := absent head List.mem_cons_self
      simp only [lookup, Array.getD_eq_getD_getElem?, Array.getElem?_setIfInBounds, other,
        ↓reduceIte]

theorem fill_present (start : Array (Option R1CS.LinearCombination))
    (writes : List (Nat × R1CS.LinearCombination)) (write : Nat × R1CS.LinearCombination)
    (member : write ∈ writes) (inside : write.1 < start.size)
    (unique : ∀ other ∈ writes, other.1 = write.1 → other = write) :
    lookup (fill start writes) write.1 = some write.2 := by
  induction writes generalizing start with
  | nil => simp at member
  | cons head rest hypothesis =>
      show lookup (fill (start.setIfInBounds head.1 (some head.2)) rest) write.1 = _
      by_cases later : write ∈ rest
      · exact hypothesis _ later (by simpa using inside)
          fun other otherMember => unique other (List.mem_cons_of_mem _ otherMember)
      · have here : write = head := by
          rcases List.mem_cons.mp member with equal | inRest
          · exact equal
          · exact absurd inRest later
        subst here
        rw [fill_absent _ rest write.1 fun other otherMember same =>
          later (unique other (List.mem_cons_of_mem _ otherMember) same ▸ otherMember)]
        simp only [lookup, Array.getD_eq_getD_getElem?, Array.getElem?_setIfInBounds, ↓reduceIte,
          inside]
        rfl

/-- A column that no write names has no form. -/
theorem lookup_table_none (size : Nat) (writes : List (Nat × R1CS.LinearCombination))
    (column : Nat) (absent : ∀ write ∈ writes, write.1 ≠ column) :
    lookup (table size writes) column = none := by
  rw [table, fill_absent _ _ _ absent]
  simp only [lookup, Array.getD_eq_getD_getElem?, Array.getElem?_replicate]
  split <;> rfl

/-- A column that one write names, below the size, has that write's form. -/
theorem lookup_table_some (size : Nat) (writes : List (Nat × R1CS.LinearCombination))
    (write : Nat × R1CS.LinearCombination) (member : write ∈ writes) (inside : write.1 < size)
    (unique : ∀ other ∈ writes, other.1 = write.1 → other = write) :
    lookup (table size writes) write.1 = some write.2 :=
  fill_present _ writes write member (by simpa using inside) unique

end NightstreamFPrime.Export.Generic.Substitute
