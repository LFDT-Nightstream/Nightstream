import Mathlib.Tactic.Ring
import NightstreamFPrime.Export.Generic.Substitute
import NightstreamFPrime.Layout.R1CS.Support

set_option autoImplicit false

namespace NightstreamFPrime.Export.Generic.NativePower

open NightstreamFPrime.Spec NightstreamFPrime.Circuit NightstreamFPrime.Layout
open Fin.CommRing

structure Chain where
  input : R1CS.LinearCombination
  x2 : Nat
  x4 : Nat
  x6 : Nat
  output : Nat

def Chain.live (chain : Chain) (column : Nat) : Prop :=
  column ≠ chain.x2 ∧ column ≠ chain.x4 ∧ column ≠ chain.x6

def Chain.Valid (chain : Chain) : Prop :=
  chain.x2 ≠ chain.x4 ∧ chain.x2 ≠ chain.x6 ∧ chain.x4 ≠ chain.x6 ∧
    chain.live chain.output ∧ chain.input.VarsSatisfy chain.live

def Chain.rows (chain : Chain) : List R1CS.Row :=
  [⟨chain.input, chain.input, .ofVar chain.x2⟩,
   ⟨.ofVar chain.x2, .ofVar chain.x2, .ofVar chain.x4⟩,
   ⟨.ofVar chain.x4, .ofVar chain.x2, .ofVar chain.x6⟩,
   ⟨.ofVar chain.x6, chain.input, .ofVar chain.output⟩]

def Chain.Holds (chain : Chain) (env : Env) : Prop :=
  (chain.input.eval env) ^ 7 = env chain.output

def Chain.restore (chain : Chain) (env : Env) : Env := fun column =>
  if column = chain.x2 then (chain.input.eval env) ^ 2
  else if column = chain.x4 then (chain.input.eval env) ^ 4
  else if column = chain.x6 then (chain.input.eval env) ^ 6
  else env column

theorem Chain.restore_live (chain : Chain) (env : Env) (column : Nat)
    (live : chain.live column) : chain.restore env column = env column := by
  simp [Chain.restore, live.1, live.2.1, live.2.2]

theorem Chain.restore_input (chain : Chain) (env : Env) (valid : chain.Valid) :
    chain.input.eval (chain.restore env) = chain.input.eval env := by
  exact chain.input.eval_eq_of_agree chain.live _ _ valid.2.2.2.2
    (chain.restore_live env)

theorem Chain.sound (chain : Chain) (env : Env)
    (holds : R1CS.RowsHold env chain.rows) : chain.Holds env := by
  have equations :
      chain.input.eval env * chain.input.eval env = env chain.x2 ∧
      env chain.x2 * env chain.x2 = env chain.x4 ∧
      env chain.x4 * env chain.x2 = env chain.x6 ∧
      env chain.x6 * chain.input.eval env = env chain.output := by
    simpa [Chain.rows, R1CS.RowsHold, R1CS.Row.Holds] using holds
  rcases equations with ⟨h2, h4, h6, h7⟩
  unfold Chain.Holds
  calc
    chain.input.eval env ^ 7 =
        (chain.input.eval env * chain.input.eval env) *
        (chain.input.eval env * chain.input.eval env) *
        (chain.input.eval env * chain.input.eval env) * chain.input.eval env := by ring
    _ = env chain.output := by rw [h2, h4, h6, h7]

theorem Chain.complete (chain : Chain) (env : Env) (valid : chain.Valid)
    (holds : chain.Holds env) : R1CS.RowsHold (chain.restore env) chain.rows := by
  rcases valid with ⟨h24, h26, h46, output, input⟩
  have hx := chain.restore_input env ⟨h24, h26, h46, output, input⟩
  have hy := chain.restore_live env chain.output output
  have h2 : chain.restore env chain.x2 = chain.input.eval env ^ 2 := by
    simp [Chain.restore]
  have h4 : chain.restore env chain.x4 = chain.input.eval env ^ 4 := by
    simp [Chain.restore, Ne.symm h24]
  have h6 : chain.restore env chain.x6 = chain.input.eval env ^ 6 := by
    simp [Chain.restore, Ne.symm h26, Ne.symm h46]
  simp only [Chain.rows, R1CS.RowsHold, List.mem_cons,
    forall_eq_or_imp, R1CS.Row.Holds, R1CS.LinearCombination.eval_ofVar]
  rw [hx, hy, h2, h4, h6]
  refine ⟨by ring, by ring, by ring, ?_, ?_⟩
  · calc
      chain.input.eval env ^ 6 * chain.input.eval env = chain.input.eval env ^ 7 := by ring
      _ = env chain.output := holds
  · intro row member
    cases member

theorem Chain.restore_source (chain : Chain) (kept : List R1CS.Row) (env : Env)
    (valid : chain.Valid) (framed : ∀ row ∈ kept, row.VarsSatisfy chain.live)
    (native : chain.Holds env) (holds : R1CS.RowsHold env kept) :
    R1CS.RowsHold (chain.restore env) (chain.rows ++ kept) := by
  apply (R1CS.rowsHold_append _ _ _).mpr
  exact ⟨chain.complete env valid native,
    R1CS.rowsHold_of_agree kept chain.live env (chain.restore env) framed
      (chain.restore_live env) holds⟩

def affineForms (column : Nat) (form : R1CS.LinearCombination) :
    Nat → Option R1CS.LinearCombination := fun index =>
  if index = column then some form else none

def affineRow (column : Nat) (form : R1CS.LinearCombination) : R1CS.Row :=
  ⟨form, .one, .ofVar column⟩

theorem affine_restore_source (column : Nat) (form : R1CS.LinearCombination)
    (kept : List R1CS.Row) (env : Env)
    (acyclic : form.VarsSatisfy (fun index => index ≠ column))
    (holds : R1CS.RowsHold env (kept.map (Substitute.row (affineForms column form)))) :
    R1CS.RowsHold (Substitute.refill (affineForms column form) env)
      (affineRow column form :: kept) := by
  have agree : ∀ index, index ≠ column →
      Substitute.refill (affineForms column form) env index = env index := by
    intro index different
    simp [Substitute.refill, affineForms, different]
  have formEq := form.eval_eq_of_agree (fun index => index ≠ column)
    (Substitute.refill (affineForms column form) env) env acyclic agree
  have tail := (Substitute.rowsHold_substitute (affineForms column form) env kept).mp holds
  intro row member
  rcases List.mem_cons.mp member with rfl | later
  · simp [affineRow, R1CS.Row.Holds, formEq, Substitute.refill, affineForms]
  · exact tail row later

theorem affine_complete (column : Nat) (form : R1CS.LinearCombination)
    (kept : List R1CS.Row) (env : Env)
    (source : R1CS.RowsHold env (affineRow column form :: kept)) :
    R1CS.RowsHold env (kept.map (Substitute.row (affineForms column form))) := by
  have defining := source (affineRow column form) List.mem_cons_self
  have equation : form.eval env = env column := by
    simpa [affineRow, R1CS.Row.Holds] using defining
  have restored : Substitute.refill (affineForms column form) env = env := by
    funext index
    by_cases equal : index = column
    · simp [Substitute.refill, affineForms, equal, equation]
    · simp [Substitute.refill, affineForms, equal]
  apply (Substitute.rowsHold_substitute (affineForms column form) env kept).mpr
  rw [restored]
  exact fun row member => source row (List.mem_cons_of_mem _ member)

end NightstreamFPrime.Export.Generic.NativePower
