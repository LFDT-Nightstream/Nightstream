-- Hoare-style soundness lemmas for the wasm zkVM's gadgets, stated against
-- zkLean's predicate-transformer framework (`Std.Do`).
--
-- This file owns:
--   * The base soundness lemma for `constrainR1CS` — "if zkLean's evaluator
--     accepts this single R1CS op, then `a.eval * b.eval = c.eval`." zkLean
--     itself doesn't ship this lemma; we prove it once, modelled on the
--     `constrainEq2.soundness` example in zkLean's `CircuitSoundness.lean`.
--   * Per-`Instr`-constructor soundness lemmas, each proven by chaining
--     `mspec` applications of `constrainR1CS.soundness` (one per row) and
--     closing with field arithmetic. Adding a new gadget adds one Hoare
--     triple + a short chain proof — no per-shape "bridge" theorems needed.

import Std.Do
import zkLean.Semantics
import zkLean.Formalism
import zkLean.SimpSets
import Mathlib.Tactic.LinearCombination
import WasmCircuit.Field
import WasmCircuit.Gadgets

-- Simp-set wiring (mirrors zkLean's `CircuitSoundness.lean` example).
attribute [simp_FreeM] bind
attribute [simp_FreeM] default
attribute [simp_FreeM] Cslib.FreeM.bind
attribute [simp_FreeM] Cslib.FreeM.foldFreeM

attribute [simp_Triple] Std.Do.Triple
attribute [simp_Triple] Std.Do.SPred.entails
attribute [simp_Triple] Std.Do.PredTrans.apply
attribute [simp_Triple] Std.Do.PredTrans.pure
attribute [simp_Triple] Std.Do.wp

attribute [simp_circuit] runZKBuilder

open Std Do

namespace WasmCircuit.Gadgets

open WasmCircuit (Fq)

/-- Helper for the predicate-transformer match collapse: when the inner Option
    is `if cond then some _ else none`, the match reduces to a plain `if`.
    Copied from zkLean's `CircuitSoundness.lean` example. -/
private lemma match_if {α : Type} (cond : Prop) [Decidable cond] (a b : β) (s1 : α) :
    PredTrans.pushOption.match_1 _
      (if cond then some s1 else none)
      (fun _ => a)
      (fun () => b)
    = (if cond then a else b) := by
  split
  · grind
  · grind

/-- Single-constraint soundness: if zkLean's evaluator accepts
    `constrainR1CS a b c`, then the R1CS equation holds at `eval`-level.
    zkLean defines this fact computationally in `ZKOpInterp` but doesn't ship
    it as a named theorem — this is that theorem.

    Modelled directly on `constrainEq2.soundness` from zkLean's
    `examples/CircuitSoundness.lean`. -/
theorem constrainR1CS_soundness (a b c : ZKExpr Fq) :
    ⦃ λ _s => ⌜True⌝ ⦄
    ZKBuilder.constrainR1CS a b c
    ⦃ ⇓? _r _s => ⌜a.eval * b.eval = c.eval⌝ ⦄ := by
  mintro _ ∀s
  simp [simp_FreeM, simp_ZKBuilder, simp_Triple, simp_circuit, wpZKBuilder,
        OptionT.mk, ExceptConds.true, ExceptConds.const, liftM, monadLift,
        MonadLift.monadLift, StateT.run, StateT.pure, bind, StateT.bind,
        pure]
  rw [ite_apply]
  simp [StateT.pure, StateT.lift, match_if]
  split
  · aesop
  · simp

/-- Soundness for the zero-test gadget. Given that the constant-one column
    carries `1`, zkLean's evaluator on the gadget's two-row chain forces the
    `is_zero` witness column to be the zero-indicator of `value`.

    Proof structure: unfold `instrToBuilder .ZeroTest` to its two-constraint
    sequence, then `mspec` once per constraint to land each row's R1CS
    equation into a pure hypothesis. The rest is field arithmetic. -/
theorem zeroTest_soundness
    (alloc : Array (ZKExpr Fq)) (v inv iz : Nat)
    (h_one : (alloc[constOneCol]!).eval = 1) :
    ⦃ λ _s => ⌜True⌝ ⦄
    instrToBuilder alloc (.ZeroTest v inv iz)
    ⦃ ⇓? _r _s =>
        ⌜((alloc[v]!).eval = 0 → (alloc[iz]!).eval = 1) ∧
         ((alloc[v]!).eval ≠ 0 → (alloc[iz]!).eval = 0)⌝ ⦄ := by
  mintro _ ∀s0
  mvcgen [instrToBuilder, instrToRows, rowToBuilder, List.forM]
  mspec (constrainR1CS_soundness _ _ _)
  mrename_i hRow1
  mpure hRow1
  mspec (constrainR1CS_soundness _ _ _)
  mrename_i hRow2
  mpure hRow2
  -- Goal now reduces to the post over `pure PUnit.unit`; mvcgen collapses it.
  mvcgen
  -- Field arithmetic from the two row equalities.
  simp [sparseRowToExpr, coeffExpr, ZKExpr.eval,
        Int.cast_one, Int.cast_neg, one_mul, add_zero, h_one] at hRow1 hRow2
  refine ⟨?_, ?_⟩
  · intro hv
    rw [hv, zero_mul] at hRow1
    linear_combination hRow1
  · intro hv
    rcases hRow2 with hv0 | hiz0
    · exact absurd hv0 hv
    · exact hiz0

end WasmCircuit.Gadgets
