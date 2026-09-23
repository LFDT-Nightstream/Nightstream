import NightstreamFPrime.Spec.Folding.Nifs.NonInteractive.PiRlcWideSampler.Conditional
import NightstreamFPrime.Spec.Folding.Nifs.NonInteractive.PiRlcWideSampler.Transcript

/-! A classical ideal block-oracle experiment. A query names the complete
history of rate-block additions before a read, including zero additions for
pure permutation advances. Histories are authoritative input data, not
carried digests. Every fresh query returns one joint uniform four-field
block; repeated queries return the same block.

The query budget counts all calls made by the experiment, including
adversarial calls and repeats. The comparison experiment first samples a
uniform scalar tape and then consistent raw preimages. Its advantage differs
by at most q * distance even when the caller observes every raw lane.

This module defines the ideal experiment. It does not assert that concrete
Poseidon2 executions have its law, prove a Fiat–Shamir extraction theorem,
or supply a quantum-query bound. Those are explicit security boundaries. -/

namespace NightstreamFPrime.Spec.Folding.Nifs.NonInteractive.PiRlcWideSampler.OracleModel

open NightstreamFPrime.Spec

abbrev Query := {history : List Draw // history ≠ []}
abbrev Cache := Query → Option Draw

def zeroBlock : Draw := fun _ => 0

/-- The existing scalar-domain block, padded to the four-lane rate. -/
def scalarDomain (coordinate : Fin 17) : Draw := fun lane =>
  if lane.val = 0 then Poseidon2.ofNat 4
  else if lane.val = 1 then Poseidon2.ofNat coordinate.val else 0

/-- Output position is represented by the actual zero-addition advances,
so two descriptions of the same normalized history name the same query. -/
def scalarQuery (history : List Draw) (coordinate : Fin 17) (position : Nat) : Query :=
  ⟨history ++ [scalarDomain coordinate] ++ List.replicate position zeroBlock, by
    intro empty
    have lengths := congrArg List.length empty
    simp at lengths⟩

def empty : Cache := fun _ => none

def ask (cache : Cache) (query : Query) (fresh : Draw) : Draw × Cache :=
  match cache query with
  | some previous => (previous, cache)
  | none => (fresh, fun candidate => if candidate = query then some fresh else cache candidate)

theorem ask_saved (cache : Cache) (query : Query) (fresh : Draw) :
    (ask cache query fresh).2 query = some (ask cache query fresh).1 := by
  cases previous : cache query <;> simp [ask, previous]

theorem ask_preserves (cache : Cache) (query previous : Query) (fresh saved : Draw)
    (known : cache previous = some saved) : (ask cache query fresh).2 previous = some saved := by
  cases current : cache query with
  | some value => simpa only [ask, current] using known
  | none =>
      have different : previous ≠ query := by
        intro same
        subst previous
        rw [current] at known
        cases known
      simpa only [ask, current, if_neg different] using known

theorem ask_repeated (cache : Cache) (query : Query) (first second : Draw) :
    (ask (ask cache query first).2 query second).1 = (ask cache query first).1 := by
  rw [ask, ask_saved]

structure Program (State : Type*) where
  request : State → Option Query
  advance : State → Draw → State

structure Outcome (State : Type*) where
  state : State
  cache : Cache
  queries : Nat

/-- q is an experiment parameter, not a deployment limit. A caller can
stop early; unused tape cells have no effect. -/
def run {State : Type*} (program : Program State) (state : State) (cache : Cache) :
    (q : Nat) → (Fin q → Draw) → Outcome State
  | 0, _ => ⟨state, cache, 0⟩
  | count + 1, tape =>
      match program.request state with
      | none => ⟨state, cache, 0⟩
      | some query =>
          let answer := ask cache query (tape 0)
          let later := run program (program.advance state answer.1) answer.2 count
            (fun index => tape index.succ)
          {later with queries := later.queries + 1}

theorem run_query_bound {State : Type*} (program : Program State) (state : State) (cache : Cache)
    (q : Nat) (tape : Fin q → Draw) : (run program state cache q tape).queries ≤ q := by
  induction q generalizing state cache with
  | zero => exact Nat.le_refl 0
  | succ count ih =>
      simp only [run]
      cases request : program.request state with
      | none => exact Nat.zero_le _
      | some query => exact Nat.add_le_add_right (ih _ _ _) 1

theorem run_preserves {State : Type*} (program : Program State) (state : State) (cache : Cache)
    (q : Nat) (tape : Fin q → Draw) (query : Query) (value : Draw)
    (known : cache query = some value) : (run program state cache q tape).cache query = some value := by
  induction q generalizing state cache with
  | zero => exact known
  | succ count ih =>
      simp only [run]
      cases request : program.request state with
      | none => exact known
      | some next => exact ih _ _ _ (ask_preserves cache next query (tape 0) value known)

/-- Adaptive keys, raw-lane observations, and repeated-query consistency
are all included in the deterministic test of the complete draw tape. -/
theorem run_bias_bound {State : Type*} (program : Program State) (initial : State) (q : Nat)
    (test : Outcome State → ℝ) (nonnegative : ∀ result, 0 ≤ test result)
    (atMostOne : ∀ result, test result ≤ 1) :
    |average (fun tape : Fin q → Draw => test (run program initial empty q tape)) -
      balancedAverage (fun tape : Fin q → Draw => test (run program initial empty q tape))| ≤
        q * distance := by
  simpa only [Fintype.card_fin] using
    raw_blocks_difference_abs_le (fun tape : Fin q → Draw => test (run program initial empty q tape))
      (fun _ => nonnegative _) (fun _ => atMostOne _)

/-- An explicitly supplied block-oracle approximation error remains a
separate term. No value for it is asserted for concrete Poseidon2. -/
theorem under_block_oracle_assumption {State : Type*} (program : Program State)
    (initial : State) (q : Nat) (test : Outcome State → ℝ)
    (nonnegative : ∀ result, 0 ≤ test result) (atMostOne : ∀ result, test result ≤ 1)
    (actualRate modelError : ℝ)
    (model : |actualRate - average (fun tape : Fin q → Draw =>
      test (run program initial empty q tape))| ≤ modelError) :
    |actualRate - balancedAverage (fun tape : Fin q → Draw =>
      test (run program initial empty q tape))| ≤ modelError + q * distance := by
  have comparison := run_bias_bound program initial q test nonnegative atMostOne
  exact (abs_sub_le _ _ _).trans (add_le_add model comparison)

end NightstreamFPrime.Spec.Folding.Nifs.NonInteractive.PiRlcWideSampler.OracleModel
