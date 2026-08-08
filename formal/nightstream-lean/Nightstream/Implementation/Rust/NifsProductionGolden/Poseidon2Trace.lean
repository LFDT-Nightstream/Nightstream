import Mathlib.Data.List.OfFn
import Nightstream.Implementation.Rust.NifsProductionGolden.Receipt
import Nightstream.Implementation.Rust.PiCcsExecution.CachedDuplex

/-!
Compact certificates for one canonical width-8 Poseidon2 permutation.

Rust records the state after the initial linear layer and after each of the
thirty rounds. Lean checks each local round equation. The soundness theorem
then reconstructs the canonical cached permutation without evaluating one
large recursive term.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Rust.NifsProductionGolden.Poseidon2Trace

open Nightstream.Implementation.R1CS.Canonical.Poseidon2Core
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Matrices
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Schedule
open Nightstream.Implementation.Rust.NifsProductionGolden
open Nightstream.Implementation.Rust.PiCcsExecution.CachedDuplex

abbrev Values := Fin width -> Nat

def stateAt (trace : RawPermutationTrace) (index : Nat) : Values :=
  fun lane => (trace.states.getD index []).getD lane.val 0

def initialState (trace : RawPermutationTrace) (round : Nat) : Values :=
  stateAt trace round

def partialState (trace : RawPermutationTrace) (round : Nat) : Values :=
  stateAt trace (4 + round)

def terminalState (trace : RawPermutationTrace) (round : Nat) : Values :=
  stateAt trace (26 + round)

def output (trace : RawPermutationTrace) : Values :=
  terminalState trace 4

def valuesMatch (left right : Values) : Bool :=
  decide (List.ofFn left = List.ofFn right)

theorem valuesMatch_sound (left right : Values)
    (checked : valuesMatch left right = true) : left = right := by
  apply List.ofFn_injective
  exact of_decide_eq_true checked

def initialRoundsCheck (constants : Constants) (trace : RawPermutationTrace) : Bool :=
  (List.range 4).all fun round =>
    valuesMatch (initialState trace (round + 1))
      (fullRoundCached (constants.initial round) (initialState trace round))

def partialRoundsCheck (constants : Constants) (trace : RawPermutationTrace) : Bool :=
  (List.range 22).all fun round =>
    valuesMatch (partialState trace (round + 1))
      (partialRoundCached (constants.internal round) (partialState trace round))

def terminalRoundsCheck (constants : Constants) (trace : RawPermutationTrace) : Bool :=
  (List.range 4).all fun round =>
    valuesMatch (terminalState trace (round + 1))
      (fullRoundCached (constants.terminal round) (terminalState trace round))

def check (constants : Constants) (input : Values)
    (trace : RawPermutationTrace) : Bool :=
  permutationTraceShapeCheck trace &&
    (valuesMatch (initialState trace 0) (applyMatrixCached externalMatrix input) &&
    (initialRoundsCheck constants trace &&
    (partialRoundsCheck constants trace &&
    terminalRoundsCheck constants trace)))

structure Valid (constants : Constants) (input : Values)
    (trace : RawPermutationTrace) : Prop where
  initialZero :
    initialState trace 0 = applyMatrixCached externalMatrix input
  initialRound : forall round, round < 4 ->
    initialState trace (round + 1) =
      fullRoundCached (constants.initial round) (initialState trace round)
  partialRound : forall round, round < 22 ->
    partialState trace (round + 1) =
      partialRoundCached (constants.internal round) (partialState trace round)
  terminalRound : forall round, round < 4 ->
    terminalState trace (round + 1) =
      fullRoundCached (constants.terminal round) (terminalState trace round)

theorem check_sound (constants : Constants) (input : Values)
    (trace : RawPermutationTrace) (checked : check constants input trace = true) :
    Valid constants input trace := by
  have components :
      permutationTraceShapeCheck trace = true /\
      valuesMatch (initialState trace 0)
        (applyMatrixCached externalMatrix input) = true /\
      initialRoundsCheck constants trace = true /\
      partialRoundsCheck constants trace = true /\
      terminalRoundsCheck constants trace = true := by
    simpa only [check, Bool.and_eq_true] using checked
  rcases components with ⟨_, initialZero, initialRounds, partialRounds,
    terminalRounds⟩
  simp only [initialRoundsCheck] at initialRounds
  simp only [partialRoundsCheck] at partialRounds
  simp only [terminalRoundsCheck] at terminalRounds
  refine {
    initialZero := valuesMatch_sound _ _ initialZero
    initialRound := ?_
    partialRound := ?_
    terminalRound := ?_ }
  · intro round roundLt
    have member : round ∈ List.range 4 := List.mem_range.mpr roundLt
    exact valuesMatch_sound _ _
      ((List.all_eq_true.mp initialRounds) round member)
  · intro round roundLt
    have member : round ∈ List.range 22 := List.mem_range.mpr roundLt
    exact valuesMatch_sound _ _
      ((List.all_eq_true.mp partialRounds) round member)
  · intro round roundLt
    have member : round ∈ List.range 4 := List.mem_range.mpr roundLt
    exact valuesMatch_sound _ _
      ((List.all_eq_true.mp terminalRounds) round member)

theorem initialState_eq (constants : Constants) (input : Values)
    (trace : RawPermutationTrace) (valid : Valid constants input trace) :
    forall round, round <= 4 ->
      initialState trace round = initialCached constants input round := by
  intro round roundLe
  induction round with
  | zero => exact valid.initialZero
  | succ round inductionHypothesis =>
      have roundLt : round < 4 := Nat.lt_of_succ_le roundLe
      rw [initialCached]
      rw [<- inductionHypothesis (Nat.le_of_succ_le roundLe)]
      exact valid.initialRound round roundLt

theorem partialState_eq (constants : Constants) (input : Values)
    (trace : RawPermutationTrace) (valid : Valid constants input trace) :
    forall round, round <= 22 ->
      partialState trace round = partialCached constants input round := by
  intro round roundLe
  induction round with
  | zero =>
      change initialState trace 4 = initialCached constants input halfFullRounds
      simpa using initialState_eq constants input trace valid 4 (by decide)
  | succ round inductionHypothesis =>
      have roundLt : round < 22 := Nat.lt_of_succ_le roundLe
      rw [partialCached]
      rw [<- inductionHypothesis (Nat.le_of_succ_le roundLe)]
      exact valid.partialRound round roundLt

theorem terminalState_eq (constants : Constants) (input : Values)
    (trace : RawPermutationTrace) (valid : Valid constants input trace) :
    forall round, round <= 4 ->
      terminalState trace round = terminalCached constants input round := by
  intro round roundLe
  induction round with
  | zero =>
      change partialState trace 22 = partialCached constants input partialRounds
      simpa using partialState_eq constants input trace valid 22 (by decide)
  | succ round inductionHypothesis =>
      have roundLt : round < 4 := Nat.lt_of_succ_le roundLe
      rw [terminalCached]
      rw [<- inductionHypothesis (Nat.le_of_succ_le roundLe)]
      exact valid.terminalRound round roundLt

theorem output_eq_referenceCached (constants : Constants) (input : Values)
    (trace : RawPermutationTrace) (valid : Valid constants input trace) :
    output trace = referencePermutationCached constants input := by
  exact terminalState_eq constants input trace valid 4 (by decide)

theorem output_eq_reference (constants : Constants) (input : Values)
    (trace : RawPermutationTrace) (checked : check constants input trace = true) :
    output trace =
      Nightstream.Implementation.R1CS.Canonical.Poseidon2Reference.referencePermutation
        constants input := by
  rw [<- referencePermutationCached_eq_reference]
  exact output_eq_referenceCached constants input trace
    (check_sound constants input trace checked)

end Nightstream.Implementation.Rust.NifsProductionGolden.Poseidon2Trace
