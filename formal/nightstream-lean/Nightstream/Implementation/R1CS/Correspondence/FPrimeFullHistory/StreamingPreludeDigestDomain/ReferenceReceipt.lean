import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingPreludeDigestDomain.Model

/-!
Contract: structural round receipts for fixed-input Poseidon2 reference
permutations.

Owns a generic induction from one checked state per round to the handwritten
reference evaluator. It does not own concrete receipt values or R1CS rows.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 10000

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPreludeDigestDomain

open Nightstream.Implementation.R1CS.Canonical
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Core
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Eval
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Matrices
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Reference
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Schedule

def valuesOf (values : List Nat) : Values :=
  fun lane => values.getD lane.val 0

def receiptState (states : List (List Nat)) (round : Nat) : Values :=
  valuesOf (states.getD round [])

def runIndexed
    (step : Nat → Values → Values) (initial : Values) : Nat → Values
  | 0 => initial
  | round + 1 => step round (runIndexed step initial round)

structure PhaseReceipt.Valid
    (states : List (List Nat)) (count : Nat)
    (initial : Values) (step : Nat → Values → Values) : Prop where
  lengthExact : states.length = count + 1
  initialExact : receiptState states 0 = initial
  stepExact : ∀ round, round < count →
    receiptState states (round + 1) = step round (receiptState states round)

theorem PhaseReceipt.Valid.finalExact
    {states : List (List Nat)} {count : Nat}
    {initial : Values} {step : Nat → Values → Values}
    (valid : PhaseReceipt.Valid states count initial step) :
    receiptState states count = runIndexed step initial count := by
  have all : ∀ round, round ≤ count →
      receiptState states round = runIndexed step initial round := by
    intro round bound
    induction round with
    | zero => exact valid.initialExact
    | succ previous hypothesis =>
        rw [valid.stepExact previous (by omega), runIndexed,
          hypothesis (by omega)]
  exact all count (Nat.le_refl count)

theorem runInitial_eq_refInitial
    (constants : Constants) (input : Values) (round : Nat) :
    runIndexed (fun index => fullRoundValues (constants.initial index))
        (applyMatrixValues externalMatrix input) round =
      refInitial constants input round := by
  induction round with
  | zero => rfl
  | succ previous hypothesis =>
      rw [runIndexed, refInitial, hypothesis]

theorem runPartial_eq_refPartial
    (constants : Constants) (input : Values) (round : Nat) :
    runIndexed (fun index => partialRoundValues (constants.internal index))
        (refInitial constants input halfFullRounds) round =
      refPartial constants input round := by
  induction round with
  | zero => rfl
  | succ previous hypothesis =>
      rw [runIndexed, refPartial, hypothesis]

theorem runTerminal_eq_refTerminal
    (constants : Constants) (input : Values) (round : Nat) :
    runIndexed (fun index => fullRoundValues (constants.terminal index))
        (refPartial constants input partialRounds) round =
      refTerminal constants input round := by
  induction round with
  | zero => rfl
  | succ previous hypothesis =>
      rw [runIndexed, refTerminal, hypothesis]

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPreludeDigestDomain
