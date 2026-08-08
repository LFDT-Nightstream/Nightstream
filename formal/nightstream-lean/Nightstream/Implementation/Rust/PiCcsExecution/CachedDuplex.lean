import Nightstream.Implementation.R1CS.Canonical.Poseidon2Duplex

/-!
Array-cached execution of the canonical Poseidon2 duplex.

Owns: an executable permutation that materializes every eight-lane round
state once, the matching overwrite duplex, and pointwise equality proofs to
the canonical value-level reference.

Does not own: constants, transcript framing, receipt parsing, or challenges.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Rust.PiCcsExecution.CachedDuplex

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Core
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Eval
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Matrices
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Reference
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Schedule

abbrev DuplexState :=
  Nightstream.Implementation.R1CS.Canonical.Poseidon2Duplex.State

/-- Materialize all eight lanes once and expose the array as a value function. -/
def cacheValues (values : Values) : Values :=
  let array := Array.ofFn values
  fun lane => array.getD lane.val 0

theorem cacheValues_eq (values : Values) : cacheValues values = values := by
  funext lane
  simp [cacheValues]

def applyMatrixCached
    (matrix : Fin width -> Fin width -> Nat) (values : Values) : Values :=
  cacheValues (applyMatrixValues matrix values)

theorem applyMatrixCached_eq
    (matrix : Fin width -> Fin width -> Nat) (values : Values) :
    applyMatrixCached matrix values = applyMatrixValues matrix values :=
  cacheValues_eq _

def fullRoundCached
    (roundConstants : Fin width -> Nat) (state : Values) : Values :=
  cacheValues (fullRoundValues roundConstants state)

theorem fullRoundCached_eq
    (roundConstants : Fin width -> Nat) (state : Values) :
    fullRoundCached roundConstants state = fullRoundValues roundConstants state :=
  cacheValues_eq _

def partialRoundCached (roundConstant : Nat) (state : Values) : Values :=
  cacheValues (partialRoundValues roundConstant state)

theorem partialRoundCached_eq (roundConstant : Nat) (state : Values) :
    partialRoundCached roundConstant state =
      partialRoundValues roundConstant state :=
  cacheValues_eq _

/-- Cached counterpart of the reference initial-full-round recursion. -/
def initialCached (constants : Constants) (input : Values) : Nat -> Values
  | 0 => applyMatrixCached externalMatrix input
  | round + 1 =>
      fullRoundCached (constants.initial round)
        (initialCached constants input round)

theorem initialCached_eq_reference
    (constants : Constants) (input : Values) (round : Nat) :
    initialCached constants input round = refInitial constants input round := by
  induction round with
  | zero => exact applyMatrixCached_eq externalMatrix input
  | succ round inductionHypothesis =>
      simp only [initialCached, refInitial]
      rw [inductionHypothesis, fullRoundCached_eq]

/-- Cached counterpart of the reference partial-round recursion. -/
def partialCached (constants : Constants) (input : Values) : Nat -> Values
  | 0 => initialCached constants input halfFullRounds
  | round + 1 =>
      partialRoundCached (constants.internal round)
        (partialCached constants input round)

theorem partialCached_eq_reference
    (constants : Constants) (input : Values) (round : Nat) :
    partialCached constants input round = refPartial constants input round := by
  induction round with
  | zero => exact initialCached_eq_reference constants input halfFullRounds
  | succ round inductionHypothesis =>
      simp only [partialCached, refPartial]
      rw [inductionHypothesis, partialRoundCached_eq]

/-- Cached counterpart of the reference terminal-full-round recursion. -/
def terminalCached (constants : Constants) (input : Values) : Nat -> Values
  | 0 => partialCached constants input partialRounds
  | round + 1 =>
      fullRoundCached (constants.terminal round)
        (terminalCached constants input round)

theorem terminalCached_eq_reference
    (constants : Constants) (input : Values) (round : Nat) :
    terminalCached constants input round = refTerminal constants input round := by
  induction round with
  | zero => exact partialCached_eq_reference constants input partialRounds
  | succ round inductionHypothesis =>
      simp only [terminalCached, refTerminal]
      rw [inductionHypothesis, fullRoundCached_eq]

/-- Complete cached permutation. -/
def referencePermutationCached
    (constants : Constants) (input : Values) : Values :=
  terminalCached constants input halfFullRounds

theorem referencePermutationCached_eq_reference
    (constants : Constants) (input : Values) :
    referencePermutationCached constants input =
      referencePermutation constants input :=
  terminalCached_eq_reference constants input halfFullRounds

/-- Cached permutation and cursor reset. -/
def permute (constants : Constants) (state : DuplexState) : DuplexState :=
  { lanes := referencePermutationCached constants state.lanes
    absorbed := 0 }

theorem permute_eq_reference (constants : Constants) (state : DuplexState) :
    permute constants state =
      Nightstream.Implementation.R1CS.Canonical.Poseidon2Duplex.permute
        constants state := by
  unfold permute
  unfold Nightstream.Implementation.R1CS.Canonical.Poseidon2Duplex.permute
  rw [referencePermutationCached_eq_reference]

def guarded (constants : Constants) (state : DuplexState) : DuplexState :=
  if Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge.rate <=
      state.absorbed then
    permute constants state
  else
    state

theorem guarded_eq_reference (constants : Constants) (state : DuplexState) :
    guarded constants state =
      Nightstream.Implementation.R1CS.Canonical.Poseidon2Duplex.guarded
        constants state := by
  unfold guarded
  unfold Nightstream.Implementation.R1CS.Canonical.Poseidon2Duplex.guarded
  split
  · exact permute_eq_reference constants state
  · rfl

/-- Cached overwrite absorption. -/
def absorbElem
    (constants : Constants) (value : Nat) (state : DuplexState) : DuplexState :=
  let target := guarded constants state
  { lanes := fun lane =>
      if lane.val = target.absorbed then
        value % goldilocksP
      else
        target.lanes lane
    absorbed := target.absorbed + 1 }

theorem absorbElem_eq_reference
    (constants : Constants) (value : Nat) (state : DuplexState) :
    absorbElem constants value state =
      Nightstream.Implementation.R1CS.Canonical.Poseidon2Duplex.absorbElem
        constants value state := by
  unfold absorbElem
  unfold Nightstream.Implementation.R1CS.Canonical.Poseidon2Duplex.absorbElem
  rw [guarded_eq_reference]

/-- Cached left-to-right list absorption. -/
def absorbList (constants : Constants) : List Nat -> DuplexState -> DuplexState
  | [], state => state
  | value :: values, state =>
      absorbList constants values (absorbElem constants value state)

theorem absorbList_eq_reference
    (constants : Constants) (values : List Nat) (state : DuplexState) :
    absorbList constants values state =
      Nightstream.Implementation.R1CS.Canonical.Poseidon2Duplex.absorbList
        constants values state := by
  induction values generalizing state with
  | nil => rfl
  | cons value values inductionHypothesis =>
      simp only [absorbList]
      simp only [Nightstream.Implementation.R1CS.Canonical.Poseidon2Duplex.absorbList]
      rw [absorbElem_eq_reference, inductionHypothesis]

/-- Cached pre-squeeze domain gate. -/
def gate (constants : Constants) (state : DuplexState) : DuplexState :=
  permute constants (absorbElem constants 1 state)

theorem gate_eq_reference (constants : Constants) (state : DuplexState) :
    gate constants state =
      Nightstream.Implementation.R1CS.Canonical.Poseidon2Duplex.gate
        constants state := by
  unfold gate
  unfold Nightstream.Implementation.R1CS.Canonical.Poseidon2Duplex.gate
  rw [absorbElem_eq_reference, permute_eq_reference]

end Nightstream.Implementation.Rust.PiCcsExecution.CachedDuplex
