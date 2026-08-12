/-!
Contract: countermodels for an F-prime output hash that depends on the
consumed fresh claim.

HyperNova Construction 2 requires the next invocation to replay the state
hash from the verifier keys, invocation index, initial state, current state,
updated running claims, and program counter. The consumed fresh claim is not
part of that next paper state. This file proves that a claim-dependent hash
cannot, in general, be replayed from the next state alone.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.NebulaV2.FPrimeStateHashCountermodels

/-- A hash is replayable by the next invocation only if one function of the
paper state alone reproduces it for every consumed claim. -/
def ReplayableFromState
    {Claim State Digest : Type}
    (hash : Claim -> State -> Digest) : Prop :=
  exists replay : State -> Digest,
    forall claim state, replay state = hash claim state

/-- If two fresh claims give different digests for the same next paper state,
then no next-state-only replay function exists. -/
theorem not_replayable_of_claim_variation
    {Claim State Digest : Type}
    {hash : Claim -> State -> Digest}
    {left right : Claim} {state : State}
    (different : hash left state ≠ hash right state) :
    ¬ ReplayableFromState hash := by
  intro replayable
  obtain ⟨replay, exact⟩ := replayable
  apply different
  exact (exact left state).symm.trans (exact right state)

/-- Small concrete model of the rejected design: the digest is the consumed
claim, while the next paper state contains no claim coordinate. -/
def claimDependentHash (claim : Bool) (_state : Unit) : Bool := claim

theorem claimDependentHash_varies_at_same_state :
    claimDependentHash false () ≠ claimDependentHash true () := by
  decide

theorem claimDependentHash_not_replayable :
    ¬ ReplayableFromState claimDependentHash :=
  not_replayable_of_claim_variation claimDependentHash_varies_at_same_state

/-- Control model: a state-only hash is replayable for every consumed claim. -/
def stateOnlyHash (_claim : Bool) (state : Bool) : Bool := state

theorem stateOnlyHash_replayable : ReplayableFromState stateOnlyHash := by
  exact ⟨id, by intro claim state; rfl⟩

/-! ## Omitting the initial state permits cross-execution splicing -/

/-- Minimal Construction-2 state with an immutable initial value and a
current value. -/
structure TwoPointState where
  initial : Bool
  current : Bool
deriving DecidableEq, Repr

/-- Rejected hash shape: it authenticates the current value but omits the
initial value. -/
def hashWithoutInitial (state : TwoPointState) : Bool := state.current

def falseStartAtTrue : TwoPointState :=
  { initial := false, current := true }

def trueStartAtTrue : TwoPointState :=
  { initial := true, current := true }

theorem different_initial_executions :
    falseStartAtTrue ≠ trueStartAtTrue := by
  decide

/-- Two executions with different starts but the same current value have the
same rejected digest. This is the exact splice that including `z0` prevents. -/
theorem hashWithoutInitial_allows_cross_execution_splice :
    hashWithoutInitial falseStartAtTrue =
      hashWithoutInitial trueStartAtTrue := by
  rfl

/-- Control shape: the full pair is injective before any cryptographic hash
is applied. -/
def fullStateFrame (state : TwoPointState) : Bool × Bool :=
  (state.initial, state.current)

theorem fullStateFrame_separates_the_two_starts :
    fullStateFrame falseStartAtTrue ≠
      fullStateFrame trueStartAtTrue := by
  decide

/-! ## Every Construction-2 hash coordinate is necessary -/

/-- Finite countermodel for the six authority-bearing coordinates in the
Construction-2 state-hash preimage. `verifierKeys` represents the complete
ordered verifier-key vector, not one digest chosen by the prover. -/
structure CompleteHashFrame where
  verifierKeys : Bool
  iteration : Bool
  initialState : Bool
  currentState : Bool
  running : Bool
  programCounter : Bool
deriving DecidableEq, Repr

def zeroFrame : CompleteHashFrame :=
  ⟨false, false, false, false, false, false⟩

def changedVerifierKeys : CompleteHashFrame :=
  { zeroFrame with verifierKeys := true }

def changedIteration : CompleteHashFrame :=
  { zeroFrame with iteration := true }

def changedInitialState : CompleteHashFrame :=
  { zeroFrame with initialState := true }

def changedCurrentState : CompleteHashFrame :=
  { zeroFrame with currentState := true }

def changedRunning : CompleteHashFrame :=
  { zeroFrame with running := true }

def changedProgramCounter : CompleteHashFrame :=
  { zeroFrame with programCounter := true }

/-- Setting one coordinate to a constant models an encoder that omits that
coordinate before the cryptographic hash. -/
def withoutVerifierKeys (frame : CompleteHashFrame) : CompleteHashFrame :=
  { frame with verifierKeys := false }

def withoutIteration (frame : CompleteHashFrame) : CompleteHashFrame :=
  { frame with iteration := false }

def withoutInitialState (frame : CompleteHashFrame) : CompleteHashFrame :=
  { frame with initialState := false }

def withoutCurrentState (frame : CompleteHashFrame) : CompleteHashFrame :=
  { frame with currentState := false }

def withoutRunning (frame : CompleteHashFrame) : CompleteHashFrame :=
  { frame with running := false }

def withoutProgramCounter (frame : CompleteHashFrame) : CompleteHashFrame :=
  { frame with programCounter := false }

/-- Omitting the verifier-key vector permits a proof-system substitution
before any cryptographic collision is needed. -/
theorem omitting_verifier_keys_aliases_distinct_frames :
    changedVerifierKeys ≠ zeroFrame /\
      withoutVerifierKeys changedVerifierKeys =
        withoutVerifierKeys zeroFrame := by
  decide

/-- Omitting the iteration permits two invocation positions to authenticate
to the same encoded frame. -/
theorem omitting_iteration_aliases_distinct_frames :
    changedIteration ≠ zeroFrame /\
      withoutIteration changedIteration = withoutIteration zeroFrame := by
  decide

/-- Omitting the initial state permits two executions to be spliced. -/
theorem omitting_initial_state_aliases_distinct_frames :
    changedInitialState ≠ zeroFrame /\
      withoutInitialState changedInitialState =
        withoutInitialState zeroFrame := by
  decide

/-- Omitting the current state permits application-state substitution. -/
theorem omitting_current_state_aliases_distinct_frames :
    changedCurrentState ≠ zeroFrame /\
      withoutCurrentState changedCurrentState =
        withoutCurrentState zeroFrame := by
  decide

/-- Omitting the running vector permits accumulator substitution. -/
theorem omitting_running_aliases_distinct_frames :
    changedRunning ≠ zeroFrame /\
      withoutRunning changedRunning = withoutRunning zeroFrame := by
  decide

/-- Omitting the one-based prior program counter permits selection of the
wrong NIFS relation and verifier key. -/
theorem omitting_program_counter_aliases_distinct_frames :
    changedProgramCounter ≠ zeroFrame /\
      withoutProgramCounter changedProgramCounter =
        withoutProgramCounter zeroFrame := by
  decide

/-! ## Missing running-state aliases permit accumulator substitution -/

/-- Minimal model with one running value authenticated by the prior F-prime
state hash and a second running value consumed by NIFS. -/
structure SplitRunningAuthority where
  authenticatedRunning : Bool
  nifsRunning : Bool
deriving DecidableEq, Repr

/-- The rejected relation checks each value in its own subsystem but does not
require both subsystems to use the same value. -/
def WeakChecks (value : SplitRunningAuthority) : Prop :=
  value.authenticatedRunning = false /\ value.nifsRunning = true

/-- The required physical alias between the state-hash input and the NIFS
input. -/
def ExactRunningAlias (value : SplitRunningAuthority) : Prop :=
  value.authenticatedRunning = value.nifsRunning

def substitutedRunning : SplitRunningAuthority :=
  { authenticatedRunning := false, nifsRunning := true }

/-- Without the alias, the prior-state check and the NIFS check can both pass
while they refer to different running accumulators. -/
theorem weak_checks_allow_running_substitution :
    WeakChecks substitutedRunning /\
      ¬ ExactRunningAlias substitutedRunning := by
  simp [WeakChecks, ExactRunningAlias, substitutedRunning]

/-- Adding the exact alias rejects the substitution deterministically. -/
theorem exact_running_alias_rejects_substitution :
    ¬ (WeakChecks substitutedRunning /\
      ExactRunningAlias substitutedRunning) := by
  simp [WeakChecks, ExactRunningAlias, substitutedRunning]

end Nightstream.Implementation.NebulaV2.FPrimeStateHashCountermodels
