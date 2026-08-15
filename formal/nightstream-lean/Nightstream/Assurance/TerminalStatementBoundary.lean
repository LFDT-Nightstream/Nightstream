import Nightstream.SuperNeo.CheckPlan

/-!
Exact control-flow model for the eleven terminal statement guards.

Assurance tier: model-level.

Owns: the verifier-recomputed terminal statement equations and one isolated
removal counterexample for each guard.

Does not own: Poseidon2 collision resistance, accumulator binding, the public
input codec, Rust refinement, the terminal R1CS, Spartan, WHIR, or a deployed
verifier key. The expected values in this model stand for verifier-owned
recomputation; separate theorems must justify those computations.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Assurance.TerminalStatementBoundary

open Nightstream.SuperNeo.CheckPlan

inductive Guard where
  | runningClaimCount
  | verifierKey
  | initialSemanticState
  | initialBoundary
  | programCounter
  | counters
  | freshBoundary
  | runningAccumulator
  | semanticState
  | stateXOut
  | freshPublicLink
deriving DecidableEq, Repr

def guardName : Guard → String
  | .runningClaimCount => "terminal.statement.running_claim_count"
  | .verifierKey => "terminal.statement.verifier_key"
  | .initialSemanticState => "terminal.statement.initial_semantic_state"
  | .initialBoundary => "terminal.statement.initial_boundary"
  | .programCounter => "terminal.statement.program_counter"
  | .counters => "terminal.statement.counters"
  | .freshBoundary => "terminal.statement.fresh_boundary"
  | .runningAccumulator => "terminal.statement.running_accumulator"
  | .semanticState => "terminal.statement.semantic_state"
  | .stateXOut => "terminal.statement.state_x_out"
  | .freshPublicLink => "terminal.statement.fresh_public_link"

def guards : List Guard := [
  .runningClaimCount,
  .verifierKey,
  .initialSemanticState,
  .initialBoundary,
  .programCounter,
  .counters,
  .freshBoundary,
  .runningAccumulator,
  .semanticState,
  .stateXOut,
  .freshPublicLink
]

def guardNames : List String := guards.map guardName

theorem guardNames_exact :
    guardNames = [
      "terminal.statement.running_claim_count",
      "terminal.statement.verifier_key",
      "terminal.statement.initial_semantic_state",
      "terminal.statement.initial_boundary",
      "terminal.statement.program_counter",
      "terminal.statement.counters",
      "terminal.statement.fresh_boundary",
      "terminal.statement.running_accumulator",
      "terminal.statement.semantic_state",
      "terminal.statement.state_x_out",
      "terminal.statement.fresh_public_link"
    ] := by
  rfl

/-- Abstract terminal statement plus verifier-recomputed expected values. -/
structure Candidate where
  runningClaimCount : Nat
  expectedRunningClaimCount : Nat
  verifierKey : Bool
  expectedVerifierKey : Bool
  initialSemanticState : Bool
  expectedInitialSemanticState : Bool
  initialBoundary : Bool
  expectedInitialBoundary : Bool
  programCounter : Nat
  chunkCount : Nat
  stepCount : Nat
  freshBoundary : Bool
  publicTrace : Bool
  expectedFreshBoundary : Bool
  runningAccumulator : Bool
  expectedRunningAccumulator : Bool
  stateless : Bool
  semanticState : Bool
  stateXOut : Bool
  expectedStateXOut : Bool
  freshPublicLinkAccepted : Bool
deriving DecidableEq, Repr

def semantics : Guard → Candidate → Prop
  | .runningClaimCount, candidate =>
      candidate.runningClaimCount = candidate.expectedRunningClaimCount
  | .verifierKey, candidate =>
      candidate.verifierKey = candidate.expectedVerifierKey
  | .initialSemanticState, candidate =>
      candidate.initialSemanticState = candidate.expectedInitialSemanticState
  | .initialBoundary, candidate =>
      candidate.initialBoundary = candidate.expectedInitialBoundary
  | .programCounter, candidate => candidate.programCounter = 1
  | .counters, candidate =>
      candidate.chunkCount ≠ 0 ∧ candidate.stepCount ≠ 0 ∧
        candidate.chunkCount = candidate.stepCount
  | .freshBoundary, candidate =>
      candidate.freshBoundary = candidate.expectedFreshBoundary ∧
        candidate.publicTrace = candidate.expectedFreshBoundary
  | .runningAccumulator, candidate =>
      candidate.runningAccumulator = candidate.expectedRunningAccumulator
  | .semanticState, candidate =>
      candidate.stateless = false ∨
        candidate.semanticState = candidate.runningAccumulator
  | .stateXOut, candidate =>
      candidate.stateXOut = candidate.expectedStateXOut
  | .freshPublicLink, candidate => candidate.freshPublicLinkAccepted = true

def Target (candidate : Candidate) : Prop :=
  candidate.runningClaimCount = candidate.expectedRunningClaimCount ∧
  candidate.verifierKey = candidate.expectedVerifierKey ∧
  candidate.initialSemanticState = candidate.expectedInitialSemanticState ∧
  candidate.initialBoundary = candidate.expectedInitialBoundary ∧
  candidate.programCounter = 1 ∧
  candidate.chunkCount ≠ 0 ∧
  candidate.stepCount ≠ 0 ∧
  candidate.chunkCount = candidate.stepCount ∧
  candidate.freshBoundary = candidate.expectedFreshBoundary ∧
  candidate.publicTrace = candidate.expectedFreshBoundary ∧
  candidate.runningAccumulator = candidate.expectedRunningAccumulator ∧
  (candidate.stateless = false ∨
    candidate.semanticState = candidate.runningAccumulator) ∧
  candidate.stateXOut = candidate.expectedStateXOut ∧
  candidate.freshPublicLinkAccepted = true

private instance targetDecidable (candidate : Candidate) :
    Decidable (Target candidate) := by
  unfold Target
  infer_instance

def verify (candidate : Candidate) : Bool := decide (Target candidate)

theorem accepts_iff_target (candidate : Candidate) :
    Accepts semantics guards candidate ↔ Target candidate := by
  simp [Accepts, guards, semantics, Target, and_assoc]

theorem verify_eq_true_iff_target (candidate : Candidate) :
    verify candidate = true ↔ Target candidate := by
  simp [verify]

def valid : Candidate where
  runningClaimCount := 1
  expectedRunningClaimCount := 1
  verifierKey := false
  expectedVerifierKey := false
  initialSemanticState := false
  expectedInitialSemanticState := false
  initialBoundary := false
  expectedInitialBoundary := false
  programCounter := 1
  chunkCount := 1
  stepCount := 1
  freshBoundary := false
  publicTrace := false
  expectedFreshBoundary := false
  runningAccumulator := false
  expectedRunningAccumulator := false
  stateless := true
  semanticState := false
  stateXOut := false
  expectedStateXOut := false
  freshPublicLinkAccepted := true

def removalWitness : Guard → Candidate
  | .runningClaimCount => { valid with runningClaimCount := 2 }
  | .verifierKey => { valid with verifierKey := true }
  | .initialSemanticState => { valid with initialSemanticState := true }
  | .initialBoundary => { valid with initialBoundary := true }
  | .programCounter => { valid with programCounter := 2 }
  | .counters => { valid with chunkCount := 2 }
  | .freshBoundary => { valid with freshBoundary := true }
  | .runningAccumulator =>
      { valid with runningAccumulator := true, semanticState := true }
  | .semanticState => { valid with semanticState := true }
  | .stateXOut => { valid with stateXOut := true }
  | .freshPublicLink => { valid with freshPublicLinkAccepted := false }

theorem removalWitness_accepts_without (removed : Guard) :
    Accepts semantics (without guards removed) (removalWitness removed) := by
  cases removed <;>
    intro retained member <;>
    cases retained <;>
    simp [without, guards, semantics, removalWitness, valid] at member ⊢

theorem removalWitness_rejects_target (removed : Guard) :
    ¬ Target (removalWitness removed) := by
  cases removed <;> simp [Target, removalWitness, valid]

theorem retained_necessary (guard : Guard) :
    NecessaryForSoundness semantics Target guards guard :=
  ⟨removalWitness guard, removalWitness_accepts_without guard,
    removalWitness_rejects_target guard⟩

/-- Model-level inclusion-minimality of the exact eleven-check statement
boundary. -/
theorem inclusionMinimalSound :
    InclusionMinimalSound semantics Target guards := by
  apply inclusionMinimalSound_of_witnesses
  · intro candidate accepted
    exact (accepts_iff_target candidate).1 accepted
  · intro guard _member
    exact retained_necessary guard

end Nightstream.Assurance.TerminalStatementBoundary
