import Nightstream.Assurance.FPrimeCircuitTrace

/-!
Contract: exact F' decider reduction into `ValidExecution`.

Owns: composition of one deployed acceptance transfer, the closed circuit
trace theorem, and `TRACE-VALID`. Every failure before the canonical F' trace
has one fixed name. A circuit-edge failure stays separate.

Does not own: an on-chain parser, a backend manifest, a verifier key, Spartan
or WHIR soundness, circuit lowering, Rust refinement, or a cryptographic
probability instantiation. A deployment closes this theorem only by proving
`AcceptanceTransfer` for its actual verifier and by bounding each reachable
`BoundaryClaim`.

Assurance tier: security reduction. No artifact, Boolean receipt, or digest is
proof authority in this module.
-/

set_option autoImplicit false

namespace Nightstream.Assurance.DeciderReduction

open Nightstream.HyperNova.Construction2
open Nightstream.Protocol.FPrime
open Nightstream.Assurance.FPrimeTrace
open Nightstream.Assurance.FPrimeCircuitTrace

universe uParams uStructure uHeader uDigest uRunning uFresh uNifsProof
  uNebulaDigest uNebulaOpen uStatement uProof

section

variable
  {Params : Type uParams}
  {StructureDigest : Type uStructure}
  {Header : Type uHeader}
  {Digest : Type uDigest}
  {Running : Type uRunning}
  {Fresh : Type uFresh}
  {NifsProof : Type uNifsProof}
  {Nebula : Type}
  {NebulaDigest : Type uNebulaDigest}
  {NebulaOpen : Type uNebulaOpen}
  {Statement : Type uStatement}
  {Proof : Type uProof}

local notation "Environment" =>
  FPrimeTrace.Environment Params StructureDigest Header Digest Running Fresh
    NifsProof Nebula NebulaDigest NebulaOpen

local notation "StepState" => State Digest Running Fresh Nebula

local notation "Invocation" =>
  FPrimeTrace.Invocation Digest Fresh NifsProof Nebula NebulaOpen

/-- Complete fixed census for failures between deployed acceptance and the
canonical closed F' trace. Keeping these constructors separate prevents one
assumption from silently covering a different bridge. -/
inductive BoundaryClaim where
  | parser
  | backendManifest
  | verifierKey
  | terminalBackend
  | publicEncoding
  | circuitManifest
  | circuitLowering
  | rustRefinement
  | paddedIdentity
  | sumCheck
  | algebraicMixing
  | coordinateFork
  | relaxedBinding
  | strongExtraction
  | poseidon2
  | fiatShamir
  | samplerExhaustion
deriving Repr, DecidableEq

/-- Concrete deployments assign a proposition to each fixed boundary name.
The mapping is data, so a review can inspect every term without changing the
reduction theorem. -/
structure BoundaryFailures (Statement : Type uStatement) (Proof : Type uProof) where
  occurs : BoundaryClaim -> Statement -> Proof -> Prop

/-- Verifier-owned semantic data needed after a proof is parsed. -/
structure Deployment
    [DecidableEq Digest]
    [DecidableEq Running]
    [DecidableEq Fresh]
    [DecidableEq NifsProof]
    [DecidableEq Nebula]
    [DecidableEq NebulaOpen]
    (environment : Environment)
    (Statement : Type uStatement)
    (Proof : Type uProof) where
  verify : Statement -> Proof -> Bool
  initialState : Statement -> StepState
  finalState : Statement -> StepState
  stepCount : Statement -> Nat
  TerminalValid : StepState -> Prop
  initialValid : forall statement,
    Step.InitialState environment.hashSemantics environment.stepSemantics
      environment.mode environment.context (initialState statement)

/-- A decoded accepted proof supplies a circuit trace and terminal fact. The
step count remains bound to the public statement. The trace may contain only
the named circuit-edge failure type supplied by the concrete circuit proof. -/
structure AcceptedCandidate
    [DecidableEq Digest]
    [DecidableEq Running]
    [DecidableEq Fresh]
    [DecidableEq NifsProof]
    [DecidableEq Nebula]
    [DecidableEq NebulaOpen]
    (environment : Environment)
    (deployment : Deployment environment Statement Proof)
    (CircuitBad : StepState -> StepState -> Invocation -> Prop)
    (statement : Statement) where
  schedule : List Nat
  trace : CandidateTrace environment (deployment.initialState statement)
    CircuitBad schedule (deployment.finalState statement)
  countBound : schedule.length = deployment.stepCount statement
  terminal : deployment.TerminalValid (deployment.finalState statement)

/-- The only deployment-specific theorem input. It starts at the actual
Boolean verifier. Acceptance must yield the exact decoded candidate or one
fixed boundary failure. It cannot return an unclassified failure. -/
def AcceptanceTransfer
    [DecidableEq Digest]
    [DecidableEq Running]
    [DecidableEq Fresh]
    [DecidableEq NifsProof]
    [DecidableEq Nebula]
    [DecidableEq NebulaOpen]
    (environment : Environment)
    (deployment : Deployment environment Statement Proof)
    (boundaries : BoundaryFailures Statement Proof)
    (CircuitBad : StepState -> StepState -> Invocation -> Prop) : Prop :=
  forall statement proof,
    deployment.verify statement proof = true ->
      Nonempty (AcceptedCandidate environment deployment CircuitBad statement) \/
        Exists fun claim => boundaries.occurs claim statement proof

/-- Exact bad-event result of the composed reduction. Deployment failures and
circuit-edge failures cannot be confused. -/
inductive BadEvent
    (boundaries : BoundaryFailures Statement Proof)
    (CircuitBad : StepState -> StepState -> Invocation -> Prop)
    (statement : Statement)
    (proof : Proof) : Prop where
  | boundary (claim : BoundaryClaim)
      (failure : boundaries.occurs claim statement proof)
  | circuit (failure : HasBad CircuitBad)

/-- Semantic execution selected by the public statement. -/
def StatementValid
    [DecidableEq Digest]
    [DecidableEq Running]
    [DecidableEq Fresh]
    [DecidableEq NifsProof]
    [DecidableEq Nebula]
    [DecidableEq NebulaOpen]
    (environment : Environment)
    (deployment : Deployment environment Statement Proof)
    (statement : Statement) : Prop :=
  ValidExecution (Edge environment) deployment.TerminalValid
    (deployment.initialState statement) (deployment.finalState statement)
    (deployment.stepCount statement)

/-- `DEC-SOUND`: actual deployed acceptance implies the exact semantic
execution or one reachable, fixed-name failure. The proof uses the circuit
trace closure theorem and `TRACE-VALID`; it does not accept a caller-provided
semantic conclusion. -/
theorem sound_or_bad
    [DecidableEq Digest]
    [DecidableEq Running]
    [DecidableEq Fresh]
    [DecidableEq NifsProof]
    [DecidableEq Nebula]
    [DecidableEq NebulaOpen]
    (environment : Environment)
    (deployment : Deployment environment Statement Proof)
    (boundaries : BoundaryFailures Statement Proof)
    (CircuitBad : StepState -> StepState -> Invocation -> Prop)
    (transfer : AcceptanceTransfer environment deployment boundaries CircuitBad)
    (statement : Statement)
    (proof : Proof)
    (accepted : deployment.verify statement proof = true) :
    StatementValid environment deployment statement \/
      BadEvent boundaries CircuitBad statement proof := by
  rcases transfer statement proof accepted with candidate | boundary
  · rcases candidate with ⟨candidate⟩
    rcases candidate_sound_or_bad environment
      (deployment.initialState statement) (deployment.finalState statement)
      CircuitBad candidate.schedule candidate.trace with trace | circuit
    · left
      have valid := accepted_trace_valid_execution environment
        (deployment.initialState statement) (deployment.finalState statement)
        candidate.schedule (deployment.initialValid statement) trace
        deployment.TerminalValid candidate.terminal
      simpa [StatementValid, candidate.countBound] using valid
    · exact Or.inr (.circuit circuit)
  · rcases boundary with ⟨claim, failure⟩
    exact Or.inr (.boundary claim failure)

/-- The composed theorem inhabits the repository's public verifier-reduction
target without weakening its semantic conclusion. -/
theorem verifierReductionTarget
    [DecidableEq Digest]
    [DecidableEq Running]
    [DecidableEq Fresh]
    [DecidableEq NifsProof]
    [DecidableEq Nebula]
    [DecidableEq NebulaOpen]
    (environment : Environment)
    (deployment : Deployment environment Statement Proof)
    (boundaries : BoundaryFailures Statement Proof)
    (CircuitBad : StepState -> StepState -> Invocation -> Prop)
    (transfer : AcceptanceTransfer environment deployment boundaries CircuitBad) :
    VerifierReductionTarget deployment.verify deployment.initialState
      deployment.finalState deployment.stepCount (Edge environment)
      deployment.TerminalValid (BadEvent boundaries CircuitBad) := by
  intro statement proof accepted
  exact sound_or_bad environment deployment boundaries CircuitBad transfer
    statement proof accepted

end

end Nightstream.Assurance.DeciderReduction
