import Nightstream.Protocol.FPrime.Step

/-!
Contract: Rust-shaped F' lifecycle verifier refinement.

`verify` mirrors the checks jointly owned by lifecycle replay and
`paper::f_prime::native::verify`: exact Initial/NoFold versus
Active/Recursive dispatch, canonical entry state, the delayed prior-fresh
link, verifier-driven NIFS output, nonempty installed batch, semantic and
Nebula advance, deterministic state advance, and recomputed `x_out`.

The implementation is deliberately separate from `Step.checkLocal`: it runs an
ordered error-producing program. `verify_eq_ok_iff_checkLocal` proves the two
executables accept exactly the same inputs. Counter representation/overflow is
owned by `FPR-COUNTER-REFINE`; this module uses the mathematical `Nat` state
shared with the M3 relation.
-/

namespace Nightstream.Implementation.Rust.FPrime

open Nightstream.HyperNova.Construction2
open Nightstream.Protocol.FPrime

universe uDigest uParams uStructure uHeader uRunning uFresh uNifsProof
  uNebulaDigest uNebulaOpen

inductive Error where
  | variant
  | entryState
  | priorLatestEmpty
  | priorLink
  | nifs
  | emptyStep
  | semanticAdvance
  | nebulaAdvance
  | nextState
  | xOut
deriving Repr, DecidableEq

def runChecks : List (Bool × Error) → Except Error Unit
  | [] => .ok ()
  | (true, _) :: checks => runChecks checks
  | (false, error) :: _ => .error error

theorem runChecks_eq_ok_iff (checks : List (Bool × Error)) :
    runChecks checks = .ok () ↔ checks.all (fun check => check.1) = true := by
  induction checks with
  | nil => simp [runChecks]
  | cons check checks inductionHypothesis =>
      cases check with
      | mk passes error => cases passes <;> simp [runChecks, inductionHypothesis]

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

local notation "HashSemantics" =>
  XOut.Semantics Params StructureDigest Header Digest Nebula NebulaDigest

local notation "StepSemantics" =>
  Step.Semantics Digest Running Fresh NifsProof Nebula NebulaOpen

local notation "StepState" => State Digest Running Fresh Nebula

local notation "StepInput" => Step.Input Fresh Nebula NebulaOpen

local notation "StepProof" => Step.Proof Digest NifsProof NebulaOpen

local notation "StepContext" =>
  XOut.Context Params StructureDigest Header Digest

def checkPinned
    [DecidableEq Digest]
    (hashSemantics : HashSemantics)
    (mode : XOut.Mode)
    (context : StepContext)
    (state : StepState) : Bool :=
  decide (state.z0 = XOut.initialBoundary hashSemantics context) &&
  decide (state.initialSemanticState = context.initialSemanticState) &&
  decide (state.publicTrace = state.zi) &&
  match mode with
  | .stateless => decide (state.semanticState = state.accumulatorDigest)
  | .stateful => true

theorem checkPinned_eq_true_iff
    [DecidableEq Digest]
    (hashSemantics : HashSemantics)
    (mode : XOut.Mode)
    (context : StepContext)
    (state : StepState) :
    checkPinned hashSemantics mode context state = true ↔
      XOut.StatePinned hashSemantics mode context state := by
  cases mode with
  | stateless =>
      constructor
      · intro accepted
        have fields :
            state.z0 = XOut.initialBoundary hashSemantics context ∧
            state.initialSemanticState = context.initialSemanticState ∧
            state.publicTrace = state.zi ∧
            state.semanticState = state.accumulatorDigest := by
          simpa [checkPinned, and_assoc] using accepted
        exact {
          initialBoundaryPinned := fields.1
          initialSemanticStatePinned := fields.2.1
          publicTraceMirrorsBoundary := fields.2.2.1
          statelessSemanticEqualsAccumulator := fun _ => fields.2.2.2
        }
      · intro pinned
        simp [checkPinned, pinned.initialBoundaryPinned,
          pinned.initialSemanticStatePinned, pinned.publicTraceMirrorsBoundary,
          pinned.statelessSemanticEqualsAccumulator rfl]
  | stateful =>
      constructor
      · intro accepted
        have fields :
            state.z0 = XOut.initialBoundary hashSemantics context ∧
            state.initialSemanticState = context.initialSemanticState ∧
            state.publicTrace = state.zi := by
          simpa [checkPinned, and_assoc] using accepted
        exact {
          initialBoundaryPinned := fields.1
          initialSemanticStatePinned := fields.2.1
          publicTraceMirrorsBoundary := fields.2.2
          statelessSemanticEqualsAccumulator := by simp
        }
      · intro pinned
        simp [checkPinned, pinned.initialBoundaryPinned,
          pinned.initialSemanticStatePinned, pinned.publicTraceMirrorsBoundary]

def checkInitial
    [DecidableEq Digest]
    [DecidableEq Running]
    [DecidableEq Fresh]
    [DecidableEq Nebula]
    (hashSemantics : HashSemantics)
    (stepSemantics : StepSemantics)
    (_mode : XOut.Mode)
    (context : StepContext)
    (state : StepState) : Bool :=
  decide (state.pc = 1) &&
  decide (state.chunkCount = 0) &&
  decide (state.stepCount = 0) &&
  decide (state.z0 = state.zi) &&
  decide (state.z0 = XOut.initialBoundary hashSemantics context) &&
  decide (state.publicTrace = XOut.publicTraceSeed hashSemantics context) &&
  decide (state.initialSemanticState = context.initialSemanticState) &&
  decide (state.accumulatorDigest = stepSemantics.initialAccumulatorDigest) &&
  decide (state.nebula = stepSemantics.initialNebula) &&
  decide (state.proof = .initial) &&
  decide (state.semanticState = state.initialSemanticState)

theorem checkInitial_eq_true_iff
    [DecidableEq Digest]
    [DecidableEq Running]
    [DecidableEq Fresh]
    [DecidableEq Nebula]
    (hashSemantics : HashSemantics)
    (stepSemantics : StepSemantics)
    (mode : XOut.Mode)
    (context : StepContext)
    (state : StepState) :
    checkInitial hashSemantics stepSemantics mode context state = true ↔
      Step.InitialState hashSemantics stepSemantics mode context state := by
  simp [checkInitial, Step.InitialState, and_assoc]

def checkActive
    [DecidableEq Digest]
    [DecidableEq Running]
    [DecidableEq Fresh]
    [DecidableEq Nebula]
    (hashSemantics : HashSemantics)
    (stepSemantics : StepSemantics)
    (mode : XOut.Mode)
    (context : StepContext)
    (state : StepState)
    (running : Running)
    (latest : List Fresh) : Bool :=
  decide (state.pc = 1) &&
  decide (state.chunkCount ≠ 0) &&
  decide (state.stepCount ≠ 0) &&
  decide (state.proof = .active running latest) &&
  decide (state.accumulatorDigest = stepSemantics.runningDigest running) &&
  checkPinned hashSemantics mode context state

theorem checkActive_eq_true_iff
    [DecidableEq Digest]
    [DecidableEq Running]
    [DecidableEq Fresh]
    [DecidableEq Nebula]
    (hashSemantics : HashSemantics)
    (stepSemantics : StepSemantics)
    (mode : XOut.Mode)
    (context : StepContext)
    (state : StepState)
    (running : Running)
    (latest : List Fresh) :
    checkActive hashSemantics stepSemantics mode context state running latest = true ↔
      Step.ActiveState hashSemantics stepSemantics mode context state running latest := by
  simp [checkActive, Step.ActiveState, checkPinned_eq_true_iff, and_assoc]

def checkSemanticAdvance
    [DecidableEq Digest]
    (stepSemantics : StepSemantics)
    (mode : XOut.Mode)
    (prior : StepState)
    (nextRunning : Running)
    (input : StepInput)
    (proof : StepProof) : Bool :=
  match mode with
  | .stateless => decide (proof.semanticStateDigest = stepSemantics.runningDigest nextRunning)
  | .stateful =>
      stepSemantics.applicationStep prior.semanticState input.nextLatest
        proof.semanticStateDigest

theorem checkSemanticAdvance_eq_true_iff
    [DecidableEq Digest]
    (stepSemantics : StepSemantics)
    (mode : XOut.Mode)
    (prior : StepState)
    (nextRunning : Running)
    (input : StepInput)
    (proof : StepProof) :
    checkSemanticAdvance stepSemantics mode prior nextRunning input proof = true ↔
      Step.SemanticAdvance stepSemantics mode prior nextRunning input proof := by
  cases mode <;> simp [checkSemanticAdvance, Step.SemanticAdvance]

def checkNebulaAdvance
    [DecidableEq NebulaOpen]
    (stepSemantics : StepSemantics)
    (prior : StepState)
    (input : StepInput)
    (proof : StepProof) : Bool :=
  decide (proof.nebulaOpen = input.nebulaOpen) &&
  stepSemantics.nebulaVerify prior.nebula input.nebulaOpen
    (Step.installedNebula prior input)

theorem checkNebulaAdvance_eq_true_iff
    [DecidableEq NebulaOpen]
    (stepSemantics : StepSemantics)
    (prior : StepState)
    (input : StepInput)
    (proof : StepProof) :
    checkNebulaAdvance stepSemantics prior input proof = true ↔
      Step.NebulaAdvance stepSemantics prior input proof := by
  simp [checkNebulaAdvance, Step.NebulaAdvance]

def checkFreshLinked
    (stepSemantics : StepSemantics)
    (digest : Digest)
    (fresh : List Fresh) : Bool :=
  fresh.all (stepSemantics.freshLink digest)

theorem checkFreshLinked_eq_true_iff
    (stepSemantics : StepSemantics)
    (digest : Digest)
    (fresh : List Fresh) :
    checkFreshLinked stepSemantics digest fresh = true ↔
      Step.FreshLinked stepSemantics.freshLink digest fresh := by
  rfl

def verifyBase
    [DecidableEq Digest]
    [DecidableEq Running]
    [DecidableEq Fresh]
    [DecidableEq NifsProof]
    [DecidableEq Nebula]
    [DecidableEq NebulaOpen]
    (hashSemantics : HashSemantics)
    (stepSemantics : StepSemantics)
    (mode : XOut.Mode)
    (context : StepContext)
    (prior next : StepState)
    (input : StepInput)
    (proof : StepProof) : Except Error Unit :=
  runChecks
    [ (checkInitial hashSemantics stepSemantics mode context prior,
        .entryState)
    , (decide (input.nextLatest ≠ []), .emptyStep)
    , (checkSemanticAdvance stepSemantics mode prior
        stepSemantics.emptyRunning input proof, .semanticAdvance)
    , (checkNebulaAdvance stepSemantics prior input proof, .nebulaAdvance)
    , (decide (next = Step.advancedState stepSemantics prior
        stepSemantics.emptyRunning input proof), .nextState)
    , (decide (proof.xOut = XOut.compute hashSemantics mode context next), .xOut)
    ]

theorem verifyBase_eq_ok_iff
    [DecidableEq Digest]
    [DecidableEq Running]
    [DecidableEq Fresh]
    [DecidableEq NifsProof]
    [DecidableEq Nebula]
    [DecidableEq NebulaOpen]
    (hashSemantics : HashSemantics)
    (stepSemantics : StepSemantics)
    (mode : XOut.Mode)
    (context : StepContext)
    (prior next : StepState)
    (input : StepInput)
    (proof : StepProof)
    (noFold : proof.fold = .noFold) :
    verifyBase hashSemantics stepSemantics mode context prior next input proof = .ok () ↔
      Step.BaseLocalHolds hashSemantics stepSemantics mode context
        prior next input proof := by
  simp [verifyBase, runChecks_eq_ok_iff, Step.BaseLocalHolds, noFold,
    checkInitial_eq_true_iff, checkSemanticAdvance_eq_true_iff,
    checkNebulaAdvance_eq_true_iff]

def recursiveEntryChecks
    [DecidableEq Digest]
    [DecidableEq Running]
    [DecidableEq Fresh]
    [DecidableEq Nebula]
    (hashSemantics : HashSemantics)
    (stepSemantics : StepSemantics)
    (mode : XOut.Mode)
    (context : StepContext)
    (prior : StepState)
    (running : Running)
    (latest : List Fresh) : List (Bool × Error) :=
  [ (checkActive hashSemantics stepSemantics mode context
      prior running latest, .entryState)
  , (decide (latest ≠ []), .priorLatestEmpty)
  , (checkFreshLinked stepSemantics
      (XOut.compute hashSemantics mode context prior) latest, .priorLink)
  ]

def recursiveOutputChecks
    [DecidableEq Digest]
    [DecidableEq Running]
    [DecidableEq Fresh]
    [DecidableEq Nebula]
    [DecidableEq NebulaOpen]
    (hashSemantics : HashSemantics)
    (stepSemantics : StepSemantics)
    (mode : XOut.Mode)
    (context : StepContext)
    (prior next : StepState)
    (input : StepInput)
    (proof : StepProof)
    (nextRunning : Running) : List (Bool × Error) :=
  [ (decide (input.nextLatest ≠ []), .emptyStep)
  , (checkSemanticAdvance stepSemantics mode prior
      nextRunning input proof, .semanticAdvance)
  , (checkNebulaAdvance stepSemantics prior input proof, .nebulaAdvance)
  , (decide (next = Step.advancedState stepSemantics prior
      nextRunning input proof), .nextState)
  , (decide (proof.xOut = XOut.compute hashSemantics mode context next), .xOut)
  ]

def verifyRecursive
    [DecidableEq Digest]
    [DecidableEq Running]
    [DecidableEq Fresh]
    [DecidableEq NifsProof]
    [DecidableEq Nebula]
    [DecidableEq NebulaOpen]
    (hashSemantics : HashSemantics)
    (stepSemantics : StepSemantics)
    (mode : XOut.Mode)
    (context : StepContext)
    (prior next : StepState)
    (input : StepInput)
    (proof : StepProof)
    (running : Running)
    (latest : List Fresh)
    (nifsProof : NifsProof) : Except Error Unit :=
  match runChecks (recursiveEntryChecks hashSemantics stepSemantics mode
      context prior running latest) with
  | .error error => .error error
  | .ok _ =>
      match stepSemantics.nifsVerify (Step.nifsContext stepSemantics prior input)
          running latest nifsProof with
      | none => .error .nifs
      | some nextRunning =>
          runChecks (recursiveOutputChecks hashSemantics stepSemantics mode
            context prior next input proof nextRunning)

private theorem recursiveEntryChecks_eq_ok_iff
    [DecidableEq Digest]
    [DecidableEq Running]
    [DecidableEq Fresh]
    [DecidableEq Nebula]
    (hashSemantics : HashSemantics)
    (stepSemantics : StepSemantics)
    (mode : XOut.Mode)
    (context : StepContext)
    (prior : StepState)
    (running : Running)
    (latest : List Fresh) :
    runChecks (recursiveEntryChecks hashSemantics stepSemantics mode context
        prior running latest) = .ok () ↔
      Step.ActiveState hashSemantics stepSemantics mode context prior running latest ∧
      latest ≠ [] ∧
      Step.FreshLinked stepSemantics.freshLink
        (XOut.compute hashSemantics mode context prior) latest := by
  simp [recursiveEntryChecks, runChecks_eq_ok_iff, checkActive_eq_true_iff,
    checkFreshLinked_eq_true_iff]

private theorem recursiveOutputChecks_eq_ok_iff
    [DecidableEq Digest]
    [DecidableEq Running]
    [DecidableEq Fresh]
    [DecidableEq Nebula]
    [DecidableEq NebulaOpen]
    (hashSemantics : HashSemantics)
    (stepSemantics : StepSemantics)
    (mode : XOut.Mode)
    (context : StepContext)
    (prior next : StepState)
    (input : StepInput)
    (proof : StepProof)
    (nextRunning : Running) :
    runChecks (recursiveOutputChecks hashSemantics stepSemantics mode context
        prior next input proof nextRunning) = .ok () ↔
      input.nextLatest ≠ [] ∧
      Step.SemanticAdvance stepSemantics mode prior nextRunning input proof ∧
      Step.NebulaAdvance stepSemantics prior input proof ∧
      next = Step.advancedState stepSemantics prior nextRunning input proof ∧
      proof.xOut = XOut.compute hashSemantics mode context next := by
  simp [recursiveOutputChecks, runChecks_eq_ok_iff,
    checkSemanticAdvance_eq_true_iff, checkNebulaAdvance_eq_true_iff]

theorem verifyRecursive_eq_ok_iff
    [DecidableEq Digest]
    [DecidableEq Running]
    [DecidableEq Fresh]
    [DecidableEq NifsProof]
    [DecidableEq Nebula]
    [DecidableEq NebulaOpen]
    (hashSemantics : HashSemantics)
    (stepSemantics : StepSemantics)
    (mode : XOut.Mode)
    (context : StepContext)
    (prior next : StepState)
    (input : StepInput)
    (proof : StepProof)
    (running : Running)
    (latest : List Fresh)
    (nifsProof : NifsProof)
    (recursive : proof.fold = .recursive nifsProof) :
    verifyRecursive hashSemantics stepSemantics mode context prior next input
        proof running latest nifsProof = .ok () ↔
      Step.RecursiveLocalHolds hashSemantics stepSemantics mode context
        prior next input proof running latest nifsProof := by
  unfold verifyRecursive
  have entryIff :
      runChecks (recursiveEntryChecks hashSemantics stepSemantics mode context
        prior running latest) = .ok () ↔
      Step.ActiveState hashSemantics stepSemantics mode context prior running latest ∧
      latest ≠ [] ∧
      Step.FreshLinked stepSemantics.freshLink
        (XOut.compute hashSemantics mode context prior) latest := by
    exact recursiveEntryChecks_eq_ok_iff hashSemantics stepSemantics mode
      context prior running latest
  cases entryResult : runChecks (recursiveEntryChecks hashSemantics
      stepSemantics mode context prior running latest) with
  | error error =>
      have entryInvalid : ¬(
          Step.ActiveState hashSemantics stepSemantics mode context prior running latest ∧
          latest ≠ [] ∧
          Step.FreshLinked stepSemantics.freshLink
            (XOut.compute hashSemantics mode context prior) latest) := by
        intro holds
        have accepted := entryIff.2 holds
        rw [entryResult] at accepted
        contradiction
      simp [Step.RecursiveLocalHolds, recursive]
      intro active latestNonempty priorLinked
      exact False.elim (entryInvalid ⟨active, latestNonempty, priorLinked⟩)
  | ok value =>
      cases value
      have entryHolds := entryIff.1 entryResult
      cases nifsResult : stepSemantics.nifsVerify
          (Step.nifsContext stepSemantics prior input) running latest nifsProof with
      | none =>
          simp [nifsResult, Step.RecursiveLocalHolds, recursive, entryHolds]
      | some nextRunning =>
          simp [nifsResult, Step.RecursiveLocalHolds, recursive, entryHolds,
            recursiveOutputChecks_eq_ok_iff]

/-- Joint lifecycle/native control flow. Variant mismatches reject immediately. -/
def verify
    [DecidableEq Digest]
    [DecidableEq Running]
    [DecidableEq Fresh]
    [DecidableEq NifsProof]
    [DecidableEq Nebula]
    [DecidableEq NebulaOpen]
    (hashSemantics : HashSemantics)
    (stepSemantics : StepSemantics)
    (mode : XOut.Mode)
    (context : StepContext)
    (prior next : StepState)
    (input : StepInput)
    (proof : StepProof) : Except Error Unit :=
  match prior.proof, proof.fold with
  | .initial, .noFold =>
      verifyBase hashSemantics stepSemantics mode context prior next input proof
  | .active running latest, .recursive nifsProof =>
      verifyRecursive hashSemantics stepSemantics mode context prior next input
        proof running latest nifsProof
  | _, _ => .error .variant

theorem verify_eq_ok_iff_localHolds
    [DecidableEq Digest]
    [DecidableEq Running]
    [DecidableEq Fresh]
    [DecidableEq NifsProof]
    [DecidableEq Nebula]
    [DecidableEq NebulaOpen]
    (hashSemantics : HashSemantics)
    (stepSemantics : StepSemantics)
    (mode : XOut.Mode)
    (context : StepContext)
    (prior next : StepState)
    (input : StepInput)
    (proof : StepProof) :
    verify hashSemantics stepSemantics mode context prior next input proof = .ok () ↔
      Step.LocalHolds hashSemantics stepSemantics mode context prior next input proof := by
  cases priorProof : prior.proof with
  | initial =>
      cases foldProof : proof.fold with
      | noFold =>
          simpa [verify, Step.LocalHolds, priorProof, foldProof] using
            verifyBase_eq_ok_iff hashSemantics stepSemantics mode context
              prior next input proof foldProof
      | recursive nifsProof => simp [verify, Step.LocalHolds, priorProof, foldProof]
  | active running latest =>
      cases foldProof : proof.fold with
      | noFold => simp [verify, Step.LocalHolds, priorProof, foldProof]
      | recursive nifsProof =>
          simpa [verify, Step.LocalHolds, priorProof, foldProof] using
            verifyRecursive_eq_ok_iff hashSemantics stepSemantics mode context
              prior next input proof running latest nifsProof foldProof

/-- `RUST-REFINE` step slice: native success is exactly M3 local acceptance. -/
theorem verify_eq_ok_iff_checkLocal
    [DecidableEq Digest]
    [DecidableEq Running]
    [DecidableEq Fresh]
    [DecidableEq NifsProof]
    [DecidableEq Nebula]
    [DecidableEq NebulaOpen]
    (hashSemantics : HashSemantics)
    (stepSemantics : StepSemantics)
    (mode : XOut.Mode)
    (context : StepContext)
    (prior next : StepState)
    (input : StepInput)
    (proof : StepProof) :
    verify hashSemantics stepSemantics mode context prior next input proof = .ok () ↔
      Step.checkLocal hashSemantics stepSemantics mode context prior next input proof = true := by
  rw [verify_eq_ok_iff_localHolds, Step.checkLocal_eq_true_iff_localHolds]

/-- A successful native producer plus its verifier-owned consumer/terminal link
closes the full M3 step relation. -/
theorem success_with_outgoing_refines_step
    [DecidableEq Digest]
    [DecidableEq Running]
    [DecidableEq Fresh]
    [DecidableEq NifsProof]
    [DecidableEq Nebula]
    [DecidableEq NebulaOpen]
    (hashSemantics : HashSemantics)
    (stepSemantics : StepSemantics)
    (mode : XOut.Mode)
    (context : StepContext)
    (prior next : StepState)
    (input : StepInput)
    (proof : StepProof)
    (accepted :
      verify hashSemantics stepSemantics mode context prior next input proof = .ok ())
    (outgoing : Step.OutgoingLinked stepSemantics input proof) :
    Step.Holds hashSemantics stepSemantics mode context prior next input proof :=
  Step.closeLocal hashSemantics stepSemantics mode context prior next input proof
    ((verify_eq_ok_iff_localHolds hashSemantics stepSemantics mode context
      prior next input proof).1 accepted)
    outgoing

theorem invalid_has_named_rejection
    [DecidableEq Digest]
    [DecidableEq Running]
    [DecidableEq Fresh]
    [DecidableEq NifsProof]
    [DecidableEq Nebula]
    [DecidableEq NebulaOpen]
    (hashSemantics : HashSemantics)
    (stepSemantics : StepSemantics)
    (mode : XOut.Mode)
    (context : StepContext)
    (prior next : StepState)
    (input : StepInput)
    (proof : StepProof)
    (invalid : ¬ Step.LocalHolds hashSemantics stepSemantics mode context
      prior next input proof) :
    ∃ error, verify hashSemantics stepSemantics mode context prior next input proof =
      .error error := by
  have notAccepted :
      verify hashSemantics stepSemantics mode context prior next input proof ≠ .ok () := by
    intro accepted
    exact invalid ((verify_eq_ok_iff_localHolds hashSemantics stepSemantics mode
      context prior next input proof).1 accepted)
  cases result : verify hashSemantics stepSemantics mode context prior next input proof with
  | ok value =>
      cases value
      exact False.elim (notAccepted result)
  | error error => exact ⟨error, by simp⟩

end

end Nightstream.Implementation.Rust.FPrime
