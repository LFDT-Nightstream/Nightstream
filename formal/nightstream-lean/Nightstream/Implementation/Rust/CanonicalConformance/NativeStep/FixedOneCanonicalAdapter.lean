import Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.Composition
import Nightstream.Implementation.Rust.FPrime
import Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne

/-!
Contract: typed adapter from the closed direct-F' step to the frozen
fixed-one Construction-2 verifier.

Assurance tier: model-level. The receipt corollary is conditional on the
existing exact receipt checker; it is not a compiled-Rust semantics theorem.

Owns:
- an explicit fixed-one paper `Setup` and `Machine` over direct-F' data;
- totalized state, digest, fresh, proof, and witness maps;
- exact prior/next hash-preimage alignment;
- separation of the delayed outgoing fresh link from the native producer.

Does not own:
- Rust-source or compiled-Rust semantics;
- primitive hash/NIFS correctness;
- terminal ownership of the delayed outgoing link;
- concrete byte/field serialization or R1CS rows.

The only rejecting value introduced by the adapter is `none` in the paper
state/digest codomains.  It is reached only by the enumerated direct checks
below; there is no caller-selectable refinement-failure branch.

Emits constraints: no.
-/

namespace Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.FixedOneCanonicalAdapter

open Nightstream.HyperNova.Construction2
open Nightstream.HyperNova.Construction2.Paper
open Nightstream.HyperNova.NonInteractiveMultiFold
open Nightstream.Protocol.FPrime

universe uParams uStructure uHeader uDigest uRunning uFresh uNifsProof
  uNebulaDigest uNebulaOpen

/-- Verifier-owned direct-F' semantics captured by one fixed-one adapter. -/
structure Parameters
    (Params : Type uParams)
    (StructureDigest : Type uStructure)
    (Header : Type uHeader)
    (Digest : Type uDigest)
    (Running : Type uRunning)
    (Fresh : Type uFresh)
    (NifsProof : Type uNifsProof)
    (Nebula : Type)
    (NebulaDigest : Type uNebulaDigest)
    (NebulaOpen : Type uNebulaOpen) where
  hash :
    XOut.Semantics Params StructureDigest Header Digest Nebula NebulaDigest
  step :
    Step.Semantics Digest Running Fresh NifsProof Nebula NebulaOpen
  mode : XOut.Mode
  context : XOut.Context Params StructureDigest Header Digest

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

local notation "AdapterParameters" =>
  Parameters Params StructureDigest Header Digest Running Fresh NifsProof
    Nebula NebulaDigest NebulaOpen

local notation "DirectState" => State Digest Running Fresh Nebula
local notation "DirectInput" => Step.Input Fresh Nebula NebulaOpen
local notation "DirectProof" => Step.Proof Digest NifsProof NebulaOpen

/-- The paper application witness is exactly the direct verifier payload not
already present in the fixed-one public input. -/
structure Witness
    (Digest : Type uDigest)
    (Fresh : Type uFresh)
    (NifsProof : Type uNifsProof)
    (Nebula : Type)
    (NebulaOpen : Type uNebulaOpen) where
  input : Step.Input Fresh Nebula NebulaOpen
  proof : Step.Proof Digest NifsProof NebulaOpen
deriving Repr, DecidableEq

/-- The paper fresh value retains the exact claimed prior digest, NIFS
context, and ordered prior batch consumed by the recursive branch. -/
structure FreshInput
    (Digest : Type uDigest)
    (Fresh : Type uFresh)
    (Nebula : Type) where
  claimedDigest : Digest
  nifsContext : Step.NifsContext Digest Nebula
  ordered : List Fresh
deriving Repr, DecidableEq

/-- Equality of these two fields expresses both digest equality and the
nonempty all-coordinate public-link check. -/
structure Encoded (Digest : Type uDigest) where
  digest : Option Digest
  linked : Bool
deriving Repr, DecidableEq

local notation "AdapterWitness" =>
  Witness Digest Fresh NifsProof Nebula NebulaOpen
local notation "AdapterFresh" => FreshInput Digest Fresh Nebula
local notation "AdapterEncoded" => Encoded Digest
local notation "PaperState" => Option DirectState
local notation "PaperDigest" => Option Digest
local notation "PaperProof" => Step.FoldProof NifsProof
local notation "PaperKey" =>
  XOut.Context Params StructureDigest Header Digest

/-- Canonical direct initial state determined entirely by verifier-owned
semantics and context. -/
def initialState (parameters : AdapterParameters) : DirectState :=
  let boundary := XOut.initialBoundary parameters.hash parameters.context
  {
    chunkCount := 0
    stepCount := 0
    z0 := boundary
    zi := boundary
    initialSemanticState := parameters.context.initialSemanticState
    semanticState := parameters.context.initialSemanticState
    pc := 1
    accumulatorDigest := parameters.step.initialAccumulatorDigest
    publicTrace := XOut.publicTraceSeed parameters.hash parameters.context
    proof := .initial
    nebula := parameters.step.initialNebula
  }

def runningOf
    (parameters : AdapterParameters)
    (state : DirectState) : Running :=
  match state.proof with
  | .initial => parameters.step.emptyRunning
  | .active running _ => running

def latestOf (state : DirectState) : List Fresh :=
  match state.proof with
  | .initial => []
  | .active _ latest => latest

/-- Every check after branch selection, excluding the prior recursive link
and the final XOut equality. -/
def postChecks
    [DecidableEq Digest]
    [DecidableEq NebulaOpen]
    (parameters : AdapterParameters)
    (prior : DirectState)
    (nextRunning : Running)
    (input : DirectInput)
    (proof : DirectProof) : Bool :=
  decide (input.nextLatest ≠ []) &&
  Nightstream.Implementation.Rust.FPrime.checkSemanticAdvance
    parameters.step parameters.mode prior nextRunning input proof &&
  Nightstream.Implementation.Rust.FPrime.checkNebulaAdvance
    parameters.step prior input proof &&
  Nightstream.Implementation.Rust.FPrime.checkFreshLinked
    parameters.step proof.xOut input.nextLatest

/-- Totalized application transition.  Each `none` branch is one exact
direct-F' rejection: variant, entry state, NIFS, empty next batch, semantic
advance, Nebula advance, or outgoing public link. -/
def application
    [DecidableEq Digest]
    [DecidableEq Running]
    [DecidableEq Fresh]
    [DecidableEq Nebula]
    [DecidableEq NebulaOpen]
    (parameters : AdapterParameters)
    (prior : PaperState)
    (witness : AdapterWitness) : PaperState :=
  match prior with
  | none => none
  | some state =>
      match state.proof, witness.proof.fold with
      | .initial, .noFold =>
          if Nightstream.Implementation.Rust.FPrime.checkInitial
              parameters.hash parameters.step parameters.mode
              parameters.context state &&
            postChecks parameters state parameters.step.emptyRunning
              witness.input witness.proof then
            some (Step.advancedState parameters.step state
              parameters.step.emptyRunning witness.input witness.proof)
          else
            none
      | .active running latest, .recursive nifsProof =>
          match parameters.step.nifsVerify
              (Step.nifsContext parameters.step state witness.input)
              running latest nifsProof with
          | none => none
          | some nextRunning =>
              if Nightstream.Implementation.Rust.FPrime.checkActive
                  parameters.hash parameters.step parameters.mode
                  parameters.context state running latest &&
                postChecks parameters state nextRunning witness.input
                  witness.proof then
                some (Step.advancedState parameters.step state nextRunning
                  witness.input witness.proof)
              else
                none
      | _, _ => none

def freshInput
    (parameters : AdapterParameters)
    (prior : DirectState)
    (input : DirectInput) : AdapterFresh :=
  {
    claimedDigest :=
      XOut.compute parameters.hash parameters.mode parameters.context prior
    nifsContext := Step.nifsContext parameters.step prior input
    ordered := latestOf prior
  }

def freshPublic
    (parameters : AdapterParameters)
    (fresh : AdapterFresh) : AdapterEncoded :=
  {
    digest := some fresh.claimedDigest
    linked :=
      decide (fresh.ordered ≠ []) &&
      fresh.ordered.all (parameters.step.freshLink fresh.claimedDigest)
  }

def encodeInstance (digest : PaperDigest) : AdapterEncoded :=
  {
    digest := digest
    linked := true
  }

def alignedHashPreimage
    [DecidableEq DirectState]
    [DecidableEq Running]
    (parameters : AdapterParameters)
    (preimage : HashPreimage PaperKey PaperState Running 1) : Bool :=
  match preimage.current with
  | none => false
  | some current =>
      decide (preimage.iteration = current.chunkCount) &&
      decide (preimage.z0 = some (initialState parameters)) &&
      decide
        (preimage.running
          Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected =
            runningOf parameters current) &&
      decide (preimage.pc = current.pc)

/-- The concrete paper hash consumes the complete fixed-one preimage.  It
returns a direct XOut only when every duplicated carrier coordinate agrees. -/
def paperHash
    [DecidableEq DirectState]
    [DecidableEq Running]
    (parameters : AdapterParameters)
    (preimage : HashPreimage PaperKey PaperState Running 1) : PaperDigest :=
  match preimage.current with
  | none => none
  | some current =>
      if alignedHashPreimage parameters preimage then
        some
          (XOut.compute parameters.hash parameters.mode parameters.context
            current)
      else
        none

def nifsVerifier (parameters : AdapterParameters) :
    Verifier PaperKey Running AdapterFresh PaperProof where
  verify := fun _ running fresh proof =>
    match proof with
    | .noFold => none
    | .recursive nifsProof =>
        parameters.step.nifsVerify fresh.nifsContext running fresh.ordered
          nifsProof

def setup (parameters : AdapterParameters) :
    Setup PaperKey Running AdapterFresh PaperProof 1 where
  verifierKeys := fun _ => parameters.context
  nifs := nifsVerifier parameters
  defaultRunning := parameters.step.emptyRunning

def machine
    [DecidableEq DirectState]
    [DecidableEq Running]
    [DecidableEq Digest]
    [DecidableEq Fresh]
    [DecidableEq Nebula]
    [DecidableEq NebulaOpen]
    (parameters : AdapterParameters) :
    Machine PaperKey PaperDigest PaperState AdapterWitness Running AdapterFresh
      AdapterEncoded 1 where
  control := fun _ _ =>
    Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected
  step := fun _ prior witness => application parameters prior witness
  freshPublic := freshPublic parameters
  encodeInstance := encodeInstance
  hash := paperHash parameters

def input
    (parameters : AdapterParameters)
    (prior : DirectState)
    (nextInput : DirectInput)
    (proof : DirectProof) :
    Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.Input
      PaperState AdapterWitness Running AdapterFresh PaperProof where
  iteration := prior.chunkCount
  z0 := some (initialState parameters)
  zi := some prior
  running := fun _ => runningOf parameters prior
  fresh := freshInput parameters prior nextInput
  witness := ⟨nextInput, proof⟩
  nifsProof := proof.fold

def output
    (parameters : AdapterParameters)
    (next : DirectState)
    (proof : DirectProof) :
    Output PaperDigest PaperState Running 1 where
  zNext := some next
  runningNext := fun _ => runningOf parameters next
  pcNext :=
    Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected
  x := some proof.xOut

@[simp] theorem initialState_holds
    (parameters : AdapterParameters) :
    Step.InitialState parameters.hash parameters.step parameters.mode
      parameters.context (initialState parameters) := by
  rcases parameters with ⟨hash, step, mode, context⟩
  cases mode <;>
    simp [initialState, Step.InitialState]

theorem initialState_eq_iff
    (parameters : AdapterParameters)
    (state : DirectState) :
    state = initialState parameters ↔
      Step.InitialState parameters.hash parameters.step parameters.mode
        parameters.context state := by
  constructor
  · intro equality
    subst state
    exact initialState_holds parameters
  · intro holds
    have ziBoundary :
        state.zi =
          XOut.initialBoundary parameters.hash parameters.context :=
      holds.2.2.2.1.symm.trans holds.2.2.2.2.1
    rcases parameters with ⟨hash, step, mode, context⟩
    cases state
    cases mode <;>
      simp_all [initialState, Step.InitialState]

theorem postChecks_eq_true_iff
    [DecidableEq Digest]
    [DecidableEq NebulaOpen]
    (parameters : AdapterParameters)
    (prior : DirectState)
    (nextRunning : Running)
    (nextInput : DirectInput)
    (proof : DirectProof) :
    postChecks parameters prior nextRunning nextInput proof = true ↔
      nextInput.nextLatest ≠ [] ∧
      Step.SemanticAdvance parameters.step parameters.mode prior nextRunning
        nextInput proof ∧
      Step.NebulaAdvance parameters.step prior nextInput proof ∧
      Step.OutgoingLinked parameters.step nextInput proof := by
  simp [postChecks,
    Nightstream.Implementation.Rust.FPrime.checkSemanticAdvance_eq_true_iff,
    Nightstream.Implementation.Rust.FPrime.checkNebulaAdvance_eq_true_iff,
    Nightstream.Implementation.Rust.FPrime.checkFreshLinked_eq_true_iff,
    Step.OutgoingLinked, and_assoc]

/-- Direct application obligations excluding the prior recursive link and
the final XOut equality, which are owned by the paper recursive branch and
paper output hash respectively. -/
def ApplicationCore
    (parameters : AdapterParameters)
    (prior next : DirectState)
    (nextInput : DirectInput)
    (proof : DirectProof) : Prop :=
  match prior.proof, proof.fold with
  | .initial, .noFold =>
      Step.InitialState parameters.hash parameters.step parameters.mode
        parameters.context prior ∧
      nextInput.nextLatest ≠ [] ∧
      Step.SemanticAdvance parameters.step parameters.mode prior
        parameters.step.emptyRunning nextInput proof ∧
      Step.NebulaAdvance parameters.step prior nextInput proof ∧
      Step.OutgoingLinked parameters.step nextInput proof ∧
      Step.advancedState parameters.step prior
        parameters.step.emptyRunning nextInput proof = next
  | .active running latest, .recursive nifsProof =>
      Step.ActiveState parameters.hash parameters.step parameters.mode
        parameters.context prior running latest ∧
      match parameters.step.nifsVerify
          (Step.nifsContext parameters.step prior nextInput)
          running latest nifsProof with
      | none => False
      | some nextRunning =>
          nextInput.nextLatest ≠ [] ∧
          Step.SemanticAdvance parameters.step parameters.mode prior
            nextRunning nextInput proof ∧
          Step.NebulaAdvance parameters.step prior nextInput proof ∧
          Step.OutgoingLinked parameters.step nextInput proof ∧
          Step.advancedState parameters.step prior nextRunning nextInput
            proof = next
  | _, _ => False

theorem application_eq_some_iff
    [DecidableEq Digest]
    [DecidableEq Running]
    [DecidableEq Fresh]
    [DecidableEq Nebula]
    [DecidableEq NebulaOpen]
    (parameters : AdapterParameters)
    (prior next : DirectState)
    (nextInput : DirectInput)
    (proof : DirectProof) :
    application parameters (some prior) ⟨nextInput, proof⟩ = some next ↔
      ApplicationCore parameters prior next nextInput proof := by
  cases priorProof : prior.proof with
  | initial =>
      cases foldProof : proof.fold with
      | noFold =>
          simp [application, ApplicationCore, priorProof, foldProof,
            Nightstream.Implementation.Rust.FPrime.checkInitial_eq_true_iff,
            postChecks_eq_true_iff, and_assoc]
      | recursive nifsProof =>
          simp [application, ApplicationCore, priorProof, foldProof]
  | active running latest =>
      cases foldProof : proof.fold with
      | noFold =>
          simp [application, ApplicationCore, priorProof, foldProof]
      | recursive nifsProof =>
          cases nifsResult :
              parameters.step.nifsVerify
                (Step.nifsContext parameters.step prior nextInput)
                running latest nifsProof with
          | none =>
              simp [application, ApplicationCore, priorProof, foldProof,
                nifsResult]
          | some nextRunning =>
              simp [ApplicationCore, priorProof, foldProof, nifsResult,
                application,
                Nightstream.Implementation.Rust.FPrime.checkActive_eq_true_iff,
                postChecks_eq_true_iff, and_assoc]

theorem paperHash_eq_some_iff
    [DecidableEq DirectState]
    [DecidableEq Running]
    (parameters : AdapterParameters)
    (preimage : HashPreimage PaperKey PaperState Running 1)
    (digest : Digest) :
    paperHash parameters preimage = some digest ↔
      ∃ current,
        preimage.current = some current ∧
        alignedHashPreimage parameters preimage = true ∧
        digest =
          XOut.compute parameters.hash parameters.mode parameters.context
            current := by
  cases currentEq : preimage.current with
  | none =>
      simp [paperHash, currentEq]
  | some current =>
      cases alignedEq : alignedHashPreimage parameters preimage with
      | false =>
          simp [paperHash, currentEq, alignedEq]
      | true =>
          simp only [paperHash, currentEq, alignedEq, ↓reduceIte,
            Option.some.injEq, exists_eq_left', true_and]
          constructor <;> intro equality <;> exact equality.symm

theorem priorHash_eq_computed
    [DecidableEq DirectState]
    [DecidableEq Running]
    (parameters : AdapterParameters)
    (prior : DirectState)
    (nextInput : DirectInput)
    (proof : DirectProof)
    (pc : prior.pc = 1) :
    paperHash parameters
        (Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.priorHashPreimage
          (setup parameters) (input parameters prior nextInput proof)) =
      some
        (XOut.compute parameters.hash parameters.mode parameters.context
          prior) := by
  simp [paperHash, alignedHashPreimage,
    Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.priorHashPreimage,
    setup, input, pc]

theorem nextHash_eq_computed
    [DecidableEq DirectState]
    [DecidableEq Running]
    (parameters : AdapterParameters)
    (prior next : DirectState)
    (nextInput : DirectInput)
    (proof : DirectProof)
    (chunkCount : next.chunkCount = prior.chunkCount + 1)
    (pc : next.pc = 1) :
    paperHash parameters
        (nextHashPreimage
          (setup parameters)
          ((input parameters prior nextInput proof).toGeneric
            (Key := PaperKey))
          (output parameters next proof)) =
      some
        (XOut.compute parameters.hash parameters.mode parameters.context
          next) := by
  simp [paperHash, alignedHashPreimage, nextHashPreimage, setup, input,
    output,
    Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.Input.toGeneric,
    chunkCount, pc]

theorem freshPublic_eq_encode_prior_iff
    [DecidableEq DirectState]
    [DecidableEq Running]
    (parameters : AdapterParameters)
    (prior : DirectState)
    (nextInput : DirectInput)
    (proof : DirectProof)
    (pc : prior.pc = 1) :
    freshPublic parameters (freshInput parameters prior nextInput) =
        encodeInstance
          (paperHash parameters
            (Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.priorHashPreimage
              (setup parameters) (input parameters prior nextInput proof))) ↔
      latestOf prior ≠ [] ∧
      Step.FreshLinked parameters.step.freshLink
        (XOut.compute parameters.hash parameters.mode parameters.context prior)
        (latestOf prior) := by
  rw [priorHash_eq_computed parameters prior nextInput proof pc]
  simp [freshPublic, freshInput, encodeInstance, Step.FreshLinked]

theorem active_freshPublic_eq_encode_prior_iff
    [DecidableEq DirectState]
    [DecidableEq Running]
    (parameters : AdapterParameters)
    (prior : DirectState)
    (nextInput : DirectInput)
    (proof : DirectProof)
    (running : Running)
    (latest : List Fresh)
    (active : prior.proof = .active running latest)
    (pc : prior.pc = 1) :
    freshPublic parameters (freshInput parameters prior nextInput) =
        encodeInstance
          (paperHash parameters
            (Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.priorHashPreimage
              (setup parameters) (input parameters prior nextInput proof))) ↔
      latest ≠ [] ∧
      Step.FreshLinked parameters.step.freshLink
        (XOut.compute parameters.hash parameters.mode parameters.context prior)
        latest := by
  simpa [latestOf, active] using
    freshPublic_eq_encode_prior_iff parameters prior nextInput proof pc

@[simp] theorem runningOf_advancedState
    (parameters : AdapterParameters)
    (prior : DirectState)
    (nextRunning : Running)
    (nextInput : DirectInput)
    (proof : DirectProof) :
    runningOf parameters
        (Step.advancedState parameters.step prior nextRunning nextInput proof) =
      nextRunning :=
  rfl

@[simp] theorem latestOf_advancedState
    (parameters : AdapterParameters)
    (prior : DirectState)
    (nextRunning : Running)
    (nextInput : DirectInput)
    (proof : DirectProof) :
    latestOf
        (Step.advancedState parameters.step prior nextRunning nextInput proof) =
      nextInput.nextLatest :=
  rfl

/-- Branch obligations supplied outside `application`: the recursive prior
batch is nonempty and linked to the exact prior XOut. -/
def PriorBranch
    (parameters : AdapterParameters)
    (prior : DirectState)
    (proof : DirectProof) : Prop :=
  match prior.proof, proof.fold with
  | .initial, .noFold => True
  | .active _ latest, .recursive _ =>
      latest ≠ [] ∧
      Step.FreshLinked parameters.step.freshLink
        (XOut.compute parameters.hash parameters.mode parameters.context prior)
        latest
  | _, _ => False

theorem holds_iff_applicationCore_priorBranch_xOut
    (parameters : AdapterParameters)
    (prior next : DirectState)
    (nextInput : DirectInput)
    (proof : DirectProof) :
    Step.Holds parameters.hash parameters.step parameters.mode
        parameters.context prior next nextInput proof ↔
      ApplicationCore parameters prior next nextInput proof ∧
      PriorBranch parameters prior proof ∧
      proof.xOut =
        XOut.compute parameters.hash parameters.mode parameters.context next := by
  cases priorProof : prior.proof with
  | initial =>
      cases foldProof : proof.fold with
      | noFold =>
          constructor
          · intro holds
            simp only [Step.Holds, priorProof, foldProof] at holds
            rcases holds with
              ⟨initial, _, nextNonempty, semantic, nebula, nextState, xOut,
                outgoing⟩
            refine ⟨?_, ?_, xOut⟩
            · simp only [ApplicationCore, priorProof, foldProof]
              exact ⟨initial, nextNonempty, semantic, nebula, outgoing,
                nextState.symm⟩
            · simp [PriorBranch, priorProof, foldProof]
          · rintro ⟨applicationCore, _, xOut⟩
            simp only [ApplicationCore, priorProof, foldProof] at applicationCore
            rcases applicationCore with
              ⟨initial, nextNonempty, semantic, nebula, outgoing, nextState⟩
            simp only [Step.Holds, priorProof, foldProof, Step.BaseHolds]
            exact ⟨initial, True.intro, nextNonempty, semantic, nebula,
              nextState.symm, xOut, outgoing⟩
      | recursive nifsProof =>
          simp [Step.Holds, ApplicationCore, PriorBranch, priorProof,
            foldProof]
  | active running latest =>
      cases foldProof : proof.fold with
      | noFold =>
          simp [Step.Holds, ApplicationCore, PriorBranch, priorProof,
            foldProof]
      | recursive nifsProof =>
          cases nifsResult :
              parameters.step.nifsVerify
                (Step.nifsContext parameters.step prior nextInput)
                running latest nifsProof with
          | none =>
              simp [Step.Holds, Step.RecursiveHolds, ApplicationCore,
                PriorBranch, priorProof, foldProof, nifsResult]
          | some nextRunning =>
              constructor
              · intro holds
                simp only [Step.Holds, priorProof, foldProof,
                  Step.RecursiveHolds, nifsResult] at holds
                rcases holds with
                  ⟨active, _, latestNonempty, priorLinked, nextNonempty,
                    semantic, nebula, nextState, xOut, outgoing⟩
                refine ⟨?_, ?_, xOut⟩
                · simp only [ApplicationCore, priorProof, foldProof,
                    nifsResult]
                  exact ⟨active, nextNonempty, semantic, nebula, outgoing,
                    nextState.symm⟩
                · simpa [PriorBranch, priorProof, foldProof] using
                    And.intro latestNonempty priorLinked
              · rintro ⟨applicationCore, priorBranch, xOut⟩
                simp only [ApplicationCore, priorProof, foldProof,
                  nifsResult] at applicationCore
                rcases applicationCore with
                  ⟨active, nextNonempty, semantic, nebula, outgoing,
                    nextState⟩
                have prior :
                    latest ≠ [] ∧
                    Step.FreshLinked parameters.step.freshLink
                      (XOut.compute parameters.hash parameters.mode
                        parameters.context prior)
                      latest := by
                  simpa [PriorBranch, priorProof, foldProof] using priorBranch
                simp only [Step.Holds, priorProof, foldProof,
                  Step.RecursiveHolds, nifsResult]
                exact ⟨active, True.intro, prior.1, prior.2, nextNonempty,
                  semantic, nebula, nextState.symm, xOut, outgoing⟩

theorem some_eq_application_iff
    [DecidableEq Digest]
    [DecidableEq Running]
    [DecidableEq Fresh]
    [DecidableEq Nebula]
    [DecidableEq NebulaOpen]
    (parameters : AdapterParameters)
    (prior next : DirectState)
    (nextInput : DirectInput)
    (proof : DirectProof) :
    some next = application parameters (some prior) ⟨nextInput, proof⟩ ↔
      ApplicationCore parameters prior next nextInput proof := by
  rw [eq_comm]
  exact application_eq_some_iff parameters prior next nextInput proof

theorem applicationCore_coordinates
    (parameters : AdapterParameters)
    (prior next : DirectState)
    (nextInput : DirectInput)
    (proof : DirectProof)
    (applicationCore :
      ApplicationCore parameters prior next nextInput proof) :
    prior.pc = 1 ∧
      next.chunkCount = prior.chunkCount + 1 ∧
      next.pc = 1 := by
  cases priorProof : prior.proof with
  | initial =>
      cases foldProof : proof.fold with
      | noFold =>
          simp only [ApplicationCore, priorProof, foldProof] at applicationCore
          rcases applicationCore with
            ⟨initial, _, _, _, _, nextState⟩
          have priorPc : prior.pc = 1 := initial.1
          subst next
          simp [Step.advancedState, priorPc]
      | recursive nifsProof =>
          simp [ApplicationCore, priorProof, foldProof] at applicationCore
  | active running latest =>
      cases foldProof : proof.fold with
      | noFold =>
          simp [ApplicationCore, priorProof, foldProof] at applicationCore
      | recursive nifsProof =>
          cases nifsResult :
              parameters.step.nifsVerify
                (Step.nifsContext parameters.step prior nextInput)
                running latest nifsProof with
          | none =>
              simp [ApplicationCore, priorProof, foldProof, nifsResult] at applicationCore
          | some nextRunning =>
              simp only [ApplicationCore, priorProof, foldProof, nifsResult] at applicationCore
              rcases applicationCore with
                ⟨active, _, _, _, _, nextState⟩
              have priorPc : prior.pc = 1 := active.1
              subst next
              simp [Step.advancedState, priorPc]

theorem mapped_outputHash_iff_of_applicationCore
    [DecidableEq DirectState]
    [DecidableEq Running]
    [DecidableEq Digest]
    [DecidableEq Fresh]
    [DecidableEq Nebula]
    [DecidableEq NebulaOpen]
    (parameters : AdapterParameters)
    (prior next : DirectState)
    (nextInput : DirectInput)
    (proof : DirectProof)
    (applicationCore :
      ApplicationCore parameters prior next nextInput proof) :
    (output parameters next proof).x =
        (machine parameters).hash
          (nextHashPreimage
            (setup parameters)
            ((input parameters prior nextInput proof).toGeneric
              (Key := PaperKey))
            (output parameters next proof)) ↔
      proof.xOut =
        XOut.compute parameters.hash parameters.mode parameters.context next := by
  have coordinates :=
    applicationCore_coordinates parameters prior next nextInput proof
      applicationCore
  simp only [output, machine]
  change some proof.xOut = paperHash parameters _ ↔ _
  have hashEq :
      paperHash parameters
          (nextHashPreimage
            (setup parameters)
            ((input parameters prior nextInput proof).toGeneric
              (Key := PaperKey))
            {
              zNext := some next
              runningNext := fun _ => runningOf parameters next
              pcNext :=
                Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected
              x := some proof.xOut
            }) =
        some
          (XOut.compute parameters.hash parameters.mode parameters.context
            next) := by
    simpa [output] using
      nextHash_eq_computed parameters prior next nextInput proof
        coordinates.2.1 coordinates.2.2
  rw [hashEq]
  simp

/-- The exact mapped fixed-one paper transition is the closed direct-F'
relation, including the delayed outgoing fresh link. -/
theorem transition_iff_holds
    [DecidableEq Digest]
    [DecidableEq Running]
    [DecidableEq Fresh]
    [DecidableEq NifsProof]
    [DecidableEq Nebula]
    [DecidableEq NebulaOpen]
    (parameters : AdapterParameters)
    (prior next : DirectState)
    (nextInput : DirectInput)
    (proof : DirectProof) :
    Transition
        (setup parameters)
        (machine parameters)
        Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected
        ((input parameters prior nextInput proof).toGeneric
          (Key := PaperKey))
        (output parameters next proof) ↔
      Step.Holds parameters.hash parameters.step parameters.mode
        parameters.context prior next nextInput proof := by
  rw [holds_iff_applicationCore_priorBranch_xOut]
  constructor
  · intro transition
    rcases transition with
      ⟨_, _, applicationEquation, outputHash, paperBranch⟩
    have applicationCore :
        ApplicationCore parameters prior next nextInput proof := by
      apply
        (some_eq_application_iff parameters prior next nextInput proof).1
      simpa [machine, input, output,
        Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.Input.toGeneric]
        using applicationEquation
    have xOut :
        proof.xOut =
          XOut.compute parameters.hash parameters.mode parameters.context
            next :=
      (mapped_outputHash_iff_of_applicationCore parameters prior next
        nextInput proof applicationCore).1 outputHash
    refine ⟨applicationCore, ?_, xOut⟩
    cases priorProof : prior.proof with
    | initial =>
        cases foldProof : proof.fold with
        | noFold =>
            simp [PriorBranch, priorProof, foldProof]
        | recursive nifsProof =>
            simp [ApplicationCore, priorProof, foldProof] at applicationCore
    | active running latest =>
        cases foldProof : proof.fold with
        | noFold =>
            simp [ApplicationCore, priorProof, foldProof] at applicationCore
        | recursive nifsProof =>
            cases nifsResult :
                parameters.step.nifsVerify
                  (Step.nifsContext parameters.step prior nextInput)
                  running latest nifsProof with
            | none =>
                simp [ApplicationCore, priorProof, foldProof, nifsResult] at applicationCore
            | some nextRunning =>
                simp only [ApplicationCore, priorProof, foldProof,
                  nifsResult] at applicationCore
                have active := applicationCore.1
                rcases paperBranch with base | recursive
                · have chunkZero : prior.chunkCount = 0 := by
                    simpa [input,
                      Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.Input.toGeneric]
                      using base.1
                  exact False.elim (active.2.1 chunkZero)
                · rcases recursive with
                    ⟨_, _, priorPublic, _, _⟩
                  have priorPublic' :
                      freshPublic parameters
                          (freshInput parameters prior nextInput) =
                        encodeInstance
                          (paperHash parameters
                            (Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.priorHashPreimage
                              (setup parameters)
                              (input parameters prior nextInput proof))) := by
                    simpa [machine, input,
                      Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.Input.toGeneric]
                      using priorPublic
                  have linked :=
                    (active_freshPublic_eq_encode_prior_iff parameters prior
                      nextInput proof running latest priorProof active.1).1
                      priorPublic'
                  simpa [PriorBranch, priorProof, foldProof] using linked
  · rintro ⟨applicationCore, priorBranch, xOut⟩
    unfold Nightstream.HyperNova.Construction2.Paper.Transition
    refine ⟨?_, ?_, ?_, ?_, ?_⟩
    · simp [machine]
    · simp [output]
    · have applicationEquation :=
        (some_eq_application_iff parameters prior next nextInput proof).2
          applicationCore
      simpa [machine, input, output,
        Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.Input.toGeneric]
        using applicationEquation
    · exact
        (mapped_outputHash_iff_of_applicationCore parameters prior next
          nextInput proof applicationCore).2 xOut
    · cases priorProof : prior.proof with
      | initial =>
          cases foldProof : proof.fold with
          | noFold =>
              simp only [ApplicationCore, priorProof, foldProof] at applicationCore
              rcases applicationCore with
                ⟨initial, _, _, _, _, nextState⟩
              left
              refine ⟨?_, ?_, ?_⟩
              · simpa [input,
                  Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.Input.toGeneric]
                  using initial.2.1
              · have priorEq :
                    prior = initialState parameters :=
                  (initialState_eq_iff parameters prior).2 initial
                simp [input,
                  Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.Input.toGeneric,
                  priorEq]
              · funext slot
                rw [← nextState]
                simp [output, setup]
          | recursive nifsProof =>
              simp [ApplicationCore, priorProof, foldProof] at applicationCore
      | active running latest =>
          cases foldProof : proof.fold with
          | noFold =>
              simp [ApplicationCore, priorProof, foldProof] at applicationCore
          | recursive nifsProof =>
              cases nifsResult :
                  parameters.step.nifsVerify
                    (Step.nifsContext parameters.step prior nextInput)
                    running latest nifsProof with
              | none =>
                  simp [ApplicationCore, priorProof, foldProof, nifsResult] at applicationCore
              | some nextRunning =>
                  simp only [ApplicationCore, priorProof, foldProof,
                    nifsResult] at applicationCore
                  rcases applicationCore with
                    ⟨active, _, _, _, _, nextState⟩
                  have linked :
                      latest ≠ [] ∧
                      Step.FreshLinked parameters.step.freshLink
                        (XOut.compute parameters.hash parameters.mode
                          parameters.context prior)
                        latest := by
                    simpa [PriorBranch, priorProof, foldProof] using priorBranch
                  let priorPcValid :
                      InRange 1
                        ((input parameters prior nextInput proof).toGeneric
                          (Key := PaperKey)).priorPc :=
                    Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.Input.toGeneric_priorPcValid
                      (Key := PaperKey)
                      (input parameters prior nextInput proof)
                  right
                  refine ⟨priorPcValid, ?_, ?_, ?_, ?_⟩
                  · have positive : 0 < prior.chunkCount :=
                      Nat.pos_of_ne_zero active.2.1
                    simpa [input,
                      Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.Input.toGeneric]
                      using positive
                  · have priorPublic :=
                      (active_freshPublic_eq_encode_prior_iff parameters prior
                        nextInput proof running latest priorProof active.1).2
                        linked
                    simpa [machine, input,
                      Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.Input.toGeneric]
                      using priorPublic
                  · unfold Accepts
                    rw [
                      Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selectedIndex_eq_selected
                    ]
                    have nextRunningEq :
                        runningOf parameters next = nextRunning := by
                      rw [← nextState]
                      rfl
                    have priorRunningEq :
                        runningOf parameters prior = running := by
                      simp [runningOf, priorProof]
                    simpa [setup, nifsVerifier, input, output, freshInput,
                      latestOf,
                      Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.Input.toGeneric,
                      priorProof, foldProof, priorRunningEq, nextRunningEq]
                      using nifsResult
                  · intro slot different
                    have same :
                        slot =
                          selectedIndex priorPcValid := by
                      rw [
                        Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selectedIndex_eq_selected
                      ]
                      exact
                        Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.fin_eq_selected
                          slot
                    exact False.elim (different same)

theorem canonicalAccepts_iff_holds
    [DecidableEq Digest]
    [DecidableEq Running]
    [DecidableEq Fresh]
    [DecidableEq NifsProof]
    [DecidableEq Nebula]
    [DecidableEq NebulaOpen]
    (parameters : AdapterParameters)
    (prior next : DirectState)
    (nextInput : DirectInput)
    (proof : DirectProof) :
    Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.Accepts
        (setup parameters)
        (machine parameters)
        (input parameters prior nextInput proof)
        (output parameters next proof) ↔
      Step.Holds parameters.hash parameters.step parameters.mode
        parameters.context prior next nextInput proof := by
  rw [
    Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.accepts_iff_transition
  ]
  exact transition_iff_holds parameters prior next nextInput proof

end

/-- Native receipt semantics instantiated as one exact fixed-one paper
machine. Primitive results remain the receipt's typed operations; lifecycle
calls come from the separate boundary receipt. -/
def nativeParameters
    (receipt : Receipt)
    (boundary : BoundaryReceipt) :
    Parameters Params StructureDigest Header Digest Running Fresh NifsProof
      Nebula NebulaDigest NebulaOpen where
  hash := boundaryHashSemantics receipt boundary
  step := boundaryStepSemantics receipt boundary
  mode := receipt.mode
  context := receipt.context

/-- Universal source-level refinement: native producer acceptance plus every
explicit lifecycle boundary and the delayed consumer/terminal link is exactly
the frozen fixed-one executable verifier. -/
theorem nativeAccepted_with_boundaries_and_outgoing_iff_canonicalAccepts
    (receipt : Receipt)
    (boundary : BoundaryReceipt)
    (next : NativeState)
    (wellFormed : ReceiptWellFormed receipt = true) :
    (NativeAccepted receipt next ∧
      EntryAuthority
        (boundaryHashSemantics receipt boundary)
        (boundaryStepSemantics receipt boundary)
        receipt.mode receipt.context receipt.prior ∧
      IncomingPriorLinked
        (boundaryHashSemantics receipt boundary)
        (boundaryStepSemantics receipt boundary)
        receipt.mode receipt.context receipt.prior ∧
      StatefulSemanticBound
        (boundaryStepSemantics receipt boundary)
        receipt.mode receipt.prior receipt.input receipt.proof ∧
      NebulaAdvanceBound
        (boundaryStepSemantics receipt boundary)
        receipt.prior receipt.input ∧
      Step.OutgoingLinked
        (boundaryStepSemantics receipt boundary)
        receipt.input receipt.proof) ↔
      Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.Accepts
        (setup (nativeParameters receipt boundary))
        (machine (nativeParameters receipt boundary))
        (input (nativeParameters receipt boundary)
          receipt.prior receipt.input receipt.proof)
        (output (nativeParameters receipt boundary) next receipt.proof) := by
  constructor
  · rintro ⟨native, entry, incoming, stateful, nebula, outgoing⟩
    have localHolds :=
      (nativeAccepted_with_boundaries_iff_localHolds receipt boundary next
        wellFormed).1
        ⟨native, entry, incoming, stateful, nebula⟩
    have holds :
        Step.Holds
          (boundaryHashSemantics receipt boundary)
          (boundaryStepSemantics receipt boundary)
          receipt.mode receipt.context receipt.prior next receipt.input
          receipt.proof :=
      Step.closeLocal
        (boundaryHashSemantics receipt boundary)
        (boundaryStepSemantics receipt boundary)
        receipt.mode receipt.context receipt.prior next receipt.input
        receipt.proof localHolds outgoing
    exact
      (canonicalAccepts_iff_holds
        (nativeParameters receipt boundary)
        receipt.prior next receipt.input receipt.proof).2 holds
  · intro accepted
    have holds :=
      (canonicalAccepts_iff_holds
        (nativeParameters receipt boundary)
        receipt.prior next receipt.input receipt.proof).1 accepted
    have split :=
      (Step.holds_iff_local_and_outgoing
        (boundaryHashSemantics receipt boundary)
        (boundaryStepSemantics receipt boundary)
        receipt.mode receipt.context receipt.prior next receipt.input
        receipt.proof).1 holds
    have native :=
      (nativeAccepted_with_boundaries_iff_localHolds receipt boundary next
        wellFormed).2 split.1
    exact ⟨native.1, native.2.1, native.2.2.1, native.2.2.2.1,
      native.2.2.2.2, split.2⟩

/-- Receipt-level corollary.  Once exact execution/call conservation is
checked, the Rust-recorded accepted result and explicit lifecycle closures
are equivalent to frozen canonical acceptance. -/
theorem checkedRecorded_with_boundaries_and_outgoing_iff_canonicalAccepts
    (receipt : Receipt)
    (boundary : BoundaryReceipt)
    (next : NativeState)
    (checked : check receipt = true) :
    (receipt.outcome = .accepted next ∧
      EntryAuthority
        (boundaryHashSemantics receipt boundary)
        (boundaryStepSemantics receipt boundary)
        receipt.mode receipt.context receipt.prior ∧
      IncomingPriorLinked
        (boundaryHashSemantics receipt boundary)
        (boundaryStepSemantics receipt boundary)
        receipt.mode receipt.context receipt.prior ∧
      StatefulSemanticBound
        (boundaryStepSemantics receipt boundary)
        receipt.mode receipt.prior receipt.input receipt.proof ∧
      NebulaAdvanceBound
        (boundaryStepSemantics receipt boundary)
        receipt.prior receipt.input ∧
      Step.OutgoingLinked
        (boundaryStepSemantics receipt boundary)
        receipt.input receipt.proof) ↔
      Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.Accepts
        (setup (nativeParameters receipt boundary))
        (machine (nativeParameters receipt boundary))
        (input (nativeParameters receipt boundary)
          receipt.prior receipt.input receipt.proof)
        (output (nativeParameters receipt boundary) next receipt.proof) := by
  have replay :=
    (check_eq_true_iff_oracleReplayConforms receipt).1 checked
  have wellFormed : ReceiptWellFormed receipt = true := replay.1
  constructor
  · rintro ⟨recorded, entry, incoming, stateful, nebula, outgoing⟩
    have native :=
      check_and_recordedAccepted_implies_nativeAccepted receipt next checked
        recorded
    exact
      (nativeAccepted_with_boundaries_and_outgoing_iff_canonicalAccepts
        receipt boundary next wellFormed).1
        ⟨native, entry, incoming, stateful, nebula, outgoing⟩
  · intro accepted
    have native :=
      (nativeAccepted_with_boundaries_and_outgoing_iff_canonicalAccepts
        receipt boundary next wellFormed).2 accepted
    have nativeRecorded : nativeOutcome receipt = receipt.outcome :=
      replay.2.2
    have recorded : receipt.outcome = .accepted next := by
      unfold NativeAccepted at native
      unfold nativeOutcome at nativeRecorded
      rw [native.1] at nativeRecorded
      simpa using nativeRecorded.symm
    exact ⟨recorded, native.2.1, native.2.2.1, native.2.2.2.1,
      native.2.2.2.2.1, native.2.2.2.2.2⟩

end Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.FixedOneCanonicalAdapter
