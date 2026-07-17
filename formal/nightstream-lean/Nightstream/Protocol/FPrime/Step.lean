import Nightstream.Protocol.FPrime.XOut
import Nightstream.HyperNova.Construction2.Default

/-!
Contract: direct-F' base and recursive semantic steps.

The relation mirrors Rust branch selection and the augmented Construction-2
function rather than treating `advance_state` alone as correctness. It checks:

- the true initial state and empty-running base specialization;
- active-branch prior `x_out` links and recomputed running authority;
- NIFS.V over the prior `(running, latest)` pair;
- the application semantic transition in stateful mode;
- nonempty newly installed fresh instances;
- counter, boundary, accumulator, semantic, Nebula, and proof-state advance;
- recomputation of the public `x_out` output.

`LocalHolds` is the standalone native/circuit boundary. The newly installed
batch is linked to that `x_out` one invocation later, or by the terminal fold;
`Holds = LocalHolds ∧ OutgoingLinked` is the closed edge used by trace
assurance. Keeping these predicates separate prevents the circuit theorem from
claiming a constraint that the producer step does not emit.

NIFS, application, fresh-link, and Nebula checkers are executable parameters.
Their Boolean success is an explicit premise of the resulting semantic step;
none is replaced by an `accepted_implies_valid` field.

| Stage path | Mathematical obligation | Authority class | Local owner |
|---|---|---|---|
| `fprime.step.input` | carry the fold proof, fresh batch, and optional Nebula transition data | checked payload | `Proof`, `Input` |
| `fprime.step.nifs.context` | derive the exact NIFS transcript prefix from the prior state and verifier input | computed | `NifsContext`, `nifsContext` |
| `fprime.step.state.advance` | compute the candidate next Construction-2 state | computed | `installedNebula`, `advancedState` |
| `fprime.step.application` | bind fresh links, application semantics, and optional Nebula advancement | checked | `FreshLinked`, `SemanticAdvance`, `NebulaAdvance` |
| `fprime.step.base` | enforce the true initial-state, no-fold specialization | checked | `InitialState`, `BaseHolds`, `BaseLocalHolds` |
| `fprime.step.recursive` | verify the selected fold and the active-state transition | checked | `ActiveState`, `RecursiveHolds`, `RecursiveLocalHolds` |
| `fprime.step.local` | expose exactly the constraints owned by one producer invocation | checked | `LocalHolds` |
| `fprime.step.outgoing` | bind the newly installed batch at the next edge or terminal fold | checked | `OutgoingLinked`, `Holds`, `closeLocal` |
| `fprime.step.check` | execute the base/recursive and local/closed predicates without hidden acceptance premises | computed checker | `check`, `checkLocal` |
| `fprime.step.check.exact` | equate Boolean acceptance with the corresponding logical relation | derived | `checkLocal_eq_true_iff_localHolds`, `check_eq_true_iff_holds` |
| `fprime.step.soundness` | derive branch-specific semantic and state-pinning facts | derived | `fPrimeBase_sound`, `fPrimeRecursive_sound`, `next_state_pinned`, `holds_advance_facts` |

Maps to:
- HyperNova Construction 2 steps 1-5.
- `paper::f_prime::native::{prove_with_semantic_state,verify}`.
- `construction2::transition::{state_base_case_check,advance_state,compute_x_out}`.
- `paper::nifs::verifier::verify`.
-/

namespace Nightstream.Protocol.FPrime.Step

open Nightstream.HyperNova.Construction2
open Nightstream.Protocol.FPrime

universe uDigest uParams uStructure uHeader uRunning uFresh uNifsProof
  uNebulaDigest uNebulaOpen

inductive FoldProof (NifsProof : Type uNifsProof) where
  | noFold
  | recursive (proof : NifsProof)
deriving Repr, DecidableEq

/-- Public/advice object checked by one F' verifier invocation. -/
structure Proof
    (Digest : Type uDigest)
    (NifsProof : Type uNifsProof)
    (NebulaOpen : Type uNebulaOpen) where
  fold : FoldProof NifsProof
  nebulaOpen : Option NebulaOpen
  semanticStateDigest : Digest
  xOut : Digest
deriving Repr, DecidableEq

/-- Verifier-owned inputs accompanying the proof. -/
structure Input
    (Fresh : Type uFresh)
    (Nebula : Type)
    (NebulaOpen : Type uNebulaOpen) where
  nextLatest : List Fresh
  nebulaOpen : Option NebulaOpen
  /-- `none` carries the prior lane; `some` installs the checked next lane. -/
  nebulaNext : Option Nebula
deriving Repr, DecidableEq

/-- Variable portion of Rust's fresh per-step NIFS transcript prefix. -/
structure NifsContext
    (Digest : Type uDigest)
    (Nebula : Type) where
  chunkCount : Nat
  stepCount : Nat
  z0 : Digest
  zi : Digest
  initialSemanticState : Digest
  semanticState : Digest
  pc : Nat
  accumulatorDigest : Digest
  publicTrace : Digest
  nebula : Option Nebula
  nextChunkDigest : Digest
deriving Repr, DecidableEq

/-- Executable semantic components owned by the concrete F' specialization. -/
structure Semantics
    (Digest : Type uDigest)
    (Running : Type uRunning)
    (Fresh : Type uFresh)
    (NifsProof : Type uNifsProof)
    (Nebula : Type)
    (NebulaOpen : Type uNebulaOpen) where
  emptyRunning : Running
  /-- Verifier-owned base memory lane; `none` for a plain chain. -/
  initialNebula : Option Nebula
  runningDigest : Running → Digest
  chunkDigest : Nat → List Fresh → Digest
  freshLink : Digest → Fresh → Bool
  /-- `none` rejects; `some nextRunning` is the verifier-computed accumulator. -/
  nifsVerify : NifsContext Digest Nebula →
    Running → List Fresh → NifsProof → Option Running
  applicationStep : Digest → List Fresh → Digest → Bool
  nebulaVerify : Option Nebula → Option NebulaOpen → Option Nebula → Bool

def nifsContext
    {Digest : Type uDigest}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {NifsProof : Type uNifsProof}
    {Nebula : Type}
    {NebulaOpen : Type uNebulaOpen}
    (semantics : Semantics Digest Running Fresh NifsProof Nebula NebulaOpen)
    (prior : State Digest Running Fresh Nebula)
    (input : Input Fresh Nebula NebulaOpen) : NifsContext Digest Nebula where
  chunkCount := prior.chunkCount
  stepCount := prior.stepCount
  z0 := prior.z0
  zi := prior.zi
  initialSemanticState := prior.initialSemanticState
  semanticState := prior.semanticState
  pc := prior.pc
  accumulatorDigest := prior.accumulatorDigest
  publicTrace := prior.publicTrace
  nebula := prior.nebula
  nextChunkDigest := semantics.chunkDigest prior.stepCount input.nextLatest

def installedNebula
    {Fresh : Type uFresh}
    {Nebula : Type}
    {NebulaOpen : Type uNebulaOpen}
    {Digest : Type uDigest}
    {Running : Type uRunning}
    (prior : State Digest Running Fresh Nebula)
    (input : Input Fresh Nebula NebulaOpen) : Option Nebula :=
  input.nebulaNext.or prior.nebula

/-- The deterministic state written by Rust `advance_state`. -/
def advancedState
    {Digest : Type uDigest}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {NifsProof : Type uNifsProof}
    {Nebula : Type}
    {NebulaOpen : Type uNebulaOpen}
    (semantics : Semantics Digest Running Fresh NifsProof Nebula NebulaOpen)
    (prior : State Digest Running Fresh Nebula)
    (nextRunning : Running)
    (input : Input Fresh Nebula NebulaOpen)
    (proof : Proof Digest NifsProof NebulaOpen) :
    State Digest Running Fresh Nebula where
  chunkCount := prior.chunkCount + 1
  stepCount := prior.stepCount + input.nextLatest.length
  z0 := prior.z0
  zi := semantics.chunkDigest prior.stepCount input.nextLatest
  initialSemanticState := prior.initialSemanticState
  semanticState := proof.semanticStateDigest
  pc := prior.pc
  accumulatorDigest := semantics.runningDigest nextRunning
  publicTrace := semantics.chunkDigest prior.stepCount input.nextLatest
  proof := .active nextRunning input.nextLatest
  nebula := installedNebula prior input

def FreshLinked
    {Digest : Type uDigest}
    {Fresh : Type uFresh}
    (semantics : Digest → Fresh → Bool)
    (digest : Digest)
    (fresh : List Fresh) : Prop :=
  fresh.all (semantics digest) = true

def SemanticAdvance
    {Digest : Type uDigest}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {NifsProof : Type uNifsProof}
    {Nebula : Type}
    {NebulaOpen : Type uNebulaOpen}
    (semantics : Semantics Digest Running Fresh NifsProof Nebula NebulaOpen)
    (mode : XOut.Mode)
    (prior : State Digest Running Fresh Nebula)
    (nextRunning : Running)
    (input : Input Fresh Nebula NebulaOpen)
    (proof : Proof Digest NifsProof NebulaOpen) : Prop :=
  match mode with
  | .stateless =>
      proof.semanticStateDigest = semantics.runningDigest nextRunning
  | .stateful =>
      semantics.applicationStep prior.semanticState input.nextLatest
        proof.semanticStateDigest = true

def NebulaAdvance
    {Digest : Type uDigest}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {NifsProof : Type uNifsProof}
    {Nebula : Type}
    {NebulaOpen : Type uNebulaOpen}
    (semantics : Semantics Digest Running Fresh NifsProof Nebula NebulaOpen)
    (prior : State Digest Running Fresh Nebula)
    (input : Input Fresh Nebula NebulaOpen)
    (proof : Proof Digest NifsProof NebulaOpen) : Prop :=
  proof.nebulaOpen = input.nebulaOpen ∧
  semantics.nebulaVerify prior.nebula input.nebulaOpen
    (installedNebula prior input) = true

/-- True Rust base state, including empty accumulator authority. -/
def InitialState
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
    (hashSemantics : XOut.Semantics
      Params StructureDigest Header Digest Nebula NebulaDigest)
    (stepSemantics : Semantics Digest Running Fresh NifsProof Nebula NebulaOpen)
    (mode : XOut.Mode)
    (context : XOut.Context Params StructureDigest Header Digest)
    (state : State Digest Running Fresh Nebula) : Prop :=
  state.pc = 1 ∧
  state.chunkCount = 0 ∧
  state.stepCount = 0 ∧
  state.z0 = state.zi ∧
  state.z0 = XOut.initialBoundary hashSemantics context ∧
  state.publicTrace = XOut.publicTraceSeed hashSemantics context ∧
  state.initialSemanticState = context.initialSemanticState ∧
  state.accumulatorDigest = stepSemantics.runningDigest stepSemantics.emptyRunning ∧
  state.nebula = stepSemantics.initialNebula ∧
  state.proof = .initial ∧
  match mode with
  | .stateless => state.semanticState = state.accumulatorDigest
  | .stateful => state.semanticState = state.initialSemanticState

/-- Active states must be pinned and bind the carried running accumulator. -/
def ActiveState
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
    (hashSemantics : XOut.Semantics
      Params StructureDigest Header Digest Nebula NebulaDigest)
    (stepSemantics : Semantics Digest Running Fresh NifsProof Nebula NebulaOpen)
    (mode : XOut.Mode)
    (context : XOut.Context Params StructureDigest Header Digest)
    (state : State Digest Running Fresh Nebula)
    (running : Running)
    (latest : List Fresh) : Prop :=
  state.pc = 1 ∧
  state.chunkCount ≠ 0 ∧
  state.stepCount ≠ 0 ∧
  state.proof = .active running latest ∧
  state.accumulatorDigest = stepSemantics.runningDigest running ∧
  XOut.StatePinned hashSemantics mode context state

def BaseHolds
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
    (hashSemantics : XOut.Semantics
      Params StructureDigest Header Digest Nebula NebulaDigest)
    (stepSemantics : Semantics Digest Running Fresh NifsProof Nebula NebulaOpen)
    (mode : XOut.Mode)
    (context : XOut.Context Params StructureDigest Header Digest)
    (prior next : State Digest Running Fresh Nebula)
    (input : Input Fresh Nebula NebulaOpen)
    (proof : Proof Digest NifsProof NebulaOpen) : Prop :=
  InitialState hashSemantics stepSemantics mode context prior ∧
  proof.fold = .noFold ∧
  input.nextLatest ≠ [] ∧
  SemanticAdvance stepSemantics mode prior stepSemantics.emptyRunning input proof ∧
  NebulaAdvance stepSemantics prior input proof ∧
  next = advancedState stepSemantics prior stepSemantics.emptyRunning input proof ∧
  proof.xOut = XOut.compute hashSemantics mode context next ∧
  FreshLinked stepSemantics.freshLink proof.xOut input.nextLatest

def RecursiveHolds
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
    (hashSemantics : XOut.Semantics
      Params StructureDigest Header Digest Nebula NebulaDigest)
    (stepSemantics : Semantics Digest Running Fresh NifsProof Nebula NebulaOpen)
    (mode : XOut.Mode)
    (context : XOut.Context Params StructureDigest Header Digest)
    (prior next : State Digest Running Fresh Nebula)
    (input : Input Fresh Nebula NebulaOpen)
    (proof : Proof Digest NifsProof NebulaOpen)
    (running : Running)
    (latest : List Fresh)
    (nifsProof : NifsProof) : Prop :=
  ActiveState hashSemantics stepSemantics mode context prior running latest ∧
  proof.fold = .recursive nifsProof ∧
  latest ≠ [] ∧
  FreshLinked stepSemantics.freshLink
    (XOut.compute hashSemantics mode context prior) latest ∧
  match stepSemantics.nifsVerify (nifsContext stepSemantics prior input)
      running latest nifsProof with
  | none => False
  | some nextRunning =>
      input.nextLatest ≠ [] ∧
      SemanticAdvance stepSemantics mode prior nextRunning input proof ∧
      NebulaAdvance stepSemantics prior input proof ∧
      next = advancedState stepSemantics prior nextRunning input proof ∧
      proof.xOut = XOut.compute hashSemantics mode context next ∧
      FreshLinked stepSemantics.freshLink proof.xOut input.nextLatest

/-- What one standalone native/R1CS F' invocation establishes immediately.

The newly installed batch is committed by its count and chunk digest, but its
`fresh.public = enc_inst(x_out)` relation is checked one invocation later (or
by the terminal fold for the trailing batch). That delayed link is therefore
not part of this local relation. -/
def BaseLocalHolds
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
    (hashSemantics : XOut.Semantics
      Params StructureDigest Header Digest Nebula NebulaDigest)
    (stepSemantics : Semantics Digest Running Fresh NifsProof Nebula NebulaOpen)
    (mode : XOut.Mode)
    (context : XOut.Context Params StructureDigest Header Digest)
    (prior next : State Digest Running Fresh Nebula)
    (input : Input Fresh Nebula NebulaOpen)
    (proof : Proof Digest NifsProof NebulaOpen) : Prop :=
  InitialState hashSemantics stepSemantics mode context prior ∧
  proof.fold = .noFold ∧
  input.nextLatest ≠ [] ∧
  SemanticAdvance stepSemantics mode prior stepSemantics.emptyRunning input proof ∧
  NebulaAdvance stepSemantics prior input proof ∧
  next = advancedState stepSemantics prior stepSemantics.emptyRunning input proof ∧
  proof.xOut = XOut.compute hashSemantics mode context next

def RecursiveLocalHolds
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
    (hashSemantics : XOut.Semantics
      Params StructureDigest Header Digest Nebula NebulaDigest)
    (stepSemantics : Semantics Digest Running Fresh NifsProof Nebula NebulaOpen)
    (mode : XOut.Mode)
    (context : XOut.Context Params StructureDigest Header Digest)
    (prior next : State Digest Running Fresh Nebula)
    (input : Input Fresh Nebula NebulaOpen)
    (proof : Proof Digest NifsProof NebulaOpen)
    (running : Running)
    (latest : List Fresh)
    (nifsProof : NifsProof) : Prop :=
  ActiveState hashSemantics stepSemantics mode context prior running latest ∧
  proof.fold = .recursive nifsProof ∧
  latest ≠ [] ∧
  FreshLinked stepSemantics.freshLink
    (XOut.compute hashSemantics mode context prior) latest ∧
  match stepSemantics.nifsVerify (nifsContext stepSemantics prior input)
      running latest nifsProof with
  | none => False
  | some nextRunning =>
      input.nextLatest ≠ [] ∧
      SemanticAdvance stepSemantics mode prior nextRunning input proof ∧
      NebulaAdvance stepSemantics prior input proof ∧
      next = advancedState stepSemantics prior nextRunning input proof ∧
      proof.xOut = XOut.compute hashSemantics mode context next

/-- The one-step-delayed recursive-link obligation. -/
def OutgoingLinked
    {Digest : Type uDigest}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {NifsProof : Type uNifsProof}
    {Nebula : Type}
    {NebulaOpen : Type uNebulaOpen}
    (stepSemantics : Semantics Digest Running Fresh NifsProof Nebula NebulaOpen)
    (input : Input Fresh Nebula NebulaOpen)
    (proof : Proof Digest NifsProof NebulaOpen) : Prop :=
  FreshLinked stepSemantics.freshLink proof.xOut input.nextLatest

/-- Exact branch dispatch for the obligations owned by a standalone step. -/
def LocalHolds
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
    (hashSemantics : XOut.Semantics
      Params StructureDigest Header Digest Nebula NebulaDigest)
    (stepSemantics : Semantics Digest Running Fresh NifsProof Nebula NebulaOpen)
    (mode : XOut.Mode)
    (context : XOut.Context Params StructureDigest Header Digest)
    (prior next : State Digest Running Fresh Nebula)
    (input : Input Fresh Nebula NebulaOpen)
    (proof : Proof Digest NifsProof NebulaOpen) : Prop :=
  match prior.proof, proof.fold with
  | .initial, .noFold =>
      BaseLocalHolds hashSemantics stepSemantics mode context prior next input proof
  | .active running latest, .recursive nifsProof =>
      RecursiveLocalHolds hashSemantics stepSemantics mode context prior next input proof
        running latest nifsProof
  | _, _ => False

/-- Exact base/recursive variant dispatch. -/
def Holds
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
    (hashSemantics : XOut.Semantics
      Params StructureDigest Header Digest Nebula NebulaDigest)
    (stepSemantics : Semantics Digest Running Fresh NifsProof Nebula NebulaOpen)
    (mode : XOut.Mode)
    (context : XOut.Context Params StructureDigest Header Digest)
    (prior next : State Digest Running Fresh Nebula)
    (input : Input Fresh Nebula NebulaOpen)
    (proof : Proof Digest NifsProof NebulaOpen) : Prop :=
  match prior.proof, proof.fold with
  | .initial, .noFold =>
      BaseHolds hashSemantics stepSemantics mode context prior next input proof
  | .active running latest, .recursive nifsProof =>
      RecursiveHolds hashSemantics stepSemantics mode context prior next input proof
        running latest nifsProof
  | _, _ => False

theorem holds_iff_local_and_outgoing
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
    (hashSemantics : XOut.Semantics
      Params StructureDigest Header Digest Nebula NebulaDigest)
    (stepSemantics : Semantics Digest Running Fresh NifsProof Nebula NebulaOpen)
    (mode : XOut.Mode)
    (context : XOut.Context Params StructureDigest Header Digest)
    (prior next : State Digest Running Fresh Nebula)
    (input : Input Fresh Nebula NebulaOpen)
    (proof : Proof Digest NifsProof NebulaOpen) :
    Holds hashSemantics stepSemantics mode context prior next input proof ↔
      LocalHolds hashSemantics stepSemantics mode context prior next input proof ∧
      OutgoingLinked stepSemantics input proof := by
  cases priorProof : prior.proof with
  | initial =>
      cases foldProof : proof.fold with
      | noFold =>
          simp only [Holds, LocalHolds, priorProof, foldProof, BaseHolds,
            BaseLocalHolds, OutgoingLinked]
          simp only [and_assoc]
      | recursive nifsProof =>
          simp [Holds, LocalHolds, priorProof, foldProof]
  | active running latest =>
      cases foldProof : proof.fold with
      | noFold =>
          simp [Holds, LocalHolds, priorProof, foldProof]
      | recursive nifsProof =>
          simp only [Holds, LocalHolds, priorProof, foldProof, RecursiveHolds,
            RecursiveLocalHolds, OutgoingLinked]
          cases stepSemantics.nifsVerify (nifsContext stepSemantics prior input)
              running latest nifsProof <;> simp only [and_assoc, false_and, and_false]

/-- Public transition facts common to the base and recursive branches. -/
structure AdvanceFacts
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
    (hashSemantics : XOut.Semantics
      Params StructureDigest Header Digest Nebula NebulaDigest)
    (stepSemantics : Semantics Digest Running Fresh NifsProof Nebula NebulaOpen)
    (mode : XOut.Mode)
    (context : XOut.Context Params StructureDigest Header Digest)
    (prior next : State Digest Running Fresh Nebula)
    (input : Input Fresh Nebula NebulaOpen)
    (proof : Proof Digest NifsProof NebulaOpen) : Prop where
  freshNonempty : input.nextLatest ≠ []
  chunkCount : next.chunkCount = prior.chunkCount + 1
  stepCount : next.stepCount = prior.stepCount + input.nextLatest.length
  initialBoundary : next.z0 = prior.z0
  initialSemanticState :
    next.initialSemanticState = prior.initialSemanticState
  programCounter : next.pc = prior.pc
  publicTrace : next.publicTrace = next.zi
  installedAccumulator : ∃ running,
    next.proof = .active running input.nextLatest ∧
    next.accumulatorDigest = stepSemantics.runningDigest running
  installedNebula : next.nebula = Step.installedNebula prior input
  recomputedXOut : proof.xOut = XOut.compute hashSemantics mode context next
  outgoingFreshLinked :
    FreshLinked stepSemantics.freshLink proof.xOut input.nextLatest
  pinned : XOut.StatePinned hashSemantics mode context next

private instance localHoldsDecidable
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
    [DecidableEq Digest]
    [DecidableEq Running]
    [DecidableEq Fresh]
    [DecidableEq NifsProof]
    [DecidableEq Nebula]
    [DecidableEq NebulaOpen]
    (hashSemantics : XOut.Semantics
      Params StructureDigest Header Digest Nebula NebulaDigest)
    (stepSemantics : Semantics Digest Running Fresh NifsProof Nebula NebulaOpen)
    (mode : XOut.Mode)
    (context : XOut.Context Params StructureDigest Header Digest)
    (prior next : State Digest Running Fresh Nebula)
    (input : Input Fresh Nebula NebulaOpen)
    (proof : Proof Digest NifsProof NebulaOpen) :
    Decidable (LocalHolds hashSemantics stepSemantics mode context
      prior next input proof) := by
  cases priorProof : prior.proof with
  | initial =>
      cases foldProof : proof.fold with
      | noFold =>
          simp only [LocalHolds, priorProof, foldProof, BaseLocalHolds,
            InitialState, SemanticAdvance, NebulaAdvance]
          cases mode <;> infer_instance
      | recursive nifsProof =>
          simp only [LocalHolds, priorProof, foldProof]
          infer_instance
  | active running latest =>
      cases foldProof : proof.fold with
      | noFold =>
          simp only [LocalHolds, priorProof, foldProof]
          infer_instance
      | recursive nifsProof =>
          simp only [LocalHolds, priorProof, foldProof, RecursiveLocalHolds,
            ActiveState, SemanticAdvance, NebulaAdvance, FreshLinked]
          cases stepSemantics.nifsVerify (nifsContext stepSemantics prior input)
              running latest nifsProof <;>
            cases mode <;> infer_instance

private instance holdsDecidable
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
    [DecidableEq Digest]
    [DecidableEq Running]
    [DecidableEq Fresh]
    [DecidableEq NifsProof]
    [DecidableEq Nebula]
    [DecidableEq NebulaOpen]
    (hashSemantics : XOut.Semantics
      Params StructureDigest Header Digest Nebula NebulaDigest)
    (stepSemantics : Semantics Digest Running Fresh NifsProof Nebula NebulaOpen)
    (mode : XOut.Mode)
    (context : XOut.Context Params StructureDigest Header Digest)
    (prior next : State Digest Running Fresh Nebula)
    (input : Input Fresh Nebula NebulaOpen)
    (proof : Proof Digest NifsProof NebulaOpen) :
    Decidable (Holds hashSemantics stepSemantics mode context prior next input proof) := by
  cases priorProof : prior.proof with
  | initial =>
      cases foldProof : proof.fold with
      | noFold =>
          simp only [Holds, priorProof, foldProof, BaseHolds, InitialState,
            SemanticAdvance, NebulaAdvance, FreshLinked]
          cases mode <;> infer_instance
      | recursive nifsProof =>
          simp only [Holds, priorProof, foldProof]
          infer_instance
  | active running latest =>
      cases foldProof : proof.fold with
      | noFold =>
          simp only [Holds, priorProof, foldProof]
          infer_instance
      | recursive nifsProof =>
          simp only [Holds, priorProof, foldProof, RecursiveHolds, ActiveState,
            SemanticAdvance, NebulaAdvance, FreshLinked]
          cases stepSemantics.nifsVerify (nifsContext stepSemantics prior input)
              running latest nifsProof <;>
            cases mode <;> infer_instance

/-- Executable verifier for the M3 semantic step relation. -/
def check
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
    [DecidableEq Digest]
    [DecidableEq Running]
    [DecidableEq Fresh]
    [DecidableEq NifsProof]
    [DecidableEq Nebula]
    [DecidableEq NebulaOpen]
    (hashSemantics : XOut.Semantics
      Params StructureDigest Header Digest Nebula NebulaDigest)
    (stepSemantics : Semantics Digest Running Fresh NifsProof Nebula NebulaOpen)
    (mode : XOut.Mode)
    (context : XOut.Context Params StructureDigest Header Digest)
    (prior next : State Digest Running Fresh Nebula)
    (input : Input Fresh Nebula NebulaOpen)
    (proof : Proof Digest NifsProof NebulaOpen) : Bool :=
  decide (Holds hashSemantics stepSemantics mode context prior next input proof)

/-- Executable checker for the obligations owned by one standalone native or
R1CS F' step. The trailing recursive link is closed by the consumer step or
terminal fold. -/
def checkLocal
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
    [DecidableEq Digest]
    [DecidableEq Running]
    [DecidableEq Fresh]
    [DecidableEq NifsProof]
    [DecidableEq Nebula]
    [DecidableEq NebulaOpen]
    (hashSemantics : XOut.Semantics
      Params StructureDigest Header Digest Nebula NebulaDigest)
    (stepSemantics : Semantics Digest Running Fresh NifsProof Nebula NebulaOpen)
    (mode : XOut.Mode)
    (context : XOut.Context Params StructureDigest Header Digest)
    (prior next : State Digest Running Fresh Nebula)
    (input : Input Fresh Nebula NebulaOpen)
    (proof : Proof Digest NifsProof NebulaOpen) : Bool :=
  decide (LocalHolds hashSemantics stepSemantics mode context prior next input proof)

theorem check_sound
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
    [DecidableEq Digest]
    [DecidableEq Running]
    [DecidableEq Fresh]
    [DecidableEq NifsProof]
    [DecidableEq Nebula]
    [DecidableEq NebulaOpen]
    (hashSemantics : XOut.Semantics
      Params StructureDigest Header Digest Nebula NebulaDigest)
    (stepSemantics : Semantics Digest Running Fresh NifsProof Nebula NebulaOpen)
    (mode : XOut.Mode)
    (context : XOut.Context Params StructureDigest Header Digest)
    (prior next : State Digest Running Fresh Nebula)
    (input : Input Fresh Nebula NebulaOpen)
    (proof : Proof Digest NifsProof NebulaOpen)
    (accepted :
      check hashSemantics stepSemantics mode context prior next input proof = true) :
    Holds hashSemantics stepSemantics mode context prior next input proof := by
  unfold check at accepted
  exact of_decide_eq_true accepted

theorem checkLocal_sound
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
    [DecidableEq Digest]
    [DecidableEq Running]
    [DecidableEq Fresh]
    [DecidableEq NifsProof]
    [DecidableEq Nebula]
    [DecidableEq NebulaOpen]
    (hashSemantics : XOut.Semantics
      Params StructureDigest Header Digest Nebula NebulaDigest)
    (stepSemantics : Semantics Digest Running Fresh NifsProof Nebula NebulaOpen)
    (mode : XOut.Mode)
    (context : XOut.Context Params StructureDigest Header Digest)
    (prior next : State Digest Running Fresh Nebula)
    (input : Input Fresh Nebula NebulaOpen)
    (proof : Proof Digest NifsProof NebulaOpen)
    (accepted : checkLocal hashSemantics stepSemantics mode context
      prior next input proof = true) :
    LocalHolds hashSemantics stepSemantics mode context prior next input proof := by
  unfold checkLocal at accepted
  exact of_decide_eq_true accepted

theorem checkLocal_eq_true_iff_localHolds
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
    [DecidableEq Digest]
    [DecidableEq Running]
    [DecidableEq Fresh]
    [DecidableEq NifsProof]
    [DecidableEq Nebula]
    [DecidableEq NebulaOpen]
    (hashSemantics : XOut.Semantics
      Params StructureDigest Header Digest Nebula NebulaDigest)
    (stepSemantics : Semantics Digest Running Fresh NifsProof Nebula NebulaOpen)
    (mode : XOut.Mode)
    (context : XOut.Context Params StructureDigest Header Digest)
    (prior next : State Digest Running Fresh Nebula)
    (input : Input Fresh Nebula NebulaOpen)
    (proof : Proof Digest NifsProof NebulaOpen) :
    checkLocal hashSemantics stepSemantics mode context prior next input proof = true ↔
      LocalHolds hashSemantics stepSemantics mode context prior next input proof := by
  unfold checkLocal
  exact decide_eq_true_iff

/-- The executable checker is exact, not merely a one-way rejection filter. -/
theorem check_eq_true_iff_holds
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
    [DecidableEq Digest]
    [DecidableEq Running]
    [DecidableEq Fresh]
    [DecidableEq NifsProof]
    [DecidableEq Nebula]
    [DecidableEq NebulaOpen]
    (hashSemantics : XOut.Semantics
      Params StructureDigest Header Digest Nebula NebulaDigest)
    (stepSemantics : Semantics Digest Running Fresh NifsProof Nebula NebulaOpen)
    (mode : XOut.Mode)
    (context : XOut.Context Params StructureDigest Header Digest)
    (prior next : State Digest Running Fresh Nebula)
    (input : Input Fresh Nebula NebulaOpen)
    (proof : Proof Digest NifsProof NebulaOpen) :
    check hashSemantics stepSemantics mode context prior next input proof = true ↔
      Holds hashSemantics stepSemantics mode context prior next input proof := by
  unfold check
  exact decide_eq_true_iff

/-- `FPR-BASE`: checker acceptance in the initial/NoFold branch gives the full base relation. -/
theorem fPrimeBase_sound
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
    [DecidableEq Digest]
    [DecidableEq Running]
    [DecidableEq Fresh]
    [DecidableEq NifsProof]
    [DecidableEq Nebula]
    [DecidableEq NebulaOpen]
    (hashSemantics : XOut.Semantics
      Params StructureDigest Header Digest Nebula NebulaDigest)
    (stepSemantics : Semantics Digest Running Fresh NifsProof Nebula NebulaOpen)
    (mode : XOut.Mode)
    (context : XOut.Context Params StructureDigest Header Digest)
    (prior next : State Digest Running Fresh Nebula)
    (input : Input Fresh Nebula NebulaOpen)
    (proof : Proof Digest NifsProof NebulaOpen)
    (priorInitial : prior.proof = .initial)
    (proofNoFold : proof.fold = .noFold)
    (accepted :
      check hashSemantics stepSemantics mode context prior next input proof = true) :
    BaseHolds hashSemantics stepSemantics mode context prior next input proof := by
  have holds := check_sound hashSemantics stepSemantics mode context
    prior next input proof accepted
  unfold Holds at holds
  rw [priorInitial, proofNoFold] at holds
  exact holds

/-- `FPR-RECURSIVE`: accepted Active/Recursive verification exposes every obligation. -/
theorem fPrimeRecursive_sound
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
    [DecidableEq Digest]
    [DecidableEq Running]
    [DecidableEq Fresh]
    [DecidableEq NifsProof]
    [DecidableEq Nebula]
    [DecidableEq NebulaOpen]
    (hashSemantics : XOut.Semantics
      Params StructureDigest Header Digest Nebula NebulaDigest)
    (stepSemantics : Semantics Digest Running Fresh NifsProof Nebula NebulaOpen)
    (mode : XOut.Mode)
    (context : XOut.Context Params StructureDigest Header Digest)
    (prior next : State Digest Running Fresh Nebula)
    (input : Input Fresh Nebula NebulaOpen)
    (proof : Proof Digest NifsProof NebulaOpen)
    (running : Running)
    (latest : List Fresh)
    (nifsProof : NifsProof)
    (priorActive : prior.proof = .active running latest)
    (proofRecursive : proof.fold = .recursive nifsProof)
    (accepted :
      check hashSemantics stepSemantics mode context prior next input proof = true) :
    RecursiveHolds hashSemantics stepSemantics mode context prior next input proof
      running latest nifsProof := by
  have holds := check_sound hashSemantics stepSemantics mode context
    prior next input proof accepted
  unfold Holds at holds
  rw [priorActive, proofRecursive] at holds
  exact holds

/-- Standalone base-circuit acceptance establishes the local base relation;
the outgoing link is intentionally a separate, delayed obligation. -/
theorem fPrimeBaseLocal_sound
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
    [DecidableEq Digest]
    [DecidableEq Running]
    [DecidableEq Fresh]
    [DecidableEq NifsProof]
    [DecidableEq Nebula]
    [DecidableEq NebulaOpen]
    (hashSemantics : XOut.Semantics
      Params StructureDigest Header Digest Nebula NebulaDigest)
    (stepSemantics : Semantics Digest Running Fresh NifsProof Nebula NebulaOpen)
    (mode : XOut.Mode)
    (context : XOut.Context Params StructureDigest Header Digest)
    (prior next : State Digest Running Fresh Nebula)
    (input : Input Fresh Nebula NebulaOpen)
    (proof : Proof Digest NifsProof NebulaOpen)
    (priorInitial : prior.proof = .initial)
    (proofNoFold : proof.fold = .noFold)
    (accepted : checkLocal hashSemantics stepSemantics mode context
      prior next input proof = true) :
    BaseLocalHolds hashSemantics stepSemantics mode context
      prior next input proof := by
  have localProof := checkLocal_sound hashSemantics stepSemantics mode context
    prior next input proof accepted
  unfold LocalHolds at localProof
  rw [priorInitial, proofNoFold] at localProof
  exact localProof

/-- Standalone recursive-circuit acceptance establishes the local recursive
relation, including the prior link and NIFS result but not the new batch's
one-step-delayed link. -/
theorem fPrimeRecursiveLocal_sound
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
    [DecidableEq Digest]
    [DecidableEq Running]
    [DecidableEq Fresh]
    [DecidableEq NifsProof]
    [DecidableEq Nebula]
    [DecidableEq NebulaOpen]
    (hashSemantics : XOut.Semantics
      Params StructureDigest Header Digest Nebula NebulaDigest)
    (stepSemantics : Semantics Digest Running Fresh NifsProof Nebula NebulaOpen)
    (mode : XOut.Mode)
    (context : XOut.Context Params StructureDigest Header Digest)
    (prior next : State Digest Running Fresh Nebula)
    (input : Input Fresh Nebula NebulaOpen)
    (proof : Proof Digest NifsProof NebulaOpen)
    (running : Running)
    (latest : List Fresh)
    (nifsProof : NifsProof)
    (priorActive : prior.proof = .active running latest)
    (proofRecursive : proof.fold = .recursive nifsProof)
    (accepted : checkLocal hashSemantics stepSemantics mode context
      prior next input proof = true) :
    RecursiveLocalHolds hashSemantics stepSemantics mode context
      prior next input proof running latest nifsProof := by
  have localProof := checkLocal_sound hashSemantics stepSemantics mode context
    prior next input proof accepted
  unfold LocalHolds at localProof
  rw [priorActive, proofRecursive] at localProof
  exact localProof

/-- Consumer/terminal link rows close a locally valid step into the strong
edge relation used by trace assurance. -/
theorem closeLocal
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
    (hashSemantics : XOut.Semantics
      Params StructureDigest Header Digest Nebula NebulaDigest)
    (stepSemantics : Semantics Digest Running Fresh NifsProof Nebula NebulaOpen)
    (mode : XOut.Mode)
    (context : XOut.Context Params StructureDigest Header Digest)
    (prior next : State Digest Running Fresh Nebula)
    (input : Input Fresh Nebula NebulaOpen)
    (proof : Proof Digest NifsProof NebulaOpen)
    (localProof : LocalHolds hashSemantics stepSemantics mode context
      prior next input proof)
    (outgoing : OutgoingLinked stepSemantics input proof) :
    Holds hashSemantics stepSemantics mode context prior next input proof :=
  (holds_iff_local_and_outgoing hashSemantics stepSemantics mode context
    prior next input proof).2 ⟨localProof, outgoing⟩

private theorem advancedState_pinned
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
    (hashSemantics : XOut.Semantics
      Params StructureDigest Header Digest Nebula NebulaDigest)
    (stepSemantics : Semantics Digest Running Fresh NifsProof Nebula NebulaOpen)
    (mode : XOut.Mode)
    (context : XOut.Context Params StructureDigest Header Digest)
    (prior : State Digest Running Fresh Nebula)
    (nextRunning : Running)
    (input : Input Fresh Nebula NebulaOpen)
    (proof : Proof Digest NifsProof NebulaOpen)
    (initialBoundaryPinned :
      prior.z0 = XOut.initialBoundary hashSemantics context)
    (initialSemanticStatePinned :
      prior.initialSemanticState = context.initialSemanticState)
    (semanticAdvance : SemanticAdvance stepSemantics mode prior nextRunning input proof) :
    XOut.StatePinned hashSemantics mode context
      (advancedState stepSemantics prior nextRunning input proof) := by
  refine {
    initialBoundaryPinned := initialBoundaryPinned
    initialSemanticStatePinned := initialSemanticStatePinned
    publicTraceMirrorsBoundary := rfl
    statelessSemanticEqualsAccumulator := ?_
  }
  intro stateless
  cases mode
  · exact semanticAdvance
  · cases stateless

/-- Every valid F' step produces a state suitable for the next recursive link. -/
theorem next_state_pinned
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
    (hashSemantics : XOut.Semantics
      Params StructureDigest Header Digest Nebula NebulaDigest)
    (stepSemantics : Semantics Digest Running Fresh NifsProof Nebula NebulaOpen)
    (mode : XOut.Mode)
    (context : XOut.Context Params StructureDigest Header Digest)
    (prior next : State Digest Running Fresh Nebula)
    (input : Input Fresh Nebula NebulaOpen)
    (proof : Proof Digest NifsProof NebulaOpen)
    (holds : Holds hashSemantics stepSemantics mode context prior next input proof) :
    XOut.StatePinned hashSemantics mode context next := by
  cases priorProof : prior.proof with
  | initial =>
      cases foldProof : proof.fold with
      | noFold =>
          have base :
              BaseHolds hashSemantics stepSemantics mode context
                prior next input proof := by
            simpa [Holds, priorProof, foldProof] using holds
          rcases base with
            ⟨initial, _, _, semanticAdvance, _, nextEq, _, _⟩
          subst next
          exact advancedState_pinned hashSemantics stepSemantics mode context prior
            stepSemantics.emptyRunning input proof initial.2.2.2.2.1
            initial.2.2.2.2.2.2.1 semanticAdvance
      | recursive nifsProof =>
          simp [Holds, priorProof, foldProof] at holds
  | active running latest =>
      cases foldProof : proof.fold with
      | noFold =>
          simp [Holds, priorProof, foldProof] at holds
      | recursive nifsProof =>
          have recursive :
              RecursiveHolds hashSemantics stepSemantics mode context
                prior next input proof running latest nifsProof := by
            simpa [Holds, priorProof, foldProof] using holds
          rcases recursive with ⟨active, _, _, _, verified⟩
          cases verifierEq : stepSemantics.nifsVerify
              (nifsContext stepSemantics prior input) running latest nifsProof with
          | none => simp [verifierEq] at verified
          | some nextRunning =>
            simp only [verifierEq] at verified
            rcases verified with
              ⟨_, semanticAdvance, _, nextEq, _, _⟩
            subst next
            exact advancedState_pinned hashSemantics stepSemantics mode context prior
              nextRunning input proof active.2.2.2.2.2.initialBoundaryPinned
              active.2.2.2.2.2.initialSemanticStatePinned semanticAdvance

/-- A valid branch exposes the exact state advance used by trace induction. -/
theorem holds_advance_facts
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
    (hashSemantics : XOut.Semantics
      Params StructureDigest Header Digest Nebula NebulaDigest)
    (stepSemantics : Semantics Digest Running Fresh NifsProof Nebula NebulaOpen)
    (mode : XOut.Mode)
    (context : XOut.Context Params StructureDigest Header Digest)
    (prior next : State Digest Running Fresh Nebula)
    (input : Input Fresh Nebula NebulaOpen)
    (proof : Proof Digest NifsProof NebulaOpen)
    (holds : Holds hashSemantics stepSemantics mode context prior next input proof) :
    AdvanceFacts hashSemantics stepSemantics mode context prior next input proof := by
  have nextPinned := next_state_pinned hashSemantics stepSemantics mode context
    prior next input proof holds
  cases priorProof : prior.proof with
  | initial =>
      cases foldProof : proof.fold with
      | noFold =>
          have base :
              BaseHolds hashSemantics stepSemantics mode context
                prior next input proof := by
            simpa [Holds, priorProof, foldProof] using holds
          rcases base with
            ⟨_, _, freshNonempty, _, _, nextEq, recomputedXOut,
              outgoingFreshLinked⟩
          subst next
          exact {
            freshNonempty := freshNonempty
            chunkCount := rfl
            stepCount := rfl
            initialBoundary := rfl
            initialSemanticState := rfl
            programCounter := rfl
            publicTrace := rfl
            installedAccumulator := ⟨stepSemantics.emptyRunning, rfl, rfl⟩
            installedNebula := rfl
            recomputedXOut := recomputedXOut
            outgoingFreshLinked := outgoingFreshLinked
            pinned := nextPinned
          }
      | recursive nifsProof =>
          simp [Holds, priorProof, foldProof] at holds
  | active running latest =>
      cases foldProof : proof.fold with
      | noFold =>
          simp [Holds, priorProof, foldProof] at holds
      | recursive nifsProof =>
          have recursive :
              RecursiveHolds hashSemantics stepSemantics mode context
                prior next input proof running latest nifsProof := by
            simpa [Holds, priorProof, foldProof] using holds
          rcases recursive with ⟨_, _, _, _, verified⟩
          cases verifierEq : stepSemantics.nifsVerify
              (nifsContext stepSemantics prior input) running latest nifsProof with
          | none => simp [verifierEq] at verified
          | some nextRunning =>
              simp only [verifierEq] at verified
              rcases verified with
                ⟨freshNonempty, _, _, nextEq, recomputedXOut,
                  outgoingFreshLinked⟩
              subst next
              exact {
                freshNonempty := freshNonempty
                chunkCount := rfl
                stepCount := rfl
                initialBoundary := rfl
                initialSemanticState := rfl
                programCounter := rfl
                publicTrace := rfl
                installedAccumulator := ⟨nextRunning, rfl, rfl⟩
                installedNebula := rfl
                recomputedXOut := recomputedXOut
                outgoingFreshLinked := outgoingFreshLinked
                pinned := nextPinned
              }

end Nightstream.Protocol.FPrime.Step
