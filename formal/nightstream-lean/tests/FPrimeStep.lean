import Nightstream.Assurance.FPrimeTrace
import Nightstream.HyperNova.Construction2.Default
import Nightstream.Implementation.Rust.FPrime

/-!
Executable M3 witnesses for the true base branch, recursive NIFS branch, stateful
application transition, Nebula carry, outgoing recursive link, and a two-step
trace. Each rejection mutates one authority-bearing part of an honest step.
-/

namespace NightstreamTests.FPrimeStep

open Nightstream.HyperNova.Construction2
open Nightstream.Protocol.FPrime
open Nightstream.Assurance.FPrimeTrace

structure Fresh where
  link : Nat
  payload : Nat
deriving Repr, DecidableEq

def optionCode : Option Nat → Nat
  | none => 0
  | some value => value + 1

def toyHash : XOut.Message Nat Nat Nat Nat Nat → Nat
  | .verifier preimage =>
      100000 + preimage.params + 3 * preimage.structureDigest +
        5 * preimage.piCcsHeader + 7 * optionCode preimage.publicInputLength +
        11 * preimage.initialSemanticState
  | .initialBoundary preimage =>
      200000 + preimage.structureDigest + 13 * optionCode preimage.publicInputLength
  | .publicTraceSeed preimage =>
      300000 + preimage.structureDigest
  | .stateOutput preimage =>
      400000 + preimage.vkFsDigest + 17 * preimage.piCcsHeader +
        19 * preimage.chunkCount + 23 * preimage.stepCount + 29 * preimage.pc +
        31 * preimage.currentBoundary + 37 * optionCode preimage.semanticState +
        41 * preimage.construction2Accumulator + 43 * optionCode preimage.nebula

def hashSemantics : XOut.Semantics Nat Nat Nat Nat Nat Nat where
  hash := toyHash
  nebulaDigest := id

def context : XOut.Context Nat Nat Nat Nat where
  params := 2
  structureDigest := 3
  piCcsHeader := 5
  publicInputLength := some 7
  initialSemanticState := 11

def payloadSum (fresh : List Fresh) : Nat :=
  fresh.foldl (fun total item => total + item.payload) 0

def expectedNifsProof
    (transcript : Step.NifsContext Nat Nat)
    (running : Nat)
    (latest : List Fresh) : Nat :=
  running + latest.length + transcript.chunkCount + transcript.stepCount +
    transcript.zi + transcript.nextChunkDigest

def stepSemantics : Step.Semantics Nat Nat Fresh Nat Nat Nat where
  emptyRunning := 0
  initialNebula := none
  runningDigest := id
  chunkDigest := fun start fresh => 1000 + 100 * start + fresh.length
  freshLink := fun digest fresh => decide (fresh.link = digest)
  nifsVerify := fun transcript running latest proof =>
    if proof = expectedNifsProof transcript running latest then
      some (running + latest.length)
    else
      none
  applicationStep := fun prior fresh next =>
    decide (next = prior + payloadSum fresh)
  nebulaVerify := fun prior opening next =>
    match opening with
    | none => decide (next = prior)
    | some opened => decide (next = some opened)

abbrev TestState := State Nat Nat Fresh Nat
abbrev TestInput := Step.Input Fresh Nat Nat
abbrev TestProof := Step.Proof Nat Nat Nat

def initial : TestState where
  chunkCount := 0
  stepCount := 0
  z0 := XOut.initialBoundary hashSemantics context
  zi := XOut.initialBoundary hashSemantics context
  initialSemanticState := 11
  semanticState := 0
  pc := 1
  accumulatorDigest := 0
  publicTrace := XOut.publicTraceSeed hashSemantics context
  proof := .initial
  nebula := none

def baseInputFor (link : Nat) : TestInput where
  nextLatest := [{ link := link, payload := 5 }, { link := link, payload := 7 }]
  nebulaOpen := none
  nebulaNext := none

def baseProofFor (xOut : Nat) : TestProof where
  fold := .noFold
  nebulaOpen := none
  semanticStateDigest := 0
  xOut := xOut

def baseTemplateNext : TestState :=
  Step.advancedState stepSemantics initial 0 (baseInputFor 0) (baseProofFor 0)

def baseXOut : Nat := XOut.compute hashSemantics .stateless context baseTemplateNext
def baseInput : TestInput := baseInputFor baseXOut
def baseProof : TestProof := baseProofFor baseXOut
def afterBase : TestState :=
  Step.advancedState stepSemantics initial 0 baseInput baseProof

theorem initialValid :
    Step.InitialState hashSemantics stepSemantics .stateless context initial := by
  simp [Step.InitialState, initial, hashSemantics, context, stepSemantics,
    XOut.initialBoundary, XOut.publicTraceSeed]

example : Step.check hashSemantics stepSemantics .stateless context
    initial afterBase baseInput baseProof = true := by native_decide

-- Base authority and exact branch selection.
example : Step.check hashSemantics stepSemantics .stateless context
    { initial with z0 := initial.z0 + 1, zi := initial.zi + 1 }
    afterBase baseInput baseProof = false := by native_decide

example : Step.check hashSemantics stepSemantics .stateless context
    { initial with publicTrace := initial.publicTrace + 1 }
    afterBase baseInput baseProof = false := by native_decide

example : Step.check hashSemantics stepSemantics .stateless context
    { initial with accumulatorDigest := 1, semanticState := 1 }
    afterBase baseInput baseProof = false := by native_decide

example : Step.check hashSemantics stepSemantics .stateless context
    { initial with nebula := some 9 }
    afterBase baseInput baseProof = false := by native_decide

example : Step.check hashSemantics stepSemantics .stateless context
    initial afterBase { baseInput with nextLatest := [] } baseProof = false := by native_decide

example : Step.check hashSemantics stepSemantics .stateless context
    initial afterBase baseInput { baseProof with fold := .recursive 0 } = false := by native_decide

example : Step.check hashSemantics stepSemantics .stateless context
    initial afterBase baseInput { baseProof with semanticStateDigest := 1 } = false := by native_decide

example : Step.check hashSemantics stepSemantics .stateless context
    initial afterBase baseInput { baseProof with xOut := baseXOut + 1 } = false := by native_decide

example : Step.check hashSemantics stepSemantics .stateless context
    initial afterBase { baseInput with nextLatest := [{ link := 0, payload := 5 }] }
    baseProof = false := by native_decide

-- The standalone step owns no outgoing public-link check. A same-length batch
-- with bad links still satisfies LocalHolds, then fails the closed relation.
def unlinkedBaseInput : TestInput :=
  { baseInput with nextLatest :=
      [{ link := baseXOut + 1, payload := 5 },
       { link := baseXOut + 1, payload := 7 }] }

def afterUnlinkedBase : TestState :=
  Step.advancedState stepSemantics initial 0 unlinkedBaseInput baseProof

example : Step.checkLocal hashSemantics stepSemantics .stateless context
    initial afterUnlinkedBase unlinkedBaseInput baseProof = true := by native_decide

example : Step.check hashSemantics stepSemantics .stateless context
    initial afterUnlinkedBase unlinkedBaseInput baseProof = false := by native_decide

-- Recursive branch: the prior latest is linked to the prior state's x_out,
-- NIFS computes running=2, then one fresh item is installed and Nebula=9.
def recursiveInputFor (link : Nat) : TestInput where
  nextLatest := [{ link := link, payload := 3 }]
  nebulaOpen := some 9
  nebulaNext := some 9

def recursiveTemplateNext : TestState :=
  Step.advancedState stepSemantics afterBase 2 (recursiveInputFor 0) {
    fold := .recursive 0
    nebulaOpen := some 9
    semanticStateDigest := 2
    xOut := 0
  }

def recursiveXOut : Nat :=
  XOut.compute hashSemantics .stateless context recursiveTemplateNext

def recursiveInput : TestInput := recursiveInputFor recursiveXOut

def recursiveNifsProof : Nat :=
  expectedNifsProof (Step.nifsContext stepSemantics afterBase recursiveInput)
    0 baseInput.nextLatest

def recursiveProofFor (xOut : Nat) : TestProof where
  fold := .recursive recursiveNifsProof
  nebulaOpen := some 9
  semanticStateDigest := 2
  xOut := xOut

def recursiveProof : TestProof := recursiveProofFor recursiveXOut
def afterRecursive : TestState :=
  Step.advancedState stepSemantics afterBase 2 recursiveInput recursiveProof

example : Step.check hashSemantics stepSemantics .stateless context
    afterBase afterRecursive recursiveInput recursiveProof = true := by native_decide

-- Prior recursive link, accumulator authority, NIFS result, and output link.
def wrongPriorLink : TestState :=
  { afterBase with proof := (.active 0
      [{ link := baseXOut + 1, payload := 5 }, { link := baseXOut, payload := 7 }]) }

example : Step.check hashSemantics stepSemantics .stateless context
    wrongPriorLink afterRecursive recursiveInput recursiveProof = false := by native_decide

example : Step.check hashSemantics stepSemantics .stateless context
    { afterBase with accumulatorDigest := 1 }
    afterRecursive recursiveInput recursiveProof = false := by native_decide

example : Step.check hashSemantics stepSemantics .stateless context
    afterBase afterRecursive recursiveInput
      { recursiveProof with fold := .recursive (recursiveNifsProof + 1) } = false := by
  native_decide

def replayTranscriptPrior : TestState :=
  { afterBase with zi := afterBase.zi + 1, publicTrace := afterBase.publicTrace + 1 }

-- The NIFS replay context changes even when running/latest stay fixed.
example : stepSemantics.nifsVerify
    (Step.nifsContext stepSemantics replayTranscriptPrior recursiveInput)
    0 baseInput.nextLatest recursiveNifsProof = none := by native_decide

example : Step.check hashSemantics stepSemantics .stateless context
    afterBase afterRecursive { recursiveInput with nextLatest := [] }
      recursiveProof = false := by native_decide

example : Step.check hashSemantics stepSemantics .stateless context
    afterBase afterRecursive recursiveInput
      { recursiveProof with semanticStateDigest := 3 } = false := by native_decide

example : Step.check hashSemantics stepSemantics .stateless context
    afterBase afterRecursive recursiveInput
      { recursiveProof with nebulaOpen := none } = false := by native_decide

example : Step.check hashSemantics stepSemantics .stateless context
    afterBase { afterRecursive with stepCount := 4 }
      recursiveInput recursiveProof = false := by native_decide

example : Step.check hashSemantics stepSemantics .stateless context
    afterBase afterRecursive
      { recursiveInput with nextLatest := [{ link := 0, payload := 3 }] }
      recursiveProof = false := by native_decide

example : Step.check hashSemantics stepSemantics .stateless context
    afterBase afterRecursive recursiveInput
      { recursiveProof with fold := .noFold } = false := by native_decide

-- Stateful mode starts at the verifier-owned state 11 and applies 5+7.
def statefulInitial : TestState := { initial with semanticState := 11 }

def statefulProofFor (xOut : Nat) : TestProof where
  fold := .noFold
  nebulaOpen := none
  semanticStateDigest := 23
  xOut := xOut

def statefulTemplateNext : TestState :=
  Step.advancedState stepSemantics statefulInitial 0
    (baseInputFor 0) (statefulProofFor 0)

def statefulXOut : Nat :=
  XOut.compute hashSemantics .stateful context statefulTemplateNext

def statefulInput : TestInput := baseInputFor statefulXOut
def statefulProof : TestProof := statefulProofFor statefulXOut
def afterStateful : TestState :=
  Step.advancedState stepSemantics statefulInitial 0 statefulInput statefulProof

example : Step.check hashSemantics stepSemantics .stateful context
    statefulInitial afterStateful statefulInput statefulProof = true := by native_decide

example : Step.check hashSemantics stepSemantics .stateful context
    { statefulInitial with semanticState := 12 }
    afterStateful statefulInput statefulProof = false := by native_decide

example : Step.check hashSemantics stepSemantics .stateful context
    statefulInitial afterStateful statefulInput
      { statefulProof with semanticStateDigest := 22 } = false := by native_decide

-- FPR-BASE-SPEC: the paper pair is materially replicated into its configured
-- vector; Rust's empty product is separately valid by zero arity.
def defaultRelation (system : Nat) (claim witness : Nat → Nat) : Prop :=
  claim system = witness system

def paperDefault :
    Default.DefaultPair Nat (Nat → Nat) (Nat → Nat) defaultRelation where
  claim := fun _ => 0
  witness := fun _ => 0
  satisfies := by intro; rfl

example : Default.AllPairs (defaultRelation 7)
    (List.replicate 3 paperDefault.claim)
    (List.replicate 3 paperDefault.witness) :=
  Default.replicatedDefault_allPairs defaultRelation paperDefault 7 3

def impossibleRelation (_system _claim _witness : Nat) : Prop := False

example : Default.ZeroAritySpecialization impossibleRelation 7
    (Default.emptyRunning (Claim := Nat) (Witness := Nat) (Parent := Nat)) :=
  Default.emptyRunning_zeroArity impossibleRelation 7

example : ¬ Default.ShapeValid ({
    claims := []
    witnesses := []
    parentAuthority := some 9
  } : Default.RunningProduct Nat Nat Nat) :=
  Default.empty_claims_with_parent_rejected 9

def environment : Environment Nat Nat Nat Nat Nat Fresh Nat Nat Nat Nat where
  hashSemantics := hashSemantics
  stepSemantics := stepSemantics
  mode := .stateless
  context := context

def baseInvocation : Invocation Nat Fresh Nat Nat Nat where
  input := baseInput
  proof := baseProof

def recursiveInvocation : Invocation Nat Fresh Nat Nat Nat where
  input := recursiveInput
  proof := recursiveProof

def honestTrace : AcceptedTrace environment initial [2, 1] afterRecursive :=
  .snoc
    (.snoc (next := afterBase) .nil baseInvocation (by native_decide))
    recursiveInvocation
    (by native_decide)

example : CounterRefines [2, 1] afterRecursive :=
  (accepted_trace_sound environment initial afterRecursive [2, 1]
    initialValid honestTrace).counterRefinement

example : Nightstream.Assurance.Reachable (Edge environment) initial 2 afterRecursive :=
  (accepted_trace_sound environment initial afterRecursive [2, 1]
    initialValid honestTrace).exactReachability

example : Nightstream.Assurance.ValidExecution (Edge environment)
    (fun state => state = afterRecursive) initial afterRecursive 2 :=
  accepted_trace_valid_execution environment initial afterRecursive [2, 1]
    initialValid honestTrace (fun state => state = afterRecursive) rfl

-- A terminal counter forgery cannot satisfy the trace result.
example : ¬ CounterRefines [2, 1] { afterRecursive with stepCount := 4 } := by
  simp [CounterRefines, afterRecursive, Step.advancedState]

def rustResultCode : Except Nightstream.Implementation.Rust.FPrime.Error Unit → Nat
  | .ok _ => 0
  | .error .variant => 1
  | .error .entryState => 2
  | .error .priorLatestEmpty => 3
  | .error .priorLink => 4
  | .error .nifs => 5
  | .error .emptyStep => 6
  | .error .semanticAdvance => 7
  | .error .nebulaAdvance => 8
  | .error .nextState => 9
  | .error .xOut => 10

def rustVerify (prior next : TestState) (input : TestInput) (proof : TestProof) :
    Except Nightstream.Implementation.Rust.FPrime.Error Unit :=
  Nightstream.Implementation.Rust.FPrime.verify hashSemantics stepSemantics
    .stateless context prior next input proof

example : rustResultCode (rustVerify initial afterBase baseInput baseProof) = 0 := by
  native_decide

example : rustResultCode (rustVerify afterBase afterRecursive recursiveInput recursiveProof) = 0 := by
  native_decide

example : rustResultCode (rustVerify initial afterBase baseInput
    { baseProof with fold := .recursive 0 }) = 1 := by
  native_decide

example : rustResultCode (rustVerify { initial with pc := 2 }
    afterBase baseInput baseProof) = 2 := by native_decide

def emptyPrior : TestState := { afterBase with proof := .active 0 [] }

def emptyPriorProof : TestProof := {
  recursiveProof with
  fold := .recursive (expectedNifsProof
    (Step.nifsContext stepSemantics emptyPrior recursiveInput) 0 [])
}

example : rustResultCode (rustVerify emptyPrior afterRecursive
    recursiveInput emptyPriorProof) = 3 := by native_decide

example : rustResultCode (rustVerify wrongPriorLink afterRecursive
    recursiveInput recursiveProof) = 4 := by native_decide

example : rustResultCode (rustVerify afterBase afterRecursive recursiveInput
      { recursiveProof with fold := .recursive (recursiveNifsProof + 1) }) = 5 := by
  native_decide

def emptyInstallInput : TestInput := { recursiveInput with nextLatest := [] }

def emptyInstallProof : TestProof := {
  recursiveProof with
  fold := .recursive (expectedNifsProof
    (Step.nifsContext stepSemantics afterBase emptyInstallInput)
    0 baseInput.nextLatest)
}

example : rustResultCode (rustVerify afterBase afterRecursive
    emptyInstallInput emptyInstallProof) = 6 := by native_decide

example : rustResultCode (rustVerify afterBase afterRecursive recursiveInput
      { recursiveProof with semanticStateDigest := 3 }) = 7 := by native_decide

example : rustResultCode (rustVerify afterBase afterRecursive recursiveInput
      { recursiveProof with nebulaOpen := none }) = 8 := by native_decide

example : rustResultCode (rustVerify afterBase { afterRecursive with stepCount := 4 }
      recursiveInput recursiveProof) = 9 := by native_decide

example : rustResultCode (rustVerify afterBase afterRecursive recursiveInput
      { recursiveProof with xOut := recursiveXOut + 1 }) = 10 := by native_decide

end NightstreamTests.FPrimeStep
