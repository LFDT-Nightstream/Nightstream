import Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.ObservedTrace

/-!
Contract: conserved proof-free receipts for the public native F' step.

Owns:
- branch-tagged differential cases over the compact atoms from `Core`;
- equality-keyed primitive receipts and fail-closed lookup;
- exact observed event/call conservation and recorded-outcome oracle replay;
- separately typed lifecycle-boundary calls.

Does not own:
- the receipt-free checker or its success theorem (owned by `Core`);
- Rust serialization, generated fixtures, or Rust-source conformance;
- R1CS rows, terminal verification, or application `Machine.step`.

Emits constraints: no.

| Stage path | Obligation | Authority class | Owner |
|---|---|---|---|
| `fprime.native.receipt.dispatch` | redundant branch tag agrees with typed input | checked | `ReceiptWellFormed` |
| `fprime.native.receipt.events` | actual call-site trace projects to exactly the reached calls | checked | `ControlFlowAndCallConservation` |
| `fprime.native.receipt.lookup` | missing or mismatched input returns poison/rejection | computed | `lookup*` |
| `fprime.native.receipt.outcome` | recorded result equals normalized oracle replay | derived | `check_eq_true_iff_oracleReplayConforms` |
| `fprime.native.boundary` | lifecycle-only calls remain outside native receipts | direct dataflow | `BoundaryReceipt` |
-/

namespace Nightstream.Implementation.Rust.CanonicalConformance.NativeStep

open Nightstream.HyperNova.Construction2
open Nightstream.Protocol.FPrime

abbrev NativeNifsContext := Step.NifsContext Digest Nebula
abbrev NativeHashMessage :=
  XOut.Message Params StructureDigest Header Digest NebulaDigest

inductive Outcome where
  | accepted (next : NativeState)
  | rejected (error : Error)
deriving Repr, DecidableEq

/-- Explicit branch tag exported beside the tagged state and proof. -/
inductive Dispatch where
  | initialNoFold
  | initialRecursive (proof : NifsProof)
  | activeNoFold (running : Running) (latest : List Fresh)
  | activeRecursive
      (running : Running)
      (latest : List Fresh)
      (proof : NifsProof)
deriving Repr, DecidableEq

def dispatchOf (prior : NativeState) (proof : NativeProof) : Dispatch :=
  match prior.proof, proof.fold with
  | .initial, .noFold => .initialNoFold
  | .initial, .recursive nifsProof => .initialRecursive nifsProof
  | .active running latest, .noFold => .activeNoFold running latest
  | .active running latest, .recursive nifsProof =>
      .activeRecursive running latest nifsProof

/-- Exact native calls, including their returned values, in execution order. -/
inductive Call where
  | chunkDigest
      (stepCount : Nat)
      (nextLatest : List Fresh)
      (output : Digest)
  | nifs
      (context : NativeNifsContext)
      (running : Running)
      (latest : List Fresh)
      (proof : NifsProof)
      (output : Option Running)
  | runningDigest (running : Running) (output : Digest)
  | hash (input : NativeHashMessage) (output : Digest)
  | nebulaDigest (input : Nebula) (output : NebulaDigest)
deriving Repr, DecidableEq

/-- Equality-normalized verifier oracle for materialized running digests. -/
abbrev RunningDigestTable := List (Running × Digest)

/-- Calls not owned by the native step.  A lifecycle differential may supply
these separately when composing the native result with `Step.LocalHolds`. -/
inductive BoundaryCall where
  | hash (input : NativeHashMessage) (output : Digest)
  | nebulaDigest (input : Nebula) (output : NebulaDigest)
  | runningDigest (running : Running) (output : Digest)
  | freshLink (digest : Digest) (fresh : Fresh)
  | application
      (priorSemantic : Digest)
      (nextLatest : List Fresh)
      (nextSemantic : Digest)
  | nebulaVerify
      (prior : Option Nebula)
      (openPayload : Option NebulaOpen)
      (next : Option Nebula)
deriving Repr, DecidableEq

structure BoundaryReceipt where
  initialNebula : Option Nebula
  calls : List BoundaryCall
deriving Repr, DecidableEq

/-- One compact, proof-free native-step differential case. -/
structure Receipt where
  mode : XOut.Mode
  context : NativeContext
  vkFsDigest : Digest
  relationColumns : Nat
  rawEncoding : RawEncodingTable
  emptyRunning : Running
  /-- Verifier-owned digest of Rust's empty accumulator handle. -/
  initialAccumulatorDigest : Digest
  /-- Equality oracle for the verifier-owned initial-boundary digest. -/
  initialBoundaryDigest : Digest
  /-- Equality oracle for the verifier-owned public-trace seed. -/
  publicTraceSeed : Digest
  /-- Equality oracle for materialized running accumulators. -/
  runningDigests : RunningDigestTable
  prior : NativeState
  input : NativeInput
  proof : NativeProof
  dispatch : Dispatch
  calls : List Call
  observed : ObservedTrace
  outcome : Outcome
deriving Repr, DecidableEq

private def lookupChunkDigest
    (calls : List Call)
    (stepCount : Nat)
    (nextLatest : List Fresh) : Digest :=
  match calls with
  | [] => poison .digest
  | .chunkDigest foundStep foundLatest output :: rest =>
      if foundStep = stepCount ∧ foundLatest = nextLatest then output
      else lookupChunkDigest rest stepCount nextLatest
  | _ :: rest => lookupChunkDigest rest stepCount nextLatest

private def lookupNifs
    (calls : List Call)
    (context : NativeNifsContext)
    (running : Running)
    (latest : List Fresh)
    (proof : NifsProof) : Option Running :=
  match calls with
  | [] => none
  | .nifs foundContext foundRunning foundLatest foundProof output :: rest =>
      if foundContext = context ∧ foundRunning = running ∧
          foundLatest = latest ∧ foundProof = proof then
        output
      else
        lookupNifs rest context running latest proof
  | _ :: rest => lookupNifs rest context running latest proof

private def lookupRunningDigest
    (table : RunningDigestTable)
    (running : Running) : Digest :=
  match table with
  | [] => poison .digest
  | (found, output) :: rest =>
      if found = running then output else lookupRunningDigest rest running

private def lookupHash
    (calls : List Call)
    (input : NativeHashMessage) : Digest :=
  match calls with
  | [] => poison .digest
  | .hash found output :: rest =>
      if found = input then output else lookupHash rest input
  | _ :: rest => lookupHash rest input

private def receiptHash
    (receipt : Receipt)
    (input : NativeHashMessage) : Digest :=
  match input with
  | .initialBoundary preimage =>
      if preimage = XOut.initialBoundaryPreimage receipt.context then
        receipt.initialBoundaryDigest
      else
        poison .digest
  | .publicTraceSeed preimage =>
      if preimage.structureDigest = receipt.context.structureDigest then
        receipt.publicTraceSeed
      else
        poison .digest
  | _ => lookupHash receipt.calls input

private def lookupNebulaDigest
    (calls : List Call)
    (input : Nebula) : NebulaDigest :=
  match calls with
  | [] => poison .nebulaDigest
  | .nebulaDigest found output :: rest =>
      if found = input then output else lookupNebulaDigest rest input
  | _ :: rest => lookupNebulaDigest rest input

def hashSemantics (receipt : Receipt) :
    XOut.Semantics Params StructureDigest Header Digest Nebula NebulaDigest where
  hash := receiptHash receipt
  nebulaDigest := lookupNebulaDigest receipt.calls

def stepSemantics (receipt : Receipt) :
    Step.Semantics Digest Running Fresh NifsProof Nebula NebulaOpen where
  emptyRunning := receipt.emptyRunning
  initialAccumulatorDigest := receipt.initialAccumulatorDigest
  initialNebula := none
  runningDigest := lookupRunningDigest receipt.runningDigests
  chunkDigest := lookupChunkDigest receipt.calls
  freshLink := fun _ _ => false
  nifsVerify := lookupNifs receipt.calls
  applicationStep := fun _ _ _ => false
  nebulaVerify := fun _ _ _ => false

private def boundaryHash
    (receipt : Receipt)
    (boundary : BoundaryReceipt)
    (input : NativeHashMessage) : Digest :=
  let native := (hashSemantics receipt).hash input
  if live native then
    native
  else
    let rec go : List BoundaryCall → Digest
      | [] => poison .digest
      | .hash found output :: rest =>
          if found = input then output else go rest
      | _ :: rest => go rest
    go boundary.calls

private def boundaryNebulaDigest
    (receipt : Receipt)
    (boundary : BoundaryReceipt)
    (input : Nebula) : NebulaDigest :=
  let native := lookupNebulaDigest receipt.calls input
  if live native then
    native
  else
    let rec go : List BoundaryCall → NebulaDigest
      | [] => poison .nebulaDigest
      | .nebulaDigest found output :: rest =>
          if found = input then output else go rest
      | _ :: rest => go rest
    go boundary.calls

private def boundaryRunningDigest
    (receipt : Receipt)
    (boundary : BoundaryReceipt)
    (running : Running) : Digest :=
  let native := lookupRunningDigest receipt.runningDigests running
  if live native then
    native
  else
    let rec go : List BoundaryCall → Digest
      | [] => poison .digest
      | .runningDigest found output :: rest =>
          if found = running then output else go rest
      | _ :: rest => go rest
    go boundary.calls

private def boundaryFreshLink
    (boundary : BoundaryReceipt)
    (digest : Digest)
    (fresh : Fresh) : Bool :=
  boundary.calls.any fun call =>
    match call with
    | .freshLink foundDigest foundFresh =>
        decide (foundDigest = digest ∧ foundFresh = fresh)
    | _ => false

private def boundaryApplication
    (boundary : BoundaryReceipt)
    (priorSemantic : Digest)
    (nextLatest : List Fresh)
    (nextSemantic : Digest) : Bool :=
  boundary.calls.any fun call =>
    match call with
    | .application foundPrior foundLatest foundNext =>
        decide (foundPrior = priorSemantic ∧ foundLatest = nextLatest ∧
          foundNext = nextSemantic)
    | _ => false

private def boundaryNebulaVerify
    (boundary : BoundaryReceipt)
    (prior : Option Nebula)
    (openPayload : Option NebulaOpen)
    (next : Option Nebula) : Bool :=
  boundary.calls.any fun call =>
    match call with
    | .nebulaVerify foundPrior foundOpen foundNext =>
        decide (foundPrior = prior ∧ foundOpen = openPayload ∧
          foundNext = next)
    | _ => false

def boundaryHashSemantics (receipt : Receipt) (boundary : BoundaryReceipt) :
    XOut.Semantics Params StructureDigest Header Digest Nebula NebulaDigest where
  hash := boundaryHash receipt boundary
  nebulaDigest := boundaryNebulaDigest receipt boundary

def boundaryStepSemantics (receipt : Receipt) (boundary : BoundaryReceipt) :
    Step.Semantics Digest Running Fresh NifsProof Nebula NebulaOpen where
  emptyRunning := receipt.emptyRunning
  initialAccumulatorDigest := receipt.initialAccumulatorDigest
  initialNebula := boundary.initialNebula
  runningDigest := boundaryRunningDigest receipt boundary
  chunkDigest := lookupChunkDigest receipt.calls
  freshLink := boundaryFreshLink boundary
  nifsVerify := lookupNifs receipt.calls
  applicationStep := boundaryApplication boundary
  nebulaVerify := boundaryNebulaVerify boundary

/-- A live native hash result cannot be replaced by lifecycle fallback data. -/
theorem boundaryHash_eq_native_of_live
    (receipt : Receipt)
    (boundary : BoundaryReceipt)
    (input : NativeHashMessage)
    (resultLive : live ((hashSemantics receipt).hash input) = true) :
    (boundaryHashSemantics receipt boundary).hash input =
      (hashSemantics receipt).hash input := by
  change live (receiptHash receipt input) = true at resultLive
  change boundaryHash receipt boundary input =
    receiptHash receipt input
  simp only [boundaryHash, hashSemantics, resultLive, ↓reduceIte]

/-- A live native Nebula digest cannot be replaced by lifecycle fallback data. -/
theorem boundaryNebulaDigest_eq_native_of_live
    (receipt : Receipt)
    (boundary : BoundaryReceipt)
    (input : Nebula)
    (resultLive :
      live ((hashSemantics receipt).nebulaDigest input) = true) :
    (boundaryHashSemantics receipt boundary).nebulaDigest input =
      (hashSemantics receipt).nebulaDigest input := by
  change live (lookupNebulaDigest receipt.calls input) = true at resultLive
  change boundaryNebulaDigest receipt boundary input =
    lookupNebulaDigest receipt.calls input
  simp only [boundaryNebulaDigest, resultLive, ↓reduceIte]

/-- A live native running digest cannot be replaced by lifecycle fallback data. -/
theorem boundaryRunningDigest_eq_native_of_live
    (receipt : Receipt)
    (boundary : BoundaryReceipt)
    (running : Running)
    (resultLive :
      live ((stepSemantics receipt).runningDigest running) = true) :
    (boundaryStepSemantics receipt boundary).runningDigest running =
      (stepSemantics receipt).runningDigest running := by
  change live (lookupRunningDigest receipt.runningDigests running) = true at resultLive
  change boundaryRunningDigest receipt boundary running =
    lookupRunningDigest receipt.runningDigests running
  simp only [boundaryRunningDigest, resultLive, ↓reduceIte]

/-- Calls and outcome emitted by one execution of the typed native checker. -/
structure Run where
  calls : List Call
  outcome : Outcome
deriving Repr, DecidableEq

private def maxUInt64 : Nat :=
  18446744073709551615

private def countersFit (prior : NativeState) (input : NativeInput) : Bool :=
  decide (prior.chunkCount < maxUInt64 ∧
    prior.stepCount + input.nextLatest.length ≤ maxUInt64)

private def finishRun
    (receipt : Receipt)
    (calls : List Call)
    (nextRunning : Running) : Run :=
  if receipt.proof.nebulaOpen = receipt.input.nebulaOpen then
    let semantics := stepSemantics receipt
    let runningDigest := semantics.runningDigest nextRunning
    let calls := calls ++ [.runningDigest nextRunning runningDigest]
    let next := nativeAdvancedState semantics receipt.mode receipt.prior
      nextRunning receipt.input receipt.proof
    if nativeSemanticCheck semantics receipt.mode nextRunning receipt.proof then
      let hash := hashSemantics receipt
      let verifierInput : NativeHashMessage := .verifier receipt.context
      let verifierOutput := hash.hash verifierInput
      let calls := calls ++ [.hash verifierInput verifierOutput]
      let nebulaOutput := next.nebula.map hash.nebulaDigest
      let calls :=
        match next.nebula, nebulaOutput with
        | some lane, some digest => calls ++ [.nebulaDigest lane digest]
        | _, _ => calls
      let stateInput : NativeHashMessage :=
        .stateOutput {
          vkFsDigest := verifierOutput
          piCcsHeader := receipt.context.piCcsHeader
          chunkCount := next.chunkCount
          stepCount := next.stepCount
          pc := next.pc
          currentBoundary := next.zi
          semanticState := match receipt.mode with
            | .stateless => none
            | .stateful => some next.semanticState
          construction2Accumulator := next.accumulatorDigest
          nebula := nebulaOutput
        }
      let stateOutput := hash.hash stateInput
      let calls := calls ++ [.hash stateInput stateOutput]
      if stateOutput = receipt.proof.xOut then
        ⟨calls, .accepted next⟩
      else
        ⟨calls, .rejected .xOutMismatch⟩
    else
      ⟨calls, .rejected .statelessSemanticInvariantViolated⟩
  else
    ⟨calls, .rejected .nebulaOpenMismatch⟩

/-- Exact source-order execution for the bounded, materialized-running profile.
The receipt supplies primitive results; the checker derives every control-flow
decision and every equality itself. -/
def run (receipt : Receipt) : Run :=
  if receipt.prior.pc ≠ 1 then
    ⟨[], .rejected .pcOutOfRange⟩
  else if branchShape receipt.prior = false then
    ⟨[], .rejected .baseCaseMismatch⟩
  else if nativeStateAuthorityCheck (hashSemantics receipt)
      (stepSemantics receipt) receipt.mode receipt.context receipt.prior = false then
    ⟨[], .rejected .stateAuthorityMismatch⟩
  else if receipt.input.nextLatest = [] then
    ⟨[], .rejected .emptyStep⟩
  else
    let semantics := stepSemantics receipt
    let chunkDigest :=
      semantics.chunkDigest receipt.prior.stepCount receipt.input.nextLatest
    let calls := [
      .chunkDigest receipt.prior.stepCount receipt.input.nextLatest chunkDigest
    ]
    match receipt.prior.proof, receipt.proof.fold with
    | .initial, .noFold =>
        finishRun receipt calls receipt.emptyRunning
    | .active running latest, .recursive nifsProof =>
        let context := Step.nifsContext semantics receipt.prior receipt.input
        let output := semantics.nifsVerify context running latest nifsProof
        let calls := calls ++ [.nifs context running latest nifsProof output]
        match output with
        | none => ⟨calls, .rejected .nifsRejected⟩
        | some nextRunning => finishRun receipt calls nextRunning
    | _, _ => ⟨calls, .rejected .foldProofVariantMismatch⟩

private def observedNifsContext
    (receipt : Receipt)
    (chunk : ChunkDigestCall) : NativeNifsContext where
  chunkCount := receipt.prior.chunkCount
  stepCount := receipt.prior.stepCount
  z0 := receipt.prior.z0
  zi := receipt.prior.zi
  initialSemanticState := receipt.prior.initialSemanticState
  semanticState := receipt.prior.semanticState
  pc := receipt.prior.pc
  accumulatorDigest := receipt.prior.accumulatorDigest
  publicTrace := receipt.prior.publicTrace
  nebula := receipt.prior.nebula
  nextChunkDigest := chunk.output

private def observedStateOutputPreimage?
    (receipt : Receipt) :
    Option (XOut.XOutPreimage Digest Header NebulaDigest) :=
  match receipt.observed.advancedState,
      receipt.observed.verifierDigestRead,
      receipt.observed.piCcsHeaderRead with
  | some next, some verifierDigest, some header =>
      some {
        vkFsDigest := verifierDigest
        piCcsHeader := header
        chunkCount := next.chunkCount
        stepCount := next.stepCount
        pc := next.pc
        currentBoundary := next.zi
        semanticState := match receipt.mode with
          | .stateless => none
          | .stateful => some next.semanticState
        construction2Accumulator := next.accumulatorDigest
        nebula := receipt.observed.nebulaDigest.map (·.output)
      }
  | _, _, _ => none

/-- Projection from the rich call-site trace to the primitive-result oracle
used by the receipt-free checker.  This is a structural projection, not a
claim that the observed primitive results are cryptographically correct. -/
def observedSemanticCalls (receipt : Receipt) : List Call :=
  let chunkCalls :=
    match receipt.observed.chunkDigest with
    | none => []
    | some chunk =>
        [.chunkDigest chunk.startIndex chunk.orderedClaims chunk.output]
  let nifsCalls :=
    match receipt.observed.chunkDigest, receipt.observed.nifsCall with
    | some chunk, some nifs =>
        [.nifs (observedNifsContext receipt chunk) nifs.running nifs.fresh
          nifs.proof nifs.outcome]
    | _, _ => []
  let runningCalls :=
    match receipt.observed.runningDigest with
    | none => []
    | some call => [.runningDigest call.running call.output]
  let hashCalls :=
    match observedStateOutputPreimage? receipt,
        receipt.observed.verifierDigestRead,
        receipt.observed.stateXOutHash with
    | some preimage, some verifierDigest, some stateHash =>
        [.hash (.verifier receipt.context) verifierDigest] ++
        (match receipt.observed.nebulaDigest with
          | none => []
          | some nebula => [.nebulaDigest nebula.input nebula.output]) ++
        [.hash (.stateOutput preimage) stateHash.outputDigest]
    | _, _, _ => []
  chunkCalls ++ nifsCalls ++ runningCalls ++ hashCalls

private def observedExecutionOrder
    (trace : ObservedTrace) : List EventKind :=
  (match trace.chunkDigest with
    | none => []
    | some _ => [.chunkDigest, .dispatch]) ++
  (match trace.transcript with
    | none => []
    | some transcript =>
        [.transcriptStarted] ++
        List.replicate transcript.orderedAppends.length .transcriptAppend ++
        [.transcriptPrefix]) ++
  (match trace.nifsCall with
    | none => []
    | some _ => [.nifsVerify]) ++
  (match trace.runningDigest with
    | none => []
    | some _ => [.runningDigest]) ++
  (match trace.advancedState with
    | none => []
    | some _ => [.stateAdvanced]) ++
  (match trace.verifierDigestRead with
    | none => []
    | some _ => [.verifierDigestRead]) ++
  (match trace.piCcsHeaderRead with
    | none => []
    | some _ => [.piCcsHeaderRead]) ++
  (match trace.nebulaDigest with
    | none => []
    | some _ => [.nebulaDigest]) ++
  (match trace.stateXOutHash with
    | none => []
    | some _ => [.stateXOutHash])

private def expectedEventsForCall
    (receipt : Receipt) : Call → List EventKind
  | .chunkDigest _ _ _ => [.chunkDigest, .dispatch]
  | .nifs context _ _ _ _ =>
      [.transcriptStarted] ++
      List.replicate
        (expectedTranscriptAppends receipt.rawEncoding receipt.vkFsDigest
          receipt.context.piCcsHeader receipt.prior
          context.nextChunkDigest).length
        .transcriptAppend ++
      [.transcriptPrefix, .nifsVerify]
  | .runningDigest _ _ => [.runningDigest, .stateAdvanced]
  | .hash (.verifier _) _ => [.verifierDigestRead, .piCcsHeaderRead]
  | .hash (.stateOutput _) _ => [.stateXOutHash]
  | .hash _ _ => []
  | .nebulaDigest _ _ => [.nebulaDigest]

private def expectedExecutionOrder (receipt : Receipt) : List EventKind :=
  receipt.calls.flatMap (expectedEventsForCall receipt)

private def expectedFinalStage : Outcome → ExecutionStage
  | .accepted _ => .complete
  | .rejected .pcOutOfRange => .entry
  | .rejected .baseCaseMismatch => .entry
  | .rejected .stateAuthorityMismatch => .entry
  | .rejected .emptyStep => .entry
  | .rejected .foldProofVariantMismatch => .dispatch
  | .rejected .nifsRejected => .nifs
  | .rejected .nebulaOpenMismatch => .nebula
  | .rejected .statelessSemanticInvariantViolated => .semantic
  | .rejected .xOutMismatch => .xOut

private def transcriptPrefixCheck (receipt : Receipt) : Bool :=
  match receipt.observed.nifsCall, receipt.observed.transcript,
      receipt.observed.chunkDigest with
  | none, none, _ => true
  | some _, some transcript, some chunk =>
      decide (transcript.label = .fPrimeStep) &&
      decide (transcript.orderedAppends =
        expectedTranscriptAppends receipt.rawEncoding receipt.vkFsDigest
          receipt.context.piCcsHeader receipt.prior chunk.output) &&
      decide (transcript.prefixSnapshot.state.length = 8) &&
      RawFieldsWellFormed transcript.prefixSnapshot.state
  | _, _, _ => false

private def runningDigestCallCheck (receipt : Receipt) : Bool :=
  match receipt.observed.runningDigest with
  | none => true
  | some call => decide (call.relationColumns = receipt.relationColumns)

private def advancedStateCheck (receipt : Receipt) : Bool :=
  match receipt.observed.runningDigest, receipt.observed.advancedState with
  | none, none => true
  | some call, some actual =>
      decide (actual = nativeAdvancedState (stepSemantics receipt)
        receipt.mode receipt.prior call.running receipt.input receipt.proof)
  | _, _ => false

private def verifierDigestReadCheck (receipt : Receipt) : Bool :=
  match receipt.observed.verifierDigestRead with
  | none => true
  | some digest => decide (digest = receipt.vkFsDigest)

private def nebulaRawEncodingCheck (receipt : Receipt) : Bool :=
  match receipt.observed.nebulaDigest with
  | none => true
  | some call =>
      decide
        (lookupRawFields receipt.rawEncoding (.nebula call.input) =
          lookupRawFields receipt.rawEncoding (.nebulaDigest call.output))

private def stateXOutHashCheck (receipt : Receipt) : Bool :=
  match receipt.observed.stateXOutHash,
      observedStateOutputPreimage? receipt with
  | none, none => true
  | some call, some preimage =>
      decide (call.rawPreimage =
        encodeStateXOutPreimage receipt.rawEncoding preimage) &&
      decide (call.outputDigest = call.output)
  | _, _ => false

/-- Truthful scope of the instrumented receipt: control-flow and exact
call-argument/result conservation.  Primitive correctness is intentionally not
part of this proposition. -/
def ControlFlowAndCallConservation (receipt : Receipt) : Prop :=
  receipt.dispatch = dispatchOf receipt.prior receipt.proof ∧
  RawEncodingTableWellFormed receipt.rawEncoding = true ∧
  observedSemanticCalls receipt = receipt.calls ∧
  (run receipt).calls = receipt.calls ∧
  receipt.observed.executionOrder =
    observedExecutionOrder receipt.observed ∧
  receipt.observed.executionOrder = expectedExecutionOrder receipt ∧
  receipt.observed.finalStage = expectedFinalStage (run receipt).outcome ∧
  transcriptPrefixCheck receipt = true ∧
  runningDigestCallCheck receipt = true ∧
  advancedStateCheck receipt = true ∧
  verifierDigestReadCheck receipt = true ∧
  nebulaRawEncodingCheck receipt = true ∧
  stateXOutHashCheck receipt = true

instance (receipt : Receipt) :
    Decidable (ControlFlowAndCallConservation receipt) := by
  unfold ControlFlowAndCallConservation
  infer_instance

def controlFlowAndCallConservationCheck (receipt : Receipt) : Bool :=
  decide (ControlFlowAndCallConservation receipt)

theorem controlFlowAndCallConservationCheck_eq_true_iff
    (receipt : Receipt) :
    controlFlowAndCallConservationCheck receipt = true ↔
      ControlFlowAndCallConservation receipt := by
  simp [controlFlowAndCallConservationCheck]

private def optionAtomsLive {sort : AtomSort}
    (value : Option (Atom sort)) : Bool :=
  match value with
  | none => true
  | some atom => live atom

private def stateAtomsLive (state : NativeState) : Bool :=
  live state.z0 &&
  live state.zi &&
  live state.initialSemanticState &&
  live state.semanticState &&
  live state.accumulatorDigest &&
  live state.publicTrace &&
  optionAtomsLive state.nebula &&
  match state.proof with
  | .initial => true
  | .active running latest =>
      live running && latest.all live

private def inputAtomsLive (input : NativeInput) : Bool :=
  input.nextLatest.all live &&
  optionAtomsLive input.nebulaOpen &&
  optionAtomsLive input.nebulaNext

private def proofAtomsLive (proof : NativeProof) : Bool :=
  (match proof.fold with
    | .noFold => true
    | .recursive nifsProof => live nifsProof) &&
  optionAtomsLive proof.nebulaOpen &&
  live proof.semanticStateDigest &&
  live proof.xOut

private def callAtomsLive : Call → Bool
  | .chunkDigest _ latest output =>
      latest.all live && live output
  | .nifs context running latest proof output =>
      live context.z0 && live context.zi &&
      live context.initialSemanticState &&
      live context.semanticState && live context.accumulatorDigest &&
      live context.publicTrace && live context.nextChunkDigest &&
      optionAtomsLive context.nebula && live running &&
      latest.all live && live proof &&
      optionAtomsLive output
  | .runningDigest running output => live running && live output
  | .hash _ output => live output
  | .nebulaDigest input output => live input && live output

private def runningDigestTableAtomsLive
    (table : RunningDigestTable) : Bool :=
  table.all fun entry => live entry.1 && live entry.2

private def rawEncodingKeyAtomsLive : RawEncodingKey → Bool
  | .digest value => live value
  | .header value => live value
  | .nebula value => live value
  | .nebulaDigest value => live value

private def rawEncodingAtomsLive (table : RawEncodingTable) : Bool :=
  table.all fun entry => rawEncodingKeyAtomsLive entry.key

private def observedAtomsLive (trace : ObservedTrace) : Bool :=
  (match trace.chunkDigest with
    | none => true
    | some call => call.orderedClaims.all live && live call.output) &&
  (match trace.nifsCall with
    | none => true
    | some call =>
        live call.running && call.fresh.all live && live call.proof &&
          optionAtomsLive call.outcome) &&
  (match trace.runningDigest with
    | none => true
    | some call => live call.running && live call.output) &&
  (match trace.advancedState with
    | none => true
    | some state => stateAtomsLive state) &&
  optionAtomsLive trace.verifierDigestRead &&
  optionAtomsLive trace.piCcsHeaderRead &&
  (match trace.nebulaDigest with
    | none => true
    | some call => live call.input && live call.output) &&
  (match trace.stateXOutHash with
    | none => true
    | some call => live call.outputDigest && live call.output)

private def outcomeAtomsLive : Outcome → Bool
  | .accepted next => stateAtomsLive next
  | .rejected _ => true

/-- Exact liveness needed to prevent boundary fallback from changing a hash
call reached by an accepted native execution. -/
def NativeCallBindingCheck (receipt : Receipt) (next : NativeState) : Bool :=
  let nativeHash := hashSemantics receipt
  let nativeStep := stepSemantics receipt
  let runningDigestLive :=
    match next.proof with
    | .initial => false
    | .active running _ => live (nativeStep.runningDigest running)
  let verifierInput : NativeHashMessage := .verifier receipt.context
  let verifierOutput := nativeHash.hash verifierInput
  let nebulaLive :=
    match next.nebula with
    | none => true
    | some lane => live (nativeHash.nebulaDigest lane)
  let stateInput : NativeHashMessage :=
    .stateOutput (XOut.preimage nativeHash receipt.mode receipt.context next)
  runningDigestLive &&
    (live verifierOutput && (nebulaLive && live (nativeHash.hash stateInput)))

def NativeCallBinding (receipt : Receipt) (next : NativeState) : Prop :=
  NativeCallBindingCheck receipt next = true

private def finishHashCandidate
    (receipt : Receipt)
    (nextRunning : Running) : Option NativeState :=
  if receipt.proof.nebulaOpen = receipt.input.nebulaOpen then
    if nativeSemanticCheck (stepSemantics receipt) receipt.mode nextRunning
        receipt.proof then
      some (nativeAdvancedState (stepSemantics receipt) receipt.mode
        receipt.prior nextRunning receipt.input receipt.proof)
    else
      none
  else
    none

private def foldedRunning? (receipt : Receipt) : Option Running :=
  if receipt.prior.pc ≠ 1 ∨ branchShape receipt.prior = false ∨
      receipt.input.nextLatest = [] then
    none
  else
    match receipt.prior.proof, receipt.proof.fold with
    | .initial, .noFold =>
        some receipt.emptyRunning
    | .active running latest, .recursive nifsProof =>
        (stepSemantics receipt).nifsVerify
          (Step.nifsContext (stepSemantics receipt) receipt.prior
            receipt.input) running latest nifsProof
    | _, _ => none

private def runningPhaseCallBindingCheck (receipt : Receipt) : Bool :=
  match foldedRunning? receipt with
  | none => true
  | some nextRunning =>
      live ((stepSemantics receipt).runningDigest nextRunning)

private def hashCandidate? (receipt : Receipt) : Option NativeState :=
  match foldedRunning? receipt with
  | none => none
  | some nextRunning => finishHashCandidate receipt nextRunning

private def hashPhaseCallBindingCheck (receipt : Receipt) : Bool :=
  match hashCandidate? receipt with
  | none => true
  | some next => NativeCallBindingCheck receipt next

/-- State-authority values are verifier inputs, not calls made by `verify_step`.
This check keeps lifecycle fallback from supplying a missing verifier value. -/
def StateAuthorityPrimitiveBindingCheck (receipt : Receipt) : Bool :=
  live (XOut.initialBoundary (hashSemantics receipt) receipt.context) &&
  live (XOut.publicTraceSeed (hashSemantics receipt) receipt.context) &&
  live (stepSemantics receipt).initialAccumulatorDigest &&
  match receipt.prior.proof with
  | .initial => true
  | .active running _ =>
      live ((stepSemantics receipt).runningDigest running)

/-- Materialized identifiers carried by the compact receipt. -/
private def receiptAtomsLive (receipt : Receipt) : Bool :=
  live receipt.context.params &&
  live receipt.context.structureDigest &&
  live receipt.context.piCcsHeader &&
  live receipt.context.initialSemanticState &&
  live receipt.vkFsDigest &&
  decide (receipt.relationColumns ≠ 0) &&
  rawEncodingAtomsLive receipt.rawEncoding &&
  live receipt.emptyRunning &&
  live receipt.initialAccumulatorDigest &&
  live receipt.initialBoundaryDigest &&
  live receipt.publicTraceSeed &&
  runningDigestTableAtomsLive receipt.runningDigests &&
  stateAtomsLive receipt.prior &&
  inputAtomsLive receipt.input &&
  proofAtomsLive receipt.proof &&
  observedAtomsLive receipt.observed &&
  outcomeAtomsLive receipt.outcome &&
  receipt.calls.all callAtomsLive

private def receiptCountersFit (receipt : Receipt) : Bool :=
  match (run receipt).outcome with
  | .accepted _ | .rejected .statelessSemanticInvariantViolated |
      .rejected .xOutMismatch =>
      countersFit receipt.prior receipt.input
  | _ => true

/-- Receipt validity includes bounded counters, materialized identifiers, and
exact ordered call conservation. -/
private def receiptShapeWellFormed (receipt : Receipt) : Bool :=
  receiptAtomsLive receipt &&
    (controlFlowAndCallConservationCheck receipt &&
      receiptCountersFit receipt)

def ReceiptWellFormed (receipt : Receipt) : Bool :=
  receiptShapeWellFormed receipt &&
    (StateAuthorityPrimitiveBindingCheck receipt &&
      (runningPhaseCallBindingCheck receipt &&
        hashPhaseCallBindingCheck receipt))

theorem stateAuthorityPrimitiveBinding_of_wellFormed
    (receipt : Receipt)
    (wellFormed : ReceiptWellFormed receipt = true) :
    StateAuthorityPrimitiveBindingCheck receipt = true := by
  simp only [ReceiptWellFormed, Bool.and_eq_true] at wellFormed
  exact wellFormed.2.1

theorem initialBoundary_live_of_wellFormed
    (receipt : Receipt)
    (wellFormed : ReceiptWellFormed receipt = true) :
    live (XOut.initialBoundary (hashSemantics receipt) receipt.context) = true := by
  have binding :=
    stateAuthorityPrimitiveBinding_of_wellFormed receipt wellFormed
  simp only [StateAuthorityPrimitiveBindingCheck, Bool.and_eq_true] at binding
  exact binding.1.1.1

theorem publicTraceSeed_live_of_wellFormed
    (receipt : Receipt)
    (wellFormed : ReceiptWellFormed receipt = true) :
    live (XOut.publicTraceSeed (hashSemantics receipt) receipt.context) = true := by
  have binding :=
    stateAuthorityPrimitiveBinding_of_wellFormed receipt wellFormed
  simp only [StateAuthorityPrimitiveBindingCheck, Bool.and_eq_true] at binding
  exact binding.1.1.2

theorem priorRunningDigest_live_of_wellFormed
    (receipt : Receipt)
    (running : Running)
    (latest : List Fresh)
    (wellFormed : ReceiptWellFormed receipt = true)
    (priorProof : receipt.prior.proof = .active running latest) :
    live ((stepSemantics receipt).runningDigest running) = true := by
  have binding :=
    stateAuthorityPrimitiveBinding_of_wellFormed receipt wellFormed
  simp only [StateAuthorityPrimitiveBindingCheck, Bool.and_eq_true] at binding
  simpa only [priorProof] using binding.2

theorem controlFlowAndCallConservation_of_wellFormed
    (receipt : Receipt)
    (wellFormed : ReceiptWellFormed receipt = true) :
    ControlFlowAndCallConservation receipt := by
  simp only [ReceiptWellFormed, Bool.and_eq_true] at wellFormed
  have shape : receiptShapeWellFormed receipt = true :=
    wellFormed.1
  simp only [receiptShapeWellFormed, Bool.and_eq_true] at shape
  have control :
      controlFlowAndCallConservationCheck receipt = true :=
    shape.2.1
  exact
    (controlFlowAndCallConservationCheck_eq_true_iff receipt).1 control

def nativeOutcome (receipt : Receipt) : Outcome :=
  match nativeVerifyStep (hashSemantics receipt) (stepSemantics receipt)
      receipt.mode receipt.context receipt.prior receipt.input receipt.proof with
  | .ok next => .accepted next
  | .error error => .rejected error

/-- The two normalized Lean control-flow replays agree with the recorded Rust
outcome.  Primitive results remain receipt-supplied oracles. -/
def OracleReplayConforms (receipt : Receipt) : Prop :=
  ReceiptWellFormed receipt = true ∧
  (run receipt).outcome = receipt.outcome ∧
  nativeOutcome receipt = receipt.outcome

def check (receipt : Receipt) : Bool :=
  ReceiptWellFormed receipt &&
    (decide ((run receipt).outcome = receipt.outcome) &&
      decide (nativeOutcome receipt = receipt.outcome))

theorem check_eq_true_iff_oracleReplayConforms (receipt : Receipt) :
    check receipt = true ↔ OracleReplayConforms receipt := by
  simp [check, OracleReplayConforms]

/-- Semantic acceptance of the source-level native step, independent of an
externally recorded outcome. -/
def NativeAccepted (receipt : Receipt) (next : NativeState) : Prop :=
  nativeVerifyStep (hashSemantics receipt) (stepSemantics receipt)
    receipt.mode receipt.context receipt.prior receipt.input receipt.proof =
      .ok next

theorem nativeAccepted_iff_nativeHolds (receipt : Receipt) (next : NativeState) :
    NativeAccepted receipt next ↔
      NativeHolds (hashSemantics receipt) (stepSemantics receipt)
        receipt.mode receipt.context receipt.prior next receipt.input
        receipt.proof := by
  exact nativeVerifyStep_eq_ok_iff (hashSemantics receipt)
    (stepSemantics receipt) receipt.mode receipt.context receipt.prior next
    receipt.input receipt.proof

theorem check_and_recordedAccepted_implies_nativeAccepted
    (receipt : Receipt)
    (next : NativeState)
    (checked : check receipt = true)
    (recorded : receipt.outcome = .accepted next) :
    NativeAccepted receipt next := by
  have replay := (check_eq_true_iff_oracleReplayConforms receipt).1 checked
  have nativeRecorded := replay.2.2
  rw [recorded] at nativeRecorded
  unfold nativeOutcome at nativeRecorded
  cases nativeResult :
      nativeVerifyStep (hashSemantics receipt) (stepSemantics receipt)
        receipt.mode receipt.context receipt.prior receipt.input receipt.proof with
  | error error =>
      simp [nativeResult] at nativeRecorded
  | ok actual =>
      have actualEq : actual = next := by
        simpa [nativeResult] using nativeRecorded
      simpa [NativeAccepted, actualEq] using nativeResult

private theorem foldedRunning_eq_some_of_shape
    (receipt : Receipt)
    (nextRunning : Running)
    (entry : NativeEntryShape receipt.prior)
    (nextNonempty : receipt.input.nextLatest ≠ [])
    (folded :
      NativeFoldedTo (stepSemantics receipt) receipt.prior receipt.input
        receipt.proof nextRunning) :
    foldedRunning? receipt = some nextRunning := by
  rcases entry with ⟨pc, shape⟩
  have notEarly :
      ¬(receipt.prior.pc ≠ 1 ∨ branchShape receipt.prior = false ∨
        receipt.input.nextLatest = []) := by
    simp [pc, shape, nextNonempty]
  simp only [foldedRunning?, notEarly, ↓reduceIte]
  cases priorProof : receipt.prior.proof with
  | initial =>
      cases foldProof : receipt.proof.fold with
      | noFold =>
          have runningEq : nextRunning = receipt.emptyRunning := by
            simpa [NativeFoldedTo, stepSemantics, priorProof, foldProof]
              using folded
          simp [runningEq]
      | recursive nifsProof =>
          simp [NativeFoldedTo, priorProof, foldProof] at folded
  | active running latest =>
      cases foldProof : receipt.proof.fold with
      | noFold =>
          simp [NativeFoldedTo, priorProof, foldProof] at folded
      | recursive nifsProof =>
          have nifs :
              (stepSemantics receipt).nifsVerify
                  (Step.nifsContext (stepSemantics receipt) receipt.prior
                    receipt.input) running latest nifsProof =
                some nextRunning := by
            simpa [NativeFoldedTo, priorProof, foldProof] using folded
          exact nifs

/-- A reached folded output has a live native running digest in every
well-formed receipt, before any lifecycle semantic fallback is considered. -/
theorem outputRunningDigest_live_of_wellFormed_and_folded
    (receipt : Receipt)
    (nextRunning : Running)
    (wellFormed : ReceiptWellFormed receipt = true)
    (entry : NativeEntryShape receipt.prior)
    (nextNonempty : receipt.input.nextLatest ≠ [])
    (folded :
      NativeFoldedTo (stepSemantics receipt) receipt.prior receipt.input
        receipt.proof nextRunning) :
    live ((stepSemantics receipt).runningDigest nextRunning) = true := by
  have runningCheck : runningPhaseCallBindingCheck receipt = true := by
    simp only [ReceiptWellFormed, Bool.and_eq_true] at wellFormed
    exact wellFormed.2.2.1
  have candidate :=
    foldedRunning_eq_some_of_shape receipt nextRunning entry nextNonempty
      folded
  simpa [runningPhaseCallBindingCheck, candidate] using runningCheck

private theorem hashCandidate_eq_some_of_nativeHolds
    (receipt : Receipt)
    (hash :
      XOut.Semantics Params StructureDigest Header Digest Nebula NebulaDigest)
    (next : NativeState)
    (native :
      NativeHolds hash (stepSemantics receipt) receipt.mode receipt.context
        receipt.prior next receipt.input receipt.proof) :
    hashCandidate? receipt = some next := by
  rcases native with
    ⟨entry, _authority, nextNonempty, nextRunning, folded, openPayload, semantic,
      nextState, xOut⟩
  have runningCandidate :=
    foldedRunning_eq_some_of_shape receipt nextRunning entry nextNonempty
      folded
  simp [hashCandidate?, runningCandidate, finishHashCandidate, openPayload,
    semantic, nextState]

theorem nativeCallBinding_of_wellFormed_and_accepted
    (receipt : Receipt)
    (next : NativeState)
    (wellFormed : ReceiptWellFormed receipt = true)
    (accepted : NativeAccepted receipt next) :
    NativeCallBinding receipt next := by
  have native :=
    (nativeAccepted_iff_nativeHolds receipt next).1 accepted
  have bindingCheck :
      hashPhaseCallBindingCheck receipt = true :=
    by
      simp only [ReceiptWellFormed, Bool.and_eq_true] at wellFormed
      exact wellFormed.2.2.2
  have candidate :=
    hashCandidate_eq_some_of_nativeHolds receipt (hashSemantics receipt) next
      native
  simpa [hashPhaseCallBindingCheck, candidate, NativeCallBinding] using
    bindingCheck

theorem nativeCallBinding_of_wellFormed_and_nativeHolds
    (receipt : Receipt)
    (hash :
      XOut.Semantics Params StructureDigest Header Digest Nebula NebulaDigest)
    (next : NativeState)
    (wellFormed : ReceiptWellFormed receipt = true)
    (native :
      NativeHolds hash (stepSemantics receipt) receipt.mode receipt.context
        receipt.prior next receipt.input receipt.proof) :
    NativeCallBinding receipt next := by
  have bindingCheck :
      hashPhaseCallBindingCheck receipt = true :=
    by
      simp only [ReceiptWellFormed, Bool.and_eq_true] at wellFormed
      exact wellFormed.2.2.2
  have candidate :=
    hashCandidate_eq_some_of_nativeHolds receipt hash next native
  simpa [hashPhaseCallBindingCheck, candidate, NativeCallBinding] using
    bindingCheck


end Nightstream.Implementation.Rust.CanonicalConformance.NativeStep
