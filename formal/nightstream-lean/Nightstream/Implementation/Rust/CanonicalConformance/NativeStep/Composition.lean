import Nightstream.Implementation.Rust.CanonicalConformance.NativeStep

/-!
Contract: lifecycle composition for the public native F' step.

Owns:
- transfer from live native primitive results to lifecycle semantics;
- the exact lifecycle obligations omitted by the public native call;
- equivalence between native acceptance plus those obligations and
  `Step.LocalHolds`.

Does not own:
- receipt execution, lookup, conservation, or well-formedness;
- generated Rust cases or Rust-source conformance;
- the outgoing one-fold-delayed fresh link or terminal closure.

Emits constraints: no.

| Stage path | Obligation | Authority class | Owner |
|---|---|---|---|
| `fprime.native.compose.hash` | live native hash calls survive boundary fallback | direct dataflow | `xOut_boundary_eq_native_of_callBinding` |
| `fprime.native.compose.entry` | lifecycle reconstructs the full base/active entry state | checked | `EntryAuthority` |
| `fprime.native.compose.incoming` | lifecycle binds the previous fresh batch | checked | `IncomingPriorLinked` |
| `fprime.native.compose.application` | stateful application and Nebula relations hold | checked | `StatefulSemanticBound`, `NebulaAdvanceBound` |
| `fprime.native.compose.exact` | native acceptance plus explicit boundaries iff local paper step | derived | `nativeAccepted_with_boundaries_iff_localHolds` |
-/

namespace Nightstream.Implementation.Rust.CanonicalConformance.NativeStep

open Nightstream.HyperNova.Construction2
open Nightstream.Protocol.FPrime

private theorem xOut_boundary_eq_native_of_liveness
    (receipt : Receipt)
    (boundary : BoundaryReceipt)
    (next : NativeState)
    (verifierLive :
      live ((hashSemantics receipt).hash (.verifier receipt.context)) = true)
    (nebulaEq :
      next.nebula.map (boundaryHashSemantics receipt boundary).nebulaDigest =
        next.nebula.map (hashSemantics receipt).nebulaDigest)
    (stateLive :
      live ((hashSemantics receipt).hash
        (.stateOutput (XOut.preimage (hashSemantics receipt) receipt.mode
          receipt.context next))) = true) :
    XOut.compute (boundaryHashSemantics receipt boundary) receipt.mode
        receipt.context next =
      XOut.compute (hashSemantics receipt) receipt.mode receipt.context next := by
  have verifierEq :
      XOut.verifierDigest (boundaryHashSemantics receipt boundary)
          receipt.context =
        XOut.verifierDigest (hashSemantics receipt) receipt.context := by
    unfold XOut.verifierDigest
    exact boundaryHash_eq_native_of_live receipt boundary
      (.verifier receipt.context) verifierLive
  have preimageEq :
      XOut.preimage (boundaryHashSemantics receipt boundary) receipt.mode
          receipt.context next =
        XOut.preimage (hashSemantics receipt) receipt.mode receipt.context
          next := by
    cases receipt.mode <;>
      simp only [XOut.preimage, verifierEq, nebulaEq]
  unfold XOut.compute
  rw [preimageEq]
  exact boundaryHash_eq_native_of_live receipt boundary
    (.stateOutput (XOut.preimage (hashSemantics receipt) receipt.mode
      receipt.context next)) stateLive

theorem xOut_boundary_eq_native_of_callBinding
    (receipt : Receipt)
    (boundary : BoundaryReceipt)
    (next : NativeState)
    (binding : NativeCallBinding receipt next) :
    XOut.compute (boundaryHashSemantics receipt boundary) receipt.mode
        receipt.context next =
      XOut.compute (hashSemantics receipt) receipt.mode receipt.context next := by
  simp only [NativeCallBinding, NativeCallBindingCheck, Bool.and_eq_true] at binding
  have nebulaEq :
      next.nebula.map (boundaryHashSemantics receipt boundary).nebulaDigest =
        next.nebula.map (hashSemantics receipt).nebulaDigest := by
    cases nebula : next.nebula with
    | none =>
        rfl
    | some lane =>
        have nebulaLive :
            live ((hashSemantics receipt).nebulaDigest lane) = true := by
          simpa only [nebula] using binding.2.2.1
        exact congrArg some
          (boundaryNebulaDigest_eq_native_of_live receipt boundary lane
            nebulaLive)
  exact xOut_boundary_eq_native_of_liveness receipt boundary next binding.2.1
    nebulaEq binding.2.2.2

private theorem outputRunningDigest_live_of_callBinding
    (receipt : Receipt)
    (next : NativeState)
    (nextRunning : Running)
    (binding : NativeCallBinding receipt next)
    (nextProof :
      next.proof = .active nextRunning receipt.input.nextLatest) :
    live ((stepSemantics receipt).runningDigest nextRunning) = true := by
  simp only [NativeCallBinding, NativeCallBindingCheck, Bool.and_eq_true] at binding
  simpa only [nextProof] using binding.1

private theorem nativeStateAuthority_boundaryStep_iff_of_wellFormed
    (receipt : Receipt)
    (boundary : BoundaryReceipt)
    (hash :
      XOut.Semantics Params StructureDigest Header Digest Nebula NebulaDigest)
    (wellFormed : ReceiptWellFormed receipt = true) :
    NativeStateAuthority hash (boundaryStepSemantics receipt boundary)
        receipt.mode receipt.context receipt.prior ↔
      NativeStateAuthority hash (stepSemantics receipt) receipt.mode
        receipt.context receipt.prior := by
  cases priorProof : receipt.prior.proof with
  | initial =>
      simp [NativeStateAuthority, priorProof, boundaryStepSemantics,
        stepSemantics]
  | active running latest =>
      have runningLive :=
        priorRunningDigest_live_of_wellFormed receipt running latest wellFormed
          priorProof
      have runningEq :=
        boundaryRunningDigest_eq_native_of_live receipt boundary running
          runningLive
      cases receipt.mode <;>
        simp only [NativeStateAuthority, priorProof] <;>
        rw [runningEq]

private theorem nativeStateAuthority_boundary_iff_of_wellFormed
    (receipt : Receipt)
    (boundary : BoundaryReceipt)
    (wellFormed : ReceiptWellFormed receipt = true) :
    NativeStateAuthority (hashSemantics receipt) (stepSemantics receipt)
        receipt.mode receipt.context receipt.prior ↔
      NativeStateAuthority (boundaryHashSemantics receipt boundary)
        (boundaryStepSemantics receipt boundary) receipt.mode receipt.context
        receipt.prior := by
  have initialBoundaryEq :
      XOut.initialBoundary (boundaryHashSemantics receipt boundary)
          receipt.context =
        XOut.initialBoundary (hashSemantics receipt) receipt.context := by
    unfold XOut.initialBoundary
    exact boundaryHash_eq_native_of_live receipt boundary
      (.initialBoundary (XOut.initialBoundaryPreimage receipt.context))
      (by
        simpa only [XOut.initialBoundary] using
          initialBoundary_live_of_wellFormed receipt wellFormed)
  have publicTraceEq :
      XOut.publicTraceSeed (boundaryHashSemantics receipt boundary)
          receipt.context =
        XOut.publicTraceSeed (hashSemantics receipt) receipt.context := by
    unfold XOut.publicTraceSeed
    exact boundaryHash_eq_native_of_live receipt boundary
      (.publicTraceSeed {
        structureDigest := receipt.context.structureDigest
      })
      (by
        simpa only [XOut.publicTraceSeed] using
          publicTraceSeed_live_of_wellFormed receipt wellFormed)
  cases priorProof : receipt.prior.proof with
  | initial =>
      simp only [NativeStateAuthority, priorProof]
      rw [initialBoundaryEq, publicTraceEq]
      rfl
  | active running latest =>
      have runningLive :=
        priorRunningDigest_live_of_wellFormed receipt running latest wellFormed
          priorProof
      have runningEq :=
        boundaryRunningDigest_eq_native_of_live receipt boundary running
          runningLive
      cases receipt.mode <;>
        simp only [NativeStateAuthority, priorProof] <;>
        rw [initialBoundaryEq, runningEq]

private theorem nativeHolds_boundaryStep_implies_nativeStep_of_wellFormed
    (receipt : Receipt)
    (boundary : BoundaryReceipt)
    (hash :
      XOut.Semantics Params StructureDigest Header Digest Nebula NebulaDigest)
    (next : NativeState)
    (wellFormed : ReceiptWellFormed receipt = true)
    (native :
      NativeHolds hash (boundaryStepSemantics receipt boundary) receipt.mode
        receipt.context receipt.prior next receipt.input receipt.proof) :
    NativeHolds hash (stepSemantics receipt) receipt.mode receipt.context
      receipt.prior next receipt.input receipt.proof := by
  rcases native with
    ⟨entry, authority, nextNonempty, nextRunning, folded, openPayload, semantic,
      nextState, xOut⟩
  have authority' :
      NativeStateAuthority hash (stepSemantics receipt) receipt.mode
        receipt.context receipt.prior :=
    (nativeStateAuthority_boundaryStep_iff_of_wellFormed receipt boundary hash
      wellFormed).1 authority
  have folded' :
      NativeFoldedTo (stepSemantics receipt) receipt.prior receipt.input
        receipt.proof nextRunning := by
    simpa [boundaryStepSemantics, stepSemantics] using folded
  have runningLive :=
    outputRunningDigest_live_of_wellFormed_and_folded receipt nextRunning
      wellFormed entry nextNonempty folded'
  have runningEq :=
    boundaryRunningDigest_eq_native_of_live receipt boundary nextRunning
      runningLive
  have semantic' :
      nativeSemanticCheck (stepSemantics receipt) receipt.mode nextRunning
        receipt.proof = true := by
    simpa [nativeSemanticCheck, runningEq] using semantic
  have nextState' :
      nativeAdvancedState (stepSemantics receipt) receipt.mode receipt.prior
          nextRunning receipt.input receipt.proof = next := by
    simpa [nativeAdvancedState, runningEq] using nextState
  refine ⟨entry, authority', nextNonempty, nextRunning, folded', openPayload, semantic',
    nextState', ?_⟩
  rw [nextState']
  simpa [nextState] using xOut

private theorem nativeHolds_boundary_iff
    (receipt : Receipt)
    (boundary : BoundaryReceipt)
    (next : NativeState)
    (wellFormed : ReceiptWellFormed receipt = true)
    (binding : NativeCallBinding receipt next) :
    NativeHolds (hashSemantics receipt) (stepSemantics receipt)
        receipt.mode receipt.context receipt.prior next receipt.input
        receipt.proof ↔
      NativeHolds (boundaryHashSemantics receipt boundary)
        (boundaryStepSemantics receipt boundary) receipt.mode receipt.context
        receipt.prior next receipt.input receipt.proof := by
  have xOutEq :=
    xOut_boundary_eq_native_of_callBinding receipt boundary next binding
  constructor
  · rintro ⟨entry, authority, nextNonempty, nextRunning, folded, openPayload,
      semantic, nextState, xOut⟩
    have authority' :
        NativeStateAuthority (boundaryHashSemantics receipt boundary)
          (boundaryStepSemantics receipt boundary) receipt.mode receipt.context
          receipt.prior :=
      (nativeStateAuthority_boundary_iff_of_wellFormed receipt boundary
        wellFormed).1 authority
    have nextProof :
        next.proof = .active nextRunning receipt.input.nextLatest := by
      rw [← nextState]
      rfl
    have runningLive :=
      outputRunningDigest_live_of_callBinding receipt next nextRunning binding
        nextProof
    have runningEq :=
      boundaryRunningDigest_eq_native_of_live receipt boundary nextRunning
        runningLive
    have folded' :
        NativeFoldedTo (boundaryStepSemantics receipt boundary) receipt.prior
          receipt.input receipt.proof nextRunning := by
      simpa [boundaryStepSemantics, stepSemantics] using folded
    have semantic' :
        nativeSemanticCheck (boundaryStepSemantics receipt boundary)
          receipt.mode nextRunning receipt.proof = true := by
      simpa [nativeSemanticCheck, runningEq] using semantic
    have nextState' :
        nativeAdvancedState (boundaryStepSemantics receipt boundary)
            receipt.mode receipt.prior nextRunning receipt.input
            receipt.proof = next := by
      simpa [nativeAdvancedState, runningEq] using nextState
    refine ⟨entry, authority', nextNonempty, nextRunning, folded', openPayload, semantic',
      nextState', ?_⟩
    rw [nextState']
    calc
      receipt.proof.xOut =
          XOut.compute (hashSemantics receipt) receipt.mode receipt.context
            next := by simpa [nextState] using xOut
      _ = XOut.compute (boundaryHashSemantics receipt boundary) receipt.mode
            receipt.context next := xOutEq.symm
  · rintro ⟨entry, authority, nextNonempty, nextRunning, folded, openPayload,
      semantic, nextState, xOut⟩
    have authority' :
        NativeStateAuthority (hashSemantics receipt) (stepSemantics receipt)
          receipt.mode receipt.context receipt.prior :=
      (nativeStateAuthority_boundary_iff_of_wellFormed receipt boundary
        wellFormed).2 authority
    have nextProof :
        next.proof = .active nextRunning receipt.input.nextLatest := by
      rw [← nextState]
      rfl
    have runningLive :=
      outputRunningDigest_live_of_callBinding receipt next nextRunning binding
        nextProof
    have runningEq :=
      boundaryRunningDigest_eq_native_of_live receipt boundary nextRunning
        runningLive
    have folded' :
        NativeFoldedTo (stepSemantics receipt) receipt.prior receipt.input
          receipt.proof nextRunning := by
      simpa [boundaryStepSemantics, stepSemantics] using folded
    have semantic' :
        nativeSemanticCheck (stepSemantics receipt) receipt.mode nextRunning
          receipt.proof = true := by
      simpa [nativeSemanticCheck, runningEq] using semantic
    have nextState' :
        nativeAdvancedState (stepSemantics receipt) receipt.mode receipt.prior
            nextRunning receipt.input receipt.proof = next := by
      simpa [nativeAdvancedState, runningEq] using nextState
    refine ⟨entry, authority', nextNonempty, nextRunning, folded', openPayload, semantic',
      nextState', ?_⟩
    rw [nextState']
    calc
      receipt.proof.xOut =
          XOut.compute (boundaryHashSemantics receipt boundary) receipt.mode
            receipt.context next := by simpa [nextState] using xOut
      _ = XOut.compute (hashSemantics receipt) receipt.mode receipt.context
            next := xOutEq

/-- State authority supplied by lifecycle reconstruction, not `verify_step`. -/
def EntryAuthority
    {Params : Type}
    {StructureDigest : Type}
    {Header : Type}
    {Digest : Type}
    {Running : Type}
    {Fresh : Type}
    {NifsProof : Type}
    {Nebula : Type}
    {NebulaDigest : Type}
    {NebulaOpen : Type}
    (hash :
      XOut.Semantics Params StructureDigest Header Digest Nebula NebulaDigest)
    (semantics :
      Step.Semantics Digest Running Fresh NifsProof Nebula NebulaOpen)
    (mode : XOut.Mode)
    (context : XOut.Context Params StructureDigest Header Digest)
    (prior : State Digest Running Fresh Nebula) : Prop :=
  match prior.proof with
  | .initial => Step.InitialState hash semantics mode context prior
  | .active running latest =>
      Step.ActiveState hash semantics mode context prior running latest

private theorem nativeStateAuthority_of_entryAuthority
    (hash :
      XOut.Semantics Params StructureDigest Header Digest Nebula NebulaDigest)
    (semantics :
      Step.Semantics Digest Running Fresh NifsProof Nebula NebulaOpen)
    (mode : XOut.Mode)
    (context : NativeContext)
    (prior : NativeState)
    (entry : EntryAuthority hash semantics mode context prior) :
    NativeStateAuthority hash semantics mode context prior := by
  cases priorProof : prior.proof with
  | initial =>
      have initial : Step.InitialState hash semantics mode context prior := by
        simpa [EntryAuthority, priorProof] using entry
      rcases initial with
        ⟨_pc, _chunk, _step, _z, initialBoundary, publicTrace,
          initialSemantic, accumulator, _initialNebula, _proofState,
          semanticInitial⟩
      refine ⟨initialBoundary, initialSemantic, ?_⟩
      simp only [priorProof]
      exact ⟨accumulator, semanticInitial, publicTrace⟩
  | active running latest =>
      have active :
          Step.ActiveState hash semantics mode context prior running latest := by
        simpa [EntryAuthority, priorProof] using entry
      rcases active with
        ⟨_pc, _chunk, _step, _proofState, accumulator, pinned⟩
      refine ⟨pinned.initialBoundaryPinned,
        pinned.initialSemanticStatePinned, ?_⟩
      simp only [priorProof]
      refine ⟨accumulator, ?_⟩
      cases mode with
      | stateless =>
          exact pinned.statelessSemanticEqualsAccumulator rfl
      | stateful =>
          trivial

/-- The delayed prior fresh link is owned by the consumer/lifecycle replay. -/
def IncomingPriorLinked
    {Params : Type}
    {StructureDigest : Type}
    {Header : Type}
    {Digest : Type}
    {Running : Type}
    {Fresh : Type}
    {NifsProof : Type}
    {Nebula : Type}
    {NebulaDigest : Type}
    {NebulaOpen : Type}
    (hash :
      XOut.Semantics Params StructureDigest Header Digest Nebula NebulaDigest)
    (semantics :
      Step.Semantics Digest Running Fresh NifsProof Nebula NebulaOpen)
    (mode : XOut.Mode)
    (context : XOut.Context Params StructureDigest Header Digest)
    (prior : State Digest Running Fresh Nebula) : Prop :=
  match prior.proof with
  | .initial => True
  | .active _ latest =>
      latest ≠ [] ∧ Step.FreshLinked semantics.freshLink
        (XOut.compute hash mode context prior) latest

/-- Stateful application authenticity is external to the public native step. -/
def StatefulSemanticBound
    {Digest : Type}
    {Running : Type}
    {Fresh : Type}
    {NifsProof : Type}
    {Nebula : Type}
    {NebulaOpen : Type}
    (semantics :
      Step.Semantics Digest Running Fresh NifsProof Nebula NebulaOpen)
    (mode : XOut.Mode)
    (prior : State Digest Running Fresh Nebula)
    (input : Step.Input Fresh Nebula NebulaOpen)
    (proof : Step.Proof Digest NifsProof NebulaOpen) : Prop :=
  mode = .stateful →
    semantics.applicationStep prior.semanticState input.nextLatest
      proof.semanticStateDigest = true

/-- The native call owns open-payload equality; relation verification is
external and supplied separately. -/
def NebulaAdvanceBound
    {Digest : Type}
    {Running : Type}
    {Fresh : Type}
    {NifsProof : Type}
    {Nebula : Type}
    {NebulaOpen : Type}
    (semantics :
      Step.Semantics Digest Running Fresh NifsProof Nebula NebulaOpen)
    (prior : State Digest Running Fresh Nebula)
    (input : Step.Input Fresh Nebula NebulaOpen) : Prop :=
  semantics.nebulaVerify prior.nebula input.nebulaOpen
    (Step.installedNebula prior input) = true

private theorem nativeAdvancedState_eq_advancedState
    (semantics :
      Step.Semantics Digest Running Fresh NifsProof Nebula NebulaOpen)
    (mode : XOut.Mode)
    (prior : NativeState)
    (nextRunning : Running)
    (input : NativeInput)
    (proof : NativeProof)
    (semantic :
      nativeSemanticCheck semantics mode nextRunning proof = true) :
    nativeAdvancedState semantics mode prior nextRunning input proof =
      Step.advancedState semantics prior nextRunning input proof := by
  have bound :
      NativeSemanticBound semantics mode nextRunning proof :=
    (nativeSemanticCheck_eq_true_iff semantics mode nextRunning proof).1
      semantic
  cases mode with
  | stateless =>
      simp [NativeSemanticBound] at bound
      simp [nativeAdvancedState, Step.advancedState, bound]
  | stateful =>
      rfl

theorem nativeHolds_with_boundaries_iff_localHolds
    (hash :
      XOut.Semantics Params StructureDigest Header Digest Nebula NebulaDigest)
    (semantics :
      Step.Semantics Digest Running Fresh NifsProof Nebula NebulaOpen)
    (mode : XOut.Mode)
    (context : NativeContext)
    (prior next : NativeState)
    (input : NativeInput)
    (proof : NativeProof) :
    (NativeHolds hash semantics mode context prior next input proof ∧
      EntryAuthority hash semantics mode context prior ∧
      IncomingPriorLinked hash semantics mode context prior ∧
      StatefulSemanticBound semantics mode prior input proof ∧
      NebulaAdvanceBound semantics prior input) ↔
      Step.LocalHolds hash semantics mode context prior next input proof := by
  constructor
  · rintro ⟨native, entry, incoming, stateful, nebula⟩
    rcases native with
      ⟨entryShape, _nativeAuthority, nextNonempty, nextRunning, folded, openPayload,
        semantic, nextNative, xOut⟩
    have semanticBound :
        Step.SemanticAdvance semantics mode prior nextRunning input proof := by
      cases mode with
      | stateless =>
          exact (nativeSemanticCheck_eq_true_iff semantics .stateless
            nextRunning proof).1 semantic
      | stateful =>
          exact stateful rfl
    have nebulaBound : Step.NebulaAdvance semantics prior input proof :=
      ⟨openPayload, nebula⟩
    have advanceEq :=
      nativeAdvancedState_eq_advancedState semantics mode prior nextRunning
        input proof semantic
    have nextStep :
        next = Step.advancedState semantics prior nextRunning input proof :=
      nextNative.symm.trans advanceEq
    have xOutNext : proof.xOut = XOut.compute hash mode context next := by
      rw [← nextNative]
      exact xOut
    cases priorProof : prior.proof with
    | initial =>
        cases foldProof : proof.fold with
        | noFold =>
            have runningEq : nextRunning = semantics.emptyRunning := by
              simpa [NativeFoldedTo, priorProof, foldProof] using folded
            subst nextRunning
            have initial :
                Step.InitialState hash semantics mode context prior := by
              simpa [EntryAuthority, priorProof] using entry
            simp only [Step.LocalHolds, priorProof, foldProof]
            unfold Step.BaseLocalHolds
            exact ⟨initial, foldProof, nextNonempty, semanticBound, nebulaBound,
              nextStep, xOutNext⟩
        | recursive nifsProof =>
            simp [NativeFoldedTo, priorProof, foldProof] at folded
    | active running latest =>
        cases foldProof : proof.fold with
        | noFold =>
            simp [NativeFoldedTo, priorProof, foldProof] at folded
        | recursive nifsProof =>
            have nifs :
                semantics.nifsVerify (Step.nifsContext semantics prior input)
                  running latest nifsProof = some nextRunning := by
              simpa [NativeFoldedTo, priorProof, foldProof] using folded
            have active :
                Step.ActiveState hash semantics mode context prior running
                  latest := by
              simpa [EntryAuthority, priorProof] using entry
            have priorLinked :
                latest ≠ [] ∧ Step.FreshLinked semantics.freshLink
                  (XOut.compute hash mode context prior) latest := by
              simpa [IncomingPriorLinked, priorProof] using incoming
            simp only [Step.LocalHolds, priorProof, foldProof]
            unfold Step.RecursiveLocalHolds
            refine ⟨active, foldProof, priorLinked.1, priorLinked.2, ?_⟩
            rw [nifs]
            exact ⟨nextNonempty, semanticBound, nebulaBound, nextStep,
              xOutNext⟩
  · intro localHolds
    cases priorProof : prior.proof with
    | initial =>
        cases foldProof : proof.fold with
        | noFold =>
            have base :
                Step.BaseLocalHolds hash semantics mode context prior next input
                  proof := by
              simpa [Step.LocalHolds, priorProof, foldProof] using localHolds
            rcases base with
              ⟨initial, _, nextNonempty, semanticBound, nebulaBound, nextStep,
                xOutNext⟩
            have semantic :
                nativeSemanticCheck semantics mode semantics.emptyRunning
                    proof = true := by
              cases mode with
              | stateless =>
                  simpa [nativeSemanticCheck, Step.SemanticAdvance] using
                    semanticBound
              | stateful =>
                  rfl
            have advanceEq :=
              nativeAdvancedState_eq_advancedState semantics mode prior
                semantics.emptyRunning input proof semantic
            have nextNative :
                nativeAdvancedState semantics mode prior semantics.emptyRunning
                    input proof = next :=
              advanceEq.trans nextStep.symm
            have initialAuthority := initial
            have native :
                NativeHolds hash semantics mode context prior next input
                  proof := by
              refine ⟨?_,
                nativeStateAuthority_of_entryAuthority hash semantics mode
                  context prior (by
                    simpa [EntryAuthority, priorProof] using initialAuthority),
                nextNonempty, semantics.emptyRunning, ?_, ?_,
                semantic, nextNative, ?_⟩
              · rcases initial with
                  ⟨pc, chunk, step, z, initialBoundary, publicTrace,
                    initialSemantic, accumulator, initialNebula, proofState,
                    modeState⟩
                exact ⟨pc, by
                  simp [branchShape, priorProof, chunk, step, z]⟩
              · simp [NativeFoldedTo, priorProof, foldProof]
              · exact nebulaBound.1
              · rw [nextNative]
                exact xOutNext
            refine ⟨native, ?_, ?_, ?_, ?_⟩
            · simpa [EntryAuthority, priorProof] using initialAuthority
            · simp [IncomingPriorLinked, priorProof]
            · intro statefulMode
              simpa [Step.SemanticAdvance, statefulMode] using semanticBound
            · exact nebulaBound.2
        | recursive nifsProof =>
            simp [Step.LocalHolds, priorProof, foldProof] at localHolds
    | active running latest =>
        cases foldProof : proof.fold with
        | noFold =>
            simp [Step.LocalHolds, priorProof, foldProof] at localHolds
        | recursive nifsProof =>
            cases nifsResult :
                semantics.nifsVerify (Step.nifsContext semantics prior input)
                  running latest nifsProof with
            | none =>
                simp [Step.LocalHolds, Step.RecursiveLocalHolds, priorProof,
                  foldProof, nifsResult] at localHolds
            | some nextRunning =>
                have recursive :
                    Step.RecursiveLocalHolds hash semantics mode context prior
                      next input proof running latest nifsProof := by
                  simpa [Step.LocalHolds, priorProof, foldProof] using localHolds
                simp only [Step.RecursiveLocalHolds, nifsResult] at recursive
                rcases recursive with
                  ⟨active, _, latestNonempty, priorLinked, nextNonempty,
                    semanticBound, nebulaBound, nextStep, xOutNext⟩
                have semantic :
                    nativeSemanticCheck semantics mode nextRunning proof =
                      true := by
                  cases mode with
                  | stateless =>
                      simpa [nativeSemanticCheck, Step.SemanticAdvance] using
                        semanticBound
                  | stateful =>
                      rfl
                have advanceEq :=
                  nativeAdvancedState_eq_advancedState semantics mode prior
                    nextRunning input proof semantic
                have nextNative :
                    nativeAdvancedState semantics mode prior nextRunning input
                        proof = next :=
                  advanceEq.trans nextStep.symm
                have activeAuthority := active
                have native :
                    NativeHolds hash semantics mode context prior next input
                      proof := by
                  refine ⟨?_,
                    nativeStateAuthority_of_entryAuthority hash semantics mode
                      context prior (by
                        simpa [EntryAuthority, priorProof] using activeAuthority),
                    nextNonempty, nextRunning, ?_, nebulaBound.1,
                    semantic, nextNative, ?_⟩
                  · rcases active with
                      ⟨pc, chunk, step, proofState, accumulator, pinned⟩
                    exact ⟨pc, by
                      simp [branchShape, priorProof, chunk, step]⟩
                  · simpa [NativeFoldedTo, priorProof, foldProof] using
                      nifsResult
                  · rw [nextNative]
                    exact xOutNext
                refine ⟨native, ?_, ?_, ?_, ?_⟩
                · simpa [EntryAuthority, priorProof] using activeAuthority
                · simpa [IncomingPriorLinked, priorProof] using
                    And.intro latestNonempty priorLinked
                · intro statefulMode
                  simpa [Step.SemanticAdvance, statefulMode] using semanticBound
                · exact nebulaBound.2

theorem nativeAccepted_with_boundaries_iff_localHolds
    (receipt : Receipt)
    (boundary : BoundaryReceipt)
    (next : NativeState)
    (wellFormed : ReceiptWellFormed receipt = true) :
    (NativeAccepted receipt next ∧
      EntryAuthority (boundaryHashSemantics receipt boundary)
        (boundaryStepSemantics receipt boundary) receipt.mode receipt.context
        receipt.prior ∧
      IncomingPriorLinked (boundaryHashSemantics receipt boundary)
        (boundaryStepSemantics receipt boundary) receipt.mode receipt.context
        receipt.prior ∧
      StatefulSemanticBound (boundaryStepSemantics receipt boundary)
        receipt.mode receipt.prior receipt.input receipt.proof ∧
      NebulaAdvanceBound (boundaryStepSemantics receipt boundary)
        receipt.prior receipt.input) ↔
      Step.LocalHolds (boundaryHashSemantics receipt boundary)
        (boundaryStepSemantics receipt boundary) receipt.mode receipt.context
        receipt.prior next receipt.input receipt.proof := by
  constructor
  · rintro ⟨accepted, entry, incoming, stateful, nebula⟩
    have binding :=
      nativeCallBinding_of_wellFormed_and_accepted receipt next wellFormed
        accepted
    have native :=
      (nativeAccepted_iff_nativeHolds receipt next).1 accepted
    have boundaryNative :=
      (nativeHolds_boundary_iff receipt boundary next wellFormed binding).1
        native
    exact
      (nativeHolds_with_boundaries_iff_localHolds
        (boundaryHashSemantics receipt boundary)
        (boundaryStepSemantics receipt boundary) receipt.mode receipt.context
        receipt.prior next receipt.input receipt.proof).1
        ⟨boundaryNative, entry, incoming, stateful, nebula⟩
  · intro localHolds
    have composed :=
      (nativeHolds_with_boundaries_iff_localHolds
        (boundaryHashSemantics receipt boundary)
        (boundaryStepSemantics receipt boundary) receipt.mode receipt.context
        receipt.prior next receipt.input receipt.proof).2 localHolds
    rcases composed with
      ⟨boundaryNative, entry, incoming, stateful, nebula⟩
    have boundaryNativeStep :
        NativeHolds (boundaryHashSemantics receipt boundary)
          (stepSemantics receipt) receipt.mode receipt.context receipt.prior
          next receipt.input receipt.proof :=
      nativeHolds_boundaryStep_implies_nativeStep_of_wellFormed receipt
        boundary (boundaryHashSemantics receipt boundary) next wellFormed
        boundaryNative
    have binding :=
      nativeCallBinding_of_wellFormed_and_nativeHolds receipt
        (boundaryHashSemantics receipt boundary) next wellFormed
        boundaryNativeStep
    have native :=
      (nativeHolds_boundary_iff receipt boundary next wellFormed binding).2
        boundaryNative
    have accepted :=
      (nativeAccepted_iff_nativeHolds receipt next).2 native
    exact ⟨accepted, entry, incoming, stateful, nebula⟩

end Nightstream.Implementation.Rust.CanonicalConformance.NativeStep
