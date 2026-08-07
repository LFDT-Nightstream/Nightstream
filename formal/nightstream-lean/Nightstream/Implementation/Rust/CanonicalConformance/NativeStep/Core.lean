import Nightstream.Protocol.FPrime.Step

/-!
Contract: receipt-free executable semantics for public native F' verification.

Owns:
- compact typed atoms used by native-step differential cases;
- exact source-order base/recursive control flow for
  `paper::construction2::verify_step`;
- entry-state authority over verifier-owned seeds and the carried accumulator;
- a logical success predicate and its executable iff theorem.

Does not own:
- Rust serialization, generated cases, or Rust-source refinement;
- receipt lookup, call conservation, lifecycle replay, or terminal checks;
- application `Machine.step`, prior fresh-link authority, stateful application
  authenticity, or Nebula-relation verification;
- counter-overflow and deferred/unmaterialized-running behavior.

Emits constraints: no.

| Stage path | Obligation | Authority class | Owner |
|---|---|---|---|
| `fprime.native.entry` | pc, branch shape, verifier seeds, and carried accumulator authority | checked | `nativeVerifyStep` |
| `fprime.native.fold` | exact Initial/NoFold or Active/Recursive dispatch | checked | `NativeFoldedTo` |
| `fprime.native.advance` | verifier-computed next carrier | computed | `nativeAdvancedState` |
| `fprime.native.semantic` | stateless accumulator/semantic equality | checked | `NativeSemanticBound` |
| `fprime.native.output` | recomputed `x_out` equality | checked | `finishNative` |
| `fprime.native.exact` | executable success iff typed obligations | derived | `nativeVerifyStep_eq_ok_iff` |
-/

namespace Nightstream.Implementation.Rust.CanonicalConformance.NativeStep

open Nightstream.HyperNova.Construction2
open Nightstream.Protocol.FPrime

inductive AtomSort where
  | params
  | structureDigest
  | header
  | digest
  | running
  | fresh
  | nifsProof
  | nebula
  | nebulaDigest
  | nebulaOpen
deriving Repr, DecidableEq

/-- A normalized equality identifier.  Zero is reserved as failed lookup. -/
structure Atom (sort : AtomSort) where
  value : Nat
deriving Repr, DecidableEq

abbrev Params := Atom .params
abbrev StructureDigest := Atom .structureDigest
abbrev Header := Atom .header
abbrev Digest := Atom .digest
/-- Every value of this sort denotes a materialized running accumulator. -/
abbrev Running := Atom .running
abbrev Fresh := Atom .fresh
abbrev NifsProof := Atom .nifsProof
abbrev Nebula := Atom .nebula
abbrev NebulaDigest := Atom .nebulaDigest
abbrev NebulaOpen := Atom .nebulaOpen

abbrev NativeState := State Digest Running Fresh Nebula
abbrev NativeInput := Step.Input Fresh Nebula NebulaOpen
abbrev NativeProof := Step.Proof Digest NifsProof NebulaOpen
abbrev NativeContext := XOut.Context Params StructureDigest Header Digest

def poison (sort : AtomSort) : Atom sort :=
  ⟨0⟩

def live {sort : AtomSort} (value : Atom sort) : Bool :=
  decide (value.value ≠ 0)

/-- Stable source-level rejection classes used by differential cases. -/
inductive Error where
  | pcOutOfRange
  | baseCaseMismatch
  | stateAuthorityMismatch
  | emptyStep
  | foldProofVariantMismatch
  | nifsRejected
  | nebulaOpenMismatch
  | statelessSemanticInvariantViolated
  | xOutMismatch
deriving Repr, DecidableEq

def branchShape (prior : NativeState) : Bool :=
  match prior.proof with
  | .initial =>
      decide (prior.chunkCount = 0 ∧ prior.stepCount = 0 ∧
        prior.z0 = prior.zi)
  | .active _ _ =>
      decide (prior.chunkCount ≠ 0 ∧ prior.stepCount ≠ 0)

/-- Entry authority checked by Rust before it consumes the new batch.
All compact `Running` atoms are materialized, so an active accumulator digest
must be recomputed from the carried running value. -/
def NativeStateAuthority
    (hash :
      XOut.Semantics Params StructureDigest Header Digest Nebula NebulaDigest)
    (semantics :
      Step.Semantics Digest Running Fresh NifsProof Nebula NebulaOpen)
    (mode : XOut.Mode)
    (context : NativeContext)
    (prior : NativeState) : Prop :=
  prior.z0 = XOut.initialBoundary hash context ∧
  prior.initialSemanticState = context.initialSemanticState ∧
  match prior.proof with
  | .initial =>
      prior.accumulatorDigest = semantics.initialAccumulatorDigest ∧
      prior.semanticState = prior.initialSemanticState ∧
      prior.publicTrace = XOut.publicTraceSeed hash context
  | .active running _ =>
      prior.accumulatorDigest = semantics.runningDigest running ∧
      match mode with
      | .stateless => prior.semanticState = prior.accumulatorDigest
      | .stateful => True

def nativeStateAuthorityCheck
    (hash :
      XOut.Semantics Params StructureDigest Header Digest Nebula NebulaDigest)
    (semantics :
      Step.Semantics Digest Running Fresh NifsProof Nebula NebulaOpen)
    (mode : XOut.Mode)
    (context : NativeContext)
    (prior : NativeState) : Bool :=
  decide
      (prior.z0 = XOut.initialBoundary hash context ∧
       prior.initialSemanticState = context.initialSemanticState) &&
  match prior.proof with
  | .initial =>
      decide
        (prior.accumulatorDigest =
            semantics.initialAccumulatorDigest ∧
         prior.semanticState = prior.initialSemanticState ∧
         prior.publicTrace = XOut.publicTraceSeed hash context)
  | .active running _ =>
      decide (prior.accumulatorDigest = semantics.runningDigest running) &&
      match mode with
      | .stateless => decide (prior.semanticState = prior.accumulatorDigest)
      | .stateful => true

theorem nativeStateAuthorityCheck_eq_true_iff
    (hash :
      XOut.Semantics Params StructureDigest Header Digest Nebula NebulaDigest)
    (semantics :
      Step.Semantics Digest Running Fresh NifsProof Nebula NebulaOpen)
    (mode : XOut.Mode)
    (context : NativeContext)
    (prior : NativeState) :
    nativeStateAuthorityCheck hash semantics mode context prior = true ↔
      NativeStateAuthority hash semantics mode context prior := by
  cases proofCase : prior.proof <;> cases mode <;>
    simp [nativeStateAuthorityCheck, NativeStateAuthority, proofCase, and_assoc]

/-- State advance performed by the public native verifier.  In stateless mode
the semantic lane is verifier-derived, not copied from the proof. -/
def nativeAdvancedState
    (semantics :
      Step.Semantics Digest Running Fresh NifsProof Nebula NebulaOpen)
    (mode : XOut.Mode)
    (prior : NativeState)
    (nextRunning : Running)
    (input : NativeInput)
    (proof : NativeProof) : NativeState where
  chunkCount := prior.chunkCount + 1
  stepCount := prior.stepCount + input.nextLatest.length
  z0 := prior.z0
  zi := semantics.chunkDigest prior.stepCount input.nextLatest
  initialSemanticState := prior.initialSemanticState
  semanticState := match mode with
    | .stateless => semantics.runningDigest nextRunning
    | .stateful => proof.semanticStateDigest
  pc := prior.pc
  accumulatorDigest := semantics.runningDigest nextRunning
  publicTrace := semantics.chunkDigest prior.stepCount input.nextLatest
  proof := .active nextRunning input.nextLatest
  nebula := Step.installedNebula prior input

/-- The source-level base/active entry shape, excluding lifecycle authority. -/
def NativeEntryShape (prior : NativeState) : Prop :=
  prior.pc = 1 ∧ branchShape prior = true

/-- The native branch selection and verifier-computed running output. -/
def NativeFoldedTo
    (semantics :
      Step.Semantics Digest Running Fresh NifsProof Nebula NebulaOpen)
    (prior : NativeState)
    (input : NativeInput)
    (proof : NativeProof)
    (nextRunning : Running) : Prop :=
  match prior.proof, proof.fold with
  | .initial, .noFold => nextRunning = semantics.emptyRunning
  | .active running latest, .recursive nifsProof =>
      semantics.nifsVerify (Step.nifsContext semantics prior input)
        running latest nifsProof = some nextRunning
  | _, _ => False

/-- In stateless mode the public native call derives the semantic lane from
the folded accumulator.  Stateful application authenticity is external. -/
def NativeSemanticBound
    (semantics :
      Step.Semantics Digest Running Fresh NifsProof Nebula NebulaOpen)
    (mode : XOut.Mode)
    (nextRunning : Running)
    (proof : NativeProof) : Prop :=
  match mode with
  | .stateless =>
      proof.semanticStateDigest = semantics.runningDigest nextRunning
  | .stateful => True

def nativeSemanticCheck
    (semantics :
      Step.Semantics Digest Running Fresh NifsProof Nebula NebulaOpen)
    (mode : XOut.Mode)
    (nextRunning : Running)
    (proof : NativeProof) : Bool :=
  match mode with
  | .stateless =>
      decide
        (proof.semanticStateDigest = semantics.runningDigest nextRunning)
  | .stateful => true

theorem nativeSemanticCheck_eq_true_iff
    (semantics :
      Step.Semantics Digest Running Fresh NifsProof Nebula NebulaOpen)
    (mode : XOut.Mode)
    (nextRunning : Running)
    (proof : NativeProof) :
    nativeSemanticCheck semantics mode nextRunning proof = true ↔
      NativeSemanticBound semantics mode nextRunning proof := by
  cases mode <;> simp [nativeSemanticCheck, NativeSemanticBound]

/-- Receipt-free logical characterization of public native-step success. -/
def NativeHolds
    (hash :
      XOut.Semantics Params StructureDigest Header Digest Nebula NebulaDigest)
    (semantics :
      Step.Semantics Digest Running Fresh NifsProof Nebula NebulaOpen)
    (mode : XOut.Mode)
    (context : NativeContext)
    (prior next : NativeState)
    (input : NativeInput)
    (proof : NativeProof) : Prop :=
  NativeEntryShape prior ∧
  NativeStateAuthority hash semantics mode context prior ∧
  input.nextLatest ≠ [] ∧
  ∃ nextRunning,
    NativeFoldedTo semantics prior input proof nextRunning ∧
    proof.nebulaOpen = input.nebulaOpen ∧
    nativeSemanticCheck semantics mode nextRunning proof = true ∧
    nativeAdvancedState semantics mode prior nextRunning input proof = next ∧
    proof.xOut = XOut.compute hash mode context
      (nativeAdvancedState semantics mode prior nextRunning input proof)

private def finishNative
    (hash :
      XOut.Semantics Params StructureDigest Header Digest Nebula NebulaDigest)
    (semantics :
      Step.Semantics Digest Running Fresh NifsProof Nebula NebulaOpen)
    (mode : XOut.Mode)
    (context : NativeContext)
    (prior : NativeState)
    (input : NativeInput)
    (proof : NativeProof)
    (nextRunning : Running) : Except Error NativeState :=
  if proof.nebulaOpen = input.nebulaOpen then
    let next :=
      nativeAdvancedState semantics mode prior nextRunning input proof
    if nativeSemanticCheck semantics mode nextRunning proof then
      if proof.xOut = XOut.compute hash mode context next then
        .ok next
      else
        .error .xOutMismatch
    else
      .error .statelessSemanticInvariantViolated
  else
    .error .nebulaOpenMismatch

private theorem finishNative_eq_ok_iff
    (hash :
      XOut.Semantics Params StructureDigest Header Digest Nebula NebulaDigest)
    (semantics :
      Step.Semantics Digest Running Fresh NifsProof Nebula NebulaOpen)
    (mode : XOut.Mode)
    (context : NativeContext)
    (prior next : NativeState)
    (input : NativeInput)
    (proof : NativeProof)
    (nextRunning : Running) :
    finishNative hash semantics mode context prior input proof nextRunning =
        .ok next ↔
      proof.nebulaOpen = input.nebulaOpen ∧
      nativeSemanticCheck semantics mode nextRunning proof = true ∧
      nativeAdvancedState semantics mode prior nextRunning input proof = next ∧
      proof.xOut = XOut.compute hash mode context
        (nativeAdvancedState semantics mode prior nextRunning input proof) := by
  by_cases openPayload : proof.nebulaOpen = input.nebulaOpen
  · cases semantic :
        nativeSemanticCheck semantics mode nextRunning proof with
    | false =>
        simp [finishNative, openPayload, semantic]
    | true =>
      by_cases xOut :
          proof.xOut = XOut.compute hash mode context
            (nativeAdvancedState semantics mode prior nextRunning input proof)
      · simp [finishNative, openPayload, semantic, xOut]
      · simp [finishNative, openPayload, semantic, xOut]
  · simp [finishNative, openPayload]

/-- Receipt-free executable semantics for the public Rust `verify_step` call.
Counter overflow and unmaterialized running values are excluded by the compact
receipt profile and therefore do not appear as branches here. -/
def nativeVerifyStep
    (hash :
      XOut.Semantics Params StructureDigest Header Digest Nebula NebulaDigest)
    (semantics :
      Step.Semantics Digest Running Fresh NifsProof Nebula NebulaOpen)
    (mode : XOut.Mode)
    (context : NativeContext)
    (prior : NativeState)
    (input : NativeInput)
    (proof : NativeProof) : Except Error NativeState :=
  if prior.pc ≠ 1 then
    .error .pcOutOfRange
  else if branchShape prior = false then
    .error .baseCaseMismatch
  else if nativeStateAuthorityCheck hash semantics mode context prior = false then
    .error .stateAuthorityMismatch
  else if input.nextLatest = [] then
    .error .emptyStep
  else
    match prior.proof, proof.fold with
    | .initial, .noFold =>
        finishNative hash semantics mode context prior input proof
          semantics.emptyRunning
    | .active running latest, .recursive nifsProof =>
        match semantics.nifsVerify (Step.nifsContext semantics prior input)
            running latest nifsProof with
        | none => .error .nifsRejected
        | some nextRunning =>
            finishNative hash semantics mode context prior input proof nextRunning
    | _, _ => .error .foldProofVariantMismatch

/-- Exact success theorem for the receipt-free native checker. -/
theorem nativeVerifyStep_eq_ok_iff
    (hash :
      XOut.Semantics Params StructureDigest Header Digest Nebula NebulaDigest)
    (semantics :
      Step.Semantics Digest Running Fresh NifsProof Nebula NebulaOpen)
    (mode : XOut.Mode)
    (context : NativeContext)
    (prior next : NativeState)
    (input : NativeInput)
    (proof : NativeProof) :
    nativeVerifyStep hash semantics mode context prior input proof = .ok next ↔
      NativeHolds hash semantics mode context prior next input proof := by
  unfold nativeVerifyStep NativeHolds NativeEntryShape
  by_cases pc : prior.pc = 1
  · have pcNotNe : ¬ prior.pc ≠ 1 := by simp [pc]
    simp only [pc, true_and]
    cases shape : branchShape prior with
    | false => simp
    | true =>
        simp only [true_and]
        cases authority :
            nativeStateAuthorityCheck hash semantics mode context prior with
        | false =>
            have notAuthority :
                ¬ NativeStateAuthority hash semantics mode context prior := by
              intro holds
              have checkTrue :=
                (nativeStateAuthorityCheck_eq_true_iff
                  hash semantics mode context prior).2 holds
              simp [authority] at checkTrue
            simp [notAuthority]
        | true =>
            have authorityHolds :
                NativeStateAuthority hash semantics mode context prior := by
              exact
                (nativeStateAuthorityCheck_eq_true_iff
                  hash semantics mode context prior).1 authority
            simp only [authorityHolds, true_and]
            by_cases empty : input.nextLatest = []
            · simp [empty]
            · simp only [empty, ↓reduceIte]
              cases priorProof : prior.proof with
              | initial =>
                  cases foldProof : proof.fold with
                  | noFold =>
                      simpa [NativeFoldedTo, priorProof, foldProof, empty] using
                        finishNative_eq_ok_iff hash semantics mode context prior
                          next input proof semantics.emptyRunning
                  | recursive nifsProof =>
                      simp [NativeFoldedTo, priorProof, foldProof, empty]
              | active running latest =>
                  cases foldProof : proof.fold with
                  | noFold =>
                      simp [NativeFoldedTo, priorProof, foldProof, empty]
                  | recursive nifsProof =>
                      cases nifsResult :
                          semantics.nifsVerify
                            (Step.nifsContext semantics prior input)
                            running latest nifsProof with
                      | none =>
                          simp [NativeFoldedTo, priorProof, foldProof,
                            nifsResult, empty]
                      | some nextRunning =>
                          simpa [NativeFoldedTo, priorProof, foldProof,
                            nifsResult, empty]
                            using
                              finishNative_eq_ok_iff hash semantics mode context
                                prior next input proof nextRunning
  · have pcNe : prior.pc ≠ 1 := pc
    simp [pc]

end Nightstream.Implementation.Rust.CanonicalConformance.NativeStep
