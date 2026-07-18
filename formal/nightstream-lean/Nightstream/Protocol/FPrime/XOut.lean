import Nightstream.HyperNova.Construction2.State

/-!
Contract: Construction-2 `x_out` authority and collision reduction.

This module mirrors the ownership of Rust `state_x_out_digest_with_mode`.
Directly absorbed coordinates appear in `XOutPreimage`. Coordinates Rust omits
for cost or fixed-point reasons are not silently treated as hashed: `z0`, the
initial semantic state, and `publicTrace` are carried by `StatePinned` through
verifier-derived/transitive or equality constraints. The running/latest proof
payload is not part of this preimage. Therefore F' step semantics must
separately bind the exact running payload to `accumulatorDigest`; merely
recomputing a non-injective parent-only handle is insufficient.

The hash is one domain-separated message function. Its message constructors
make verifier-key, initial-boundary, and state-output domains disjoint without
introducing a second hash family. A same-output disagreement reduces to an
explicit collision; no collision-resistance conclusion is assumed locally.

| Stage path | Mathematical obligation | Authority class | Local owner |
|---|---|---|---|
| `fprime.x_out.mode` | distinguish stateless and stateful semantic-state layouts | verifier-owned configuration | `Mode` |
| `fprime.x_out.verifier` | name the exact verifier material absorbed into the verifier digest | direct dataflow | `VerifierPreimage`, `verifierDigest` |
| `fprime.x_out.initial` | derive the initial boundary and public-trace seed from verifier-owned structure | computed | `InitialBoundaryPreimage`, `PublicTraceSeedPreimage`, `initialBoundary`, `publicTraceSeed` |
| `fprime.x_out.preimage` | assemble the exact compact state-output absorb sequence | computed | `XOutPreimage`, `preimage` |
| `fprime.x_out.domain` | separate verifier, initial-boundary, trace-seed, and state-output messages | checked by construction | `Message` |
| `fprime.x_out.hash` | compute the public output through the sole abstract hash interface | computed | `Semantics`, `compute` |
| `fprime.x_out.pinning` | retain omitted state coordinates through explicit verifier-derived equalities | checked | `StatePinned` |
| `fprime.x_out.authority` | project exactly the hashed and transitively pinned fields; this is not a binding claim for the omitted proof payload | computed | `AuthorityView`, `authorityView` |
| `fprime.x_out.collision` | name hash or Nebula-digest disagreement instead of assuming injectivity | security boundary | `HashCollision`, `NebulaDigestCollision`, `BindingFailure` |
| `fprime.x_out.reduction` | reduce equal outputs to equal authority views or an explicit binding failure | derived | `xOut_binding_or_collision` |

Maps to:
- `paper::digest::{vk_fs_digest, initial_boundary_digest,
  state_x_out_digest_with_mode}`
- `construction2::transition::compute_x_out`
- `paper::f_prime::digest_circuit::enforce_state_x_out_digest_*`
-/

namespace Nightstream.Protocol.FPrime.XOut

open Nightstream.HyperNova.Construction2

universe uDigest uParams uStructure uHeader uRunning uFresh uNebulaDigest

/-- Verifier-owned semantic-state layout. Stateless mode omits a duplicate lane. -/
inductive Mode where
  | stateless
  | stateful
deriving Repr, DecidableEq

/-- Raw verifier-owned material absorbed into `vk_fs_digest`. -/
structure VerifierPreimage
    (Params : Type uParams)
    (StructureDigest : Type uStructure)
    (Header : Type uHeader)
    (Digest : Type uDigest) where
  params : Params
  structureDigest : StructureDigest
  piCcsHeader : Header
  publicInputLength : Option Nat
  initialSemanticState : Digest
deriving Repr, DecidableEq

/-- Inputs from which Rust deterministically derives `z0`. -/
structure InitialBoundaryPreimage
    (StructureDigest : Type uStructure) where
  structureDigest : StructureDigest
  publicInputLength : Option Nat
deriving Repr, DecidableEq

/-- Verifier-derived seed for the initial public-trace coordinate. -/
structure PublicTraceSeedPreimage
    (StructureDigest : Type uStructure) where
  structureDigest : StructureDigest
deriving Repr, DecidableEq

/-- Exact typed shape of the compact `state_x_out` absorb sequence. -/
structure XOutPreimage
    (Digest : Type uDigest)
    (Header : Type uHeader)
    (NebulaDigest : Type uNebulaDigest) where
  vkFsDigest : Digest
  piCcsHeader : Header
  chunkCount : Nat
  stepCount : Nat
  pc : Nat
  currentBoundary : Digest
  /-- `none` is stateless; `some d` is the independently authenticated stateful lane. -/
  semanticState : Option Digest
  construction2Accumulator : Digest
  /-- Present-only marker and lane digest in the Rust encoding. -/
  nebula : Option NebulaDigest
deriving Repr, DecidableEq

/-- One Poseidon2 family with constructor-level domain separation. -/
inductive Message
    (Params : Type uParams)
    (StructureDigest : Type uStructure)
    (Header : Type uHeader)
    (Digest : Type uDigest)
    (NebulaDigest : Type uNebulaDigest) where
  | verifier (preimage : VerifierPreimage Params StructureDigest Header Digest)
  | initialBoundary (preimage : InitialBoundaryPreimage StructureDigest)
  | publicTraceSeed (preimage : PublicTraceSeedPreimage StructureDigest)
  | stateOutput (preimage : XOutPreimage Digest Header NebulaDigest)
deriving Repr, DecidableEq

/-- The only hash operation in the model; production instantiates it with Poseidon2. -/
structure Semantics
    (Params : Type uParams)
    (StructureDigest : Type uStructure)
    (Header : Type uHeader)
    (Digest : Type uDigest)
    (Nebula : Type)
    (NebulaDigest : Type uNebulaDigest) where
  hash : Message Params StructureDigest Header Digest NebulaDigest → Digest
  nebulaDigest : Nebula → NebulaDigest

abbrev Context
    (Params : Type uParams)
    (StructureDigest : Type uStructure)
    (Header : Type uHeader)
    (Digest : Type uDigest) :=
  VerifierPreimage Params StructureDigest Header Digest

def initialBoundaryPreimage
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Header : Type uHeader}
    {Digest : Type uDigest}
    (context : Context Params StructureDigest Header Digest) :
    InitialBoundaryPreimage StructureDigest where
  structureDigest := context.structureDigest
  publicInputLength := context.publicInputLength

def verifierDigest
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Header : Type uHeader}
    {Digest : Type uDigest}
    {Nebula : Type}
    {NebulaDigest : Type uNebulaDigest}
    (semantics : Semantics Params StructureDigest Header Digest Nebula NebulaDigest)
    (context : Context Params StructureDigest Header Digest) : Digest :=
  semantics.hash (.verifier context)

def initialBoundary
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Header : Type uHeader}
    {Digest : Type uDigest}
    {Nebula : Type}
    {NebulaDigest : Type uNebulaDigest}
    (semantics : Semantics Params StructureDigest Header Digest Nebula NebulaDigest)
    (context : Context Params StructureDigest Header Digest) : Digest :=
  semantics.hash (.initialBoundary (initialBoundaryPreimage context))

def publicTraceSeed
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Header : Type uHeader}
    {Digest : Type uDigest}
    {Nebula : Type}
    {NebulaDigest : Type uNebulaDigest}
    (semantics : Semantics Params StructureDigest Header Digest Nebula NebulaDigest)
    (context : Context Params StructureDigest Header Digest) : Digest :=
  semantics.hash (.publicTraceSeed {
    structureDigest := context.structureDigest
  })

def preimage
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Header : Type uHeader}
    {Digest : Type uDigest}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {Nebula : Type}
    {NebulaDigest : Type uNebulaDigest}
    (semantics : Semantics Params StructureDigest Header Digest Nebula NebulaDigest)
    (mode : Mode)
    (context : Context Params StructureDigest Header Digest)
    (state : State Digest Running Fresh Nebula) :
    XOutPreimage Digest Header NebulaDigest where
  vkFsDigest := verifierDigest semantics context
  piCcsHeader := context.piCcsHeader
  chunkCount := state.chunkCount
  stepCount := state.stepCount
  pc := state.pc
  currentBoundary := state.zi
  semanticState := match mode with
    | .stateless => none
    | .stateful => some state.semanticState
  construction2Accumulator := state.accumulatorDigest
  nebula := state.nebula.map semantics.nebulaDigest

/-- Two states with equal hash-visible fields have the same state-output
preimage. The proof payload and the fields owned by `StatePinned` are absent
because `preimage` does not read them. -/
theorem preimage_eq_of_visible_fields
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Header : Type uHeader}
    {Digest : Type uDigest}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {Nebula : Type}
    {NebulaDigest : Type uNebulaDigest}
    (semantics :
      Semantics Params StructureDigest Header Digest Nebula NebulaDigest)
    (mode : Mode)
    (context : Context Params StructureDigest Header Digest)
    (left right : State Digest Running Fresh Nebula)
    (chunkCount : left.chunkCount = right.chunkCount)
    (stepCount : left.stepCount = right.stepCount)
    (pc : left.pc = right.pc)
    (currentBoundary : left.zi = right.zi)
    (semanticState :
      (match mode with
        | .stateless => none
        | .stateful => some left.semanticState) =
      (match mode with
        | .stateless => none
        | .stateful => some right.semanticState))
    (accumulatorDigest :
      left.accumulatorDigest = right.accumulatorDigest)
    (nebulaDigest :
      left.nebula.map semantics.nebulaDigest =
        right.nebula.map semantics.nebulaDigest) :
    preimage semantics mode context left =
      preimage semantics mode context right := by
  rw [XOutPreimage.mk.injEq]
  exact ⟨rfl, rfl, chunkCount, stepCount, pc, currentBoundary,
    semanticState, accumulatorDigest, nebulaDigest⟩

def compute
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Header : Type uHeader}
    {Digest : Type uDigest}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {Nebula : Type}
    {NebulaDigest : Type uNebulaDigest}
    (semantics : Semantics Params StructureDigest Header Digest Nebula NebulaDigest)
    (mode : Mode)
    (context : Context Params StructureDigest Header Digest)
    (state : State Digest Running Fresh Nebula) : Digest :=
  semantics.hash (.stateOutput (preimage semantics mode context state))

/-- Obligations for coordinates deliberately omitted from the compact preimage. -/
structure StatePinned
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Header : Type uHeader}
    {Digest : Type uDigest}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {Nebula : Type}
    {NebulaDigest : Type uNebulaDigest}
    (semantics : Semantics Params StructureDigest Header Digest Nebula NebulaDigest)
    (mode : Mode)
    (context : Context Params StructureDigest Header Digest)
    (state : State Digest Running Fresh Nebula) : Prop where
  initialBoundaryPinned : state.z0 = initialBoundary semantics context
  initialSemanticStatePinned :
    state.initialSemanticState = context.initialSemanticState
  publicTraceMirrorsBoundary : state.publicTrace = state.zi
  statelessSemanticEqualsAccumulator :
    mode = .stateless → state.semanticState = state.accumulatorDigest

instance statePinnedDecidable
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Header : Type uHeader}
    {Digest : Type uDigest}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {Nebula : Type}
    {NebulaDigest : Type uNebulaDigest}
    [DecidableEq Digest]
    (semantics : Semantics Params StructureDigest Header Digest Nebula NebulaDigest)
    (mode : Mode)
    (context : Context Params StructureDigest Header Digest)
    (state : State Digest Running Fresh Nebula) :
    Decidable (StatePinned semantics mode context state) := by
  apply decidable_of_iff
    (state.z0 = initialBoundary semantics context ∧
      state.initialSemanticState = context.initialSemanticState ∧
      state.publicTrace = state.zi ∧
      (mode = .stateless → state.semanticState = state.accumulatorDigest))
  constructor
  · intro fields
    exact {
      initialBoundaryPinned := fields.1
      initialSemanticStatePinned := fields.2.1
      publicTraceMirrorsBoundary := fields.2.2.1
      statelessSemanticEqualsAccumulator := fields.2.2.2
    }
  · intro pinned
    exact ⟨pinned.initialBoundaryPinned, pinned.initialSemanticStatePinned,
      pinned.publicTraceMirrorsBoundary,
      pinned.statelessSemanticEqualsAccumulator⟩

/-- Canonical authority view recovered from a pinned state and verifier context. -/
structure AuthorityView
    (Params : Type uParams)
    (StructureDigest : Type uStructure)
    (Header : Type uHeader)
    (Digest : Type uDigest)
    (Nebula : Type)
    (NebulaDigest : Type uNebulaDigest) where
  verifier : VerifierPreimage Params StructureDigest Header Digest
  mode : Mode
  chunkCount : Nat
  stepCount : Nat
  initialBoundary : Digest
  currentBoundary : Digest
  initialSemanticState : Digest
  semanticState : Digest
  pc : Nat
  construction2Accumulator : Digest
  publicTrace : Digest
  nebulaDigest : Option NebulaDigest
  nebula : Option Nebula
deriving Repr, DecidableEq

namespace AuthorityView

theorem eq_of_fields
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Header : Type uHeader}
    {Digest : Type uDigest}
    {Nebula : Type}
    {NebulaDigest : Type uNebulaDigest}
    {left right : AuthorityView Params StructureDigest Header Digest Nebula NebulaDigest}
    (verifier : left.verifier = right.verifier)
    (mode : left.mode = right.mode)
    (chunkCount : left.chunkCount = right.chunkCount)
    (stepCount : left.stepCount = right.stepCount)
    (initialBoundary : left.initialBoundary = right.initialBoundary)
    (currentBoundary : left.currentBoundary = right.currentBoundary)
    (initialSemanticState :
      left.initialSemanticState = right.initialSemanticState)
    (semanticState : left.semanticState = right.semanticState)
    (pc : left.pc = right.pc)
    (construction2Accumulator :
      left.construction2Accumulator = right.construction2Accumulator)
    (publicTrace : left.publicTrace = right.publicTrace)
    (nebulaDigest : left.nebulaDigest = right.nebulaDigest)
    (nebula : left.nebula = right.nebula) :
    left = right := by
  cases left
  cases right
  simp_all

end AuthorityView

def authorityView
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Header : Type uHeader}
    {Digest : Type uDigest}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {Nebula : Type}
    {NebulaDigest : Type uNebulaDigest}
    (semantics : Semantics Params StructureDigest Header Digest Nebula NebulaDigest)
    (mode : Mode)
    (context : Context Params StructureDigest Header Digest)
    (state : State Digest Running Fresh Nebula) :
    AuthorityView Params StructureDigest Header Digest Nebula NebulaDigest where
  verifier := context
  mode := mode
  chunkCount := state.chunkCount
  stepCount := state.stepCount
  initialBoundary := state.z0
  currentBoundary := state.zi
  initialSemanticState := state.initialSemanticState
  semanticState := state.semanticState
  pc := state.pc
  construction2Accumulator := state.accumulatorDigest
  publicTrace := state.publicTrace
  nebulaDigest := state.nebula.map semantics.nebulaDigest
  nebula := state.nebula

/-- Collision event for the single domain-separated Poseidon2 message family. -/
def HashCollision
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Header : Type uHeader}
    {Digest : Type uDigest}
    {NebulaDigest : Type uNebulaDigest}
    (hash : Message Params StructureDigest Header Digest NebulaDigest → Digest) : Prop :=
  ∃ left right, left ≠ right ∧ hash left = hash right

/-- Collision event in the separately compressed Nebula memory lane. -/
def NebulaDigestCollision
    {Nebula : Type}
    {NebulaDigest : Type uNebulaDigest}
    (digest : Nebula → NebulaDigest) : Prop :=
  ∃ left right, left ≠ right ∧ digest left = digest right

/-- Every compression boundary whose failure can explain an `x_out` ambiguity. -/
def BindingFailure
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Header : Type uHeader}
    {Digest : Type uDigest}
    {Nebula : Type}
    {NebulaDigest : Type uNebulaDigest}
    (semantics : Semantics Params StructureDigest Header Digest Nebula NebulaDigest) : Prop :=
  HashCollision semantics.hash ∨ NebulaDigestCollision semantics.nebulaDigest

private theorem option_eq_or_digest_collision
    {Nebula : Type}
    {NebulaDigest : Type uNebulaDigest}
    (digest : Nebula → NebulaDigest)
    (left right : Option Nebula)
    (sameDigest : left.map digest = right.map digest) :
    left = right ∨ NebulaDigestCollision digest := by
  classical
  cases left with
  | none =>
      cases right with
      | none => exact Or.inl rfl
      | some right => simp at sameDigest
  | some left =>
      cases right with
      | none => simp at sameDigest
      | some right =>
          have digestEq : digest left = digest right := Option.some.inj sameDigest
          by_cases laneEq : left = right
          · exact Or.inl (congrArg some laneEq)
          · exact Or.inr ⟨left, right, laneEq, digestEq⟩

private theorem authority_eq_of_preimage_eq
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Header : Type uHeader}
    {Digest : Type uDigest}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {Nebula : Type}
    {NebulaDigest : Type uNebulaDigest}
    (semantics : Semantics Params StructureDigest Header Digest Nebula NebulaDigest)
    (leftMode rightMode : Mode)
    (context : Context Params StructureDigest Header Digest)
    (leftState rightState : State Digest Running Fresh Nebula)
    (leftPinned : StatePinned semantics leftMode context leftState)
    (rightPinned : StatePinned semantics rightMode context rightState)
    (samePreimage :
      preimage semantics leftMode context leftState =
        preimage semantics rightMode context rightState)
    (sameNebula : leftState.nebula = rightState.nebula) :
    authorityView semantics leftMode context leftState =
      authorityView semantics rightMode context rightState := by
  have chunkCountEq := congrArg XOutPreimage.chunkCount samePreimage
  have stepCountEq := congrArg XOutPreimage.stepCount samePreimage
  have pcEq := congrArg XOutPreimage.pc samePreimage
  have boundaryEq := congrArg XOutPreimage.currentBoundary samePreimage
  have semanticEq := congrArg XOutPreimage.semanticState samePreimage
  have accumulatorEq :=
    congrArg XOutPreimage.construction2Accumulator samePreimage
  have nebulaDigestEq := congrArg XOutPreimage.nebula samePreimage
  have initialBoundaryEq : leftState.z0 = rightState.z0 :=
    leftPinned.initialBoundaryPinned.trans
      rightPinned.initialBoundaryPinned.symm
  have initialSemanticEq :
      leftState.initialSemanticState = rightState.initialSemanticState :=
    leftPinned.initialSemanticStatePinned.trans
      rightPinned.initialSemanticStatePinned.symm
  have publicTraceEq : leftState.publicTrace = rightState.publicTrace :=
    leftPinned.publicTraceMirrorsBoundary.trans
      (boundaryEq.trans rightPinned.publicTraceMirrorsBoundary.symm)
  cases leftMode <;> cases rightMode
  · have stateSemanticEq : leftState.semanticState = rightState.semanticState :=
      (leftPinned.statelessSemanticEqualsAccumulator rfl).trans
        (accumulatorEq.trans
          (rightPinned.statelessSemanticEqualsAccumulator rfl).symm)
    exact AuthorityView.eq_of_fields rfl rfl chunkCountEq stepCountEq
      initialBoundaryEq boundaryEq initialSemanticEq stateSemanticEq pcEq
      accumulatorEq publicTraceEq nebulaDigestEq sameNebula
  · cases semanticEq
  · cases semanticEq
  · have stateSemanticEq : leftState.semanticState = rightState.semanticState :=
      Option.some.inj semanticEq
    exact AuthorityView.eq_of_fields rfl rfl chunkCountEq stepCountEq
      initialBoundaryEq boundaryEq initialSemanticEq stateSemanticEq pcEq
      accumulatorEq publicTraceEq nebulaDigestEq sameNebula

/--
`x_out` binding theorem. Equal outputs either recover the entire canonical
authority view—including transitively pinned fields—or exhibit a collision in
the exact domain-separated hash family.
-/
theorem xOut_binding_or_collision
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Header : Type uHeader}
    {Digest : Type uDigest}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {Nebula : Type}
    {NebulaDigest : Type uNebulaDigest}
    (semantics : Semantics Params StructureDigest Header Digest Nebula NebulaDigest)
    (leftMode rightMode : Mode)
    (leftContext rightContext : Context Params StructureDigest Header Digest)
    (leftState rightState : State Digest Running Fresh Nebula)
    (leftPinned : StatePinned semantics leftMode leftContext leftState)
    (rightPinned : StatePinned semantics rightMode rightContext rightState)
    (sameOutput :
      compute semantics leftMode leftContext leftState =
        compute semantics rightMode rightContext rightState) :
    authorityView semantics leftMode leftContext leftState =
        authorityView semantics rightMode rightContext rightState ∨
      BindingFailure semantics := by
  classical
  let leftMessage : Message Params StructureDigest Header Digest NebulaDigest :=
    .stateOutput (preimage semantics leftMode leftContext leftState)
  let rightMessage : Message Params StructureDigest Header Digest NebulaDigest :=
    .stateOutput (preimage semantics rightMode rightContext rightState)
  by_cases sameMessage : leftMessage = rightMessage
  · have samePreimage :
        preimage semantics leftMode leftContext leftState =
          preimage semantics rightMode rightContext rightState := by
      exact Message.stateOutput.inj sameMessage
    let leftVerifier : Message Params StructureDigest Header Digest NebulaDigest :=
      .verifier leftContext
    let rightVerifier : Message Params StructureDigest Header Digest NebulaDigest :=
      .verifier rightContext
    by_cases sameVerifier : leftVerifier = rightVerifier
    · have sameContext : leftContext = rightContext :=
        Message.verifier.inj sameVerifier
      subst rightContext
      have sameNebulaDigest :
          leftState.nebula.map semantics.nebulaDigest =
            rightState.nebula.map semantics.nebulaDigest := by
        exact congrArg XOutPreimage.nebula samePreimage
      rcases option_eq_or_digest_collision semantics.nebulaDigest
          leftState.nebula rightState.nebula sameNebulaDigest with
        sameNebula | nebulaCollision
      · left
        exact authority_eq_of_preimage_eq semantics leftMode rightMode leftContext
          leftState rightState leftPinned rightPinned samePreimage sameNebula
      · exact Or.inr (Or.inr nebulaCollision)
    · right
      left
      refine ⟨leftVerifier, rightVerifier, sameVerifier, ?_⟩
      have verifierDigestEq := congrArg XOutPreimage.vkFsDigest samePreimage
      exact verifierDigestEq
  · right
    left
    exact ⟨leftMessage, rightMessage, sameMessage, sameOutput⟩

end Nightstream.Protocol.FPrime.XOut
