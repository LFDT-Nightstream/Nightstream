import Nightstream.Protocol.FPrime.ConcretePhi81.Outer
import Nightstream.Protocol.FPrime.XOut
import Nightstream.SuperNeo.Folding.PiDEC.Necessity.ProductionChildSubstitution

/-!
F-prime consequence of the production Phi81 child-substitution witness.

Assurance tier: model-level.

Owns: two distinct complete active running slots with the same checked
PiRLC parent; strict incoming PiDEC acceptance for both slots; inequality of
their paper-visible child products; and equality of every parent-only
Construction-2 accumulator handle and resulting `x_out` message.

Does not own: Rust/R1CS decoding, concrete Poseidon2 arithmetic, an executable
lifecycle exploit, probability, costs, or row removal.

Emits constraints: no.

Authority boundary: the two `x_out` messages below are definitionally equal.
This is not a hash collision. The omitted running-child vectors are distinct,
but both are valid strict-PiDEC decompositions of the exact same parent.

| Stage path | Mathematical obligation | Authority class | Rust owner | Lean owner | Status |
|---|---|---|---|---|---|
| `fprime.recursive.running.children` | retain the exact fourteen-child accumulator | public authority | `AccumulatorHandle::from_running_parts` caller | `paperRunning_ne` | omitted by parent-only handle |
| `fprime.recursive.running.parent_dec` | both substituted child vectors pass strict incoming PiDEC | checked | `nifs::circuit` | `leftIncomingAccepted`, `rightIncomingAccepted` | retained but insufficient |
| `fprime.recursive.running.handle` | a function of only the common parent cannot bind the child vector | impossibility | `accumulator_digest_from_running_parts` | `no_parentOnlyAccumulator_binds` | unsafe boundary |
| `fprime.x_out.preimage.construction2_accumulator` | parent-only handles produce identical state-output messages | computed | `state_x_out_digest_with_mode` | `stateOutputMessages_eq` | exact alias |
-/

namespace Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ParentOnlyAuthority

open Nightstream.HyperNova.Construction2
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiDEC.Necessity.ProductionChildSubstitution

namespace Substitution

def shape : SemanticShape :=
  FPrimeCarrier270.PaddedIdentityEvaluation.semanticShape
abbrev Statement := Fixture.Statement

def publicRingColumns : Nat := FPrimeCarrier270.publicRingColumns

def publicFits :
    ringDegree * publicRingColumns <= shape.carrierWidth := by
  decide

/-- Exact one-slot active accumulator type at the counterexample profile. -/
abbrev Slot := Outer.Slot shape publicRingColumns publicFits 0

/-- First valid strict-radix representation of the common parent. -/
def leftSlot : Slot where
  parent := Fixture.parent
  children := Fixture.leftChildren

/-- Distinct valid strict-radix representation of the common parent. -/
def rightSlot : Slot where
  parent := Fixture.parent
  children := Fixture.rightChildren

@[simp] theorem leftSlot_parent : leftSlot.parent = Fixture.parent := rfl

@[simp] theorem rightSlot_parent : rightSlot.parent = Fixture.parent := rfl

@[simp] theorem leftSlot_children :
    leftSlot.children = Fixture.leftChildren := rfl

@[simp] theorem rightSlot_children :
    rightSlot.children = Fixture.rightChildren := rfl

/-- The rich accumulator slots differ even though their cached parent is
literally the same statement. -/
theorem slots_ne : leftSlot ≠ rightSlot := by
  intro equal
  apply Fixture.children_ne
  exact congrArg (fun slot : Slot => slot.children) equal

/-- Strict incoming PiDEC accepts the left running slot. -/
theorem leftIncomingAccepted :
    PiDEC.Accepted Fixture.algebra {
      parent := leftSlot.parent
      children := leftSlot.children
    } := by
  simpa [leftSlot] using Fixture.leftAccepted

/-- Strict incoming PiDEC also accepts the substituted right running slot. -/
theorem rightIncomingAccepted :
    PiDEC.Accepted Fixture.algebra {
      parent := rightSlot.parent
      children := rightSlot.children
    } := by
  simpa [rightSlot] using Fixture.rightAccepted

/-- A one-slot outer running product is enough to expose the F-prime state
alias. -/
abbrev Running :=
  Outer.Running shape publicRingColumns publicFits 0 1

def leftRunning : Running := fun _ => leftSlot

def rightRunning : Running := fun _ => rightSlot

theorem running_ne : leftRunning ≠ rightRunning := by
  intro equal
  apply slots_ne
  exact congrFun equal (0 : Fin 1)

/-- Projection to the paper carrier removes only the parent cache, not the
authoritative children; the two paper running products remain distinct. -/
theorem paperRunning_ne :
    leftRunning.toPaper ≠ rightRunning.toPaper := by
  intro equal
  have atSlot := congrFun equal (0 : Fin 1)
  change Fixture.leftChildren = Fixture.rightChildren at atSlot
  exact Fixture.children_ne atSlot

/-- The authority claim required to use a parent-only accumulator handle as a
binding commitment to one complete running slot. -/
def ParentOnlyAccumulatorBinds
    {Digest : Type}
    (handle : Statement -> Digest) : Prop :=
  ∀ left right : Slot,
    PiDEC.Accepted Fixture.algebra {
      parent := left.parent
      children := left.children
    } ->
    PiDEC.Accepted Fixture.algebra {
      parent := right.parent
      children := right.children
    } ->
    handle left.parent = handle right.parent ->
    left = right

/-- No function of only the cached parent can bind a complete F-prime running
slot, even when both slots pass the exact incoming PiDEC check. -/
theorem no_parentOnlyAccumulator_binds
    {Digest : Type}
    (handle : Statement -> Digest) :
    ¬ ParentOnlyAccumulatorBinds handle := by
  intro binds
  exact slots_ne
    (binds leftSlot rightSlot leftIncomingAccepted rightIncomingAccepted rfl)

universe uParams uStructure uHeader uDigest uNebulaDigest

/-- Opaque proof-payload tag used to keep the large concrete slot out of the
`x_out` type. `slot` below gives its exact semantic interpretation. -/
inductive RunningPayload where
  | left
  | right
deriving DecidableEq

namespace RunningPayload

def slot : RunningPayload -> Slot
  | .left => leftSlot
  | .right => rightSlot

end RunningPayload

def leftPayload : RunningPayload := .left

def rightPayload : RunningPayload := .right

theorem payloads_ne : leftPayload ≠ rightPayload := by
  decide

theorem payloadSlots_ne :
    leftPayload.slot ≠ rightPayload.slot := by
  exact slots_ne

/-- Stateless Construction-2 state used to isolate the accumulator binding.
All non-accumulator coordinates are verifier-derived or identical. -/
def xOutState
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Header : Type uHeader}
    {Digest : Type uDigest}
    {NebulaDigest : Type uNebulaDigest}
    (semantics :
      XOut.Semantics Params StructureDigest Header Digest Unit NebulaDigest)
    (context : XOut.Context Params StructureDigest Header Digest)
    (accumulatorDigest : Digest)
    (payload : RunningPayload) :
    State Digest RunningPayload Unit Unit where
  chunkCount := 1
  stepCount := 1
  z0 := XOut.initialBoundary semantics context
  zi := context.initialSemanticState
  initialSemanticState := context.initialSemanticState
  semanticState := accumulatorDigest
  pc := 1
  accumulatorDigest := accumulatorDigest
  publicTrace := context.initialSemanticState
  proof := .active payload [()]
  nebula := none

theorem xOutState_pinned
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Header : Type uHeader}
    {Digest : Type uDigest}
    {NebulaDigest : Type uNebulaDigest}
    (semantics :
      XOut.Semantics Params StructureDigest Header Digest Unit NebulaDigest)
    (context : XOut.Context Params StructureDigest Header Digest)
    (accumulatorDigest : Digest)
    (payload : RunningPayload) :
    XOut.StatePinned semantics .stateless context
      (xOutState semantics context accumulatorDigest payload) := by
  exact {
    initialBoundaryPinned := rfl
    initialSemanticStatePinned := rfl
    publicTraceMirrorsBoundary := rfl
    statelessSemanticEqualsAccumulator := fun _ => rfl
  }

/-- Extract the active running payload only; this makes state inequality
independent of the list payload. -/
def proofRunning? : ProofState RunningPayload Unit -> Option RunningPayload
  | .initial => none
  | .active running _ => some running

/-- The two Construction-2 states are genuinely different because their
complete running accumulators differ. -/
theorem xOutStates_ne
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Header : Type uHeader}
    {Digest : Type uDigest}
    {NebulaDigest : Type uNebulaDigest}
    (semantics :
      XOut.Semantics Params StructureDigest Header Digest Unit NebulaDigest)
    (context : XOut.Context Params StructureDigest Header Digest)
    (accumulatorDigest : Digest) :
    xOutState semantics context accumulatorDigest leftPayload ≠
      xOutState semantics context accumulatorDigest rightPayload := by
  intro equal
  have proofEqual := congrArg
    (fun state : State Digest RunningPayload Unit Unit =>
      proofRunning? state.proof)
    equal
  change some leftPayload = some rightPayload at proofEqual
  exact payloads_ne (Option.some.inj proofEqual)

/-- Despite distinct full states, parent-only authority yields the exact same
typed `x_out` preimage. -/
theorem xOutPreimages_eq
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Header : Type uHeader}
    {Digest : Type uDigest}
    {NebulaDigest : Type uNebulaDigest}
    (semantics :
      XOut.Semantics Params StructureDigest Header Digest Unit NebulaDigest)
    (context : XOut.Context Params StructureDigest Header Digest)
    (accumulatorDigest : Digest) :
    XOut.preimage semantics .stateless context
        (xOutState semantics context accumulatorDigest leftPayload) =
      XOut.preimage semantics .stateless context
        (xOutState semantics context accumulatorDigest rightPayload) := by
  apply XOut.preimage_eq_of_visible_fields
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl

/-- The hash inputs themselves are identical, so equal outputs require no
Poseidon2 collision assumption. -/
theorem stateOutputMessages_eq
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Header : Type uHeader}
    {Digest : Type uDigest}
    {NebulaDigest : Type uNebulaDigest}
    (semantics :
      XOut.Semantics Params StructureDigest Header Digest Unit NebulaDigest)
    (context : XOut.Context Params StructureDigest Header Digest)
    (accumulatorDigest : Digest) :
    @XOut.Message.stateOutput Params StructureDigest Header Digest NebulaDigest
        (XOut.preimage semantics .stateless context
          (xOutState semantics context accumulatorDigest leftPayload)) =
      @XOut.Message.stateOutput Params StructureDigest Header Digest NebulaDigest
        (XOut.preimage semantics .stateless context
          (xOutState semantics context accumulatorDigest rightPayload)) := by
  rw [xOutPreimages_eq]

/-- Consequently every hash function—not merely Poseidon2—returns the same
`x_out` for the two distinct running states. -/
theorem xOut_eq
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Header : Type uHeader}
    {Digest : Type uDigest}
    {NebulaDigest : Type uNebulaDigest}
    (semantics :
      XOut.Semantics Params StructureDigest Header Digest Unit NebulaDigest)
    (context : XOut.Context Params StructureDigest Header Digest)
    (accumulatorDigest : Digest) :
    XOut.compute semantics .stateless context
        (xOutState semantics context accumulatorDigest leftPayload) =
      XOut.compute semantics .stateless context
        (xOutState semantics context accumulatorDigest rightPayload) := by
  rw [XOut.compute, XOut.compute, xOutPreimages_eq]

end Substitution

end Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ParentOnlyAuthority
