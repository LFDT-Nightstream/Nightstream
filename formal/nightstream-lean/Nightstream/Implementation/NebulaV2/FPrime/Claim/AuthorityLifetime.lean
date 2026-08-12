import Mathlib.Algebra.Field.TransferInstance
import Nightstream.Implementation.NebulaV2.FPrime.Manifest.BaseStateAuthority
import Nightstream.Implementation.NebulaV2.FPrime.Claim.DelayedTrace
import Nightstream.Implementation.NebulaV2.FPrime.Claim.FreshLink
import Nightstream.Implementation.NebulaV2.FPrime.Claim.GlobalFPrime
import Nightstream.Implementation.NebulaV2.FPrime.Manifest.RecursiveStateAuthority
import Nightstream.Implementation.NebulaV2.FPrime.Manifest.TerminalStateAuthority

/-!
Contract: exact delayed full-claim authority over one Nebula V2 lifetime.

Assurance tier: implementation model and cryptographic boundary.

Owns one ordered consumer record per exact delayed receipt, construction of
recursive and terminal consumer records from satisfying manifests, the exact
540-coordinate producer-to-claim link before every receipt, derivation of the
four public-state wrapper placements, terminal no-output shape, and an
induction that binds every exact receipt to its producer state or returns the
first named two-stage Poseidon2 collision.

Does not own a generated base artifact, producer-side fresh-link row
refinement, NIFS extraction, recursive-size closure, terminal backend
verification, Poseidon2 collision resistance, or Rust conformance.

Emits constraints: no new rows. Each boundary owns the four equality rows from
`StateAuthorityBoundaryRows`.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.NebulaV2.FullClaimAuthorityLifetime

universe uParams uStructure uHeader uRunning uNifsProof uNebulaDigest
  uNebulaOpen

open Nightstream.Implementation.NebulaV2.FullClaimEnvelope
open Nightstream.Implementation.NebulaV2.FullClaimNifsReceipt
open Nightstream.Implementation.NebulaV2.StateAuthorityBoundaryRows
open Nightstream.Protocol.NebulaV2
open Nightstream.Protocol.NebulaV2.FPrime
open Nightstream.SuperNeo.Concrete

/-- The executable coefficient pair is transported through the proved exact
equivalence to the mathematical Goldilocks quadratic field. -/
noncomputable local instance concreteKField : Field K :=
  Nightstream.Implementation.NebulaV2.ConcreteField.superNeoEquiv.field

/-- One invocation that consumes one exact selected-verifier receipt. A
nonterminal invocation has one outgoing authority. The terminal invocation has
no outgoing authority. `carriesIncoming` is constructed from mandatory local
rows by the manifest adapters below. -/
structure ConsumingInvocation {widths : CompilerWidths}
    (selected : SelectedVerifier widths) where
  receipt : Receipt selected
  incoming : Authority
  outgoing : Option Authority
  carriesIncoming : StateAuthorityFullClaim.Carries incoming receipt.envelope

namespace ConsumingInvocation

/-- One satisfying recursive manifest constructs a nonterminal consumer. Both
authority values come from the same assignment and the exact receipt is the
one accepted and consumed by that manifest. -/
def ofRecursive
    {widths : CompilerWidths}
    {artifact : RecursiveManifestSchema.Artifact widths}
    {selected : SelectedVerifier widths} {assignment : Nat → Nat}
    {call : RecursiveManifestNifsCall.Call artifact selected assignment}
    (carry : call.CarryBlocks)
    (satisfies : Nightstream.Implementation.R1CS.Satisfies
      artifact.programRows assignment) :
    ConsumingInvocation selected where
  receipt := call.receiptOfRows satisfies
  incoming := RecursiveManifestStateAuthority.priorAuthority carry satisfies
  outgoing := some
    (RecursiveManifestStateAuthority.outgoingAuthority carry satisfies)
  carriesIncoming :=
    RecursiveManifestStateAuthority.exactReceiptCarriesPriorAuthority
      carry satisfies

/-- One satisfying terminal manifest constructs the unique consumer with no
outgoing fresh claim. -/
def ofTerminal
    {widths : CompilerWidths}
    {fullShape operationsShape snapshotShape :
      Nightstream.SuperNeo.Concrete.Phi81Relation.Shape}
    {artifact : TerminalManifestSchema.Artifact widths fullShape
      operationsShape snapshotShape}
    {selected : SelectedVerifier widths} {assignment : Nat → Nat}
    {call : TerminalManifestNifsCall.Call artifact selected assignment}
    (carry : call.CarryBlocks)
    (satisfies : Nightstream.Implementation.R1CS.Satisfies
      artifact.programRows assignment) :
    ConsumingInvocation selected where
  receipt := call.receiptOfRows satisfies
  incoming := TerminalManifestStateAuthority.incomingAuthority carry satisfies
  outgoing := none
  carriesIncoming :=
    TerminalManifestStateAuthority.exactReceiptCarriesIncomingAuthority
      carry satisfies

@[simp] theorem ofRecursive_outgoing
    {widths : CompilerWidths}
    {artifact : RecursiveManifestSchema.Artifact widths}
    {selected : SelectedVerifier widths} {assignment : Nat → Nat}
    {call : RecursiveManifestNifsCall.Call artifact selected assignment}
    (carry : call.CarryBlocks)
    (satisfies : Nightstream.Implementation.R1CS.Satisfies
      artifact.programRows assignment) :
    (ofRecursive carry satisfies).outgoing = some
      (RecursiveManifestStateAuthority.outgoingAuthority carry satisfies) :=
  rfl

@[simp] theorem ofTerminal_outgoing
    {widths : CompilerWidths}
    {fullShape operationsShape snapshotShape :
      Nightstream.SuperNeo.Concrete.Phi81Relation.Shape}
    {artifact : TerminalManifestSchema.Artifact widths fullShape
      operationsShape snapshotShape}
    {selected : SelectedVerifier widths} {assignment : Nat → Nat}
    {call : TerminalManifestNifsCall.Call artifact selected assignment}
    (carry : call.CarryBlocks)
    (satisfies : Nightstream.Implementation.R1CS.Satisfies
      artifact.programRows assignment) :
    (ofTerminal carry satisfies).outgoing = none :=
  rfl

end ConsumingInvocation

/-- Candidate lifetime extracted from wrapper and local rows. The first
authority is the base producer output. Each boundary compares that producer
output with the exact incoming state carried by the next receipt. A recursive
consumer supplies the next producer output. The last consumer is terminal and
must not produce another claim. -/
inductive Candidate {widths : CompilerWidths}
    (selected : SelectedVerifier widths) :
    Authority → List (ConsumingInvocation selected) → Prop
  | terminal
      {producer : Authority} {invocation : ConsumingInvocation selected}
      (boundary : Boundary producer invocation.incoming)
      (noOutgoing : invocation.outgoing = none) :
      Candidate selected producer [invocation]
  | recursive
      {producer nextProducer : Authority}
      {invocation : ConsumingInvocation selected}
      {rest : List (ConsumingInvocation selected)}
      (boundary : Boundary producer invocation.incoming)
      (continues : invocation.outgoing = some nextProducer)
      (tail : Candidate selected nextProducer rest) :
      Candidate selected producer (invocation :: rest)

/-- Collision-free lifetime result. In addition to exact typed state equality,
each constructor records that the producer state carries the complete exact
receipt, including that receipt's memory suffix. -/
inductive Exact {widths : CompilerWidths}
    (selected : SelectedVerifier widths) :
    Authority → List (ConsumingInvocation selected) → Prop
  | terminal
      {producer : Authority} {invocation : ConsumingInvocation selected}
      (same : Same producer invocation.incoming)
      (carriesProduced :
        StateAuthorityFullClaim.Carries producer invocation.receipt.envelope)
      (noOutgoing : invocation.outgoing = none) :
      Exact selected producer [invocation]
  | recursive
      {producer nextProducer : Authority}
      {invocation : ConsumingInvocation selected}
      {rest : List (ConsumingInvocation selected)}
      (same : Same producer invocation.incoming)
      (carriesProduced :
        StateAuthorityFullClaim.Carries producer invocation.receipt.envelope)
      (continues : invocation.outgoing = some nextProducer)
      (tail : Exact selected nextProducer rest) :
      Exact selected producer (invocation :: rest)

namespace Candidate

/-- A candidate always contains at least the terminal consumer. -/
theorem nonempty
    {widths : CompilerWidths} {selected : SelectedVerifier widths}
    {producer : Authority} {invocations : List (ConsumingInvocation selected)}
    (candidate : Candidate selected producer invocations) :
    invocations ≠ [] := by
  cases candidate <;> simp

/-- Global delayed-authority induction. No exact state link is an assumption:
every one is derived from four equality rows. If a link cannot recover the
typed state, the result names the concrete inner or outer Poseidon2 collision.
The exact receipt and its memory suffix remain the same typed object. -/
theorem sound_or_collision
    {widths : CompilerWidths} {selected : SelectedVerifier widths}
    {producer : Authority} {invocations : List (ConsumingInvocation selected)}
    (candidate : Candidate selected producer invocations) :
    Exact selected producer invocations ∨ StateAuthorityBoundaryRows.Failure := by
  induction candidate with
  | @terminal producer invocation boundary noOutgoing =>
      rcases boundary.sound with same | failure
      · exact Or.inl (.terminal same
          (StateAuthorityFullClaim.carries_of_same same
            invocation.carriesIncoming) noOutgoing)
      · exact Or.inr failure
  | @recursive producer nextProducer invocation rest boundary continues tail
      inductionHypothesis =>
      rcases boundary.sound with same | failure
      · rcases inductionHypothesis with exactTail | failure
        · exact Or.inl (.recursive same
            (StateAuthorityFullClaim.carries_of_same same
              invocation.carriesIncoming)
            continues exactTail)
        · exact Or.inr failure
      · exact Or.inr failure

end Candidate

namespace Exact

/-- The first exact delayed edge binds the complete head receipt to the base or
previous recursive producer state. -/
theorem headCarriesProducer
    {widths : CompilerWidths} {selected : SelectedVerifier widths}
    {producer : Authority} {head : ConsumingInvocation selected}
    {tail : List (ConsumingInvocation selected)}
    (exact : Exact selected producer (head :: tail)) :
    StateAuthorityFullClaim.Carries producer head.receipt.envelope := by
  cases exact with
  | terminal _ carriesProduced _ => exact carriesProduced
  | recursive _ carriesProduced _ _ => exact carriesProduced

/-- The final exact consumer has no outgoing claim. -/
theorem lastHasNoOutgoing
    {widths : CompilerWidths} {selected : SelectedVerifier widths}
    {producer : Authority} {invocations : List (ConsumingInvocation selected)}
    (exact : Exact selected producer invocations) :
    ∃ front last,
      invocations = front ++ [last] ∧ last.outgoing = none := by
  induction exact with
  | @terminal producer invocation same carriesProduced noOutgoing =>
      exact ⟨[], invocation, rfl, noOutgoing⟩
  | @recursive producer nextProducer invocation rest same carriesProduced
      continues tail inductionHypothesis =>
      rcases inductionHypothesis with ⟨front, last, restExact, noOutgoing⟩
      refine ⟨invocation :: front, last, ?_, noOutgoing⟩
      simp [restExact]

end Exact

/-- The initial producer is not a free authority value. It is the normalized
output extracted from one satisfying base manifest assignment. The call also
contains the mandatory initial-memory and segment-opening placements. -/
structure BaseProducer (widths : CompilerWidths) where
  artifact : BaseManifestSchema.Artifact widths
  assignment : Nat → Nat
  call : BaseManifestStateAuthority.Call artifact assignment
  satisfies : Nightstream.Implementation.R1CS.Satisfies
    artifact.programRows assignment

namespace BaseProducer

/-- Exact authority emitted by the mandatory base state-output rows. -/
def authority {widths : CompilerWidths} (base : BaseProducer widths) :
    Authority :=
  base.call.outgoingAuthority base.satisfies

/-- The same base assignment derives the verifier-authoritative initial carry. -/
theorem initialExact {widths : CompilerWidths} (base : BaseProducer widths) :
    InitialMemoryCarryRows.Exact base.call.initialValue
      base.call.initialMemoryRoot :=
  base.call.initialExact base.satisfies

/-- The same base assignment opens the first segment from that initial carry. -/
theorem opensExactInitialCarry
    {widths : CompilerWidths} (base : BaseProducer widths) :
    ∃ (canOpen :
        (MemoryOpenSegmentSound.closedOfWire base.call.initialValue).CanOpen)
      (activeCountInRange :
        base.call.outgoingValue.segmentActiveAccessCount < operationCountLimit)
      (endTimestampInRange :
        (MemoryOpenSegmentSound.closedOfWire base.call.initialValue).globalTimestamp +
            base.call.outgoingValue.segmentActiveAccessCount < timestampLimit)
      (stepBound :
        base.call.outgoingValue.stepIndex < Lifecycle.claimsPerSegment),
      base.call.initialValue.phase = .closed ∧
        base.call.outgoingValue.phase = .active ∧
        Carry.active
            (MemoryOpenSegmentSound.activeOfWire base.call.outgoingValue stepBound) =
          MemoryOpenSegment.openCarryFor base.artifact.profile
            base.call.openingAuthority base.call.headers base.call.outgoingValue.dPre
            base.call.outgoingValue.segmentActiveAccessCount
            (MemoryOpenSegmentSound.closedOfWire base.call.initialValue) canOpen
            activeCountInRange endTimestampInRange :=
  base.call.opensExactInitialCarry base.satisfies

end BaseProducer

/-- One nonterminal edge derived from a satisfying recursive manifest. The
producer-side premise is the exact executable 540-coordinate fresh-public
link for the same full claim consumed by this manifest. It is not a state
equality or a digest-equality premise. -/
structure RecursiveEdge {widths : CompilerWidths}
    (selected : SelectedVerifier widths) (producer : Authority) where
  artifact : RecursiveManifestSchema.Artifact widths
  assignment : Nat → Nat
  call : RecursiveManifestNifsCall.Call artifact selected assignment
  carry : call.CarryBlocks
  satisfies : Nightstream.Implementation.R1CS.Satisfies
    artifact.programRows assignment
  producerLinked :
    Nightstream.Protocol.FPrime.Step.FreshLinked FullClaimFreshLink.check
      (StateAuthorityFullClaim.canonicalDigest producer)
      [(call.receiptOfRows satisfies).envelope]

namespace RecursiveEdge

/-- Construct the only release-facing recursive edge from a closed delayed
F-prime producer. The complete receipt envelope is the producer's claim type
parameter; there is no claim-equality transport premise. -/
def ofDelayedProducer
    {widths : CompilerWidths}
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Header : Type uHeader}
    {Running : Type uRunning}
    {NifsProof : Type uNifsProof}
    {Nebula : Type}
    {NebulaDigest : Type uNebulaDigest}
    {NebulaOpen : Type uNebulaOpen}
    {configuration :
      Nightstream.Protocol.FPrime.DelayedTrace.Configuration Params
        StructureDigest Header Digest.Value Running (Value widths) NifsProof
        Nebula NebulaDigest NebulaOpen}
    {producer : Authority}
    {artifact : RecursiveManifestSchema.Artifact widths}
    {selected : SelectedVerifier widths} {assignment : Nat → Nat}
    (call : RecursiveManifestNifsCall.Call artifact selected assignment)
    (carry : call.CarryBlocks)
    (satisfies : Nightstream.Implementation.R1CS.Satisfies
      artifact.programRows assignment)
    (produced : FullClaimDelayedTrace.Producer configuration producer
      (call.receiptOfRows satisfies).envelope)
    (outgoing : produced.invocation.OutgoingLinked) :
    RecursiveEdge selected producer where
  artifact := artifact
  assignment := assignment
  call := call
  carry := carry
  satisfies := satisfies
  producerLinked := produced.freshLinked_of_outgoing outgoing

/-- The selected identity is already owned by the exact manifest call. -/
theorem matchesSelected
    {widths : CompilerWidths} {selected : SelectedVerifier widths}
    {producer : Authority} (edge : RecursiveEdge selected producer) :
    edge.artifact.MatchesSelected selected :=
  edge.call.identity

def invocation
    {widths : CompilerWidths} {selected : SelectedVerifier widths}
    {producer : Authority} (edge : RecursiveEdge selected producer) :
    ConsumingInvocation selected :=
  ConsumingInvocation.ofRecursive edge.carry edge.satisfies

def nextProducer
    {widths : CompilerWidths} {selected : SelectedVerifier widths}
    {producer : Authority} (edge : RecursiveEdge selected producer) :
    Authority :=
  RecursiveManifestStateAuthority.outgoingAuthority edge.carry edge.satisfies

/-- The producer-side full-carrier relation is recovered from the executable
fresh-public link. -/
theorem producerCarries
    {widths : CompilerWidths} {selected : SelectedVerifier widths}
    {producer : Authority} (edge : RecursiveEdge selected producer) :
    StateAuthorityFullClaim.Carries producer edge.invocation.receipt.envelope :=
  FullClaimFreshLink.carries_of_freshLinked edge.producerLinked

/-- The same complete claim is carried by the producer link and by the
consumer's mandatory prior-state rows. Therefore the exact typed states are
equal or one named two-stage Poseidon2 collision occurred. -/
theorem sameOrFailure
    {widths : CompilerWidths} {selected : SelectedVerifier widths}
    {producer : Authority} (edge : RecursiveEdge selected producer) :
    StateAuthorityBoundaryRows.Same producer edge.invocation.incoming ∨
      StateAuthorityBoundaryRows.Failure :=
  StateAuthorityFullClaim.same_claim_authority_eq_or_failure
    edge.producerCarries edge.invocation.carriesIncoming

/-- Once exact typed equality is recovered from the complete claim, the
manifest's four mandatory wrapper rows derive the named public-state
placement and construct the concrete row boundary. -/
def boundary
    {widths : CompilerWidths} {selected : SelectedVerifier widths}
    {producer : Authority} (edge : RecursiveEdge selected producer)
    (same : StateAuthorityBoundaryRows.Same producer
      edge.invocation.incoming) :
    Boundary producer edge.invocation.incoming := by
  have placed := RecursiveManifestStateAuthority.previousStatePlaced_of_same
    edge.carry producer edge.satisfies same
  exact RecursiveManifestStateAuthority.boundaryFromPrevious edge.carry
    producer placed edge.satisfies

@[simp] theorem invocation_outgoing
    {widths : CompilerWidths} {selected : SelectedVerifier widths}
    {producer : Authority} (edge : RecursiveEdge selected producer) :
    edge.invocation.outgoing = some edge.nextProducer :=
  rfl

end RecursiveEdge

/-- The unique last edge is derived from one satisfying terminal manifest and
the exact producer-side fresh-public link for the same trailing full claim. -/
structure TerminalEdge {widths : CompilerWidths}
    (selected : SelectedVerifier widths) (producer : Authority) where
  fullShape : Nightstream.SuperNeo.Concrete.Phi81Relation.Shape
  operationsShape : Nightstream.SuperNeo.Concrete.Phi81Relation.Shape
  snapshotShape : Nightstream.SuperNeo.Concrete.Phi81Relation.Shape
  artifact : TerminalManifestSchema.Artifact widths fullShape operationsShape
    snapshotShape
  assignment : Nat → Nat
  call : TerminalManifestNifsCall.Call artifact selected assignment
  carry : call.CarryBlocks
  satisfies : Nightstream.Implementation.R1CS.Satisfies
    artifact.programRows assignment
  producerLinked :
    Nightstream.Protocol.FPrime.Step.FreshLinked FullClaimFreshLink.check
      (StateAuthorityFullClaim.canonicalDigest producer)
      [(call.receiptOfRows satisfies).envelope]

namespace TerminalEdge

/-- Construct the terminal edge from the exact trailing delayed producer and
the same complete typed claim consumed by the terminal manifest. -/
def ofDelayedProducer
    {widths : CompilerWidths}
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Header : Type uHeader}
    {Running : Type uRunning}
    {NifsProof : Type uNifsProof}
    {Nebula : Type}
    {NebulaDigest : Type uNebulaDigest}
    {NebulaOpen : Type uNebulaOpen}
    {configuration :
      Nightstream.Protocol.FPrime.DelayedTrace.Configuration Params
        StructureDigest Header Digest.Value Running (Value widths) NifsProof
        Nebula NebulaDigest NebulaOpen}
    {producer : Authority}
    {fullShape operationsShape snapshotShape :
      Nightstream.SuperNeo.Concrete.Phi81Relation.Shape}
    {artifact : TerminalManifestSchema.Artifact widths fullShape
      operationsShape snapshotShape}
    {selected : SelectedVerifier widths} {assignment : Nat → Nat}
    (call : TerminalManifestNifsCall.Call artifact selected assignment)
    (carry : call.CarryBlocks)
    (satisfies : Nightstream.Implementation.R1CS.Satisfies
      artifact.programRows assignment)
    (produced : FullClaimDelayedTrace.Producer configuration producer
      (call.receiptOfRows satisfies).envelope)
    (outgoing : produced.invocation.OutgoingLinked) :
    TerminalEdge selected producer where
  fullShape := fullShape
  operationsShape := operationsShape
  snapshotShape := snapshotShape
  artifact := artifact
  assignment := assignment
  call := call
  carry := carry
  satisfies := satisfies
  producerLinked := produced.freshLinked_of_outgoing outgoing

/-- The terminal call owns the exact selected-profile identity. -/
theorem matchesSelected
    {widths : CompilerWidths} {selected : SelectedVerifier widths}
    {producer : Authority} (edge : TerminalEdge selected producer) :
    edge.artifact.MatchesSelected selected :=
  edge.call.identity

def invocation
    {widths : CompilerWidths} {selected : SelectedVerifier widths}
    {producer : Authority} (edge : TerminalEdge selected producer) :
    ConsumingInvocation selected :=
  ConsumingInvocation.ofTerminal edge.carry edge.satisfies

/-- Exact producer carrier recovered from the trailing fresh-public link. -/
theorem producerCarries
    {widths : CompilerWidths} {selected : SelectedVerifier widths}
    {producer : Authority} (edge : TerminalEdge selected producer) :
    StateAuthorityFullClaim.Carries producer edge.invocation.receipt.envelope :=
  FullClaimFreshLink.carries_of_freshLinked edge.producerLinked

/-- The trailing full claim binds the terminal incoming state to the last
producer state, modulo the two named state-hash collision events. -/
theorem sameOrFailure
    {widths : CompilerWidths} {selected : SelectedVerifier widths}
    {producer : Authority} (edge : TerminalEdge selected producer) :
    StateAuthorityBoundaryRows.Same producer edge.invocation.incoming ∨
      StateAuthorityBoundaryRows.Failure :=
  StateAuthorityFullClaim.same_claim_authority_eq_or_failure
    edge.producerCarries edge.invocation.carriesIncoming

/-- The terminal wrapper placement and boundary follow from the exact claim
link and the mandatory four equality rows. -/
def boundary
    {widths : CompilerWidths} {selected : SelectedVerifier widths}
    {producer : Authority} (edge : TerminalEdge selected producer)
    (same : StateAuthorityBoundaryRows.Same producer
      edge.invocation.incoming) :
    Boundary producer edge.invocation.incoming := by
  have placed := TerminalManifestStateAuthority.previousStatePlaced_of_same
    edge.carry producer edge.satisfies same
  exact TerminalManifestStateAuthority.boundaryFromPrevious edge.carry
    producer placed edge.satisfies

@[simp] theorem invocation_outgoing
    {widths : CompilerWidths} {selected : SelectedVerifier widths}
    {producer : Authority} (edge : TerminalEdge selected producer) :
    edge.invocation.outgoing = none :=
  rfl

end TerminalEdge

/-- Release-facing authority chain. Every boundary is constructed from one
matching recursive or terminal manifest; callers cannot insert a standalone
four-row gadget that is absent from the selected relation. -/
inductive ManifestCandidate {widths : CompilerWidths}
    (selected : SelectedVerifier widths) :
    Authority → List (ConsumingInvocation selected) → Prop
  | terminal {producer : Authority}
      (edge : TerminalEdge selected producer) :
      ManifestCandidate selected producer [edge.invocation]
  | recursive {producer : Authority}
      {rest : List (ConsumingInvocation selected)}
      (edge : RecursiveEdge selected producer)
      (tail : ManifestCandidate selected edge.nextProducer rest) :
      ManifestCandidate selected producer (edge.invocation :: rest)

namespace ManifestCandidate

theorem sound_or_collision
    {widths : CompilerWidths} {selected : SelectedVerifier widths}
    {producer : Authority} {invocations : List (ConsumingInvocation selected)}
    (candidate : ManifestCandidate selected producer invocations) :
    Exact selected producer invocations ∨ StateAuthorityBoundaryRows.Failure := by
  induction candidate with
  | terminal edge =>
      rcases edge.sameOrFailure with same | failure
      · exact Or.inl (.terminal same edge.producerCarries
          edge.invocation_outgoing)
      · exact Or.inr failure
  | recursive edge tail inductionHypothesis =>
      rcases edge.sameOrFailure with same | failure
      · rcases inductionHypothesis with exactTail | failure
        · exact Or.inl (.recursive same edge.producerCarries
            edge.invocation_outgoing exactTail)
        · exact Or.inr failure
      · exact Or.inr failure

end ManifestCandidate

/-- The release-facing lifetime object uses one receipt list for both the
semantic delayed memory schedule and the row-derived authority schedule. -/
structure RowBoundChain
    {widths : CompilerWidths} (selected : SelectedVerifier widths)
    (initial final : FPrime.ClosedCarry Digest.Value) (segmentCount : Nat) where
  delayed : FullClaimGlobalFPrime.Chain selected initial final segmentCount
  positiveSegments : 0 < segmentCount
  base : BaseProducer widths
  baseMatchesSelected : base.artifact.MatchesSelected selected
  invocations : List (ConsumingInvocation selected)
  receiptsExact : invocations.map ConsumingInvocation.receipt = delayed.receipts
  authority : ManifestCandidate selected base.authority invocations

namespace RowBoundChain

theorem exactInvocationCount
    {widths : CompilerWidths} {selected : SelectedVerifier widths}
    {initial final : FPrime.ClosedCarry Digest.Value} {segmentCount : Nat}
    (chain : RowBoundChain selected initial final segmentCount) :
    chain.invocations.length = segmentCount * Lifecycle.claimsPerSegment := by
  have receiptLengths := congrArg List.length chain.receiptsExact
  simp only [List.length_map] at receiptLengths
  rw [receiptLengths]
  exact chain.delayed.exactClaimCount

theorem completeDelayedSchedule
    {widths : CompilerWidths} {selected : SelectedVerifier widths}
    {initial final : FPrime.ClosedCarry Digest.Value} {segmentCount : Nat}
    (chain : RowBoundChain selected initial final segmentCount) :
    Lifecycle.CompleteSchedule chain.invocations.length := by
  rw [chain.exactInvocationCount]
  have positiveClaims : 0 < segmentCount * Lifecycle.claimsPerSegment :=
    Nat.mul_pos chain.positiveSegments (by decide)
  exact Lifecycle.completeSchedule positiveClaims

theorem authoritySoundOrCollision
    {widths : CompilerWidths} {selected : SelectedVerifier widths}
    {initial final : FPrime.ClosedCarry Digest.Value} {segmentCount : Nat}
    (chain : RowBoundChain selected initial final segmentCount) :
    Exact selected chain.base.authority chain.invocations ∨
      StateAuthorityBoundaryRows.Failure :=
  chain.authority.sound_or_collision

end RowBoundChain

end Nightstream.Implementation.NebulaV2.FullClaimAuthorityLifetime
