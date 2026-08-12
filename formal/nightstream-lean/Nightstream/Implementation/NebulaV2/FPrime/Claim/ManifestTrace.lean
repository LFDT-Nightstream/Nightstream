import Nightstream.Implementation.NebulaV2.FPrime.Claim.AuthorityLifetime

/-!
Contract: one structurally paired delayed F-prime and V2 manifest trace.

Assurance tier: implementation model and explicit terminal boundary.

Owns one exact local F-prime producer for each exact recursive or terminal
manifest receipt, exact identity of each complete receipt claim by dependent
typing, exact state identity between adjacent local invocations, derivation of
all nonterminal delayed links from the next invocation, and construction of
the release-facing full-claim authority chain.

Does not own generated producer-side rows, generated NIFS verifier rows,
refinement from terminal public-input bytes to the typed last-producer state,
recursive size closure, cryptographic extraction, or Rust conformance.

The trace has no list-equality premise, no receipt-equality premise, and no
outgoing-link premise. The terminal constructor contains only the placement
of the typed last-producer digest in the terminal manifest's public-input
columns. The manifest rows derive the trailing link from that placement.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.NebulaV2.FullClaimManifestTrace

open Nightstream.Implementation.NebulaV2.FullClaimEnvelope
open Nightstream.Implementation.NebulaV2.FullClaimNifsReceipt
open Nightstream.Implementation.NebulaV2.FullClaimAuthorityLifetime
open Nightstream.Protocol.FPrime.DelayedTrace
open Nightstream.Protocol.NebulaV2
open Nightstream.SuperNeo.Concrete

universe uParams uStructure uHeader uRunning uNifsProof uNebulaDigest
  uNebulaOpen

/-- The executable coefficient pair is transported through the proved exact
equivalence to the mathematical Goldilocks quadratic field. -/
noncomputable local instance concreteKField : Field K :=
  Nightstream.Implementation.NebulaV2.ConcreteField.superNeoEquiv.field

/-- One nonterminal producer and the exact satisfying recursive manifest that
consumes its complete claim one invocation later. The receipt envelope occurs
directly in the producer type. -/
structure RecursiveNode
    {widths : CompilerWidths}
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Header : Type uHeader}
    {Running : Type uRunning}
    {NifsProof : Type uNifsProof}
    {Nebula : Type}
    {NebulaDigest : Type uNebulaDigest}
    {NebulaOpen : Type uNebulaOpen}
    (configuration : Configuration Params StructureDigest Header Digest.Value
      Running (Value widths) NifsProof Nebula NebulaDigest NebulaOpen)
    (selected : SelectedVerifier widths)
    (producer : StateAuthorityBoundaryRows.Authority) where
  artifact : RecursiveManifestSchema.Artifact widths
  assignment : Nat → Nat
  call : RecursiveManifestNifsCall.Call artifact selected assignment
  carry : call.CarryBlocks
  satisfies : Nightstream.Implementation.R1CS.Satisfies
    artifact.programRows assignment
  produced : FullClaimDelayedTrace.Producer configuration producer
    (call.receiptOfRows satisfies).envelope

namespace RecursiveNode

def invocation
    {widths : CompilerWidths}
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Header : Type uHeader}
    {Running : Type uRunning}
    {NifsProof : Type uNifsProof}
    {Nebula : Type}
    {NebulaDigest : Type uNebulaDigest}
    {NebulaOpen : Type uNebulaOpen}
    {configuration : Configuration Params StructureDigest Header Digest.Value
      Running (Value widths) NifsProof Nebula NebulaDigest NebulaOpen}
    {selected : SelectedVerifier widths}
    {producer : StateAuthorityBoundaryRows.Authority}
    (node : RecursiveNode configuration selected producer) :
    Invocation configuration :=
  node.produced.invocation

def consumer
    {widths : CompilerWidths}
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Header : Type uHeader}
    {Running : Type uRunning}
    {NifsProof : Type uNifsProof}
    {Nebula : Type}
    {NebulaDigest : Type uNebulaDigest}
    {NebulaOpen : Type uNebulaOpen}
    {configuration : Configuration Params StructureDigest Header Digest.Value
      Running (Value widths) NifsProof Nebula NebulaDigest NebulaOpen}
    {selected : SelectedVerifier widths}
    {producer : StateAuthorityBoundaryRows.Authority}
    (node : RecursiveNode configuration selected producer) :
    ConsumingInvocation selected :=
  ConsumingInvocation.ofRecursive node.carry node.satisfies

def nextProducer
    {widths : CompilerWidths}
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Header : Type uHeader}
    {Running : Type uRunning}
    {NifsProof : Type uNifsProof}
    {Nebula : Type}
    {NebulaDigest : Type uNebulaDigest}
    {NebulaOpen : Type uNebulaOpen}
    {configuration : Configuration Params StructureDigest Header Digest.Value
      Running (Value widths) NifsProof Nebula NebulaDigest NebulaOpen}
    {selected : SelectedVerifier widths}
    {producer : StateAuthorityBoundaryRows.Authority}
    (node : RecursiveNode configuration selected producer) :
    StateAuthorityBoundaryRows.Authority :=
  RecursiveManifestStateAuthority.outgoingAuthority node.carry node.satisfies

/-- Exact adjacency supplies this node's delayed link. -/
theorem outgoingLinked_of_next
    {widths : CompilerWidths}
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Header : Type uHeader}
    {Running : Type uRunning}
    {NifsProof : Type uNifsProof}
    {Nebula : Type}
    {NebulaDigest : Type uNebulaDigest}
    {NebulaOpen : Type uNebulaOpen}
    {configuration : Configuration Params StructureDigest Header Digest.Value
      Running (Value widths) NifsProof Nebula NebulaDigest NebulaOpen}
    {selected : SelectedVerifier widths}
    {producer : StateAuthorityBoundaryRows.Authority}
    (node : RecursiveNode configuration selected producer)
    (next : Invocation configuration)
    (continuous : node.invocation.next = next.prior) :
    node.invocation.OutgoingLinked :=
  node.invocation.outgoingLinked_of_next next continuous

/-- The release-facing edge is derived after the next invocation closes the
current producer. No caller supplies a claim equality. -/
def edgeOfNext
    {widths : CompilerWidths}
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Header : Type uHeader}
    {Running : Type uRunning}
    {NifsProof : Type uNifsProof}
    {Nebula : Type}
    {NebulaDigest : Type uNebulaDigest}
    {NebulaOpen : Type uNebulaOpen}
    {configuration : Configuration Params StructureDigest Header Digest.Value
      Running (Value widths) NifsProof Nebula NebulaDigest NebulaOpen}
    {selected : SelectedVerifier widths}
    {producer : StateAuthorityBoundaryRows.Authority}
    (node : RecursiveNode configuration selected producer)
    (next : Invocation configuration)
    (continuous : node.invocation.next = next.prior) :
    RecursiveEdge selected producer :=
  RecursiveEdge.ofDelayedProducer node.call node.carry node.satisfies
    node.produced (node.outgoingLinked_of_next next continuous)

end RecursiveNode

/-- The last producer and exact satisfying terminal manifest. The only
cross-object input is placement of the typed producer digest in the terminal
manifest's four named public-state columns. This is a public-input refinement
fact, not a fresh-link, claim-carrier, or state-equality conclusion. -/
structure TerminalNode
    {widths : CompilerWidths}
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Header : Type uHeader}
    {Running : Type uRunning}
    {NifsProof : Type uNifsProof}
    {Nebula : Type}
    {NebulaDigest : Type uNebulaDigest}
    {NebulaOpen : Type uNebulaOpen}
    (configuration : Configuration Params StructureDigest Header Digest.Value
      Running (Value widths) NifsProof Nebula NebulaDigest NebulaOpen)
    (selected : SelectedVerifier widths)
    (producer : StateAuthorityBoundaryRows.Authority) where
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
  produced : FullClaimDelayedTrace.Producer configuration producer
    (call.receiptOfRows satisfies).envelope
  producerStatePlaced : TerminalManifestStateAuthority.PreviousStatePlaced
    (artifact := artifact) (assignment := assignment) producer

namespace TerminalNode

def invocation
    {widths : CompilerWidths}
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Header : Type uHeader}
    {Running : Type uRunning}
    {NifsProof : Type uNifsProof}
    {Nebula : Type}
    {NebulaDigest : Type uNebulaDigest}
    {NebulaOpen : Type uNebulaOpen}
    {configuration : Configuration Params StructureDigest Header Digest.Value
      Running (Value widths) NifsProof Nebula NebulaDigest NebulaOpen}
    {selected : SelectedVerifier widths}
    {producer : StateAuthorityBoundaryRows.Authority}
    (node : TerminalNode configuration selected producer) :
    Invocation configuration :=
  node.produced.invocation

def consumer
    {widths : CompilerWidths}
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Header : Type uHeader}
    {Running : Type uRunning}
    {NifsProof : Type uNifsProof}
    {Nebula : Type}
    {NebulaDigest : Type uNebulaDigest}
    {NebulaOpen : Type uNebulaOpen}
    {configuration : Configuration Params StructureDigest Header Digest.Value
      Running (Value widths) NifsProof Nebula NebulaDigest NebulaOpen}
    {selected : SelectedVerifier widths}
    {producer : StateAuthorityBoundaryRows.Authority}
    (node : TerminalNode configuration selected producer) :
    ConsumingInvocation selected :=
  ConsumingInvocation.ofTerminal node.carry node.satisfies

/-- The terminal manifest's mandatory four equality rows bind the typed last
producer digest to the recomputed state carried by the trailing claim. -/
def boundary
    {widths : CompilerWidths}
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Header : Type uHeader}
    {Running : Type uRunning}
    {NifsProof : Type uNifsProof}
    {Nebula : Type}
    {NebulaDigest : Type uNebulaDigest}
    {NebulaOpen : Type uNebulaOpen}
    {configuration : Configuration Params StructureDigest Header Digest.Value
      Running (Value widths) NifsProof Nebula NebulaDigest NebulaOpen}
    {selected : SelectedVerifier widths}
    {producer : StateAuthorityBoundaryRows.Authority}
    (node : TerminalNode configuration selected producer) :
    StateAuthorityBoundaryRows.Boundary producer
      (TerminalManifestStateAuthority.incomingAuthority node.carry
        node.satisfies) :=
  TerminalManifestStateAuthority.boundaryFromPrevious node.carry producer
    node.producerStatePlaced node.satisfies

/-- The terminal rows and public producer placement derive the complete
540-coordinate carrier for the exact trailing claim. -/
theorem producerCarries
    {widths : CompilerWidths}
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Header : Type uHeader}
    {Running : Type uRunning}
    {NifsProof : Type uNifsProof}
    {Nebula : Type}
    {NebulaDigest : Type uNebulaDigest}
    {NebulaOpen : Type uNebulaOpen}
    {configuration : Configuration Params StructureDigest Header Digest.Value
      Running (Value widths) NifsProof Nebula NebulaDigest NebulaOpen}
    {selected : SelectedVerifier widths}
    {producer : StateAuthorityBoundaryRows.Authority}
    (node : TerminalNode configuration selected producer) :
    StateAuthorityFullClaim.Carries producer
      (node.call.receiptOfRows node.satisfies).envelope := by
  exact StateAuthorityFullClaim.carries_of_digest_eq node.boundary.digest_eq
    (TerminalManifestStateAuthority.exactReceiptCarriesIncomingAuthority
      node.carry node.satisfies)

/-- Terminal delayed consumption is a row-derived conclusion. The trace does
not receive this link as a premise. -/
theorem trailingLink
    {widths : CompilerWidths}
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Header : Type uHeader}
    {Running : Type uRunning}
    {NifsProof : Type uNifsProof}
    {Nebula : Type}
    {NebulaDigest : Type uNebulaDigest}
    {NebulaOpen : Type uNebulaOpen}
    {configuration : Configuration Params StructureDigest Header Digest.Value
      Running (Value widths) NifsProof Nebula NebulaDigest NebulaOpen}
    {selected : SelectedVerifier widths}
    {producer : StateAuthorityBoundaryRows.Authority}
    (node : TerminalNode configuration selected producer) :
    node.produced.invocation.OutgoingLinked :=
  node.produced.outgoing_of_carries node.producerCarries

def edge
    {widths : CompilerWidths}
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Header : Type uHeader}
    {Running : Type uRunning}
    {NifsProof : Type uNifsProof}
    {Nebula : Type}
    {NebulaDigest : Type uNebulaDigest}
    {NebulaOpen : Type uNebulaOpen}
    {configuration : Configuration Params StructureDigest Header Digest.Value
      Running (Value widths) NifsProof Nebula NebulaDigest NebulaOpen}
    {selected : SelectedVerifier widths}
    {producer : StateAuthorityBoundaryRows.Authority}
    (node : TerminalNode configuration selected producer) :
    TerminalEdge selected producer :=
  { fullShape := node.fullShape
    operationsShape := node.operationsShape
    snapshotShape := node.snapshotShape
    artifact := node.artifact
    assignment := node.assignment
    call := node.call
    carry := node.carry
    satisfies := node.satisfies
    producerLinked := FullClaimFreshLink.freshLinked_of_carries
      node.producerCarries }

end TerminalNode

/-- One exact paired lifetime. Each constructor places the producer and the
manifest consumer in the same dependent object. -/
inductive Candidate
    {widths : CompilerWidths}
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Header : Type uHeader}
    {Running : Type uRunning}
    {NifsProof : Type uNifsProof}
    {Nebula : Type}
    {NebulaDigest : Type uNebulaDigest}
    {NebulaOpen : Type uNebulaOpen}
    (configuration : Configuration Params StructureDigest Header Digest.Value
      Running (Value widths) NifsProof Nebula NebulaDigest NebulaOpen)
    (selected : SelectedVerifier widths) :
    StateAuthorityBoundaryRows.Authority → Invocation configuration →
      List (Invocation configuration) →
      List (ConsumingInvocation selected) → Type _
  | terminal {producer : StateAuthorityBoundaryRows.Authority}
      (node : TerminalNode configuration selected producer) :
      Candidate configuration selected producer node.invocation []
        [node.consumer]
  | recursive
      {producer : StateAuthorityBoundaryRows.Authority}
      {nextInvocation : Invocation configuration}
      {restInvocations : List (Invocation configuration)}
      {rest : List (ConsumingInvocation selected)}
      (node : RecursiveNode configuration selected producer)
      (continuous : node.invocation.next = nextInvocation.prior)
      (tail : Candidate configuration selected node.nextProducer
        nextInvocation restInvocations rest) :
      Candidate configuration selected producer node.invocation
        (nextInvocation :: restInvocations)
        (node.consumer :: rest)

namespace Candidate

/-- Forget manifest data and recover the exact generic delayed trace. Every
nonterminal link is still derived by `DelayedTrace.Candidate.closeAll`. -/
def toDelayed
    {widths : CompilerWidths}
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Header : Type uHeader}
    {Running : Type uRunning}
    {NifsProof : Type uNifsProof}
    {Nebula : Type}
    {NebulaDigest : Type uNebulaDigest}
    {NebulaOpen : Type uNebulaOpen}
    {configuration : Configuration Params StructureDigest Header Digest.Value
      Running (Value widths) NifsProof Nebula NebulaDigest NebulaOpen}
    {selected : SelectedVerifier widths}
    {producer : StateAuthorityBoundaryRows.Authority}
    {first : Invocation configuration}
    {restInvocations : List (Invocation configuration)}
    {consumers : List (ConsumingInvocation selected)}
    (candidate : Candidate configuration selected producer first
      restInvocations consumers) :
    Nightstream.Protocol.FPrime.DelayedTrace.Candidate configuration first
      restInvocations := by
  induction candidate with
  | terminal node =>
      exact .terminal node.invocation node.trailingLink
  | recursive node continuous tail inductionHypothesis =>
      exact .recursive continuous inductionHypothesis

/-- Construct the release-facing authority chain without receipt-list or
claim-equality transport. -/
def toManifest
    {widths : CompilerWidths}
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Header : Type uHeader}
    {Running : Type uRunning}
    {NifsProof : Type uNifsProof}
    {Nebula : Type}
    {NebulaDigest : Type uNebulaDigest}
    {NebulaOpen : Type uNebulaOpen}
    {configuration : Configuration Params StructureDigest Header Digest.Value
      Running (Value widths) NifsProof Nebula NebulaDigest NebulaOpen}
    {selected : SelectedVerifier widths}
    {producer : StateAuthorityBoundaryRows.Authority}
    {first : Invocation configuration}
    {restInvocations : List (Invocation configuration)}
    {consumers : List (ConsumingInvocation selected)}
    (candidate : Candidate configuration selected producer first
      restInvocations consumers) :
    ManifestCandidate selected producer consumers := by
  induction candidate with
  | terminal node =>
      exact .terminal node.edge
  | @recursive producer nextInvocation restInvocations rest node continuous tail
      inductionHypothesis =>
      exact .recursive (node.edgeOfNext nextInvocation continuous)
        inductionHypothesis

/-- All paired producers have their delayed checks closed. -/
def closed
    {widths : CompilerWidths}
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Header : Type uHeader}
    {Running : Type uRunning}
    {NifsProof : Type uNifsProof}
    {Nebula : Type}
    {NebulaDigest : Type uNebulaDigest}
    {NebulaOpen : Type uNebulaOpen}
    {configuration : Configuration Params StructureDigest Header Digest.Value
      Running (Value widths) NifsProof Nebula NebulaDigest NebulaOpen}
    {selected : SelectedVerifier widths}
    {producer : StateAuthorityBoundaryRows.Authority}
    {first : Invocation configuration}
    {restInvocations : List (Invocation configuration)}
    {consumers : List (ConsumingInvocation selected)}
    (candidate : Candidate configuration selected producer first
      restInvocations consumers) :
    Nightstream.Protocol.FPrime.DelayedTrace.Closed configuration first
      restInvocations :=
  candidate.toDelayed.closeAll

/-- Paired trace soundness. The result names only the two state-hash collision
events that can prevent exact typed state recovery. -/
theorem authoritySoundOrCollision
    {widths : CompilerWidths}
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Header : Type uHeader}
    {Running : Type uRunning}
    {NifsProof : Type uNifsProof}
    {Nebula : Type}
    {NebulaDigest : Type uNebulaDigest}
    {NebulaOpen : Type uNebulaOpen}
    {configuration : Configuration Params StructureDigest Header Digest.Value
      Running (Value widths) NifsProof Nebula NebulaDigest NebulaOpen}
    {selected : SelectedVerifier widths}
    {producer : StateAuthorityBoundaryRows.Authority}
    {first : Invocation configuration}
    {restInvocations : List (Invocation configuration)}
    {consumers : List (ConsumingInvocation selected)}
    (candidate : Candidate configuration selected producer first
      restInvocations consumers) :
    Exact selected producer consumers ∨ StateAuthorityBoundaryRows.Failure :=
  candidate.toManifest.sound_or_collision

/-- One producer exists for each consumer. -/
theorem exactProducerCount
    {widths : CompilerWidths}
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Header : Type uHeader}
    {Running : Type uRunning}
    {NifsProof : Type uNifsProof}
    {Nebula : Type}
    {NebulaDigest : Type uNebulaDigest}
    {NebulaOpen : Type uNebulaOpen}
    {configuration : Configuration Params StructureDigest Header Digest.Value
      Running (Value widths) NifsProof Nebula NebulaDigest NebulaOpen}
    {selected : SelectedVerifier widths}
    {producer : StateAuthorityBoundaryRows.Authority}
    {first : Invocation configuration}
    {restInvocations : List (Invocation configuration)}
    {consumers : List (ConsumingInvocation selected)}
    (candidate : Candidate configuration selected producer first
      restInvocations consumers) :
    restInvocations.length + 1 = consumers.length := by
  induction candidate with
  | terminal node => rfl
  | recursive node continuous tail inductionHypothesis =>
      simp [inductionHypothesis]

end Candidate

/-- The receipt order owned by a paired trace. It is a projection of the
consumer list, not a second caller-supplied list. -/
def receipts {widths : CompilerWidths} {selected : SelectedVerifier widths}
    (consumers : List (ConsumingInvocation selected)) :
    List (FullClaimNifsReceipt.Receipt selected) :=
  consumers.map (fun invocation => invocation.receipt)

/-- Release-facing lifetime object with one structural receipt order. The
global memory execution consumes the direct projection of the paired manifest
trace. No equality proof connects two independently supplied lists. -/
structure ExactChain
    {widths : CompilerWidths}
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Header : Type uHeader}
    {Running : Type uRunning}
    {NifsProof : Type uNifsProof}
    {Nebula : Type}
    {NebulaDigest : Type uNebulaDigest}
    {NebulaOpen : Type uNebulaOpen}
    (configuration : Configuration Params StructureDigest Header Digest.Value
      Running (Value widths) NifsProof Nebula NebulaDigest NebulaOpen)
    (selected : SelectedVerifier widths)
    (initial final : Nightstream.Protocol.NebulaV2.FPrime.ClosedCarry
      Digest.Value)
    (segmentCount : Nat) where
  positiveSegments : 0 < segmentCount
  base : BaseProducer widths
  baseMatchesSelected : base.artifact.MatchesSelected selected
  first : Invocation configuration
  restInvocations : List (Invocation configuration)
  consumers : List (ConsumingInvocation selected)
  trace : Candidate configuration selected base.authority first
    restInvocations consumers
  /-- Public-state refinement for the unique base invocation. The generated
  base wrapper must place the canonical initial proof tag in this exact prior
  state. The local F-prime relation derives `noFold`; no branch conclusion is
  stored here. -/
  firstPriorInitial : first.prior.proof = .initial
  memory : Nightstream.Protocol.NebulaV2.GlobalFPrime.Chain
    (protocolSchema widths (PackedProof selected)) Digest.Value
    (VerifyClaim selected) initial
    (FullClaimGlobalFPrime.verifiedReceipts (receipts consumers)) final
    segmentCount

namespace ExactChain

/-- The single structural consumer list has the exact V2 lifetime length. -/
theorem exactConsumerCount
    {widths : CompilerWidths}
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Header : Type uHeader}
    {Running : Type uRunning}
    {NifsProof : Type uNifsProof}
    {Nebula : Type}
    {NebulaDigest : Type uNebulaDigest}
    {NebulaOpen : Type uNebulaOpen}
    {configuration : Configuration Params StructureDigest Header Digest.Value
      Running (Value widths) NifsProof Nebula NebulaDigest NebulaOpen}
    {selected : SelectedVerifier widths}
    {initial final : Nightstream.Protocol.NebulaV2.FPrime.ClosedCarry
      Digest.Value}
    {segmentCount : Nat}
    (chain : ExactChain configuration selected initial final segmentCount) :
    chain.consumers.length = segmentCount * Lifecycle.claimsPerSegment := by
  have exact := chain.memory.exactClaimCount
  simpa [FullClaimGlobalFPrime.verifiedReceipts, receipts] using exact

/-- The producer trace and consumer trace have the same exact lifetime
length. -/
theorem exactProducerCount
    {widths : CompilerWidths}
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Header : Type uHeader}
    {Running : Type uRunning}
    {NifsProof : Type uNifsProof}
    {Nebula : Type}
    {NebulaDigest : Type uNebulaDigest}
    {NebulaOpen : Type uNebulaOpen}
    {configuration : Configuration Params StructureDigest Header Digest.Value
      Running (Value widths) NifsProof Nebula NebulaDigest NebulaOpen}
    {selected : SelectedVerifier widths}
    {initial final : Nightstream.Protocol.NebulaV2.FPrime.ClosedCarry
      Digest.Value}
    {segmentCount : Nat}
    (chain : ExactChain configuration selected initial final segmentCount) :
    chain.restInvocations.length + 1 =
      segmentCount * Lifecycle.claimsPerSegment := by
  rw [chain.trace.exactProducerCount, chain.exactConsumerCount]

/-- Positive segment count gives the exact base/recursive/terminal schedule. -/
theorem completeDelayedSchedule
    {widths : CompilerWidths}
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Header : Type uHeader}
    {Running : Type uRunning}
    {NifsProof : Type uNifsProof}
    {Nebula : Type}
    {NebulaDigest : Type uNebulaDigest}
    {NebulaOpen : Type uNebulaOpen}
    {configuration : Configuration Params StructureDigest Header Digest.Value
      Running (Value widths) NifsProof Nebula NebulaDigest NebulaOpen}
    {selected : SelectedVerifier widths}
    {initial final : Nightstream.Protocol.NebulaV2.FPrime.ClosedCarry
      Digest.Value}
    {segmentCount : Nat}
    (chain : ExactChain configuration selected initial final segmentCount) :
    Lifecycle.CompleteSchedule chain.consumers.length := by
  rw [chain.exactConsumerCount]
  exact Lifecycle.completeSchedule
    (Nat.mul_pos chain.positiveSegments (by decide))

/-- All local producers, including the trailing one, are closed in exact
delayed order. -/
def closedTrace
    {widths : CompilerWidths}
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Header : Type uHeader}
    {Running : Type uRunning}
    {NifsProof : Type uNifsProof}
    {Nebula : Type}
    {NebulaDigest : Type uNebulaDigest}
    {NebulaOpen : Type uNebulaOpen}
    {configuration : Configuration Params StructureDigest Header Digest.Value
      Running (Value widths) NifsProof Nebula NebulaDigest NebulaOpen}
    {selected : SelectedVerifier widths}
    {initial final : Nightstream.Protocol.NebulaV2.FPrime.ClosedCarry
      Digest.Value}
    {segmentCount : Nat}
    (chain : ExactChain configuration selected initial final segmentCount) :
    Nightstream.Protocol.FPrime.DelayedTrace.Closed configuration chain.first
      chain.restInvocations :=
  chain.trace.closed

/-- Every producer after the first is forced onto the recursive branch by
exact adjacent-state identity. This theorem needs no branch premise. -/
theorem tailProducersAreRecursive
    {widths : CompilerWidths}
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Header : Type uHeader}
    {Running : Type uRunning}
    {NifsProof : Type uNifsProof}
    {Nebula : Type}
    {NebulaDigest : Type uNebulaDigest}
    {NebulaOpen : Type uNebulaOpen}
    {configuration : Configuration Params StructureDigest Header Digest.Value
      Running (Value widths) NifsProof Nebula NebulaDigest NebulaOpen}
    {selected : SelectedVerifier widths}
    {initial final : Nightstream.Protocol.NebulaV2.FPrime.ClosedCarry
      Digest.Value}
    {segmentCount : Nat}
    (chain : ExactChain configuration selected initial final segmentCount) :
    ∀ invocation ∈ chain.restInvocations, invocation.IsRecursive :=
  chain.trace.toDelayed.rest_isRecursive

/-- The complete branch schedule follows from the canonical initial proof tag
and the local F-prime relation. The structure does not store `IsBase` or the
`noFold` conclusion. -/
theorem exactBranchSchedule
    {widths : CompilerWidths}
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Header : Type uHeader}
    {Running : Type uRunning}
    {NifsProof : Type uNifsProof}
    {Nebula : Type}
    {NebulaDigest : Type uNebulaDigest}
    {NebulaOpen : Type uNebulaOpen}
    {configuration : Configuration Params StructureDigest Header Digest.Value
      Running (Value widths) NifsProof Nebula NebulaDigest NebulaOpen}
    {selected : SelectedVerifier widths}
    {initial final : Nightstream.Protocol.NebulaV2.FPrime.ClosedCarry
      Digest.Value}
    {segmentCount : Nat}
    (chain : ExactChain configuration selected initial final segmentCount) :
    chain.first.IsBase ∧
      ∀ invocation ∈ chain.restInvocations, invocation.IsRecursive :=
  chain.trace.toDelayed.exactBranchSchedule
    (chain.first.isBase_of_prior_initial chain.firstPriorInitial)

/-- Complete-claim authority follows from the same paired trace, modulo the
two explicit Poseidon2 state-hash collision events. -/
theorem authoritySoundOrCollision
    {widths : CompilerWidths}
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Header : Type uHeader}
    {Running : Type uRunning}
    {NifsProof : Type uNifsProof}
    {Nebula : Type}
    {NebulaDigest : Type uNebulaDigest}
    {NebulaOpen : Type uNebulaOpen}
    {configuration : Configuration Params StructureDigest Header Digest.Value
      Running (Value widths) NifsProof Nebula NebulaDigest NebulaOpen}
    {selected : SelectedVerifier widths}
    {initial final : Nightstream.Protocol.NebulaV2.FPrime.ClosedCarry
      Digest.Value}
    {segmentCount : Nat}
    (chain : ExactChain configuration selected initial final segmentCount) :
    Exact selected chain.base.authority chain.consumers ∨
      StateAuthorityBoundaryRows.Failure :=
  chain.trace.authoritySoundOrCollision

/-- The initial carry and first segment opening are derived from the same
base manifest that supplies the first producer authority. -/
theorem baseAuthority
    {widths : CompilerWidths}
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Header : Type uHeader}
    {Running : Type uRunning}
    {NifsProof : Type uNifsProof}
    {Nebula : Type}
    {NebulaDigest : Type uNebulaDigest}
    {NebulaOpen : Type uNebulaOpen}
    {configuration : Configuration Params StructureDigest Header Digest.Value
      Running (Value widths) NifsProof Nebula NebulaDigest NebulaOpen}
    {selected : SelectedVerifier widths}
    {initial final : Nightstream.Protocol.NebulaV2.FPrime.ClosedCarry
      Digest.Value}
    {segmentCount : Nat}
    (chain : ExactChain configuration selected initial final segmentCount) :
    InitialMemoryCarryRows.Exact chain.base.call.initialValue
        chain.base.call.initialMemoryRoot ∧
      chain.base.artifact.MatchesSelected selected :=
  ⟨chain.base.initialExact, chain.baseMatchesSelected⟩

end ExactChain

end Nightstream.Implementation.NebulaV2.FullClaimManifestTrace
