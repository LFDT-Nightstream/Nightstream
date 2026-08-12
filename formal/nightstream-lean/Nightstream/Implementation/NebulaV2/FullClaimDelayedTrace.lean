import Nightstream.Implementation.NebulaV2.FullClaimFreshLink
import Nightstream.Protocol.FPrime.DelayedTrace

/-!
Contract: exact factor-one V2 specialization of delayed F-prime trace closure.

Assurance tier: implementation model.

Owns the fixed V2 fresh-link function over a complete full-claim envelope,
one exact singleton producer view, and conversion of the generic delayed
trace conclusion into the 540-coordinate authority-bearing carrier relation.

Does not own generated producer-side rows, placement of a manifest's outgoing
state columns into the generic F-prime invocation, NIFS extraction, terminal
backend verification, or Rust conformance.

The producer view assumes only exact dataflow: the singleton claim installed
by the local invocation, equality of its row-derived `x_out` with the typed
authority digest, and selection of the fixed executable V2 link function. It
does not assume `Carries`, typed-state equality, or verifier acceptance.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.NebulaV2.FullClaimDelayedTrace

open Nightstream.Implementation.NebulaV2.FullClaimEnvelope
open Nightstream.Protocol.FPrime
open Nightstream.Protocol.FPrime.DelayedTrace
open Nightstream.Protocol.NebulaV2

universe uParams uStructure uHeader uRunning uNifsProof uNebulaDigest
  uNebulaOpen

/-- The only delayed fresh-public function admitted by V2. It unwraps the
canonical four-lane protocol digest and runs the complete 540-coordinate
claim-dependent check. -/
def check {widths : CompilerWidths}
    (digest : Digest.Value) (claim : Value widths) : Bool :=
  FullClaimFreshLink.check digest.lanes claim

theorem check_authority_eq_true_iff_carries
    {widths : CompilerWidths}
    {authority : StateAuthorityBoundaryRows.Authority}
    {claim : Value widths} :
    check (StateAuthorityFullClaim.digestValue authority) claim = true ↔
      StateAuthorityFullClaim.Carries authority claim := by
  exact FullClaimFreshLink.check_authority_eq_true_iff_carries

theorem singleton_freshLinked_iff_carries
    {widths : CompilerWidths}
    {authority : StateAuthorityBoundaryRows.Authority}
    {claim : Value widths} :
    Step.FreshLinked check (StateAuthorityFullClaim.digestValue authority)
        [claim] ↔
      StateAuthorityFullClaim.Carries authority claim := by
  simpa [check, StateAuthorityFullClaim.digestValue] using
    (FullClaimFreshLink.singleton_freshLinked_iff_carries
      (authority := authority) (claim := claim))

/-- One local producer interpreted as the exact factor-one V2 producer. The
complete claim is a type parameter, so a consumer cannot later substitute a
different claim by supplying an equality proof. The three fields are direct
dataflow obligations for a generated artifact. -/
structure Producer
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
    (authority : StateAuthorityBoundaryRows.Authority)
    (claim : Value widths) where
  invocation : Invocation configuration
  singleton : invocation.input.nextLatest = [claim]
  selectedLink : configuration.stepSemantics.freshLink = check
  exactXOut : invocation.proof.xOut =
    StateAuthorityFullClaim.digestValue authority

namespace Producer

/-- An exact complete-claim carrier constructs the producer's delayed link.
This is the converse of `carries_of_outgoing` and uses only the producer's
three exact dataflow fields. -/
theorem outgoing_of_carries
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
    {authority : StateAuthorityBoundaryRows.Authority}
    {claim : Value widths}
    (producer : Producer configuration authority claim)
    (carries : StateAuthorityFullClaim.Carries authority claim) :
    producer.invocation.OutgoingLinked := by
  unfold Invocation.OutgoingLinked Step.OutgoingLinked
  rw [producer.selectedLink, producer.singleton, producer.exactXOut]
  exact singleton_freshLinked_iff_carries.mpr carries

/-- Delayed closure of this exact producer derives the full authority-bearing
carrier for its complete claim. -/
theorem carries_of_outgoing
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
    {authority : StateAuthorityBoundaryRows.Authority}
    {claim : Value widths}
    (producer : Producer configuration authority claim)
    (outgoing : producer.invocation.OutgoingLinked) :
    StateAuthorityFullClaim.Carries authority claim := by
  have linked := outgoing
  unfold Invocation.OutgoingLinked Step.OutgoingLinked at linked
  rw [producer.selectedLink, producer.singleton, producer.exactXOut] at linked
  exact singleton_freshLinked_iff_carries.mp linked

/-- The same result in the exact Boolean form required by a recursive or
terminal manifest edge. -/
theorem freshLinked_of_outgoing
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
    {authority : StateAuthorityBoundaryRows.Authority}
    {claim : Value widths}
    (producer : Producer configuration authority claim)
    (outgoing : producer.invocation.OutgoingLinked) :
    Step.FreshLinked FullClaimFreshLink.check
      (StateAuthorityFullClaim.canonicalDigest authority) [claim] :=
  FullClaimFreshLink.freshLinked_of_carries
    (producer.carries_of_outgoing outgoing)

end Producer

end Nightstream.Implementation.NebulaV2.FullClaimDelayedTrace
