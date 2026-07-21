import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.ActiveBoundary
import Nightstream.Implementation.R1CS.Correspondence.Gadgets.PiDecTypedCarrier
import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Authority.DelayedProduction

/-!
Exact active-result bridge for the decoded strict-`PiDEC` carrier.

Assurance tier: model-level.

Owns: the minimal family-payload seam between a typed strict-`PiDEC` source
view and the `FixedActive.resultOf` computed from one accepted certificate;
recovery of the exact parent and ordered children from that seam; and rewrite
of the existing outgoing delayed-state check onto the decoded result.

Does not own: generated column identity, Rust or selective-R1CS refinement,
assignment-to-certificate decoding, transcript refinement, commitment
binding, costs, or row removal. In particular, no current artifact proves
`ParentPointBound` or `ChildPayloadsBound` below.

The two deliberately exposed decoder obligations are exact and sufficient:

* the strict parent `r` columns decode to the derived `PiRLC` point;
* each strict child commitment/public-input/evaluation payload decodes to
  `certificate.piDecPayloads` at the same ordered child index.

The relation structure is verifier-owned and definitionally shared. Parent
commitment, public input, and evaluations need no duplicate decoder premise:
strict `PiDEC` acceptance makes them uniquely recoverable from the equal
accepted child family. Delayed `y_zcol` remains in
`DelayedProduction.outgoingPending`, computed from the same certificate; it
is not added to the paper CE carrier.

Emits constraints: no.

| Stage path | Mathematical obligation | Authority class |
|---|---|---|
| `nifs.pi_dec.result.decode` | decode the strict parent and fourteen children as one `FoldResult` | computed |
| `nifs.pi_dec.result.point` | identify the decoded parent point with the derived `PiRLC` point | open decoder boundary |
| `nifs.pi_dec.result.children` | identify every decoded child payload with the certificate payload | open decoder boundary |
| `nifs.pi_dec.result.family` | reduce exact result identity to the minimal family carrier | derived |
| `nifs.pi_dec.result.exact` | recover the exact `FixedActive.resultOf` | derived |
| `f_prime.state.out.decoded` | reuse the outgoing state check with that exact decoded result and unchanged pending sidecar | derived |
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Nifs.PiDec.ActiveResultBridge

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.ChildPayloadAuthority
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier
open Nightstream.Protocol.FPrime.ConcretePhi81
open Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics
open Nightstream.Protocol.FPrime.ConcretePhi81.ActiveEvaluator
open Nightstream.Protocol.FPrime.ConcretePhi81.AccumulatorBinding.DelayedPending
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual
open Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual.CombinedNc
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.PackedWitness

universe uState

variable {shape : SemanticShape}
variable {State : Type uState}
variable {publicRingColumns verifierRows : Nat}
variable {publicFits :
  ringDegree * publicRingColumns <= shape.carrierWidth}
variable {layout : PiDecStrictCompiler.Layout}

private theorem familyPayload_eq_of_fields
    {left right :
      FamilyPayload (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)}
    (constraintSystem :
      left.constraintSystem = right.constraintSystem)
    (point : left.point = right.point)
    (children : left.children = right.children) :
    left = right := by
  cases left
  cases right
  cases constraintSystem
  cases point
  cases children
  rfl

private theorem foldResult_eq_of_fields
    {left right :
      FixedActive.FoldResult shape publicRingColumns publicFits verifierRows}
    (parent : left.parent = right.parent)
    (children : left.children = right.children) :
    left = right := by
  cases left
  cases right
  cases parent
  cases children
  rfl

/-- One strict source assignment decoded directly into the result type used by
the active NIFS lifecycle. Sidecars do not enter this value. -/
def decodedFoldResult
    (profile : PiDecTypedCarrier.Profile
      (RelationShape shape publicRingColumns publicFits) verifierRows layout)
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (assignment : Nat -> Nat) :
    FixedActive.FoldResult shape publicRingColumns publicFits verifierRows := {
  parent := PiDecTypedCarrier.decodedParent profile context.system assignment
  children := PiDecTypedCarrier.decodedOutput profile context.system assignment
}

/-- First missing physical decoder fact: the strict parent point is the point
computed by the same certificate's `PiRLC` execution. -/
def ParentPointBound
    (profile : PiDecTypedCarrier.Profile
      (RelationShape shape publicRingColumns publicFits) verifierRows layout)
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (assignment : Nat -> Nat)
    (certificate : FixedActive.Certificate context) : Prop :=
  (decodedFoldResult profile context assignment).parent.point =
    (derive context certificate).piRlcOutput.point

/-- Second missing physical decoder fact: the three child-specific paper
fields are exactly the ordered payloads read by the active certificate. -/
def ChildPayloadsBound
    (profile : PiDecTypedCarrier.Profile
      (RelationShape shape publicRingColumns publicFits) verifierRows layout)
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (assignment : Nat -> Nat)
    (certificate : FixedActive.Certificate context) : Prop :=
  forall child,
    PiDecChildPayload.ofStatement
        ((decodedFoldResult profile context assignment).children child) =
      certificate.piDecPayloads child

/-- Minimal exact family equality needed to compare two accepted `PiDEC`
views. It carries the shared structure and point once and only the three
child-specific paper fields thereafter. -/
def FamilyPayloadBound
    (profile : PiDecTypedCarrier.Profile
      (RelationShape shape publicRingColumns publicFits) verifierRows layout)
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (assignment : Nat -> Nat)
    (certificate : FixedActive.Certificate context) : Prop :=
  familyPayload
      (decodedFoldResult profile context assignment).parent
      (decodedFoldResult profile context assignment).children =
    familyPayload
      (FixedActive.resultOf context certificate).parent
      (FixedActive.resultOf context certificate).children

private theorem decodedFoldResult_parent_constraintSystem
    (profile : PiDecTypedCarrier.Profile
      (RelationShape shape publicRingColumns publicFits) verifierRows layout)
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (assignment : Nat -> Nat) :
    (decodedFoldResult profile context assignment).parent.constraintSystem =
      context.system := by
  rfl

private theorem resultOf_parent_constraintSystem
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (certificate : FixedActive.Certificate context) :
    (FixedActive.resultOf context certificate).parent.constraintSystem =
      context.system := by
  rfl

private theorem resultOf_child_payload
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (certificate : FixedActive.Certificate context)
    (child : Fin productionGlobalParams.k) :
    PiDecChildPayload.ofStatement
        ((FixedActive.resultOf context certificate).children child) =
      certificate.piDecPayloads child := by
  rfl

/-- The two missing decoder facts are exactly sufficient for equality of the
minimal family carriers. No parent commitment, public-input, evaluation, or
implementation-sidecar equality is assumed. -/
theorem familyPayloadBound_of_decoderFacts
    (profile : PiDecTypedCarrier.Profile
      (RelationShape shape publicRingColumns publicFits) verifierRows layout)
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (assignment : Nat -> Nat)
    (certificate : FixedActive.Certificate context)
    (pointBound : ParentPointBound profile context assignment certificate)
    (payloadsBound : ChildPayloadsBound profile context assignment certificate) :
    FamilyPayloadBound profile context assignment certificate := by
  unfold FamilyPayloadBound
  apply familyPayload_eq_of_fields
  · exact (decodedFoldResult_parent_constraintSystem profile context
      assignment).trans
        (resultOf_parent_constraintSystem context certificate).symm
  · exact pointBound
  · unfold familyPayload payloadList
    dsimp only
    apply congrArg List.ofFn
    funext child
    calc
      PiDecChildPayload.ofStatement
          ((decodedFoldResult profile context assignment).children child) =
          certificate.piDecPayloads child := payloadsBound child
      _ = PiDecChildPayload.ofStatement
          ((FixedActive.resultOf context certificate).children child) := by
        exact (resultOf_child_payload context certificate child).symm

/-- Equality of the minimal exact family carrier plus acceptance on both sides
recovers the complete computed fold result. The decoded side obtains ordinary
recomposition acceptance from the stronger paper verifier; the physical side
uses the retained tail acceptance of the same active certificate. -/
theorem decodedFoldResult_eq_resultOf
    (profile : PiDecTypedCarrier.Profile
      (RelationShape shape publicRingColumns publicFits) verifierRows layout)
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (assignment : Nat -> Nat)
    (certificate : FixedActive.Certificate context)
    (typedAccepted : PiDecTypedCarrier.Accepted profile context.key
      context.system assignment)
    (tailAccepted : TailAccepted context certificate)
    (familyBound : FamilyPayloadBound profile context assignment certificate) :
    decodedFoldResult profile context assignment =
      FixedActive.resultOf context certificate := by
  have decodedAccepted :
      PiDEC.Accepted (decAlgebra context.key) {
        parent := (decodedFoldResult profile context assignment).parent
        children := (decodedFoldResult profile context assignment).children
      } := by
    change PiDEC.Accepted (decAlgebra context.key) {
      parent := PiDecTypedCarrier.decodedParent profile context.system assignment
      children := PiDecTypedCarrier.decodedOutput profile context.system assignment
    }
    exact (PiDecTypedCarrier.accepted_refines_paper profile context.key
      context.system assignment typedAccepted).toRecompositionAccepted
  have physicalAccepted :
      PiDEC.Accepted (decAlgebra context.key) {
        parent := (FixedActive.resultOf context certificate).parent
        children := (FixedActive.resultOf context certificate).children
      } := by
    change PiDEC.Accepted (decAlgebra context.key)
      ((derive context certificate).piDecAttempt certificate)
    exact tailAccepted.piDec
  have exactFields := parent_children_eq_of_familyPayload_eq
    (by decide : 0 < productionGlobalParams.k)
    decodedAccepted physicalAccepted familyBound
  apply foldResult_eq_of_fields
  · exact exactFields.1
  · exact exactFields.2

/-! ## Active lifecycle specialization -/

universe uOuterKey uAppState uWitness uDigest uTranscriptState uEncoding

variable {OuterKey : Type uOuterKey}
variable {AppState : Type uAppState}
variable {Witness : Type uWitness}
variable {Digest : Type uDigest}
variable {TranscriptState : Type uTranscriptState}
variable {Encoding : Type uEncoding}

/-- The public active claims checker supplies the physical tail acceptance;
the two explicit decoder facts then identify the strict decoded result with
the lifecycle's verifier-computed result. -/
theorem claimsAccepted_decodedFoldResult_eq_resultOf
    [DecidableEq Digest]
    (scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (PendingFamilyPayload
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) Encoding Digest)
    (incomingStateDigest outgoingStateDigest : Digest)
    (machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows 1)
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows 1)
    (input :
      ProductionContext.Input OuterKey AppState Witness shape
        publicRingColumns publicFits verifierRows)
    (template : Data shape)
    (witnesses : Fin shape.runningCount -> Matrix shape)
    (certificate : FixedActive.Certificate
      (ProductionContext.full setup input))
    (profile : PiDecTypedCarrier.Profile
      (RelationShape shape publicRingColumns publicFits) verifierRows layout)
    (assignment : Nat -> Nat)
    (claimsAccepted : ActiveBoundary.ClaimsAccepted scheme
      incomingStateDigest outgoingStateDigest machine setup input template
      witnesses certificate)
    (typedAccepted : PiDecTypedCarrier.Accepted profile
      (ProductionContext.full setup input).key
      (ProductionContext.full setup input).system assignment)
    (pointBound : ParentPointBound profile
      (ProductionContext.full setup input) assignment certificate)
    (payloadsBound : ChildPayloadsBound profile
      (ProductionContext.full setup input) assignment certificate) :
    decodedFoldResult profile (ProductionContext.full setup input) assignment =
      ActiveBoundary.resultOf setup input certificate := by
  have messageAccepted : ProductionNifs.MessageAccepted
      (ProductionContext.full setup input) certificate :=
    (PackedWitnessProduction.messageCheck_eq_true_iff_accepted
      (ProductionContext.canonical setup input) certificate).mp
        claimsAccepted.nifs
  exact decodedFoldResult_eq_resultOf profile
    (ProductionContext.full setup input) assignment certificate typedAccepted
    messageAccepted.tail
    (familyPayloadBound_of_decoderFacts profile
      (ProductionContext.full setup input) assignment certificate pointBound
      payloadsBound)

/-- The existing outgoing state check can be read over the exact decoded
paper result. The delayed block-lane value remains the unchanged
certificate-computed `outgoingPending`, so all three surfaces come from one
accepted transition. -/
theorem claimsAccepted_outgoingState_rewrite
    [DecidableEq Digest]
    (scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (PendingFamilyPayload
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) Encoding Digest)
    (incomingStateDigest outgoingStateDigest : Digest)
    (machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows 1)
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows 1)
    (input :
      ProductionContext.Input OuterKey AppState Witness shape
        publicRingColumns publicFits verifierRows)
    (template : Data shape)
    (witnesses : Fin shape.runningCount -> Matrix shape)
    (certificate : FixedActive.Certificate
      (ProductionContext.full setup input))
    (profile : PiDecTypedCarrier.Profile
      (RelationShape shape publicRingColumns publicFits) verifierRows layout)
    (assignment : Nat -> Nat)
    (claimsAccepted : ActiveBoundary.ClaimsAccepted scheme
      incomingStateDigest outgoingStateDigest machine setup input template
      witnesses certificate)
    (typedAccepted : PiDecTypedCarrier.Accepted profile
      (ProductionContext.full setup input).key
      (ProductionContext.full setup input).system assignment)
    (pointBound : ParentPointBound profile
      (ProductionContext.full setup input) assignment certificate)
    (payloadsBound : ChildPayloadsBound profile
      (ProductionContext.full setup input) assignment certificate) :
    ProductionChecker.stateBindingCheck scheme outgoingStateDigest
        (decodedFoldResult profile
          (ProductionContext.full setup input) assignment).parent
        (decodedFoldResult profile
          (ProductionContext.full setup input) assignment).children
        (some (Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Authority.DelayedProduction.outgoingPending
          (ProductionContext.full setup input) certificate)) = true := by
  have resultEq := claimsAccepted_decodedFoldResult_eq_resultOf scheme
    incomingStateDigest outgoingStateDigest machine setup input template
    witnesses certificate profile assignment claimsAccepted typedAccepted
    pointBound payloadsBound
  rw [resultEq]
  exact claimsAccepted.outgoingState

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Nifs.PiDec.ActiveResultBridge
