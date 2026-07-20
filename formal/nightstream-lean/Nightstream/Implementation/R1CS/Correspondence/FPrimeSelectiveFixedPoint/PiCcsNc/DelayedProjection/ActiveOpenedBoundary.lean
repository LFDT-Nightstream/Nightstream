import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.ActiveBoundary
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.PackedWitnessCommitment

/-!
Active delayed-projection edge with actual packed-matrix openings.

Assurance tier: model-level with artifact-checked matrix geometry.  Native
combined-NC extraction, key serialization, and commitment binding remain
explicit boundaries.

Owns: elimination of the successor raw-child commitment premise from one
active delayed edge.  The exact fourteen `matrixCommit` equations derive that
premise from the same packed matrices consumed by combined NC.

Does not own: Rust transcript integration, extraction of those matrices from
SumCheck acceptance, the predecessor canonical-parent opening, Ajtai/MSIS
hardness, `y_ring`, costs, or row-removal permission.

Emits constraints: none; active correspondence theorem only.

| Stable stage path | Obligation | Authority class |
|---|---|---|
| `f_prime.pi_ccs_nc.delayed.active.opened_children` | actual successor matrices open the fourteen public running commitments | typed external boundary |
| `f_prime.pi_ccs_nc.delayed.active.opened_edge` | the opened successor closes the predecessor packed equation or a named event | derived/security partition |
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.ActiveOpenedBoundary

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Authority
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane
open Nightstream.Protocol.FPrime.ConcretePhi81
open Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics
open Nightstream.Protocol.FPrime.ConcretePhi81.ActiveEvaluator
open Nightstream.Protocol.FPrime.ConcretePhi81.AccumulatorBinding.DelayedPending
open Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual
open Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual.CombinedNc
open PackedWitness

universe uOuterKey uAppState uWitness uDigest uTranscriptState uEncoding

variable {OuterKey : Type uOuterKey}
variable {AppState : Type uAppState}
variable {Witness : Type uWitness}
variable {Digest : Type uDigest}
variable {TranscriptState : Type uTranscriptState}
variable {Encoding : Type uEncoding}
variable {shape : SemanticShape}
variable {publicRingColumns verifierRows : Nat}
variable {publicFits :
  ringDegree * publicRingColumns <= shape.carrierWidth}

/-- Binding failures still external after the actual successor matrices have
discharged raw-child commitment authority. -/
inductive BindingFailure
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows 1)
    (previousInput nextInput :
      ProductionContext.Input OuterKey AppState Witness shape
        publicRingColumns publicFits verifierRows)
    (previousTemplate : Data shape)
    (previousWitnesses : Fin shape.runningCount -> Matrix shape)
    (previousCertificate : FixedActive.Certificate
      (ProductionContext.full setup previousInput)) : Prop where
  | canonicalParent
      (failure :
        ¬ DelayedRawChildren.CanonicalParentBinding
          (ProductionContext.full setup previousInput)
          (decodedData previousTemplate previousWitnesses)
          previousCertificate) :
      BindingFailure setup previousInput nextInput previousTemplate
        previousWitnesses previousCertificate
  | verifierKeyContinuity
      (failure : (ProductionContext.full setup nextInput).key ≠
        (ProductionContext.full setup previousInput).key) :
      BindingFailure setup previousInput nextInput previousTemplate
        previousWitnesses previousCertificate

/-- One active claims-level edge with actual successor matrix openings.

Unlike `ActiveBoundary.ParentOpeningActiveBindingFailure`, the result has no
raw-child commitment-mismatch constructor: `nextOpened` derives that fact
from the full packed witnesses before the delayed theorem is invoked. -/
theorem claimsAcceptedPair_of_nextPacked_of_openedPackedWitnesses_implies_previousPacked_or_namedFailure
    [DecidableEq Digest]
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    (scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (PendingFamilyPayload
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) Encoding Digest)
    (previousIncomingDigest sharedStateDigest nextOutgoingDigest : Digest)
    (machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows 1)
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows 1)
    (previousInput :
      ProductionContext.Input OuterKey AppState Witness shape
        publicRingColumns publicFits verifierRows)
    (previousTemplate : Data shape)
    (previousWitnesses : Fin shape.runningCount -> Matrix shape)
    (previousCertificate : FixedActive.Certificate
      (ProductionContext.full setup previousInput))
    (previousAccepted : ActiveBoundary.ClaimsAccepted scheme
      previousIncomingDigest sharedStateDigest machine setup previousInput
      previousTemplate previousWitnesses previousCertificate)
    (nextInput :
      ProductionContext.Input OuterKey AppState Witness shape
        publicRingColumns publicFits verifierRows)
    (nextTemplate : Data shape)
    (nextWitnesses : Fin shape.runningCount -> Matrix shape)
    (nextCertificate : FixedActive.Certificate
      (ProductionContext.full setup nextInput))
    (nextAccepted : ActiveBoundary.ClaimsAccepted scheme sharedStateDigest
      nextOutgoingDigest machine setup nextInput nextTemplate nextWitnesses
      nextCertificate)
    (nextPacked : Terminal.PackedYZcolBoundAtBlock
      (ProductionContext.full setup nextInput).covers
      (decodedData nextTemplate nextWitnesses)
      (ProductionPiCcs.ncPoint (ProductionContext.full setup nextInput)
        nextCertificate).block nextCertificate.piCcs.output)
    (nextOpened : forall child,
      PackedWitnessCommitment.matrixCommit
          (ProductionContext.full setup nextInput).key
          (nextWitnesses
            ((ProductionContext.full setup nextInput).alignment.semanticRunningIndex
              child)) =
        ((ProductionContext.full setup nextInput).input.running child).commitment) :
    Terminal.PackedYZcolBoundAtBlock
        (ProductionContext.full setup previousInput).covers
        (decodedData previousTemplate previousWitnesses)
        (ProductionPiCcs.ncPoint (ProductionContext.full setup previousInput)
          previousCertificate).block previousCertificate.piCcs.output ∨
      ProductionBoundary.ParentOpeningRecursiveBadEvent scheme
        (ProductionContext.full setup previousInput)
        (decodedData previousTemplate previousWitnesses) previousCertificate
        (ProductionContext.full setup nextInput)
        (decodedData nextTemplate nextWitnesses) nextCertificate ∨
      BindingFailure setup previousInput nextInput previousTemplate
        previousWitnesses previousCertificate := by
  by_cases sameKey : (ProductionContext.full setup nextInput).key =
      (ProductionContext.full setup previousInput).key
  · rcases
        PackedWitnessProduction.messageCheckedPair_of_nextPacked_of_openedPackedWitnesses_implies_previousPacked_or_parentBindingFailure
          noZeroDivisors scheme sharedStateDigest
          (ProductionContext.canonical setup previousInput) previousTemplate
          previousWitnesses previousCertificate previousAccepted.nifs
          previousAccepted.outgoingState
          (ProductionContext.canonical setup nextInput) nextTemplate
          nextWitnesses nextCertificate nextAccepted.nifs nextPacked
          nextAccepted.incomingState sameKey nextOpened with
      packed | bad | parentFailure
    · exact Or.inl (by
        simpa [ProductionPiCcs.ncPoint] using packed)
    · exact Or.inr (Or.inl bad)
    · exact Or.inr (Or.inr (.canonicalParent parentFailure))
  · exact Or.inr (Or.inr (.verifierKeyContinuity sameKey))

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.ActiveOpenedBoundary
