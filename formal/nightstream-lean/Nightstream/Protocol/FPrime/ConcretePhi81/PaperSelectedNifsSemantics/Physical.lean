import Nightstream.Protocol.FPrime.ConcretePhi81.PaperSelectedNifsSemantics
import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive.PaperProfile.PhysicalTransition

/-!
Physical projection into the paper-exact selected NIFS family.

Assurance tier: model-level.

Owns: the thin public-family projection of fixed-active physical paper
soundness for the actual verifier-visible child vector.

Does not own: generated rows, production column binding, deterministic child
openings, canonical private child equality, Rust refinement, costs, or row
removal.

Authority boundary: the theorem preserves the physical theorem's exact
`yRing` equation failure and typed `PiCcsBadEvent`. It adds only the internal
context witnesses required to hide physical verifier state behind the public
Construction-2 edge.

Emits constraints: no.
-/

namespace Nightstream.Protocol.FPrime.ConcretePhi81.PaperSelectedNifsSemantics.Physical

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.Nifs
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive.PaperProfile.PhysicalTransition
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier

universe uOuterKey uTranscriptState

variable {OuterKey : Type uOuterKey}
variable {TranscriptState : Type uTranscriptState}
variable {shape : SemanticShape}
variable {publicRingColumns verifierRows slotCount : Nat}
variable {publicFits :
  ringDegree * publicRingColumns <= shape.carrierWidth}
variable {setup : Setup OuterKey TranscriptState shape publicRingColumns
  publicFits verifierRows slotCount}
variable {key : OuterKey}
variable {slot : Fin slotCount}
variable {source : Source shape publicRingColumns publicFits verifierRows}
variable {incomingParent :
  Phi81Relation.CEStatement
    (RelationShape shape publicRingColumns publicFits)
    (CommitmentValue verifierRows)}
variable {polynomial : PiCCS.SplitNc.Verifier.PublicInput shape}
variable {priorState : TranscriptState}

local notation "selectedContext" =>
  contextOf setup key slot source incomingParent polynomial priorState

/-- Physical acceptance reaches the paper-exact public child edge, or exposes
exactly the remaining `yRing` equation failure or typed `Pi_CCS` bad event.
The selected target is the actual physical `Pi_DEC` output family. -/
theorem transition_or_yRingUnbound_or_badEvent_of_physical
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    (data : Data shape)
    (certificate : FixedActive.Certificate selectedContext)
    (input : SemanticInput selectedContext data)
    (canonicalPublicInput : forall child,
      (outputChildren selectedContext certificate child).publicInput =
        (FixedActive.PaperProfile.decPublicInputSplit
          (FixedActive.paperProfileOf selectedContext)).split
          (derive selectedContext certificate).piRlcOutput.publicInput child)
    (parentEvaluationSize :
      (derive selectedContext certificate).piRlcOutput.evaluations.size =
        (FixedActive.PaperProfile.decEvaluationArity
          (FixedActive.paperProfileOf selectedContext)).count
          (derive selectedContext certificate).piRlcOutput.constraintSystem)
    (childEvaluationSize : forall child,
      (outputChildren selectedContext certificate child).evaluations.size =
        (FixedActive.PaperProfile.decEvaluationArity
          (FixedActive.paperProfileOf selectedContext)).count
          (derive selectedContext certificate).piRlcOutput.constraintSystem)
    (packed :
      Polynomial.Nc.BlockLane.Terminal.PackedYZcolBoundAtBlock
        (selectedContext).covers data
        (derive selectedContext certificate).piCcs.ncPoint.block
        certificate.piCcs.output)
    (accepted : Accepted selectedContext certificate) :
    Transition setup key slot source
        (outputChildren selectedContext certificate) ∨
      ¬ certificate.piCcs.output.yRing =
        Polynomial.Fe.sourceYRingAt data
          (derive selectedContext certificate).piCcs.fePoint.row ∨
      PiCcsBadEvent selectedContext data certificate := by
  rcases
      accepted_implies_transition_or_yRingUnbound_or_badEvent_of_packedYZcolBound
          noZeroDivisors selectedContext data certificate input
          canonicalPublicInput parentEvaluationSize childEvaluationSize packed
          accepted with
    paper | yRingUnbound | bad
  · exact Or.inl
      (transition_of_paper (incomingParent := incomingParent)
        (polynomial := polynomial) (priorState := priorState) paper)
  · exact Or.inr (Or.inl yRingUnbound)
  · exact Or.inr (Or.inr bad)

end Nightstream.Protocol.FPrime.ConcretePhi81.PaperSelectedNifsSemantics.Physical
