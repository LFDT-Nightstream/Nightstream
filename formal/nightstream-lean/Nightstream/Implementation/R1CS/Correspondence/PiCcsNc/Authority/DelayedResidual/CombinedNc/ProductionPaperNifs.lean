import Nightstream.Implementation.R1CS.Correspondence.PiCcsNc.Authority.DelayedResidual.CombinedNc.ProductionNifs
import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Authority.DelayedRawChildren
import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive.CanonicalOpening.SourceInput
import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive.PaperProfile.PhysicalTransition

/-!
Paper-exact production NIFS refinement with delayed packed projection.

Protocol: SuperNeo Sections 7.3--7.5.
Phase: production raw-source `Pi_CCS -> Pi_RLC -> Pi_DEC`.
Constraint family: typed semantic composition only; this file emits no rows.

Assurance tier: model-level, pending the three exact `Pi_DEC` shape facts from
the production checker/artifact.

Owns: the minimal positive result of raw production acceptance; its partition
from exact `yRing` and `Pi_CCS` failure outcomes; and promotion to the
independent paper transition over the actual public children once delayed
packed authority is supplied.

Does not own: running/lifecycle sidecars, child CE openings, deterministic
private child splitting, generated rows, Rust refinement, costs, or row
removal.

Authority boundary: raw `Pi_CCS` reads authoritative `Sources.Data`. The
positive record retains no running field and no generic refinement predicate.
The delayed packed equation is combined with the exact `yRing` equation only
at promotion; it is never treated as a child sidecar or digest authority.
-/

namespace Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual.CombinedNc.ProductionPaperNifs

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Authority
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane

universe uState

variable {shape : SemanticShape}
variable {State : Type uState}
variable {publicRingColumns verifierRows : Nat}
variable {publicFits :
  ringDegree * publicRingColumns <= shape.carrierWidth}

/-- Public, executable obligations for one opening-derived paper step.

The incoming running family is deliberately absent.  Once the step's raw NC
truth is known, it is reconstructed from `carrier.opening` and proved by
`SourceInput.Carrier.childSourcesValid_of_ncTruth`. -/
structure PaperStepAccepted
    (carrier : FixedActive.CanonicalOpening.SourceInput.Carrier shape
      publicRingColumns publicFits)
    (context : FixedActive.CanonicalOpening.Context shape State
      publicRingColumns publicFits verifierRows)
    (certificate : FixedActive.Certificate (carrier.install context).full) :
    Prop where
  piCcs : ProductionPiCcs.MessageAccepted
    (carrier.install context).full certificate
  sampler : Sampler.CertificateAccepted
    (carrier.install context).full certificate
  paperOutput : FixedActive.PaperProfile.PhysicalOutput.PaperOutputEquations
    (carrier.install context).full certificate
  canonicalParent : DelayedRawChildren.CanonicalParentBinding
    (carrier.install context).full carrier.data certificate

namespace PaperStepAccepted

/-- The operational paper-output equations imply ordinary public `PiDEC`
recomposition for the exact parent and child family in the certificate. -/
def piDecAccepted
    {carrier : FixedActive.CanonicalOpening.SourceInput.Carrier shape
      publicRingColumns publicFits}
    {context : FixedActive.CanonicalOpening.Context shape State
      publicRingColumns publicFits verifierRows}
    {certificate : FixedActive.Certificate (carrier.install context).full}
    (accepted : PaperStepAccepted carrier context certificate) :
    PiDEC.Accepted (decAlgebra (carrier.install context).full.key)
      ((derive (carrier.install context).full certificate).piDecAttempt
        certificate) := by
  exact
    ((FixedActive.PaperProfile.PhysicalOutput.paperOutputAccepted_iff_equations
      (carrier.install context).full certificate).2
      accepted.paperOutput).toRecompositionAccepted

end PaperStepAccepted

/-- Minimal positive production result needed by the independent paper
transition. The actual child family remains certificate-derived; no private
opening or canonical-child equality is retained. -/
structure PaperRefinement
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (data : Data shape)
    (certificate : FixedActive.Certificate context) : Prop where
  paper : Semantics.Paper.Holds data
  input : SemanticInput context data
  yRing : ProductionPiCcs.YRingBound context data certificate
  sampler : Sampler.CertificateAccepted context certificate
  paperOutput : FixedActive.PaperProfile.PhysicalOutput.PaperOutputEquations
    context certificate
  canonicalParent : DelayedRawChildren.CanonicalParentBinding context data
    certificate

namespace PaperRefinement

/-- Delayed packed authority completes the exact two-component physical output
binding. -/
def outputBound
    {context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows}
    {data : Data shape}
    {certificate : FixedActive.Certificate context}
    (refinement : PaperRefinement context data certificate)
    (packed : Terminal.PackedYZcolBoundAtBlock context.covers data
      (derive context certificate).piCcs.ncPoint.block
      certificate.piCcs.output) :
    OutputBound context data certificate :=
  ⟨refinement.yRing, packed⟩

/-- Packed delayed authority promotes the positive raw-production result to
the independent fixed-active paper transition for the actual public children.
No running-state or child-opening premise enters the paper target. -/
theorem toPaperTransition
    {context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows}
    {data : Data shape}
    {certificate : FixedActive.Certificate context}
    (refinement : PaperRefinement context data certificate)
    (packed : Terminal.PackedYZcolBoundAtBlock context.covers data
      (derive context certificate).piCcs.ncPoint.block
      certificate.piCcs.output) :
    FixedActive.PaperProfile.Transition (FixedActive.paperProfileOf context)
      context.input (outputChildren context certificate) := by
  let output := refinement.outputBound packed
  let semanticWitness := CertificateRefinement.semanticWitness certificate
  let witness := FixedActive.paperWitnessOf semanticWitness
  have systemEq :
      context.system = SemanticFold.systemOf context data := by
    simpa [Context.system, SemanticFold.systemOf] using
      (refinement.input.sources.fresh
        ⟨0, FixedActive.arity.freshPositive⟩).constraintSystem
  have outputsEq :
      (derive context certificate).piCcsOutputs =
        SemanticFold.outputs context data semanticWitness := by
    change
      OutputProduct.materialize publicRingColumns publicFits context.alignment
          context.input (derive context certificate).piCcs.fePoint.row
          certificate.piCcs.output =
        PiCCS.honestOutputs (semantics context.key) context.input
          (InputAuthority.productAssignments data context.alignment)
          (derive context certificate).piCcs.fePoint.row
    simpa [semantics] using
      (Protocol.OutputRefinement.materializedOutputs_eq_honestOutputs_of_yRingEq
        publicRingColumns publicFits (commit context.key) data
        context.alignment context.input
        (derive context certificate).piCcs.fePoint.row certificate.piCcs.output
        production_norm_stages.1 refinement.paper refinement.input.sources
        output.1)
  have parentEq :
      (derive context certificate).piRlcOutput =
        FixedActive.PaperProfile.parentOf (FixedActive.paperProfileOf context)
          context.input data witness := by
    change
      PiRLC.combinedOutput (rlcAlgebra context.key)
          context.system (derive context certificate).piCcs.fePoint.row
          (derive context certificate).piCcsOutputs
          certificate.piRlcChallenges =
        PiRLC.combinedOutput (rlcAlgebra context.key)
          (SemanticFold.systemOf context data)
          (derive context certificate).piCcs.fePoint.row
          (SemanticFold.outputs context data semanticWitness)
          certificate.piRlcChallenges
    rw [systemEq, outputsEq]
  have physicalPiDec :=
    (FixedActive.PaperProfile.PhysicalOutput.paperOutputAccepted_iff_equations
      context certificate).2 refinement.paperOutput
  have paperPiDec :
      PiDEC.PaperVerifier.OutputAccepted
        (FixedActive.PaperProfile.decAlgebra
          (FixedActive.paperProfileOf context))
        (FixedActive.PaperProfile.decPublicInputSplit
          (FixedActive.paperProfileOf context))
        (FixedActive.PaperProfile.decEvaluationArity
          (FixedActive.paperProfileOf context))
        (FixedActive.PaperProfile.parentOf (FixedActive.paperProfileOf context)
          context.input data witness)
        (outputChildren context certificate) := by
    rw [← parentEq]
    exact physicalPiDec
  exact ⟨data, witness, {
    paper := refinement.paper
    input := refinement.input.sources
    challengesValid := by
      simpa [witness, semanticWitness, FixedActive.paperWitnessOf] using
        (Sampler.certificateAccepted_challengesValid refinement.sampler)
    piDecAccepted := paperPiDec
  }⟩

end PaperRefinement

/-- Raw production acceptance partitions into the minimal positive paper
record, the exact current-step `yRing` mismatch, or the existing typed raw
`Pi_CCS` bad event. There is no output-unbound or implementation-refinement
branch. -/
theorem accepted_implies_paperRefinement_or_yRingUnbound_or_badEvent
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (data : Data shape)
    (certificate : FixedActive.Certificate context)
    (input : SemanticInput context data)
    (canonicalPublicInput : forall child,
      (outputChildren context certificate child).publicInput =
        (FixedActive.PaperProfile.decPublicInputSplit
          (FixedActive.paperProfileOf context)).split
          (derive context certificate).piRlcOutput.publicInput child)
    (parentEvaluationSize :
      (derive context certificate).piRlcOutput.evaluations.size =
        (FixedActive.PaperProfile.decEvaluationArity
          (FixedActive.paperProfileOf context)).count
          (derive context certificate).piRlcOutput.constraintSystem)
    (childEvaluationSize : forall child,
      (outputChildren context certificate child).evaluations.size =
        (FixedActive.PaperProfile.decEvaluationArity
          (FixedActive.paperProfileOf context)).count
          (derive context certificate).piRlcOutput.constraintSystem)
    (canonicalParent : DelayedRawChildren.CanonicalParentBinding context data
      certificate)
    (accepted : ProductionNifs.Accepted context data certificate) :
    PaperRefinement context data certificate ∨
      ProductionPiCcs.YRingUnbound context data certificate ∨
      ProductionPiCcs.BadEvent context data certificate := by
  rcases
      ProductionPiCcs.accepted_implies_paper_and_yRingBound_or_yRingUnbound_or_badEvent
        noZeroDivisors context data certificate input.publicInput
        accepted.piCcs with
    paper | yRingUnbound | bad
  · exact Or.inl {
      paper := paper.1
      input := input
      yRing := paper.2
      sampler := accepted.sampler
      paperOutput :=
        FixedActive.PaperProfile.PhysicalOutput.equations_of_tail context
          certificate accepted.tail canonicalPublicInput
          parentEvaluationSize childEvaluationSize
      canonicalParent := canonicalParent
    }
  · exact Or.inr (Or.inl yRingUnbound)
  · exact Or.inr (Or.inr bad)

/-- Once the delayed packed equation is supplied by the successor or terminal
step, the claims-only checker becomes the raw-source paper refinement.  Every
source/input fact and the canonical parent opening is computed or checked by
`PaperStepAccepted`; no running-authority or child-sidecar premise occurs. -/
theorem paperStepAccepted_and_packed_implies_refinement_or_yRingUnbound_or_badEvent
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    (carrier : FixedActive.CanonicalOpening.SourceInput.Carrier shape
      publicRingColumns publicFits)
    (context : FixedActive.CanonicalOpening.Context shape State
      publicRingColumns publicFits verifierRows)
    (certificate : FixedActive.Certificate (carrier.install context).full)
    (accepted : PaperStepAccepted carrier context certificate)
    (packed : Terminal.PackedYZcolBoundAtBlock
      (carrier.install context).full.covers carrier.data
      (ProductionPiCcs.ncPoint (carrier.install context).full
        certificate).block
      certificate.piCcs.output) :
    PaperRefinement (carrier.install context).full carrier.data certificate ∨
      ProductionPiCcs.YRingUnbound (carrier.install context).full carrier.data
        certificate ∨
      ProductionPiCcs.BadEvent (carrier.install context).full carrier.data
        certificate := by
  have raw : ProductionPiCcs.Accepted (carrier.install context).full
      carrier.data certificate :=
    ProductionPiCcs.accepted_of_messageAccepted_and_packed
      (carrier.install context).full carrier.data certificate accepted.piCcs
      packed
  rcases
      ProductionPiCcs.accepted_implies_paper_and_yRingBound_or_yRingUnbound_or_badEvent
        noZeroDivisors (carrier.install context).full carrier.data certificate
        (carrier.semanticInput context).publicInput raw with
    paper | yRingUnbound | bad
  · exact Or.inl {
      paper := paper.1
      input := carrier.semanticInput context
      yRing := paper.2
      sampler := accepted.sampler
      paperOutput := accepted.paperOutput
      canonicalParent := accepted.canonicalParent
    }
  · exact Or.inr (Or.inl yRingUnbound)
  · exact Or.inr (Or.inr bad)

end Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual.CombinedNc.ProductionPaperNifs
