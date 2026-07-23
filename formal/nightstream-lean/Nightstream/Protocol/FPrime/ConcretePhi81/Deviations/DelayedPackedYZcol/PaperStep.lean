import Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc
import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Authority.DelayedRawChildren
import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive.CanonicalOpening.SourceInput
import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive.PaperProfile.PhysicalTransition

/-!
One opening-derived production step, refined to the independent paper
transition after its delayed packed-`yZcol` equation is closed.

Assurance tier: model-level registered production deviation.

Owns: the executable obligations retained by one production paper step; their
minimal positive refinement record; and promotion to
`FixedActive.PaperProfile.Transition` once the successor or terminal boundary
supplies the delayed packed equation.

Does not own: lifecycle continuity, construction of the delayed packed
equation, terminal closure, concrete assignment decoding, generated rows,
commitment binding, Rust/R1CS conformance, costs, or row removal.

Emits constraints: no.

Authority boundary: raw `Pi_CCS` acceptance is imported only through the
protocol-owned block/lane combined-NC surface. Its NC terminal reads
authoritative `Sources.Data`; neither an incoming child `y_zcol` sidecar nor a
generic implementation-refinement premise occurs here.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `fprime.delayed.paper_step.receipt` | retain the opening-derived production checks needed by one step | checked | `PaperStepAccepted` |
| `fprime.delayed.paper_step.refinement` | isolate the minimal paper obligations produced by those checks | derived | `PaperRefinement` |
| `fprime.delayed.paper_step.transition` | add the delayed packed-output equation and construct the independent paper transition | derived | `toPaperTransition` |
| `fprime.delayed.paper_step.soundness` | accepted production step plus packed binding yields the paper transition or a named `yRing`/algebraic event | derived/security partition | `accepted_and_packed_implies_transition_or_yRingUnbound_or_badEvent` |
-/

namespace Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.DelayedPackedYZcol.PaperStep

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

namespace ProductionPiCcs

export
  Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc
  (MessageAccepted Accepted BadEvent YRingUnbound YRingBound
    accepted_of_messageAccepted_and_packed
    accepted_implies_paper_and_yRingBound_or_yRingUnbound_or_badEvent)

end ProductionPiCcs

universe uState

variable {shape : SemanticShape}
variable {State : Type uState}
variable {publicRingColumns verifierRows : Nat}
variable {publicFits :
  ringDegree * publicRingColumns <= shape.carrierWidth}

/-- Public, executable obligations for one opening-derived paper step.

The incoming running family is deliberately absent. Once the raw NC truth is
known, it is reconstructed from `carrier.opening` by the canonical source
input bridge. -/
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

/-- The operational paper-output equations imply ordinary public `Pi_DEC`
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

/-- A packed-closed, opening-derived step partitions directly into the
independent paper transition, the exact current-step `yRing` mismatch, or the
typed raw-`Pi_CCS` bad event.

There is no generic output-unbound or implementation-refinement branch. -/
theorem accepted_and_packed_implies_transition_or_yRingUnbound_or_badEvent
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    (carrier : FixedActive.CanonicalOpening.SourceInput.Carrier shape
      publicRingColumns publicFits)
    (context : FixedActive.CanonicalOpening.Context shape State
      publicRingColumns publicFits verifierRows)
    (certificate : FixedActive.Certificate (carrier.install context).full)
    (accepted : PaperStepAccepted carrier context certificate)
    (packed : Terminal.PackedYZcolBoundAtBlock
      (carrier.install context).full.covers carrier.data
      (derive (carrier.install context).full certificate).piCcs.ncPoint.block
      certificate.piCcs.output) :
    FixedActive.PaperProfile.Transition
        (FixedActive.paperProfileOf (carrier.install context).full)
        (carrier.install context).full.input
        (outputChildren (carrier.install context).full certificate) ∨
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
  · exact Or.inl (PaperRefinement.toPaperTransition {
      paper := paper.1
      input := carrier.semanticInput context
      yRing := paper.2
      sampler := accepted.sampler
      paperOutput := accepted.paperOutput
      canonicalParent := accepted.canonicalParent
    } packed)
  · exact Or.inr (Or.inl yRingUnbound)
  · exact Or.inr (Or.inr bad)

end Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.DelayedPackedYZcol.PaperStep
