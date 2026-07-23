import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive.PaperProfile.PhysicalOutput

/-!
Physical fixed-active acceptance to the independent paper transition.

Protocol: SuperNeo Sections 7.3--7.5.
Phase: one physical `Pi_CCS -> Pi_RLC -> Pi_DEC` execution.
Constraint family: typed semantic composition only; this file emits no rows.

Assurance tier: model-level.

Owns: direct refinement from physical acceptance to the paper-profile
transition over the actual public `Pi_DEC` children, modulo the existing
output-binding and `Pi_CCS` bad-event outcomes.

Does not own: lifecycle/running-state sidecars, deterministic child openings,
canonical private splitting, generated rows, Rust refinement, costs, or row
removal.

Authority boundary: source data bind both public input surfaces explicitly.
The physical parent is identified with the paper parent using only source
binding and the positive `yRing` half of `OutputBound`. The target remains the
actual verifier-visible child family; no child CE opening is assumed.
-/

namespace Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive.PaperProfile.PhysicalTransition

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier

universe uState

variable {shape : SemanticShape}
variable {State : Type uState}
variable {publicRingColumns verifierRows : Nat}
variable {publicFits :
  ringDegree * publicRingColumns <= shape.carrierWidth}

/-- Physical fixed-active acceptance refines the independent paper transition
for the actual public children, or exposes exactly the existing output-binding
failure or typed `Pi_CCS` bad event. Neither `ChildOpenings` nor the stronger
deterministic `canonicalTarget` relation occurs in the statement. -/
theorem accepted_implies_transition_or_outputUnbound_or_badEvent
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
    (accepted : Accepted context certificate) :
    FixedActive.PaperProfile.Transition (FixedActive.paperProfileOf context)
        context.input (outputChildren context certificate) ∨
      ¬ OutputBound context data certificate ∨
      PiCcsBadEvent context data certificate := by
  rcases accepted_implies_paper_or_outputUnbound_or_badEvent noZeroDivisors
      input.publicInput accepted with paper | outputUnbound | bad
  · by_cases output : OutputBound context data certificate
    · apply Or.inl
      let semanticWitness := CertificateRefinement.semanticWitness certificate
      let witness := FixedActive.paperWitnessOf semanticWitness
      have systemEq :
          context.system = SemanticFold.systemOf context data := by
        simpa [Context.system, SemanticFold.systemOf] using
          (input.sources.fresh ⟨0, FixedActive.arity.freshPositive⟩).constraintSystem
      have outputsEq :
          (derive context certificate).piCcsOutputs =
            SemanticFold.outputs context data semanticWitness := by
        change
          OutputProduct.materialize publicRingColumns publicFits
              context.alignment context.input
              (derive context certificate).piCcs.fePoint.row
              certificate.piCcs.output =
            PiCCS.honestOutputs (ConcretePhi81.semantics context.key)
              context.input
              (InputAuthority.productAssignments data context.alignment)
              (derive context certificate).piCcs.fePoint.row
        simpa [ConcretePhi81.semantics] using
          (Protocol.OutputRefinement.materializedOutputs_eq_honestOutputs_of_yRingEq
            publicRingColumns publicFits (ConcretePhi81.commit context.key) data
            context.alignment context.input
            (derive context certificate).piCcs.fePoint.row
            certificate.piCcs.output production_norm_stages.1 paper
            input.sources output.1)
      have parentEq :
          (derive context certificate).piRlcOutput =
            FixedActive.PaperProfile.parentOf
              (FixedActive.paperProfileOf context) context.input data witness := by
        change
          PiRLC.combinedOutput (ConcretePhi81.rlcAlgebra context.key)
              context.system (derive context certificate).piCcs.fePoint.row
              (derive context certificate).piCcsOutputs
              certificate.piRlcChallenges =
            PiRLC.combinedOutput (ConcretePhi81.rlcAlgebra context.key)
              (SemanticFold.systemOf context data)
              (derive context certificate).piCcs.fePoint.row
              (SemanticFold.outputs context data semanticWitness)
              certificate.piRlcChallenges
        rw [systemEq, outputsEq]
      have physicalPiDec :=
        PhysicalOutput.paperOutputAccepted_of_tail context certificate
          accepted.tail canonicalPublicInput parentEvaluationSize
          childEvaluationSize
      have paperPiDec :
          PiDEC.PaperVerifier.OutputAccepted
            (FixedActive.PaperProfile.decAlgebra
              (FixedActive.paperProfileOf context))
            (FixedActive.PaperProfile.decPublicInputSplit
              (FixedActive.paperProfileOf context))
            (FixedActive.PaperProfile.decEvaluationArity
              (FixedActive.paperProfileOf context))
            (FixedActive.PaperProfile.parentOf
              (FixedActive.paperProfileOf context) context.input data witness)
            (outputChildren context certificate) := by
        rw [← parentEq]
        exact physicalPiDec
      exact ⟨data, witness, {
        paper := paper
        input := input.sources
        challengesValid := by
          simpa [witness, semanticWitness, FixedActive.paperWitnessOf] using
            (Sampler.certificateAccepted_challengesValid accepted.sampler)
        piDecAccepted := paperPiDec
      }⟩
    · exact Or.inr (Or.inl output)
  · exact Or.inr (Or.inl outputUnbound)
  · exact Or.inr (Or.inr bad)

/-- Once the independently established packed projection is supplied, the
generic output-bound branch sharpens to exactly the remaining `yRing`
equation. The packed theorem is consumed only to reconstruct the pair defining
`OutputBound`; it is not copied into the paper target. -/
theorem accepted_implies_transition_or_yRingUnbound_or_badEvent_of_packedYZcolBound
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
    (packed :
      Polynomial.Nc.BlockLane.Terminal.PackedYZcolBoundAtBlock context.covers
        data (derive context certificate).piCcs.ncPoint.block
        certificate.piCcs.output)
    (accepted : Accepted context certificate) :
    FixedActive.PaperProfile.Transition (FixedActive.paperProfileOf context)
        context.input (outputChildren context certificate) ∨
      ¬ certificate.piCcs.output.yRing =
        Polynomial.Fe.sourceYRingAt data
          (derive context certificate).piCcs.fePoint.row ∨
      PiCcsBadEvent context data certificate := by
  rcases accepted_implies_transition_or_outputUnbound_or_badEvent
      noZeroDivisors context data certificate input canonicalPublicInput
      parentEvaluationSize childEvaluationSize accepted with
    transition | outputUnbound | bad
  · exact Or.inl transition
  · apply Or.inr
    apply Or.inl
    intro yRing
    exact outputUnbound ⟨yRing, packed⟩
  · exact Or.inr (Or.inr bad)

end Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive.PaperProfile.PhysicalTransition
