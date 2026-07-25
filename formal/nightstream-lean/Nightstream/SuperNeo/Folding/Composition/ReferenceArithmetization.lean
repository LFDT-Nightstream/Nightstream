import Nightstream.SuperNeo.Folding.Composition

/-!
Composition against one verifier-derived reference arithmetization.

Assurance tier: model-level.

Owns: replacement of the caller-supplied `rewindArithmetization` callback when
one concrete assignment vector both opens the PiRLC inputs and carries a
kernel-checked PiCCS arithmetization. Disagreement with the extractor is
returned as the existing relaxed-binding collision.

Does not own: construction of the reference opening, a commitment binding
bound, sampling probability, Fiat--Shamir, Rust, R1CS, costs, or rows.
-/

set_option autoImplicit false

namespace Nightstream.SuperNeo.Folding.Composition.ReferenceArithmetization

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Folding

universe uStructure uAssignment uPublicInput uPoint uEvaluation uCommitment
universe uScalar uChallenge uValue

/-- Strong composition with a concrete reference assignment vector. No
rewind callback is retained: unequal valid openings are classified by the
existing relaxed-binding event. -/
theorem fold_extraction_or_bad_event
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    {Challenge : Type uChallenge}
    {Value : Type uValue}
    (semantics : RelationSemantics
      Structure Assignment PublicInput Point Evaluation Commitment)
    (params : GlobalParams)
    (sumcheckOps : SumCheck.Ops Challenge Value)
    (rlcAlgebra : PiRLC.Algebra
      Structure Assignment PublicInput Point Evaluation Commitment Scalar
      semantics params)
    (decAlgebra : PiDEC.Algebra
      Structure Assignment PublicInput Point Evaluation Commitment semantics
      params)
    (bindingOps : PiRLC.RelaxedBindingOps Assignment Commitment Scalar)
    (arity : BatchArity params)
    (sampling : PiRLC.SamplingBoundary arity.total)
    (ccsAttempt : PiCCS.Attempt
      Structure PublicInput Point Evaluation Commitment Challenge Value params
      arity)
    (rlcAttempt : PiRLC.Attempt
      Structure PublicInput Point Evaluation Commitment Scalar params arity)
    (decAttempt : PiDEC.Attempt
      Structure PublicInput Point Evaluation Commitment params)
    (finalAssignments : Fin params.k -> Assignment)
    (referenceAssignments : Fin arity.total -> Assignment)
    (kPositive : 0 < params.k)
    (sameRlcInputs : ∀ i, rlcAttempt.inputs i = ccsAttempt.outputs i)
    (sameDecParent : decAttempt.parent = rlcAttempt.output)
    (ccsAccepted : PiCCS.Accepted sumcheckOps ccsAttempt)
    (rlcAccepted : PiRLC.Accepted rlcAlgebra rlcAttempt)
    (decAccepted : PiDEC.Accepted decAlgebra decAttempt)
    (finalValid : ∀ i,
      CE.Holds semantics params (decAttempt.children i) (finalAssignments i))
    (extractor : WeakExtractor semantics params rlcAlgebra rlcAttempt sampling)
    (uniqueness : PiRLC.UniquenessBridge semantics params bindingOps
      (n := arity.total))
    (referenceAmbient :
      PiRLC.AmbientOpenings semantics params rlcAttempt.inputs
        referenceAssignments)
    (referenceArithmetization :
      PiCCS.Arithmetization semantics params sumcheckOps ccsAttempt
        referenceAssignments) :
    Nonempty (ExtractedBatch semantics params ccsAttempt) ∨
      BadEvent semantics params bindingOps sampling ccsAttempt
        rlcAttempt.inputs := by
  have parentValid : CE.Holds semantics params decAttempt.parent
      (decAlgebra.recomposeAssignment finalAssignments) :=
    PiDEC.reduce_knowledge semantics params decAlgebra decAttempt
      finalAssignments kPositive decAccepted finalValid
  have rlcOutputValid : CE.Holds semantics params rlcAttempt.output
      (decAlgebra.recomposeAssignment finalAssignments) := by
    simpa [sameDecParent] using parentValid
  let outcome := extractor.run false
    (decAlgebra.recomposeAssignment finalAssignments) rlcAccepted rlcOutputValid
  cases outcomeEq : outcome with
  | failed failure =>
      exact Or.inr (Or.inr (Or.inl failure))
  | extracted result =>
      rcases PiRLC.same_phi_extractions_unique_or_collision semantics params
          bindingOps uniqueness rlcAttempt.inputs rlcAttempt.inputs
          result.assignments referenceAssignments rfl result.valid
          referenceAmbient with sameAssignments | collision
      · have outputValid :
            PiCCS.AmbientOutputsHold semantics params ccsAttempt
              result.assignments := by
          intro i
          have extracted := result.valid i
          simpa [PiRLC.ambientInput, PiCCS.relaxedOutput, sameRlcInputs i] using
            extracted
        have arithmetization :
            PiCCS.Arithmetization semantics params sumcheckOps ccsAttempt
              result.assignments := by
          rw [sameAssignments]
          exact referenceArithmetization
        rcases PiCCS.strong_extract_or_bad_event semantics params sumcheckOps
            ccsAttempt result.assignments ccsAccepted arithmetization
            outputValid with valid | bad
        · exact Or.inl ⟨⟨result.assignments, valid, outputValid⟩⟩
        · exact Or.inr (Or.inl bad)
      · exact Or.inr (Or.inr (Or.inr collision))

end Nightstream.SuperNeo.Folding.Composition.ReferenceArithmetization
