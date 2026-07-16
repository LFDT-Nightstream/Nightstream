import Nightstream.SuperNeo.Folding.PiCCS
import Nightstream.SuperNeo.Folding.PiRLC
import Nightstream.SuperNeo.Folding.PiDEC

/-!
Composition of the joint Π_CCS batch, Π_RLC, and Π_DEC.

The theorem follows the knowledge direction from valid final `CE(b)^k`
witnesses.  Both Π_CCS and Π_RLC use the same `BatchArity`, so bootstrap
consumes no fictional running claims and active mode recovers the paper's full
`K+k` product.  Every probabilistic or cryptographic exit is explicit.

The strongest success result retains one extracted assignment vector together
with both source validity and generic ambient CE validity for the complete
Π_CCS output product.  It does not identify generic CE evaluations with the
Phi81 `yRing` representation and does not cover the separate `yZcol` sidecar.

Owns: knowledge composition and the exact success/bad-event boundary.

Does not own: Fiat--Shamir, a concrete commitment reduction, Phi81 output
refinement, SplitNC sidecars, Rust, R1CS, or row removal.

Emits constraints: no.

| Protocol | Phase | Family | Mathematical obligation |
|---|---|---|---|
| `Pi_DEC` | reverse knowledge | parent opening | valid fresh children and accepted recomposition open the combined parent |
| `Pi_RLC` | rewind | ambient extraction | two extractor forks open the complete Pi_RLC input product or expose sampling failure |
| `Pi_RLC` | rewind | uniqueness | unequal extracted assignment vectors expose relaxed binding |
| `Pi_CCS` | strong extraction | source/output product | the same assignment vector opens every source and every generic ambient CE output |
| composition | success | `ExtractedBatch` | retain assignments, source validity, and ambient output validity together |
| composition | projection | `InputsValid` | deliberately forget output validity for source-only consumers |
| composition | failure | `BadEvent` | expose Pi_CCS, sampling, or relaxed-binding failure without adding another event |
-/

namespace Nightstream.SuperNeo.Folding.Composition

universe uStructure uAssignment uPublicInput uPoint uEvaluation uCommitment
  uScalar uChallenge uValue

/-- Explicit cryptographic boundary for the Appendix D.5 rewinding extractor. -/
structure WeakExtractor
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    (semantics : RelationSemantics
      Structure Assignment PublicInput Point Evaluation Commitment)
    (params : GlobalParams)
    {arity : BatchArity params}
    (algebra : PiRLC.Algebra
      Structure Assignment PublicInput Point Evaluation Commitment Scalar semantics params)
    (attempt : PiRLC.Attempt
      Structure PublicInput Point Evaluation Commitment Scalar params arity)
    (sampling : PiRLC.SamplingBoundary arity.total) where
  /-- `fork = false/true` represents the two independent extractor executions
  used by weak-reduction witness uniqueness. -/
  run : Bool → ∀ outputAssignment,
    PiRLC.Accepted algebra attempt →
    CE.Holds semantics params attempt.output outputAssignment →
    PiRLC.ExtractionOutcome semantics params attempt.inputs sampling

/-- A successful composed extraction opens every source in the joint batch. -/
def InputsValid
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Challenge : Type uChallenge}
    {Value : Type uValue}
    (semantics : RelationSemantics
      Structure Assignment PublicInput Point Evaluation Commitment)
    (params : GlobalParams)
    {arity : BatchArity params}
    (attempt : PiCCS.Attempt
      Structure PublicInput Point Evaluation Commitment Challenge Value params arity) : Prop :=
  ∃ assignments : Fin arity.total → Assignment,
    ∀ i, (attempt.inputs.source i).Holds semantics params (assignments i)

/-- One successful composed extraction, without discarding the output product
that the weak extractor opened.

`ambientOutputsValid` is validity in the generic `RelationSemantics` CE model at
the verifier-owned ambient norm bound. It is not a Phi81-output theorem: a later
refinement must identify `semantics.evaluations` with the exact `yRing`
representation. `yZcol` is not a `CE.Instance` field and remains a separate
SplitNC authority obligation. -/
structure ExtractedBatch
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Challenge : Type uChallenge}
    {Value : Type uValue}
    (semantics : RelationSemantics
      Structure Assignment PublicInput Point Evaluation Commitment)
    (params : GlobalParams)
    {arity : BatchArity params}
    (attempt : PiCCS.Attempt
      Structure PublicInput Point Evaluation Commitment Challenge Value params arity) where
  assignments : Fin arity.total → Assignment
  sourcesValid : ∀ i,
    (attempt.inputs.source i).Holds semantics params (assignments i)
  ambientOutputsValid :
    PiCCS.AmbientOutputsHold semantics params attempt assignments

/-- Every failure branch is a named paper event; none is `accepted → valid`. -/
def BadEvent
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
    (bindingOps : PiRLC.RelaxedBindingOps Assignment Commitment Scalar)
    {arity : BatchArity params}
    (sampling : PiRLC.SamplingBoundary arity.total)
    (ccsAttempt : PiCCS.Attempt
      Structure PublicInput Point Evaluation Commitment Challenge Value params arity)
    (rlcInputs : Fin arity.total →
      CE.Instance Structure PublicInput Point Evaluation Commitment) : Prop :=
  PiCCS.BadEvent semantics params ccsAttempt ∨
  sampling.Failure ∨
  ∃ i, Nonempty
    (PiRLC.RelaxedBindingCollision semantics params bindingOps
      (rlcInputs i).commitment)

private theorem extractedBatch_of_ambient_extraction
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Challenge : Type uChallenge}
    {Value : Type uValue}
    (semantics : RelationSemantics
      Structure Assignment PublicInput Point Evaluation Commitment)
    (params : GlobalParams)
    (ops : SumCheck.Ops Challenge Value)
    {arity : BatchArity params}
    (ccsAttempt : PiCCS.Attempt
      Structure PublicInput Point Evaluation Commitment Challenge Value params arity)
    (rlcInputs : Fin arity.total →
      CE.Instance Structure PublicInput Point Evaluation Commitment)
    (sameInputs : ∀ i, rlcInputs i = ccsAttempt.outputs i)
    (accepted : PiCCS.Accepted ops ccsAttempt)
    (assignments : Fin arity.total → Assignment)
    (ambientValid : PiRLC.AmbientOpenings semantics params rlcInputs assignments)
    (arithmetization :
      PiCCS.Arithmetization semantics params ops ccsAttempt assignments) :
    Nonempty (ExtractedBatch semantics params ccsAttempt) ∨
      PiCCS.BadEvent semantics params ccsAttempt := by
  have outputValid :
      PiCCS.AmbientOutputsHold semantics params ccsAttempt assignments := by
    intro i
    have extracted := ambientValid i
    simpa [PiRLC.ambientInput, PiCCS.relaxedOutput, sameInputs i] using extracted
  rcases PiCCS.strong_extract_or_bad_event semantics params ops ccsAttempt
      assignments accepted arithmetization outputValid with valid | bad
  · exact Or.inl ⟨⟨assignments, valid, outputValid⟩⟩
  · exact Or.inr bad

/-- The joint Π_CCS output projection is exactly the Π_RLC input projection. -/
theorem shared_phi
    {Structure : Type uStructure}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Challenge : Type uChallenge}
    {Value : Type uValue}
    {params : GlobalParams}
    {arity : BatchArity params}
    (ccsAttempt : PiCCS.Attempt
      Structure PublicInput Point Evaluation Commitment Challenge Value params arity)
    (rlcInputs : Fin arity.total →
      CE.Instance Structure PublicInput Point Evaluation Commitment)
    (sameInputs : ∀ i, rlcInputs i = ccsAttempt.outputs i) :
    PiRLC.phi rlcInputs = PiCCS.phi ccsAttempt := by
  funext i
  simp [PiRLC.phi, PiCCS.phi, sameInputs i]

/--
Strongest model-level SuperNeo multi-fold knowledge theorem for
`Π_DEC ∘ Π_RLC ∘ Π_CCS` at either production running mode.

The successful branch retains the exact assignment vector used for both the
source openings and the complete generic ambient CE output product. The latter
does not by itself establish Phi81 `yRing` or SplitNC `yZcol` authority.
-/
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
      Structure Assignment PublicInput Point Evaluation Commitment Scalar semantics params)
    (decAlgebra : PiDEC.Algebra
      Structure Assignment PublicInput Point Evaluation Commitment semantics params)
    (bindingOps : PiRLC.RelaxedBindingOps Assignment Commitment Scalar)
    (arity : BatchArity params)
    (sampling : PiRLC.SamplingBoundary arity.total)
    (ccsAttempt : PiCCS.Attempt
      Structure PublicInput Point Evaluation Commitment Challenge Value params arity)
    (rlcAttempt : PiRLC.Attempt
      Structure PublicInput Point Evaluation Commitment Scalar params arity)
    (decAttempt : PiDEC.Attempt
      Structure PublicInput Point Evaluation Commitment params)
    (finalAssignments : Fin params.k → Assignment)
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
    (rewindArithmetization : ∀ leftAssignments rightAssignments,
      PiRLC.AmbientOpenings semantics params rlcAttempt.inputs leftAssignments →
      PiRLC.AmbientOpenings semantics params rlcAttempt.inputs rightAssignments →
      leftAssignments = rightAssignments →
      PiCCS.Arithmetization semantics params sumcheckOps
        ccsAttempt leftAssignments) :
    Nonempty (ExtractedBatch semantics params ccsAttempt) ∨
      BadEvent semantics params bindingOps sampling ccsAttempt rlcAttempt.inputs := by
  have parentValid : CE.Holds semantics params decAttempt.parent
      (decAlgebra.recomposeAssignment finalAssignments) :=
    PiDEC.reduce_knowledge semantics params decAlgebra decAttempt finalAssignments
      kPositive decAccepted finalValid
  have rlcOutputValid : CE.Holds semantics params rlcAttempt.output
      (decAlgebra.recomposeAssignment finalAssignments) := by
    simpa [sameDecParent] using parentValid
  let leftOutcome := extractor.run false
    (decAlgebra.recomposeAssignment finalAssignments) rlcAccepted rlcOutputValid
  let rightOutcome := extractor.run true
    (decAlgebra.recomposeAssignment finalAssignments) rlcAccepted rlcOutputValid
  cases leftEq : leftOutcome with
  | failed leftFailure =>
      exact Or.inr (Or.inr (Or.inl leftFailure))
  | extracted leftResult =>
      cases rightEq : rightOutcome with
      | failed rightFailure =>
          exact Or.inr (Or.inr (Or.inl rightFailure))
      | extracted rightResult =>
          rcases PiRLC.same_phi_extractions_unique_or_collision semantics params
              bindingOps uniqueness rlcAttempt.inputs rlcAttempt.inputs
              leftResult.assignments rightResult.assignments rfl
              leftResult.valid rightResult.valid with sameAssignments | collision
          · rcases extractedBatch_of_ambient_extraction semantics params sumcheckOps
                ccsAttempt rlcAttempt.inputs sameRlcInputs ccsAccepted
                leftResult.assignments leftResult.valid
                (rewindArithmetization leftResult.assignments
                  rightResult.assignments leftResult.valid rightResult.valid
                  sameAssignments) with valid | bad
            · exact Or.inl valid
            · exact Or.inr (Or.inl bad)
          · exact Or.inr (Or.inr (Or.inr collision))

/-- Source-validity projection of `fold_extraction_or_bad_event`.

This corollary deliberately forgets the successful extractor's generic ambient
output validity. Consumers that reason about the output product must use the
stronger theorem instead. -/
theorem fold_knowledge_or_bad_event
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
      Structure Assignment PublicInput Point Evaluation Commitment Scalar semantics params)
    (decAlgebra : PiDEC.Algebra
      Structure Assignment PublicInput Point Evaluation Commitment semantics params)
    (bindingOps : PiRLC.RelaxedBindingOps Assignment Commitment Scalar)
    (arity : BatchArity params)
    (sampling : PiRLC.SamplingBoundary arity.total)
    (ccsAttempt : PiCCS.Attempt
      Structure PublicInput Point Evaluation Commitment Challenge Value params arity)
    (rlcAttempt : PiRLC.Attempt
      Structure PublicInput Point Evaluation Commitment Scalar params arity)
    (decAttempt : PiDEC.Attempt
      Structure PublicInput Point Evaluation Commitment params)
    (finalAssignments : Fin params.k → Assignment)
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
    (rewindArithmetization : ∀ leftAssignments rightAssignments,
      PiRLC.AmbientOpenings semantics params rlcAttempt.inputs leftAssignments →
      PiRLC.AmbientOpenings semantics params rlcAttempt.inputs rightAssignments →
      leftAssignments = rightAssignments →
      PiCCS.Arithmetization semantics params sumcheckOps
        ccsAttempt leftAssignments) :
    InputsValid semantics params ccsAttempt ∨
      BadEvent semantics params bindingOps sampling ccsAttempt rlcAttempt.inputs := by
  rcases fold_extraction_or_bad_event semantics params sumcheckOps rlcAlgebra
      decAlgebra bindingOps arity sampling ccsAttempt rlcAttempt decAttempt
      finalAssignments kPositive sameRlcInputs sameDecParent ccsAccepted
      rlcAccepted decAccepted finalValid extractor uniqueness
      rewindArithmetization with extracted | bad
  · rcases extracted with ⟨witness⟩
    exact Or.inl ⟨witness.assignments, witness.sourcesValid⟩
  · exact Or.inr bad

end Nightstream.SuperNeo.Folding.Composition
