import Nightstream.SuperNeo.Folding.Composition

/-!
Candidate SuperNeo NIFS composition over the current abstract phase models.

Owns: one typed verifier attempt for `Pi_CCS -> Pi_RLC -> Pi_DEC`, the two
cross-phase equalities, model completeness from valid source openings, and a
conditional knowledge direction to valid NIFS inputs or named bad events.

Does not own: Fiat-Shamir, Poseidon2, Rust layouts, R1CS rows, production
public-input packing, HyperNova state, or permission to remove constraints.

Emits constraints: no.

Authority boundary: `Accepted` is defined only from the three abstract phase
acceptance predicates and exact statement equality at their two interfaces.
No implementation verifier, digest, row artifact, or historical constraint
count appears here. However, the current `PiCCS` phase is the abstract
two-chain FE/NC model; equivalence to Section 7.3's one joint `Q` polynomial,
finite certificates, verifier-derived points, and concrete Fiat--Shamir remain
open. Therefore `PaperNifsTransition` is a candidate composition target, not
yet end-to-end paper assurance.

| Protocol phase | Mathematical obligation | Lean owner |
|---|---|---|
| `piCcs` | fresh/running sources reduce to a shared-point CE product | `PiCCS.Accepted` |
| `piRlc` | one challenge vector combines every public CE component | `PiRLC.Accepted` |
| `piDec` | the combined CE parent recomposes from `k` fresh children | `PiDEC.Accepted` |
| `ccsToRlc` | the complete Pi_CCS output product is the Pi_RLC input product | `Wiring` |
| `rlcToDec` | the Pi_RLC output is literally the Pi_DEC parent | `Wiring` |
| completeness | honest source openings construct all three accepted phases and the external transition | `complete`, `paperNifsTransition_complete` |
| external relation | hides phase advice while fixing the NIFS input and output products | `PaperNifsTransition` |
| knowledge | accepted final children recover the original sources or a named bad event | `accepted_inputsValid_or_badEvent` |
-/

namespace Nightstream.SuperNeo.Folding.Nifs

universe uStructure uAssignment uPublicInput uPoint uEvaluation uCommitment
  uScalar uChallenge uValue

/-- Public data consumed and produced by one paper NIFS verifier execution. -/
structure Attempt
    (Structure : Type uStructure)
    (PublicInput : Type uPublicInput)
    (Point : Type uPoint)
    (Evaluation : Type uEvaluation)
    (Commitment : Type uCommitment)
    (Scalar : Type uScalar)
    (Challenge : Type uChallenge)
    (Value : Type uValue)
    (params : GlobalParams)
    (arity : BatchArity params) where
  piCcs : PiCCS.Attempt
    Structure PublicInput Point Evaluation Commitment Challenge Value params arity
  piRlc : PiRLC.Attempt
    Structure PublicInput Point Evaluation Commitment Scalar params arity
  piDec : PiDEC.Attempt
    Structure PublicInput Point Evaluation Commitment params

/-- Exact statement identity at both sequential-composition boundaries. -/
structure Wiring
    {Structure : Type uStructure}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    {Challenge : Type uChallenge}
    {Value : Type uValue}
    {params : GlobalParams}
    {arity : BatchArity params}
    (attempt : Attempt
      Structure PublicInput Point Evaluation Commitment Scalar Challenge Value
        params arity) : Prop where
  ccsToRlc : forall index, attempt.piRlc.inputs index = attempt.piCcs.outputs index
  rlcToDec : attempt.piDec.parent = attempt.piRlc.output

/-- Independent paper verifier acceptance for the composed NIFS transition. -/
structure Accepted
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    {Challenge : Type uChallenge}
    {Value : Type uValue}
    {semantics : RelationSemantics
      Structure Assignment PublicInput Point Evaluation Commitment}
    {params : GlobalParams}
    {arity : BatchArity params}
    (sumcheckOps : SumCheck.Ops Challenge Value)
    (rlcAlgebra : PiRLC.Algebra
      Structure Assignment PublicInput Point Evaluation Commitment Scalar semantics params)
    (decAlgebra : PiDEC.Algebra
      Structure Assignment PublicInput Point Evaluation Commitment semantics params)
    (attempt : Attempt
      Structure PublicInput Point Evaluation Commitment Scalar Challenge Value
        params arity) : Prop where
  wiring : Wiring attempt
  piCcs : PiCCS.Accepted sumcheckOps attempt.piCcs
  piRlc : PiRLC.Accepted rlcAlgebra attempt.piRlc
  piDec : PiDEC.Accepted decAlgebra attempt.piDec

/--
Candidate public NIFS transition relation.

The existential contains only abstract phase attempts. The externally visible
input is the complete Pi_CCS source product and the externally visible output
is the complete vector of fresh Pi_DEC children. The legacy `Paper` name does
not discharge the open joint-Q/finite-certificate boundaries above.
-/
def PaperNifsTransition
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    {Challenge : Type uChallenge}
    {Value : Type uValue}
    {semantics : RelationSemantics
      Structure Assignment PublicInput Point Evaluation Commitment}
    {params : GlobalParams}
    {arity : BatchArity params}
    (sumcheckOps : SumCheck.Ops Challenge Value)
    (rlcAlgebra : PiRLC.Algebra
      Structure Assignment PublicInput Point Evaluation Commitment Scalar semantics params)
    (decAlgebra : PiDEC.Algebra
      Structure Assignment PublicInput Point Evaluation Commitment semantics params)
    (input : PiCCS.InputProduct
      Structure PublicInput Point Evaluation Commitment params arity)
    (output : Fin params.k ->
      CE.Instance Structure PublicInput Point Evaluation Commitment) : Prop :=
  exists attempt : Attempt
      Structure PublicInput Point Evaluation Commitment Scalar Challenge Value
        params arity,
    attempt.piCcs.inputs = input /\
      attempt.piDec.children = output /\
      Accepted sumcheckOps rlcAlgebra decAlgebra attempt

/-- Any accepted phase composition exposes one independent public transition. -/
theorem paperNifsTransition_of_accepted
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    {Challenge : Type uChallenge}
    {Value : Type uValue}
    {semantics : RelationSemantics
      Structure Assignment PublicInput Point Evaluation Commitment}
    {params : GlobalParams}
    {arity : BatchArity params}
    (sumcheckOps : SumCheck.Ops Challenge Value)
    (rlcAlgebra : PiRLC.Algebra
      Structure Assignment PublicInput Point Evaluation Commitment Scalar semantics params)
    (decAlgebra : PiDEC.Algebra
      Structure Assignment PublicInput Point Evaluation Commitment semantics params)
    (attempt : Attempt
      Structure PublicInput Point Evaluation Commitment Scalar Challenge Value
        params arity)
    (accepted : Accepted sumcheckOps rlcAlgebra decAlgebra attempt) :
    PaperNifsTransition sumcheckOps rlcAlgebra decAlgebra
      attempt.piCcs.inputs attempt.piDec.children := by
  exact ⟨attempt, rfl, rfl, accepted⟩

/--
Knowledge soundness of the paper NIFS composition.

The extractor, uniqueness bridge, and rewind arithmetization are the explicit
computational/probabilistic boundaries from SuperNeo's strong/weak reduction
composition. The theorem never concludes source validity from public equations
alone.
-/
theorem accepted_inputsValid_or_badEvent
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
    (attempt : Attempt
      Structure PublicInput Point Evaluation Commitment Scalar Challenge Value
        params arity)
    (finalAssignments : Fin params.k -> Assignment)
    (kPositive : 0 < params.k)
    (accepted : Accepted sumcheckOps rlcAlgebra decAlgebra attempt)
    (finalValid : forall index,
      CE.Holds semantics params (attempt.piDec.children index)
        (finalAssignments index))
    (extractor : Composition.WeakExtractor
      semantics params rlcAlgebra attempt.piRlc sampling)
    (uniqueness : PiRLC.UniquenessBridge semantics params bindingOps
      (n := arity.total))
    (rewindArithmetization : forall leftAssignments rightAssignments,
      PiRLC.AmbientOpenings semantics params attempt.piRlc.inputs leftAssignments ->
      PiRLC.AmbientOpenings semantics params attempt.piRlc.inputs rightAssignments ->
      leftAssignments = rightAssignments ->
      PiCCS.Arithmetization semantics params sumcheckOps
        attempt.piCcs leftAssignments) :
    Composition.InputsValid semantics params attempt.piCcs \/
      Composition.BadEvent semantics params bindingOps sampling
        attempt.piCcs attempt.piRlc.inputs := by
  exact Composition.fold_knowledge_or_bad_event
    semantics params sumcheckOps rlcAlgebra decAlgebra bindingOps arity sampling
    attempt.piCcs attempt.piRlc attempt.piDec finalAssignments kPositive
    accepted.wiring.ccsToRlc accepted.wiring.rlcToDec
    accepted.piCcs accepted.piRlc accepted.piDec finalValid extractor uniqueness
    rewindArithmetization

/--
Perfect completeness of the independent three-phase NIFS semantics.

Given honest source openings, honest SumCheck transcripts, one verifier-owned
shared point, and valid Pi_RLC challenges, the canonical Pi_CCS outputs,
Pi_RLC combination, and Pi_DEC digit children form an accepted NIFS attempt.
-/
theorem complete
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
    (arity : BatchArity params)
    (system : Structure)
    (point : Point)
    (input : PiCCS.InputProduct
      Structure PublicInput Point Evaluation Commitment params arity)
    (sourceAssignments : Fin arity.total -> Assignment)
    (fe nc : SumCheck.Instance Challenge Value)
    (challenges : Fin arity.total -> Scalar)
    (sourceFresh : forall index, (input.source index).stage = .fresh)
    (sourceValid : forall index,
      (input.source index).Holds semantics params (sourceAssignments index))
    (sameStructure : forall index,
      (input.source index).constraintSystem = system)
    (pointValid : semantics.evaluationPointValid system point)
    (feTruth : SumCheck.TruthPath sumcheckOps fe)
    (ncTruth : SumCheck.TruthPath sumcheckOps nc)
    (feHonest : SumCheck.Honest fe)
    (ncHonest : SumCheck.Honest nc)
    (challengesValid : forall index,
      rlcAlgebra.challengeValid (challenges index)) :
    let ccsOutputs := PiCCS.honestOutputs semantics input sourceAssignments point
    let ccsAttempt : PiCCS.Attempt
        Structure PublicInput Point Evaluation Commitment Challenge Value params arity := {
      inputs := input
      outputs := ccsOutputs
      fe := fe
      nc := nc
    }
    let rlcOutput := PiRLC.combinedOutput
      rlcAlgebra system point ccsOutputs challenges
    let rlcAttempt : PiRLC.Attempt
        Structure PublicInput Point Evaluation Commitment Scalar params arity := {
      inputs := ccsOutputs
      challenges := challenges
      output := rlcOutput
    }
    let combinedAssignment := PiRLC.combinedWitness
      rlcAlgebra challenges sourceAssignments
    let decAttempt : PiDEC.Attempt
        Structure PublicInput Point Evaluation Commitment params := {
      parent := rlcOutput
      children := PiDEC.childrenOf decAlgebra rlcOutput combinedAssignment
    }
    let nifsAttempt : Attempt
        Structure PublicInput Point Evaluation Commitment Scalar Challenge Value
          params arity := {
      piCcs := ccsAttempt
      piRlc := rlcAttempt
      piDec := decAttempt
    }
    Accepted sumcheckOps rlcAlgebra decAlgebra nifsAttempt /\
      forall index, CE.Holds semantics params (nifsAttempt.piDec.children index)
        (decAlgebra.splitAssignment combinedAssignment index) := by
  dsimp only
  have sourcePointValid : forall index,
      semantics.evaluationPointValid (input.source index).constraintSystem point := by
    intro index
    simpa [sameStructure index] using pointValid
  have ccsComplete := PiCCS.complete
    semantics params sumcheckOps arity input sourceAssignments point fe nc
    sourceFresh sourceValid sourcePointValid feTruth ncTruth feHonest ncHonest
  have rlcInputFresh : forall index,
      (PiCCS.honestOutputs semantics input sourceAssignments point index).stage =
        .fresh := by
    intro index
    rfl
  have rlcSameStructure : forall index,
      (PiCCS.honestOutputs semantics input sourceAssignments point index).constraintSystem =
        system := by
    intro index
    exact sameStructure index
  have rlcSamePoint : forall index,
      (PiCCS.honestOutputs semantics input sourceAssignments point index).point = point := by
    intro index
    rfl
  have rlcComplete := PiRLC.complete
    semantics params rlcAlgebra arity system point
    (PiCCS.honestOutputs semantics input sourceAssignments point)
    challenges sourceAssignments rlcInputFresh rlcSameStructure rlcSamePoint
    challengesValid ccsComplete.2 pointValid
  have decComplete := PiDEC.complete
    semantics params decAlgebra
    (PiRLC.combinedOutput rlcAlgebra system point
      (PiCCS.honestOutputs semantics input sourceAssignments point) challenges)
    (PiRLC.combinedWitness rlcAlgebra challenges sourceAssignments)
    rfl rlcComplete.2
  constructor
  · exact {
      wiring := {
        ccsToRlc := fun _ => rfl
        rlcToDec := rfl
      }
      piCcs := ccsComplete.1
      piRlc := rlcComplete.1
      piDec := decComplete.1
    }
  · exact decComplete.2

/-- Honest source openings realize the externally visible NIFS transition. -/
theorem paperNifsTransition_complete
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
    (arity : BatchArity params)
    (system : Structure)
    (point : Point)
    (input : PiCCS.InputProduct
      Structure PublicInput Point Evaluation Commitment params arity)
    (sourceAssignments : Fin arity.total -> Assignment)
    (fe nc : SumCheck.Instance Challenge Value)
    (challenges : Fin arity.total -> Scalar)
    (sourceFresh : forall index, (input.source index).stage = .fresh)
    (sourceValid : forall index,
      (input.source index).Holds semantics params (sourceAssignments index))
    (sameStructure : forall index,
      (input.source index).constraintSystem = system)
    (pointValid : semantics.evaluationPointValid system point)
    (feTruth : SumCheck.TruthPath sumcheckOps fe)
    (ncTruth : SumCheck.TruthPath sumcheckOps nc)
    (feHonest : SumCheck.Honest fe)
    (ncHonest : SumCheck.Honest nc)
    (challengesValid : forall index,
      rlcAlgebra.challengeValid (challenges index)) :
    PaperNifsTransition sumcheckOps rlcAlgebra decAlgebra input
      (PiDEC.childrenOf decAlgebra
        (PiRLC.combinedOutput rlcAlgebra system point
          (PiCCS.honestOutputs semantics input sourceAssignments point) challenges)
        (PiRLC.combinedWitness rlcAlgebra challenges sourceAssignments)) := by
  let ccsOutputs := PiCCS.honestOutputs semantics input sourceAssignments point
  let ccsAttempt : PiCCS.Attempt
      Structure PublicInput Point Evaluation Commitment Challenge Value params arity := {
    inputs := input
    outputs := ccsOutputs
    fe := fe
    nc := nc
  }
  let rlcOutput := PiRLC.combinedOutput
    rlcAlgebra system point ccsOutputs challenges
  let rlcAttempt : PiRLC.Attempt
      Structure PublicInput Point Evaluation Commitment Scalar params arity := {
    inputs := ccsOutputs
    challenges := challenges
    output := rlcOutput
  }
  let combinedAssignment := PiRLC.combinedWitness
    rlcAlgebra challenges sourceAssignments
  let decAttempt : PiDEC.Attempt
      Structure PublicInput Point Evaluation Commitment params := {
    parent := rlcOutput
    children := PiDEC.childrenOf decAlgebra rlcOutput combinedAssignment
  }
  let nifsAttempt : Attempt
      Structure PublicInput Point Evaluation Commitment Scalar Challenge Value
        params arity := {
    piCcs := ccsAttempt
    piRlc := rlcAttempt
    piDec := decAttempt
  }
  have honest := complete semantics params sumcheckOps rlcAlgebra decAlgebra
    arity system point input sourceAssignments fe nc challenges sourceFresh
    sourceValid sameStructure pointValid feTruth ncTruth feHonest ncHonest
    challengesValid
  dsimp only at honest
  have transition := paperNifsTransition_of_accepted
    sumcheckOps rlcAlgebra decAlgebra nifsAttempt honest.1
  simpa [nifsAttempt, decAttempt, combinedAssignment, rlcAttempt, rlcOutput,
    ccsAttempt, ccsOutputs] using transition

end Nightstream.SuperNeo.Folding.Nifs
