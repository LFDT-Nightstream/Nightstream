import Nightstream.SuperNeo.Folding.Nifs

/-!
Shared-carrier semantics for the two internal SuperNeo NIFS boundaries.

Owns: one representation in which the complete Pi_CCS output product is the
Pi_RLC input product by construction, and the Pi_RLC output is the Pi_DEC
parent by construction. It also proves exact equivalence with the independent
candidate NIFS relation whenever the latter's explicit wiring checks hold.

Does not own: Fiat--Shamir, Poseidon2, concrete encodings, Rust allocation,
R1CS column aliasing, row counts, or permission to remove a constraint.

Emits constraints: no.

Authority boundary: this file removes duplicate *semantic data*, not verifier
obligations. Each phase still has its complete independent acceptance
predicate. A production circuit may share columns at these boundaries only
after an exact Rust/R1CS refinement theorem proves that its aliases implement
this representation.

| Protocol | Phase boundary | Shared mathematical object | Derived consumer | Lean owner |
|---|---|---|---|---|
| NIFS | `Pi_CCS -> Pi_RLC` | complete vector `piCcs.outputs` | `piRlc.inputs` | `SharedAttempt.piRlc` |
| NIFS | `Pi_RLC -> Pi_DEC` | one combined CE instance `piRlcOutput` | `piDec.parent` | `SharedAttempt.piDec` |
| NIFS | all phases | shared phase data plus challenges and DEC children | ordinary three-phase attempt | `SharedAttempt.toAttempt` |
| assurance | definitional wiring | both boundary equalities hold by reflexivity | `Nifs.Wiring` | `SharedAttempt.wiring` |
| assurance | normalization | every accepted explicitly wired attempt equals its shared form | ordinary attempt | `normalize_toAttempt_eq` |
| assurance | public relation | shared and explicit candidate transition relations are equivalent | `PaperNifsTransition` | `sharedPaperNifsTransition_iff_paperNifsTransition` |
-/

namespace Nightstream.SuperNeo.Folding.Nifs

universe uStructure uAssignment uPublicInput uPoint uEvaluation uCommitment
  uScalar uChallenge uValue

/--
One NIFS attempt with both sequential phase boundaries shared by construction.

There is no separately supplied `PiRLC.inputs` or `PiDEC.parent` field. Those
views are derived below from the unique authoritative carriers.
-/
structure SharedAttempt
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
  piRlcChallenges : Fin arity.total -> Scalar
  piRlcOutput : CE.Instance Structure PublicInput Point Evaluation Commitment
  piDecChildren : Fin params.k ->
    CE.Instance Structure PublicInput Point Evaluation Commitment

namespace SharedAttempt

/-- The Pi_RLC view; its input vector is the unique Pi_CCS output vector. -/
def piRlc
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
    (attempt : SharedAttempt
      Structure PublicInput Point Evaluation Commitment Scalar Challenge Value
        params arity) :
    PiRLC.Attempt
      Structure PublicInput Point Evaluation Commitment Scalar params arity := {
  inputs := attempt.piCcs.outputs
  challenges := attempt.piRlcChallenges
  output := attempt.piRlcOutput
}

/-- The Pi_DEC view; its parent is the unique Pi_RLC output instance. -/
def piDec
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
    (attempt : SharedAttempt
      Structure PublicInput Point Evaluation Commitment Scalar Challenge Value
        params arity) :
    PiDEC.Attempt Structure PublicInput Point Evaluation Commitment params := {
  parent := attempt.piRlcOutput
  children := attempt.piDecChildren
}

/-- Forget only that the two phase boundaries were shared by construction. -/
def toAttempt
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
    (attempt : SharedAttempt
      Structure PublicInput Point Evaluation Commitment Scalar Challenge Value
        params arity) :
    Attempt Structure PublicInput Point Evaluation Commitment Scalar Challenge
      Value params arity := {
  piCcs := attempt.piCcs
  piRlc := attempt.piRlc
  piDec := attempt.piDec
}

/-- Both explicit NIFS wiring obligations hold definitionally. -/
theorem wiring
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
    (attempt : SharedAttempt
      Structure PublicInput Point Evaluation Commitment Scalar Challenge Value
        params arity) :
    Wiring attempt.toAttempt := by
  exact {
    ccsToRlc := fun _ => rfl
    rlcToDec := rfl
  }

end SharedAttempt

/-- Phase acceptance over the shared representation. Wiring is not a field
because it is a definitional property of `SharedAttempt`. -/
structure SharedAccepted
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
    (attempt : SharedAttempt
      Structure PublicInput Point Evaluation Commitment Scalar Challenge Value
        params arity) : Prop where
  piCcs : PiCCS.Accepted sumcheckOps attempt.piCcs
  piRlc : PiRLC.Accepted rlcAlgebra attempt.piRlc
  piDec : PiDEC.Accepted decAlgebra attempt.piDec

/-- Shared acceptance implies ordinary acceptance without any additional
boundary premise. -/
theorem SharedAccepted.toAccepted
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
    {sumcheckOps : SumCheck.Ops Challenge Value}
    {rlcAlgebra : PiRLC.Algebra
      Structure Assignment PublicInput Point Evaluation Commitment Scalar semantics params}
    {decAlgebra : PiDEC.Algebra
      Structure Assignment PublicInput Point Evaluation Commitment semantics params}
    {attempt : SharedAttempt
      Structure PublicInput Point Evaluation Commitment Scalar Challenge Value
        params arity}
    (accepted : SharedAccepted sumcheckOps rlcAlgebra decAlgebra attempt) :
    Accepted sumcheckOps rlcAlgebra decAlgebra attempt.toAttempt := by
  exact {
    wiring := attempt.wiring
    piCcs := accepted.piCcs
    piRlc := accepted.piRlc
    piDec := accepted.piDec
  }

/-- Canonical shared form of an ordinary attempt. Boundary copies are discarded
in favor of the preceding phase's authoritative output. -/
def normalize
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
        params arity) :
    SharedAttempt Structure PublicInput Point Evaluation Commitment Scalar
      Challenge Value params arity := {
  piCcs := attempt.piCcs
  piRlcChallenges := attempt.piRlc.challenges
  piRlcOutput := attempt.piRlc.output
  piDecChildren := attempt.piDec.children
}

/-- Explicit wiring is exactly the condition under which canonical shared
normalization reconstructs the original ordinary attempt. -/
theorem normalize_toAttempt_eq
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
        params arity)
    (wired : Wiring attempt) :
    (normalize attempt).toAttempt = attempt := by
  cases attempt with
  | mk piCcs piRlc piDec =>
      cases piRlc with
      | mk inputs challenges output =>
          cases piDec with
          | mk parent children =>
              have inputsEq : inputs = piCcs.outputs :=
                funext wired.ccsToRlc
              have parentEq : parent = output := wired.rlcToDec
              subst inputs
              subst parent
              rfl

/-- Every accepted ordinary attempt has an accepted shared normalization. -/
theorem accepted_normalize
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
    {sumcheckOps : SumCheck.Ops Challenge Value}
    {rlcAlgebra : PiRLC.Algebra
      Structure Assignment PublicInput Point Evaluation Commitment Scalar semantics params}
    {decAlgebra : PiDEC.Algebra
      Structure Assignment PublicInput Point Evaluation Commitment semantics params}
    {attempt : Attempt
      Structure PublicInput Point Evaluation Commitment Scalar Challenge Value
        params arity}
    (accepted : Accepted sumcheckOps rlcAlgebra decAlgebra attempt) :
    SharedAccepted sumcheckOps rlcAlgebra decAlgebra (normalize attempt) := by
  have normalizedAccepted : Accepted sumcheckOps rlcAlgebra decAlgebra
      (normalize attempt).toAttempt := by
    rw [normalize_toAttempt_eq attempt accepted.wiring]
    exact accepted
  exact {
    piCcs := normalizedAccepted.piCcs
    piRlc := normalizedAccepted.piRlc
    piDec := normalizedAccepted.piDec
  }

/-- Candidate public NIFS transition with internal phase-boundary sharing made
explicit in the witness type. -/
def SharedPaperNifsTransition
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
  exists attempt : SharedAttempt
      Structure PublicInput Point Evaluation Commitment Scalar Challenge Value
        params arity,
    attempt.piCcs.inputs = input /\
      attempt.piDecChildren = output /\
      SharedAccepted sumcheckOps rlcAlgebra decAlgebra attempt

/-- Sharing the internal carriers does not strengthen or weaken the independent
candidate NIFS transition relation. -/
theorem sharedPaperNifsTransition_iff_paperNifsTransition
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
      CE.Instance Structure PublicInput Point Evaluation Commitment) :
    SharedPaperNifsTransition sumcheckOps rlcAlgebra decAlgebra input output <->
      PaperNifsTransition sumcheckOps rlcAlgebra decAlgebra input output := by
  constructor
  · rintro ⟨attempt, inputEq, outputEq, accepted⟩
    exact ⟨attempt.toAttempt, inputEq, outputEq, accepted.toAccepted⟩
  · rintro ⟨attempt, inputEq, outputEq, accepted⟩
    exact ⟨normalize attempt, inputEq, outputEq, accepted_normalize accepted⟩

end Nightstream.SuperNeo.Folding.Nifs
