import Nightstream.SuperNeo.Folding.Nifs.NonInteractive.Replay

/-!
Partial replay wrapper for the challenge carriers currently present in
`Nifs.Attempt`.

Owns: one fixed `Nifs.Attempt`, public input/output identity, setup-structure
binding, the current candidate phase acceptance, canonical partial transcript
replay, and projection back to that same interactive candidate relation.

Does not own: PiCCS alpha/gamma carriers, concrete encoding or hashing,
random-oracle security, bounded PiRLC sampling, backend refinement, cost
accounting, or check-removal authority.

Emits backend obligations: no.

Authority boundary: `ReplayBoundExecution` is only a wrapper around
`Nifs.Accepted`. It additionally requires the represented SumCheck and PiRLC
challenge carriers to replay from the typed partial prefix. It does not bind
SumCheck initial/terminal values, verifier parameters, exact round counts, the
paper joint-Q/production SplitNc relation, output-projection sufficiency, or a
concrete oracle. A constant or colliding `Replay.Oracle` remains possible.

| Protocol | Phase | Constraint family | Mathematical obligation |
|---|---|---|---|
| NIFS | setup | statement identity | context, attempt, and external input are literally the same product |
| NIFS | setup | structure authority | every source is bound to the context's expected structure |
| NIFS | PiCCS/PiRLC/PiDEC | fixed attempt | `ReplayBoundAttempt` binds one specified attempt rather than hiding it existentially |
| NIFS | PiCCS/PiRLC/PiDEC | core acceptance | retain the independent `Nifs.Accepted` proof without redefining its equations |
| NIFS | Fiat--Shamir replay | challenge carrier equality | canonical carried events replay from the typed context seed and exact message chain |
| NIFS | candidate transition | input/output projection | expose the existing candidate `Nifs.PaperNifsTransition`; joint-Q and security bridges remain open |
-/

namespace Nightstream.SuperNeo.Folding.Nifs.NonInteractive

universe uPublicParameters uVerifierKey uStructure uAssignment uPublicInput
  uPoint uEvaluation uCommitment uScalar uChallenge uValue uDigest

/-- Obligations for one fixed, reviewable NIFS attempt. -/
def ReplayBoundAttempt
    {PublicParameters : Type uPublicParameters}
    {VerifierKey : Type uVerifierKey}
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
      Structure Assignment PublicInput Point Evaluation Commitment Scalar
        semantics params)
    (decAlgebra : PiDEC.Algebra
      Structure Assignment PublicInput Point Evaluation Commitment semantics params)
    (oracle : Replay.Oracle
      PublicParameters VerifierKey Structure PublicInput Point Evaluation
        Commitment Scalar Challenge Value params arity)
    (context : Context
      PublicParameters VerifierKey Structure PublicInput Point Evaluation
        Commitment params arity)
    (attempt : Nifs.Attempt
      Structure PublicInput Point Evaluation Commitment Scalar Challenge Value
        params arity)
    (input : PiCCS.InputProduct
      Structure PublicInput Point Evaluation Commitment params arity)
    (output : Fin params.k ->
      CE.Instance Structure PublicInput Point Evaluation Commitment) : Prop :=
  context.statement = input /\
    attempt.piCcs.inputs = input /\
    attempt.piDec.children = output /\
    (forall index,
      (attempt.piCcs.inputs.source index).constraintSystem =
        context.relationStructure) /\
    Nifs.Accepted sumcheckOps rlcAlgebra decAlgebra attempt /\
    Replay.AcceptsCanonical oracle context attempt

/-- Public projection which hides only the fixed attempt certificate. -/
def ReplayBoundExecution
    {PublicParameters : Type uPublicParameters}
    {VerifierKey : Type uVerifierKey}
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
      Structure Assignment PublicInput Point Evaluation Commitment Scalar
        semantics params)
    (decAlgebra : PiDEC.Algebra
      Structure Assignment PublicInput Point Evaluation Commitment semantics params)
    (oracle : Replay.Oracle
      PublicParameters VerifierKey Structure PublicInput Point Evaluation
        Commitment Scalar Challenge Value params arity)
    (context : Context
      PublicParameters VerifierKey Structure PublicInput Point Evaluation
        Commitment params arity)
    (input : PiCCS.InputProduct
      Structure PublicInput Point Evaluation Commitment params arity)
    (output : Fin params.k ->
      CE.Instance Structure PublicInput Point Evaluation Commitment) : Prop :=
  exists attempt : Nifs.Attempt
      Structure PublicInput Point Evaluation Commitment Scalar Challenge Value
        params arity,
    ReplayBoundAttempt sumcheckOps rlcAlgebra decAlgebra oracle context
      attempt input output

/-- A fixed replay-bound attempt projects only to the existing candidate NIFS
relation. This theorem is bookkeeping; it is not recursive-verifier
soundness. -/
theorem replayBoundAttempt_implies_candidateNifsTransition
    {PublicParameters : Type uPublicParameters}
    {VerifierKey : Type uVerifierKey}
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
      Structure Assignment PublicInput Point Evaluation Commitment Scalar
        semantics params)
    (decAlgebra : PiDEC.Algebra
      Structure Assignment PublicInput Point Evaluation Commitment semantics params)
    (oracle : Replay.Oracle
      PublicParameters VerifierKey Structure PublicInput Point Evaluation
        Commitment Scalar Challenge Value params arity)
    (context : Context
      PublicParameters VerifierKey Structure PublicInput Point Evaluation
        Commitment params arity)
    (attempt : Nifs.Attempt
      Structure PublicInput Point Evaluation Commitment Scalar Challenge Value
        params arity)
    (input : PiCCS.InputProduct
      Structure PublicInput Point Evaluation Commitment params arity)
    (output : Fin params.k ->
      CE.Instance Structure PublicInput Point Evaluation Commitment)
    (execution : ReplayBoundAttempt
      sumcheckOps rlcAlgebra decAlgebra oracle context attempt input output) :
    Nifs.PaperNifsTransition
      sumcheckOps rlcAlgebra decAlgebra input output := by
  rcases execution with
    ⟨_, attemptInput, attemptOutput, _, coreAccepted, _⟩
  exact ⟨attempt, attemptInput, attemptOutput, coreAccepted⟩

/-- The existential replay wrapper projects to the same candidate relation. -/
theorem replayBoundExecution_implies_candidateNifsTransition
    {PublicParameters : Type uPublicParameters}
    {VerifierKey : Type uVerifierKey}
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
      Structure Assignment PublicInput Point Evaluation Commitment Scalar
        semantics params)
    (decAlgebra : PiDEC.Algebra
      Structure Assignment PublicInput Point Evaluation Commitment semantics params)
    (oracle : Replay.Oracle
      PublicParameters VerifierKey Structure PublicInput Point Evaluation
        Commitment Scalar Challenge Value params arity)
    (context : Context
      PublicParameters VerifierKey Structure PublicInput Point Evaluation
        Commitment params arity)
    (input : PiCCS.InputProduct
      Structure PublicInput Point Evaluation Commitment params arity)
    (output : Fin params.k ->
      CE.Instance Structure PublicInput Point Evaluation Commitment)
    (execution : ReplayBoundExecution
      sumcheckOps rlcAlgebra decAlgebra oracle context input output) :
    Nifs.PaperNifsTransition
      sumcheckOps rlcAlgebra decAlgebra input output := by
  rcases execution with
    ⟨attempt, fixedExecution⟩
  exact replayBoundAttempt_implies_candidateNifsTransition
    sumcheckOps rlcAlgebra decAlgebra oracle context attempt input output
    fixedExecution

/--
Constructor for one fixed replay-bound attempt in this partial slice.

Core acceptance and setup binding remain independent semantic obligations. The
only transcript premise is exact equality between carried events and the
oracle's deterministic materialization; replay itself is then constructed.
This is not completeness or soundness and does not discharge the open
refinement boundaries listed in the event-carrier contract or bridge spec.
-/
theorem replayBoundAttempt_of_core_and_carrierAgreement
    {PublicParameters : Type uPublicParameters}
    {VerifierKey : Type uVerifierKey}
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
      Structure Assignment PublicInput Point Evaluation Commitment Scalar
        semantics params)
    (decAlgebra : PiDEC.Algebra
      Structure Assignment PublicInput Point Evaluation Commitment semantics params)
    (oracle : Replay.Oracle
      PublicParameters VerifierKey Structure PublicInput Point Evaluation
        Commitment Scalar Challenge Value params arity)
    (context : Context
      PublicParameters VerifierKey Structure PublicInput Point Evaluation
        Commitment params arity)
    (input : PiCCS.InputProduct
      Structure PublicInput Point Evaluation Commitment params arity)
    (output : Fin params.k ->
      CE.Instance Structure PublicInput Point Evaluation Commitment)
    (attempt : Nifs.Attempt
      Structure PublicInput Point Evaluation Commitment Scalar Challenge Value
        params arity)
    (contextInput : context.statement = input)
    (attemptInput : attempt.piCcs.inputs = input)
    (attemptOutput : attempt.piDec.children = output)
    (sourceStructure : forall index,
      (attempt.piCcs.inputs.source index).constraintSystem =
        context.relationStructure)
    (coreAccepted : Nifs.Accepted
      sumcheckOps rlcAlgebra decAlgebra attempt)
    (carrierAgreement : canonicalEvents attempt =
      Replay.materializedSchedule oracle context attempt) :
    ReplayBoundAttempt
      sumcheckOps rlcAlgebra decAlgebra oracle context attempt input output := by
  exact ⟨contextInput, attemptInput, attemptOutput, sourceStructure,
    coreAccepted, Replay.accepts_of_carrierAgreement
      oracle context attempt carrierAgreement⟩

/-- Public existential constructor from the same explicit premises. -/
theorem replayBoundExecution_of_core_and_carrierAgreement
    {PublicParameters : Type uPublicParameters}
    {VerifierKey : Type uVerifierKey}
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
      Structure Assignment PublicInput Point Evaluation Commitment Scalar
        semantics params)
    (decAlgebra : PiDEC.Algebra
      Structure Assignment PublicInput Point Evaluation Commitment semantics params)
    (oracle : Replay.Oracle
      PublicParameters VerifierKey Structure PublicInput Point Evaluation
        Commitment Scalar Challenge Value params arity)
    (context : Context
      PublicParameters VerifierKey Structure PublicInput Point Evaluation
        Commitment params arity)
    (input : PiCCS.InputProduct
      Structure PublicInput Point Evaluation Commitment params arity)
    (output : Fin params.k ->
      CE.Instance Structure PublicInput Point Evaluation Commitment)
    (attempt : Nifs.Attempt
      Structure PublicInput Point Evaluation Commitment Scalar Challenge Value
        params arity)
    (contextInput : context.statement = input)
    (attemptInput : attempt.piCcs.inputs = input)
    (attemptOutput : attempt.piDec.children = output)
    (sourceStructure : forall index,
      (attempt.piCcs.inputs.source index).constraintSystem =
        context.relationStructure)
    (coreAccepted : Nifs.Accepted
      sumcheckOps rlcAlgebra decAlgebra attempt)
    (carrierAgreement : canonicalEvents attempt =
      Replay.materializedSchedule oracle context attempt) :
    ReplayBoundExecution
      sumcheckOps rlcAlgebra decAlgebra oracle context input output := by
  exact ⟨attempt,
    replayBoundAttempt_of_core_and_carrierAgreement
      sumcheckOps rlcAlgebra decAlgebra oracle context input output attempt
      contextInput attemptInput attemptOutput sourceStructure coreAccepted
      carrierAgreement⟩

end Nightstream.SuperNeo.Folding.Nifs.NonInteractive
