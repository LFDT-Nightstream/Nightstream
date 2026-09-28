import NightstreamFPrime.Spec.Folding.Nifs.PaperWeakSuffix
import NightstreamFPrime.Spec.Folding.PiRLC.PaperCompleteness

/-!
Honest completeness and public-coin ownership of the same PiRLC/PiDEC suffix.
The public verifier takes only the public challenge vector and child messages.
Final witness checks belong to the extractor, not this verifier decision.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Folding.Nifs.PaperWeakCompleteness

open NightstreamFPrime.Spec
open _root_.NightstreamFPrime.Spec.Folding
open PiRLC.PaperForkExtraction PiRLC.CoordinateForkLaw

variable {Structure Assignment PublicInput Point Evaluation Commitment Scalar : Type*}
  {semantics : RelationSemantics Structure Assignment PublicInput Point Evaluation Commitment}
  {params : GlobalParams} {arity : BatchArity params}
  (rlc : PiRLC.Algebra Structure Assignment PublicInput Point Evaluation Commitment Scalar semantics params)
  (batch : InputBatch Structure PublicInput Point Evaluation Commitment params arity)
  (dec : PiDEC.Algebra Structure Assignment PublicInput Point Evaluation Commitment semantics params)
  (publicSplit : PiDEC.PaperVerifier.PublicInputSplit dec)
  (evaluationArity : PiDEC.PaperVerifier.EvaluationArity semantics)

/-- The verifier computes the parent from public coins, then receives only
the PiDEC public child messages. There is no private witness argument. -/
def publicAttempt (vector : Fin arity.total → Challenge rlc)
    (messages : Fin params.k → PiDEC.PaperVerifier.ChildMessage Evaluation Commitment) :
    PiDEC.PaperVerifier.Attempt Structure PublicInput Point Evaluation Commitment params where
  parent := PiRLC.combinedOutput rlc batch.system batch.point batch.inputs (scalarVector rlc vector)
  messages := messages

/-- The suffix extractor and the public verifier use exactly the same public
attempt; the reply's private assignments cannot change it. -/
theorem attempt_eq_publicAttempt (vector : Fin arity.total → Challenge rlc)
    (reply : PaperWeakSuffix.Reply Assignment Evaluation Commitment params) :
    PaperWeakSuffix.attempt rlc batch vector reply = publicAttempt rlc batch vector reply.messages := rfl

/-- The actual PiDEC decision procedure supplies the deterministic final check
after the sole public random message, the PiRLC challenge vector. -/
def publicAccepts [DecidableEq Commitment] [DecidableEq Evaluation]
    (vector : Fin arity.total → Challenge rlc)
    (messages : Fin params.k → PiDEC.PaperVerifier.ChildMessage Evaluation Commitment) : Bool :=
  letI := PiDEC.PaperVerifier.acceptedDecision dec publicSplit evaluationArity
    (publicAttempt rlc batch vector messages)
  decide (PiDEC.PaperVerifier.Accepted dec publicSplit evaluationArity
    (publicAttempt rlc batch vector messages))

theorem publicAccepts_iff [DecidableEq Commitment] [DecidableEq Evaluation]
    (vector : Fin arity.total → Challenge rlc)
    (messages : Fin params.k → PiDEC.PaperVerifier.ChildMessage Evaluation Commitment) :
    publicAccepts rlc batch dec publicSplit evaluationArity vector messages = true ↔
      PiDEC.PaperVerifier.Accepted dec publicSplit evaluationArity
        (publicAttempt rlc batch vector messages) := by
  letI := PiDEC.PaperVerifier.acceptedDecision dec publicSplit evaluationArity
    (publicAttempt rlc batch vector messages)
  unfold publicAccepts
  exact decide_eq_true_iff

/-- The honest prover splits exactly the assignment combined under the
verifier's public vector. Both messages and witnesses come from that split. -/
def honestReply (vector : Fin arity.total → Challenge rlc)
    (assignments : Fin arity.total → Assignment) :
    PaperWeakSuffix.Reply Assignment Evaluation Commitment params where
  messages := PiDEC.PaperVerifier.honestMessages dec
    (PiRLC.combinedOutput rlc batch.system batch.point batch.inputs (scalarVector rlc vector))
    (PiRLC.combinedWitness rlc (scalarVector rlc vector) assignments)
  assignments := dec.splitAssignment (PiRLC.combinedWitness rlc (scalarVector rlc vector) assignments)

/-- Existing PiRLC completeness supplies the parent opening; existing PiDEC
completeness supplies all final child openings. Neither is an output premise. -/
theorem honest_complete (vector : Fin arity.total → Challenge rlc)
    (assignments : Fin arity.total → Assignment)
    (inputFresh : ∀ index, (batch.inputs index).stage = .fresh)
    (inputHolds : ∀ index, CE.Holds semantics params (batch.inputs index) (assignments index)) :
    let reply := honestReply rlc batch dec vector assignments
    PiDEC.PaperVerifier.Accepted dec publicSplit evaluationArity
      (PaperWeakSuffix.attempt rlc batch vector reply) ∧
    ∀ child, CE.Holds semantics params
      (PiDEC.PaperVerifier.children publicSplit (PaperWeakSuffix.attempt rlc batch vector reply) child)
      (reply.assignments child) := by
  let context : PiRLC.PaperCompleteness.Context Structure Assignment PublicInput Point Evaluation
      Commitment Scalar := {
    semantics := semantics
    params := params
    arity := arity
    algebra := rlc
    evaluationCount := evaluationArity.count
    evaluationsSize := evaluationArity.evaluations_size
  }
  let coins : PiRLC.PaperCompleteness.PublicCoins context := {
    challenges := scalarVector rlc vector
    valid := fun index => (vector index).property
  }
  have parent := PiRLC.PaperCompleteness.honestResponse_success_of_inputHolds
    context batch assignments coins inputFresh inputHolds
  exact PiDEC.PaperVerifier.complete semantics params dec publicSplit evaluationArity
    (PiRLC.combinedOutput rlc batch.system batch.point batch.inputs (scalarVector rlc vector))
    (PiRLC.combinedWitness rlc (scalarVector rlc vector) assignments) rfl parent

/-- The same public verifier accepts the honest suffix on every allowed
public vector, with no conditioning or private verifier randomness. -/
theorem honest_public_accepts [DecidableEq Commitment] [DecidableEq Evaluation]
    (vector : Fin arity.total → Challenge rlc) (assignments : Fin arity.total → Assignment)
    (inputFresh : ∀ index, (batch.inputs index).stage = .fresh)
    (inputHolds : ∀ index, CE.Holds semantics params (batch.inputs index) (assignments index)) :
    publicAccepts rlc batch dec publicSplit evaluationArity vector
      (honestReply rlc batch dec vector assignments).messages = true := by
  apply (publicAccepts_iff rlc batch dec publicSplit evaluationArity vector _).mpr
  exact (honest_complete rlc batch dec publicSplit evaluationArity vector assignments inputFresh inputHolds).1

/-- PiRLC's existing public-coin theorem identifies its public output with
the exact parent consumed by the final PiDEC verifier. -/
theorem public_parent
    (context : PiRLC.PaperCompleteness.Context Structure Assignment PublicInput Point Evaluation
      Commitment Scalar)
    (inputs : InputBatch Structure PublicInput Point Evaluation Commitment context.params context.arity)
    (assignments : Fin context.arity.total → Assignment)
    (vector : Fin context.arity.total → Challenge context.algebra)
    (messages : Fin context.params.k → PiDEC.PaperVerifier.ChildMessage Evaluation Commitment) :
    (PiRLC.PaperCompleteness.honestResponse context assignments
      ⟨scalarVector context.algebra vector, fun index => (vector index).property⟩).output
      context.algebra inputs = (publicAttempt context.algebra inputs vector messages).parent :=
  PiRLC.PaperCompleteness.publicCoin context inputs assignments
    ⟨scalarVector context.algebra vector, fun index => (vector index).property⟩

end NightstreamFPrime.Spec.Folding.Nifs.PaperWeakCompleteness
