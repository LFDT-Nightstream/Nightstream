import NightstreamFPrime.Spec.Folding.Nifs.PaperWeakOracle
import NightstreamFPrime.Spec.Folding.Nifs.PaperWeakLaw
import NightstreamFPrime.Spec.Folding.PiRLC.CoordinateExtraction

/-!
The weak NIFS extractor resumes the actual PiRLC/PiDEC suffix at each queried
challenge vector. Its base success event is the original accepted final
output with that invocation's own child witnesses. The uniform interactive
loss is transferred to the existing stopped extractor and its returned list.
Fiat–Shamir replay and bounded Poseidon sampling are separate obligations.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Folding.Nifs.PaperWeakExtraction

attribute [local instance] Classical.propDecidable

open scoped BigOperators
open NightstreamFPrime.Spec
open _root_.NightstreamFPrime.Spec.Folding
open PiRLC.PaperForkExtraction PiRLC.CoordinateForkLaw
open PiRLC.CoordinateOracle PiRLC.CoordinateChargedOracle PiRLC.CoordinateCheckedCalls

variable {Tape Structure Assignment PublicInput Point Evaluation Commitment Scalar : Type*}
  {semantics : RelationSemantics Structure Assignment PublicInput Point Evaluation Commitment}
  {params : GlobalParams} {arity : BatchArity params}
  (rlc : PiRLC.Algebra Structure Assignment PublicInput Point Evaluation Commitment Scalar semantics params)
  (batch : InputBatch Structure PublicInput Point Evaluation Commitment params arity)
  (dec : PiDEC.Algebra Structure Assignment PublicInput Point Evaluation Commitment semantics params)
  (publicSplit : PiDEC.PaperVerifier.PublicInputSplit dec)
  (evaluationArity : PiDEC.PaperVerifier.EvaluationArity semantics)

/-- The original suffix success event uses its public verifier acceptance and
the final child witnesses returned by this same invocation. -/
def FinalOutputSuccess (vector : Fin arity.total → Challenge rlc) :
    Option (PaperWeakSuffix.Reply Assignment Evaluation Commitment params) → Prop
  | none => False
  | some reply =>
      PiDEC.PaperVerifier.Accepted dec publicSplit evaluationArity
        (PaperWeakSuffix.attempt rlc batch vector reply) ∧
      ∀ child, CE.Holds semantics params
        (PiDEC.PaperVerifier.children publicSplit
          (PaperWeakSuffix.attempt rlc batch vector reply) child)
        (reply.assignments child)

/-- Returning a parent has exactly the original final-output success event.
The reverse direction does not assume an intermediate parent opening. -/
theorem run_return_iff
    (program : PaperWeakSuffix.Program (arity := arity) rlc)
    (correct : PaperWeakSuffix.Correct rlc batch dec publicSplit evaluationArity program)
    (call : PaperWeakOracle.Call (Tape := Tape) (arity := arity) rlc)
    (vector : Fin arity.total → Challenge rlc) (tape : Tape) :
    (PaperWeakOracle.run rlc program call vector tape).value.isSome = true ↔
      FinalOutputSuccess rlc batch dec publicSplit evaluationArity vector (call vector tape).value := by
  change (PaperWeakSuffix.finish rlc program vector (call vector tape).value).value.isSome = true ↔ _
  cases issued : (call vector tape).value with
  | none => simp only [PaperWeakSuffix.finish, FinalOutputSuccess, Option.isSome_none, Bool.false_eq_true]
  | some reply =>
      change (PaperWeakSuffix.finish rlc program vector (some reply)).value.isSome = true ↔
        PiDEC.PaperVerifier.Accepted dec publicSplit evaluationArity
          (PaperWeakSuffix.attempt rlc batch vector reply) ∧ _
      rw [← correct.check vector reply]
      cases checked : (program.check vector reply).value <;> simp [PaperWeakSuffix.finish, checked]

variable [Fintype (Challenge rlc)] [Nonempty (Challenge rlc)] [Fintype Assignment]

/-- Fresh private tapes and a uniform public challenge vector are sampled
independently. Neither distribution is conditioned on output validity. -/
noncomputable def successProbability (tapes : PMF Tape)
    (call : PaperWeakOracle.Call (Tape := Tape) (arity := arity) rlc) : ℝ := by
  classical
  exact 𝔼 vector, ∑' tape, (tapes tape).toReal *
    (if FinalOutputSuccess rlc batch dec publicSplit evaluationArity vector (call vector tape).value
      then (1 : ℝ) else 0)

omit [Fintype (Challenge rlc)] [Nonempty (Challenge rlc)] [Fintype Assignment] in
private theorem accepted_run
    (program : PaperWeakSuffix.Program (arity := arity) rlc)
    (correct : PaperWeakSuffix.Correct rlc batch dec publicSplit evaluationArity program)
    (kPositive : 0 < params.k)
    (call : PaperWeakOracle.Call (Tape := Tape) (arity := arity) rlc)
    (checker : Response Assignment Scalar params arity → CheckResult)
    (check_spec : ∀ found, (checker found).accepted = true ↔
      found.Success semantics params rlc batch)
    (vector : Fin arity.total → Challenge rlc) (tape : Tape) :
    accepted (oracleCheck rlc (fun found => (checker found).accepted)) vector
      (PaperWeakOracle.run rlc program call vector tape).value =
      (PaperWeakOracle.run rlc program call vector tape).value.isSome := by
  cases returned : (PaperWeakOracle.run rlc program call vector tape).value with
  | none => simp [accepted]
  | some assignment =>
      have valid := PaperWeakOracle.run_returns_parent rlc batch dec publicSplit evaluationArity
        program correct kPositive call vector tape assignment returned
      have checked := (check_spec (response rlc vector assignment)).mpr valid
      simpa only [accepted, Option.map_some, Option.getD_some, Option.isSome_some]
        using checked

omit [Fintype (Challenge rlc)] [Nonempty (Challenge rlc)] in
/-- At each public vector, accepted parent mass equals the actual original
suffix success probability. The extra CE check changes only the clock. -/
theorem acceptance_eq
    (tapes : PMF Tape) (program : PaperWeakSuffix.Program (arity := arity) rlc)
    (correct : PaperWeakSuffix.Correct rlc batch dec publicSplit evaluationArity program)
    (kPositive : 0 < params.k)
    (call : PaperWeakOracle.Call (Tape := Tape) (arity := arity) rlc)
    (recomposeBound : Nat)
    (bounded : ∀ assignments, (program.recompose assignments).work ≤ recomposeBound)
    (baseSummable : ∀ vector, Summable fun tape =>
      (tapes tape).toReal * (PaperWeakOracle.baseWork rlc program call vector tape : ℝ))
    (checker : Response Assignment Scalar params arity → CheckResult)
    (check_spec : ∀ found, (checker found).accepted = true ↔
      found.Success semantics params rlc batch)
    (vector : Fin arity.total → Challenge rlc) :
    (line (withChecker
      (PaperWeakOracle.law rlc tapes program call recomposeBound bounded baseSummable)
      (PiRLC.CoordinateExtraction.typedChecker rlc checker)).oracle
      (oracleCheck rlc (fun found => (checker found).accepted))).acceptance vector =
      ∑' tape, (tapes tape).toReal *
        (if FinalOutputSuccess rlc batch dec publicSplit evaluationArity vector (call vector tape).value
          then (1 : ℝ) else 0) := by
  classical
  let law := PaperWeakOracle.law rlc tapes program call recomposeBound bounded baseSummable
  let verify := oracleCheck rlc (fun found => (checker found).accepted)
  change (∑ result, responseAcceptedMass
    (withChecker law (PiRLC.CoordinateExtraction.typedChecker rlc checker)).oracle verify vector result) = _
  calc
    _ = ∑ result, law.oracle.mass vector result *
        (if accepted verify vector result then (1 : ℝ) else 0) := by
      apply Finset.sum_congr rfl
      intro result _
      simp only [responseAcceptedMass, withChecker_response]
      cases accepted verify vector result <;> simp
    _ = ∑' tape, (tapes tape).toReal *
        (if accepted verify vector (PaperWeakOracle.run rlc program call vector tape).value
          then (1 : ℝ) else 0) :=
      PaperWeakOracle.law_response_mean rlc tapes program call recomposeBound bounded baseSummable
        vector (fun result => if accepted verify vector result then (1 : ℝ) else 0)
    _ = _ := by
      apply tsum_congr
      intro tape
      rw [accepted_run rlc batch dec publicSplit evaluationArity program correct kPositive call
        checker check_spec vector tape]
      by_cases success : FinalOutputSuccess rlc batch dec publicSplit evaluationArity vector
          (call vector tape).value
      · have returned := (run_return_iff rlc batch dec publicSplit evaluationArity
          program correct call vector tape).mpr success
        simp only [returned, success, ↓reduceIte]
      · have rejected : ¬ (PaperWeakOracle.run rlc program call vector tape).value.isSome = true :=
          fun returned => success ((run_return_iff rlc batch dec publicSplit evaluationArity
            program correct call vector tape).mp returned)
        simp only [rejected, success, Bool.false_eq_true, ↓reduceIte]

omit [Nonempty (Challenge rlc)] in
/-- The coordinate extractor starts from the same unconditioned success
probability as the original PiRLC/PiDEC suffix. -/
theorem rate_eq_successProbability
    (tapes : PMF Tape) (program : PaperWeakSuffix.Program (arity := arity) rlc)
    (correct : PaperWeakSuffix.Correct rlc batch dec publicSplit evaluationArity program)
    (kPositive : 0 < params.k)
    (call : PaperWeakOracle.Call (Tape := Tape) (arity := arity) rlc)
    (recomposeBound : Nat)
    (bounded : ∀ assignments, (program.recompose assignments).work ≤ recomposeBound)
    (baseSummable : ∀ vector, Summable fun tape =>
      (tapes tape).toReal * (PaperWeakOracle.baseWork rlc program call vector tape : ℝ))
    (checker : Response Assignment Scalar params arity → CheckResult)
    (check_spec : ∀ found, (checker found).accepted = true ↔
      found.Success semantics params rlc batch) :
    (line (withChecker
      (PaperWeakOracle.law rlc tapes program call recomposeBound bounded baseSummable)
      (PiRLC.CoordinateExtraction.typedChecker rlc checker)).oracle
      (oracleCheck rlc (fun found => (checker found).accepted))).rate =
      successProbability rlc batch dec publicSplit evaluationArity tapes call := by
  classical
  simp only [PiRLC.CoordinateRetry.Line.rate, PiRLC.CoordinateRetry.Line.weight,
    successProbability, Fintype.expect_eq_sum_div_card, Finset.sum_div]
  apply Finset.sum_congr rfl
  intro vector _
  exact congrArg (fun value : ℝ => value / Fintype.card (Fin arity.total → Challenge rlc))
    (acceptance_eq rlc batch dec publicSplit evaluationArity tapes program correct kPositive call
      recomposeBound bounded baseSummable checker check_spec vector)

variable [DecidableEq Scalar]

/-- The real stopped extractor loses at most one challenge collision per
coordinate from the original suffix success probability. The returned-list
law is the same law consumed by the exact source-opening theorem. -/
theorem weak_success_bound
    (laws : ExtractionAlgebra semantics params rlc)
    (strongSet : StrongSetUnits laws.ring rlc.challengeValid)
    (tapes : PMF Tape) (program : PaperWeakSuffix.Program (arity := arity) rlc)
    (correct : PaperWeakSuffix.Correct rlc batch dec publicSplit evaluationArity program)
    (kPositive : 0 < params.k)
    (call : PaperWeakOracle.Call (Tape := Tape) (arity := arity) rlc)
    (recomposeBound : Nat)
    (bounded : ∀ assignments, (program.recompose assignments).work ≤ recomposeBound)
    (baseSummable : ∀ vector, Summable fun tape =>
      (tapes tape).toReal * (PaperWeakOracle.baseWork rlc program call vector tape : ℝ))
    (checker : Response Assignment Scalar params arity → CheckResult)
    (check_spec : ∀ found, (checker found).accepted = true ↔
      found.Success semantics params rlc batch)
    (extraction : PiRLC.PaperForkExtractionWork.Primitives Scalar Assignment)
    (extractionCorrect : PiRLC.PaperForkExtractionWork.Correct laws.ring laws.assignmentModule extraction)
    (bounds : PiRLC.PaperForkExtractionWork.PrimitiveBounds)
    (extractionBounded : PiRLC.PaperForkExtractionWork.Bounded laws.ring extraction bounds) :
    let oracleLaw := PaperWeakOracle.law rlc tapes program call recomposeBound bounded baseSummable
    let charged := withChecker oracleLaw (PiRLC.CoordinateExtraction.typedChecker rlc checker)
    let verify := oracleCheck rlc (fun found => (checker found).accepted)
    successProbability rlc batch dec publicSplit evaluationArity tapes call -
        (arity.total : ℝ) / Fintype.card (Challenge rlc) ≤
      PaperWeakLaw.successProbability charged.oracle verify extraction := by
  dsimp only
  rw [PaperWeakLaw.successProbability_eq_returningProbability]
  have lower := (PiRLC.CoordinateExtraction.bounds_and_openings rlc batch laws strongSet
    (PaperWeakOracle.law rlc tapes program call recomposeBound bounded baseSummable)
    checker check_spec extraction extractionCorrect bounds extractionBounded).1
  rw [rate_eq_successProbability rlc batch dec publicSplit evaluationArity tapes program correct
    kPositive call recomposeBound bounded baseSummable checker check_spec] at lower
  exact lower

end NightstreamFPrime.Spec.Folding.Nifs.PaperWeakExtraction
