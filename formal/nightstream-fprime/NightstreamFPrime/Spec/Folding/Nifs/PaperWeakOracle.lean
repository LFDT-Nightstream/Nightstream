import NightstreamFPrime.Spec.Folding.Nifs.PaperWeakSuffix
import NightstreamFPrime.Spec.Folding.PiRLC.CoordinateChargedOracle
import Mathlib.Probability.ProbabilityMassFunction.Basic
import Mathlib.Topology.Algebra.InfiniteSum.Real
import Mathlib.Topology.Algebra.InfiniteSum.Constructions
import Mathlib.Analysis.Normed.Group.InfiniteSum
import Mathlib.Tactic.Ring

/-!
The clocked PiRLC response law is the pushforward of actual resumed suffix
calls. Each query uses fresh private coins from the same PMF and the captured
post-PiCCS continuation. Rejected final replies become counted aborts. The
joint response/clock law retains arbitrary correlation and has no per-call
or per-context runtime cap.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Folding.Nifs.PaperWeakOracle

open NightstreamFPrime.Spec
open _root_.NightstreamFPrime.Spec.Folding
open PiRLC.PaperForkExtraction PiRLC.CoordinateForkLaw
open PaperWeakSuffix
open _root_.NightstreamFPrime.Spec.Folding.PiRLC.PaperForkExtractionWork (Result)

private theorem weights_hasSum {Tape : Type*} (tapes : PMF Tape) :
    HasSum (fun tape => (tapes tape).toReal) 1 := by
  have summable := ENNReal.summable_toReal tapes.tsum_coe_ne_top
  have total : (∑' tape, (tapes tape).toReal) = 1 := by
    rw [← ENNReal.tsum_toReal_eq tapes.apply_ne_top, tapes.tsum_coe, ENNReal.toReal_one]
  exact total ▸ summable.hasSum

private noncomputable def pushMass {Tape Output : Type*} (tapes : PMF Tape)
    (route : Tape → Output) (output : Output) : ℝ :=
  ∑' tape : {tape // route tape = output}, (tapes tape.val).toReal

private theorem pushMass_hasSum {Tape Output : Type*} (tapes : PMF Tape)
    (route : Tape → Output) : HasSum (pushMass tapes route) 1 :=
  (weights_hasSum tapes).tsum_fiberwise route

private theorem pushMass_value_hasSum {Tape Output : Type*} (tapes : PMF Tape)
    (route : Tape → Output) (value : Output → ℝ)
    (summable : Summable fun tape => (tapes tape).toReal * value (route tape)) :
    HasSum (fun output => pushMass tapes route output * value output)
      (∑' tape, (tapes tape).toReal * value (route tape)) := by
  apply (summable.hasSum.tsum_fiberwise route).congr_fun
  intro output
  symm
  change (∑' tape : {tape // route tape = output},
    (tapes tape.val).toReal * value (route tape.val)) = _
  calc
    _ = ∑' tape : {tape // route tape = output}, (tapes tape.val).toReal * value output := by
      apply tsum_congr
      intro tape
      rw [tape.property]
    _ = _ := tsum_mul_right

variable {Tape Structure Assignment PublicInput Point Evaluation Commitment Scalar : Type*}
  {semantics : RelationSemantics Structure Assignment PublicInput Point Evaluation Commitment}
  {params : GlobalParams} {arity : BatchArity params}
  (rlc : PiRLC.Algebra Structure Assignment PublicInput Point Evaluation Commitment Scalar semantics params)

abbrev Call := (Fin arity.total → Challenge rlc) → Tape →
  Result (Option (Reply Assignment Evaluation Commitment params))

/-- One actual adversary/verifier continuation followed by the checked final
output consumer. The outer return is charged on every branch. -/
def run (program : Program (arity := arity) rlc) (call : Call (Tape := Tape) (arity := arity) rlc)
    (vector : Fin arity.total → Challenge rlc) (tape : Tape) : Result (Option Assignment) :=
  let issued := call vector tape
  let finished := finish rlc program vector issued.value
  ⟨finished.value, issued.work + finished.work + 1⟩

/-- Work from the same original suffix call and final-output check. -/
def baseWork (program : Program (arity := arity) rlc) (call : Call (Tape := Tape) (arity := arity) rlc)
    (vector : Fin arity.total → Challenge rlc) (tape : Tape) : Nat :=
  (call vector tape).work + checkerWork rlc program vector (call vector tape).value

theorem run_work_le (program : Program (arity := arity) rlc) (call : Call (Tape := Tape) (arity := arity) rlc)
    (recomposeBound : Nat)
    (bounded : ∀ assignments, (program.recompose assignments).work ≤ recomposeBound)
    (vector : Fin arity.total → Challenge rlc) (tape : Tape) :
    (run rlc program call vector tape).work ≤ baseWork rlc program call vector tape + recomposeBound + 3 := by
  have work := finish_work_le rlc program recomposeBound bounded vector (call vector tape).value
  dsimp only [run, baseWork]
  omega

theorem run_returns_parent
    (batch : InputBatch Structure PublicInput Point Evaluation Commitment params arity)
    (dec : PiDEC.Algebra Structure Assignment PublicInput Point Evaluation Commitment semantics params)
    (publicSplit : PiDEC.PaperVerifier.PublicInputSplit dec)
    (evaluationArity : PiDEC.PaperVerifier.EvaluationArity semantics)
    (program : Program (arity := arity) rlc)
    (correct : Correct rlc batch dec publicSplit evaluationArity program)
    (kPositive : 0 < params.k) (call : Call (Tape := Tape) (arity := arity) rlc)
    (vector : Fin arity.total → Challenge rlc) (tape : Tape) (assignment : Assignment)
    (returned : (run rlc program call vector tape).value = some assignment) :
    (response rlc vector assignment).Success semantics params rlc batch :=
  finish_returns_parent rlc batch dec publicSplit evaluationArity program correct kPositive
    vector (call vector tape).value assignment returned

private def route (program : Program (arity := arity) rlc) (call : Call (Tape := Tape) (arity := arity) rlc)
    (vector : Fin arity.total → Challenge rlc) (tape : Tape) : Option Assignment × Nat :=
  ((run rlc program call vector tape).value, (run rlc program call vector tape).work)

/-- Every atom contains exactly the private tapes that produced this observed
parent or abort and this observed work. No output is sampled uniformly. -/
noncomputable def mass (tapes : PMF Tape) (program : Program (arity := arity) rlc)
    (call : Call (Tape := Tape) (arity := arity) rlc) (vector : Fin arity.total → Challenge rlc)
    (result : Option Assignment) (steps : Nat) : ℝ :=
  pushMass tapes (route rlc program call vector) (result, steps)

private theorem query_work_le (program : Program (arity := arity) rlc) (call : Call (Tape := Tape) (arity := arity) rlc)
    (recomposeBound : Nat)
    (bounded : ∀ assignments, (program.recompose assignments).work ≤ recomposeBound)
    (vector : Fin arity.total → Challenge rlc) (tape : Tape) :
    ((run rlc program call vector tape).work : ℝ) + 1 ≤
      (baseWork rlc program call vector tape : ℝ) + ((recomposeBound + 4 : Nat) : ℝ) := by
  have work := run_work_le rlc program call recomposeBound bounded vector tape
  have natural : (run rlc program call vector tape).work + 1 ≤
      baseWork rlc program call vector tape + (recomposeBound + 4) := by omega
  exact_mod_cast natural

private theorem query_work_summable (tapes : PMF Tape) (program : Program (arity := arity) rlc)
    (call : Call (Tape := Tape) (arity := arity) rlc) (recomposeBound : Nat)
    (bounded : ∀ assignments, (program.recompose assignments).work ≤ recomposeBound)
    (vector : Fin arity.total → Challenge rlc)
    (baseSummable : Summable fun tape =>
      (tapes tape).toReal * (baseWork rlc program call vector tape : ℝ)) :
    Summable fun tape => (tapes tape).toReal * (((run rlc program call vector tape).work : ℝ) + 1) := by
  have envelope := baseSummable.add ((weights_hasSum tapes).summable.mul_right
    ((recomposeBound + 4 : Nat) : ℝ))
  apply Summable.of_nonneg_of_le
    (fun tape => mul_nonneg ENNReal.toReal_nonneg (by positivity)) _ envelope
  intro tape
  simpa only [mul_add] using mul_le_mul_of_nonneg_left
    (query_work_le rlc program call recomposeBound bounded vector tape) ENNReal.toReal_nonneg

variable [Fintype Assignment]

/-- The existing PiRLC law is constructed from actual suffix calls. The sole
moment premise is on the original call/check work; recomposition is bounded
by its executed primitive. Neither total extractor work nor a fork is assumed. -/
noncomputable def law (tapes : PMF Tape) (program : Program (arity := arity) rlc)
    (call : Call (Tape := Tape) (arity := arity) rlc) (recomposeBound : Nat)
    (bounded : ∀ assignments, (program.recompose assignments).work ≤ recomposeBound)
    (baseSummable : ∀ vector, Summable fun tape =>
      (tapes tape).toReal * (baseWork rlc program call vector tape : ℝ)) :
    PiRLC.CoordinateChargedOracle.Law (Fin arity.total) (Challenge rlc) Assignment where
  mass := mass rlc tapes program call
  nonnegative _ _ _ := tsum_nonneg fun _ => ENNReal.toReal_nonneg
  summable vector result := (pushMass_hasSum tapes (route rlc program call vector)).summable.prod_factor result
  normalized vector := by
    have normalized := pushMass_hasSum tapes (route rlc program call vector)
    calc
      _ = ∑' result, ∑' steps, mass rlc tapes program call vector result steps := (tsum_fintype _).symm
      _ = ∑' output, pushMass tapes (route rlc program call vector) output :=
        normalized.summable.tsum_prod.symm
      _ = 1 := normalized.tsum_eq
  workSummable vector result :=
    (pushMass_value_hasSum tapes (route rlc program call vector)
      (fun output => (output.2 : ℝ) + 1)
      (query_work_summable rlc tapes program call recomposeBound bounded vector (baseSummable vector))
      ).summable.prod_factor result

/-- Exact mean of the real suffix/query program, including the PiRLC query
transition. The probability and work use the same private tapes. -/
theorem law_meanWork (tapes : PMF Tape) (program : Program (arity := arity) rlc)
    (call : Call (Tape := Tape) (arity := arity) rlc) (recomposeBound : Nat)
    (bounded : ∀ assignments, (program.recompose assignments).work ≤ recomposeBound)
    (baseSummable : ∀ vector, Summable fun tape =>
      (tapes tape).toReal * (baseWork rlc program call vector tape : ℝ))
    (vector : Fin arity.total → Challenge rlc) :
    (law rlc tapes program call recomposeBound bounded baseSummable).meanWork vector =
      ∑' tape, (tapes tape).toReal * (((run rlc program call vector tape).work : ℝ) + 1) := by
  have summed := pushMass_value_hasSum tapes (route rlc program call vector)
    (fun output => (output.2 : ℝ) + 1)
    (query_work_summable rlc tapes program call recomposeBound bounded vector (baseSummable vector))
  change (∑ result, ∑' steps : Nat,
    mass rlc tapes program call vector result steps * ((steps : ℝ) + 1)) = _
  calc
    _ = ∑' result : Option Assignment, ∑' steps : Nat,
        mass rlc tapes program call vector result steps * ((steps : ℝ) + 1) :=
      (tsum_fintype _).symm
    _ = ∑' output : Option Assignment × Nat,
        pushMass tapes (route rlc program call vector) output * ((output.2 : ℝ) + 1) :=
      summed.summable.tsum_prod.symm
    _ = _ := summed.tsum_eq

/-- Erasing the clock preserves every observable of the actual response.
The finite response carrier makes the observable integrable; the private
tape carrier and the clock remain unrestricted. -/
theorem law_response_mean (tapes : PMF Tape) (program : Program (arity := arity) rlc)
    (call : Call (Tape := Tape) (arity := arity) rlc) (recomposeBound : Nat)
    (bounded : ∀ assignments, (program.recompose assignments).work ≤ recomposeBound)
    (baseSummable : ∀ vector, Summable fun tape =>
      (tapes tape).toReal * (baseWork rlc program call vector tape : ℝ))
    (vector : Fin arity.total → Challenge rlc) (value : Option Assignment → ℝ) :
    (∑ result, (law rlc tapes program call recomposeBound bounded baseSummable).oracle.mass
      vector result * value result) =
      ∑' tape, (tapes tape).toReal * value (run rlc program call vector tape).value := by
  have moment : Summable fun tape =>
      (tapes tape).toReal * value (run rlc program call vector tape).value := by
    apply Summable.of_norm_bounded ((weights_hasSum tapes).summable.mul_right
      (∑ result, |value result|))
    intro tape
    rw [Real.norm_eq_abs, abs_mul, abs_of_nonneg ENNReal.toReal_nonneg]
    exact mul_le_mul_of_nonneg_left
      (Finset.single_le_sum (fun result _ => abs_nonneg (value result))
        (Finset.mem_univ (run rlc program call vector tape).value)) ENNReal.toReal_nonneg
  have summed := pushMass_value_hasSum tapes (route rlc program call vector)
    (fun output => value output.1) moment
  change (∑ result, (∑' steps : Nat, mass rlc tapes program call vector result steps) *
    value result) = _
  calc
    _ = ∑ result, ∑' steps : Nat, mass rlc tapes program call vector result steps * value result := by
      apply Finset.sum_congr rfl
      intro result _
      exact tsum_mul_right.symm
    _ = ∑' result : Option Assignment, ∑' steps : Nat,
        mass rlc tapes program call vector result steps * value result :=
      (tsum_fintype _).symm
    _ = ∑' output : Option Assignment × Nat,
        pushMass tapes (route rlc program call vector) output * value output.1 :=
      summed.summable.tsum_prod.symm
    _ = _ := summed.tsum_eq

/-- The mean is bounded by the original call/check clock plus the executed
recomposition bound and the four charged control transitions. This is a
pointwise mean comparison, not a uniform adversary runtime premise. -/
theorem law_meanWork_le (tapes : PMF Tape) (program : Program (arity := arity) rlc)
    (call : Call (Tape := Tape) (arity := arity) rlc) (recomposeBound : Nat)
    (bounded : ∀ assignments, (program.recompose assignments).work ≤ recomposeBound)
    (baseSummable : ∀ vector, Summable fun tape =>
      (tapes tape).toReal * (baseWork rlc program call vector tape : ℝ))
    (vector : Fin arity.total → Challenge rlc) :
    (law rlc tapes program call recomposeBound bounded baseSummable).meanWork vector ≤
      (∑' tape, (tapes tape).toReal * (baseWork rlc program call vector tape : ℝ)) +
        ((recomposeBound + 4 : Nat) : ℝ) := by
  rw [law_meanWork]
  have constant := (weights_hasSum tapes).summable.mul_right ((recomposeBound + 4 : Nat) : ℝ)
  have envelope := (baseSummable vector).add constant
  calc
    _ ≤ ∑' tape, ((tapes tape).toReal * (baseWork rlc program call vector tape : ℝ) +
        (tapes tape).toReal * ((recomposeBound + 4 : Nat) : ℝ)) := by
      apply Summable.tsum_le_tsum _
        (query_work_summable rlc tapes program call recomposeBound bounded vector (baseSummable vector))
        envelope
      intro tape
      simpa only [mul_add] using mul_le_mul_of_nonneg_left
        (query_work_le rlc program call recomposeBound bounded vector tape) ENNReal.toReal_nonneg
    _ = _ := by
      rw [Summable.tsum_add (baseSummable vector) constant, tsum_mul_right,
        (weights_hasSum tapes).tsum_eq, one_mul]

omit [Fintype Assignment] in
/-- Invalid parent responses have zero mass because every returned parent was
derived from this invocation's checked final child witness. -/
theorem mass_invalid_parent_zero
    (batch : InputBatch Structure PublicInput Point Evaluation Commitment params arity)
    (dec : PiDEC.Algebra Structure Assignment PublicInput Point Evaluation Commitment semantics params)
    (publicSplit : PiDEC.PaperVerifier.PublicInputSplit dec)
    (evaluationArity : PiDEC.PaperVerifier.EvaluationArity semantics)
    (program : Program (arity := arity) rlc)
    (correct : Correct rlc batch dec publicSplit evaluationArity program)
    (kPositive : 0 < params.k) (tapes : PMF Tape) (call : Call (Tape := Tape) (arity := arity) rlc)
    (vector : Fin arity.total → Challenge rlc) (assignment : Assignment) (steps : Nat)
    (invalid : ¬ (response rlc vector assignment).Success semantics params rlc batch) :
    mass rlc tapes program call vector (some assignment) steps = 0 := by
  calc
    _ = ∑' tape : {tape // route rlc program call vector tape = (some assignment, steps)}, (0 : ℝ) := by
      apply tsum_congr
      intro tape
      have returned : (run rlc program call vector tape.val).value = some assignment :=
        congrArg Prod.fst tape.property
      exact False.elim (invalid (run_returns_parent rlc batch dec publicSplit evaluationArity
        program correct kPositive call vector tape.val assignment returned))
    _ = 0 := tsum_zero

end NightstreamFPrime.Spec.Folding.Nifs.PaperWeakOracle
