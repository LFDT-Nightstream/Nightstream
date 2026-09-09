import NightstreamFPrime.Spec.Folding.Nifs.PaperWeakAlgorithm

/-!
An abort-only extension for unreachable continuation inputs. Its call returns
none without reading its private tape. The callbacks required by Algorithm
are never invoked: the run theorem and response law prove that fact. Their
zero labels do not assert a cost for evaluating the semantic predicates.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Folding.Nifs.PaperWeakAbort

attribute [local instance] Classical.propDecidable
open scoped BigOperators
open NightstreamFPrime.Spec
open _root_.NightstreamFPrime.Spec.Folding
open PiRLC.PaperForkExtraction PiRLC.CoordinateForkLaw PiRLC.CoordinateCheckedCalls

variable {Tape Structure Assignment PublicInput Point Evaluation Commitment Scalar : Type*}
  {semantics : RelationSemantics Structure Assignment PublicInput Point Evaluation Commitment}
  {params : GlobalParams} {arity : BatchArity params}
  (rlc : PiRLC.Algebra Structure Assignment PublicInput Point Evaluation Commitment Scalar semantics params)
  (batch : InputBatch Structure PublicInput Point Evaluation Commitment params arity)
  (dec : PiDEC.Algebra Structure Assignment PublicInput Point Evaluation Commitment semantics params)
  (publicSplit : PiDEC.PaperVerifier.PublicInputSplit dec)
  (evaluationArity : PiDEC.PaperVerifier.EvaluationArity semantics)

private noncomputable def unusedProgram : PaperWeakSuffix.Program (arity := arity) rlc where
  check vector reply := ⟨decide (PaperWeakExtraction.FinalOutputSuccess rlc batch dec
    publicSplit evaluationArity vector (some reply)), 0⟩
  recompose assignments := ⟨dec.recomposeAssignment assignments, 0⟩

/-- One unused tape value is sufficient to describe the deterministic abort
as a PMF on the same tape type as the supported continuations. -/
noncomputable def algorithm (abortTape : Tape) :
    PaperWeakAlgorithm.Algorithm Tape rlc batch dec publicSplit evaluationArity where
  tapes := PMF.pure abortTape
  rawCall _ _ := ⟨none, 0⟩
  suffixProgram := unusedProgram rlc batch dec publicSplit evaluationArity
  suffixCorrect := {
    check := by
      intro vector reply
      exact decide_eq_true_iff
    recompose := fun _ => rfl
  }
  recomposeBound := 0
  recomposeBounded := fun _ => Nat.le_refl 0
  baseSummable := by
    intro vector
    simp only [PaperWeakOracle.baseWork, PaperWeakSuffix.checkerWork, Nat.zero_add,
      Nat.cast_zero, mul_zero]
    exact summable_zero
  parentChecker response := ⟨decide (response.Success semantics params rlc batch), 0⟩
  parentChecker_spec := fun _ => decide_eq_true_iff

/-- No raw suffix invocation or final witness is produced off support. -/
theorem rawCall_eq (abortTape : Tape) (vector : Fin arity.total → Challenge rlc) (tape : Tape) :
    ((algorithm rlc batch dec publicSplit evaluationArity abortTape).rawCall vector tape).value = none ∧
      ((algorithm rlc batch dec publicSplit evaluationArity abortTape).rawCall vector tape).work = 0 :=
  ⟨rfl, rfl⟩

/-- The only executed steps are the abort dispatch and outer return. Neither
semantic callback is evaluated. The query driver adds its separate next step. -/
theorem run_eq (abortTape : Tape) (vector : Fin arity.total → Challenge rlc) (tape : Tape) :
    PaperWeakOracle.run rlc
      (algorithm rlc batch dec publicSplit evaluationArity abortTape).suffixProgram
      (algorithm rlc batch dec publicSplit evaluationArity abortTape).rawCall vector tape =
      ⟨none, 2⟩ := rfl

private theorem weights_sum (tapes : PMF Tape) : (∑' tape, (tapes tape).toReal) = 1 := by
  rw [← ENNReal.tsum_toReal_eq tapes.apply_ne_top, tapes.tsum_coe, ENNReal.toReal_one]

variable [Fintype Assignment]

/-- The finite query mean comes from the constant observed abort clock. -/
theorem oracle_meanWork (abortTape : Tape) (vector : Fin arity.total → Challenge rlc) :
    (algorithm rlc batch dec publicSplit evaluationArity abortTape).oracleLaw.meanWork vector = 3 := by
  rw [PaperWeakAlgorithm.Algorithm.oracle_meanWork]
  change (∑' tape, ((PMF.pure abortTape) tape).toReal * ((2 : ℝ) + 1)) = 3
  rw [tsum_mul_right, weights_sum, one_mul]
  norm_num

private theorem response_mean (abortTape : Tape) (vector : Fin arity.total → Challenge rlc)
    (value : Option Assignment → ℝ) :
    (∑ result, (algorithm rlc batch dec publicSplit evaluationArity abortTape).chargedOracle.mass
      vector result * value result) = value none := by
  simp only [PaperWeakAlgorithm.Algorithm.chargedOracle, withChecker_response]
  rw [PaperWeakAlgorithm.Algorithm.oracleLaw, PaperWeakOracle.law_response_mean]
  change (∑' tape, ((PMF.pure abortTape) tape).toReal * value none) = value none
  rw [tsum_mul_right, weights_sum, one_mul]

/-- The derived charged response law is exactly the pure-none law. No kernel
or normalization assertion is supplied as a premise. -/
theorem charged_mass (abortTape : Tape) (vector : Fin arity.total → Challenge rlc)
    (result : Option Assignment) :
    (algorithm rlc batch dec publicSplit evaluationArity abortTape).chargedOracle.mass vector result =
      if result = none then 1 else 0 := by
  have mean := response_mean rlc batch dec publicSplit evaluationArity abortTape vector
    (fun observed => if observed = result then (1 : ℝ) else 0)
  simpa [eq_comm] using mean

/-- Parent checking contributes no work because the abort has no assignment
to check. The full checked query therefore still costs exactly three steps. -/
theorem charged_meanWork (abortTape : Tape) (vector : Fin arity.total → Challenge rlc) :
    (withChecker (algorithm rlc batch dec publicSplit evaluationArity abortTape).oracleLaw
      (PiRLC.CoordinateExtraction.typedChecker rlc
        (algorithm rlc batch dec publicSplit evaluationArity abortTape).parentChecker)).meanWork vector = 3 := by
  rw [withChecker_meanWork, oracle_meanWork]
  have noCheck : (∑ result,
      (algorithm rlc batch dec publicSplit evaluationArity abortTape).oracleLaw.oracle.mass vector result *
      (checkerWork (PiRLC.CoordinateExtraction.typedChecker rlc
        (algorithm rlc batch dec publicSplit evaluationArity abortTape).parentChecker) vector result : ℝ)) = 0 := by
    apply Finset.sum_eq_zero
    intro result _
    cases result <;> simp only [checkerWork, PiRLC.CoordinateExtraction.typedChecker,
      algorithm, Nat.cast_zero, mul_zero]
  rw [noCheck, add_zero]

end NightstreamFPrime.Spec.Folding.Nifs.PaperWeakAbort
