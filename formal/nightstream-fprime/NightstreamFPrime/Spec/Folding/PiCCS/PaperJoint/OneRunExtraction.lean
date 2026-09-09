import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.CheckedWitnessExtraction
import Mathlib.Algebra.Polynomial.Eval.Defs

/-!
The B.2 extractor invokes the interactive adversary/verifier once, checks its
actual output, and projects that witness. Call, check, and access primitives
return their values and clocks together. The call clock includes the selected
interactive implementation's representation and coin-reading work.

Runtime is averaged on the same private tape and independent uniform verifier
coins as the proved success event. Expected polynomial time is conditional on
polynomial bounds for the computed call/check mean and the correct accessor.
No uniform law or Fiat–Shamir transfer is asserted for Poseidon2 sampling.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.OneRunExtraction

open NightstreamFPrime.Spec
open StrongReduction ConcreteCarrier WitnessProjection
open CheckedWitnessExtraction
open _root_.NightstreamFPrime.Spec.Folding.PiRLC.PaperForkExtractionWork (Result)

abbrev Call (Tape : Type*) (shape : Shape) (carrier : Phi81Relation.Shape) :=
  Tape → CubePoint K shape.cubeVariables → K → CubePoint K shape.cubeVariables →
    Result (Outcome shape carrier)

/-- Erasing only the clock preserves the same private tape, verifier coins,
causal verifier transcript, and returned output witness. -/
def CallCorrect {Tape : Type*} {shape : Shape} {carrier : Phi81Relation.Shape}
    {width : Nat} (call : Call Tape shape carrier)
    (prover : Tape → CausalExecution.Prover shape carrier.carrierWidth width) : Prop :=
  ∀ tape alpha gamma point,
    (call tape alpha gamma point).value = CausalExecution.run (prover tape) alpha gamma point

/-- The single call is executed before the same returned output is checked.
The outer result return costs one step on every branch. -/
def run {Tape : Type*} {shape : Shape} {carrier : Phi81Relation.Shape}
    (call : Call Tape shape carrier) (program : Program shape carrier)
    (tape : Tape) (alpha : CubePoint K shape.cubeVariables) (gamma : K)
    (point : CubePoint K shape.cubeVariables) : Result (Option (SourceWitness shape carrier)) :=
  let issued := call tape alpha gamma point
  let finished := finish program issued.value
  ⟨finished.value, issued.work + finished.work + 1⟩

theorem run_value {Tape : Type*} {shape : Shape} {carrier : Phi81Relation.Shape}
    {width : Nat} (call : Call Tape shape carrier) (program : Program shape carrier)
    (prover : Tape → CausalExecution.Prover shape carrier.carrierWidth width)
    (callCorrect : CallCorrect call prover) (tape : Tape)
    (alpha : CubePoint K shape.cubeVariables) (gamma : K) (point : CubePoint K shape.cubeVariables) :
    (run call program tape alpha gamma point).value =
      (finish program (CausalExecution.run (prover tape) alpha gamma point)).value := by
  change (finish program (call tape alpha gamma point).value).value = _
  rw [callCorrect tape alpha gamma point]

/-- Actual call and check work before projection. Abort has no checker work. -/
def baseClock {Tape : Type*} {shape : Shape} {carrier : Phi81Relation.Shape}
    (call : Call Tape shape carrier) (program : Program shape carrier)
    (tape : Tape) (alpha : CubePoint K shape.cubeVariables) (gamma : K)
    (point : CubePoint K shape.cubeVariables) : ℝ :=
  ((call tape alpha gamma point).work +
    checkerWork program (call tape alpha gamma point).value : Nat)

/-- Total clock of the executed one-call program, including every exit. -/
def totalClock {Tape : Type*} {shape : Shape} {carrier : Phi81Relation.Shape}
    (call : Call Tape shape carrier) (program : Program shape carrier)
    (tape : Tape) (alpha : CubePoint K shape.cubeVariables) (gamma : K)
    (point : CubePoint K shape.cubeVariables) : ℝ :=
  (run call program tape alpha gamma point).work

theorem run_work_le {Tape : Type*} {shape : Shape} {carrier : Phi81Relation.Shape}
    (call : Call Tape shape carrier) (program : Program shape carrier)
    (accessBound : Nat) (bounded : CostedWitnessProjection.Bounded program.access accessBound)
    (tape : Tape) (alpha : CubePoint K shape.cubeVariables) (gamma : K)
    (point : CubePoint K shape.cubeVariables) :
    totalClock call program tape alpha gamma point ≤ baseClock call program tape alpha gamma point +
      ((CostedWitnessProjection.workBound shape carrier accessBound + 3 : Nat) : ℝ) := by
  have finishBound := finish_work_le program accessBound bounded (call tape alpha gamma point).value
  have naturalBound : (run call program tape alpha gamma point).work ≤
      (call tape alpha gamma point).work + checkerWork program (call tape alpha gamma point).value +
        (CostedWitnessProjection.workBound shape carrier accessBound + 3) := by
    dsimp only [run]
    omega
  change ((run call program tape alpha gamma point).work : ℝ) ≤
    (((call tape alpha gamma point).work +
      checkerWork program (call tape alpha gamma point).value : Nat) : ℝ) +
    ((CostedWitnessProjection.workBound shape carrier accessBound + 3 : Nat) : ℝ)
  exact_mod_cast naturalBound

variable {Tape Commitment : Type*} {shape : Shape} {carrier : Phi81Relation.Shape}
  {blockCount width : Nat}
  (tapes : PMF Tape) (call : Call Tape shape carrier) (program : Program shape carrier)
  (prover : Tape → CausalExecution.Prover shape carrier.carrierWidth width)
  (callCorrect : CallCorrect call prover)
  (commit : Phi81Relation.Assignment carrier → Commitment) (params : GlobalParams)
  (statement : Statement K Commitment (Phi81Relation.PublicInput carrier)
    shape carrier.carrierWidth blockCount baseOps)
  (correct : CheckedWitnessExtraction.Correct (width := width) program commit params statement)

include callCorrect correct in
/-- The successful source event belongs to the exact run whose clock is charged. -/
theorem run_source_iff (tape : Tape) (alpha : CubePoint K shape.cubeVariables)
    (gamma : K) (point : CubePoint K shape.cubeVariables) :
    SourceReturned commit params statement (run call program tape alpha gamma point).value ↔
      StrongProbability.RelaxedSuccess (width := width) (openingMaps commit) params statement
        (CausalExecution.run (prover tape) alpha gamma point) ∧
      StrongProbability.SourceValid (openingMaps commit) params statement
        (CausalExecution.run (prover tape) alpha gamma point) := by
  rw [run_value call program prover callCorrect]
  exact finish_source_iff program commit params statement correct _

attribute [local instance] Classical.propDecidable

/-- Probability that this executed extractor returns a valid source witness. -/
noncomputable def successProbability : ℝ :=
  StrongProbability.clockMean tapes fun tape alpha gamma point =>
    if SourceReturned commit params statement (run call program tape alpha gamma point).value then 1 else 0

include callCorrect correct in
theorem successProbability_eq :
    successProbability tapes call program commit params statement =
      StrongProbability.sourceProbability tapes prover (openingMaps commit) params statement := by
  let value : Outcome shape carrier → ℝ := fun outcome =>
    if StrongProbability.RelaxedSuccess (width := width) (openingMaps commit) params statement outcome ∧
      StrongProbability.SourceValid (openingMaps commit) params statement outcome then 1 else 0
  have integrands :
      (fun tape alpha gamma point =>
        if SourceReturned commit params statement (run call program tape alpha gamma point).value
        then (1 : ℝ) else 0) =
      (fun tape alpha gamma point => value (CausalExecution.run (prover tape) alpha gamma point)) := by
    funext tape alpha gamma point
    have equivalent := run_source_iff call program prover callCorrect commit params statement correct
      tape alpha gamma point
    by_cases success :
        StrongProbability.RelaxedSuccess (width := width) (openingMaps commit) params statement
          (CausalExecution.run (prover tape) alpha gamma point) ∧
        StrongProbability.SourceValid (openingMaps commit) params statement
          (CausalExecution.run (prover tape) alpha gamma point)
    · have returned := equivalent.mpr success
      simp only [value, if_pos success, if_pos returned]
    · have rejected : ¬ SourceReturned commit params statement
          (run call program tape alpha gamma point).value :=
        fun returned => success (equivalent.mp returned)
      simp only [value, if_neg success, if_neg rejected]
  change StrongProbability.clockMean tapes _ = _
  calc
    _ = StrongProbability.clockMean tapes
        (fun tape alpha gamma point => value (CausalExecution.run (prover tape) alpha gamma point)) :=
      congrArg (StrongProbability.clockMean tapes) integrands
    _ = StrongProbability.executionMean tapes prover value :=
      StrongProbability.clockMean_run_eq_executionMean tapes prover value
    _ = StrongProbability.sourceProbability tapes prover (openingMaps commit) params statement := by
      apply StrongProbability.executionMean_eq_sourceProbability tapes prover
        (openingMaps commit) params statement value
      · intro outcome success
        simp only [value, if_pos success]
      · intro outcome failure
        simp only [value, if_neg failure]

/-- Summability is derived for the total clock, so a divergent real sum cannot
masquerade as a finite expected runtime. Oracle work may be unbounded per call. -/
theorem expected_work_bound (accessBound : Nat)
    (bounded : CostedWitnessProjection.Bounded program.access accessBound)
    (baseSummable : Summable fun tape =>
      (tapes tape).toReal * StrongProbability.verifierMean (baseClock call program tape)) :
    Summable (fun tape => (tapes tape).toReal * StrongProbability.verifierMean (totalClock call program tape)) ∧
      StrongProbability.clockMean tapes (totalClock call program) ≤
        StrongProbability.clockMean tapes (baseClock call program) +
          ((CostedWitnessProjection.workBound shape carrier accessBound + 3 : Nat) : ℝ) := by
  apply StrongProbability.clockMean_le_add_const tapes
    (baseClock call program) (totalClock call program) _
  · intro tape alpha gamma point
    exact Nat.cast_nonneg _
  · exact baseSummable
  · exact run_work_le call program accessBound bounded

/-- Explicit paper EPT/PPT premises bound the actual call/check mean and the
access implementation. The displayed polynomial charges all projection work. -/
theorem expected_work_polynomial_bound (accessBound : Nat)
    (bounded : CostedWitnessProjection.Bounded program.access accessBound)
    (baseSummable : Summable fun tape =>
      (tapes tape).toReal * StrongProbability.verifierMean (baseClock call program tape))
    (securityParameter : Nat) (callPolynomial accessPolynomial : Polynomial ℝ)
    (callPPT : StrongProbability.clockMean tapes (baseClock call program) ≤
      callPolynomial.eval (securityParameter : ℝ))
    (accessPPT : (accessBound : ℝ) ≤ accessPolynomial.eval (securityParameter : ℝ)) :
    Summable (fun tape => (tapes tape).toReal * StrongProbability.verifierMean (totalClock call program tape)) ∧
      StrongProbability.clockMean tapes (totalClock call program) ≤
        (callPolynomial +
          Polynomial.C (shape.freshCount : ℝ) *
            (Polynomial.C (privateWidth carrier : ℝ) * (accessPolynomial + Polynomial.C 1) + Polynomial.C 2) +
          Polynomial.C (shape.runningCount : ℝ) *
            (Polynomial.C (carrier.carrierWidth : ℝ) * (accessPolynomial + Polynomial.C 1) + Polynomial.C 2) +
          Polynomial.C 6).eval (securityParameter : ℝ) := by
  have actual := expected_work_bound tapes call program accessBound bounded baseSummable
  refine ⟨actual.1, ?_⟩
  have fresh := mul_le_mul_of_nonneg_left accessPPT
    (by positivity : (0 : ℝ) ≤ (shape.freshCount : ℝ) * (privateWidth carrier : ℝ))
  have running := mul_le_mul_of_nonneg_left accessPPT
    (by positivity : (0 : ℝ) ≤ (shape.runningCount : ℝ) * (carrier.carrierWidth : ℝ))
  have bound := actual.2
  unfold CostedWitnessProjection.workBound at bound
  push_cast at bound
  simp only [Polynomial.eval_add, Polynomial.eval_mul, Polynomial.eval_C]
  nlinarith

end NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.OneRunExtraction
