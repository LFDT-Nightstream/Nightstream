import NightstreamFPrime.Spec.Folding.PiRLC.CoordinateForkEvent
import NightstreamFPrime.Spec.Folding.PiRLC.CoordinateTerminalLaw
import NightstreamFPrime.Spec.Folding.PiRLC.CoordinateTerminalSemantics
import Mathlib.Algebra.Polynomial.Eval.Defs

/-!
Interactive PiRLC extraction for one fixed oracle context. The same clocked
experiment supplies the probability, invocation count, expected total work,
and exact source-opening consumer. The actual CE checker and four extraction
primitives return their values and work together.

Expected polynomial time follows from polynomial bounds on the computed
uniform-call mean and the correct primitive implementations. Individual oracle
calls can have unbounded work. No uniform or Fiat–Shamir law is asserted for
the bounded Poseidon sampler; its exact shortfall remains a separate event.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Folding.PiRLC.CoordinateExtraction

open scoped BigOperators
open NightstreamFPrime.Spec
open PaperForkExtraction CoordinateForkLaw CoordinateForkEvent
open CoordinateOracle CoordinateOracleStar CoordinateOracleCost
open PaperForkExtractionWork CoordinateChargedOracle CoordinateCheckedCalls
open CoordinateRetryWork CoordinateTerminalLaw CoordinateTerminalProgram
open CoordinateTerminalSemantics

variable {Structure Assignment PublicInput Point Evaluation Commitment Scalar : Type*}
  {semantics : RelationSemantics
    Structure Assignment PublicInput Point Evaluation Commitment}
  {params : GlobalParams} {arity : BatchArity params}
  (algebra : Algebra Structure Assignment PublicInput Point Evaluation Commitment
    Scalar semantics params)
  (batch : InputBatch Structure PublicInput Point Evaluation Commitment params arity)

/-- The sampled fork feeds the existing inverse-difference extractor with all
base and fork response values preserved. -/
theorem event_implies_extracted_openings
    (laws : ExtractionAlgebra semantics params algebra)
    (strongSet : StrongSetUnits laws.ring algebra.challengeValid)
    (vector : Fin arity.total → Challenge algebra) (initial : Option Assignment)
    (outputs : Fin arity.total → Outcome
      (Challenge := Challenge algebra) (Assignment := Assignment))
    (event : Event algebra batch vector initial outputs) :
    ∃ fork : CompleteFork semantics params algebra batch,
      fork.base.challenges = scalarVector algebra vector ∧
      some fork.base.assignment = initial ∧
      (∀ coordinate,
        (fork.forks coordinate).challenges = Function.update (scalarVector algebra vector)
          coordinate (outputs coordinate).1.val ∧
        some (fork.forks coordinate).assignment = (outputs coordinate).2) ∧
      (∀ coordinate, PaperCorrections.CorrectedAmbientHolds semantics params
        (batch.inputs coordinate) (extractedAssignment laws strongSet fork coordinate)) := by
  rcases event with ⟨fork, baseVector, baseAssignment, responses⟩
  exact ⟨fork, baseVector, baseAssignment, responses,
    completeFork_implies_correctedAmbientHolds semantics params arity algebra laws strongSet batch fork⟩

variable [DecidableEq Scalar] [Fintype (Challenge algebra)]
  [Nonempty (Challenge algebra)] [Fintype Assignment]

/-- The CE checker is invoked on this exact typed response. Its returned Bool
and work remain part of the same call. -/
def typedChecker
    (checker : Response Assignment Scalar params arity → CheckResult) :
    Checker (Fin arity.total) (Challenge algebra) Assignment :=
  fun vector assignment => checker (CoordinateForkLaw.response algebra vector assignment)

/-- Total expected work of the charged query loop followed by the actual
endpoint program. Initial rejection and failed endpoint checks are included. -/
noncomputable def expectedTotalWork
    (oracleLaw : Law (Fin arity.total) (Challenge algebra) Assignment)
    (checker : Response Assignment Scalar params arity → CheckResult)
    (program : Primitives Scalar Assignment) : ℝ :=
  let typed := typedChecker algebra checker
  let charged := withChecker oracleLaw typed
  expectedQueryWork charged (CoordinateCheckedCalls.check typed) +
    expectedTerminalWork charged.oracle (CoordinateCheckedCalls.check typed)
      (fun vector initial outputs => (finish program vector initial outputs).work)

/-- Expected work comes from the actual transition and terminal clocks. The
terminal bound covers failures and uses only the bounded executed primitives. -/
theorem expectedTotalWork_bound
    (laws : ExtractionAlgebra semantics params algebra)
    (strongSet : StrongSetUnits laws.ring algebra.challengeValid)
    (oracleLaw : Law (Fin arity.total) (Challenge algebra) Assignment)
    (checker : Response Assignment Scalar params arity → CheckResult)
    (program : Primitives Scalar Assignment)
    (correct : Correct laws.ring laws.assignmentModule program)
    (bounds : PrimitiveBounds) (bounded : Bounded laws.ring program bounds) :
    expectedTotalWork algebra oracleLaw checker program ≤
      ((arity.total : ℝ) + 1) *
        (𝔼 vector, (withChecker oracleLaw (typedChecker algebra checker)).meanWork vector) +
      ((arity.total * (bounds.coordinateWork + 3) + 2 : Nat) : ℝ) := by
  let charged := withChecker oracleLaw (typedChecker algebra checker)
  let verify := CoordinateCheckedCalls.check (typedChecker algebra checker)
  have queries : expectedQueryWork charged verify ≤
      ((arity.total : ℝ) + 1) * (𝔼 vector, charged.meanWork vector) := by
    simpa only [Fintype.card_fin] using expectedQueryWork_bound charged verify
  have terminal : expectedTerminalWork charged.oracle verify
      (fun vector initial outputs => (finish program vector initial outputs).work) ≤
      ((arity.total * (bounds.coordinateWork + 3) + 2 : Nat) : ℝ) :=
    expectedTerminalWork_le charged.oracle verify _ _ (by omega)
      (fun vector initial outputs => finish_work_le laws.ring laws.assignmentModule
        program correct strongSet bounds bounded vector initial outputs)
  exact _root_.add_le_add queries terminal

/-- The explicit paper EPT/PPT premises bound the computed call mean and the
actual primitive clocks by polynomials in the security parameter. The whole
charged extractor then has the displayed polynomial expected-work bound. -/
theorem expectedTotalWork_polynomial_bound
    (laws : ExtractionAlgebra semantics params algebra)
    (strongSet : StrongSetUnits laws.ring algebra.challengeValid)
    (oracleLaw : Law (Fin arity.total) (Challenge algebra) Assignment)
    (checker : Response Assignment Scalar params arity → CheckResult)
    (program : Primitives Scalar Assignment)
    (correct : Correct laws.ring laws.assignmentModule program)
    (bounds : PrimitiveBounds) (bounded : Bounded laws.ring program bounds)
    (securityParameter : Nat) (callPolynomial primitivePolynomial : Polynomial ℝ)
    (callPPT : (𝔼 vector, (withChecker oracleLaw (typedChecker algebra checker)).meanWork vector) ≤
      callPolynomial.eval (securityParameter : ℝ))
    (primitivePPT : (bounds.coordinateWork : ℝ) ≤
      primitivePolynomial.eval (securityParameter : ℝ)) :
    expectedTotalWork algebra oracleLaw checker program ≤
      (Polynomial.C ((arity.total : ℝ) + 1) * callPolynomial +
        Polynomial.C (arity.total : ℝ) * (primitivePolynomial + Polynomial.C 3) +
        Polynomial.C 2).eval (securityParameter : ℝ) := by
  have total := expectedTotalWork_bound algebra laws strongSet oracleLaw checker program correct bounds bounded
  have queries := mul_le_mul_of_nonneg_left callPPT (by positivity : 0 ≤ (arity.total : ℝ) + 1)
  have primitives := mul_le_mul_of_nonneg_left
    (_root_.add_le_add_right primitivePPT 3) (Nat.cast_nonneg arity.total : (0 : ℝ) ≤ arity.total)
  simp only [Polynomial.eval_add, Polynomial.eval_mul, Polynomial.eval_C]
  push_cast at total
  nlinarith

/-- Every positive-mass returned value is the exact extracted source vector.
The program's own distinctness checks recover the same successful fork event;
the CompleteFork is proof evidence and is never chosen to compute the output. -/
theorem positive_return_implies_openings
    (laws : ExtractionAlgebra semantics params algebra)
    (strongSet : StrongSetUnits laws.ring algebra.challengeValid)
    (oracle : Oracle (Fin arity.total) (Challenge algebra) Assignment)
    (checker : Response Assignment Scalar params arity → CheckResult)
    (check_spec : ∀ found, (checker found).accepted = true ↔
      found.Success semantics params algebra batch)
    (program : Primitives Scalar Assignment)
    (correct : Correct laws.ring laws.assignmentModule program)
    (vector : Fin arity.total → Challenge algebra) (initial : Option Assignment)
    (outputs : Fin arity.total → Outcome
      (Challenge := Challenge algebra) (Assignment := Assignment)) (values : List Assignment)
    (positive : 0 < endpointMass oracle
      (oracleCheck algebra (fun found => (checker found).accepted)) vector initial outputs)
    (returned : (finish program vector initial outputs).value = some values) :
    ∃ fork : CompleteFork semantics params algebra batch,
      fork.base.challenges = scalarVector algebra vector ∧
      some fork.base.assignment = initial ∧
      (∀ coordinate,
        (fork.forks coordinate).challenges = Function.update (scalarVector algebra vector)
          coordinate (outputs coordinate).1.val ∧
        some (fork.forks coordinate).assignment = (outputs coordinate).2) ∧
      values = List.ofFn (extractedAssignment laws strongSet fork) ∧
      (∀ coordinate, PaperCorrections.CorrectedAmbientHolds semantics params
        (batch.inputs coordinate) (extractedAssignment laws strongSet fork coordinate)) := by
  obtain ⟨_base, _initial, _present, different⟩ :=
    finish_returns_implies_endpoints program vector initial outputs values returned
  have mass := endpointMass_eq_forkMass oracle
    (oracleCheck algebra (fun found => (checker found).accepted)) vector initial outputs
    (fun coordinate equal => different coordinate (congrArg Subtype.val equal).symm)
  rw [mass] at positive
  have event := positive_outcome_implies_event algebra batch oracle
    (fun found => (checker found).accepted) check_spec vector initial outputs positive
  obtain ⟨fork, baseVector, baseAssignment, responses, exactReturn, openings⟩ :=
    event_implies_result algebra batch laws strongSet program correct vector initial outputs event
  exact ⟨fork, baseVector, baseAssignment, responses,
    Option.some.inj (returned.symm.trans exactReturn), openings⟩

/-- The same implemented extractor supplies the paper loss, all oracle calls,
expected total work, and validity of every positive-mass returned source
opening. Initial rejection and repeated-coordinate failure cost are included.
Polynomial bounds on this computed mean and the primitives are supplied by
`expectedTotalWork_polynomial_bound`; no positive success premise is needed. -/
theorem bounds_and_openings
    (laws : ExtractionAlgebra semantics params algebra)
    (strongSet : StrongSetUnits laws.ring algebra.challengeValid)
    (oracleLaw : Law (Fin arity.total) (Challenge algebra) Assignment)
    (checker : Response Assignment Scalar params arity → CheckResult)
    (check_spec : ∀ found, (checker found).accepted = true ↔
      found.Success semantics params algebra batch)
    (program : Primitives Scalar Assignment)
    (correct : Correct laws.ring laws.assignmentModule program)
    (bounds : PrimitiveBounds) (bounded : Bounded laws.ring program bounds) :
    let charged := withChecker oracleLaw (typedChecker algebra checker)
    let verify := oracleCheck algebra (fun found => (checker found).accepted)
    ((line charged.oracle verify).rate - (arity.total : ℝ) / Fintype.card (Challenge algebra) ≤
      returningProbability charged.oracle verify
        (fun vector initial outputs => (finish program vector initial outputs).value.isSome)) ∧
    (1 + ∑' priorCalls : Nat, invocationTail charged.oracle verify priorCalls ≤ (arity.total : ℝ) + 1) ∧
    (expectedTotalWork algebra oracleLaw checker program ≤
      ((arity.total : ℝ) + 1) * (𝔼 vector, charged.meanWork vector) +
        ((arity.total * (bounds.coordinateWork + 3) + 2 : Nat) : ℝ)) ∧
    (∀ vector initial outputs values,
      0 < endpointMass charged.oracle verify vector initial outputs →
      (finish program vector initial outputs).value = some values →
      ∃ fork : CompleteFork semantics params algebra batch,
        fork.base.challenges = scalarVector algebra vector ∧
        some fork.base.assignment = initial ∧
        (∀ coordinate,
          (fork.forks coordinate).challenges = Function.update (scalarVector algebra vector)
            coordinate (outputs coordinate).1.val ∧
          some (fork.forks coordinate).assignment = (outputs coordinate).2) ∧
        values = List.ofFn (extractedAssignment laws strongSet fork) ∧
        (∀ coordinate, PaperCorrections.CorrectedAmbientHolds semantics params
          (batch.inputs coordinate) (extractedAssignment laws strongSet fork coordinate))) := by
  dsimp only
  let charged := withChecker oracleLaw (typedChecker algebra checker)
  let verify := oracleCheck algebra (fun found => (checker found).accepted)
  refine ⟨?_, ?_, expectedTotalWork_bound algebra laws strongSet oracleLaw checker program correct bounds bounded, ?_⟩
  · have lower := returningProbability_lower_bound charged.oracle verify
      (fun vector initial outputs => (finish program vector initial outputs).value.isSome)
      (fun vector initial outputs positive =>
        event_implies_returns algebra batch laws strongSet program correct vector initial outputs
          (positive_outcome_implies_event algebra batch charged.oracle
            (fun found => (checker found).accepted) check_spec vector initial outputs positive))
    simpa only [Fintype.card_fin] using lower
  · simpa only [Fintype.card_fin] using oracleCalls_bound charged.oracle verify
  · intro vector initial outputs values positive returned
    exact positive_return_implies_openings algebra batch laws strongSet charged.oracle checker check_spec
      program correct vector initial outputs values positive returned

end NightstreamFPrime.Spec.Folding.PiRLC.CoordinateExtraction
