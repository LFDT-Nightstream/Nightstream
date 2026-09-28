import NightstreamFPrime.Spec.Folding.Nifs.PaperWeakExtraction

/-!
One captured post-PiCCS continuation and its executed checks. The oracle law,
success probability and weak-extraction bound are derived from these fields.
No fork, parent-opening family, response law or claimed work total is supplied.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Folding.Nifs.PaperWeakAlgorithm

open NightstreamFPrime.Spec
open _root_.NightstreamFPrime.Spec.Folding
open PiRLC.PaperForkExtraction PiRLC.CoordinateForkLaw PiRLC.CoordinateCheckedCalls

variable (Tape : Type*)
  {Structure Assignment PublicInput Point Evaluation Commitment Scalar : Type*}
  {semantics : RelationSemantics Structure Assignment PublicInput Point Evaluation Commitment}
  {params : GlobalParams} {arity : BatchArity params}
  (rlc : PiRLC.Algebra Structure Assignment PublicInput Point Evaluation Commitment Scalar semantics params)
  (batch : InputBatch Structure PublicInput Point Evaluation Commitment params arity)
  (dec : PiDEC.Algebra Structure Assignment PublicInput Point Evaluation Commitment semantics params)
  (publicSplit : PiDEC.PaperVerifier.PublicInputSplit dec)
  (evaluationArity : PiDEC.PaperVerifier.EvaluationArity semantics)

/-- Data of the actual resumed suffix, indexed by its exact public batch and
verifier. Private coins are fresh on each call. Only the original call/check
first moment is required; the final extractor moment is derived. -/
structure Algorithm where
  tapes : PMF Tape
  rawCall : PaperWeakOracle.Call (Tape := Tape) (arity := arity) rlc
  suffixProgram : PaperWeakSuffix.Program (arity := arity) rlc
  suffixCorrect : PaperWeakSuffix.Correct rlc batch dec publicSplit evaluationArity suffixProgram
  recomposeBound : Nat
  recomposeBounded : ∀ assignments, (suffixProgram.recompose assignments).work ≤ recomposeBound
  baseSummable : ∀ vector, Summable fun tape =>
    (tapes tape).toReal * (PaperWeakOracle.baseWork rlc suffixProgram rawCall vector tape : ℝ)
  parentChecker : Response Assignment Scalar params arity → CheckResult
  parentChecker_spec : ∀ found, (parentChecker found).accepted = true ↔
    found.Success semantics params rlc batch

namespace Algorithm

variable {Tape rlc batch dec publicSplit evaluationArity}
  (algorithm : Algorithm Tape rlc batch dec publicSplit evaluationArity)

/-- The parent check is the Bool projection of the same costed checker. -/
def check : (Fin arity.total → Challenge rlc) → Assignment → Bool :=
  oracleCheck rlc (fun found => (algorithm.parentChecker found).accepted)

variable [Fintype Assignment]

/-- Actual suffix tapes are pushed forward to the returned parent and clock. -/
noncomputable def oracleLaw :
    PiRLC.CoordinateChargedOracle.Law (Fin arity.total) (Challenge rlc) Assignment :=
  PaperWeakOracle.law rlc algorithm.tapes algorithm.suffixProgram algorithm.rawCall
    algorithm.recomposeBound algorithm.recomposeBounded algorithm.baseSummable

/-- The stopped-search response law erases the clock after adding the actual
parent-check work. Abort mass is retained. -/
noncomputable def chargedOracle :
    PiRLC.CoordinateOracle.Oracle (Fin arity.total) (Challenge rlc) Assignment :=
  (withChecker algorithm.oracleLaw
    (PiRLC.CoordinateExtraction.typedChecker rlc algorithm.parentChecker)).oracle

/-- Exact one-query work before the extra parent check, over the same tapes
that determine the returned parent or abort. -/
theorem oracle_meanWork (vector : Fin arity.total → Challenge rlc) :
    algorithm.oracleLaw.meanWork vector =
      ∑' tape, (algorithm.tapes tape).toReal *
        (((PaperWeakOracle.run rlc algorithm.suffixProgram algorithm.rawCall vector tape).work : ℝ) + 1) :=
  PaperWeakOracle.law_meanWork rlc algorithm.tapes algorithm.suffixProgram algorithm.rawCall
    algorithm.recomposeBound algorithm.recomposeBounded algorithm.baseSummable vector

/-- The original call/check mean controls the derived suffix mean. It is not
a uniform polynomial bound on adversary contexts. -/
theorem oracle_meanWork_le (vector : Fin arity.total → Challenge rlc) :
    algorithm.oracleLaw.meanWork vector ≤
      (∑' tape, (algorithm.tapes tape).toReal *
        (PaperWeakOracle.baseWork rlc algorithm.suffixProgram algorithm.rawCall vector tape : ℝ)) +
      ((algorithm.recomposeBound + 4 : Nat) : ℝ) :=
  PaperWeakOracle.law_meanWork_le rlc algorithm.tapes algorithm.suffixProgram algorithm.rawCall
    algorithm.recomposeBound algorithm.recomposeBounded algorithm.baseSummable vector

variable [Fintype (Challenge rlc)] [Nonempty (Challenge rlc)]

/-- Original accepted final-output probability under independent public
challenge coins and this continuation's private tapes. -/
noncomputable def successProbability : ℝ :=
  PaperWeakExtraction.successProbability rlc batch dec publicSplit evaluationArity
    algorithm.tapes algorithm.rawCall

theorem base_rate (kPositive : 0 < params.k) :
    (PiRLC.CoordinateOracle.line algorithm.chargedOracle algorithm.check).rate =
      algorithm.successProbability :=
  PaperWeakExtraction.rate_eq_successProbability rlc batch dec publicSplit evaluationArity
    algorithm.tapes algorithm.suffixProgram algorithm.suffixCorrect kPositive algorithm.rawCall
    algorithm.recomposeBound algorithm.recomposeBounded algorithm.baseSummable
    algorithm.parentChecker algorithm.parentChecker_spec

variable [DecidableEq Scalar]

/-- Derived weak bound for this actual continuation and the existing charged
coordinate extractor. The returned list is computed from its observed endpoint. -/
theorem weak_success_bound
    (laws : ExtractionAlgebra semantics params rlc)
    (strongSet : StrongSetUnits laws.ring rlc.challengeValid)
    (kPositive : 0 < params.k)
    (extraction : PiRLC.PaperForkExtractionWork.Primitives Scalar Assignment)
    (extractionCorrect : PiRLC.PaperForkExtractionWork.Correct laws.ring laws.assignmentModule extraction)
    (bounds : PiRLC.PaperForkExtractionWork.PrimitiveBounds)
    (extractionBounded : PiRLC.PaperForkExtractionWork.Bounded laws.ring extraction bounds) :
    algorithm.successProbability - (arity.total : ℝ) / Fintype.card (Challenge rlc) ≤
      PaperWeakLaw.successProbability algorithm.chargedOracle algorithm.check extraction :=
  PaperWeakExtraction.weak_success_bound rlc batch dec publicSplit evaluationArity laws strongSet
    algorithm.tapes algorithm.suffixProgram algorithm.suffixCorrect kPositive algorithm.rawCall
    algorithm.recomposeBound algorithm.recomposeBounded algorithm.baseSummable
    algorithm.parentChecker algorithm.parentChecker_spec extraction extractionCorrect bounds extractionBounded

end Algorithm

end NightstreamFPrime.Spec.Folding.Nifs.PaperWeakAlgorithm
