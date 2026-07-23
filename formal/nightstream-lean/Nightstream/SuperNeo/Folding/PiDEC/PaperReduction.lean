import Nightstream.SuperNeo.Folding.PiDEC.PaperVerifier
import Nightstream.SuperNeo.InteractiveReduction.Paper

/-!
Paper reduction-of-knowledge theorem for SuperNeo Pi_DEC.

Source: SuperNeo Section 7.5, Theorem 7, and Appendix D.6.

Owns: the exact operational success experiment, the straight-line
recomposition extractor, perfect completeness, and the zero-loss quantitative
knowledge inequality.

Does not own: Pi_CCS, Pi_RLC, commitment binding, Fiat--Shamir, a concrete
field/commitment implementation, Rust, R1CS, or costs.

Emits constraints: no.

The adversary may induce any probability experiment over operational outputs.
The sole extractor performs the paper's fixed `k`-term radix recomposition.
No commitment-binding event is required for Theorem 7: the extracted parent
opening is the recomposition itself.
-/

namespace Nightstream.SuperNeo.Folding.PiDEC.PaperReduction

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.InteractiveReduction.Paper

universe uStructure uAssignment uPublicInput uPoint uEvaluation uCommitment
  uWeight

/-- All verifier-owned operations needed by the exact paper reduction. -/
structure Context
    (Structure : Type uStructure)
    (Assignment : Type uAssignment)
    (PublicInput : Type uPublicInput)
    (Point : Type uPoint)
    (Evaluation : Type uEvaluation)
    (Commitment : Type uCommitment) where
  semantics : RelationSemantics
    Structure Assignment PublicInput Point Evaluation Commitment
  params : GlobalParams
  algebra : PiDEC.Algebra Structure Assignment PublicInput Point Evaluation
    Commitment semantics params
  publicSplit : PaperVerifier.PublicInputSplit algebra
  evaluationArity : PaperVerifier.EvaluationArity semantics
  kPositive : 0 < params.k

/-- One adversarial operational output and the witnesses claimed for all
children. -/
structure Execution
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    (context : Context Structure Assignment PublicInput Point Evaluation
      Commitment) where
  attempt : PaperVerifier.Attempt Structure PublicInput Point Evaluation
    Commitment context.params
  childAssignments : Fin context.params.k -> Assignment

/-- The Definition-5 success event: the exact verifier accepts and the
adversarial output belongs to the target `CE(b)^k` relation. -/
def Success
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    (context : Context Structure Assignment PublicInput Point Evaluation
      Commitment)
    (execution : Execution context) : Prop :=
  PaperVerifier.Accepted context.algebra context.evaluationArity
      execution.attempt /\
    forall child,
      CE.Holds context.semantics context.params
        (PaperVerifier.children context.publicSplit execution.attempt child)
        (execution.childAssignments child)

/-- Event that the straight-line extractor produced a valid source witness. -/
def ExtractedSource
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    (context : Context Structure Assignment PublicInput Point Evaluation
      Commitment)
    (execution : Execution context) : Prop :=
  CE.Holds context.semantics context.params execution.attempt.parent
    (context.algebra.recomposeAssignment execution.childAssignments)

/-- Pointwise straight-line extraction, exactly Appendix D.6. -/
theorem success_implies_extractedSource
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    (context : Context Structure Assignment PublicInput Point Evaluation
      Commitment)
    (execution : Execution context)
    (success : Success context execution) :
    ExtractedSource context execution := by
  exact PaperVerifier.reduce_knowledge context.semantics context.params
    context.algebra context.publicSplit context.evaluationArity
    execution.attempt execution.childAssignments context.kPositive
    success.1 success.2

/-- Honest operational execution produced from one valid combined parent. -/
def honestExecution
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    (context : Context Structure Assignment PublicInput Point Evaluation
      Commitment)
    (parent : CE.Instance Structure PublicInput Point Evaluation Commitment)
    (assignment : Assignment) : Execution context where
  attempt := PaperVerifier.honestAttempt context.algebra parent assignment
  childAssignments := context.algebra.splitAssignment assignment

/-- Perfect-completeness proposition computed by the Pi_DEC knowledge game. -/
def PerfectComplete
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    (context : Context Structure Assignment PublicInput Point Evaluation
      Commitment) : Prop :=
  forall parent assignment,
    parent.stage = NormStage.combined ->
    CE.Holds context.semantics context.params parent assignment ->
      Success context (honestExecution context parent assignment)

/-- Honest Pi_DEC execution is accepted and every output child is valid. -/
theorem perfectComplete
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    (context : Context Structure Assignment PublicInput Point Evaluation
      Commitment) :
    PerfectComplete context := by
  intro parent assignment parentCombined parentValid
  simpa [Success, honestExecution] using
    (PaperVerifier.complete context.semantics context.params context.algebra
      context.publicSplit context.evaluationArity parent assignment
      parentCombined parentValid)

/-- There is exactly one extraction algorithm: Appendix D.6's straight-line
recomposition. -/
inductive Extractor where
  | straightLine
deriving Repr, DecidableEq

/-- Symbolic extractor work is one pass over the `k` children. -/
def Extractor.work
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    (context : Context Structure Assignment PublicInput Point Evaluation
      Commitment) : Extractor -> Nat
  | .straightLine => context.params.k

theorem straightLine_work
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    (context : Context Structure Assignment PublicInput Point Evaluation
      Commitment) :
    Extractor.work context .straightLine = context.params.k := by
  rfl

/-- Definition-5 game induced by arbitrary probability experiments over exact
Pi_DEC executions. -/
def knowledgeGame
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Weight : Type uWeight}
    (context : Context Structure Assignment PublicInput Point Evaluation
      Commitment)
    (scale : ProbabilityScale Weight) :
    KnowledgeGame Weight (ProbabilityExperiment scale (Execution context))
      Extractor where
  perfectComplete := PerfectComplete context
  publicCoin := True
  adversaryExpectedPolynomialTime := fun _ => True
  extractorExpectedPolynomialTime := fun _ extractor =>
    extractor = .straightLine
  extractionEligible := fun _ => True
  adversarySuccess := fun experiment =>
    experiment.probability (Success context)
  sourceWitnessExtracted := fun experiment _ =>
    experiment.probability (ExtractedSource context)

/-- Obligation 3: exact Pi_DEC is a zero-loss reduction of knowledge. -/
theorem reductionOfKnowledge
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Weight : Type uWeight}
    (context : Context Structure Assignment PublicInput Point Evaluation
      Commitment)
    (scale : ProbabilityScale Weight) :
    ReductionOfKnowledge scale (knowledgeGame context scale) scale.zero := by
  refine ⟨perfectComplete context, True.intro, ?_⟩
  intro experiment _ _
  refine ⟨.straightLine, rfl, ?_⟩
  rw [scale.subtract_zero]
  exact experiment.monotone fun execution success =>
    success_implies_extractedSource context execution success

end Nightstream.SuperNeo.Folding.PiDEC.PaperReduction
