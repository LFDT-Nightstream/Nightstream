import Nightstream.SuperNeo.Folding.PiDEC.PaperReduction

/-!
Focused interface regression for the paper-exact quantitative reduction of
knowledge of `Pi_DEC`.
-/

namespace tests.PiDECPaperReduction

open Nightstream.SuperNeo.InteractiveReduction.Paper
open Nightstream.SuperNeo.Folding.PiDEC.PaperReduction

universe uStructure uAssignment uPublicInput uPoint uEvaluation uCommitment
  uWeight

variable
  {Structure : Type uStructure}
  {Assignment : Type uAssignment}
  {PublicInput : Type uPublicInput}
  {Point : Type uPoint}
  {Evaluation : Type uEvaluation}
  {Commitment : Type uCommitment}
  {Weight : Type uWeight}

#check success_implies_extractedSource
#check perfectComplete
#check straightLine_work
#check reductionOfKnowledge

example
    (context : Context Structure Assignment PublicInput Point Evaluation
      Commitment)
    (execution : Execution context)
    (success : Success context execution) :
    ExtractedSource context execution :=
  success_implies_extractedSource context execution success

example
    (context : Context Structure Assignment PublicInput Point Evaluation
      Commitment) :
    PerfectComplete context :=
  perfectComplete context

example
    (context : Context Structure Assignment PublicInput Point Evaluation
      Commitment)
    (scale : ProbabilityScale Weight) :
    ReductionOfKnowledge scale (knowledgeGame context scale) scale.zero :=
  reductionOfKnowledge context scale

end tests.PiDECPaperReduction
