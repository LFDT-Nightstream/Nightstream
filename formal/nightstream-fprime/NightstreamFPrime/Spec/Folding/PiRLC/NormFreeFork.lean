import NightstreamFPrime.Spec.Folding.PiRLC.PaperForkExtraction

/-! SuperNeo v1.2 Lemma 8 at any output norm. The extraction of Appendix B.3 reads the
commitments, public inputs and evaluations of the fork outputs, never their norms. So a fork of
the relation with the norm check removed extracts every input, and the extracted witnesses keep
the ambient bound of `ExtractionAlgebra.correctedNormCoverage`. A chain extracts refresh
outputs above `B`; their forks have this form. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Folding.PiRLC.NormFreeFork

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiRLC.PaperForkExtraction

universe uStructure uAssignment uPublicInput uPoint uEvaluation uCommitment uScalar

variable {Structure : Type uStructure} {Assignment : Type uAssignment}
  {PublicInput : Type uPublicInput} {Point : Type uPoint} {Evaluation : Type uEvaluation}
  {Commitment : Type uCommitment} {Scalar : Type uScalar}
  {semantics : RelationSemantics Structure Assignment PublicInput Point Evaluation Commitment}
  {params : GlobalParams} {arity : BatchArity params}
  {algebra : Algebra Structure Assignment PublicInput Point Evaluation Commitment Scalar
    semantics params}
  {batch : InputBatch Structure PublicInput Point Evaluation Commitment params arity}

/-- The relation with the norm check removed. -/
def normFree
    (semantics : RelationSemantics Structure Assignment PublicInput Point Evaluation Commitment) :
    RelationSemantics Structure Assignment PublicInput Point Evaluation Commitment :=
  { semantics with normBounded := fun _ _ => True }

def normFreeAlgebra
    (algebra : Algebra Structure Assignment PublicInput Point Evaluation Commitment Scalar
      semantics params) :
    Algebra Structure Assignment PublicInput Point Evaluation Commitment Scalar
      (normFree semantics) params :=
  { algebra with norm_growth := fun _ _ _ _ _ => trivial }

def normFreeLaws (laws : ExtractionAlgebra semantics params algebra) :
    ExtractionAlgebra (normFree semantics) params (normFreeAlgebra algebra) :=
  { laws with correctedNormCoverage := fun _ => trivial }

/-- A fork at the output stage bound is a fork of the relation without norms. -/
def forgetNorms (fork : CompleteFork semantics params algebra batch) :
    CompleteFork (normFree semantics) params (normFreeAlgebra algebra) batch where
  base := fork.base
  forks := fork.forks
  baseSuccess := And.intro (And.intro fork.baseSuccess.1.1
    (And.intro fork.baseSuccess.1.2.1 trivial)) fork.baseSuccess.2
  forkSuccess coordinate := And.intro (And.intro (fork.forkSuccess coordinate).1.1
    (And.intro (fork.forkSuccess coordinate).1.2.1 trivial)) (fork.forkSuccess coordinate).2
  baseStrong := fork.baseStrong
  forkStrong := fork.forkStrong
  agreeExcept := fork.agreeExcept
  changed := fork.changed

/-- A response whose output opens at any norm bound, such as `B + cap · T · (b − 1)`, succeeds
without norms. -/
theorem success_at (bound : Nat) (response : Response Assignment Scalar params arity)
    (opens : Opening.Holds semantics bound (response.output algebra batch).commitment
      (response.output algebra batch).publicInput response.assignment)
    (pointValid : semantics.evaluationPointValid (response.output algebra batch).constraintSystem
      (response.output algebra batch).point)
    (evaluations : semantics.evaluations (response.output algebra batch).constraintSystem
      response.assignment (response.output algebra batch).point =
        (response.output algebra batch).evaluations) :
    response.Success (normFree semantics) params (normFreeAlgebra algebra) batch :=
  And.intro (And.intro opens.1 (And.intro opens.2.1 trivial)) (And.intro pointValid evaluations)

/-- SuperNeo v1.2 Lemma 8 for a fork whose outputs satisfy no norm bound. -/
theorem correctedAmbientHolds (laws : ExtractionAlgebra semantics params algebra)
    (strongSet : StrongSetUnits laws.ring algebra.challengeValid)
    (fork : CompleteFork (normFree semantics) params (normFreeAlgebra algebra) batch) :
    ∀ coordinate, PaperCorrections.CorrectedAmbientHolds semantics params
      (batch.inputs coordinate)
      (extractedAssignment (normFreeLaws laws) strongSet fork coordinate) := by
  intro coordinate
  obtain ⟨⟨commits, projects, _⟩, pointValid, evaluations⟩ :=
    completeFork_implies_correctedAmbientHolds (normFree semantics) params arity
      (normFreeAlgebra algebra) (normFreeLaws laws) strongSet batch fork coordinate
  unfold PaperCorrections.CorrectedAmbientHolds
  exact ⟨⟨commits, projects, laws.correctedNormCoverage _⟩, pointValid, evaluations⟩

end NightstreamFPrime.Spec.Folding.PiRLC.NormFreeFork
