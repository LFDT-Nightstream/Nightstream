import Nightstream.SuperNeo.Folding.PiRLC.PaperCompleteness
import Nightstream.SuperNeo.Folding.PiRLC.PaperForkCollision
import Nightstream.SuperNeo.InteractiveReduction.CoordinateForking

/-!
Operational paper `Pi_RLC` weak reduction, conditional only on Theorem 10's
generic finite-uniform coordinate-forking contract and relaxed binding.

Source: SuperNeo Lemma 4, Appendix C Theorem 10, and Appendix D.5.

Owns: vector-at-once adversary oracles, the exact `Response.Success` verifier,
conversion of accepted coordinate forks to the deterministic paper extractor,
corrected ambient extraction, direct paired relaxed-binding collisions, and a
concrete `WeakGame` whose completeness/public-coin fields come from the paper
protocol.

Does not own: a proof of the generic finite-uniform coordinate-fork theorem,
the relaxed-binding assumption, source-relation validity, Fiat--Shamir, Rust,
R1CS, artifacts, or costs.

Emits constraints: no.

The headline theorem is conditional on the explicitly named generic
`Theorem10Contract`.  It therefore does not unconditionally discharge frozen
obligation 2 until that contract is instantiated by the finite-uniform theorem.
It does prove that no additional `Pi_RLC`-specific extraction or paired
witness-disagreement premise is needed.
-/

namespace Nightstream.SuperNeo.Folding.PiRLC.PaperWeakReduction

open Nightstream.SuperNeo
open Nightstream.SuperNeo.InteractiveReduction.Paper
open Nightstream.SuperNeo.InteractiveReduction.CoordinateForking
open Nightstream.SuperNeo.Folding.PiRLC.PaperForkExtraction
open Nightstream.SuperNeo.Folding.PiRLC.PaperForkCollision
open Nightstream.SuperNeo.Folding.PiRLC.PaperCompleteness

universe uWeight uStructure uAssignment uPublicInput uPoint uEvaluation
  uCommitment uScalar uExtractor

/-- One complete paper challenge vector. -/
abbrev Challenge
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    (context : Context Structure Assignment PublicInput Point
      Evaluation Commitment Scalar) :=
  ChallengeVector Scalar context.arity.total

/-- An operational adversary chooses a public input batch and answers each
complete challenge vector with one assignment. -/
structure Adversary
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    (context : Context Structure Assignment PublicInput Point
      Evaluation Commitment Scalar) where
  batch : InputBatch Structure PublicInput Point Evaluation Commitment
    context.params context.arity
  oracle : (Fin context.arity.total -> Scalar) -> Assignment

/-- The exact response made from a verifier challenge and one oracle answer. -/
def response
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    (context : Context Structure Assignment PublicInput Point
      Evaluation Commitment Scalar)
    (challenges : Challenge context)
    (assignment : Assignment) :
    Response Assignment Scalar context.params context.arity where
  challenges := challenges
  assignment := assignment

/-- The operational verifier predicate is literally `Response.Success` for
the response above.  Its public output is definitionally
`PiRLC.combinedOutput`; the oracle supplies only the assignment. -/
def verifies
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    (context : Context Structure Assignment PublicInput Point
      Evaluation Commitment Scalar)
    (adversary : Adversary context)
    (challenges : Challenge context)
    (assignment : Assignment) : Prop :=
  (response context challenges assignment).Success context.semantics
    context.params context.algebra adversary.batch

/-- The generic accepted-fork predicate specialized to the exact paper
verifier and the adversary's vector-at-once oracle. -/
abbrev AcceptedFork
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    (context : Context Structure Assignment PublicInput Point
      Evaluation Commitment Scalar)
    (adversary : Adversary context)
    (sample : ForkSample Scalar context.arity.total) :=
  AcceptedCoordinateFork
    context.algebra.challengeValid (verifies context adversary)
      adversary.oracle sample

/-- A base accepted response plus one accepted coordinate fork is exactly the
deterministic `CompleteFork` consumed by Appendix D.5's algebra. -/
def acceptedFork_to_completeFork
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    (context : Context Structure Assignment PublicInput Point
      Evaluation Commitment Scalar)
    (adversary : Adversary context)
    (sample : ForkSample Scalar context.arity.total)
    (accepted : AcceptedFork context adversary sample) :
    CompleteFork context.semantics context.params context.algebra
      adversary.batch where
  base := response context sample.base (adversary.oracle sample.base)
  forks := fun coordinate =>
    response context (sample.forks coordinate)
      (adversary.oracle (sample.forks coordinate))
  baseSuccess := accepted.baseAccepted
  forkSuccess := accepted.forkAccepted
  baseStrong := accepted.baseValid
  forkStrong := accepted.forkValid
  agreeExcept := accepted.agreeExcept
  changed := accepted.changed

/-- The actual assignment family returned by Appendix D.5 for one accepted
coordinate fork. -/
def extractedFamily
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    (context : Context Structure Assignment PublicInput Point
      Evaluation Commitment Scalar)
    (laws : ExtractionAlgebra context.semantics context.params context.algebra)
    (strongSet : StrongSetUnits laws.ring context.algebra.challengeValid)
    (adversary : Adversary context)
    (sample : ForkSample Scalar context.arity.total)
    (accepted : AcceptedFork context adversary sample) :
    Fin context.arity.total -> Assignment :=
  fun coordinate =>
    extractedAssignment laws strongSet
      (acceptedFork_to_completeFork context adversary sample accepted)
      coordinate

/-- The weak extractor's real semantic success event.  It records both the
accepted operational fork and corrected ambient membership of the assignment
family computed from that same fork. -/
def ExtractsCorrectedAmbient
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    (context : Context Structure Assignment PublicInput Point
      Evaluation Commitment Scalar)
    (laws : ExtractionAlgebra context.semantics context.params context.algebra)
    (strongSet : StrongSetUnits laws.ring context.algebra.challengeValid)
    (adversary : Adversary context)
    (sample : ForkSample Scalar context.arity.total) : Prop :=
  exists accepted : AcceptedFork context adversary sample,
    forall coordinate,
      PaperCorrections.CorrectedAmbientHolds context.semantics context.params
        (adversary.batch.inputs coordinate)
        (extractedFamily context laws strongSet adversary sample accepted
          coordinate)

/-- Accepted operational forks always produce corrected ambient openings.
No source-validity or target conclusion is a premise. -/
theorem acceptedFork_extracts_correctedAmbient
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    (context : Context Structure Assignment PublicInput Point
      Evaluation Commitment Scalar)
    (laws : ExtractionAlgebra context.semantics context.params context.algebra)
    (strongSet : StrongSetUnits laws.ring context.algebra.challengeValid)
    (adversary : Adversary context)
    (sample : ForkSample Scalar context.arity.total)
    (accepted : AcceptedFork context adversary sample) :
    ExtractsCorrectedAmbient context laws strongSet adversary sample := by
  refine ⟨accepted, ?_⟩
  exact completeFork_implies_correctedAmbientHolds context.semantics
    context.params context.arity context.algebra laws strongSet adversary.batch
    (acceptedFork_to_completeFork context adversary sample accepted)

/-- The paper's paired `(B,B')` adversary. -/
structure PairedAdversary
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    (context : Context Structure Assignment PublicInput Point
      Evaluation Commitment Scalar) where
  left : Adversary context
  right : Adversary context

/-- Literal equality of the two input commitment projections. -/
def SamePhi
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    {context : Context Structure Assignment PublicInput Point
      Evaluation Commitment Scalar}
    (paired : PairedAdversary context) : Prop :=
  PiRLC.phi paired.left.batch.inputs = PiRLC.phi paired.right.batch.inputs

/-- The two chosen coordinate extractors return different non-bottom
assignment families. -/
def PairedWitnessDisagreement
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    (context : Context Structure Assignment PublicInput Point
      Evaluation Commitment Scalar)
    (laws : ExtractionAlgebra context.semantics context.params context.algebra)
    (strongSet : StrongSetUnits laws.ring context.algebra.challengeValid)
    (paired : PairedAdversary context)
    (samples : ForkSample Scalar context.arity.total ×
      ForkSample Scalar context.arity.total) : Prop :=
  exists leftAccepted : AcceptedFork context paired.left samples.1,
    exists rightAccepted : AcceptedFork context paired.right samples.2,
      exists coordinate,
        extractedFamily context laws strongSet paired.left samples.1
            leftAccepted coordinate ≠
          extractedFamily context laws strongSet paired.right samples.2
            rightAccepted coordinate

/-- The exact execution-dependent collision event.  It retains the two sampled
forks through their accepted-fork proofs, their explicit differing coordinate,
and the collision receipt whose deltas/openings are definitionally those
forks' values. -/
def PairedForkCollisionReceipt
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    (context : Context Structure Assignment PublicInput Point
      Evaluation Commitment Scalar)
    (laws : ExtractionAlgebra context.semantics context.params context.algebra)
    (strongSet : StrongSetUnits laws.ring context.algebra.challengeValid)
    (ops : RelaxedBindingOps Assignment Commitment Scalar)
    (paired : PairedAdversary context)
    (samples : ForkSample Scalar context.arity.total ×
      ForkSample Scalar context.arity.total) : Prop :=
  exists leftAccepted : AcceptedFork context paired.left samples.1,
    exists rightAccepted : AcceptedFork context paired.right samples.2,
      exists coordinate,
        (extractedFamily context laws strongSet paired.left samples.1
              leftAccepted coordinate ≠
            extractedFamily context laws strongSet paired.right samples.2
              rightAccepted coordinate) ∧
          CoordinateForkCollisionReceipt laws ops
            (acceptedFork_to_completeFork context paired.left samples.1
              leftAccepted)
            (acceptedFork_to_completeFork context paired.right samples.2
              rightAccepted)
            coordinate

/-- Different accepted extraction families at the same `phi` construct the
literal relaxed-binding collision; disagreement probability is never a
premise. -/
theorem pairedDisagreement_implies_collisionReceipt
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    (context : Context Structure Assignment PublicInput Point
      Evaluation Commitment Scalar)
    (laws : ExtractionAlgebra context.semantics context.params context.algebra)
    (strongSet : StrongSetUnits laws.ring context.algebra.challengeValid)
    (ops : RelaxedBindingOps Assignment Commitment Scalar)
    (bindingLaws : RelaxedBindingLaws context.semantics context.params
      context.algebra laws ops)
    (paired : PairedAdversary context)
    (samePhi : SamePhi paired)
    (samples : ForkSample Scalar context.arity.total ×
      ForkSample Scalar context.arity.total)
    (disagreement : PairedWitnessDisagreement context laws strongSet paired
      samples) :
    PairedForkCollisionReceipt context laws strongSet ops paired samples := by
  rcases disagreement with
    ⟨leftAccepted, rightAccepted, coordinate, different⟩
  refine ⟨leftAccepted, rightAccepted, coordinate, different, ?_⟩
  exact coordinate_differingExtractions_imply_collisionReceipt
    context.semantics context.params context.arity context.algebra laws
    strongSet ops bindingLaws paired.left.batch paired.right.batch
    (acceptedFork_to_completeFork context paired.left samples.1 leftAccepted)
    (acceptedFork_to_completeFork context paired.right samples.2 rightAccepted)
    samePhi coordinate different

/-- The generic Theorem-10 contract specialized definitionally to `Pi_RLC`'s
operational verifier and assignment oracle. -/
abbrev ForkingContract
    {Weight : Type uWeight}
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    {Extractor : Type uExtractor}
    (scale : ProbabilityScale Weight)
    (ratio : Nat -> Nat -> Weight)
    (context : Context Structure Assignment PublicInput Point
      Evaluation Commitment Scalar) :=
  Theorem10Contract (Extractor := Extractor) scale ratio context.arity.total
    context.algebra.challengeValid (verifies context)
    (fun adversary => adversary.oracle)

/-- Actual probability experiments for paired executions and the only
binding-security premise: an upper bound on the literal indexed collision
event. -/
structure OperationalGame
    {Weight : Type uWeight}
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    {Extractor : Type uExtractor}
    (scale : ProbabilityScale Weight)
    (ratio : Nat -> Nat -> Weight)
    (context : Context Structure Assignment PublicInput Point
      Evaluation Commitment Scalar)
    (laws : ExtractionAlgebra context.semantics context.params context.algebra)
    (strongSet : StrongSetUnits laws.ring context.algebra.challengeValid)
    (ops : RelaxedBindingOps Assignment Commitment Scalar)
    (relaxedBindingError : Weight) where
  theorem10 : ForkingContract scale ratio context (Extractor := Extractor)
  pairedForks : PairedAdversary context -> Extractor -> Extractor ->
    ProbabilityExperiment scale
      (ForkSample Scalar context.arity.total ×
        ForkSample Scalar context.arity.total)
  relaxedBindingBound : forall paired leftExtractor rightExtractor,
    theorem10.adversaryExpectedPolynomialTime paired.left ->
    theorem10.adversaryExpectedPolynomialTime paired.right ->
    theorem10.extractorExpectedPolynomialTime paired.left leftExtractor ->
    theorem10.extractorExpectedPolynomialTime paired.right rightExtractor ->
    SamePhi paired ->
    scale.le
      ((pairedForks paired leftExtractor rightExtractor).probability
        (PairedForkCollisionReceipt context laws strongSet ops paired))
      relaxedBindingError

/-- The independently defined operational `WeakGame`. -/
def weakGame
    {Weight : Type uWeight}
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    {Extractor : Type uExtractor}
    (scale : ProbabilityScale Weight)
    (ratio : Nat -> Nat -> Weight)
    (context : Context Structure Assignment PublicInput Point
      Evaluation Commitment Scalar)
    (laws : ExtractionAlgebra context.semantics context.params context.algebra)
    (strongSet : StrongSetUnits laws.ring context.algebra.challengeValid)
    (ops : RelaxedBindingOps Assignment Commitment Scalar)
    (relaxedBindingError : Weight)
    (game : OperationalGame (Extractor := Extractor) scale ratio context laws
      strongSet ops
      relaxedBindingError) :
    WeakGame Weight (Adversary context) (PairedAdversary context) Extractor where
  perfectComplete := PerfectComplete context
  publicCoin := PublicCoin context
  adversaryExpectedPolynomialTime :=
    game.theorem10.adversaryExpectedPolynomialTime
  pairedAdversaryExpectedPolynomialTime := fun paired =>
    game.theorem10.adversaryExpectedPolynomialTime paired.left ∧
      game.theorem10.adversaryExpectedPolynomialTime paired.right
  extractorExpectedPolynomialTime :=
    game.theorem10.extractorExpectedPolynomialTime
  extractionEligible := fun _ => True
  adversarySuccess := fun adversary =>
    (game.theorem10.uniformChallenges adversary).probability fun challenges =>
      verifies context adversary challenges (adversary.oracle challenges)
  ambientSourceWitnessExtracted := fun adversary extractor =>
    (game.theorem10.forkSamples adversary extractor).probability
      (ExtractsCorrectedAmbient context laws strongSet adversary)
  left := PairedAdversary.left
  right := PairedAdversary.right
  samePhiInputsAlways := SamePhi
  pairedWitnessDisagreement := fun paired leftExtractor rightExtractor =>
    (game.pairedForks paired leftExtractor rightExtractor).probability
      (PairedWitnessDisagreement context laws strongSet paired)

/-- Appendix D.5's weak-reduction theorem, with exactly the corrected
coordinate-fork loss and relaxed-binding loss.

This theorem does not assume `Weak`, source validity, the desired extraction
inequality, or a paired witness-disagreement bound.  The only probabilistic
premises are the generic finite-uniform Theorem-10 contract stored in `game`
and its literal relaxed-binding collision bound. -/
theorem paperWeak
    {Weight : Type uWeight}
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    {Extractor : Type uExtractor}
    (scale : ProbabilityScale Weight)
    (ratio : Nat -> Nat -> Weight)
    (context : Context Structure Assignment PublicInput Point
      Evaluation Commitment Scalar)
    (laws : ExtractionAlgebra context.semantics context.params context.algebra)
    (strongSet : StrongSetUnits laws.ring context.algebra.challengeValid)
    (ops : RelaxedBindingOps Assignment Commitment Scalar)
    (bindingLaws : RelaxedBindingLaws context.semantics context.params
      context.algebra laws ops)
    (relaxedBindingError : Weight)
    (game : OperationalGame (Extractor := Extractor) scale ratio context laws
      strongSet ops
      relaxedBindingError) :
    Weak scale
      (weakGame (Extractor := Extractor) scale ratio context laws strongSet ops
        relaxedBindingError game)
      (correctedLoss ratio context.arity.total
        game.theorem10.challengeSetCardinality)
      relaxedBindingError := by
  refine ⟨perfectComplete context, publicCoin context, ?_⟩
  let chooseExtractor : Adversary context -> Extractor :=
    game.theorem10.coordinateExtractor
  refine ⟨chooseExtractor, ?_, ?_⟩
  · intro adversary adversaryEpt _eligible
    refine ⟨game.theorem10.extractor_ept adversary adversaryEpt, ?_⟩
    exact scale.le_trans (game.theorem10.lowerBound adversary adversaryEpt)
      ((game.theorem10.forkSamples adversary (chooseExtractor adversary)).monotone
        (fun sample accepted =>
          acceptedFork_extracts_correctedAmbient context laws strongSet
            adversary sample accepted))
  · intro paired pairedEpt samePhi
    have leftExtractorEpt :=
      game.theorem10.extractor_ept paired.left pairedEpt.1
    have rightExtractorEpt :=
      game.theorem10.extractor_ept paired.right pairedEpt.2
    exact scale.le_trans
      ((game.pairedForks paired (chooseExtractor paired.left)
        (chooseExtractor paired.right)).monotone
          (fun samples disagreement =>
            pairedDisagreement_implies_collisionReceipt context laws strongSet
              ops bindingLaws paired samePhi samples disagreement))
      (game.relaxedBindingBound paired (chooseExtractor paired.left)
        (chooseExtractor paired.right) pairedEpt.1 pairedEpt.2
        leftExtractorEpt rightExtractorEpt samePhi)

end Nightstream.SuperNeo.Folding.PiRLC.PaperWeakReduction
