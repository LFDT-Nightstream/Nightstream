import Nightstream.SuperNeo.Folding.PiRLC.PaperWeakReduction
import Nightstream.SuperNeo.InteractiveReduction.CoordinateForking.FiniteUniform.Experiment.Headline
import Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.Product

/-!
Finite-uniform instantiation of the paper `Pi_RLC` weak reduction.

Source: SuperNeo Definition 9, Section 7.4, Appendix C Theorem 10, and
Appendix D.5.

Owns: the exact finite challenge experiment for the operational `Pi_RLC`
verifier, the cyclic coordinate extractor with its concrete query bound, the
independent two-extractor experiment used by witness uniqueness, and the
resulting `Weak` theorem.

Does not own: the relaxed-binding assumption, field/ring/commitment laws,
Fiat--Shamir, a frozen-protocol facade, Rust, R1CS, artifacts, or costs.

Emits constraints: no.

The finite theorem proves both the sharp coordinate-fork loss
`ell / |C|` and the conservative Appendix-D.5 loss `(ell + 1) / |C|`.
The headline theorem deliberately uses the latter, matching the frozen
`Pi_RLC` budget.  Relaxed binding appears only as Definition 9's paired
witness-uniqueness error; it is not charged again as extraction error.
-/

set_option autoImplicit false

namespace Nightstream.SuperNeo.Folding.PiRLC.PaperWeakFiniteUniform

open Nightstream.SuperNeo.InteractiveReduction.Paper
open Nightstream.SuperNeo.InteractiveReduction.CoordinateForking
open Nightstream.SuperNeo.InteractiveReduction.CoordinateForking.FiniteUniform
open Nightstream.SuperNeo.InteractiveReduction.FiniteUniform
open Nightstream.SuperNeo.Folding.PiRLC.PaperCompleteness
open Nightstream.SuperNeo.Folding.PiRLC.PaperForkCollision
open Nightstream.SuperNeo.Folding.PiRLC.PaperForkExtraction
open Nightstream.SuperNeo.Folding.PiRLC.PaperWeakReduction

universe uStructure uAssignment uPublicInput uPoint uEvaluation uCommitment
  uScalar

/-- Executable finite verifier data for one paper `Pi_RLC` context.

`accepts_iff` prevents the Boolean experiment from weakening or replacing
the semantic verifier.  `adversaryExpectedPolynomialTime` is kept as an
explicit complexity predicate; extractor EPT below combines it with the
proved query bound instead of replacing complexity by `True`. -/
structure VerifierData
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    (context : Context Structure Assignment PublicInput Point Evaluation
      Commitment Scalar) where
  alphabet : Support Scalar
  alphabetValid : forall scalar,
    scalar ∈ alphabet.values -> context.algebra.challengeValid scalar
  accepts : Adversary context -> Challenge context -> Bool
  accepts_iff : forall adversary challenges,
    accepts adversary challenges = true <->
      verifies context adversary challenges (adversary.oracle challenges)
  adversaryExpectedPolynomialTime : Adversary context -> Prop

/-- The canonical finite coordinate extractor is tied to the adversary whose
oracle it rewinds. -/
structure Extractor
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    (context : Context Structure Assignment PublicInput Point Evaluation
      Commitment Scalar) where
  target : Adversary context

/-- Uniform paper challenge vectors over exactly the declared alphabet. -/
def uniformChallengeExperiment
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    {context : Context Structure Assignment PublicInput Point Evaluation
      Commitment Scalar}
    (verifier : VerifierData context) :
    Experiment (Challenge context) :=
  uniformChallenges verifier.alphabet context.arity.total

/-- The actual cyclic coordinate-fork distribution, pushed forward to the
base-and-forks sample consumed by the Appendix-D.5 extractor. -/
def forkSampleExperiment
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    {context : Context Structure Assignment PublicInput Point Evaluation
      Commitment Scalar}
    (verifier : VerifierData context)
    (adversary : Adversary context) :
    Experiment (ForkSample Scalar context.arity.total) :=
  (forkExperiment verifier.alphabet context.arity.total).map fun seed =>
    (run (verifier.accepts adversary) seed.val).sample

/-- Extractor EPT means that it targets this adversary, the adversary is EPT,
and its exact finite query trace has expectation at most `ell + 1`. -/
def ExtractorExpectedPolynomialTime
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    {context : Context Structure Assignment PublicInput Point Evaluation
      Commitment Scalar}
    (verifier : VerifierData context)
    (adversary : Adversary context)
    (extractor : Extractor context) : Prop :=
  extractor.target = adversary /\
  verifier.adversaryExpectedPolynomialTime adversary /\
  (forkExperiment verifier.alphabet context.arity.total).ExpectedQueriesAtMost
    (fun seed => (run (verifier.accepts adversary) seed.val).trace)
    (context.arity.total + 1)

/-- Exact finite query-cost predicate used by the generic Theorem-10
interface. -/
def ExpectedQueriesAtMost
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    {context : Context Structure Assignment PublicInput Point Evaluation
      Commitment Scalar}
    (verifier : VerifierData context)
    (adversary : Adversary context)
    (_extractor : Extractor context)
    (bound : Nat) : Prop :=
  (forkExperiment verifier.alphabet context.arity.total).ExpectedQueriesAtMost
    (fun seed => (run (verifier.accepts adversary) seed.val).trace) bound

private def finiteForking
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    {context : Context Structure Assignment PublicInput Point Evaluation
      Commitment Scalar}
    (verifier : VerifierData context)
    (adversary : Adversary context) :=
  finite_uniform_coordinate_forking verifier.alphabet context.arity.total
    context.algebra.challengeValid (verifies context adversary)
    adversary.oracle (verifier.accepts adversary)
    (verifier.accepts_iff adversary) verifier.alphabetValid

/-- Appendix C's generic contract, now instantiated by the concrete finite
cyclic extractor.  Its lower bound is the conservative D.5 term and its EPT
claim contains the actual query-cost certificate. -/
noncomputable def theorem10Contract
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    {context : Context Structure Assignment PublicInput Point Evaluation
      Commitment Scalar}
    (verifier : VerifierData context) :
    ForkingContract scale ratio context (Extractor := Extractor context) where
  challengeSetCardinality := verifier.alphabet.cardinality
  challengeSetCardinality_pos := verifier.alphabet.cardinality_pos
  uniformChallenges := fun _ =>
    (uniformChallengeExperiment verifier).toProbabilityExperiment
  forkSamples := fun adversary _ =>
    (forkSampleExperiment verifier adversary).toProbabilityExperiment
  coordinateExtractor := fun adversary => ⟨adversary⟩
  adversaryExpectedPolynomialTime :=
    verifier.adversaryExpectedPolynomialTime
  extractorExpectedPolynomialTime :=
    ExtractorExpectedPolynomialTime verifier
  expectedQueriesAtMost := ExpectedQueriesAtMost verifier
  extractor_ept := by
    intro adversary adversaryEpt
    exact ⟨rfl, adversaryEpt, (finiteForking verifier adversary).2.2⟩
  expectedQueries_le := by
    intro adversary _adversaryEpt
    exact (finiteForking verifier adversary).2.2
  lowerBound := by
    intro adversary _adversaryEpt
    simpa [uniformChallengeExperiment, forkSampleExperiment,
      correctedLoss, scale] using (finiteForking verifier adversary).2.1

/-- Two independent executions of the canonical coordinate extractor.  The
left seed is the outer lexicographic component and the right seed is the
inner component; neither side is derived from the other's transcript. -/
def pairedForkExperiment
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    {context : Context Structure Assignment PublicInput Point Evaluation
      Commitment Scalar}
    (verifier : VerifierData context)
    (paired : PairedAdversary context) :
    Experiment (ForkSample Scalar context.arity.total ×
      ForkSample Scalar context.arity.total) :=
  (((forkSeedSupport verifier.alphabet context.arity.total).product
      (forkSeedSupport verifier.alphabet context.arity.total)).uniform).map
    fun seeds =>
      ((run (verifier.accepts paired.left) seeds.1.val).sample,
        (run (verifier.accepts paired.right) seeds.2.val).sample)

/-- The permitted relaxed-binding assumption, stated only for the literal
collision receipt constructed from two independent accepted extractions at
the same paper projection `PiRLC.phi` (the vector of input commitments). -/
structure RelaxedBindingSecurity
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    {context : Context Structure Assignment PublicInput Point Evaluation
      Commitment Scalar}
    (laws : ExtractionAlgebra context.semantics context.params context.algebra)
    (strongSet : StrongSetUnits laws.ring context.algebra.challengeValid)
    (ops : RelaxedBindingOps Assignment Commitment Scalar)
    (verifier : VerifierData context)
    (relaxedBindingError : Rat) where
  collisionBound : forall paired,
    verifier.adversaryExpectedPolynomialTime paired.left ->
    verifier.adversaryExpectedPolynomialTime paired.right ->
    SamePhi paired ->
    (pairedForkExperiment verifier paired).probability
        (PairedForkCollisionReceipt context laws strongSet ops paired) <=
      relaxedBindingError

/-- The fully finite operational game.  No Theorem-10 lower bound or
extractor-runtime claim remains as a premise. -/
noncomputable def operationalGame
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    {context : Context Structure Assignment PublicInput Point Evaluation
      Commitment Scalar}
    (laws : ExtractionAlgebra context.semantics context.params context.algebra)
    (strongSet : StrongSetUnits laws.ring context.algebra.challengeValid)
    (ops : RelaxedBindingOps Assignment Commitment Scalar)
    (verifier : VerifierData context)
    (relaxedBindingError : Rat)
    (binding : RelaxedBindingSecurity laws strongSet ops verifier
      relaxedBindingError) :
    OperationalGame (Extractor := Extractor context) scale ratio context laws strongSet ops
      relaxedBindingError where
  theorem10 := theorem10Contract verifier
  pairedForks := fun paired _leftExtractor _rightExtractor =>
    (pairedForkExperiment verifier paired).toProbabilityExperiment
  relaxedBindingBound := by
    intro paired _leftExtractor _rightExtractor leftEpt rightEpt
      _leftExtractorEpt _rightExtractorEpt samePhi
    exact binding.collisionBound paired leftEpt rightEpt samePhi

/-- The concrete finite `WeakGame` consumed by SuperNeo's frozen weak target.
Its projection is definitionally `PiRLC.phi`, not a caller-selected summary.

The inherited eligibility predicate is total deliberately: the finite
coordinate-fork inequality is proved for every adversary, including those
below Definition 9's nonnegligible-success threshold.  This is a strict
strengthening of the paper implication, not a placeholder for an unproved
success condition. -/
noncomputable def weakGame
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    {context : Context Structure Assignment PublicInput Point Evaluation
      Commitment Scalar}
    (laws : ExtractionAlgebra context.semantics context.params context.algebra)
    (strongSet : StrongSetUnits laws.ring context.algebra.challengeValid)
    (ops : RelaxedBindingOps Assignment Commitment Scalar)
    (verifier : VerifierData context)
    (relaxedBindingError : Rat)
    (binding : RelaxedBindingSecurity laws strongSet ops verifier
      relaxedBindingError) :
    WeakGame Rat (Adversary context) (PairedAdversary context)
      (Extractor context) :=
  PaperWeakReduction.weakGame scale ratio context laws strongSet ops
    relaxedBindingError
    (operationalGame laws strongSet ops verifier relaxedBindingError binding)

/-- SuperNeo Lemma 4 / Appendix D.5 for the actual finite-uniform challenge
and cyclic coordinate-fork experiments.

The extraction loss is exactly `(ell + 1) / |C|`.  The only remaining
security premise is relaxed binding for the concrete paired collision event,
and that error occupies only Definition 9's witness-uniqueness slot.  Perfect
completeness, public coins, coordinate extraction, corrected ambient
membership, and extractor query complexity are all conclusions. -/
theorem paperWeak
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    {context : Context Structure Assignment PublicInput Point Evaluation
      Commitment Scalar}
    (laws : ExtractionAlgebra context.semantics context.params context.algebra)
    (strongSet : StrongSetUnits laws.ring context.algebra.challengeValid)
    (ops : RelaxedBindingOps Assignment Commitment Scalar)
    (bindingLaws : RelaxedBindingLaws context.semantics context.params
      context.algebra laws ops)
    (verifier : VerifierData context)
    (relaxedBindingError : Rat)
    (binding : RelaxedBindingSecurity laws strongSet ops verifier
      relaxedBindingError) :
    Weak scale
      (weakGame laws strongSet ops verifier relaxedBindingError binding)
      (ratio (context.arity.total + 1) verifier.alphabet.cardinality)
      relaxedBindingError := by
  simpa [weakGame, correctedLoss] using
    (PaperWeakReduction.paperWeak scale ratio context laws strongSet ops
      bindingLaws relaxedBindingError
      (operationalGame laws strongSet ops verifier relaxedBindingError binding))

end Nightstream.SuperNeo.Folding.PiRLC.PaperWeakFiniteUniform
