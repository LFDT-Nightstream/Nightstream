import Nightstream.SuperNeo.Folding.PiRLC.PaperWeakFiniteUniform

/-!
Finite outer-prefix lifting for the paper `Pi_RLC` weak reduction.

Source: SuperNeo Theorem 6 and Appendix D.3.

Owns: the actual `B'_2` experiment in which a fresh first-stage prefix chooses
the public `Pi_RLC` batch, followed by the finite-uniform challenge/fork
experiment; exact averaging of extraction and relaxed-binding bounds; and
independent prefix/fork sampling for the paired same-`phi` experiment.

Does not own: construction of a concrete `Pi_CCS` prefix, the `Pi_CCS` strong
game, `Pi_DEC`, commitment security, Fiat--Shamir, Rust, R1CS, or costs.

The loss is not multiplied by the number of prefixes.  Each component obeys
the same pointwise inequality and the outer experiment is its literal uniform
mixture.
-/

set_option autoImplicit false

namespace Nightstream.SuperNeo.Folding.PiRLC.PaperWeakFiniteMixture

open Nightstream.SuperNeo.InteractiveReduction.Paper
open Nightstream.SuperNeo.InteractiveReduction.FiniteUniform
open Nightstream.SuperNeo.Folding.PiRLC.PaperCompleteness
open Nightstream.SuperNeo.Folding.PiRLC.PaperForkCollision
open Nightstream.SuperNeo.Folding.PiRLC.PaperForkExtraction
open Nightstream.SuperNeo.Folding.PiRLC.PaperWeakReduction
open Nightstream.SuperNeo.Folding.PiRLC.PaperWeakFiniteUniform

universe uStructure uAssignment uPublicInput uPoint uEvaluation uCommitment
  uScalar uPrefix

/-- A D.3 weak adversary first samples a first-stage prefix, then exposes the
fixed-batch `Pi_RLC` adversary determined by that prefix. -/
structure Adversary
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    (context : Context Structure Assignment PublicInput Point Evaluation
      Commitment Scalar)
    (Prefix : Type uPrefix) where
  prefixes : Support Prefix
  /-- Whether the preceding reduction accepted this prefix.  Sequential
  composition aborts before `Pi_RLC` when this bit is false. -/
  enabled : Prefix -> Bool
  component : Prefix -> PaperWeakReduction.Adversary context

/-- The lifted extractor is the canonical coordinate extractor applied after
the sampled prefix. -/
structure Extractor
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    (context : Context Structure Assignment PublicInput Point Evaluation
      Commitment Scalar)
    (Prefix : Type uPrefix) where
  target : Adversary context Prefix

/-- Every prefix component is admitted by the fixed-batch runtime predicate.
This is the literal closure condition needed to average its security bound. -/
def AdversaryExpectedPolynomialTime
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    {Prefix : Type uPrefix}
    {context : Context Structure Assignment PublicInput Point Evaluation
      Commitment Scalar}
    (verifier : VerifierData context)
    (adversary : Adversary context Prefix) : Prop :=
  forall outer, outer ∈ adversary.prefixes.values ->
    verifier.adversaryExpectedPolynomialTime (adversary.component outer)

/-- Runtime evidence retains the exact per-prefix query bound.  This uniform
bound is stronger than merely bounding its finite outer average. -/
def ExtractorExpectedPolynomialTime
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    {Prefix : Type uPrefix}
    {context : Context Structure Assignment PublicInput Point Evaluation
      Commitment Scalar}
    (verifier : VerifierData context)
    (adversary : Adversary context Prefix)
    (extractor : Extractor context Prefix) : Prop :=
  extractor.target = adversary /\
  AdversaryExpectedPolynomialTime verifier adversary /\
  forall outer, outer ∈ adversary.prefixes.values ->
    ExpectedQueriesAtMost verifier (adversary.component outer)
      ⟨adversary.component outer⟩ (context.arity.total + 1)

/-- Prefix-tagged verifier challenges.  Tagging the outcome lets the event
read the exact component adversary without hiding the outer sample. -/
def challengeMixture
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    {Prefix : Type uPrefix}
    {context : Context Structure Assignment PublicInput Point Evaluation
      Commitment Scalar}
    (verifier : VerifierData context)
    (adversary : Adversary context Prefix) :
    Mixture Prefix (Prefix × Challenge context) where
  prefixes := adversary.prefixes
  component := fun outer =>
    (uniformChallengeExperiment verifier).map fun challenges =>
      (outer, challenges)

/-- Literal operational success after the sampled prefix. -/
def Success
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    {Prefix : Type uPrefix}
    {context : Context Structure Assignment PublicInput Point Evaluation
      Commitment Scalar}
    (adversary : Adversary context Prefix)
    (outcome : Prefix × Challenge context) : Prop :=
  adversary.enabled outcome.1 = true /\
    verifies context (adversary.component outcome.1) outcome.2
      ((adversary.component outcome.1).oracle outcome.2)

/-- Prefix-tagged coordinate-fork samples. -/
def forkMixture
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    {Prefix : Type uPrefix}
    {context : Context Structure Assignment PublicInput Point Evaluation
      Commitment Scalar}
    (verifier : VerifierData context)
    (adversary : Adversary context Prefix) :
    Mixture Prefix
      (Prefix ×
        Nightstream.SuperNeo.InteractiveReduction.CoordinateForking.ForkSample
          Scalar context.arity.total) where
  prefixes := adversary.prefixes
  component := fun outer =>
    (forkSampleExperiment verifier (adversary.component outer)).map fun sample =>
      (outer, sample)

/-- The weak extractor's actual ambient source-membership event. -/
def Extracts
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    {Prefix : Type uPrefix}
    {context : Context Structure Assignment PublicInput Point Evaluation
      Commitment Scalar}
    (laws : ExtractionAlgebra context.semantics context.params context.algebra)
    (strongSet : StrongSetUnits laws.ring context.algebra.challengeValid)
    (adversary : Adversary context Prefix)
    (outcome : Prefix ×
      Nightstream.SuperNeo.InteractiveReduction.CoordinateForking.ForkSample
        Scalar context.arity.total) : Prop :=
  adversary.enabled outcome.1 = true /\
    ExtractsCorrectedAmbient context laws strongSet
      (adversary.component outcome.1) outcome.2

/-- Two independently sampled first-stage prefixes. -/
structure PairedAdversary
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    (context : Context Structure Assignment PublicInput Point Evaluation
      Commitment Scalar)
    (Prefix : Type uPrefix) where
  left : Adversary context Prefix
  right : Adversary context Prefix

namespace PairedAdversary

/-- Fixed-batch paired adversary selected by one outer prefix pair. -/
def fixed
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    {Prefix : Type uPrefix}
    {context : Context Structure Assignment PublicInput Point Evaluation
      Commitment Scalar}
    (paired : PairedAdversary context Prefix)
    (prefixes : Prefix × Prefix) : PaperWeakReduction.PairedAdversary context where
  left := paired.left.component prefixes.1
  right := paired.right.component prefixes.2

end PairedAdversary

/-- Definition 9's shared-projection condition holds for every independently
sampled pair of prefixes. -/
def SamePhi
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    {Prefix : Type uPrefix}
    {context : Context Structure Assignment PublicInput Point Evaluation
      Commitment Scalar}
    (paired : PairedAdversary context Prefix) : Prop :=
  forall leftPrefix,
    leftPrefix ∈ paired.left.prefixes.values ->
  forall rightPrefix,
    rightPrefix ∈ paired.right.prefixes.values ->
      PaperWeakReduction.SamePhi
        (paired.fixed (leftPrefix, rightPrefix))

/-- The product support is independent in both prefix coordinates. -/
def pairedPrefixSupport
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    {Prefix : Type uPrefix}
    {context : Context Structure Assignment PublicInput Point Evaluation
      Commitment Scalar}
    (paired : PairedAdversary context Prefix) : Support (Prefix × Prefix) :=
  paired.left.prefixes.product paired.right.prefixes

/-- After the independent prefix pair, run two independent coordinate forks. -/
def pairedMixture
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    {Prefix : Type uPrefix}
    {context : Context Structure Assignment PublicInput Point Evaluation
      Commitment Scalar}
    (verifier : VerifierData context)
    (paired : PairedAdversary context Prefix) :
    Mixture (Prefix × Prefix)
      ((Prefix × Prefix) ×
        (Nightstream.SuperNeo.InteractiveReduction.CoordinateForking.ForkSample
            Scalar context.arity.total ×
          Nightstream.SuperNeo.InteractiveReduction.CoordinateForking.ForkSample
            Scalar context.arity.total)) where
  prefixes := pairedPrefixSupport paired
  component := fun prefixes =>
    (pairedForkExperiment verifier (paired.fixed prefixes)).map fun samples =>
      (prefixes, samples)

/-- Literal disagreement of the two assignment families extracted after the
two independently sampled prefixes. -/
def PairedDisagreement
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    {Prefix : Type uPrefix}
    {context : Context Structure Assignment PublicInput Point Evaluation
      Commitment Scalar}
    (laws : ExtractionAlgebra context.semantics context.params context.algebra)
    (strongSet : StrongSetUnits laws.ring context.algebra.challengeValid)
    (paired : PairedAdversary context Prefix)
    (outcome : (Prefix × Prefix) ×
      (Nightstream.SuperNeo.InteractiveReduction.CoordinateForking.ForkSample
          Scalar context.arity.total ×
        Nightstream.SuperNeo.InteractiveReduction.CoordinateForking.ForkSample
          Scalar context.arity.total)) : Prop :=
  paired.left.enabled outcome.1.1 = true /\
    paired.right.enabled outcome.1.2 = true /\
      PairedWitnessDisagreement context laws strongSet
        (paired.fixed outcome.1) outcome.2

/-- One fixed-prefix extraction inequality, named separately so the outer
mixture proof never asks elaboration to normalize the whole weak game. -/
private theorem componentExtractionBound
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
    (verifier : VerifierData context)
    (adversary : PaperWeakReduction.Adversary context)
    (adversaryEpt : verifier.adversaryExpectedPolynomialTime adversary) :
    (uniformChallengeExperiment verifier).probability
          (fun challenges => verifies context adversary challenges
            (adversary.oracle challenges)) -
        ratio (context.arity.total + 1) verifier.alphabet.cardinality <=
      (forkSampleExperiment verifier adversary).probability
        (ExtractsCorrectedAmbient context laws strongSet adversary) := by
  have lower := (theorem10Contract verifier).lowerBound adversary adversaryEpt
  have extractionMonotone :=
    (forkSampleExperiment verifier adversary).probability_mono
      (fun sample accepted =>
        acceptedFork_extracts_correctedAmbient context laws strongSet
          adversary sample accepted)
  exact scale.le_trans lower extractionMonotone

/-- A rejected preceding prefix contributes no sequential-composition mass.
An accepted prefix reduces to the ordinary fixed-batch `Pi_RLC` bound. -/
private theorem gatedComponentExtractionBound
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
    (verifier : VerifierData context)
    (adversary : PaperWeakReduction.Adversary context)
    (enabled : Bool)
    (adversaryEpt : verifier.adversaryExpectedPolynomialTime adversary) :
    (uniformChallengeExperiment verifier).probability
          (fun challenges => enabled = true /\
            verifies context adversary challenges
              (adversary.oracle challenges)) -
        ratio (context.arity.total + 1) verifier.alphabet.cardinality <=
      (forkSampleExperiment verifier adversary).probability
        (fun sample => enabled = true /\
          ExtractsCorrectedAmbient context laws strongSet adversary sample) := by
  cases enabled with
  | true =>
      simpa using
        componentExtractionBound laws strongSet verifier adversary adversaryEpt
  | false =>
      have successZero :
          (uniformChallengeExperiment verifier).probability
              (fun challenges => false = true /\
                verifies context adversary challenges
                  (adversary.oracle challenges)) = 0 := by
        simpa using
          (uniformChallengeExperiment verifier).probability_false
      have extractionZero :
          (forkSampleExperiment verifier adversary).probability
              (fun sample => false = true /\
                ExtractsCorrectedAmbient context laws strongSet adversary
                  sample) = 0 := by
        simpa using
          (forkSampleExperiment verifier adversary).probability_false
      rw [successZero, extractionZero, Rat.sub_eq_add_neg, Rat.zero_add]
      have lossNonnegative :
          0 <= ratio (context.arity.total + 1)
            verifier.alphabet.cardinality := by
        unfold ratio
        rw [Rat.div_def]
        exact Rat.mul_nonneg Rat.natCast_nonneg
          (Rat.le_of_lt (Rat.inv_pos.mpr
            (Rat.natCast_pos.mpr verifier.alphabet.cardinality_pos)))
      simpa using Rat.neg_le_neg lossNonnegative

/-- One fixed prefix pair obeys the same relaxed-binding bound. -/
private theorem componentDisagreementBound
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
      relaxedBindingError)
    (paired : PaperWeakReduction.PairedAdversary context)
    (leftEpt : verifier.adversaryExpectedPolynomialTime paired.left)
    (rightEpt : verifier.adversaryExpectedPolynomialTime paired.right)
    (samePhi : PaperWeakReduction.SamePhi paired) :
    (pairedForkExperiment verifier paired).probability
        (PairedWitnessDisagreement context laws strongSet paired) <=
      relaxedBindingError := by
  have collisionMonotone :=
    (pairedForkExperiment verifier paired).probability_mono
      (fun samples disagreement =>
        pairedDisagreement_implies_collisionReceipt context laws strongSet ops
          bindingLaws paired samePhi samples disagreement)
  exact scale.le_trans collisionMonotone
    (binding.collisionBound paired leftEpt rightEpt samePhi)

/-- Prefix rejection can only remove paired disagreement outcomes. -/
private theorem gatedComponentDisagreementBound
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
      relaxedBindingError)
    (paired : PaperWeakReduction.PairedAdversary context)
    (leftEnabled rightEnabled : Bool)
    (leftEpt : verifier.adversaryExpectedPolynomialTime paired.left)
    (rightEpt : verifier.adversaryExpectedPolynomialTime paired.right)
    (samePhi : PaperWeakReduction.SamePhi paired) :
    (pairedForkExperiment verifier paired).probability
          (fun samples => leftEnabled = true /\ rightEnabled = true /\
            PairedWitnessDisagreement context laws strongSet paired samples) <=
        relaxedBindingError := by
  apply scale.le_trans
    ((pairedForkExperiment verifier paired).probability_mono
      (fun _ disagreement => disagreement.2.2))
  exact componentDisagreementBound laws strongSet ops bindingLaws verifier
    relaxedBindingError binding paired leftEpt rightEpt samePhi

/-- The finite weak game after averaging over the first-stage prefix. -/
noncomputable def weakGame
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    {Prefix : Type uPrefix}
    {context : Context Structure Assignment PublicInput Point Evaluation
      Commitment Scalar}
    (laws : ExtractionAlgebra context.semantics context.params context.algebra)
    (strongSet : StrongSetUnits laws.ring context.algebra.challengeValid)
    (verifier : VerifierData context) :
    WeakGame Rat (Adversary context Prefix) (PairedAdversary context Prefix)
      (Extractor context Prefix) where
  perfectComplete := PerfectComplete context
  publicCoin := PublicCoin context
  adversaryExpectedPolynomialTime :=
    AdversaryExpectedPolynomialTime verifier
  pairedAdversaryExpectedPolynomialTime := fun paired =>
    AdversaryExpectedPolynomialTime verifier paired.left /\
      AdversaryExpectedPolynomialTime verifier paired.right
  extractorExpectedPolynomialTime :=
    ExtractorExpectedPolynomialTime verifier
  extractionEligible := fun _ => True
  adversarySuccess := fun adversary =>
    (challengeMixture verifier adversary).probability (Success adversary)
  ambientSourceWitnessExtracted := fun adversary _ =>
    (forkMixture verifier adversary).probability
      (Extracts laws strongSet adversary)
  left := PairedAdversary.left
  right := PairedAdversary.right
  samePhiInputsAlways := SamePhi
  pairedWitnessDisagreement := fun paired _ _ =>
    (pairedMixture verifier paired).probability
      (PairedDisagreement laws strongSet paired)

/-- Appendix D.3's outer-prefix adversary preserves the finite-uniform
`Pi_RLC` weak guarantee exactly. -/
theorem paperWeak
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    {Prefix : Type uPrefix}
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
    Weak scale (weakGame (Prefix := Prefix) laws strongSet verifier)
      (ratio (context.arity.total + 1) verifier.alphabet.cardinality)
      relaxedBindingError := by
  refine ⟨PaperCompleteness.perfectComplete context,
    PaperCompleteness.publicCoin context, ?_⟩
  let chooseExtractor : Adversary context Prefix -> Extractor context Prefix :=
    fun adversary => ⟨adversary⟩
  refine ⟨chooseExtractor, ?_, ?_⟩
  · intro adversary adversaryEpt _eligible
    refine ⟨?_, ?_⟩
    · refine ⟨rfl, adversaryEpt, ?_⟩
      intro outer member
      exact (theorem10Contract verifier).expectedQueries_le
        (adversary.component outer) (adversaryEpt outer member)
    · change
        (challengeMixture verifier adversary).probability (Success adversary) -
              ratio (context.arity.total + 1)
                verifier.alphabet.cardinality <=
          (forkMixture verifier adversary).probability
            (Extracts laws strongSet adversary)
      exact Mixture.loss_le_of_component_pairs
        (challengeMixture verifier adversary)
        (forkMixture verifier adversary) rfl
        (Success adversary) (Extracts laws strongSet adversary)
        (ratio (context.arity.total + 1) verifier.alphabet.cardinality)
        (fun outer member => by
          change
            (uniformChallengeExperiment verifier).probability
                  (fun challenges => adversary.enabled outer = true /\
                    verifies context (adversary.component outer) challenges
                      ((adversary.component outer).oracle challenges)) -
                ratio (context.arity.total + 1)
                  verifier.alphabet.cardinality <=
              (forkSampleExperiment verifier
                  (adversary.component outer)).probability
                (fun sample => adversary.enabled outer = true /\
                  ExtractsCorrectedAmbient context laws strongSet
                    (adversary.component outer) sample)
          exact gatedComponentExtractionBound laws strongSet verifier
            (adversary.component outer) (adversary.enabled outer)
            (adversaryEpt outer member))
  · intro paired pairedEpt samePhi
    change
      (pairedMixture verifier paired).probability
          (PairedDisagreement laws strongSet paired) <= relaxedBindingError
    exact Mixture.probability_le_of_components
      (pairedMixture verifier paired)
      (PairedDisagreement laws strongSet paired) relaxedBindingError
      (fun prefixes member => by
        change prefixes ∈
          (paired.left.prefixes.product paired.right.prefixes).values at member
        have prefixMembers :=
          (Support.mem_product_iff paired.left.prefixes
            paired.right.prefixes prefixes).mp member
        let fixed := paired.fixed prefixes
        have leftEpt : verifier.adversaryExpectedPolynomialTime fixed.left :=
          pairedEpt.1 prefixes.1 prefixMembers.1
        have rightEpt : verifier.adversaryExpectedPolynomialTime fixed.right :=
          pairedEpt.2 prefixes.2 prefixMembers.2
        have fixedSamePhi : PaperWeakReduction.SamePhi fixed :=
          samePhi prefixes.1 prefixMembers.1 prefixes.2 prefixMembers.2
        change
          (pairedForkExperiment verifier fixed).probability
              (fun samples => paired.left.enabled prefixes.1 = true /\
                paired.right.enabled prefixes.2 = true /\
                  PairedWitnessDisagreement context laws strongSet fixed
                    samples) <= relaxedBindingError
        exact gatedComponentDisagreementBound laws strongSet ops bindingLaws
          verifier relaxedBindingError binding fixed
          (paired.left.enabled prefixes.1) (paired.right.enabled prefixes.2)
          leftEpt rightEpt fixedSamePhi)

end Nightstream.SuperNeo.Folding.PiRLC.PaperWeakFiniteMixture
