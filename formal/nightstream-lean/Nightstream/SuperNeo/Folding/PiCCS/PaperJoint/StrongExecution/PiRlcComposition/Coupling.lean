import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.PiRlcComposition.Repeated

/-!
Concrete strong--weak coupling for paper `Pi_CCS` followed by paper `Pi_RLC`.

Owns: assembly of the operational one-run and two-run identities into the
generic SuperNeo Appendix-D.3 coupling interface.

Does not own: either component reduction theorem, `Pi_DEC`, asymptotic
sampling, Fiat--Shamir, Rust, R1CS, artifacts, or costs.
-/

set_option autoImplicit false

namespace Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.PiRlcComposition.Coupling

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongReduction
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.PiRlcBatch
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.PiRlcComposition
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.PiRlcComposition.Repeated
open Nightstream.SuperNeo.Folding.PiRLC.PaperCompleteness
open Nightstream.SuperNeo.Folding.PiRLC.PaperForkExtraction
open Nightstream.SuperNeo.InteractiveReduction.CoordinateForking.FiniteUniform
open Nightstream.SuperNeo.InteractiveReduction.FiniteUniform
open Nightstream.SuperNeo.InteractiveReduction.Paper
open MatrixCoefficientSource
open PaperLinearAlgebra

universe uExtension uCommitment uPublicInput uScalar uProverSeed uProverTape

/-- The Appendix-D.3 coupling contains no probability assumptions: both
probability fields are the exact finite equalities proved by the operational
leaves. -/
noncomputable def operationalCoupling
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    [DecidableEq Extension]
    {shape : Shape}
    {columns blockCount : Nat}
    {ProverSeed : Type uProverSeed}
    {ProverTape : Type uProverTape}
    (context : CompatibleContext Extension Commitment PublicInput Scalar
      shape columns blockCount)
    (laws : ExtractionAlgebra context.piRlc.semantics context.piRlc.params
      context.piRlc.algebra)
    (strongSet : StrongSetUnits laws.ring
      context.piRlc.algebra.challengeValid)
    (extensionAlphabet : Support Extension)
    (verifier :
      Nightstream.SuperNeo.Folding.PiRLC.PaperWeakFiniteUniform.VerifierData
        context.piRlc)
    (strongAdversaryExpectedPolynomialTime :
      OperationalExperiment.Adversary context.piCcs ProverSeed
        (ForkSeed verifier.alphabet context.arity.total) ProverTape -> Prop)
    (successFloor : Rat) :
    Nightstream.SuperNeo.InteractiveReduction.StrongWeakComposition.Coupling
      Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.scale
      (FinitePaperStrong.finiteStrongGame context.piCcs extensionAlphabet
        strongAdversaryExpectedPolynomialTime successFloor)
      (Nightstream.SuperNeo.Folding.PiRLC.PaperWeakFiniteMixture.weakGame
        (Prefix := PrefixSeed Extension shape ProverSeed)
        laws strongSet verifier)
      (Adversary context ProverSeed ProverTape) where
  toWeak := fun adversary => toWeak context extensionAlphabet adversary
  toStrong := fun adversary _extractor =>
    toStrong context laws strongSet verifier adversary
  paired := fun adversary _extractor =>
    pairedWeak context extensionAlphabet adversary
  pairedLeft := by
    intro adversary _extractor
    rfl
  pairedRight := by
    intro adversary _extractor
    rfl
  pairedExpectedPolynomialTime := by
    intro adversary _extractor adversaryExpected _extractorExpected
    exact ⟨adversaryExpected, adversaryExpected⟩
  pairedSamePhi := by
    intro adversary _extractor leftPrefix _leftMember rightPrefix _rightMember
    simpa [Nightstream.SuperNeo.Folding.PiRLC.PaperWeakFiniteMixture.SamePhi,
      Nightstream.SuperNeo.Folding.PiRLC.PaperWeakFiniteMixture.PairedAdversary.fixed,
      pairedWeak, toWeak, component] using
      context.repeatedBatch_samePhi
        (prefixExecution context adversary leftPrefix)
        (prefixExecution context adversary rightPrefix)
  intermediateProbability := by
    intro adversary _extractor
    simpa [FinitePaperStrong.finiteStrongGame,
      Nightstream.SuperNeo.Folding.PiRLC.PaperWeakFiniteMixture.weakGame] using
      intermediateProbability context laws strongSet extensionAlphabet verifier
        adversary
  repeatedWitnessProbability := by
    intro adversary _extractor
    simpa [FinitePaperStrong.finiteStrongGame,
      Nightstream.SuperNeo.Folding.PiRLC.PaperWeakFiniteMixture.weakGame] using
      repeatedWitnessProbability context laws strongSet extensionAlphabet
        verifier adversary

/-- Ordinary rational subtraction satisfies the two arithmetic laws used by
strong--weak composition. -/
def rationalScaleLaws :
    Nightstream.SuperNeo.InteractiveReduction.StrongWeakComposition.ScaleLaws
      Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.scale where
  subtract_mono_left := by
    intro left right error ordered
    change left - error <= right - error
    simpa [Rat.sub_eq_add_neg] using
      (Rat.add_le_add_right (c := -error)).mpr ordered
  subtract_subtract := by
    intro probability first second
    change (probability - first) - second = probability - (first + second)
    simp [Rat.sub_eq_add_neg, Rat.neg_add, Rat.add_assoc]

/-- Finite operational SuperNeo Theorem 6 for the concrete `Pi_CCS` to
`Pi_RLC` composition.  The exact loss is the coordinate-fork term, the two
named `Pi_CCS` intrinsic budgets, and the relaxed-binding disagreement budget
conditioned once by the strong-game success floor. -/
theorem finiteReductionOfKnowledge
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    [DecidableEq Extension]
    {shape : Shape}
    {columns blockCount : Nat}
    {ProverSeed : Type uProverSeed}
    {ProverTape : Type uProverTape}
    (context : CompatibleContext Extension Commitment PublicInput Scalar
      shape columns blockCount)
    (laws : ExtractionAlgebra context.piRlc.semantics context.piRlc.params
      context.piRlc.algebra)
    (strongSet : StrongSetUnits laws.ring
      context.piRlc.algebra.challengeValid)
    (extensionAlphabet : Support Extension)
    (verifier :
      Nightstream.SuperNeo.Folding.PiRLC.PaperWeakFiniteUniform.VerifierData
        context.piRlc)
    (strongAdversaryExpectedPolynomialTime :
      OperationalExperiment.Adversary context.piCcs ProverSeed
        (ForkSeed verifier.alphabet context.arity.total) ProverTape -> Prop)
    (successFloor relaxedBindingError mixingBudget sumCheckBudget : Rat)
    (ops : PiRLC.RelaxedBindingOps (Assignment F columns) Commitment Scalar)
    (bindingLaws :
      Nightstream.SuperNeo.Folding.PiRLC.PaperForkCollision.RelaxedBindingLaws
        context.piRlc.semantics context.piRlc.params context.piRlc.algebra laws
        ops)
    (binding :
      Nightstream.SuperNeo.Folding.PiRLC.PaperWeakFiniteUniform.RelaxedBindingSecurity
        laws strongSet ops verifier relaxedBindingError)
    (ambientAdmissible : context.piCcs.params.b <=
      Nightstream.SuperNeo.Folding.PiRLC.PaperCorrections.correctedAmbientBoundFor
        context.piCcs.params)
    (contracts : FinitePaperStrong.NamedSecurityContracts context.piCcs
      extensionAlphabet strongAdversaryExpectedPolynomialTime mixingBudget
      sumCheckBudget) :
    ReductionOfKnowledge
      Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.scale
      (Nightstream.SuperNeo.InteractiveReduction.StrongWeakComposition.knowledgeGame
        Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.scale
        (FinitePaperStrong.finiteStrongGame context.piCcs extensionAlphabet
          strongAdversaryExpectedPolynomialTime successFloor)
        (Nightstream.SuperNeo.Folding.PiRLC.PaperWeakFiniteMixture.weakGame
          (Prefix := PrefixSeed Extension shape ProverSeed)
          laws strongSet verifier)
        (operationalCoupling context laws strongSet extensionAlphabet verifier
          strongAdversaryExpectedPolynomialTime successFloor))
      (ratio (context.arity.total + 1) verifier.alphabet.cardinality +
        ((mixingBudget + sumCheckBudget) +
          relaxedBindingError / successFloor)) := by
  exact
    Nightstream.SuperNeo.InteractiveReduction.StrongWeakComposition.reductionOfKnowledge
      Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.scale
      rationalScaleLaws
      (FinitePaperStrong.finiteStrongGame context.piCcs extensionAlphabet
        strongAdversaryExpectedPolynomialTime successFloor)
      (Nightstream.SuperNeo.Folding.PiRLC.PaperWeakFiniteMixture.weakGame
        (Prefix := PrefixSeed Extension shape ProverSeed)
        laws strongSet verifier)
      (operationalCoupling context laws strongSet extensionAlphabet verifier
        strongAdversaryExpectedPolynomialTime successFloor)
      (fun raw floor => raw / floor)
      successFloor
      (mixingBudget + sumCheckBudget)
      (ratio (context.arity.total + 1) verifier.alphabet.cardinality)
      relaxedBindingError
      (FinitePaperStrong.finitePaperStrong context.piCcs extensionAlphabet
        strongAdversaryExpectedPolynomialTime successFloor relaxedBindingError
        mixingBudget sumCheckBudget ambientAdmissible contracts)
      (Nightstream.SuperNeo.Folding.PiRLC.PaperWeakFiniteMixture.paperWeak
        laws strongSet ops bindingLaws verifier relaxedBindingError binding)

end Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.PiRlcComposition.Coupling
