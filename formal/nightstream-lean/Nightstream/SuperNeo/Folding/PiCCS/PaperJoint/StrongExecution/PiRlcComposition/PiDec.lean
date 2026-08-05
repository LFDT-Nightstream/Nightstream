import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.PiRlcComposition.Coupling
import Nightstream.SuperNeo.Folding.PiDEC.PaperReduction
import Nightstream.SuperNeo.InteractiveReduction.KnowledgeComposition

/-!
Operational composition of paper `Pi_CCS`, `Pi_RLC`, and `Pi_DEC`.

Source: SuperNeo Sections 7.3--7.5, Theorems 6--7, and Appendices D.3--D.6.

Owns: the exact verifier-computed `Pi_RLC` parent passed to `Pi_DEC`, the
adversarial child messages and child witnesses, abort after a rejected
`Pi_CCS` prefix, reverse-order straight-line extraction, and the final finite
knowledge theorem with `Pi_DEC`'s zero loss.

Does not own: HyperNova/NIFS, Fiat--Shamir, commitment internals, Rust, R1CS,
artifacts, minimality, or costs.

Emits constraints: no.

Authority boundary: the adversary cannot supply the `Pi_RLC` parent or its
opening.  It supplies only the `Pi_DEC` child messages and child assignments.
The verifier constructs the parent from the exact `K+k` batch and challenge
vector; the `Pi_DEC` extractor recomposes the child assignments, and that
recomposition is the only assignment passed backward to `Pi_RLC`.
-/

set_option autoImplicit false

namespace Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.PiRlcComposition.PiDec

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongReduction
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.PiRlcBatch
open Nightstream.SuperNeo.Folding.PiRLC.PaperForkExtraction
open Nightstream.SuperNeo.InteractiveReduction.CoordinateForking.FiniteUniform
open Nightstream.SuperNeo.InteractiveReduction.FiniteUniform
open Nightstream.SuperNeo.InteractiveReduction.Paper
open MatrixCoefficientSource
open PaperLinearAlgebra

universe uExtension uCommitment uPublicInput uScalar uProverSeed uProverTape
  uStructure uAssignment uPoint uEvaluation uWeight

/-- `Pi_DEC` operations over exactly the semantics and parameters shared by
the preceding `Pi_CCS`/`Pi_RLC` context.  No equality proof transports a
separately chosen relation into the composition. -/
structure CompatiblePiDecContext
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {shape : Shape}
    {columns blockCount : Nat}
    (context : CompatibleContext Extension Commitment PublicInput Scalar
      shape columns blockCount) where
  algebra : PiDEC.Algebra
    (MatrixSource F shape columns blockCount)
    (Assignment F columns)
    PublicInput
    (CubePoint Extension shape.cubeVariables)
    (EvaluationFamily Extension shape)
    Commitment
    context.piRlc.semantics
    context.piRlc.params
  publicSplit : PiDEC.PaperVerifier.PublicInputSplit algebra
  evaluationArity : PiDEC.PaperVerifier.EvaluationArity
    context.piRlc.semantics
  kPositive : 0 < context.piRlc.params.k

namespace CompatiblePiDecContext

/-- The exact paper-`Pi_DEC` context selected by compatibility above. -/
def paper
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {shape : Shape}
    {columns blockCount : Nat}
    {context : CompatibleContext Extension Commitment PublicInput Scalar
      shape columns blockCount}
    (piDec : CompatiblePiDecContext context) :
    PiDEC.PaperReduction.Context
      (MatrixSource F shape columns blockCount)
      (Assignment F columns)
      PublicInput
      (CubePoint Extension shape.cubeVariables)
      (EvaluationFamily Extension shape)
      Commitment where
  semantics := context.piRlc.semantics
  params := context.piRlc.params
  algebra := piDec.algebra
  publicSplit := piDec.publicSplit
  evaluationArity := piDec.evaluationArity
  kPositive := piDec.kPositive

end CompatiblePiDecContext

/-- The only second-stage data supplied by the adversary.  The parent is
intentionally absent. -/
structure Reply
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {shape : Shape}
    {columns blockCount : Nat}
    (context : CompatibleContext Extension Commitment PublicInput Scalar
      shape columns blockCount) where
  messages : Fin context.piRlc.params.k ->
    PiDEC.PaperVerifier.ChildMessage
      (EvaluationFamily Extension shape) Commitment
  childAssignments : Fin context.piRlc.params.k -> Assignment F columns

/-- One sequential adversary.  Its first-stage strategy is causal, while its
second-stage reply receives the completed prefix and verifier challenge. -/
structure Adversary
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {shape : Shape}
    {columns blockCount : Nat}
    (context : CompatibleContext Extension Commitment PublicInput Scalar
      shape columns blockCount)
    (ProverSeed : Type uProverSeed)
    (ProverTape : Type uProverTape) where
  proverSupport : Support ProverSeed
  strategy : Strategy Extension shape ProverTape
  proverTape : ProverSeed -> ProverTape
  reply : PrefixExecution Extension shape ->
    Nightstream.SuperNeo.Folding.PiRLC.PaperWeakReduction.Challenge
      context.piRlc -> Reply context

/-- Reverse-order extraction turns the `Pi_DEC` child witnesses into the sole
assignment oracle exposed to `Pi_RLC`. -/
def toPiRlc
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {shape : Shape}
    {columns blockCount : Nat}
    {ProverSeed : Type uProverSeed}
    {ProverTape : Type uProverTape}
    (context : CompatibleContext Extension Commitment PublicInput Scalar
      shape columns blockCount)
    (piDec : CompatiblePiDecContext context)
    (adversary : Adversary context ProverSeed ProverTape) :
    PiRlcComposition.Adversary context ProverSeed ProverTape where
  proverSupport := adversary.proverSupport
  strategy := adversary.strategy
  proverTape := adversary.proverTape
  oracle := fun causalRun challenges =>
    piDec.algebra.recomposeAssignment
      (adversary.reply causalRun challenges).childAssignments

/-- The public `CE(B)` parent passed into `Pi_DEC` is computed by the
verifier's exact `Pi_RLC` combination. -/
def combinedParent
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {shape : Shape}
    {columns blockCount : Nat}
    (context : CompatibleContext Extension Commitment PublicInput Scalar
      shape columns blockCount)
    (causalRun : PrefixExecution Extension shape)
    (challenges :
      Nightstream.SuperNeo.Folding.PiRLC.PaperWeakReduction.Challenge
        context.piRlc) :
    CE.Instance
      (MatrixSource F shape columns blockCount)
      PublicInput
      (CubePoint Extension shape.cubeVariables)
      (EvaluationFamily Extension shape)
      Commitment :=
  PiRLC.combinedOutput context.piRlc.algebra
    (context.batchOfPrefix causalRun).system
    (context.batchOfPrefix causalRun).point
    (context.batchOfPrefix causalRun).inputs challenges

/-- The exact operational `Pi_DEC` execution for one accepted-prefix
challenge.  Only child messages and child witnesses come from the adversary. -/
def piDecExecution
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {shape : Shape}
    {columns blockCount : Nat}
    {ProverSeed : Type uProverSeed}
    {ProverTape : Type uProverTape}
    (context : CompatibleContext Extension Commitment PublicInput Scalar
      shape columns blockCount)
    (piDec : CompatiblePiDecContext context)
    (adversary : Adversary context ProverSeed ProverTape)
    (causalRun : PrefixExecution Extension shape)
    (challenges :
      Nightstream.SuperNeo.Folding.PiRLC.PaperWeakReduction.Challenge
        context.piRlc) :
    PiDEC.PaperReduction.Execution piDec.paper :=
  let reply := adversary.reply causalRun challenges
  {
    attempt := {
      parent := combinedParent context causalRun challenges
      messages := reply.messages
    }
    childAssignments := reply.childAssignments
  }

/-- A prior rejection is represented by `none`; no arbitrary invalid
`Pi_DEC` execution is invented to absorb rejected probability mass. -/
def optionalPiDecExecution
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
    (piDec : CompatiblePiDecContext context)
    (adversary : Adversary context ProverSeed ProverTape)
    (outcome : PrefixSeed Extension shape ProverSeed ×
      Nightstream.SuperNeo.Folding.PiRLC.PaperWeakReduction.Challenge
        context.piRlc) :
    Option (PiDEC.PaperReduction.Execution piDec.paper) :=
  let causalRun := PiRlcComposition.prefixExecution context
    (toPiRlc context piDec adversary) outcome.1
  if acceptedCheck context.piCcs causalRun = true then
    some (piDecExecution context piDec adversary causalRun outcome.2)
  else
    none

/-- Exact finite second-stage distribution, with prior rejection preserved as
an explicit abort outcome. -/
def piDecMixture
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
    (piDec : CompatiblePiDecContext context)
    (extensionAlphabet : Support Extension)
    (verifier :
      Nightstream.SuperNeo.Folding.PiRLC.PaperWeakFiniteUniform.VerifierData
        context.piRlc)
    (adversary : Adversary context ProverSeed ProverTape) :
    Mixture (PrefixSeed Extension shape ProverSeed)
      (Option (PiDEC.PaperReduction.Execution piDec.paper)) :=
  (Nightstream.SuperNeo.Folding.PiRLC.PaperWeakFiniteMixture.challengeMixture
    verifier
    (PiRlcComposition.toWeak context extensionAlphabet
      (toPiRlc context piDec adversary))).map
    (optionalPiDecExecution context piDec adversary)

/-- Success in the abort-aware final stage. -/
def AbortingSuccess
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    (context : PiDEC.PaperReduction.Context Structure Assignment PublicInput
      Point Evaluation Commitment) :
    Option (PiDEC.PaperReduction.Execution context) -> Prop
  | none => False
  | some execution => PiDEC.PaperReduction.Success context execution

/-- Parent-opening extraction in the abort-aware final stage. -/
def AbortingExtractedSource
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    (context : PiDEC.PaperReduction.Context Structure Assignment PublicInput
      Point Evaluation Commitment) :
    Option (PiDEC.PaperReduction.Execution context) -> Prop
  | none => False
  | some execution => PiDEC.PaperReduction.ExtractedSource context execution

/-- Exact `Pi_DEC` knowledge game lifted through sequential abort. -/
def abortingKnowledgeGame
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Weight : Type uWeight}
    (context : PiDEC.PaperReduction.Context Structure Assignment PublicInput
      Point Evaluation Commitment)
    (scale : ProbabilityScale Weight) :
    KnowledgeGame Weight
      (ProbabilityExperiment scale
        (Option (PiDEC.PaperReduction.Execution context)))
      PiDEC.PaperReduction.Extractor where
  perfectComplete := PiDEC.PaperReduction.PerfectComplete context
  publicCoin := True
  adversaryExpectedPolynomialTime := fun _ => True
  extractorExpectedPolynomialTime := fun _ extractor =>
    extractor = .straightLine
  extractionEligible := fun _ => True
  adversarySuccess := fun experiment =>
    experiment.probability (AbortingSuccess context)
  sourceWitnessExtracted := fun experiment _ =>
    experiment.probability (AbortingExtractedSource context)

/-- The paper's pointwise zero-loss extractor remains zero-loss when an
earlier stage may abort. -/
theorem abortingReductionOfKnowledge
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Weight : Type uWeight}
    (context : PiDEC.PaperReduction.Context Structure Assignment PublicInput
      Point Evaluation Commitment)
    (scale : ProbabilityScale Weight) :
    ReductionOfKnowledge scale (abortingKnowledgeGame context scale)
      scale.zero := by
  refine ⟨PiDEC.PaperReduction.perfectComplete context, True.intro, ?_⟩
  intro experiment _ _
  refine ⟨PiDEC.PaperReduction.Extractor.straightLine, rfl, ?_⟩
  rw [scale.subtract_zero]
  exact experiment.monotone fun outcome success => by
    cases outcome with
    | none => exact False.elim success
    | some execution =>
        exact PiDEC.PaperReduction.success_implies_extractedSource context
          execution success

/-- Pointwise intermediate identity: `Pi_RLC` succeeds on the assignment
recomposed by the `Pi_DEC` extractor exactly when that extractor opens the
verifier-computed parent. -/
theorem weakSuccess_iff_abortingExtractedSource
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
    (piDec : CompatiblePiDecContext context)
    (extensionAlphabet : Support Extension)
    (adversary : Adversary context ProverSeed ProverTape)
    (outcome : PrefixSeed Extension shape ProverSeed ×
      Nightstream.SuperNeo.Folding.PiRLC.PaperWeakReduction.Challenge
        context.piRlc) :
    Nightstream.SuperNeo.Folding.PiRLC.PaperWeakFiniteMixture.Success
        (PiRlcComposition.toWeak context extensionAlphabet
          (toPiRlc context piDec adversary)) outcome <->
      AbortingExtractedSource piDec.paper
        (optionalPiDecExecution context piDec adversary outcome) := by
  by_cases accepted : acceptedCheck context.piCcs
      (PiRlcComposition.prefixExecution context
        (toPiRlc context piDec adversary) outcome.1) = true
  · simp [toPiRlc] at accepted
    simp [Nightstream.SuperNeo.Folding.PiRLC.PaperWeakFiniteMixture.Success,
      PiRlcComposition.toWeak, PiRlcComposition.component,
      Nightstream.SuperNeo.Folding.PiRLC.PaperWeakReduction.verifies,
      Nightstream.SuperNeo.Folding.PiRLC.PaperWeakReduction.response,
      PiRLC.PaperForkExtraction.Response.Success,
      PiRLC.PaperForkExtraction.Response.output,
      optionalPiDecExecution, accepted,
      AbortingExtractedSource, PiDEC.PaperReduction.ExtractedSource,
      piDecExecution, combinedParent, toPiRlc,
      CompatiblePiDecContext.paper]
    rfl
  · simp [toPiRlc] at accepted
    simp [Nightstream.SuperNeo.Folding.PiRLC.PaperWeakFiniteMixture.Success,
      PiRlcComposition.toWeak, optionalPiDecExecution, accepted, toPiRlc,
      AbortingExtractedSource]

/-- Exact equality of the intermediate probabilities, derived from the
pointwise verifier/extractor identity over the same finite distribution. -/
theorem intermediateProbability
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
    (piDec : CompatiblePiDecContext context)
    (extensionAlphabet : Support Extension)
    (verifier :
      Nightstream.SuperNeo.Folding.PiRLC.PaperWeakFiniteUniform.VerifierData
        context.piRlc)
    (adversary : Adversary context ProverSeed ProverTape) :
    (Nightstream.SuperNeo.Folding.PiRLC.PaperWeakFiniteMixture.challengeMixture
      verifier
      (PiRlcComposition.toWeak context extensionAlphabet
        (toPiRlc context piDec adversary))).probability
        (Nightstream.SuperNeo.Folding.PiRLC.PaperWeakFiniteMixture.Success
          (PiRlcComposition.toWeak context extensionAlphabet
            (toPiRlc context piDec adversary))) =
      (piDecMixture context piDec extensionAlphabet verifier adversary).probability
        (AbortingExtractedSource piDec.paper) := by
  unfold piDecMixture
  rw [Mixture.map_probability]
  congr 1
  funext outcome
  exact propext (weakSuccess_iff_abortingExtractedSource context piDec
    extensionAlphabet adversary outcome)

/-- Exact coupling used for reverse-order `Pi_DEC` then `Pi_RLC`/`Pi_CCS`
extraction.  Its probability field is a theorem above, not caller evidence. -/
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
    (piDec : CompatiblePiDecContext context)
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
        (ForkSeed verifier.alphabet context.arity.total) ProverTape -> Prop) :
    Nightstream.SuperNeo.InteractiveReduction.KnowledgeComposition.Coupling
      scale
      (Nightstream.SuperNeo.InteractiveReduction.StrongWeakComposition.knowledgeGame
        scale
        (FinitePaperStrong.successGatedFiniteStrongGame context.piCcs
          extensionAlphabet strongAdversaryExpectedPolynomialTime)
        (Nightstream.SuperNeo.Folding.PiRLC.PaperWeakFiniteMixture.weakGame
          (Prefix := PrefixSeed Extension shape ProverSeed)
          laws strongSet verifier)
        (PiRlcComposition.Coupling.operationalCoupling context laws strongSet
          extensionAlphabet verifier strongAdversaryExpectedPolynomialTime))
      (abortingKnowledgeGame piDec.paper scale)
      (Adversary context ProverSeed ProverTape) where
  toSecond := fun adversary =>
    (piDecMixture context piDec extensionAlphabet verifier adversary).toProbabilityExperiment
  toFirst := fun adversary _extractor => toPiRlc context piDec adversary
  intermediateProbability := by
    intro adversary _extractor
    simpa [Nightstream.SuperNeo.InteractiveReduction.StrongWeakComposition.knowledgeGame,
      Nightstream.SuperNeo.Folding.PiRLC.PaperWeakFiniteMixture.weakGame,
      abortingKnowledgeGame, Mixture.toProbabilityExperiment] using
      intermediateProbability context piDec extensionAlphabet verifier adversary

/-- Full finite paper composition through `Pi_DEC`.  The final loss is
unchanged because Theorem 7 is exact and straight-line. -/
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
    (piDec : CompatiblePiDecContext context)
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
    (relaxedBindingRaw relaxedBindingRoot mixingBudget sumCheckBudget : Rat)
    (rootNonnegative : 0 <= relaxedBindingRoot)
    (rawBinding_le_rootSquare :
      relaxedBindingRaw <= relaxedBindingRoot * relaxedBindingRoot)
    (mixingNonnegative : 0 <= mixingBudget)
    (sumCheckNonnegative : 0 <= sumCheckBudget)
    (ops : PiRLC.RelaxedBindingOps (Assignment F columns) Commitment Scalar)
    (bindingLaws :
      Nightstream.SuperNeo.Folding.PiRLC.PaperForkCollision.RelaxedBindingLaws
        context.piRlc.semantics context.piRlc.params context.piRlc.algebra laws
        ops)
    (binding :
      Nightstream.SuperNeo.Folding.PiRLC.PaperWeakFiniteUniform.RelaxedBindingSecurity
        laws strongSet ops verifier relaxedBindingRaw)
    (ambientAdmissible : context.piCcs.params.b <=
      Nightstream.SuperNeo.Folding.PiRLC.PaperCorrections.correctedAmbientBoundFor
        context.piCcs.params)
    (contracts : FinitePaperStrong.NamedSecurityContracts context.piCcs
      extensionAlphabet strongAdversaryExpectedPolynomialTime mixingBudget
      sumCheckBudget) :
    ReductionOfKnowledge scale
      (Nightstream.SuperNeo.InteractiveReduction.KnowledgeComposition.knowledgeGame
        scale
        (Nightstream.SuperNeo.InteractiveReduction.StrongWeakComposition.knowledgeGame
          scale
          (FinitePaperStrong.successGatedFiniteStrongGame context.piCcs
            extensionAlphabet strongAdversaryExpectedPolynomialTime)
          (Nightstream.SuperNeo.Folding.PiRLC.PaperWeakFiniteMixture.weakGame
            (Prefix := PrefixSeed Extension shape ProverSeed)
            laws strongSet verifier)
          (PiRlcComposition.Coupling.operationalCoupling context laws strongSet
            extensionAlphabet verifier strongAdversaryExpectedPolynomialTime))
        (abortingKnowledgeGame piDec.paper scale)
        (operationalCoupling context piDec laws strongSet extensionAlphabet
          verifier strongAdversaryExpectedPolynomialTime))
      (ratio (context.arity.total + 1) verifier.alphabet.cardinality +
        ((mixingBudget + sumCheckBudget) +
          relaxedBindingRoot)) := by
  have composed :=
    Nightstream.SuperNeo.InteractiveReduction.KnowledgeComposition.reductionOfKnowledge
      scale PiRlcComposition.Coupling.rationalScaleLaws
      (Nightstream.SuperNeo.InteractiveReduction.StrongWeakComposition.knowledgeGame
        scale
        (FinitePaperStrong.successGatedFiniteStrongGame context.piCcs
          extensionAlphabet strongAdversaryExpectedPolynomialTime)
        (Nightstream.SuperNeo.Folding.PiRLC.PaperWeakFiniteMixture.weakGame
          (Prefix := PrefixSeed Extension shape ProverSeed)
          laws strongSet verifier)
        (PiRlcComposition.Coupling.operationalCoupling context laws strongSet
          extensionAlphabet verifier strongAdversaryExpectedPolynomialTime))
      (abortingKnowledgeGame piDec.paper scale)
      (operationalCoupling context piDec laws strongSet extensionAlphabet
        verifier strongAdversaryExpectedPolynomialTime)
      (ratio (context.arity.total + 1) verifier.alphabet.cardinality +
        ((mixingBudget + sumCheckBudget) +
          relaxedBindingRoot))
      scale.zero
      (PiRlcComposition.Coupling.finiteReductionOfKnowledge context laws
        strongSet extensionAlphabet verifier
        strongAdversaryExpectedPolynomialTime relaxedBindingRaw
        relaxedBindingRoot mixingBudget sumCheckBudget rootNonnegative
        rawBinding_le_rootSquare mixingNonnegative sumCheckNonnegative ops
        bindingLaws binding ambientAdmissible contracts)
      (abortingReductionOfKnowledge piDec.paper scale)
  simpa only [scale, Rat.zero_add] using composed

end Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.PiRlcComposition.PiDec
