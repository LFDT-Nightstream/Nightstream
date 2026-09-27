import NightstreamFPrime.Lifecycle.Nifs.SupportedExtraction
import NightstreamFPrime.Spec.Folding.PiDEC.OutputWitnessConsumer
import NightstreamFPrime.Spec.Folding.Nifs.NonInteractive.PiRlcSampler.OracleModel

/-!
An explicit classical FS/SuperNeo game-transfer hypothesis. The owner
approved the parametric boundary in FIAT_SHAMIR_MODEL.md on 2026-09-11 UTC.
The real event runs the actual ProductionKey verifier: additive Poseidon2
absorption, existing domain labels, complete C output absorption, total four-field
R sampling, and the actual PiDEC attempt. It includes valid witnesses for
all sixteen returned children; bare public acceptance is not this event.

The translated side is the existing checked causal prefix and supported
R/D provider under the real law's context marginal, with the same public
input and key. FiatShamirModel assumes only a symbolic success transfer.
It supplies no source witness, checker correctness, call refinement, or work
bound. This is an additional game-transfer assumption, not Poseidon2 collision
resistance or an application of the printed CO25 overwrite-sponge theorem.

No oracle simulator, repeated-prefix cache, adaptive-query bound, or executable
adversary translation is constructed here. Q, g, and deltaFS have no defaults.
Any concrete model must separately supply their values, query inflation,
and replay scope. The local implementation premises remain explicit. The
prepared work theorem charges the supplied preparation call and existing
extractor clocks; it makes no machine-time or unprovided-translator claim.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Lifecycle.Nifs.FiatShamirTransfer

open scoped BigOperators
attribute [local instance] Classical.propDecidable

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open StrongReduction ConcreteCarrier
open _root_.NightstreamFPrime.Spec.Folding.Nifs
open _root_.NightstreamFPrime.Lifecycle.PaperAlgebra
open PiRLC.PaperForkExtraction PiRLC.PaperForkExtractionWork
open PiRLC.CoordinateForkLaw

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth}

/-- Existing proof data paired with witnesses for the existing ordered output.
This adds no protocol message, claimed verifier output, or representation. -/
structure RealOutput (relation : ProductionKey.LogicalRelation logicalWidth publicFits) where
  proof : PaperNonInteractive.Proof K PaperAlgebra.Commitment productionShape
    (ProductionKey.degreeBound relation)
  children : Fin productionShape.runningCount →
    PaperAlgebra.Assignment (logicalWidth := logicalWidth) (publicFits := publicFits)

variable
  (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
  (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))

/-- The actual NIFS verifier accepts, and the supplied witnesses open its
exact sixteen returned children. The same proof supplies the PiDEC attempt. -/
def RealSuccess
    (running : Lifecycle.Running (logicalWidth := logicalWidth) (publicFits := publicFits))
    (fresh : Lifecycle.Fresh (logicalWidth := logicalWidth) (publicFits := publicFits)) :
    Option (RealOutput relation) → Prop
  | none => False
  | some output =>
      let key := ProductionKey.key relation ajtai
      ∃ result attempt,
        PaperNonInteractive.verify key running fresh output.proof = some result ∧
        key.piDecAttempt running fresh output.proof = some attempt ∧
        ∀ child, CE.Holds key.piRlcSemantics key.params
          (PiDEC.OutputWitnessConsumer.runningStatement key result child) (output.children child)

/-- The real event's witnesses are for the exact verifier-computed PiDEC
children, in their original order, with no assumed output correspondence. -/
theorem realSuccess_implies_exact_children
    (running : Lifecycle.Running (logicalWidth := logicalWidth) (publicFits := publicFits))
    (fresh : Lifecycle.Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
    (output : RealOutput relation) (success : RealSuccess relation ajtai running fresh (some output)) :
    let key := ProductionKey.key relation ajtai
    ∃ result attempt,
      PaperNonInteractive.verify key running fresh output.proof = some result ∧
      key.piDecAttempt running fresh output.proof = some attempt ∧
      ∀ child, CE.Holds key.piRlcSemantics key.params
        (PiDEC.PaperVerifier.children key.piDecPublicInputSplit attempt child)
        (output.children (Fin.cast key.outputCount_eq child)) := by
  dsimp only
  rcases success with ⟨result, attempt, accepted, attemptEq, valid⟩
  refine ⟨result, attempt, accepted, attemptEq, ?_⟩
  intro child
  rw [← PiDEC.OutputWitnessConsumer.runningStatement_eq_child
    (ProductionKey.key relation ajtai) running fresh output.proof result attempt attemptEq accepted child]
  exact valid (Fin.cast (ProductionKey.key relation ajtai).outputCount_eq child)

variable {Context : Type*}
  (running : Context → Lifecycle.Running
    (logicalWidth := logicalWidth) (publicFits := publicFits))
  (fresh : Context → Lifecycle.Fresh
    (logicalWidth := logicalWidth) (publicFits := publicFits))

/-- Success mass under the supplied classical adversary output law.
There is no caller-supplied scalar standing in for verifier success. -/
noncomputable def realSuccessProbability
    (law : PMF (Context × Option (RealOutput relation))) : ℝ :=
  ∑' outcome, if RealSuccess relation ajtai (running outcome.1) (fresh outcome.1) outcome.2
    then (law outcome).toReal else 0

/-- The translated experiment keeps the same context and hence the same
running/fresh public input law. The relation and Ajtai key are shared parameters. -/
noncomputable def contextLaw (law : PMF (Context × Option (RealOutput relation))) : PMF Context :=
  law.map Prod.fst

variable {State Tape : Type*}
  [DecidableEq RingF]
  [Fintype (Challenge (ProductionKey.key relation ajtai).piRlcAlgebra)]
  [Nonempty (Challenge (ProductionKey.key relation ajtai).piRlcAlgebra)]
  (law : PMF (Context × Option (RealOutput relation)))
  (originalFirstPhase : Context → InteractivePrefix.Prover State productionShape 9)
  (abortTape : Tape)
  (provider : SupportedContinuation.Provider Tape relation ajtai running fresh (contextLaw relation law)
    (InteractiveComposition.firstPhase originalFirstPhase (SupportedExtraction.publicCheck running)))

/-- Exactly the success mean consumed by SupportedExtraction, on the
supported R/D continuation of the same checked causal prefix. -/
noncomputable def originalSuccessProbability : ℝ :=
  StrongProbability.clockMean (contextLaw relation law)
    (InteractiveComposition.originalSuccess relation ajtai running fresh originalFirstPhase
      (SupportedExtraction.publicCheck running)
      (SupportedContinuation.extension relation ajtai running fresh (contextLaw relation law)
        (InteractiveComposition.firstPhase originalFirstPhase (SupportedExtraction.publicCheck running))
        abortTape provider))

/-- Owner-approved additional classical game-transfer assumption. Its sole field
transfers success to the typed interactive experiment. No source conclusion,
local correctness, replay/query theorem, or time bound is assumed here.
The admitted adversaries, history depth and total-query interpretation are
specified in FIAT_SHAMIR_MODEL.md. Neither a model instance nor numerical
functions g and deltaFS are supplied by this module. -/
structure FiatShamirModel (g : Nat → ℝ → ℝ) (deltaFS : Nat → ℝ) (Q : Nat) : Prop where
  successTransfer :
    g Q (realSuccessProbability relation ajtai running fresh law) - deltaFS Q ≤
      originalSuccessProbability relation ajtai running fresh law originalFirstPhase abortTape provider

/-- Separate the proved sampler loss from the externally supplied FS error.
`sampleQueries Q` counts block-oracle calls in the translated experiment,
including adversarial calls and replays; it is not assumed to be 17. -/
noncomputable def samplerTransferError (deltaFS : Nat → ℝ) (sampleQueries : Nat → Nat) (Q : Nat) : ℝ :=
  deltaFS Q + sampleQueries Q * NonInteractive.PiRlcSampler.distance

/-- Construct the existing transfer interface from a specified block-oracle
experiment. The statistical comparison is proved, including repeated queries
and observations of raw lanes. The two experiment correspondences remain the
external FS applicability obligations: this theorem does not infer them for
concrete Poseidon2, or assume that an arbitrary `g` is Lipschitz. No source
witness, local verifier fact, or extractor correctness is assumed here. -/
theorem FiatShamirModel.of_blockOracle {OracleState : Type*}
    (g : Nat → ℝ → ℝ) (deltaFS : Nat → ℝ) (sampleQueries : Nat → Nat) (Q : Nat)
    (experiment : NonInteractive.PiRlcSampler.OracleModel.Program OracleState)
    (initial : OracleState)
    (test : NonInteractive.PiRlcSampler.OracleModel.Outcome OracleState → ℝ)
    (nonnegative : ∀ outcome, 0 ≤ test outcome) (atMostOne : ∀ outcome, test outcome ≤ 1)
    (rawTransfer :
      g Q (realSuccessProbability relation ajtai running fresh law) - deltaFS Q ≤
        NonInteractive.PiRlcSampler.average (fun tape => test
          (NonInteractive.PiRlcSampler.OracleModel.run experiment initial
            NonInteractive.PiRlcSampler.OracleModel.empty (sampleQueries Q) tape)))
    (balancedTransfer :
      NonInteractive.PiRlcSampler.balancedAverage (fun tape => test
        (NonInteractive.PiRlcSampler.OracleModel.run experiment initial
          NonInteractive.PiRlcSampler.OracleModel.empty (sampleQueries Q) tape)) ≤
        originalSuccessProbability relation ajtai running fresh law originalFirstPhase abortTape provider) :
    FiatShamirModel relation ajtai running fresh law originalFirstPhase abortTape provider
      g (samplerTransferError deltaFS sampleQueries) Q := by
  constructor
  have comparison := (abs_le.mp (NonInteractive.PiRlcSampler.OracleModel.run_bias_bound
    experiment initial (sampleQueries Q) test nonnegative atMostOne)).2
  dsimp only [samplerTransferError]
  linarith

/-- The sampler contribution over a history is charged for every translated
experiment's block-query budget. No independence between history visits is
needed to sum these per-visit bounds. -/
theorem samplerTransferError_sum {depth : Nat} (deltaFS : Nat → ℝ)
    (sampleQueries : Nat → Nat) (queries : Fin depth → Nat) :
    ∑ j, samplerTransferError deltaFS sampleQueries (queries j) =
      (∑ j, deltaFS (queries j)) +
        (∑ j, (sampleQueries (queries j) : ℝ)) * NonInteractive.PiRlcSampler.distance := by
  simp only [samplerTransferError, Finset.sum_add_distrib, Finset.sum_mul]

variable
  (g : Nat → ℝ → ℝ) (deltaFS : Nat → ℝ) (Q : Nat)
  (model : FiatShamirModel relation ajtai running fresh law originalFirstPhase abortTape provider g deltaFS Q)
  (program : Primitives RingF
    (PaperAlgebra.Assignment (logicalWidth := logicalWidth) (publicFits := publicFits)))
  (sourceProgram : Context → CheckedWitnessExtraction.Program productionShape
    (FullShape logicalWidth publicFits))
  (correct : Correct (PaperExtractionAlgebra.extractionAlgebra ajtai).ring
    (PaperExtractionAlgebra.extractionAlgebra ajtai).assignmentModule program)
  (bounds : PrimitiveBounds)
  (bounded : Bounded (PaperExtractionAlgebra.extractionAlgebra ajtai).ring program bounds)
  (sourceCorrect : ∀ context, CheckedWitnessExtraction.Correct (width := 9)
    (sourceProgram context) (PaperAlgebra.openingMaps ajtai).commit productionGlobalParams
    ((ProductionKey.key relation ajtai).statement (running context) (fresh context)))

include model correct bounded sourceCorrect in
/-- Algebraic composition with the existing actual binding-event bound.
Every implementation premise remains outside FiatShamirModel. -/
theorem returned_source_bound_with_binding :
    let continuation := SupportedContinuation.extension relation ajtai running fresh (contextLaw relation law)
      (InteractiveComposition.firstPhase originalFirstPhase (SupportedExtraction.publicCheck running))
      abortTape provider
    g Q (realSuccessProbability relation ajtai running fresh law) - deltaFS Q -
      InteractiveComposition.weakLoss relation ajtai -
      Real.sqrt (InteractiveAgreement.bindingProbability relation ajtai running fresh
        originalFirstPhase (SupportedExtraction.publicCheck running) continuation program
        (contextLaw relation law) + IndependentExecution.testError productionShape 9) ≤
      InteractiveOutput.returnedSourceProbability relation ajtai running fresh originalFirstPhase
        (SupportedExtraction.publicCheck running) continuation program sourceProgram (contextLaw relation law) := by
  dsimp only
  have transfer := model.successTransfer
  unfold originalSuccessProbability at transfer
  have extracted := SupportedExtraction.returned_source_bound_with_binding relation ajtai running fresh
    (contextLaw relation law) originalFirstPhase abortTape provider program sourceProgram
    correct bounds bounded sourceCorrect
  exact (sub_le_sub_right (sub_le_sub_right transfer _) _).trans extracted

include model correct bounded sourceCorrect in
/-- The same-key MSIS success event is the existing executable reduction's
event. Its probability is not replaced by a numerical hardness estimate. -/
theorem returned_source_bound_with_msis :
    let continuation := SupportedContinuation.extension relation ajtai running fresh (contextLaw relation law)
      (InteractiveComposition.firstPhase originalFirstPhase (SupportedExtraction.publicCheck running))
      abortTape provider
    g Q (realSuccessProbability relation ajtai running fresh law) - deltaFS Q -
      InteractiveComposition.weakLoss relation ajtai -
      Real.sqrt (BindingProbability.successProbability ajtai program relation running fresh
        originalFirstPhase (SupportedExtraction.publicCheck running) continuation
        (fun context => (sourceProgram context).access) (contextLaw relation law) * PaperProfile.arity.total +
        IndependentExecution.testError productionShape 9) ≤
      InteractiveOutput.returnedSourceProbability relation ajtai running fresh originalFirstPhase
        (SupportedExtraction.publicCheck running) continuation program sourceProgram (contextLaw relation law) := by
  dsimp only
  have transfer := model.successTransfer
  unfold originalSuccessProbability at transfer
  have extracted := SupportedExtraction.returned_source_bound_with_msis relation ajtai running fresh
    (contextLaw relation law) originalFirstPhase abortTape provider program sourceProgram
    correct bounds bounded sourceCorrect
  exact (sub_le_sub_right (sub_le_sub_right transfer _) _).trans extracted

include model correct bounded sourceCorrect in
/-- Transfer the v1.2 additive bound without changing the approved FS
model. The MSIS term is the actual stopped reduction under the same context
law; efficient translation and query applicability remain external. -/
theorem returned_source_bound_with_adaptive_msis :
    let continuation := SupportedContinuation.extension relation ajtai running fresh (contextLaw relation law)
      (InteractiveComposition.firstPhase originalFirstPhase (SupportedExtraction.publicCheck running))
      abortTape provider
    g Q (realSuccessProbability relation ajtai running fresh law) - deltaFS Q -
      InteractiveComposition.weakLoss relation ajtai - IndependentExecution.testError productionShape 9 -
      AdaptiveBindingProbability.successProbability relation ajtai program running fresh
        originalFirstPhase (SupportedExtraction.publicCheck running) continuation sourceProgram
        (contextLaw relation law) * PaperProfile.arity.total ≤
      InteractiveOutput.returnedSourceProbability relation ajtai running fresh originalFirstPhase
        (SupportedExtraction.publicCheck running) continuation program sourceProgram (contextLaw relation law) := by
  dsimp only
  have transfer := model.successTransfer
  unfold originalSuccessProbability at transfer
  have extracted := SupportedExtraction.returned_source_bound_with_adaptive_msis relation ajtai running fresh
    (contextLaw relation law) originalFirstPhase abortTape provider program sourceProgram
    correct bounds bounded sourceCorrect
  exact (sub_le_sub_right (sub_le_sub_right (sub_le_sub_right transfer _) _) _).trans extracted

include model correct bounded sourceCorrect in
/-- A supplied bound on that exact same-key success probability can be used
without changing the FS hypothesis or its real verifier-success event. -/
theorem returned_source_bound_of_msis
    (epsilonMSIS : ℝ)
    (msisBound :
      let continuation := SupportedContinuation.extension relation ajtai running fresh (contextLaw relation law)
        (InteractiveComposition.firstPhase originalFirstPhase (SupportedExtraction.publicCheck running))
        abortTape provider
      BindingProbability.successProbability ajtai program relation running fresh originalFirstPhase
        (SupportedExtraction.publicCheck running) continuation
        (fun context => (sourceProgram context).access) (contextLaw relation law) ≤ epsilonMSIS) :
    let continuation := SupportedContinuation.extension relation ajtai running fresh (contextLaw relation law)
      (InteractiveComposition.firstPhase originalFirstPhase (SupportedExtraction.publicCheck running))
      abortTape provider
    g Q (realSuccessProbability relation ajtai running fresh law) - deltaFS Q -
      InteractiveComposition.weakLoss relation ajtai -
      Real.sqrt (epsilonMSIS * PaperProfile.arity.total + IndependentExecution.testError productionShape 9) ≤
      InteractiveOutput.returnedSourceProbability relation ajtai running fresh originalFirstPhase
        (SupportedExtraction.publicCheck running) continuation program sourceProgram (contextLaw relation law) := by
  dsimp only at msisBound ⊢
  have extracted := returned_source_bound_with_msis relation ajtai running fresh law
    originalFirstPhase abortTape provider g deltaFS Q model program sourceProgram correct bounds
    bounded sourceCorrect
  have errorBound := Real.sqrt_le_sqrt (_root_.add_le_add
    (mul_le_mul_of_nonneg_right msisBound (Nat.cast_nonneg PaperProfile.arity.total))
    (le_refl (IndependentExecution.testError productionShape 9)))
  exact (sub_le_sub_left errorBound _).trans extracted

include model correct bounded sourceCorrect in
/-- Compose with the existing prepared source/MSIS work theorem. The supplied
preparation call generates the same context law; its charged translation work
and every local value/moment premise are separate from FiatShamirModel.
The result concerns the declared clocks of these calls, not an unprovided
simulator, oracle-query count, or compiled execution-time bound. -/
theorem prepared_probability_and_expected_work {SetupTape : Type*}
    (setupTapes : PMF SetupTape) (prepare : SetupTape → Result Context)
    (preparedContexts : contextLaw relation law = ContextPreparation.contexts setupTapes prepare)
    (preparationSummable : Summable fun tape => (setupTapes tape).toReal * (prepare tape).work)
    (call : Context → CubePoint K productionShape.cubeVariables → K →
      CubePoint K productionShape.cubeVariables → Result (Option (Probe K productionShape × State)))
    (callCorrect : ∀ context alpha gamma point,
      (call context alpha gamma point).value = InteractivePrefix.run
        (InteractiveComposition.firstPhase originalFirstPhase
          (SupportedExtraction.publicCheck running) context) alpha gamma point)
    (accessBound : Nat)
    (accessBounded : ∀ context, CostedWitnessProjection.Bounded (sourceProgram context).access accessBound)
    (securityParameter : Nat)
    (preparationPolynomial basePolynomial primitivePolynomial accessPolynomial : Polynomial ℝ) :
    let continuation := SupportedContinuation.extension relation ajtai running fresh (contextLaw relation law)
      (InteractiveComposition.firstPhase originalFirstPhase (SupportedExtraction.publicCheck running))
      abortTape provider
    let base := InteractiveWork.baseClock relation ajtai running fresh continuation call program sourceProgram
    let bindingTotal := BindingWork.totalClock ajtai program relation running fresh originalFirstPhase
      (SupportedExtraction.publicCheck running) continuation call sourceProgram
    let total := ContextPreparation.clock prepare
      (fun context => StrongProbability.verifierMean (bindingTotal context))
    let sourcePolynomial := Polynomial.C ((PaperProfile.arity.total : ℝ) + 1) * basePolynomial +
      Polynomial.C (PaperProfile.arity.total : ℝ) * (primitivePolynomial + Polynomial.C 3) +
      Polynomial.C (productionShape.freshCount : ℝ) *
        (Polynomial.C (WitnessProjection.privateWidth (FullShape logicalWidth publicFits) : ℝ) *
          (accessPolynomial + Polynomial.C 6) + Polynomial.C 9) +
      Polynomial.C (productionShape.runningCount : ℝ) *
        (Polynomial.C ((FullShape logicalWidth publicFits).carrierWidth : ℝ) *
          (accessPolynomial + Polynomial.C 6) + Polynomial.C 9) + Polynomial.C 13
    Summable (fun context => (contextLaw relation law context).toReal *
      StrongProbability.verifierMean (base context)) →
    StrongProbability.clockMean (contextLaw relation law) base ≤ basePolynomial.eval (securityParameter : ℝ) →
    (bounds.coordinateWork : ℝ) ≤ primitivePolynomial.eval (securityParameter : ℝ) →
    (accessBound : ℝ) ≤ accessPolynomial.eval (securityParameter : ℝ) →
    (∑' tape, (setupTapes tape).toReal * (prepare tape).work) ≤
      preparationPolynomial.eval (securityParameter : ℝ) →
    (g Q (realSuccessProbability relation ajtai running fresh law) - deltaFS Q -
      InteractiveComposition.weakLoss relation ajtai -
      Real.sqrt ((∑' tape, (setupTapes tape).toReal * BindingProbability.localSuccessProbability ajtai program
        (sourceProgram (prepare tape).value).access relation running fresh originalFirstPhase
        (SupportedExtraction.publicCheck running) continuation (prepare tape).value) * PaperProfile.arity.total +
          IndependentExecution.testError productionShape 9) ≤
      InteractiveOutput.returnedSourceProbability relation ajtai running fresh originalFirstPhase
        (SupportedExtraction.publicCheck running) continuation program sourceProgram (contextLaw relation law)) ∧
    Summable (fun tape => (setupTapes tape).toReal * total tape) ∧
    (∑' tape, (setupTapes tape).toReal * total tape) ≤
      (preparationPolynomial + Polynomial.C 2 * sourcePolynomial + Polynomial.C 3 * primitivePolynomial +
        Polynomial.C ((FullShape logicalWidth publicFits).carrierWidth : ℝ) *
          (accessPolynomial + Polynomial.C 6) + Polynomial.C 13).eval (securityParameter : ℝ) := by
  dsimp only
  intro baseSummable basePPT primitivePPT accessPPT preparationPPT
  have checked := SupportedExtraction.msis_probability_and_expected_work relation ajtai running fresh
    (contextLaw relation law) originalFirstPhase abortTape provider program sourceProgram
    correct bounds bounded sourceCorrect setupTapes prepare preparedContexts preparationSummable
    call callCorrect accessBound accessBounded securityParameter
    preparationPolynomial basePolynomial primitivePolynomial accessPolynomial
    baseSummable basePPT primitivePPT accessPPT preparationPPT
  have transfer := model.successTransfer
  unfold originalSuccessProbability at transfer
  exact ⟨(sub_le_sub_right (sub_le_sub_right transfer _) _).trans checked.2.1, checked.2.2⟩

end NightstreamFPrime.Lifecycle.Nifs.FiatShamirTransfer
