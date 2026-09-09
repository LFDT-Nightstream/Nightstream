import NightstreamFPrime.Spec.Folding.Nifs.InteractiveDistribution

/-!
SuperNeo B.1's strong-extractor loss for the sequential returned-value law.
The suffix kernel is analyzed through a proved causal coupling; the source
and relaxed probabilities below measure the original sequential experiment.
The weak suffix's concrete success loss is supplied by PaperWeakExtraction.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Folding.Nifs.PaperCompositionProbability

open scoped BigOperators
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open StrongReduction ConcreteCarrier

attribute [local instance] Classical.propDecidable

variable {Context State Endpoint Commitment PublicInput : Type*}
  [Fintype Endpoint] [DecidableEq Endpoint]
  {shape : Shape} {columns blockCount width : Nat}
  (contexts : PMF Context)
  (firstPhase : Context → InteractivePrefix.Prover State shape width)
  (abortEndpoint : Endpoint)
  (suffixLaw : Context → PublicCoins K shape → FullOutputCoordinates.FullOutput K shape →
    State → PMF Endpoint)
  (consume : Context → PublicCoins K shape → FullOutputCoordinates.FullOutput K shape → State →
    Endpoint → Option (OutputWitness shape columns))
  (maps : Context → OpeningMaps Commitment PublicInput columns) (params : GlobalParams)
  (statement : Context → Statement K Commitment PublicInput shape columns blockCount baseOps)

/-- All returns of the sequential experiment use this context and these C coins. -/
noncomputable def eventProbability
    (event : Context → Option (Probe K shape × OutputWitness shape columns) → Prop) : ℝ :=
  ∑' context, (contexts context).toReal * StrongProbability.verifierMean
    (InteractiveDistribution.sequentialMean (firstPhase context)
      (suffixLaw context) (consume context) (fun outcome => if event context outcome then 1 else 0))

noncomputable def relaxedProbability : ℝ :=
  eventProbability contexts firstPhase suffixLaw consume fun context =>
    StrongProbability.RelaxedSuccess (width := width) (maps context) params (statement context)

noncomputable def sourceProbability : ℝ :=
  eventProbability contexts firstPhase suffixLaw consume fun context outcome =>
    StrongProbability.RelaxedSuccess (width := width) (maps context) params (statement context) outcome ∧
      StrongProbability.SourceValid (maps context) params (statement context) outcome

/-- Both rewound calls keep the original context, and independently sample
the verifier coins and their selected suffix outcomes. -/
noncomputable def disagreementProbability : ℝ :=
  StrongProbability.globalDisagreementProbability contexts
    (fun context => InteractiveDistribution.tapes (firstPhase context) abortEndpoint (suffixLaw context))
    (fun context => InteractiveDistribution.coupled (firstPhase context) (consume context))
    maps params statement

theorem relaxedProbability_eq :
    relaxedProbability contexts firstPhase suffixLaw consume maps params statement =
      StrongProbability.globalSuccessProbability contexts
        (fun context => InteractiveDistribution.tapes (firstPhase context) abortEndpoint (suffixLaw context))
        (fun context => InteractiveDistribution.coupled (firstPhase context) (consume context))
        maps params statement := by
  unfold relaxedProbability eventProbability StrongProbability.globalSuccessProbability
  congr 1
  funext context
  unfold StrongProbability.successProbability
  dsimp only
  rw [InteractiveDistribution.executionMean_eq_sequentialMean]
  rfl

theorem sourceProbability_eq :
    sourceProbability contexts firstPhase suffixLaw consume maps params statement =
      StrongProbability.globalSourceProbability contexts
        (fun context => InteractiveDistribution.tapes (firstPhase context) abortEndpoint (suffixLaw context))
        (fun context => InteractiveDistribution.coupled (firstPhase context) (consume context))
        maps params statement := by
  unfold sourceProbability eventProbability StrongProbability.globalSourceProbability
  congr 1
  funext context
  unfold StrongProbability.sourceProbability
  dsimp only
  rw [InteractiveDistribution.executionMean_eq_sequentialMean]
  rfl

/-- The strong proof applies to the actual sequential returned witnesses,
with its exact measured disagreement loss and PiCCS test error. -/
theorem source_success_ge
    (freshBound : params.b = 2)
    (constantLaw : ∀ context,
      MatrixCoefficientSource.ConstantTermLaw baseOps (statement context).matrixSource.kernel)
    (degreeCovers : ∀ context,
      ((statement context).verifierInput K.embed).sumcheckDegreeBound ≤ width) :
    relaxedProbability contexts firstPhase suffixLaw consume maps params statement -
      Real.sqrt (disagreementProbability contexts firstPhase abortEndpoint suffixLaw consume
        maps params statement + IndependentExecution.testError shape width) ≤
      sourceProbability contexts firstPhase suffixLaw consume maps params statement := by
  rw [relaxedProbability_eq contexts firstPhase abortEndpoint suffixLaw consume maps params statement,
    sourceProbability_eq contexts firstPhase abortEndpoint suffixLaw consume maps params statement]
  exact StrongProbability.source_success_ge contexts _ _ maps params statement
    freshBound constantLaw degreeCovers

/-- Integrate the proved weak suffix loss on the same original context and
verifier coins, then apply the strong reduction. The concrete caller supplies
this local inequality from its actual suffix algorithm, not as a hardness premise. -/
theorem source_success_ge_from_weak
    (originalClock : Context → CubePoint K shape.cubeVariables → K →
      CubePoint K shape.cubeVariables → ℝ)
    (weakLoss : ℝ)
    (originalNonnegative : ∀ context alpha gamma point, 0 ≤ originalClock context alpha gamma point)
    (localWeak : ∀ context alpha gamma point,
      originalClock context alpha gamma point - weakLoss ≤
        InteractiveDistribution.sequentialMean (firstPhase context) (suffixLaw context) (consume context)
          (fun outcome => if StrongProbability.RelaxedSuccess (width := width)
            (maps context) params (statement context) outcome then (1 : ℝ) else 0)
          alpha gamma point)
    (freshBound : params.b = 2)
    (constantLaw : ∀ context,
      MatrixCoefficientSource.ConstantTermLaw baseOps (statement context).matrixSource.kernel)
    (degreeCovers : ∀ context,
      ((statement context).verifierInput K.embed).sumcheckDegreeBound ≤ width) :
    StrongProbability.clockMean contexts originalClock - weakLoss -
      Real.sqrt (disagreementProbability contexts firstPhase abortEndpoint suffixLaw consume
        maps params statement + IndependentExecution.testError shape width) ≤
      sourceProbability contexts firstPhase suffixLaw consume maps params statement := by
  let relaxed := fun context =>
    InteractiveDistribution.sequentialMean (firstPhase context) (suffixLaw context) (consume context)
      (fun outcome => if StrongProbability.RelaxedSuccess (width := width)
        (maps context) params (statement context) outcome then (1 : ℝ) else 0)
  have bounded : ∀ context alpha gamma point,
      0 ≤ relaxed context alpha gamma point ∧ relaxed context alpha gamma point ≤ 1 := by
    intro context alpha gamma point
    apply InteractiveDistribution.sequentialMean_range
    intro outcome
    split_ifs <;> norm_num
  have summed := StrongProbability.clockMean_summable_of_bounded contexts relaxed 1 bounded
  have weak := StrongProbability.clockMean_le_add_const contexts relaxed originalClock weakLoss
    originalNonnegative summed (by
      intro context alpha gamma point
      have localBound := localWeak context alpha gamma point
      change originalClock context alpha gamma point ≤
        relaxed context alpha gamma point + weakLoss
      dsimp only [relaxed]
      linarith)
  have exactMean : StrongProbability.clockMean contexts relaxed =
      relaxedProbability contexts firstPhase suffixLaw consume maps params statement := rfl
  rw [exactMean] at weak
  have strong := source_success_ge contexts firstPhase abortEndpoint suffixLaw consume
    maps params statement freshBound constantLaw degreeCovers
  linarith [weak.2]

end NightstreamFPrime.Spec.Folding.Nifs.PaperCompositionProbability
