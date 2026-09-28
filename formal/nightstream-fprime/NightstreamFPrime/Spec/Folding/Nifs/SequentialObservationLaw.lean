import NightstreamFPrime.Spec.Folding.Nifs.PaperCompositionAgreement
import NightstreamFPrime.Spec.Folding.Nifs.VerifierCoinLaw

/-!
The actual sequential NIFS law before endpoint evidence is erased. It runs
the existing checked prefix and the suffix selected by its captured receipt.
The retained observation is the existing binding-reduction input. Aborts
remain outcomes. This probability law is not a runtime table sampler.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Folding.Nifs.SequentialObservationLaw

open scoped BigOperators ENNReal
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open StrongReduction PaperCompositionAgreement

attribute [local instance] Classical.propDecidable

variable {State Endpoint : Type*} {shape : Shape} {width : Nat}
  (firstPhase : InteractivePrefix.Prover State shape width)
  (suffixLaw : PublicCoins K shape → FullOutputCoordinates.FullOutput K shape →
    State → PMF Endpoint)

private noncomputable def atCoins (coins : PublicCoins K shape) :
    PMF (Observation State Endpoint shape) :=
  match InteractivePrefix.run firstPhase coins.alpha coins.gamma coins.roundPoint with
  | none => PMF.pure none
  | some receipt =>
      (suffixLaw receipt.1.coins receipt.1.response.fullOutput receipt.2).map
        fun endpoint => some (receipt, endpoint)

/-- One actual prefix and its selected weak-extractor return, with the
complete receipt and endpoint retained for the computed binding reduction. -/
noncomputable def law : PMF (Observation State Endpoint shape) :=
  (VerifierCoinLaw.law shape).bind fun request =>
    atCoins firstPhase suffixLaw (VerifierCoinSpace.coins request)

/-- Every possible returned observation has the endpoint support required
by the existing binding reduction. This includes the aborted-prefix case. -/
theorem supported (observation : Observation State Endpoint shape)
    (positive : observation ∈ (law firstPhase suffixLaw).support) :
    Supported suffixLaw observation := by
  obtain ⟨request, _, reached⟩ := (PMF.mem_support_bind_iff _ _ observation).mp positive
  let coins := VerifierCoinSpace.coins request
  change observation ∈ (atCoins firstPhase suffixLaw coins).support at reached
  unfold atCoins at reached
  cases returned : InteractivePrefix.run firstPhase coins.alpha coins.gamma coins.roundPoint with
  | none =>
      rw [returned, PMF.mem_support_pure_iff] at reached
      subst observation
      exact True.intro
  | some receipt =>
      rw [returned, PMF.mem_support_map_iff] at reached
      obtain ⟨endpoint, endpointSupported, rfl⟩ := reached
      exact ENNReal.toReal_pos
        ((PMF.mem_support_iff _ _).mp endpointSupported)
        ((suffixLaw receipt.1.coins receipt.1.response.fullOutput receipt.2).apply_ne_top endpoint)

private theorem finite_bind_hasSum {Input Output : Type*} [Fintype Input]
    (distribution : PMF Input) (next : Input → PMF Output)
    (value : Output → ℝ) (means : Input → ℝ)
    (each : ∀ input, HasSum (fun output => (next input output).toReal * value output)
      (means input)) :
    HasSum (fun output => ((distribution.bind next) output).toReal * value output)
      (∑ input, (distribution input).toReal * means input) := by
  have mass (output : Output) :
      ((distribution.bind next) output).toReal =
        ∑ input, (distribution input).toReal * (next input output).toReal := by
    rw [PMF.bind_apply, tsum_fintype, ENNReal.toReal_sum]
    · simp only [ENNReal.toReal_mul]
    · intro input _
      exact ENNReal.mul_ne_top (distribution.apply_ne_top input) ((next input).apply_ne_top output)
  have summed := hasSum_sum (s := Finset.univ)
    (fun input _ => (each input).mul_left (distribution input).toReal)
  simpa only [mass, Finset.sum_mul, mul_assoc] using summed

private theorem pure_value_hasSum {Output : Type*} (output : Output) (value : Output → ℝ) :
    HasSum (fun result => ((PMF.pure output) result).toReal * value result) (value output) := by
  classical
  simpa only [PMF.pure_apply_self, ENNReal.toReal_one, one_mul] using
    (hasSum_single (f := fun result => ((PMF.pure output) result).toReal * value result)
      output (fun result different => by
      change ((PMF.pure output) result).toReal * value result = 0
      rw [PMF.pure_apply_of_ne output result different, ENNReal.toReal_zero, zero_mul]))

variable [Fintype Endpoint]

private theorem atCoins_value_hasSum (value : Observation State Endpoint shape → ℝ)
    (coins : PublicCoins K shape) :
    HasSum (fun observation =>
      ((atCoins firstPhase suffixLaw coins) observation).toReal * value observation)
      (endpointMean firstPhase suffixLaw value coins.alpha coins.gamma coins.roundPoint) := by
  cases returned : InteractivePrefix.run firstPhase coins.alpha coins.gamma coins.roundPoint with
  | none =>
      simpa only [atCoins, endpointMean, returned] using pure_value_hasSum none value
  | some receipt =>
      simp only [atCoins, endpointMean, returned, PMF.map, Function.comp_def]
      exact finite_bind_hasSum _ _ _ _ (fun endpoint =>
        pure_value_hasSum (some (receipt, endpoint)) value)

/-- Every observable has the existing sequential endpoint mean. Finite
verifier requests and finite suffix endpoints give this exact sum without
an extra boundedness, independence or distribution premise. -/
theorem value_hasSum (value : Observation State Endpoint shape → ℝ) :
    HasSum (fun observation => (law firstPhase suffixLaw observation).toReal * value observation)
      (StrongProbability.verifierMean (endpointMean firstPhase suffixLaw value)) := by
  rw [VerifierCoinLaw.verifierMean_eq_requestMean, tsum_fintype]
  exact finite_bind_hasSum _ _ _ _ (fun request =>
    atCoins_value_hasSum firstPhase suffixLaw value (VerifierCoinSpace.coins request))

/-- Two fresh complete calls have exactly the existing pair mean, while
retaining the receipts and endpoints consumed by the binding reduction. -/
theorem pairMean_eq
    (value : Observation State Endpoint shape → Observation State Endpoint shape → ℝ) :
    (∑' left, (law firstPhase suffixLaw left).toReal *
      ∑' right, (law firstPhase suffixLaw right).toReal * value left right) =
      PaperCompositionAgreement.pairMean firstPhase suffixLaw value := by
  simp_rw [(value_hasSum firstPhase suffixLaw (value _)).tsum_eq]
  exact (value_hasSum firstPhase suffixLaw _).tsum_eq

variable [DecidableEq Endpoint]

/-- Erasing endpoint evidence only after the actual call recovers the
checked strong extractor's execution mean. Its causal coupling is used as
a proof of this equality, rather than as a runtime source of responses. -/
theorem outputMean_eq {columns : Nat} (abortEndpoint : Endpoint)
    (consume : PublicCoins K shape → FullOutputCoordinates.FullOutput K shape → State →
      Endpoint → Option (OutputWitness shape columns))
    (value : Option (Probe K shape × OutputWitness shape columns) → ℝ) :
    (∑' observation, (law firstPhase suffixLaw observation).toReal *
      value (outputOf consume observation)) =
      StrongProbability.executionMean
        (InteractiveDistribution.tapes firstPhase abortEndpoint suffixLaw)
        (InteractiveDistribution.coupled firstPhase consume) value := by
  rw [InteractiveDistribution.executionMean_eq_sequentialMean]
  have same : InteractiveDistribution.sequentialMean firstPhase suffixLaw consume value =
      endpointMean firstPhase suffixLaw (fun observation => value (outputOf consume observation)) := by
    funext alpha gamma point
    cases returned : InteractivePrefix.run firstPhase alpha gamma point <;>
      simp only [InteractiveDistribution.sequentialMean, endpointMean, outputOf, returned]
  rw [same]
  exact (value_hasSum firstPhase suffixLaw _).tsum_eq

/-- The linear theorem's normalized term uses these actual retained NIFS
observations: successful disagreement in two fresh calls divided by this
context's one-call relaxed success. No global conditioning is substituted. -/
theorem retryDisagreement_eq {Commitment PublicInput : Type*} {columns blockCount : Nat}
    (abortEndpoint : Endpoint)
    (consume : PublicCoins K shape → FullOutputCoordinates.FullOutput K shape → State →
      Endpoint → Option (OutputWitness shape columns))
    (maps : OpeningMaps Commitment PublicInput columns) (params : GlobalParams)
    (statement : Statement K Commitment PublicInput shape columns blockCount ConcreteCarrier.baseOps) :
    StrongProbability.retryDisagreementProbability
      (InteractiveDistribution.tapes firstPhase abortEndpoint suffixLaw)
      (InteractiveDistribution.coupled firstPhase consume) maps params statement =
      (∑' left, (law firstPhase suffixLaw left).toReal *
        ∑' right, (law firstPhase suffixLaw right).toReal *
          (if StrongProbability.SuccessfulDisagreement (width := width) maps params statement
            (outputOf consume left) (outputOf consume right) then (1 : ℝ) else 0)) /
      (∑' observation, (law firstPhase suffixLaw observation).toReal *
        (if StrongProbability.RelaxedSuccess (width := width) maps params statement
          (outputOf consume observation) then (1 : ℝ) else 0)) := by
  unfold StrongProbability.retryDisagreementProbability
  rw [pairMean_eq, ← disagreementProbability_eq_pairMean firstPhase abortEndpoint suffixLaw consume]
  congr 1
  exact (outputMean_eq firstPhase suffixLaw abortEndpoint consume _).symm

end NightstreamFPrime.Spec.Folding.Nifs.SequentialObservationLaw
