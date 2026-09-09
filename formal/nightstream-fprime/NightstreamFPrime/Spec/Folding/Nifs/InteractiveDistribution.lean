import NightstreamFPrime.Spec.Folding.Nifs.InteractivePrefix
import NightstreamFPrime.Spec.Folding.Nifs.VerifierCoinSpace
import NightstreamFPrime.Spec.Folding.Nifs.SuffixCoinCoupling

/-!
The two experiments in SuperNeo B.1 have the same returned-value law.
The operational experiment first runs PiCCS and then samples the suffix for
that exact public output and captured state. A finite private-tape coupling
lets the existing causal strong-reduction theorem analyze that experiment.
No table is evaluated by the operational extractor.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Folding.Nifs.InteractiveDistribution

open scoped BigOperators
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open StrongReduction

variable {State Endpoint : Type*} [Fintype Endpoint] [DecidableEq Endpoint]
  {shape : Shape} {columns width : Nat}
  (firstPhase : InteractivePrefix.Prover State shape width)
  (abortEndpoint : Endpoint)
  (suffixLaw : PublicCoins K shape → FullOutputCoordinates.FullOutput K shape → State → PMF Endpoint)
  (consume : PublicCoins K shape → FullOutputCoordinates.FullOutput K shape → State →
    Endpoint → Option (OutputWitness shape columns))

noncomputable def lawAt (input : VerifierCoinSpace.Request shape) : PMF Endpoint :=
  let coins := VerifierCoinSpace.coins input
  match InteractivePrefix.run firstPhase coins.alpha coins.gamma coins.roundPoint with
  | none => PMF.pure abortEndpoint
  | some receipt => suffixLaw receipt.1.coins receipt.1.response.fullOutput receipt.2

noncomputable def tapes : PMF (VerifierCoinSpace.Request shape → Endpoint) := by
  classical
  exact SuffixCoinCoupling.tableLaw (lawAt firstPhase abortEndpoint suffixLaw)

/-- Only the selected suffix choice can affect the relayed final witness. -/
def coupled (table : VerifierCoinSpace.Request shape → Endpoint) :
    CausalExecution.Prover shape columns width :=
  InteractivePrefix.relay firstPhase fun coins output state =>
    consume coins output state (table (VerifierCoinSpace.request coins))

/-- Conditional mean in the sequential experiment, with prefix aborts and
all suffix outcomes retained. `value` can be any bounded success event. -/
noncomputable def sequentialMean
    (value : Option (Probe K shape × OutputWitness shape columns) → ℝ)
    (alpha : CubePoint K shape.cubeVariables) (gamma : K)
    (point : CubePoint K shape.cubeVariables) : ℝ :=
  match InteractivePrefix.run firstPhase alpha gamma point with
  | none => value none
  | some receipt =>
      ∑ endpoint,
        (suffixLaw receipt.1.coins receipt.1.response.fullOutput receipt.2 endpoint).toReal *
          value ((consume receipt.1.coins receipt.1.response.fullOutput receipt.2 endpoint).map
            fun witness => (receipt.1, witness))

omit [DecidableEq Endpoint] in
private theorem weights_sum (law : PMF Endpoint) : ∑ endpoint, (law endpoint).toReal = 1 := by
  rw [← ENNReal.toReal_sum (fun endpoint _ => law.apply_ne_top endpoint)]
  have total : ∑ endpoint, law endpoint = 1 := by
    simpa only [tsum_fintype] using law.tsum_coe
  rw [total, ENNReal.toReal_one]

omit [DecidableEq Endpoint] in
/-- A success indicator remains between zero and one under the exact suffix law. -/
theorem sequentialMean_range
    (value : Option (Probe K shape × OutputWitness shape columns) → ℝ)
    (range : ∀ outcome, 0 ≤ value outcome ∧ value outcome ≤ 1)
    (alpha : CubePoint K shape.cubeVariables) (gamma : K)
    (point : CubePoint K shape.cubeVariables) :
    0 ≤ sequentialMean firstPhase suffixLaw consume value alpha gamma point ∧
      sequentialMean firstPhase suffixLaw consume value alpha gamma point ≤ 1 := by
  unfold sequentialMean
  split
  · exact range none
  · constructor
    · exact Finset.sum_nonneg fun endpoint _ =>
        mul_nonneg ENNReal.toReal_nonneg (range _).1
    · calc
        _ ≤ ∑ endpoint, (suffixLaw _ _ _ endpoint).toReal * 1 := by
          apply Finset.sum_le_sum
          intro endpoint _
          exact mul_le_mul_of_nonneg_left (range _).2 ENNReal.toReal_nonneg
        _ = 1 := by simp only [mul_one, weights_sum]

/-- This pointwise equality is the selected-entry coupling, including the
case where the original PiCCS prefix aborts. -/
theorem selected_mean
    (value : Option (Probe K shape × OutputWitness shape columns) → ℝ)
    (alpha : CubePoint K shape.cubeVariables) (gamma : K)
    (point : CubePoint K shape.cubeVariables) :
    (∑ table, (tapes firstPhase abortEndpoint suffixLaw table).toReal *
      value (CausalExecution.run (coupled firstPhase consume table) alpha gamma point)) =
      sequentialMean firstPhase suffixLaw consume value alpha gamma point := by
  classical
  simp only [coupled, InteractivePrefix.run_relay]
  cases returned : InteractivePrefix.run firstPhase alpha gamma point with
  | none =>
      simp only [returned, Option.bind_none, sequentialMean, ← Finset.sum_mul,
        weights_sum, one_mul]
  | some receipt =>
      have same := InteractivePrefix.run_coins firstPhase alpha gamma point receipt returned
      have selected : lawAt firstPhase abortEndpoint suffixLaw
          (VerifierCoinSpace.request receipt.1.coins) =
          suffixLaw receipt.1.coins receipt.1.response.fullOutput receipt.2 := by
        simp only [lawAt, VerifierCoinSpace.coins_request, same, returned]
      simp only [returned, Option.bind_some, sequentialMean]
      change (∑ table,
        (SuffixCoinCoupling.tableLaw (lawAt firstPhase abortEndpoint suffixLaw) table).toReal *
          value ((consume receipt.1.coins receipt.1.response.fullOutput receipt.2
            (table (VerifierCoinSpace.request receipt.1.coins))).map
              fun witness => (receipt.1, witness))) = _
      have equal := SuffixCoinCoupling.selected_mean
        (lawAt firstPhase abortEndpoint suffixLaw) (VerifierCoinSpace.request receipt.1.coins)
        (fun endpoint => value
          ((consume receipt.1.coins receipt.1.response.fullOutput receipt.2 endpoint).map
            fun witness => (receipt.1, witness)))
      simpa only [selected] using equal

/-- Public coins may be averaged before or after the coupled private tape.
This is the distribution equality required by the strong-weak composition. -/
theorem executionMean_eq_sequentialMean
    (value : Option (Probe K shape × OutputWitness shape columns) → ℝ) :
    StrongProbability.executionMean (tapes firstPhase abortEndpoint suffixLaw)
      (coupled firstPhase consume) value =
      StrongProbability.verifierMean (sequentialMean firstPhase suffixLaw consume value) := by
  rw [← StrongProbability.clockMean_run_eq_executionMean,
    StrongProbability.clockMean_fintype]
  congr 1
  funext alpha gamma point
  exact selected_mean firstPhase abortEndpoint suffixLaw consume value alpha gamma point

end NightstreamFPrime.Spec.Folding.Nifs.InteractiveDistribution
