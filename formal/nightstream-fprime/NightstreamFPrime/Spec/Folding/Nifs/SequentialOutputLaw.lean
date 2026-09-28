import NightstreamFPrime.Spec.Folding.Nifs.VerifierCoinLaw
import NightstreamFPrime.Spec.Folding.Nifs.InteractiveDistribution
import NightstreamFPrime.Spec.Folding.Nifs.PaperCompositionProbability

/-!
The mathematical law of the existing sequential NIFS output. It preserves
the context, runs the actual prefix, and uses the suffix law for that exact
receipt. Prefix and suffix aborts remain outputs. No coupled endpoint table
is sampled, and no executable sampler, work, or security claim is added.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Folding.Nifs.SequentialOutputLaw

open scoped BigOperators ENNReal
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open StrongReduction

attribute [local instance] Classical.propDecidable

private theorem event_ne_top {Sample : Type*} (distribution : PMF Sample)
    (event : Set Sample) : distribution.toOuterMeasure event ≠ ∞ := by
  rw [PMF.toOuterMeasure_apply]
  exact distribution.tsum_coe_indicator_ne_top event

private theorem bind_event_toReal {Sample Output : Type*}
    (distribution : PMF Sample) (next : Sample → PMF Output) (event : Set Output) :
    ((distribution.bind next).toOuterMeasure event).toReal =
      ∑' sample, (distribution sample).toReal * ((next sample).toOuterMeasure event).toReal := by
  rw [PMF.toOuterMeasure_bind_apply,
    ENNReal.tsum_toReal_eq (fun sample =>
      ENNReal.mul_ne_top (distribution.apply_ne_top sample) (event_ne_top (next sample) event))]
  simp only [ENNReal.toReal_mul]

private theorem event_toReal {Sample : Type*} (distribution : PMF Sample)
    (event : Set Sample) :
    (distribution.toOuterMeasure event).toReal =
      ∑' sample, (distribution sample).toReal * (if sample ∈ event then 1 else 0) := by
  classical
  have finite (sample : Sample) : event.indicator distribution sample ≠ ∞ := by
    by_cases member : sample ∈ event
    · simpa only [Set.indicator_of_mem member] using distribution.apply_ne_top sample
    · rw [Set.indicator_of_notMem member]
      exact ENNReal.zero_ne_top
  rw [PMF.toOuterMeasure_apply, ENNReal.tsum_toReal_eq finite]
  apply tsum_congr
  intro sample
  by_cases member : sample ∈ event
  · simp only [Set.indicator_of_mem member, if_pos member, mul_one]
  · simp only [Set.indicator_of_notMem member, if_neg member, ENNReal.toReal_zero, mul_zero]

variable {Context State Endpoint : Type*} {shape : Shape} {columns width : Nat}
  (contexts : PMF Context)
  (firstPhase : Context → InteractivePrefix.Prover State shape width)
  (suffixLaw : Context → PublicCoins K shape → FullOutputCoordinates.FullOutput K shape →
    State → PMF Endpoint)
  (consume : Context → PublicCoins K shape → FullOutputCoordinates.FullOutput K shape → State →
    Endpoint → Option (OutputWitness shape columns))

private noncomputable def atCoins (context : Context) (coins : PublicCoins K shape) :
    PMF (Context × Option (Probe K shape × OutputWitness shape columns)) :=
  match InteractivePrefix.run (firstPhase context) coins.alpha coins.gamma coins.roundPoint with
  | none => PMF.pure (context, none)
  | some receipt =>
      (suffixLaw context receipt.1.coins receipt.1.response.fullOutput receipt.2).map fun endpoint =>
        (context,
          (consume context receipt.1.coins receipt.1.response.fullOutput receipt.2 endpoint).map
            fun witness => (receipt.1, witness))

/-- Context, verifier coins, and the actual receipt's suffix are drawn in
that order. The returned pair retains its context even when either phase aborts. -/
noncomputable def law : PMF (Context × Option (Probe K shape × OutputWitness shape columns)) :=
  contexts.bind fun context =>
    (VerifierCoinLaw.law shape).bind fun request =>
      atCoins firstPhase suffixLaw consume context (VerifierCoinSpace.coins request)

private theorem atCoins_context_marginal (context : Context) (coins : PublicCoins K shape) :
    (atCoins firstPhase suffixLaw consume context coins).map Prod.fst = PMF.pure context := by
  unfold atCoins
  split
  · simp only [PMF.pure_map]
  · rw [PMF.map_comp]
    exact PMF.map_const _ context

/-- Taking the context of the returned pair gives its original PMF, including
all prefix and suffix aborts. No support or acceptance restriction is needed. -/
theorem context_marginal :
    (law contexts firstPhase suffixLaw consume).map Prod.fst = contexts := by
  simp only [law, PMF.map_bind, atCoins_context_marginal, PMF.bind_const, PMF.bind_pure]

private theorem bind_eq_on_support {Sample Result : Type*}
    (distribution : PMF Sample) (left right : Sample → PMF Result)
    (same : ∀ sample ∈ distribution.support, left sample = right sample) :
    distribution.bind left = distribution.bind right := by
  apply PMF.ext
  intro result
  simp only [PMF.bind_apply]
  apply tsum_congr
  intro sample
  by_cases zero : distribution sample = 0
  · simp only [zero, zero_mul]
  · rw [same sample ((distribution.mem_support_iff sample).mpr zero)]

/-- Only the continuation of an actual receipt in a supported context can
affect the sequential output law. All other contexts and aborts retain their
original mass; no conditional law or independent-execution premise is used. -/
theorem law_eq_of_suffix_eq_on_return
    (otherSuffix : Context → PublicCoins K shape → FullOutputCoordinates.FullOutput K shape →
      State → PMF Endpoint)
    (same : ∀ context ∈ contexts.support,
      ∀ alpha gamma point (receipt : Probe K shape × State),
        InteractivePrefix.run (firstPhase context) alpha gamma point = some receipt →
        suffixLaw context receipt.1.coins receipt.1.response.fullOutput receipt.2 =
          otherSuffix context receipt.1.coins receipt.1.response.fullOutput receipt.2) :
    law contexts firstPhase suffixLaw consume = law contexts firstPhase otherSuffix consume := by
  unfold law
  apply bind_eq_on_support
  intro context supported
  apply congrArg (PMF.bind (VerifierCoinLaw.law shape))
  funext request
  let coins := VerifierCoinSpace.coins request
  change atCoins firstPhase suffixLaw consume context coins =
    atCoins firstPhase otherSuffix consume context coins
  unfold atCoins
  cases returned : InteractivePrefix.run (firstPhase context)
      coins.alpha coins.gamma coins.roundPoint with
  | none => rfl
  | some receipt =>
      dsimp only
      rw [same context supported coins.alpha coins.gamma coins.roundPoint receipt returned]

variable [Fintype Endpoint]

private theorem atCoins_event_eq (context : Context) (coins : PublicCoins K shape)
    (event : Context → Option (Probe K shape × OutputWitness shape columns) → Prop) :
    ((atCoins firstPhase suffixLaw consume context coins).toOuterMeasure
      {sample | event sample.1 sample.2}).toReal =
      InteractiveDistribution.sequentialMean (firstPhase context) (suffixLaw context) (consume context)
        (fun outcome => if event context outcome then 1 else 0)
        coins.alpha coins.gamma coins.roundPoint := by
  classical
  cases returned : InteractivePrefix.run (firstPhase context)
      coins.alpha coins.gamma coins.roundPoint with
  | none =>
      simp only [atCoins, InteractiveDistribution.sequentialMean, returned,
        PMF.toOuterMeasure_pure_apply, Set.mem_setOf_eq]
      by_cases accepted : event context none
      · simp only [if_pos accepted, ENNReal.toReal_one]
      · simp only [if_neg accepted, ENNReal.toReal_zero]
  | some receipt =>
      simp only [atCoins, InteractiveDistribution.sequentialMean, returned,
        PMF.toOuterMeasure_map_apply, event_toReal, tsum_fintype,
        Set.mem_preimage, Set.mem_setOf_eq]

/-- Every event has exactly its existing sequential probability. The
context and all aborts are retained; there is no extra distribution premise. -/
theorem eventProbability_eq
    (event : Context → Option (Probe K shape × OutputWitness shape columns) → Prop) :
    ((law contexts firstPhase suffixLaw consume).toOuterMeasure
      {sample | event sample.1 sample.2}).toReal =
      PaperCompositionProbability.eventProbability contexts firstPhase suffixLaw consume event := by
  rw [law, bind_event_toReal]
  unfold PaperCompositionProbability.eventProbability
  apply tsum_congr
  intro context
  apply congrArg (fun mean : ℝ => (contexts context).toReal * mean)
  rw [bind_event_toReal]
  simp only [atCoins_event_eq]
  exact (VerifierCoinLaw.verifierMean_eq_requestMean
    (InteractiveDistribution.sequentialMean (firstPhase context)
      (suffixLaw context) (consume context) (fun outcome => if event context outcome then 1 else 0))).symm

end NightstreamFPrime.Spec.Folding.Nifs.SequentialOutputLaw
