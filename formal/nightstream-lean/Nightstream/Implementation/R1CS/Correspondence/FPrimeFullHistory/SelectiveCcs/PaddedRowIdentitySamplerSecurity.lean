import Mathlib.Tactic.NormNum
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.PaddedRowIdentityPoseidon2
import Nightstream.SuperNeo.InteractiveReduction.FiniteUniform

/-!
Contract: finite failure bound for the selected bounded full-field `Pi_RLC`
sampler.

Owns:
- the exact 15-source, 54-coefficient, three-attempt parameter tuple;
- the equivalence between one coefficient failure and three rejections;
- the exact `810/q^3` per-fold exhaustion expression; and
- the proved 182-bit rational upper bound.

Does not own: a proof that concrete Poseidon2 candidate calls are independent
uniform Goldilocks elements, collision resistance, the random-oracle
assumption, low-norm invertibility, Rust, or R1CS correspondence.

Assurance tier: security-reduced. The finite rational arithmetic is proved in
Lean. `GoldilocksRandomOracleSamplerContract` is the precise remaining
probability premise for the complete sampler event. An external random-oracle
argument must derive this bound from uniformity and independence. Those
distribution facts are not encoded as fields of this Lean proposition.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentitySamplerSecurity

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.Nifs
open Nightstream.SuperNeo.InteractiveReduction.FiniteUniform
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentityPoseidon2

/-- Exact event that all three indexed candidates for one coefficient reject. -/
def ThreeRejections
    (state : State)
    (source : Fin PaperProfile.arity.total)
    (coefficient : Fin samplerCoefficientCount) : Prop :=
  candidateAccepted
      (candidateValue state source coefficient firstAttempt) = false /\
    candidateAccepted
      (candidateValue state source coefficient secondAttempt) = false /\
    candidateAccepted
      (candidateValue state source coefficient thirdAttempt) = false

theorem sampleCoefficient_eq_none_iff_threeRejections
    (state : State)
    (source : Fin PaperProfile.arity.total)
    (coefficient : Fin samplerCoefficientCount) :
    sampleCoefficient state source coefficient = none ↔
      ThreeRejections state source coefficient := by
  cases first : candidateAccepted
      (candidateValue state source coefficient firstAttempt) <;>
    cases second : candidateAccepted
      (candidateValue state source coefficient secondAttempt) <;>
    cases third : candidateAccepted
      (candidateValue state source coefficient thirdAttempt) <;>
    simp [sampleCoefficient, ThreeRejections, first, second, third]

/-- A shortfall witness names one coefficient whose three attempts reject. -/
theorem shortfall_requires_three_rejections
    {state : State} (shortfall : SamplerShortfall state) :
    Exists fun source : Fin PaperProfile.arity.total =>
      Exists fun coefficient : Fin samplerCoefficientCount =>
        ThreeRejections state source coefficient := by
  rcases shortfall with ⟨source, coefficient, failed⟩
  exact ⟨source, coefficient,
    (sampleCoefficient_eq_none_iff_threeRejections
      state source coefficient).mp failed⟩

/-- One uniform candidate rejects with probability exactly `1/q`. -/
def singleCandidateRejectionProbability : Rat :=
  (1 : Rat) / (goldilocksModulus : Rat)

/-- Three independent candidate rejections exhaust one coefficient. -/
def singleCoefficientExhaustionBound : Rat :=
  (1 : Rat) / ((goldilocksModulus ^ samplerAttemptCount : Nat) : Rat)

/-- There are exactly `15 * 54 = 810` sampled coefficients per fold. -/
def sampledCoefficientCount : Nat :=
  PaperProfile.arity.total * samplerCoefficientCount

/-- Union bound for all coefficient-exhaustion events in one fold. -/
def completeSamplerShortfallBound : Rat :=
  (sampledCoefficientCount : Rat) * singleCoefficientExhaustionBound

/-- Target used to report the exact selected loss in whole security bits. -/
def samplerSecurityTarget : Rat :=
  (1 : Rat) / (((2 : Nat) ^ 182 : Nat) : Rat)

theorem selected_sampler_parameters :
    PaperProfile.arity.total = 15 /\
    samplerCoefficientCount = 54 /\
    samplerAttemptCount = 3 /\
    sampledCoefficientCount = 810 := by
  decide

/-- The exact accepted-domain size is divisible by five. This is the
arithmetic premise behind uniform mod-5 digits after rejection. -/
theorem accepted_domain_divisible_by_five :
    goldilocksModulus - 1 = acceptedQuotientCount * 5 :=
  acceptedDomain_factorization

/-- The complete per-fold sampler loss is at most `2^-182`. -/
theorem completeSamplerShortfallBound_le_target :
    completeSamplerShortfallBound <= samplerSecurityTarget := by
  norm_num [completeSamplerShortfallBound, sampledCoefficientCount,
    singleCoefficientExhaustionBound, samplerSecurityTarget,
    samplerAttemptCount, samplerCoefficientCount, PaperProfile.arity,
    goldilocksModulus]

/-- Final random-oracle probability premise needed by the complete selected
sampler. It records the result of the uniformity, independence, and finite
union-bound argument. It does not encode those distribution hypotheses
separately, because they are not derivable from deterministic Poseidon2 inside
Lean. -/
def GoldilocksRandomOracleSamplerContract
    (experiment : Experiment State) : Prop :=
  experiment.probability SamplerShortfall <= completeSamplerShortfallBound

theorem samplerShortfall_probability_le
    (experiment : Experiment State)
    (distribution : GoldilocksRandomOracleSamplerContract experiment) :
    experiment.probability SamplerShortfall <=
      completeSamplerShortfallBound :=
  distribution

/-- Under the named full-field random-oracle premise, the complete bounded
sampler fails with probability at most `2^-182`. -/
theorem samplerShortfall_probability_le_182_bits
    (experiment : Experiment State)
    (distribution : GoldilocksRandomOracleSamplerContract experiment) :
    experiment.probability SamplerShortfall <= samplerSecurityTarget :=
  Rat.le_trans
    (samplerShortfall_probability_le experiment distribution)
    completeSamplerShortfallBound_le_target

end Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentitySamplerSecurity
