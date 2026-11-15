import NightstreamFPrime.Export.BoundedDepth.RandomOracleSetup
import NightstreamFPrime.Spec.AjtaiSetupV1.Programming
import NightstreamFPrime.Spec.AjtaiSetupV1.Rejection
import NightstreamFPrime.Spec.Folding.Schedule
import NightstreamFPrime.Spec.RingStrongSet.LowNormInvertibility
import NightstreamFPrime.Spec.SumCheck.Abstract
import NightstreamFPrime.Spec.SumCheck.FieldCausal

/-! # Claims: foundations

The algebra, setup and sum-check facts that the other claims use.

- Paper: SuperNeo v1.2 Theorem 4 (`low_norm_invertible`); the sum-check protocol.
- Assumptions: none for the algebra and sum-check claims. The `setup_*` claims hold in the
  experiment of one uniform SHAKE128 stream (`[KECCAK]`, `SECURITY_MODEL.md` trust item 1).
- Not a claim: the hardness of Module-SIS (`[MSIS]`). The reduction from a binding collision to a
  kernel vector below `chain_norm` is the definition
  `Spec.RingRelation.PiRLCAlgebra.Binding.chainCollision_to_shortKernel`, which returns the
  vector, so it is not a theorem of this folder. -/

namespace NightstreamFPrime.Claims.Foundations

open Spec
attribute [local instance] Classical.propDecidable

/-- SuperNeo v1.2 Theorem 4, low-norm invertibility for the strong sampling set, at the
production ring. -/
theorem low_norm_invertible :
    RingStrongSet.LowNormInvertibility :=
  @Spec.RingStrongSet.lowNormInvertibility

/-- The chain norm of the Module-SIS instance at the production cap is 25,165,824. -/
theorem chain_norm :
    Folding.Schedule.msisChainBound productionGlobalParams
          Folding.Schedule.productionLightFoldCap =
        25165824 :=
  @Spec.Folding.Schedule.production_msisChainBound

/-- The fallback bound of one key element of the Ajtai setup is below `2^-1155`. -/
theorem setup_fallback_bound :
    AjtaiSetupV1.Rejection.fallbackBound < 1 / 2 ^ 1155 :=
  @Spec.AjtaiSetupV1.Rejection.fallbackBound_lt

/-- The element of a uniform stream is within `2 · fallbackBound` of a uniform ring element, in
total absolute difference of the two laws. -/
theorem setup_element_bias :
    ∑ value : RingF,
          |AjtaiSetupV1.Programming.sampledWeight AjtaiSetupV1.Rejection.element value -
              1 / (↑(Fintype.card RingF) : ℚ)| ≤
        2 * AjtaiSetupV1.Rejection.fallbackBound :=
  @Spec.AjtaiSetupV1.Programming.element_bias_le

/-- At the bounded-depth scheme's key width the programming error is below `2^-1129`. -/
theorem setup_error :
    Export.BoundedDepth.RandomOracleSetup.programmingError
          Export.BoundedDepth.StepRelation.keyWidth
          Export.BoundedDepth.StepRelation.keyPublicFits <
        1 / 2 ^ 1129 :=
  @Export.BoundedDepth.RandomOracleSetup.programmingError_lt

/-- An accepted sum-check transcript on a truth path with a false claim has a round with a bad
challenge. -/
theorem sumcheck_sound.{uChallenge, uValue}
    {Challenge : Type uChallenge}
    {Value : Type uValue}
    (ops : SumCheck.Ops Challenge Value)
    (transcript : SumCheck.Instance Challenge Value)
    (accepted : SumCheck.Accepted ops transcript)
    (truth : SumCheck.TruthPath ops transcript)
    (falseClaim : ¬SumCheck.Claim.True transcript) :
    ∃ (round : SumCheck.Round Challenge Value), SumCheck.BadChallenge transcript round :=
  @Spec.SumCheck.false_acceptance_implies_bad_challenge Challenge Value ops transcript accepted
    truth falseClaim

/-- A uniform challenge from `samples` hits a bad point of one round with probability at most
`degree / |samples|`. -/
theorem sumcheck_hit_probability
    {degree : ℕ}
    (samples : Finset K)
    (q : List K → K)
    (fixed : List K)
    (remaining : ℕ)
    (message : SumCheck.Finite.FixedPolynomial K degree)
    (semantic : SumCheck.Finite.FixedPolynomial K degree)
    (represents : SumCheck.Finite.FixedPhase.Represents SumCheck.Finite.FieldRoots.ops semantic
        (SumCheck.Finite.FieldCausal.expected q fixed remaining)) :
    (samples.expect fun (challenge : K) =>
          if SumCheck.Finite.FieldCausal.Hit q fixed remaining message challenge then 1 else 0) ≤
        HDiv.hDiv (α := ℝ) (↑degree : ℝ) (↑samples.card : ℝ) :=
  @Spec.SumCheck.Finite.FieldCausal.hit_probability_le degree samples q fixed remaining message
    semantic represents

end NightstreamFPrime.Claims.Foundations
