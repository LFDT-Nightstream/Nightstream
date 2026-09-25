import Mathlib.SetTheory.Cardinal.Finite
import NightstreamFPrime.Spec.Profile
import NightstreamFPrime.Spec.Phi81StrongSet

/-! Counts the actual sampler carrier and its injective Phi81 image, then
connects that count to the selected production profile. No set is enumerated. -/

namespace NightstreamFPrime.Spec.Phi81StrongSet

open Folding.Nifs.NonInteractive.PiRlcSampler
open ProductionAlphabet ProductionStrongSet

/-- The sampler has five choices at each of its 54 coefficient positions. -/
theorem scalar_cardinality : Nat.card Scalar = productionChallengeSetCardinality := by
  change Nat.card (Fin coefficientCount → Fin alphabetSize) = _
  rw [Nat.card_fun, Nat.card_fin, Nat.card_fin]
  rfl

/-- Ring membership has exactly the sampler's cardinality because the
embedding is injective and membership is precisely its image. -/
theorem productionMember_cardinality :
    Nat.card {value : RingF // ProductionMember value} =
      productionChallengeSetCardinality := by
  let embed : Scalar → {value : RingF // ProductionMember value} :=
    fun scalar => ⟨embedScalar scalar, embedScalar_member scalar⟩
  have bijective : Function.Bijective embed := by
    constructor
    · intro left right equal
      exact embedScalar_injective (congrArg Subtype.val equal)
    · rintro ⟨value, scalar, equal⟩
      refine ⟨scalar, ?_⟩
      exact Subtype.ext equal.symm
  calc
    Nat.card {value : RingF // ProductionMember value} = Nat.card Scalar :=
      (Nat.card_eq_of_bijective embed bijective).symm
    _ = productionChallengeSetCardinality := scalar_cardinality

/-- The profile's bit descriptor is the floor of the binary logarithm of
the actual set size, expressed with exact integer inequalities. -/
theorem productionMember_cardinality_bits :
    2 ^ productionProfile.challengeSetBitsFloor ≤
        Nat.card {value : RingF // ProductionMember value} ∧
      Nat.card {value : RingF // ProductionMember value} <
        2 ^ (productionProfile.challengeSetBitsFloor + 1) := by
  rw [productionMember_cardinality]
  decide

end NightstreamFPrime.Spec.Phi81StrongSet
