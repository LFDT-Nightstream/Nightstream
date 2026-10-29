import Mathlib.SetTheory.Cardinal.Finite
import NightstreamFPrime.Spec.Profile
import NightstreamFPrime.Spec.Phi81StrongSet

namespace NightstreamFPrime.Spec.Phi81StrongSet

theorem scalar_cardinality : Nat.card Scalar = productionChallengeSetCardinality := by
  change Nat.card (Fin coefficientCount → Fin alphabetSize) = _
  rw [Nat.card_fun, Nat.card_fin, Nat.card_fin]
  rfl

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

theorem productionMember_cardinality_bits :
    2 ^ productionProfile.challengeSetBitsFloor ≤
        Nat.card {value : RingF // ProductionMember value} ∧
      Nat.card {value : RingF // ProductionMember value} <
        2 ^ (productionProfile.challengeSetBitsFloor + 1) := by
  rw [productionMember_cardinality]
  decide

end NightstreamFPrime.Spec.Phi81StrongSet
