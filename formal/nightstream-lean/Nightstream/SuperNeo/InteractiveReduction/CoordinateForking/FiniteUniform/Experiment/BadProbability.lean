import Nightstream.SuperNeo.InteractiveReduction.CoordinateForking.FiniteUniform.Experiment

/-! Sharp finite-uniform probability bound for failed coordinate forks. -/

set_option autoImplicit false

namespace Nightstream.SuperNeo.InteractiveReduction.CoordinateForking.FiniteUniform

open Nightstream.SuperNeo.InteractiveReduction.FiniteUniform
open Nightstream.SuperNeo.InteractiveReduction.CoordinateForking.FiniteWords

universe uScalar

variable {Scalar : Type uScalar} {coordinates : Nat}

theorem bad_probability_le_sharp
    (alphabet : Support Scalar)
    (accepts : ChallengeVector Scalar coordinates -> Bool) :
    (forkExperiment alphabet coordinates).probabilityBool
        (fun seed => wordBad accepts seed.val) ≤
      ratio coordinates alphabet.cardinality := by
  let words := vectors (cyclicPointSupport alphabet).values coordinates
  have numeratorCount :
      (forkExperiment alphabet coordinates).countBool
          (fun seed => wordBad accepts seed.val) =
        words.countP (wordBad accepts) := by
    unfold Experiment.countBool forkExperiment forkSeedSupport Support.uniform
    change words.attach.countP
        (fun seed => wordBad accepts seed.val) =
      words.countP (wordBad accepts)
    exact List.countP_attach (l := words) (p := wordBad accepts)
  have supportCardinality :
      (forkExperiment alphabet coordinates).support.cardinality =
        alphabet.cardinality ^ coordinates := by
    unfold forkExperiment forkSeedSupport Support.uniform
    change (pointWordSupport alphabet coordinates).values.attach.length =
      alphabet.cardinality ^ coordinates
    rw [List.length_attach]
    change (vectors (cyclicPointSupport alphabet).values coordinates).length =
      alphabet.cardinality ^ coordinates
    rw [vectors_length]
    have receipt :
        (cyclicPointSupport alphabet).values.length =
          alphabet.cardinality := by
      simpa [Support.cardinality] using
        cyclicPointSupport_cardinality alphabet
    rw [receipt]
  unfold Experiment.probabilityBool
  rw [numeratorCount, supportCardinality]
  cases coordinates with
  | zero =>
      have countZero : words.countP (wordBad accepts) = 0 := by
        exact Nat.eq_zero_of_le_zero
          (by simpa [words] using wordBad_count_le alphabet accepts)
      rw [countZero]
      simp [ratio, Rat.div_def]
  | succ coordinates =>
      apply (div_le_iff_of_pos (Rat.natCast_pos.mpr
        (Nat.pow_pos alphabet.cardinality_pos))).mpr
      have castBound := Rat.natCast_le_natCast.mpr
        (wordBad_count_le alphabet accepts)
      refine Rat.le_trans castBound ?_
      simp only [Nat.add_sub_cancel, Nat.pow_succ, Rat.natCast_mul]
      unfold ratio
      have cardinalityNe : (alphabet.cardinality : Rat) ≠ 0 :=
        Rat.ne_of_gt (Rat.natCast_pos.mpr alphabet.cardinality_pos)
      calc
        ((coordinates + 1 : Nat) : Rat) *
              (alphabet.cardinality ^ coordinates : Nat) =
            ((((coordinates + 1 : Nat) : Rat) /
              (alphabet.cardinality : Rat)) *
              (alphabet.cardinality : Rat)) *
                (alphabet.cardinality ^ coordinates : Nat) := by
          rw [Rat.div_mul_cancel cardinalityNe]
        _ = (((coordinates + 1 : Nat) : Rat) /
              (alphabet.cardinality : Rat)) *
            ((alphabet.cardinality : Rat) *
              (alphabet.cardinality ^ coordinates : Nat)) :=
          Rat.mul_assoc _ _ _
        _ = (((coordinates + 1 : Nat) : Rat) /
              (alphabet.cardinality : Rat)) *
            ((alphabet.cardinality ^ coordinates : Nat) *
              (alphabet.cardinality : Rat)) := by
          rw [Rat.mul_comm (alphabet.cardinality : Rat)
            (alphabet.cardinality ^ coordinates : Nat)]
        _ ≤ _ := Rat.le_refl

end Nightstream.SuperNeo.InteractiveReduction.CoordinateForking.FiniteUniform
