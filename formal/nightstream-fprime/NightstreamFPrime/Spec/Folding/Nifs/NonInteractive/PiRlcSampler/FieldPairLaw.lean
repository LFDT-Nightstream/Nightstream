import Mathlib.Logic.Equiv.Fin.Basic
import Mathlib.SetTheory.Cardinal.Finite
import Mathlib.Tactic.Ring
import NightstreamFPrime.Spec.AjtaiSetupV1.ReductionBias
import NightstreamFPrime.Spec.Folding.Nifs.NonInteractive.PiRlcSampler.ProductionAlphabet

/-!
Owns the exact finite comparison law for the ordered low two 16-bit chunks
of a uniform Goldilocks element. Counts use complete residue blocks and the
single final input; no field-sized set is enumerated. This module assigns no
probability law to Poseidon2 and does not change the bounded sampler.
-/

namespace NightstreamFPrime.Spec.Folding.Nifs.NonInteractive.PiRlcSampler.FieldPairLaw

open ProductionAlphabet

/-- The joint carrier of the two 16-bit candidates read from one field lane. -/
def pairModulus : Nat := chunkModulus * chunkModulus

theorem pairModulus_eq_pow : pairModulus = 2 ^ 32 := by decide

theorem goldilocks_decomposition :
    goldilocksModulus = pairModulus * (pairModulus - 1) + 1 := by decide

private theorem goldilocks_quotient :
    goldilocksModulus / pairModulus = pairModulus - 1 := by decide

private theorem goldilocks_remainder :
    goldilocksModulus % pairModulus = 1 := by decide

/-- Canonical low-32-bit residue of a field element. -/
def low32 (value : F) : Fin pairModulus :=
  ⟨value.val % pairModulus, Nat.mod_lt _ (by decide)⟩

/-- Little-endian pair order: first chunk plus `2^16` times second chunk. -/
def pairEquiv : Chunk × Chunk ≃ Fin pairModulus :=
  (Equiv.prodComm Chunk Chunk).trans finProdFinEquiv

/-- The two ordered candidates, before rejection or reduction modulo five. -/
def candidates (value : F) : Chunk × Chunk :=
  pairEquiv.symm (low32 value)

/-- These are exactly bits 0--15 and bits 16--31 of the canonical lane. -/
theorem candidate_values (value : F) :
    (candidates value).1.val = value.val % chunkModulus ∧
      (candidates value).2.val = (value.val / chunkModulus) % chunkModulus := by
  change (value.val % (chunkModulus * chunkModulus)) % chunkModulus =
      value.val % chunkModulus ∧
    (value.val % (chunkModulus * chunkModulus)) / chunkModulus =
      (value.val / chunkModulus) % chunkModulus
  exact ⟨Nat.mod_mul_right_mod _ _ _, Nat.mod_mul_right_div_self _ _ _⟩

theorem candidates_eq_iff (value : F) (pair : Chunk × Chunk) :
    candidates value = pair ↔ low32 value = pairEquiv pair :=
  pairEquiv.symm_apply_eq

private theorem pair_zero_iff (pair : Chunk × Chunk) :
    (pairEquiv pair).val = 0 ↔ pair.1.val = 0 ∧ pair.2.val = 0 := by
  change pair.1.val + 65536 * pair.2.val = 0 ↔ _
  omega

private theorem residue_block_count (residue : Fin pairModulus) :
    Nat.count (fun value => value % pairModulus = residue.val) pairModulus = 1 := by
  rw [Nat.count_eq_card_filter_range]
  have singleton : (Finset.range pairModulus).filter
      (fun value => value % pairModulus = residue.val) = {residue.val} := by
    ext value
    simp only [Finset.mem_filter, Finset.mem_range, Finset.mem_singleton]
    constructor
    · rintro ⟨below, same⟩
      simpa only [Nat.mod_eq_of_lt below] using same
    · intro same
      subst value
      exact ⟨residue.isLt, Nat.mod_eq_of_lt residue.isLt⟩
  rw [singleton, Finset.card_singleton]

private theorem residue_preimage_count (residue : Fin pairModulus) :
    Nat.count (fun value => value % pairModulus = residue.val) goldilocksModulus =
      pairModulus - 1 + if residue.val = 0 then 1 else 0 := by
  rw [AjtaiSetupV1.ReductionBias.event_count goldilocksModulus pairModulus
    (fun value => value = residue.val), goldilocks_quotient,
    goldilocks_remainder, residue_block_count, Nat.mul_one, Nat.count_one]
  simp only [Nat.zero_mod, eq_comm]

private def fieldFiberEquiv (residue : Fin pairModulus) :
    {value : Nat // value < goldilocksModulus ∧ value % pairModulus = residue.val} ≃
      {value : F // low32 value = residue} where
  toFun value := ⟨⟨value.val, value.property.1⟩, Fin.ext value.property.2⟩
  invFun value := ⟨value.val.val, value.val.isLt, congrArg Fin.val value.property⟩
  left_inv value := by apply Subtype.ext; rfl
  right_inv value := by apply Subtype.ext; apply Fin.ext; rfl

/-- Each nonzero low-32 residue has `M-1` field preimages; zero has one more. -/
theorem field_preimage_count (residue : Fin pairModulus) :
    Nat.card {value : F // low32 value = residue} =
      pairModulus - 1 + if residue.val = 0 then 1 else 0 := by
  letI := Nat.CountSet.fintype
    (fun value => value % pairModulus = residue.val) goldilocksModulus
  rw [← Nat.card_congr (fieldFiberEquiv residue), Nat.card_eq_fintype_card,
    ← Nat.count_eq_card_fintype]
  exact residue_preimage_count residue

theorem field_zero_preimage_count :
    Nat.card {value : F // low32 value = ⟨0, by decide⟩} = pairModulus := by
  rw [field_preimage_count]
  norm_num [pairModulus, chunkModulus]

theorem field_nonzero_preimage_count (residue : Fin pairModulus)
    (nonzero : residue.val ≠ 0) :
    Nat.card {value : F // low32 value = residue} = pairModulus - 1 := by
  simp only [field_preimage_count, nonzero, if_false, Nat.add_zero]

/-- The exceptional pair is exactly `(0,0)` in the ordered candidate carrier. -/
theorem candidate_preimage_count (pair : Chunk × Chunk) :
    Nat.card {value : F // candidates value = pair} =
      pairModulus - 1 + if pair.1.val = 0 ∧ pair.2.val = 0 then 1 else 0 := by
  rw [Nat.card_congr (Equiv.subtypeEquivRight
    (fun value => candidates_eq_iff value pair)), field_preimage_count]
  simp only [pair_zero_iff]

/-- Point mass in the comparison experiment that samples one uniform field element. -/
noncomputable def candidateMass (pair : Chunk × Chunk) : ℚ :=
  (Nat.card {value : F // candidates value = pair} : ℚ) / goldilocksModulus

/-- The pair law is a uniform component of weight `1-1/q` plus mass `1/q`
at `(0,0)`. It is not the independent uniform 16-bit pair law. -/
theorem candidate_mass_eq_mixture (pair : Chunk × Chunk) :
    candidateMass pair =
      (1 - 1 / (goldilocksModulus : ℚ)) / pairModulus +
        if pair.1.val = 0 ∧ pair.2.val = 0 then 1 / (goldilocksModulus : ℚ) else 0 := by
  unfold candidateMass
  rw [candidate_preimage_count]
  split_ifs <;> norm_num [pairModulus, chunkModulus, goldilocksModulus]

/-- The complete mixture law for every event on low-32 residues, expressed
as exact rational frequencies. No distribution assumption occurs in the type. -/
theorem event_frequency_eq_mixture (event : Nat → Prop) [DecidablePred event] :
    AjtaiSetupV1.ReductionBias.frequency goldilocksModulus pairModulus event =
      (1 - 1 / (goldilocksModulus : ℚ)) *
        AjtaiSetupV1.ReductionBias.frequency pairModulus pairModulus event +
      if event 0 then 1 / (goldilocksModulus : ℚ) else 0 := by
  unfold AjtaiSetupV1.ReductionBias.frequency
  rw [AjtaiSetupV1.ReductionBias.event_count goldilocksModulus pairModulus,
    goldilocks_quotient, goldilocks_remainder, Nat.count_one]
  simp only [Nat.zero_mod]
  split_ifs <;> norm_num [pairModulus, chunkModulus, goldilocksModulus] <;> ring

/-- Exact total-variation distance from the uniform ordered candidate pair. -/
def pairDeviation : ℚ :=
  ((pairModulus : ℚ) - 1) / ((pairModulus : ℚ) * goldilocksModulus)

private theorem block_count_bounds (event : Nat → Prop) [DecidablePred event] :
    (if event 0 then 1 else 0) ≤
        Nat.count (fun value => event (value % pairModulus)) pairModulus ∧
      Nat.count (fun value => event (value % pairModulus)) pairModulus ≤
        pairModulus - 1 + (if event 0 then 1 else 0) := by
  have split := Nat.count_succ'
    (fun value => event (value % pairModulus)) (pairModulus - 1)
  have successor : pairModulus - 1 + 1 = pairModulus := by decide
  rw [successor] at split
  simp only [Nat.zero_mod] at split
  have tailBound :
      Nat.count (fun value => event ((value + 1) % pairModulus)) (pairModulus - 1) ≤
        pairModulus - 1 := Nat.count_le _
  by_cases zeroEvent : event 0 <;>
    simp only [zeroEvent, if_true, if_false] at split ⊢ <;> omega

/-- Sharp event bound for the uniform-field comparison law. -/
theorem event_frequency_error_le (event : Nat → Prop) [DecidablePred event] :
    |AjtaiSetupV1.ReductionBias.frequency goldilocksModulus pairModulus event -
      AjtaiSetupV1.ReductionBias.frequency pairModulus pairModulus event| ≤
        pairDeviation := by
  let count := Nat.count (fun value => event (value % pairModulus)) pairModulus
  have bounds := block_count_bounds event
  have lower : ((if event 0 then 1 else 0 : Nat) : ℚ) ≤ (count : ℚ) := by
    exact_mod_cast bounds.1
  have upper : (count : ℚ) ≤
      ((pairModulus - 1 + (if event 0 then 1 else 0) : Nat) : ℚ) := by
    exact_mod_cast bounds.2
  rw [event_frequency_eq_mixture]
  change |(1 - 1 / (goldilocksModulus : ℚ)) * ((count : ℚ) / pairModulus) +
      (if event 0 then 1 / (goldilocksModulus : ℚ) else 0) -
      (count : ℚ) / pairModulus| ≤ pairDeviation
  by_cases zeroEvent : event 0
  · simp only [zeroEvent, if_true] at lower upper ⊢
    norm_num [pairDeviation, pairModulus, chunkModulus, goldilocksModulus] at lower upper ⊢
    have lowerQ : (1 : ℚ) ≤ count := by exact_mod_cast lower
    have upperQ : (count : ℚ) ≤ 4294967296 := by exact_mod_cast upper
    rw [abs_le]
    constructor <;> linarith only [lowerQ, upperQ]
  · simp only [zeroEvent, if_false] at lower upper ⊢
    norm_num [pairDeviation, pairModulus, chunkModulus, goldilocksModulus] at lower upper ⊢
    have lowerQ : (0 : ℚ) ≤ count := Nat.cast_nonneg _
    have upperQ : (count : ℚ) ≤ 4294967295 := by exact_mod_cast upper
    rw [abs_le]
    constructor <;> linarith only [lowerQ, upperQ]

/-- The zero-residue event attains the sharp bound. -/
theorem zero_event_frequency_error :
    |AjtaiSetupV1.ReductionBias.frequency goldilocksModulus pairModulus (fun value => value = 0) -
      AjtaiSetupV1.ReductionBias.frequency pairModulus pairModulus (fun value => value = 0)| =
        pairDeviation := by
  have blockCount : Nat.count (fun value => value % pairModulus = 0) pairModulus = 1 :=
    residue_block_count ⟨0, by decide⟩
  rw [event_frequency_eq_mixture]
  unfold AjtaiSetupV1.ReductionBias.frequency
  rw [blockCount]
  norm_num [pairDeviation, pairModulus, chunkModulus, goldilocksModulus]

/-- The least common event bound is exactly `pairDeviation`, the finite
total-variation characterization. No supremum or field enumeration is needed. -/
theorem event_bound_iff (bound : ℚ) :
    (∀ (event : Nat → Prop) [DecidablePred event],
      |AjtaiSetupV1.ReductionBias.frequency goldilocksModulus pairModulus event -
        AjtaiSetupV1.ReductionBias.frequency pairModulus pairModulus event| ≤ bound) ↔
      pairDeviation ≤ bound := by
  constructor
  · intro everyEvent
    have zeroBound := everyEvent (fun value : Nat => value = 0)
    rw [zero_event_frequency_error] at zeroBound
    exact zeroBound
  · intro enough event decidableEvent
    exact (event_frequency_error_le event).trans enough

/-- Any deterministic decoding of the ordered pair preserves the event
bound. The target type may include an explicit reject or abort value. -/
theorem decoded_event_frequency_error_le {Output : Type*}
    (decode : Chunk × Chunk → Output) (event : Output → Prop) [DecidablePred event] :
    let preimage := fun value : Nat =>
      event (decode (pairEquiv.symm
        ⟨value % pairModulus, Nat.mod_lt _ (by decide)⟩))
    |AjtaiSetupV1.ReductionBias.frequency goldilocksModulus pairModulus preimage -
      AjtaiSetupV1.ReductionBias.frequency pairModulus pairModulus preimage| ≤
        pairDeviation := by
  exact event_frequency_error_le _

end NightstreamFPrime.Spec.Folding.Nifs.NonInteractive.PiRlcSampler.FieldPairLaw
