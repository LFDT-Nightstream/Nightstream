import NightstreamFPrime.Spec.Folding.Nifs.NonInteractive.PiRlcSampler.ShortfallBound

/-!
Owns the scalar shortfall bound for 32 independent uniform Goldilocks
lanes. A finite bijection couples each ordered candidate pair to uniform
bits and can only remove rejections. No law is assigned to Poseidon2.
-/

namespace NightstreamFPrime.Spec.Folding.Nifs.NonInteractive.PiRlcSampler.FieldShortfall

open ProductionAlphabet Sampling FieldPairLaw

local instance : NeZero pairModulus := ⟨by decide⟩
local instance : NeZero chunkModulus := ⟨by decide⟩

/-- One field lane supplies two of the scalar's 64 candidate positions. -/
def fieldLaneCount : Nat := candidateBound / 2

abbrev FieldWindow := Fin fieldLaneCount → F
abbrev PairWindow := Fin fieldLaneCount → Fin pairModulus

theorem fieldLaneCount_eq : fieldLaneCount = 32 := by decide

/-- Flatten ordered low/high pairs in lane order into the actual candidate carrier. -/
def pairWindowEquiv : PairWindow ≃ ShortfallBound.Window :=
  ((Equiv.piCongrRight fun _ : Fin fieldLaneCount =>
      pairEquiv.symm.trans (finTwoArrowEquiv Chunk).symm).trans
    (Equiv.curry (Fin fieldLaneCount) (Fin 2) Chunk).symm).trans
      (Equiv.arrowCongr finProdFinEquiv (Equiv.refl Chunk))

def fieldCandidates (fields : FieldWindow) : ShortfallBound.Window :=
  pairWindowEquiv fun lane => low32 (fields lane)

private abbrev Blocks := (Fin (pairModulus - 1) × Fin pairModulus) ⊕ Fin 1

private def fieldBlocks : Blocks ≃ F :=
  ((Equiv.sumCongr finProdFinEquiv (Equiv.refl (Fin 1))).trans finSumFinEquiv).trans
    (finCongr (by
      rw [Nat.mul_comm]
      exact goldilocks_decomposition.symm))

private theorem low32_complete_block (block : Fin (pairModulus - 1))
    (residue : Fin pairModulus) :
    low32 (fieldBlocks (Sum.inl (block, residue))) = residue := by
  apply Fin.ext
  change (residue.val + pairModulus * block.val) % pairModulus = residue.val
  exact (Nat.add_mul_mod_self_left _ _ _).trans (Nat.mod_eq_of_lt residue.isLt)

private theorem low32_final_block (last : Fin 1) :
    low32 (fieldBlocks (Sum.inr last)) = 0 := by
  apply Fin.ext
  change ((pairModulus - 1) * pairModulus + last.val) % pairModulus = 0
  have zero : last.val = 0 := by omega
  simp [zero]

private def swapBlocks : Blocks × Fin pairModulus → Blocks × Fin pairModulus
  | (Sum.inl (block, residue), auxiliary) => (Sum.inl (block, auxiliary), residue)
  | (Sum.inr last, auxiliary) => (Sum.inr last, auxiliary)

private def blockSwap : Blocks × Fin pairModulus ≃ Blocks × Fin pairModulus where
  toFun := swapBlocks
  invFun := swapBlocks
  left_inv := by rintro ⟨(⟨block, residue⟩ | last), auxiliary⟩ <;> rfl
  right_inv := by rintro ⟨(⟨block, residue⟩ | last), auxiliary⟩ <;> rfl

/-- Swap a complete block's residue with the auxiliary pair. At the final
field value the field residue is zero and the auxiliary pair is unchanged. -/
def laneCoupling : F × Fin pairModulus ≃ F × Fin pairModulus :=
  ((Equiv.prodCongr fieldBlocks.symm (Equiv.refl (Fin pairModulus))).trans
    blockSwap).trans (Equiv.prodCongr fieldBlocks (Equiv.refl (Fin pairModulus)))

/-- A coupled field pair either equals its uniform-bit pair or is `(0,0)`. -/
theorem lane_coupling_residue (value : F) (auxiliary : Fin pairModulus) :
    low32 value = (laneCoupling (value, auxiliary)).2 ∨ low32 value = 0 := by
  obtain ⟨block, rfl⟩ := fieldBlocks.surjective value
  rcases block with ⟨block, residue⟩ | last
  · left
    rw [low32_complete_block]
    simp [laneCoupling, blockSwap, swapBlocks]
  · exact Or.inr (low32_final_block last)

/-- The coupled pair differs only at the single final field value and a
nonzero auxiliary residue. This retains the exact single-pair variation. -/
theorem lane_coupling_disagrees_iff (value : F) (auxiliary : Fin pairModulus) :
    low32 value ≠ (laneCoupling (value, auxiliary)).2 ↔
      value.val = goldilocksModulus - 1 ∧ auxiliary.val ≠ 0 := by
  obtain ⟨block, rfl⟩ := fieldBlocks.surjective value
  rcases block with ⟨block, residue⟩ | last
  · have below : (fieldBlocks (Sum.inl (block, residue))).val < goldilocksModulus - 1 := by
      change residue.val + pairModulus * block.val < goldilocksModulus - 1
      rw [goldilocks_decomposition, Nat.add_sub_cancel,
        Nat.mul_comm pairModulus (pairModulus - 1)]
      exact (finProdFinEquiv (block, residue)).isLt
    simp [low32_complete_block, laneCoupling, blockSwap, swapBlocks, ne_of_lt below]
  · have final : (fieldBlocks (Sum.inr last)).val = goldilocksModulus - 1 := by
      have zero : last.val = 0 := by omega
      change (pairModulus - 1) * pairModulus + last.val = goldilocksModulus - 1
      rw [zero, Nat.add_zero, goldilocks_decomposition, Nat.add_sub_cancel,
        Nat.mul_comm pairModulus (pairModulus - 1)]
    have coupled : (laneCoupling (fieldBlocks (Sum.inr last), auxiliary)).2 = auxiliary := by
      simp [laneCoupling, blockSwap, swapBlocks]
    rw [low32_final_block, coupled, final]
    simp only [eq_self_iff_true, true_and]
    constructor
    · intro different zero
      apply different
      apply Fin.ext
      exact zero.symm
    · intro nonzero equal
      exact nonzero (congrArg Fin.val equal).symm

/-- Independent copies of the finite lane bijection. -/
def windowCoupling : FieldWindow × PairWindow ≃ FieldWindow × PairWindow :=
  let split := Equiv.arrowProdEquivProdArrow (Fin fieldLaneCount)
    (fun _ => F) (fun _ => Fin pairModulus)
  (split.symm.trans (Equiv.piCongrRight fun _ => laneCoupling)).trans split

private theorem zero_pair_does_not_reject (side : Fin 2) :
    (finTwoArrowEquiv Chunk).symm (pairEquiv.symm (0 : Fin pairModulus)) side ≠
      ShortfallBound.rejection := by
  have pairZero : pairEquiv.symm (0 : Fin pairModulus) = ((0 : Chunk), 0) := by
    apply pairEquiv.symm_apply_eq.mpr
    rfl
  have constantZero : (finTwoArrowEquiv Chunk).symm ((0 : Chunk), 0) =
      fun _ => 0 := by
    apply (finTwoArrowEquiv Chunk).symm_apply_eq.mpr
    rfl
  rw [pairZero, constantZero]
  change (0 : Chunk) ≠ ShortfallBound.rejection
  decide

/-- The field comparison can only remove rejected positions, including
when both candidates from one lane occur in the rejection event. -/
theorem field_rejections_subset (fields : FieldWindow) (auxiliary : PairWindow) :
    ShortfallBound.rejectedPositions (fieldCandidates fields) ⊆
      ShortfallBound.rejectedPositions
        (pairWindowEquiv (windowCoupling (fields, auxiliary)).2) := by
  intro index member
  have rejected := (Finset.mem_filter.mp member).2
  apply Finset.mem_filter.mpr
  refine ⟨Finset.mem_univ _, ?_⟩
  let laneSide : Fin fieldLaneCount × Fin 2 := finProdFinEquiv.symm index
  change (finTwoArrowEquiv Chunk).symm
    (pairEquiv.symm (low32 (fields laneSide.1))) laneSide.2 =
      ShortfallBound.rejection at rejected
  change (finTwoArrowEquiv Chunk).symm
    (pairEquiv.symm (laneCoupling (fields laneSide.1, auxiliary laneSide.1)).2)
      laneSide.2 = ShortfallBound.rejection
  rcases lane_coupling_residue (fields laneSide.1) (auxiliary laneSide.1) with same | zero
  · rw [same] at rejected
    exact rejected
  · rw [zero] at rejected
    exact (zero_pair_does_not_reject laneSide.2 rejected).elim

private theorem field_shortfall_implies_coupled (fields : FieldWindow)
    (auxiliary : PairWindow)
    (failure : FirstAccepted.Shortfall verifier coefficientCount
      (List.ofFn (fieldCandidates fields))) :
    FirstAccepted.Shortfall verifier coefficientCount
      (List.ofFn (pairWindowEquiv (windowCoupling (fields, auxiliary)).2)) := by
  apply (ShortfallBound.shortfall_iff_eleven_rejections _).mpr
  exact ((ShortfallBound.shortfall_iff_eleven_rejections _).mp failure).trans
    (Finset.card_le_card (field_rejections_subset fields auxiliary))

private theorem coupling_card_le {Left Right : Type*} [Finite Left] [Finite Right]
    (coupling : Left × Right ≃ Left × Right)
    (leftEvent : Left → Prop) (rightEvent : Right → Prop)
    (preserves : ∀ left right, leftEvent left → rightEvent (coupling (left, right)).2) :
    Nat.card {left // leftEvent left} * Nat.card Right ≤
      Nat.card Left * Nat.card {right // rightEvent right} := by
  let injection : {left // leftEvent left} × Right →
      Left × {right // rightEvent right} := fun input =>
    ((coupling (input.1.val, input.2)).1,
      ⟨(coupling (input.1.val, input.2)).2,
        preserves input.1.val input.2 input.1.property⟩)
  have injective : Function.Injective injection := by
    intro first second same
    have sameImage := congrArg
      (fun output : Left × {right // rightEvent right} => (output.1, output.2.val)) same
    change coupling (first.1.val, first.2) =
      coupling (second.1.val, second.2) at sameImage
    have sameInput := coupling.injective sameImage
    apply Prod.ext
    · exact Subtype.ext (congrArg (fun input : Left × Right => input.1) sameInput)
    · exact congrArg (fun input : Left × Right => input.2) sameInput
  simpa only [Nat.card_prod] using Nat.card_le_card_of_injective injection injective

theorem field_window_cardinality : Nat.card FieldWindow = goldilocksModulus ^ fieldLaneCount := by
  rw [Nat.card_fun, Nat.card_fin, Nat.card_fin]

/-- Count the injection from field failures with auxiliary pairs to bit
failures with auxiliary fields. Both denominators remain exact. -/
theorem field_shortfall_count_comparison :
    Nat.card {fields : FieldWindow // FirstAccepted.Shortfall verifier coefficientCount
        (List.ofFn (fieldCandidates fields))} * chunkModulus ^ candidateBound ≤
      goldilocksModulus ^ fieldLaneCount *
        Nat.card {window : ShortfallBound.Window //
          FirstAccepted.Shortfall verifier coefficientCount (List.ofFn window)} := by
  have pairCard : Nat.card PairWindow = chunkModulus ^ candidateBound :=
    (Nat.card_congr pairWindowEquiv).trans ShortfallBound.window_cardinality
  have eventCard :
      Nat.card {pairs : PairWindow // FirstAccepted.Shortfall verifier coefficientCount
        (List.ofFn (pairWindowEquiv pairs))} =
      Nat.card {window : ShortfallBound.Window //
        FirstAccepted.Shortfall verifier coefficientCount (List.ofFn window)} :=
    Nat.card_congr (Equiv.subtypeEquivOfSubtype
      (p := fun window : ShortfallBound.Window =>
        FirstAccepted.Shortfall verifier coefficientCount (List.ofFn window)) pairWindowEquiv)
  have bound := coupling_card_le windowCoupling
    (fun fields => FirstAccepted.Shortfall verifier coefficientCount
      (List.ofFn (fieldCandidates fields)))
    (fun pairs => FirstAccepted.Shortfall verifier coefficientCount
      (List.ofFn (pairWindowEquiv pairs))) field_shortfall_implies_coupled
  rw [pairCard, field_window_cardinality, eventCard] at bound
  exact bound

/-- Scalar failure frequency for 32 IID uniform fields. Within each lane,
the two ordered candidates retain the exact Goldilocks low32 law. -/
noncomputable def iidFieldShortfallProbability : ℚ :=
  (Nat.card {fields : FieldWindow // FirstAccepted.Shortfall verifier coefficientCount
    (List.ofFn (fieldCandidates fields))} : ℚ) /
      (goldilocksModulus : ℚ) ^ fieldLaneCount

theorem iid_field_shortfall_probability_le_iid_bit :
    iidFieldShortfallProbability ≤ ShortfallBound.iidBitShortfallProbability := by
  have counted :
      (Nat.card {fields : FieldWindow // FirstAccepted.Shortfall verifier coefficientCount
        (List.ofFn (fieldCandidates fields))} : ℚ) * (chunkModulus : ℚ) ^ candidateBound ≤
      (goldilocksModulus : ℚ) ^ fieldLaneCount *
        (Nat.card {window : ShortfallBound.Window //
          FirstAccepted.Shortfall verifier coefficientCount (List.ofFn window)} : ℚ) := by
    exact_mod_cast field_shortfall_count_comparison
  have fieldPositive : 0 < (goldilocksModulus : ℚ) ^ fieldLaneCount :=
    pow_pos (by norm_num [goldilocksModulus]) _
  have bitPositive : 0 < (chunkModulus : ℚ) ^ candidateBound :=
    pow_pos (by norm_num [chunkModulus]) _
  unfold iidFieldShortfallProbability ShortfallBound.iidBitShortfallProbability
  apply (div_le_div_iff₀ fieldPositive bitPositive).mpr
  exact counted.trans_eq (mul_comm _ _)

/-- The existing eleven-rejection bound transfers with no added bias term. -/
theorem iid_field_shortfall_probability_le :
    iidFieldShortfallProbability ≤
      (Nat.choose candidateBound 11 : ℚ) / (chunkModulus : ℚ) ^ 11 :=
  iid_field_shortfall_probability_le_iid_bit.trans
    ShortfallBound.iid_bit_shortfall_probability_le

end NightstreamFPrime.Spec.Folding.Nifs.NonInteractive.PiRlcSampler.FieldShortfall
