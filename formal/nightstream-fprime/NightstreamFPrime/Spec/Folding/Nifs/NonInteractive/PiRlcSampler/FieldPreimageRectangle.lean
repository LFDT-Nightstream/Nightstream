import Mathlib.Logic.Equiv.Prod
import Mathlib.Logic.Equiv.Sum
import NightstreamFPrime.Spec.Folding.Nifs.NonInteractive.PiRlcSampler.FieldPairLaw

/-!
Owns arithmetic indexing of the four candidate classes used by the success
and abort counting tables, and the exact ordered field-preimage rectangle
bijection. Common ranks put the low index first, then high, then field block;
the optional final rank is the extra field `q-1`. No words are enumerated.
Random sampling, stored execution costs, complete decoder fibers, and any
Poseidon2 distribution or Fiat–Shamir model remain separate obligations.
-/

namespace NightstreamFPrime.Spec.Folding.Nifs.NonInteractive.PiRlcSampler.FieldPreimageRectangle

open ProductionAlphabet FieldPairLaw

/-- Exactly the candidate classes needed by the two counting tables. -/
inductive ChunkClass where
  | reject
  | residue (digit : Coefficient)
  | accepted
  | all

namespace ChunkClass

def size : ChunkClass → Nat
  | .reject => 1
  | .residue _ => acceptedQuotientCount
  | .accepted => rejectionBucket
  | .all => chunkModulus

def Member : ChunkClass → Chunk → Prop
  | .reject, chunk => chunk.val = rejectionBucket
  | .residue digit, chunk => verifier.accepts chunk = true ∧ symbol chunk = digit
  | .accepted, chunk => verifier.accepts chunk = true
  | .all, _ => True

instance (kind : ChunkClass) (chunk : Chunk) : Decidable (kind.Member chunk) := by
  cases kind <;> unfold Member <;> infer_instance

def HasZero : ChunkClass → Prop
  | .reject => False
  | .residue digit => digit.val = 0
  | .accepted => True
  | .all => True

instance (kind : ChunkClass) : Decidable kind.HasZero := by
  cases kind <;> unfold HasZero <;> infer_instance

private def rejectionEquiv : Fin 1 ≃ {chunk : Chunk // chunk.val = rejectionBucket} where
  toFun _ := ⟨⟨rejectionBucket, by decide⟩, rfl⟩
  invFun _ := ⟨0, by decide⟩
  left_inv index := by apply Fin.ext; have := index.isLt; omega
  right_inv chunk := by apply Subtype.ext; apply Fin.ext; exact chunk.property.symm

private def residueEquiv (digit : Coefficient) :
    Fin acceptedQuotientCount ≃
      {chunk : Chunk // verifier.accepts chunk = true ∧ symbol chunk = digit} where
  toFun index :=
    ⟨(combine (index, digit)).val, (combine (index, digit)).property,
      congrArg Prod.snd (factor_combine (index, digit))⟩
  invFun chunk := (factor ⟨chunk.val, chunk.property.1⟩).1
  left_inv index := congrArg Prod.fst (factor_combine (index, digit))
  right_inv chunk := by
    apply Subtype.ext
    let accepted : AcceptedChunk := ⟨chunk.val, chunk.property.1⟩
    have coordinates : ((factor accepted).1, digit) = factor accepted := by
      apply Prod.ext
      · rfl
      · exact chunk.property.2.symm
    change (combine ((factor accepted).1, digit)).val = accepted.val
    exact congrArg (fun value : AcceptedChunk => value.val)
      ((congrArg combine coordinates).trans (combine_factor accepted))

private def acceptedEquiv : Fin rejectionBucket ≃ AcceptedChunk where
  toFun index :=
    ⟨⟨index.val, by have := index.isLt; change index.val < 65536; change index.val < 65535 at this; omega⟩,
      (accepts_eq_true_iff _).mpr index.isLt⟩
  invFun chunk := ⟨chunk.val.val, chunk.val_lt_rejectionBucket⟩
  left_inv _ := rfl
  right_inv _ := rfl

private def allEquiv : Chunk ≃ {_chunk : Chunk // True} where
  toFun chunk := ⟨chunk, trivial⟩
  invFun chunk := chunk.val
  left_inv _ := rfl
  right_inv _ := rfl

/-- Class indexing uses a constant, `5*index+digit`, or the identity. Its
inverse uses the existing accepted quotient, or the identity. -/
def indexEquiv (kind : ChunkClass) : Fin kind.size ≃ {chunk : Chunk // kind.Member chunk} :=
  match kind with
  | .reject => rejectionEquiv
  | .residue digit => residueEquiv digit
  | .accepted => acceptedEquiv
  | .all => allEquiv

theorem index_value (kind : ChunkClass) (index : Fin kind.size) :
    (kind.indexEquiv index).val.val =
      match kind with
      | .reject => rejectionBucket
      | .residue digit => index.val * alphabetSize + digit.val
      | .accepted => index.val
      | .all => index.val := by
  cases kind <;> rfl

theorem member_zero_iff (kind : ChunkClass) :
    kind.Member ⟨0, by decide⟩ ↔ kind.HasZero := by
  cases kind with
  | reject => decide
  | residue digit =>
      change (verifier.accepts ⟨0, by decide⟩ = true ∧
        symbol ⟨0, by decide⟩ = digit) ↔ digit.val = 0
      constructor
      · rintro ⟨_, same⟩
        exact (congrArg Fin.val same).symm
      · intro zero
        exact ⟨by decide, Fin.ext zero.symm⟩
  | accepted => decide
  | all => decide

theorem member_card (kind : ChunkClass) :
    Nat.card {chunk : Chunk // kind.Member chunk} = kind.size := by
  rw [← Nat.card_congr kind.indexEquiv, Nat.card_fin]

end ChunkClass

/-- One extra field exists exactly when both candidate classes contain zero. -/
def zeroBonus (low high : ChunkClass) : Nat :=
  if low.HasZero ∧ high.HasZero then 1 else 0

/-- The common words have `M-1` preimages; the zero word has one extra. -/
def size (low high : ChunkClass) : Nat :=
  (pairModulus - 1) * (low.size * high.size) + zeroBonus low high

def Member (low high : ChunkClass) (value : F) : Prop :=
  low.Member (candidates value).1 ∧ high.Member (candidates value).2

private abbrev Blocks := (Fin (pairModulus - 1) × Fin pairModulus) ⊕ Fin 1

private def fieldBlocks : Blocks ≃ F :=
  ((Equiv.sumCongr finProdFinEquiv (Equiv.refl (Fin 1))).trans finSumFinEquiv).trans
    (finCongr (by rw [Nat.mul_comm]; exact goldilocks_decomposition.symm))

private theorem fieldBlocks_common_candidates (block : Fin (pairModulus - 1))
    (word : Fin pairModulus) :
    candidates (fieldBlocks (Sum.inl (block, word))) = pairEquiv.symm word := by
  apply congrArg pairEquiv.symm
  apply Fin.ext
  change (word.val + pairModulus * block.val) % pairModulus = word.val
  exact (Nat.add_mul_mod_self_left _ _ _).trans (Nat.mod_eq_of_lt word.isLt)

private theorem fieldBlocks_extra_candidates (last : Fin 1) :
    candidates (fieldBlocks (Sum.inr last)) = (⟨0, by decide⟩, ⟨0, by decide⟩) := by
  rw [candidates_eq_iff]
  apply Fin.ext
  change ((pairModulus - 1) * pairModulus + last.val) % pairModulus = 0
  simp

private def PairMember (low high : ChunkClass) (pair : Chunk × Chunk) : Prop :=
  low.Member pair.1 ∧ high.Member pair.2

private def blockMember (low high : ChunkClass) : Blocks → Prop
  | .inl (_, word) => PairMember low high (pairEquiv.symm word)
  | .inr _ => low.HasZero ∧ high.HasZero

private theorem blockMember_iff (low high : ChunkClass) (block : Blocks) :
    blockMember low high block ↔ Member low high (fieldBlocks block) := by
  cases block with
  | inl coordinate =>
      unfold blockMember Member
      rw [fieldBlocks_common_candidates]
      rfl
  | inr last =>
      unfold blockMember Member
      rw [fieldBlocks_extra_candidates]
      exact and_congr low.member_zero_iff.symm high.member_zero_iff.symm

private def pairClassEquiv (low high : ChunkClass) :
    (Fin low.size × Fin high.size) ≃
      {word : Fin pairModulus // PairMember low high (pairEquiv.symm word)} :=
  ((Equiv.prodCongr low.indexEquiv high.indexEquiv).trans
    Equiv.subtypeProdEquivProd.symm).trans
      (Equiv.subtypeEquiv pairEquiv (by
        intro pair
        change PairMember low high pair ↔ PairMember low high (pairEquiv.symm (pairEquiv pair))
        rw [pairEquiv.symm_apply_apply]))

private def commonClassEquiv (low high : ChunkClass) :
    (Fin (pairModulus - 1) × (Fin low.size × Fin high.size)) ≃
      {coordinate : Fin (pairModulus - 1) × Fin pairModulus //
        blockMember low high (Sum.inl coordinate)} where
  toFun coordinate :=
    ⟨(coordinate.1, (pairClassEquiv low high coordinate.2).val),
      (pairClassEquiv low high coordinate.2).property⟩
  invFun coordinate :=
    (coordinate.val.1, (pairClassEquiv low high).symm ⟨coordinate.val.2, coordinate.property⟩)
  left_inv coordinate := by
    apply Prod.ext
    · rfl
    · exact (pairClassEquiv low high).symm_apply_apply coordinate.2
  right_inv coordinate := by
    apply Subtype.ext
    apply Prod.ext
    · rfl
    · exact congrArg Subtype.val
        ((pairClassEquiv low high).apply_symm_apply ⟨coordinate.val.2, coordinate.property⟩)

private def conditionalOneEquiv (condition : Prop) [Decidable condition] :
    Fin (if condition then 1 else 0) ≃ {_last : Fin 1 // condition} where
  toFun index :=
    have holds : condition := by
      by_contra absent
      have bound := index.isLt
      simp only [absent, if_false] at bound
      omega
    ⟨⟨index.val, by simpa only [holds, if_true] using index.isLt⟩, holds⟩
  invFun last := ⟨last.val.val, by simpa only [last.property, if_true] using last.val.isLt⟩
  left_inv _ := rfl
  right_inv _ := rfl

private abbrev Coordinates (low high : ChunkClass) :=
  (Fin (pairModulus - 1) × (Fin low.size × Fin high.size)) ⊕ Fin (zeroBonus low high)

private def coordinateFields (low high : ChunkClass) :
    Coordinates low high ≃ {value : F // Member low high value} :=
  (((Equiv.sumCongr (commonClassEquiv low high)
    (conditionalOneEquiv (low.HasZero ∧ high.HasZero))).trans
      (Equiv.subtypeSum (p := blockMember low high)).symm).trans
        (Equiv.subtypeEquiv fieldBlocks (blockMember_iff low high)))

private def pairRanks (low high : Nat) : Fin low × Fin high ≃ Fin (low * high) :=
  ((Equiv.prodComm _ _).trans finProdFinEquiv).trans (finCongr (Nat.mul_comm high low))

private def coordinateRanks (low high : ChunkClass) : Coordinates low high ≃ Fin (size low high) :=
  (Equiv.sumCongr
    ((Equiv.prodCongr (Equiv.refl _) (pairRanks low.size high.size)).trans finProdFinEquiv)
    (Equiv.refl _)).trans finSumFinEquiv

/-- Exact local rank/unrank interface for the ordered rectangle. -/
def rectangleEquiv (low high : ChunkClass) :
    Fin (size low high) ≃ {value : F // Member low high value} :=
  (coordinateRanks low high).symm.trans (coordinateFields low high)

/-- A common rank has low index fastest, then high, then field block. -/
def commonRank (low high : ChunkClass) (block : Fin (pairModulus - 1))
    (lowIndex : Fin low.size) (highIndex : Fin high.size) : Fin (size low high) :=
  coordinateRanks low high (Sum.inl (block, (lowIndex, highIndex)))

theorem common_rank_value (low high : ChunkClass) (block : Fin (pairModulus - 1))
    (lowIndex : Fin low.size) (highIndex : Fin high.size) :
    (commonRank low high block lowIndex highIndex).val =
      lowIndex.val + low.size * highIndex.val + (low.size * high.size) * block.val := rfl

theorem common_unrank_value (low high : ChunkClass) (block : Fin (pairModulus - 1))
    (lowIndex : Fin low.size) (highIndex : Fin high.size) :
    (rectangleEquiv low high (commonRank low high block lowIndex highIndex)).val.val =
      (low.indexEquiv lowIndex).val.val + chunkModulus * (high.indexEquiv highIndex).val.val +
        pairModulus * block.val := by
  change (coordinateFields low high
    ((coordinateRanks low high).symm
      (coordinateRanks low high (Sum.inl (block, (lowIndex, highIndex)))))).val.val = _
  rw [(coordinateRanks low high).symm_apply_apply]
  rfl

/-- The extra rank is present only when zero belongs to both classes. -/
def extraRank (low high : ChunkClass) (bothZero : low.HasZero ∧ high.HasZero) :
    Fin (size low high) :=
  coordinateRanks low high (Sum.inr ⟨0, by simp only [zeroBonus, bothZero]; decide⟩)

theorem extra_rank_value (low high : ChunkClass) (bothZero : low.HasZero ∧ high.HasZero) :
    (extraRank low high bothZero).val = (pairModulus - 1) * (low.size * high.size) := by
  change (pairModulus - 1) * (low.size * high.size) + 0 = _
  exact Nat.add_zero _

theorem extra_unrank_value (low high : ChunkClass) (bothZero : low.HasZero ∧ high.HasZero) :
    (rectangleEquiv low high (extraRank low high bothZero)).val.val = goldilocksModulus - 1 := by
  change (coordinateFields low high
    ((coordinateRanks low high).symm
      (coordinateRanks low high (Sum.inr ⟨0, by simp only [zeroBonus, bothZero]; decide⟩)))).val.val = _
  rw [(coordinateRanks low high).symm_apply_apply]
  change (pairModulus - 1) * pairModulus + 0 = goldilocksModulus - 1
  rw [goldilocks_decomposition, Nat.add_sub_cancel, Nat.add_zero, Nat.mul_comm]

theorem rank_unrank (low high : ChunkClass) (rank : Fin (size low high)) :
    (rectangleEquiv low high).symm (rectangleEquiv low high rank) = rank :=
  (rectangleEquiv low high).symm_apply_apply rank

theorem unrank_rank (low high : ChunkClass) (value : {value : F // Member low high value}) :
    rectangleEquiv low high ((rectangleEquiv low high).symm value) = value :=
  (rectangleEquiv low high).apply_symm_apply value

theorem rectangle_card (low high : ChunkClass) :
    Nat.card {value : F // Member low high value} =
      (pairModulus - 1) * (low.size * high.size) +
        if low.HasZero ∧ high.HasZero then 1 else 0 := by
  rw [← Nat.card_congr (rectangleEquiv low high), Nat.card_fin]
  rfl

end NightstreamFPrime.Spec.Folding.Nifs.NonInteractive.PiRlcSampler.FieldPreimageRectangle
