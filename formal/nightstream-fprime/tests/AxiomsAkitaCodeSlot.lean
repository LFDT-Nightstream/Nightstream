import tests.AxiomAudit
import NightstreamFPrime.Spec.Compression.Akita.Replay.CodeSlot

/-! Axiom audits for the shared pool of the fold challenge sampler of the replay
(`Replay/SamplerPool.lean`), the slot type `CodeSlot` with the fold decoders of the code
(`Replay/CodeSlot.lean`), and the fair decoders from a slot type with `FieldSlot`
(`Extract/Decoders.lean`), with the guard that the laws of `CodeSlot.open` apply to the
unfiltered group `(64, 31, 10)` of the code. -/

#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Extract.expect_read
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Extract.mass_read
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Extract.Fair.comp
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Extract.FieldSlot
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Extract.FieldSlot.exists_field
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Extract.FieldSlot.self
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Extract.FieldSlot.nonempty
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Extract.FieldSlot.exists_read
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Extract.exists_fair_field
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Extract.exists_fair_fields
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Extract.exists_fair_field_family
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Extract.exists_fair_field_base
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Extract.exists_fair_field_challenges
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.rank
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.pool
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.pool_accepts
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.rank_congr
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.rank_pool
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.pool_rank
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.pool_of_not
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.rank_lt_rank
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.rank_inj
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.ranked
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.ranked_eq
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.ranked_of_not
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.swapPool
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.swapPool_involutive
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.expect_pool_pair
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.expect_pool
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.rank_zero
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.rank_succ
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.none_of_rejected
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.getElem?_filterMap_rank
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.exists_rank
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.length_filterMap_ofFn
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.filterMap_pool
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.words
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.filterMap_words
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.sample_pool
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.sample_slots
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.sample_isSome_iff
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.slotDecode_mk
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.filtered64
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.filtered128
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Kept
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Fill
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Kept.fintype
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Kept.decidableEq
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Fill.fintype
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Fill.decidableEq
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Fill.nonempty
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.CodeSlot
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.CodeSlot.nonempty
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.CodeSlot.fieldSlot
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.splitWord
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.splitWord_snd
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Fill.decode
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.CodeSlot.open
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.CodeSlot.kept64
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.CodeSlot.kept128
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.slotChance_comp_le
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.card_open_fibre
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.card_kept64_fibre
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.card_kept128_fibre
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.pow_le_field
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.open_isSome
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.open_iff
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.open_mem
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.signFair_open
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.slotChance_open
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.slotChance_open_field
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.card_fill
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Fill.decode_isSome
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Fill.decode_mem
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Fill.decode_surjective
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.slotChance_fill
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.kept_nonempty
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.kept64_mem
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.kept128_mem
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.kept64_surjective
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.kept128_surjective
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.kept64_isSome
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.kept128_isSome
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.slotChance_kept64
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.slotChance_kept128
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.slotChance_kept64_le
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.slotChance_kept128_le
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.slotChance_kept64_field
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.slotChance_kept128_field

section AkitaCodeSlotGuards

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Compression.Akita
open NightstreamFPrime.Spec.Compression.Akita.Extract (slotChance FieldSlot)
open NightstreamFPrime.Spec.Compression.Akita.Replay

/-- `CodeSlot E` is a nonempty finite type with no premise on the filters, and it gives uniform
elements of `E`. -/
example {E : Type} [Fintype E] [Nonempty E] :
    Nonempty (CodeSlot E) ∧ Nonempty (Fintype (CodeSlot E)) ∧ FieldSlot (CodeSlot E) E :=
  ⟨inferInstance, ⟨inferInstance⟩, inferInstance⟩

/-- The laws of `CodeSlot.open` apply to the unfiltered group `(64, 31, 10)` of the code, whose
words have 128 bits. -/
example {E : Type} [Fintype E] [Nonempty E] :
    ∃ fits : FoldGroup.candidateBits ⟨1, 64, 64, 31, 10, false⟩ ≤ 131,
      (∀ s : CodeSlot E, (CodeSlot.open ⟨1, 64, 64, 31, 10, false⟩ fits s).isSome) ∧
      SignFair (CodeSlot.open (E := E) ⟨1, 64, 64, 31, 10, false⟩ fits) ∧
      slotChance (CodeSlot.open (E := E) ⟨1, 64, 64, 31, 10, false⟩ fits) ≤
        1 / (fieldModulus : ℝ) ^ 2 := by
  have wide : FoldGroup.candidateBits ⟨1, 64, 64, 31, 10, false⟩ = 128 := by
    rw [FoldGroup.candidateBits, ite_eq_right (by decide),
      Nat.log_eq_of_pow_le_of_lt_pow (m := 87) (by decide) (by decide)]
    rfl
  exact ⟨wide.trans_le (by decide), fun s => open_isSome _ rfl (by decide) s,
    signFair_open _ rfl, slotChance_open_field _ _ wide.ge⟩

end AkitaCodeSlotGuards
