import CompressionLeanRust.Rounds
import CompressionLeanRust.Positions
import CompressionLean.Ports.Shrink.Diagonal

/-! The translated functions of `neo_verified::diagonal` equal their copies in
`CompressionLean/Ports/Shrink/Diagonal.lean` over `Cubic`: `walk_bits` is `walkBits`, `track_factor`
`trackFactor`, `tracks_step` `tracksStep`, `next_below` `nextBelow`, `walk_step` `walkStep`,
`walk_states` `walk` and `diagonal` `diagonal`. The shrink verifier's block part reads `diagonal`.

A copy reads an input `Ext` as `cubicOf` reads it and gives a value as `cubicWords`; every output is a
constant, a sum, a product or a reduced word, so its words are canonical. Rust computes `walkStep` at
each target state as the copy's sum, carry by carry, so the proof needs no reordering. A state table
is read by carry, then below or not (`statesOf`); `walk_step` reads only the carries below
`2^tracks.len()`, as the copy does.

Rust panics where `2^tracks.len()` passes `usize::MAX`, where a carry sum overflows, and where the
states before a step are fewer than the carries; `Complete` assumes none of these, and the verifier
passes one or two tracks. -/

namespace CompressionLeanRust

open Aeneas Aeneas.Std Result
open neo_verified
open CompressionLean
open NightstreamFPrime.Spec (goldilocksModulus)

/-- The tracks as the copy reads them. -/
noncomputable def tracksOf (tracks : List diagonal.Track) : List (Shrink.Track Cubic) :=
  tracks.map fun track => (track.point.val.map cubicOf, track.offset.val)

theorem tracksOf_length (tracks : List diagonal.Track) : (tracksOf tracks).length = tracks.length := by
  simp [tracksOf]

/-! ## Bits and carries -/

theorem size_eq_log (v : ℕ) (h : 0 < v) : Nat.size v = Nat.log 2 v + 1 := by
  apply le_antisymm
  · exact Nat.size_le.mpr (Nat.lt_pow_succ_log_self (by norm_num) v)
  · exact Nat.lt_size.mpr (Nat.pow_log_le_self 2 (by omega))

theorem bit_length_eq (v : Usize) : ∃ r, diagonal.bit_length v = ok r ∧ r.val = Nat.size v.val := by
  unfold diagonal.bit_length
  have hn := numBits_le
  have hp := numBits_pos
  have hpow : UScalarTy.Usize.numBits < 2 ^ UScalarTy.Usize.numBits := Nat.lt_two_pow_self
  have hbits : (UScalar.cast .Usize core.num.Usize.BITS).val = UScalarTy.Usize.numBits := by
    rw [UScalar.cast_val_eq, bits_val, Nat.mod_eq_of_lt hpow]
  have hlz : (core.num.Usize.leading_zeros v).val =
      if v.val = 0 then UScalarTy.Usize.numBits else UScalarTy.Usize.numBits - Nat.log 2 v.val - 1 := by
    split
    · exact leading_zeros_zero v (by assumption)
    · exact leading_zeros_val v (by omega)
  have hcast : (UScalar.cast .Usize (core.num.Usize.leading_zeros v)).val = (core.num.Usize.leading_zeros v).val := by
    rw [UScalar.cast_val_eq]
    apply Nat.mod_eq_of_lt
    calc (core.num.Usize.leading_zeros v).val ≤ UScalarTy.Usize.numBits := by rw [hlz]; split <;> omega
      _ < 2 ^ UScalarTy.Usize.numBits := hpow
  have hlog : v.val ≠ 0 → Nat.log 2 v.val < UScalarTy.Usize.numBits := fun h => Nat.log_lt_of_lt_pow h v.hBounds
  refine ⟨core.num.Usize.wrapping_sub (UScalar.cast .Usize core.num.Usize.BITS)
    (UScalar.cast .Usize (core.num.Usize.leading_zeros v)), by simp only [lift, bind_ok], ?_⟩
  rw [wrapping_sub_val _ _ (by rw [hcast, hbits, hlz]; split <;> omega), hcast, hbits, hlz]
  split
  · next h => rw [h]; simp
  · next h =>
    rw [size_eq_log v.val (by omega)]
    have := hlog h
    omega

theorem bit_at_eq (v t : Usize) : ∃ r, diagonal.bit_at v t = ok r ∧ r.val = v.val / 2 ^ t.val % 2 := by
  unfold diagonal.bit_at
  rw [index_bit_eq, bind_ok]
  rw [Nat.testBit_eq_decide_div_mod_eq]
  by_cases h : v.val / 2 ^ t.val % 2 = 1
  · exact ⟨1#usize, by simp [h], by simp [h]⟩
  · exact ⟨0#usize, by simp [h], by simp; omega⟩


theorem carry_states_spec (tracks : Slice diagonal.Track) :
    (∀ y, diagonal.carry_states tracks = ok y → y.val = 2 ^ tracks.val.length) ∧
    (2 ^ tracks.val.length ≤ Usize.max → ∃ y, diagonal.carry_states tracks = ok y) := by
  have hlen := tracks.property
  unfold diagonal.carry_states diagonal.carry_states_loop
  have key := loop_ok (fun a : Usize × Usize => diagonal.carry_states_loop.body tracks a.1 a.2)
    (fun n a => a.2.val ≤ tracks.val.length ∧ n = tracks.val.length - a.2.val ∧ a.1.val = 2 ^ a.2.val)
    (fun y : Usize => y.val = 2 ^ tracks.val.length)
    (fun _ => 2 ^ tracks.val.length ≤ Usize.max)
    (by
      rintro n ⟨st, i⟩ c ⟨hi, rfl, hv⟩ hbody
      dsimp only at hi hv hbody ⊢
      by_cases hlt : i.val < tracks.val.length
      · right
        have hmax : i.val < Usize.max := by scalar_tac
        simp only [diagonal.carry_states_loop.body, Slice.len,
          show i < Usize.ofNatCore tracks.val.length _ by scalar_tac, ite_true, lift, bind_ok] at hbody
        obtain ⟨st1, h1, hbody⟩ := bind_eq_ok hbody
        rw [Result.ok.injEq] at hbody
        subst hbody
        refine ⟨(st1, core.num.Usize.wrapping_add i 1#usize), tracks.val.length - (i.val + 1), rfl, by omega, ?_,
          by rw [add_one_val i hmax], ?_⟩
        · rw [add_one_val i hmax]; omega
        · rw [mul_eq_ok h1, hv, add_one_val i hmax, pow_succ]
          rfl
      · left
        simp only [diagonal.carry_states_loop.body, Slice.len,
          show ¬ i < Usize.ofNatCore tracks.val.length _ by scalar_tac, ite_false, Result.ok.injEq] at hbody
        subst hbody
        exact ⟨_, rfl, by rw [hv]; congr 1; omega⟩)
    (by
      rintro n ⟨st, i⟩ ⟨hi, rfl, hv⟩ hP
      dsimp only at hi hv hP ⊢
      by_cases hlt : i.val < tracks.val.length
      · have hpow : 2 ^ (i.val + 1) ≤ 2 ^ tracks.val.length := Nat.pow_le_pow_right (by omega) hlt
        obtain ⟨st1, h1, -⟩ := mul_ok st 2#usize (by rw [hv]; simp; rw [← pow_succ]; omega)
        exact ⟨ControlFlow.cont (st1, core.num.Usize.wrapping_add i 1#usize), by
          simp only [diagonal.carry_states_loop.body, Slice.len,
            show i < Usize.ofNatCore tracks.val.length _ by scalar_tac, ite_true, lift, bind_ok, h1]⟩
      · exact ⟨ControlFlow.done st, by
          simp only [diagonal.carry_states_loop.body, Slice.len,
            show ¬ i < Usize.ofNatCore tracks.val.length _ by scalar_tac, ite_false]⟩)
    (fun _ _ _ _ hP _ => hP) _ (1#usize, 0#usize) ⟨by simp, rfl, by simp⟩
  exact ⟨key.1, fun h => let ⟨y, hy, _⟩ := key.2 h; ⟨y, hy⟩⟩

/-! ## `walk_bits` -/

theorem walk_bits_eq (tracks : Slice diagonal.Track) (count : Usize) :
    ∃ r, diagonal.walk_bits tracks count = ok r ∧ r.val = Shrink.walkBits (tracksOf tracks.val) count.val := by
  have hlen := tracks.property
  obtain ⟨b0, hb0, hb0v⟩ := bit_length_eq count
  unfold diagonal.walk_bits diagonal.walk_bits_loop
  rw [hb0, bind_ok]
  set ms := (tracksOf tracks.val).map fun track => max track.1.length (Nat.size track.2) with hms
  have hmsl : ms.length = tracks.val.length := by simp [hms, tracksOf]
  obtain ⟨y, hy, hq⟩ := loop_total (fun a : Usize × Usize => diagonal.walk_bits_loop.body tracks a.1 a.2)
    (fun n a => a.2.val ≤ tracks.val.length ∧ n = a.2.val ∧ a.1.val = (ms.drop a.2.val).foldr max (Nat.size count.val))
    (fun y : Usize => y.val = ms.foldr max (Nat.size count.val))
    (by
      rintro n ⟨bits, i⟩ ⟨hi, rfl, hv⟩
      dsimp only at hi hv ⊢
      by_cases hpos : 0 < i.val
      · right
        have hsub : (core.num.Usize.wrapping_sub i 1#usize).val = i.val - 1 := wrapping_sub_val i 1#usize (by simp; omega)
        have hlt : i.val - 1 < tracks.val.length := by omega
        obtain ⟨o, ho, hov⟩ := bit_length_eq tracks.val[i.val - 1].offset
        refine ⟨(core.cmp.impls.OrdUsize.max (core.cmp.impls.OrdUsize.max (alloc.vec.Vec.len tracks.val[i.val - 1].point) o) bits,
          core.num.Usize.wrapping_sub i 1#usize), i.val - 1, ?_, by omega, by rw [hsub]; omega, by rw [hsub], ?_⟩
        · simp only [diagonal.walk_bits_loop.body, show (0#usize) < i by scalar_tac, ite_true, lift, bind_ok,
            slice_index tracks _ (by rw [hsub]; exact hlt)]
          simp only [hsub]
          rw [ho, bind_ok]
        · dsimp only
          have hd : ms.drop (i.val - 1) = ms[i.val - 1]'(by rw [hmsl]; omega) :: ms.drop i.val := by
            rw [List.drop_eq_getElem_cons (by rw [hmsl]; omega), Nat.sub_add_cancel (by omega)]
          rw [core.cmp.impls.OrdUsize.max_val, core.cmp.impls.OrdUsize.max_val, hsub, hv, hov, alloc.vec.Vec.len_val,
            hd, List.foldr_cons]
          simp [hms, tracksOf]
      · left
        refine ⟨bits, ?_, ?_⟩
        · simp only [diagonal.walk_bits_loop.body, show ¬ (0#usize) < i by scalar_tac, ite_false]
        · rw [hv, show i.val = 0 by omega, List.drop_zero])
    _ (b0, Slice.len tracks) ⟨by simp, rfl, by rw [hb0v, Slice.len_val, List.drop_of_length_le hmsl.le]; rfl⟩
  exact ⟨y, hy, hq⟩


/-! ## `track_factor`, `next_below` and `tracks_step` -/

theorem usize_eq_one (r : Usize) : r = 1#usize ↔ r.val = 1 := by
  constructor
  · rintro rfl; rfl
  · intro e; exact UScalar.val_eq_imp _ _ (by simp [e])

theorem canon_one : Canon field.EXT_ONE := by
  unfold field.EXT_ONE field.GL_ONE field.GL_ZERO
  exact ⟨by decide, by decide, by decide⟩

theorem canon_zero : Canon field.EXT_ZERO := by
  unfold field.EXT_ZERO field.GL_ZERO
  exact ⟨by decide, by decide, by decide⟩

theorem track_factor_eq (point : Slice field.Ext) (t bit : Usize) :
    ∃ e, diagonal.track_factor point t bit = ok e ∧ Canon e ∧
      cubicOf e = Shrink.trackFactor (point.val.map cubicOf) t.val bit.val := by
  unfold diagonal.track_factor
  by_cases ht : t.val < point.val.length
  · have hget : (point.val.map cubicOf).getD t.val 0 = cubicOf point.val[t.val] := by
      rw [List.getD_eq_getElem _ _ (by simpa using ht), List.getElem_map]
    by_cases hb : bit.val = 1
    · refine ⟨extRed point.val[t.val], ?_, canon_extRed _, ?_⟩
      · simp only [Slice.len, show t < Usize.ofNatCore point.val.length _ by scalar_tac, ite_true,
          (usize_eq_one bit).mpr hb, slice_index point t ht, ext_reduce_eq, bind_ok]
      · simp only [Shrink.trackFactor, List.length_map, ht, ite_true, hb, hget, cubicOf_extRed]
    · refine ⟨extSub field.EXT_ONE point.val[t.val], ?_, canon_extSub _ _, ?_⟩
      · simp only [Slice.len, show t < Usize.ofNatCore point.val.length _ by scalar_tac, ite_true,
          show ¬ bit = 1#usize from fun h => hb ((usize_eq_one bit).mp h), ite_false, slice_index point t ht,
          ext_sub_eq, bind_ok]
      · simp only [Shrink.trackFactor, List.length_map, ht, ite_true, hb, ite_false, hget, extSub_cubic, cubicOf_one]
  · by_cases hb : bit.val = 1
    · refine ⟨field.EXT_ZERO, ?_, canon_zero, ?_⟩
      · simp only [Slice.len, show ¬ t < Usize.ofNatCore point.val.length _ by scalar_tac, ite_false,
          (usize_eq_one bit).mpr hb, ite_true]
      · simp only [Shrink.trackFactor, List.length_map, ht, ite_false, hb, ite_true, cubicOf_zero]
    · refine ⟨field.EXT_ONE, ?_, canon_one, ?_⟩
      · simp only [Slice.len, show ¬ t < Usize.ofNatCore point.val.length _ by scalar_tac, ite_false,
          show ¬ bit = 1#usize from fun h => hb ((usize_eq_one bit).mp h)]
      · simp only [Shrink.trackFactor, List.length_map, ht, ite_false, hb, cubicOf_one]

theorem next_below_eq (cb b below : Usize) :
    ∃ r, diagonal.next_below cb b below = ok r ∧ r.val = Shrink.nextBelow cb.val b.val below.val := by
  unfold diagonal.next_below Shrink.nextBelow
  by_cases h1 : b.val < cb.val
  · exact ⟨1#usize, by simp [show b < cb by scalar_tac], by simp [h1]⟩
  · by_cases h2 : cb.val < b.val
    · exact ⟨0#usize, by simp [show ¬ b < cb by scalar_tac, show cb < b by scalar_tac], by simp [h1, h2]⟩
    · exact ⟨below, by simp [show ¬ b < cb by scalar_tac, show ¬ cb < b by scalar_tac], by simp [h1, h2]⟩

theorem usize_shr_one (e : Usize) : (core.num.Usize.wrapping_shr e 1#u32).val = e.val / 2 := by
  have hn := numBits_le
  have hp := numBits_pos
  simp only [UScalar.val, core.num.Usize.wrapping_shr_bv_eq]
  have h1 : (1#u32).bv.toNat % System.Platform.numBits = 1 := by
    rcases System.Platform.numBits_eq with h | h <;> rw [h] <;> rfl
  rw [h1, BitVec.ushiftRight_eq, BitVec.toNat_ushiftRight, Nat.shiftRight_eq_div_pow, pow_one]

theorem usize_is_multiple_of_two (e : Usize) : core.num.Usize.is_multiple_of e 2#usize = ok (e.val % 2 == 0) := rfl


theorem tracksStep_carry_lt (t b : ℕ) (hb : b ≤ 1) :
    ∀ (T : List (Shrink.Track Cubic)) (c : ℕ), (Shrink.tracksStep t b T c).2 < 2 ^ T.length
  | [], _ => by simp [Shrink.tracksStep]
  | (point, offset) :: rest, c => by
      have ih := tracksStep_carry_lt t b hb rest (c / 2)
      simp only [Shrink.tracksStep, List.length_cons, pow_succ]
      have : (offset / 2 ^ t % 2 + b + c % 2) / 2 ≤ 1 := by omega
      omega

theorem tracksOf_drop (tracks : List diagonal.Track) (i : ℕ) (h : i < tracks.length) :
    (tracksOf tracks).drop i = (tracks[i].point.val.map cubicOf, tracks[i].offset.val) :: (tracksOf tracks).drop (i + 1) := by
  rw [List.drop_eq_getElem_cons (by rw [tracksOf_length]; exact h)]
  simp [tracksOf]

theorem tracks_step_spec (tracks : Slice diagonal.Track) (t b carry : Usize) :
    (∀ y, diagonal.tracks_step tracks t b carry = ok y → Canon y.1 ∧
      cubicOf y.1 = (Shrink.tracksStep t.val b.val (tracksOf tracks.val) carry.val).1 ∧
      y.2.val = (Shrink.tracksStep t.val b.val (tracksOf tracks.val) carry.val).2) ∧
    (b.val ≤ 1 → 2 ^ tracks.val.length ≤ Usize.max → ∃ y, diagonal.tracks_step tracks t b carry = ok y) := by
  have hlen := tracks.property
  unfold diagonal.tracks_step diagonal.tracks_step_loop
  have key := loop_ok
    (fun a : field.Ext × Usize × Usize => diagonal.tracks_step_loop.body tracks t b carry a.1 a.2.1 a.2.2)
    (fun n a => a.2.2.val ≤ tracks.val.length ∧ n = a.2.2.val ∧ Canon a.1 ∧
      cubicOf a.1 = (Shrink.tracksStep t.val b.val ((tracksOf tracks.val).drop a.2.2.val) (carry.val / 2 ^ a.2.2.val)).1 ∧
      a.2.1.val = (Shrink.tracksStep t.val b.val ((tracksOf tracks.val).drop a.2.2.val) (carry.val / 2 ^ a.2.2.val)).2)
    (fun y : field.Ext × Usize => Canon y.1 ∧
      cubicOf y.1 = (Shrink.tracksStep t.val b.val (tracksOf tracks.val) carry.val).1 ∧
      y.2.val = (Shrink.tracksStep t.val b.val (tracksOf tracks.val) carry.val).2)
    (fun _ => b.val ≤ 1 ∧ 2 ^ tracks.val.length ≤ Usize.max)
    (by
      rintro n ⟨factor, nc, i⟩ c ⟨hi, rfl, hc, hv1, hv2⟩ hbody
      dsimp only at hi hc hv1 hv2 hbody ⊢
      by_cases hpos : 0 < i.val
      · right
        have hsub : (core.num.Usize.wrapping_sub i 1#usize).val = i.val - 1 :=
          wrapping_sub_val i 1#usize (by simp; omega)
        have hlt : i.val - 1 < tracks.val.length := by omega
        obtain ⟨ob, hob, hobv⟩ := bit_at_eq tracks.val[i.val - 1].offset t
        obtain ⟨cb, hcb, hcbv⟩ := bit_at_eq carry (core.num.Usize.wrapping_sub i 1#usize)
        simp only [diagonal.tracks_step_loop.body, show (0#usize) < i by scalar_tac, ite_true, lift, bind_ok,
          slice_index tracks _ (by rw [hsub]; exact hlt)] at hbody
        simp only [hsub] at hbody
        rw [hob, bind_ok] at hbody
        obtain ⟨s1, hs1, hbody⟩ := bind_eq_ok hbody
        rw [hcb, bind_ok] at hbody
        obtain ⟨sum, hsum, hbody⟩ := bind_eq_ok hbody
        rw [usize_is_multiple_of_two, bind_ok] at hbody
        have es1 := add_eq_ok hs1
        have esum := add_eq_ok hsum
        obtain ⟨bt, hbt, hbtv⟩ : ∃ bt : Usize, (if (sum.val % 2 == 0) = true then ok 0#usize else ok 1#usize :
            Result Usize) = ok bt ∧ bt.val = sum.val % 2 := by
          by_cases hs : sum.val % 2 = 0
          · exact ⟨0#usize, by simp [hs], by simp [hs]⟩
          · exact ⟨1#usize, by simp [hs], by simp; omega⟩
        rw [hbt, bind_ok] at hbody
        obtain ⟨e, he, hce, hve⟩ := track_factor_eq (alloc.vec.Vec.deref tracks.val[i.val - 1].point) t bt
        rw [he, bind_ok, ext_mul_eq, bind_ok] at hbody
        obtain ⟨m2, hm2, hbody⟩ := bind_eq_ok hbody
        obtain ⟨nc1, hnc1, hbody⟩ := bind_eq_ok hbody
        rw [Result.ok.injEq] at hbody
        subst hbody
        have em2 := mul_eq_ok hm2
        have enc1 := add_eq_ok hnc1
        refine ⟨_, i.val - 1, rfl, by omega, by rw [hsub]; omega, by rw [hsub], canon_extMul _ _, ?_, ?_⟩
        · dsimp only
          rw [hsub, tracksOf_drop tracks.val (i.val - 1) hlt, Nat.sub_add_cancel hpos, extMul_cubic, hve, hv1,
            deref_val]
          simp only [Shrink.tracksStep]
          rw [hbtv, esum, es1, hobv, hcbv, hsub, div_pow_succ, Nat.sub_add_cancel hpos]
        · dsimp only
          rw [hsub, tracksOf_drop tracks.val (i.val - 1) hlt, Nat.sub_add_cancel hpos, enc1, usize_shr_one, em2, hv2]
          simp only [Shrink.tracksStep]
          rw [esum, es1, hobv, hcbv, hsub, div_pow_succ, Nat.sub_add_cancel hpos]
          simp only [show (2#usize).val = 2 by simp]
          ring
      · left
        simp only [diagonal.tracks_step_loop.body, show ¬ (0#usize) < i by scalar_tac, ite_false,
          Result.ok.injEq] at hbody
        subst hbody
        have hi0 : i.val = 0 := by omega
        refine ⟨_, rfl, hc, ?_, ?_⟩
        · rw [hv1, hi0]; simp
        · rw [hv2, hi0]; simp)
    (by
      rintro n ⟨factor, nc, i⟩ ⟨hi, rfl, hc, hv1, hv2⟩ ⟨hb, hmax⟩
      dsimp only at hi hc hv1 hv2 ⊢
      by_cases hpos : 0 < i.val
      · have hsub : (core.num.Usize.wrapping_sub i 1#usize).val = i.val - 1 :=
          wrapping_sub_val i 1#usize (by simp; omega)
        have hlt : i.val - 1 < tracks.val.length := by omega
        have hm := usize_max_ge
        obtain ⟨ob, hob, hobv⟩ := bit_at_eq tracks.val[i.val - 1].offset t
        obtain ⟨cb, hcb, hcbv⟩ := bit_at_eq carry (core.num.Usize.wrapping_sub i 1#usize)
        have hob2 : ob.val ≤ 1 := by rw [hobv]; omega
        have hcb2 : cb.val ≤ 1 := by rw [hcbv]; omega
        obtain ⟨s1, hs1, es1⟩ := add_ok ob b (by omega)
        obtain ⟨sum, hsum, esum⟩ := add_ok s1 cb (by omega)
        obtain ⟨bt, hbt, -⟩ : ∃ bt : Usize, (if (sum.val % 2 == 0) = true then ok 0#usize else ok 1#usize :
            Result Usize) = ok bt ∧ bt.val = sum.val % 2 := by
          by_cases hs : sum.val % 2 = 0
          · exact ⟨0#usize, by simp [hs], by simp [hs]⟩
          · exact ⟨1#usize, by simp [hs], by simp; omega⟩
        obtain ⟨e, he, -, -⟩ := track_factor_eq (alloc.vec.Vec.deref tracks.val[i.val - 1].point) t bt
        have hncb := tracksStep_carry_lt t.val b.val hb ((tracksOf tracks.val).drop i.val) (carry.val / 2 ^ i.val)
        rw [← hv2, List.length_drop, tracksOf_length] at hncb
        have hpow : 2 ^ (tracks.val.length - i.val + 1) ≤ 2 ^ tracks.val.length := Nat.pow_le_pow_right (by omega) (by omega)
        rw [pow_succ] at hpow
        obtain ⟨m2, hm2, em2⟩ := mul_ok nc 2#usize (by simp; omega)
        obtain ⟨nc1, hnc1, -⟩ := add_ok (core.num.Usize.wrapping_shr sum 1#u32) m2 (by rw [usize_shr_one, em2]; simp; omega)
        refine ⟨ControlFlow.cont (extMul e factor, nc1, core.num.Usize.wrapping_sub i 1#usize), ?_⟩
        simp only [diagonal.tracks_step_loop.body, show (0#usize) < i by scalar_tac, ite_true, lift, bind_ok,
          slice_index tracks _ (by rw [hsub]; exact hlt)]
        simp only [hsub]
        rw [hob, bind_ok, hs1, bind_ok, hcb, bind_ok, hsum, bind_ok, usize_is_multiple_of_two, bind_ok, hbt, bind_ok,
          he, bind_ok, ext_mul_eq, bind_ok, hm2, bind_ok, hnc1, bind_ok]
      · exact ⟨ControlFlow.done (factor, nc), by
          simp only [diagonal.tracks_step_loop.body, show ¬ (0#usize) < i by scalar_tac, ite_false]⟩)
    (fun _ _ _ _ hP _ => hP) _ (field.EXT_ONE, 0#usize, Slice.len tracks)
    ⟨by simp, rfl, canon_one, by rw [Slice.len_val, List.drop_of_length_le (by rw [tracksOf_length])]; simp [Shrink.tracksStep, cubicOf_one],
      by rw [Slice.len_val, List.drop_of_length_le (by rw [tracksOf_length])]; simp [Shrink.tracksStep]⟩
  exact ⟨key.1, fun h1 h2 => let ⟨y, hy, _⟩ := key.2 ⟨h1, h2⟩; ⟨y, hy⟩⟩


/-! ## `walk_step` -/

theorem array_index {α : Type} {n : Usize} (a : Aeneas.Std.Array α n) (i : Usize) (h : i.val < a.val.length) :
    Aeneas.Std.Array.index_usize a i = ok a.val[i.val] := by
  unfold Aeneas.Std.Array.index_usize
  rw [Aeneas.Std.Array.getElem?_Usize_eq, List.getElem?_eq_getElem h]

/-- The walk's states as the copy reads them: carry `c`, below or not `bl`. -/
noncomputable def statesOf (states : List (Aeneas.Std.Array field.Ext 2#usize)) (c bl : ℕ) : Cubic :=
  match states[c]? with
  | some a => (a.val.map cubicOf).getD bl 0
  | none => 0

/-- One term of `walkStep`, with `cb` the bit of `count`. -/
noncomputable def stepTerm (T : List (Shrink.Track Cubic)) (t cb : ℕ) (S : ℕ → ℕ → Cubic) (target tb carry below b : ℕ) :
    Cubic :=
  if (Shrink.tracksStep t b T carry).2 = target ∧ Shrink.nextBelow cb b below = tb then
    S carry below * (Shrink.tracksStep t b T carry).1
  else 0

theorem usize_eq_iff (a b : Usize) : a = b ↔ a.val = b.val :=
  ⟨fun h => h ▸ rfl, fun h => UScalar.val_eq_imp a b h⟩

theorem statesOf_get (states : List (Aeneas.Std.Array field.Ext 2#usize)) (c : ℕ) (h : c < states.length) (bl : ℕ)
    (hbl : bl < states[c].val.length) : statesOf states c bl = cubicOf states[c].val[bl] := by
  simp [statesOf, List.getElem?_eq_getElem h, List.getElem?_eq_getElem hbl]

theorem bind_exists {α β : Type} {x : Result α} {f : α → Result β} (a : α) (hx : x = ok a)
    (hf : ∃ c, f a = ok c) : ∃ c, x >>= f = ok c := by
  obtain ⟨c, hc⟩ := hf
  exact ⟨c, by rw [hx]; simpa using hc⟩

/-- The innermost loop of `step_sum`: the bits `b` of one carry and one below. -/
theorem step_b_spec (tracks : Slice diagonal.Track) (cb t : Usize) (states : Slice (Aeneas.Std.Array field.Ext 2#usize))
    (target tb carry below : Usize) (total : field.Ext) (hc : Canon total) (hbelow : below.val < 2) :
    (∀ y, diagonal.step_sum_loop0_loop0_loop0 tracks cb t states target tb total carry below 0#usize = ok y →
      Canon y ∧ cubicOf y = cubicOf total + ∑ b ∈ Finset.range 2,
        stepTerm (tracksOf tracks.val) t.val cb.val (statesOf states.val) target.val tb.val carry.val below.val b) ∧
    (2 ^ tracks.val.length ≤ Usize.max → carry.val < states.val.length →
      ∃ y, diagonal.step_sum_loop0_loop0_loop0 tracks cb t states target tb total carry below 0#usize = ok y) := by
  unfold diagonal.step_sum_loop0_loop0_loop0
  have key := loop_ok
    (fun a : field.Ext × Usize =>
      diagonal.step_sum_loop0_loop0_loop0.body tracks cb t states target tb carry below a.1 a.2)
    (fun n a => a.2.val ≤ 2 ∧ n = 2 - a.2.val ∧ Canon a.1 ∧ cubicOf a.1 = cubicOf total + ∑ b ∈ Finset.range a.2.val,
        stepTerm (tracksOf tracks.val) t.val cb.val (statesOf states.val) target.val tb.val carry.val below.val b)
    (fun y : field.Ext => Canon y ∧ cubicOf y = cubicOf total + ∑ b ∈ Finset.range 2,
        stepTerm (tracksOf tracks.val) t.val cb.val (statesOf states.val) target.val tb.val carry.val below.val b)
    (fun _ => 2 ^ tracks.val.length ≤ Usize.max ∧ carry.val < states.val.length)
    (by
      rintro n ⟨tot, b⟩ c ⟨hb, rfl, hct, hv⟩ hbody
      dsimp only at hb hct hv hbody ⊢
      by_cases hlt : b.val < 2
      · right
        have h2 : (2#usize).val = 2 := by simp
        simp only [diagonal.step_sum_loop0_loop0_loop0.body, show b < 2#usize by scalar_tac, ite_true] at hbody
        obtain ⟨⟨factor, nc⟩, hts, hbody⟩ := bind_eq_ok hbody
        obtain ⟨hcf, hvf, hvn⟩ := (tracks_step_spec tracks t b carry).1 _ hts
        obtain ⟨tot1, ht1, hbody⟩ := bind_eq_ok hbody
        simp only [lift, bind_ok, Result.ok.injEq] at hbody
        subst hbody
        obtain ⟨r, hr, hrv⟩ := next_below_eq cb b below
        have hstep : Canon tot1 ∧ cubicOf tot1 = cubicOf tot +
            stepTerm (tracksOf tracks.val) t.val cb.val (statesOf states.val) target.val tb.val carry.val below.val b.val := by
          unfold stepTerm
          by_cases hn : nc = target
          · rw [ite_eq_left hn, hr, bind_ok] at ht1
            by_cases hrt : r = tb
            · rw [ite_eq_left hrt] at ht1
              by_cases hcs : carry.val < states.val.length
              · rw [slice_index states carry hcs, bind_ok, array_index _ below (by simp; omega), bind_ok, ext_mul_eq,
                  bind_ok, ext_add_eq, Result.ok.injEq] at ht1
                subst ht1
                refine ⟨canon_extAdd _ _, ?_⟩
                rw [ite_eq_left ⟨by rw [← hvn, hn], by rw [← hrv, hrt]⟩, extAdd_cubic, extMul_cubic, hvf,
                  statesOf_get states.val carry.val hcs below.val (by simp; omega)]
              · rw [slice_index_fail states carry (by omega)] at ht1
                exact absurd ht1 (by simp)
            · rw [ite_eq_right hrt, Result.ok.injEq] at ht1
              subst ht1
              refine ⟨hct, ?_⟩
              rw [ite_eq_right (fun h => hrt ((usize_eq_iff r tb).mpr (by rw [hrv]; exact h.2))), add_zero]
          · rw [ite_eq_right hn, Result.ok.injEq] at ht1
            subst ht1
            refine ⟨hct, ?_⟩
            rw [ite_eq_right (fun h => hn ((usize_eq_iff nc target).mpr (by rw [hvn]; exact h.1))), add_zero]
        refine ⟨_, 2 - (b.val + 1), rfl, by omega, by rw [add_one_val b (by scalar_tac)]; omega,
          by rw [add_one_val b (by scalar_tac)], hstep.1, ?_⟩
        dsimp only
        rw [add_one_val b (by scalar_tac), Finset.sum_range_succ, ← add_assoc, ← hv, hstep.2]
      · left
        simp only [diagonal.step_sum_loop0_loop0_loop0.body, show ¬ b < 2#usize by scalar_tac, ite_false,
          Result.ok.injEq] at hbody
        subst hbody
        exact ⟨_, rfl, hct, by rw [hv, show b.val = 2 by omega]⟩)
    (by
      rintro n ⟨tot, b⟩ ⟨hb, rfl, hct, hv⟩ ⟨hpow, hcs⟩
      dsimp only at hb hct hv ⊢
      by_cases hlt : b.val < 2
      · obtain ⟨⟨factor, nc⟩, hts⟩ := (tracks_step_spec tracks t b carry).2 (by omega) hpow
        obtain ⟨r, hr, -⟩ := next_below_eq cb b below
        obtain ⟨tot1, ht1⟩ : ∃ tot1, (if nc = target then do
            let i ← diagonal.next_below cb b below
            if i = tb then do
              let a ← Slice.index_usize states carry
              let e ← Aeneas.Std.Array.index_usize a below
              let e1 ← field.ext_mul e factor
              field.ext_add tot e1
            else ok tot
          else ok tot) = ok tot1 := by
          by_cases hn : nc = target
          · rw [ite_eq_left hn, hr, bind_ok]
            by_cases hrt : r = tb
            · rw [ite_eq_left hrt, slice_index states carry hcs, bind_ok, array_index _ below (by simp; omega), bind_ok,
                ext_mul_eq, bind_ok, ext_add_eq]
              exact ⟨_, rfl⟩
            · rw [ite_eq_right hrt]; exact ⟨_, rfl⟩
          · rw [ite_eq_right hn]; exact ⟨_, rfl⟩
        simp only [diagonal.step_sum_loop0_loop0_loop0.body, show b < 2#usize by scalar_tac, ite_true]
        exact bind_exists _ hts (bind_exists _ ht1 ⟨ControlFlow.cont (tot1, core.num.Usize.wrapping_add b 1#usize),
          by simp [lift]⟩)
      · exact ⟨ControlFlow.done tot, by
          simp only [diagonal.step_sum_loop0_loop0_loop0.body, show ¬ b < 2#usize by scalar_tac, ite_false]⟩)
    (fun _ _ _ _ hP _ => hP) _ (total, 0#usize) ⟨by simp, rfl, hc, by simp⟩
  exact ⟨key.1, fun h1 h2 => let ⟨y, hy, _⟩ := key.2 ⟨h1, h2⟩; ⟨y, hy⟩⟩


/-- A translated loop `for k < N { total = inner(total, k) }` that adds `g k` to a field value. -/
theorem sum_loop {body : field.Ext × Usize → Result (ControlFlow (field.Ext × Usize) field.Ext)}
    {inner : field.Ext → Usize → Result field.Ext} (N : Usize) (g : ℕ → Cubic) (P : Prop)
    (hbody : ∀ tot k, body (tot, k) = if k < N then do
        let total1 ← inner tot k
        let k1 ← lift (core.num.Usize.wrapping_add k 1#usize)
        ok (ControlFlow.cont (total1, k1))
      else ok (ControlFlow.done tot))
    (hinner : ∀ tot k y, Canon tot → k.val < N.val → inner tot k = ok y → Canon y ∧ cubicOf y = cubicOf tot + g k.val)
    (hok : P → ∀ tot k, Canon tot → k.val < N.val → ∃ y, inner tot k = ok y)
    (tot0 : field.Ext) (hc : Canon tot0) :
    (∀ y, loop body (tot0, 0#usize) = ok y → Canon y ∧ cubicOf y = cubicOf tot0 + ∑ k ∈ Finset.range N.val, g k) ∧
    (P → ∃ y, loop body (tot0, 0#usize) = ok y) := by
  have key := loop_ok body
    (fun n a => a.2.val ≤ N.val ∧ n = N.val - a.2.val ∧ Canon a.1 ∧
      cubicOf a.1 = cubicOf tot0 + ∑ k ∈ Finset.range a.2.val, g k)
    (fun y => Canon y ∧ cubicOf y = cubicOf tot0 + ∑ k ∈ Finset.range N.val, g k) (fun _ => P)
    (by
      rintro n ⟨tot, k⟩ c ⟨hk, rfl, hct, hv⟩ hb
      dsimp only at hk hct hv hb ⊢
      rw [hbody] at hb
      by_cases hlt : k.val < N.val
      · right
        have hmax : k.val < Usize.max := by scalar_tac
        rw [ite_eq_left (show k < N by scalar_tac)] at hb
        obtain ⟨tot1, h1, hb⟩ := bind_eq_ok hb
        simp only [lift, bind_ok, Result.ok.injEq] at hb
        subst hb
        obtain ⟨hc1, hv1⟩ := hinner tot k tot1 hct hlt h1
        refine ⟨_, N.val - (k.val + 1), rfl, by omega, by rw [add_one_val k hmax]; omega, by rw [add_one_val k hmax],
          hc1, ?_⟩
        dsimp only
        rw [add_one_val k hmax, Finset.sum_range_succ, ← add_assoc, ← hv, hv1]
      · left
        rw [ite_eq_right (show ¬ k < N by scalar_tac), Result.ok.injEq] at hb
        subst hb
        exact ⟨_, rfl, hct, by rw [hv, show k.val = N.val by omega]⟩)
    (by
      rintro n ⟨tot, k⟩ ⟨hk, rfl, hct, hv⟩ hP
      dsimp only at hk hct hv ⊢
      rw [hbody]
      by_cases hlt : k.val < N.val
      · obtain ⟨tot1, h1⟩ := hok hP tot k hct hlt
        rw [ite_eq_left (show k < N by scalar_tac), h1, bind_ok]
        exact ⟨ControlFlow.cont (tot1, core.num.Usize.wrapping_add k 1#usize), by simp [lift]⟩
      · rw [ite_eq_right (show ¬ k < N by scalar_tac)]
        exact ⟨_, rfl⟩)
    (fun _ _ _ _ hP _ => hP) _ (tot0, 0#usize) ⟨by simp, rfl, hc, by simp⟩
  exact ⟨key.1, fun h => let ⟨y, hy, _⟩ := key.2 h; ⟨y, hy⟩⟩

theorem step_below_spec (tracks : Slice diagonal.Track) (cb t : Usize) (states : Slice (Aeneas.Std.Array field.Ext 2#usize))
    (target tb carry : Usize) (total : field.Ext) (hc : Canon total) :
    (∀ y, diagonal.step_sum_loop0_loop0 tracks cb t states target tb total carry 0#usize = ok y →
      Canon y ∧ cubicOf y = cubicOf total + ∑ below ∈ Finset.range 2, ∑ b ∈ Finset.range 2,
        stepTerm (tracksOf tracks.val) t.val cb.val (statesOf states.val) target.val tb.val carry.val below b) ∧
    (2 ^ tracks.val.length ≤ Usize.max → carry.val < states.val.length →
      ∃ y, diagonal.step_sum_loop0_loop0 tracks cb t states target tb total carry 0#usize = ok y) := by
  have key := sum_loop (body := fun a => diagonal.step_sum_loop0_loop0.body tracks cb t states target tb carry a.1 a.2)
    (inner := fun tot k => diagonal.step_sum_loop0_loop0_loop0 tracks cb t states target tb tot carry k 0#usize)
    2#usize (fun below => ∑ b ∈ Finset.range 2,
        stepTerm (tracksOf tracks.val) t.val cb.val (statesOf states.val) target.val tb.val carry.val below b)
    (2 ^ tracks.val.length ≤ Usize.max ∧ carry.val < states.val.length) (fun _ _ => rfl)
    (fun tot k y hct hk h => (step_b_spec tracks cb t states target tb carry k tot hct (by simpa using hk)).1 y h)
    (fun hP tot k hct hk => (step_b_spec tracks cb t states target tb carry k tot hct (by simpa using hk)).2 hP.1 hP.2)
    total hc
  refine ⟨fun y h => ?_, fun h1 h2 => key.2 ⟨h1, h2⟩⟩
  have := key.1 y h
  simpa using this

theorem step_carry_spec (tracks : Slice diagonal.Track) (cb t : Usize) (states : Slice (Aeneas.Std.Array field.Ext 2#usize))
    (target tb carries : Usize) :
    (∀ y, diagonal.step_sum_loop0 tracks cb t states target tb carries field.EXT_ZERO 0#usize = ok y →
      Canon y ∧ cubicOf y = ∑ carry ∈ Finset.range carries.val, ∑ below ∈ Finset.range 2, ∑ b ∈ Finset.range 2,
        stepTerm (tracksOf tracks.val) t.val cb.val (statesOf states.val) target.val tb.val carry below b) ∧
    (2 ^ tracks.val.length ≤ Usize.max → carries.val ≤ states.val.length →
      ∃ y, diagonal.step_sum_loop0 tracks cb t states target tb carries field.EXT_ZERO 0#usize = ok y) := by
  have key := sum_loop (body := fun a => diagonal.step_sum_loop0.body tracks cb t states target tb carries a.1 a.2)
    (inner := fun tot k => diagonal.step_sum_loop0_loop0 tracks cb t states target tb tot k 0#usize)
    carries (fun carry => ∑ below ∈ Finset.range 2, ∑ b ∈ Finset.range 2,
        stepTerm (tracksOf tracks.val) t.val cb.val (statesOf states.val) target.val tb.val carry below b)
    (2 ^ tracks.val.length ≤ Usize.max ∧ carries.val ≤ states.val.length) (fun _ _ => rfl)
    (fun tot k y hct _ h => (step_below_spec tracks cb t states target tb k tot hct).1 y h)
    (fun hP tot k hct hk => (step_below_spec tracks cb t states target tb k tot hct).2 hP.1 (by omega))
    field.EXT_ZERO canon_zero
  refine ⟨fun y h => ?_, fun h1 h2 => key.2 ⟨h1, h2⟩⟩
  have := key.1 y h
  rwa [cubicOf_zero, zero_add] at this


theorem step_sum_spec (tracks : Slice diagonal.Track) (cb t : Usize) (states : Slice (Aeneas.Std.Array field.Ext 2#usize))
    (target tb : Usize) :
    (∀ y, diagonal.step_sum tracks cb t states target tb = ok y →
      Canon y ∧ cubicOf y = ∑ carry ∈ Finset.range (2 ^ tracks.val.length), ∑ below ∈ Finset.range 2,
        ∑ b ∈ Finset.range 2,
          stepTerm (tracksOf tracks.val) t.val cb.val (statesOf states.val) target.val tb.val carry below b) ∧
    (2 ^ tracks.val.length ≤ Usize.max → 2 ^ tracks.val.length ≤ states.val.length →
      ∃ y, diagonal.step_sum tracks cb t states target tb = ok y) := by
  unfold diagonal.step_sum
  constructor
  · intro y h
    obtain ⟨carries, hcs, h⟩ := bind_eq_ok h
    have hcv := (carry_states_spec tracks).1 carries hcs
    rw [← hcv]
    exact (step_carry_spec tracks cb t states target tb carries).1 y h
  · intro h1 h2
    obtain ⟨carries, hcs⟩ := (carry_states_spec tracks).2 h1
    have hcv := (carry_states_spec tracks).1 carries hcs
    obtain ⟨y, hy⟩ := (step_carry_spec tracks cb t states target tb carries).2 h1 (by rw [hcv]; exact h2)
    exact ⟨y, by rw [hcs, bind_ok, hy]⟩

theorem walkStep_eq (T : List (Shrink.Track Cubic)) (count t : ℕ) (S : ℕ → ℕ → Cubic) (c bl : ℕ) :
    Shrink.walkStep T count t S c bl = ∑ carry ∈ Finset.range (2 ^ T.length), ∑ below ∈ Finset.range 2,
      ∑ b ∈ Finset.range 2, stepTerm T t (count / 2 ^ t % 2) S c bl carry below b := rfl

theorem walkStep_congr (T : List (Shrink.Track Cubic)) (count t : ℕ) (S S' : ℕ → ℕ → Cubic)
    (h : ∀ c bl, c < 2 ^ T.length → bl < 2 → S c bl = S' c bl) (c bl : ℕ) :
    Shrink.walkStep T count t S c bl = Shrink.walkStep T count t S' c bl := by
  rw [walkStep_eq, walkStep_eq]
  apply Finset.sum_congr rfl
  intro carry hc
  apply Finset.sum_congr rfl
  intro below hb
  apply Finset.sum_congr rfl
  intro b _
  rw [Finset.mem_range] at hc hb
  simp only [stepTerm, h carry below hc hb]

/-- The states as the copy reads them, by carry: below and not. -/
noncomputable def stateRows (states : List (Aeneas.Std.Array field.Ext 2#usize)) : List (List Cubic) :=
  states.map fun a => a.val.map cubicOf

theorem statesOf_rows (states : List (Aeneas.Std.Array field.Ext 2#usize)) (n : ℕ) (f : ℕ → ℕ → Cubic)
    (h : stateRows states = (List.range n).map fun c => [f c 0, f c 1]) (c bl : ℕ) (hc : c < n) (hbl : bl < 2) :
    statesOf states c bl = f c bl := by
  have hlen : states.length = n := by simpa [stateRows] using congrArg List.length h
  have hc' : c < states.length := by omega
  have hrow := congrArg (fun l => l[c]?) h
  simp only [stateRows, List.getElem?_map, List.getElem?_eq_getElem hc', List.getElem?_range hc, Option.map_some,
    Option.some.injEq] at hrow
  simp only [statesOf, List.getElem?_eq_getElem hc', hrow]
  rcases (show bl = 0 ∨ bl = 1 by omega) with rfl | rfl <;> rfl

theorem walk_step_spec (tracks : Slice diagonal.Track) (count t : Usize) (states : Slice (Aeneas.Std.Array field.Ext 2#usize)) :
    (∀ y, diagonal.walk_step tracks count t states = ok y →
      stateRows y.val = (List.range (2 ^ tracks.val.length)).map (fun c =>
        [Shrink.walkStep (tracksOf tracks.val) count.val t.val (statesOf states.val) c 0,
          Shrink.walkStep (tracksOf tracks.val) count.val t.val (statesOf states.val) c 1]) ∧
      ∀ a ∈ y.val, ∀ e ∈ a.val, Canon e) ∧
    (2 ^ tracks.val.length ≤ Usize.max → 2 ^ tracks.val.length ≤ states.val.length →
      ∃ y, diagonal.walk_step tracks count t states = ok y) := by
  set T := tracksOf tracks.val
  have hT : T.length = tracks.val.length := tracksOf_length _
  set f : ℕ → List Cubic := fun c =>
    [Shrink.walkStep T count.val t.val (statesOf states.val) c 0, Shrink.walkStep T count.val t.val (statesOf states.val) c 1]
  obtain ⟨cb, hcb, hcbv⟩ := bit_at_eq count t
  have hsum (target tb : Usize) (y : field.Ext) (h : diagonal.step_sum tracks cb t states target tb = ok y) :
      Canon y ∧ cubicOf y = Shrink.walkStep T count.val t.val (statesOf states.val) target.val tb.val := by
    obtain ⟨hc, hv⟩ := (step_sum_spec tracks cb t states target tb).1 y h
    rw [walkStep_eq, hT, ← hcbv]
    exact ⟨hc, hv⟩
  unfold diagonal.walk_step
  constructor
  · intro y h
    obtain ⟨carries, hcs, h⟩ := bind_eq_ok h
    have hcv := (carry_states_spec tracks).1 carries hcs
    rw [hcb, bind_ok] at h
    unfold diagonal.walk_step_loop at h
    have key := loop_sound
      (fun a : alloc.vec.Vec (Aeneas.Std.Array field.Ext 2#usize) × Usize =>
        diagonal.walk_step_loop.body tracks t states carries cb a.1 a.2)
      (fun n a => a.2.val ≤ carries.val ∧ n = carries.val - a.2.val ∧
        stateRows a.1.val = (List.range a.2.val).map f ∧ ∀ r ∈ a.1.val, ∀ e ∈ r.val, Canon e)
      (fun y => stateRows y.val = (List.range carries.val).map f ∧ ∀ r ∈ y.val, ∀ e ∈ r.val, Canon e)
      (by
        rintro n ⟨nx, target⟩ c ⟨hk, rfl, hv, hcn⟩ hb
        dsimp only at hk hv hcn hb ⊢
        by_cases hlt : target.val < carries.val
        · right
          have hmax : target.val < Usize.max := by scalar_tac
          simp only [diagonal.walk_step_loop.body, show target < carries by scalar_tac, ite_true] at hb
          obtain ⟨y0, h0, hb⟩ := bind_eq_ok hb
          obtain ⟨y1, h1, hb⟩ := bind_eq_ok hb
          obtain ⟨nx1, hpush, hb⟩ := bind_eq_ok hb
          simp only [lift, bind_ok, Result.ok.injEq] at hb
          subst hb
          obtain ⟨hc0, hv0⟩ := hsum target 0#usize y0 h0
          obtain ⟨hc1, hv1⟩ := hsum target 1#usize y1 h1
          refine ⟨_, carries.val - (target.val + 1), rfl, by omega, by rw [add_one_val target hmax]; omega,
            by rw [add_one_val target hmax], ?_, ?_⟩
          · dsimp only
            rw [vec_push_val hpush, add_one_val target hmax, List.range_succ, List.map_append, ← hv, stateRows,
              List.map_append, ← stateRows]
            simp [f, Aeneas.Std.Array.make, hv0, hv1]
          · dsimp only
            rw [vec_push_val hpush]
            intro a ha e he
            simp only [List.mem_append, List.mem_singleton] at ha
            rcases ha with ha | rfl
            · exact hcn a ha e he
            · have hmk : (Aeneas.Std.Array.make 2#usize [y0, y1]).val = [y0, y1] := by simp [Aeneas.Std.Array.make]
              rw [hmk, List.mem_cons, List.mem_singleton] at he
              rcases he with rfl | rfl
              · exact hc0
              · exact hc1
        · left
          simp only [diagonal.walk_step_loop.body, show ¬ target < carries by scalar_tac, ite_false,
            Result.ok.injEq] at hb
          subst hb
          exact ⟨_, rfl, by rw [hv, show target.val = carries.val by omega], hcn⟩)
      _ (alloc.vec.Vec.new _, 0#usize) y ⟨by simp, rfl, by simp [stateRows, alloc.vec.Vec.new], by
        simp [alloc.vec.Vec.new]⟩ h
    rw [hcv] at key
    exact key
  · intro h1 h2
    obtain ⟨carries, hcs⟩ := (carry_states_spec tracks).2 h1
    have hcv := (carry_states_spec tracks).1 carries hcs
    rw [hcs, bind_ok, hcb, bind_ok]
    unfold diagonal.walk_step_loop
    refine loop_total
      (fun a : alloc.vec.Vec (Aeneas.Std.Array field.Ext 2#usize) × Usize =>
        diagonal.walk_step_loop.body tracks t states carries cb a.1 a.2)
      (fun n a => a.2.val ≤ carries.val ∧ n = carries.val - a.2.val ∧ a.1.val.length = a.2.val) (fun _ => True)
      ?_ _ (alloc.vec.Vec.new _, 0#usize) ⟨by simp, rfl, by simp [alloc.vec.Vec.new]⟩ |>.imp fun _ h => h.1
    rintro n ⟨nx, target⟩ ⟨hk, rfl, hl⟩
    dsimp only at hk hl ⊢
    by_cases hlt : target.val < carries.val
    · right
      have hmax : target.val < Usize.max := by scalar_tac
      obtain ⟨y0, h0⟩ := (step_sum_spec tracks cb t states target 0#usize).2 h1 h2
      obtain ⟨y1, h1'⟩ := (step_sum_spec tracks cb t states target 1#usize).2 h1 h2
      obtain ⟨nx1, hpush, hnx1⟩ := vec_push_ok nx (Aeneas.Std.Array.make 2#usize [y0, y1]) (by rw [hl]; omega)
      refine ⟨(nx1, core.num.Usize.wrapping_add target 1#usize), carries.val - (target.val + 1), ?_, by omega,
        by rw [add_one_val target hmax]; omega, by rw [add_one_val target hmax], by
          rw [hnx1, add_one_val target hmax]; simp [hl]⟩
      simp only [diagonal.walk_step_loop.body, show target < carries by scalar_tac, ite_true, h0, bind_ok, h1',
        hpush, lift]
    · left
      exact ⟨nx, by simp only [diagonal.walk_step_loop.body, show ¬ target < carries by scalar_tac, ite_false],
        trivial⟩


/-! ## `walk_states` and `diagonal` -/

/-- The walk's states before any bit, by carry. -/
noncomputable def walkRows (T : List (Shrink.Track Cubic)) (count width : ℕ) : List (List Cubic) :=
  (List.range (2 ^ T.length)).map fun c => [Shrink.walk T count width c 0, Shrink.walk T count width c 1]

/-- The state of carry `carry` before any bit. -/
def initRow (carry : Usize) : Aeneas.Std.Array field.Ext 2#usize :=
  if carry = 0#usize then Aeneas.Std.Array.make 2#usize [field.EXT_ONE, field.EXT_ZERO]
  else Aeneas.Std.Array.repeat 2#usize field.EXT_ZERO

theorem walk_states_init (carries : Usize) (T : List (Shrink.Track Cubic)) (count : ℕ) :
    (∀ y, diagonal.walk_states_loop0 carries (alloc.vec.Vec.new _) 0#usize = ok y →
      stateRows y.val = (List.range carries.val).map (fun c => [Shrink.walk T count 0 c 0, Shrink.walk T count 0 c 1]) ∧
      ∀ r ∈ y.val, ∀ e ∈ r.val, Canon e) ∧
    (carries.val ≤ Usize.max → ∃ y, diagonal.walk_states_loop0 carries (alloc.vec.Vec.new _) 0#usize = ok y) := by
  have hrow (carry : Usize) :
      stateRows [initRow carry] = [[Shrink.walk T count 0 carry.val 0, Shrink.walk T count 0 carry.val 1]] := by
    unfold initRow
    by_cases h : carry = 0#usize
    · have h0 : carry.val = 0 := by rw [h]; rfl
      simp [stateRows, h, Shrink.walk, cubicOf_one, cubicOf_zero]
    · have h0 : carry.val ≠ 0 := fun e => h ((usize_eq_iff carry 0#usize).mpr (by rw [e]; rfl))
      simp [stateRows, h, h0, Shrink.walk, Aeneas.Std.Array.repeat, cubicOf_zero]
  have hcanon (carry : Usize) : ∀ e ∈ (initRow carry).val, Canon e := by
    intro e he
    unfold initRow at he
    by_cases h : carry = 0#usize
    · simp only [h, ite_true, Aeneas.Std.Array.make] at he
      simp at he
      rcases he with rfl | rfl
      · exact canon_one
      · exact canon_zero
    · simp only [h, ite_false, Aeneas.Std.Array.repeat] at he
      simp at he
      rw [he]
      exact canon_zero
  unfold diagonal.walk_states_loop0
  have key := loop_ok
    (fun a : alloc.vec.Vec (Aeneas.Std.Array field.Ext 2#usize) × Usize => diagonal.walk_states_loop0.body carries a.1 a.2)
    (fun n a => a.2.val ≤ carries.val ∧ n = carries.val - a.2.val ∧ a.1.val.length = a.2.val ∧
      stateRows a.1.val = (List.range a.2.val).map (fun c => [Shrink.walk T count 0 c 0, Shrink.walk T count 0 c 1]) ∧
      ∀ r ∈ a.1.val, ∀ e ∈ r.val, Canon e)
    (fun y => stateRows y.val = (List.range carries.val).map (fun c => [Shrink.walk T count 0 c 0, Shrink.walk T count 0 c 1]) ∧
      ∀ r ∈ y.val, ∀ e ∈ r.val, Canon e)
    (fun _ => carries.val ≤ Usize.max)
    (by
      rintro n ⟨st, carry⟩ c ⟨hk, rfl, hl, hv, hcn⟩ hb
      dsimp only at hk hl hv hcn hb ⊢
      by_cases hlt : carry.val < carries.val
      · right
        have hmax : carry.val < Usize.max := by scalar_tac
        simp only [diagonal.walk_states_loop0.body, show carry < carries by scalar_tac, ite_true] at hb
        obtain ⟨st1, hpush, hb⟩ := bind_eq_ok hb
        simp only [lift, bind_ok, Result.ok.injEq] at hb
        subst hb
        have hst1 : st1.val = st.val ++ [initRow carry] := by
          unfold initRow
          by_cases h : carry = 0#usize
          · rw [ite_eq_left h] at hpush ⊢; exact vec_push_val hpush
          · rw [ite_eq_right h] at hpush ⊢; exact vec_push_val hpush
        refine ⟨_, carries.val - (carry.val + 1), rfl, by omega, by rw [add_one_val carry hmax]; omega,
          by rw [add_one_val carry hmax], by simp [hst1, add_one_val carry hmax, hl], ?_, ?_⟩
        · dsimp only
          rw [hst1, add_one_val carry hmax, List.range_succ, List.map_append, ← hv, stateRows, List.map_append,
            ← stateRows, ← stateRows, hrow carry]
          rfl
        · dsimp only
          rw [hst1]
          intro r hr e he
          rcases List.mem_append.mp hr with hr | hr
          · exact hcn r hr e he
          · rw [List.mem_singleton] at hr
            subst hr
            exact hcanon carry e he
      · left
        simp only [diagonal.walk_states_loop0.body, show ¬ carry < carries by scalar_tac, ite_false,
          Result.ok.injEq] at hb
        subst hb
        exact ⟨_, rfl, by rw [hv, show carry.val = carries.val by omega], hcn⟩)
    (by
      rintro n ⟨st, carry⟩ ⟨hk, rfl, hl, hv, hcn⟩ hP
      dsimp only at hk hl hv hcn ⊢
      by_cases hlt : carry.val < carries.val
      · have hlen : st.val.length < Usize.max := by omega
        by_cases h : carry = 0#usize
        · obtain ⟨st1, hpush, -⟩ := vec_push_ok st (Aeneas.Std.Array.make 2#usize [field.EXT_ONE, field.EXT_ZERO]) hlen
          exact ⟨ControlFlow.cont (st1, core.num.Usize.wrapping_add carry 1#usize), by
            simp only [diagonal.walk_states_loop0.body, show carry < carries by scalar_tac, ite_true, ite_eq_left h,
              hpush, bind_ok, lift]⟩
        · obtain ⟨st1, hpush, -⟩ := vec_push_ok st (Aeneas.Std.Array.repeat 2#usize field.EXT_ZERO) hlen
          exact ⟨ControlFlow.cont (st1, core.num.Usize.wrapping_add carry 1#usize), by
            simp only [diagonal.walk_states_loop0.body, show carry < carries by scalar_tac, ite_true, ite_eq_right h,
              hpush, bind_ok, lift]⟩
      · exact ⟨ControlFlow.done st, by
          simp only [diagonal.walk_states_loop0.body, show ¬ carry < carries by scalar_tac, ite_false]⟩)
    (fun _ _ _ _ hP _ => hP) _ (alloc.vec.Vec.new _, 0#usize)
    ⟨by simp, rfl, by simp [alloc.vec.Vec.new], by simp [stateRows, alloc.vec.Vec.new], by simp [alloc.vec.Vec.new]⟩
  exact ⟨key.1, fun h => let ⟨y, hy, _⟩ := key.2 h; ⟨y, hy⟩⟩


theorem walk_states_spec (tracks : Slice diagonal.Track) (count width : Usize) :
    (∀ y, diagonal.walk_states tracks count width = ok y →
      stateRows y.val = walkRows (tracksOf tracks.val) count.val width.val ∧ ∀ r ∈ y.val, ∀ e ∈ r.val, Canon e) ∧
    (2 ^ tracks.val.length ≤ Usize.max → ∃ y, diagonal.walk_states tracks count width = ok y) := by
  set T := tracksOf tracks.val
  have hT : T.length = tracks.val.length := tracksOf_length _
  have hstep (st : alloc.vec.Vec (Aeneas.Std.Array field.Ext 2#usize)) (t : Usize)
      (hv : stateRows st.val = walkRows T count.val t.val) (y : alloc.vec.Vec (Aeneas.Std.Array field.Ext 2#usize))
      (h : diagonal.walk_step tracks count t (alloc.vec.Vec.deref st) = ok y) :
      stateRows y.val = walkRows T count.val (t.val + 1) ∧ ∀ r ∈ y.val, ∀ e ∈ r.val, Canon e := by
    obtain ⟨hy, hc⟩ := (walk_step_spec tracks count t (alloc.vec.Vec.deref st)).1 y h
    refine ⟨?_, hc⟩
    rw [hy, walkRows, hT]
    apply List.map_congr_left
    intro c hc'
    rw [List.mem_range] at hc'
    have hS : ∀ c bl, c < 2 ^ T.length → bl < 2 →
        statesOf (alloc.vec.Vec.deref st).val c bl = Shrink.walk T count.val t.val c bl := by
      intro c bl hc hbl
      rw [deref_val]
      exact statesOf_rows st.val (2 ^ T.length) (Shrink.walk T count.val t.val) hv c bl hc hbl
    rw [walkStep_congr T count.val t.val _ _ hS, walkStep_congr T count.val t.val _ _ hS]
    rfl
  unfold diagonal.walk_states
  constructor
  · intro y h
    obtain ⟨carries, hcs, h⟩ := bind_eq_ok h
    have hcv := (carry_states_spec tracks).1 carries hcs
    obtain ⟨st0, h0, h⟩ := bind_eq_ok h
    obtain ⟨hv0, hc0⟩ := (walk_states_init carries T count.val).1 st0 h0
    unfold diagonal.walk_states_loop1 at h
    exact loop_sound
      (fun a : alloc.vec.Vec (Aeneas.Std.Array field.Ext 2#usize) × Usize =>
        diagonal.walk_states_loop1.body tracks count width a.1 a.2)
      (fun n a => a.2.val ≤ width.val ∧ n = width.val - a.2.val ∧ stateRows a.1.val = walkRows T count.val a.2.val ∧
        ∀ r ∈ a.1.val, ∀ e ∈ r.val, Canon e)
      (fun y => stateRows y.val = walkRows T count.val width.val ∧ ∀ r ∈ y.val, ∀ e ∈ r.val, Canon e)
      (by
        rintro n ⟨st, t⟩ c ⟨hk, rfl, hv, hcn⟩ hb
        dsimp only at hk hv hcn hb ⊢
        by_cases hlt : t.val < width.val
        · right
          have hmax : t.val < Usize.max := by scalar_tac
          simp only [diagonal.walk_states_loop1.body, show t < width by scalar_tac, ite_true] at hb
          obtain ⟨st1, h1, hb⟩ := bind_eq_ok hb
          simp only [lift, bind_ok, Result.ok.injEq] at hb
          subst hb
          obtain ⟨hv1, hc1⟩ := hstep st t hv st1 h1
          exact ⟨_, width.val - (t.val + 1), rfl, by omega, by rw [add_one_val t hmax]; omega,
            by rw [add_one_val t hmax], by rw [add_one_val t hmax]; exact hv1, hc1⟩
        · left
          simp only [diagonal.walk_states_loop1.body, show ¬ t < width by scalar_tac, ite_false,
            Result.ok.injEq] at hb
          subst hb
          exact ⟨_, rfl, by rw [hv, show t.val = width.val by omega], hcn⟩)
      _ (st0, 0#usize) y ⟨by simp, rfl, by rw [hv0, hcv, walkRows, hT]; rfl, hc0⟩ h
  · intro hP
    obtain ⟨carries, hcs⟩ := (carry_states_spec tracks).2 hP
    have hcv := (carry_states_spec tracks).1 carries hcs
    obtain ⟨st0, h0⟩ := (walk_states_init carries T count.val).2 (by rw [hcv]; exact hP)
    obtain ⟨hv0, -⟩ := (walk_states_init carries T count.val).1 st0 h0
    rw [hcs, bind_ok, h0, bind_ok]
    unfold diagonal.walk_states_loop1
    refine (loop_total
      (fun a : alloc.vec.Vec (Aeneas.Std.Array field.Ext 2#usize) × Usize =>
        diagonal.walk_states_loop1.body tracks count width a.1 a.2)
      (fun n a => a.2.val ≤ width.val ∧ n = width.val - a.2.val ∧ stateRows a.1.val = walkRows T count.val a.2.val)
      (fun _ => True) ?_ _ (st0, 0#usize) ⟨by simp, rfl, by rw [hv0, hcv, walkRows, hT]; rfl⟩).imp fun _ h => h.1
    rintro n ⟨st, t⟩ ⟨hk, rfl, hv⟩
    dsimp only at hk hv ⊢
    by_cases hlt : t.val < width.val
    · right
      have hmax : t.val < Usize.max := by scalar_tac
      have hlen : st.val.length = 2 ^ tracks.val.length := by
        have := congrArg List.length hv
        simpa [stateRows, walkRows, hT] using this
      obtain ⟨st1, h1⟩ := (walk_step_spec tracks count t (alloc.vec.Vec.deref st)).2 hP (by rw [deref_val, hlen])
      obtain ⟨hv1, -⟩ := hstep st t hv st1 h1
      refine ⟨(st1, core.num.Usize.wrapping_add t 1#usize), width.val - (t.val + 1), ?_, by omega,
        by rw [add_one_val t hmax]; omega, by rw [add_one_val t hmax], by rw [add_one_val t hmax]; exact hv1⟩
      simp only [diagonal.walk_states_loop1.body, show t < width by scalar_tac, ite_true, h1, bind_ok, lift]
    · left
      exact ⟨st, by simp only [diagonal.walk_states_loop1.body, show ¬ t < width by scalar_tac, ite_false], trivial⟩

theorem vec_index0 {α : Type} (v : alloc.vec.Vec α) (h : 0 < v.val.length) :
    alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice α) v 0#usize = ok v.val[0] :=
  vec_index v 0#usize h

theorem array_index1 {α : Type} {n : Usize} (a : Aeneas.Std.Array α n) (h : 1 < a.val.length) :
    Aeneas.Std.Array.index_usize a 1#usize = ok a.val[1] :=
  array_index a 1#usize h

theorem diagonal_spec (tracks : Slice diagonal.Track) (count : Usize) :
    (∀ y, diagonal.diagonal tracks count = ok y → Canon y ∧ cubicOf y = Shrink.diagonal (tracksOf tracks.val) count.val) ∧
    (2 ^ tracks.val.length ≤ Usize.max → ∃ y, diagonal.diagonal tracks count = ok y) := by
  obtain ⟨w, hw, hwv⟩ := walk_bits_eq tracks count
  have hrow (st : alloc.vec.Vec (Aeneas.Std.Array field.Ext 2#usize)) (hv : stateRows st.val = walkRows (tracksOf tracks.val) count.val w.val) :
      ∃ h0 : 0 < st.val.length, ∃ h1 : 1 < (st.val[0]'h0).val.length,
        cubicOf ((st.val[0]'h0).val[1]'h1) = Shrink.diagonal (tracksOf tracks.val) count.val := by
    have hlen : st.val.length = 2 ^ (tracksOf tracks.val).length := by
      have := congrArg List.length hv
      simp only [stateRows, walkRows, List.length_map, List.length_range] at this
      exact this
    have h0 : 0 < st.val.length := by rw [hlen]; positivity
    have hr := congrArg (fun l => l[0]?) hv
    simp only [stateRows, walkRows, List.getElem?_map, List.getElem?_eq_getElem h0,
      List.getElem?_range (show 0 < 2 ^ (tracksOf tracks.val).length by positivity), Option.map_some,
      Option.some.injEq] at hr
    have h1 : 1 < (st.val[0]'h0).val.length := by simp
    refine ⟨h0, h1, ?_⟩
    have := congrArg (fun l => l[1]?) hr
    simp only [List.getElem?_map, List.getElem?_eq_getElem h1, Option.map_some] at this
    simp only [Shrink.diagonal, ← hwv]
    simpa using this
  unfold diagonal.diagonal
  rw [hw, bind_ok]
  constructor
  · intro y h
    obtain ⟨st, hst, h⟩ := bind_eq_ok h
    obtain ⟨hv, hc⟩ := (walk_states_spec tracks count w).1 st hst
    obtain ⟨h0, h1, hval⟩ := hrow st hv
    rw [vec_index0 st h0, bind_ok, array_index1 _ h1, Result.ok.injEq] at h
    subst h
    exact ⟨hc _ (List.getElem_mem h0) _ (List.getElem_mem h1), hval⟩
  · intro hP
    obtain ⟨st, hst⟩ := (walk_states_spec tracks count w).2 hP
    obtain ⟨hv, -⟩ := (walk_states_spec tracks count w).1 st hst
    obtain ⟨h0, h1, -⟩ := hrow st hv
    exact ⟨_, by rw [hst, bind_ok, vec_index0 st h0, bind_ok, array_index1 _ h1]⟩


/-! ## `Sound` and `Complete` -/

/-- A factor and a carry, as their words and value. -/
def extUsizeValue (x : field.Ext × Usize) : (ℕ × ℕ × ℕ) × ℕ := (extValue x.1, x.2.val)

theorem extUsizeValue_injective : Function.Injective extUsizeValue := by
  rintro ⟨a, u⟩ ⟨b, v⟩ h
  simp only [extUsizeValue, Prod.mk.injEq] at h
  rw [extValue_injective h.1, usizeValue_injective h.2]

/-- The walk's states, by carry, as their words. -/
def statesValue (v : alloc.vec.Vec (Aeneas.Std.Array field.Ext 2#usize)) : List (List (ℕ × ℕ × ℕ)) :=
  v.val.map fun a => a.val.map extValue

theorem statesValue_injective : Function.Injective statesValue := by
  intro a b h
  have row : Function.Injective fun r : Aeneas.Std.Array field.Ext 2#usize => r.val.map extValue := by
    intro r q hrq
    exact Aeneas.Std.Array.ext r q ((List.map_injective_iff.mpr extValue_injective) hrq)
  exact vec_val_injective ((List.map_injective_iff.mpr row) h)

theorem statesValue_rows (v : alloc.vec.Vec (Aeneas.Std.Array field.Ext 2#usize)) (rows : List (List Cubic))
    (h : stateRows v.val = rows) (hc : ∀ r ∈ v.val, ∀ e ∈ r.val, Canon e) :
    statesValue v = rows.map fun row => row.map cubicWords := by
  rw [← h, stateRows, List.map_map, statesValue]
  apply List.map_congr_left
  intro r hr
  rw [Function.comp_apply, List.map_map]
  exact List.map_congr_left fun e he => (cubicWords_cubicOf (hc r hr e he)).symm

theorem walk_bits_sound : Sound (fun x : Slice diagonal.Track × Usize => diagonal.walk_bits x.1 x.2) usizeValue
    (fun x => Shrink.walkBits (tracksOf x.1.val) x.2.val) :=
  sound_of_eq fun x => walk_bits_eq x.1 x.2

theorem walk_bits_complete : Complete (fun _ => True)
    (fun x : Slice diagonal.Track × Usize => diagonal.walk_bits x.1 x.2) usizeValue
    (fun x => Shrink.walkBits (tracksOf x.1.val) x.2.val) :=
  complete_of_eq walk_bits_sound (emptySlice, 0#usize) trivial fun x _ => walk_bits_eq x.1 x.2

theorem track_factor_sound : Sound
    (fun x : Slice field.Ext × Usize × Usize => diagonal.track_factor x.1 x.2.1 x.2.2) extValue
    (fun x => cubicWords (Shrink.trackFactor (x.1.val.map cubicOf) x.2.1.val x.2.2.val)) :=
  sound_of_eq fun x =>
    let ⟨_, h, hc, hv⟩ := track_factor_eq x.1 x.2.1 x.2.2
    ⟨_, h, by rw [← hv, cubicWords_cubicOf hc]⟩

theorem track_factor_complete : Complete (fun _ => True)
    (fun x : Slice field.Ext × Usize × Usize => diagonal.track_factor x.1 x.2.1 x.2.2) extValue
    (fun x => cubicWords (Shrink.trackFactor (x.1.val.map cubicOf) x.2.1.val x.2.2.val)) :=
  complete_of_eq track_factor_sound (emptySlice, 0#usize, 0#usize) trivial fun x _ =>
    let ⟨_, h, hc, hv⟩ := track_factor_eq x.1 x.2.1 x.2.2
    ⟨_, h, by rw [← hv, cubicWords_cubicOf hc]⟩

theorem next_below_sound : Sound (fun x : Usize × Usize × Usize => diagonal.next_below x.1 x.2.1 x.2.2) usizeValue
    (fun x => Shrink.nextBelow x.1.val x.2.1.val x.2.2.val) :=
  sound_of_eq fun x => next_below_eq x.1 x.2.1 x.2.2

theorem next_below_complete : Complete (fun _ => True)
    (fun x : Usize × Usize × Usize => diagonal.next_below x.1 x.2.1 x.2.2) usizeValue
    (fun x => Shrink.nextBelow x.1.val x.2.1.val x.2.2.val) :=
  complete_of_eq next_below_sound (0#usize, 0#usize, 0#usize) trivial fun x _ => next_below_eq x.1 x.2.1 x.2.2

theorem tracks_step_sound : Sound
    (fun x : Slice diagonal.Track × Usize × Usize × Usize => diagonal.tracks_step x.1 x.2.1 x.2.2.1 x.2.2.2)
    extUsizeValue
    (fun x => (cubicWords (Shrink.tracksStep x.2.1.val x.2.2.1.val (tracksOf x.1.val) x.2.2.2.val).1,
      (Shrink.tracksStep x.2.1.val x.2.2.1.val (tracksOf x.1.val) x.2.2.2.val).2)) := by
  rintro ⟨tracks, t, b, carry⟩ y h
  obtain ⟨hc, hv1, hv2⟩ := (tracks_step_spec tracks t b carry).1 y h
  show (extValue y.1, y.2.val) = (cubicWords (Shrink.tracksStep t.val b.val (tracksOf tracks.val) carry.val).1,
    (Shrink.tracksStep t.val b.val (tracksOf tracks.val) carry.val).2)
  rw [← hv1, cubicWords_cubicOf hc, ← hv2]

/-- The walk adds bits `b` of 0 or 1, and the verifier passes one or two tracks. -/
theorem tracks_step_complete : Complete
    (fun x : Slice diagonal.Track × Usize × Usize × Usize => x.2.2.1.val ≤ 1 ∧ 2 ^ x.1.val.length ≤ Usize.max)
    (fun x : Slice diagonal.Track × Usize × Usize × Usize => diagonal.tracks_step x.1 x.2.1 x.2.2.1 x.2.2.2)
    extUsizeValue
    (fun x => (cubicWords (Shrink.tracksStep x.2.1.val x.2.2.1.val (tracksOf x.1.val) x.2.2.2.val).1,
      (Shrink.tracksStep x.2.1.val x.2.2.1.val (tracksOf x.1.val) x.2.2.2.val).2)) :=
  complete_of tracks_step_sound ⟨(emptySlice, 0#usize, 0#usize, 0#usize), by
      have := usize_max_ge
      simp only [zero_usize_val, emptySlice]
      simp; omega⟩
    fun x hx => (tracks_step_spec x.1 x.2.1 x.2.2.1 x.2.2.2).2 hx.1 hx.2

theorem walk_step_sound : Sound
    (fun x : Slice diagonal.Track × Usize × Usize × Slice (Aeneas.Std.Array field.Ext 2#usize) =>
      diagonal.walk_step x.1 x.2.1 x.2.2.1 x.2.2.2) statesValue
    (fun x => (List.range (2 ^ x.1.val.length)).map fun c =>
      [cubicWords (Shrink.walkStep (tracksOf x.1.val) x.2.1.val x.2.2.1.val (statesOf x.2.2.2.val) c 0),
        cubicWords (Shrink.walkStep (tracksOf x.1.val) x.2.1.val x.2.2.1.val (statesOf x.2.2.2.val) c 1)]) := by
  rintro ⟨tracks, count, t, states⟩ y h
  obtain ⟨hv, hc⟩ := (walk_step_spec tracks count t states).1 y h
  rw [statesValue_rows y _ hv hc, List.map_map]
  rfl

/-- The verifier passes one or two tracks and the states of the step before, one per carry. -/
theorem walk_step_complete : Complete
    (fun x : Slice diagonal.Track × Usize × Usize × Slice (Aeneas.Std.Array field.Ext 2#usize) =>
      2 ^ x.1.val.length ≤ Usize.max ∧ 2 ^ x.1.val.length ≤ x.2.2.2.val.length)
    (fun x : Slice diagonal.Track × Usize × Usize × Slice (Aeneas.Std.Array field.Ext 2#usize) =>
      diagonal.walk_step x.1 x.2.1 x.2.2.1 x.2.2.2) statesValue
    (fun x => (List.range (2 ^ x.1.val.length)).map fun c =>
      [cubicWords (Shrink.walkStep (tracksOf x.1.val) x.2.1.val x.2.2.1.val (statesOf x.2.2.2.val) c 0),
        cubicWords (Shrink.walkStep (tracksOf x.1.val) x.2.1.val x.2.2.1.val (statesOf x.2.2.2.val) c 1)]) :=
  complete_of walk_step_sound ⟨(emptySlice, 0#usize, 0#usize,
      Slice.from [Aeneas.Std.Array.repeat 2#usize field.EXT_ZERO] (by have := usize_max_ge; simp; omega)), by
      have := usize_max_ge
      simp [emptySlice]; omega⟩
    fun x hx => (walk_step_spec x.1 x.2.1 x.2.2.1 x.2.2.2).2 hx.1 hx.2

theorem walk_states_sound : Sound
    (fun x : Slice diagonal.Track × Usize × Usize => diagonal.walk_states x.1 x.2.1 x.2.2) statesValue
    (fun x => (List.range (2 ^ x.1.val.length)).map fun c =>
      [cubicWords (Shrink.walk (tracksOf x.1.val) x.2.1.val x.2.2.val c 0),
        cubicWords (Shrink.walk (tracksOf x.1.val) x.2.1.val x.2.2.val c 1)]) := by
  rintro ⟨tracks, count, width⟩ y h
  obtain ⟨hv, hc⟩ := (walk_states_spec tracks count width).1 y h
  rw [statesValue_rows y _ hv hc, walkRows, List.map_map, tracksOf_length]
  rfl

/-- The verifier passes one or two tracks. -/
theorem walk_states_complete : Complete
    (fun x : Slice diagonal.Track × Usize × Usize => 2 ^ x.1.val.length ≤ Usize.max)
    (fun x : Slice diagonal.Track × Usize × Usize => diagonal.walk_states x.1 x.2.1 x.2.2) statesValue
    (fun x => (List.range (2 ^ x.1.val.length)).map fun c =>
      [cubicWords (Shrink.walk (tracksOf x.1.val) x.2.1.val x.2.2.val c 0),
        cubicWords (Shrink.walk (tracksOf x.1.val) x.2.1.val x.2.2.val c 1)]) :=
  complete_of walk_states_sound ⟨(emptySlice, 0#usize, 0#usize), by have := usize_max_ge; simp [emptySlice]; omega⟩
    fun x hx => (walk_states_spec x.1 x.2.1 x.2.2).2 hx

theorem diagonal_sound : Sound (fun x : Slice diagonal.Track × Usize => diagonal.diagonal x.1 x.2) extValue
    (fun x => cubicWords (Shrink.diagonal (tracksOf x.1.val) x.2.val)) := by
  rintro ⟨tracks, count⟩ y h
  obtain ⟨hc, hv⟩ := (diagonal_spec tracks count).1 y h
  show extValue y = cubicWords (Shrink.diagonal (tracksOf tracks.val) count.val)
  rw [← hv, cubicWords_cubicOf hc]

/-- The verifier passes one or two tracks. -/
theorem diagonal_complete : Complete (fun x : Slice diagonal.Track × Usize => 2 ^ x.1.val.length ≤ Usize.max)
    (fun x : Slice diagonal.Track × Usize => diagonal.diagonal x.1 x.2) extValue
    (fun x => cubicWords (Shrink.diagonal (tracksOf x.1.val) x.2.val)) :=
  complete_of diagonal_sound ⟨(emptySlice, 0#usize), by have := usize_max_ge; simp [emptySlice]; omega⟩
    fun x hx => (diagonal_spec x.1 x.2).2 hx

end CompressionLeanRust
