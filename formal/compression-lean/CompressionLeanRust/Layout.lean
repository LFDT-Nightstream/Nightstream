import NeoVerified.Funs
import CompressionLean.Ports.Shrink.Layout
import CompressionLeanRust.Shape

namespace CompressionLeanRust

open Aeneas Aeneas.Std Result
open CompressionLean.Shrink (wide narrow variablesFor)
open neo_verified

theorem numBits_eq : UScalarTy.Usize.numBits = System.Platform.numBits := by simp

theorem numBits_le : UScalarTy.Usize.numBits ≤ 64 := by
  cases System.Platform.numBits_eq <;> simp_all

theorem numBits_pos : 1 ≤ UScalarTy.Usize.numBits := by
  cases System.Platform.numBits_eq <;> simp_all

theorem bits_val : core.num.Usize.BITS.val = UScalarTy.Usize.numBits := by
  simp [core.num.Usize.BITS]

theorem leading_zeros_val (x : Usize) (h : 0 < x.val) :
    (core.num.Usize.leading_zeros x).val = UScalarTy.Usize.numBits - Nat.log 2 x.val - 1 := by
  have hb : x.bv ≠ 0 := by
    intro e; simp [UScalar.val, e] at h
  have hn := numBits_le
  have he := numBits_eq
  unfold core.num.Usize.leading_zeros
  simp only [UScalar.val, BitVec.natCast_eq_ofNat, BitVec.toNat_ofNat, BitVec.leadingZeros, hb, ite_false,
    UScalarTy.U32_numBits_eq, UScalarTy.Usize_numBits_eq]
  apply Nat.mod_eq_of_lt
  omega

theorem leading_zeros_zero (x : Usize) (h : x.val = 0) :
    (core.num.Usize.leading_zeros x).val = UScalarTy.Usize.numBits := by
  have hn := numBits_le
  have he := numBits_eq
  have hb : x.bv = 0 := BitVec.eq_of_toNat_eq (by simpa using h)
  simp only [core.num.Usize.leading_zeros, BitVec.leadingZeros, hb, ite_true, UScalar.val,
    BitVec.natCast_eq_ofNat, BitVec.toNat_ofNat, UScalarTy.U32_numBits_eq, UScalarTy.Usize_numBits_eq]
  exact Nat.mod_eq_of_lt (by omega)

theorem usize_eq_zero (r : Usize) : r = 0#usize ↔ r.val = 0 := by
  constructor
  · rintro rfl; rfl
  · intro e; exact UScalar.val_eq_imp _ _ (by simp [e])

theorem clog_eq_log_pred {n : ℕ} (h : 2 ≤ n) : Nat.clog 2 n = Nat.log 2 (n - 1) + 1 := by
  apply le_antisymm
  · apply Nat.clog_le_of_le_pow
    have := Nat.lt_pow_succ_log_self (by decide : 1 < 2) (n - 1)
    rw [Nat.succ_eq_add_one] at this
    omega
  · have h1 : 2 ^ Nat.log 2 (n - 1) ≤ n - 1 := Nat.pow_log_le_self 2 (by omega)
    have := (Nat.lt_clog_iff_pow_lt (by decide : 1 < 2)).mpr (show 2 ^ Nat.log 2 (n - 1) < n by omega)
    omega

theorem checked_sub_some {ty : UScalarTy} (x y : UScalar ty) (h : y.val ≤ x.val) :
    ∃ z : UScalar ty, core.num.checked_sub_UScalar x y = some z ∧ z.val = x.val - y.val := by
  have := core.num.checked_sub_UScalar_bv_spec x y
  split at this
  · rename_i z hz; exact ⟨z, hz, this.2.1⟩
  · omega

theorem checked_sub_none {ty : UScalarTy} (x y : UScalar ty) (h : x.val < y.val) :
    core.num.checked_sub_UScalar x y = none := by
  have := core.num.checked_sub_UScalar_bv_spec x y
  split at this
  · omega
  · assumption

theorem unwrap_some {T : Type} (z : T) : core.option.Option.unwrap (some z) = ok z := rfl

theorem unwrap_none {T : Type} : core.option.Option.unwrap (none : Option T) = fail .panic := rfl

theorem floor_log2_eq (n : Usize) (h : 0 < n.val) :
    ∃ r, layout.floor_log2 n = ok r ∧ r.val = Nat.log 2 n.val := by
  have hlog : Nat.log 2 n.val < UScalarTy.Usize.numBits := Nat.log_lt_of_lt_pow (by omega) n.hBounds
  have hn := numBits_pos
  have h1 : (1#u32).val = 1 := rfl
  have hlz := leading_zeros_val n h
  have hbits := bits_val
  obtain ⟨a, ha, hav⟩ := checked_sub_some core.num.Usize.BITS 1#u32 (by omega)
  obtain ⟨b, hb, hbv⟩ := checked_sub_some a (core.num.Usize.leading_zeros n) (by omega)
  refine ⟨b, ?_, by omega⟩
  simp only [layout.floor_log2, U32.checked_sub, lift, bind_ok, ha, hb, unwrap_some]

theorem floor_log2_zero (n : Usize) (h : n.val = 0) : layout.floor_log2 n = fail .panic := by
  have hn := numBits_pos
  have h1 : (1#u32).val = 1 := rfl
  have hlz := leading_zeros_zero n h
  have hbits := bits_val
  obtain ⟨a, ha, hav⟩ := checked_sub_some core.num.Usize.BITS 1#u32 (by omega)
  have hb := checked_sub_none a (core.num.Usize.leading_zeros n) (by omega)
  simp only [layout.floor_log2, U32.checked_sub, lift, bind_ok, ha, hb, unwrap_some, unwrap_none]

theorem ceil_log2_eq (n : Usize) (h : 0 < n.val) :
    ∃ r, layout.ceil_log2 n = ok r ∧ r.val = Nat.clog 2 n.val := by
  have hn := numBits_pos
  have h1 : (1#usize).val = 1 := rfl
  have hbits := bits_val
  obtain ⟨m, hm, hmv⟩ := checked_sub_some n 1#usize (by omega)
  rw [h1] at hmv
  have hlz : (core.num.Usize.leading_zeros m).val =
      if m.val = 0 then UScalarTy.Usize.numBits else UScalarTy.Usize.numBits - Nat.log 2 m.val - 1 := by
    split
    · exact leading_zeros_zero m (by assumption)
    · exact leading_zeros_val m (by omega)
  have hlog : Nat.log 2 m.val < UScalarTy.Usize.numBits := by
    by_cases hm0 : m.val = 0
    · simp [hm0]
    · exact Nat.log_lt_of_lt_pow hm0 m.hBounds
  obtain ⟨b, hb, hbv⟩ := checked_sub_some core.num.Usize.BITS (core.num.Usize.leading_zeros m)
    (by rw [hlz, hbits]; split <;> omega)
  refine ⟨b, ?_, ?_⟩
  · simp only [layout.ceil_log2, U32.checked_sub, Usize.checked_sub, lift, bind_ok, hm, hb, unwrap_some]
  · rw [hbv, hlz, hbits]
    split
    · rename_i h0
      rw [Nat.clog_of_right_le_one (by omega)]; omega
    · rw [clog_eq_log_pred (by omega), ← hmv]; omega

theorem two_pow_eq (k : U32) (h : k.val < UScalarTy.Usize.numBits) :
    ∃ p, layout.two_pow k = ok p ∧ p.val = 2 ^ k.val := by
  have hn := numBits_pos
  have h1 : (1#u32).val = 1 := rfl
  have hbits := bits_val
  obtain ⟨a, ha, hav⟩ := checked_sub_some core.num.Usize.BITS 1#u32 (by omega)
  obtain ⟨c, hc, -⟩ := checked_sub_some a k (by omega)
  refine ⟨core.num.Usize.wrapping_shl 1#usize k,
    by simp only [layout.two_pow, U32.checked_sub, lift, bind_ok, ha, hc, unwrap_some], ?_⟩
  have he := numBits_eq
  have hk : k.val % System.Platform.numBits = k.val := Nat.mod_eq_of_lt (by omega)
  have hlt : 2 ^ k.val < 2 ^ System.Platform.numBits := Nat.pow_lt_pow_right (by decide) (by omega)
  simp only [UScalar.val, core.num.Usize.wrapping_shl_bv_eq, BitVec.shiftLeft_eq, BitVec.toNat_shiftLeft]
  have h1 : (Usize.bv 1#usize).toNat = 1 := by simp
  have hkv : k.bv.toNat = k.val := rfl
  rw [h1, hkv, hk, Nat.one_shiftLeft, he, Nat.mod_eq_of_lt hlt]

theorem split_ok (size : Usize) (h : 0 < size.val) :
    ∃ a, layout.split size = ok a ∧ usizeValues a = [wide size.val, narrow size.val] := by
  have hlog : Nat.log 2 size.val < UScalarTy.Usize.numBits := Nat.log_lt_of_lt_pow (by omega) size.hBounds
  have hpow : 2 ^ Nat.log 2 size.val ≤ size.val := Nat.pow_log_le_self 2 (by omega)
  have hsucc : size.val < 2 ^ Nat.log 2 size.val * 2 := by
    simpa [pow_succ] using Nat.lt_pow_succ_log_self (by decide : 1 < 2) size.val
  obtain ⟨i, hi, hiv⟩ := floor_log2_eq size h
  obtain ⟨w, hw, hwv⟩ := two_pow_eq i (by omega)
  rw [hiv] at hwv
  obtain ⟨r, hr, hrv⟩ := checked_sub_some size w (by omega)
  have hnarrow : narrow size.val = if r.val = 0 then 0 else 2 ^ Nat.clog 2 r.val := by
    rw [narrow, hrv, hwv]; rfl
  by_cases hr0 : r.val = 0
  · refine ⟨Aeneas.Std.Array.make 2#usize [w, 0#usize], ?_, ?_⟩
    · simp only [layout.split, hi, hw, Usize.checked_sub, lift, bind_ok, hr, unwrap_some,
        (usize_eq_zero r).mpr hr0, ite_true]
    · simp [usizeValues, Aeneas.Std.Array.make, hnarrow, hr0, wide, hwv]
  · have hclog : Nat.clog 2 r.val ≤ Nat.log 2 size.val :=
      Nat.clog_le_of_le_pow (by omega)
    obtain ⟨j, hj, hjv⟩ := ceil_log2_eq r (by omega)
    obtain ⟨p, hp, hpv⟩ := two_pow_eq j (by omega)
    refine ⟨Aeneas.Std.Array.make 2#usize [w, p], ?_, ?_⟩
    · have hne : ¬ r = 0#usize := fun e => hr0 ((usize_eq_zero r).mp e)
      simp only [layout.split, hi, hw, Usize.checked_sub, lift, bind_ok, hr, unwrap_some, hne,
        ite_false, hj, hp]
    · simp [usizeValues, Aeneas.Std.Array.make, hnarrow, hr0, wide, hwv, hpv, hjv]

theorem split_zero (size : Usize) (h : size.val = 0) : layout.split size = fail .panic := by
  simp only [layout.split, floor_log2_zero size h, bind_fail]

theorem split_sound : Sound layout.split usizeValues (fun s => [wide s.val, narrow s.val]) := by
  intro s y hy
  by_cases hs : s.val = 0
  · rw [split_zero s hs] at hy; exact absurd hy fail_not_ok
  · obtain ⟨a, ha, hav⟩ := split_ok s (by omega)
    rw [ha, Result.ok.injEq] at hy
    rw [← hy, hav]

theorem split_complete :
    Complete (fun s : Usize => 0 < s.val) layout.split usizeValues (fun s => [wide s.val, narrow s.val]) :=
  ⟨⟨1#usize, by decide⟩, split_ok⟩

theorem variables_ok (count : Usize) :
    ∃ v, layout.variables count = ok v ∧ v.val = variablesFor count.val := by
  have hle := numBits_le
  have hn := numBits_pos
  by_cases hc : count.val ≤ 2
  · refine ⟨1#usize, ?_, ?_⟩
    · have : count ≤ 2#usize := by scalar_tac
      simp only [layout.variables, this, ite_true]
    · have := Nat.clog_le_of_le_pow (show count.val ≤ 2 ^ 1 by rw [pow_one]; exact hc)
      rw [variablesFor, Nat.max_eq_left this]; rfl
  · obtain ⟨r, hr, hrv⟩ := ceil_log2_eq count (by omega)
    have hcl : Nat.clog 2 count.val ≤ UScalarTy.Usize.numBits := Nat.clog_le_of_le_pow count.hBounds.le
    have hpos : 0 < Nat.clog 2 count.val := Nat.clog_pos (by decide) (by omega)
    have hlt : r.val < 2 ^ UScalarTy.Usize.numBits := by
      have : UScalarTy.Usize.numBits < 2 ^ UScalarTy.Usize.numBits := Nat.lt_two_pow_self
      omega
    refine ⟨UScalar.cast .Usize r, ?_, ?_⟩
    · have : ¬ count ≤ 2#usize := by scalar_tac
      simp only [layout.variables, this, ite_false, hr, bind_ok]
    · rw [UScalar.cast_val_eq, Nat.mod_eq_of_lt hlt, hrv, variablesFor, Nat.max_eq_right hpos]

theorem variables_sound : Sound layout.variables usizeValue (fun c => variablesFor c.val) := by
  intro c y hy
  obtain ⟨v, hv, hvv⟩ := variables_ok c
  rw [hv, Result.ok.injEq] at hy
  rw [← hy]; exact hvv

theorem variables_complete :
    Complete (fun _ : Usize => True) layout.variables usizeValue (fun c => variablesFor c.val) :=
  ⟨⟨0#usize, trivial⟩, fun c _ => variables_ok c⟩

end CompressionLeanRust
