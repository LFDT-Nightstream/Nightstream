import CompressionLeanRust.Backend
import CompressionLeanRust.Layout
import CompressionLeanRust.Poseidon2

/-! The translated helpers of `neo_verified::algebra`, run on `build`, equal their copies in
`CompressionLean/Ports/Circuit/Algebra.lean`.

Each `…_eq` theorem states that the translation returns the copy's run on every input. A copy
reads a translated field value as `wordOf` or `extOf` reads it, a slice as its list, and an index
as its value. The translation of `eq_table` can fail where `Vec::push` passes `usize::MAX`
entries; its `Complete` premise is `2^n ≤ usize::MAX` for a point of `n` coordinates. Every other
helper always returns, so its premise is `True`. -/

namespace CompressionLeanRust

open Aeneas Aeneas.Std Result
open neo_verified
open CompressionLean
open CompressionLean.Circuit

/-! ## Constants and proof words -/

theorem constant_eq (s : Builder) (v : field.Gl) :
    algebra.constant build s v = ok ((Build.const (wordOf v)).run s) := by
  simp only [algebra.constant, gl_word_eq, bind_ok, build_constant, wordOf_red]

theorem private_eq (s : Builder) (v : field.Gl) : algebra.private build s v = ok (Build.priv.run s) := by
  simp only [algebra.private, gl_word_eq, bind_ok, build_private]

theorem scale_eq (s : Builder) (a : Var) (v : field.Gl) :
    algebra.scale build s a v = ok ((Build.scale a (wordOf v)).run s) := by
  simp only [algebra.scale, gl_word_eq, bind_ok, build_scale, wordOf_red]

theorem ext_constant_eq (s : Builder) (v : field.Ext) :
    algebra.ext_constant build s v = ok ((Algebra.extConstant (extOf v)).run s) := by
  have words : (fun i : Fin 3 => wordOf ([redW v.c0, redW v.c1, redW v.c2].getD i 0#u64)) =
      ![wordOf v.c0, wordOf v.c1, wordOf v.c2] := by
    funext i
    fin_cases i <;> simp [wordOf_red]
  simp only [algebra.ext_constant, gl_word_eq, bind_ok, build_ext_constant, Algebra.extConstant, extOf,
    Aeneas.Std.Array.make]
  rw [← words]
  rfl

theorem extOf_one : extOf field.EXT_ONE = 1 := by
  unfold extOf field.EXT_ONE field.GL_ONE field.GL_ZERO
  rfl

theorem ext_zero_eq (s : Builder) : algebra.ext_zero build s = ok (Algebra.extZero.run s) := by
  rw [algebra.ext_zero, ext_constant_eq, extOf_zero]
  rfl

theorem ext_one_eq (s : Builder) : algebra.ext_one build s = ok (Algebra.extOne.run s) := by
  rw [algebra.ext_one, ext_constant_eq, extOf_one]
  rfl

theorem handle_make (x y z : Var) : handle (Aeneas.Std.Array.make 3#usize [x, y, z]) = ![x, y, z] := by
  funext i
  fin_cases i <;> rfl

theorem private_ext_eq (s : Builder) (v : field.Ext) :
    algebra.private_ext build s v = ok (Algebra.privateExt.run s) := by
  simp only [algebra.private_ext, private_eq, bind_ok, uncurry_eq, build_ext, handle_make]
  rfl

theorem lift_word_eq (s : Builder) (a : Var) : algebra.lift_word build s a = ok ((Algebra.lift a).run s) := by
  simp only [algebra.lift_word, build_constant, bind_ok, uncurry_eq, build_ext, handle_make]
  rfl

theorem ext_scale_constant_eq (s : Builder) (a : ExtVar) (v : field.Gl) :
    algebra.ext_scale_constant build s a v = ok ((Algebra.extScaleConstant a (wordOf v)).run s) := by
  simp only [algebra.ext_scale_constant, constant_eq, bind_ok, uncurry_eq, build_ext_scale]
  rfl

theorem one_minus_eq (s : Builder) (x : ExtVar) : algebra.one_minus build s x = ok ((Algebra.oneMinus x).run s) := by
  simp only [algebra.one_minus, ext_one_eq, bind_ok, uncurry_eq, build_ext_sub]
  rfl

theorem eq1_eq (s : Builder) (x y : ExtVar) : algebra.eq1 build s x y = ok ((Algebra.eq1 x y).run s) := by
  simp only [algebra.eq1, ext_one_eq, build_ext_add, build_ext_sub, build_ext_mul, bind_ok, uncurry_eq]
  rfl

/-! ## Loops over two slices -/

theorem zip_length_le (x y : Slice ExtVar) : (x.val.zip y.val).length ≤ Usize.max := by
  rw [List.length_zip]
  have := x.property
  omega

theorem zip_index (x y : Slice ExtVar) (i : Usize) (h : i.val < (x.val.zip y.val).length) :
    Slice.index_usize x i = ok (x.val.zip y.val)[i.val].1 ∧ Slice.index_usize y i = ok (x.val.zip y.val)[i.val].2 := by
  rw [List.length_zip] at h
  rw [slice_index x i (by omega), slice_index y i (by omega), List.getElem_zip]
  exact ⟨rfl, rfl⟩

theorem dot_eq (s : Builder) (x y : Slice ExtVar) :
    algebra.dot build s x y = ok ((Algebra.dot x.val.toArray y.val.toArray).run s) := by
  unfold algebra.dot algebra.dot_loop
  rw [ext_zero_eq, bind_ok, uncurry_eq]
  rw [loop_forIn _ (fun s r i => (s, r, i)) (fun r s => (r, s)) (x.val.zip y.val)
    (fun p total => do let term ← Build.extMul p.1 p.2; Build.extAdd total term) (zip_length_le x y)
    ?step ?stop _ _ _ 0#usize rfl]
  case step =>
    intro s r i h
    obtain ⟨hx, hy⟩ := zip_index x y i h
    rw [List.length_zip] at h
    simp only [algebra.dot_loop.body, Slice.len, show i < Usize.ofNatCore x.val.length _ by scalar_tac,
      show i < Usize.ofNatCore y.val.length _ by scalar_tac, ite_true, hx, hy, build_ext_mul, build_ext_add,
      bind_ok, uncurry_eq, lift]
    rfl
  case stop =>
    intro s r i h
    rw [List.length_zip] at h
    by_cases hx : i.val < x.val.length
    · simp only [algebra.dot_loop.body, Slice.len, show i < Usize.ofNatCore x.val.length _ by scalar_tac,
        show ¬ i < Usize.ofNatCore y.val.length _ by scalar_tac, ite_true, ite_false]
    · simp only [algebra.dot_loop.body, Slice.len, show ¬ i < Usize.ofNatCore x.val.length _ by scalar_tac,
        ite_false]
  simp only [Algebra.dot, StateT.run_bind, List.zip_toArray, List.forIn_toArray]
  rfl

theorem eq_eval_eq (s : Builder) (x y : Slice ExtVar) :
    algebra.eq_eval build s x y = ok ((Algebra.eqEval x.val.toArray y.val.toArray).run s) := by
  unfold algebra.eq_eval algebra.eq_eval_loop
  rw [ext_one_eq, bind_ok, uncurry_eq]
  rw [loop_forIn _ (fun s r i => (s, r, i)) (fun r s => (r, s)) (x.val.zip y.val)
    (fun p total => do let factor ← Algebra.eq1 p.1 p.2; Build.extMul total factor) (zip_length_le x y)
    ?step ?stop _ _ _ 0#usize rfl]
  case step =>
    intro s r i h
    obtain ⟨hx, hy⟩ := zip_index x y i h
    rw [List.length_zip] at h
    simp only [algebra.eq_eval_loop.body, Slice.len, show i < Usize.ofNatCore x.val.length _ by scalar_tac,
      show i < Usize.ofNatCore y.val.length _ by scalar_tac, ite_true, hx, hy, eq1_eq, build_ext_mul,
      bind_ok, uncurry_eq, lift]
    rfl
  case stop =>
    intro s r i h
    rw [List.length_zip] at h
    by_cases hx : i.val < x.val.length
    · simp only [algebra.eq_eval_loop.body, Slice.len, show i < Usize.ofNatCore x.val.length _ by scalar_tac,
        show ¬ i < Usize.ofNatCore y.val.length _ by scalar_tac, ite_true, ite_false]
    · simp only [algebra.eq_eval_loop.body, Slice.len, show ¬ i < Usize.ofNatCore x.val.length _ by scalar_tac,
        ite_false]
  simp only [Algebra.eqEval, StateT.run_bind, List.zip_toArray, List.forIn_toArray]
  rfl

/-! ## `eq_at` -/

theorem index_bit_eq (index t : Usize) : algebra.index_bit index t = ok (index.val.testBit t.val) := by
  unfold algebra.index_bit
  have hn := numBits_le
  have hp := numBits_pos
  have hpow : UScalarTy.Usize.numBits < 2 ^ UScalarTy.Usize.numBits := Nat.lt_two_pow_self
  have hbits : (UScalar.cast .Usize core.num.Usize.BITS).val = UScalarTy.Usize.numBits := by
    rw [UScalar.cast_val_eq, bits_val, Nat.mod_eq_of_lt hpow]
  simp only [lift, bind_ok]
  by_cases h : t.val < UScalarTy.Usize.numBits
  · rw [ite_eq_left (by rw [UScalar.lt_equiv, hbits]; exact h)]
    have hcast : (UScalar.cast .U32 t).bv.toNat % System.Platform.numBits = t.val := by
      rw [show (UScalar.cast .U32 t).bv.toNat = (UScalar.cast .U32 t).val from rfl, UScalar.cast_val_eq]
      simp only [UScalarTy.U32_numBits_eq]
      rw [← numBits_eq, Nat.mod_eq_of_lt (by omega : t.val < 2 ^ 32), Nat.mod_eq_of_lt h]
    have hshr : (core.num.Usize.wrapping_shr index (UScalar.cast .U32 t)).val = index.val / 2 ^ t.val := by
      simp only [UScalar.val, core.num.Usize.wrapping_shr_bv_eq, hcast]
      rw [BitVec.ushiftRight_eq, BitVec.toNat_ushiftRight, Nat.shiftRight_eq_div_pow]
    simp only [core.num.Usize.is_multiple_of, UScalar.is_multiple_of, hshr, bind_ok,
      Nat.testBit_eq_decide_div_mod_eq]
    congr 1
    by_cases e : index.val / 2 ^ t.val % 2 = 0 <;> simp [e]
    omega
  · rw [ite_eq_right (by rw [UScalar.lt_equiv, hbits]; exact h)]
    congr 1
    symm
    apply Nat.testBit_lt_two_pow
    calc index.val < 2 ^ UScalarTy.Usize.numBits := index.hBounds
      _ ≤ 2 ^ t.val := Nat.pow_le_pow_right (by omega) (by omega)

theorem eq_at_eq (s : Builder) (point : Slice ExtVar) (index : Usize) :
    algebra.eq_at build s point index = ok ((Algebra.eqAt point.val.toArray index.val).run s) := by
  have hlen : point.val.zipIdx.length ≤ Usize.max := by rw [List.length_zipIdx]; exact point.property
  unfold algebra.eq_at algebra.eq_at_loop
  rw [ext_one_eq, bind_ok, uncurry_eq]
  rw [loop_forIn _ (fun s r i => (s, r, i)) (fun r s => (r, s)) point.val.zipIdx
    (fun p total => do
      let factor ← if index.val.testBit p.2 then pure p.1 else Algebra.oneMinus p.1
      Build.extMul total factor) hlen ?step ?stop _ _ _ 0#usize rfl]
  case step =>
    intro s r i h
    rw [List.length_zipIdx] at h
    have hx : point.val.zipIdx[i.val] = (point.val[i.val], i.val) := by simp
    simp only [algebra.eq_at_loop.body, Slice.len, show i < Usize.ofNatCore point.val.length _ by scalar_tac,
      ite_true, index_bit_eq, bind_ok, slice_index point i h, hx]
    by_cases c : index.val.testBit i.val
    · simp only [c, ite_true, bind_ok, uncurry_eq, build_ext_mul, lift]
      rfl
    · simp only [c, Bool.false_eq_true, ite_false, one_minus_eq, bind_ok, uncurry_eq, build_ext_mul, lift]
      rfl
  case stop =>
    intro s r i h
    rw [List.length_zipIdx] at h
    simp only [algebra.eq_at_loop.body, Slice.len, show ¬ i < Usize.ofNatCore point.val.length _ by scalar_tac,
      ite_false]
  simp only [Algebra.eqAt, StateT.run_bind, Array.zipIdx_toArray, List.forIn_toArray, map_bind, bind_pure_comp,
    apply_ite (Functor.map ForInStep.yield)]
  rfl

theorem sub_one_val (i : Usize) (h : 1 ≤ i.val) : (core.num.Usize.wrapping_sub i 1#usize).val = i.val - 1 := by
  have hle : (1#usize).bv ≤ i.bv := by
    rw [BitVec.le_def]
    exact h
  have e := congrArg BitVec.toNat (core.num.Usize.wrapping_sub_bv_eq i 1#usize)
  rw [BitVec.toNat_sub_of_le hle] at e
  exact e

theorem horner_eq (s : Builder) (coefficients : Slice ExtVar) (x : ExtVar) :
    algebra.horner build s coefficients x = ok ((Algebra.horner coefficients.val.toArray x).run s) := by
  unfold algebra.horner algebra.horner_loop
  rw [ext_zero_eq, bind_ok, uncurry_eq]
  rw [loop_forIn_rel _ (fun a s r k => a.1 = s ∧ a.2.1 = r ∧ a.2.2.val + k = coefficients.val.length)
    (fun r s => (r, s)) coefficients.val.reverse (fun c total => do
      let scaled ← Build.extMul total x
      Build.extAdd scaled c) ?step ?stop coefficients.val.reverse.length _ _ _ 0 ⟨rfl, rfl, by simp⟩ (by simp)]
  case step =>
    rintro ⟨b, total, i⟩ s r k h ⟨rfl, rfl, hk⟩
    simp only [List.length_reverse] at h hk
    have hpos : 0#usize < i := by scalar_tac
    have hsub := sub_one_val i (by omega)
    have hidx : coefficients.val.reverse[k] = coefficients.val[i.val - 1] := by
      rw [List.getElem_reverse]
      congr 1
      omega
    refine ⟨(_, _, core.num.Usize.wrapping_sub i 1#usize), ?_, rfl, rfl, by simp only [hsub]; omega⟩
    simp only [algebra.horner_loop.body, hpos, ite_true, lift, bind_ok, build_ext_mul, uncurry_eq,
      slice_index coefficients _ (by rw [hsub]; omega), hsub, hidx, build_ext_add]
    rfl
  case stop =>
    rintro ⟨b, total, i⟩ s r k ⟨rfl, rfl, hk⟩ h
    simp only [List.length_reverse] at h hk
    have : ¬ 0#usize < i := by scalar_tac
    simp only [algebra.horner_loop.body, this, ite_false]
  simp only [Algebra.horner, StateT.run_bind, List.drop_zero, List.reverse_toArray, List.forIn_toArray]
  rfl

/-! ## Loops with a field accumulator -/

theorem wordOf_one : wordOf 1#u64 = 1 := rfl

theorem wordOf_gl_one : wordOf field.GL_ONE = 1 := one_exact.wordOf

/-- A canonical word as a translated `Gl`. -/
def glOfWord (w : Replay.Word) : field.Gl := ⟨w.word.toBitVec⟩

theorem glOfWord_exact (w : Replay.Word) : Exact (glOfWord w) w := rfl

theorem glOfWord_of_exact {c : U64} {w : Replay.Word} (h : Exact c w) : c = glOfWord w := by
  have e := congrArg UInt64.toBitVec h
  cases c
  exact congrArg _ e

theorem power_from_bits_eq (s : Builder) (generator : field.Gl) (bits : Slice Var) :
    algebra.power_from_bits build s generator bits =
      ok ((Algebra.powerFromBits (wordOf generator) bits.val.toArray).run s) := by
  unfold algebra.power_from_bits algebra.power_from_bits_loop
  rw [build_constant, bind_ok, uncurry_eq]
  rw [loop_forIn_rel _ (fun a s r k => a.1 = s ∧ a.2.1 = r.1 ∧ wordOf a.2.2.1 = r.2 ∧ a.2.2.2.val = k)
    (fun r s => (r.1, s)) bits.val (fun bit r => do
      let one ← Build.const 1
      let factorMinus ← Build.scale bit (r.2 - 1)
      let factor ← Build.add one factorMinus
      let total ← Build.mul r.1 factor
      pure (total, r.2 * r.2)) ?step ?stop bits.val.length _ _ (_, wordOf generator) 0 ⟨rfl, rfl, rfl, rfl⟩
    (by simp)]
  case step =>
    rintro ⟨b, total, base, i⟩ s r k h ⟨rfl, rfl, hbase, rfl⟩
    simp only at h hbase
    have hmax : i.val < Usize.max := by have := bits.property; omega
    refine ⟨(_, _, glMul base base, core.num.Usize.wrapping_add i 1#usize), ?_, rfl, rfl, ?_, add_one_val i hmax⟩
    · simp only [algebra.power_from_bits_loop.body, Slice.len,
        show i < Usize.ofNatCore bits.val.length _ by scalar_tac, ite_true, build_constant, bind_ok, uncurry_eq,
        slice_index bits i h, gl_sub_eq, scale_eq, build_add, build_mul, gl_mul_eq, lift, wordOf_one,
        (glSub_exact base field.GL_ONE).wordOf, wordOf_gl_one, hbase]
      rfl
    · simp only [(glMul_exact base base).wordOf, hbase]
      rfl
  case stop =>
    rintro ⟨b, total, base, i⟩ s r k ⟨rfl, rfl, -, rfl⟩ h
    simp only at h
    simp only [algebra.power_from_bits_loop.body, Slice.len,
      show ¬ i < Usize.ofNatCore bits.val.length _ by scalar_tac, ite_false]
  simp only [Algebra.powerFromBits, StateT.run_bind, List.drop_zero, List.forIn_toArray, map_bind, map_pure,
    wordOf_one]
  rfl

/-! ## `interpolate` -/

/-- One step of the copy's inner loop of `interpolate` at node `j`, for the value at `i`. -/
def interpolateStep (x : ExtVar) (i j : ℕ) (r : ExtVar × Replay.Word) : Build (ExtVar × Replay.Word) :=
  if i ≠ j then do
    let node ← Algebra.extConstant (Replay.Ext.ofGl (Replay.Word.ofNat j))
    let factor ← Build.extSub x node
    let numerator ← Build.extMul r.1 factor
    pure (numerator, r.2 * (Replay.Word.ofNat i - Replay.Word.ofNat j))
  else pure r

theorem wordOf_cast (j : Usize) : wordOf (UScalar.cast .U64 j) = Replay.Word.ofNat j.val := by
  unfold wordOf
  congr 1
  rw [UScalar.cast_val_eq]
  have := j.hBounds
  have := numBits_le
  simp only [UScalarTy.U64_numBits_eq]
  apply Nat.mod_eq_of_lt
  calc j.val < 2 ^ UScalarTy.Usize.numBits := j.hBounds
    _ ≤ 2 ^ 64 := Nat.pow_le_pow_right (by omega) numBits_le

theorem extOf_node (j : Usize) :
    extOf ⟨redW (redW (UScalar.cast .U64 j)), field.GL_ZERO, field.GL_ZERO⟩ =
      Replay.Ext.ofGl (Replay.Word.ofNat j.val) := by
  unfold extOf Replay.Ext.ofGl
  rw [wordOf_red, wordOf_red, wordOf_cast, wordOf_zero]

theorem interpolate_inner_eq (s : Builder) (x : ExtVar) (n i : Usize) (numerator : ExtVar) (denominator : Replay.Word) :
    algebra.interpolate_loop0_loop0 build s x n i numerator (glOfWord denominator) 0#usize =
      ok (let p := (forIn (List.range n.val) (numerator, denominator)
          fun j r => ForInStep.yield <$> interpolateStep x i.val j r).run s; (p.2, p.1.1, glOfWord p.1.2)) := by
  unfold algebra.interpolate_loop0_loop0
  rw [loop_forIn_rel _ (fun a s r k => a.1 = s ∧ a.2.1 = r.1 ∧ a.2.2.1 = glOfWord r.2 ∧ a.2.2.2.val = k)
    (fun r s => (s, r.1, glOfWord r.2)) (List.range n.val) (interpolateStep x i.val) ?step ?stop n.val _ s (numerator, denominator) 0
    ⟨rfl, rfl, rfl, rfl⟩ (by simp)]
  · simp only [List.drop_zero]
  case step =>
    rintro ⟨b, num, den, j⟩ s r k h ⟨rfl, rfl, hden, rfl⟩
    simp only [List.length_range] at h hden ⊢
    have hmax : j.val < Usize.max := by have := n.hBounds; scalar_tac
    have hj : j < n := by scalar_tac
    simp only [List.getElem_range]
    by_cases e : i.val = j.val
    · have hij : (i != j) = false := by
        simp only [bne_eq_false_iff_eq]
        exact UScalar.val_eq_imp i j e
      refine ⟨(b, r.1, den, core.num.Usize.wrapping_add j 1#usize), ?_, ?_⟩
      · simp only [algebra.interpolate_loop0_loop0.body, hj, ite_true, hij, Bool.false_eq_true, ite_false,
          bind_ok, uncurry_eq, lift]
      · simp only [interpolateStep, e, ne_eq, not_true_eq_false, ite_false, StateT.run_pure, hden]
        exact ⟨rfl, rfl, rfl, add_one_val j hmax⟩
    · have hij : (i != j) = true := by
        simp only [bne_iff_ne, ne_eq]
        intro h'
        exact e (congrArg UScalar.val h')
      obtain ⟨⟨node, s1⟩, h1⟩ : ∃ p, (Algebra.extConstant (Replay.Ext.ofGl (Replay.Word.ofNat j.val))).run b = p :=
        ⟨_, rfl⟩
      obtain ⟨⟨factor, s2⟩, h2⟩ : ∃ p, (Build.extSub x node).run s1 = p := ⟨_, rfl⟩
      obtain ⟨⟨num, s3⟩, h3⟩ : ∃ p, (Build.extMul r.1 factor).run s2 = p := ⟨_, rfl⟩
      refine ⟨(s3, num, glMul den (glSub (redW (UScalar.cast .U64 i)) (redW (UScalar.cast .U64 j))),
        core.num.Usize.wrapping_add j 1#usize), ?_, ?_⟩
      · simp only [algebra.interpolate_loop0_loop0.body, hj, ite_true, hij, lift, bind_ok, reduce_eq,
          ext_of_gl_eq, ext_constant_eq, extOf_node, uncurry_eq, build_ext_sub, build_ext_mul, gl_sub_eq,
          gl_mul_eq, h1, h2, h3]
      · simp only [interpolateStep, ne_eq, e, not_false_eq_true, ite_true, StateT.run_bind, h1, id_bind, h2, h3,
          StateT.run_pure]
        refine ⟨rfl, rfl, ?_, add_one_val j hmax⟩
        subst hden
        apply glOfWord_of_exact
        have hm := glMul_exact (glOfWord r.2) (glSub (redW (UScalar.cast .U64 i)) (redW (UScalar.cast .U64 j)))
        rw [(glSub_exact _ _).wordOf, wordOf_red, wordOf_red, wordOf_cast, wordOf_cast,
          (glOfWord_exact _).wordOf] at hm
        exact hm
  case stop =>
    rintro ⟨b, num, den, j⟩ s r k ⟨rfl, rfl, hden, rfl⟩ h
    simp only [List.length_range] at h hden
    have hj : ¬ j < n := by scalar_tac
    simp only [algebra.interpolate_loop0_loop0.body, hj, ite_false, hden]

theorem gl_one_word : field.GL_ONE = glOfWord 1 := glOfWord_of_exact one_exact

theorem interpolate_eq (s : Builder) (values : Slice ExtVar) (x : ExtVar) :
    algebra.interpolate build s values x = ok ((Algebra.interpolate values.val.toArray x).run s) := by
  have hlen : values.val.zipIdx.length ≤ Usize.max := by rw [List.length_zipIdx]; exact values.property
  have hn : (Slice.len values).val = values.val.length := Slice.len_val values
  unfold algebra.interpolate algebra.interpolate_loop0
  rw [ext_zero_eq, bind_ok, uncurry_eq]
  rw [loop_forIn _ (fun s r i => (s, r, i)) (fun r s => (r, s)) values.val.zipIdx
    (fun p total => do
      let numerator ← Algebra.extOne
      let q ← forIn (List.range values.val.length) (numerator, (1 : Replay.Word))
        fun j r => ForInStep.yield <$> interpolateStep x p.2 j r
      let weight ← Algebra.extScaleConstant q.1 q.2.inv
      let term ← Build.extMul p.1 weight
      Build.extAdd total term) hlen ?step ?stop _ _ _ 0#usize rfl]
  case step =>
    intro s r i h
    rw [List.length_zipIdx] at h
    have hx : values.val.zipIdx[i.val] = (values.val[i.val], i.val) := by simp
    obtain ⟨⟨one, s1⟩, h1⟩ : ∃ p, Algebra.extOne.run s = p := ⟨_, rfl⟩
    obtain ⟨⟨⟨num, den⟩, s2⟩, h2⟩ : ∃ p, (forIn (List.range values.val.length) (one, (1 : Replay.Word))
        fun j r => ForInStep.yield <$> interpolateStep x i.val j r).run s1 = p := ⟨_, rfl⟩
    obtain ⟨c, hc, hexact⟩ := gl_inv_exact (glOfWord den)
    rw [(glOfWord_exact den).wordOf] at hexact
    have inner := interpolate_inner_eq s1 x (Slice.len values) i one 1
    rw [hn, h2, ← gl_one_word] at inner
    simp only [algebra.interpolate_loop0.body, show i < Slice.len values by scalar_tac, ite_true, ext_one_eq,
      h1, bind_ok, uncurry_eq, inner, hc, ext_scale_constant_eq, slice_index values i h, build_ext_mul,
      build_ext_add, lift, hx, hexact.wordOf]
    simp only [StateT.run_bind, h1, id_bind, h2]
  case stop =>
    intro s r i h
    rw [List.length_zipIdx] at h
    simp only [algebra.interpolate_loop0.body, show ¬ i < Slice.len values by scalar_tac, ite_false]
  simp only [Algebra.interpolate, StateT.run_bind, Array.zipIdx_toArray, List.forIn_toArray, List.size_toArray,
    Std.Legacy.Range.forIn_eq_forIn_range', map_bind, map_pure, bind_pure_comp, apply_ite (Functor.map ForInStep.yield),
    interpolateStep, Functor.map_map, Std.Legacy.Range.size, ← List.range_eq_range', Nat.sub_zero,
    Nat.add_sub_cancel, Nat.div_one]
  rfl

/-! ## `K ⊗ Ext` -/

/-- A translated `Kx` as the copies' `KxVar`. -/
def kxOf (k : algebra.Kx Builder Var ExtVar) : KxVar := ⟨k.re, k.im⟩

/-- A translated `[F; 2]` as the copies' `KVar`. -/
def kOf (k : Aeneas.Std.Array Var 2#usize) : KVar := (k.val[0]!, k.val[1]!)

theorem kx_zero_eq (s : Builder) :
    algebra.Kx.zero build s = ok (let p := KxVar.zero.run s; (⟨p.1.re, p.1.im⟩, p.2)) := by
  simp only [algebra.Kx.zero, ext_zero_eq, bind_ok, uncurry_eq]
  rfl

theorem kx_one_eq (s : Builder) :
    algebra.Kx.one build s = ok (let p := KxVar.one.run s; (⟨p.1.re, p.1.im⟩, p.2)) := by
  simp only [algebra.Kx.one, ext_one_eq, ext_zero_eq, bind_ok, uncurry_eq]
  rfl

theorem kx_from_k_eq (s : Builder) (k : Aeneas.Std.Array Var 2#usize) :
    algebra.Kx.from_k build s k = ok (let p := (KxVar.fromK (kOf k)).run s; (⟨p.1.re, p.1.im⟩, p.2)) := by
  simp only [algebra.Kx.from_k, index_eq k 0#usize (by decide), index_eq k 1#usize (by decide), lift_word_eq,
    bind_ok, uncurry_eq]
  rfl

theorem kx_add_eq (s : Builder) (a c : algebra.Kx Builder Var ExtVar) :
    algebra.Kx.add build a s c = ok (let p := (KxVar.add (kxOf a) (kxOf c)).run s; (⟨p.1.re, p.1.im⟩, p.2)) := by
  simp only [algebra.Kx.add, build_ext_add, bind_ok, uncurry_eq]
  rfl

theorem kx_mul_eq (s : Builder) (a c : algebra.Kx Builder Var ExtVar) :
    algebra.Kx.mul build a s c = ok (let p := (KxVar.mul (kxOf a) (kxOf c)).run s; (⟨p.1.re, p.1.im⟩, p.2)) := by
  simp only [algebra.Kx.mul, build_ext_mul, build_ext_add, reduce_eq, ext_scale_constant_eq, wordOf_red,
    bind_ok, uncurry_eq]
  rfl

theorem kx_scale_eq (s : Builder) (a : algebra.Kx Builder Var ExtVar) (by_ : ExtVar) :
    algebra.Kx.scale build a s by_ = ok (let p := (KxVar.scale (kxOf a) by_).run s; (⟨p.1.re, p.1.im⟩, p.2)) := by
  simp only [algebra.Kx.scale, build_ext_mul, bind_ok, uncurry_eq]
  rfl

theorem kx_one_minus_eq (s : Builder) (a : algebra.Kx Builder Var ExtVar) :
    algebra.Kx.one_minus build a s = ok (let p := (KxVar.oneMinus (kxOf a)).run s; (⟨p.1.re, p.1.im⟩, p.2)) := by
  simp only [algebra.Kx.one_minus, one_minus_eq, ext_zero_eq, build_ext_sub, bind_ok, uncurry_eq]
  rfl

/-! ## `eq_table` -/

theorem vec_val_injective {α : Type} : Function.Injective (alloc.vec.Vec.val (α := α)) := by
  rintro ⟨⟨n, l, hb⟩⟩ ⟨⟨m, k, hc⟩⟩ h
  simp only [alloc.vec.Vec.val, Slice.val] at h
  have hn : n = m := by
    rw [← Aeneas.Data.ListN.ListN_length l, ← Aeneas.Data.ListN.ListN_length k]
    exact congrArg List.length h
  subst hn
  rw [Aeneas.Data.ListN.ListN.toList_inj l k h]

/-- An array as a translated `Vec`; empty when it is too long for one. -/
def vecOfArray (r : _root_.Array ExtVar) : alloc.vec.Vec ExtVar :=
  if h : r.toList.length ≤ Usize.max then alloc.vec.Vec.from r.toList h else alloc.vec.Vec.new ExtVar

theorem vecOfArray_val (v : alloc.vec.Vec ExtVar) : vecOfArray v.val.toArray = v := by
  apply vec_val_injective
  unfold vecOfArray
  rw [dite_eq_left (by simp)]
  simp

theorem vec_index {α : Type} (v : alloc.vec.Vec α) (i : Usize) (h : i.val < v.val.length) :
    alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice α) v i = ok v.val[i.val] :=
  slice_index v.slice i h

theorem vec_push_val {α : Type} {v w : alloc.vec.Vec α} {x : α} (h : alloc.vec.Vec.push v x = ok w) :
    w.val = v.val ++ [x] := by
  unfold alloc.vec.Vec.push at h
  dsimp only at h
  split at h
  · rw [Result.ok.injEq] at h
    subst h
    simp
  · exact absurd h fail_not_ok

theorem vec_push_ok {α : Type} (v : alloc.vec.Vec α) (x : α) (h : v.val.length < Usize.max) :
    ∃ w, alloc.vec.Vec.push v x = ok w ∧ w.val = v.val ++ [x] := by
  unfold alloc.vec.Vec.push
  dsimp only
  rw [dite_eq_left (by simp; omega)]
  exact ⟨_, rfl, by simp⟩

/-- One step of the copy's inner loops of `eqTable`: append `value·by`. -/
def pushProduct (by_ : ExtVar) (value : ExtVar) (next : _root_.Array ExtVar) : Build (_root_.Array ExtVar) := do
  let product ← Build.extMul value by_
  pure (next.push product)

/-- The state of an inner loop: the builder, the table so far and the index. -/
def InnerState (a : Builder × alloc.vec.Vec ExtVar × Usize) (s : Builder) (r : _root_.Array ExtVar) (k : ℕ) :
    Prop :=
  a.1 = s ∧ a.2.1.val.toArray = r ∧ a.2.2.val = k

theorem inner0_sound (table : alloc.vec.Vec ExtVar) (by_ : ExtVar) :
    ∀ s next (y : Builder × alloc.vec.Vec ExtVar),
      algebra.eq_table_loop0_loop0 build s table by_ next 0#usize = ok y →
      (let p := (forIn table.val next.val.toArray fun x r => ForInStep.yield <$> pushProduct by_ x r).run s;
        y.1 = p.2 ∧ y.2.val.toArray = p.1) := by
  intro s next y h
  unfold algebra.eq_table_loop0_loop0 at h
  have := loop_forIn_rel_sound _ InnerState (fun y r s => y.1 = s ∧ y.2.val.toArray = r) table.val
    (pushProduct by_) ?step ?stop table.val.length (s, next, 0#usize) s next.val.toArray 0 y ⟨rfl, rfl, rfl⟩
    (by simp) h
  · simpa using this
  case step =>
    rintro ⟨b, acc, i⟩ s r k c h ⟨rfl, rfl, rfl⟩ hc
    dsimp only at h
    have hi : i < alloc.vec.Vec.len table := by scalar_tac
    simp only [algebra.eq_table_loop0_loop0.body, hi, ite_true, vec_index table i h, build_ext_mul, bind_ok,
      uncurry_eq] at hc
    obtain ⟨w, hw, hc⟩ := bind_eq_ok hc
    simp only [lift, bind_ok, Result.ok.injEq] at hc
    subst hc
    have hmax : i.val < Usize.max := by have : table.val.length ≤ Usize.max := Slice.property table.slice; omega
    refine ⟨_, rfl, rfl, ?_, add_one_val i hmax⟩
    rw [vec_push_val hw]
    simp only [pushProduct, StateT.run_bind, StateT.run_pure, id_bind, List.push_toArray]
    rfl
  case stop =>
    rintro ⟨b, acc, i⟩ s r k c ⟨rfl, rfl, rfl⟩ h hc
    dsimp only at h
    have hi : ¬ i < alloc.vec.Vec.len table := by scalar_tac
    simp only [algebra.eq_table_loop0_loop0.body, hi, ite_false, Result.ok.injEq] at hc
    exact ⟨_, hc.symm, rfl, rfl⟩

theorem inner0_eq (table : alloc.vec.Vec ExtVar) (by_ : ExtVar) (s : Builder) (next : alloc.vec.Vec ExtVar)
    (fits : next.val.length + table.val.length ≤ Usize.max) :
    algebra.eq_table_loop0_loop0 build s table by_ next 0#usize =
      ok (let p := (forIn table.val next.val.toArray fun x r => ForInStep.yield <$> pushProduct by_ x r).run s;
        (p.2, vecOfArray p.1)) := by
  unfold algebra.eq_table_loop0_loop0
  rw [loop_forIn_rel _ (fun a s r k => InnerState a s r k ∧ r.size + (table.val.length - k) ≤ Usize.max)
    (fun r s => (s, vecOfArray r)) table.val (pushProduct by_) ?step ?stop
    table.val.length (s, next, 0#usize) s next.val.toArray 0 ⟨⟨rfl, rfl, rfl⟩, by simpa using fits⟩ (by simp)]
  · simp
  case step =>
    rintro ⟨b, acc, i⟩ s r k h ⟨⟨rfl, rfl, rfl⟩, hfit⟩
    dsimp only at h hfit
    have hi : i < alloc.vec.Vec.len table := by scalar_tac
    simp only [List.size_toArray] at hfit
    have hmax : i.val < Usize.max := by have : table.val.length ≤ Usize.max := Slice.property table.slice; omega
    obtain ⟨⟨prod, b1⟩, hp⟩ : ∃ p, (Build.extMul (table.val[i.val]'h) by_).run b = p := ⟨_, rfl⟩
    obtain ⟨w, hw, hval⟩ := vec_push_ok acc prod (by omega)
    refine ⟨(b1, w, core.num.Usize.wrapping_add i 1#usize), ?_, ⟨?_, ?_, add_one_val i hmax⟩, ?_⟩
    · simp only [algebra.eq_table_loop0_loop0.body, hi, ite_true, vec_index table i h, build_ext_mul, bind_ok,
        uncurry_eq, lift, hp, hw]
    · simp only [pushProduct, StateT.run_bind, hp, id_bind, StateT.run_pure]
      rfl
    · rw [hval]
      simp only [pushProduct, StateT.run_bind, hp, id_bind, StateT.run_pure, List.push_toArray]
      rfl
    · simp only [pushProduct, StateT.run_bind, hp, id_bind, StateT.run_pure]
      show (acc.val.toArray.push prod).size + _ ≤ _
      rw [Array.size_push, List.size_toArray]
      omega
  case stop =>
    rintro ⟨b, acc, i⟩ s r k ⟨⟨rfl, rfl, rfl⟩, -⟩ h
    dsimp only at h
    have hi : ¬ i < alloc.vec.Vec.len table := by scalar_tac
    simp only [algebra.eq_table_loop0_loop0.body, hi, ite_false, vecOfArray_val]

theorem inner1_sound (table : alloc.vec.Vec ExtVar) (by_ : ExtVar) :
    ∀ s next (y : Builder × alloc.vec.Vec ExtVar),
      algebra.eq_table_loop0_loop1 build s table by_ next 0#usize = ok y →
      (let p := (forIn table.val next.val.toArray fun x r => ForInStep.yield <$> pushProduct by_ x r).run s;
        y.1 = p.2 ∧ y.2.val.toArray = p.1) := by
  intro s next y h
  unfold algebra.eq_table_loop0_loop1 at h
  have := loop_forIn_rel_sound _ InnerState (fun y r s => y.1 = s ∧ y.2.val.toArray = r) table.val
    (pushProduct by_) ?step ?stop table.val.length (s, next, 0#usize) s next.val.toArray 0 y ⟨rfl, rfl, rfl⟩
    (by simp) h
  · simpa using this
  case step =>
    rintro ⟨b, acc, i⟩ s r k c h ⟨rfl, rfl, rfl⟩ hc
    dsimp only at h
    have hi : i < alloc.vec.Vec.len table := by scalar_tac
    simp only [algebra.eq_table_loop0_loop1.body, hi, ite_true, vec_index table i h, build_ext_mul, bind_ok,
      uncurry_eq] at hc
    obtain ⟨w, hw, hc⟩ := bind_eq_ok hc
    simp only [lift, bind_ok, Result.ok.injEq] at hc
    subst hc
    have hmax : i.val < Usize.max := by have : table.val.length ≤ Usize.max := Slice.property table.slice; omega
    refine ⟨_, rfl, rfl, ?_, add_one_val i hmax⟩
    rw [vec_push_val hw]
    simp only [pushProduct, StateT.run_bind, StateT.run_pure, id_bind, List.push_toArray]
    rfl
  case stop =>
    rintro ⟨b, acc, i⟩ s r k c ⟨rfl, rfl, rfl⟩ h hc
    dsimp only at h
    have hi : ¬ i < alloc.vec.Vec.len table := by scalar_tac
    simp only [algebra.eq_table_loop0_loop1.body, hi, ite_false, Result.ok.injEq] at hc
    exact ⟨_, hc.symm, rfl, rfl⟩

theorem inner1_eq (table : alloc.vec.Vec ExtVar) (by_ : ExtVar) (s : Builder) (next : alloc.vec.Vec ExtVar)
    (fits : next.val.length + table.val.length ≤ Usize.max) :
    algebra.eq_table_loop0_loop1 build s table by_ next 0#usize =
      ok (let p := (forIn table.val next.val.toArray fun x r => ForInStep.yield <$> pushProduct by_ x r).run s;
        (p.2, vecOfArray p.1)) := by
  unfold algebra.eq_table_loop0_loop1
  rw [loop_forIn_rel _ (fun a s r k => InnerState a s r k ∧ r.size + (table.val.length - k) ≤ Usize.max)
    (fun r s => (s, vecOfArray r)) table.val (pushProduct by_) ?step ?stop
    table.val.length (s, next, 0#usize) s next.val.toArray 0 ⟨⟨rfl, rfl, rfl⟩, by simpa using fits⟩ (by simp)]
  · simp
  case step =>
    rintro ⟨b, acc, i⟩ s r k h ⟨⟨rfl, rfl, rfl⟩, hfit⟩
    dsimp only at h hfit
    have hi : i < alloc.vec.Vec.len table := by scalar_tac
    simp only [List.size_toArray] at hfit
    have hmax : i.val < Usize.max := by have : table.val.length ≤ Usize.max := Slice.property table.slice; omega
    obtain ⟨⟨prod, b1⟩, hp⟩ : ∃ p, (Build.extMul (table.val[i.val]'h) by_).run b = p := ⟨_, rfl⟩
    obtain ⟨w, hw, hval⟩ := vec_push_ok acc prod (by omega)
    refine ⟨(b1, w, core.num.Usize.wrapping_add i 1#usize), ?_, ⟨?_, ?_, add_one_val i hmax⟩, ?_⟩
    · simp only [algebra.eq_table_loop0_loop1.body, hi, ite_true, vec_index table i h, build_ext_mul, bind_ok,
        uncurry_eq, lift, hp, hw]
    · simp only [pushProduct, StateT.run_bind, hp, id_bind, StateT.run_pure]
      rfl
    · rw [hval]
      simp only [pushProduct, StateT.run_bind, hp, id_bind, StateT.run_pure, List.push_toArray]
      rfl
    · simp only [pushProduct, StateT.run_bind, hp, id_bind, StateT.run_pure]
      show (acc.val.toArray.push prod).size + _ ≤ _
      rw [Array.size_push, List.size_toArray]
      omega
  case stop =>
    rintro ⟨b, acc, i⟩ s r k ⟨⟨rfl, rfl, rfl⟩, -⟩ h
    dsimp only at h
    have hi : ¬ i < alloc.vec.Vec.len table := by scalar_tac
    simp only [algebra.eq_table_loop0_loop1.body, hi, ite_false, vecOfArray_val]

/-- One step of the copy's outer loop of `eqTable`. -/
def tableStep (x : ExtVar) (table : _root_.Array ExtVar) : Build (_root_.Array ExtVar) := do
  let low ← Algebra.oneMinus x
  let next ← forIn table.toList #[] fun v n => ForInStep.yield <$> pushProduct low v n
  forIn table.toList next fun v n => ForInStep.yield <$> pushProduct x v n

theorem push_products_size (by_ : ExtVar) : ∀ (l : List ExtVar) (acc : _root_.Array ExtVar) (s : Builder),
    ((forIn l acc fun v n => ForInStep.yield <$> pushProduct by_ v n).run s).1.size = acc.size + l.length := by
  intro l
  induction l with
  | nil => intro acc s; rfl
  | cons v l ih =>
    intro acc s
    rw [List.forIn_cons]
    exact (ih (acc.push ((Build.extMul v by_).run s).1) ((Build.extMul v by_).run s).2).trans
      (by rw [Array.size_push, List.length_cons]; omega)

theorem vecOfArray_toArray (r : _root_.Array ExtVar) (h : r.size ≤ Usize.max) : (vecOfArray r).val.toArray = r := by
  unfold vecOfArray
  rw [dite_eq_left (by simpa using h)]
  simp

theorem vec_new_val : (alloc.vec.Vec.new ExtVar).val.toArray = #[] := rfl

theorem eq_table_loop_sound (point : Slice ExtVar) : ∀ s table (y : alloc.vec.Vec ExtVar × Builder),
    algebra.eq_table_loop0 build s point table 0#usize = ok y →
      (let p := (forIn point.val table.val.toArray fun x r => ForInStep.yield <$> tableStep x r).run s;
        y.2 = p.2 ∧ y.1.val.toArray = p.1) := by
  intro s table y h
  unfold algebra.eq_table_loop0 at h
  have := loop_forIn_rel_sound _ InnerState (fun y r s => y.2 = s ∧ y.1.val.toArray = r) point.val tableStep
    ?step ?stop point.val.length (s, table, 0#usize) s table.val.toArray 0 y ⟨rfl, rfl, rfl⟩ (by simp) h
  · simpa using this
  case step =>
    rintro ⟨b, tab, t⟩ s r k c h ⟨rfl, rfl, rfl⟩ hc
    dsimp only at h
    have ht : t < Slice.len point := by scalar_tac
    simp only [algebra.eq_table_loop0.body, ht, ite_true, slice_index point t h, one_minus_eq, bind_ok,
      uncurry_eq] at hc
    obtain ⟨⟨b2, next⟩, h0, hc⟩ := bind_eq_ok hc
    simp only [uncurry_eq] at hc
    obtain ⟨⟨b3, next1⟩, h1, hc⟩ := bind_eq_ok hc
    simp only [uncurry_eq, lift, bind_ok, Result.ok.injEq] at hc
    subst hc
    obtain ⟨e2, n2⟩ := inner0_sound tab _ _ _ _ h0
    obtain ⟨e3, n3⟩ := inner1_sound tab _ _ _ _ h1
    have hmax : t.val < Usize.max := by have := point.property; omega
    refine ⟨_, rfl, ?_, ?_, add_one_val t hmax⟩
    · simp only [tableStep, StateT.run_bind, id_bind] at e2 e3 n2 n3 ⊢
      rw [e3, n2, e2, vec_new_val]
    · simp only [tableStep, StateT.run_bind, id_bind] at e2 e3 n2 n3 ⊢
      rw [n3, n2, e2, vec_new_val]
  case stop =>
    rintro ⟨b, tab, t⟩ s r k c ⟨rfl, rfl, rfl⟩ h hc
    dsimp only at h
    have ht : ¬ t < Slice.len point := by scalar_tac
    simp only [algebra.eq_table_loop0.body, ht, ite_false, Result.ok.injEq] at hc
    exact ⟨_, hc.symm, rfl, rfl⟩

theorem eq_table_loop_succeeds (point : Slice ExtVar) (fits : 2 ^ point.val.length ≤ Usize.max) :
    ∀ s (table : alloc.vec.Vec ExtVar), table.val.length = 1 →
      ∃ y, algebra.eq_table_loop0 build s point table 0#usize = ok y := by
  intro s table h1
  unfold algebra.eq_table_loop0
  rw [loop_forIn_rel _ (fun a s r k => InnerState a s r k ∧ r.size = 2 ^ k ∧ k ≤ point.val.length)
    (fun r s => (vecOfArray r, s)) point.val tableStep ?step ?stop point.val.length (s, table, 0#usize) s
    table.val.toArray 0 ⟨⟨rfl, rfl, rfl⟩, by simp [h1], by omega⟩ (by simp)]
  · exact ⟨_, rfl⟩
  case step =>
    rintro ⟨b, tab, t⟩ s r k h ⟨⟨rfl, rfl, rfl⟩, hsize, -⟩
    dsimp only at h hsize
    simp only [List.size_toArray] at hsize
    have ht : t < Slice.len point := by scalar_tac
    have hpow : 2 ^ (t.val + 1) ≤ Usize.max :=
      le_trans (Nat.pow_le_pow_right (by omega) (by omega)) fits
    have hmax : t.val < Usize.max := by have := point.property; omega
    have e0 := inner0_eq tab ((Algebra.oneMinus (point.val[t.val]'h)).run b).1 ((Algebra.oneMinus (point.val[t.val]'h)).run b).2
      (alloc.vec.Vec.new ExtVar) (by simp only [hsize]; rw [pow_succ] at hpow; simp; omega)
    obtain ⟨⟨arr, b2⟩, hp⟩ : ∃ p, (forIn tab.val (alloc.vec.Vec.new ExtVar).val.toArray
        fun x r => ForInStep.yield <$> pushProduct ((Algebra.oneMinus (point.val[t.val]'h)).run b).1 x r).run
        ((Algebra.oneMinus (point.val[t.val]'h)).run b).2 = p := ⟨_, rfl⟩
    have arr_size : arr.size = 2 ^ t.val := by
      have := push_products_size ((Algebra.oneMinus (point.val[t.val]'h)).run b).1 tab.val
        (alloc.vec.Vec.new ExtVar).val.toArray ((Algebra.oneMinus (point.val[t.val]'h)).run b).2
      rw [hp] at this
      simpa [hsize] using this
    rw [hp] at e0
    dsimp only at e0
    have hv : (vecOfArray arr).val.length = arr.size := by
      have := congrArg _root_.Array.size (vecOfArray_toArray arr (by rw [arr_size, pow_succ] at *; omega))
      simpa using this
    have e1 := inner1_eq tab (point.val[t.val]'h) b2 (vecOfArray arr)
      (by rw [hv, arr_size, hsize]; rw [pow_succ] at hpow; omega)
    rw [vecOfArray_toArray arr (by rw [arr_size]; rw [pow_succ] at hpow; omega)] at e1
    obtain ⟨⟨arr2, b3⟩, hq⟩ : ∃ p, (forIn tab.val arr
        fun x r => ForInStep.yield <$> pushProduct (point.val[t.val]'h) x r).run b2 = p := ⟨_, rfl⟩
    have arr2_size : arr2.size = 2 ^ (t.val + 1) := by
      have := push_products_size (point.val[t.val]'h) tab.val arr b2
      rw [hq, arr_size, hsize] at this
      rw [this, pow_succ]
      omega
    rw [hq] at e1
    dsimp only at e1
    refine ⟨(b3, vecOfArray arr2, core.num.Usize.wrapping_add t 1#usize), ?_, ⟨⟨?_, ?_, add_one_val t hmax⟩, ?_, ?_⟩⟩
    · simp only [algebra.eq_table_loop0.body, ht, ite_true, slice_index point t h, one_minus_eq, bind_ok,
        uncurry_eq, lift]
      rw [e0, bind_ok, uncurry_eq]
      dsimp only
      rw [e1, bind_ok, uncurry_eq]
    · simp only [tableStep, StateT.run_bind, id_bind]
      rw [show (#[] : _root_.Array ExtVar) = (alloc.vec.Vec.new ExtVar).val.toArray from rfl, hp]
      simp only [hq]
    · rw [vecOfArray_toArray arr2 (by omega)]
      simp only [tableStep, StateT.run_bind, id_bind]
      rw [show (#[] : _root_.Array ExtVar) = (alloc.vec.Vec.new ExtVar).val.toArray from rfl, hp]
      simp only [hq]
    · rw [← arr2_size]
      simp only [tableStep, StateT.run_bind, id_bind]
      rw [show (#[] : _root_.Array ExtVar) = (alloc.vec.Vec.new ExtVar).val.toArray from rfl, hp]
      simp only [hq]
    · dsimp only
      omega
  case stop =>
    rintro ⟨b, tab, t⟩ s r k ⟨⟨rfl, rfl, rfl⟩, -⟩ h
    dsimp only at h
    have ht : ¬ t < Slice.len point := by scalar_tac
    simp only [algebra.eq_table_loop0.body, ht, ite_false, vecOfArray_val]

/-! ## `Sound` and `Complete` -/

section

def zeroExt : ExtVar := fun _ => 0

theorem ext_constant_sound : Sound (fun x : Builder × field.Ext => algebra.ext_constant build x.1 x.2) built
    (fun x => (Algebra.extConstant (extOf x.2)).run x.1) :=
  sound_eq _ (fun x => ext_constant_eq x.1 x.2) fun _ => rfl
theorem ext_constant_complete : Complete (fun _ => True) (fun x : Builder × field.Ext => algebra.ext_constant build x.1 x.2)
    built (fun x => (Algebra.extConstant (extOf x.2)).run x.1) :=
  complete_eq _ (fun x => ext_constant_eq x.1 x.2) (fun _ => rfl) ({}, field.EXT_ZERO)

theorem ext_zero_sound : Sound (algebra.ext_zero build) built (fun s => Algebra.extZero.run s) :=
  sound_eq _ ext_zero_eq fun _ => rfl
theorem ext_zero_complete : Complete (fun _ => True) (algebra.ext_zero build) built (fun s => Algebra.extZero.run s) :=
  complete_eq _ ext_zero_eq (fun _ => rfl) {}

theorem ext_one_sound : Sound (algebra.ext_one build) built (fun s => Algebra.extOne.run s) :=
  sound_eq _ ext_one_eq fun _ => rfl
theorem ext_one_complete : Complete (fun _ => True) (algebra.ext_one build) built (fun s => Algebra.extOne.run s) :=
  complete_eq _ ext_one_eq (fun _ => rfl) {}

theorem private_ext_sound : Sound (fun x : Builder × field.Ext => algebra.private_ext build x.1 x.2) built
    (fun x => Algebra.privateExt.run x.1) :=
  sound_eq _ (fun x => private_ext_eq x.1 x.2) fun _ => rfl
theorem private_ext_complete : Complete (fun _ => True) (fun x : Builder × field.Ext => algebra.private_ext build x.1 x.2)
    built (fun x => Algebra.privateExt.run x.1) :=
  complete_eq _ (fun x => private_ext_eq x.1 x.2) (fun _ => rfl) ({}, field.EXT_ZERO)

theorem lift_word_sound : Sound (fun x : Builder × Var => algebra.lift_word build x.1 x.2) built
    (fun x => (Algebra.lift x.2).run x.1) :=
  sound_eq _ (fun x => lift_word_eq x.1 x.2) fun _ => rfl
theorem lift_word_complete : Complete (fun _ => True) (fun x : Builder × Var => algebra.lift_word build x.1 x.2) built
    (fun x => (Algebra.lift x.2).run x.1) :=
  complete_eq _ (fun x => lift_word_eq x.1 x.2) (fun _ => rfl) ({}, 0)

theorem ext_scale_constant_sound :
    Sound (fun x : Builder × ExtVar × field.Gl => algebra.ext_scale_constant build x.1 x.2.1 x.2.2) built
      (fun x => (Algebra.extScaleConstant x.2.1 (wordOf x.2.2)).run x.1) :=
  sound_eq _ (fun x => ext_scale_constant_eq x.1 x.2.1 x.2.2) fun _ => rfl
theorem ext_scale_constant_complete : Complete (fun _ => True)
    (fun x : Builder × ExtVar × field.Gl => algebra.ext_scale_constant build x.1 x.2.1 x.2.2) built
      (fun x => (Algebra.extScaleConstant x.2.1 (wordOf x.2.2)).run x.1) :=
  complete_eq _ (fun x => ext_scale_constant_eq x.1 x.2.1 x.2.2) (fun _ => rfl) ({}, zeroExt, field.GL_ZERO)

theorem one_minus_sound : Sound (fun x : Builder × ExtVar => algebra.one_minus build x.1 x.2) built
    (fun x => (Algebra.oneMinus x.2).run x.1) :=
  sound_eq _ (fun x => one_minus_eq x.1 x.2) fun _ => rfl
theorem one_minus_complete : Complete (fun _ => True) (fun x : Builder × ExtVar => algebra.one_minus build x.1 x.2)
    built (fun x => (Algebra.oneMinus x.2).run x.1) :=
  complete_eq _ (fun x => one_minus_eq x.1 x.2) (fun _ => rfl) ({}, zeroExt)

theorem eq1_sound : Sound (fun x : Builder × ExtVar × ExtVar => algebra.eq1 build x.1 x.2.1 x.2.2) built
    (fun x => (Algebra.eq1 x.2.1 x.2.2).run x.1) :=
  sound_eq _ (fun x => eq1_eq x.1 x.2.1 x.2.2) fun _ => rfl
theorem eq1_complete : Complete (fun _ => True) (fun x : Builder × ExtVar × ExtVar => algebra.eq1 build x.1 x.2.1 x.2.2)
    built (fun x => (Algebra.eq1 x.2.1 x.2.2).run x.1) :=
  complete_eq _ (fun x => eq1_eq x.1 x.2.1 x.2.2) (fun _ => rfl) ({}, zeroExt, zeroExt)

theorem dot_sound : Sound (fun x : Builder × Slice ExtVar × Slice ExtVar => algebra.dot build x.1 x.2.1 x.2.2) built
    (fun x => (Algebra.dot x.2.1.val.toArray x.2.2.val.toArray).run x.1) :=
  sound_eq _ (fun x => dot_eq x.1 x.2.1 x.2.2) fun _ => rfl
theorem dot_complete : Complete (fun _ => True)
    (fun x : Builder × Slice ExtVar × Slice ExtVar => algebra.dot build x.1 x.2.1 x.2.2) built
    (fun x => (Algebra.dot x.2.1.val.toArray x.2.2.val.toArray).run x.1) :=
  complete_eq _ (fun x => dot_eq x.1 x.2.1 x.2.2) (fun _ => rfl) ({}, emptySlice, emptySlice)

theorem eq_eval_sound :
    Sound (fun x : Builder × Slice ExtVar × Slice ExtVar => algebra.eq_eval build x.1 x.2.1 x.2.2) built
      (fun x => (Algebra.eqEval x.2.1.val.toArray x.2.2.val.toArray).run x.1) :=
  sound_eq _ (fun x => eq_eval_eq x.1 x.2.1 x.2.2) fun _ => rfl
theorem eq_eval_complete : Complete (fun _ => True)
    (fun x : Builder × Slice ExtVar × Slice ExtVar => algebra.eq_eval build x.1 x.2.1 x.2.2) built
    (fun x => (Algebra.eqEval x.2.1.val.toArray x.2.2.val.toArray).run x.1) :=
  complete_eq _ (fun x => eq_eval_eq x.1 x.2.1 x.2.2) (fun _ => rfl) ({}, emptySlice, emptySlice)

theorem eq_at_sound : Sound (fun x : Builder × Slice ExtVar × Usize => algebra.eq_at build x.1 x.2.1 x.2.2) built
    (fun x => (Algebra.eqAt x.2.1.val.toArray x.2.2.val).run x.1) :=
  sound_eq _ (fun x => eq_at_eq x.1 x.2.1 x.2.2) fun _ => rfl
theorem eq_at_complete : Complete (fun _ => True)
    (fun x : Builder × Slice ExtVar × Usize => algebra.eq_at build x.1 x.2.1 x.2.2) built
    (fun x => (Algebra.eqAt x.2.1.val.toArray x.2.2.val).run x.1) :=
  complete_eq _ (fun x => eq_at_eq x.1 x.2.1 x.2.2) (fun _ => rfl) ({}, emptySlice, 0#usize)

theorem interpolate_sound :
    Sound (fun x : Builder × Slice ExtVar × ExtVar => algebra.interpolate build x.1 x.2.1 x.2.2) built
      (fun x => (Algebra.interpolate x.2.1.val.toArray x.2.2).run x.1) :=
  sound_eq _ (fun x => interpolate_eq x.1 x.2.1 x.2.2) fun _ => rfl
theorem interpolate_complete : Complete (fun _ => True)
    (fun x : Builder × Slice ExtVar × ExtVar => algebra.interpolate build x.1 x.2.1 x.2.2) built
    (fun x => (Algebra.interpolate x.2.1.val.toArray x.2.2).run x.1) :=
  complete_eq _ (fun x => interpolate_eq x.1 x.2.1 x.2.2) (fun _ => rfl) ({}, emptySlice, zeroExt)

theorem horner_sound : Sound (fun x : Builder × Slice ExtVar × ExtVar => algebra.horner build x.1 x.2.1 x.2.2) built
    (fun x => (Algebra.horner x.2.1.val.toArray x.2.2).run x.1) :=
  sound_eq _ (fun x => horner_eq x.1 x.2.1 x.2.2) fun _ => rfl
theorem horner_complete : Complete (fun _ => True)
    (fun x : Builder × Slice ExtVar × ExtVar => algebra.horner build x.1 x.2.1 x.2.2) built
    (fun x => (Algebra.horner x.2.1.val.toArray x.2.2).run x.1) :=
  complete_eq _ (fun x => horner_eq x.1 x.2.1 x.2.2) (fun _ => rfl) ({}, emptySlice, zeroExt)

theorem power_from_bits_sound :
    Sound (fun x : Builder × field.Gl × Slice Var => algebra.power_from_bits build x.1 x.2.1 x.2.2) built
      (fun x => (Algebra.powerFromBits (wordOf x.2.1) x.2.2.val.toArray).run x.1) :=
  sound_eq _ (fun x => power_from_bits_eq x.1 x.2.1 x.2.2) fun _ => rfl
theorem power_from_bits_complete : Complete (fun _ => True)
    (fun x : Builder × field.Gl × Slice Var => algebra.power_from_bits build x.1 x.2.1 x.2.2) built
    (fun x => (Algebra.powerFromBits (wordOf x.2.1) x.2.2.val.toArray).run x.1) :=
  complete_eq _ (fun x => power_from_bits_eq x.1 x.2.1 x.2.2) (fun _ => rfl) ({}, field.GL_ZERO, emptySlice)

theorem kx_zero_sound : Sound (algebra.Kx.zero build) builtKx (fun s => KxVar.zero.run s) :=
  sound_eq _ kx_zero_eq fun _ => rfl
theorem kx_zero_complete : Complete (fun _ => True) (algebra.Kx.zero build) builtKx (fun s => KxVar.zero.run s) :=
  complete_eq _ kx_zero_eq (fun _ => rfl) {}

theorem kx_one_sound : Sound (algebra.Kx.one build) builtKx (fun s => KxVar.one.run s) :=
  sound_eq _ kx_one_eq fun _ => rfl
theorem kx_one_complete : Complete (fun _ => True) (algebra.Kx.one build) builtKx (fun s => KxVar.one.run s) :=
  complete_eq _ kx_one_eq (fun _ => rfl) {}

theorem kx_from_k_sound : Sound (fun x : Builder × Aeneas.Std.Array Var 2#usize => algebra.Kx.from_k build x.1 x.2)
    builtKx (fun x => (KxVar.fromK (kOf x.2)).run x.1) :=
  sound_eq _ (fun x => kx_from_k_eq x.1 x.2) fun _ => rfl
theorem kx_from_k_complete : Complete (fun _ => True)
    (fun x : Builder × Aeneas.Std.Array Var 2#usize => algebra.Kx.from_k build x.1 x.2)
    builtKx (fun x => (KxVar.fromK (kOf x.2)).run x.1) :=
  complete_eq _ (fun x => kx_from_k_eq x.1 x.2) (fun _ => rfl) ({}, Aeneas.Std.Array.make 2#usize [0, 0])

def zeroKx : algebra.Kx Builder Var ExtVar := ⟨zeroExt, zeroExt⟩

theorem kx_add_sound : Sound (fun x : algebra.Kx Builder Var ExtVar × Builder × algebra.Kx Builder Var ExtVar =>
    algebra.Kx.add build x.1 x.2.1 x.2.2) builtKx (fun x => (KxVar.add (kxOf x.1) (kxOf x.2.2)).run x.2.1) :=
  sound_eq _ (fun x => kx_add_eq x.2.1 x.1 x.2.2) fun _ => rfl
theorem kx_add_complete : Complete (fun _ => True)
    (fun x : algebra.Kx Builder Var ExtVar × Builder × algebra.Kx Builder Var ExtVar =>
      algebra.Kx.add build x.1 x.2.1 x.2.2) builtKx (fun x => (KxVar.add (kxOf x.1) (kxOf x.2.2)).run x.2.1) :=
  complete_eq _ (fun x => kx_add_eq x.2.1 x.1 x.2.2) (fun _ => rfl) (zeroKx, {}, zeroKx)

theorem kx_mul_sound : Sound (fun x : algebra.Kx Builder Var ExtVar × Builder × algebra.Kx Builder Var ExtVar =>
    algebra.Kx.mul build x.1 x.2.1 x.2.2) builtKx (fun x => (KxVar.mul (kxOf x.1) (kxOf x.2.2)).run x.2.1) :=
  sound_eq _ (fun x => kx_mul_eq x.2.1 x.1 x.2.2) fun _ => rfl
theorem kx_mul_complete : Complete (fun _ => True)
    (fun x : algebra.Kx Builder Var ExtVar × Builder × algebra.Kx Builder Var ExtVar =>
      algebra.Kx.mul build x.1 x.2.1 x.2.2) builtKx (fun x => (KxVar.mul (kxOf x.1) (kxOf x.2.2)).run x.2.1) :=
  complete_eq _ (fun x => kx_mul_eq x.2.1 x.1 x.2.2) (fun _ => rfl) (zeroKx, {}, zeroKx)

theorem kx_scale_sound : Sound (fun x : algebra.Kx Builder Var ExtVar × Builder × ExtVar =>
    algebra.Kx.scale build x.1 x.2.1 x.2.2) builtKx (fun x => (KxVar.scale (kxOf x.1) x.2.2).run x.2.1) :=
  sound_eq _ (fun x => kx_scale_eq x.2.1 x.1 x.2.2) fun _ => rfl
theorem kx_scale_complete : Complete (fun _ => True)
    (fun x : algebra.Kx Builder Var ExtVar × Builder × ExtVar =>
      algebra.Kx.scale build x.1 x.2.1 x.2.2) builtKx (fun x => (KxVar.scale (kxOf x.1) x.2.2).run x.2.1) :=
  complete_eq _ (fun x => kx_scale_eq x.2.1 x.1 x.2.2) (fun _ => rfl) (zeroKx, {}, zeroExt)

theorem kx_one_minus_sound : Sound (fun x : algebra.Kx Builder Var ExtVar × Builder =>
    algebra.Kx.one_minus build x.1 x.2) builtKx (fun x => (KxVar.oneMinus (kxOf x.1)).run x.2) :=
  sound_eq _ (fun x => kx_one_minus_eq x.2 x.1) fun _ => rfl
theorem kx_one_minus_complete : Complete (fun _ => True)
    (fun x : algebra.Kx Builder Var ExtVar × Builder => algebra.Kx.one_minus build x.1 x.2) builtKx
    (fun x => (KxVar.oneMinus (kxOf x.1)).run x.2) :=
  complete_eq _ (fun x => kx_one_minus_eq x.2 x.1) (fun _ => rfl) (zeroKx, {})

theorem eq_table_sound : Sound (fun x : Builder × Slice ExtVar => algebra.eq_table build x.1 x.2) builtVec
    (fun x => (Algebra.eqTable x.2.val.toArray).run x.1) := by
  rintro ⟨s, point⟩ y h
  simp only [algebra.eq_table, ext_one_eq, bind_ok, uncurry_eq, alloc.vec.FromVecArray.from] at h
  obtain ⟨e2, e1⟩ := eq_table_loop_sound point _ _ y h
  simp only [builtVec]
  rw [e1, e2]
  simp only [Algebra.eqTable, StateT.run_bind, List.forIn_toArray, tableStep, Array.forIn_toList, map_bind,
    bind_pure_comp, pushProduct, Functor.map_map]
  rfl

theorem eq_table_complete : Complete (fun x : Builder × Slice ExtVar => 2 ^ x.2.val.length ≤ Usize.max)
    (fun x : Builder × Slice ExtVar => algebra.eq_table build x.1 x.2) builtVec
    (fun x => (Algebra.eqTable x.2.val.toArray).run x.1) :=
  complete_of eq_table_sound ⟨({}, emptySlice), by simp [emptySlice]; scalar_tac⟩ fun x hx => by
    simp only [algebra.eq_table, ext_one_eq, bind_ok, uncurry_eq, alloc.vec.FromVecArray.from]
    exact eq_table_loop_succeeds x.2 hx _ _ (by simp)

end

end CompressionLeanRust
