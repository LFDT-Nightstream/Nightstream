import CompressionLeanRust.Native
import CompressionLeanRust.Algebra

/-! The translated `eq_at`, `eq_eval` and `eq_table` of `neo_verified::algebra`, run on `nativeB` (the
verifier's `Native`), equal `Lib.eqAt` and `Lib.eqEval` of `CompressionLean/Lib/Cube.lean` over
`Cubic`. The shrink verifier runs them for its outer check, its public value and its eq tables.

A copy reads an input `Ext` as `cubicOf` reads it, so a word of `p` or more reads as its value mod p,
and it gives its value as `cubicWords`, the canonical words of a `Cubic`. Every output is `1` or a
product of `Native`, so its words are canonical (`Canon`), and `extValue` reads them as `cubicWords`
does (`cubicWords_cubicOf`). `eq_at` and `eq_eval` always return, so their `Complete` premise is
`True`. `eq_table` can fail where `Vec::push` passes `usize::MAX` entries; its premise is
`2^n ≤ usize::MAX` for a point of `n` coordinates. -/

namespace CompressionLeanRust

open Aeneas Aeneas.Std Result
open neo_verified
open CompressionLean
open NightstreamFPrime.Spec (goldilocksModulus)
open Polynomial

/-- Every word of `e` is below `p`. -/
def Canon (e : field.Ext) : Prop :=
  e.c0.val < goldilocksModulus ∧ e.c1.val < goldilocksModulus ∧ e.c2.val < goldilocksModulus

theorem cubic_monic : cubic.Monic := by
  unfold cubic
  refine ((monic_X_pow 3).sub_of_left ?_).sub_of_left ?_
  · rw [degree_X, degree_X_pow]; decide
  · rw [degree_one, degree_sub_eq_left_of_degree_lt (by rw [degree_X, degree_X_pow]; decide), degree_X_pow]
    decide

/-- The canonical words of a `Cubic`: its coordinates in the basis `1, x₀, x₀²`. -/
noncomputable def cubicWords (c : Cubic) : ℕ × ℕ × ℕ :=
  let r := AdjoinRoot.modByMonicHom cubic_monic c
  ((r.coeff 0).val, (r.coeff 1).val, (r.coeff 2).val)

theorem cubicWords_cubicOf {e : field.Ext} (h : Canon e) : cubicWords (cubicOf e) = extValue e := by
  have hmk : cubicOf e = AdjoinRoot.mk cubic (C (glOf e.c0) + C (glOf e.c1) * X + C (glOf e.c2) * X ^ 2) := by
    simp only [cubicOf, map_add, map_mul, map_pow, AdjoinRoot.mk_C, AdjoinRoot.mk_X, AdjoinRoot.algebraMap_eq]
    rfl
  have hdeg : (C (glOf e.c0) + C (glOf e.c1) * X + C (glOf e.c2) * X ^ 2).degree < cubic.degree := by
    rw [degree_eq_natDegree cubic_ne_zero, cubic_natDegree]
    refine (degree_le_of_natDegree_le (show _ ≤ 2 by compute_degree)).trans_lt ?_
    exact_mod_cast (by norm_num : (2 : ℕ) < 3)
  obtain ⟨h0, h1, h2⟩ := h
  simp only [cubicWords, hmk, AdjoinRoot.modByMonicHom_mk, (modByMonic_eq_self_iff cubic_monic).mpr hdeg]
  simp [coeff_X_pow, coeff_X, glOf, extValue, ZMod.val_natCast_of_lt h0, ZMod.val_natCast_of_lt h1,
    ZMod.val_natCast_of_lt h2]

theorem redW_lt (w : U64) : (redW w).val < goldilocksModulus := by
  rw [show (redW w).val = (wordOf w).toNat from (show Exact (redW w) (wordOf w) from toU_red_word w).val]
  exact (wordOf w).canonical

theorem glSub_lt (a b : field.Gl) : (glSub a b).val < goldilocksModulus :=
  (glSub_exact a b).val ▸ (wordOf a - wordOf b).canonical

theorem glMul_lt (a b : field.Gl) : (glMul a b).val < goldilocksModulus :=
  (glMul_exact a b).val ▸ (wordOf a * wordOf b).canonical

theorem canon_extAdd (a b : field.Ext) : Canon (extAdd a b) := ⟨glAdd_lt _ _, glAdd_lt _ _, glAdd_lt _ _⟩

theorem canon_extSub (a b : field.Ext) : Canon (extSub a b) := ⟨glSub_lt _ _, glSub_lt _ _, glSub_lt _ _⟩

theorem canon_extMul (a b : field.Ext) : Canon (extMul a b) := ⟨glAdd_lt _ _, glAdd_lt _ _, glAdd_lt _ _⟩

theorem native_ext_add (u : native.Native) (a b : field.Ext) :
    native.Native.Insts.Neo_verifiedBackendBackendGlExt.ext_add u a b = ok (extAdd a b, u) := by
  simp only [native.Native.Insts.Neo_verifiedBackendBackendGlExt.ext_add, ext_add_eq, bind_ok]

theorem native_ext_sub (u : native.Native) (a b : field.Ext) :
    native.Native.Insts.Neo_verifiedBackendBackendGlExt.ext_sub u a b = ok (extSub a b, u) := by
  simp only [native.Native.Insts.Neo_verifiedBackendBackendGlExt.ext_sub, ext_sub_eq, bind_ok]

theorem native_ext_mul (u : native.Native) (a b : field.Ext) :
    native.Native.Insts.Neo_verifiedBackendBackendGlExt.ext_mul u a b = ok (extMul a b, u) := by
  simp only [native.Native.Insts.Neo_verifiedBackendBackendGlExt.ext_mul, ext_mul_eq, bind_ok]

/-- An extension constant as `Native` holds it. -/
def constW (v : field.Ext) : field.Ext := ⟨redW (redW v.c0), redW (redW v.c1), redW (redW v.c2)⟩

theorem ext_constant_native (u : native.Native) (v : field.Ext) :
    algebra.ext_constant nativeB u v = ok (constW v, u) := by
  simp only [algebra.ext_constant, gl_word_eq, bind_ok, native.Native.Insts.Neo_verifiedBackendBackendGlExt.ext_constant,
    reduce_eq, index_eq (n := 3#usize) _ 0#usize (by simp), index_eq (n := 3#usize) _ 1#usize (by simp),
    index_eq (n := 3#usize) _ 2#usize (by simp)]
  rfl

theorem canon_constW (v : field.Ext) : Canon (constW v) := ⟨redW_lt _, redW_lt _, redW_lt _⟩

theorem cubicOf_constW (v : field.Ext) : cubicOf (constW v) = cubicOf v := by
  simp only [cubicOf, constW, glOf_red]

theorem ext_one_native (u : native.Native) : algebra.ext_one nativeB u = ok (constW field.EXT_ONE, u) := by
  rw [algebra.ext_one, ext_constant_native]

theorem ext_zero_native (u : native.Native) : algebra.ext_zero nativeB u = ok (constW field.EXT_ZERO, u) := by
  rw [algebra.ext_zero, ext_constant_native]

theorem one_minus_native (u : native.Native) (x : field.Ext) :
    algebra.one_minus nativeB u x = ok (extSub (constW field.EXT_ONE) x, u) := by
  simp only [algebra.one_minus, ext_one_native, bind_ok, uncurry_eq, native_ext_sub]


/-! ## `eq_at` -/

theorem cubicOf_oneMinus (x : field.Ext) : cubicOf (extSub (constW field.EXT_ONE) x) = 1 - cubicOf x := by
  rw [extSub_cubic, cubicOf_constW, cubicOf_one]

theorem eqAt_cons (c : Cubic) (rest : List Cubic) (j : ℕ) :
    Lib.eqAt (c :: rest) j = (if j % 2 = 1 then c else 1 - c) * Lib.eqAt rest (j / 2) := rfl

theorem factor_cubic (x : field.Ext) (i t : ℕ) :
    cubicOf (if i.testBit t then x else extSub (constW field.EXT_ONE) x) =
      if i / 2 ^ t % 2 = 1 then cubicOf x else 1 - cubicOf x := by
  rw [Nat.testBit_eq_decide_div_mod_eq]
  by_cases h : i / 2 ^ t % 2 = 1 <;> simp [h, cubicOf_oneMinus]

theorem div_pow_succ (i t : ℕ) : i / 2 ^ t / 2 = i / 2 ^ (t + 1) := by
  rw [Nat.div_div_eq_div_mul, pow_succ]

theorem drop_map_cons {α β : Type} (f : α → β) (l : List α) (t : ℕ) (h : t < l.length) :
    (l.map f).drop t = f l[t] :: (l.map f).drop (t + 1) := by
  rw [List.drop_eq_getElem_cons (by simpa using h), List.getElem_map]

/-- One step of `eq_at`'s loop: multiply by `point[t]` or `1 - point[t]`, by bit `t` of `index`. -/
theorem eq_at_body (point : Slice field.Ext) (index : Usize) (b : native.Native) (total : field.Ext) (t : Usize)
    (ht : t.val < point.val.length) :
    algebra.eq_at_loop.body nativeB point index b total t =
      ok (.cont (b, extMul total (if index.val.testBit t.val then point.val[t.val]
        else extSub (constW field.EXT_ONE) point.val[t.val]), core.num.Usize.wrapping_add t 1#usize)) := by
  by_cases hbit : index.val.testBit t.val <;>
  simp [algebra.eq_at_loop.body, Slice.len, show t < Usize.ofNatCore point.val.length _ by scalar_tac,
    index_bit_eq, hbit, slice_index point t ht, one_minus_native, native_ext_mul, lift]

theorem eq_at_native (u : native.Native) (point : Slice field.Ext) (index : Usize) :
    ∃ e, algebra.eq_at nativeB u point index = ok (e, u) ∧ Canon e ∧
      cubicOf e = Lib.eqAt (point.val.map cubicOf) index.val := by
  have hlen := point.property
  simp only [algebra.eq_at, ext_one_native, bind_ok, uncurry_eq]
  unfold algebra.eq_at_loop
  obtain ⟨y, hy, hq⟩ := loop_total
    (fun a : native.Native × field.Ext × Usize => algebra.eq_at_loop.body nativeB point index a.1 a.2.1 a.2.2)
    (fun n a => a.2.2.val ≤ point.val.length ∧ n = point.val.length - a.2.2.val ∧ Canon a.2.1 ∧
      cubicOf a.2.1 * Lib.eqAt ((point.val.map cubicOf).drop a.2.2.val) (index.val / 2 ^ a.2.2.val) =
        Lib.eqAt (point.val.map cubicOf) index.val)
    (fun y : field.Ext × native.Native => Canon y.1 ∧ cubicOf y.1 = Lib.eqAt (point.val.map cubicOf) index.val)
    (by
      rintro n ⟨b, total, t⟩ ⟨ht, rfl, hc, hv⟩
      dsimp only at ht hc hv ⊢
      by_cases hlt : t.val < point.val.length
      · right
        have hmax : t.val < Usize.max := by scalar_tac
        refine ⟨_, point.val.length - (t.val + 1), eq_at_body point index b total t hlt, by omega, ?_,
          by rw [add_one_val t hmax], canon_extMul _ _, ?_⟩
        · rw [add_one_val t hmax]; omega
        · dsimp only
          rw [add_one_val t hmax, ← hv, drop_map_cons cubicOf point.val t.val hlt, eqAt_cons, div_pow_succ,
            extMul_cubic, factor_cubic, mul_assoc]
      · left
        have hend : t.val = point.val.length := by omega
        refine ⟨(total, b), ?_, hc, ?_⟩
        · simp only [algebra.eq_at_loop.body, Slice.len, show ¬ t < Usize.ofNatCore point.val.length _ by scalar_tac,
            ite_false]
        · rw [← hv, hend, List.drop_eq_nil_of_le (by simp)]
          simp [Lib.eqAt])
    _ (u, constW field.EXT_ONE, 0#usize)
    ⟨by simp, rfl, canon_constW _, by simp [cubicOf_constW, cubicOf_one]⟩
  exact ⟨y.1, hy, hq⟩


/-! ## `eq_eval` -/

theorem eq1_native (u : native.Native) (x y : field.Ext) :
    ∃ e, algebra.eq1 nativeB u x y = ok (e, u) ∧ cubicOf e = cubicOf x * cubicOf y + (1 - cubicOf x) * (1 - cubicOf y) := by
  refine ⟨extSub (extAdd (extAdd (extMul x y) (extMul x y)) (constW field.EXT_ONE)) (extAdd x y), ?_, ?_⟩
  · simp only [algebra.eq1, native_ext_mul, native_ext_add, ext_one_native, native_ext_sub, bind_ok, uncurry_eq]
  · simp only [extSub_cubic, extAdd_cubic, extMul_cubic, cubicOf_constW, cubicOf_one]
    ring

theorem eqEval_cons (r : Cubic) (ρ : List Cubic) (x : Cubic) (xs : List Cubic) :
    Lib.eqEval (r :: ρ) (x :: xs) = (r * x + (1 - r) * (1 - x)) * Lib.eqEval ρ xs := rfl

theorem eq_eval_native (u : native.Native) (x y : Slice field.Ext) :
    ∃ e, algebra.eq_eval nativeB u x y = ok (e, u) ∧ Canon e ∧
      cubicOf e = Lib.eqEval (x.val.map cubicOf) (y.val.map cubicOf) := by
  have hx := x.property
  simp only [algebra.eq_eval, ext_one_native, bind_ok, uncurry_eq]
  unfold algebra.eq_eval_loop
  obtain ⟨r, hr, hq⟩ := loop_total
    (fun a : native.Native × field.Ext × Usize => algebra.eq_eval_loop.body nativeB x y a.1 a.2.1 a.2.2)
    (fun n a => a.2.2.val ≤ x.val.length ∧ a.2.2.val ≤ y.val.length ∧
      n = min x.val.length y.val.length - a.2.2.val ∧ Canon a.2.1 ∧
      cubicOf a.2.1 * Lib.eqEval ((x.val.map cubicOf).drop a.2.2.val) ((y.val.map cubicOf).drop a.2.2.val) =
        Lib.eqEval (x.val.map cubicOf) (y.val.map cubicOf))
    (fun r : field.Ext × native.Native => Canon r.1 ∧ cubicOf r.1 = Lib.eqEval (x.val.map cubicOf) (y.val.map cubicOf))
    (by
      rintro n ⟨b, total, t⟩ ⟨htx, hty, rfl, hc, hv⟩
      dsimp only at htx hty hc hv ⊢
      by_cases hlt : t.val < x.val.length ∧ t.val < y.val.length
      · right
        have hmax : t.val < Usize.max := by scalar_tac
        obtain ⟨e, he, hev⟩ := eq1_native b x.val[t.val] y.val[t.val]
        refine ⟨(b, extMul total e, core.num.Usize.wrapping_add t 1#usize),
          min x.val.length y.val.length - (t.val + 1), ?_, by omega, ?_, ?_, by rw [add_one_val t hmax],
          canon_extMul _ _, ?_⟩
        · simp only [algebra.eq_eval_loop.body, Slice.len, show t < Usize.ofNatCore x.val.length _ by scalar_tac,
            show t < Usize.ofNatCore y.val.length _ by scalar_tac, ite_true, slice_index x t hlt.1,
            slice_index y t hlt.2, he, native_ext_mul, bind_ok, uncurry_eq, lift]
        · rw [add_one_val t hmax]; omega
        · rw [add_one_val t hmax]; omega
        · dsimp only
          rw [add_one_val t hmax, ← hv, drop_map_cons cubicOf x.val t.val hlt.1, drop_map_cons cubicOf y.val t.val hlt.2,
            eqEval_cons, extMul_cubic, hev, mul_assoc]
      · left
        refine ⟨(total, b), ?_, hc, ?_⟩
        · by_cases h1 : t.val < x.val.length
          · simp only [algebra.eq_eval_loop.body, Slice.len, show t < Usize.ofNatCore x.val.length _ by scalar_tac,
              show ¬ t < Usize.ofNatCore y.val.length _ by scalar_tac, ite_true, ite_false]
          · simp only [algebra.eq_eval_loop.body, Slice.len, show ¬ t < Usize.ofNatCore x.val.length _ by scalar_tac,
              ite_false]
        · rw [← hv]
          rcases not_and_or.mp hlt with h | h
          · rw [List.drop_eq_nil_of_le (by simp; omega)]
            simp [Lib.eqEval]
          · rw [List.drop_eq_nil_of_le (show (y.val.map cubicOf).length ≤ t.val by simp; omega)]
            cases ((x.val.map cubicOf).drop t.val) <;> simp [Lib.eqEval])
    _ (u, constW field.EXT_ONE, 0#usize)
    ⟨by simp, by simp, rfl, canon_constW _, by simp [cubicOf_constW, cubicOf_one]⟩
  exact ⟨r.1, hr, hq⟩

/-! ## `eq_table` -/

/-- `loop_sound`, and `loop_total` where `P` holds: `P` makes each step return and is kept. -/
theorem loop_ok {α β : Type} (body : α → Result (ControlFlow α β)) (I : ℕ → α → Prop) (Q : β → Prop) (P : α → Prop)
    (h : ∀ n a c, I n a → body a = ok c → (∃ b, c = .done b ∧ Q b) ∨ ∃ a' m, c = .cont a' ∧ m < n ∧ I m a')
    (succeeds : ∀ n a, I n a → P a → ∃ c, body a = ok c)
    (keeps : ∀ n a a', I n a → P a → body a = ok (.cont a') → P a') (n : ℕ) (a : α) (hI : I n a) :
    (∀ y, loop body a = ok y → Q y) ∧ (P a → ∃ y, loop body a = ok y ∧ Q y) := by
  refine ⟨fun y hy => loop_sound body I Q h n a y hI hy, fun hP => ?_⟩
  refine loop_total body (fun n a => I n a ∧ P a) Q ?_ n a ⟨hI, hP⟩
  rintro n a ⟨hI, hP⟩
  obtain ⟨c, hc⟩ := succeeds n a hI hP
  rcases h n a c hI hc with ⟨b, rfl, hQ⟩ | ⟨a', m, rfl, hm, hI'⟩
  · exact Or.inl ⟨b, hc, hQ⟩
  · exact Or.inr ⟨a', m, hc, hm, hI', keeps n a a' hI hP hc⟩

theorem vec_index_lt {α : Type} (v : alloc.vec.Vec α) (i : Usize) : i < alloc.vec.Vec.len v ↔ i.val < v.val.length := by
  rw [UScalar.lt_equiv, alloc.vec.Vec.len_val]

/-- An inner loop of `eq_table` on `Native` appends `table[i]·by` for each `i` from its start. -/
theorem push_products {body : alloc.vec.Vec field.Ext → field.Ext → native.Native → alloc.vec.Vec field.Ext → Usize →
      Result (ControlFlow (native.Native × alloc.vec.Vec field.Ext × Usize) (native.Native × alloc.vec.Vec field.Ext))}
    (hbody : ∀ table by_ b next i, body table by_ b next i =
      if i < alloc.vec.Vec.len table then do
        let t ← alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice field.Ext) table i
        let (t1, b1) ← native.Native.Insts.Neo_verifiedBackendBackendGlExt.ext_mul b t by_
        let next1 ← alloc.vec.Vec.push next t1
        let i2 ← lift (core.num.Usize.wrapping_add i 1#usize)
        ok (ControlFlow.cont (b1, next1, i2))
      else ok (ControlFlow.done (b, next)))
    (table : alloc.vec.Vec field.Ext) (by_ : field.Ext) (b : native.Native) (next : alloc.vec.Vec field.Ext) :
    (∀ y, loop (fun a : native.Native × alloc.vec.Vec field.Ext × Usize => body table by_ a.1 a.2.1 a.2.2)
        (b, next, 0#usize) = ok y → y.2.val = next.val ++ table.val.map (fun v => extMul v by_)) ∧
    (next.val.length + table.val.length ≤ Usize.max →
      ∃ y, loop (fun a : native.Native × alloc.vec.Vec field.Ext × Usize => body table by_ a.1 a.2.1 a.2.2)
        (b, next, 0#usize) = ok y ∧ y.2.val = next.val ++ table.val.map (fun v => extMul v by_)) := by
  have htab := table.property
  have := loop_ok (fun a : native.Native × alloc.vec.Vec field.Ext × Usize => body table by_ a.1 a.2.1 a.2.2)
    (fun n a => a.2.2.val ≤ table.val.length ∧ n = table.val.length - a.2.2.val ∧
      a.2.1.val = next.val ++ (table.val.take a.2.2.val).map (fun v => extMul v by_))
    (fun y : native.Native × alloc.vec.Vec field.Ext => y.2.val = next.val ++ table.val.map (fun v => extMul v by_))
    (fun a => next.val.length + table.val.length ≤ Usize.max)
    (by
      rintro n ⟨b, nx, i⟩ c ⟨hi, rfl, hv⟩ hc
      dsimp only at hi hv hc ⊢
      rw [hbody] at hc
      by_cases hlt : i.val < table.val.length
      · right
        have hmax : i.val < Usize.max := by scalar_tac
        rw [ite_eq_left_iff.mpr (fun h => absurd ((vec_index_lt table i).mpr hlt) h)] at hc
        simp only [vec_index table i hlt, native_ext_mul, bind_ok, uncurry_eq, lift] at hc
        obtain ⟨w, hw, hc⟩ := bind_eq_ok hc
        rw [Result.ok.injEq] at hc
        subst hc
        refine ⟨(b, w, core.num.Usize.wrapping_add i 1#usize), table.val.length - (i.val + 1), rfl, by omega, ?_,
          by rw [add_one_val i hmax], ?_⟩
        · rw [add_one_val i hmax]; omega
        · rw [vec_push_val hw, add_one_val i hmax, hv, List.take_add_one, List.getElem?_eq_getElem hlt,
            Option.toList_some, List.map_append, List.map_singleton, List.append_assoc]
      · left
        rw [ite_eq_right_iff.mpr (fun h => absurd ((vec_index_lt table i).mp h) hlt), Result.ok.injEq] at hc
        subst hc
        refine ⟨_, rfl, ?_⟩
        rw [hv, List.take_of_length_le (by omega)])
    (by
      rintro n ⟨b, nx, i⟩ ⟨hi, rfl, hv⟩ hP
      dsimp only at hi hv hP ⊢
      rw [hbody]
      by_cases hlt : i.val < table.val.length
      · rw [ite_eq_left_iff.mpr (fun h => absurd ((vec_index_lt table i).mpr hlt) h)]
        have hlen : nx.val.length < Usize.max := by rw [hv]; simp; omega
        obtain ⟨w, hw, -⟩ := vec_push_ok nx (extMul table.val[i.val] by_) hlen
        exact ⟨ControlFlow.cont (b, w, core.num.Usize.wrapping_add i 1#usize),
          by simp only [vec_index table i hlt, native_ext_mul, bind_ok, uncurry_eq, lift, hw]⟩
      · exact ⟨_, ite_eq_right_iff.mpr (fun h => absurd ((vec_index_lt table i).mp h) hlt)⟩)
    (fun _ _ _ _ hP _ => hP) _ (b, next, 0#usize) ⟨by simp, rfl, by simp⟩
  exact this


theorem inner0_native (table : alloc.vec.Vec field.Ext) (by_ : field.Ext) (b : native.Native)
    (next : alloc.vec.Vec field.Ext) :
    (∀ y, algebra.eq_table_loop0_loop0 nativeB b table by_ next 0#usize = ok y →
      y.2.val = next.val ++ table.val.map (fun v => extMul v by_)) ∧
    (next.val.length + table.val.length ≤ Usize.max → ∃ y, algebra.eq_table_loop0_loop0 nativeB b table by_ next 0#usize = ok y ∧
      y.2.val = next.val ++ table.val.map (fun v => extMul v by_)) :=
  push_products (body := fun table by_ b next i => algebra.eq_table_loop0_loop0.body nativeB table by_ b next i)
    (fun _ _ _ _ _ => rfl) table by_ b next

theorem inner1_native (table : alloc.vec.Vec field.Ext) (by_ : field.Ext) (b : native.Native)
    (next : alloc.vec.Vec field.Ext) :
    (∀ y, algebra.eq_table_loop0_loop1 nativeB b table by_ next 0#usize = ok y →
      y.2.val = next.val ++ table.val.map (fun v => extMul v by_)) ∧
    (next.val.length + table.val.length ≤ Usize.max → ∃ y, algebra.eq_table_loop0_loop1 nativeB b table by_ next 0#usize = ok y ∧
      y.2.val = next.val ++ table.val.map (fun v => extMul v by_)) :=
  push_products (body := fun table by_ b next i => algebra.eq_table_loop0_loop1.body nativeB table by_ b next i)
    (fun _ _ _ _ _ => rfl) table by_ b next

theorem eqAt_append_single (p : List Cubic) (c : Cubic) (i : ℕ) :
    Lib.eqAt (p ++ [c]) i = Lib.eqAt p (i % 2 ^ p.length) * (if i / 2 ^ p.length % 2 = 1 then c else 1 - c) := by
  induction p generalizing i with
  | nil => simp [Lib.eqAt]
  | cons r rest ih =>
    rw [List.cons_append, eqAt_cons, ih, List.length_cons, eqAt_cons, pow_succ', Nat.mod_mul_right_mod,
      Nat.mod_mul_right_div_self, ← Nat.div_div_eq_div_mul, mul_assoc]

/-- The table of `eq_table` after the first `t` coordinates of `l`. -/
noncomputable def tableAt (l : List Cubic) (t : ℕ) : List Cubic := (List.range (2 ^ t)).map (Lib.eqAt (l.take t))

theorem tableAt_succ (l : List Cubic) (t : ℕ) (h : t < l.length) :
    tableAt l (t + 1) = (tableAt l t).map (· * (1 - l[t])) ++ (tableAt l t).map (· * l[t]) := by
  have htake : l.take (t + 1) = l.take t ++ [l[t]] := by
    rw [List.take_add_one, List.getElem?_eq_getElem h, Option.toList_some]
  have hlen : (l.take t).length = t := by rw [List.length_take]; omega
  unfold tableAt
  rw [pow_succ, mul_two, List.range_add, List.map_append, List.map_map, List.map_map, List.map_map, htake]
  congr 1
  · apply List.map_congr_left
    intro i hi
    rw [List.mem_range] at hi
    simp only [Function.comp, eqAt_append_single, hlen, Nat.mod_eq_of_lt hi, Nat.div_eq_of_lt hi]
    simp
  · apply List.map_congr_left
    intro j hj
    rw [List.mem_range] at hj
    have hmod : (2 ^ t + j) % 2 ^ t = j := by rw [Nat.add_mod_left, Nat.mod_eq_of_lt hj]
    have hdiv : (2 ^ t + j) / 2 ^ t = 1 := by
      rw [Nat.add_div_left _ (by positivity), Nat.div_eq_of_lt hj]
    simp only [Function.comp, eqAt_append_single, hlen, hmod, hdiv]
    simp

theorem eq_table_native (u : native.Native) (point : Slice field.Ext) :
    (∀ y, algebra.eq_table nativeB u point = ok y →
      y.1.val.map cubicOf = (List.range (2 ^ point.val.length)).map (Lib.eqAt (point.val.map cubicOf)) ∧
        ∀ e ∈ y.1.val, Canon e) ∧
    (2 ^ point.val.length ≤ Usize.max → ∃ y, algebra.eq_table nativeB u point = ok y) := by
  have hlen := point.property
  set l := point.val.map cubicOf with hl
  have hll : l.length = point.val.length := by simp [hl]
  have hone : ∃ v, alloc.vec.FromVecArray.from (Aeneas.Std.Array.make 1#usize [constW field.EXT_ONE]) = ok v ∧
      v.val = [constW field.EXT_ONE] := ⟨_, rfl, rfl⟩
  obtain ⟨v0, hv0, hv0val⟩ := hone
  simp only [algebra.eq_table, ext_one_native, bind_ok, uncurry_eq, hv0]
  unfold algebra.eq_table_loop0
  have key := loop_ok
    (fun a : native.Native × alloc.vec.Vec field.Ext × Usize => algebra.eq_table_loop0.body nativeB point a.1 a.2.1 a.2.2)
    (fun n a => a.2.2.val ≤ point.val.length ∧ n = point.val.length - a.2.2.val ∧
      a.2.1.val.map cubicOf = tableAt l a.2.2.val ∧ ∀ e ∈ a.2.1.val, Canon e)
    (fun y : alloc.vec.Vec field.Ext × native.Native =>
      y.1.val.map cubicOf = (List.range (2 ^ point.val.length)).map (Lib.eqAt l) ∧ ∀ e ∈ y.1.val, Canon e)
    (fun _ => 2 ^ point.val.length ≤ Usize.max)
    (by
      rintro n ⟨b, table, t⟩ c ⟨ht, rfl, hv, hc⟩ hbody
      dsimp only at ht hv hc hbody ⊢
      by_cases hlt : t.val < point.val.length
      · right
        have hmax : t.val < Usize.max := by scalar_tac
        simp only [algebra.eq_table_loop0.body, Slice.len, show t < Usize.ofNatCore point.val.length _ by scalar_tac,
          ite_true, slice_index point t hlt, one_minus_native, bind_ok, uncurry_eq, lift] at hbody
        obtain ⟨r1, h1, hbody⟩ := bind_eq_ok hbody
        obtain ⟨⟨b3, next1⟩, h2, hbody⟩ := bind_eq_ok hbody
        have hbody : ok (ControlFlow.cont (b3, next1, core.num.Usize.wrapping_add t 1#usize)) = ok c := hbody
        rw [Result.ok.injEq] at hbody
        subst hbody
        have e1 := (inner0_native table _ b _).1 r1 h1
        have e2 := (inner1_native table _ r1.1 r1.2).1 _ h2
        refine ⟨_, point.val.length - (t.val + 1), rfl, by omega, ?_, by rw [add_one_val t hmax], ?_, ?_⟩
        · rw [add_one_val t hmax]; omega
        · dsimp only
          have hlt' : t.val < l.length := by rw [hll]; exact hlt
          rw [add_one_val t hmax, e2, e1, tableAt_succ l t.val hlt', ← hv]
          simp [List.map_map, Function.comp_def, extMul_cubic, cubicOf_oneMinus, hl]
        · dsimp only
          rw [e2, e1]
          intro e he
          simp only [List.mem_append, List.mem_map] at he
          rcases he with (he | ⟨v, -, rfl⟩) | ⟨v, -, rfl⟩
          · exact absurd he List.not_mem_nil
          · exact canon_extMul _ _
          · exact canon_extMul _ _
      · left
        simp only [algebra.eq_table_loop0.body, Slice.len, show ¬ t < Usize.ofNatCore point.val.length _ by scalar_tac,
          ite_false, Result.ok.injEq] at hbody
        subst hbody
        have hend : t.val = point.val.length := by omega
        refine ⟨_, rfl, ?_, hc⟩
        rw [hv, hend, tableAt, ← hll, List.take_length])
    (by
      rintro n ⟨b, table, t⟩ ⟨ht, rfl, hv, hc⟩ hP
      dsimp only at ht hv hc hP ⊢
      have htab : table.val.length = 2 ^ t.val := by
        have := congrArg List.length hv
        simpa [tableAt] using this
      by_cases hlt : t.val < point.val.length
      · have hpow : 2 ^ (t.val + 1) ≤ 2 ^ point.val.length := Nat.pow_le_pow_right (by omega) hlt
        obtain ⟨r1, h1, e1⟩ := (inner0_native table (extSub (constW field.EXT_ONE) point.val[t.val]) b
          (alloc.vec.Vec.new field.Ext)).2 (by simp [alloc.vec.Vec.new])
        obtain ⟨r2, h2, -⟩ := (inner1_native table point.val[t.val] r1.1 r1.2).2
          (by rw [e1]; simp [alloc.vec.Vec.new, htab]; rw [pow_succ] at hpow; omega)
        exact ⟨ControlFlow.cont (r2.1, r2.2, core.num.Usize.wrapping_add t 1#usize), by
          simp only [algebra.eq_table_loop0.body, Slice.len,
          show t < Usize.ofNatCore point.val.length _ by scalar_tac, ite_true, slice_index point t hlt,
          one_minus_native, bind_ok, uncurry_eq, lift, h1, h2]⟩
      · exact ⟨ControlFlow.done (table, b), by simp only [algebra.eq_table_loop0.body, Slice.len,
          show ¬ t < Usize.ofNatCore point.val.length _ by scalar_tac, ite_false]⟩)
    (fun _ _ _ _ hP _ => hP) _ (u, v0, 0#usize)
    ⟨by simp, rfl, by simp [hv0val, tableAt, cubicOf_constW, cubicOf_one, Lib.eqAt], by
      simp only [hv0val, List.mem_singleton]; rintro e rfl; exact canon_constW _⟩
  exact ⟨key.1, fun h => let ⟨y, hy, _⟩ := key.2 h; ⟨y, hy⟩⟩


/-! ## `Sound` and `Complete` -/

/-- A run on `nativeB` that returns an extension value, as its words. -/
def nativeExt (x : field.Ext × native.Native) : ℕ × ℕ × ℕ := extValue x.1

theorem nativeExt_injective : Function.Injective nativeExt := by
  rintro ⟨a, u⟩ ⟨b, v⟩ h
  have hab : a = b := extValue_injective h
  rw [hab]

/-- A run on `nativeB` that returns extension values, as their words. -/
def nativeExts (x : alloc.vec.Vec field.Ext × native.Native) : List (ℕ × ℕ × ℕ) := x.1.val.map extValue

theorem nativeExts_injective : Function.Injective nativeExts := by
  rintro ⟨a, u⟩ ⟨b, v⟩ h
  have hab : a = b := vec_val_injective ((List.map_injective_iff.mpr extValue_injective) h)
  rw [hab]

theorem eq_at_native_sound : Sound (fun x : native.Native × Slice field.Ext × Usize => algebra.eq_at nativeB x.1 x.2.1 x.2.2)
    nativeExt (fun x => cubicWords (Lib.eqAt (x.2.1.val.map cubicOf) x.2.2.val)) :=
  sound_of_eq fun x =>
    let ⟨_, h, hc, hv⟩ := eq_at_native x.1 x.2.1 x.2.2
    ⟨_, h, by rw [nativeExt, ← cubicWords_cubicOf hc, hv]⟩

theorem eq_at_native_complete : Complete (fun _ => True)
    (fun x : native.Native × Slice field.Ext × Usize => algebra.eq_at nativeB x.1 x.2.1 x.2.2)
    nativeExt (fun x => cubicWords (Lib.eqAt (x.2.1.val.map cubicOf) x.2.2.val)) :=
  complete_of_eq eq_at_native_sound ((), emptySlice, 0#usize) trivial fun x _ =>
    let ⟨_, h, hc, hv⟩ := eq_at_native x.1 x.2.1 x.2.2
    ⟨_, h, by rw [nativeExt, ← cubicWords_cubicOf hc, hv]⟩

theorem eq_eval_native_sound : Sound
    (fun x : native.Native × Slice field.Ext × Slice field.Ext => algebra.eq_eval nativeB x.1 x.2.1 x.2.2)
    nativeExt (fun x => cubicWords (Lib.eqEval (x.2.1.val.map cubicOf) (x.2.2.val.map cubicOf))) :=
  sound_of_eq fun x =>
    let ⟨_, h, hc, hv⟩ := eq_eval_native x.1 x.2.1 x.2.2
    ⟨_, h, by rw [nativeExt, ← cubicWords_cubicOf hc, hv]⟩

theorem eq_eval_native_complete : Complete (fun _ => True)
    (fun x : native.Native × Slice field.Ext × Slice field.Ext => algebra.eq_eval nativeB x.1 x.2.1 x.2.2)
    nativeExt (fun x => cubicWords (Lib.eqEval (x.2.1.val.map cubicOf) (x.2.2.val.map cubicOf))) :=
  complete_of_eq eq_eval_native_sound ((), emptySlice, emptySlice) trivial fun x _ =>
    let ⟨_, h, hc, hv⟩ := eq_eval_native x.1 x.2.1 x.2.2
    ⟨_, h, by rw [nativeExt, ← cubicWords_cubicOf hc, hv]⟩

theorem eq_table_native_sound : Sound (fun x : native.Native × Slice field.Ext => algebra.eq_table nativeB x.1 x.2)
    nativeExts (fun x => (List.range (2 ^ x.2.val.length)).map fun i => cubicWords (Lib.eqAt (x.2.val.map cubicOf) i)) := by
  intro x y h
  obtain ⟨hv, hc⟩ := (eq_table_native x.1 x.2).1 y h
  show y.1.val.map extValue = (List.range (2 ^ x.2.val.length)).map (cubicWords ∘ Lib.eqAt (x.2.val.map cubicOf))
  rw [← List.map_map, ← hv, List.map_map]
  exact List.map_congr_left fun e he => (cubicWords_cubicOf (hc e he)).symm

/-- The verifier's tables have at most `2^11` entries: halves of `rx` and `ry`, and block strides. -/
theorem eq_table_native_complete : Complete (fun x : native.Native × Slice field.Ext => 2 ^ x.2.val.length ≤ Usize.max)
    (fun x : native.Native × Slice field.Ext => algebra.eq_table nativeB x.1 x.2)
    nativeExts (fun x => (List.range (2 ^ x.2.val.length)).map fun i => cubicWords (Lib.eqAt (x.2.val.map cubicOf) i)) :=
  complete_of eq_table_native_sound ⟨((), emptySlice), by simp [emptySlice]; scalar_tac⟩
    fun x hx => (eq_table_native x.1 x.2).2 hx

end CompressionLeanRust
