import CompressionLeanRust.Cube
import CompressionLean.Game.Path
import CompressionLean.Game.SumCheck

/-! The translated `round_claim` and `replay_claim` of `neo_verified::sumcheck`, which the shrink
verifier runs on its sum-check rounds, equal their copies over `Cubic`.

`round_claim` is the round polynomial of `Lib.roundValues` at the challenge: `interpolate` on `Native`
is the Lagrange sum over the nodes `0, …, d + 1`. `replay_claim` takes its challenges as input; its
copy is where `Game.sumcheck` ends (`gameEnd`) when the prover sends the rounds and the draws are the
challenges. The verifier draws the challenges from the messages alone, so drawing them first changes
no draw.

A copy reads an input `Ext` as `cubicOf` reads it and gives its value as `cubicWords`. With no round,
`replay_claim` returns its claim with canonical words. A message of fewer values than `degree`, an
empty message, or fewer challenges than rounds makes Rust panic, so `Sound` holds there with nothing to
show; `Complete` assumes none of these, as the verifier's shape check ensures. -/

namespace CompressionLeanRust

open Aeneas Aeneas.Std Result
open neo_verified
open CompressionLean
open NightstreamFPrime.Spec (goldilocksModulus)
open Polynomial

local notation "ι" => algebraMap Gl Cubic

/-! ## `interpolate` on `Native` -/

theorem glOf_cast (j : Usize) : glOf (UScalar.cast .U64 j) = (j.val : Gl) := by
  unfold glOf
  congr 1
  rw [UScalar.cast_val_eq]
  simp only [UScalarTy.U64_numBits_eq]
  apply Nat.mod_eq_of_lt
  calc j.val < 2 ^ UScalarTy.Usize.numBits := j.hBounds
    _ ≤ 2 ^ 64 := Nat.pow_le_pow_right (by omega) numBits_le

/-- The node `j` as `Native` holds it. -/
def nodeW (j : Usize) : field.Ext := constW ⟨redW (redW (UScalar.cast .U64 j)), field.GL_ZERO, field.GL_ZERO⟩

theorem cubicOf_nodeW (j : Usize) : cubicOf (nodeW j) = (j.val : Cubic) := by
  rw [nodeW, cubicOf_constW, cubicOf_ofGl, glOf_red, glOf_cast, map_natCast]

/-- The numerator of node `i`'s weight after the nodes below `k`. -/
noncomputable def numerator (i k : ℕ) (x : Cubic) : Cubic := ∏ j ∈ Finset.range k, if i = j then 1 else x - j

/-- The denominator of node `i`'s weight after the nodes below `k`, in `Gl`. -/
noncomputable def denominator (i k : ℕ) : Gl := ∏ j ∈ Finset.range k, if i = j then 1 else (i : Gl) - j

theorem interpolate_inner_native (x : field.Ext) (n i : Usize) (b : native.Native) :
    ∃ y, algebra.interpolate_loop0_loop0 nativeB b x n i (constW field.EXT_ONE) field.GL_ONE 0#usize = ok y ∧
      cubicOf y.2.1 = numerator i.val n.val (cubicOf x) ∧ glOf y.2.2 = denominator i.val n.val := by
  unfold algebra.interpolate_loop0_loop0
  obtain ⟨y, hy, hq⟩ := loop_total
    (fun a : native.Native × field.Ext × field.Gl × Usize =>
      algebra.interpolate_loop0_loop0.body nativeB x n i a.1 a.2.1 a.2.2.1 a.2.2.2)
    (fun m a => a.2.2.2.val ≤ n.val ∧ m = n.val - a.2.2.2.val ∧
      cubicOf a.2.1 = numerator i.val a.2.2.2.val (cubicOf x) ∧ glOf a.2.2.1 = denominator i.val a.2.2.2.val)
    (fun y : native.Native × field.Ext × field.Gl =>
      cubicOf y.2.1 = numerator i.val n.val (cubicOf x) ∧ glOf y.2.2 = denominator i.val n.val)
    (by
      rintro m ⟨b, num, den, j⟩ ⟨hj, rfl, hnum, hden⟩
      dsimp only at hj hnum hden ⊢
      by_cases hlt : j.val < n.val
      · right
        have hmax : j.val < Usize.max := by scalar_tac
        by_cases e : i.val = j.val
        · have hij : (i != j) = false := by
            simp only [bne_eq_false_iff_eq]
            exact UScalar.val_eq_imp i j e
          refine ⟨(b, num, den, core.num.Usize.wrapping_add j 1#usize), n.val - (j.val + 1), ?_, by omega,
            by rw [add_one_val j hmax]; omega, by rw [add_one_val j hmax], ?_, ?_⟩
          · simp only [algebra.interpolate_loop0_loop0.body, show j < n by scalar_tac, ite_true, hij,
              Bool.false_eq_true, ite_false, bind_ok, uncurry_eq, lift]
          · rw [add_one_val j hmax, numerator, Finset.prod_range_succ, ← numerator, ← hnum, ite_eq_left e, mul_one]
          · rw [add_one_val j hmax, denominator, Finset.prod_range_succ, ← denominator, ← hden, ite_eq_left e, mul_one]
        · have hij : (i != j) = true := by
            simp only [bne_iff_ne, ne_eq]
            intro h'
            exact e (congrArg UScalar.val h')
          refine ⟨(b, extMul num (extSub x (nodeW j)),
            glMul den (glSub (redW (UScalar.cast .U64 i)) (redW (UScalar.cast .U64 j))),
            core.num.Usize.wrapping_add j 1#usize), n.val - (j.val + 1), ?_, by omega,
            by rw [add_one_val j hmax]; omega, by rw [add_one_val j hmax], ?_, ?_⟩
          · simp only [algebra.interpolate_loop0_loop0.body, show j < n by scalar_tac, ite_true, hij, lift, bind_ok,
              reduce_eq, ext_of_gl_eq, ext_constant_native, uncurry_eq, native_ext_sub, native_ext_mul, gl_sub_eq,
              gl_mul_eq, nodeW]
          · rw [add_one_val j hmax, numerator, Finset.prod_range_succ, ← numerator, ← hnum, ite_eq_right e, extMul_cubic,
              extSub_cubic, cubicOf_nodeW]
          · rw [add_one_val j hmax, denominator, Finset.prod_range_succ, ← denominator, ← hden, ite_eq_right e, glMul_field,
              glSub_field, glOf_red, glOf_red, glOf_cast, glOf_cast]
      · left
        have hend : j.val = n.val := by omega
        refine ⟨(b, num, den), ?_, ?_, ?_⟩
        · simp only [algebra.interpolate_loop0_loop0.body, show ¬ j < n by scalar_tac, ite_false]
        · rw [hnum, hend]
        · rw [hden, hend])
    _ (b, constW field.EXT_ONE, field.GL_ONE, 0#usize)
    ⟨by simp, rfl, by simp [numerator, cubicOf_constW, cubicOf_one], by simp [denominator, glOf_one]⟩
  exact ⟨y, hy, hq⟩


theorem ext_scale_constant_native (u : native.Native) (a : field.Ext) (g : field.Gl) :
    algebra.ext_scale_constant nativeB u a g = ok (extScale a (redW (redW g)), u) := by
  simp only [algebra.ext_scale_constant, algebra.constant, gl_word_eq, native_constant, bind_ok, uncurry_eq,
    native.Native.Insts.Neo_verifiedBackendBackendGlExt.ext_scale, ext_scale_eq]

/-- The weight of node `i` among the first `n` at `x`. -/
noncomputable def weight (n i : ℕ) (x : Cubic) : Cubic :=
  numerator i n x * (∏ j ∈ Finset.range n, if i = j then (1 : Cubic) else (i : Cubic) - j)⁻¹

theorem ι_denominator (i n : ℕ) : ι (denominator i n) = ∏ j ∈ Finset.range n, if i = j then (1 : Cubic) else (i : Cubic) - j := by
  rw [denominator, map_prod]
  apply Finset.prod_congr rfl
  intro j _
  split <;> simp

/-- `interpolate` on `Native`: the sum of each value times its node's weight at `x`. -/
theorem interpolate_native (u : native.Native) (values : Slice field.Ext) (x : field.Ext) :
    ∃ e, algebra.interpolate nativeB u values x = ok (e, u) ∧ Canon e ∧
      cubicOf e = ∑ k ∈ Finset.range values.val.length,
        (values.val.map cubicOf).getD k 0 * weight values.val.length k (cubicOf x) := by
  have hlen := values.property
  have hn : (Slice.len values).val = values.val.length := Slice.len_val values
  simp only [algebra.interpolate, ext_zero_native, bind_ok, uncurry_eq]
  unfold algebra.interpolate_loop0
  obtain ⟨y, hy, hq⟩ := loop_total
    (fun a : native.Native × field.Ext × Usize =>
      algebra.interpolate_loop0.body nativeB values x (Slice.len values) a.1 a.2.1 a.2.2)
    (fun m a => a.2.2.val ≤ values.val.length ∧ m = values.val.length - a.2.2.val ∧ Canon a.2.1 ∧
      cubicOf a.2.1 = ∑ k ∈ Finset.range a.2.2.val,
        (values.val.map cubicOf).getD k 0 * weight values.val.length k (cubicOf x))
    (fun y : field.Ext × native.Native => Canon y.1 ∧ cubicOf y.1 = ∑ k ∈ Finset.range values.val.length,
        (values.val.map cubicOf).getD k 0 * weight values.val.length k (cubicOf x))
    (by
      rintro m ⟨b, total, i⟩ ⟨hi, rfl, hc, hv⟩
      dsimp only at hi hc hv ⊢
      by_cases hlt : i.val < values.val.length
      · right
        have hmax : i.val < Usize.max := by scalar_tac
        obtain ⟨⟨b2, num, den⟩, hin, hnum, hden⟩ := interpolate_inner_native x (Slice.len values) i b
        obtain ⟨g, hg, -⟩ := gl_inv_exact den
        have hgf := gl_inv_field den g hg
        refine ⟨(b2, extAdd total (extMul values.val[i.val] (extScale num (redW (redW g)))),
          core.num.Usize.wrapping_add i 1#usize), values.val.length - (i.val + 1), ?_, by omega,
          by rw [add_one_val i hmax]; omega, by rw [add_one_val i hmax], canon_extAdd _ _, ?_⟩
        · simp only [algebra.interpolate_loop0.body, show i < Slice.len values by scalar_tac, ite_true,
            ext_one_native, bind_ok, uncurry_eq, hin, hg, ext_scale_constant_native, slice_index values i hlt,
            native_ext_mul, native_ext_add, lift]
        · rw [add_one_val i hmax, Finset.sum_range_succ, ← hv, extAdd_cubic, extMul_cubic, extScale_cubic,
            glOf_red, glOf_red, hgf, hnum, hden, hn, map_inv₀, ι_denominator, weight, List.getD_eq_getElem _ _ (by simpa using hlt),
            List.getElem_map]
      · left
        have hend : i.val = values.val.length := by omega
        refine ⟨(total, b), ?_, hc, ?_⟩
        · simp only [algebra.interpolate_loop0.body, show ¬ i < Slice.len values by scalar_tac, ite_false]
        · rw [hv, hend])
    _ (u, constW field.EXT_ZERO, 0#usize)
    ⟨by simp, rfl, canon_constW _, by simp [cubicOf_constW, cubicOf_zero]⟩
  exact ⟨y.1, hy, hq⟩


/-! ## `round_claim` -/

theorem slice_index_fail {α : Type} (d : Slice α) (t : Usize) (ht : d.val.length ≤ t.val) :
    Slice.index_usize d t = fail .arrayOutOfBounds := by
  unfold Slice.index_usize
  rw [Slice.getElem?_Usize_eq, List.getElem?_eq_none ht]

/-- `Bridge.interpolate_eval_range`, which this library does not import. -/
theorem lagrange_eval_range {L : Type} [Field L] (n : ℕ) (r : Fin n → L) (c : L) :
    (Lagrange.interpolate Finset.univ (fun i : Fin n => ((i : ℕ) : L)) r).eval c =
      ∑ i : Fin n, r i * ((∏ j ∈ Finset.range n, if (i : ℕ) = j then 1 else c - j) *
        (∏ j ∈ Finset.range n, if (i : ℕ) = j then 1 else ((i : ℕ) : L) - j)⁻¹) := by
  have erase (i : Fin n) (f : ℕ → L) :
      ∏ j ∈ (Finset.univ : Finset (Fin n)).erase i, f j = ∏ j ∈ Finset.range n, if (i : ℕ) = j then 1 else f j := by
    rw [← Fin.prod_univ_eq_prod_range (fun j => if (i : ℕ) = j then 1 else f j) n,
      ← Finset.prod_erase_mul _ _ (Finset.mem_univ i), ite_eq_left rfl, mul_one]
    apply Finset.prod_congr rfl
    intro j member
    rw [ite_eq_right fun same => (Finset.mem_erase.mp member).1 (Fin.ext same.symm)]
  rw [Lagrange.interpolate_apply, Polynomial.eval_finsetSum]
  apply Finset.sum_congr rfl
  intro i _
  rw [eval_mul, eval_C, Lagrange.basis, Polynomial.eval_prod]
  congr 1
  simp only [Lagrange.basisDivisor, eval_mul, eval_C, eval_sub, eval_X]
  rw [Finset.prod_mul_distrib, Finset.prod_inv_distrib, mul_comm, erase i (fun j => c - j),
    erase i (fun j => ((i : ℕ) : L) - j)]

/-- A round message as the copy's `d + 1` values. -/
noncomputable def messageOf (d : ℕ) (m : List field.Ext) : Fin (d + 1) → Cubic := fun j => (m.map cubicOf).getD j 0

theorem take_drop_one_succ {α : Type} (l : List α) (j : ℕ) (h1 : 1 ≤ j) (hj : j < l.length) :
    (l.drop 1).take j = (l.drop 1).take (j - 1) ++ [l[j]] := by
  obtain ⟨k, rfl⟩ : ∃ k, j = k + 1 := ⟨j - 1, by omega⟩
  rw [List.take_add_one, Nat.add_sub_cancel, List.getElem?_drop, List.getElem?_eq_getElem (by omega)]
  simp [Nat.add_comm]

theorem round_values_loop (degree : Usize) (message : Slice field.Ext) (base : alloc.vec.Vec field.Ext)
    (hone : 1 ≤ message.val.length) :
    (∀ v, sumcheck.round_claim_loop degree message base 1#usize = ok v →
      v.val = base.val ++ (message.val.drop 1).take (degree.val - 1) ∧
        (degree.val ≤ message.val.length ∨ degree.val ≤ 1)) ∧
    (degree.val ≤ message.val.length → base.val.length + degree.val ≤ Usize.max + 1 →
      ∃ v, sumcheck.round_claim_loop degree message base 1#usize = ok v) := by
  have hlen := message.property
  unfold sumcheck.round_claim_loop
  have key := loop_ok
    (fun a : alloc.vec.Vec field.Ext × Usize => sumcheck.round_claim_loop.body degree message a.1 a.2)
    (fun m a => 1 ≤ a.2.val ∧ (a.2.val ≤ degree.val ∨ a.2.val = 1) ∧ a.2.val ≤ message.val.length ∧
      m = degree.val - a.2.val ∧ a.1.val = base.val ++ (message.val.drop 1).take (a.2.val - 1))
    (fun v : alloc.vec.Vec field.Ext =>
      v.val = base.val ++ (message.val.drop 1).take (degree.val - 1) ∧
        (degree.val ≤ message.val.length ∨ degree.val ≤ 1))
    (fun _ => degree.val ≤ message.val.length ∧ base.val.length + degree.val ≤ Usize.max + 1)
    (by
      rintro m ⟨vs, j⟩ c ⟨h1, hj, hjm, rfl, hv⟩ hbody
      dsimp only at h1 hj hjm hv hbody ⊢
      by_cases hlt : j.val < degree.val
      · right
        have hmax : j.val < Usize.max := by scalar_tac
        by_cases hin : j.val < message.val.length
        · simp only [sumcheck.round_claim_loop.body, show j < degree by scalar_tac, ite_true, slice_index message j hin,
            bind_ok, lift] at hbody
          obtain ⟨w, hw, hbody⟩ := bind_eq_ok hbody
          rw [Result.ok.injEq] at hbody
          subst hbody
          refine ⟨(w, core.num.Usize.wrapping_add j 1#usize), degree.val - (j.val + 1), rfl, by omega, ?_, ?_, ?_,
            by rw [add_one_val j hmax], ?_⟩
          · rw [add_one_val j hmax]; omega
          · rw [add_one_val j hmax]; omega
          · rw [add_one_val j hmax]; omega
          · rw [vec_push_val hw, hv, add_one_val j hmax, Nat.add_sub_cancel,
              take_drop_one_succ message.val j.val h1 hin, List.append_assoc]
        · simp only [sumcheck.round_claim_loop.body, show j < degree by scalar_tac, ite_true,
            slice_index_fail message j (by omega)] at hbody
          exact absurd hbody (by simp)
      · left
        simp only [sumcheck.round_claim_loop.body, show ¬ j < degree by scalar_tac, ite_false,
          Result.ok.injEq] at hbody
        subst hbody
        refine ⟨_, rfl, ?_, by omega⟩
        rw [hv, show j.val - 1 = degree.val - 1 by omega])
    (by
      rintro m ⟨vs, j⟩ ⟨h1, hj, hjm, rfl, hv⟩ ⟨hd, hb⟩
      dsimp only at h1 hj hjm hv ⊢
      by_cases hlt : j.val < degree.val
      · have hin : j.val < message.val.length := by omega
        have hpush : vs.val.length < Usize.max := by
          rw [hv]; simp; omega
        obtain ⟨w, hw, -⟩ := vec_push_ok vs message.val[j.val] hpush
        exact ⟨ControlFlow.cont (w, core.num.Usize.wrapping_add j 1#usize), by
          simp only [sumcheck.round_claim_loop.body, show j < degree by scalar_tac, ite_true,
            slice_index message j hin, bind_ok, lift, hw]⟩
      · exact ⟨ControlFlow.done vs, by
          simp only [sumcheck.round_claim_loop.body, show ¬ j < degree by scalar_tac, ite_false]⟩)
    (fun _ _ _ _ hP _ => hP) _ (base, 1#usize) ⟨by simp, by simp, by simpa using hone, rfl, by simp⟩
  exact ⟨key.1, fun hd hb => let ⟨v, hv, _⟩ := key.2 ⟨hd, hb⟩; ⟨v, hv⟩⟩


/-- One round of the copy: the round polynomial of `Lib.roundValues` at `r`. -/
noncomputable def roundAt (d : ℕ) (claim : Cubic) (message : Fin (d + 1) → Cubic) (r : Cubic) : Cubic :=
  (Lagrange.interpolate Finset.univ (Lib.node d) (Lib.roundValues d claim message)).eval r

theorem roundList_get (d : ℕ) (msg : List field.Ext) (claim : field.Ext) (h1 : 1 ≤ msg.length)
    (hd : d + 1 ≤ msg.length ∨ d = 0) (k : Fin (d + 2)) :
    (([msg[0], extSub claim msg[0]] ++ (msg.drop 1).take d).map cubicOf).getD k 0 =
      Lib.roundValues d (cubicOf claim) (messageOf d msg) k := by
  obtain ⟨k, hk⟩ := k
  rcases (show k = 0 ∨ k = 1 ∨ 2 ≤ k by omega) with rfl | rfl | two
  · simp [Lib.roundValues, messageOf, List.getD_eq_getElem?_getD, List.getElem?_map, List.getElem?_eq_getElem h1]
  · simp [Lib.roundValues, messageOf, List.getD_eq_getElem?_getD, List.getElem?_map, List.getElem?_eq_getElem h1,
      extSub_cubic]
  · have hk1 : k - 1 < msg.length := by omega
    simp only [Lib.roundValues]
    rw [dite_eq_right (by omega), ite_eq_right (by omega)]
    simp only [messageOf]
    rw [List.map_append, List.getD_eq_getElem?_getD, List.getElem?_append_right (by simp; omega),
      List.getElem?_map, List.getElem?_take, ite_eq_left (by simp; omega), List.getElem?_drop,
      List.getD_eq_getElem?_getD, List.getElem?_map]
    simp only [List.length_map, List.length_cons, List.length_nil]
    rw [show 1 + (k - (0 + 1 + 1)) = k - 1 by omega]

theorem roundAt_sum (d : ℕ) (claim : Cubic) (message : Fin (d + 1) → Cubic) (r : Cubic) :
    roundAt d claim message r =
      ∑ k ∈ Finset.range (d + 2), (fun k => if h : k < d + 2 then Lib.roundValues d claim message ⟨k, h⟩ else 0) k *
        weight (d + 2) k r := by
  rw [roundAt, show Lib.node (K := Cubic) d = fun i : Fin (d + 2) => ((i : ℕ) : Cubic) from rfl, lagrange_eval_range,
    ← Fin.sum_univ_eq_sum_range]
  apply Finset.sum_congr rfl
  intro i _
  simp [weight, numerator, i.isLt]


theorem zero_usize_val : (0#usize).val = 0 := by simp

theorem slice_index0 {α : Type} (d : Slice α) (h : 0 < d.val.length) : Slice.index_usize d 0#usize = ok d.val[0] :=
  slice_index d 0#usize h

theorem usize_max_ge : 4294967295 ≤ Usize.max := by
  rw [Usize.max_def]
  rcases System.Platform.numBits_eq with h | h <;> simp [Usize.numBits, h]

theorem round_claim_spec (degree : Usize) (message : Slice field.Ext) (claim r : field.Ext) :
    (∀ e, sumcheck.round_claim degree message claim r = ok e → Canon e ∧
      cubicOf e = roundAt (degree.val - 1) (cubicOf claim) (messageOf (degree.val - 1) message.val) (cubicOf r)) ∧
    (1 ≤ message.val.length → degree.val ≤ message.val.length → degree.val < Usize.max →
      ∃ e, sumcheck.round_claim degree message claim r = ok e) := by
  have hlen := message.property
  constructor
  · intro e h
    unfold sumcheck.round_claim at h
    by_cases h1 : 1 ≤ message.val.length
    · rw [slice_index0 message (by omega), bind_ok] at h
      obtain ⟨v1, hv1, h⟩ := bind_eq_ok h
      rw [ext_sub_eq, bind_ok] at h
      obtain ⟨v2, hv2, h⟩ := bind_eq_ok h
      obtain ⟨v3, hv3, h⟩ := bind_eq_ok h
      obtain ⟨⟨e2, u⟩, hint, h⟩ := bind_eq_ok h
      have h : ok e2 = ok e := h
      rw [Result.ok.injEq] at h
      subst h
      have e1 := vec_push_val hv1
      have e2 := vec_push_val hv2
      obtain ⟨e3, hdeg⟩ := (round_values_loop degree message v2 h1).1 v3 hv3
      obtain ⟨e', he', hc, hv⟩ := interpolate_native () (alloc.vec.Vec.deref v3) r
      have hder : (alloc.vec.Vec.deref v3).val = v3.val := by simp [alloc.vec.Vec.deref]
      rw [hder] at hv
      rw [hint, Result.ok.injEq, Prod.mk.injEq] at he'
      obtain ⟨rfl, -⟩ := he'
      refine ⟨hc, ?_⟩
      have hval : v3.val = [message.val[0], extSub claim message.val[0]] ++ (message.val.drop 1).take (degree.val - 1) := by
        rw [e3, e2, e1]
        rfl
      have hn : v3.val.length = degree.val - 1 + 2 := by
        rw [hval]
        simp only [List.length_append, List.length_cons, List.length_nil, List.length_take, List.length_drop]
        omega
      rw [hv, roundAt_sum, hn]
      apply Finset.sum_congr rfl
      intro k hk
      rw [Finset.mem_range] at hk
      simp only [hk, ↓reduceDIte]
      rw [hval, ← roundList_get (degree.val - 1) message.val claim h1 (by omega) ⟨k, hk⟩]
    · rw [slice_index_fail message 0#usize (by rw [zero_usize_val]; omega)] at h
      exact absurd h (by simp)
  · intro h1 hd hmax
    have hm := usize_max_ge
    obtain ⟨v1, hv1, e1⟩ := vec_push_ok (alloc.vec.Vec.new field.Ext) message.val[0]
      (lt_of_lt_of_le (by norm_num) hm)
    obtain ⟨v2, hv2, e2⟩ := vec_push_ok v1 (extSub claim message.val[0]) (by rw [e1]; simp [alloc.vec.Vec.new]; omega)
    obtain ⟨v3, hv3⟩ := (round_values_loop degree message v2 h1).2 hd (by rw [e2, e1]; simp [alloc.vec.Vec.new]; omega)
    obtain ⟨e', he', -⟩ := interpolate_native () (alloc.vec.Vec.deref v3) r
    refine ⟨e', ?_⟩
    unfold sumcheck.round_claim
    rw [slice_index0 message (by omega), bind_ok, hv1, bind_ok, ext_sub_eq, bind_ok, hv2, bind_ok, hv3,
      bind_ok]
    show (do let (e2, _) ← algebra.interpolate nativeB () (alloc.vec.Vec.deref v3) r; ok e2) = ok e'
    rw [he', bind_ok]
    rfl


/-! ## `replay_claim` and `Game.sumcheck` -/

theorem reaches_unique {α : Type} {g : Game α} {ms ds : List (Σ C : Type, C)} {a b : α}
    (ha : Game.Reaches g ms ds a) (hb : Game.Reaches g ms ds b) : a = b := by
  obtain ⟨p, hp, hd, hm⟩ := ha
  obtain ⟨q, hq, hd', hm'⟩ := hb
  obtain rfl := Game.Pos.eq_of_msgs_draws (hm.trans hm'.symm) (hd.trans hd'.symm)
  rw [hp] at hq
  cases hq
  rfl

open Classical in
/-- Where `g` ends when the prover sends `ms` and the draws are `ds`, if that play ends. -/
noncomputable def gameEnd {α : Type} (g : Game α) (ms ds : List (Σ C : Type, C)) : Option α :=
  if h : ∃ a, Game.Reaches g ms ds a then some h.choose else none

theorem gameEnd_eq {α : Type} {g : Game α} {ms ds : List (Σ C : Type, C)} {a : α}
    (h : Game.Reaches g ms ds a) : gameEnd g ms ds = some a := by
  unfold gameEnd
  split
  · next h' => exact congrArg some (reaches_unique h'.choose_spec h)
  · next h' => exact absurd ⟨a, h⟩ h'

/-- The claim after rounds `rs`, each a message and its draw. -/
noncomputable def claimAfter (d : ℕ) (rs : List ((Fin (d + 1) → Cubic) × Cubic)) (claim : Cubic) : Cubic :=
  rs.foldl (fun c p => roundAt d c p.1 p.2) claim

theorem sumcheck_path {W : Type} (worlds : Finset W) (d : ℕ) (round : W → List Cubic → Cubic[X])
    (extra : List Cubic → Finset Cubic) :
    ∀ (rs : List ((Fin (d + 1) → Cubic) × Cubic)) (point : List Cubic) (claim : Cubic),
      Game.Reaches (Game.sumcheck worlds d round extra rs.length point claim)
        (rs.map fun p => ⟨Fin (d + 1) → Cubic, p.1⟩) (rs.map fun p => ⟨Cubic, p.2⟩)
        (point ++ rs.map Prod.snd, claimAfter d rs claim)
  | [], point, claim => by
      simpa [claimAfter, Game.sumcheck] using Game.Reaches.pure (point, claim)
  | (m, c) :: rest, point, claim => by
      rw [List.length_cons, Game.sumcheck, List.map_cons, List.map_cons]
      apply Game.Reaches.recv m
      apply Game.Reaches.draw c
      have h := sumcheck_path worlds d round extra rest (point ++ [c]) (roundAt d claim m c)
      rw [List.append_assoc, List.singleton_append] at h
      exact h

/-- `a` with canonical words, as `ext_reduce` returns it. -/
def extRed (a : field.Ext) : field.Ext := ⟨redW a.c0, redW a.c1, redW a.c2⟩

theorem ext_reduce_eq (a : field.Ext) : field.ext_reduce a = ok (extRed a) := by
  simp only [field.ext_reduce, reduce_word_eq, bind_ok]
  rfl

theorem canon_extRed (a : field.Ext) : Canon (extRed a) := ⟨redW_lt _, redW_lt _, redW_lt _⟩

theorem cubicOf_extRed (a : field.Ext) : cubicOf (extRed a) = cubicOf a := by
  simp only [cubicOf, extRed, glOf_red]

/-- Round `i` of `rounds` with its draw, as the copy reads them. -/
noncomputable def roundPairs (d : ℕ) (rounds : List (alloc.vec.Vec field.Ext)) (point : List field.Ext) :
    List ((Fin (d + 1) → Cubic) × Cubic) :=
  (rounds.zip point).map fun p => (messageOf d p.1.val, cubicOf p.2)

theorem roundPairs_take_succ (d : ℕ) (rounds : List (alloc.vec.Vec field.Ext)) (point : List field.Ext) (i : ℕ)
    (h1 : i < rounds.length) (h2 : i < point.length) :
    (roundPairs d rounds point).take (i + 1) =
      (roundPairs d rounds point).take i ++ [(messageOf d rounds[i].val, cubicOf point[i])] := by
  rw [List.take_add_one]
  congr 1
  have hz : (rounds.zip point)[i]? = some (rounds[i], point[i]) := by
    rw [List.getElem?_eq_getElem (by simp; omega), List.getElem_zip]
  simp [roundPairs, hz]

theorem claimAfter_snoc (d : ℕ) (rs : List ((Fin (d + 1) → Cubic) × Cubic)) (p : (Fin (d + 1) → Cubic) × Cubic)
    (claim : Cubic) : claimAfter d (rs ++ [p]) claim = roundAt d (claimAfter d rs claim) p.1 p.2 := by
  simp [claimAfter, List.foldl_append]

theorem deref_val {α : Type} (v : alloc.vec.Vec α) : (alloc.vec.Vec.deref v).val = v.val := by
  simp [alloc.vec.Vec.deref]

theorem replay_claim_loop_spec (degree : Usize) (rounds : Slice (alloc.vec.Vec field.Ext)) (point : Slice field.Ext)
    (claim : field.Ext) (hc : Canon claim) :
    (∀ y, sumcheck.replay_claim_loop degree rounds point claim 0#usize = ok y →
      rounds.val.length ≤ point.val.length ∧ Canon y ∧
        cubicOf y = claimAfter (degree.val - 1) (roundPairs (degree.val - 1) rounds.val point.val) (cubicOf claim)) ∧
    (rounds.val.length ≤ point.val.length → (∀ m ∈ rounds.val, 1 ≤ m.val.length ∧ degree.val ≤ m.val.length) →
      degree.val < Usize.max → ∃ y, sumcheck.replay_claim_loop degree rounds point claim 0#usize = ok y) := by
  have hlen := rounds.property
  unfold sumcheck.replay_claim_loop
  have key := loop_ok
    (fun a : field.Ext × Usize => sumcheck.replay_claim_loop.body degree rounds point a.1 a.2)
    (fun m a => a.2.val ≤ rounds.val.length ∧ a.2.val ≤ point.val.length ∧ m = rounds.val.length - a.2.val ∧
      Canon a.1 ∧ cubicOf a.1 = claimAfter (degree.val - 1)
        ((roundPairs (degree.val - 1) rounds.val point.val).take a.2.val) (cubicOf claim))
    (fun y : field.Ext => rounds.val.length ≤ point.val.length ∧ Canon y ∧
      cubicOf y = claimAfter (degree.val - 1) (roundPairs (degree.val - 1) rounds.val point.val) (cubicOf claim))
    (fun _ => rounds.val.length ≤ point.val.length ∧ (∀ m ∈ rounds.val, 1 ≤ m.val.length ∧ degree.val ≤ m.val.length) ∧
      degree.val < Usize.max)
    (by
      rintro m ⟨cl, i⟩ c ⟨hi, hip, rfl, hcl, hv⟩ hbody
      dsimp only at hi hip hcl hv hbody ⊢
      by_cases hlt : i.val < rounds.val.length
      · right
        have hmax : i.val < Usize.max := by scalar_tac
        by_cases hpt : i.val < point.val.length
        · simp only [sumcheck.replay_claim_loop.body, Slice.len,
            show i < Usize.ofNatCore rounds.val.length _ by scalar_tac, ite_true, slice_index rounds i hlt,
            slice_index point i hpt, bind_ok, lift] at hbody
          obtain ⟨cl1, h1, hbody⟩ := bind_eq_ok hbody
          rw [Result.ok.injEq] at hbody
          subst hbody
          obtain ⟨hc1, hv1⟩ := (round_claim_spec degree _ cl point.val[i.val]).1 cl1 h1
          refine ⟨(cl1, core.num.Usize.wrapping_add i 1#usize), rounds.val.length - (i.val + 1), rfl, by omega, ?_, ?_,
            by rw [add_one_val i hmax], hc1, ?_⟩
          · rw [add_one_val i hmax]; omega
          · rw [add_one_val i hmax]; omega
          · dsimp only
            rw [add_one_val i hmax, roundPairs_take_succ _ rounds.val point.val i.val hlt hpt, claimAfter_snoc, ← hv,
              hv1, deref_val]
        · simp only [sumcheck.replay_claim_loop.body, Slice.len,
            show i < Usize.ofNatCore rounds.val.length _ by scalar_tac, ite_true, slice_index rounds i hlt,
            slice_index_fail point i (by omega), bind_ok] at hbody
          exact absurd hbody (by simp)
      · left
        simp only [sumcheck.replay_claim_loop.body, Slice.len,
          show ¬ i < Usize.ofNatCore rounds.val.length _ by scalar_tac, ite_false, Result.ok.injEq] at hbody
        subst hbody
        have hend : i.val = rounds.val.length := by omega
        refine ⟨_, rfl, by omega, hcl, ?_⟩
        rw [hv, hend, List.take_of_length_le]
        simp [roundPairs])
    (by
      rintro m ⟨cl, i⟩ ⟨hi, hip, rfl, hcl, hv⟩ ⟨hpl, hms, hmax⟩
      dsimp only at hi hip hcl hv ⊢
      by_cases hlt : i.val < rounds.val.length
      · have hpt : i.val < point.val.length := by omega
        obtain ⟨hm1, hmd⟩ := hms _ (List.getElem_mem hlt)
        obtain ⟨cl1, h1⟩ := (round_claim_spec degree (alloc.vec.Vec.deref rounds.val[i.val]) cl point.val[i.val]).2
          (by simpa [alloc.vec.Vec.deref] using hm1) (by simpa [alloc.vec.Vec.deref] using hmd) hmax
        exact ⟨ControlFlow.cont (cl1, core.num.Usize.wrapping_add i 1#usize), by
          simp only [sumcheck.replay_claim_loop.body, Slice.len,
            show i < Usize.ofNatCore rounds.val.length _ by scalar_tac, ite_true, slice_index rounds i hlt,
            slice_index point i hpt, bind_ok, lift, h1]⟩
      · exact ⟨ControlFlow.done cl, by
          simp only [sumcheck.replay_claim_loop.body, Slice.len,
            show ¬ i < Usize.ofNatCore rounds.val.length _ by scalar_tac, ite_false]⟩)
    (fun _ _ _ _ hP _ => hP) _ (claim, 0#usize) ⟨by simp, by simp, rfl, hc, by simp [claimAfter]⟩
  exact ⟨key.1, fun h1 h2 h3 => let ⟨y, hy, _⟩ := key.2 ⟨h1, h2, h3⟩; ⟨y, hy⟩⟩


theorem zip_take_right {α β : Type} : ∀ (l₁ : List α) (l₂ : List β), l₁.zip l₂ = l₁.zip (l₂.take l₁.length)
  | [], _ => rfl
  | _ :: _, [] => rfl
  | _ :: xs, _ :: ys => by simp [zip_take_right xs ys]

theorem replay_claim_spec (degree : Usize) (rounds : Slice (alloc.vec.Vec field.Ext)) (claim : field.Ext)
    (point : Slice field.Ext) (y : field.Ext) (h : sumcheck.replay_claim degree rounds claim point = ok y) :
    rounds.val.length ≤ point.val.length ∧ Canon y ∧
      cubicOf y = claimAfter (degree.val - 1) (roundPairs (degree.val - 1) rounds.val point.val) (cubicOf claim) := by
  unfold sumcheck.replay_claim at h
  rw [ext_reduce_eq, bind_ok] at h
  have := (replay_claim_loop_spec degree rounds point (extRed claim) (canon_extRed claim)).1 y h
  rwa [cubicOf_extRed] at this

/-- The messages that `Game.sumcheck` receives: each round as `d + 1` values. -/
noncomputable def roundMsgs (d : ℕ) (rounds : List (alloc.vec.Vec field.Ext)) : List (Σ C : Type, C) :=
  rounds.map fun m => ⟨Fin (d + 1) → Cubic, messageOf d m.val⟩

/-- The draws of `Game.sumcheck`: one challenge per round. -/
noncomputable def roundDraws (point : List field.Ext) : List (Σ C : Type, C) :=
  point.map fun c => ⟨Cubic, cubicOf c⟩

theorem replay_claim_sound : Sound
    (fun x : Usize × Slice (alloc.vec.Vec field.Ext) × field.Ext × Slice field.Ext =>
      sumcheck.replay_claim x.1 x.2.1 x.2.2.1 x.2.2.2) extValue
    (fun x => ((gameEnd (Game.sumcheck (∅ : Finset Unit) (x.1.val - 1) (fun _ _ => (0 : Cubic[X])) (fun _ => ∅)
        x.2.1.val.length [] (cubicOf x.2.2.1)) (roundMsgs (x.1.val - 1) x.2.1.val)
        (roundDraws (x.2.2.2.val.take x.2.1.val.length))).map fun r => cubicWords r.2).getD (0, 0, 0)) := by
  rintro ⟨degree, rounds, claim, point⟩ y h
  obtain ⟨hle, hc, hv⟩ := replay_claim_spec degree rounds claim point y h
  have hp := sumcheck_path (∅ : Finset Unit) (degree.val - 1) (fun _ _ => (0 : Cubic[X])) (fun _ => ∅)
    (roundPairs (degree.val - 1) rounds.val point.val) [] (cubicOf claim)
  have hlen : (roundPairs (degree.val - 1) rounds.val point.val).length = rounds.val.length := by
    simp [roundPairs]
    omega
  have hm : (roundPairs (degree.val - 1) rounds.val point.val).map
      (fun p => (⟨Fin (degree.val - 1 + 1) → Cubic, p.1⟩ : Σ C : Type, C)) = roundMsgs (degree.val - 1) rounds.val := by
    rw [roundPairs, List.map_map, roundMsgs]
    conv_rhs => rw [← List.map_fst_zip hle]
    rw [List.map_map]
    rfl
  have hd : (roundPairs (degree.val - 1) rounds.val point.val).map (fun p => (⟨Cubic, p.2⟩ : Σ C : Type, C)) =
      roundDraws (point.val.take rounds.val.length) := by
    rw [roundPairs, List.map_map, roundDraws, zip_take_right rounds.val point.val]
    conv_rhs => rw [← List.map_snd_zip (l₁ := rounds.val) (l₂ := point.val.take rounds.val.length) (by simp)]
    rw [List.map_map]
    rfl
  rw [hlen, hm, hd] at hp
  dsimp only
  rw [gameEnd_eq hp, Option.map_some, Option.getD_some, ← cubicWords_cubicOf hc, hv]

/-- The verifier draws one challenge per round and checks each message's length first: `DEGREE`
values in the outer rounds and 2 in the inner ones. -/
theorem replay_claim_complete : Complete
    (fun x : Usize × Slice (alloc.vec.Vec field.Ext) × field.Ext × Slice field.Ext =>
      x.2.1.val.length ≤ x.2.2.2.val.length ∧ (∀ m ∈ x.2.1.val, 1 ≤ m.val.length ∧ x.1.val ≤ m.val.length) ∧
        x.1.val < Usize.max)
    (fun x : Usize × Slice (alloc.vec.Vec field.Ext) × field.Ext × Slice field.Ext =>
      sumcheck.replay_claim x.1 x.2.1 x.2.2.1 x.2.2.2) extValue
    (fun x => ((gameEnd (Game.sumcheck (∅ : Finset Unit) (x.1.val - 1) (fun _ _ => (0 : Cubic[X])) (fun _ => ∅)
        x.2.1.val.length [] (cubicOf x.2.2.1)) (roundMsgs (x.1.val - 1) x.2.1.val)
        (roundDraws (x.2.2.2.val.take x.2.1.val.length))).map fun r => cubicWords r.2).getD (0, 0, 0)) := by
  refine complete_of replay_claim_sound ⟨(0#usize, emptySlice, field.EXT_ZERO, emptySlice), ?_⟩ ?_
  · refine ⟨by simp [emptySlice], by simp [emptySlice], ?_⟩
    have := usize_max_ge
    simp only [zero_usize_val]
    omega
  · rintro ⟨degree, rounds, claim, point⟩ ⟨h1, h2, h3⟩
    unfold sumcheck.replay_claim
    rw [ext_reduce_eq, bind_ok]
    exact (replay_claim_loop_spec degree rounds point (extRed claim) (canon_extRed claim)).2 h1 h2 h3


/-! ## `Sound` and `Complete` of `round_claim` -/

theorem round_claim_sound : Sound
    (fun x : Usize × Slice field.Ext × field.Ext × field.Ext => sumcheck.round_claim x.1 x.2.1 x.2.2.1 x.2.2.2) extValue
    (fun x => cubicWords ((Lagrange.interpolate Finset.univ (Lib.node (x.1.val - 1))
      (Lib.roundValues (x.1.val - 1) (cubicOf x.2.2.1) (messageOf (x.1.val - 1) x.2.1.val))).eval (cubicOf x.2.2.2))) := by
  rintro ⟨degree, message, claim, r⟩ y h
  obtain ⟨hc, hv⟩ := (round_claim_spec degree message claim r).1 y h
  rw [← cubicWords_cubicOf hc, hv]
  rfl

/-- A message of one value. -/
def oneSlice : Slice field.Ext := Slice.from [field.EXT_ZERO] (by have := usize_max_ge; simp; omega)

/-- The verifier checks each message's length first: `DEGREE` values in the outer rounds and 2 in the
inner ones, with `degree` the same number. -/
theorem round_claim_complete : Complete
    (fun x : Usize × Slice field.Ext × field.Ext × field.Ext =>
      1 ≤ x.2.1.val.length ∧ x.1.val ≤ x.2.1.val.length ∧ x.1.val < Usize.max)
    (fun x : Usize × Slice field.Ext × field.Ext × field.Ext => sumcheck.round_claim x.1 x.2.1 x.2.2.1 x.2.2.2) extValue
    (fun x => cubicWords ((Lagrange.interpolate Finset.univ (Lib.node (x.1.val - 1))
      (Lib.roundValues (x.1.val - 1) (cubicOf x.2.2.1) (messageOf (x.1.val - 1) x.2.1.val))).eval (cubicOf x.2.2.2))) := by
  refine complete_of round_claim_sound ⟨(1#usize, oneSlice, field.EXT_ZERO, field.EXT_ZERO), ?_⟩ ?_
  · have := usize_max_ge
    refine ⟨by simp [oneSlice], by simp [oneSlice], ?_⟩
    simp only [show (1#usize).val = 1 by simp]
    omega
  · rintro ⟨degree, message, claim, r⟩ ⟨h1, h2, h3⟩
    exact (round_claim_spec degree message claim r).2 h1 h2 h3

end CompressionLeanRust
