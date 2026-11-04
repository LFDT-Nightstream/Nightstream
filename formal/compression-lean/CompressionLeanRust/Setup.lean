import CompressionLeanRust.Native
import CompressionLeanRust.Algebra
import CompressionLean.Ports.Replay.Stages
import CompressionLean.Ports.Circuit.Honest

/-! The setup leaf check of `neo_verified::setup` and the generator that it reads.

`two_adic_generator` equals `Replay.twoAdicGenerator`, Plonky3's `two_adic_generator`, where it
returns; it panics for more than 32 bits, as Plonky3 does. `path_ok`, run on `nativeB`, equals
`Replay.pathOk`, the path check that `MerkleBinds` reads: it reads its words as `merkle_root_native`
does, the levels it reads are the shorter of its bits and its path, and it rejects with
`"setup leaf path"` exactly where the copy's lanes differ. It always returns, so its `Complete` premise
is `True`. -/

namespace CompressionLeanRust

open Aeneas Aeneas.Std Result
open neo_verified
open CompressionLean
open NightstreamFPrime.Spec (goldilocksModulus)

/-! ## `two_adic_generator` -/

theorem squarings_succ (x : Replay.Word) (n : ℕ) :
    (List.range (n + 1)).foldl (fun g _ => g * g) x = (List.range n).foldl (fun g _ => g * g) (x * x) := by
  rw [List.range_succ_eq_map, List.foldl_cons, List.foldl_map]

theorem two_adic_generator_loop_eq (squarings : Usize) (g : field.Gl) (x : Replay.Word) (hx : Exact g x) :
    ∃ c, field.two_adic_generator_loop squarings g = ok c ∧
      Exact c ((List.range squarings.val).foldl (fun g _ => g * g) x) := by
  unfold field.two_adic_generator_loop
  exact loop_total (fun (a : Usize × field.Gl) => field.two_adic_generator_loop.body a.1 a.2)
    (fun m a => a.1.val = m ∧ ∃ w, Exact a.2 w ∧
      (List.range m).foldl (fun g _ => g * g) w = (List.range squarings.val).foldl (fun g _ => g * g) x)
    (fun c => Exact c ((List.range squarings.val).foldl (fun g _ => g * g) x))
    (by
      rintro m ⟨sq, g⟩ ⟨hm, w, hw, hf⟩
      dsimp only at hm hw hf ⊢
      by_cases hz : m = 0
      · left
        subst hz
        refine ⟨g, by simp [field.two_adic_generator_loop.body, show ¬ sq > 0#usize by scalar_tac], ?_⟩
        rw [← hf]
        exact hw
      · right
        obtain ⟨m', rfl⟩ : ∃ m', m = m' + 1 := ⟨m - 1, by omega⟩
        refine ⟨(core.num.Usize.wrapping_sub sq 1#usize, glMul g g), m', ?_, by omega, ?_, w * w, ?_, ?_⟩
        · simp [field.two_adic_generator_loop.body, show sq > 0#usize by scalar_tac, gl_mul_eq, lift_ok]
        · dsimp only
          rw [wrapping_sub_val _ _ (by simp; omega)]
          scalar_tac
        · rw [Exact, glMul_exact g g, hw.wordOf]
        · rw [← hf, squarings_succ])
    squarings.val (squarings, g) ⟨rfl, x, hx, rfl⟩

theorem generator_exact : Exact 1753635133440165772#u64 (Replay.Word.ofNat 0x185629dcda58878c) := by
  unfold Exact
  rfl

theorem two_adic_generator_eq (bits : Usize) (h : bits.val ≤ 32) :
    ∃ c, field.two_adic_generator bits = ok c ∧ Exact c (Replay.twoAdicGenerator bits.val) := by
  obtain ⟨z, hz, hzv⟩ := checked_sub_some 32#usize bits (by simp; omega)
  obtain ⟨c, hc, hx⟩ := two_adic_generator_loop_eq z 1753635133440165772#u64 _ generator_exact
  refine ⟨c, ?_, ?_⟩
  · simp only [field.two_adic_generator, Usize.checked_sub, lift_ok, bind_ok, hz, unwrap_some, hc]
  · have : z.val = 32 - bits.val := by simpa using hzv
    rw [this] at hx
    exact hx

theorem two_adic_generator_returns (bits : Usize) (c : field.Gl) (h : field.two_adic_generator bits = ok c) :
    bits.val ≤ 32 := by
  by_contra hb
  rw [field.two_adic_generator, Usize.checked_sub, lift_ok, bind_ok,
    checked_sub_none 32#usize bits (by simp; omega), unwrap_none] at h
  simp at h

theorem two_adic_generator_sound : Sound (fun x : Usize => field.two_adic_generator x) glValue
    (fun x => (Replay.twoAdicGenerator x.val).toNat) :=
  sound_of_eq_on (fun x => x.val ≤ 32) (fun x h => let ⟨c, hc, hx⟩ := two_adic_generator_eq x h; ⟨c, hc, hx.val⟩)
    two_adic_generator_returns

/-- Plonky3's generator takes at most 32 bits; the verifier asks for `setup variables + 1` and for
`FOLD` bits, and `Relation::new` refuses more than 31 setup variables. -/
theorem two_adic_generator_complete : Complete (fun x : Usize => x.val ≤ 32)
    (fun x : Usize => field.two_adic_generator x) glValue (fun x => (Replay.twoAdicGenerator x.val).toNat) :=
  complete_of_eq two_adic_generator_sound 32#usize (by simp)
    fun x h => let ⟨c, hc, hx⟩ := two_adic_generator_eq x h; ⟨c, hc, hx.val⟩

/-! ## `path_ok` on `Native` -/

/-- A `Result<(), Rejected>` run on `Native`, as an `Except`. -/
def nativeChecked (x : core.result.Result Unit backend.Rejected × native.Native) : Except backend.Rejected Unit :=
  match x.1 with
  | .Ok u => .ok u
  | .Err e => .error e

theorem nativeChecked_injective : Function.Injective nativeChecked := by
  rintro ⟨a, ⟨⟩⟩ ⟨b, ⟨⟩⟩ h
  cases a <;> cases b <;> simp_all [nativeChecked]

theorem word_sub_eq_zero (x y : Replay.Word) : (x - y).toNat = 0 ↔ x = y := by
  constructor
  · intro h
    apply toNat_of_cast
    have h0 : ((x - y).toNat : Gl) = 0 := by rw [h]; simp
    rw [cast_sub] at h0
    exact sub_eq_zero.mp h0
  · rintro rfl
    have hz : x - x = 0 := toNat_of_cast (by rw [cast_sub, sub_self]; rfl)
    rw [hz]
    rfl

theorem word_beq_iff (x y : Replay.Word) : (x == y) = true ↔ x = y := by
  show (x.word == y.word) = true ↔ _
  rw [beq_iff_eq]
  constructor
  · intro h
    cases x
    cases y
    simp_all
  · rintro rfl
    rfl

theorem native_assert_zero (u : native.Native) (a : field.Gl) (what : Str) :
    native.Native.Insts.Neo_verifiedBackendBackendGlExt.assert_zero u a what =
      ok (if redW a = 0#u64 then .Ok () else .Err what, u) := by
  simp only [native.Native.Insts.Neo_verifiedBackendBackendGlExt.assert_zero, gl_word_eq, bind_ok]
  split <;> rfl

/-- `Native::assert_equal` passes where the two words are equal mod p. -/
theorem sub_zero_iff (a b : field.Gl) : (redW (glSub a b) = 0#u64) = ((wordOf a == wordOf b) = true) := by
  have hx : Exact (redW (glSub a b)) (wordOf a - wordOf b) := by
    unfold Exact
    rw [toU_red_word, (glSub_exact a b).wordOf]
  rw [u64_eq_zero, hx.val, word_sub_eq_zero, word_beq_iff]

/-- The leaf that `path_ok` reads, as copy words: its values, and the levels of its path that it
reads, the shorter of its bits and its path. -/
def leafWords (bits values : Slice field.Gl) (path : Slice (Aeneas.Std.Array field.Gl 4#usize)) : Replay.Leaf :=
  ⟨(values.val.map wordOf).toArray, ((path.val.take (min bits.val.length path.val.length)).map wordsOf).toArray⟩

theorem path_ok_native (u : native.Native) (root : Aeneas.Std.Array field.Gl 4#usize) (bits values : Slice field.Gl)
    (path : Slice (Aeneas.Std.Array field.Gl 4#usize)) :
    ∃ y, setup.path_ok nativeB u root bits values path = ok y ∧
      nativeChecked y = if Replay.pathOk (wordsOf root) (bitIndex bits.val) (leafWords bits values path) then .ok ()
        else .error (toStr "setup leaf path") := by
  obtain ⟨⟨d, u1⟩, hh, hv1⟩ := hash_leaf_native u values
  obtain ⟨hd, _⟩ := words_of_values d _ hv1
  obtain ⟨⟨m, u2⟩, hm, hmv⟩ := merkle_root_native u1 d bits path
  have hcomp : nativeValues (m, u2) = (Replay.merkleRoot (Replay.hashLeaf (values.val.map wordOf).toArray)
      (bitIndex bits.val) (leafWords bits values path).path).toList.map Replay.Word.toNat := by
    rw [hmv]
    have hw : wordsOf d = Replay.hashLeaf (values.val.map wordOf).toArray := by
      rw [wordsOf, hd, Array.toArray_toList]
    split
    · next h0 =>
      have hp : (leafWords bits values path).path = #[] := by simp [leafWords, h0]
      rw [hp]
      simpa [Replay.merkleRoot, nativeValues] using hv1
    · rw [hw]
      rfl
  obtain ⟨hmw, _⟩ := words_of_values m _ hcomp
  obtain ⟨c0, c1, c2, c3, hm4⟩ := list4 m.val (by simp)
  obtain ⟨r0, r1, r2, r3, hr4⟩ := list4 root.val (by simp)
  have hW : Replay.merkleRoot (Replay.hashLeaf (leafWords bits values path).values) (bitIndex bits.val)
      (leafWords bits values path).path = #[wordOf c0, wordOf c1, wordOf c2, wordOf c3] := by
    apply Array.toList_inj.mp
    show (Replay.merkleRoot (Replay.hashLeaf (values.val.map wordOf).toArray) _ _).toList = _
    rw [← hmw, hm4]
    rfl
  have hR : wordsOf root = #[wordOf r0, wordOf r1, wordOf r2, wordOf r3] := by
    simp [wordsOf, hr4]
  have hcopy : Replay.pathOk (wordsOf root) (bitIndex bits.val) (leafWords bits values path) =
      ((wordOf c0 == wordOf r0) && (wordOf c1 == wordOf r1) && (wordOf c2 == wordOf r2) &&
        (wordOf c3 == wordOf r3)) := by
    simp only [Replay.pathOk, hW, hR]
    simp [List.range_succ, Bool.and_assoc]
  have i0 := index_eq m 0#usize (by simp)
  have i1 := index_eq m 1#usize (by simp)
  have i2 := index_eq m 2#usize (by simp)
  have i3 := index_eq m 3#usize (by simp)
  have j0 := index_eq root 0#usize (by simp)
  have j1 := index_eq root 1#usize (by simp)
  have j2 := index_eq root 2#usize (by simp)
  have j3 := index_eq root 3#usize (by simp)
  simp only [hm4, hr4] at i0 i1 i2 i3 j0 j1 j2 j3
  rw [hcopy]
  cases e0 : (wordOf c0 == wordOf r0) <;> cases e1 : (wordOf c1 == wordOf r1) <;>
    cases e2 : (wordOf c2 == wordOf r2) <;> cases e3 : (wordOf c3 == wordOf r3) <;>
    simp [setup.path_ok, hh, hm, i0, i1, i2, i3, j0, j1, j2, j3, native_sub, native_assert_zero, sub_zero_iff,
      e0, e1, e2, e3,  core.result.Result.Insts.CoreOpsTry.branch, nativeChecked,
      core.result.Result.Insts.CoreOpsTry_traitFromResidualResult.from_residual]

theorem path_ok_native_sound : Sound
    (fun x : native.Native × Aeneas.Std.Array field.Gl 4#usize × Slice field.Gl × Slice field.Gl ×
      Slice (Aeneas.Std.Array field.Gl 4#usize) => setup.path_ok nativeB x.1 x.2.1 x.2.2.1 x.2.2.2.1 x.2.2.2.2)
    nativeChecked
    (fun x => if Replay.pathOk (wordsOf x.2.1) (bitIndex x.2.2.1.val) (leafWords x.2.2.1 x.2.2.2.1 x.2.2.2.2)
      then .ok () else .error (toStr "setup leaf path")) :=
  sound_of_eq fun x => path_ok_native x.1 x.2.1 x.2.2.1 x.2.2.2.1 x.2.2.2.2

theorem path_ok_native_complete : Complete (fun _ => True)
    (fun x : native.Native × Aeneas.Std.Array field.Gl 4#usize × Slice field.Gl × Slice field.Gl ×
      Slice (Aeneas.Std.Array field.Gl 4#usize) => setup.path_ok nativeB x.1 x.2.1 x.2.2.1 x.2.2.2.1 x.2.2.2.2)
    nativeChecked
    (fun x => if Replay.pathOk (wordsOf x.2.1) (bitIndex x.2.2.1.val) (leafWords x.2.2.1 x.2.2.2.1 x.2.2.2.2)
      then .ok () else .error (toStr "setup leaf path")) :=
  complete_of_eq path_ok_native_sound ((), Aeneas.Std.Array.repeat 4#usize 0#u64, emptySlice, emptySlice, emptySlice)
    trivial fun x _ => path_ok_native x.1 x.2.1 x.2.2.1 x.2.2.2.1 x.2.2.2.2

/-! ## `check` on `build`

`Circuit.setupCheck` is cut into the steps below, each the body of one of its loops, and
`setupCheck_steps` shows that the cut is the copy. Each translated loop of `check` equals the copy's
loop over the same list. -/

section
open CompressionLean.Circuit

/-- The copy's path check: the leaf hash, the root along the path, and the four lane checks. -/
def setupPath (root : DigestVar) (bits : _root_.Array Var) (leaf : LeafView) : Build Unit := do
  let digest ← hashLeaf leaf.values
  let computed ← merkleRoot digest bits leaf.path
  for lane in List.finRange 4 do
    Build.assertEqual (computed lane) (root lane)

/-- The copy's monomials after one more coordinate `a` of `alpha`. -/
def monomialStep (a : ExtVar) (monomials : _root_.Array ExtVar) : Build (_root_.Array ExtVar) := do
  let mut scaled := #[]
  for m in monomials do
    scaled := scaled.push (← Build.extMul m a)
  pure (monomials ++ scaled)

/-- The copy's coefficient of one monomial, and the next power of `yInverse`. -/
def coefficientStep (yInverse : Var) (monomial : ExtVar) (r : Var × _root_.Array ExtVar) :
    Build (Var × _root_.Array ExtVar) := do
  let c ← Build.extScale monomial r.1
  let scale ← Build.mul r.1 yInverse
  pure (scale, r.2.push c)

/-- One term of the copy's weight at `v`. -/
def weightTerm (omega : Replay.Word) (v : ℕ) (x : ExtVar × ℕ) (total : ExtVar) : Build ExtVar := do
  let term ← Algebra.extScaleConstant x.1 (omega.pow ((2 ^ Replay.foldVars - x.2) * v % 2 ^ Replay.foldVars))
  Build.extAdd total term

/-- The copy's weight at `v`. -/
def weightStep (coefficients : _root_.Array ExtVar) (omega : Replay.Word) (v : ℕ) (weight : _root_.Array ExtVar) :
    Build (_root_.Array ExtVar) := do
  let total ← Algebra.extZero
  let total ← forIn coefficients.zipIdx total fun x t => ForInStep.yield <$> weightTerm omega v x t
  pure (weight.push total)

/-- One value of an oracle, weighted by `w`, added to the combined oracle. -/
def combineTerm (w : ExtVar) (x : Var × ℕ) (combined : _root_.Array ExtVar) : Build (_root_.Array ExtVar) := do
  let term ← Build.extScale w x.1
  let sum ← Build.extAdd combined[x.2]! term
  pure (combined.set! x.2 sum)

/-- The next oracle's values, weighted by `w`, added to the combined oracle. -/
def oracleStep (values : _root_.Array Var) (w : ExtVar) (r : ℕ × _root_.Array ExtVar) :
    Build (ℕ × _root_.Array ExtVar) := do
  let combined ← forIn (values.extract (2 ^ Replay.foldVars * r.1)
    (2 ^ Replay.foldVars * r.1 + 2 ^ Replay.foldVars)).zipIdx r.2 fun x c => ForInStep.yield <$> combineTerm w x c
  pure (r.1 + 1, combined)

/-- The folded value of one group of oracles. -/
def groupStep (values : _root_.Array Var) (weight : _root_.Array ExtVar) (weights : _root_.Array ExtVar)
    (r : ℕ × _root_.Array ExtVar) : Build (ℕ × _root_.Array ExtVar) := do
  let zero ← Algebra.extZero
  let p ← forIn weights (r.1, _root_.Array.replicate (2 ^ Replay.foldVars) zero) fun w q =>
    ForInStep.yield <$> oracleStep values w q
  let total ← Algebra.dot weight p.2
  pure (p.1, r.2.push total)

theorem setupCheck_steps (root : DigestVar) (vars : ℕ) (bits : _root_.Array Var) (leaf : LeafView)
    (groups : _root_.Array (_root_.Array ExtVar)) (alpha : _root_.Array ExtVar) :
    setupCheck root vars bits leaf groups alpha = (do
      setupPath root bits leaf
      let rotation ← Algebra.powerFromBits (Replay.twoAdicGenerator (vars + 1)).inv bits
      let yInverse ← Build.scale rotation Replay.generator.inv
      let one ← Algebra.extOne
      let monomials ← forIn alpha #[one] fun a m => ForInStep.yield <$> monomialStep a m
      let scale ← Build.const (Replay.Word.ofNat (2 ^ Replay.foldVars)).inv
      let r ← forIn monomials (scale, #[]) fun m r => ForInStep.yield <$> coefficientStep yInverse m r
      let weight ← forIn [:2 ^ Replay.foldVars] #[] fun v w =>
        ForInStep.yield <$> weightStep r.2 (Replay.twoAdicGenerator Replay.foldVars) v w
      let f ← forIn groups (0, #[]) fun ws r => ForInStep.yield <$> groupStep leaf.values weight ws r
      pure f.2) := by
  unfold setupCheck setupPath
  simp only [bind_assoc]
  rfl

/-- The leaf that `check` reads, as the copy's view. -/
def leafView (values : Slice Var) (path : Slice (Aeneas.Std.Array Var 4#usize)) : LeafView :=
  ⟨values.val.toArray, (path.val.map digestOf).toArray⟩

theorem digestOf_lane (a : Aeneas.Std.Array Var 4#usize) (k : Fin 4) :
    digestOf a k = a.val[k.val]'(by rw [a.property]; exact k.isLt) := by
  simp [digestOf, List.getD_eq_getElem?_getD]

@[simp] theorem build_assert_equal (s : Builder) (a b : Var) (what : Str) :
    build.assert_equal s a b what = ok (checked (Build.assertEqual a b) s) := rfl

theorem path_ok_build (s : Builder) (root : Aeneas.Std.Array Var 4#usize) (bits values : Slice Var)
    (path : Slice (Aeneas.Std.Array Var 4#usize)) :
    setup.path_ok build s root bits values path =
      ok (.Ok (), ((setupPath (digestOf root) bits.val.toArray (leafView values path)).run s).2) := by
  obtain ⟨⟨d, s1⟩, hh, hd⟩ := hash_leaf_eq s values
  obtain ⟨⟨m, s2⟩, hm, hmd⟩ := merkle_root_eq s1 d bits path
  simp only [builtDigest] at hd hmd
  have i0 := index_eq m 0#usize (by simp)
  have i1 := index_eq m 1#usize (by simp)
  have i2 := index_eq m 2#usize (by simp)
  have i3 := index_eq m 3#usize (by simp)
  have j0 := index_eq root 0#usize (by simp)
  have j1 := index_eq root 1#usize (by simp)
  have j2 := index_eq root 2#usize (by simp)
  have j3 := index_eq root 3#usize (by simp)
  have hf : List.finRange 4 = [0, 1, 2, 3] := rfl
  simp [setup.path_ok, hh, hm, i0, i1, i2, i3, j0, j1, j2, j3, setupPath, leafView, StateT.run_bind, ← hd, ← hmd,
    checked, id_bind, core.result.Result.Insts.CoreOpsTry.branch, hf, List.foldlM_cons, digestOf_lane]

/-- The number of oracles that the groups cover. -/
def oracleCount {α : Type} (groups : List (alloc.vec.Vec α)) : ℕ := (groups.map (·.val.length)).sum

theorem oracleCount_succ {α : Type} (groups : List (alloc.vec.Vec α)) (g : ℕ) (h : g < groups.length) :
    oracleCount (groups.take (g + 1)) = oracleCount (groups.take g) + groups[g].val.length := by
  unfold oracleCount
  rw [List.map_take, List.map_take, List.sum_take_succ _ _ (by simpa using h)]
  simp

theorem check_loop0_sound {α : Type} (groups : Slice (alloc.vec.Vec α)) (o : Usize)
    (h : setup.check_loop0 groups 0#usize 0#usize = ok o) : o.val = oracleCount groups.val := by
  unfold setup.check_loop0 at h
  exact loop_sound (fun (a : Usize × Usize) => setup.check_loop0.body groups a.1 a.2)
    (fun n a => a.2.val ≤ groups.val.length ∧ n = groups.val.length - a.2.val ∧
      a.1.val = oracleCount (groups.val.take a.2.val))
    (fun o => o.val = oracleCount groups.val)
    (by
      rintro n ⟨o, g⟩ c ⟨hg, rfl, ho⟩ hc
      dsimp only at hg ho ⊢
      by_cases hlt : g.val < groups.val.length
      · right
        have hb : g < Slice.len groups := (slice_lt groups g).mpr hlt
        simp only [setup.check_loop0.body, hb, ite_true, slice_index groups g hlt, bind_ok, lift_ok] at hc
        obtain ⟨o', ho', hc⟩ := bind_eq_ok hc
        simp only [Result.ok.injEq] at hc
        subst hc
        have hmax : g.val < Usize.max := by have := groups.property; omega
        have h1 := add_one_val g hmax
        refine ⟨_, groups.val.length - (g.val + 1), rfl, by omega, by dsimp only; omega, by dsimp only; omega, ?_⟩
        dsimp only
        rw [add_eq_ok ho', h1, oracleCount_succ _ _ hlt, ho, alloc.vec.Vec.len_val]
      · left
        have hb : ¬ g < Slice.len groups := by rw [slice_lt]; exact hlt
        simp only [setup.check_loop0.body, hb, ite_false, Result.ok.injEq] at hc
        refine ⟨o, hc.symm, ?_⟩
        rw [ho, List.take_of_length_le (by omega)])
    (groups.val.length) (0#usize, 0#usize) o ⟨by simp, by simp, by simp [oracleCount]⟩ h

theorem check_loop0_eq {α : Type} (groups : Slice (alloc.vec.Vec α)) (fits : oracleCount groups.val ≤ Usize.max) :
    ∃ o, setup.check_loop0 groups 0#usize 0#usize = ok o := by
  unfold setup.check_loop0
  obtain ⟨o, ho, -⟩ := loop_total (fun (a : Usize × Usize) => setup.check_loop0.body groups a.1 a.2)
    (fun n a => a.2.val ≤ groups.val.length ∧ n = groups.val.length - a.2.val ∧
      a.1.val = oracleCount (groups.val.take a.2.val))
    (fun _ => True)
    (by
      rintro n ⟨o, g⟩ ⟨hg, rfl, ho⟩
      dsimp only at hg ho ⊢
      by_cases hlt : g.val < groups.val.length
      · right
        have hb : g < Slice.len groups := (slice_lt groups g).mpr hlt
        have hle : oracleCount (groups.val.take (g.val + 1)) ≤ oracleCount groups.val := by
          have := List.sum_take_add_sum_drop (groups.val.map (·.val.length)) (g.val + 1)
          simp only [oracleCount, List.map_take] at this ⊢
          omega
        obtain ⟨o', ho', hv⟩ := add_ok o (alloc.vec.Vec.len (groups.val[g.val]'hlt))
          (by rw [alloc.vec.Vec.len_val, ho, ← oracleCount_succ _ _ hlt]; omega)
        have hmax : g.val < Usize.max := by have := groups.property; omega
        have h1 := add_one_val g hmax
        refine ⟨(o', core.num.Usize.wrapping_add g 1#usize), groups.val.length - (g.val + 1), ?_, by omega,
          by dsimp only; omega, by dsimp only; omega, ?_⟩
        · simp only [setup.check_loop0.body, hb, ite_true, slice_index groups g hlt, bind_ok, lift_ok, ho']
        · dsimp only
          rw [hv, h1, oracleCount_succ _ _ hlt, ho, alloc.vec.Vec.len_val]
      · left
        have hb : ¬ g < Slice.len groups := by rw [slice_lt]; exact hlt
        exact ⟨o, by simp only [setup.check_loop0.body, hb, ite_false], trivial⟩)
    (groups.val.length) (0#usize, 0#usize) ⟨by simp, by simp, by simp [oracleCount]⟩
  exact ⟨o, ho⟩

theorem vecOfArray_toList (r : _root_.Array ExtVar) (h : r.size ≤ Usize.max) : (vecOfArray r).val = r.toList := by
  unfold vecOfArray
  rw [dite_eq_left (by simpa using h)]
  simp

/-- The copy's `for` loop over `x :: l` runs `g x` and then the loop over `l`. -/
theorem forIn_cons_run {γ ρ : Type} (g : γ → ρ → Build ρ) (x : γ) (l : List γ) (r : ρ) (s : Builder) :
    (forIn (x :: l) r fun x r => ForInStep.yield <$> g x r).run s =
      (forIn l ((g x r).run s).1 fun x r => ForInStep.yield <$> g x r).run ((g x r).run s).2 := by
  simp only [List.forIn_cons, StateT.run_bind, StateT.run_map]
  rfl

theorem pushProduct_size (by_ : ExtVar) : ∀ (l : List ExtVar) (init : _root_.Array ExtVar) (s : Builder),
    ((forIn l init fun x r => ForInStep.yield <$> pushProduct by_ x r).run s).1.size = init.size + l.length
  | [], init, s => rfl
  | x :: l, init, s => by
    rw [forIn_cons_run, pushProduct_size by_ l]
    simp only [pushProduct, StateT.run_bind, StateT.run_pure, id_bind, id_pure, Array.size_push, List.length_cons]
    omega

/-! ### The monomials `α^[u]` -/

theorem scaled_loop_eq (alpha : Aeneas.Std.Array ExtVar 3#usize) (i : Usize) (hi : i.val < 3)
    (monomials : alloc.vec.Vec ExtVar) (s : Builder) (scaled : alloc.vec.Vec ExtVar)
    (fits : scaled.val.length + monomials.val.length ≤ Usize.max) :
    setup.check_loop1_loop0 build s alpha monomials i scaled 0#usize =
      ok (let p := (forIn monomials.val scaled.val.toArray fun x r =>
        ForInStep.yield <$> pushProduct alpha.val[i.val]! x r).run s; (p.2, vecOfArray p.1)) := by
  unfold setup.check_loop1_loop0
  rw [loop_forIn_rel _ (fun a s r k => InnerState a s r k ∧ r.size + (monomials.val.length - k) ≤ Usize.max)
    (fun r s => (s, vecOfArray r)) monomials.val (pushProduct alpha.val[i.val]!) ?step ?stop
    monomials.val.length (s, scaled, 0#usize) s scaled.val.toArray 0 ⟨⟨rfl, rfl, rfl⟩, by simpa using fits⟩ (by simp)]
  · simp
  case step =>
    rintro ⟨b, acc, j⟩ s r k h ⟨⟨rfl, rfl, rfl⟩, hfit⟩
    dsimp only at h hfit
    have hj : j < alloc.vec.Vec.len monomials := by scalar_tac
    simp only [List.size_toArray] at hfit
    have hmax : j.val < Usize.max := by have : monomials.val.length ≤ Usize.max := monomials.property; omega
    obtain ⟨⟨prod, b1⟩, hp⟩ : ∃ p, (Build.extMul (monomials.val[j.val]'h) alpha.val[i.val]!).run b = p := ⟨_, rfl⟩
    obtain ⟨w, hw, hval⟩ := vec_push_ok acc prod (by omega)
    refine ⟨(b1, w, core.num.Usize.wrapping_add j 1#usize), ?_, ⟨⟨?_, ?_, add_one_val j hmax⟩, ?_⟩⟩
    · simp only [setup.check_loop1_loop0.body, hj, ite_true, vec_index monomials j h, index_eq alpha i (by simpa),
        build_ext_mul, bind_ok, uncurry_eq, lift, hp, hw]
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
    rintro ⟨b, acc, j⟩ s r k ⟨⟨rfl, rfl, rfl⟩, -⟩ h
    dsimp only at h
    have hj : ¬ j < alloc.vec.Vec.len monomials := by scalar_tac
    simp only [setup.check_loop1_loop0.body, hj, ite_false, vecOfArray_val]

theorem append_loop_eq {α : Type} (monomials scaled : alloc.vec.Vec α)
    (fits : monomials.val.length + scaled.val.length ≤ Usize.max) :
    ∃ w, setup.check_loop1_loop1 monomials scaled 0#usize = ok w ∧ w.val = monomials.val ++ scaled.val := by
  unfold setup.check_loop1_loop1
  exact loop_total (fun (a : alloc.vec.Vec α × Usize) => setup.check_loop1_loop1.body scaled a.1 a.2)
    (fun n a => a.2.val ≤ scaled.val.length ∧ n = scaled.val.length - a.2.val ∧
      a.1.val = monomials.val ++ scaled.val.take a.2.val)
    (fun w => w.val = monomials.val ++ scaled.val)
    (by
      rintro n ⟨acc, k⟩ ⟨hk, rfl, hacc⟩
      dsimp only at hk hacc ⊢
      by_cases hlt : k.val < scaled.val.length
      · right
        have hb : k < alloc.vec.Vec.len scaled := by scalar_tac
        have hmax : k.val < Usize.max := by have : scaled.val.length ≤ Usize.max := scaled.property; omega
        obtain ⟨w, hw, hval⟩ := vec_push_ok acc (scaled.val[k.val]'hlt) (by
          rw [hacc, List.length_append, List.length_take]; omega)
        have h1 := add_one_val k hmax
        refine ⟨(w, core.num.Usize.wrapping_add k 1#usize), scaled.val.length - (k.val + 1), ?_, by omega,
          by dsimp only; omega, by dsimp only; omega, ?_⟩
        · simp only [setup.check_loop1_loop1.body, hb, ite_true, vec_index scaled k hlt, bind_ok, lift, hw]
        · dsimp only
          rw [hval, hacc, h1, List.take_add_one, List.getElem?_eq_getElem hlt]
          simp
      · left
        have hb : ¬ k < alloc.vec.Vec.len scaled := by scalar_tac
        refine ⟨acc, by simp only [setup.check_loop1_loop1.body, hb, ite_false], ?_⟩
        rw [hacc, List.take_of_length_le (by omega)])
    (scaled.val.length) (monomials, 0#usize) ⟨by simp, by simp, by simp⟩

theorem monomialStep_run (a : ExtVar) (m : _root_.Array ExtVar) (s : Builder) :
    (monomialStep a m).run s =
      (let p := (forIn m.toList #[] fun x r => ForInStep.yield <$> pushProduct a x r).run s; (m ++ p.1, p.2)) := by
  simp only [monomialStep, StateT.run_bind, ← Array.forIn_toList, StateT.run_pure, id_bind]
  rfl

theorem monomials_size : ∀ (l : List ExtVar) (init : _root_.Array ExtVar) (s : Builder),
    ((forIn l init fun a m => ForInStep.yield <$> monomialStep a m).run s).1.size = init.size * 2 ^ l.length
  | [], init, s => by simp; rfl
  | x :: l, init, s => by
    rw [forIn_cons_run, monomials_size l, monomialStep_run]
    simp only [Array.size_append, pushProduct_size, List.length_cons, pow_succ]
    simp
    ring

theorem monomials_loop_eq (alpha : Aeneas.Std.Array ExtVar 3#usize) (s : Builder) (one : ExtVar) :
    setup.check_loop1 build s alpha (vecOfArray #[one]) 0#usize =
      ok (let p := (forIn alpha.val.toArray #[one] fun a m => ForInStep.yield <$> monomialStep a m).run s;
        (p.2, vecOfArray p.1)) := by
  have hal : alpha.val.length = 3 := alpha.property
  unfold setup.check_loop1
  rw [loop_forIn_rel (fun (a : Builder × alloc.vec.Vec ExtVar × Usize) =>
      setup.check_loop1.body build alpha a.1 a.2.1 a.2.2)
    (fun a s r k => a.1 = s ∧ a.2.1.val.toArray = r ∧ a.2.2.val = k ∧ r.size = 2 ^ k)
    (fun r s => (s, vecOfArray r)) alpha.val monomialStep ?step ?stop
    3 (s, vecOfArray #[one], 0#usize) s #[one] 0
    ⟨rfl, by rw [vecOfArray_toList _ (by simp; scalar_tac)], rfl, rfl⟩ (by simp [hal]), List.forIn_toArray]
  · simp
  case step =>
    rintro ⟨b, mon, i⟩ s r k h ⟨rfl, rfl, rfl, hsize⟩
    dsimp only at h hsize ⊢
    have hk : i.val < 3 := by omega
    have hi : i < degrees.FOLD := by simp only [degrees.FOLD]; scalar_tac
    have hU : 16 ≤ Usize.max := by scalar_tac
    rw [List.size_toArray] at hsize
    have hpow : 2 ^ i.val ≤ 4 := by
      calc 2 ^ i.val ≤ 2 ^ 2 := Nat.pow_le_pow_right (by omega) (by omega)
        _ = 4 := rfl
    have hnew : (alloc.vec.Vec.new ExtVar).val.toArray = #[] := rfl
    have hsc := scaled_loop_eq alpha i hk mon b (alloc.vec.Vec.new ExtVar) (by
      show 0 + mon.val.length ≤ Usize.max
      omega)
    rw [hnew] at hsc
    set p := (forIn mon.val (#[] : _root_.Array ExtVar) fun x r =>
      ForInStep.yield <$> pushProduct alpha.val[i.val]! x r).run b with hp
    have hps : p.1.size = mon.val.length := by rw [hp, pushProduct_size]; simp
    have hsz : mon.val.length = 2 ^ i.val := by simpa using hsize
    have hsc2 : setup.check_loop1_loop0 build b alpha mon i (alloc.vec.Vec.new ExtVar) 0#usize =
        ok (p.2, vecOfArray p.1) := hsc
    obtain ⟨w, hw, hwv⟩ := append_loop_eq mon (vecOfArray p.1) (by
      rw [vecOfArray_toList _ (by omega), Array.length_toList (xs := p.1)]
      omega)
    have hmax : i.val < Usize.max := by omega
    have hget : alpha.val[i.val]'h = alpha.val[i.val]! := (getElem!_pos alpha.val i.val h).symm
    have hrun : (monomialStep (alpha.val[i.val]'h) mon.val.toArray).run b = (mon.val.toArray ++ p.1, p.2) := by
      rw [monomialStep_run, hget]
    refine ⟨(p.2, w, core.num.Usize.wrapping_add i 1#usize), ?_, ?_, ?_, add_one_val i hmax, ?_⟩
    · simp only [setup.check_loop1.body, hi, ite_true]
      rw [hsc2, bind_ok]
      show (do
        let monomials1 ← setup.check_loop1_loop1 mon (vecOfArray p.1) 0#usize
        let i1 ← lift (core.num.Usize.wrapping_add i 1#usize)
        ok (ControlFlow.cont (p.2, monomials1, i1))) = _
      simp only [hw, bind_tc_ok, lift]
    · rw [hrun]
    · rw [hrun, hwv, vecOfArray_toList _ (by omega)]
      simp
    · rw [hrun, Array.size_append, hps, List.size_toArray, hsz, pow_succ]
      omega
  case stop =>
    rintro ⟨b, mon, i⟩ s r k ⟨rfl, rfl, rfl, -⟩ h
    dsimp only at h
    have hi : ¬ i < degrees.FOLD := by simp only [degrees.FOLD]; scalar_tac
    simp only [setup.check_loop1.body, hi, ite_false, vecOfArray_val]

/-! ### The coefficients `8^{-1}·y^{-u}·α^[u]` -/

theorem coefficients_loop_eq (yInverse : Var) (monomials : alloc.vec.Vec ExtVar) (s : Builder) (step : Var) :
    setup.check_loop2 build s yInverse monomials step (alloc.vec.Vec.new ExtVar) 0#usize =
      ok (let p := (forIn monomials.val (step, (#[] : _root_.Array ExtVar)) fun m r =>
        ForInStep.yield <$> coefficientStep yInverse m r).run s; (p.2, vecOfArray p.1.2)) := by
  unfold setup.check_loop2
  rw [loop_forIn_rel (fun (a : Builder × Var × alloc.vec.Vec ExtVar × Usize) =>
      setup.check_loop2.body build yInverse monomials a.1 a.2.1 a.2.2.1 a.2.2.2)
    (fun a s r k => a.1 = s ∧ a.2.1 = r.1 ∧ a.2.2.1.val.toArray = r.2 ∧ a.2.2.2.val = k ∧ r.2.size = k)
    (fun r s => (s, vecOfArray r.2)) monomials.val (coefficientStep yInverse) ?step ?stop
    monomials.val.length (s, step, alloc.vec.Vec.new ExtVar, 0#usize) s (step, #[]) 0
    ⟨rfl, rfl, rfl, rfl, rfl⟩ (by simp)]
  · simp
  case step =>
    rintro ⟨b, st, co, m⟩ s ⟨sc, cs⟩ k h ⟨rfl, rfl, rfl, rfl, hsize⟩
    dsimp only at h hsize ⊢
    have hm : m < alloc.vec.Vec.len monomials := by scalar_tac
    have hmax : m.val < Usize.max := by have : monomials.val.length ≤ Usize.max := monomials.property; omega
    have hco : co.val.length = m.val := by simpa using hsize
    obtain ⟨⟨c, b1⟩, hc⟩ : ∃ p, (Build.extScale (monomials.val[m.val]'h) st).run b = p := ⟨_, rfl⟩
    obtain ⟨⟨st1, b2⟩, hst⟩ : ∃ p, (Build.mul st yInverse).run b1 = p := ⟨_, rfl⟩
    obtain ⟨w, hw, hval⟩ := vec_push_ok co c (by omega)
    have hrun : (coefficientStep yInverse (monomials.val[m.val]'h) (st, co.val.toArray)).run b =
        ((st1, co.val.toArray.push c), b2) := by
      simp only [coefficientStep, StateT.run_bind, hc, hst, id_bind, StateT.run_pure, id_pure]
    refine ⟨(b2, st1, w, core.num.Usize.wrapping_add m 1#usize), ?_, ?_, ?_, ?_, add_one_val m hmax, ?_⟩
    · simp only [setup.check_loop2.body, hm, ite_true, vec_index monomials m h, build_ext_scale, build_mul,
        bind_ok, uncurry_eq, lift, hc, hst, hw]
    · rw [hrun]
    · rw [hrun]
    · rw [hrun, hval]
      simp
    · rw [hrun]
      simp [hco]
  case stop =>
    rintro ⟨b, st, co, m⟩ s ⟨sc, cs⟩ k ⟨rfl, rfl, rfl, rfl, -⟩ h
    dsimp only at h
    have hm : ¬ m < alloc.vec.Vec.len monomials := by scalar_tac
    simp only [setup.check_loop2.body, hm, ite_false, vecOfArray_val]

/-! ### The weights -/

theorem exponent_val (u v : Usize) (hu : u.val < 8) (hv : v.val < 8) :
    ∃ c, core.option.Option.unwrap (Usize.checked_mul (core.num.Usize.wrapping_sub 8#usize u) v) = ok c ∧
      (UScalar.cast .U64 (c &&& core.num.Usize.wrapping_sub 8#usize 1#usize)).val = (8 - u.val) * v.val % 8 := by
  have h8 : (core.num.Usize.wrapping_sub 8#usize u).val = 8 - u.val := wrapping_sub_val _ _ (by simp; omega)
  have h7 : (core.num.Usize.wrapping_sub 8#usize 1#usize).val = 7 := wrapping_sub_val _ _ (by simp)
  obtain ⟨c, hc, hcv⟩ := mul_ok (core.num.Usize.wrapping_sub 8#usize u) v (by
    rw [h8]
    have : (8 - u.val) * v.val ≤ 8 * 8 := Nat.mul_le_mul (show 8 - u.val ≤ 8 by omega) (show v.val ≤ 8 by omega)
    have : 64 ≤ Usize.max := by scalar_tac
    omega)
  refine ⟨c, hc, ?_⟩
  have hand : (c &&& core.num.Usize.wrapping_sub 8#usize 1#usize).val = (8 - u.val) * v.val % 8 := by
    rw [UScalar.val_and, h7, hcv, h8, show (7 : ℕ) = 2 ^ 3 - 1 from rfl, Nat.and_two_pow_sub_one_eq_mod]
  rw [UScalar.cast_val_eq, hand]
  exact Nat.mod_eq_of_lt (lt_of_lt_of_le (Nat.mod_lt _ (by omega)) (by simp))

theorem weight_inner_eq (coefficients : alloc.vec.Vec ExtVar) (hc8 : coefficients.val.length = 8)
    (omega : field.Gl) (v : Usize) (hv : v.val < 8) (s : Builder) (total : ExtVar) :
    setup.check_loop3_loop0 build 8#usize s coefficients omega v total 0#usize =
      ok (let p := (forIn coefficients.val.zipIdx total fun x t =>
        ForInStep.yield <$> weightTerm (wordOf omega) v.val x t).run s; (p.2, p.1)) := by
  unfold setup.check_loop3_loop0
  rw [loop_forIn_rel (fun (a : Builder × ExtVar × Usize) =>
      setup.check_loop3_loop0.body build 8#usize coefficients omega v a.1 a.2.1 a.2.2)
    (fun a s r k => a.1 = s ∧ a.2.1 = r ∧ a.2.2.val = k)
    (fun r s => (s, r)) coefficients.val.zipIdx (weightTerm (wordOf omega) v.val) ?step ?stop
    8 (s, total, 0#usize) s total 0 ⟨rfl, rfl, rfl⟩ (by simp [hc8])]
  · simp
  case step =>
    rintro ⟨b, t, u⟩ s r k h ⟨rfl, rfl, rfl⟩
    dsimp only at h ⊢
    rw [List.length_zipIdx] at h
    have hu : u < alloc.vec.Vec.len coefficients := by scalar_tac
    have hu8 : u.val < 8 := by omega
    have hmax : u.val < Usize.max := by scalar_tac
    obtain ⟨c, hc, hcv⟩ := exponent_val u v hu8 hv
    obtain ⟨g, hg, hgx⟩ := gl_pow_exact _ omega _ hcv
    obtain ⟨⟨term, b1⟩, ht⟩ : ∃ p,
      (Algebra.extScaleConstant (coefficients.val[u.val]'h) (wordOf g)).run b = p := ⟨_, rfl⟩
    obtain ⟨⟨t1, b2⟩, ht1⟩ : ∃ p, (Build.extAdd t term).run b1 = p := ⟨_, rfl⟩
    have hf : (2 : ℕ) ^ Replay.foldVars = 8 := rfl
    have hrun : (weightTerm (wordOf omega) v.val (coefficients.val[u.val]'h, u.val) t).run b = (t1, b2) := by
      simp only [weightTerm, StateT.run_bind, hf, ← hgx.wordOf, ht, id_bind, ht1]
    refine ⟨(b2, t1, core.num.Usize.wrapping_add u 1#usize), ?_, ?_, ?_, add_one_val u hmax⟩
    · simp only [setup.check_loop3_loop0.body, hu, ite_true, lift, bind_ok, hc, vec_index coefficients u h, hg,
        ext_scale_constant_eq, uncurry_eq, ht, build_ext_add, ht1]
    · simp only [List.getElem_zipIdx, zero_add, hrun]
    · simp only [List.getElem_zipIdx, zero_add, hrun]
  case stop =>
    rintro ⟨b, t, u⟩ s r k ⟨rfl, rfl, rfl⟩ h
    dsimp only at h
    rw [List.length_zipIdx] at h
    have hu : ¬ u < alloc.vec.Vec.len coefficients := by scalar_tac
    simp only [setup.check_loop3_loop0.body, hu, ite_false]

theorem weightStep_run (coefficients : alloc.vec.Vec ExtVar) (omegaW : Replay.Word) (v : ℕ)
    (weight : _root_.Array ExtVar) (s : Builder) :
    (weightStep coefficients.val.toArray omegaW v weight).run s =
      (let z := Algebra.extZero.run s
       let p := (forIn coefficients.val.zipIdx z.1 fun x t => ForInStep.yield <$> weightTerm omegaW v x t).run z.2
       (weight.push p.1, p.2)) := by
  simp only [weightStep, StateT.run_bind, Array.zipIdx_toArray, List.forIn_toArray, id_bind, StateT.run_pure,
    id_pure]

theorem weights_loop_eq (coefficients : alloc.vec.Vec ExtVar) (hc8 : coefficients.val.length = 8)
    (omega : field.Gl) (s : Builder) :
    setup.check_loop3 build 8#usize s coefficients omega (alloc.vec.Vec.new ExtVar) 0#usize =
      ok (let p := (forIn (List.range 8) (#[] : _root_.Array ExtVar) fun v w =>
        ForInStep.yield <$> weightStep coefficients.val.toArray (wordOf omega) v w).run s; (p.2, vecOfArray p.1)) := by
  unfold setup.check_loop3
  rw [loop_forIn_rel (fun (a : Builder × alloc.vec.Vec ExtVar × Usize) =>
      setup.check_loop3.body build 8#usize coefficients omega a.1 a.2.1 a.2.2)
    (fun a s r k => a.1 = s ∧ a.2.1.val.toArray = r ∧ a.2.2.val = k ∧ r.size = k)
    (fun r s => (s, vecOfArray r)) (List.range 8) (weightStep coefficients.val.toArray (wordOf omega)) ?step ?stop
    8 (s, alloc.vec.Vec.new ExtVar, 0#usize) s #[] 0 ⟨rfl, rfl, rfl, rfl⟩ (by simp)]
  · simp
  case step =>
    rintro ⟨b, w, v⟩ s r k h ⟨rfl, rfl, rfl, hsize⟩
    dsimp only at h hsize ⊢
    rw [List.length_range] at h
    have hv : v < 8#usize := by scalar_tac
    have hmax : v.val < Usize.max := by scalar_tac
    have hw : w.val.length = v.val := by simpa using hsize
    obtain ⟨⟨z, b1⟩, hz⟩ : ∃ p, Algebra.extZero.run b = p := ⟨_, rfl⟩
    set p := (forIn coefficients.val.zipIdx z fun x t =>
      ForInStep.yield <$> weightTerm (wordOf omega) v.val x t).run b1 with hp
    have hin : setup.check_loop3_loop0 build 8#usize b1 coefficients omega v z 0#usize = ok (p.2, p.1) :=
      weight_inner_eq coefficients hc8 omega v (by scalar_tac) b1 z
    obtain ⟨w', hw', hval⟩ := vec_push_ok w p.1 (by scalar_tac)
    have hrun : (weightStep coefficients.val.toArray (wordOf omega) v.val w.val.toArray).run b =
        (w.val.toArray.push p.1, p.2) := by
      rw [weightStep_run, hz]
    refine ⟨(p.2, w', core.num.Usize.wrapping_add v 1#usize), ?_, ?_, ?_, add_one_val v hmax, ?_⟩
    · simp only [setup.check_loop3.body, hv, ite_true, ext_zero_eq, hz, bind_ok, uncurry_eq]
      rw [hin, bind_ok]
      show (do
        let weight1 ← alloc.vec.Vec.push w p.1
        let v1 ← lift (core.num.Usize.wrapping_add v 1#usize)
        ok (ControlFlow.cont (p.2, weight1, v1))) = _
      simp only [hw', bind_tc_ok, lift]
    · rw [List.getElem_range, hrun]
    · rw [List.getElem_range, hrun, hval]
      simp
    · rw [List.getElem_range, hrun]
      simp [hw]
  case stop =>
    rintro ⟨b, w, v⟩ s r k ⟨rfl, rfl, rfl, -⟩ h
    dsimp only at h
    rw [List.length_range] at h
    have hv : ¬ v < 8#usize := by scalar_tac
    simp only [setup.check_loop3.body, hv, ite_false, vecOfArray_val]

/-! ### The folded values -/

theorem vec_index_mut {α : Type} (v : alloc.vec.Vec α) (i : Usize) (h : i.val < v.val.length) :
    ∃ back : α → alloc.vec.Vec α,
      alloc.vec.Vec.index_mut (core.slice.index.SliceIndexUsizeSlice α) v i = ok (v.val[i.val], back) ∧
        ∀ x, (back x).val = v.val.set i.val x := by
  refine ⟨alloc.vec.Vec.set v i, ?_, fun x => by simp [alloc.vec.Vec.set]⟩
  have hi := vec_index v i h
  rw [alloc.vec.Vec.index_slice_index] at hi
  rw [alloc.vec.Vec.index_mut_slice_index, alloc.vec.Vec.index_mut_usize, hi, bind_tc_ok]

theorem replicate_loop_eq {α : Type} (zero : α) :
    ∃ w, setup.check_loop4_loop0 8#usize zero (alloc.vec.Vec.new α) 0#usize = ok w ∧ w.val = List.replicate 8 zero := by
  unfold setup.check_loop4_loop0
  exact loop_total (fun (a : alloc.vec.Vec α × Usize) => setup.check_loop4_loop0.body 8#usize zero a.1 a.2)
    (fun n a => a.2.val ≤ 8 ∧ n = 8 - a.2.val ∧ a.1.val = List.replicate a.2.val zero)
    (fun w => w.val = List.replicate 8 zero)
    (by
      rintro n ⟨acc, v⟩ ⟨hv, rfl, hacc⟩
      dsimp only at hv hacc ⊢
      by_cases hlt : v.val < 8
      · right
        have hb : v < 8#usize := by scalar_tac
        have hmax : v.val < Usize.max := by scalar_tac
        obtain ⟨w, hw, hval⟩ := vec_push_ok acc zero (by rw [hacc, List.length_replicate]; scalar_tac)
        have h1 := add_one_val v hmax
        refine ⟨(w, core.num.Usize.wrapping_add v 1#usize), 8 - (v.val + 1), ?_, by omega,
          by dsimp only; omega, by dsimp only; omega, ?_⟩
        · simp only [setup.check_loop4_loop0.body, hb, ite_true, hw, bind_ok, lift]
        · dsimp only
          rw [hval, hacc, h1, List.replicate_succ']
      · left
        have hb : ¬ v < 8#usize := by scalar_tac
        refine ⟨acc, by simp only [setup.check_loop4_loop0.body, hb, ite_false], ?_⟩
        rw [hacc, show v.val = 8 by omega])
    8 (alloc.vec.Vec.new α, 0#usize) ⟨by simp, by simp, by simp [alloc.vec.Vec.new]⟩

/-- The values of the oracle that starts at `start`, as the copy reads them: at most eight. -/
def oracleValues (values : Slice Var) (start : ℕ) : List (Var × ℕ) := ((values.val.drop start).take 8).zipIdx

theorem combine_loop_eq (values : Slice Var) (groups : Slice (alloc.vec.Vec ExtVar)) (g k start : Usize)
    (hg : g.val < groups.val.length) (hk : k.val < (groups.val[g.val]'hg).val.length)
    (hs : start.val + 8 ≤ values.val.length) (s : Builder) (combined : alloc.vec.Vec ExtVar)
    (hc : combined.val.length = 8) :
    setup.check_loop4_loop1_loop0 build 8#usize s values groups g combined k start 0#usize =
      ok (let p := (forIn (oracleValues values start.val) combined.val.toArray fun x c =>
        ForInStep.yield <$> combineTerm ((groups.val[g.val]'hg).val[k.val]'hk) x c).run s;
        (p.2, vecOfArray p.1)) := by
  have hlen : (oracleValues values start.val).length = 8 := by simp [oracleValues]; omega
  unfold setup.check_loop4_loop1_loop0
  rw [loop_forIn_rel (fun (a : Builder × alloc.vec.Vec ExtVar × Usize) =>
      setup.check_loop4_loop1_loop0.body build 8#usize values groups g k start a.1 a.2.1 a.2.2)
    (fun a s r j => a.1 = s ∧ a.2.1.val.toArray = r ∧ a.2.2.val = j ∧ r.size = 8)
    (fun r s => (s, vecOfArray r)) (oracleValues values start.val)
    (combineTerm ((groups.val[g.val]'hg).val[k.val]'hk)) ?step ?stop
    8 (s, combined, 0#usize) s combined.val.toArray 0 ⟨rfl, rfl, rfl, by simpa using hc⟩ (by simp [hlen])]
  · simp
  case step =>
    rintro ⟨b, co, v⟩ s r j h ⟨rfl, rfl, rfl, hsize⟩
    dsimp only at h hsize ⊢
    rw [hlen] at h
    have hco : co.val.length = 8 := by simpa using hsize
    have hv : v < 8#usize := by scalar_tac
    have hmax : v.val < Usize.max := by scalar_tac
    have hadd : (core.num.Usize.wrapping_add start v).val = start.val + v.val :=
      wrapping_add_le _ _ (by have := values.property; omega)
    have hin : (core.num.Usize.wrapping_add start v).val < values.val.length := by omega
    have hx : (oracleValues values start.val)[v.val]'(by rw [hlen]; exact h) =
        (values.val[(core.num.Usize.wrapping_add start v).val]'hin, v.val) := by
      simp [oracleValues, hadd]
    obtain ⟨⟨term, b1⟩, ht⟩ : ∃ p, (Build.extScale ((groups.val[g.val]'hg).val[k.val]'hk)
      (values.val[(core.num.Usize.wrapping_add start v).val]'hin)).run b = p := ⟨_, rfl⟩
    set q := (Build.extAdd (co.val[v.val]'(by omega)) term).run b1 with hq
    obtain ⟨back, hback, hbv⟩ := vec_index_mut co v (by omega)
    have hrun : (combineTerm ((groups.val[g.val]'hg).val[k.val]'hk)
        (values.val[(core.num.Usize.wrapping_add start v).val]'hin, v.val) co.val.toArray).run b =
        (co.val.toArray.setIfInBounds v.val q.1, q.2) := by
      have hget : co.val.toArray[v.val]! = co.val[v.val]'(by omega) := by
        rw [getElem!_pos co.val.toArray v.val (by simp; omega)]
        simp
      simp only [combineTerm, StateT.run_bind, ht, id_bind, hget, ← hq, StateT.run_pure, id_pure,
        Array.set!_eq_setIfInBounds]
    refine ⟨(q.2, back q.1, core.num.Usize.wrapping_add v 1#usize), ?_, ?_, ?_, add_one_val v hmax, ?_⟩
    · simp only [setup.check_loop4_loop1_loop0.body, hv, ite_true, slice_index groups g hg, vec_index _ k hk,
        lift, bind_ok, slice_index values _ hin, build_ext_scale, ht, vec_index co v (by omega), build_ext_add,
        hback]
      rfl
    · rw [hx, hrun]
    · rw [hx, hrun, hbv]
      simp
    · rw [hx, hrun]
      simp [hco]
  case stop =>
    rintro ⟨b, co, v⟩ s r j ⟨rfl, rfl, rfl, -⟩ h
    dsimp only at h
    rw [hlen] at h
    have hv : ¬ v < 8#usize := by scalar_tac
    simp only [setup.check_loop4_loop1_loop0.body, hv, ite_false, vecOfArray_val]

theorem combine_size (w : ExtVar) : ∀ (l : List (Var × ℕ)) (c : _root_.Array ExtVar) (s : Builder),
    ((forIn l c fun x c => ForInStep.yield <$> combineTerm w x c).run s).1.size = c.size
  | [], c, s => rfl
  | x :: l, c, s => by
    rw [forIn_cons_run, combine_size w l]
    have hrun : ((combineTerm w x c).run s).1 = c.setIfInBounds x.2
        ((Build.extAdd c[x.2]! ((Build.extScale w x.1).run s).1).run ((Build.extScale w x.1).run s).2).1 := by
      simp only [combineTerm, StateT.run_bind, id_bind, StateT.run_pure, id_pure, Array.set!_eq_setIfInBounds]
    rw [hrun, Array.size_setIfInBounds]

/-- A natural number as a translated `usize`, modulo its width. -/
def usizeOf (n : ℕ) : Usize := ⟨BitVec.ofNat _ n⟩

theorem usizeOf_val (u : Usize) : usizeOf u.val = u := by
  apply UScalar.val_eq_imp
  show (BitVec.ofNat _ u.bv.toNat).toNat = u.bv.toNat
  rw [BitVec.toNat_ofNat]
  exact Nat.mod_eq_of_lt u.bv.isLt

theorem oracleStep_run (values : Slice Var) (w : ExtVar) (o : ℕ) (c : _root_.Array ExtVar) (s : Builder) :
    (oracleStep values.val.toArray w (o, c)).run s =
      (let p := (forIn (oracleValues values (o * 8)) c fun x c => ForInStep.yield <$> combineTerm w x c).run s;
        ((o + 1, p.1), p.2)) := by
  simp only [oracleStep, StateT.run_bind, List.extract_toArray, Array.zipIdx_toArray, List.forIn_toArray, id_bind,
    StateT.run_pure, id_pure, oracleValues, List.extract_eq_take_drop]
  rw [show (2 : ℕ) ^ Replay.foldVars = 8 from rfl, show 8 * o + 8 - 8 * o = 8 by omega, mul_comm]

theorem oracles_loop_eq (values : Slice Var) (groups : Slice (alloc.vec.Vec ExtVar)) (g : Usize)
    (hg : g.val < groups.val.length) (oracle : Usize)
    (hfit : 8 * (oracle.val + (groups.val[g.val]'hg).val.length) ≤ values.val.length) (s : Builder)
    (combined : alloc.vec.Vec ExtVar) (hc : combined.val.length = 8) :
    setup.check_loop4_loop1 build 8#usize s values groups oracle g combined 0#usize =
      ok (let p := (forIn (groups.val[g.val]'hg).val (oracle.val, combined.val.toArray) fun w r =>
        ForInStep.yield <$> oracleStep values.val.toArray w r).run s; (p.2, usizeOf p.1.1, vecOfArray p.1.2)) := by
  unfold setup.check_loop4_loop1
  rw [loop_forIn_rel (fun (a : Builder × Usize × alloc.vec.Vec ExtVar × Usize) =>
      setup.check_loop4_loop1.body build 8#usize values groups g a.1 a.2.1 a.2.2.1 a.2.2.2)
    (fun a s r j => a.1 = s ∧ a.2.1.val = r.1 ∧ a.2.2.1.val.toArray = r.2 ∧ a.2.2.2.val = j ∧ r.2.size = 8 ∧
      r.1 = oracle.val + j)
    (fun r s => (s, usizeOf r.1, vecOfArray r.2)) (groups.val[g.val]'hg).val (oracleStep values.val.toArray) ?step
    ?stop _ (s, oracle, combined, 0#usize) s (oracle.val, combined.val.toArray) 0
    ⟨rfl, rfl, rfl, rfl, by simpa using hc, rfl⟩ rfl]
  · simp
  case step =>
    rintro ⟨b, o, co, k⟩ s ⟨ro, rc⟩ j h ⟨rfl, ho, rfl, rfl, hsize, hro⟩
    dsimp only at h ho hsize hro ⊢
    subst ho
    have hco : co.val.length = 8 := by simpa using hsize
    have hk : k < alloc.vec.Vec.len (groups.val[g.val]'hg) := by scalar_tac
    have hU : values.val.length ≤ Usize.max := values.property
    obtain ⟨st, hst, hstv⟩ := mul_ok o 8#usize (by simp; omega)
    have hmax : k.val < Usize.max := by scalar_tac
    have homax : o.val < Usize.max := by omega
    have hcl := combine_loop_eq values groups g k st hg h (by simp at hstv; omega) b co hco
    set p := (forIn (oracleValues values st.val) co.val.toArray fun x c =>
      ForInStep.yield <$> combineTerm ((groups.val[g.val]'hg).val[k.val]'h) x c).run b with hp
    have hps : p.1.size = 8 := by rw [hp, combine_size]; simpa using hco
    have hrun : (oracleStep values.val.toArray ((groups.val[g.val]'hg).val[k.val]'h) (o.val, co.val.toArray)).run b =
        ((o.val + 1, p.1), p.2) := by
      rw [oracleStep_run, show o.val * 8 = st.val by simp at hstv; omega]
    refine ⟨(p.2, core.num.Usize.wrapping_add o 1#usize, vecOfArray p.1, core.num.Usize.wrapping_add k 1#usize),
      ?_, ?_, ?_, ?_, add_one_val k hmax, ?_, ?_⟩
    · simp only [setup.check_loop4_loop1.body, slice_index groups g hg, bind_ok, hk, ite_true, lift, hst]
      rw [hcl, bind_ok]
      rfl
    · rw [hrun]
    · rw [hrun, add_one_val o homax]
    · rw [hrun, vecOfArray_toList _ (by omega)]
    · rw [hrun]
      simpa using hps
    · rw [hrun]
      dsimp only
      omega
  case stop =>
    rintro ⟨b, o, co, k⟩ s ⟨ro, rc⟩ j ⟨rfl, ho, rfl, rfl, -, -⟩ h
    dsimp only at h ho
    have hk : ¬ k < alloc.vec.Vec.len (groups.val[g.val]'hg) := by scalar_tac
    simp only [setup.check_loop4_loop1.body, slice_index groups g hg, bind_ok, hk, ite_false, vecOfArray_val, ← ho,
      usizeOf_val]

theorem oracles_shape (values : Slice Var) : ∀ (l : List ExtVar) (o : ℕ) (c : _root_.Array ExtVar) (s : Builder),
    ((forIn l (o, c) fun w r => ForInStep.yield <$> oracleStep values.val.toArray w r).run s).1.1 = o + l.length ∧
    ((forIn l (o, c) fun w r => ForInStep.yield <$> oracleStep values.val.toArray w r).run s).1.2.size = c.size
  | [], o, c, s => ⟨rfl, rfl⟩
  | x :: l, o, c, s => by
    rw [forIn_cons_run, oracleStep_run]
    obtain ⟨h1, h2⟩ := oracles_shape values l (o + 1)
      ((forIn (oracleValues values (o * 8)) c fun x' c => ForInStep.yield <$> combineTerm x x' c).run s).1
      ((forIn (oracleValues values (o * 8)) c fun x' c => ForInStep.yield <$> combineTerm x x' c).run s).2
    exact ⟨by rw [h1, List.length_cons]; omega, by rw [h2, combine_size]⟩

theorem deref_val {α : Type} (v : alloc.vec.Vec α) : (alloc.vec.Vec.deref v).val = v.val := by
  simp [alloc.vec.Vec.deref]

theorem usizeOf_val_of_le (n : ℕ) (h : n ≤ Usize.max) : (usizeOf n).val = n := by
  show (BitVec.ofNat _ n).toNat = n
  rw [BitVec.toNat_ofNat]
  apply Nat.mod_eq_of_lt
  have hm := Usize.max_def
  have hp : 0 < 2 ^ Usize.numBits := Nat.pos_of_ne_zero (by simp)
  show n < 2 ^ UScalarTy.Usize.numBits
  rw [← Usize.numBits_def]
  omega

theorem groupStep_run (values : Slice Var) (weight : alloc.vec.Vec ExtVar) (ws : alloc.vec.Vec ExtVar) (o : ℕ)
    (folded : _root_.Array ExtVar) (s : Builder) :
    (groupStep values.val.toArray weight.val.toArray ws.val.toArray (o, folded)).run s =
      (let z := Algebra.extZero.run s
       let p := (forIn ws.val (o, _root_.Array.replicate 8 z.1) fun w r =>
         ForInStep.yield <$> oracleStep values.val.toArray w r).run z.2
       let d := (Algebra.dot weight.val.toArray p.1.2).run p.2
       ((p.1.1, folded.push d.1), d.2)) := by
  simp only [groupStep, StateT.run_bind, List.forIn_toArray, id_bind, StateT.run_pure, id_pure,
    show (2 : ℕ) ^ Replay.foldVars = 8 from rfl]

theorem groups_loop_eq (values : Slice Var) (groups : Slice (alloc.vec.Vec ExtVar)) (weight : alloc.vec.Vec ExtVar)
    (hcover : 8 * oracleCount groups.val = values.val.length) (s : Builder) :
    setup.check_loop4 build 8#usize s values groups weight 0#usize (alloc.vec.Vec.new ExtVar) 0#usize =
      ok (let p := (forIn (groups.val.map (·.val.toArray)) (0, (#[] : _root_.Array ExtVar)) fun ws r =>
        ForInStep.yield <$> groupStep values.val.toArray weight.val.toArray ws r).run s; (p.2, vecOfArray p.1.2)) := by
  unfold setup.check_loop4
  rw [loop_forIn_rel (fun (a : Builder × Usize × alloc.vec.Vec ExtVar × Usize) =>
      setup.check_loop4.body build 8#usize values groups weight a.1 a.2.1 a.2.2.1 a.2.2.2)
    (fun a s r j => a.1 = s ∧ a.2.1.val = r.1 ∧ a.2.2.1.val.toArray = r.2 ∧ a.2.2.2.val = j ∧
      r.1 = oracleCount (groups.val.take j) ∧ r.2.size = j)
    (fun r s => (s, vecOfArray r.2)) (groups.val.map (·.val.toArray)) (groupStep values.val.toArray weight.val.toArray)
    ?step ?stop _ (s, 0#usize, alloc.vec.Vec.new ExtVar, 0#usize) s (0, #[]) 0
    ⟨rfl, rfl, rfl, rfl, by simp [oracleCount], rfl⟩ rfl]
  · simp
  case step =>
    rintro ⟨b, o, fo, g⟩ s ⟨ro, rf⟩ j h ⟨rfl, ho, rfl, rfl, hro, hsize⟩
    dsimp only at h ho hro hsize ⊢
    subst ho
    rw [List.length_map] at h
    have hg : g < Slice.len groups := (slice_lt groups g).mpr h
    have hmax : g.val < Usize.max := by have := groups.property; omega
    have hfo : fo.val.length = g.val := by simpa using hsize
    have hle : oracleCount (groups.val.take (g.val + 1)) ≤ oracleCount groups.val := by
      have := List.sum_take_add_sum_drop (groups.val.map (·.val.length)) (g.val + 1)
      simp only [oracleCount, List.map_take] at this ⊢
      omega
    have hfit : 8 * (o.val + (groups.val[g.val]'h).val.length) ≤ values.val.length := by
      rw [hro, ← oracleCount_succ _ _ h]
      omega
    obtain ⟨⟨z, b1⟩, hz⟩ : ∃ q, Algebra.extZero.run b = q := ⟨_, rfl⟩
    obtain ⟨w, hw, hwv⟩ := replicate_loop_eq z
    have hwa : w.val.toArray = _root_.Array.replicate 8 z := by rw [hwv]; rfl
    have hol := oracles_loop_eq values groups g h o hfit b1 w (by simp [hwv])
    obtain ⟨⟨⟨o', c'⟩, b2⟩, hpq⟩ : ∃ q, (forIn (groups.val[g.val]'h).val (o.val, w.val.toArray) fun w r =>
      ForInStep.yield <$> oracleStep values.val.toArray w r).run b1 = q := ⟨_, rfl⟩
    rw [hpq] at hol
    obtain ⟨ho', hc'⟩ := oracles_shape values (groups.val[g.val]'h).val o.val w.val.toArray b1
    rw [hpq] at ho' hc'
    simp only [List.size_toArray, hwv, List.length_replicate] at hc'
    dsimp only at ho'
    obtain ⟨⟨d, b3⟩, hd⟩ : ∃ q, (Algebra.dot weight.val.toArray c').run b2 = q := ⟨_, rfl⟩
    obtain ⟨f1, hf1, hf1v⟩ := vec_push_ok fo d (by scalar_tac)
    have hrun : (groupStep values.val.toArray weight.val.toArray (groups.val[g.val]'h).val.toArray
        (o.val, fo.val.toArray)).run b = ((o', fo.val.toArray.push d), b3) := by
      rw [groupStep_run, hz]
      simp only [← hwa, hpq, hd]
    refine ⟨(b3, usizeOf o', f1, core.num.Usize.wrapping_add g 1#usize), ?_, ?_, ?_, ?_, add_one_val g hmax,
      ?_, ?_⟩
    · simp only [setup.check_loop4.body, hg, ite_true, ext_zero_eq, hz, bind_ok, uncurry_eq]
      rw [hw, bind_ok, hol, bind_ok]
      simp only [dot_eq, deref_val, vecOfArray_toList c' (by scalar_tac), Array.toArray_toList, hd, bind_ok,
        uncurry_eq, hf1, lift]
    · simp only [List.getElem_map, hrun]
    · simp only [List.getElem_map, hrun]
      rw [ho']
      exact usizeOf_val_of_le _ (by
        have := oracleCount_succ groups.val g.val h
        have := values.property
        omega)
    · simp only [List.getElem_map, hrun, hf1v]
      simp
    · simp only [List.getElem_map, hrun]
      rw [ho', hro, oracleCount_succ _ _ h]
    · simp only [List.getElem_map, hrun]
      simp [hfo]
  case stop =>
    rintro ⟨b, o, fo, g⟩ s ⟨ro, rf⟩ j ⟨rfl, ho, rfl, rfl, -, -⟩ h
    dsimp only at h ho
    rw [List.length_map] at h
    have hg : ¬ g < Slice.len groups := by rw [slice_lt]; omega
    simp only [setup.check_loop4.body, hg, ite_false, vecOfArray_val]

/-! ### `check` -/

theorem coefficients_size (y : Var) : ∀ (l : List ExtVar) (r : Var × _root_.Array ExtVar) (s : Builder),
    ((forIn l r fun m r => ForInStep.yield <$> coefficientStep y m r).run s).1.2.size = r.2.size + l.length
  | [], r, s => rfl
  | x :: l, r, s => by
    rw [forIn_cons_run, coefficients_size y l]
    simp only [coefficientStep, StateT.run_bind, id_bind, StateT.run_pure, id_pure, Array.size_push,
      List.length_cons]
    omega

theorem groups_size (values : _root_.Array Var) (weight : _root_.Array ExtVar) :
    ∀ (l : List (_root_.Array ExtVar)) (r : ℕ × _root_.Array ExtVar) (s : Builder),
      ((forIn l r fun ws r => ForInStep.yield <$> groupStep values weight ws r).run s).1.2.size = r.2.size + l.length
  | [], r, s => rfl
  | x :: l, r, s => by
    rw [forIn_cons_run, groups_size values weight l]
    simp only [groupStep, StateT.run_bind, id_bind, StateT.run_pure, id_pure, Array.size_push, List.length_cons]
    omega

theorem weights_size (coefficients : _root_.Array ExtVar) (omegaW : Replay.Word) :
    ∀ (l : List ℕ) (w : _root_.Array ExtVar) (s : Builder),
      ((forIn l w fun v w => ForInStep.yield <$> weightStep coefficients omegaW v w).run s).1.size = w.size + l.length
  | [], w, s => rfl
  | x :: l, w, s => by
    rw [forIn_cons_run, weights_size coefficients omegaW l]
    simp only [weightStep, StateT.run_bind, id_bind, StateT.run_pure, id_pure, Array.size_push, List.length_cons]
    omega

theorem forIn_leaf {β : Type} (init : β) (f : ℕ → β → Build (ForInStep β)) :
    forIn [:2 ^ Replay.foldVars] init f = forIn (List.range 8) init f := by
  rw [Std.Legacy.Range.forIn_eq_forIn_range', List.range_eq_range']
  rfl

theorem leaf_eq : setup.LEAF = ok 8#usize := by
  have hw : 3 < UScalarTy.Usize.numBits := by
    rcases System.Platform.numBits_eq with h | h <;> simp [h]
  have h8 : 8 < 2 ^ UScalarTy.Usize.numBits := lt_of_lt_of_le (by decide) (Nat.pow_le_pow_right (by omega) hw)
  unfold setup.LEAF degrees.FOLD
  simp only [HShiftLeft.hShiftLeft, UScalar.shiftLeft_UScalar, UScalar.shiftLeft, show (3#usize).val = 3 from rfl, hw,
    ite_true, Result.ok.injEq]
  apply UScalar.val_eq_imp
  simp only [UScalar.val, BitVec.shiftLeft_eq, BitVec.toNat_shiftLeft]
  simp [Nat.shiftLeft_eq]

/-- A run of `check` on `build` that returns a `Result<Vec<E>, Rejected>`, as an `Except`. -/
def builtChecked (x : core.result.Result (alloc.vec.Vec ExtVar) backend.Rejected × Builder) :
    Except backend.Rejected (_root_.Array ExtVar) × Builder :=
  (match x.1 with
    | .Ok v => .ok v.val.toArray
    | .Err e => .error e, x.2)

theorem builtChecked_injective : Function.Injective builtChecked := by
  rintro ⟨a, s⟩ ⟨b, t⟩ h
  simp only [builtChecked, Prod.mk.injEq] at h
  obtain ⟨h1, rfl⟩ := h
  cases a <;> cases b <;> simp_all
  exact vec_val_injective h1

theorem check_eq (s : Builder) (root : Aeneas.Std.Array Var 4#usize) (vars : Usize) (bits values : Slice Var)
    (path : Slice (Aeneas.Std.Array Var 4#usize)) (groups : Slice (alloc.vec.Vec ExtVar))
    (alpha : Aeneas.Std.Array ExtVar 3#usize) (hcover : 8 * oracleCount groups.val = values.val.length)
    (hvars : vars.val + 1 ≤ 32) :
    ∃ y, setup.check build s root vars bits values path groups alpha = ok y ∧
      builtChecked y = (let p := (setupCheck (digestOf root) vars.val bits.val.toArray (leafView values path)
        (groups.val.map (·.val.toArray)).toArray alpha.val.toArray).run s; (.ok p.1, p.2)) := by
  have hvl : values.val.length ≤ Usize.max := values.property
  obtain ⟨o, ho⟩ := check_loop0_eq groups (by omega)
  have hov := check_loop0_sound groups o ho
  obtain ⟨c, hc, hcv⟩ := mul_ok o 8#usize (by
    show o.val * 8 ≤ Usize.max
    rw [hov]
    omega)
  have h8v : (8#usize).val = 8 := rfl
  have h1v : (1#usize).val = 1 := rfl
  have hceq : c = Slice.len values := UScalar.val_eq_imp _ _ (by rw [hcv, Slice.len_val, h8v, hov]; simp; omega)
  have hU : 64 ≤ Usize.max := by scalar_tac
  obtain ⟨c1, hc1, hc1v⟩ := add_ok vars 1#usize (by rw [h1v]; omega)
  rw [h1v] at hc1v
  obtain ⟨g, hg, hgx⟩ := two_adic_generator_eq c1 (by omega)
  obtain ⟨order, horder, hox⟩ := gl_inv_exact g
  have hpath := path_ok_build s root bits values path
  set s1 := ((setupPath (digestOf root) bits.val.toArray (leafView values path)).run s).2 with hs1
  obtain ⟨⟨rot, s2⟩, hr⟩ : ∃ q, (Algebra.powerFromBits (wordOf order) bits.val.toArray).run s1 = q := ⟨_, rfl⟩
  obtain ⟨g2, hg2, hg2x⟩ := gl_inv_exact (redW setup.GENERATOR)
  obtain ⟨⟨yInv, s3⟩, hy⟩ : ∃ q, (Build.scale rot (wordOf g2)).run s2 = q := ⟨_, rfl⟩
  obtain ⟨⟨one, s4⟩, h1⟩ : ∃ q, Algebra.extOne.run s3 = q := ⟨_, rfl⟩
  obtain ⟨mv, hmv, hmvv⟩ := vec_push_ok (alloc.vec.Vec.new ExtVar) one (by simp [alloc.vec.Vec.new]; omega)
  have hmv' : mv = vecOfArray #[one] :=
    vec_val_injective (by rw [hmvv, vecOfArray_toList _ (by simp; omega)]; rfl)
  have hmon := monomials_loop_eq alpha s4 one
  set pm := (forIn alpha.val.toArray #[one] fun a m => ForInStep.yield <$> monomialStep a m).run s4 with hpm
  have hpm8 : pm.1.size = 8 := by
    rw [hpm, List.forIn_toArray, monomials_size]
    simp [alpha.property]
  obtain ⟨g4, hg4, hg4x⟩ := gl_inv_exact (redW (UScalar.cast .U64 8#usize))
  obtain ⟨⟨st, s6⟩, hst⟩ : ∃ q, (Build.const (wordOf g4)).run pm.2 = q := ⟨_, rfl⟩
  have hco := coefficients_loop_eq yInv (vecOfArray pm.1) s6 st
  set pc := (forIn (vecOfArray pm.1).val (st, (#[] : _root_.Array ExtVar)) fun m r =>
    ForInStep.yield <$> coefficientStep yInv m r).run s6 with hpc
  have hpc8 : pc.1.2.size = 8 := by
    rw [hpc, coefficients_size, vecOfArray_toList _ (by omega)]
    simp [hpm8]
  obtain ⟨om, hom, homx⟩ := two_adic_generator_eq degrees.FOLD (by simp [degrees.FOLD])
  have hwe := weights_loop_eq (vecOfArray pc.1.2) (by rw [vecOfArray_toList _ (by omega)]; simp [hpc8]) om pc.2
  set pw := (forIn (List.range 8) (#[] : _root_.Array ExtVar) fun v w =>
    ForInStep.yield <$> weightStep (vecOfArray pc.1.2).val.toArray (wordOf om) v w).run pc.2 with hpw
  have hgr := groups_loop_eq values groups (vecOfArray pw.1) hcover pw.2
  set pg := (forIn (groups.val.map (·.val.toArray)) (0, (#[] : _root_.Array ExtVar)) fun ws r =>
    ForInStep.yield <$> groupStep values.val.toArray (vecOfArray pw.1).val.toArray ws r).run pw.2 with hpg
  refine ⟨(.Ok (vecOfArray pg.1.2), pg.2), ?_, ?_⟩
  · unfold setup.check
    simp only [ho, bind_ok, leaf_eq, lift, hc, hceq, massert, ite_true, hc1, hg, horder, hpath, uncurry_eq,
      core.result.Result.Insts.CoreOpsTry.branch, power_from_bits_eq, hr, reduce_eq, hg2, scale_eq, hy, ext_one_eq,
      h1, hmv, hmv', hmon, hg4, constant_eq, hst, hco, hom, hwe, hgr]
  · have hordW : wordOf order = (Replay.twoAdicGenerator (vars.val + 1)).inv := by
      rw [hox.wordOf, hgx.wordOf, hc1v]
    have hgenW : wordOf g2 = Replay.generator.inv := by
      rw [hg2x.wordOf, wordOf_red]
      unfold setup.GENERATOR
      rfl
    have hconstW : wordOf g4 = (Replay.Word.ofNat (2 ^ Replay.foldVars)).inv := by
      rw [hg4x.wordOf, wordOf_red, wordOf_cast]
      rfl
    have homW : wordOf om = Replay.twoAdicGenerator Replay.foldVars := by
      rw [homx.wordOf]
      unfold degrees.FOLD
      rfl
    have hgs : pg.1.2.size ≤ Usize.max := by
      rw [hpg, groups_size]
      have := groups.property
      simp only [List.size_toArray, List.length_nil, List.length_map]
      omega
    have hpc' : (forIn pm.1 (st, (#[] : _root_.Array ExtVar)) fun m r =>
        ForInStep.yield <$> coefficientStep yInv m r).run s6 = pc := by
      rw [hpc, vecOfArray_toList _ (by omega), Array.forIn_toList]
    have hpw' : (forIn (List.range 8) (#[] : _root_.Array ExtVar) fun v w =>
        ForInStep.yield <$> weightStep pc.1.2 (wordOf om) v w).run pc.2 = pw := by
      rw [hpw, vecOfArray_toList _ (by omega), Array.toArray_toList]
    have hpws : pw.1.size ≤ Usize.max := by
      rw [hpw, weights_size]
      simp only [List.size_toArray, List.length_nil, List.length_range]
      omega
    have hpg' : (forIn (groups.val.map (·.val.toArray)).toArray (0, (#[] : _root_.Array ExtVar)) fun ws r =>
        ForInStep.yield <$> groupStep (leafView values path).values pw.1 ws r).run pw.2 = pg := by
      rw [hpg, List.forIn_toArray, vecOfArray_toList _ hpws, Array.toArray_toList, leafView]
    simp only [builtChecked, vecOfArray_toList _ hgs, Array.toArray_toList]
    rw [setupCheck_steps]
    simp only [StateT.run_bind, ← hordW, ← hgenW, ← hconstW, ← homW, forIn_leaf, id_bind, ← hs1, hr, hy, h1, ← hpm,
      hst, hpc', hpw', hpg', StateT.run_pure, id_pure]

theorem check_returns (s : Builder) (root : Aeneas.Std.Array Var 4#usize) (vars : Usize) (bits values : Slice Var)
    (path : Slice (Aeneas.Std.Array Var 4#usize)) (groups : Slice (alloc.vec.Vec ExtVar))
    (alpha : Aeneas.Std.Array ExtVar 3#usize) (y : core.result.Result (alloc.vec.Vec ExtVar) backend.Rejected × Builder)
    (h : setup.check build s root vars bits values path groups alpha = ok y) :
    8 * oracleCount groups.val = values.val.length ∧ vars.val + 1 ≤ 32 := by
  unfold setup.check at h
  obtain ⟨o, ho, h⟩ := bind_eq_ok h
  have hov := check_loop0_sound groups o ho
  rw [leaf_eq, bind_ok, lift_ok, bind_ok] at h
  obtain ⟨i1, hi1, h⟩ := bind_eq_ok h
  have hi1v := mul_eq_ok hi1
  obtain ⟨u, hu, h⟩ := bind_eq_ok h
  have hi1e : i1 = Slice.len values := by
    unfold massert at hu
    split at hu
    · assumption
    · exact absurd hu fail_not_ok
  rw [lift_ok, bind_ok] at h
  obtain ⟨i3, hi3, h⟩ := bind_eq_ok h
  have hi3v := add_eq_ok hi3
  obtain ⟨g, hg, -⟩ := bind_eq_ok h
  have h32 := two_adic_generator_returns i3 g hg
  have hlen : i1.val = values.val.length := by rw [hi1e, Slice.len_val]
  refine ⟨?_, ?_⟩
  · rw [← hlen, hi1v, hov, show (8#usize).val = 8 from rfl]
    ring
  · rw [hi3v] at h32
    simpa using h32

/-- The arguments of `check`, as a tuple. -/
abbrev CheckArgs := Builder × Aeneas.Std.Array Var 4#usize × Usize × Slice Var × Slice Var ×
  Slice (Aeneas.Std.Array Var 4#usize) × Slice (alloc.vec.Vec ExtVar) × Aeneas.Std.Array ExtVar 3#usize

theorem check_sound : Sound
    (fun x : CheckArgs => setup.check build x.1 x.2.1 x.2.2.1 x.2.2.2.1 x.2.2.2.2.1 x.2.2.2.2.2.1 x.2.2.2.2.2.2.1
      x.2.2.2.2.2.2.2) builtChecked
    (fun x => let p := (setupCheck (digestOf x.2.1) x.2.2.1.val x.2.2.2.1.val.toArray
      (leafView x.2.2.2.2.1 x.2.2.2.2.2.1) (x.2.2.2.2.2.2.1.val.map (·.val.toArray)).toArray
      x.2.2.2.2.2.2.2.val.toArray).run x.1; (.ok p.1, p.2)) :=
  sound_of_eq_on (fun x => 8 * oracleCount x.2.2.2.2.2.2.1.val = x.2.2.2.2.1.val.length ∧ x.2.2.1.val + 1 ≤ 32)
    (fun x h => check_eq x.1 x.2.1 x.2.2.1 x.2.2.2.1 x.2.2.2.2.1 x.2.2.2.2.2.1 x.2.2.2.2.2.2.1 x.2.2.2.2.2.2.2 h.1 h.2)
    (fun x y h => check_returns x.1 x.2.1 x.2.2.1 x.2.2.2.1 x.2.2.2.2.1 x.2.2.2.2.2.1 x.2.2.2.2.2.2.1 x.2.2.2.2.2.2.2 y h)

/-- The groups cover the leaf's values, eight per oracle, and the setup has at most 31 variables.
The verifier meets both: `LeafView::read` rejects a leaf without `(κ + RUN_ORACLES)·8` values, the
groups are the `κ` commitment rows and the `RUN_ORACLES` run weights, and `Relation::new` refuses more
than 31 setup variables. -/
theorem check_complete : Complete
    (fun x : CheckArgs => 8 * oracleCount x.2.2.2.2.2.2.1.val = x.2.2.2.2.1.val.length ∧ x.2.2.1.val + 1 ≤ 32)
    (fun x : CheckArgs => setup.check build x.1 x.2.1 x.2.2.1 x.2.2.2.1 x.2.2.2.2.1 x.2.2.2.2.2.1 x.2.2.2.2.2.2.1
      x.2.2.2.2.2.2.2) builtChecked
    (fun x => let p := (setupCheck (digestOf x.2.1) x.2.2.1.val x.2.2.2.1.val.toArray
      (leafView x.2.2.2.2.1 x.2.2.2.2.2.1) (x.2.2.2.2.2.2.1.val.map (·.val.toArray)).toArray
      x.2.2.2.2.2.2.2.val.toArray).run x.1; (.ok p.1, p.2)) :=
  complete_of_eq check_sound ({}, Aeneas.Std.Array.make 4#usize [0, 0, 0, 0], 0#usize, emptySlice, emptySlice,
      emptySlice, emptySlice, Aeneas.Std.Array.make 3#usize [zeroExt, zeroExt, zeroExt])
    (by simp [oracleCount, emptySlice])
    (fun x h => check_eq x.1 x.2.1 x.2.2.1 x.2.2.2.1 x.2.2.2.2.1 x.2.2.2.2.2.1 x.2.2.2.2.2.2.1 x.2.2.2.2.2.2.2 h.1 h.2)

end

end CompressionLeanRust
