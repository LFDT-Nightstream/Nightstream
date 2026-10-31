import CompressionLeanRust.Algebra
import CompressionLeanRust.Loops

/-! The translated `compress`, `hash_leaf` and `merkle_root` of `neo_verified::hash`, run on `build`,
equal their copies in `CompressionLean/Ports/Circuit/Hash.lean`. A copy reads a translated `[F; 4]`
as `digestOf` reads it and a slice as its list. The three always return, so their `Complete` premise
is `True`.

Each copy is first put in a normal form (`hashLeaf_eq`, `merkleRoot_eq`): a `for` loop over a list
whose body is one named step (`leafStep`, `levelStep`, `laneStep`). `loop_forIn_spec` then relates
each translated loop that threads the builder to that `for` loop. `Native.lean` proves the same three
functions, run on `Native`, against their `Replay` copies. -/

namespace CompressionLeanRust

open Aeneas Aeneas.Std Result
open neo_verified
open CompressionLean
open CompressionLean.Circuit

/-- A translated `[F; 4]` as the copies' digest. -/
def digestOf (a : Aeneas.Std.Array Var 4#usize) : DigestVar := fun i => a.val.getD i 0

/-- A run that returns a translated digest, as the copy's run. -/
def builtDigest (x : Aeneas.Std.Array Var 4#usize × Builder) : DigestVar × Builder := (digestOf x.1, x.2)

theorem builtDigest_injective : Function.Injective builtDigest := by
  rintro ⟨a, s⟩ ⟨c, t⟩ h
  simp only [builtDigest, Prod.mk.injEq] at h
  obtain ⟨h, rfl⟩ := h
  congr 1
  apply Aeneas.Std.Array.ext
  apply List.ext_getElem (by rw [a.property, c.property])
  intro i hi _
  have := congrFun h ⟨i, by rw [a.property] at hi; exact hi⟩
  simp only [digestOf] at this
  rwa [List.getD_eq_getElem _ _ hi, List.getD_eq_getElem _ _ (by rw [c.property]; rw [a.property] at hi; exact hi)]
    at this

theorem digest_len : circuit.DIGEST_LEN = 4#usize := by
  unfold circuit.DIGEST_LEN
  rfl

theorem rate_eq : circuit.RATE = 12#usize := by
  unfold circuit.RATE
  rfl

/-- One step of `compress`'s lane loop. -/
def compressStep {α : Type} [Inhabited α] (left right : Aeneas.Std.Array α 4#usize) (k : ℕ)
    (state : Aeneas.Std.Array α 16#usize) : Aeneas.Std.Array α 16#usize :=
  setN (setN state k left.val[k]!) (4 + k) right.val[k]!

theorem compress_loop_eq {α : Type} [Inhabited α] (left right : Aeneas.Std.Array α 4#usize)
    (state : Aeneas.Std.Array α 16#usize) :
    hash.compress_loop left right state 0#usize = ok (upto 4 1 (compressStep left right) 4 0 state) := by
  unfold hash.compress_loop
  refine loop_upto _ (fun (a : Aeneas.Std.Array α 16#usize) (i : Usize) => (a, i)) 4 1 1#usize rfl (by omega)
    (by omega) (compressStep left right) id ?_ ?_ 4 state 0#usize 0 rfl rfl (by omega)
  · intro a i h _
    have h4 : 4 + i.val < 16 := by omega
    have hadd : (core.num.Usize.wrapping_add 4#usize i).val = 4 + i.val := by
      rw [wrapping_add_val _ _ (by scalar_tac)]
      rfl
    simp only [hash.compress_loop.body, digest_len, show i < 4#usize by scalar_tac, ite_true,
      index_eq left i (by scalar_tac), index_eq right i (by scalar_tac), update_eq a i _ (by scalar_tac), bind_ok,
      lift]
    rw [update_eq _ _ _ (by rw [hadd]; scalar_tac), hadd, bind_ok]
    rfl
  · intro a i h
    simp only [hash.compress_loop.body, digest_len, show ¬ i < 4#usize by scalar_tac, ite_false]
    rfl

theorem compress_lanes (left right : Aeneas.Std.Array Var 4#usize) (z : Var) :
    lanesOf (upto 4 1 (compressStep left right) 4 0 (Aeneas.Std.Array.repeat 16#usize z)) = fun (lane : Fin 16) =>
      if h : lane.val < 4 then digestOf left ⟨lane, h⟩
      else if h' : lane.val < 8 then digestOf right ⟨lane - 4, by omega⟩ else z := by
  have hl := left.property
  have hr := right.property
  obtain ⟨l, hl'⟩ := left
  obtain ⟨r, hr'⟩ := right
  funext lane
  fin_cases lane <;> simp [lanesOf, upto, compressStep, digestOf, Aeneas.Std.Array.repeat]

theorem compress_eq (s : Builder) (left right : Aeneas.Std.Array Var 4#usize) :
    ∃ y, hash.compress build s left right = ok y ∧
      builtDigest y = (compress (digestOf left) (digestOf right)).run s := by
  obtain ⟨⟨z, s1⟩, hz⟩ : ∃ p, (Build.const (wordOf 0#u64)).run s = p := ⟨_, rfl⟩
  obtain ⟨⟨out, s2⟩, hp⟩ : ∃ p, (Build.permute (lanesOf (upto 4 1 (compressStep left right) 4 0
      (Aeneas.Std.Array.repeat 16#usize z)))).run s1 = p := ⟨_, rfl⟩
  have hm : ((fun x => lanes16 x) <$> Build.permute (lanesOf (upto 4 1 (compressStep left right) 4 0
      (Aeneas.Std.Array.repeat 16#usize z)))).run s1 = (lanes16 out, s2) := by
    rw [StateT.run_map, hp]
    rfl
  unfold hash.compress
  simp only [build_constant, hz, bind_ok, uncurry_eq, compress_loop_eq, build_permute, hm,
    index_eq (lanes16 out) 0#usize (by scalar_tac), index_eq (lanes16 out) 1#usize (by scalar_tac),
    index_eq (lanes16 out) 2#usize (by scalar_tac), index_eq (lanes16 out) 3#usize (by scalar_tac)]
  refine ⟨_, rfl, ?_⟩
  simp only [compress, StateT.run_bind, show (0 : Replay.Word) = wordOf 0#u64 from rfl, hz, id_bind]
  rw [← compress_lanes, hp]
  simp only [builtDigest]
  refine Prod.ext ?_ rfl
  funext i
  fin_cases i <;> rfl

theorem compress_sound : Sound (fun x : Builder × Aeneas.Std.Array Var 4#usize × Aeneas.Std.Array Var 4#usize =>
    hash.compress build x.1 x.2.1 x.2.2) builtDigest (fun x => (compress (digestOf x.2.1) (digestOf x.2.2)).run x.1) := by
  rintro ⟨s, l, r⟩ y h
  obtain ⟨y', h', hv⟩ := compress_eq s l r
  simp only at h
  rw [h', Result.ok.injEq] at h
  subst h
  exact hv

theorem compress_complete : Complete (fun _ => True)
    (fun x : Builder × Aeneas.Std.Array Var 4#usize × Aeneas.Std.Array Var 4#usize => hash.compress build x.1 x.2.1 x.2.2)
    builtDigest (fun x => (compress (digestOf x.2.1) (digestOf x.2.2)).run x.1) :=
  complete_of compress_sound
    ⟨({}, Aeneas.Std.Array.make 4#usize [0, 0, 0, 0], Aeneas.Std.Array.make 4#usize [0, 0, 0, 0]), trivial⟩
    fun x _ => let ⟨y, h, _⟩ := compress_eq x.1 x.2.1 x.2.2; ⟨y, h⟩

/-! ## `hash_leaf` -/

/-- `state` with lane `l` set to `chunk[l]`, as the copy's inner loop of `hashLeaf` writes it. -/
def fill {α : Type} (state chunk : List α) : List α :=
  chunk.zipIdx.foldl (fun st x => st.set x.2 x.1) state

theorem fill_snoc {α : Type} (state chunk : List α) (v : α) :
    fill state (chunk ++ [v]) = (fill state chunk).set chunk.length v := by
  simp [fill, List.zipIdx_append, List.foldl_append]

theorem fill_length {α : Type} (state chunk : List α) : (fill state chunk).length = state.length := by
  induction chunk using List.reverseRecOn with
  | nil => rfl
  | append_singleton c v ih => rw [fill_snoc, List.length_set, ih]

theorem hash_leaf_inner {α : Type} (values : Slice α) (state : Aeneas.Std.Array α 16#usize) (start : Usize)
    (hs : start.val < values.val.length) :
    ∃ st m, hash.hash_leaf_loop0_loop0 values state start 0#usize = ok (st, m) ∧
      st.val = fill state.val ((values.val.drop start.val).take 12) ∧
      m.val = min 12 (values.val.length - start.val) := by
  unfold hash.hash_leaf_loop0_loop0
  have hlen := values.property
  have hsub : (core.num.Usize.wrapping_sub (Slice.len values) start).val = values.val.length - start.val := by
    rw [wrapping_sub_val _ _ (by simp; omega)]
    simp
  have hfill := fill_length state.val ((values.val.drop start.val).take 12)
  have key := loop_total
    (fun (a : Aeneas.Std.Array α 16#usize × Usize) => hash.hash_leaf_loop0_loop0.body values start a.1 a.2)
    (fun n (a : Aeneas.Std.Array α 16#usize × Usize) => a.2.val ≤ min 12 (values.val.length - start.val) ∧
      n = min 12 (values.val.length - start.val) - a.2.val ∧
      a.1.val = fill state.val ((values.val.drop start.val).take a.2.val))
    (fun (b : Aeneas.Std.Array α 16#usize × Usize) =>
      b.1.val = fill state.val ((values.val.drop start.val).take 12) ∧ b.2.val = min 12 (values.val.length - start.val))
    (by
      rintro n ⟨st, lane⟩ ⟨hle, rfl, hst⟩
      dsimp only at hle hst ⊢
      by_cases hm : lane.val < min 12 (values.val.length - start.val)
      · right
        have hv : start.val + lane.val < values.val.length := by omega
        have hadd : (core.num.Usize.wrapping_add start lane).val = start.val + lane.val := by
          rw [wrapping_add_le _ _ (by omega)]
        have hl1 := add_one_val lane (by omega)
        refine ⟨(setN st lane.val values.val[start.val + lane.val], core.num.Usize.wrapping_add lane 1#usize),
          min 12 (values.val.length - start.val) - (lane.val + 1), ?_, by omega, by dsimp only; omega,
          by dsimp only; omega, ?_⟩
        · simp only [hash.hash_leaf_loop0_loop0.body, rate_eq, show lane < 12#usize by scalar_tac, ite_true, lift,
            bind_ok, show lane < core.num.Usize.wrapping_sub (Slice.len values) start by scalar_tac,
            slice_index values _ (by rw [hadd]; exact hv), hadd, update_eq st lane _ (by scalar_tac)]
        · dsimp only
          rw [setN_val, hst, hl1, take_drop_succ _ _ _ hv, fill_snoc, List.length_take, List.length_drop,
            Nat.min_eq_left (by omega)]
      · left
        have heq : lane.val = min 12 (values.val.length - start.val) := by omega
        refine ⟨(st, lane), ?_, ?_, heq⟩
        · by_cases h12 : lane.val < 12
          · simp only [hash.hash_leaf_loop0_loop0.body, rate_eq, show lane < 12#usize by scalar_tac, ite_true, lift,
              bind_ok, show ¬ lane < core.num.Usize.wrapping_sub (Slice.len values) start by scalar_tac, ite_false]
          · simp only [hash.hash_leaf_loop0_loop0.body, rate_eq, show ¬ lane < 12#usize by scalar_tac, ite_false]
        · rw [hst, heq, ← List.length_drop, take_min_length])
    _ (state, 0#usize) ⟨by simp, rfl, by simp [fill]⟩
  obtain ⟨y, hy, hQ⟩ := key
  exact ⟨y.1, y.2, hy, hQ⟩

/-- One chunk of the copy's `hashLeaf`: fill, then permute. -/
def leafStep (chunk : _root_.Array Var) (state : _root_.Array Var) : Build (_root_.Array Var) := do
  let out ← Build.permute fun lane => (_root_.Array.mk (fill state.toList chunk.toList))[lane]!
  pure (_root_.Array.ofFn out)

theorem chunks_length (values : List Var) : (chunks values.toArray).toList.length = (values.length + 11) / 12 := by
  simp [chunks, rate]

theorem chunks_getElem (values : List Var) (k : ℕ) (h : k < (chunks values.toArray).toList.length) :
    ((chunks values.toArray).toList[k]).toList = (values.drop (12 * k)).take 12 := by
  simp [chunks, rate]

theorem hash_leaf_loop_eq (s : Builder) (values : Slice Var) (state : Aeneas.Std.Array Var 16#usize) :
    ∃ y, hash.hash_leaf_loop0 build s values state 0#usize = ok y ∧
      (let p := (forIn (chunks values.val.toArray).toList (_root_.Array.mk state.val) fun chunk r =>
        ForInStep.yield <$> leafStep chunk r).run s; y.1 = p.2 ∧ y.2.val = p.1.toList) := by
  unfold hash.hash_leaf_loop0
  have hlen := values.property
  have := loop_forIn_spec (fun (a : Builder × Aeneas.Std.Array Var 16#usize × Usize) =>
      hash.hash_leaf_loop0.body build values a.1 a.2.1 a.2.2)
    (fun a s r k => a.1 = s ∧ a.2.1.val = r.toList ∧ a.2.2.val = min (12 * k) values.val.length)
    (fun b r s => b.1 = s ∧ b.2.val = r.toList) (chunks values.val.toArray).toList leafStep ?_ ?_
    (s, state, 0#usize) s (_root_.Array.mk state.val) 0 ⟨rfl, rfl, by simp⟩
  · simpa using this
  · rintro ⟨b, st, start⟩ s r k hk ⟨rfl, hst, hstart⟩
    dsimp only at hst hstart ⊢
    rw [chunks_length] at hk
    have hs : start.val < values.val.length := by omega
    have h12 : start.val = 12 * k := by omega
    obtain ⟨st1, m, hin, hst1, hm⟩ := hash_leaf_inner values st start hs
    have hfill : st1.val = fill r.toList ((chunks values.val.toArray).toList[k]).toList := by
      rw [hst1, hst, chunks_getElem _ _ (by rw [chunks_length]; omega), h12]
    obtain ⟨⟨out, s2⟩, hp⟩ : ∃ p, (Build.permute (lanesOf st1)).run b = p := ⟨_, rfl⟩
    have hperm : build.permute b st1 = ok (lanes16 out, s2) := by
      rw [build_permute, StateT.run_map, hp]
      rfl
    have hrun : (leafStep (chunks values.val.toArray).toList[k] r).run b = (_root_.Array.ofFn out, s2) := by
      simp only [leafStep, StateT.run_bind, ← lanesOf_eq st1 _ hfill, hp]
      rfl
    refine ⟨(s2, lanes16 out, core.num.Usize.wrapping_add start m), ?_, ?_, ?_, ?_⟩
    · have hlt : start < Slice.len values := by rw [UScalar.lt_equiv, Slice.len_val]; exact hs
      simp only [hash.hash_leaf_loop0.body, hlt, ite_true, hin, bind_ok, uncurry_eq, hperm, lift]
    · rw [hrun]
    · rw [hrun]
      show (lanes16 out).val = (_root_.Array.ofFn out).toList
      rw [Array.toList_ofFn]
      rfl
    · dsimp only
      rw [wrapping_add_le _ _ (by omega)]
      omega
  · rintro ⟨b, st, start⟩ s r k ⟨rfl, hst, hstart⟩ hk
    dsimp only at hst hstart
    rw [chunks_length] at hk
    refine ⟨(b, st), ?_, rfl, hst⟩
    have hlt : ¬ start < Slice.len values := by
      rw [UScalar.lt_equiv, Slice.len_val, Slice.length]
      omega
    simp only [hash.hash_leaf_loop0.body, hlt, ite_false]

theorem toList_foldl_set (xs : List (Var × ℕ)) (state : _root_.Array Var) :
    (xs.foldl (fun b a => b.set! a.2 a.1) state).toList = xs.foldl (fun st x => st.set x.2 x.1) state.toList := by
  induction xs generalizing state with
  | nil => rfl
  | cons x xs ih =>
    simp only [List.foldl_cons]
    rw [ih, Array.set!_eq_setIfInBounds, Array.toList_setIfInBounds]

theorem fill_eq (state chunk : _root_.Array Var) :
    (forIn chunk.zipIdx state fun x (r : _root_.Array Var) => match x with
      | (value, lane) => (pure (ForInStep.yield (r.set! lane value)) : Build (ForInStep (_root_.Array Var)))) =
      pure ⟨fill state.toList chunk.toList⟩ := by
  show (forIn chunk.zipIdx state fun x (r : _root_.Array Var) =>
    (pure (ForInStep.yield (r.set! x.2 x.1)) : Build (ForInStep (_root_.Array Var)))) = _
  simp only [Array.forIn_pure_yield_eq_foldl]
  rw [← Array.foldl_toList, fill, Array.toList_zipIdx]
  congr 1
  exact Array.toList_inj.mp (by rw [toList_foldl_set])

theorem hashLeaf_eq (values : _root_.Array Var) :
    hashLeaf values =
      (do
        let z ← Build.const 0
        let st ← forIn (chunks values).toList (_root_.Array.replicate 16 z) fun chunk r =>
          ForInStep.yield <$> leafStep chunk r
        pure fun (lane : Fin 4) => st[lane.val]!) := by
  unfold hashLeaf
  refine bind_congr fun z => ?_
  rw [Array.forIn_toList]
  refine congrArg (· >>= _) ?_
  refine congrArg (forIn _ _) ?_
  funext chunk r
  simp only [fill_eq, leafStep, pure_bind, map_bind, map_pure]

theorem hash_leaf_eq (s : Builder) (values : Slice Var) :
    ∃ y, hash.hash_leaf build s values = ok y ∧ builtDigest y = (hashLeaf values.val.toArray).run s := by
  obtain ⟨⟨z, s1⟩, hz⟩ : ∃ p, (Build.const (wordOf 0#u64)).run s = p := ⟨_, rfl⟩
  obtain ⟨y, hy, hy1, hy2⟩ := hash_leaf_loop_eq s1 values (Aeneas.Std.Array.repeat 16#usize z)
  have hinit : _root_.Array.mk (Aeneas.Std.Array.repeat 16#usize z).val = _root_.Array.replicate 16 z := by
    exact Array.toList_inj.mp (by simp [Aeneas.Std.Array.repeat])
  rw [hinit] at hy1 hy2
  refine ⟨(Aeneas.Std.Array.make 4#usize [y.2.val[0]!, y.2.val[1]!, y.2.val[2]!, y.2.val[3]!], y.1), ?_, ?_⟩
  · unfold hash.hash_leaf
    simp only [build_constant, hz, bind_ok, uncurry_eq, hy, index_eq y.2 0#usize (by scalar_tac),
      index_eq y.2 1#usize (by scalar_tac), index_eq y.2 2#usize (by scalar_tac), index_eq y.2 3#usize (by scalar_tac)]
    rfl
  · rw [hashLeaf_eq]
    obtain ⟨⟨r, s2⟩, hp⟩ : ∃ p, (forIn (chunks values.val.toArray).toList (_root_.Array.replicate 16 z)
        fun chunk r => ForInStep.yield <$> leafStep chunk r).run s1 = p := ⟨_, rfl⟩
    rw [hp] at hy1 hy2
    simp only [StateT.run_bind, show (0 : Replay.Word) = wordOf 0#u64 from rfl, hz, hp, id_bind]
    simp only [builtDigest, hy1]
    refine Prod.ext ?_ rfl
    funext lane
    fin_cases lane <;> simp [digestOf, hy2, Aeneas.Std.Array.make, getElem!_def, id_pure]

theorem hash_leaf_sound : Sound (fun x : Builder × Slice Var => hash.hash_leaf build x.1 x.2) builtDigest
    (fun x => (hashLeaf x.2.val.toArray).run x.1) :=
  sound_of_eq fun x => hash_leaf_eq x.1 x.2

theorem hash_leaf_complete : Complete (fun _ => True) (fun x : Builder × Slice Var => hash.hash_leaf build x.1 x.2)
    builtDigest (fun x => (hashLeaf x.2.val.toArray).run x.1) :=
  complete_of_eq hash_leaf_sound ({}, emptySlice) trivial fun x _ => hash_leaf_eq x.1 x.2

/-! ## `merkle_root` -/

theorem digestOf_getElem (a : Aeneas.Std.Array Var 4#usize) (k : ℕ) (h : k < 4) :
    digestOf a ⟨k, h⟩ = a.val[k]! := by
  simp [digestOf, List.getD_eq_getElem?_getD, getElem!_def]
  cases a.val[k]? <;> rfl

theorem ofFn_digestOf (a : Aeneas.Std.Array Var 4#usize) : (_root_.Array.ofFn (digestOf a)).toList = a.val := by
  apply List.ext_getElem (by simp [a.property])
  intro i h1 h2
  have : i < 4 := by rw [a.property] at h2; exact h2
  simp only [Array.toList_ofFn, List.getElem_ofFn, digestOf, List.getD_eq_getElem?_getD,
    List.getElem?_eq_getElem h2, Option.getD_some]

theorem digestOf_eq (a : Aeneas.Std.Array Var 4#usize) (r : _root_.Array Var) (h : a.val = r.toList) :
    digestOf a = fun lane => r[lane]! := by
  funext lane
  simp [digestOf, h, List.getD_eq_getElem?_getD, getElem!_def]
  cases r[lane.val]? <;> rfl

/-- One lane of a level of the copy's `merkleRoot`. -/
def laneStep (bit : Var) (node sibling : DigestVar) (lane : Fin 4) (r : _root_.Array Var × _root_.Array Var) :
    Build (_root_.Array Var × _root_.Array Var) := do
  let l ← Build.select bit (node lane) (sibling lane)
  let both ← Build.add (node lane) (sibling lane)
  let x ← Build.sub both (r.1.set! lane l)[lane]!
  pure (r.1.set! lane l, r.2.set! lane x)

/-- One level of the copy's `merkleRoot`. -/
def levelStep (x : Var × DigestVar) (node : DigestVar) : Build DigestVar := do
  let lr ← forIn (List.finRange 4) (_root_.Array.ofFn node, _root_.Array.ofFn node) fun lane r =>
    ForInStep.yield <$> laneStep x.1 node x.2 lane r
  compress (fun lane => lr.1[lane]!) (fun lane => lr.2[lane]!)

theorem merkleRoot_eq (leaf : DigestVar) (bits : _root_.Array Var) (path : _root_.Array DigestVar) :
    merkleRoot leaf bits path = forIn (bits.zip path).toList leaf fun x node => ForInStep.yield <$> levelStep x node := by
  unfold merkleRoot
  rw [Array.forIn_toList, bind_pure]
  refine congrArg (forIn _ _) ?_
  funext x node
  obtain ⟨bit, sibling⟩ := x
  simp only [levelStep, laneStep, map_bind, bind_pure_comp, Functor.map_map]

theorem merkle_inner_eq (s : Builder) (node sibling : Aeneas.Std.Array Var 4#usize) (bit : Var) :
    ∃ y, hash.merkle_root_loop0_loop0 build s node bit sibling node node 0#usize = ok y ∧
      (let p := (forIn (List.finRange 4) (_root_.Array.ofFn (digestOf node), _root_.Array.ofFn (digestOf node))
          fun lane r => ForInStep.yield <$> laneStep bit (digestOf node) (digestOf sibling) lane r).run s;
        y.1 = p.2 ∧ y.2.1.val = p.1.1.toList ∧ y.2.2.val = p.1.2.toList) := by
  unfold hash.merkle_root_loop0_loop0
  have := loop_forIn_spec (fun (a : Builder × Aeneas.Std.Array Var 4#usize × Aeneas.Std.Array Var 4#usize × Usize) =>
      hash.merkle_root_loop0_loop0.body build node bit sibling a.1 a.2.1 a.2.2.1 a.2.2.2)
    (fun a s r k => a.1 = s ∧ a.2.1.val = r.1.toList ∧ a.2.2.1.val = r.2.toList ∧ a.2.2.2.val = k)
    (fun b r s => b.1 = s ∧ b.2.1.val = r.1.toList ∧ b.2.2.val = r.2.toList) (List.finRange 4)
    (laneStep bit (digestOf node) (digestOf sibling)) ?_ ?_
    (s, node, node, 0#usize) s (_root_.Array.ofFn (digestOf node), _root_.Array.ofFn (digestOf node)) 0
    ⟨rfl, (ofFn_digestOf node).symm, (ofFn_digestOf node).symm, rfl⟩
  · simpa using this
  · rintro ⟨b, left, right, lane⟩ s r k hk ⟨rfl, hl, hr, hlane⟩
    dsimp only at hl hr hlane
    rw [List.length_finRange] at hk
    obtain ⟨⟨l, s1⟩, h1⟩ : ∃ p, (Build.select bit node.val[k]! sibling.val[k]!).run b = p := ⟨_, rfl⟩
    obtain ⟨⟨both, s2⟩, h2⟩ : ∃ p, (Build.add node.val[k]! sibling.val[k]!).run s1 = p := ⟨_, rfl⟩
    obtain ⟨⟨x, s3⟩, h3⟩ : ∃ p, (Build.sub both l).run s2 = p := ⟨_, rfl⟩
    have hk4 : k < r.1.size := by rw [← Array.length_toList, ← hl, left.property]; exact hk
    have hlk : (setN left k l).val[k]! = l := by
      simp [setN_val, hl, hk4]
    have hrun : (laneStep bit (digestOf node) (digestOf sibling) ⟨k, hk⟩ r).run b =
        ((r.1.set! k l, r.2.set! k x), s3) := by
      have hset : (r.1.set! k l)[(⟨k, hk⟩ : Fin 4)]! = l := by simp [hk4]
      simp only [laneStep, StateT.run_bind, digestOf_getElem node k hk, digestOf_getElem sibling k hk, h1, id_bind, h2,
        hset, h3]
      rfl
    refine ⟨(s3, setN left k l, setN right k x, core.num.Usize.wrapping_add lane 1#usize), ?_, ?_, ?_, ?_, ?_⟩
    · simp only [hash.merkle_root_loop0_loop0.body, digest_len, show lane < 4#usize by scalar_tac, ite_true,
        index_eq node lane (by scalar_tac), index_eq sibling lane (by scalar_tac), hlane, build_select, h1, bind_ok,
        uncurry_eq, update_eq left lane _ (by scalar_tac), build_add, h2, index_eq (setN left k l) lane (by scalar_tac),
        hlk, build_sub, h3, update_eq right lane _ (by scalar_tac), lift]
    all_goals simp only [List.getElem_finRange, Fin.cast_mk]
    · rw [hrun]
    · rw [hrun]
      simp [hl, Array.toList_setIfInBounds]
    · rw [hrun]
      simp [hr, Array.toList_setIfInBounds]
    · rw [add_one_val lane (by scalar_tac), hlane]
  · rintro ⟨b, left, right, lane⟩ s r k ⟨rfl, hl, hr, hlane⟩ hk
    dsimp only at hl hr hlane
    rw [List.length_finRange] at hk
    refine ⟨(b, left, right), ?_, rfl, hl, hr⟩
    simp only [hash.merkle_root_loop0_loop0.body, digest_len, show ¬ lane < 4#usize by scalar_tac, ite_false]

theorem level_run (b : Builder) (node sibling : Aeneas.Std.Array Var 4#usize) (bit : Var) :
    ∃ y, (do
        let p ← hash.merkle_root_loop0_loop0 build b node bit sibling node node 0#usize
        hash.compress build p.1 p.2.1 p.2.2) = ok y ∧
      builtDigest y = (levelStep (bit, digestOf sibling) (digestOf node)).run b := by
  obtain ⟨y, hy, hy1, hy2, hy3⟩ := merkle_inner_eq b node sibling bit
  obtain ⟨y', hy', hv⟩ := compress_eq y.1 y.2.1 y.2.2
  refine ⟨y', by rw [hy, bind_ok, hy'], ?_⟩
  obtain ⟨⟨lr, s1⟩, hp⟩ : ∃ p, (forIn (List.finRange 4) (_root_.Array.ofFn (digestOf node), _root_.Array.ofFn (digestOf node))
      fun lane r => ForInStep.yield <$> laneStep bit (digestOf node) (digestOf sibling) lane r).run b = p := ⟨_, rfl⟩
  rw [hp] at hy1 hy2 hy3
  rw [hv, digestOf_eq _ _ hy2, digestOf_eq _ _ hy3, hy1]
  simp only [levelStep, StateT.run_bind, hp, id_bind]

theorem merkle_root_eq (s : Builder) (leaf : Aeneas.Std.Array Var 4#usize) (bits : Slice Var)
    (path : Slice (Aeneas.Std.Array Var 4#usize)) :
    ∃ y, hash.merkle_root build s leaf bits path = ok y ∧
      builtDigest y = (merkleRoot (digestOf leaf) bits.val.toArray (path.val.map digestOf).toArray).run s := by
  unfold hash.merkle_root hash.merkle_root_loop0
  have := loop_forIn_spec (fun (a : Builder × Aeneas.Std.Array Var 4#usize × Usize) =>
      hash.merkle_root_loop0.body build bits path a.1 a.2.1 a.2.2)
    (fun a s r k => a.1 = s ∧ digestOf a.2.1 = r ∧ a.2.2.val = k)
    (fun b r s => digestOf b.1 = r ∧ b.2 = s) (bits.val.zip (path.val.map digestOf)) levelStep ?_ ?_
    (s, leaf, 0#usize) s (digestOf leaf) 0 ⟨rfl, rfl, rfl⟩
  · obtain ⟨y, hy, h1, h2⟩ := this
    refine ⟨(y.1, y.2), hy, ?_⟩
    rw [merkleRoot_eq]
    simp only [List.drop_zero] at h1 h2
    simp only [builtDigest, h1, h2, Array.toList_zip]
    rfl
  · rintro ⟨b, node, level⟩ s r k hk ⟨rfl, rfl, hlevel⟩
    dsimp only at hlevel
    simp only [List.length_zip, List.length_map] at hk
    have hb : level < Slice.len bits := by rw [UScalar.lt_equiv, Slice.len_val, Slice.length]; omega
    have hpth : level < Slice.len path := by rw [UScalar.lt_equiv, Slice.len_val, Slice.length]; omega
    obtain ⟨y, hy, hv⟩ := level_run b node path.val[k] bits.val[k]
    obtain ⟨p, hp, hy⟩ := bind_eq_ok hy
    refine ⟨(y.2, y.1, core.num.Usize.wrapping_add level 1#usize), ?_, ?_, ?_, ?_⟩
    · simp only [hash.merkle_root_loop0.body, hb, hpth, ite_true, slice_index bits level (by omega),
        slice_index path level (by omega), hlevel, hp, bind_ok, uncurry_eq, hy, lift]
    · simp only [List.getElem_zip, List.getElem_map, ← hv]
      rfl
    · simp only [List.getElem_zip, List.getElem_map, ← hv]
      rfl
    · rw [add_one_val level (by scalar_tac), hlevel]
  · rintro ⟨b, node, level⟩ s r k ⟨rfl, rfl, hlevel⟩ hk
    dsimp only at hlevel
    simp only [List.length_zip, List.length_map] at hk
    refine ⟨(node, b), ?_, rfl, rfl⟩
    by_cases hb : level < Slice.len bits
    · have hpth : ¬ level < Slice.len path := by rw [UScalar.lt_equiv, Slice.len_val, Slice.length]; scalar_tac
      simp only [hash.merkle_root_loop0.body, hb, hpth, ite_true, ite_false]
    · simp only [hash.merkle_root_loop0.body, hb, ite_false]

theorem merkle_root_sound : Sound (fun x : Builder × Aeneas.Std.Array Var 4#usize × Slice Var ×
      Slice (Aeneas.Std.Array Var 4#usize) => hash.merkle_root build x.1 x.2.1 x.2.2.1 x.2.2.2) builtDigest
    (fun x => (merkleRoot (digestOf x.2.1) x.2.2.1.val.toArray (x.2.2.2.val.map digestOf).toArray).run x.1) :=
  sound_of_eq fun x => merkle_root_eq x.1 x.2.1 x.2.2.1 x.2.2.2

theorem merkle_root_complete : Complete (fun _ => True) (fun x : Builder × Aeneas.Std.Array Var 4#usize × Slice Var ×
      Slice (Aeneas.Std.Array Var 4#usize) => hash.merkle_root build x.1 x.2.1 x.2.2.1 x.2.2.2) builtDigest
    (fun x => (merkleRoot (digestOf x.2.1) x.2.2.1.val.toArray (x.2.2.2.val.map digestOf).toArray).run x.1) :=
  complete_of_eq merkle_root_sound ({}, Aeneas.Std.Array.make 4#usize [0, 0, 0, 0], emptySlice, emptySlice) trivial
    fun x _ => merkle_root_eq x.1 x.2.1 x.2.2.1 x.2.2.2

end CompressionLeanRust
