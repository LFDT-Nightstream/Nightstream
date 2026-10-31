import CompressionLeanRust.Hash

/-! The translated `hash_leaf`, `compress` and `merkle_root` of `neo_verified::hash`, run on `nativeB`
(Aeneas's translation of `Native`, the verifier's backend on values), equal their copies
`Replay.hashLeaf`, `Replay.compress` and `Replay.merkleRoot` in
`CompressionLean/Ports/Replay/Transcript.lean`. These are the Merkle hash that `MerkleBinds` reads.

A copy reads each input word as `wordOf` reads it, so a word of `p` or more reads as its value mod p,
as `Native` reads it. Each output word of a permutation is canonical, and `nativeValues` reads it word
for word. `Replay.merkleRoot` reads a natural index and every level of its path; Rust reads one field
bit per level, zero or not as `Native::select` reads it (`bitIndex`), and stops at the shorter of its
bits and its path. With no level Rust returns the leaf as given, so the copy returns the leaf's words
there. The three always return, so their `Complete` premise is `True`. -/

namespace CompressionLeanRust

open Aeneas Aeneas.Std Result
open neo_verified
open CompressionLean
open NightstreamFPrime.Spec (goldilocksModulus)

abbrev nativeB := native.Native.Insts.Neo_verifiedBackendBackendGlExt

/-- A run on `nativeB` that returns words, as their values. -/
def nativeValues {n : Usize} (x : Aeneas.Std.Array field.Gl n × native.Native) : List ℕ := glValues x.1

theorem nativeValues_injective {n : Usize} : Function.Injective (@nativeValues n) := by
  rintro ⟨a, u⟩ ⟨b, v⟩ h
  have hab : a = b := glValues_injective h
  rw [hab]

theorem native_constant (u : native.Native) (v : U64) :
    native.Native.Insts.Neo_verifiedBackendBackendGlExt.constant u v = ok (redW v, u) := by
  simp only [native.Native.Insts.Neo_verifiedBackendBackendGlExt.constant, reduce_eq, bind_ok]

theorem native_permute (u : native.Native) (state : Aeneas.Std.Array field.Gl 16#usize) :
    native.Native.Insts.Neo_verifiedBackendBackendGlExt.permute u state = ok (permuteW state, u) := by
  simp only [native.Native.Insts.Neo_verifiedBackendBackendGlExt.permute, permute_eq, bind_ok]

theorem redW_zero : redW 0#u64 = 0#u64 := by
  simp [redW, field.MODULUS]

theorem wordOf_zero' : wordOf 0#u64 = 0 := rfl

theorem toNat_injective : Function.Injective Replay.Word.toNat := by
  rintro ⟨a, ha⟩ ⟨b, hb⟩ h
  simp only [Replay.Word.toNat] at h
  simp only [Replay.Word.mk.injEq]
  exact UInt64.toNat_inj.mp h

theorem toNat_wordOf {g : field.Gl} (h : g.val < goldilocksModulus) : (wordOf g).toNat = g.val := by
  rw [Replay.Word.toNat, wordOf, Replay.Word.ofNat, ofNat64_toNat, Nat.mod_eq_of_lt h]

/-- The permutation's output words, read as copy words, are `Replay.permute` of its input words. -/
theorem permute_words (state : Aeneas.Std.Array field.Gl 16#usize) :
    (permuteW state).val.map wordOf = (Replay.permute (state.val.map wordOf).toArray).toList := by
  have hs : glValues (permuteW state) = (Replay.permute (state.val.map wordOf).toArray).toList.map Replay.Word.toNat :=
    permute_sound state (permuteW state) (permute_eq state)
  have hc := permuteW_canonical state
  apply (List.map_injective_iff.mpr toNat_injective)
  rw [← hs, List.map_map]
  simp only [glValues]
  exact List.map_congr_left fun g hg => toNat_wordOf (hc g hg)

/-! ## `hash_leaf` -/

theorem fill_map {α β : Type} (f : α → β) (state chunk : List α) :
    (fill state chunk).map f = fill (state.map f) (chunk.map f) := by
  induction chunk using List.reverseRecOn with
  | nil => rfl
  | append_singleton c v ih => rw [fill_snoc, List.map_set, ih, List.map_append, List.map_singleton, fill_snoc,
      List.length_map]

/-- The copy's inner loop of `Replay.hashLeaf` is `fill`. -/
theorem replay_fill (state : _root_.Array Replay.Word) (c : List Replay.Word) (n : ℕ) (h : n ≤ c.length) :
    ((List.range n).foldl (fun s i => s.set! i c.toArray[i]!) state).toList = fill state.toList (c.take n) := by
  induction n with
  | zero => rfl
  | succ n ih =>
    rw [List.range_succ, List.foldl_append, List.foldl_cons, List.foldl_nil, Array.set!_eq_setIfInBounds,
      Array.toList_setIfInBounds, ih (by omega), List.take_add_one, List.getElem?_eq_getElem (by omega),
      Option.toList_some, fill_snoc, List.length_take, Nat.min_eq_left (by omega)]
    simp [getElem!_pos c n (by omega)]

/-- `Replay.hashLeaf`'s state after its first `k` chunks. -/
def leafState (vs : _root_.Array Replay.Word) (k : ℕ) : _root_.Array Replay.Word :=
  ((List.range k).map fun c => vs.extract (Replay.rate * c) (Replay.rate * c + Replay.rate)).foldl
    (fun state chunk => Replay.permute ((List.range chunk.size).foldl (fun s i => s.set! i chunk[i]!) state))
    (_root_.Array.replicate Replay.width 0)

theorem hashLeaf_state (vs : _root_.Array Replay.Word) :
    Replay.hashLeaf vs = (leafState vs ((vs.size + 11) / 12)).extract 0 4 := rfl

theorem leafState_step (vs : _root_.Array Replay.Word) (k : ℕ) :
    leafState vs (k + 1) = Replay.permute ((List.range (vs.extract (12 * k) (12 * k + 12)).size).foldl
      (fun s i => s.set! i (vs.extract (12 * k) (12 * k + 12))[i]!) (leafState vs k)) := by
  simp only [leafState, List.range_succ, List.map_append, List.foldl_append, List.map_singleton, List.foldl_cons,
    List.foldl_nil]
  rfl

theorem leafState_succ (vs : List Replay.Word) (k : ℕ) :
    leafState vs.toArray (k + 1) =
      Replay.permute (fill (leafState vs.toArray k).toList ((vs.drop (12 * k)).take 12)).toArray := by
  have hchunk : vs.toArray.extract (12 * k) (12 * k + 12) = ((vs.drop (12 * k)).take 12).toArray :=
    Array.toList_inj.mp (by simp [List.extract_eq_take_drop])
  rw [leafState_step, hchunk, List.size_toArray]
  congr 1
  exact Array.toList_inj.mp (by rw [replay_fill _ _ _ le_rfl, List.take_length])

theorem hash_leaf_native_loop (u : native.Native) (values : Slice field.Gl) (state : Aeneas.Std.Array field.Gl 16#usize)
    (hs : state.val.map wordOf = (leafState (values.val.map wordOf).toArray 0).toList)
    (hc : ∀ g ∈ state.val, g.val < goldilocksModulus) :
    ∃ y, hash.hash_leaf_loop0 nativeB u values state 0#usize = ok y ∧
      y.2.val.map wordOf = (leafState (values.val.map wordOf).toArray ((values.val.length + 11) / 12)).toList ∧
      ∀ g ∈ y.2.val, g.val < goldilocksModulus := by
  unfold hash.hash_leaf_loop0
  have hlen := values.property
  exact loop_total (fun (a : native.Native × Aeneas.Std.Array field.Gl 16#usize × Usize) =>
      hash.hash_leaf_loop0.body nativeB values a.1 a.2.1 a.2.2)
    (fun n a => ∃ k, k ≤ (values.val.length + 11) / 12 ∧ n = (values.val.length + 11) / 12 - k ∧
      a.2.2.val = min (12 * k) values.val.length ∧
      a.2.1.val.map wordOf = (leafState (values.val.map wordOf).toArray k).toList ∧
      ∀ g ∈ a.2.1.val, g.val < goldilocksModulus)
    (fun y => y.2.val.map wordOf = (leafState (values.val.map wordOf).toArray ((values.val.length + 11) / 12)).toList ∧
      ∀ g ∈ y.2.val, g.val < goldilocksModulus)
    (by
      rintro n ⟨b, st, start⟩ ⟨k, hk, rfl, hstart, hw, hcan⟩
      dsimp only at hstart hw hcan ⊢
      by_cases hlt : k < (values.val.length + 11) / 12
      · right
        have hs' : start.val < values.val.length := by omega
        have h12 : start.val = 12 * k := by omega
        obtain ⟨st1, m, hin, hst1, hm⟩ := hash_leaf_inner values st start hs'
        have hb : start < Slice.len values := by rw [slice_lt]; exact hs'
        refine ⟨(b, permuteW st1, core.num.Usize.wrapping_add start m), (values.val.length + 11) / 12 - (k + 1),
          ?_, by omega, k + 1, by omega, rfl, ?_, ?_, permuteW_canonical st1⟩
        · simp only [hash.hash_leaf_loop0.body, hb, ite_true, hin, bind_ok, uncurry_eq, native_permute, lift]
        · dsimp only
          rw [wrapping_add_le _ _ (by omega)]
          omega
        · dsimp only
          rw [permute_words, hst1, fill_map, hw, leafState_succ, h12, List.map_take, List.map_drop]
      · left
        have hk' : k = (values.val.length + 11) / 12 := by omega
        have hb : ¬ start < Slice.len values := by rw [slice_lt]; omega
        refine ⟨(b, st), ?_, by rw [← hk']; exact hw, hcan⟩
        simp only [hash.hash_leaf_loop0.body, hb, ite_false])
    _ (u, state, 0#usize) ⟨0, by omega, rfl, by simp, hs, hc⟩

/-- The first four words of a canonical state, as `Replay` reads them. -/
theorem take4_values (st : Aeneas.Std.Array field.Gl 16#usize) (hc : ∀ g ∈ st.val, g.val < goldilocksModulus) :
    glValues (Aeneas.Std.Array.make 4#usize [st.val[0]!, st.val[1]!, st.val[2]!, st.val[3]!]) =
      ((st.val.map wordOf).take 4).map Replay.Word.toNat := by
  obtain ⟨g0, g1, g2, g3, g4, g5, g6, g7, g8, g9, g10, g11, g12, g13, g14, g15, hg⟩ := list16 st.val (by simp)
  rw [hg] at hc
  simp only [List.mem_cons, List.not_mem_nil, or_false, forall_eq_or_imp, forall_eq] at hc
  simp only [glValues, Aeneas.Std.Array.make, Aeneas.Std.Array.from_val, hg]
  simp [glValue, toNat_wordOf hc.1, toNat_wordOf hc.2.1, toNat_wordOf hc.2.2.1, toNat_wordOf hc.2.2.2.1]

theorem hash_leaf_native (u : native.Native) (values : Slice field.Gl) :
    ∃ y, hash.hash_leaf nativeB u values = ok y ∧
      nativeValues y = (Replay.hashLeaf (values.val.map wordOf).toArray).toList.map Replay.Word.toNat := by
  have h0 : (Aeneas.Std.Array.repeat 16#usize (redW 0#u64)).val.map wordOf =
      (leafState (values.val.map wordOf).toArray 0).toList := by
    simp [redW_zero, leafState, Aeneas.Std.Array.repeat, Replay.width, wordOf_zero']
  have hc0 : ∀ g ∈ (Aeneas.Std.Array.repeat 16#usize (redW 0#u64)).val, g.val < goldilocksModulus := by
    simp [redW_zero, Aeneas.Std.Array.repeat, goldilocksModulus]
  obtain ⟨y, hy, hw, hc⟩ := hash_leaf_native_loop u values _ h0 hc0
  refine ⟨(Aeneas.Std.Array.make 4#usize [y.2.val[0]!, y.2.val[1]!, y.2.val[2]!, y.2.val[3]!], y.1), ?_, ?_⟩
  · unfold hash.hash_leaf
    simp only [native_constant, bind_ok, uncurry_eq, hy, index_eq y.2 0#usize (by scalar_tac),
      index_eq y.2 1#usize (by scalar_tac), index_eq y.2 2#usize (by scalar_tac), index_eq y.2 3#usize (by scalar_tac)]
    rfl
  · rw [nativeValues, take4_values y.2 hc, hashLeaf_state, List.size_toArray, List.length_map, Array.toList_extract,
      ← hw]
    simp [List.extract_eq_take_drop]

theorem hash_leaf_native_sound : Sound (fun x : native.Native × Slice field.Gl => hash.hash_leaf nativeB x.1 x.2)
    nativeValues (fun x => (Replay.hashLeaf (x.2.val.map wordOf).toArray).toList.map Replay.Word.toNat) :=
  sound_of_eq fun x => hash_leaf_native x.1 x.2

theorem hash_leaf_native_complete : Complete (fun _ => True)
    (fun x : native.Native × Slice field.Gl => hash.hash_leaf nativeB x.1 x.2) nativeValues
    (fun x => (Replay.hashLeaf (x.2.val.map wordOf).toArray).toList.map Replay.Word.toNat) :=
  complete_of_eq hash_leaf_native_sound ((), emptySlice) trivial fun x _ => hash_leaf_native x.1 x.2

/-! ## `compress` -/

theorem list4 {α : Type} (l : List α) (h : l.length = 4) : ∃ a b c d : α, l = [a, b, c, d] := by
  rcases l with _ | ⟨a, _ | ⟨b, _ | ⟨c, _ | ⟨d, _ | ⟨_, _⟩⟩⟩⟩⟩ <;> simp at h
  exact ⟨a, b, c, d, rfl⟩

theorem compress_state {α : Type} [Inhabited α] (left right : Aeneas.Std.Array α 4#usize) (z : α) :
    (upto 4 1 (compressStep left right) 4 0 (Aeneas.Std.Array.repeat 16#usize z)).val =
      left.val ++ right.val ++ List.replicate 8 z := by
  obtain ⟨a, b, c, d, hl⟩ := list4 left.val (by simp)
  obtain ⟨e, f, g, h, hr⟩ := list4 right.val (by simp)
  simp [upto, compressStep, hl, hr, Aeneas.Std.Array.repeat]

theorem compress_native (u : native.Native) (left right : Aeneas.Std.Array field.Gl 4#usize) :
    ∃ y, hash.compress nativeB u left right = ok y ∧
      nativeValues y = (Replay.compress (left.val.map wordOf).toArray (right.val.map wordOf).toArray).toList.map
        Replay.Word.toNat := by
  have hs := compress_state left right (redW 0#u64)
  obtain ⟨st, hst⟩ : ∃ st, st = upto 4 1 (compressStep left right) 4 0 (Aeneas.Std.Array.repeat 16#usize (redW 0#u64)) :=
    ⟨_, rfl⟩
  rw [← hst] at hs
  have hc := permuteW_canonical st
  refine ⟨(Aeneas.Std.Array.make 4#usize [(permuteW st).val[0]!, (permuteW st).val[1]!, (permuteW st).val[2]!,
    (permuteW st).val[3]!], u), ?_, ?_⟩
  · unfold hash.compress
    simp only [native_constant, bind_ok, uncurry_eq, compress_loop_eq, ← hst, native_permute,
      index_eq (permuteW st) 0#usize (by scalar_tac), index_eq (permuteW st) 1#usize (by scalar_tac),
      index_eq (permuteW st) 2#usize (by scalar_tac), index_eq (permuteW st) 3#usize (by scalar_tac)]
    rfl
  · have ha : ((left.val ++ right.val ++ List.replicate 8 (redW 0#u64)).map wordOf).toArray =
        (left.val.map wordOf).toArray ++ (right.val.map wordOf).toArray ++ _root_.Array.replicate (Replay.width - 8) 0 :=
      Array.toList_inj.mp (by simp [redW_zero, wordOf_zero', Replay.width])
    rw [nativeValues, take4_values _ hc, permute_words, hs, Replay.compress, ha]
    simp [List.extract_eq_take_drop]

theorem compress_native_sound : Sound
    (fun x : native.Native × Aeneas.Std.Array field.Gl 4#usize × Aeneas.Std.Array field.Gl 4#usize =>
      hash.compress nativeB x.1 x.2.1 x.2.2) nativeValues
    (fun x => (Replay.compress (x.2.1.val.map wordOf).toArray (x.2.2.val.map wordOf).toArray).toList.map
      Replay.Word.toNat) :=
  sound_of_eq fun x => compress_native x.1 x.2.1 x.2.2

theorem compress_native_complete : Complete (fun _ => True)
    (fun x : native.Native × Aeneas.Std.Array field.Gl 4#usize × Aeneas.Std.Array field.Gl 4#usize =>
      hash.compress nativeB x.1 x.2.1 x.2.2) nativeValues
    (fun x => (Replay.compress (x.2.1.val.map wordOf).toArray (x.2.2.val.map wordOf).toArray).toList.map
      Replay.Word.toNat) :=
  complete_of_eq compress_native_sound ((), Aeneas.Std.Array.repeat 4#usize 0#u64, Aeneas.Std.Array.repeat 4#usize 0#u64)
    trivial fun x _ => compress_native x.1 x.2.1 x.2.2

/-! ## `merkle_root` -/

theorem native_select (u : native.Native) (bit z o : field.Gl) :
    native.Native.Insts.Neo_verifiedBackendBackendGlExt.select u bit z o = ok (if redW bit = 0#u64 then z else o, u) := by
  simp only [native.Native.Insts.Neo_verifiedBackendBackendGlExt.select, gl_word_eq, bind_ok]
  split <;> rfl

theorem native_add (u : native.Native) (a b : field.Gl) :
    native.Native.Insts.Neo_verifiedBackendBackendGlExt.add u a b = ok (glAdd a b, u) := by
  simp only [native.Native.Insts.Neo_verifiedBackendBackendGlExt.add, gl_add_eq, bind_ok]

theorem native_sub (u : native.Native) (a b : field.Gl) :
    native.Native.Insts.Neo_verifiedBackendBackendGlExt.sub u a b = ok (glSub a b, u) := by
  simp only [native.Native.Insts.Neo_verifiedBackendBackendGlExt.sub, gl_sub_eq, bind_ok]

theorem toNat_of_cast {x y : Replay.Word} (h : (x.toNat : Gl) = (y.toNat : Gl)) : x = y := by
  apply toNat_injective
  have := congrArg ZMod.val h
  rw [ZMod.val_natCast, ZMod.val_natCast] at this
  have hx : x.toNat < goldilocksModulus := x.canonical
  have hy : y.toNat < goldilocksModulus := y.canonical
  rwa [Nat.mod_eq_of_lt hx, Nat.mod_eq_of_lt hy] at this

theorem add_sub_left (x y : Replay.Word) : x + y - x = y :=
  toNat_of_cast (by rw [cast_sub, cast_add]; ring)

theorem add_sub_right (x y : Replay.Word) : x + y - y = x :=
  toNat_of_cast (by rw [cast_sub, cast_add]; ring)

/-- Words whose values a list of copy words gives are those words. -/
theorem words_of_values {n : Usize} (a : Aeneas.Std.Array field.Gl n) (ws : List Replay.Word)
    (h : glValues a = ws.map Replay.Word.toNat) :
    a.val.map wordOf = ws ∧ ∀ g ∈ a.val, g.val < goldilocksModulus := by
  have hc : ∀ g ∈ a.val, g.val < goldilocksModulus := by
    intro g hg
    have : g.val ∈ ws.map Replay.Word.toNat := h ▸ List.mem_map_of_mem hg
    obtain ⟨w, _, hw⟩ := List.mem_map.mp this
    exact hw ▸ w.canonical
  refine ⟨List.map_injective_iff.mpr toNat_injective ?_, hc⟩
  rw [List.map_map, ← h]
  exact List.map_congr_left fun g hg => toNat_wordOf (hc g hg)

/-- Lane `k` of a level of `merkle_root` on `Native`: the left lane is the node's where `bit` is zero
and the sibling's otherwise, and the right lane is the node plus the sibling minus the left lane. -/
def laneN (node sibling : Aeneas.Std.Array field.Gl 4#usize) (bit : field.Gl) (k : ℕ)
    (lr : Aeneas.Std.Array field.Gl 4#usize × Aeneas.Std.Array field.Gl 4#usize) :
    Aeneas.Std.Array field.Gl 4#usize × Aeneas.Std.Array field.Gl 4#usize :=
  let v := if redW bit = 0#u64 then node.val[k]! else sibling.val[k]!
  (setN lr.1 k v, setN lr.2 k (glSub (glAdd node.val[k]! sibling.val[k]!) v))

theorem setN_get {α : Type} [Inhabited α] {n : Usize} (a : Aeneas.Std.Array α n) (k : ℕ) (x : α) (h : k < n.val) :
    (setN a k x).val[k]! = x := by
  simp [setN_val, h]

theorem merkle_inner_native (u : native.Native) (node sibling : Aeneas.Std.Array field.Gl 4#usize) (bit : field.Gl) :
    hash.merkle_root_loop0_loop0 nativeB u node bit sibling node node 0#usize =
      ok (u, upto 4 1 (laneN node sibling bit) 4 0 (node, node)) := by
  unfold hash.merkle_root_loop0_loop0
  refine loop_upto _ (fun (lr : Aeneas.Std.Array field.Gl 4#usize × Aeneas.Std.Array field.Gl 4#usize) (i : Usize) =>
    (u, lr.1, lr.2, i)) 4 1 1#usize rfl (by omega) (by omega) (laneN node sibling bit) (fun lr => (u, lr.1, lr.2))
    ?_ ?_ 4 (node, node) 0#usize 0 rfl rfl (by omega)
  · rintro ⟨l, r⟩ i h _
    simp only [hash.merkle_root_loop0_loop0.body, digest_len, show i < 4#usize by scalar_tac, ite_true,
      index_eq node i (by scalar_tac), index_eq sibling i (by scalar_tac), native_select, bind_ok, uncurry_eq,
      update_eq l i _ (by scalar_tac), native_add, index_eq (setN l i.val _) i (by scalar_tac),
      setN_get l i.val _ (by scalar_tac), native_sub, update_eq r i _ (by scalar_tac), lift]
    rfl
  · rintro ⟨l, r⟩ i h
    simp only [hash.merkle_root_loop0_loop0.body, digest_len, show ¬ i < 4#usize by scalar_tac, ite_false]

/-- The two halves of a level, read as copy words: the node and the sibling in the order of `bit`. -/
theorem merkle_level_words (node sibling : Aeneas.Std.Array field.Gl 4#usize) (bit : field.Gl) :
    (upto 4 1 (laneN node sibling bit) 4 0 (node, node)).1.val.map wordOf =
        (if redW bit = 0#u64 then node.val else sibling.val).map wordOf ∧
      (upto 4 1 (laneN node sibling bit) 4 0 (node, node)).2.val.map wordOf =
        (if redW bit = 0#u64 then sibling.val else node.val).map wordOf := by
  obtain ⟨a, b, c, d, hn⟩ := list4 node.val (by simp)
  obtain ⟨e, f, g, h, hs⟩ := list4 sibling.val (by simp)
  by_cases hb : redW bit = 0#u64 <;>
    simp [upto, laneN, hb, hn, hs, wordOf_glAdd, wordOf_glSub, add_sub_left, add_sub_right]

/-- The index whose bit `l` is one where `bits[l]` is not zero mod p, as `Native::select` reads a bit. -/
def bitIndex : List field.Gl → ℕ
  | [] => 0
  | b :: bs => (if redW b = 0#u64 then 0 else 1) + 2 * bitIndex bs

theorem bitIndex_bit (bits : List field.Gl) (l : ℕ) (h : l < bits.length) :
    bitIndex bits / 2 ^ l % 2 = if redW bits[l] = 0#u64 then 0 else 1 := by
  induction bits generalizing l with
  | nil => simp at h
  | cons b bs ih =>
    cases l with
    | zero =>
      simp only [bitIndex, pow_zero, Nat.div_one, List.getElem_cons_zero]
      split <;> omega
    | succ l =>
      have hl := ih l (by simpa using h)
      simp only [bitIndex, List.getElem_cons_succ, pow_succ, mul_comm (2 ^ l) 2, ← Nat.div_div_eq_div_mul]
      rw [show ((if redW b = 0#u64 then 0 else 1) + 2 * bitIndex bs) / 2 = bitIndex bs by split <;> omega]
      exact hl

/-- A translated digest as copy words. -/
def wordsOf {n : Usize} (a : Aeneas.Std.Array field.Gl n) : _root_.Array Replay.Word := (a.val.map wordOf).toArray

/-- `Replay.merkleRoot`'s node after its first `k` levels. -/
def rootState (leaf : _root_.Array Replay.Word) (index : ℕ) (path : _root_.Array (_root_.Array Replay.Word)) (k : ℕ) :
    _root_.Array Replay.Word :=
  (List.range k).foldl
    (fun node level => if (index / 2 ^ level) % 2 = 0 then Replay.compress node path[level]!
      else Replay.compress path[level]! node) leaf

theorem merkleRoot_state (leaf : _root_.Array Replay.Word) (index : ℕ) (path : _root_.Array (_root_.Array Replay.Word)) :
    Replay.merkleRoot leaf index path = rootState leaf index path path.size := rfl

theorem rootState_succ (leaf : _root_.Array Replay.Word) (index : ℕ) (path : _root_.Array (_root_.Array Replay.Word))
    (k : ℕ) : rootState leaf index path (k + 1) =
      if (index / 2 ^ k) % 2 = 0 then Replay.compress (rootState leaf index path k) path[k]!
      else Replay.compress path[k]! (rootState leaf index path k) := by
  simp only [rootState, List.range_succ, List.foldl_append, List.foldl_cons, List.foldl_nil]

theorem merkle_root_native_loop (u : native.Native) (leaf : Aeneas.Std.Array field.Gl 4#usize) (bits : Slice field.Gl)
    (path : Slice (Aeneas.Std.Array field.Gl 4#usize)) :
    ∃ y, hash.merkle_root_loop0 nativeB u bits path leaf 0#usize = ok y ∧
      y.1.val.map wordOf = (rootState (wordsOf leaf) (bitIndex bits.val)
        ((path.val.take (min bits.val.length path.val.length)).map wordsOf).toArray
        (min bits.val.length path.val.length)).toList ∧
      (min bits.val.length path.val.length = 0 → y.1 = leaf) ∧
      (0 < min bits.val.length path.val.length → ∀ g ∈ y.1.val, g.val < goldilocksModulus) := by
  unfold hash.merkle_root_loop0
  have hb := bits.property
  have hp := path.property
  exact loop_total (fun (a : native.Native × Aeneas.Std.Array field.Gl 4#usize × Usize) =>
      hash.merkle_root_loop0.body nativeB bits path a.1 a.2.1 a.2.2)
    (fun n a => a.2.2.val ≤ min bits.val.length path.val.length ∧ n = min bits.val.length path.val.length - a.2.2.val ∧
      a.2.1.val.map wordOf = (rootState (wordsOf leaf) (bitIndex bits.val)
        ((path.val.take (min bits.val.length path.val.length)).map wordsOf).toArray a.2.2.val).toList ∧
      (a.2.2.val = 0 → a.2.1 = leaf) ∧ (0 < a.2.2.val → ∀ g ∈ a.2.1.val, g.val < goldilocksModulus))
    (fun y => y.1.val.map wordOf = (rootState (wordsOf leaf) (bitIndex bits.val)
        ((path.val.take (min bits.val.length path.val.length)).map wordsOf).toArray
        (min bits.val.length path.val.length)).toList ∧
      (min bits.val.length path.val.length = 0 → y.1 = leaf) ∧
      (0 < min bits.val.length path.val.length → ∀ g ∈ y.1.val, g.val < goldilocksModulus))
    (by
      rintro n ⟨b, node, level⟩ ⟨hle, rfl, hw, h0, hc⟩
      dsimp only at hle hw h0 hc ⊢
      by_cases hlt : level.val < min bits.val.length path.val.length
      · right
        have hbi : level.val < bits.val.length := by omega
        have hpi : level.val < path.val.length := by omega
        obtain ⟨hl, hr⟩ := merkle_level_words node (path.val[level.val]'hpi) (bits.val[level.val]'hbi)
        obtain ⟨y, hy, hv⟩ := compress_native b
          (upto 4 1 (laneN node (path.val[level.val]'hpi) (bits.val[level.val]'hbi)) 4 0 (node, node)).1
          (upto 4 1 (laneN node (path.val[level.val]'hpi) (bits.val[level.val]'hbi)) 4 0 (node, node)).2
        obtain ⟨hyw, hyc⟩ := words_of_values y.1 _ hv
        have hl1 := add_one_val level (by omega)
        have hbl : level < Slice.len bits := (slice_lt bits level).mpr hbi
        have hpl : level < Slice.len path := (slice_lt path level).mpr hpi
        refine ⟨(y.2, y.1, core.num.Usize.wrapping_add level 1#usize),
          min bits.val.length path.val.length - (level.val + 1), ?_, by omega, by dsimp only; omega,
          by dsimp only; omega, ?_, fun h => by dsimp only at h; omega, fun _ => hyc⟩
        · simp only [hash.merkle_root_loop0.body, hbl, hpl, ite_true, slice_index bits level hbi,
            slice_index path level hpi, merkle_inner_native, bind_ok, uncurry_eq, lift, hy]
        · have hP : ((path.val.take (min bits.val.length path.val.length)).map wordsOf).toArray[level.val]! =
              wordsOf (path.val[level.val]'hpi) := by
            simp [hlt]
          have hnode : rootState (wordsOf leaf) (bitIndex bits.val)
              ((path.val.take (min bits.val.length path.val.length)).map wordsOf).toArray level.val =
              (node.val.map wordOf).toArray := by
            rw [hw, Array.toArray_toList]
          dsimp only
          rw [hyw, hl1, rootState_succ, bitIndex_bit bits.val level.val hbi, hP, hnode]
          by_cases hz : redW (bits.val[level.val]'hbi) = 0#u64
          · simp only [hz, ite_true] at hl hr ⊢
            rw [hl, hr]
            rfl
          · simp only [hz, ite_false, one_ne_zero] at hl hr ⊢
            rw [hl, hr]
            rfl
      · left
        have hnot : ¬ (level < Slice.len bits ∧ level < Slice.len path) := by rw [slice_lt, slice_lt]; omega
        have hk : level.val = min bits.val.length path.val.length := by omega
        refine ⟨(node, b), ?_, by rw [hw, hk], fun h => h0 (by omega), fun h => hc (by omega)⟩
        by_cases hbl : level < Slice.len bits
        · have hpl : ¬ level < Slice.len path := fun h => hnot ⟨hbl, h⟩
          simp only [hash.merkle_root_loop0.body, hbl, hpl, ite_true, ite_false]
        · simp only [hash.merkle_root_loop0.body, hbl, ite_false])
    (min bits.val.length path.val.length) (u, leaf, 0#usize)
    ⟨by simp, by simp, by simp [rootState, wordsOf], fun _ => rfl, fun h => by simp at h⟩

/-- `merkle_root` on `Native` returns `Replay.merkleRoot` of the words it reads: the leaf, the index
whose bits `Native::select` reads, and the levels that it reads, which are the shorter of `bits` and
`path`. With no level it returns the leaf as given, where `Replay` reads the leaf mod p. -/
theorem merkle_root_native (u : native.Native) (leaf : Aeneas.Std.Array field.Gl 4#usize) (bits : Slice field.Gl)
    (path : Slice (Aeneas.Std.Array field.Gl 4#usize)) :
    ∃ y, hash.merkle_root nativeB u leaf bits path = ok y ∧
      nativeValues y = if min bits.val.length path.val.length = 0 then glValues leaf
        else (Replay.merkleRoot (wordsOf leaf) (bitIndex bits.val)
          ((path.val.take (min bits.val.length path.val.length)).map wordsOf).toArray).toList.map Replay.Word.toNat := by
  obtain ⟨y, hy, hw, h0, hc⟩ := merkle_root_native_loop u leaf bits path
  refine ⟨y, hy, ?_⟩
  by_cases hn : min bits.val.length path.val.length = 0
  · rw [ite_eq_left hn, nativeValues, h0 hn]
  · rw [ite_eq_right hn, nativeValues, merkleRoot_state]
    have hsize : ((path.val.take (min bits.val.length path.val.length)).map wordsOf).toArray.size =
        min bits.val.length path.val.length := by simp
    rw [hsize, ← hw, List.map_map]
    exact List.map_congr_left fun g hg => (toNat_wordOf (hc (by omega) g hg)).symm

theorem merkle_root_native_sound : Sound
    (fun x : native.Native × Aeneas.Std.Array field.Gl 4#usize × Slice field.Gl ×
      Slice (Aeneas.Std.Array field.Gl 4#usize) => hash.merkle_root nativeB x.1 x.2.1 x.2.2.1 x.2.2.2) nativeValues
    (fun x => if min x.2.2.1.val.length x.2.2.2.val.length = 0 then glValues x.2.1
      else (Replay.merkleRoot (wordsOf x.2.1) (bitIndex x.2.2.1.val)
        ((x.2.2.2.val.take (min x.2.2.1.val.length x.2.2.2.val.length)).map wordsOf).toArray).toList.map
          Replay.Word.toNat) :=
  sound_of_eq fun x => merkle_root_native x.1 x.2.1 x.2.2.1 x.2.2.2

theorem merkle_root_native_complete : Complete (fun _ => True)
    (fun x : native.Native × Aeneas.Std.Array field.Gl 4#usize × Slice field.Gl ×
      Slice (Aeneas.Std.Array field.Gl 4#usize) => hash.merkle_root nativeB x.1 x.2.1 x.2.2.1 x.2.2.2) nativeValues
    (fun x => if min x.2.2.1.val.length x.2.2.2.val.length = 0 then glValues x.2.1
      else (Replay.merkleRoot (wordsOf x.2.1) (bitIndex x.2.2.1.val)
        ((x.2.2.2.val.take (min x.2.2.1.val.length x.2.2.2.val.length)).map wordsOf).toArray).toList.map
          Replay.Word.toNat) :=
  complete_of_eq merkle_root_native_sound ((), Aeneas.Std.Array.repeat 4#usize 0#u64, emptySlice, emptySlice) trivial
    fun x _ => merkle_root_native x.1 x.2.1 x.2.2.1 x.2.2.2

end CompressionLeanRust
