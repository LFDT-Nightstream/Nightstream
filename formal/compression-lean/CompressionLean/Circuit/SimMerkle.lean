import CompressionLean.Circuit.SimHash

/-!
Owns the simulation of the leaf sponge (`hashLeaf`), node compression
(`compress`) and Merkle paths (`merkleRoot`) of `Hash.lean` against the
replay's (`Replay/Transcript.lean`), and the F′ permutation on words
(`permutes_poseidon2`). `SimHash.lean` owns the duplex challenger.

- `IsDigest`: four handles are a digest of four words.
- `Sim.merkleRoot` takes the path words as given: the computed root is the
  replay's root of the words that the path handles hold. It does not say
  where those words come from.
- `permutes_poseidon2`: on words, `Spec.Poseidon2.permute`, the permutation
  of the executor of `lower_sound`, is the replay's
  (`State64.permute64_denote`).
-/

namespace CompressionLean.Circuit

open CompressionLean.Lowering
open CompressionLean.Lib
open CompressionLean.Replay (Word)
open NightstreamFPrime.Spec (goldilocksModulus)
open NightstreamFPrime.Export.NativePoseidon2 (State64)

variable {F : Type} [CommRing F]

/-- The four handles of `d` are the four words of `g`. -/
def IsDigest (c : Ctx F) (n : ℕ) (d : DigestVar) (g : Array Word) : Prop :=
  g.size = 4 ∧ ∀ lane : Fin 4, IsWord c n (d lane) g[lane]!

theorem IsDigest.mono {c : Ctx F} {n m : ℕ} {d : DigestVar} {g : Array Word} (h : IsDigest c n d g) (le : n ≤ m) :
    IsDigest c m d g :=
  ⟨h.1, fun lane => (h.2 lane).mono le⟩

/-- The replay's sponge state after the first `k` chunks of `gs`. -/
private def leafState (gs : Array Word) (k : ℕ) : Array Word :=
  ((List.range k).map fun j => gs.extract (Replay.rate * j) (Replay.rate * j + Replay.rate)).foldl
    (fun state chunk => Replay.permute ((List.range chunk.size).foldl (fun s i => s.set! i chunk[i]!) state))
    (Array.replicate Replay.width 0)

private theorem leafState_succ (gs : Array Word) (k : ℕ) :
    leafState gs (k + 1) =
      Replay.permute ((List.range (gs.extract (Replay.rate * k) (Replay.rate * k + Replay.rate)).size).foldl
        (fun s i => s.set! i (gs.extract (Replay.rate * k) (Replay.rate * k + Replay.rate))[i]!) (leafState gs k)) := by
  rw [leafState, List.range_succ, List.map_append, List.foldl_append]
  rfl

/-- The replay's node after the first `k` levels of `gpath`. -/
private def merkleNode (gleaf : Array Word) (index : ℕ) (gpath : Array (Array Word)) (k : ℕ) : Array Word :=
  (List.range k).foldl
    (fun node level =>
      if index / 2 ^ level % 2 = 0 then Replay.compress node gpath[level]! else Replay.compress gpath[level]! node)
    gleaf

private theorem merkleNode_succ (gleaf : Array Word) (index : ℕ) (gpath : Array (Array Word)) (k : ℕ) :
    merkleNode gleaf index gpath (k + 1) =
      if index / 2 ^ k % 2 = 0 then Replay.compress (merkleNode gleaf index gpath k) gpath[k]!
      else Replay.compress gpath[k]! (merkleNode gleaf index gpath k) := by
  rw [merkleNode, List.range_succ, List.foldl_append]
  rfl

private theorem bit_cases (x : ℕ) (gn gp : Array Word) :
    ∃ b : Bool, Word.ofNat (x % 2) = (if b then 1 else 0) ∧
      (if x % 2 = 0 then Replay.compress gn gp else Replay.compress gp gn) =
        if b then Replay.compress gp gn else Replay.compress gn gp := by
  rcases Nat.mod_two_eq_zero_or_one x with h | h
  · exact ⟨false, by rw [h]; rfl, by simp [h]⟩
  · exact ⟨true, by rw [h]; rfl, by simp [h]⟩

attribute [local irreducible] Replay.permute

namespace Sim

variable {c : Ctx F} {n : ℕ} {P : Prop}

/-- `compress` (`hash.rs:104-111`): `[left, right, 0…]`, permute, the first
four lanes. -/
theorem compress {l r : DigestVar} {gl gr : Array Word} (hp : Permutes c.perm) (hl : P → IsDigest c n l gl)
    (hr : P → IsDigest c n r gr) :
    Sim c n P (Circuit.compress l r) (fun res m => IsDigest c m res (Replay.compress gl gr)) True := by
  unfold Circuit.compress
  refine (Sim.const 0).bind₀ fun z m₁ l₁ => ?_
  refine (Sim.permute (gs := gl ++ gr ++ Array.replicate (Replay.width - 8) 0) hp fun h k => ?_).bind₀
    fun lanes m₂ l₂ => Sim.pure fun h => ⟨by simp [Replay.compress, permute_size], fun lane => ?_⟩
  · obtain ⟨⟨sl, wl⟩, ⟨sr, wr⟩⟩ := And.intro (hl h.1) (hr h.1)
    rw [Fin.getElem!_fin, getElem!_append, getElem!_append, Array.size_append, sl, sr]
    by_cases k4 : k.val < 4
    · rw [dif_pos k4, if_pos (by omega), if_pos k4]
      simpa using (wl ⟨k, k4⟩).mono l₁
    by_cases k8 : k.val < 8
    · rw [dif_neg k4, dif_pos k8, if_pos (by omega), if_neg k4]
      simpa using (wr ⟨k - 4, by omega⟩).mono l₁
    · rw [dif_neg k4, dif_neg k8, if_neg (by omega), getElem!_pos _ _ (by simp [Replay.width]; omega),
        Array.getElem_replicate]
      exact h.2
  · have lt := lane.isLt
    have e : (Replay.compress gl gr)[lane]! =
        (Replay.permute (gl ++ gr ++ Array.replicate (Replay.width - 8) 0))[lane]! := by
      rw [Fin.getElem!_fin, Fin.getElem!_fin, Replay.compress,
        getElem!_pos _ _ (by rw [Array.size_extract, permute_size]; omega),
        getElem!_pos _ _ (by rw [permute_size]; omega), Array.getElem_extract]
      simp
    rw [e]
    exact h.2 ⟨lane, by omega⟩

/-- `hash_leaf` (`hash.rs:93-101`): overwrite the lanes of each chunk of
`RATE` words and permute; the digest is the first four lanes. -/
theorem hashLeaf {vs : Array Var} {gs : Array Word} (hp : Permutes c.perm) (hv : P → IsWords c n vs gs) :
    Sim c n P (Circuit.hashLeaf vs) (fun r m => IsDigest c m r (Replay.hashLeaf gs)) True := by
  unfold Circuit.hashLeaf
  refine (Sim.const 0).bind₀ fun z m₁ l₁ => ?_
  dsimp only
  rw [← Array.forIn_toList]
  refine ((Sim.pre (Sim.loop (P := P ∧ IsWord c m₁ z 0)
      (I := fun k acc m => IsWords c m acc (leafState gs k) ∧ (leafState gs k).size = 16) (D := fun _ => True)
      fun k hk acc m hm => ?step) fun h => ⟨h, ?init⟩).bind fun st m₂ l₂ => Sim.pure fun h => ?fin).post
    (fun _ _ _ h => h) fun _ => by simp
  case init =>
    exact ⟨IsWords.replicate h.2 16, by simp [leafState, Replay.width]⟩
  case fin =>
    obtain ⟨⟨hP, -⟩, w, size⟩ := h
    have e : (chunks vs).toList.length = (gs.size + Replay.rate - 1) / Replay.rate := by
      simp [chunks, (hv hP).1]
      rfl
    have hl : Replay.hashLeaf gs = (leafState gs ((gs.size + Replay.rate - 1) / Replay.rate)).extract 0 4 := rfl
    rw [e] at w size
    rw [hl]
    refine ⟨by simp [size], fun lane => ?_⟩
    rw [Fin.getElem!_fin, getElem!_extract_zero _ lane.isLt]
    exact w.get (by rw [size]; omega)
  case step =>
    have get : (chunks vs).toList[k] = vs.extract (Replay.rate * k) (Replay.rate * k + Replay.rate) := by
      simp [chunks]
      rfl
    rw [get, ← Array.forIn_toList, Array.toList_zipIdx,
      forIn_pure (fun x s => s.set! x.2 x.1) (fun x s => by cases x; rfl), pure_bind, foldl_zipIdx_set]
    simp only [Array.getElem!_toList, Array.length_toList]
    have wc : P → IsWords c n (vs.extract (Replay.rate * k) (Replay.rate * k + Replay.rate))
        (gs.extract (Replay.rate * k) (Replay.rate * k + Replay.rate)) := fun h => (hv h).extract _ _
    let t := (List.range (gs.extract (Replay.rate * k) (Replay.rate * k + Replay.rate)).size).foldl
      (fun s i => s.set! i (gs.extract (Replay.rate * k) (Replay.rate * k + Replay.rate))[i]!) (leafState gs k)
    have ws : (P ∧ IsWord c m₁ z 0) ∧ IsWords c m acc (leafState gs k) ∧ (leafState gs k).size = 16 →
        IsWords c m ((List.range (vs.extract (Replay.rate * k) (Replay.rate * k + Replay.rate)).size).foldl
          (fun s i => s.set! i (vs.extract (Replay.rate * k) (Replay.rate * k + Replay.rate))[i]!) acc) t :=
      fun h => by
        rw [(wc h.1.1).1]
        exact h.2.1.setRange _ (fun i => i) fun i hi => ((wc h.1.1).get hi).mono (l₁.trans hm)
    have wt : (leafState gs k).size = 16 → t.size = 16 := fun h => by simp only [t, size_setRange, h]
    refine (Sim.permute (gs := t) hp fun h j => (ws h).get (by rw [wt h.2.2]; exact j.isLt)).bind₀
      fun o m' l' => Sim.pure fun h => ⟨rfl, ?_⟩
    rw [leafState_succ]
    exact ⟨IsWords.ofFn (permute_size _) h.2, permute_size _⟩

/-- `merkle_root` (`hash.rs:115-129`): at each level, `select` puts the node
left when the bit is zero, the other side is `node + sibling - left`, and the
two compress. The replay branches on digit `level` of `index`. -/
theorem merkleRoot [CharP F goldilocksModulus] {leaf : DigestVar} {bits : Array Var} {path : Array DigestVar}
    {gleaf : Array Word} {index : ℕ} {gpath : Array (Array Word)} (hp : Permutes c.perm)
    (hleaf : P → IsDigest c n leaf gleaf)
    (hbits : P → ∀ i < bits.size, IsWord c n bits[i]! (Word.ofNat (index / 2 ^ i % 2)))
    (hpath : P → ∀ k < path.size, IsDigest c n path[k]! gpath[k]!) (hsize : bits.size = path.size)
    (hgsize : path.size = gpath.size) :
    Sim c n P (Circuit.merkleRoot leaf bits path)
      (fun res m => IsDigest c m res (Replay.merkleRoot gleaf index gpath)) True := by
  unfold Circuit.merkleRoot
  dsimp only
  rw [← Array.forIn_toList]
  refine ((Sim.pre (Sim.loop (P := P) (I := fun k node m => IsDigest c m node (merkleNode gleaf index gpath k))
      (D := fun _ => True) fun k hk node m hm => ?step) fun h => ⟨h, hleaf h⟩).bind fun res m l =>
      Sim.pure fun h => ?fin).post (fun _ _ _ h => h) fun _ => by simp
  case fin =>
    have e : (bits.zip path).toList.length = gpath.size := by simp [hsize, hgsize]
    rw [e] at h
    exact h.2
  case step =>
    have kb : k < bits.size := by simp at hk; omega
    have kp : k < path.size := by omega
    have get : (bits.zip path).toList[k] = (bits[k]!, path[k]!) := by simp [getElem!_pos, kb, kp]
    rw [get]
    dsimp only
    generalize hgn : merkleNode gleaf index gpath k = gn
    obtain ⟨b, hbit, hnext⟩ := bit_cases (index / 2 ^ k) gn gpath[k]!
    rw [merkleNode_succ, hgn, hnext]
    have hs : P → IsDigest c n path[k]! gpath[k]! := fun h => hpath h k kp
    have hb : P → IsWord c n bits[k]! (if b then 1 else 0) := fun h => hbit ▸ hbits h k kb
    generalize gpath[k]! = gp at hs hnext
    have swap : (if b then Replay.compress gp gn else Replay.compress gn gp) =
        Replay.compress (if b then gp else gn) (if b then gn else gp) := by cases b <;> rfl
    rw [swap]
    refine (Sim.bind (Sim.pre (Sim.loop (P := P ∧ IsDigest c m node gn)
        (I := fun j (acc : Array Var × Array Var) m' => acc.1.size = 4 ∧ acc.2.size = 4 ∧
          ∀ i < j, IsWord c m' acc.1[i]! (if b then gp else gn)[i]! ∧
            IsWord c m' acc.2[i]! (if b then gn else gp)[i]!)
        (D := fun _ => True) fun j hj acc m' hm' => ?lane)
        fun h => ⟨h, by simp, by simp, fun i hi => absurd hi (by omega)⟩)
      fun acc m' l' => (Sim.compress hp ?left ?right).bind₀ fun res m'' l'' => Sim.pure fun h => ⟨rfl, h.2⟩).post
      (fun _ _ _ h => h) fun _ => by simp
    case left =>
      exact fun h => ⟨by cases b <;> simp [h.1.2.1, (hs h.1.1).1], fun lane => (h.2.2.2 lane (by simp)).1⟩
    case right =>
      exact fun h => ⟨by cases b <;> simp [h.1.2.1, (hs h.1.1).1], fun lane => (h.2.2.2 lane (by simp)).2⟩
    case lane =>
      have j4 : j < 4 := by simpa using hj
      rw [show (List.finRange 4)[j] = ⟨j, j4⟩ by simp]
      refine (Sim.select (b := b) (fun h => (hb h.1.1).mono (hm.trans hm')) (fun h => (h.1.2.2 ⟨j, j4⟩).mono hm')
        fun h => ((hs h.1.1).2 ⟨j, j4⟩).mono (hm.trans hm')).bind₀ fun x m₁ l₁ => ?_
      refine (Sim.add (fun h => (h.1.1.2.2 ⟨j, j4⟩).mono (hm'.trans l₁))
        fun h => ((hs h.1.1.1).2 ⟨j, j4⟩).mono (hm.trans (hm'.trans l₁))).bind₀ fun both m₂ l₂ => ?_
      refine (Sim.sub (gb := if b then gp[(⟨j, j4⟩ : Fin 4)]! else gn[(⟨j, j4⟩ : Fin 4)]!) (fun h => h.2)
        fun h => ?_).bind₀ fun y m₃ l₃ => Sim.pure fun h => ⟨rfl, ?_⟩
      · rw [Fin.getElem!_fin, getElem!_set! _ _ _ _ (by rw [h.1.1.2.1]; exact j4), if_pos rfl]
        exact h.1.2.mono l₂
      · obtain ⟨⟨⟨⟨-, s₁, s₂, prev⟩, wx⟩, -⟩, wy⟩ := h
        simp only [Fin.getElem!_fin] at wx wy
        dsimp only [ForInStep.value]
        refine ⟨by rw [Array.size_set!, s₁], by rw [Array.size_set!, s₂], fun i hi => ?_⟩
        rw [getElem!_set! _ _ _ _ (by rw [s₁]; omega), getElem!_set! _ _ _ _ (by rw [s₂]; omega)]
        by_cases e : j = i
        · subst e
          rw [if_pos rfl, if_pos rfl]
          refine ⟨(wx.mono (l₂.trans l₃)).congr ?_, wy.congr ?_⟩ <;> cases b
          all_goals simp only [Bool.false_eq_true, if_true, if_false, word_sub, word_add]
          all_goals ring
        · rw [if_neg e, if_neg e]
          exact ⟨(prev i (by omega)).1.mono (l₁.trans (l₂.trans l₃)),
            (prev i (by omega)).2.mono (l₁.trans (l₂.trans l₃))⟩

end Sim

/-! ## The F′ permutation -/

section Permutation

open Fin.CommRing

private theorem word_denote (g : Word) : (Word.cast g : NightstreamFPrime.Spec.F) = g.word.denote := by
  apply Fin.ext
  simp [Word.cast, Word.toNat, UInt64.denote, NightstreamFPrime.Spec.Poseidon2.ofNat, Fin.val_natCast]

/-- `b.permute` in `hash.rs`: on words, F′'s Poseidon2 permutation is the
replay's. -/
theorem permutes_poseidon2 : Permutes (F := NightstreamFPrime.Spec.F) NightstreamFPrime.Spec.Poseidon2.permute := by
  intro s k
  unfold Replay.permute
  dsimp only
  generalize hst : (⟨_, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _⟩ : State64) = st
  have e : (List.ofFn fun i : Fin 16 => (Word.cast s[i]! : NightstreamFPrime.Spec.F)) = st.denote := by
    subst hst
    simp [List.ofFn_succ, State64.denote, word_denote]
  rw [e, ← State64.permute64_denote]
  generalize State64.permute64 st = out
  fin_cases k <;> simp [State64.denote, word_denote]

end Permutation

end CompressionLean.Circuit
