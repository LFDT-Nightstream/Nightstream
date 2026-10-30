import CompressionLean.Sim.Hash

namespace CompressionLean.Sim

open CompressionLean.Circuit
open CompressionLean.Lib
open CompressionLean.Replay (Word)
open NightstreamFPrime.Spec (goldilocksModulus)
open NightstreamFPrime.Export.NativePoseidon2 (State64)

variable {F : Type} [CommRing F]

def IsDigest (c : Ctx F) (n : ℕ) (d : DigestVar) (g : Array Word) : Prop :=
  g.size = 4 ∧ ∀ lane : Fin 4, IsWord c n (d lane) g[lane]!

theorem IsDigest.mono {c : Ctx F} {n m : ℕ} {d : DigestVar} {g : Array Word} (h : IsDigest c n d g)
    (le : n ≤ m := by omega) : IsDigest c m d g :=
  ⟨h.1, fun lane => (h.2 lane).mono le⟩

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

instance compress_appends (l r : DigestVar) : Appends (Circuit.compress l r) := by unfold Circuit.compress; appends
instance hashLeaf_appends (vs : Array Var) : Appends (Circuit.hashLeaf vs) := by unfold Circuit.hashLeaf; appends
instance merkleRoot_appends (leaf : DigestVar) (bits : Array Var) (path : Array DigestVar) :
    Appends (Circuit.merkleRoot leaf bits path) := by
  unfold Circuit.merkleRoot; appends

section Hashes

variable {c : Ctx F} {n : ℕ}

theorem compress_ensures {l r : DigestVar} {gl gr : Array Word} (hp : Permutes c.perm) (hl : IsDigest c n l gl)
    (hr : IsDigest c n r gr) :
    Ensures c n (Circuit.compress l r) fun res m => IsDigest c m res (Replay.compress gl gr) := by
  unfold Circuit.compress
  refine (const_ensures 0).bind fun z _ l₁ hz => ?_
  refine (permute_ensures (gs := gl ++ gr ++ Array.replicate (Replay.width - 8) 0) hp fun k => ?_).bind
    fun lanes _ _ hlanes => Ensures.pure ⟨by simp [Replay.compress, permute_size], fun lane => ?_⟩
  · obtain ⟨⟨sl, wl⟩, ⟨sr, wr⟩⟩ := And.intro hl hr
    rw [Fin.getElem!_fin, getElem!_append, getElem!_append, Array.size_append, sl, sr]
    by_cases k4 : k.val < 4
    · rw [dite_eq_left k4, ite_eq_left (by omega), ite_eq_left k4]
      simpa using (wl ⟨k, k4⟩).mono l₁
    by_cases k8 : k.val < 8
    · rw [dite_eq_right k4, dite_eq_left k8, ite_eq_left (by omega), ite_eq_right k4]
      simpa using (wr ⟨k - 4, by omega⟩).mono l₁
    · rw [dite_eq_right k4, dite_eq_right k8, ite_eq_right (by omega), getElem!_pos _ _ (by simp [Replay.width]; omega),
        Array.getElem_replicate]
      exact hz
  · have lt := lane.isLt
    have e : (Replay.compress gl gr)[lane]! =
        (Replay.permute (gl ++ gr ++ Array.replicate (Replay.width - 8) 0))[lane]! := by
      rw [Fin.getElem!_fin, Fin.getElem!_fin, Replay.compress,
        getElem!_pos _ _ (by rw [Array.size_extract, permute_size]; omega),
        getElem!_pos _ _ (by rw [permute_size]; omega), Array.getElem_extract]
      simp
    rw [e]
    exact hlanes ⟨lane, by omega⟩

theorem hashLeaf_ensures {vs : Array Var} {gs : Array Word} (hp : Permutes c.perm) (hv : IsWords c n vs gs) :
    Ensures c n (Circuit.hashLeaf vs) fun r m => IsDigest c m r (Replay.hashLeaf gs) := by
  unfold Circuit.hashLeaf
  refine (const_ensures 0).bind fun z _ _ hz => ?_
  dsimp only
  rw [← Array.forIn_toList]
  refine (Ensures.loop (fun k acc m => IsWords c m acc (leafState gs k) ∧ (leafState gs k).size = 16)
      ⟨IsWords.replicate hz 16, by simp [leafState, Replay.width]⟩ fun k _ acc m _ hacc => ?_).bind
    fun st _ _ hst => Ensures.pure ?_
  · obtain ⟨w, size⟩ := hacc
    have get : (chunks vs).toList[k] = vs.extract (Replay.rate * k) (Replay.rate * k + Replay.rate) := by
      simp [chunks]
      rfl
    rw [get, ← Array.forIn_toList, Array.toList_zipIdx,
      forIn_pure (fun x s => s.set! x.2 x.1) (fun x s => by cases x; rfl), pure_bind, foldl_zipIdx_set]
    simp only [Array.getElem!_toList, Array.length_toList]
    have wc : IsWords c n (vs.extract (Replay.rate * k) (Replay.rate * k + Replay.rate))
        (gs.extract (Replay.rate * k) (Replay.rate * k + Replay.rate)) := hv.extract _ _
    let t := (List.range (gs.extract (Replay.rate * k) (Replay.rate * k + Replay.rate)).size).foldl
      (fun s i => s.set! i (gs.extract (Replay.rate * k) (Replay.rate * k + Replay.rate))[i]!) (leafState gs k)
    have ws : IsWords c m ((List.range (vs.extract (Replay.rate * k) (Replay.rate * k + Replay.rate)).size).foldl
        (fun s i => s.set! i (vs.extract (Replay.rate * k) (Replay.rate * k + Replay.rate))[i]!) acc) t := by
      rw [wc.1]
      exact w.setRange _ (fun i => i) fun i hi => (wc.get hi).mono
    have wt : t.size = 16 := by simp only [t, size_setRange, size]
    refine (permute_ensures (gs := t) hp fun j => ws.get (by rw [wt]; exact j.isLt)).bind
      fun o _ _ ho => Ensures.pure ⟨rfl, ?_⟩
    rw [leafState_succ]
    exact ⟨IsWords.ofFn (permute_size _) ho, permute_size _⟩
  · obtain ⟨w, size⟩ := hst
    have e : (chunks vs).toList.length = (gs.size + Replay.rate - 1) / Replay.rate := by
      simp [chunks, hv.1]
      rfl
    have hl : Replay.hashLeaf gs = (leafState gs ((gs.size + Replay.rate - 1) / Replay.rate)).extract 0 4 := rfl
    rw [e] at w size
    rw [hl]
    refine ⟨by simp [size], fun lane => ?_⟩
    rw [Fin.getElem!_fin, getElem!_extract_zero _ lane.isLt]
    exact w.get (by rw [size]; omega)

theorem merkleRoot_ensures [CharP F goldilocksModulus] {leaf : DigestVar} {bits : Array Var} {path : Array DigestVar}
    {gleaf : Array Word} {index : ℕ} {gpath : Array (Array Word)} (hp : Permutes c.perm)
    (hleaf : IsDigest c n leaf gleaf)
    (hbits : ∀ i < bits.size, IsWord c n bits[i]! (Word.ofNat (index / 2 ^ i % 2)))
    (hpath : ∀ k < path.size, IsDigest c n path[k]! gpath[k]!) (hsize : bits.size = path.size)
    (hgsize : path.size = gpath.size) :
    Ensures c n (Circuit.merkleRoot leaf bits path)
      fun res m => IsDigest c m res (Replay.merkleRoot gleaf index gpath) := by
  unfold Circuit.merkleRoot
  dsimp only
  rw [← Array.forIn_toList]
  refine (Ensures.loop (fun k node m => IsDigest c m node (merkleNode gleaf index gpath k)) hleaf
    fun k hk node m _ hnode => ?_).bind fun res _ _ hres => Ensures.pure ?_
  · have kb : k < bits.size := by simp at hk; omega
    have kp : k < path.size := by omega
    have get : (bits.zip path).toList[k] = (bits[k]!, path[k]!) := by simp [getElem!_pos, kb, kp]
    rw [get]
    dsimp only
    generalize hgn : merkleNode gleaf index gpath k = gn at hnode
    obtain ⟨b, hbit, hnext⟩ := bit_cases (index / 2 ^ k) gn gpath[k]!
    rw [merkleNode_succ, hgn, hnext]
    have hs : IsDigest c n path[k]! gpath[k]! := hpath k kp
    have hb : IsWord c n bits[k]! (if b then 1 else 0) := hbit ▸ hbits k kb
    generalize gpath[k]! = gp at hs hnext
    have swap : (if b then Replay.compress gp gn else Replay.compress gn gp) =
        Replay.compress (if b then gp else gn) (if b then gn else gp) := by cases b <;> rfl
    rw [swap]
    refine (Ensures.loop (fun j (acc : Array Var × Array Var) m' => acc.1.size = 4 ∧ acc.2.size = 4 ∧
          ∀ i < j, IsWord c m' acc.1[i]! (if b then gp else gn)[i]! ∧
            IsWord c m' acc.2[i]! (if b then gn else gp)[i]!)
        ⟨by simp, by simp, fun i hi => absurd hi (by omega)⟩ fun j hj acc m' _ hacc => ?_).bind
      fun acc _ _ hacc => ?_
    · have j4 : j < 4 := by simpa using hj
      obtain ⟨s₁, s₂, prev⟩ := hacc
      rw [show (List.finRange 4)[j]'(by simpa using j4) = ⟨j, j4⟩ by simp]
      refine (select_ensures (b := b) hb.mono (hnode.2 ⟨j, j4⟩).mono (hs.2 ⟨j, j4⟩).mono).bind
        fun x _ _ wx => ?_
      refine (add_ensures (hnode.2 ⟨j, j4⟩).mono (hs.2 ⟨j, j4⟩).mono).bind fun both _ _ hboth => ?_
      refine (sub_ensures (gb := if b then gp[(⟨j, j4⟩ : Fin 4)]! else gn[(⟨j, j4⟩ : Fin 4)]!) hboth ?_).bind
        fun y _ _ wy => Ensures.pure ⟨rfl, ?_⟩
      · rw [Fin.getElem!_fin, getElem!_set! _ _ _ _ (by rw [s₁]; exact j4), ite_eq_left rfl]
        exact wx.mono
      · simp only [Fin.getElem!_fin] at wx wy
        dsimp only [ForInStep.value]
        refine ⟨by rw [Array.size_set!, s₁], by rw [Array.size_set!, s₂], fun i hi => ?_⟩
        rw [getElem!_set! _ _ _ _ (by rw [s₁]; omega), getElem!_set! _ _ _ _ (by rw [s₂]; omega)]
        by_cases e : j = i
        · subst e
          rw [ite_eq_left rfl, ite_eq_left rfl]
          refine ⟨wx.mono.congr ?_, wy.congr ?_⟩ <;> cases b
          all_goals simp only [Bool.false_eq_true, ite_true, ite_false, word_sub, word_add]
          all_goals ring
        · rw [ite_eq_right e, ite_eq_right e]
          exact ⟨(prev i (by omega)).1.mono, (prev i (by omega)).2.mono⟩
    · obtain ⟨s₁, s₂, all⟩ := hacc
      refine (compress_ensures hp ⟨?_, fun lane => (all lane (by simp)).1⟩
        ⟨?_, fun lane => (all lane (by simp)).2⟩).bind fun node' _ _ hnode' => Ensures.pure ⟨rfl, hnode'⟩
      · cases b <;> simp [hnode.1, hs.1]
      · cases b <;> simp [hnode.1, hs.1]
  · have e : (bits.zip path).toList.length = gpath.size := by simp [hsize, hgsize]
    rw [e] at hres
    exact hres

end Hashes

section Permutation

open Fin.CommRing

private theorem word_denote (g : Word) : (Word.cast g : NightstreamFPrime.Spec.F) = g.word.denote := by
  apply Fin.ext
  simp [Word.cast, Word.toNat, UInt64.denote, NightstreamFPrime.Spec.Poseidon2.ofNat, Fin.val_natCast]

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

end CompressionLean.Sim
