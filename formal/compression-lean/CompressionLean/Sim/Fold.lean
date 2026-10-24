import CompressionLean.Sim.Hash
import CompressionLean.Ports.Circuit.Fold

/-!
Owns the simulation of the fold transcript of `Fold.lean` (`fold.rs`): the
additive sponge (`absorb`, `absorbConstants`), `squeezeDigest` and the
layer-1 `handoff`. The replay takes the layer-1 seed as an input, so this
file also owns the transcript on words that names the seed (`absorbWords`,
`handoffWords`).
-/

namespace CompressionLean.Sim
open CompressionLean.Circuit
open CompressionLean.Replay (Word)
open NightstreamFPrime.Spec (goldilocksModulus)

variable {F : Type} [CommRing F]

section FoldTranscript

open CompressionLean.Circuit.FoldTranscript

/-- One chunk added into the rate lanes, then a permutation (`fold.rs:39-46`). -/
def _root_.CompressionLean.Circuit.FoldTranscript.absorbChunk (state chunk : Array Word) : Array Word :=
  Replay.permute ((List.range chunk.size).foldl (fun s i => s.set! i (s[i]! + chunk[i]!)) state)

/-- `absorb` on words. -/
def _root_.CompressionLean.Circuit.FoldTranscript.absorbWords (state words : Array Word) : Array Word :=
  (List.range ((words.size + rate - 1) / rate)).foldl
    (fun s k => absorbChunk s (words.extract (rate * k) (rate * k + rate))) state

/-- The seed that `handoff` returns: the first four lanes after the layer-1
domain chunk (`fold.rs:68-72`). -/
def _root_.CompressionLean.Circuit.FoldTranscript.handoffWords (state : Array Word) : Array Word :=
  (absorbWords state (domainChunk compressDomain)).extract 0 4

theorem _root_.CompressionLean.Circuit.FoldTranscript.absorbWords_size {gs : Array Word} (gw : Array Word) (h : gs.size = 16) : (absorbWords gs gw).size = 16 := by
  unfold absorbWords
  generalize List.range _ = ks
  induction ks generalizing gs with
  | nil => exact h
  | cons k ks ih => exact ih (permute_size _)

end FoldTranscript

/-! ## Programs -/

instance FoldTranscript.new_appends : Appends FoldTranscript.new := by unfold FoldTranscript.new; appends
instance FoldTranscript.absorb_appends (t : FoldTranscript) (words : Array Var) : Appends (t.absorb words) := by
  unfold FoldTranscript.absorb; appends
instance FoldTranscript.absorbConstants_appends (t : FoldTranscript) (words : Array Word) :
    Appends (t.absorbConstants words) := by
  unfold FoldTranscript.absorbConstants; appends
instance FoldTranscript.squeezeDigest_appends (t : FoldTranscript) : Appends t.squeezeDigest := by
  unfold FoldTranscript.squeezeDigest; appends
instance FoldTranscript.handoff_appends (t : FoldTranscript) : Appends t.handoff := by
  unfold FoldTranscript.handoff; appends

attribute [local irreducible] Replay.permute

/-- `new` (`fold.rs:27-30`). -/
theorem FoldTranscript.new_ensures {c : Ctx F} {n : ℕ} :
    Ensures c n FoldTranscript.new fun t m => IsWords c m t.state (Array.replicate 16 0) :=
  (const_ensures 0).bind fun _ _ _ hz => Ensures.pure (IsWords.replicate hz 16)

/-- `squeeze_digest` (`fold.rs:55-59`). -/
theorem FoldTranscript.squeezeDigest_ensures {c : Ctx F} {n : ℕ} {t : FoldTranscript} {gs : Array Word}
    (hp : Permutes c.perm) (ht : IsWords c n t.state gs) (hsize : gs.size = 16) :
    Ensures c n t.squeezeDigest fun res m =>
      (∀ lane : Fin 4, IsWord c m (res.1 lane) gs[lane]!) ∧ IsWords c m res.2.state (Replay.permute gs) := by
  unfold FoldTranscript.squeezeDigest
  refine (permute_ensures (state := fun lane => t.state[lane]!) (gs := gs) hp fun lane =>
    ht.get (by rw [hsize]; exact lane.isLt)).bind fun outs _ _ houts =>
      Ensures.pure ⟨fun lane => ?_, IsWords.ofFn (permute_size _) houts⟩
  exact (ht.get (by rw [hsize]; omega)).mono

section Absorb

variable [CharP F goldilocksModulus] {c : Ctx F} {n : ℕ}

/-- `absorb` (`fold.rs:39-46`). -/
theorem FoldTranscript.absorb_ensures {t : FoldTranscript} {words : Array Var} {gs gw : Array Word}
    (hp : Permutes c.perm) (ht : IsWords c n t.state gs) (hsize : gs.size = 16) (hw : IsWords c n words gw) :
    Ensures c n (t.absorb words) fun t' m => IsWords c m t'.state (FoldTranscript.absorbWords gs gw) := by
  unfold FoldTranscript.absorb
  dsimp only
  rw [← Array.forIn_toList]
  have hchunks : (chunks words).toList =
      (List.range ((words.size + rate - 1) / rate)).map fun k => words.extract (rate * k) (rate * k + rate) := by
    simp [chunks]
  rw [hchunks]
  let chunk (k : ℕ) := gw.extract (rate * k) (rate * k + rate)
  let after (k : ℕ) := (List.range k).foldl (fun s k => FoldTranscript.absorbChunk s (chunk k)) gs
  have small : ∀ k, (words.extract (rate * k) (rate * k + rate)).size ≤ rate := fun k => by
    simp only [Array.size_extract]
    omega
  refine (Ensures.loop (fun k acc m => IsWords c m acc (after k) ∧ (after k).size = 16)
    ⟨by simpa [after] using ht, by simpa [after] using hsize⟩ fun k _ acc m _ hacc => ?_).bind
    fun acc _ _ hacc => Ensures.pure ?_
  · obtain ⟨wacc, sacc⟩ := hacc
    simp only [List.getElem_map, List.getElem_range]
    rw [← Array.forIn_toList, Array.toList_zipIdx]
    generalize hwk : words.extract (rate * k) (rate * k + rate) = wk
    have wsmall : wk.size ≤ 12 := hwk ▸ small k
    have hwk' : IsWords c m wk (chunk k) := hwk ▸ (hw.extract _ _).mono
    let inner (j : ℕ) := (List.range j).foldl (fun s i => s.set! i (s[i]! + (chunk k)[i]!)) (after k)
    have inner_size : ∀ j, (inner j).size = (after k).size := by
      intro j
      induction j with
      | zero => rfl
      | succ j ih =>
          simp only [inner, List.range_succ, List.foldl_append, List.foldl_cons, List.foldl_nil,
            Array.size_set!] at ih ⊢
          exact ih
    refine (Ensures.loop (fun j a m' => IsWords c m' a (inner j)) wacc fun j hj a m' _ ha => ?_).bind
      fun a _ _ ha => ?_
    · simp only [List.getElem_zipIdx, Nat.zero_add]
      have hj' : j < wk.size := by simpa using hj
      refine (add_ensures (a := a[j]!) (b := wk.toList[j]) (ga := (inner j)[j]!) (gb := (chunk k)[j]!)
        (ha.get (by rw [inner_size, sacc]; omega)) ?_).bind fun sum _ _ hsum => Ensures.pure ⟨rfl, ?_⟩
      · rw [Array.getElem_toList, ← getElem!_pos wk j hj']
        exact hwk'.mono.get (by rw [← hwk'.1]; exact hj')
      · have e : inner (j + 1) = (inner j).set! j ((inner j)[j]! + (chunk k)[j]!) := by
          simp only [inner, List.range_succ, List.foldl_append, List.foldl_cons, List.foldl_nil]
        rw [e]
        exact ha.mono.set hsum j
    · refine (permute_ensures (state := fun lane => a[lane]!) (gs := inner wk.size) hp fun lane => ?_).bind
        fun outs _ _ houts => Ensures.pure ⟨rfl, ?_, ?_⟩
      · simp only [List.length_zipIdx, Array.length_toList] at ha
        exact ha.get (by rw [inner_size, sacc]; exact lane.isLt)
      · have size : wk.size = (chunk k).size := hwk'.1
        have e : after (k + 1) = Replay.permute (inner wk.size) := by
          simp only [after, inner, List.range_succ, List.foldl_append, List.foldl_cons, List.foldl_nil,
            FoldTranscript.absorbChunk, size]
        rw [e]
        exact IsWords.ofFn (permute_size _) houts
      · simp only [after, List.range_succ, List.foldl_append, List.foldl_cons, List.foldl_nil,
          FoldTranscript.absorbChunk, permute_size]
  · have count : (List.map (fun k => words.extract (rate * k) (rate * k + rate))
        (List.range ((words.size + rate - 1) / rate))).length = (gw.size + rate - 1) / rate := by
      simp [hw.1]
    rw [count] at hacc
    exact hacc.1

/-- `absorb_constants` (`fold.rs:49-52`). -/
theorem FoldTranscript.absorbConstants_ensures {t : FoldTranscript} {gs : Array Word} (words : Array Word)
    (hp : Permutes c.perm) (ht : IsWords c n t.state gs) (hsize : gs.size = 16) :
    Ensures c n (t.absorbConstants words) fun t' m => IsWords c m t'.state (FoldTranscript.absorbWords gs words) := by
  unfold FoldTranscript.absorbConstants
  exact (Ensures.mapM (fun g v m => IsWord c m v g) (fun _ _ _ _ h le => h.mono le) fun g _ _ _ => const_ensures g).bind
    fun _ _ _ hs => FoldTranscript.absorb_ensures hp ht.mono hsize (IsWords.of_forall₂ hs)

/-- `handoff` (`fold.rs:68-72`): the layer-1 seed. -/
theorem FoldTranscript.handoff_ensures {t : FoldTranscript} {gs : Array Word} (hp : Permutes c.perm)
    (ht : IsWords c n t.state gs) (hsize : gs.size = 16) :
    Ensures c n t.handoff fun r m => ∀ lane : Fin 4, IsWord c m (r lane) (FoldTranscript.handoffWords gs)[lane]! := by
  unfold FoldTranscript.handoff
  refine (FoldTranscript.absorbConstants_ensures (domainChunk compressDomain) hp ht hsize).bind fun _ _ _ ht' => ?_
  refine (FoldTranscript.squeezeDigest_ensures hp ht' (FoldTranscript.absorbWords_size _ hsize)).bind
    fun res _ _ hres => Ensures.pure fun lane => ?_
  have e : (FoldTranscript.handoffWords gs)[lane]! =
      (FoldTranscript.absorbWords gs (domainChunk compressDomain))[lane]! := by
    have size := FoldTranscript.absorbWords_size (domainChunk compressDomain) hsize
    unfold FoldTranscript.handoffWords
    rw [getElem!_pos _ _ (by simp only [Array.size_extract, size]; omega), getElem!_pos _ _ (by omega)]
    simp [Array.getElem_extract]
  rw [e]
  exact hres.1 lane

end Absorb

end CompressionLean.Sim
