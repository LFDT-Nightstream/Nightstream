import CompressionLean.Circuit.SimHash
import CompressionLean.Circuit.Fold

/-!
Owns the simulation of the fold transcript of `Fold.lean` (`fold.rs`): the
additive sponge (`absorb`, `absorbConstants`), `squeezeDigest` and the
layer-1 `handoff`. The replay takes the layer-1 seed as an input, so this
file also owns the transcript on words that names the seed (`absorbWords`,
`handoffWords`).
-/

namespace CompressionLean.Circuit
open CompressionLean.Lowering
open CompressionLean.Replay (Word)
open NightstreamFPrime.Spec (goldilocksModulus)

variable {F : Type} [CommRing F]

namespace FoldTranscript

/-- One chunk added into the rate lanes, then a permutation (`fold.rs:39-46`). -/
def absorbChunk (state chunk : Array Word) : Array Word :=
  Replay.permute ((List.range chunk.size).foldl (fun s i => s.set! i (s[i]! + chunk[i]!)) state)

/-- `absorb` on words. -/
def absorbWords (state words : Array Word) : Array Word :=
  (List.range ((words.size + rate - 1) / rate)).foldl
    (fun s k => absorbChunk s (words.extract (rate * k) (rate * k + rate))) state

/-- The seed that `handoff` returns: the first four lanes after the layer-1
domain chunk (`fold.rs:68-72`). -/
def handoffWords (state : Array Word) : Array Word :=
  (absorbWords state (domainChunk compressDomain)).extract 0 4

theorem absorbWords_size {gs : Array Word} (gw : Array Word) (h : gs.size = 16) : (absorbWords gs gw).size = 16 := by
  unfold absorbWords
  generalize List.range _ = ks
  induction ks generalizing gs with
  | nil => exact h
  | cons k ks ih => exact ih (permute_size _)

end FoldTranscript

namespace Sim

variable [CharP F goldilocksModulus] {c : Ctx F} {n : ℕ} {P : Prop}

attribute [local irreducible] Replay.permute

/-- `absorb` (`fold.rs:39-46`). -/
theorem foldAbsorb {t : FoldTranscript} {words : Array Var} {gs gw : Array Word} (hp : Permutes c.perm)
    (ht : P → IsWords c n t.state gs ∧ gs.size = 16) (hw : P → IsWords c n words gw) :
    Sim c n P (t.absorb words) (fun t' m => IsWords c m t'.state (FoldTranscript.absorbWords gs gw)) True := by
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
  refine (((Sim.loop (c := c) (P := P) (D := fun _ => True)
      (I := fun k acc m => IsWords c m acc (after k) ∧ (after k).size = 16) fun k hk acc m le => ?_).pre
      (fun h => ⟨h, by simpa [after] using ht h⟩)).bind fun acc m le => Sim.pure (c := c) fun h => ?_).post
      (fun _ _ _ h => h) (fun _ => by simp)
  · simp only [List.getElem_map, List.getElem_range]
    rw [← Array.forIn_toList, Array.toList_zipIdx]
    generalize hwk : words.extract (rate * k) (rate * k + rate) = wk
    have wsmall : wk.size ≤ 12 := hwk ▸ small k
    have hwk' : P → IsWords c m wk (chunk k) := fun h => hwk ▸ ((hw h).extract _ _).mono le
    let inner (j : ℕ) := (List.range j).foldl (fun s i => s.set! i (s[i]! + (chunk k)[i]!)) (after k)
    have inner_size : ∀ j, (inner j).size = (after k).size := by
      intro j
      induction j with
      | zero => rfl
      | succ j ih =>
          simp only [inner, List.range_succ, List.foldl_append, List.foldl_cons, List.foldl_nil,
            Array.size_set!] at ih ⊢
          exact ih
    refine (Sim.bind (C' := True) ((Sim.loop (c := c) (P := P ∧ IsWords c m acc (after k) ∧ (after k).size = 16)
        (D := fun _ => True) (I := fun j a m' => IsWords c m' a (inner j)) fun j hj a m' le' => ?_).pre
        (fun h => ⟨h, h.2.1⟩)) fun a m₂ le₂ => ?_).post (fun _ _ _ h => h) (fun _ => by simp)
    · simp only [List.getElem_zipIdx, Nat.zero_add]
      have hj' : j < wk.size := by simpa using hj
      refine ((Sim.add (a := a[j]!) (b := wk.toList[j]) (ga := (inner j)[j]!) (gb := (chunk k)[j]!)
          (fun h => h.2.get (by rw [inner_size, h.1.2.2]; omega))
          (fun h => by
            rw [Array.getElem_toList, ← getElem!_pos wk j hj']
            exact (((hwk' h.1.1).mono le').get (by rw [← (hwk' h.1.1).1]; exact hj')))).bind
        fun sum m₃ le₃ => Sim.pure (c := c) fun h => ⟨rfl, ?_⟩).post (fun _ _ _ h => h) (fun _ => by simp)
      have e : inner (j + 1) = (inner j).set! j ((inner j)[j]! + (chunk k)[j]!) := by
        simp only [inner, List.range_succ, List.foldl_append, List.foldl_cons, List.foldl_nil]
      rw [e]
      exact (h.1.2.mono le₃).set h.2 j
    · refine ((Sim.permute (state := fun lane => a[lane]!) (gs := inner wk.size) hp fun h lane =>
          ?_).bind fun outs m₃ le₃ => Sim.pure (c := c) fun h => ⟨rfl, ?_, ?_⟩).post (fun _ _ _ h => h) (fun _ => by simp)
      · simp only [List.length_zipIdx, Array.length_toList] at h
        exact h.2.get (by rw [inner_size, h.1.2.2]; exact lane.isLt)
      · have size : wk.size = (chunk k).size := (hwk' h.1.1.1).1
        have e : after (k + 1) = Replay.permute (inner wk.size) := by
          simp only [after, inner, List.range_succ, List.foldl_append, List.foldl_cons, List.foldl_nil,
            FoldTranscript.absorbChunk, size]
        rw [e]
        exact IsWords.ofFn (permute_size _) h.2
      · simp only [after, List.range_succ, List.foldl_append, List.foldl_cons, List.foldl_nil,
          FoldTranscript.absorbChunk, permute_size]
  · have count : (List.map (fun k => words.extract (rate * k) (rate * k + rate))
        (List.range ((words.size + rate - 1) / rate))).length = (gw.size + rate - 1) / rate := by
      simp [(hw h.1).1]
    rw [count] at h
    exact h.2.1


omit [CharP F goldilocksModulus] in
/-- `new` (`fold.rs:27-30`). -/
theorem foldNew : Sim c n P FoldTranscript.new (fun t m => IsWords c m t.state (Array.replicate 16 0)) True :=
  ((Sim.const 0).bind fun _ _ _ => Sim.pure fun h => IsWords.replicate h.2 16).post (fun _ _ _ h => h)
    (fun _ => by simp)

/-- `absorb_constants` (`fold.rs:49-52`). -/
theorem foldAbsorbConstants {t : FoldTranscript} {gs : Array Word} (words : Array Word) (hp : Permutes c.perm)
    (ht : P → IsWords c n t.state gs ∧ gs.size = 16) :
    Sim c n P (t.absorbConstants words) (fun t' m => IsWords c m t'.state (FoldTranscript.absorbWords gs words))
      True := by
  unfold FoldTranscript.absorbConstants
  refine ((Sim.mapM (fun g v m => IsWord c m v g) (fun _ _ _ _ h le => h.mono le) fun g _ m _ => Sim.const g).bind
    fun hs m le => foldAbsorb hp (fun h => ⟨(ht h.1).1.mono le, (ht h.1).2⟩)
      fun h => IsWords.of_forall₂ h.2).post (fun _ _ _ h => h) (fun _ => by simp)

omit [CharP F goldilocksModulus] in
/-- `squeeze_digest` (`fold.rs:55-59`). -/
theorem squeezeDigest {t : FoldTranscript} {gs : Array Word} (hp : Permutes c.perm)
    (ht : P → IsWords c n t.state gs ∧ gs.size = 16) :
    Sim c n P t.squeezeDigest
      (fun res m => (∀ lane : Fin 4, IsWord c m (res.1 lane) gs[lane]!) ∧ IsWords c m res.2.state (Replay.permute gs))
      True := by
  unfold FoldTranscript.squeezeDigest
  refine ((Sim.permute (state := fun lane => t.state[lane]!) (gs := gs) hp fun h lane =>
    (ht h).1.get (by rw [(ht h).2]; exact lane.isLt)).bind fun outs m le => Sim.pure fun h => ⟨fun lane => ?_,
      IsWords.ofFn (permute_size _) h.2⟩).post (fun _ _ _ h => h) (fun _ => by simp)
  exact ((ht h.1).1.get (by rw [(ht h.1).2]; omega)).mono le

/-- `handoff` (`fold.rs:68-72`): the layer-1 seed. -/
theorem handoff {t : FoldTranscript} {gs : Array Word} (hp : Permutes c.perm)
    (ht : P → IsWords c n t.state gs ∧ gs.size = 16) :
    Sim c n P t.handoff (fun r m => ∀ lane : Fin 4, IsWord c m (r lane) (FoldTranscript.handoffWords gs)[lane]!)
      True := by
  unfold FoldTranscript.handoff
  refine ((foldAbsorbConstants (domainChunk compressDomain) hp ht).bind fun t' m le =>
    ((squeezeDigest hp fun h => ⟨h.2, FoldTranscript.absorbWords_size _ (ht h.1).2⟩).bind fun res m' le' =>
      Sim.pure fun h lane => ?_)).post (fun _ _ _ h => h) (fun _ => by simp)
  have e : (FoldTranscript.handoffWords gs)[lane]! =
      (FoldTranscript.absorbWords gs (domainChunk compressDomain))[lane]! := by
    have size := FoldTranscript.absorbWords_size (domainChunk compressDomain) (ht h.1.1).2
    unfold FoldTranscript.handoffWords
    rw [getElem!_pos _ _ (by simp only [Array.size_extract, size]; omega), getElem!_pos _ _ (by omega)]
    simp [Array.getElem_extract]
  rw [e]
  exact h.2.1 lane
end Sim

end CompressionLean.Circuit
