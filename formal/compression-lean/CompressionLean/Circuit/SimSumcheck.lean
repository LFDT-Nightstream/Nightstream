import CompressionLean.Circuit.SimMerkle
import CompressionLean.Circuit.Sumcheck
import CompressionLean.Bridge.Run

/-!
Owns the simulation of the sum-check replay of `Sumcheck.lean`
(`sumcheck.rs`) against the replay's (`Replay/Gkr.lean`): the loop that
observes each value of an array (`Sim.observeExts`), and `Sim.sumcheckReplay`,
which gives the replay's run. `Sim.elim` fixes the replay values that a loop
step names by an existential. Each round message must be nonempty: both read
`message[0]!`, where Rust's `message[0]` panics.
-/

namespace CompressionLean.Replay

/-- Replaying `l₁ ++ l₂` replays `l₁`, then `l₂` from its point and claim. -/
theorem sumcheckRounds_append (l₁ l₂ : List (Array Ext)) (point : Array Ext) (claim : Ext) :
    sumcheckRounds (l₁ ++ l₂) point claim =
      sumcheckRounds l₁ point claim >>= fun r => sumcheckRounds l₂ r.1 r.2 := by
  induction l₁ generalizing point claim with
  | nil => simp [sumcheckRounds]
  | cons msg rest ih => simp [sumcheckRounds, ih, bind_assoc]

end CompressionLean.Replay

namespace CompressionLean.Circuit

open CompressionLean.Replay (Ext Runs)
open NightstreamFPrime.Spec (goldilocksModulus)

variable {F : Type} [CommRing F]

namespace IsExts

variable {c : Ctx F} {n : ℕ} {xs : Array ExtVar} {gs : Array Ext}

theorem empty : IsExts c n #[] #[] := ⟨rfl, fun _ h => absurd h (by simp)⟩

theorem append (h : IsExts c n xs gs) {ys : Array ExtVar} {gy : Array Ext} (hy : IsExts c n ys gy) :
    IsExts c n (xs ++ ys) (gs ++ gy) := by
  refine ⟨by simp [h.1, hy.1], fun i hi => ?_⟩
  rw [Array.size_append] at hi
  rw [getElem!_pos _ _ (by simp; omega), getElem!_pos _ _ (by simp [← h.1, ← hy.1]; omega), Array.getElem_append,
    Array.getElem_append]
  by_cases lt : i < xs.size
  · rw [dif_pos lt, dif_pos (h.1 ▸ lt), ← getElem!_pos, ← getElem!_pos]
    exact h.2 i lt
  · rw [dif_neg lt, dif_neg (h.1 ▸ lt), ← getElem!_pos, ← getElem!_pos, ← h.1]
    exact hy.2 _ (by omega)

theorem extract (h : IsExts c n xs gs) (a : ℕ) : IsExts c n (xs.extract a) (gs.extract a) := by
  refine ⟨by simp [h.1], fun i hi => ?_⟩
  rw [Array.size_extract] at hi
  rw [getElem!_pos _ _ (by simp; omega), getElem!_pos _ _ (by simp [← h.1]; omega), Array.getElem_extract,
    Array.getElem_extract, ← getElem!_pos, ← getElem!_pos]
  exact h.2 _ (by omega)

end IsExts

namespace Sim

variable [CharP F goldilocksModulus] {c : Ctx F} {n : ℕ} {P : Prop}

omit [CharP F goldilocksModulus] in
/-- Fix the witness of an existential precondition. Any `x₀` gives the
instructions, which do not depend on it. -/
theorem elim {α ι : Type} {b : Build α} {Q : α → ℕ → Prop} {C : Prop} {R : ι → Prop} (x₀ : ι)
    (h : ∀ x, Sim c n (R x) b Q C) (hp : P → ∃ x, R x) : Sim c n P b Q C := by
  intro p E hp' hE
  obtain ⟨new, hi, hl, -⟩ := h x₀ p E hp' hE
  refine ⟨new, hi, hl, fun hP hF => ?_⟩
  obtain ⟨x, hx⟩ := hp hP
  obtain ⟨new', hi', -, hs⟩ := h x p E hp' hE
  obtain rfl : new = new' := List.toArray_inj ((Array.append_right_inj _).mp (hi.symm.trans hi'))
  exact hs hx hF

/-- `for &value in message { duplex.observe_ext(b, value) }`
(`sumcheck.rs:36-38`): the replay's `observeExts`. -/
theorem observeExts {d : Duplex} {r : Replay.Duplex} {vs : Array ExtVar} {gs : Array Ext} (hp : Permutes c.perm)
    (hd : P → IsDuplex c n d r) (hv : P → IsExts c n vs gs) :
    Sim c n P
      ((forIn vs PUnit.unit fun value _ => (do Duplex.observeExt value; Pure.pure (ForInStep.yield PUnit.unit))).run d)
      (fun res m => IsDuplex c m res.2 ((gs.flatMap fun g => #[g.c0, g.c1, g.c2]).foldl Replay.Duplex.observe r))
      True := by
  rw [← Array.forIn_toList]
  refine ((loopT (fun k _ e m => IsDuplex c m e
      (((gs.toList.take k).flatMap fun g => [g.c0, g.c1, g.c2]).foldl Replay.Duplex.observe r)) (fun _ => True)
    fun k hk _ e m le => ?_).pre fun h => ⟨h, by simpa using hd h⟩).post (fun h _ _ q => ?_)
    fun _ => ⟨fun _ => trivial, fun _ _ _ => trivial⟩
  · have hk' : k < vs.size := by simpa using hk
    rw [StateT.run_bind]
    refine (observeExt (e := gs[k]!) hp (fun h => h.2) fun h => ?_).bind₀ fun res m' le' => ?_
    · rw [Array.getElem_toList, ← getElem!_pos vs k hk']
      exact ((hv h.1).2 k hk').mono le
    · rw [StateT.run_pure]
      refine Sim.pure fun h => ⟨rfl, ?_⟩
      have hg : k < gs.size := (hv h.1.1).1 ▸ hk'
      rw [List.take_add_one, List.getElem?_eq_getElem (by simpa using hg), Option.toList_some, List.flatMap_append,
        List.foldl_append, Array.getElem_toList, ← getElem!_pos gs k hg]
      simp only [List.flatMap_cons, List.flatMap_nil, List.append_nil, List.foldl_cons, List.foldl_nil]
      exact h.2
  · rw [show vs.toList.length = gs.toList.length by simp [(hv h).1], List.take_length] at q
    rw [← Array.foldl_toList, Array.toList_flatMap]
    exact q

/-- `replay` (`sumcheck.rs:24-44`): the point and the final claim are the
replay's run, which succeeds, and the duplexes agree. Each round message is
nonempty. -/
theorem sumcheckReplay {rounds : Array (Array ExtVar)} {grounds : Array (Array Ext)} {claim : ExtVar} {gclaim : Ext}
    {d : Duplex} {t : Replay.Transcript} (hp : Permutes c.perm) (hd : P → IsDuplex c n d t.duplex)
    (hr : P → rounds.size = grounds.size ∧ ∀ i < rounds.size, IsExts c n rounds[i]! grounds[i]!)
    (hne : P → ∀ g ∈ grounds, 0 < g.size) (hc : P → IsExt c n claim gclaim) :
    Sim c n P ((Circuit.sumcheckReplay rounds claim).run d)
      (fun res m => ∃ point final t', Runs (Replay.sumcheckReplay grounds gclaim) t (point, final) t' ∧
        IsExts c m res.1.1 point ∧ IsExt c m res.1.2 final ∧ IsDuplex c m res.2 t'.duplex) True := by
  unfold Circuit.sumcheckReplay
  dsimp only
  refine Sim.tbind₀ (Q := fun r m => ∃ point final t',
      Runs (Replay.sumcheckRounds grounds.toList #[] gclaim) t (point, final) t' ∧
        IsExts c m r.1.1 point ∧ IsExt c m r.1.2 final ∧ IsDuplex c m r.2 t'.duplex) ?_ fun res e m le => ?_
  · rw [← Array.forIn_toList (xs := rounds)]
    refine ((loopT (fun k acc e m => ∃ point final t',
        Runs (Replay.sumcheckRounds (grounds.toList.take k) #[] gclaim) t (point, final) t' ∧
          IsExts c m acc.1 point ∧ IsExt c m acc.2 final ∧ IsDuplex c m e t'.duplex) (fun _ => True)
      fun k hk acc e m le => ?_).pre
        fun h => ⟨h, #[], gclaim, t, Replay.runs_pure _ _, IsExts.empty, hc h, hd h⟩).post
        (fun h r m q => ?_) fun _ => ⟨fun _ => trivial, fun _ _ _ => trivial⟩
    · have hk' : k < rounds.size := by simpa using hk
      have hmsg : P → IsExts c n rounds.toList[k] grounds[k]! := fun h => by
        rw [Array.getElem_toList, ← getElem!_pos rounds k hk']
        exact (hr h).2 k hk'
      have hpos : P → 0 < rounds.toList[k].size := fun h => by
        rw [(hmsg h).1, getElem!_pos grounds k ((hr h).1 ▸ hk')]
        exact hne h _ (Array.getElem_mem _)
      refine Sim.elim (R := fun w : Array Ext × Ext × Replay.Transcript => P ∧
          Runs (Replay.sumcheckRounds (grounds.toList.take k) #[] gclaim) t (w.1, w.2.1) w.2.2 ∧
          IsExts c m acc.1 w.1 ∧ IsExt c m acc.2 w.2.1 ∧ IsDuplex c m e w.2.2.duplex) (#[], 0, t)
        (fun w => ?_) fun ⟨hP, gpt, gcl, tk, h⟩ => ⟨(gpt, gcl, tk), hP, h⟩
      obtain ⟨gpt, gcl, tk⟩ := w
      dsimp only
      have h0 : P → IsExt c m rounds.toList[k][0]! (grounds[k]!)[0]! := fun h => ((hmsg h).2 0 (hpos h)).mono le
      refine Sim.tbind₀ (Sim.lift (Sim.extSub (fun h => h.2.2.2.1) fun h => h0 h.1)) fun atOne e₁ m₁ l₁ => ?_
      refine Sim.tbind₀ (observeExts hp (fun h => h.2.2 ▸ h.1.2.2.2.2.mono l₁)
        fun h => (hmsg h.1.1).mono (le.trans l₁)) fun u e₂ m₂ l₂ => ?_
      refine Sim.tbind₀ (Sim.sampleExt hp fun h => h.2) fun r e₃ m₃ l₃ => ?_
      refine Sim.tbind₀ (Sim.lift (Sim.interpolate (gvalues := Replay.roundValues grounds[k]! gcl) (fun h => ?_)
        fun h => h.2.1)) fun cl' e₄ m₄ l₄ => ?_
      · unfold Replay.roundValues
        exact ((IsExts.empty.push ((h0 h.1.1.1.1).mono (l₁.trans (l₂.trans l₃)))).push
          (h.1.1.2.1.mono (l₂.trans l₃))).append
          (((hmsg h.1.1.1.1).mono (le.trans (l₁.trans (l₂.trans l₃)))).extract 1)
      rw [StateT.run_pure]
      refine Sim.pure fun h => ⟨rfl, ?_⟩
      have hg : k < grounds.size := (hr h.1.1.1.1.1).1 ▸ hk'
      rw [List.take_add_one, List.getElem?_eq_getElem (by simpa using hg), Option.toList_some,
        Replay.sumcheckRounds_append, Array.getElem_toList, ← getElem!_pos grounds k hg]
      exact ⟨_, _, _, Replay.runs_bind h.1.1.1.1.2.1 rfl,
        (h.1.1.1.1.2.2.1.mono (l₁.trans (l₂.trans (l₃.trans l₄)))).push (h.1.2.1.mono l₄), h.2.1,
        h.2.2 ▸ h.1.2.2.mono l₄⟩
    · rw [show rounds.toList.length = grounds.toList.length by simp [(hr h).1], List.take_length] at q
      exact q
  · rw [StateT.run_pure]
    exact Sim.pure fun h => h.2

end Sim

end CompressionLean.Circuit
