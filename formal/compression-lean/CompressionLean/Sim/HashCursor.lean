import CompressionLean.Sim.HashState

namespace CompressionLean.Sim

open CompressionLean.Circuit
open CompressionLean.Lib
open CompressionLean.Replay (Word)
open NightstreamFPrime.Spec (goldilocksModulus)

variable {F : Type} [CommRing F] [CharP F goldilocksModulus] {c : Ctx F} {n : ℕ}

attribute [local irreducible] Replay.permute

instance Duplex.sampleIf_appends (enabled : Var) (d : Duplex) : Appends ((Duplex.sampleIf enabled).run d) := by
  unfold Duplex.sampleIf; appends

private theorem bool_mul (a b : Bool) :
    (Word.cast (if a then 1 else 0) : F) * Word.cast (if b then 1 else 0) =
      Word.cast (if a && b then 1 else 0) := by
  cases a <;> cases b <;> simp

theorem Duplex.sampleIf_ensures {d : Duplex} {r : Replay.Duplex} {enabled : Var} {take : Bool}
    (hp : Permutes c.perm) (hd : IsDuplex c n d r)
    (hen : IsWord c n enabled (if take then 1 else 0)) (empty : r.input = #[]) :
    Ensures c n ((Duplex.sampleIf enabled).run d) fun res m =>
      IsWord c m res.1 (if take then r.sample.1 else 0) ∧
        IsDuplex c m res.2 (if take then r.sample.2 else r) := by
  let refill := take && r.output.isEmpty
  let base := if refill then r.duplexing else r
  let final := if take then { base with output := base.output.pop } else base
  have baseInput : base.input = #[] := by cases h : refill <;> simp [base, h, empty, duplexing_empty r empty]
  have baseWidth : base.state.size = 16 := by
    cases h : refill <;> simp [base, h, duplexing_empty r empty, permute_size, hd.width]
  have baseBound : base.output.size ≤ rate := by
    cases h : refill
    · simpa [base, h] using hd.outputBound
    · simp [base, h, duplexing_output_size, rate, Replay.rate]
  have basePos : take = true → 0 < base.output.size := by
    intro ht
    by_cases he : r.output.isEmpty = true
    · simp [base, refill, ht, he, duplexing_output_size, Replay.rate]
    · have hs : r.output.size ≠ 0 := by simpa [Array.isEmpty_iff_size_eq_zero] using he
      simpa [base, refill, ht, he] using Nat.pos_of_ne_zero hs
  have finalEq : final = if take then r.sample.2 else r := by
    cases take <;> simp [final, base, refill, Replay.Duplex.sample, empty]
  have sampleEq : (if take then base.output.back! else 0) = if take then r.sample.1 else 0 := by
    cases take <;> simp [base, refill, Replay.Duplex.sample, empty]
  have stateEq : (Array.range 16).map (fun i => if refill then (Replay.permute r.state)[i]! else r.state[i]!) =
      base.state := by
    cases h : refill
    · simpa [base, h] using map_range_words r.state 16 hd.width
    · simpa [base, h, duplexing_empty r empty] using map_range_words (Replay.permute r.state) 16 (permute_size _)
  have queueEq : (Array.range rate).map (fun i =>
        if refill then (Replay.permute r.state)[rate - 1 - i]! else (queueWords r.output)[i]!) =
      queueWords base.output := by
    cases h : refill
    · simpa [base, h] using map_range_words (queueWords r.output) rate (queueWords_size _)
    · simpa [base, h, duplexing_empty r empty, Replay.rate, rate] using
        (queueWords_extract (Replay.permute r.state) (permute_size _)).symm
  have cursorEq : (Array.range 13).map (fun i =>
        if refill then (if i = rate then (1 : Word) else 0) else (if i = r.output.size then 1 else 0)) =
      cursorWords base.output.size := by
    cases h : refill <;> simp [base, h, duplexing_output_size, cursorWords, Replay.rate, rate]
    intro _ _
    rfl
  have shiftedQueue : (Array.range rate).map (fun i =>
        if take then (queueWords base.output).getD (i + 1) 0 else (queueWords base.output)[i]!) =
      queueWords final.output := by
    cases h : take
    · simpa [final, h] using map_range_words (queueWords base.output) rate (queueWords_size _)
    · simpa [final, h] using queueWords_shift base.output baseBound
  have shiftedCursor : (Array.range 13).map (fun i =>
        if take then (cursorWords base.output.size).getD (i + 1) 0 else (cursorWords base.output.size)[i]!) =
      cursorWords final.output.size := by
    cases h : take
    · simpa [final, h] using map_range_words (cursorWords base.output.size) 13 (cursorWords_size _)
    · simpa [final, h] using cursorWords_shift base.output.size (basePos h) baseBound
  unfold Duplex.sampleIf
  rw [StateT.run_bind, StateT.run_get, pure_bind]
  dsimp only
  refine Ensures.liftBind (const_ensures 0) fun zeroWord m₁ le₁ hz => ?_
  refine Ensures.liftBind (const_ensures 1) fun one m₂ le₂ ho => ?_
  obtain ⟨hout, hcursor⟩ := hd.mono.promoted (hz.mono le₂) ho
  have hcursor0 := hcursor.get (i := 0) (by simp)
  have hcursor0' : IsWord c m₂
      (Array.ofFn (d.cursor.getD fun i => if i.val = d.output.size then one else zeroWord))[0]!
      (if r.output.isEmpty then 1 else 0) := by
    rw [cursorWords_get r.output.size 0 (by decide)] at hcursor0
    have e : (0 = r.output.size) ↔ r.output.isEmpty = true := by
      rw [Array.isEmpty_iff_size_eq_zero]
      exact eq_comm
    simpa only [e] using hcursor0
  refine Ensures.liftBind (mul_ensures (hen.mono (le₁.trans le₂)) hcursor0') fun ref m₃ le₃ href => ?_
  have href' : IsWord c m₃ ref (if refill then 1 else 0) := href.congr (by
    rw [word_mul]
    exact bool_mul take r.output.isEmpty)
  refine Ensures.liftBind (permute_ensures hp fun i => hd.state.mono.get (by rw [hd.width]; exact i.isLt))
    fun next m₄ le₄ hnext => ?_
  have hn := IsWords.ofFn (permute_size r.state) hnext
  refine Ensures.liftBind (mapWords_ensures (count := 16) (n := m₄)
    (f := fun i => Build.select ref d.state[i]! (Array.ofFn next)[i]!) (g := fun i =>
      if refill then (Replay.permute r.state)[i]! else r.state[i]!)
    (fun i hi m hm => select_ensures (href'.mono (le₄.trans hm))
      (hd.state.mono.get (by rw [hd.width]; exact hi)) (hn.mono.get (by rw [permute_size]; exact hi))))
    fun state m₅ le₅ hstate => ?_
  rw [stateEq] at hstate
  refine Ensures.liftBind (mapWords_ensures (count := rate) (n := m₅)
    (f := fun i => Build.select ref
      (if d.cursor.isSome then d.output else (Array.range rate).map fun j => d.output.reverse.getD j zeroWord)[i]!
      (Array.ofFn next)[rate - 1 - i]!) (g := fun i =>
      if refill then (Replay.permute r.state)[rate - 1 - i]! else (queueWords r.output)[i]!)
    (fun i hi m hm => select_ensures href'.mono (hout.mono.get (by simpa using hi))
      (hn.mono.get (by rw [permute_size]; dsimp [rate] at hi ⊢; omega))))
    fun output m₆ le₆ houtput => ?_
  rw [queueEq] at houtput
  refine Ensures.liftBind (mapWords_ensures (count := 13) (n := m₆)
    (f := fun i => Build.select ref
      (Array.ofFn (d.cursor.getD fun j => if j.val = d.output.size then one else zeroWord))[i]!
      (if i = rate then one else zeroWord)) (g := fun i =>
      if refill then (if i = rate then (1 : Word) else 0) else (if i = r.output.size then 1 else 0))
    (fun i hi m hm => ?_)) fun remaining m₇ le₇ hremaining => ?_
  · have old := (hcursor.mono (m := m) (by omega)).get (i := i) (by simpa using hi)
    rw [cursorWords_get _ _ hi] at old
    apply select_ensures href'.mono old
    split_ifs
    · exact ho.mono
    · exact hz.mono
  rw [cursorEq] at hremaining
  refine Ensures.liftBind (select_ensures hen.mono hz.mono (houtput.mono.get (i := 0) (by simp [rate])))
    fun result m₈ le₈ hresult => ?_
  have hresult' : IsWord c m₈ result (if take then r.sample.1 else 0) := by
    apply hresult.congr
    rw [← sampleEq]
    cases ht : take
    · rfl
    · rw [queueWords_head _ (basePos ht)]
  refine Ensures.liftBind (mapWords_ensures (count := rate) (n := m₈)
    (f := fun i => Build.select enabled output[i]! (output.getD (i + 1) zeroWord)) (g := fun i =>
      if take then (queueWords base.output).getD (i + 1) 0 else (queueWords base.output)[i]!)
    (fun i hi m hm => select_ensures hen.mono (houtput.mono.get (by simpa using hi))
      (houtput.mono.getD hz.mono (i + 1)))) fun output' m₉ le₉ houtput' => ?_
  rw [shiftedQueue] at houtput'
  refine Ensures.liftBind (mapWords_ensures (count := 13) (n := m₉)
    (f := fun i => Build.select enabled remaining[i]! (remaining.getD (i + 1) zeroWord)) (g := fun i =>
      if take then (cursorWords base.output.size).getD (i + 1) 0 else (cursorWords base.output.size)[i]!)
    (fun i hi m hm => select_ensures hen.mono (hremaining.mono.get (by simpa using hi))
      (hremaining.mono.getD hz.mono (i + 1)))) fun remaining' m₁₀ le₁₀ hremaining' => ?_
  rw [shiftedCursor] at hremaining'
  rw [StateT.run_bind, StateT.run_set, pure_bind, StateT.run_pure]
  refine Ensures.pure ⟨hresult'.mono, ?_⟩
  rw [← finalEq]
  refine IsDuplex.mk ?_ ?_ ⟨houtput'.mono, ?_⟩ ?_ ?_ ?_
  · cases ht : take <;> simpa [final, ht] using hstate.mono
  · have hi : final.input = #[] := by cases ht : take <;> simpa [final, ht] using baseInput
    rw [hi]
    exact IsWords.empty
  · intro i
    have h := hremaining'.get (i := i.val) (by simp)
    rw [cursorWords_get _ _ i.isLt] at h
    exact h
  · cases ht : take <;> simpa [final, ht] using baseWidth
  · cases ht : take
    · simpa [final, ht] using baseBound
    · simpa [final, ht] using (Nat.sub_le base.output.size 1).trans baseBound
  · intro _
    cases ht : take <;> simpa [final, ht] using baseInput

end CompressionLean.Sim
