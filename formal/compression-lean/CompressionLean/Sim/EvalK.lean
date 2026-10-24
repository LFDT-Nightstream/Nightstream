import CompressionLean.Sim.Ring

/-!
Owns the simulation of `eval_k` (`Mle.evalK`, `mle.rs:146-204`) against the
replay's `Replay.evalK`: the long division over the coordinate bits.

- The circuit keeps `Option` states and skips absent ones; the replay sums
  every transition. An absent state is a zero state (`IsState`), and one
  replay step is a sum over the `4·D` digits `(rem, below, bit)` in the
  circuit's order (`digitSum`).
- An identity of `Kx` holds when it holds in the challenge field `Cubic`
  (`kx_eq_of_toQ`).
-/

namespace CompressionLean.Sim

open CompressionLean.Circuit
open CompressionLean.Lib
open CompressionLean.Replay (Word Ext Kx D)
open NightstreamFPrime.Spec (goldilocksModulus)

/-! ## Identities in the challenge field -/

private theorem kx_eq_of_toQ {a b : Kx} (h : a.toQ (K := Cubic) = b.toQ) : a = b := by
  obtain ⟨ar, ai⟩ := a
  obtain ⟨br, bi⟩ := b
  have hr := congrArg QuadraticAlgebra.re h
  have hi := congrArg QuadraticAlgebra.im h
  simp only [Kx.toQ_re, Kx.toQ_im] at hr hi
  rw [Ext.eq_of_toK (K := Cubic) hr, Ext.eq_of_toK (K := Cubic) hi]

private theorem kx_add_zero (a : Kx) : a.add Kx.zero = a :=
  kx_eq_of_toQ (by rw [Kx.toQ_add, Kx.toQ_zero, add_zero])

private theorem kx_zero_add (a : Kx) : Kx.zero.add a = a :=
  kx_eq_of_toQ (by rw [Kx.toQ_add, Kx.toQ_zero, zero_add])

private theorem kx_zero_mul (a : Kx) : Kx.zero.mul a = Kx.zero :=
  kx_eq_of_toQ (by rw [Kx.toQ_mul, Kx.toQ_zero, zero_mul])

private theorem kx_zero_scale (e : Ext) : Kx.zero.scale e = Kx.zero :=
  kx_eq_of_toQ (by rw [Kx.toQ_scale, Kx.toQ_zero, zero_mul])

/-! ## Sums of `Kx` values -/

private theorem sumKx_succ (k : ℕ) (f : ℕ → Kx) : Replay.sumKx (k + 1) f = (Replay.sumKx k f).add (f k) := by
  simp only [Replay.sumKx, List.range_succ, List.foldl_append, List.foldl_cons, List.foldl_nil]

private theorem sumKx_congr {k : ℕ} {f g : ℕ → Kx} (h : ∀ i < k, f i = g i) :
    Replay.sumKx k f = Replay.sumKx k g := by
  induction k with
  | zero => simp only [Replay.sumKx, List.range_zero, List.foldl_nil]
  | succ k ih => rw [sumKx_succ, sumKx_succ, ih fun i hi => h i (by omega), h k (by omega)]

/-- A sum over `4·k` digits is the sum over `(rem, below, bit)`, digit `4·rem + 2·below + bit`. -/
private theorem sumKx_digits (k : ℕ) (f : ℕ → Kx) :
    Replay.sumKx (4 * k) f = Replay.sumKx k fun rem => Replay.sumKx 2 fun below => Replay.sumKx 2 fun bit =>
      f (4 * rem + 2 * below + bit) := by
  apply kx_eq_of_toQ
  simp only [Bridge.toQ_sumKx]
  induction k with
  | zero => simp
  | succ k ih =>
      rw [show 4 * (k + 1) = 4 * k + 4 by ring, Finset.sum_range_add, ih, Finset.sum_range_succ (n := k)]
      simp only [Finset.sum_range_succ, Finset.sum_range_zero]
      ring_nf

/-! ## The long division -/

variable {F : Type} [CommRing F]

/-- A long-division state of the circuit holds the replay's state; an absent state is zero. -/
private def IsState (c : Ctx F) (n : ℕ) : Option KxVar → Kx → Prop
  | none, g => g = Kx.zero
  | some a, g => IsKx c n a g

private theorem IsState.mono {c : Ctx F} {n m : ℕ} {o : Option KxVar} {g : Kx} (h : IsState c n o g)
    (le : n ≤ m := by omega) : IsState c m o g := by
  cases o with
  | none => exact h
  | some a => exact IsKx.mono h le

/-- The circuit's long-division states hold the replay's states `g rem below`. -/
private def IsStates (c : Ctx F) (n : ℕ) (states : Mle.DivisionStates) (g : ℕ → ℕ → Kx) : Prop :=
  states.size = D ∧
    ∀ rem < D, (states[rem]!).size = 2 ∧ ∀ below < 2, IsState c n (states[rem]!)[below]! (g rem below)

/-- The replay's next states after the first `i` digits of one step of the long division, digit
`4·rem + 2·below + bit` (`evalKStates`). -/
private def digitSum (gr : Array (Word × Word)) (gs : Array Ext) (blocks t : ℕ) (prev : Array (Array Kx))
    (i rem' below' : ℕ) : Kx :=
  Replay.sumKx i fun j => Replay.contribution (Replay.divStep blocks gs.size t (j / 4) (j / 2 % 2) (j % 2)) rem'
    below' (Replay.divFactor gr gs t (j % 2)) ((prev[j / 4]!)[j / 2 % 2]!)

private theorem digitSum_succ (gr : Array (Word × Word)) (gs : Array Ext) (blocks t : ℕ) (prev : Array (Array Kx))
    (i rem below : ℕ) :
    digitSum gr gs blocks t prev (i + 1) rem below = (digitSum gr gs blocks t prev i rem below).add
      (Replay.contribution (Replay.divStep blocks gs.size t (i / 4) (i / 2 % 2) (i % 2)) rem below
        (Replay.divFactor gr gs t (i % 2)) ((prev[i / 4]!)[i / 2 % 2]!)) :=
  sumKx_succ _ _

private theorem evalKStates_zero (gr : Array (Word × Word)) (gs : Array Ext) (blocks cb : ℕ) {rem below : ℕ} (hr : rem < D)
    (hb : below < 2) :
    ((Replay.evalKStates gr gs blocks cb 0)[rem]!)[below]! = if rem = 0 ∧ below = 0 then Kx.one else Kx.zero := by
  rw [Replay.evalKStates, getElem!_map_range _ _ hr, getElem!_map_range _ _ hb]

private theorem evalKStates_succ (gr : Array (Word × Word)) (gs : Array Ext) (blocks cb k : ℕ) {rem below : ℕ} (hr : rem < D)
    (hb : below < 2) :
    ((Replay.evalKStates gr gs blocks cb (k + 1))[rem]!)[below]! =
      digitSum gr gs blocks (cb - 1 - k) (Replay.evalKStates gr gs blocks cb k) (4 * D) rem below := by
  rw [Replay.evalKStates, getElem!_map_range _ _ hr, getElem!_map_range _ _ hb, digitSum, sumKx_digits]
  refine sumKx_congr fun r hr' => sumKx_congr fun b hb' => sumKx_congr fun t ht => ?_
  rw [show (4 * r + 2 * b + t) / 4 = r by omega, show (4 * r + 2 * b + t) / 2 % 2 = b by omega,
    show (4 * r + 2 * b + t) % 2 = t by omega]

private theorem contribution_absent (step : Option (ℕ × ℕ × ℕ)) (rem below : ℕ) (factor : ℕ → Kx) :
    Replay.contribution step rem below factor Kx.zero = Kx.zero := by
  rcases step with _ | ⟨r, b, q⟩
  · rfl
  · simp only [Replay.contribution]
    split_ifs
    · exact kx_zero_mul _
    · rfl

/-- `divFactor` as the circuit computes it: `1 - r_t` or `r_t`, then the weight of `s_t`. -/
private theorem divFactor_ifs (gr : Array (Word × Word)) (gs : Array Ext) (t bit q : ℕ) (hbit : bit < 2) :
    Replay.divFactor gr gs t bit q =
      if t < gs.size then
        (if bit = 0 then (Kx.ofK gr[t]!).oneMinus else Kx.ofK gr[t]!).scale
          (if q = 1 then gs[t]! else Replay.oneMinus gs[t]!)
      else if bit = 0 then (Kx.ofK gr[t]!).oneMinus else Kx.ofK gr[t]! := by
  unfold Replay.divFactor
  rcases (show bit = 0 ∨ bit = 1 by omega) with rfl | rfl <;> simp

/-- `divStep` as the circuit tests it: the quotient bit, then the flag. -/
private theorem divStep_ifs (blocks bb t rem below bit : ℕ) :
    Replay.divStep blocks bb t rem below bit =
      if t ≥ bb ∧ (if 2 * rem + bit ≥ D then 1 else 0) = 1 then none
      else (if below = 1 ∨ (if 2 * rem + bit ≥ D then 1 else 0) < blocks / 2 ^ t % 2 then some 1
        else if (if 2 * rem + bit ≥ D then 1 else 0) = blocks / 2 ^ t % 2 then some 0 else none).map
          fun nb => (2 * rem + bit - D * (if 2 * rem + bit ≥ D then 1 else 0), nb,
            if 2 * rem + bit ≥ D then 1 else 0) := by
  unfold Replay.divStep
  simp only [ge_iff_le]
  split_ifs <;> simp_all

namespace IsStates

variable {c : Ctx F} {n : ℕ} {next : Mle.DivisionStates} {g : ℕ → ℕ → Kx} {gr : Array (Word × Word)}
  {gs : Array Ext} {blocks t : ℕ} {prev : Array (Array Kx)} {i : ℕ}

private theorem mono (h : IsStates c n next g) {m : ℕ} (le : n ≤ m := by omega) : IsStates c m next g :=
  ⟨h.1, fun rem hr => ⟨(h.2 rem hr).1, fun below hb => ((h.2 rem hr).2 below hb).mono le⟩⟩

private theorem congr (h : IsStates c n next g) {g' : ℕ → ℕ → Kx}
    (e : ∀ rem < D, ∀ below < 2, g rem below = g' rem below) : IsStates c n next g' :=
  ⟨h.1, fun rem hr => ⟨(h.2 rem hr).1, fun below hb => e rem hr below hb ▸ (h.2 rem hr).2 below hb⟩⟩

private theorem existing (h : IsStates c n next g) {slot nb : ℕ} (hs : slot < D) (hb : nb < 2) {a : KxVar}
    (e : (next[slot]!)[nb]! = some a) : IsKx c n a (g slot nb) := by
  have := (h.2 slot hs).2 nb hb
  rwa [e] at this

private theorem absent (h : IsStates c n next g) {slot nb : ℕ} (hs : slot < D) (hb : nb < 2)
    (e : (next[slot]!)[nb]! = none) : g slot nb = Kx.zero := by
  have := (h.2 slot hs).2 nb hb
  rwa [e] at this

private theorem set (h : IsStates c n next g) (slot nb : ℕ) {a : KxVar} {v : Kx} (ha : IsKx c n a v) :
    IsStates c n (next.set! slot (next[slot]!.set! nb (some a)))
      (fun rem below => if rem = slot ∧ below = nb then v else g rem below) := by
  refine ⟨by simp [h.1], fun rem hr => ?_⟩
  rw [getElem!_set! _ _ _ _ (by rw [h.1]; exact hr)]
  by_cases e : slot = rem
  · subst e
    rw [if_pos rfl]
    refine ⟨by simp [(h.2 slot hr).1], fun below hb' => ?_⟩
    rw [getElem!_set! _ _ _ _ (by rw [(h.2 slot hr).1]; exact hb')]
    dsimp only
    by_cases e' : nb = below
    · subst e'
      rw [if_pos rfl, if_pos (⟨rfl, rfl⟩ : slot = slot ∧ nb = nb)]
      exact ha
    · rw [if_neg e', if_neg fun h' : slot = slot ∧ below = nb => e' h'.2.symm]
      exact (h.2 slot hr).2 below hb'
  · rw [if_neg e]
    exact ⟨(h.2 rem hr).1, fun below hb' => by
      dsimp only
      rw [if_neg fun h' : rem = slot ∧ below = nb => e h'.1.symm]
      exact (h.2 rem hr).2 below hb'⟩

/-- A digit that moves no state. -/
private theorem skip (h : IsStates c n next (digitSum gr gs blocks t prev i))
    (zero : ∀ rem below, Replay.contribution (Replay.divStep blocks gs.size t (i / 4) (i / 2 % 2) (i % 2)) rem below
      (Replay.divFactor gr gs t (i % 2)) ((prev[i / 4]!)[i / 2 % 2]!) = Kx.zero) :
    IsStates c n next (digitSum gr gs blocks t prev (i + 1)) :=
  h.congr fun rem _ below _ => by rw [digitSum_succ, zero, kx_add_zero]

/-- A digit `(R, B, Bt)` that adds `x·factor` to state `(slot, nb)`. -/
private theorem move {R B Bt : ℕ} (h : IsStates c n next (digitSum gr gs blocks t prev (4 * R + 2 * B + Bt)))
    (hB : B < 2) (hBt : Bt < 2) {slot nb q : ℕ} (step : Replay.divStep blocks gs.size t R B Bt = some (slot, nb, q))
    {a : KxVar} (ha : IsKx c n a ((digitSum gr gs blocks t prev (4 * R + 2 * B + Bt) slot nb).add
      (((prev[R]!)[B]!).mul (Replay.divFactor gr gs t Bt q)))) :
    IsStates c n (next.set! slot (next[slot]!.set! nb (some a)))
      (digitSum gr gs blocks t prev (4 * R + 2 * B + Bt + 1)) := by
  have d1 : (4 * R + 2 * B + Bt) / 4 = R := by omega
  have d2 : (4 * R + 2 * B + Bt) / 2 % 2 = B := by omega
  have d3 : (4 * R + 2 * B + Bt) % 2 = Bt := by omega
  refine (h.set slot nb ha).congr fun rem _ below _ => ?_
  rw [digitSum_succ, d1, d2, d3, step]
  simp only [Replay.contribution]
  by_cases e : slot = rem ∧ nb = below
  · obtain ⟨rfl, rfl⟩ := e
    rw [if_pos ⟨rfl, rfl⟩, if_pos ⟨rfl, rfl⟩]
  · rw [if_neg fun h' => e ⟨h'.1.symm, h'.2.symm⟩, if_neg e, kx_add_zero]

end IsStates

instance Mle.evalK_appends (r : Array KVar) (sBlock tau : Array ExtVar) (blocks : ℕ) :
    Appends (Mle.evalK r sBlock tau blocks) := by
  unfold Mle.evalK; appends

section EvalK

variable [CharP F goldilocksModulus] {c : Ctx F} {n : ℕ}

/-- `eval_k` (`mle.rs:146-204`): the long division's value. An absent state
of the circuit is a zero state of the replay. -/
theorem Mle.evalK_ensures {r : Array KVar} {sBlock tau : Array ExtVar} {gr : Array (Word × Word)}
    {gs gt : Array Ext} (blocks : ℕ) (rSize : Nat.log2 (D * 2 ^ gs.size).nextPowerOfTwo ≤ gr.size)
    (tauSize : D ≤ gt.size) (hr : IsKs c n r gr) (hs : IsExts c n sBlock gs) (ht : IsExts c n tau gt) :
    Ensures c n (Mle.evalK r sBlock tau blocks) fun res m => IsKx c m res (Replay.evalK gr gs gt blocks) := by
  unfold Mle.evalK Replay.evalK
  dsimp only
  generalize hcb : (D * 2 ^ sBlock.size).nextPowerOfTwo.log2 = cb
  have bits : cb = (D * 2 ^ gs.size).nextPowerOfTwo.log2 := by rw [← hcb, hs.1]
  refine KxVar.zero_ensures.bind fun zero _ _ hzero => KxVar.one_ensures.bind fun one m₂ _ hone => ?_
  refine (Ensures.loop (fun k states m => IsStates c m states fun rem below =>
      ((Replay.evalKStates gr gs blocks cb k)[rem]!)[below]!) ?_ fun k hk states m _ hstates => ?_).bind
    fun states _ _ hstates => ?_
  · refine ⟨by simp, fun rem hr => ?_⟩
    rw [getElem!_set! _ _ _ _ (by simpa using hr)]
    by_cases e : 0 = rem
    · subst e
      rw [if_pos rfl]
      refine ⟨rfl, fun below hb => ?_⟩
      beta_reduce
      rw [evalKStates_zero _ _ _ _ hr hb]
      rcases (show below = 0 ∨ below = 1 by omega) with rfl | rfl
      · show IsKx c m₂ one _
        rw [if_pos ⟨rfl, rfl⟩]
        exact hone
      · show _ = Kx.zero
        rw [if_neg (by omega)]
    · rw [if_neg e, getElem!_pos _ rem (by simpa using hr), Array.getElem_replicate]
      refine ⟨rfl, fun below hb => ?_⟩
      beta_reduce
      rw [evalKStates_zero _ _ _ _ hr hb, if_neg (by omega)]
      rcases (show below = 0 ∨ below = 1 by omega) with rfl | rfl <;> rfl
  · have hk' : k < cb := by simpa using hk
    rw [show (List.range cb).reverse[k] = cb - 1 - k by simp [List.getElem_reverse]]
    generalize ht' : cb - 1 - k = t
    have hrt : IsK c n r[t]! gr[t]! := hr.get (by omega)
    refine (KxVar.fromK_ensures (g₁ := gr[t]!.1) (g₂ := gr[t]!.2) hrt.1.mono hrt.2.mono).bind
      fun rT _ _ hrT => ?_
    refine (KxVar.oneMinus_ensures hrT).bind fun rLow m₅ _ hrLow => ?_
    obtain ⟨prev, hprev⟩ : ∃ prev, prev = Replay.evalKStates gr gs blocks cb k := ⟨_, rfl⟩
    rw [← hprev] at hstates
    refine (Ensures.loopRange (fun R next m' => IsStates c m' next (digitSum gr gs blocks t prev (4 * R))) ?_
      fun R hR next m' _ hnext => ?_).bind fun next _ _ hnext => Ensures.pure ⟨rfl, ?_⟩
    · exact ⟨by simp, fun rem hr => ⟨by simp [getElem!_pos, hr], fun below hb => by
        rw [getElem!_pos _ rem (by simpa using hr), Array.getElem_replicate]
        rcases (show below = 0 ∨ below = 1 by omega) with rfl | rfl <;> rfl⟩⟩
    · have hR' : R < D := by simpa using hR
      rw [Nat.zero_add]
      refine (Ensures.loopRange
        (fun B next' m'' => IsStates c m'' next' (digitSum gr gs blocks t prev (4 * R + 2 * B)))
        hnext fun B hB next' m'' _ hnext' => ?_).bind fun next₂ _ _ hnext₂ =>
        Ensures.pure ⟨rfl, by rw [show 4 * (R + 1) = 4 * R + 2 * (2 - 0) by omega]; exact hnext₂⟩
      have hB' : B < 2 := by simpa using hB
      rw [Nat.zero_add]
      rcases hst : (states[R]!)[B]! with _ | value
      · refine Ensures.pure ⟨rfl, ?_⟩
        have e := hstates.absent hR' hB' hst
        have z : ∀ i, i / 4 = R ∧ i / 2 % 2 = B → ∀ rem below,
            Replay.contribution (Replay.divStep blocks gs.size t (i / 4) (i / 2 % 2) (i % 2)) rem below
              (Replay.divFactor gr gs t (i % 2)) ((prev[i / 4]!)[i / 2 % 2]!) = Kx.zero :=
          fun i hi rem below => by rw [hi.1, hi.2, e]; exact contribution_absent _ _ _ _
        have two := (hnext'.skip (z _ ⟨by omega, by omega⟩)).skip (z _ ⟨by omega, by omega⟩)
        rwa [show 4 * R + 2 * B + 1 + 1 = 4 * R + 2 * (B + 1) by ring] at two
      have hvalue := hstates.existing hR' hB' hst
      refine (Ensures.loopRange
        (fun Bt next'' m''' => IsStates c m''' next'' (digitSum gr gs blocks t prev (4 * R + 2 * B + Bt)))
        hnext' fun Bt hBt next'' m''' _ hnext'' => ?_).bind fun next₃ _ _ hnext₃ =>
        Ensures.pure ⟨rfl, by rw [show 4 * R + 2 * (B + 1) = 4 * R + 2 * B + (2 - 0) by omega]; exact hnext₃⟩
      have hBt' : Bt < 2 := by simpa using hBt
      rw [Nat.zero_add]
      have dec : (4 * R + 2 * B + Bt) / 4 = R ∧ (4 * R + 2 * B + Bt) / 2 % 2 = B ∧
          (4 * R + 2 * B + Bt) % 2 = Bt := ⟨by omega, by omega, by omega⟩
      have hstep := divStep_ifs blocks gs.size t R B Bt
      have hslot : 2 * R + Bt - D * (if 2 * R + Bt ≥ D then 1 else 0) < D := by split_ifs <;> omega
      generalize (if 2 * R + Bt ≥ D then 1 else 0) = q at hstep hslot
      generalize hnbv : (if B = 1 ∨ q < blocks / 2 ^ t % 2 then some 1 else if q = blocks / 2 ^ t % 2 then some 0
        else none) = nbo at hstep
      by_cases skip : t ≥ sBlock.size ∧ q = 1
      · rw [if_pos skip]
        refine Ensures.pure ⟨rfl, hnext''.skip fun rem below => ?_⟩
        rw [dec.1, dec.2.1, dec.2.2, hstep, if_pos (by rw [← hs.1]; exact skip)]
        rfl
      rw [if_neg skip]
      rcases nbo with _ | nb
      · refine Ensures.pure ⟨rfl, hnext''.skip fun rem below => ?_⟩
        rw [dec.1, dec.2.1, dec.2.2, hstep, if_neg (by rw [← hs.1]; exact skip)]
        rfl
      dsimp only
      have hnb2 : nb < 2 := by
        split_ifs at hnbv <;> cases hnbv <;> omega
      have step : Replay.divStep blocks gs.size t R B Bt = some (2 * R + Bt - D * q, nb, q) := by
        rw [hstep, if_neg (by rw [← hs.1]; exact skip)]
        rfl
      have hX : IsKx c m₅ (if Bt = 0 then rLow else rT)
          (if Bt = 0 then (Kx.ofK gr[t]!).oneMinus else Kx.ofK gr[t]!) := by
        split_ifs
        · exact hrLow
        · exact hrT.mono
      have fac : Replay.divFactor gr gs t Bt q = if t < sBlock.size then
          (if Bt = 0 then (Kx.ofK gr[t]!).oneMinus else Kx.ofK gr[t]!).scale
            (if q = 1 then gs[t]! else Replay.oneMinus gs[t]!)
          else if Bt = 0 then (Kx.ofK gr[t]!).oneMinus else Kx.ofK gr[t]! := by
        rw [divFactor_ifs _ _ _ _ _ hBt', hs.1]
      rcases hex : (next''[2 * R + Bt - D * q]!)[nb]! with _ | existing
      · by_cases hlt : t < sBlock.size
        · rw [if_pos hlt]
          by_cases hq1 : q = 1
          · rw [if_pos hq1, pure_bind]
            refine (KxVar.scale_ensures hX.mono (hs.get (by rw [← hs.1]; exact hlt)).mono).bind
              fun factor _ _ hfactor => ?_
            refine (KxVar.mul_ensures hvalue.mono hfactor).bind fun product _ _ hproduct => ?_
            dsimp only
            rw [pure_bind]
            refine Ensures.pure ⟨rfl, hnext''.mono.move hB' hBt' step ?_⟩
            rw [hnext''.absent hslot hnb2 hex, kx_zero_add, fac, if_pos hlt, if_pos hq1]
            exact hproduct
          · rw [if_neg hq1]
            refine (oneMinus_ensures (hs.get (by rw [← hs.1]; exact hlt)).mono).bind
              fun weight _ _ hweight => ?_
            refine (KxVar.scale_ensures hX.mono hweight).bind fun factor _ _ hfactor => ?_
            refine (KxVar.mul_ensures hvalue.mono hfactor).bind fun product _ _ hproduct => ?_
            dsimp only
            rw [pure_bind]
            refine Ensures.pure ⟨rfl, hnext''.mono.move hB' hBt' step ?_⟩
            rw [hnext''.absent hslot hnb2 hex, kx_zero_add, fac, if_pos hlt, if_neg hq1]
            exact hproduct
        · rw [if_neg hlt]
          refine (KxVar.mul_ensures hvalue.mono hX.mono).bind fun product _ _ hproduct => ?_
          dsimp only
          rw [pure_bind]
          refine Ensures.pure ⟨rfl, hnext''.mono.move hB' hBt' step ?_⟩
          rw [hnext''.absent hslot hnb2 hex, kx_zero_add, fac, if_neg hlt]
          exact hproduct
      · by_cases hlt : t < sBlock.size
        · rw [if_pos hlt]
          by_cases hq1 : q = 1
          · rw [if_pos hq1, pure_bind]
            refine (KxVar.scale_ensures hX.mono (hs.get (by rw [← hs.1]; exact hlt)).mono).bind
              fun factor _ _ hfactor => ?_
            refine (KxVar.mul_ensures hvalue.mono hfactor).bind fun product _ _ hproduct => ?_
            dsimp only
            refine (KxVar.add_ensures (hnext''.existing hslot hnb2 hex).mono hproduct).bind
              fun merged _ _ hmerged => Ensures.pure ⟨rfl, hnext''.mono.move hB' hBt' step ?_⟩
            rw [fac, if_pos hlt, if_pos hq1]
            exact hmerged
          · rw [if_neg hq1]
            refine (oneMinus_ensures (hs.get (by rw [← hs.1]; exact hlt)).mono).bind
              fun weight _ _ hweight => ?_
            refine (KxVar.scale_ensures hX.mono hweight).bind fun factor _ _ hfactor => ?_
            refine (KxVar.mul_ensures hvalue.mono hfactor).bind fun product _ _ hproduct => ?_
            dsimp only
            refine (KxVar.add_ensures (hnext''.existing hslot hnb2 hex).mono hproduct).bind
              fun merged _ _ hmerged => Ensures.pure ⟨rfl, hnext''.mono.move hB' hBt' step ?_⟩
            rw [fac, if_pos hlt, if_neg hq1]
            exact hmerged
        · rw [if_neg hlt]
          refine (KxVar.mul_ensures hvalue.mono hX.mono).bind fun product _ _ hproduct => ?_
          dsimp only
          refine (KxVar.add_ensures (hnext''.existing hslot hnb2 hex).mono hproduct).bind
            fun merged _ _ hmerged => Ensures.pure ⟨rfl, hnext''.mono.move hB' hBt' step ?_⟩
          rw [fac, if_neg hlt]
          exact hmerged
    · refine hnext.congr fun rem hr below hb => ?_
      rw [show 4 * (D - 0) = 4 * D by omega, evalKStates_succ _ _ _ _ _ hr hb, ht', hprev]
  obtain ⟨last, hlast⟩ : ∃ last, last = Replay.evalKStates gr gs blocks cb cb := ⟨_, rfl⟩
  rw [show (List.range cb).reverse.length = cb by simp, ← hlast] at hstates
  refine (Ensures.loopArray (fun j total m => IsKx c m total
      (Replay.sumKx j fun rem => ((last[rem]!)[1]!).scale gt[rem]!)) hzero.mono
    fun j hj total m _ htotal => ?_).bind fun total _ _ htotal => ?_
  · have hj' : j < states.size := by simpa using hj
    rw [show states.zipIdx[j] = (states[j]!, j) by simp [getElem!_pos, hj']]
    dsimp only
    rcases hst : (states[j]!)[1]! with _ | value
    · refine Ensures.pure ⟨rfl, ?_⟩
      rw [sumKx_succ, hstates.absent (hstates.1 ▸ hj') (by omega) hst, kx_zero_scale, kx_add_zero]
      exact htotal
    · refine (KxVar.scale_ensures (hstates.existing (hstates.1 ▸ hj') (by omega) hst).mono
        (ht.get (by have := hstates.1; omega)).mono).bind fun term _ _ hterm => ?_
      exact (KxVar.add_ensures htotal.mono hterm).bind fun total' _ _ htotal' =>
        Ensures.pure ⟨rfl, by rw [sumKx_succ]; exact htotal'⟩
  have hk : ∀ j (hj : j < (r.extract cb r.size).size),
      IsK c n (r.extract cb r.size)[j] (gr.extract cb gr.size)[j]! := fun j hj => by
    have e := (hr.extract cb).get (i := j) (by rw [Array.size_extract, ← hr.1]; simpa using hj)
    rwa [getElem!_pos (r.extract _) j hj] at e
  rw [Array.size_zipIdx, hstates.1] at htotal
  refine (Ensures.loopArray (fun j acc m => IsKx c m acc (((gr.extract cb gr.size).toList.take j).foldl
      (fun total value => total.mul (Kx.ofK value).oneMinus)
      (Replay.sumKx D fun rem => ((last[rem]!)[1]!).scale gt[rem]!)))
    (by rw [List.take_zero, List.foldl_nil]; exact htotal) fun j hj acc m _ hacc => ?_).bind
    fun res _ _ hres => Ensures.pure ?_
  · refine (KxVar.fromK_ensures (g₁ := ((gr.extract cb gr.size)[j]!).1) (g₂ := ((gr.extract cb gr.size)[j]!).2)
      (hk j hj).1.mono (hk j hj).2.mono).bind fun x _ _ hx => ?_
    refine (KxVar.oneMinus_ensures hx).bind fun f _ _ hf => ?_
    refine (KxVar.mul_ensures hacc.mono hf).bind fun total' _ _ htotal' => Ensures.pure ⟨rfl, ?_⟩
    have size := hr.1
    rw [List.take_succ_eq_append_getElem (by simp only [Array.length_toList, Array.size_extract]; simp at hj; omega),
      List.foldl_append, List.foldl_cons, List.foldl_nil, Array.getElem_toList, ← getElem!_pos]
    exact htotal'
  · rw [List.take_of_length_le (by simp [hr.1]), Array.foldl_toList, hlast, ← hcb, hs.1] at hres
    exact hres

end EvalK

end CompressionLean.Sim
