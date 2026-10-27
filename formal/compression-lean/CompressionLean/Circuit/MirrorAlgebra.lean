import CompressionLean.Circuit.Mirror

/-!
Owns the simulation of the verifier's algebra against the replay's
formulas (`Replay/Formulas.lean`, `Replay/Gkr.lean`): arrays of extension
values (`IsExts`), `eq_eval`, `interpolate`, observing extension values,
and the sum-check replay as a mirrored transcript loop (`Mirrors.sumcheck`).

A circuit loop and its replay formula may group a sum or a product
differently; the two values are equal in `Cubic` (`toC_inj`).
-/

namespace CompressionLean.Circuit
open CompressionLean.Replay (Gl Ext)
open NightstreamFPrime.Spec (goldilocksModulus)

variable {F : Type} [CommRing F]

/-- `as` and `es` have one size, and each handle triple is its value. -/
def IsExts (c : Ctx F) (n : ℕ) (as : Array ExtVar) (es : Array Ext) : Prop :=
  as.size = es.size ∧ ∀ i < as.size, IsExt c n as[i]! es[i]!

namespace IsExts
variable {c : Ctx F} {n : ℕ} {as : Array ExtVar} {es : Array Ext}
theorem mono (h : IsExts c n as es) {m : ℕ} (le : n ≤ m) : IsExts c m as es := ⟨h.1, fun i hi => (h.2 i hi).mono le⟩
theorem get (h : IsExts c n as es) {i : ℕ} (hi : i < es.size) : IsExt c n as[i]! es[i]! := h.2 i (h.1 ▸ hi)
theorem empty : IsExts c n #[] #[] := ⟨rfl, fun _ h => absurd h (by simp)⟩
theorem push (h : IsExts c n as es) {a : ExtVar} {e : Ext} (ha : IsExt c n a e) :
    IsExts c n (as.push a) (es.push e) := by
  refine ⟨by simp [h.1], fun i hi => ?_⟩
  rw [Array.size_push] at hi
  rw [getElem!_pos _ _ (by simp; omega), getElem!_pos _ _ (by simp [← h.1]; omega), Array.getElem_push,
    Array.getElem_push]
  by_cases lt : i < as.size
  · rw [dif_pos lt, dif_pos (h.1 ▸ lt), ← getElem!_pos, ← getElem!_pos]
    exact h.2 i lt
  · rw [dif_neg lt, dif_neg (h.1 ▸ lt)]
    exact ha
theorem extract (h : IsExts c n as es) (a b : ℕ) : IsExts c n (as.extract a b) (es.extract a b) := by
  refine ⟨by simp [h.1], fun i hi => ?_⟩
  rw [Array.size_extract] at hi
  rw [getElem!_pos _ _ (by simp; omega), getElem!_pos _ _ (by simp [← h.1]; omega), Array.getElem_extract,
    Array.getElem_extract, ← getElem!_pos, ← getElem!_pos]
  exact h.2 _ (by omega)
theorem extractTail (h : IsExts c n as es) (a : ℕ) : IsExts c n (as.extract a) (es.extract a) := by
  have := h.extract a as.size
  rwa [h.1] at this ⊢
theorem append (h : IsExts c n as es) {bs : Array ExtVar} {fs : Array Ext} (h' : IsExts c n bs fs) :
    IsExts c n (as ++ bs) (es ++ fs) := by
  refine ⟨by simp [h.1, h'.1], fun i hi => ?_⟩
  rw [Array.size_append] at hi
  rw [getElem!_pos _ _ (by simp; omega), getElem!_pos _ _ (by simp [← h.1, ← h'.1]; omega), Array.getElem_append,
    Array.getElem_append]
  by_cases lt : i < as.size
  · rw [dif_pos lt, dif_pos (h.1 ▸ lt), ← getElem!_pos, ← getElem!_pos]
    exact h.2 i lt
  · rw [dif_neg lt, dif_neg (h.1 ▸ lt), ← getElem!_pos, ← getElem!_pos]
    simp only [h.1]
    exact h'.2 _ (by rw [← h.1]; omega)
end IsExts

theorem IsExt.of_eq {c : Ctx F} {n : ℕ} {a : ExtVar} {e e' : Ext} (h : IsExt c n a e) (he : e = e') :
    IsExt c n a e' := he ▸ h


theorem zipIdx_getElem {α : Type} [Inhabited α] (xs : Array α) {k : ℕ} (hk : k < xs.zipIdx.toList.length) :
    xs.zipIdx.toList[k] = (xs[k]!, k) := by
  simp only [Array.length_toList, Array.size_zipIdx] at hk
  rw [Array.getElem_toList, Array.getElem_zipIdx, getElem!_pos xs k hk, Nat.zero_add]

/-! ## Products and sums in `Cubic` -/

theorem toC_foldl_mul {α : Type} (f : α → Ext) : ∀ (l : List α) (init : Ext),
    toC (l.foldl (fun t x => t * f x) init) = toC init * (l.map fun x => toC (f x)).prod
  | [], _ => by simp
  | x :: l, init => by rw [List.foldl_cons, toC_foldl_mul f l, toC_mul]; simp [mul_assoc]

theorem toC_eqEvalList : ∀ l1 l2 : List Ext,
    toC (Replay.eqEvalList l1 l2) = ((l1.zip l2).map fun p => toC (Replay.eq1 p.1 p.2)).prod
  | x :: xs, y :: ys => by rw [Replay.eqEvalList, toC_mul, toC_eqEvalList xs ys]; simp
  | [], _ => by simp [Replay.eqEvalList, toC_one]
  | _ :: _, [] => by simp [Replay.eqEvalList, toC_one]

theorem zip_getElem {α β : Type} [Inhabited α] [Inhabited β] (xs : Array α) (ys : Array β) {k : ℕ}
    (hk : k < (xs.zip ys).toList.length) : (xs.zip ys).toList[k] = (xs[k]!, ys[k]!) := by
  simp only [Array.length_toList, Array.size_zip] at hk
  rw [Array.getElem_toList, Array.getElem_zip, getElem!_pos xs k (by omega), getElem!_pos ys k (by omega)]

namespace Sim
variable [CharP F goldilocksModulus] {c : Ctx F} {n : ℕ} {P : Prop}

/-- `eq_eval`: `Π_t eq(x_t, y_t)` over the common coordinates. -/
theorem eqEval {xs ys : Array ExtVar} {gxs gys : Array Ext} (hx : P → IsExts c n xs gxs)
    (hy : P → IsExts c n ys gys) :
    Sim c n P (Algebra.eqEval xs ys) (fun r m => IsExt c m r (Replay.eqEval gxs gys)) True := by
  unfold Algebra.eqEval
  simp only [← Array.forIn_toList]
  let zs := (gxs.zip gys).toList
  have len : P → (xs.zip ys).toList.length = zs.length := fun h => by simp [zs, (hx h).1, (hy h).1]
  refine (extOne.bind fun one m₁ l₁ => ((Sim.loop (I := fun k acc m => IsExt c m acc
    ((zs.take k).foldl (fun t p => t * Replay.eq1 p.1 p.2) 1)) (D := fun _ => True) fun k hk acc m le => ?_).pre
      fun h => ⟨h, by simpa using h.2⟩).bind fun total m₂ l₂ => Sim.pure fun h => ?_).post
        (fun _ _ _ h => h) fun _ => by simp
  · rw [zip_getElem xs ys hk]
    dsimp only
    have hk' : P → k < zs.length := fun h => len h ▸ hk
    have hxk : P → IsExt c m xs[k]! gxs[k]! := fun h =>
      ((hx h).get (by have := hk' h; simp [zs] at this; omega)).mono (l₁.trans le)
    have hyk : P → IsExt c m ys[k]! gys[k]! := fun h =>
      ((hy h).get (by have := hk' h; simp [zs] at this; omega)).mono (l₁.trans le)
    refine ((eq1 (fun h => hxk h.1.1) fun h => hyk h.1.1).bind fun factor m₃ l₃ =>
      (Sim.extMul (fun h => h.1.2.mono l₃) fun h => h.2).bind fun total m₄ l₄ =>
        Sim.pure fun h => ⟨rfl, ?_⟩).post (fun _ _ _ h => h) fun _ => by simp
    have hP := h.1.1.1.1
    have e : zs[k]'(hk' hP) = (gxs[k]!, gys[k]!) := by
      simp only [zs]
      exact zip_getElem gxs gys (hk' hP)
    rw [List.take_succ_eq_append_getElem (hk' hP), List.foldl_append, e]
    exact h.2
  · rw [len h.1.1, List.take_length] at h
    refine h.2.of_eq (toC_inj ?_)
    rw [Replay.eqEval, toC_eqEvalList, toC_foldl_mul, toC_one, one_mul]
    simp [zs, Array.toList_zip]

omit [CharP F goldilocksModulus] in
theorem range_foldl_succ {α : Type} (f : α → ℕ → α) (init : α) (k : ℕ) :
    (List.range (k + 1)).foldl f init = f ((List.range k).foldl f init) k := by
  rw [List.range_succ, List.foldl_append]
  rfl

theorem gl_mul_one (g : Gl) : g * 1 = g := Replay.Gl.eq_of_toF (by rw [Replay.Gl.toF_mul, Replay.Gl.toF_one, mul_one])

theorem ext_mul_one (e : Ext) : e * 1 = e := toC_inj (by rw [toC_mul, toC_one, mul_one])

/-- `interpolate`: the polynomial through `(i, values[i])` at `x`. -/
theorem interpolate {vs : Array ExtVar} {gvs : Array Ext} {x : ExtVar} {gx : Ext} (hv : P → IsExts c n vs gvs)
    (hx : P → IsExt c n x gx) :
    Sim c n P (Algebra.interpolate vs x) (fun r m => IsExt c m r (Replay.interpolate gvs gx)) True := by
  unfold Algebra.interpolate
  simp only [← Array.forIn_toList, Std.Legacy.Range.forIn_eq_forIn_range']
  let num (i : ℕ) : ℕ → Ext := fun j => if i = j then 1 else gx - Ext.ofGl (Replay.Gl.ofNat j)
  let den (i : ℕ) : ℕ → Gl := fun j => if i = j then 1 else Replay.Gl.ofNat i - Replay.Gl.ofNat j
  let term (i : ℕ) : Ext :=
    gvs[i]! * (Replay.prodRange gvs.size (num i)).scale (Replay.prodRangeGl gvs.size (den i)).inv
  refine (extZero.bind fun zero m₁ l₁ => ((Sim.loop (I := fun k acc m => IsExt c m acc (Replay.sumRange k term))
    (D := fun _ => True) fun k hk acc m le => ?_).pre fun h => ⟨h, h.2⟩).bind fun total m₂ l₂ =>
      Sim.pure fun h => ?_).post (fun _ _ _ h => h) fun _ => by simp
  · have hP : P → vs.size = gvs.size := fun h => (hv h).1
    rw [zipIdx_getElem vs hk]
    dsimp only
    rw [show ([:vs.size] : Std.Legacy.Range).size = vs.size by simp [Std.Legacy.Range.size], ← List.range_eq_range']
    have hvk : P → IsExt c m vs[k]! gvs[k]! := fun h =>
      ((hv h).get (by
        have := hk
        simp only [Array.length_toList, Array.size_zipIdx] at this
        rw [← (hv h).1]
        exact this)).mono (l₁.trans le)
    refine (extOne.bind fun one m₃ l₃ => ((Sim.loop (I := fun j (acc : ExtVar × Gl) m =>
        IsExt c m acc.1 (Replay.prodRange j (num k)) ∧ acc.2 = Replay.prodRangeGl j (den k))
      (D := fun _ => True) fun j hj acc m' le' => ?_).pre fun h => ⟨h, h.2, rfl⟩).bind fun st m₄ l₄ =>
        (extScaleConstant _ fun h => h.2.1).bind fun weight m₅ l₅ =>
        (Sim.extMul (fun h => (hvk h.1.1.1.1.1).mono (l₃.trans (l₄.trans l₅))) fun h => h.2).bind fun t m₆ l₆ =>
        (extAdd (fun h => (h.1.1.1.1.2).mono (l₃.trans (l₄.trans (l₅.trans l₆)))) fun h => h.2).bind
          fun total m₇ l₇ => Sim.pure fun h => ⟨rfl, ?_⟩).post (fun _ _ _ h => h) fun _ => by simp
    · simp only [List.getElem_range] at *
      split
      · rename_i ne
        refine (((Sim.extConstant _).bind fun node m₅ l₅ =>
          (Sim.extSub (fun h => (hx h.1.1.1.1.1).mono (l₁.trans (le.trans (l₃.trans (le'.trans l₅)))))
            fun h => h.2).bind
            fun factor m₆ l₆ => (Sim.extMul (fun h => (h.1.1.2.1).mono (l₅.trans l₆)) fun h => h.2).bind
              fun num' m₇ l₇ => Sim.pure fun h => ⟨rfl, ?_, ?_⟩).post (fun _ _ _ h => h) fun _ => by simp)
        · rw [show Replay.prodRange (j + 1) (num k) = Replay.prodRange j (num k) * num k j from
            range_foldl_succ _ _ _]
          simp only [num, if_neg ne]
          exact h.2
        · rw [show Replay.prodRangeGl (j + 1) (den k) = Replay.prodRangeGl j (den k) * den k j from
            range_foldl_succ _ _ _, ← h.1.1.1.2.2]
          simp only [den, if_neg ne]
          rfl
      · rename_i eq
        refine Sim.pure fun h => ⟨rfl, ?_, ?_⟩
        · rw [show Replay.prodRange (j + 1) (num k) = Replay.prodRange j (num k) * num k j from
            range_foldl_succ _ _ _]
          simp only [num, if_pos (not_not.mp eq), ext_mul_one]
          exact h.2.1
        · rw [show Replay.prodRangeGl (j + 1) (den k) = Replay.prodRangeGl j (den k) * den k j from
            range_foldl_succ _ _ _]
          simp only [den, if_pos (not_not.mp eq), gl_mul_one]
          exact h.2.2
    · rw [show Replay.sumRange (k + 1) term = Replay.sumRange k term + term k from range_foldl_succ _ _ _]
      obtain ⟨⟨⟨⟨⟨⟨⟨hP', -⟩, -⟩, -⟩, -, hden⟩, -⟩, -⟩, htot⟩ := h
      refine htot.of_eq ?_
      simp only [term, hden, List.length_range, hP hP']
  · have size : vs.zipIdx.toList.length = gvs.size := by simp [(hv h.1.1).1]
    rw [size] at h
    exact h.2
end Sim

/-- Observe the coordinates of `e` in order. -/
def observe3 (r : Replay.Duplex) (e : Ext) : Replay.Duplex := ((r.observe e.c0).observe e.c1).observe e.c2

theorem foldl_observe_words (es : Array Ext) (r : Replay.Duplex) :
    (es.flatMap fun e => #[e.c0, e.c1, e.c2]).foldl Replay.Duplex.observe r = es.foldl observe3 r := by
  rw [← Array.foldl_toList, ← Array.foldl_toList, Array.toList_flatMap]
  generalize es.toList = l
  induction l generalizing r with
  | nil => rfl
  | cons e l ih => simp [List.flatMap_cons, observe3, ih]

namespace Sim
variable [CharP F goldilocksModulus] {c : Ctx F} {n : ℕ} {P : Prop}

/-- `for value in values do observe_ext value`. -/
theorem observeExtsFor {d : Duplex} {r : Replay.Duplex} (hp : Permutes c.perm) (hd : P → IsDuplex c n d r)
    {vs : Array ExtVar} {es : Array Ext} (hv : P → IsExts c n vs es) :
    Sim c n P (StateT.run (forIn (m := Transcript) vs.toList PUnit.unit fun value (_ : PUnit) =>
        Duplex.observeExt value >>= fun _ => Pure.pure (ForInStep.yield PUnit.unit)) d)
      (fun res m => IsDuplex c m res.2 (es.foldl observe3 r)) True := by
  have len : P → vs.toList.length = es.toList.length := fun h => by simp [(hv h).1]
  refine ((Sim.loopT (I := fun k _ e m => IsDuplex c m e ((es.toList.take k).foldl observe3 r)) (D := fun _ => True)
    fun k hk _ e m le => ?_).pre fun h => ⟨h, by simpa using hd h⟩).post (fun h _ _ q => ?_) fun _ => by simp
  · refine Sim.tbind₀ (observeExt hp (e := es[k]!) (fun h => h.2) fun h => ?_) fun _ e' m' l' => ?_
    · rw [Array.getElem_toList, ← getElem!_pos vs k (by simpa using hk)]
      exact ((hv h.1).get (by have := len h.1; simp at this hk; omega)).mono le
    · rw [StateT.run_pure]
      refine Sim.pure fun h => ⟨rfl, ?_⟩
      have hk' : k < es.toList.length := len h.1.1 ▸ hk
      rw [List.take_succ_eq_append_getElem hk', List.foldl_append, Array.getElem_toList,
        ← getElem!_pos es k (by simpa using hk')]
      exact h.2
  · rw [len h, List.take_length, Array.foldl_toList] at q
    exact q

end Sim

namespace Mirrors
variable [CharP F goldilocksModulus] {c : Ctx F} {n : ℕ} {P : Prop}

theorem observeExts (hp : Permutes c.perm) {vs : Array ExtVar} {es : Array Ext} (hv : P → IsExts c n vs es) :
    Mirrors c n P (forIn (m := Transcript) vs.toList PUnit.unit fun value (_ : PUnit) =>
        Duplex.observeExt value >>= fun _ => Pure.pure (ForInStep.yield PUnit.unit))
      (Replay.observeExts es) fun _ _ _ => True := fun _ tr =>
  ((Sim.observeExtsFor hp (fun h => h.2) fun h => hv h.1).accepts).post fun _ _ _ q =>
    ⟨(), _, Replay.runs_observeAll _ tr, by rw [foldl_observe_words]; exact q.1, trivial⟩

omit [CharP F goldilocksModulus] in
/-- A program whose precondition cannot hold mirrors any replay program. -/
theorem of_false {α β : Type} {t : Transcript α} {x : Replay.ReplayM β} {R : α → β → ℕ → Prop}
    (h : ∀ d, Accepts c n False (t.run d) fun _ _ => True) (no : P → False) : Mirrors c n P t x R := fun d _ =>
  ((h d).pre fun h' => (no h'.1).elim).post fun h' => (no h'.1).elim

omit [CharP F goldilocksModulus] in
/-- The loop's shape: it accepts with no claim when its steps do. -/
theorem forIn_false {body : Array ExtVar → Array ExtVar × ExtVar → Transcript (ForInStep (Array ExtVar × ExtVar))}
    (step : ∀ msg gmsg st (gst : Array Ext × Ext) m P', Mirrors c m (P' ∧ (IsExts c m msg gmsg ∧ 0 < gmsg.size) ∧
      IsExts c m st.1 gst.1 ∧ IsExt c m st.2 gst.2) (body msg st)
      (Replay.observeExts gmsg >>= fun _ => Replay.sampleExt >>= fun r =>
        Pure.pure (gst.1.push r, Replay.interpolate (Replay.roundValues gmsg gst.2) r))
      fun res g m' => res = .yield res.value ∧ IsExts c m' res.value.1 g.1 ∧ IsExt c m' res.value.2 g.2) :
    ∀ (l : List (Array ExtVar)) (st : Array ExtVar × ExtVar) (n : ℕ) (d : Duplex),
      Accepts c n False ((forIn l st body).run d) fun _ _ => True
  | [], st, n, d => by rw [List.forIn_nil, StateT.run_pure]; exact Accepts.pure fun _ => trivial
  | msg :: l, st, n, d => by
      rw [List.forIn_cons]
      refine ((step msg #[] st (#[], 0) n False d ⟨Replay.Duplex.new, #[]⟩).pre fun h => h.elim).tbind
        fun r e m _ => ?_
      cases r with
      | done b => exact Accepts.pure fun _ => trivial
      | yield b => exact (forIn_false step l b m e).pre fun h => h.1.elim

omit [CharP F goldilocksModulus] in
/-- A loop of sum-check rounds against the replay's recursion. -/
theorem forIn_rounds {body : Array ExtVar → Array ExtVar × ExtVar → Transcript (ForInStep (Array ExtVar × ExtVar))}
    (step : ∀ msg gmsg st (gst : Array Ext × Ext) m P', Mirrors c m (P' ∧ (IsExts c m msg gmsg ∧ 0 < gmsg.size) ∧
      IsExts c m st.1 gst.1 ∧ IsExt c m st.2 gst.2) (body msg st)
      (Replay.observeExts gmsg >>= fun _ => Replay.sampleExt >>= fun r =>
        Pure.pure (gst.1.push r, Replay.interpolate (Replay.roundValues gmsg gst.2) r))
      fun res g m' => res = .yield res.value ∧ IsExts c m' res.value.1 g.1 ∧ IsExt c m' res.value.2 g.2) :
    ∀ (l : List (Array ExtVar)) (gl : List (Array Ext)) (st : Array ExtVar × ExtVar) (gst : Array Ext × Ext) (n : ℕ)
      (P : Prop),
      (P → l.length = gl.length ∧
        (∀ i (h : i < l.length) (h' : i < gl.length), IsExts c n l[i] gl[i] ∧ 0 < gl[i].size) ∧
        IsExts c n st.1 gst.1 ∧ IsExt c n st.2 gst.2) →
      Mirrors c n P (forIn l st body) (Replay.sumcheckRounds gl gst.1 gst.2)
        fun res g m => IsExts c m res.1 g.1 ∧ IsExt c m res.2 g.2
  | [], _ :: _, st, gst, n, P, h => by
      rw [List.forIn_nil]
      exact of_false (c := c) (fun _ => by rw [StateT.run_pure]; exact Accepts.pure fun _ => trivial)
        fun hP => absurd (h hP).1 (by simp)
  | l, [], st, gst, n, P, h => by
      cases l with
      | nil => rw [List.forIn_nil]; exact Mirrors.pure fun hP => ⟨(h hP).2.2.1, (h hP).2.2.2⟩
      | cons msg l => exact of_false (c := c) (forIn_false step _ st n) fun hP => absurd (h hP).1 (by simp)
  | msg :: l, gm :: gl, st, gst, n, P, h => by
      rw [List.forIn_cons]
      have e : Replay.sumcheckRounds (gm :: gl) gst.1 gst.2 = ((Replay.observeExts gm >>= fun _ =>
          Replay.sampleExt >>= fun r =>
            Pure.pure (gst.1.push r, Replay.interpolate (Replay.roundValues gm gst.2) r)) >>=
            fun g => Replay.sumcheckRounds gl g.1 g.2) := by
        simp [Replay.sumcheckRounds, bind_assoc]
      rw [e]
      refine Mirrors.bind ((step msg gm st gst n P).pre fun hP =>
        ⟨hP, (h hP).2.1 0 (by simp) (by simp), (h hP).2.2.1, (h hP).2.2.2⟩) fun res g m le => ?_
      cases res with
      | done b => exact of_false (fun _ => Accepts.pure fun _ => trivial) fun hP => absurd hP.2.1 (by simp)
      | yield b =>
          refine forIn_rounds step l gl b g m _ fun hP => ⟨by simpa using (h hP.1).1, fun i hi hi' => ?_, hP.2.2.1,
            hP.2.2.2⟩
          exact ⟨((h hP.1).2.1 (i + 1) (by simpa using hi) (by simpa using hi')).1.mono le,
            ((h hP.1).2.1 (i + 1) (by simpa using hi) (by simpa using hi')).2⟩

/-- `sumcheck::replay`: per round the message, a draw and the next claim. -/
theorem sumcheck (hp : Permutes c.perm) {rounds : Array (Array ExtVar)} {grounds : Array (Array Ext)}
    {claim : ExtVar} {gclaim : Ext}
    (hr : P → rounds.size = grounds.size ∧ ∀ i (h : i < rounds.size) (h' : i < grounds.size),
      IsExts c n rounds[i] grounds[i] ∧ 0 < grounds[i].size)
    (hc : P → IsExt c n claim gclaim) :
    Mirrors c n P (sumcheckReplay rounds claim) (Replay.sumcheckReplay grounds gclaim)
      fun res g m => IsExts c m res.1 g.1 ∧ IsExt c m res.2 g.2 := by
  unfold sumcheckReplay Replay.sumcheckReplay
  simp only [← Array.forIn_toList]
  rw [show Replay.sumcheckRounds grounds.toList #[] gclaim =
    Replay.sumcheckRounds grounds.toList #[] gclaim >>= Pure.pure by rw [bind_pure]]
  refine Mirrors.bind (forIn_rounds (fun msg gmsg st gst m P' => ?_) rounds.toList grounds.toList (#[], claim)
    (#[], gclaim) n P fun hP => ⟨by simp [(hr hP).1], fun i hi hi' => ?_, IsExts.empty, hc hP⟩) fun res g m _ =>
      Mirrors.pure fun h => h.2
  · have m0 : P' ∧ (IsExts c m msg gmsg ∧ 0 < gmsg.size) ∧ IsExts c m st.1 gst.1 ∧ IsExt c m st.2 gst.2 →
        IsExt c m msg[0]! gmsg[0]! := fun h => h.2.1.1.get h.2.1.2
    refine Mirrors.left (Sim.extSub (fun h => h.2.2.2) m0).accepts fun atOne m₁ l₁ => ?_
    refine Mirrors.bind (observeExts hp fun h => h.1.2.1.1.mono l₁) fun _ _ m₂ l₂ => ?_
    refine Mirrors.bind (sampleExt hp) fun r gr m₃ l₃ => ?_
    refine Mirrors.left (Sim.interpolate (gvs := Replay.roundValues gmsg gst.2) (fun h => ?_) fun h => h.2).accepts
      fun claim m₄ l₄ => Mirrors.pure fun h => ⟨rfl, (h.1.1.1.1.2.2.1.mono (l₁.trans (l₂.trans (l₃.trans l₄)))).push
        (h.1.2.mono l₄), h.2.1⟩
    have l₂₃ := l₂.trans l₃
    refine ((IsExts.empty.push ((m0 h.1.1.1).mono (l₁.trans l₂₃))).push (h.1.1.2.1.mono l₂₃)).append
      ((h.1.1.1.2.1.1.mono (l₁.trans l₂₃)).extractTail 1)
  · simp only [Array.getElem_toList]
    exact (hr hP).2 i (by simpa using hi) (by simpa using hi')

end Mirrors
end CompressionLean.Circuit
