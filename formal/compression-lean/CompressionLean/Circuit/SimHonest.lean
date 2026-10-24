import CompressionLean.Circuit.SimSumcheck
import CompressionLean.Circuit.Honest
import CompressionLean.Bridge.Run
import CompressionLean.Bridge.Setup

/-!
Owns the simulation of the honest fold of `Honest.lean` (`verifier.rs:208-249`)
against `Replay.honestStage`, and of the setup query point (`queryPoint`,
`setup.rs:203-213`) against `Replay.leafQueryPoint`.

- `honestStage_ensures`: once the circuit's checks hold, the replay's honest
  stage runs, and `IsHonestOut` gives the outputs that the openings read.
- `setupCheck_ensures`: the leaf check (`setup.rs:243-300`). The circuit's
  root assertions are the replay's `pathOk`, and its folded values are the
  replay's.
- The circuit draws the query bits with `sampleBits`, the replay draws words
  (`sampleWords`); once each 64-bit check holds, bit `i` of a query is digit
  `i` of its word (`sampleBitsList`).
-/

namespace CompressionLean.Circuit

open CompressionLean.Lib
open CompressionLean.Replay (Gl Ext Relation foldVars runOracles generator twoAdicGenerator Runs runs_bind runs_pure
  runs_sampleWord runs_record runs_sampleExt runs_observeAll runs_observeExts runs_ensure runs_liftExcept)
open CompressionLean.Bridge (two_groups)
open NightstreamFPrime.Spec (goldilocksModulus)

variable {F : Type} [CommRing F]

/-! ## Arrays -/

/-- A fold over one more element of a prefix. -/
private theorem foldl_take_succ {α β : Type} (f : β → α → β) (b : β) (l : List α) {k : ℕ}
    (hk : k < l.length) : (l.take (k + 1)).foldl f b = f ((l.take k).foldl f b) l[k] := by
  rw [List.take_add_one, List.getElem?_eq_getElem hk, Option.toList_some, List.foldl_append, List.foldl_cons,
    List.foldl_nil]

namespace IsExts

variable {c : Ctx F} {n : ℕ} {xs ys : Array ExtVar} {gs hs : Array Ext}

/-- `k` copies of one value. -/
theorem replicate {z : ExtVar} {g : Ext} (hz : IsExt c n z g) (k : ℕ) :
    IsExts c n (Array.replicate k z) (Array.replicate k g) := by
  refine ⟨by simp, fun i hi => ?_⟩
  rw [Array.size_replicate] at hi
  rw [getElem!_pos _ _ (by simpa using hi), getElem!_pos _ _ (by simpa using hi), Array.getElem_replicate,
    Array.getElem_replicate]
  exact hz

/-- The empty prefix of a table. -/
theorem range_zero (f : ℕ → Ext) : IsExts c n #[] ((Array.range 0).map f) :=
  ⟨by simp, fun _ h => absurd h (by simp)⟩

/-- Entry `i` of a table `f`. -/
theorem get_map {k : ℕ} {f : ℕ → Ext} (h : IsExts c n xs ((Array.range k).map f)) {i : ℕ} (hi : i < k) :
    IsExt c n xs[i]! (f i) := by
  have := h.get (i := i) (by simpa using hi)
  rwa [getElem!_pos ((Array.range k).map f) i (by simpa using hi), Array.getElem_map, Array.getElem_range] at this

/-- A table of equal values. -/
theorem congr_map {k : ℕ} {f g : ℕ → Ext} (h : IsExts c n xs ((Array.range k).map f)) (e : ∀ v < k, f v = g v) :
    IsExts c n xs ((Array.range k).map g) := by
  rwa [show (Array.range k).map f = (Array.range k).map g from
    Array.ext (by simp) fun i h₁ _ => by simp at h₁; simp [e i h₁]] at h

/-- `k` copies of one value as a table. -/
theorem replicate_map {z : ExtVar} {g : Ext} (hz : IsExt c n z g) {k : ℕ} {f : ℕ → Ext} (e : ∀ v < k, f v = g) :
    IsExts c n (Array.replicate k z) ((Array.range k).map f) := by
  refine ⟨by simp, fun i hi => ?_⟩
  rw [Array.size_replicate] at hi
  rw [getElem!_pos _ _ (by simpa using hi), Array.getElem_replicate, getElem!_map_range k f hi, e i hi]
  exact hz

/-- Entry `g` of nested arrays, empty past the end. -/
theorem nested {xss : Array (Array ExtVar)} {gss : Array (Array Ext)}
    (h : xss.size = gss.size ∧ ∀ g < xss.size, IsExts c n xss[g]! gss[g]!) (g : ℕ) : IsExts c n xss[g]! gss[g]! := by
  by_cases lt : g < xss.size
  · exact h.2 g lt
  · rw [getElem!_neg xss g lt, getElem!_neg gss g (by rw [← h.1]; exact lt)]
    exact IsExts.empty

/-- The last entry. -/
theorem back (h : IsExts c n xs gs) (ne : 0 < gs.size) : IsExt c n xs.back! gs.back! := by
  have e := h.1
  rw [Array.back!, Array.back!, e]
  exact h.get (by omega)

/-- Three values. -/
theorem triple {a b d : ExtVar} {ga gb gd : Ext} (ha : IsExt c n a ga) (hb : IsExt c n b gb) (hd : IsExt c n d gd) :
    IsExts c n #[a, b, d] #[ga, gb, gd] :=
  ⟨rfl, fun i hi => match i, hi with
    | 0, _ => ha
    | 1, _ => hb
    | 2, _ => hd⟩

/-- A slice. -/
theorem slice (h : IsExts c n xs gs) (a b : ℕ) : IsExts c n (xs.extract a b) (gs.extract a b) := by
  refine ⟨by simp [h.1], fun i hi => ?_⟩
  rw [Array.size_extract] at hi
  rw [getElem!_pos _ _ (by simp; omega), getElem!_pos _ _ (by simp [← h.1]; omega), Array.getElem_extract,
    Array.getElem_extract, ← getElem!_pos, ← getElem!_pos]
  exact h.2 _ (by omega)

end IsExts

/-! ## Replay facts -/

/-- `a^(e+1) = a^e·a` for the replay's power by squaring. -/
theorem ext_pow_succ (a : Ext) (e : ℕ) : a.pow (e + 1) = a.pow e * a := by
  induction e using Nat.strong_induction_on generalizing a with
  | _ e ih =>
    rw [Ext.pow.eq_def a (e + 1), if_neg (by omega)]
    rcases Nat.even_or_odd' e with ⟨k, rfl | rfl⟩
    · rw [if_pos (by omega), show (2 * k + 1) / 2 = k by omega, Ext.pow.eq_def a (2 * k)]
      rcases k with _ | k
      · rw [if_pos rfl, Ext.pow.eq_def, if_pos rfl]
      · rw [if_neg (by omega), if_neg (by omega), show 2 * (k + 1) / 2 = k + 1 by omega]
    · rw [if_neg (by omega), show (2 * k + 1 + 1) / 2 = k + 1 by omega, ih k (by omega),
        Ext.pow.eq_def a (2 * k + 1), if_neg (by omega), if_pos (by omega), show (2 * k + 1) / 2 = k by omega,
        ext_mul_assoc]

/-- Sums of functions that agree below `k`. -/
theorem sumRange_congr {k : ℕ} {f g : ℕ → Ext} (h : ∀ t < k, f t = g t) :
    Replay.sumRange k f = Replay.sumRange k g := by
  induction k with
  | zero => simp [Replay.sumRange]
  | succ k ih => rw [sumRange_succ, sumRange_succ, ih fun t ht => h t (by omega), h k (by omega)]

/-- The replay's `observeExts` absorbs each value's coordinates in order. -/
theorem foldl_coords (vs : Array Ext) (r : Replay.Duplex) :
    (vs.flatMap fun v => #[v.c0, v.c1, v.c2]).foldl Replay.Duplex.observe r =
      vs.foldl (fun r v => ((r.observe v.c0).observe v.c1).observe v.c2) r := by
  rw [← Array.foldl_toList, Array.toList_flatMap, List.foldl_flatMap, ← Array.foldl_toList]
  rfl

/-- `α^[u]` only reads the low `|α|` bits of `u`. -/
theorem monomial_add (l : List Ext) (u k : ℕ) : Replay.monomial l (u + 2 ^ l.length * k) = Replay.monomial l u := by
  induction l generalizing u with
  | nil => rfl
  | cons a l ih =>
      rw [List.length_cons, pow_succ, mul_comm (2 ^ l.length) 2, mul_assoc]
      simp only [Replay.monomial]
      rw [show (u + 2 * (2 ^ l.length * k)) / 2 = u / 2 + 2 ^ l.length * k by omega, ih,
        show (u + 2 * (2 ^ l.length * k)) % 2 = u % 2 by omega]

/-- One more variable of `α` multiplies by its bit. -/
theorem monomial_append (l : List Ext) (x : Ext) (u : ℕ) :
    Replay.monomial (l ++ [x]) u = Replay.monomial l u * (if u / 2 ^ l.length % 2 = 1 then x else 1) := by
  induction l generalizing u with
  | nil =>
      simp only [List.nil_append, Replay.monomial, List.length_nil, pow_zero, Nat.div_one]
      rw [ext_mul_one, ext_one_mul]
  | cons a l ih =>
      have e : u / 2 ^ (l.length + 1) = u / 2 / 2 ^ l.length := by rw [pow_succ', Nat.div_div_eq_div_mul]
      simp only [List.cons_append, Replay.monomial, List.length_cons, ih, e]
      rw [ext_mul_assoc]

/-- The monomials of `α` and one more variable. -/
theorem monomials_succ (l : List Ext) (x : Ext) :
    (Array.range (2 ^ l.length)).map (Replay.monomial l) ++
        (Array.range (2 ^ l.length)).map (fun u => Replay.monomial l u * x) =
      (Array.range (2 ^ (l.length + 1))).map (Replay.monomial (l ++ [x])) := by
  refine Array.ext (by simp [pow_succ]; omega) fun i h₁ h₂ => ?_
  rw [Array.getElem_map, Array.getElem_range, monomial_append]
  by_cases lt : i < 2 ^ l.length
  · rw [Array.getElem_append_left (by simpa using lt), Array.getElem_map, Array.getElem_range,
      Nat.div_eq_of_lt lt, if_neg (by omega), ext_mul_one]
  · have hi : i < 2 ^ l.length + 2 ^ l.length := by simpa using h₁
    rw [Array.getElem_append_right (by simpa using lt), Array.getElem_map, Array.getElem_range]
    simp only [Array.size_map, Array.size_range]
    rw [show i / 2 ^ l.length = 1 from Nat.div_eq_of_lt_le (by omega) (by omega), if_pos rfl]
    conv_rhs => rw [show i = i - 2 ^ l.length + 2 ^ l.length * 1 by omega, monomial_add]

/-- `x + w·0 = x`: an absent leaf value adds nothing. -/
theorem add_scale_zero (x w : Ext) : x + w.scale default = x :=
  coords_injective (F := ZMod goldilocksModulus) (by
    rw [coords_add, coords_scale]
    funext k
    simp [show (default : Gl) = 0 from rfl])

/-- Digit `i < b` of `x mod 2^b` is digit `i` of `x`. -/
theorem digit_mod (x b i : ℕ) (hi : i < b) : x % 2 ^ b / 2 ^ i % 2 = x / 2 ^ i % 2 := by
  have e := Nat.testBit_mod_two_pow x b i
  rw [decide_eq_true hi, Bool.true_and, Nat.testBit_eq_decide_div_mod_eq, Nat.testBit_eq_decide_div_mod_eq,
    decide_eq_decide] at e
  omega

/-- The circuit's lane checks (`setup.rs:255-257`) are the replay's `pathOk`. -/
theorem pathOk_of {root : Array Gl} {index : ℕ} {leaf : Replay.Leaf}
    (h : ∀ k < 4, (Replay.merkleRoot (Replay.hashLeaf leaf.values) index leaf.path)[k]! = root[k]!) :
    Replay.pathOk root index leaf = true := by
  unfold Replay.pathOk
  rw [List.all_eq_true]
  intro lane hl
  rw [h lane (List.mem_range.mp hl)]
  exact beq_self_eq_true (root[lane]!).word

/-- `setup::check` on a leaf whose path holds: one folded value per group. -/
theorem setupCheck_ok {root : Array Gl} {vars index : ℕ} {leaf : Replay.Leaf} {a b alpha : Array Ext}
    (path : Replay.pathOk root index leaf = true) :
    Replay.setupCheck root vars index leaf #[a, b] alpha = .ok
      #[Replay.sumRange (2 ^ foldVars) fun v => Replay.leafWeight alpha
          (Replay.powerFromBits (twoAdicGenerator (vars + 1)).inv index (vars + 1 - foldVars) * generator.inv)
          (twoAdicGenerator foldVars) v *
            Replay.sumRange a.size fun k => a[k]!.scale leaf.values[(0 + k) * 2 ^ foldVars + v]!,
        Replay.sumRange (2 ^ foldVars) fun v => Replay.leafWeight alpha
          (Replay.powerFromBits (twoAdicGenerator (vars + 1)).inv index (vars + 1 - foldVars) * generator.inv)
          (twoAdicGenerator foldVars) v *
            Replay.sumRange b.size fun k => b[k]!.scale leaf.values[(a.size + k) * 2 ^ foldVars + v]!] := by
  unfold Replay.setupCheck
  rw [path]
  dsimp only
  rw [two_groups]
  rfl

namespace Sim

variable [CharP F goldilocksModulus] {c : Ctx F} {n : ℕ} {P : Prop}

/-! ## Points -/

omit [CharP F goldilocksModulus] in
/-- `setup_point` (`lib.rs:730-734`): a point of at most `vars` coordinates,
padded with zeros. -/
theorem setupPoint {point : Array ExtVar} {gpoint : Array Ext} {vars : ℕ} (hp : P → IsExts c n point gpoint)
    (size : gpoint.size ≤ vars) :
    Sim c n P (Circuit.setupPoint point vars) (fun r m => IsExts c m r (Replay.padded vars gpoint)) True := by
  unfold Circuit.setupPoint
  refine extZero.bind₀ fun zero m l => Sim.pure fun h => ?_
  have hpoint := (hp h.1).mono l
  rw [Array.extract_eq_self_of_le (by rw [hpoint.1]; exact size), hpoint.1]
  exact hpoint.append (IsExts.replicate h.2 _)

/-- `combine` (`verifier.rs:334-337`). -/
theorem combinePartials {partials point : Array ExtVar} {gpartials gpoint : Array Ext}
    (hpa : P → IsExts c n partials gpartials) (hpo : P → IsExts c n point gpoint) :
    Sim c n P (Circuit.combinePartials partials point) (fun r m => IsExt c m r (Replay.combine gpartials gpoint))
      True :=
  (Sim.eqTable hpo).bind₀ fun _ _ l => Sim.dot (fun h => h.2) (fun h => (hpa h.1).mono l)

/-- `setup::query_point` (`setup.rs:203-213`): the power point of the leaf
whose index has the little-endian bits `bits`. -/
theorem queryPoint {vars index : ℕ} {bits : Array Var} (hsize : P → bits.size = vars + 1 - foldVars)
    (hb : P → ∀ i < bits.size, IsWord c n bits[i]! (Gl.ofNat (index / 2 ^ i % 2))) :
    Sim c n P (Circuit.queryPoint vars bits) (fun r m => IsExts c m r (Replay.leafQueryPoint vars index)) True := by
  unfold Circuit.queryPoint Replay.leafQueryPoint
  dsimp only
  generalize hq : Replay.powerFromBits (twoAdicGenerator (vars + 1 - foldVars)) index (vars + 1 - foldVars) *
    generator.pow (2 ^ foldVars) = q
  refine (Sim.powerFromBits _ index hb).bind₀ fun rotation m₁ l₁ => ?_
  refine (Sim.scale (generator.pow (2 ^ foldVars)) fun h => h.2).bind₀ fun start m₂ l₂ => ?_
  rw [Std.Legacy.Range.forIn_eq_forIn_range']
  refine (((Sim.loop (P := (P ∧ IsWord c m₁ rotation (Replay.powerFromBits (twoAdicGenerator (vars + 1 - foldVars))
      index bits.size)) ∧ IsWord c m₂ start (Replay.powerFromBits (twoAdicGenerator (vars + 1 - foldVars)) index
      bits.size * generator.pow (2 ^ foldVars)))
      (I := fun k (acc : Var × Array ExtVar) m => IsWord c m acc.1 (q.pow (2 ^ k)) ∧
        IsExts c m acc.2 ((Array.range k).map fun k => Ext.ofGl (q.pow (2 ^ k))))
      (D := fun _ => True) fun k hk acc m lm => ?_).pre fun h => ⟨h, ?_, IsExts.range_zero _⟩).bind
      fun res m₃ l₃ => Sim.pure fun h => ?_).post (fun _ _ _ h => h) fun _ => by simp
  · refine (Sim.liftWord fun h => h.2.1).bind₀ fun lifted m₄ l₄ => ?_
    refine (Sim.mul (fun h => h.1.2.1.mono l₄) fun h => h.1.2.1.mono l₄).bind₀ fun next m₅ l₅ =>
      Sim.pure fun h => ⟨rfl, ?_, ?_⟩
    · have e : q.pow (2 ^ k) * q.pow (2 ^ k) = q.pow (2 ^ (k + 1)) := Gl.eq_of_toF (by
        rw [Gl.toF_mul, Gl.toF_pow, Gl.toF_pow, ← pow_add, ← two_mul, ← pow_succ'])
      exact e ▸ h.2
    · rw [map_range_succ]
      exact (h.1.1.2.2.mono (l₄.trans l₅)).push (h.1.2.mono l₅)
  · rw [hsize h.1.1, hq] at h
    have e : q = q.pow (2 ^ 0) := Gl.eq_of_toF (by rw [Gl.toF_pow, pow_zero, pow_one])
    exact e ▸ h.2
  · have e := h.2.2
    simp only [List.length_range', Std.Legacy.Range.size] at e
    rw [show (vars - foldVars - 0 + 1 - 1) / 1 = vars - foldVars by omega] at e
    rw [show ((List.range (vars - foldVars)).map fun k => Ext.ofGl (q.pow (2 ^ k))).toArray =
      (Array.range (vars - foldVars)).map fun k => Ext.ofGl (q.pow (2 ^ k)) from Array.ext' (by simp)]
    exact e

/-! ## The duplex -/

/-- `observe_root` (`whir/mod.rs:230-233`): the separator words as constants,
then the root. -/
theorem observeRoot {d : Duplex} {r : Replay.Duplex} {seed : Array Gl} {root : DigestVar} {groot : Array Gl}
    (hp : Permutes c.perm) (hd : P → IsDuplex c n d r) (hr : P → IsDigest c n root groot) :
    Sim c n P ((Circuit.observeRoot seed root).run d)
      (fun res m => IsDuplex c m res.2 (groot.foldl Replay.Duplex.observe (seed.foldl Replay.Duplex.observe r)))
      True := by
  unfold Circuit.observeRoot
  rw [← Array.forIn_toList]
  refine (Sim.tbind (C' := True) ((Sim.loopT (P := P) (d := d) (I := fun k _ e m => IsDuplex c m e
      ((seed.toList.take k).foldl Replay.Duplex.observe r)) (D := fun _ => True) fun k hk _ e m lm => ?_).pre
      fun h => by exact ⟨h, hd h⟩) fun _ e m le => ?_).post (fun _ _ _ h => h) fun _ => by simp
  · refine Sim.tbind₀ (Sim.lift (Sim.const seed.toList[k])) fun w e₁ m₁ l₁ => ?_
    refine Sim.tbind₀ (Sim.observe hp (fun h => by rw [show e₁ = e from h.2.2]; exact h.1.2.mono l₁)
      fun h => h.2.1) fun _ e₂ m₂ l₂ => ?_
    rw [StateT.run_pure]
    refine Sim.pure fun h => ⟨rfl, ?_⟩
    rw [foldl_take_succ _ _ _ hk]
    exact h.2
  · refine Sim.observeSlice hp (fun h => ?_) fun h => IsWords.ofFn (hr h.1).1 fun lane => ((hr h.1).2 lane).mono le
    have e := h.2
    rw [List.take_of_length_le (by simp), Array.foldl_toList] at e
    exact e

/-- The binding of the partials (`verifier.rs:209-213`): the coordinates of
each value of each group, in order. -/
theorem observePartials {d : Duplex} {r : Replay.Duplex} {partials : Array (Array ExtVar)}
    {gpartials : Array (Array Ext)} (hp : Permutes c.perm) (hd : P → IsDuplex c n d r)
    (hpa : P → partials.size = gpartials.size ∧ ∀ g < partials.size, IsExts c n partials[g]! gpartials[g]!) :
    Sim c n P ((forIn partials PUnit.unit fun values _ => do
        forIn values PUnit.unit fun value _ => do
          Duplex.observeExt value
          Pure.pure (ForInStep.yield PUnit.unit)
        Pure.pure (ForInStep.yield PUnit.unit) : Transcript PUnit).run d)
      (fun res m => IsDuplex c m res.2 ((gpartials.flatten.flatMap fun v => #[v.c0, v.c1, v.c2]).foldl
        Replay.Duplex.observe r)) True := by
  rw [foldl_coords, Array.foldl_flatten, ← Array.foldl_toList, ← Array.forIn_toList]
  let step := fun (r : Replay.Duplex) (v : Ext) => ((r.observe v.c0).observe v.c1).observe v.c2
  refine ((Sim.loopT (P := P) (d := d) (I := fun k _ e m => IsDuplex c m e ((gpartials.toList.take k).foldl
      (fun r vs => vs.foldl step r) r)) (D := fun _ => True) fun k hk _ e m lm => ?_).pre fun h => by
        exact ⟨h, hd h⟩).post
      (fun h _ _ q => ?_) fun _ => by simp
  · rw [show partials.toList[k] = partials[k]! by simp [getElem!_pos partials k (by simpa using hk)],
      ← Array.forIn_toList]
    have hv : P → IsExts c m partials[k]! gpartials[k]! := fun h => ((hpa h).2 k (by simpa using hk)).mono lm
    refine (Sim.tbind (C' := True) ((Sim.loopT (P := P ∧ _) (d := e) (I := fun j _ e' m' => IsDuplex c m' e'
        (((gpartials[k]!).toList.take j).foldl step ((gpartials.toList.take k).foldl (fun r vs => vs.foldl step r) r)))
        (D := fun _ => True) fun j hj _ e' m' lm' => ?_).pre fun h => by exact ⟨h, h.2⟩) fun _ e' m' l' => ?_).post
        (fun _ _ _ h => h) fun _ => by simp
    · rw [show (partials[k]!).toList[j] = (partials[k]!)[j]! by
        simp [getElem!_pos (partials[k]!) j (by simpa using hj)]]
      refine Sim.tbind₀ (Sim.observeExt hp (fun h => h.2) fun h => ((hv h.1.1).2 j (by simpa using hj)).mono lm')
        fun _ e₂ m₂ l₂ => ?_
      rw [StateT.run_pure]
      refine Sim.pure fun h => ⟨rfl, ?_⟩
      have hj' : j < (gpartials[k]!).toList.length := by
        have := (hv h.1.1.1).1
        simp at hj ⊢
        omega
      rw [foldl_take_succ _ _ _ hj', Array.getElem_toList, ← getElem!_pos]
      exact h.2
    · rw [StateT.run_pure]
      refine Sim.pure fun h => ⟨rfl, ?_⟩
      have e := h.2
      have hk' : k < gpartials.toList.length := by
        have := (hpa h.1.1).1
        simp at hk ⊢
        omega
      rw [List.take_of_length_le (l := (gpartials[k]!).toList) (by simp [(hv h.1.1).1])] at e
      rw [foldl_take_succ _ _ _ hk', Array.getElem_toList, ← getElem!_pos, ← Array.foldl_toList]
      exact e
  · rw [List.take_of_length_le (by simp [(hpa h).1])] at q
    exact q

omit [CharP F goldilocksModulus] in
/-- `α`: three extension samples (`verifier.rs:214`). -/
theorem mapM_sampleExt {β : Type} (g : Array ExtVar → Transcript β) :
    ((Array.range foldVars).mapM (fun _ => Duplex.sampleExt) >>= g) =
      (Duplex.sampleExt >>= fun a => Duplex.sampleExt >>= fun b => Duplex.sampleExt >>= fun e => g #[a, b, e]) := by
  simp [Array.mapM_eq_mapM_toList, foldVars, List.range_succ]

/-- `sample_bits` for each query (`verifier.rs:235-237`) and the replay's word
draws from the same duplex. Once each decomposition's check holds, bit `i` of
query `q` is digit `i` of word `q`. -/
theorem sampleBitsList (bits : ℕ) (small : bits ≤ 64) (hp : Permutes c.perm) :
    ∀ (l : List ℕ) {n : ℕ} {P : Prop} {d : Duplex} (t : Replay.Transcript) (init : Array Gl),
      (P → IsDuplex c n d t.duplex) →
      Ensures c n P ((l.mapM fun _ => Duplex.sampleBits bits : Transcript (List (Array Var))).run d)
        (fun res m => ∃ ws : List Gl, ∃ t' : Replay.Transcript,
          Runs (l.foldlM (fun values _ => do Pure.pure (values.push (← Replay.sampleWord))) init) t
            (init ++ ws.toArray) t' ∧
          ws.length = l.length ∧ res.1.length = l.length ∧
          (∀ q < l.length, (res.1[q]!).size = bits ∧
            ∀ i < bits, IsWord c m (res.1[q]!)[i]! (Gl.ofNat ((ws[q]!).toNat / 2 ^ i % 2))) ∧
          IsDuplex c m res.2 t'.duplex)
  | [], n, P, d, t, init, hd => by
      rw [List.mapM_nil, StateT.run_pure]
      exact Ensures.pure fun h => ⟨[], t, by simpa using runs_pure init t, rfl, rfl,
        fun q hq => absurd hq (by simp), hd h⟩
  | x :: l, n, P, d, t, init, hd => by
      rw [List.mapM_cons]
      refine Ensures.tbind (Sim.sampleBits bits small hp hd) fun a e m le => ?_
      refine Ensures.tbind (sampleBitsList bits small hp l { t with duplex := t.duplex.sample.2 }
        (init.push t.duplex.sample.1) fun h => h.2.2.2) fun as e' m' le' => ?_
      rw [StateT.run_pure]
      refine Ensures.pure fun h => ?_
      obtain ⟨⟨-, size, digits, -⟩, ws, t', runs, wlen, alen, adigits, hd'⟩ := h
      refine ⟨t.duplex.sample.1 :: ws, t', ?_, by simp [wlen], by simp [alen], fun q hq => ?_, hd'⟩
      · rw [List.foldlM_cons]
        refine runs_bind (runs_bind (runs_sampleWord t) (runs_pure _ _)) ?_
        have e : init.push t.duplex.sample.1 ++ ws.toArray = init ++ (t.duplex.sample.1 :: ws).toArray :=
          Array.ext' (by simp)
        exact e ▸ runs
      · cases q with
        | zero => exact ⟨by simpa using size, fun i hi => by simpa using (digits i hi).mono le'⟩
        | succ q =>
            simp only [List.getElem!_cons_succ]
            exact adigits q (by simpa using hq)

/-- `sample_bits` for every setup query (`verifier.rs:235-237`) against the
replay's `sampleWords`. -/
theorem sampleBitsArray (bits : ℕ) (small : bits ≤ 64) (hp : Permutes c.perm) (k : ℕ) {d : Duplex}
    {t : Replay.Transcript} (hd : P → IsDuplex c n d t.duplex) :
    Ensures c n P (((Array.range k).mapM fun _ => Duplex.sampleBits bits : Transcript (Array (Array Var))).run d)
      (fun res m => ∃ ws : Array Gl, ∃ t' : Replay.Transcript, Runs (Replay.sampleWords k) t ws t' ∧
        res.1.size = ws.size ∧ (∀ q < res.1.size, (res.1[q]!).size = bits ∧
          ∀ i < bits, IsWord c m (res.1[q]!)[i]! (Gl.ofNat ((ws[q]!).toNat / 2 ^ i % 2))) ∧
        IsDuplex c m res.2 t'.duplex) := by
  rw [Array.mapM_eq_mapM_toList, StateT.run_map, Array.toList_range]
  refine (sampleBitsList bits small hp (List.range k) t #[] hd).map fun res m h => ?_
  obtain ⟨ws, t', runs, wlen, rlen, digits, hd'⟩ := h
  refine ⟨ws.toArray, { t' with draws := t'.draws.push (.words ws.toArray) }, ?_, by simp [wlen, rlen],
    fun q hq => ?_, hd'⟩
  · unfold Replay.sampleWords
    rw [Array.empty_append] at runs
    exact runs_bind runs (runs_bind (runs_record _ _) (runs_pure _ _))
  · have lt : q < (List.range k).length := by simpa [rlen] using hq
    have := digits q lt
    simpa using this

/-- The run weights `μ^i` for the run oracles (`verifier.rs:239-243`). -/
theorem runWeightsLoop {d : Duplex} {one mu : ExtVar} {gmu : Ext} (hone : P → IsExt c n one 1)
    (hmu : P → IsExt c n mu gmu) :
    Sim c n P ((forIn [1:runOracles] #[one] fun _ (weights : Array ExtVar) => do
        let next ← liftM (Build.extMul weights.back! mu)
        Pure.pure (ForInStep.yield (weights.push next)) : Transcript (Array ExtVar)).run d)
      (fun res m => IsExts c m res.1 ((Array.range runOracles).map gmu.pow) ∧ res.2 = d) True := by
  rw [Std.Legacy.Range.forIn_eq_forIn_range']
  refine ((Sim.loopT (P := P) (d := d) (I := fun k (acc : Array ExtVar) e m =>
      IsExts c m acc ((Array.range (k + 1)).map gmu.pow) ∧ e = d) (D := fun _ => True)
      fun k hk acc e m lm => ?_).pre fun h => ⟨h, ?_, rfl⟩).post (fun h _ _ q => ?_) fun _ => by simp
  · refine Sim.tbind₀ (Sim.lift (Sim.extMul (fun h => h.2.1.back (by simp)) fun h => (hmu h.1).mono lm))
      fun next e₁ m₁ l₁ => ?_
    rw [StateT.run_pure]
    refine Sim.pure fun h => ⟨rfl, ?_, h.2.2.trans h.1.2.2⟩
    have e := h.2.1
    rw [show ((Array.range (k + 1)).map gmu.pow).back! = gmu.pow k by
      rw [Array.back!, getElem!_pos _ _ (by simp)]; simp, ← ext_pow_succ] at e
    rw [map_range_succ]
    exact (h.1.2.1.mono l₁).push e
  · have p0 : gmu.pow 0 = 1 := by rw [Ext.pow.eq_def, if_pos rfl]
    rw [show Array.range (0 + 1) = #[0] by decide, Array.map_singleton, p0]
    exact ⟨rfl, fun i hi => match i, hi with | 0, _ => hone h⟩
  · simpa [runOracles, Std.Legacy.Range.size] using q

omit [CharP F goldilocksModulus] in
/-- The key root words as constants (`verifier.rs:245`). -/
theorem rootConsts {d : Duplex} (words : Array Gl) :
    Sim c n P ((forIn words (#[] : Array Var) fun word (root : Array Var) => do
        let w ← liftM (Build.const word)
        Pure.pure (ForInStep.yield (root.push w)) : Transcript (Array Var)).run d)
      (fun res m => IsWords c m res.1 words ∧ res.2 = d) True := by
  rw [← Array.forIn_toList]
  refine ((Sim.loopT (P := P) (d := d) (I := fun k (acc : Array Var) e m =>
      IsWords c m acc (words.toList.take k).toArray ∧ e = d) (D := fun _ => True)
      fun k hk acc e m lm => ?_).pre fun h => by exact ⟨h, IsWords.empty, rfl⟩).post (fun h _ _ q => ?_)
      fun _ => by simp
  · refine Sim.tbind₀ (Sim.lift (Sim.const words.toList[k])) fun w e₁ m₁ l₁ => ?_
    rw [StateT.run_pure]
    refine Sim.pure fun h => ⟨rfl, ?_, h.2.2.trans h.1.2.2⟩
    rw [List.take_add_one, List.getElem?_eq_getElem hk, Option.toList_some, ← List.push_toArray]
    exact (h.1.2.1.mono l₁).push h.2.1
  · rw [List.take_of_length_le (by simp), Array.toArray_toList] at q
    exact q

/-- The slot-block embedding `block·Π (1 - x)` over the run point's slot
tail (`verifier.rs:219-223`). -/
theorem embedLoop {d : Duplex} {xs : Array ExtVar} {gxs : Array Ext} {init : ExtVar} {ginit : Ext}
    (hx : P → IsExts c n xs gxs) (hi : P → IsExt c n init ginit) :
    Sim c n P ((forIn xs init fun x (embedded : ExtVar) => do
        let low ← liftM (Algebra.oneMinus x)
        let embedded ← liftM (Build.extMul embedded low)
        Pure.pure (ForInStep.yield embedded) : Transcript ExtVar).run d)
      (fun res m => IsExt c m res.1 (gxs.foldl (fun total x => total * Replay.oneMinus x) ginit) ∧ res.2 = d)
      True := by
  rw [← Array.forIn_toList]
  refine ((Sim.loopT (P := P) (d := d) (I := fun k (acc : ExtVar) e m =>
      IsExt c m acc ((gxs.toList.take k).foldl (fun total x => total * Replay.oneMinus x) ginit) ∧ e = d)
      (D := fun _ => True) fun k hk acc e m lm => ?_).pre fun h => by exact ⟨h, hi h, rfl⟩).post
      (fun h _ _ q => ?_) fun _ => by simp
  · have hk' : k < xs.size := by simpa using hk
    rw [show xs.toList[k] = xs[k]! by simp [getElem!_pos xs k hk']]
    refine Sim.tbind₀ (Sim.lift (Sim.oneMinus fun h => ((hx h.1).2 k hk').mono lm)) fun low e₁ m₁ l₁ => ?_
    refine Sim.tbind₀ (Sim.lift (Sim.extMul (fun h => h.1.2.1.mono l₁) fun h => h.2.1)) fun total e₂ m₂ l₂ => ?_
    rw [StateT.run_pure]
    refine Sim.pure fun h => ⟨rfl, ?_, h.2.2.trans (h.1.2.2.trans h.1.1.2.2)⟩
    have size := (hx h.1.1.1).1
    rw [foldl_take_succ _ _ _ (by simp; omega), Array.getElem_toList, ← getElem!_pos]
    exact h.2.1
  · rw [List.take_of_length_le (by simp [(hx h).1]), Array.foldl_toList] at q
    exact q

/-- The run value `c + μ·(σ + μ·(i + μ·embedded))` (`verifier.rs:224-228`). -/
theorem runValueLoop {d : Duplex} {values : Array ExtVar} {gvalues : Array Ext} {init mu : ExtVar}
    {ginit gmu : Ext} (hv : P → IsExts c n values gvalues) (hi : P → IsExt c n init ginit)
    (hmu : P → IsExt c n mu gmu) :
    Sim c n P ((forIn values init fun value (runValue : ExtVar) => do
        let scaled ← liftM (Build.extMul mu runValue)
        let runValue ← liftM (Build.extAdd value scaled)
        Pure.pure (ForInStep.yield runValue) : Transcript ExtVar).run d)
      (fun res m => IsExt c m res.1 (gvalues.foldl (fun total value => value + gmu * total) ginit) ∧ res.2 = d)
      True := by
  rw [← Array.forIn_toList]
  refine ((Sim.loopT (P := P) (d := d) (I := fun k (acc : ExtVar) e m =>
      IsExt c m acc ((gvalues.toList.take k).foldl (fun total value => value + gmu * total) ginit) ∧ e = d)
      (D := fun _ => True) fun k hk acc e m lm => ?_).pre fun h => by exact ⟨h, hi h, rfl⟩).post
      (fun h _ _ q => ?_) fun _ => by simp
  · have hk' : k < values.size := by simpa using hk
    rw [show values.toList[k] = values[k]! by simp [getElem!_pos values k hk']]
    refine Sim.tbind₀ (Sim.lift (Sim.extMul (fun h => (hmu h.1).mono lm) fun h => h.2.1)) fun scaled e₁ m₁ l₁ => ?_
    refine Sim.tbind₀ (Sim.lift (Sim.extAdd (fun h => ((hv h.1.1).2 k hk').mono (lm.trans l₁)) fun h => h.2.1))
      fun total e₂ m₂ l₂ => ?_
    rw [StateT.run_pure]
    refine Sim.pure fun h => ⟨rfl, ?_, h.2.2.trans (h.1.2.2.trans h.1.1.2.2)⟩
    have size := (hv h.1.1.1).1
    rw [foldl_take_succ _ _ _ (by simp; omega), Array.getElem_toList, ← getElem!_pos]
    exact h.2.1
  · rw [List.take_of_length_le (by simp [(hv h).1]), Array.foldl_toList] at q
    exact q

/-! ## The leaf check -/

/-- `merkle_root` (`circuit/hash.rs:115-129`) with the sizes known under `P`. -/
theorem merkleRoot' {leaf : DigestVar} {bits : Array Var} {path : Array DigestVar} {gleaf : Array Gl} {index : ℕ}
    {gpath : Array (Array Gl)} (hp : Permutes c.perm) (hleaf : P → IsDigest c n leaf gleaf)
    (hbits : P → ∀ i < bits.size, IsWord c n bits[i]! (Gl.ofNat (index / 2 ^ i % 2)))
    (hpath : P → ∀ k < path.size, IsDigest c n path[k]! gpath[k]!) (hsize : P → bits.size = path.size)
    (hgsize : P → path.size = gpath.size) :
    Sim c n P (Circuit.merkleRoot leaf bits path)
      (fun res m => IsDigest c m res (Replay.merkleRoot gleaf index gpath)) True := by
  obtain ⟨k, hk⟩ : ∃ k, k = min bits.size path.size := ⟨_, rfl⟩
  have zip : bits.zip path = (bits.extract 0 k).zip (path.extract 0 k) :=
    Array.ext (by simp [hk]) fun i h₁ h₂ => by simp
  have e : Circuit.merkleRoot leaf bits path = Circuit.merkleRoot leaf (bits.extract 0 k) (path.extract 0 k) := by
    unfold Circuit.merkleRoot
    rw [zip]
  rw [e]
  refine (Sim.merkleRoot (index := index) (gpath := (Array.range k).map fun i => gpath[i]!) hp hleaf (fun h i hi => ?_)
    (fun h i hi => ?_) (by simp [hk]) (by simp [hk])).post (fun h _ _ q => ?_) fun _ => Iff.rfl
  · have lt : i < k := by simp at hi; omega
    rw [getElem!_extract_zero bits lt]
    exact hbits h i (by simp at hi; omega)
  · have lt : i < k := by simp at hi; omega
    rw [getElem!_extract_zero path lt, getElem!_pos ((Array.range k).map fun i => gpath[i]!) i (by simpa using lt),
      Array.getElem_map, Array.getElem_range]
    exact hpath h i (by simp at hi; omega)
  · have size : k = gpath.size := by rw [hk, hsize h, Nat.min_self, hgsize h]
    have same : (Array.range k).map (fun i => gpath[i]!) = gpath :=
      Array.ext (by simp [size]) fun i h₁ h₂ => by
        rw [Array.getElem_map, Array.getElem_range, getElem!_pos gpath i h₂]
    rwa [same] at q

/-- The monomials `α^[u]` for `u < 2^|α|`, low bit first (`setup.rs:263-267`). -/
theorem monomials {alpha : Array ExtVar} {galpha : Array Ext} {one : ExtVar} (hone : P → IsExt c n one 1)
    (ha : P → IsExts c n alpha galpha) :
    Sim c n P (forIn alpha #[one] fun a (monomials : Array ExtVar) => do
        let scaled ← forIn monomials (#[] : Array ExtVar) fun m (scaled : Array ExtVar) => do
          let x ← Build.extMul m a
          Pure.pure (ForInStep.yield (scaled.push x))
        Pure.pure (ForInStep.yield (monomials ++ scaled)))
      (fun res m => IsExts c m res ((Array.range (2 ^ galpha.size)).map (Replay.monomial galpha.toList))) True := by
  rw [← Array.forIn_toList]
  refine ((Sim.loop (P := P) (I := fun j (acc : Array ExtVar) m =>
      IsExts c m acc ((Array.range (2 ^ j)).map (Replay.monomial (galpha.toList.take j))))
      (D := fun _ => True) fun j hj acc m lm => ?_).pre fun h => ⟨h, ?_⟩).post (fun h _ _ q => ?_) fun _ => by simp
  · have hj' : j < alpha.size := by simpa using hj
    rw [show alpha.toList[j] = alpha[j]! by simp [getElem!_pos alpha j hj'], ← Array.forIn_toList]
    refine (Sim.bind (C' := True) ((Sim.loop (P := P ∧ _) (I := fun i (sc : Array ExtVar) m' =>
        IsExts c m' sc ((Array.range i).map fun u => Replay.monomial (galpha.toList.take j) u * galpha[j]!))
        (D := fun _ => True) fun i hi sc m' lm' => ?_).pre fun h => by exact ⟨h, IsExts.range_zero _⟩)
        fun sc m' l' => Sim.pure fun h => ⟨rfl, ?_⟩).post (fun _ _ _ h => h) fun _ => by simp
    · have hi' : i < acc.size := by simpa using hi
      rw [show acc.toList[i] = acc[i]! by simp [getElem!_pos acc i hi']]
      refine (Sim.extMul (fun h => (h.1.2.get_map (by have := h.1.2.1; simp at this; omega)).mono lm')
        fun h => ((ha h.1.1).get (by rw [← (ha h.1.1).1]; exact hj')).mono (lm.trans lm')).bind₀
        fun x m₂ l₂ => Sim.pure fun h => ⟨rfl, ?_⟩
      rw [map_range_succ]
      exact (h.1.2.mono l₂).push h.2
    · have size := (ha h.1.1).1
      have accSize := h.1.2.1
      simp only [Array.size_map, Array.size_range] at accSize
      have e := (h.1.2.mono l').append h.2
      rw [Array.length_toList, accSize] at e
      have lj : j < galpha.toList.length := by simp; omega
      have len : (galpha.toList.take j).length = j := by simp; omega
      have succ := monomials_succ (galpha.toList.take j) galpha[j]!
      rw [len] at succ
      rw [succ] at e
      rwa [List.take_add_one, List.getElem?_eq_getElem lj, Option.toList_some, Array.getElem_toList,
        ← getElem!_pos]
  · rw [show Array.range (2 ^ 0) = #[0] by decide, Array.map_singleton, List.take_zero]
    exact ⟨rfl, fun i hi => match i, hi with | 0, _ => hone h⟩
  · rwa [Array.length_toList, (ha h).1, List.take_of_length_le (by simp)] at q

/-- The coefficients `α^[u]·2^-f·y^-u` (`setup.rs:268-273`). -/
theorem coefficients {monomials : Array ExtVar} {k : ℕ} {M : ℕ → Ext} {scale y : Var} {g0 gy : Gl}
    (hm : P → IsExts c n monomials ((Array.range k).map M)) (hs : P → IsWord c n scale g0)
    (hy : P → IsWord c n y gy) :
    Sim c n P (forIn monomials (scale, (#[] : Array ExtVar)) fun monomial (acc : Var × Array ExtVar) => do
        let coefficient ← Build.extScale monomial acc.1
        let next ← Build.mul acc.1 y
        Pure.pure (ForInStep.yield (next, acc.2.push coefficient)))
      (fun res m => IsExts c m res.2 ((Array.range k).map fun u => (M u).scale (g0 * gy.pow u))) True := by
  rw [← Array.forIn_toList]
  refine ((Sim.loop (P := P) (I := fun j (acc : Var × Array ExtVar) m => IsWord c m acc.1 (g0 * gy.pow j) ∧
      IsExts c m acc.2 ((Array.range j).map fun u => (M u).scale (g0 * gy.pow u)))
      (D := fun _ => True) fun j hj acc m lm => ?_).pre fun h => ⟨h, ?_, IsExts.range_zero _⟩).post
      (fun h _ _ q => ?_) fun _ => by simp
  · have hj' : j < monomials.size := by simpa using hj
    rw [show monomials.toList[j] = monomials[j]! by simp [getElem!_pos monomials j hj']]
    refine (Sim.extScale (fun h => ((hm h.1).get_map (by have := (hm h.1).1; simp at this; omega)).mono lm)
      fun h => h.2.1).bind₀ fun coefficient m₁ l₁ => ?_
    refine (Sim.mul (fun h => h.1.2.1.mono l₁) fun h => (hy h.1.1).mono (lm.trans l₁)).bind₀
      fun next m₂ l₂ => Sim.pure fun h => ⟨rfl, ?_, ?_⟩
    · have e : g0 * gy.pow j * gy = g0 * gy.pow (j + 1) := Gl.eq_of_toF (by
        rw [Gl.toF_mul, Gl.toF_mul, Gl.toF_mul, Gl.toF_pow, Gl.toF_pow, pow_succ, mul_assoc])
      rw [← e]
      exact h.2
    · rw [map_range_succ]
      exact (h.1.1.2.2.mono (l₁.trans l₂)).push (h.1.2.mono l₂)
  · have e : g0 * gy.pow 0 = g0 := Gl.eq_of_toF (by rw [Gl.toF_mul, Gl.toF_pow, pow_zero, mul_one])
    rw [e]
    exact hs h
  · have size : monomials.size = k := by simpa using (hm h).1
    rw [Array.length_toList, size] at q
    exact q.2

/-- The leaf weights `Σ_u coefficient_u·ω^((f - u)·v mod f)` (`setup.rs:274-284`). -/
theorem weights {coefficients : Array ExtVar} {k : ℕ} {C : ℕ → Ext} (omega : Gl)
    (hc : P → IsExts c n coefficients ((Array.range k).map C)) :
    Sim c n P (forIn [:2 ^ foldVars] (#[] : Array ExtVar) fun v (weight : Array ExtVar) => do
        let zero ← Algebra.extZero
        let total ← forIn coefficients.zipIdx zero fun x (total : ExtVar) => match x with
          | (coefficient, u) => do
            let term ← Algebra.extScaleConstant coefficient (omega.pow ((2 ^ foldVars - u) * v % 2 ^ foldVars))
            let total ← Build.extAdd total term
            Pure.pure (ForInStep.yield total)
        Pure.pure (ForInStep.yield (weight.push total)))
      (fun res m => IsExts c m res ((Array.range (2 ^ foldVars)).map fun v => Replay.sumRange k
        fun u => (C u).scale (omega.pow ((2 ^ foldVars - u) * v % 2 ^ foldVars)))) True := by
  rw [Std.Legacy.Range.forIn_eq_forIn_range']
  refine ((Sim.loop (P := P) (I := fun j (acc : Array ExtVar) m => IsExts c m acc ((Array.range j).map fun v =>
      Replay.sumRange k fun u => (C u).scale (omega.pow ((2 ^ foldVars - u) * v % 2 ^ foldVars))))
      (D := fun _ => True) fun j hj acc m lm => ?_).pre fun h => ⟨h, IsExts.range_zero _⟩).post
      (fun h _ _ q => ?_) fun _ => by simp
  · rw [show (List.range' [:2 ^ foldVars].start [:2 ^ foldVars].size [:2 ^ foldVars].step)[j] = j by simp]
    refine extZero.bind₀ fun zero m₁ l₁ => ?_
    rw [← Array.forIn_toList]
    refine (Sim.bind (C' := True) ((Sim.loop (I := fun u (total : ExtVar) m' =>
        IsExt c m' total (Replay.sumRange u fun u => (C u).scale (omega.pow ((2 ^ foldVars - u) * j % 2 ^ foldVars))))
        (D := fun _ => True) fun u hu total m' lm' => ?_).pre fun h => by exact ⟨h, h.2⟩)
        fun total m' l' => Sim.pure fun h => ⟨rfl, ?_⟩).post (fun _ _ _ h => h) fun _ => by simp
    · have hu' : u < coefficients.size := by simpa using hu
      rw [show coefficients.zipIdx.toList[u] = (coefficients[u]!, u) by simp [getElem!_pos coefficients u hu']]
      dsimp only
      refine (extScaleConstant _ fun h => ((hc h.1.1.1).get_map (by have := (hc h.1.1.1).1; simp at this; omega)).mono
        (lm.trans (l₁.trans lm'))).bind₀ fun term m₂ l₂ =>
        (extAdd (fun h => h.1.2.mono l₂) fun h => h.2).bind₀ fun total' m₃ l₃ => Sim.pure fun h => ⟨rfl, ?_⟩
      rw [sumRange_succ]
      exact h.2
    · have e := h.2
      have size : coefficients.size = k := by simpa using (hc h.1.1.1).1
      rw [Array.length_toList, Array.size_zipIdx, size] at e
      rw [map_range_succ]
      exact (h.1.1.2.mono (l₁.trans l')).push e
  · simpa [Std.Legacy.Range.size] using q

/-- One group of the leaf check (`setup.rs:287-298`): each weight times its
oracle's coset values, summed per coset position, then the dot product with
the leaf weights. -/
theorem groupStep {values : Array Var} {gvalues : Array Gl} {weight ws : Array ExtVar} {gweight : ℕ → Ext}
    {gws gfolded : Array Ext} {acc : ℕ × Array ExtVar} (hv : P → IsWords c n values gvalues)
    (hweight : P → IsExts c n weight ((Array.range (2 ^ foldVars)).map gweight)) (hws : P → IsExts c n ws gws)
    (hf : P → IsExts c n acc.2 gfolded) :
    Sim c n P (do
        let zero ← Algebra.extZero
        let s ← forIn ws (acc.1, Array.replicate (2 ^ foldVars) zero) fun w (s : ℕ × Array ExtVar) => do
          let combined ← forIn (values.extract (2 ^ foldVars * s.1) (2 ^ foldVars * s.1 + 2 ^ foldVars)).zipIdx s.2
            fun x (combined : Array ExtVar) => match x with
            | (value, v) => do
              let term ← Build.extScale w value
              let sum ← Build.extAdd combined[v]! term
              Pure.pure (ForInStep.yield (combined.set! v sum))
          Pure.pure (ForInStep.yield (s.1 + 1, combined))
        let f ← Algebra.dot weight s.2
        Pure.pure (ForInStep.yield (s.1, acc.2.push f)))
      (fun r m => r = .yield r.value ∧ r.value.1 = acc.1 + ws.size ∧
        IsExts c m r.value.2 (gfolded.push (Replay.sumRange (2 ^ foldVars) fun v => gweight v *
          Replay.sumRange gws.size fun k => gws[k]!.scale gvalues[(acc.1 + k) * 2 ^ foldVars + v]!))) True := by
  obtain ⟨S, hS⟩ : ∃ S : ℕ → ℕ → Ext, S = fun k v =>
      Replay.sumRange k fun k' => gws[k']!.scale gvalues[(acc.1 + k') * 2 ^ foldVars + v]! := ⟨_, rfl⟩
  refine extZero.bind₀ fun zero m₁ l₁ => ?_
  rw [← Array.forIn_toList]
  refine (Sim.bind (C' := True) ((Sim.loop (I := fun k (s : ℕ × Array ExtVar) m => s.1 = acc.1 + k ∧
      IsExts c m s.2 ((Array.range (2 ^ foldVars)).map (S k)))
      (D := fun _ => True) fun k hk s m lm => ?_).pre fun h => by
        exact ⟨h, rfl, IsExts.replicate_map h.2 fun v _ => by rw [hS]; simp [Replay.sumRange]⟩) fun s m₂ l₂ => ?_).post
      (fun _ _ _ h => h) fun _ => by simp
  · have hk' : k < ws.size := by simpa using hk
    rw [show ws.toList[k] = ws[k]! by simp [getElem!_pos ws k hk'], ← Array.forIn_toList]
    refine (Sim.bind (C' := True) ((Sim.loop (I := fun j (comb : Array ExtVar) m' =>
        IsExts c m' comb ((Array.range (2 ^ foldVars)).map fun v =>
          if v < j then S k v + gws[k]!.scale gvalues[2 ^ foldVars * s.1 + v]! else S k v))
        (D := fun _ => True) fun j hj comb m' lm' => ?_).pre fun h => by
          exact ⟨h, h.2.2.congr_map fun v _ => by simp⟩)
        fun comb m₃ l₃ => Sim.pure fun h => ⟨rfl, by have := h.1.2.1; simp only [ForInStep.value]; omega, ?_⟩).post
        (fun _ _ _ h => h) fun _ => by simp
    · have hj' : j < (values.extract (2 ^ foldVars * s.1) (2 ^ foldVars * s.1 + 2 ^ foldVars)).size := by
        simpa only [Array.length_toList, Array.size_zipIdx] using hj
      rw [show (values.extract (2 ^ foldVars * s.1) (2 ^ foldVars * s.1 + 2 ^ foldVars)).zipIdx.toList[j] =
        ((values.extract (2 ^ foldVars * s.1) (2 ^ foldVars * s.1 + 2 ^ foldVars))[j]!, j) by
        simp [getElem!_pos (values.extract (2 ^ foldVars * s.1) (2 ^ foldVars * s.1 + 2 ^ foldVars)) j hj']]
      dsimp only
      have small : j < 2 ^ foldVars := by simp at hj'; omega
      have inside : 2 ^ foldVars * s.1 + j < values.size := by simp at hj'; omega
      refine (Sim.extScale (gs := gvalues[2 ^ foldVars * s.1 + j]!)
        (fun h => ((hws h.1.1.1).get (by rw [← (hws h.1.1.1).1]; exact hk')).mono (l₁.trans (lm.trans lm')))
        fun h => ?_).bind₀ fun term m₄ l₄ => ?_
      · rw [getElem!_pos (values.extract (2 ^ foldVars * s.1) (2 ^ foldVars * s.1 + 2 ^ foldVars)) j hj',
          Array.getElem_extract, ← getElem!_pos]
        exact ((hv h.1.1.1).get (by rw [← (hv h.1.1.1).1]; exact inside)).mono (l₁.trans (lm.trans lm'))
      refine (Sim.extAdd (ga := S k j) (fun h => ?_) fun h => h.2).bind₀ fun sum m₅ l₅ =>
        Sim.pure fun h => ⟨rfl, ?_⟩
      · have e := h.1.2.get_map small
        rw [if_neg (lt_irrefl j)] at e
        exact e.mono l₄
      · have size : comb.size = 2 ^ foldVars := by simpa using h.1.1.2.1
        simp only [ForInStep.value]
        refine ⟨by simp [size], fun v hv => ?_⟩
        rw [Array.size_set!, size] at hv
        rw [getElem!_set! _ _ _ _ (by rw [size]; exact hv), getElem!_map_range _ _ hv]
        by_cases e : j = v
        · subst e
          rw [if_pos rfl, if_pos (Nat.lt_succ_self j)]
          exact h.2
        · rw [if_neg e]
          have old := (h.1.1.2.get_map hv).mono (l₄.trans l₅)
          by_cases lt : v < j
          · rwa [if_pos lt, if_pos (by omega)] at *
          · rwa [if_neg lt, if_neg (by omega)] at *
    · refine h.2.congr_map fun v _ => ?_
      have hs := h.1.2.1
      simp only [hS, sumRange_succ]
      split_ifs with lt
      · rw [hs, show 2 ^ foldVars * (acc.1 + k) = (acc.1 + k) * 2 ^ foldVars from Nat.mul_comm _ _]
      · have size := (hv h.1.1.1).1
        simp only [Array.length_toList, Array.size_zipIdx, Array.size_extract] at lt
        have far : ¬ 2 ^ foldVars * s.1 + v < gvalues.size := by rw [← size]; omega
        rw [show (acc.1 + k) * 2 ^ foldVars + v = 2 ^ foldVars * s.1 + v by
          rw [hs, Nat.mul_comm], getElem!_neg gvalues _ far, add_scale_zero]
  · refine (Sim.dot (fun h => (hweight h.1.1).mono (l₁.trans l₂)) fun h => h.2.2).bind₀ fun f m₃ l₃ =>
      Sim.pure fun h => ⟨rfl, by simp only [ForInStep.value]; rw [h.1.2.1, Array.length_toList], ?_⟩
    have size := (hws h.1.1.1).1
    have e := h.1.2.1
    rw [Array.length_toList, size] at e
    refine ((hf h.1.1.1).mono (l₁.trans (l₂.trans l₃))).push ?_
    refine h.2.congr ?_
    rw [Replay.dot, Array.size_map, Array.size_map, Array.size_range, Nat.min_self]
    refine congrArg coords (sumRange_congr fun v hv => ?_)
    rw [getElem!_map_range _ _ hv, getElem!_map_range _ _ hv, show ws.toList.length = gws.size by simp [size], hS]

/-- The computed root against the key root, lane by lane (`setup.rs:255-257`). -/
theorem rootEqual {computed root : DigestVar} {gcomputed groot : Array Gl} (hc : P → IsDigest c n computed gcomputed)
    (hr : P → IsDigest c n root groot) :
    Sim c n P (forIn (List.finRange 4) PUnit.unit fun lane _ => do
        Build.assertEqual (computed lane) (root lane)
        Pure.pure (ForInStep.yield PUnit.unit))
      (fun _ _ => True) (∀ k < 4, gcomputed[k]! = groot[k]!) := by
  refine ((Sim.loop (P := P) (I := fun _ _ _ => True) (D := fun k => gcomputed[k]! = groot[k]!)
    fun k hk _ m lm => ?_).pre fun h => ⟨h, trivial⟩).post (fun _ _ _ _ => trivial) fun _ => by simp
  have k4 : k < 4 := by simpa using hk
  rw [List.getElem_finRange]
  refine ((Sim.assertEqual (ga := gcomputed[k]!) (gb := groot[k]!) (fun h => ?_) fun h => ?_).bind fun _ _ _ =>
    Sim.pure fun _ => ⟨rfl, trivial⟩).post (fun _ _ _ h => h) fun _ => by simp
  · have := (hc h.1).2 ⟨k, k4⟩
    rw [Fin.getElem!_fin] at this
    exact this.mono lm
  · have := (hr h.1).2 ⟨k, k4⟩
    rw [Fin.getElem!_fin] at this
    exact this.mono lm

/-- The two groups of the leaf check (`setup.rs:285-298`): the key rows, then
the run oracles from offset `|w₀|`. -/
theorem groups {values : Array Var} {gvalues : Array Gl} {weight w₀ w₁ : Array ExtVar} {gweight : ℕ → Ext}
    {gw₀ gw₁ : Array Ext} (hv : P → IsWords c n values gvalues)
    (hweight : P → IsExts c n weight ((Array.range (2 ^ foldVars)).map gweight)) (hw₀ : P → IsExts c n w₀ gw₀)
    (hw₁ : P → IsExts c n w₁ gw₁) :
    Sim c n P (forIn #[w₀, w₁] ((0 : ℕ), (#[] : Array ExtVar)) fun ws (acc : ℕ × Array ExtVar) => do
        let zero ← Algebra.extZero
        let s ← forIn ws (acc.1, Array.replicate (2 ^ foldVars) zero) fun w (s : ℕ × Array ExtVar) => do
          let combined ← forIn (values.extract (2 ^ foldVars * s.1) (2 ^ foldVars * s.1 + 2 ^ foldVars)).zipIdx s.2
            fun x (combined : Array ExtVar) => match x with
            | (value, v) => do
              let term ← Build.extScale w value
              let sum ← Build.extAdd combined[v]! term
              Pure.pure (ForInStep.yield (combined.set! v sum))
          Pure.pure (ForInStep.yield (s.1 + 1, combined))
        let f ← Algebra.dot weight s.2
        Pure.pure (ForInStep.yield (s.1, acc.2.push f)))
      (fun r m => IsExts c m r.2
        #[Replay.sumRange (2 ^ foldVars) fun v => gweight v *
            Replay.sumRange gw₀.size fun k => gw₀[k]!.scale gvalues[(0 + k) * 2 ^ foldVars + v]!,
          Replay.sumRange (2 ^ foldVars) fun v => gweight v *
            Replay.sumRange gw₁.size fun k => gw₁[k]!.scale gvalues[(gw₀.size + k) * 2 ^ foldVars + v]!]) True := by
  rw [← Array.forIn_toList, show #[w₀, w₁].toList = [w₀, w₁] from rfl, List.forIn_cons]
  refine (groupStep (acc := (0, #[])) (gfolded := #[]) hv hweight hw₀ fun _ => IsExts.empty).bind₀
    fun r m₁ l₁ => ?_
  rcases r with b | b
  · exact (Sim.pure (Q := fun _ _ => True) fun _ => trivial).post (fun h => absurd h.2.1 (by simp))
      fun h => absurd h.2.1 (by simp)
  · dsimp only
    rw [List.forIn_cons]
    refine (groupStep (acc := b) (gfolded := #[].push _) (fun h => (hv h.1).mono l₁)
      (fun h => (hweight h.1).mono l₁) (fun h => (hw₁ h.1).mono l₁) fun h => h.2.2.2).bind₀ fun r m₂ l₂ => ?_
    rcases r with b' | b'
    · exact (Sim.pure (Q := fun _ _ => True) fun _ => trivial).post (fun h => absurd h.2.1 (by simp))
        fun h => absurd h.2.1 (by simp)
    · dsimp only
      rw [List.forIn_nil]
      refine Sim.pure fun h => ?_
      have offset : b.1 = gw₀.size := by
        have e := h.1.2.2.1
        simp only [ForInStep.value, Nat.zero_add] at e
        rw [e, (hw₀ h.1.1).1]
      have e := h.2.2.2
      simp only [ForInStep.value] at e
      rw [offset] at e
      exact e

end Sim

/-- `setup::check` (`setup.rs:243-300`): once the circuit's checks hold, the
leaf's path reaches the key root and the replay's leaf check returns the
circuit's folded values. -/
theorem setupCheck_ensures [CharP F goldilocksModulus] {c : Ctx F} {n : ℕ} {P : Prop} {root : DigestVar}
    {groot : Array Gl} {vars index : ℕ} {bits : Array Var} {leaf : LeafView} {g : Replay.Leaf}
    {w₀ w₁ alpha : Array ExtVar} {gw₀ gw₁ galpha : Array Ext} (hp : Permutes c.perm)
    (hroot : P → IsDigest c n root groot)
    (hbits : P → bits.size = vars + 1 - foldVars ∧
      ∀ i < bits.size, IsWord c n bits[i]! (Gl.ofNat (index / 2 ^ i % 2)))
    (hvalues : P → IsWords c n leaf.values g.values)
    (hpath : P → leaf.path.size = g.path.size ∧ leaf.path.size = vars + 1 - foldVars ∧
      ∀ k < leaf.path.size, IsDigest c n leaf.path[k]! g.path[k]!)
    (hw₀ : P → IsExts c n w₀ gw₀) (hw₁ : P → IsExts c n w₁ gw₁) (halpha : P → IsExts c n alpha galpha)
    (hsize : galpha.size = foldVars) :
    Ensures c n P (Circuit.setupCheck root vars bits leaf #[w₀, w₁] alpha)
      (fun res m => ∃ vals, Replay.setupCheck groot vars index g #[gw₀, gw₁] galpha = .ok vals ∧
        IsExts c m res vals) := by
  obtain ⟨Y, hY⟩ : ∃ Y, Y = Replay.powerFromBits (twoAdicGenerator (vars + 1)).inv index (vars + 1 - foldVars) *
      generator.inv := ⟨_, rfl⟩
  unfold Circuit.setupCheck
  refine (Sim.hashLeaf hp hvalues).ensures.bind fun digest m₁ l₁ => ?_
  refine (Sim.merkleRoot' (index := index) (gpath := g.path) hp (fun h => h.2.1)
    (fun h i hi => ((hbits h.1).2 i hi).mono l₁) (fun h k hk => ((hpath h.1).2.2 k hk).mono l₁)
    (fun h => by rw [(hbits h.1).1, (hpath h.1).2.1]) fun h => (hpath h.1).1).ensures.bind
    fun computed m₂ l₂ => ?_
  refine (Sim.rootEqual (fun h => h.2.1) fun h => (hroot h.1.1).mono (l₁.trans l₂)).ensures.bind
    fun _ m₃ l₃ => ?_
  refine Ensures.pre (P' := P ∧ Replay.pathOk groot index g = true) ?_ fun h => ⟨h.1.1.1, pathOk_of h.2.2⟩
  refine (Sim.powerFromBits (twoAdicGenerator (vars + 1)).inv index fun h i hi =>
    ((hbits h.1).2 i hi).mono (l₁.trans (l₂.trans l₃))).ensures.bind fun rotation m₄ l₄ => ?_
  refine (Sim.scale generator.inv fun h => h.2.1).ensures.bind fun y m₅ l₅ => ?_
  have L : n ≤ m₅ := l₁.trans (l₂.trans (l₃.trans (l₄.trans l₅)))
  refine Ensures.pre (P' := (P ∧ Replay.pathOk groot index g = true) ∧ IsWord c m₅ y Y) ?_ fun h =>
    ⟨h.1.1, by rw [hY, ← (hbits h.1.1.1).1]; exact h.2.1⟩
  refine Sim.extOne.ensures.bind fun one m₆ l₆ => ?_
  refine (Sim.monomials (fun h => h.2.1) fun h => (halpha h.1.1.1).mono (L.trans l₆)).ensures.bind
    fun monomials m₇ l₇ => ?_
  refine (Sim.const (Gl.ofNat (2 ^ foldVars)).inv).ensures.bind fun scale m₈ l₈ => ?_
  refine (Sim.coefficients (fun h => h.1.2.1.mono l₈) (fun h => h.2.1)
    fun h => h.1.1.1.2.mono (l₆.trans (l₇.trans l₈))).ensures.bind fun coefs m₉ l₉ => ?_
  refine (Sim.weights (twoAdicGenerator foldVars) fun h => h.2.1).ensures.bind fun weight m₁₀ l₁₀ => ?_
  have L' : n ≤ m₁₀ := L.trans (l₆.trans (l₇.trans (l₈.trans (l₉.trans l₁₀))))
  refine (Sim.groups (gweight := Replay.leafWeight galpha Y (twoAdicGenerator foldVars))
    (fun h => (hvalues h.1.1.1.1.1.1.1).mono L') (fun h => by have e := h.2.1; rw [hsize] at e; exact e)
    (fun h => (hw₀ h.1.1.1.1.1.1.1).mono L') fun h => (hw₁ h.1.1.1.1.1.1.1).mono L').ensures.bind
    fun s m₁₁ l₁₁ => Ensures.pure fun h => ⟨_, setupCheck_ok h.1.1.1.1.1.1.1.2, ?_⟩
  rw [← hY]
  exact h.2.1

/-- The setup queries (`verifier.rs:246-249`): once each leaf's checks hold,
the replay's fold over the queries returns the circuit's folded values. -/
theorem queriedLoop [CharP F goldilocksModulus] {c : Ctx F} {n : ℕ} {P : Prop} {d : Duplex} {rootV : Array Var}
    {groot : Array Gl} {vars : ℕ} {indexBits : Array (Array Var)} {queries : Array ℕ} {leaves : Array LeafView}
    {gleaves : Array Replay.Leaf} {w₀ w₁ alpha : Array ExtVar} {gw₀ gw₁ galpha : Array Ext} (hp : Permutes c.perm)
    (hroot : P → IsWords c n rootV groot) (root : groot.size = 4)
    (hbits : P → indexBits.size = queries.size ∧ ∀ q < indexBits.size,
      (indexBits[q]!).size = vars + 1 - foldVars ∧
      ∀ i < vars + 1 - foldVars, IsWord c n (indexBits[q]!)[i]! (Gl.ofNat (queries[q]! / 2 ^ i % 2)))
    (hleaves : P → leaves.size = gleaves.size ∧ ∀ q (hq : q < leaves.size),
      IsWords c n leaves[q].values (gleaves[q]!).values ∧ leaves[q].path.size = (gleaves[q]!).path.size ∧
      leaves[q].path.size = vars + 1 - foldVars ∧
      ∀ k < leaves[q].path.size, IsDigest c n leaves[q].path[k]! (gleaves[q]!).path[k]!)
    (hw₀ : P → IsExts c n w₀ gw₀) (hw₁ : P → IsExts c n w₁ gw₁) (halpha : P → IsExts c n alpha galpha)
    (hsize : galpha.size = foldVars) :
    Ensures c n P ((forIn (indexBits.zip leaves) (#[] : Array ExtVar) fun x (queried : Array ExtVar) => match x with
        | (bits, leaf) => do
          let r ← liftM (Circuit.setupCheck (fun lane => rootV[lane]!) vars bits leaf #[w₀, w₁] alpha)
          Pure.pure (ForInStep.yield (queried ++ r)) : Transcript (Array ExtVar)).run d)
      (fun res m => ∃ gq, (queries.zip gleaves).toList.foldlM (fun queried pair => do
          Pure.pure (queried ++ (← Replay.setupCheck groot vars pair.1 pair.2 #[gw₀, gw₁] galpha))) #[] = .ok gq ∧
        IsExts c m res.1 gq ∧ res.2 = d) := by
  rw [← Array.forIn_toList]
  refine ((Ensures.loopT (P := P) (d := d) (I := fun k (acc : Array ExtVar) e m =>
      (∃ gq, ((queries.zip gleaves).toList.take k).foldlM (fun queried pair => do
        Pure.pure (queried ++ (← Replay.setupCheck groot vars pair.1 pair.2 #[gw₀, gw₁] galpha))) #[] = .ok gq ∧
        IsExts c m acc gq) ∧ e = d) fun k hk acc e m lm => ?_).pre fun h => by
      exact ⟨h, ⟨#[], rfl, IsExts.empty⟩, rfl⟩).post fun h _ _ q => ?_
  · have hk' : k < indexBits.size ∧ k < leaves.size := by simpa using hk
    rw [show (indexBits.zip leaves).toList[k] = (indexBits[k]!, leaves[k]'hk'.2) by
      simp [getElem!_pos indexBits k hk'.1]]
    dsimp only
    refine Ensures.tbind (Ensures.lift (setupCheck_ensures (groot := groot) (index := queries[k]!)
      (g := gleaves[k]!) (gw₀ := gw₀) (gw₁ := gw₁) (galpha := galpha) hp
      (fun h => ⟨root, fun lane => by rw [Fin.getElem!_fin]; exact ((hroot h.1).get (by omega)).mono lm⟩)
      (fun h => ⟨((hbits h.1).2 k hk'.1).1, fun i hi => (((hbits h.1).2 k hk'.1).2 i
        (by rw [← ((hbits h.1).2 k hk'.1).1]; exact hi)).mono lm⟩)
      (fun h => (((hleaves h.1).2 k hk'.2).1).mono lm)
      (fun h => ⟨((hleaves h.1).2 k hk'.2).2.1, ((hleaves h.1).2 k hk'.2).2.2.1,
        fun j hj => (((hleaves h.1).2 k hk'.2).2.2.2 j hj).mono lm⟩)
      (fun h => (hw₀ h.1).mono lm) (fun h => (hw₁ h.1).mono lm) (fun h => (halpha h.1).mono lm) hsize))
      fun r e₁ m₁ l₁ => ?_
    rw [StateT.run_pure]
    refine Ensures.pure fun h => ⟨rfl, ?_, h.2.2.trans h.1.2.2⟩
    obtain ⟨⟨hP, ⟨gq, run, hq⟩, -⟩, ⟨vals, check, hv⟩, -⟩ := h
    have lt : k < (queries.zip gleaves).toList.length := by
      have := (hbits hP).1
      have := (hleaves hP).1
      simp
      omega
    refine ⟨gq ++ vals, ?_, (hq.mono l₁).append hv⟩
    rw [List.take_add_one, List.getElem?_eq_getElem lt, Option.toList_some, List.foldlM_append, run]
    simp only [Array.getElem_toList, Array.getElem_zip, List.foldlM_cons, List.foldlM_nil]
    rw [← getElem!_pos queries k (by simp at lt; omega), ← getElem!_pos gleaves k (by simp at lt; omega), check]
    rfl
  · have e₁ := (hbits h).1
    have e₂ := (hleaves h).1
    rw [List.take_of_length_le (by simp; omega)] at q
    obtain ⟨⟨gq, run, hq⟩, he⟩ := q
    exact ⟨gq, run, hq, he⟩

/-- What the openings read of the honest fold (`verifier.rs:208-249`,
`262-265`): `α`, the key and run points padded to the setup variables, the
digits of each query in its index bits, and the queried folded values. -/
def IsHonestOut (c : Ctx F) (m : ℕ) (rel : Relation) (early : Replay.EarlyOut) (lin : Replay.LinearOut)
    (v : HonestOut) (out : Replay.HonestOut) : Prop :=
  IsExts c m v.alpha out.alpha ∧
    IsExts c m v.keyPoint (Replay.padded rel.setupVars (lin.lanePoint ++ lin.blockPoint)) ∧
    IsExts c m v.runPoint (Replay.padded rel.setupVars early.openings.runPoint) ∧
    v.indexBits.size = out.queries.size ∧
    (∀ q < v.indexBits.size, (v.indexBits[q]!).size = rel.setupVars + 1 - foldVars ∧
      ∀ i < rel.setupVars + 1 - foldVars, IsWord c m (v.indexBits[q]!)[i]! (Gl.ofNat (out.queries[q]! / 2 ^ i % 2))) ∧
    IsExts c m v.queried out.queried

/-- `IsHonestOut` survives later values. -/
theorem IsHonestOut.mono {c : Ctx F} {n m : ℕ} {rel : Relation} {early : Replay.EarlyOut} {lin : Replay.LinearOut}
    {v : HonestOut} {out : Replay.HonestOut} (h : IsHonestOut c n rel early lin v out) (le : n ≤ m) :
    IsHonestOut c m rel early lin v out :=
  ⟨h.1.mono le, h.2.1.mono le, h.2.2.1.mono le, h.2.2.2.1,
    fun q hq => ⟨(h.2.2.2.2.1 q hq).1, fun i hi => ((h.2.2.2.2.1 q hq).2 i hi).mono le⟩, h.2.2.2.2.2.mono le⟩

/-- **The honest fold** (`verifier.rs:208-249`). Once the circuit's checks
hold, the replay's honest stage runs from the matching transcript, the
circuit's outputs are the replay's, and the duplexes still match. -/
theorem honestStage_ensures [CharP F goldilocksModulus] {c : Ctx F} {n : ℕ} {P : Prop} {rel : Relation}
    {input : HonestInput} {proof : Replay.Proof} {early : Replay.EarlyOut} {lin : Replay.LinearOut} {d : Duplex}
    {t : Replay.Transcript} (hp : Permutes c.perm) (hd : P → IsDuplex c n d t.duplex)
    (hlane : P → IsExts c n input.lanePoint lin.lanePoint) (hblock : P → IsExts c n input.blockPoint lin.blockPoint)
    (hrun : P → IsExts c n input.runPoint early.openings.runPoint)
    (hrow : P → IsExt c n input.setup.1 early.openings.rowClaim)
    (hslot : P → IsExt c n input.setup.2.1 early.openings.slotClaim)
    (hcoef : P → IsExt c n input.setup.2.2 early.openings.coefClaim)
    (hblockValue : P → IsExt c n input.block early.values.block)
    (hrows : P → IsExts c n input.commitmentRows ((Array.range rel.kappa).map lin.mixing.commitment))
    (hkey : P → IsExt c n input.keyValue lin.keyValue)
    (hpartials : P → input.partials.size = proof.partials.size ∧
      ∀ g < input.partials.size, IsExts c n input.partials[g]! proof.partials[g]!)
    (hp1 : P → IsDigest c n input.p1 proof.p1)
    (hleaves : P → input.leaves.size = proof.leaves.size ∧ ∀ q (hq : q < input.leaves.size),
      IsWords c n input.leaves[q].values (proof.leaves[q]!).values ∧
      input.leaves[q].path.size = (proof.leaves[q]!).path.size ∧
      input.leaves[q].path.size = rel.setupVars + 1 - foldVars ∧
      ∀ k < input.leaves[q].path.size, IsDigest c n input.leaves[q].path[k]! (proof.leaves[q]!).path[k]!)
    (keySize : (lin.lanePoint ++ lin.blockPoint).size ≤ rel.setupVars)
    (runSize : early.openings.runPoint.size ≤ rel.setupVars) (root : rel.key.root.size = 4)
    (small : rel.setupVars + 1 - foldVars ≤ 64) :
    Ensures c n P ((Circuit.honestStage rel input).run d)
      (fun res m => ∃ out t', Runs (Replay.honestStage rel proof early lin) t out t' ∧
        IsHonestOut c m rel early lin res.1 out ∧ IsDuplex c m res.2 t'.duplex) := by
  obtain ⟨lanePoint, blockPoint, runPoint, ⟨row, slot, coefficient⟩, block, commitmentRows, keyValue, partials, p1,
    leaves⟩ := input
  dsimp only at hlane hblock hrun hrow hslot hcoef hblockValue hrows hkey hpartials hp1 hleaves
  unfold Circuit.honestStage
  refine Ensures.tbind (Sim.sampleExt hp hd).ensures fun mu d₁ m₁ l₁ => ?_
  refine Ensures.tbind (Sim.observePartials hp (fun h => h.2.1.2) fun h =>
    ⟨(hpartials h.1).1, fun g hg => ((hpartials h.1).2 g hg).mono l₁⟩).ensures fun _ d₂ m₂ l₂ => ?_
  rw [Sim.mapM_sampleExt]
  refine Ensures.tbind (Sim.sampleExt hp fun h => h.2.1).ensures fun a₀ d₃ m₃ l₃ => ?_
  refine Ensures.tbind (Sim.sampleExt hp fun h => h.2.1.2).ensures fun a₁ d₄ m₄ l₄ => ?_
  refine Ensures.tbind (Sim.sampleExt hp fun h => h.2.1.2).ensures fun a₂ d₅ m₅ l₅ => ?_
  set gmu : Ext := ⟨t.duplex.sample.1, t.duplex.sample.2.sample.1, t.duplex.sample.2.sample.2.sample.1⟩
  set r₂ := (proof.partials.flatten.flatMap fun v => #[v.c0, v.c1, v.c2]).foldl Replay.Duplex.observe
    t.duplex.sample.2.sample.2.sample.2
  set ga₀ : Ext := ⟨r₂.sample.1, r₂.sample.2.sample.1, r₂.sample.2.sample.2.sample.1⟩
  set r₃ := r₂.sample.2.sample.2.sample.2
  set ga₁ : Ext := ⟨r₃.sample.1, r₃.sample.2.sample.1, r₃.sample.2.sample.2.sample.1⟩
  set r₄ := r₃.sample.2.sample.2.sample.2
  set ga₂ : Ext := ⟨r₄.sample.1, r₄.sample.2.sample.1, r₄.sample.2.sample.2.sample.1⟩
  set r₅ := r₄.sample.2.sample.2.sample.2
  have L₅ : n ≤ m₅ := l₁.trans (l₂.trans (l₃.trans (l₄.trans l₅)))
  refine Ensures.pre (P' := P ∧ IsExt c m₅ mu gmu ∧ IsExts c m₅ #[a₀, a₁, a₂] #[ga₀, ga₁, ga₂] ∧
    IsDuplex c m₅ d₅ r₅) ?_ fun h => ⟨h.1.1.1.1.1, h.1.1.1.1.2.1.1.mono (l₂.trans (l₃.trans (l₄.trans l₅))),
      IsExts.triple (h.1.1.2.1.1.mono (l₄.trans l₅)) (h.1.2.1.1.mono l₅) h.2.1.1, h.2.1.2⟩
  refine Ensures.tbind (Ensures.lift (Sim.setupPoint (fun h => ((hlane h.1).append (hblock h.1)).mono L₅)
    keySize).ensures) fun keyPoint d₆ m₆ l₆ => ?_
  refine Ensures.tbind (Ensures.lift (Sim.setupPoint (fun h => (hrun h.1.1).mono (L₅.trans l₆))
    runSize).ensures) fun runPoint' d₇ m₇ l₇ => ?_
  refine Ensures.tbind (Sim.embedLoop
    (fun h => ((hrun h.1.1.1).extract rel.key.layout.slotVariables).mono (L₅.trans (l₆.trans l₇)))
    fun h => (hblockValue h.1.1.1).mono (L₅.trans (l₆.trans l₇))).ensures fun embedded d₈ m₈ l₈ => ?_
  refine Ensures.tbind (Sim.runValueLoop (fun h => IsExts.triple
    ((hcoef h.1.1.1.1).mono (L₅.trans (l₆.trans (l₇.trans l₈))))
    ((hslot h.1.1.1.1).mono (L₅.trans (l₆.trans (l₇.trans l₈))))
    ((hrow h.1.1.1.1).mono (L₅.trans (l₆.trans (l₇.trans l₈))))) (fun h => h.2.1.1)
    fun h => h.1.1.1.2.1.mono (l₆.trans (l₇.trans l₈))).ensures fun runValue d₉ m₉ l₉ => ?_
  have L₉ : n ≤ m₉ := L₅.trans (l₆.trans (l₇.trans (l₈.trans l₉)))
  refine Ensures.pre (P' := P ∧ IsExt c m₉ mu gmu ∧ IsExts c m₉ #[a₀, a₁, a₂] #[ga₀, ga₁, ga₂] ∧
    IsExts c m₉ keyPoint (Replay.padded rel.setupVars (lin.lanePoint ++ lin.blockPoint)) ∧
    IsExts c m₉ runPoint' (Replay.padded rel.setupVars early.openings.runPoint) ∧
    IsExt c m₉ runValue (#[early.openings.coefClaim, early.openings.slotClaim, early.openings.rowClaim].foldl
      (fun total value => value + gmu * total) ((early.openings.runPoint.extract rel.key.layout.slotVariables).foldl
        (fun total x => total * Replay.oneMinus x) early.values.block)) ∧
    IsDuplex c m₉ d₉ r₅) ?_ fun h => ?_
  rotate_left
  · obtain ⟨⟨⟨⟨⟨hP, hmu, halpha, hd₅⟩, ⟨hkp, -⟩, e₆⟩, ⟨hrp, -⟩, e₇⟩, ⟨hem, e₈⟩, -⟩, ⟨hrv, e₉⟩, -⟩ := h
    refine ⟨hP, hmu.mono (l₆.trans (l₇.trans (l₈.trans l₉))), halpha.mono (l₆.trans (l₇.trans (l₈.trans l₉))),
      hkp.mono (l₇.trans (l₈.trans l₉)), hrp.mono (l₈.trans l₉), hrv, ?_⟩
    dsimp only at e₆ e₇ e₈ e₉
    rw [e₉, e₈, e₇, e₆]
    exact hd₅.mono (l₆.trans (l₇.trans (l₈.trans l₉)))
  set gkey := Replay.padded rel.setupVars (lin.lanePoint ++ lin.blockPoint)
  set grun := Replay.padded rel.setupVars early.openings.runPoint
  set grv := #[early.openings.coefClaim, early.openings.slotClaim, early.openings.rowClaim].foldl
    (fun total value => value + gmu * total) ((early.openings.runPoint.extract rel.key.layout.slotVariables).foldl
      (fun total x => total * Replay.oneMinus x) early.values.block)
  refine Ensures.tbind (Ensures.lift (Sim.combinePartials (fun h => (IsExts.nested (hpartials h.1) 0).mono L₉)
    fun h => h.2.2.2.1.slice 0 foldVars).ensures) fun keyPartial d₁₀ m₁₀ l₁₀ => ?_
  refine Ensures.tbind (Ensures.lift (Sim.assertExtEqual (fun h => h.2.1.1)
    fun h => (hkey h.1.1).mono (L₉.trans l₁₀)).ensures) fun _ d₁₁ m₁₁ l₁₁ => ?_
  refine Ensures.tbind (Ensures.lift (Sim.combinePartials
    (fun h => (IsExts.nested (hpartials h.1.1.1) 1).mono (L₉.trans (l₁₀.trans l₁₁)))
    fun h => (h.1.1.2.2.2.2.1.slice 0 foldVars).mono (l₁₀.trans l₁₁)).ensures) fun runPartial d₁₂ m₁₂ l₁₂ => ?_
  refine Ensures.tbind (Ensures.lift (Sim.assertExtEqual (fun h => h.2.1.1)
    fun h => h.1.1.1.2.2.2.2.2.1.mono (l₁₀.trans (l₁₁.trans l₁₂))).ensures) fun _ d₁₃ m₁₃ l₁₃ => ?_
  refine Ensures.tbind (Sim.observeRoot (r := r₅) hp (fun h => ?_)
    fun h => (hp1 h.1.1.1.1.1).mono (L₉.trans (l₁₀.trans (l₁₁.trans (l₁₂.trans l₁₃))))).ensures
    fun _ d₁₄ m₁₄ l₁₄ => ?_
  · obtain ⟨⟨⟨⟨⟨-, -, -, -, -, -, hd₉⟩, -, e₁₀⟩, -, e₁₁⟩, -, e₁₂⟩, -, e₁₃⟩ := h
    dsimp only at e₁₀ e₁₁ e₁₂ e₁₃
    rw [e₁₃, e₁₂, e₁₁, e₁₀]
    exact hd₉.mono (l₁₀.trans (l₁₁.trans (l₁₂.trans l₁₃)))
  set r₇ := proof.p1.foldl Replay.Duplex.observe (rel.commitmentSeed.foldl Replay.Duplex.observe r₅)
  have L₁₄ : n ≤ m₁₄ := L₉.trans (l₁₀.trans (l₁₁.trans (l₁₂.trans (l₁₃.trans l₁₄))))
  refine Ensures.pre (P' := P ∧ Replay.combine proof.partials[0]! (gkey.extract 0 foldVars) = lin.keyValue ∧
    Replay.combine proof.partials[1]! (grun.extract 0 foldVars) = grv ∧ IsExt c m₁₄ mu gmu ∧
    IsExts c m₁₄ #[a₀, a₁, a₂] #[ga₀, ga₁, ga₂] ∧ IsExts c m₁₄ keyPoint gkey ∧ IsExts c m₁₄ runPoint' grun ∧
    IsDuplex c m₁₄ d₁₄ r₇) ?_ fun h => ?_
  rotate_left
  · obtain ⟨⟨⟨⟨⟨⟨hP, hmu, halpha, hkp, hrp, -, -⟩, -, -⟩, ⟨-, keyCheck⟩, -⟩, -, -⟩, ⟨-, runCheck⟩, -⟩, hd₁₄, -⟩ := h
    have M : m₉ ≤ m₁₄ := l₁₀.trans (l₁₁.trans (l₁₂.trans (l₁₃.trans l₁₄)))
    exact ⟨hP, keyCheck, runCheck, hmu.mono M, halpha.mono M, hkp.mono M, hrp.mono M, hd₁₄⟩
  refine Ensures.tbind (Sim.sampleBitsArray (rel.setupVars + 1 - foldVars) small hp rel.queries
    (t := { duplex := r₇, draws := (((t.draws.push (.ext gmu)).push (.ext ga₀)).push (.ext ga₁)).push (.ext ga₂) })
    fun h => h.2.2.2.2.2.2.2) fun indexBits d₁₅ m₁₅ l₁₅ => ?_
  refine Ensures.exists_and (#[] : Array Gl) fun ws => ?_
  refine Ensures.exists_and t fun t₈ => ?_
  have digits : ∀ {m : ℕ} {q : ℕ}, q < ws.size → ∀ i < rel.setupVars + 1 - foldVars,
      IsWord c m (indexBits[q]!)[i]! (Gl.ofNat ((ws[q]!).toNat / 2 ^ i % 2)) →
      IsWord c m (indexBits[q]!)[i]!
        (Gl.ofNat ((ws.map fun word => word.toNat % 2 ^ (rel.setupVars + 1 - foldVars))[q]! / 2 ^ i % 2)) :=
      fun hq i hi w => by
    rw [getElem!_pos (ws.map fun word => word.toNat % 2 ^ (rel.setupVars + 1 - foldVars)) _ (by simpa using hq),
      Array.getElem_map, ← getElem!_pos ws _ hq, digit_mod _ _ _ hi]
    exact w
  refine Ensures.tbind (Ensures.lift Sim.extOne.ensures) fun one d₁₆ m₁₆ l₁₆ => ?_
  refine Ensures.tbind (Sim.runWeightsLoop (fun h => h.2.1.1)
    fun h => h.1.1.2.2.2.1.mono (l₁₅.trans l₁₆)).ensures fun runWeights d₁₇ m₁₇ l₁₇ => ?_
  refine Ensures.tbind (Sim.rootConsts rel.key.root).ensures fun rootV d₁₈ m₁₈ l₁₈ => ?_
  have L₁₈ : n ≤ m₁₈ := L₁₄.trans (l₁₅.trans (l₁₆.trans (l₁₇.trans l₁₈)))
  refine Ensures.tbind (queriedLoop (groot := rel.key.root)
    (queries := ws.map fun word => word.toNat % 2 ^ (rel.setupVars + 1 - foldVars)) (gleaves := proof.leaves)
    (gw₀ := (Array.range rel.kappa).map lin.mixing.commitment) (gw₁ := (Array.range runOracles).map gmu.pow)
    (galpha := #[ga₀, ga₁, ga₂]) hp (fun h => by exact h.2.1.1) root
    (fun h => by
      refine ⟨by rw [h.1.1.1.2.2.1, Array.size_map], fun q hq => ⟨(h.1.1.1.2.2.2.1 q hq).1, fun i hi => ?_⟩⟩
      exact digits (by rw [← h.1.1.1.2.2.1]; exact hq) i hi (((h.1.1.1.2.2.2.1 q hq).2 i hi).mono
        (l₁₆.trans (l₁₇.trans l₁₈))))
    (fun h => by
      exact ⟨(hleaves h.1.1.1.1.1).1, fun q hq => ⟨(((hleaves h.1.1.1.1.1).2 q hq).1).mono L₁₈,
        ((hleaves h.1.1.1.1.1).2 q hq).2.1, ((hleaves h.1.1.1.1.1).2 q hq).2.2.1,
        fun k hk => (((hleaves h.1.1.1.1.1).2 q hq).2.2.2 k hk).mono L₁₈⟩⟩)
    (fun h => by exact (hrows h.1.1.1.1.1).mono L₁₈) (fun h => by exact h.1.2.1.1.mono l₁₈)
    (fun h => by exact h.1.1.1.1.2.2.2.2.1.mono (l₁₅.trans (l₁₆.trans (l₁₇.trans l₁₈)))) rfl)
    fun queried d₁₉ m₁₉ l₁₉ => ?_
  refine Ensures.pure fun h => ?_
  obtain ⟨⟨⟨⟨⟨⟨hP, keyCheck, runCheck, -, halpha, hkp, hrp, -⟩, words, isize, ibits, hd₁₅⟩, -, e₁₆⟩, ⟨-, e₁₇⟩, -⟩,
    ⟨-, e₁₈⟩, -⟩, gq, qok, hq, e₁₉⟩ := h
  dsimp only at isize ibits hd₁₅ e₁₆ e₁₇ e₁₈ e₁₉ hq ⊢
  have keyOk : (Replay.combine proof.partials[0]! (gkey.extract 0 foldVars) == lin.keyValue) = true := by
    rw [keyCheck]
    exact Ext.beq_self _
  have runOk : (Replay.combine proof.partials[1]! (grun.extract 0 foldVars) == grv) = true := by
    rw [runCheck]
    exact Ext.beq_self _
  have M : m₁₄ ≤ m₁₉ := l₁₅.trans (l₁₆.trans (l₁₇.trans (l₁₈.trans l₁₉)))
  refine ⟨{ mu := gmu, alpha := #[ga₀, ga₁, ga₂], words := ws,
            queries := ws.map fun word => word.toNat % 2 ^ (rel.setupVars + 1 - foldVars), queried := gq },
    t₈, ?_, ⟨halpha.mono M, hkp.mono M, hrp.mono M, by simp [isize], fun q hq' => ⟨(ibits q hq').1, fun i hi =>
      digits (by rw [← isize]; exact hq') i hi (((ibits q hq').2 i hi).mono (l₁₆.trans (l₁₇.trans (l₁₈.trans l₁₉))))⟩,
      hq⟩, ?_⟩
  · unfold Replay.honestStage
    exact runs_bind (runs_sampleExt t) (runs_bind (runs_observeExts _ _) (runs_bind (runs_sampleExt _)
      (runs_bind (runs_sampleExt _) (runs_bind (runs_sampleExt _) (runs_bind (runs_ensure keyOk _ _)
      (runs_bind (runs_ensure runOk _ _) (runs_bind (runs_observeAll _ _) (runs_bind (runs_observeAll _ _)
      (runs_bind words (runs_bind (runs_liftExcept qok _) (runs_pure _ _)))))))))))
  · rw [e₁₉, e₁₈, e₁₇, e₁₆]
    exact hd₁₅.mono (l₁₆.trans (l₁₇.trans (l₁₈.trans l₁₉)))

end CompressionLean.Circuit
