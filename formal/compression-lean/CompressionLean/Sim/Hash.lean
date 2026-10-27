import CompressionLean.Sim.Algebra
import CompressionLean.Ports.Replay.Transcript
import CompressionLean.Lib.Array

/-!
Owns the simulation of the duplex challenger of `Hash.lean` against the
replay's (`Ports/Replay/Transcript.lean`): `observe`, `sample`, `sample_ext` and
`sample_bits`. `Merkle.lean` owns the leaf sponge, node compression and
Merkle paths.

- `IsWords`: an array of handles is an array of words, entry by entry.
- `IsDuplex`: the builder's duplex is the replay's, lane by lane.
- `bits_digits`: a decomposition that passes the 64-bit check is the binary
  one. So once its check holds, `sampleBits` gives the sample's digits
  (`Duplex.sampleBits_ensures`).
-/

namespace CompressionLean.Sim

open CompressionLean.Circuit
open CompressionLean.Lowering
open CompressionLean.Lib
open CompressionLean.Replay (Word Ext)
open NightstreamFPrime.Spec (goldilocksModulus)

variable {F : Type} [CommRing F]

/-! ## Arrays -/

/-- `hs` and `gs` have one size, and each handle is its word. -/
def IsWords (c : Ctx F) (n : ℕ) (hs : Array Var) (gs : Array Word) : Prop :=
  hs.size = gs.size ∧ ∀ i < hs.size, IsWord c n hs[i]! gs[i]!

namespace IsWords

variable {c : Ctx F} {n : ℕ} {hs : Array Var} {gs : Array Word}

theorem mono (h : IsWords c n hs gs) {m : ℕ} (le : n ≤ m := by omega) : IsWords c m hs gs :=
  ⟨h.1, fun i hi => (h.2 i hi).mono le⟩

theorem get (h : IsWords c n hs gs) {i : ℕ} (hi : i < gs.size) : IsWord c n hs[i]! gs[i]! :=
  h.2 i (h.1 ▸ hi)

theorem empty : IsWords c n #[] #[] := ⟨rfl, fun _ h => absurd h (by simp)⟩

theorem set (h : IsWords c n hs gs) {x : Var} {g : Word} (hx : IsWord c n x g) (j : ℕ) :
    IsWords c n (hs.set! j x) (gs.set! j g) := by
  refine ⟨by simp [h.1], fun i hi => ?_⟩
  rw [Array.size_set!] at hi
  rw [getElem!_set! _ _ _ _ hi, getElem!_set! _ _ _ _ (h.1 ▸ hi)]
  split_ifs
  · exact hx
  · exact h.2 i hi

theorem push (h : IsWords c n hs gs) {x : Var} {g : Word} (hx : IsWord c n x g) :
    IsWords c n (hs.push x) (gs.push g) := by
  refine ⟨by simp [h.1], fun i hi => ?_⟩
  rw [Array.size_push] at hi
  rw [getElem!_pos _ _ (by simp; omega), getElem!_pos _ _ (by simp [← h.1]; omega), Array.getElem_push,
    Array.getElem_push]
  by_cases lt : i < hs.size
  · rw [dif_pos lt, dif_pos (h.1 ▸ lt), ← getElem!_pos, ← getElem!_pos]
    exact h.2 i lt
  · rw [dif_neg lt, dif_neg (h.1 ▸ lt)]
    exact hx

theorem pop (h : IsWords c n hs gs) : IsWords c n hs.pop gs.pop := by
  refine ⟨by simp [h.1], fun i hi => ?_⟩
  rw [Array.size_pop] at hi
  rw [getElem!_pos _ _ (by simp; omega), getElem!_pos _ _ (by simp [← h.1]; omega), Array.getElem_pop,
    Array.getElem_pop, ← getElem!_pos, ← getElem!_pos]
  exact h.2 i (by omega)

theorem back (h : IsWords c n hs gs) (ne : 0 < hs.size) : IsWord c n hs.back! gs.back! := by
  have e := h.1
  rw [Array.back!, Array.back!, h.1]
  exact h.get (by omega)

theorem extract (h : IsWords c n hs gs) (a b : ℕ) : IsWords c n (hs.extract a b) (gs.extract a b) := by
  refine ⟨by simp [h.1], fun i hi => ?_⟩
  rw [Array.size_extract] at hi
  rw [getElem!_pos _ _ (by simp; omega), getElem!_pos _ _ (by simp [← h.1]; omega), Array.getElem_extract,
    Array.getElem_extract, ← getElem!_pos, ← getElem!_pos]
  exact h.2 _ (by omega)

theorem replicate {z : Var} {g : Word} (hz : IsWord c n z g) (k : ℕ) :
    IsWords c n (Array.replicate k z) (Array.replicate k g) := by
  refine ⟨by simp, fun i hi => ?_⟩
  rw [Array.size_replicate] at hi
  rw [getElem!_pos _ _ (by simpa using hi), getElem!_pos _ _ (by simpa using hi), Array.getElem_replicate,
    Array.getElem_replicate]
  exact hz

theorem ofFn {k : ℕ} {f : Fin k → Var} {gs : Array Word} (size : gs.size = k) (hf : ∀ i : Fin k, IsWord c n (f i) gs[i]!) :
    IsWords c n (Array.ofFn f) gs := by
  refine ⟨by simp [size], fun i hi => ?_⟩
  rw [Array.size_ofFn] at hi
  rw [getElem!_pos _ _ (by simpa using hi), Array.getElem_ofFn]
  exact hf ⟨i, hi⟩

/-- Overwrite lanes `lane i` for `i < k` with `f i` and `g i`. -/
theorem setRange (h : IsWords c n hs gs) (k : ℕ) (lane : ℕ → ℕ) {f : ℕ → Var} {g : ℕ → Word}
    (hf : ∀ i < k, IsWord c n (f i) (g i)) :
    IsWords c n ((List.range k).foldl (fun s i => s.set! (lane i) (f i)) hs)
      ((List.range k).foldl (fun s i => s.set! (lane i) (g i)) gs) := by
  induction k with
  | zero => exact h
  | succ k ih =>
      rw [List.range_succ, List.foldl_append, List.foldl_append]
      exact (ih fun i hi => hf i (by omega)).set (hf k (by omega)) _

/-- Entry-wise words, as `mapM` lemmas state them. -/
theorem of_forall₂ (h : List.Forall₂ (fun g v => IsWord c n v g) gs.toList hs.toList) : IsWords c n hs gs := by
  have size := h.length_eq
  simp only [Array.length_toList] at size
  refine ⟨size.symm, fun i hi => ?_⟩
  have := List.forall₂_iff_get.mp h |>.2 i (by simp [size]; omega) (by simpa using hi)
  rw [getElem!_pos hs i hi, getElem!_pos gs i (by omega)]
  simpa using this

end IsWords

/-- The `i`-th pair of `l.zipIdx` is `(l[i], i)`. -/
theorem foldl_zipIdx {α β : Type} [Inhabited α] (l : List α) (f : β → α → ℕ → β) (s : β) :
    l.zipIdx.foldl (fun s x => f s x.1 x.2) s = (List.range l.length).foldl (fun s i => f s l[i]! i) s := by
  induction l using List.reverseRecOn generalizing s with
  | nil => rfl
  | append_singleton l a ih =>
      rw [List.zipIdx_append, List.foldl_append, ih, List.length_append, List.length_singleton, List.range_succ,
        List.foldl_append]
      simp only [List.zipIdx_singleton, Nat.zero_add, List.foldl_cons, List.foldl_nil]
      congr 1
      · apply List.foldl_ext
        intro t i hi
        rw [List.getElem!_eq_getElem?_getD, List.getElem!_eq_getElem?_getD,
          List.getElem?_append_left (List.mem_range.mp hi)]
      · rw [List.getElem!_eq_getElem?_getD, List.getElem?_append_right (le_refl _), Nat.sub_self]
        rfl

theorem foldl_zipIdx_set {α : Type} [Inhabited α] (l : List α) (s : Array α) :
    l.zipIdx.foldl (fun s x => s.set! x.2 x.1) s = (List.range l.length).foldl (fun s i => s.set! i l[i]!) s :=
  foldl_zipIdx l (fun s v i => s.set! i v) s

theorem size_setRange {α : Type} (k : ℕ) (lane : ℕ → ℕ) (f : ℕ → α) (s : Array α) :
    ((List.range k).foldl (fun s i => s.set! (lane i) (f i)) s).size = s.size := by
  induction k with
  | zero => rfl
  | succ k ih => rw [List.range_succ, List.foldl_append, List.foldl_cons, List.foldl_nil, Array.size_set!, ih]

theorem range_foldl {α : Type} (a b : ℕ) (f : ℕ → α → α) (s : α) :
    (List.range' a (([a:b] : Std.Legacy.Range).size)).foldl (fun s i => f i s) s =
      (List.range (b - a)).foldl (fun s i => f (a + i) s) s := by
  rw [List.range'_eq_map_range, List.foldl_map]
  congr 2
  simp [Std.Legacy.Range.size]

/-- A loop whose body is pure is a fold. -/
theorem forIn_pure {m : Type → Type} [Monad m] [LawfulMonad m] {α β : Type} {l : List α} {init : β}
    {body : α → β → m (ForInStep β)} (f : α → β → β) (h : ∀ x s, body x s = pure (.yield (f x s))) :
    forIn l init body = pure (l.foldl (fun s x => f x s) init) := by
  rw [show body = fun x s => pure (.yield (f x s)) from funext fun x => funext fun s => h x s]
  exact List.forIn_pure_yield_eq_foldl f init

/-! ## The duplex -/

/-- The builder's duplex is the replay's, lane by lane; its state has 16
lanes. -/
structure IsDuplex (c : Ctx F) (n : ℕ) (d : Duplex) (r : Replay.Duplex) : Prop where
  state : IsWords c n d.state r.state
  input : IsWords c n d.input r.input
  output : IsWords c n d.output r.output
  width : r.state.size = 16

theorem IsDuplex.mono {c : Ctx F} {n m : ℕ} {d : Duplex} {r : Replay.Duplex} (h : IsDuplex c n d r)
    (le : n ≤ m := by omega) : IsDuplex c m d r :=
  ⟨h.state.mono le, h.input.mono le, h.output.mono le, h.width⟩

theorem permute_size (s : Array Word) : (Replay.permute s).size = 16 := rfl

/- Unifying two terms that contain the permutation must never evaluate it. -/
attribute [local irreducible] Replay.permute

/-- The duplex after a permutation, from the replay's last state `t`. -/
theorem IsDuplex.permuted {c : Ctx F} {m : ℕ} (t : Array Word) (outs : Fin 16 → Var)
    (h : ∀ k : Fin 16, IsWord c m (outs k) (Replay.permute t)[k]!) :
    IsDuplex c m ({ state := Array.ofFn outs, input := #[], output := (Array.ofFn outs).extract 0 rate } : Duplex)
      ⟨Replay.permute t, #[], (Replay.permute t).extract 0 Replay.rate⟩ := by
  have wo := IsWords.ofFn (permute_size _) h
  refine ⟨wo, IsWords.empty, ?_, permute_size _⟩
  dsimp only
  exact wo.extract 0 rate

theorem duplexing_output_size (r : Replay.Duplex) : r.duplexing.output.size = Replay.rate := by
  unfold Replay.Duplex.duplexing
  simp [permute_size, Replay.rate]

/-- Pop the last output lane of matching duplexes. -/
theorem IsDuplex.pop {c : Ctx F} {m : ℕ} {d : Duplex} {r : Replay.Duplex} (h : IsDuplex c m d r)
    (ne : 0 < r.output.size) :
    IsWord c m d.output.back! r.output.back! ∧
      IsDuplex c m { d with output := d.output.pop } { r with output := r.output.pop } :=
  ⟨h.output.back (h.output.1 ▸ ne), ⟨h.state, h.input, h.output.pop, h.width⟩⟩

/-- Matching duplexes take the same branch before a sample. -/
theorem IsDuplex.cond {c : Ctx F} {m : ℕ} {d : Duplex} {r : Replay.Duplex} (h : IsDuplex c m d r) :
    (!d.input.isEmpty || d.output.isEmpty) = (!r.input.isEmpty || r.output.isEmpty) := by
  unfold Array.isEmpty
  rw [h.input.1, h.output.1]

/-- The sum of `2^i·β i` below `k`, for bits `β`, is below `2^k` and has
digit `j` equal to `β j`. -/
theorem binary_sum (β : ℕ → ℕ) : ∀ k, (∀ i < k, β i ≤ 1) →
    ((List.range k).map fun i => 2 ^ i * β i).sum < 2 ^ k ∧
      ∀ j < k, ((List.range k).map fun i => 2 ^ i * β i).sum / 2 ^ j % 2 = β j
  | 0, _ => ⟨by simp, fun j hj => absurd hj (by omega)⟩
  | k + 1, hβ => by
      obtain ⟨lt, digits⟩ := binary_sum β k fun i hi => hβ i (by omega)
      rw [List.range_succ, List.map_append, List.sum_append, List.map_singleton, List.sum_singleton]
      generalize ((List.range k).map fun i => 2 ^ i * β i).sum = S at lt digits
      have hk := hβ k (by omega)
      refine ⟨by rw [pow_succ]; nlinarith, fun j hj => ?_⟩
      rcases Nat.lt_succ_iff_lt_or_eq.mp hj with lo | rfl
      · have e : 2 ^ k * β k = 2 ^ j * (2 * (2 ^ (k - j - 1) * β k)) := by
          rw [← mul_assoc, ← mul_assoc, ← pow_succ, ← pow_add]
          congr 2
          omega
        rw [e, Nat.add_mul_div_left _ _ (by positivity), Nat.add_mul_mod_self_left]
        exact digits j lo
      · rw [Nat.add_mul_div_left _ _ (by positivity), Nat.div_eq_of_lt lt, zero_add]
        omega

/-- **A canonical decomposition is the binary one.** If `b` passes the 64-bit
check for the word `g`, bit `i` is digit `i` of `g`. -/
theorem bits_digits [CharP F goldilocksModulus] {g : Word} {b : ℕ → F}
    (h : Bits (Word.cast g : F) 64 b) : ∀ i < 64, b i = Word.cast (Word.ofNat (g.toNat / 2 ^ i % 2)) := by
  classical
  have one_ne : (1 : F) ≠ 0 := by
    intro e
    have := (CharP.cast_eq_zero_iff F goldilocksModulus 1).mp (by simpa using e)
    norm_num [goldilocksModulus] at this
  obtain ⟨β, hβ⟩ : ∃ β : ℕ → ℕ, ∀ i < 64, b i = (β i : F) ∧ β i ≤ 1 :=
    ⟨fun i => if b i = 1 then 1 else 0, fun i hi => by
      rcases h.boolean i hi with z | o
      · have : b i ≠ 1 := z ▸ one_ne.symm
        exact ⟨by dsimp only; rw [if_neg this, Nat.cast_zero, z], by dsimp only; rw [if_neg this]; omega⟩
      · simp [o]⟩
  obtain ⟨lt, digits⟩ := binary_sum β 64 fun i hi => (hβ i hi).2
  generalize hN : ((List.range 64).map fun i => 2 ^ i * β i).sum = N at lt digits
  have cast : (Word.cast g : F) = (N : F) := by
    rw [h.sum, ← hN, Nat.cast_list_sum, List.map_map]
    refine congrArg List.sum (List.map_congr_left fun i hi => ?_)
    simp [(hβ i (List.mem_range.mp hi)).1]
  have small : N < goldilocksModulus := by
    by_contra big
    have high : N / 2 ^ 32 = 2 ^ 32 - 1 := by
      norm_num [goldilocksModulus] at big lt ⊢
      omega
    have table : ∀ k < 32, (2 ^ 32 - 1) / 2 ^ k % 2 = 1 := by decide
    have ones : ∀ j, 32 ≤ j → j < 64 → b j = 1 := fun j lo hj => by
      obtain ⟨k, rfl⟩ : ∃ k, j = 32 + k := ⟨j - 32, by omega⟩
      rw [(hβ _ hj).1, ← digits _ hj, pow_add, ← Nat.div_div_eq_div_mul, high, table k (by omega), Nat.cast_one]
    have zeros := h.canonical rfl ones
    have low : N % 2 ^ 32 = 0 := by
      apply Nat.eq_of_testBit_eq
      intro i
      rw [Nat.testBit_mod_two_pow, Nat.zero_testBit]
      by_cases hi : i < 32
      · rw [decide_eq_true hi, Bool.true_and, Nat.testBit_eq_decide_div_mod_eq, digits i (by omega)]
        have bi := (hβ i (by omega)).1
        rw [zeros i hi] at bi
        by_cases one : β i = 1
        · rw [one, Nat.cast_one] at bi
          exact absurd bi.symm one_ne
        · simp [one]
      · simp [hi]
    have : N = 2 ^ 32 * (2 ^ 32 - 1) := by
      rw [← Nat.div_add_mod N (2 ^ 32), high, low, add_zero]
    norm_num [goldilocksModulus, this] at big
  have eq : N = g.toNat := by
    rw [Word.cast, eq_comm] at cast
    have := (CharP.natCast_eq_natCast F goldilocksModulus).mp cast
    exact Nat.ModEq.eq_of_lt_of_lt this small g.canonical
  intro i hi
  rw [(hβ i hi).1, word_ofNat, ← eq, digits i hi]

/-! ## Programs -/

instance Duplex.new_appends : Appends Duplex.new := by unfold Duplex.new; appends
instance Duplex.duplexing_appends (d : Duplex) : Appends (Duplex.duplexing.run d) := by
  unfold Duplex.duplexing; appends
instance Duplex.observe_appends (v : Var) (d : Duplex) : Appends ((Duplex.observe v).run d) := by
  unfold Duplex.observe; appends
instance Duplex.observeSlice_appends (vs : Array Var) (d : Duplex) : Appends ((Duplex.observeSlice vs).run d) := by
  unfold Duplex.observeSlice Array.forM; rw [← Array.foldlM_toList]; appends
instance Duplex.observeExt_appends (v : ExtVar) (d : Duplex) : Appends ((Duplex.observeExt v).run d) := by
  unfold Duplex.observeExt; appends
instance Duplex.sample_appends (d : Duplex) : Appends (Duplex.sample.run d) := by unfold Duplex.sample; appends
instance Duplex.sampleExt_appends (d : Duplex) : Appends (Duplex.sampleExt.run d) := by
  unfold Duplex.sampleExt; appends
instance Duplex.sampleBits_appends (bits : ℕ) (d : Duplex) : Appends ((Duplex.sampleBits bits).run d) := by
  unfold Duplex.sampleBits; appends

theorem Duplex.new_ensures {c : Ctx F} {n : ℕ} :
    Ensures c n Duplex.new fun d m => IsDuplex c m d Replay.Duplex.new :=
  (const_ensures 0).bind fun _ _ _ hz => Ensures.pure ⟨IsWords.replicate hz 16, IsWords.empty, IsWords.empty, rfl⟩

section Duplex

variable [CharP F goldilocksModulus] {c : Ctx F} {n : ℕ}

/-- `duplexing`: overwrite the absorbed lanes; after input, zero the rest of
the rate and add the length tag; permute; refill the output. -/
theorem Duplex.duplexing_ensures {d : Duplex} {r : Replay.Duplex} (hp : Permutes c.perm) (hd : IsDuplex c n d r) :
    Ensures c n (Duplex.duplexing.run d) fun res m => IsDuplex c m res.2 r.duplexing := by
  unfold Duplex.duplexing
  rw [StateT.run_bind, StateT.run_get, pure_bind]
  dsimp only
  rw [← Array.forIn_toList, Array.toList_zipIdx, forIn_pure (fun x s => s.set! x.2 x.1) (fun x s => by cases x; rfl),
    StateT.run_bind, StateT.run_pure, pure_bind]
  dsimp only
  rw [foldl_zipIdx_set]
  simp only [Array.getElem!_toList, Array.length_toList]
  generalize hs₁ : (List.range d.input.size).foldl (fun s i => s.set! i d.input[i]!) d.state = s₁
  let t₁ := (List.range d.input.size).foldl (fun s i => s.set! i r.input[i]!) r.state
  have ws₁ : IsWords c n s₁ t₁ := by
    rw [← hs₁]
    exact hd.state.setRange _ (fun i => i) fun i hi => hd.input.get (by rw [← hd.input.1]; exact hi)
  have wt₁ : t₁.size = 16 := by simp only [t₁, size_setRange, hd.width]
  by_cases hpos : d.input.size > 0
  · rw [if_pos hpos]
    refine Ensures.liftBind (const_ensures 0) fun z m₁ _ hz => ?_
    rw [Std.Legacy.Range.forIn_eq_forIn_range', List.forIn_pure_yield_eq_foldl, StateT.run_bind, StateT.run_pure,
      pure_bind]
    dsimp only
    rw [range_foldl d.input.size rate (fun i (s : Array Var) => s.set! i z)]
    generalize hs₂ : (List.range (rate - d.input.size)).foldl (fun s i => s.set! (d.input.size + i) z) s₁ = s₂
    let t₂ := (List.range (rate - d.input.size)).foldl (fun s i => s.set! (d.input.size + i) (0 : Word)) t₁
    let t₃ := t₂.set! rate (t₂[rate]! + Word.ofNat d.input.size)
    have wt₂ : t₂.size = 16 := by simp only [t₂, size_setRange, wt₁]
    have ws₂ : IsWords c m₁ s₂ t₂ := by
      rw [← hs₂]
      exact ws₁.mono.setRange _ (fun i => d.input.size + i) (f := fun _ => z) (g := fun _ => 0) fun _ _ => hz
    refine Ensures.liftBind (const_ensures (Word.ofNat d.input.size)) fun t _ _ ht => ?_
    refine Ensures.liftBind (add_ensures (ws₂.mono.get (by rw [wt₂]; decide)) ht) fun sum _ _ hsum => ?_
    refine Ensures.liftBind (permute_ensures (gs := t₃) (state := fun lane => (s₂.set! rate sum)[lane]!) hp
      fun k => (ws₂.mono.set hsum rate).get (by rw [Array.size_set!, wt₂]; exact k.isLt)) fun outs _ _ houts => ?_
    rw [StateT.run_set]
    refine Ensures.pure ?_
    have hr : r.duplexing = ⟨Replay.permute t₃, #[], (Replay.permute t₃).extract 0 Replay.rate⟩ := by
      unfold Replay.Duplex.duplexing
      simp only [← hd.input.1, if_pos hpos]
      rfl
    rw [hr]
    exact IsDuplex.permuted t₃ outs houts
  · rw [if_neg hpos]
    refine Ensures.liftBind (permute_ensures (gs := t₁) (state := fun lane => s₁[lane]!) hp
      fun k => ws₁.get (by rw [wt₁]; exact k.isLt)) fun outs _ _ houts => ?_
    rw [StateT.run_set]
    refine Ensures.pure ?_
    have hr : r.duplexing = ⟨Replay.permute t₁, #[], (Replay.permute t₁).extract 0 Replay.rate⟩ := by
      unfold Replay.Duplex.duplexing
      simp only [← hd.input.1, if_neg hpos]
      rfl
    rw [hr]
    exact IsDuplex.permuted t₁ outs houts

/-- `observe`: push the word, clear the output, and duplex at a full rate. -/
theorem Duplex.observe_ensures {d : Duplex} {r : Replay.Duplex} {v : Var} {g : Word} (hp : Permutes c.perm)
    (hd : IsDuplex c n d r) (hv : IsWord c n v g) :
    Ensures c n ((Duplex.observe v).run d) fun res m => IsDuplex c m res.2 (r.observe g) := by
  unfold Duplex.observe
  rw [StateT.run_bind, StateT.run_modify, pure_bind, StateT.run_bind, StateT.run_get, pure_bind]
  dsimp only
  have hd' : IsDuplex c n ⟨d.state, d.input.push v, #[]⟩ ⟨r.state, r.input.push g, #[]⟩ :=
    ⟨hd.state, hd.input.push hv, IsWords.empty, hd.width⟩
  by_cases full : (d.input.push v).size = rate
  · rw [if_pos full]
    refine (Duplex.duplexing_ensures hp hd').post fun _ _ q => ?_
    have e : (r.input.push g).size = Replay.rate := by rw [← hd'.input.1]; exact full
    have hr : r.observe g = Replay.Duplex.duplexing ⟨r.state, r.input.push g, #[]⟩ := by
      simp only [Replay.Duplex.observe, e, beq_self_eq_true, if_true]
    rw [hr]
    exact q
  · rw [if_neg full, StateT.run_pure]
    refine Ensures.pure ?_
    have e : ¬(r.input.push g).size = Replay.rate := by rw [← hd'.input.1]; exact full
    have hr : r.observe g = ⟨r.state, r.input.push g, #[]⟩ := by
      simp only [Replay.Duplex.observe, beq_iff_eq, e, if_false]
    rw [hr]
    exact hd'

/-- `observe` of each word of a list, in order. -/
private theorem observeList_ensures (hp : Permutes c.perm) :
    ∀ (l : List Var) (gl : List Word) {n : ℕ} {d : Duplex} {r : Replay.Duplex}, IsDuplex c n d r →
      l.length = gl.length → (∀ i < l.length, IsWord c n l[i]! gl[i]!) →
      Ensures c n ((l.foldlM (fun _ v => Duplex.observe v) ()).run d)
        fun res m => IsDuplex c m res.2 (gl.foldl Replay.Duplex.observe r)
  | [], gl, _, _, _, hd, hlen, _ => by
      rw [List.foldlM_nil, StateT.run_pure, List.eq_nil_of_length_eq_zero hlen.symm]
      exact Ensures.pure hd
  | _ :: _, [], _, _, _, _, hlen, _ => absurd hlen (by simp)
  | v :: l, g :: gl, _, _, _, hd, hlen, hl => by
      rw [List.foldlM_cons, StateT.run_bind]
      refine (Duplex.observe_ensures hp hd (g := g) (by simpa using hl 0 (by simp))).bind fun res _ _ hres => ?_
      exact observeList_ensures hp l gl hres (by simpa using hlen) fun i hi => by
        simpa using (hl (i + 1) (by simp; omega)).mono

/-- `observe_slice`: the replay's `foldl observe`. -/
theorem Duplex.observeSlice_ensures {d : Duplex} {r : Replay.Duplex} {vs : Array Var} {gs : Array Word}
    (hp : Permutes c.perm) (hd : IsDuplex c n d r) (hv : IsWords c n vs gs) :
    Ensures c n ((Duplex.observeSlice vs).run d)
      fun res m => IsDuplex c m res.2 (gs.foldl Replay.Duplex.observe r) := by
  unfold Duplex.observeSlice Array.forM
  rw [← Array.foldlM_toList, ← Array.foldl_toList]
  exact observeList_ensures hp vs.toList gs.toList hd (by simpa using hv.1) fun i hi => by
    rw [Array.getElem!_toList, Array.getElem!_toList]
    exact hv.2 i (by simpa using hi)

/-- `observe_ext`: the three coordinates in order. -/
theorem Duplex.observeExt_ensures {d : Duplex} {r : Replay.Duplex} {a : ExtVar} {e : Ext} (hp : Permutes c.perm)
    (hd : IsDuplex c n d r) (ha : IsExt c n a e) :
    Ensures c n ((Duplex.observeExt a).run d)
      fun res m => IsDuplex c m res.2 (((r.observe e.c0).observe e.c1).observe e.c2) := by
  unfold Duplex.observeExt
  refine (Duplex.observe_ensures hp hd (ha 0)).tbind fun _ _ _ _ h₁ => ?_
  refine (Duplex.observe_ensures hp h₁ (ha 1).mono).tbind fun _ _ _ _ h₂ => ?_
  exact Duplex.observe_ensures hp h₂ (ha 2).mono

/-- `sample`: duplex when there is input or no output, then pop the last
output lane. -/
theorem Duplex.sample_ensures {d : Duplex} {r : Replay.Duplex} (hp : Permutes c.perm) (hd : IsDuplex c n d r) :
    Ensures c n (Duplex.sample.run d) fun res m => IsWord c m res.1 r.sample.1 ∧ IsDuplex c m res.2 r.sample.2 := by
  unfold Duplex.sample
  rw [StateT.run_bind, StateT.run_get, pure_bind]
  dsimp only
  split
  · rename_i cond
    rw [StateT.run_bind]
    refine (Duplex.duplexing_ensures hp hd).bind fun res _ _ hres => ?_
    rw [StateT.run_bind, StateT.run_get, pure_bind, StateT.run_bind, StateT.run_set, pure_bind, StateT.run_pure]
    refine Ensures.pure ?_
    have e : (!r.input.isEmpty || r.output.isEmpty) = true := hd.cond ▸ cond
    simp only [Replay.Duplex.sample, e, if_true]
    exact hres.pop (by rw [duplexing_output_size]; decide)
  · rename_i cond
    rw [StateT.run_bind, StateT.run_get, pure_bind, StateT.run_bind, StateT.run_set, pure_bind, StateT.run_pure]
    refine Ensures.pure ?_
    have e : ¬(!r.input.isEmpty || r.output.isEmpty) = true := hd.cond ▸ cond
    simp only [Replay.Duplex.sample, e]
    refine hd.pop ?_
    simp only [Bool.or_eq_true, Bool.not_eq_true', Array.isEmpty_iff_size_eq_zero, not_or] at e
    omega

/-- `sample_ext`: coordinate `i` is the `i`-th sample. -/
theorem Duplex.sampleExt_ensures {d : Duplex} {r : Replay.Duplex} (hp : Permutes c.perm) (hd : IsDuplex c n d r) :
    Ensures c n (Duplex.sampleExt.run d)
      fun res m => IsExt c m res.1 ⟨r.sample.1, r.sample.2.sample.1, r.sample.2.sample.2.sample.1⟩ ∧
        IsDuplex c m res.2 r.sample.2.sample.2.sample.2 := by
  unfold Duplex.sampleExt
  refine (Duplex.sample_ensures hp hd).tbind fun _ _ _ _ hx => ?_
  refine (Duplex.sample_ensures hp hx.2).tbind fun _ _ _ _ hy => ?_
  refine (Duplex.sample_ensures hp hy.2).tbind fun _ _ _ _ hz => ?_
  rw [StateT.run_pure]
  refine Ensures.pure ⟨fun k => ?_, hz.2⟩
  fin_cases k
  · exact hx.1.mono
  · exact hy.1.mono
  · exact hz.1

/-- `sample_bits`: the low `bits` bits of the canonical sample. Once the
decomposition's check holds, bit `i` is digit `i` of the sample. -/
theorem Duplex.sampleBits_ensures {d : Duplex} {r : Replay.Duplex} (bits : ℕ) (small : bits ≤ 64)
    (hp : Permutes c.perm) (hd : IsDuplex c n d r) :
    Ensures c n ((Duplex.sampleBits bits).run d)
      fun res m => res.1.size = bits ∧ (∀ i < bits, IsWord c m res.1[i]! (Word.ofNat (r.sample.1.toNat / 2 ^ i % 2))) ∧
        IsDuplex c m res.2 r.sample.2 := by
  unfold Duplex.sampleBits
  refine (Duplex.sample_ensures hp hd).tbind fun w e _ _ hw => ?_
  refine Ensures.liftBind (bits_ensures 64 hw.1) fun res _ _ hres => ?_
  obtain ⟨cells, check⟩ := hres
  rw [StateT.run_pure]
  refine Ensures.pure ⟨by simp [small], fun i hi => ?_, hw.2.mono⟩
  have get : ((Array.ofFn res).extract 0 bits)[i]! = res ⟨i, by omega⟩ := by
    rw [getElem!_pos _ _ (by simp; omega)]
    simp
  rw [get]
  obtain ⟨at_, lt_, val⟩ := cells ⟨i, by omega⟩
  refine ⟨lt_, ?_⟩
  rw [val, at_]
  exact bits_digits check i (by omega)

end Duplex

end CompressionLean.Sim
