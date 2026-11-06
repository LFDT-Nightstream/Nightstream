import CompressionLean.Sim.HashQueue
import CompressionLean.Ports.Replay.Transcript
import CompressionLean.Lib.Array

namespace CompressionLean.Sim

open CompressionLean.Circuit
open CompressionLean.Lowering
open CompressionLean.Lib
open CompressionLean.Replay (Word Ext)
open NightstreamFPrime.Spec (goldilocksModulus)

variable {F : Type} [CommRing F]

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
  · rw [dite_eq_left lt, dite_eq_left (h.1 ▸ lt), ← getElem!_pos, ← getElem!_pos]
    exact h.2 i lt
  · rw [dite_eq_right lt, dite_eq_right (h.1 ▸ lt)]
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

theorem reverse (h : IsWords c n hs gs) : IsWords c n hs.reverse gs.reverse := by
  refine ⟨by simpa using h.1, fun i hi => ?_⟩
  have hg : i < gs.reverse.size := by simpa [← h.1] using hi
  rw [getElem!_pos hs.reverse i hi, getElem!_pos gs.reverse i hg,
    Array.getElem_reverse, Array.getElem_reverse]
  have hk : hs.size - 1 - i < hs.size := by simp only [Array.size_reverse] at hi; omega
  have hk' : gs.size - 1 - i < gs.size := by simp only [Array.size_reverse] at hg; omega
  rw [← getElem!_pos hs _ hk, ← getElem!_pos gs _ hk', ← h.1]
  exact h.2 _ hk

theorem getD (h : IsWords c n hs gs) {z : Var} {g : Word} (hz : IsWord c n z g) (i : ℕ) :
    IsWord c n (hs.getD i z) (gs.getD i g) := by
  by_cases hi : i < hs.size
  · simpa [Array.getD, hi, ← h.1, ← getElem!_pos] using h.2 i hi
  · simpa [Array.getD, hi, ← h.1] using hz

theorem mapGetD (h : IsWords c n hs gs) {z : Var} {g : Word} (hz : IsWord c n z g) (count : ℕ) :
    IsWords c n ((Array.range count).map fun i => hs.getD i z)
      ((Array.range count).map fun i => gs.getD i g) := by
  refine ⟨by simp, fun i hi => ?_⟩
  simp only [Array.size_map, Array.size_range] at hi
  rw [getElem!_map_range _ _ hi, getElem!_map_range _ _ hi]
  exact h.getD hz i

theorem ofFn {k : ℕ} {f : Fin k → Var} {gs : Array Word} (size : gs.size = k) (hf : ∀ i : Fin k, IsWord c n (f i) gs[i]!) :
    IsWords c n (Array.ofFn f) gs := by
  refine ⟨by simp [size], fun i hi => ?_⟩
  rw [Array.size_ofFn] at hi
  rw [getElem!_pos _ _ (by simpa using hi), Array.getElem_ofFn]
  exact hf ⟨i, hi⟩

theorem setRange (h : IsWords c n hs gs) (k : ℕ) (lane : ℕ → ℕ) {f : ℕ → Var} {g : ℕ → Word}
    (hf : ∀ i < k, IsWord c n (f i) (g i)) :
    IsWords c n ((List.range k).foldl (fun s i => s.set! (lane i) (f i)) hs)
      ((List.range k).foldl (fun s i => s.set! (lane i) (g i)) gs) := by
  induction k with
  | zero => exact h
  | succ k ih =>
      rw [List.range_succ, List.foldl_append, List.foldl_append]
      exact (ih fun i hi => hf i (by omega)).set (hf k (by omega)) _

theorem of_forall₂ (h : List.Forall₂ (fun g v => IsWord c n v g) gs.toList hs.toList) : IsWords c n hs gs := by
  have size := h.length_eq
  simp only [Array.length_toList] at size
  refine ⟨size.symm, fun i hi => ?_⟩
  have := List.forall₂_iff_get.mp h |>.2 i (by simp [size]; omega) (by simpa using hi)
  rw [getElem!_pos hs i hi, getElem!_pos gs i (by omega)]
  simpa using this

end IsWords

theorem mapWords_ensures {c : Ctx F} {n count : ℕ} {f : ℕ → Build Var} {g : ℕ → Word}
    (h : ∀ i < count, ∀ m, n ≤ m → Ensures c m (f i) fun v m' => IsWord c m' v (g i))
    (hf : ∀ i, Appends (f i) := by appends) :
    Ensures c n ((Array.range count).mapM f)
      fun values m => IsWords c m values ((Array.range count).map g) := by
  refine (Ensures.mapM (fun i v m => IsWord c m v (g i))
    (fun _ _ _ _ hi le => hi.mono le) (fun i hi => h i (by simpa using hi)) hf).post fun values m hv => ?_
  apply IsWords.of_forall₂
  simpa only [Array.toList_map, List.forall₂_map_left_iff] using hv

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

theorem forIn_pure {m : Type → Type} [Monad m] [LawfulMonad m] {α β : Type} {l : List α} {init : β}
    {body : α → β → m (ForInStep β)} (f : α → β → β) (h : ∀ x s, body x s = pure (.yield (f x s))) :
    forIn l init body = pure (l.foldl (fun s x => f x s) init) := by
  rw [show body = fun x s => pure (.yield (f x s)) from funext fun x => funext fun s => h x s]
  exact List.forIn_pure_yield_eq_foldl f init

structure IsDuplex (c : Ctx F) (n : ℕ) (d : Duplex) (r : Replay.Duplex) : Prop where
  state : IsWords c n d.state r.state
  input : IsWords c n d.input r.input
  output : match d.cursor with
    | none => IsWords c n d.output r.output
    | some cursor => IsWords c n d.output (queueWords r.output) ∧
        ∀ i : Fin 13, IsWord c n (cursor i) (if i.val = r.output.size then 1 else 0)
  width : r.state.size = 16
  outputBound : r.output.size ≤ rate
  cursorInput : d.cursor.isSome = true → r.input = #[]

theorem IsDuplex.mono {c : Ctx F} {n m : ℕ} {d : Duplex} {r : Replay.Duplex} (h : IsDuplex c n d r)
    (le : n ≤ m := by omega) : IsDuplex c m d r := by
  refine ⟨h.state.mono le, h.input.mono le, ?_, h.width, h.outputBound, h.cursorInput⟩
  cases hc : d.cursor with
  | none => exact (show IsWords c n d.output r.output by simpa [hc] using h.output).mono le
  | some cursor =>
      obtain ⟨ho, hr⟩ := (show IsWords c n d.output (queueWords r.output) ∧
        ∀ i : Fin 13, IsWord c n (cursor i) (if i.val = r.output.size then 1 else 0) by simpa [hc] using h.output)
      exact ⟨ho.mono le, fun i => (hr i).mono le⟩

theorem IsDuplex.promoted {c : Ctx F} {n : ℕ} {d : Duplex} {r : Replay.Duplex}
    (h : IsDuplex c n d r) {zeroWord one : Var} (hz : IsWord c n zeroWord 0) (ho : IsWord c n one 1) :
    IsWords c n (if d.cursor.isSome then d.output else
      (Array.range rate).map fun i => d.output.reverse.getD i zeroWord) (queueWords r.output) ∧
    IsWords c n (Array.ofFn (d.cursor.getD fun i => if i.val = d.output.size then one else zeroWord))
      (cursorWords r.output.size) := by
  cases hc : d.cursor with
  | none =>
      have hout : IsWords c n d.output r.output := by simpa [hc] using h.output
      refine ⟨by simpa [hc, queueWords] using hout.reverse.mapGetD hz rate, ?_⟩
      simp only [Option.getD_none]
      apply IsWords.ofFn (by simp)
      intro i
      change IsWord c n (if i.val = d.output.size then one else zeroWord) (cursorWords r.output.size)[i.val]!
      rw [cursorWords_get _ _ i.isLt]
      by_cases hi : i.val = d.output.size
      · simpa [hi, ← hout.1] using ho
      · simpa [hi, ← hout.1] using hz
  | some cursor =>
      have hout : IsWords c n d.output (queueWords r.output) ∧
          ∀ i : Fin 13, IsWord c n (cursor i) (if i.val = r.output.size then 1 else 0) := by
        simpa [hc] using h.output
      refine ⟨by simpa [hc] using hout.1, ?_⟩
      simp only [Option.getD_some]
      apply IsWords.ofFn (by simp)
      intro i
      change IsWord c n (cursor i) (cursorWords r.output.size)[i.val]!
      rw [cursorWords_get _ _ i.isLt]
      exact hout.2 i

theorem permute_size (s : Array Word) : (Replay.permute s).size = 16 := rfl

/- Unifying two terms that contain the permutation must never evaluate it. -/
attribute [local irreducible] Replay.permute

theorem IsDuplex.permuted {c : Ctx F} {m : ℕ} (t : Array Word) (outs : Fin 16 → Var)
    (h : ∀ k : Fin 16, IsWord c m (outs k) (Replay.permute t)[k]!) :
    IsDuplex c m
      ({ state := Array.ofFn outs, input := #[], output := (Array.ofFn outs).extract 0 rate, cursor := none } : Duplex)
      ⟨Replay.permute t, #[], (Replay.permute t).extract 0 Replay.rate⟩ := by
  generalize values : Replay.permute t = words at h ⊢
  have size : words.size = 16 := by rw [← values]; exact permute_size t
  have wo : IsWords c m (Array.ofFn outs) words := IsWords.ofFn size h
  refine IsDuplex.mk wo IsWords.empty ?_ size ?_ ?_
  · simpa only [Duplex.cursor, Duplex.output, Replay.rate, rate] using wo.extract 0 rate
  · change (words.extract 0 Replay.rate).size ≤ rate
    rw [Array.size_extract, size]
    decide
  · intro _
    rfl

theorem duplexing_output_size (r : Replay.Duplex) : r.duplexing.output.size = Replay.rate := by
  unfold Replay.Duplex.duplexing
  simp [permute_size, Replay.rate]

theorem IsDuplex.pop {c : Ctx F} {m : ℕ} {d : Duplex} {r : Replay.Duplex} (h : IsDuplex c m d r)
    (static : d.cursor = none) (ne : 0 < r.output.size) :
    IsWord c m d.output.back! r.output.back! ∧
      IsDuplex c m { d with output := d.output.pop } { r with output := r.output.pop } :=
  by
  have ho : IsWords c m d.output r.output := by simpa [static] using h.output
  refine ⟨ho.back (ho.1 ▸ ne), h.state, h.input, ?_, h.width, ?_, ?_⟩
  · simpa [static] using ho.pop
  · have := h.outputBound
    simp only [Array.size_pop]
    omega
  · simp [static]

theorem IsDuplex.cond {c : Ctx F} {m : ℕ} {d : Duplex} {r : Replay.Duplex} (h : IsDuplex c m d r)
    (static : d.cursor = none) :
    (!d.input.isEmpty || d.output.isEmpty) = (!r.input.isEmpty || r.output.isEmpty) := by
  unfold Array.isEmpty
  have ho : IsWords c m d.output r.output := by simpa [static] using h.output
  rw [h.input.1, ho.1]

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
        exact ⟨by dsimp only; rw [ite_eq_right this, Nat.cast_zero, z], by dsimp only; rw [ite_eq_right this]; omega⟩
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

end CompressionLean.Sim
