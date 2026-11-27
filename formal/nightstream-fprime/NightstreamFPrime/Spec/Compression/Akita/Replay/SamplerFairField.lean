import NightstreamFPrime.Spec.Compression.Akita.Replay.SamplerFair
import NightstreamFPrime.Spec.Compression.Akita.Replay.Coupling
import NightstreamFPrime.Spec.Compression.Akita.Replay.SamplerPool

/-! The field challenge sampler of the pinned Akita code (revision `d50f700`,
`jolt_transcript`, `challenge_vector`: the first words below `q` of a pool of words of 64 bits) is
fair (Akita, ePrint 2026/1983, revision of 2026-11-23, Section 8.2 `[KECCAK]`).

- `firstOf g W p` is the list of the first `W` accepted words of `p`, decoded by `g`.
  `card_first_le`: for an injective decoder, each value of this list has at most
  `|Word|^B / |accepted|^W` pools (`pool`, `swapPool_involutive` of `Replay/SamplerPool.lean`).
- `decodePool_eq`: `decodePool` reads the first `2·len` words below `q` of its words, in pairs
  (`paired`). `decodePool_canonical`: an output has `len` elements, each with two canonical words.
- `card_decodePool_le`: each output of `decodePool` has at most `256^m / q^(2·len)` strings of
  `m` bytes. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

open NightstreamFPrime.Spec.Compression.Akita
open Bytes

section First

variable {Word α : Type} [Fintype Word] (accepts : Word → Prop) [DecidablePred accepts]
  {g : Word → Option α} {B W : ℕ}

/-- The first `W` accepted words of a pool, decoded. -/
def firstOf (g : Word → Option α) (W : ℕ) (p : Fin B → Word) : List α :=
  ((List.ofFn p).filterMap g).take W

omit [Fintype Word] in
/-- The element `r` of a full list `firstOf` is the decoded accepted word of rank `r`. -/
theorem exists_rank_first (some_iff : ∀ w, (g w).isSome ↔ accepts w) {p : Fin B → Word}
    {t : List α} (first : firstOf g W p = t) (full : t.length = W) (r : Fin W) :
    ∃ i, accepts (p i) ∧ rank accepts p i = r ∧ g (p i) = t[(r : ℕ)]? := by
  have long : (r : ℕ) < ((List.ofFn p).filterMap g).length := by
    have length := congrArg List.length first
    rw [firstOf, List.length_take, full] at length
    omega
  obtain ⟨i, accepted, same⟩ := exists_rank accepts some_iff p r long
  refine ⟨i, accepted, same, ?_⟩
  rw [← first, firstOf, List.getElem?_take, ite_eq_left r.isLt, ← same,
    getElem?_filterMap_rank accepts some_iff p i accepted]

omit [Fintype Word] in
/-- Two pools with one full list `firstOf` have the same accepted words of the ranks below
`W`. -/
theorem ranked_first (some_iff : ∀ w, (g w).isSome ↔ accepts w)
    (injective : ∀ w w', accepts w → g w = g w' → w = w') {p p' : Fin B → Word} {t : List α}
    (first : firstOf g W p = t) (first' : firstOf g W p' = t) (full : t.length = W)
    (a a' : Fin W → {w // accepts w}) : ranked accepts p a = ranked accepts p' a' := by
  funext r
  obtain ⟨i, accepted, same, read⟩ := exists_rank_first accepts some_iff first full r
  obtain ⟨i', accepted', same', read'⟩ := exists_rank_first accepts some_iff first' full r
  rw [ranked_eq accepts a accepted same, ranked_eq accepts a' accepted' same']
  exact Subtype.ext (injective _ _ accepted (read.trans read'.symm))

/-- For an injective decoder, each full value of the first `W` accepted words has at most
`|Word|^B / |accepted|^W` pools of `B` words. -/
theorem card_first_le (some_iff : ∀ w, (g w).isSome ↔ accepts w)
    (injective : ∀ w w', accepts w → g w = g w' → w = w') (t : List α) (full : t.length = W) :
    Nat.card {p : Fin B → Word // firstOf g W p = t} * Nat.card {w // accepts w} ^ W ≤
      Nat.card Word ^ B := by
  have apart : Function.Injective
      fun x : {p : Fin B → Word // firstOf g W p = t} × (Fin W → {w // accepts w}) =>
        pool accepts x.1.1 x.2 := by
    intro x y same
    have swapped : swapPool accepts (x.1.1, x.2) = swapPool accepts (y.1.1, y.2) :=
      Prod.ext same (ranked_first accepts some_iff injective x.1.2 y.1.2 full x.2 y.2)
    have equal := (swapPool_involutive accepts).injective swapped
    exact Prod.ext (Subtype.ext (Prod.mk.inj equal).1) (Prod.mk.inj equal).2
  have bound := Nat.card_le_card_of_injective _ apart
  rwa [Nat.card_prod, Nat.card_fun, Nat.card_fun, Nat.card_eq_fintype_card (α := Fin W),
    Fintype.card_fin, Nat.card_eq_fintype_card (α := Fin B), Fintype.card_fin] at bound

end First

/-- The loop of `decodePool`: the first `N` accepted values of a list, after `acc`. -/
theorem forIn_first {α : Type} (f : ℕ → Option α) (N : ℕ)
    (step : ℕ → Array α → Id (ForInStep (Array α)))
    (full : ∀ i s, s.size = N → step i s = pure (ForInStep.done s))
    (push : ∀ i s a, s.size ≠ N → f i = some a → step i s = pure (ForInStep.yield (s.push a)))
    (skip : ∀ i s, s.size ≠ N → f i = none → step i s = pure (ForInStep.yield s)) :
    ∀ (l : List ℕ) (acc : Array α), acc.size ≤ N →
      forIn (m := Id) l acc step = pure (acc ++ ((l.filterMap f).take (N - acc.size)).toArray)
  | [], acc, _ => by simp
  | i :: l, acc, room => by
      rw [List.forIn_cons]
      by_cases done : acc.size = N
      · rw [full i acc done]
        simp [done]
      · cases found : f i with
        | some a =>
            rw [push i acc a done found]
            simp only [pure_bind]
            rw [forIn_first f N step full push skip l (acc.push a) (by simp; omega),
              List.filterMap_cons_some found,
              show N - acc.size = N - (acc.push a).size + 1 by simp; omega, List.take_succ_cons]
            simp
        | none =>
            rw [skip i acc done found]
            simp only [pure_bind]
            rw [forIn_first f N step full push skip l acc room, List.filterMap_cons_none found]

/-- The elements of `E` of a list of accepted words: the words in pairs, or none when the list
has less than `2·len` words. -/
def paired (len : ℕ) (accepted : Array Fq) : Option (Array Ext) :=
  if accepted.size < 2 * len then none
  else some ((Array.range len).map fun i => ⟨accepted[2 * i]!, accepted[2 * i + 1]!⟩)

/-- `decodePool` reads the first `2·len` words below `q` of the words of its bytes, in pairs. -/
theorem decodePool_eq (bytes : ByteArray) (len : ℕ) :
    decodePool bytes len = paired len (((List.range (bytes.size / 8)).filterMap fun i =>
      Fq.ofWord? (Wire.word bytes (8 * i))).take (2 * len)).toArray := by
  unfold decodePool
  dsimp only
  rw [Std.Legacy.Range.forIn_eq_forIn_range',
    forIn_first (fun i => Fq.ofWord? (Wire.word bytes (8 * i))) (2 * len) _
      (fun i s size => by simp [size])
      (fun i s a size found => by simp [size, found])
      (fun i s size found => by simp [size, found]) _ _ (Nat.zero_le _)]
  simp [paired, Std.Legacy.Range.size, List.range_eq_range']
  rfl

/-- Two full lists of accepted words with one output of `paired` are equal. -/
theorem paired_injective {len : ℕ} {l l' : List Fq} {out : Array Ext} (full : l.length = 2 * len)
    (full' : l'.length = 2 * len) (read : paired len l.toArray = some out)
    (read' : paired len l'.toArray = some out) : l = l' := by
  unfold paired at read read'
  rw [ite_eq_right (by simp [full])] at read
  rw [ite_eq_right (by simp [full'])] at read'
  have same := (Option.some.inj read).trans (Option.some.inj read').symm
  refine List.ext_getElem (full.trans full'.symm) fun r small small' => ?_
  have half : r / 2 < len := by omega
  have entry := congrArg (fun a : Array Ext => a[r / 2]?) same
  simp [half] at entry
  have value : ∀ j (inside : j < l.length) (inside' : j < l'.length),
      l[j]?.getD default = l'[j]?.getD default → l[j] = l'[j] := by
    intro j inside inside' equal
    simpa [inside, inside'] using equal
  rcases Nat.mod_two_eq_zero_or_one r with even | odd
  · have index : 2 * (r / 2) = r := by omega
    have first := entry.1
    rw [index] at first
    exact value r small small' first
  · have index : 2 * (r / 2) + 1 = r := by omega
    have second := entry.2
    rw [index] at second
    exact value r small small' second

/-- The element of a word of 64 bits, when the word is below `q`. -/
def wordField (w : Fin (2 ^ 64)) : Option Fq := Fq.ofWord? (UInt64.ofNatLT w w.isLt)

theorem wordField_eq (w : Fin (2 ^ 64)) :
    wordField w = if (w : ℕ) < fieldModulus then some ⟨UInt64.ofNatLT w w.isLt⟩ else none :=
  rfl

theorem wordField_isSome (w : Fin (2 ^ 64)) : (wordField w).isSome ↔ (w : ℕ) < fieldModulus := by
  rw [wordField_eq]
  split <;> simp_all

theorem wordField_injective (w w' : Fin (2 ^ 64)) (small : (w : ℕ) < fieldModulus)
    (same : wordField w = wordField w') : w = w' := by
  rw [wordField_eq, ite_eq_left small] at same
  have canonical := (Fq.ofWord?_canonical
    (show Fq.ofWord? (UInt64.ofNatLT w' w'.isLt) = some _ from same.symm)).1
  exact Fin.ext (congrArg UInt64.toNat canonical)

/-- The words below `q` are `q` words. -/
theorem card_canonical : Nat.card {w : Fin (2 ^ 64) // (w : ℕ) < fieldModulus} = fieldModulus :=
  (Nat.card_congr
    { toFun := fun w => (⟨w.1, w.2⟩ : Fin fieldModulus)
      invFun := fun x => ⟨⟨x, x.isLt.trans (by decide)⟩, x.isLt⟩
      left_inv := fun _ => rfl
      right_inv := fun _ => rfl }).trans (Nat.card_eq_fintype_card.trans (Fintype.card_fin _))

/-- The accepted words of a string are the accepted windows of 64 bits of its number. -/
theorem filterMap_words_array {m : ℕ} (u : Bytes m) :
    ((List.range ((array u).size / 8)).filterMap fun i =>
        Fq.ofWord? (Wire.word (array u) (8 * i))) =
      (List.ofFn (digits (m / 8) 64 u)).filterMap wordField := by
  rw [size_array, List.ofFn_eq_map, List.filterMap_map, ← List.map_coe_finRange_eq_range,
    List.filterMap_map]
  congr 1
  funext i
  show Fq.ofWord? (Wire.word (array u) (8 * (i : ℕ))) =
    Fq.ofWord? (UInt64.ofNatLT (digits (m / 8) 64 u i) (digits (m / 8) 64 u i).isLt)
  congr 1
  apply UInt64.toNat_inj.mp
  rw [word_array]
  show _ = value u / 2 ^ ((i : ℕ) * 64) % 2 ^ 64
  rw [show (256 : ℕ) = 2 ^ 8 from rfl, ← pow_mul, ← pow_mul, Nat.mul_comm (i : ℕ) 64,
    ← Nat.mul_assoc]

/-- An accepted read of `decodePool` on a string has a full list of the first `2·len` accepted
windows, and the output is `paired` of it. -/
theorem decodePool_first {m len : ℕ} {u : Bytes m} {out : Array Ext}
    (read : decodePool (array u) len = some out) :
    (firstOf wordField (2 * len) (digits (m / 8) 64 u)).length = 2 * len ∧
      paired len (firstOf wordField (2 * len) (digits (m / 8) 64 u)).toArray = some out := by
  rw [decodePool_eq, filterMap_words_array] at read
  refine ⟨?_, read⟩
  unfold paired at read
  split at read
  · cases read
  · next long =>
    rw [List.size_toArray] at long
    exact Nat.le_antisymm (List.length_take_le _ _) (Nat.le_of_not_lt long)

/-- An output of `decodePool` has `len` elements, each with two canonical words. -/
theorem decodePool_canonical {bytes : ByteArray} {len : ℕ} {out : Array Ext}
    (read : decodePool bytes len = some out) :
    out.size = len ∧
      ∀ x ∈ out, x.c0.word.toNat < fieldModulus ∧ x.c1.word.toNat < fieldModulus := by
  rw [decodePool_eq] at read
  obtain ⟨L, taken⟩ : ∃ L, L = ((List.range (bytes.size / 8)).filterMap fun i =>
      Fq.ofWord? (Wire.word bytes (8 * i))).take (2 * len) := ⟨_, rfl⟩
  rw [← taken] at read
  have canonical : ∀ a ∈ L, a.word.toNat < fieldModulus := by
    intro a member
    rw [taken] at member
    obtain ⟨i, -, found⟩ := List.mem_filterMap.mp (List.mem_of_mem_take member)
    have read := Fq.ofWord?_canonical found
    exact read.1 ▸ read.2
  unfold paired at read
  split at read
  · cases read
  · next long =>
    rw [List.size_toArray] at long
    have each : ∀ j, j < 2 * len → (L.toArray[j]!).word.toNat < fieldModulus := by
      intro j small
      have inside : j < L.length := by omega
      rw [getElem!_pos L.toArray j (by simpa using inside)]
      exact canonical _ (by simp)
    obtain rfl := Option.some.inj read
    refine ⟨by simp, fun x member => ?_⟩
    obtain ⟨i, inside, rfl⟩ := Array.mem_map.mp member
    have small : i < len := by simpa using inside
    exact ⟨each _ (by omega), each _ (by omega)⟩

/-- `decodePool` is fair: each output has at most `256^m / q^(2·len)` strings of `m` bytes. -/
theorem card_decodePool_le (m len : ℕ) (out : Array Ext) :
    Nat.card {u : Bytes m // decodePool (array u) len = some out} * fieldModulus ^ (2 * len) ≤
      256 ^ m := by
  rcases isEmpty_or_nonempty {u : Bytes m // decodePool (array u) len = some out} with
    empty | ⟨⟨u₀, read₀⟩⟩
  · rw [Nat.card_of_isEmpty, Nat.zero_mul]
    exact Nat.zero_le _
  · obtain ⟨full₀, paired₀⟩ := decodePool_first read₀
    have inside (u : {u : Bytes m // decodePool (array u) len = some out}) :
        firstOf wordField (2 * len) (digits (m / 8) 64 u.1) =
          firstOf wordField (2 * len) (digits (m / 8) 64 u₀) :=
      paired_injective (decodePool_first u.2).1 full₀ (decodePool_first u.2).2 paired₀
    have fits : m / 8 * 64 ≤ 8 * m := by omega
    have pools := card_first_le (B := m / 8) (fun w : Fin (2 ^ 64) => (w : ℕ) < fieldModulus)
      wordField_isSome wordField_injective _ full₀
    rw [card_canonical, Nat.card_eq_fintype_card (α := Fin (2 ^ 64)), Fintype.card_fin] at pools
    calc Nat.card {u : Bytes m // decodePool (array u) len = some out} *
          fieldModulus ^ (2 * len)
        ≤ Nat.card {u : Bytes m // firstOf wordField (2 * len) (digits (m / 8) 64 u) =
            firstOf wordField (2 * len) (digits (m / 8) 64 u₀)} * fieldModulus ^ (2 * len) :=
          Nat.mul_le_mul_right _ (Nat.card_le_card_of_injective (fun u => ⟨u.1, inside u⟩)
            fun _ _ same => Subtype.ext (Subtype.mk.inj same))
      _ = 2 ^ (8 * m - m / 8 * 64) *
            (Nat.card {p : Fin (m / 8) → Fin (2 ^ 64) // firstOf wordField (2 * len) p =
              firstOf wordField (2 * len) (digits (m / 8) 64 u₀)} *
              fieldModulus ^ (2 * len)) := by
          rw [card_preimage (digits (m / 8) 64) (card_digits fits) fun p =>
            firstOf wordField (2 * len) p = firstOf wordField (2 * len) (digits (m / 8) 64 u₀),
            Nat.mul_assoc]
      _ ≤ 2 ^ (8 * m - m / 8 * 64) * (2 ^ 64) ^ (m / 8) := Nat.mul_le_mul_left _ pools
      _ = 256 ^ m := by
          rw [← pow_mul, ← pow_add, show (256 : ℕ) = 2 ^ 8 from rfl, ← pow_mul]
          congr 1
          omega

end NightstreamFPrime.Spec.Compression.Akita.Replay
