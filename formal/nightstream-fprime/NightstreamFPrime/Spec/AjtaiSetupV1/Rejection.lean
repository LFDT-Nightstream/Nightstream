import Mathlib.Algebra.BigOperators.Field
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Data.Fintype.BigOperators
import Mathlib.Data.Fintype.Fin
import Mathlib.Data.List.OfFn
import Mathlib.Data.Nat.Choose.Basic
import Mathlib.Data.Rat.Cast.Order
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.GCongr
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.NormNum.NatFactorial
import Mathlib.Tactic.Ring
import NightstreamFPrime.Spec.AjtaiSetupV1

/-! The law of one key element over uniform SHAKE128 words (premise P1, not proved here).

A stream is the `keyWords = 84` words of 64 bits that one key element reads. A word below `q` is
kept and a word of `q` or more is rejected, 59 of the `2^64` words (`rejected_eq`).

* `card_startsWith`: the number of streams whose kept words start with a list of values below `q`
  depends only on the length of the list. So the element is exactly uniform over the streams
  that keep `ringDegree = 64` words or more (`card_element`).
* `Fallback`: fewer than 64 words are kept, so 21 or more of the 84 words are rejected, and the
  last coefficients are `0`. `card_fallback_le` bounds its chance by `fallbackBound`, which is
  `C(84, 21) · 59^21 / 2^1344`, below `2^-1155` (`fallbackBound_lt`). -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.AjtaiSetupV1.Rejection

open Finset

section Counting

variable {bound : Nat}

def words {count : Nat} (stream : Fin count → Fin bound) : List Nat :=
  List.ofFn fun index => (stream index).val

theorem words_cons {count : Nat} (word : Fin bound) (stream : Fin count → Fin bound) :
    words (Fin.cons word stream : Fin (count + 1) → Fin bound) = word.val :: words stream := by
  simp [words, List.ofFn_succ]

private theorem keptWords_cons_of_lt {word : Nat} (kept : word < fieldModulus)
    (rest : List Nat) : keptWords (word :: rest) = word :: keptWords rest := by
  simp [keptWords, kept]

private theorem keptWords_cons_of_not_lt {word : Nat} (rejected : ¬ word < fieldModulus)
    (rest : List Nat) : keptWords (word :: rest) = keptWords rest := by
  simp [keptWords, rejected]

private theorem keptWords_length_le {count : Nat} (stream : Fin count → Fin bound) :
    (keptWords (words stream)).length ≤ count := by
  have short := List.length_filter_le (fun word => decide (word < fieldModulus)) (words stream)
  rw [words, List.length_ofFn] at short
  exact short

theorem card_streams (count : Nat) : Fintype.card (Fin count → Fin bound) = bound ^ count := by
  rw [Fintype.card_fun, Fintype.card_fin, Fintype.card_fin]

private theorem card_filter_cons {count : Nat} (event : (Fin (count + 1) → Fin bound) → Prop)
    [DecidablePred event] :
    (univ.filter event).card =
      ∑ word : Fin bound, (univ.filter fun stream : Fin count → Fin bound =>
        event (Fin.cons word stream)).card := by
  simp only [card_filter]
  rw [← Fintype.sum_equiv (Fin.consEquiv fun _ : Fin (count + 1) => Fin bound)
    (fun pair : Fin bound × (Fin count → Fin bound) =>
      if event (Fin.cons pair.1 pair.2) then 1 else 0)
    (fun stream => if event stream then 1 else 0) (fun _ => rfl)]
  exact Fintype.sum_prod_type _

private theorem card_kept (fits : fieldModulus ≤ bound) :
    (univ.filter fun word : Fin bound => word.val < fieldModulus).card = fieldModulus := by
  rw [Fin.card_filter_val_lt, Nat.min_eq_right fits]

private theorem card_not_kept (fits : fieldModulus ≤ bound) :
    (univ.filter fun word : Fin bound => ¬ word.val < fieldModulus).card =
      bound - fieldModulus := by
  have total := card_filter_add_card_filter_not (s := (univ : Finset (Fin bound)))
    (fun word : Fin bound => word.val < fieldModulus)
  rw [card_kept fits, card_univ, Fintype.card_fin] at total
  exact Nat.eq_sub_of_add_eq' total

private theorem sum_kept (fits : fieldModulus ≤ bound) (kept rejected : Nat) :
    (∑ word : Fin bound, if word.val < fieldModulus then kept else rejected) =
      fieldModulus * kept + (bound - fieldModulus) * rejected := by
  rw [sum_ite, sum_const_nat fun _ _ => rfl, sum_const_nat fun _ _ => rfl, card_kept fits,
    card_not_kept fits]

private theorem sum_selected (fits : fieldModulus ≤ bound) (value : Nat)
    (canonical : value < fieldModulus) (kept rejected : Nat) :
    (∑ word : Fin bound, if word.val < fieldModulus then (if word.val = value then kept else 0)
      else rejected) = (bound - fieldModulus) * rejected + kept := by
  rw [sum_ite, sum_const_nat fun _ _ => rfl, card_not_kept fits, Nat.add_comm]
  congr 1
  rw [sum_eq_single_of_mem (⟨value, Nat.lt_of_lt_of_le canonical fits⟩ : Fin bound)
    (mem_filter.mpr ⟨mem_univ _, canonical⟩)]
  · exact ite_eq_left rfl
  · intro word _ different
    exact ite_eq_right fun same => different (Fin.ext same)

/-- The number of streams of `count` words below `bound` whose kept words start with `length`
given values. -/
def startCount (bound : Nat) : Nat → Nat → Nat
  | count, 0 => bound ^ count
  | 0, _ + 1 => 0
  | count + 1, length + 1 =>
      (bound - fieldModulus) * startCount bound count (length + 1) + startCount bound count length

theorem startCount_pos (positive : 0 < bound) (count length : Nat) (fits : length ≤ count) :
    0 < startCount bound count length := by
  induction count generalizing length with
  | zero =>
      obtain rfl : length = 0 := Nat.eq_zero_of_le_zero fits
      exact Nat.pow_pos positive
  | succ count ih =>
      cases length with
      | zero => exact Nat.pow_pos positive
      | succ length =>
          exact Nat.lt_of_lt_of_le (ih length (Nat.le_of_succ_le_succ fits))
            (Nat.le_add_left _ _)

theorem card_startsWith (fits : fieldModulus ≤ bound) {count : Nat} (values : List Nat)
    (canonical : ∀ value ∈ values, value < fieldModulus) :
    (univ.filter fun stream : Fin count → Fin bound =>
      values <+: keptWords (words stream)).card = startCount bound count values.length := by
  induction count generalizing values with
  | zero =>
      cases values with
      | nil => simp [startCount]
      | cons value rest => simp [startCount, words, keptWords]
  | succ count ih =>
      cases values with
      | nil => simp [startCount, Fintype.card_fin]
      | cons value rest =>
          have restCanonical : ∀ other ∈ rest, other < fieldModulus :=
            fun other member => canonical other (List.mem_cons_of_mem _ member)
          have rejectedStep := ih (value :: rest) canonical
          rw [List.length_cons] at rejectedStep
          have each : ∀ word : Fin bound,
              (univ.filter fun stream : Fin count → Fin bound =>
                (value :: rest) <+: keptWords (words (Fin.cons word stream :
                  Fin (count + 1) → Fin bound))).card =
                if word.val < fieldModulus then
                  (if word.val = value then startCount bound count rest.length else 0)
                else startCount bound count (rest.length + 1) := by
            intro word
            by_cases kept : word.val < fieldModulus
            · rw [ite_eq_left kept]
              by_cases same : word.val = value
              · rw [ite_eq_left same, ← ih rest restCanonical]
                congr 1
                ext stream
                simp only [mem_filter, mem_univ, true_and]
                rw [words_cons, keptWords_cons_of_lt kept, List.cons_prefix_cons]
                exact and_iff_right same.symm
              · rw [ite_eq_right same]
                apply card_eq_zero.mpr
                apply filter_eq_empty_iff.mpr
                intro stream _
                rw [words_cons, keptWords_cons_of_lt kept, List.cons_prefix_cons]
                exact fun both => same both.1.symm
            · rw [ite_eq_right kept, ← rejectedStep]
              congr 1
              ext stream
              simp only [mem_filter, mem_univ, true_and]
              rw [words_cons, keptWords_cons_of_not_lt kept]
          rw [card_filter_cons, sum_congr rfl fun word _ => each word,
            sum_selected fits value (canonical value List.mem_cons_self)]
          rfl

/-- The streams of `count` words with `needed` or more rejected words. -/
theorem card_rejected_le (fits : fieldModulus ≤ bound) (count needed : Nat) :
    (univ.filter fun stream : Fin count → Fin bound =>
      needed ≤ count - (keptWords (words stream)).length).card * bound ^ needed ≤
      count.choose needed * (bound - fieldModulus) ^ needed * bound ^ count := by
  induction count generalizing needed with
  | zero =>
      cases needed with
      | zero => simp
      | succ needed => simp
  | succ count ih =>
      cases needed with
      | zero => simp [Fintype.card_fin]
      | succ needed =>
          have each : ∀ word : Fin bound,
              (univ.filter fun stream : Fin count → Fin bound =>
                needed + 1 ≤ count + 1 - (keptWords (words (Fin.cons word stream :
                  Fin (count + 1) → Fin bound))).length).card =
                if word.val < fieldModulus then
                  (univ.filter fun stream : Fin count → Fin bound =>
                    needed + 1 ≤ count - (keptWords (words stream)).length).card
                else
                  (univ.filter fun stream : Fin count → Fin bound =>
                    needed ≤ count - (keptWords (words stream)).length).card := by
            intro word
            by_cases kept : word.val < fieldModulus
            · rw [ite_eq_left kept]
              congr 1
              ext stream
              simp only [mem_filter, mem_univ, true_and]
              rw [words_cons, keptWords_cons_of_lt kept, List.length_cons]
              omega
            · rw [ite_eq_right kept]
              congr 1
              ext stream
              have short := keptWords_length_le stream
              simp only [mem_filter, mem_univ, true_and]
              rw [words_cons, keptWords_cons_of_not_lt kept]
              omega
          rw [card_filter_cons, sum_congr rfl fun word _ => each word, sum_kept fits]
          have keptBound := ih (needed + 1)
          have rejectedBound := ih needed
          set keptStreams := (univ.filter fun stream : Fin count → Fin bound =>
            needed + 1 ≤ count - (keptWords (words stream)).length).card
          set rejectedStreams := (univ.filter fun stream : Fin count → Fin bound =>
            needed ≤ count - (keptWords (words stream)).length).card
          calc (fieldModulus * keptStreams + (bound - fieldModulus) * rejectedStreams) *
                bound ^ (needed + 1)
              = fieldModulus * (keptStreams * bound ^ (needed + 1)) +
                  (bound - fieldModulus) * bound * (rejectedStreams * bound ^ needed) := by ring
            _ ≤ bound * (count.choose (needed + 1) * (bound - fieldModulus) ^ (needed + 1) *
                    bound ^ count) +
                  (bound - fieldModulus) * bound *
                    (count.choose needed * (bound - fieldModulus) ^ needed * bound ^ count) :=
                Nat.add_le_add (Nat.mul_le_mul fits keptBound)
                  (Nat.mul_le_mul_left _ rejectedBound)
            _ = (count + 1).choose (needed + 1) * (bound - fieldModulus) ^ (needed + 1) *
                  bound ^ (count + 1) := by
                rw [Nat.choose_succ_succ']
                ring

variable {count degree : Nat}

/-- The ring element that a stream gives: coefficient `lane` is the `lane`-th kept word, and `0`
when fewer words are kept. -/
def streamElement (stream : Fin count → Fin bound) : Fin degree → F :=
  fun lane => ⟨keptCoefficientNat (words stream) lane.val, keptCoefficientNat_lt _ _⟩

/-- Fewer than `degree` words of `stream` are kept, so the fallback gives a coefficient. -/
def StreamFallback (degree : Nat) (stream : Fin count → Fin bound) : Prop :=
  (keptWords (words stream)).length < degree

instance : DecidablePred (StreamFallback degree (count := count) (bound := bound)) :=
  fun _ => Nat.decLt _ _

private def coefficients (value : Fin degree → F) : List Nat :=
  List.ofFn fun lane => (value lane).val

private theorem coefficients_length (value : Fin degree → F) :
    (coefficients value).length = degree := by
  simp [coefficients]

private theorem coefficients_prefix_iff (stream : Fin count → Fin bound)
    (value : Fin degree → F) :
    coefficients value <+: keptWords (words stream) ↔
      ¬ StreamFallback degree stream ∧ streamElement stream = value := by
  have length := coefficients_length value
  rw [List.prefix_iff_eq_take, length]
  constructor
  · intro same
    have long : degree ≤ (keptWords (words stream)).length := by
      have lengths := congrArg List.length same
      rw [length, List.length_take] at lengths
      omega
    refine ⟨Nat.not_lt.mpr long, ?_⟩
    funext lane
    apply Fin.ext
    have inside : lane.val < (keptWords (words stream)).length :=
      Nat.lt_of_lt_of_le lane.isLt long
    have entry := congrArg (fun values => values[lane.val]?) same
    simp only [coefficients, List.getElem?_ofFn, lane.isLt, dite_true, List.getElem?_take,
      ite_true, List.getElem?_eq_getElem inside, Option.some.injEq] at entry
    change keptCoefficientNat (words stream) lane.val = (value lane).val
    rw [keptCoefficientNat_of_kept _ _ inside]
    exact entry.symm
  · rintro ⟨long, rfl⟩
    have longEnough : degree ≤ (keptWords (words stream)).length := Nat.not_lt.mp long
    apply List.ext_getElem
    · rw [length, List.length_take]
      omega
    · intro index inLeft inRight
      rw [length] at inLeft
      have inside : index < (keptWords (words stream)).length :=
        Nat.lt_of_lt_of_le inLeft longEnough
      simp only [coefficients, List.getElem_ofFn, List.getElem_take, streamElement]
      exact keptCoefficientNat_of_kept _ _ inside

/-- Each ring element is the element of the same number of streams that take no fallback. -/
theorem card_streamElement (fits : fieldModulus ≤ bound) (value : Fin degree → F) :
    (univ.filter fun stream : Fin count → Fin bound =>
      ¬ StreamFallback degree stream ∧ streamElement stream = value).card =
      startCount bound count degree := by
  have canonical : ∀ entry ∈ coefficients value, entry < fieldModulus := by
    intro entry member
    obtain ⟨lane, rfl⟩ := List.mem_ofFn.mp member
    exact (value lane).isLt
  calc (univ.filter fun stream : Fin count → Fin bound =>
        ¬ StreamFallback degree stream ∧ streamElement stream = value).card
      = (univ.filter fun stream : Fin count → Fin bound =>
          coefficients value <+: keptWords (words stream)).card := by
        congr 1
        ext stream
        simp only [mem_filter, mem_univ, true_and]
        exact (coefficients_prefix_iff stream value).symm
    _ = startCount bound count (coefficients value).length :=
        card_startsWith fits (coefficients value) canonical
    _ = startCount bound count degree :=
        congrArg (startCount bound count) (coefficients_length value)

/-- The chance that a uniform stream takes the fallback: `count − degree + 1` or more of its
`count` words are rejected. -/
theorem streamFallback_chance_le (fits : fieldModulus ≤ bound) (positive : 0 < bound)
    (enough : degree ≤ count) :
    ((univ.filter (StreamFallback degree (count := count) (bound := bound))).card : ℚ) /
        Fintype.card (Fin count → Fin bound) ≤
      (count.choose (count - degree + 1) : ℚ) *
        ((bound - fieldModulus : ℕ) : ℚ) ^ (count - degree + 1) /
        (bound : ℚ) ^ (count - degree + 1) := by
  have subset : (univ.filter (StreamFallback degree (count := count) (bound := bound))).card ≤
      (univ.filter fun stream : Fin count → Fin bound =>
        count - degree + 1 ≤ count - (keptWords (words stream)).length).card := by
    apply card_le_card
    intro stream member
    have short : (keptWords (words stream)).length < degree := (mem_filter.mp member).2
    refine mem_filter.mpr ⟨mem_univ _, ?_⟩
    omega
  have counted := (Nat.mul_le_mul_right (bound ^ (count - degree + 1)) subset).trans
    (card_rejected_le fits count (count - degree + 1))
  have boundPositive : (0 : ℚ) < bound := by exact_mod_cast positive
  have streams : (0 : ℚ) < Fintype.card (Fin count → Fin bound) := by
    rw [card_streams]
    exact_mod_cast Nat.pow_pos positive
  rw [div_le_div_iff₀ streams (pow_pos boundPositive _), card_streams]
  exact_mod_cast counted

/-- The stream whose words are `values`. -/
def streamOfWords (positive : 0 < bound) (values : List Nat)
    (below : ∀ value ∈ values, value < bound) : Fin count → Fin bound :=
  fun index => ⟨values.getD index.val 0, by
    rw [List.getD_eq_getElem?_getD]
    cases found : values[index.val]? with
    | none => exact positive
    | some value => exact below value (List.mem_of_getElem? found)⟩

theorem words_streamOfWords (positive : 0 < bound) (values : List Nat)
    (below : ∀ value ∈ values, value < bound) (length : values.length = count) :
    words (streamOfWords (count := count) positive values below) = values := by
  apply List.ext_getElem
  · simp [words, length]
  · intro position inLeft inRight
    simp [words, streamOfWords, List.getD_eq_getElem?_getD, List.getElem?_eq_getElem inRight]

end Counting

attribute [irreducible] startCount

def wordCount : Nat := 2 ^ 64

abbrev Word := Fin wordCount

instance : Nonempty Word := ⟨⟨0, by decide⟩⟩

theorem fieldModulus_le : fieldModulus ≤ wordCount := by decide

theorem rejected_eq : wordCount - fieldModulus = 59 := by decide

abbrev Stream := Fin keyWords → Word

theorem elementWords_below (seed : List Nat) (row block : Nat) :
    ∀ word ∈ elementWords seed row block, word < wordCount :=
  elementWords_lt seed row block

abbrev element : Stream → RingF := streamElement

abbrev Fallback : Stream → Prop := StreamFallback ringDegree

theorem card_element (value : RingF) :
    (univ.filter fun stream : Stream => ¬ Fallback stream ∧ element stream = value).card =
      startCount wordCount keyWords ringDegree :=
  card_streamElement fieldModulus_le value

theorem startCount_element_pos : 0 < startCount wordCount keyWords ringDegree :=
  startCount_pos (by decide) keyWords ringDegree (by decide)

/-- A bound on the chance that one key element takes the fallback: `keyWords − ringDegree + 1`
or more of its `keyWords` words are rejected. -/
def fallbackBound : ℚ :=
  (keyWords.choose (keyWords - ringDegree + 1) : ℚ) *
    ((wordCount - fieldModulus : ℕ) : ℚ) ^ (keyWords - ringDegree + 1) /
    (wordCount : ℚ) ^ (keyWords - ringDegree + 1)

theorem fallbackBound_nonneg : 0 ≤ fallbackBound :=
  div_nonneg (mul_nonneg (Nat.cast_nonneg _) (pow_nonneg (Nat.cast_nonneg _) _))
    (pow_nonneg (Nat.cast_nonneg _) _)

theorem card_fallback_le :
    ((univ.filter Fallback).card : ℚ) / Fintype.card Stream ≤ fallbackBound :=
  streamFallback_chance_le fieldModulus_le (by decide) (by decide)

theorem fallbackBound_lt : fallbackBound < 1 / 2 ^ 1155 := by
  have core : (Nat.choose 84 21 : ℚ) * 59 ^ 21 < 2 ^ 189 := by
    rw [Nat.choose_eq_descFactorial_div_factorial]
    norm_num [Nat.descFactorial, Nat.factorial]
  have exponent : keyWords - ringDegree + 1 = 21 := by decide
  have keys : keyWords = 84 := rfl
  have word : (wordCount : ℚ) = 2 ^ 64 := by norm_num [wordCount]
  have large : (0 : ℚ) < 2 ^ 1155 := pow_pos (by norm_num) _
  have small : (0 : ℚ) < 2 ^ 189 := pow_pos (by norm_num) _
  rw [fallbackBound, rejected_eq, exponent, keys, word, ← pow_mul,
    show 64 * 21 = 189 + 1155 from rfl, pow_add, Nat.cast_ofNat, ← div_div]
  exact div_lt_div_of_pos_right ((div_lt_one₀ small).mpr core) large

end NightstreamFPrime.Spec.AjtaiSetupV1.Rejection
