import NightstreamFPrime.Spec.Compression.Akita.Extract.Work

/-! The response search of a fold of Akita in the model of one uniform table of the oracle on keys
(ePrint 2026/1983, revision of 2026-11-23, Section 10.2: Lemma 10.7, Definition 10.8 and
Theorem 10.9, (154); `first_jointly_accepted_nonce` of the pinned Akita code,
`crates/akita-prover/src/protocol/fold_grind.rs`).

The honest prover sends a nonce before the fold challenges of a level, and each nonce gives
another stage input. So the answers of the table for the nonces are independent.

- `mass_points`: for different stage inputs, the probability that the answer at each input is in
  the set of its input is the product of the probabilities of the sets.
- `mass_misses_le` is Akita Lemma 10.7 for one transcript prefix: when one answer makes a nonce
  fail with probability at most `δ`, each of `count` nonces fails with probability at most
  `δ^count`.
- `mass_search_le` is the term of Akita (154) for a prefix that the table gives: the stage inputs
  of the nonces and the set of the failing answers are functions of the table that read only the
  answers at the inputs `stop`, and no input of a nonce is in `stop`. The set of the failing
  answers is empty for a table whose honest transcript does not reach the level.
- `mass_view_only`: a set of answers that does not read the work slot has the probability of its
  set of views. `mass_tries_le` is `mass_misses_le` for the views and the `responseTries = 2^12`
  nonces of the code. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Extract

open scoped BigOperators
open NightstreamFPrime.Spec.RandomOracle (mass mass_nonnegative Stopping overlay overlay_agree
  expect_resample)

attribute [local instance low] Classical.propDecidable

section Points

variable {Point Reply : Type} [Fintype Point] [DecidableEq Point] [Fintype Reply] [Nonempty Reply]

omit [Nonempty Reply] in
/-- The expectation of a product over the points of a uniform table is the product of the
expectations. -/
theorem expect_prod_points (value : Point → Reply → ℝ) :
    𝔼 table : Point → Reply, ∏ point, value point (table point) =
      ∏ point, 𝔼 reply, value point reply := by
  simp only [Finset.expect_eq_sum_div_card, Finset.card_univ, Fintype.card_fun]
  rw [Finset.prod_div_distrib, Finset.prod_const, Finset.card_univ, Finset.prod_univ_sum,
    Fintype.piFinset_univ]
  push_cast
  rfl

/-- Akita Lemma 10.7, "Distinct nonce inputs give independent runs": the answers of a uniform
table at different stage inputs are independent. -/
theorem mass_points {count : ℕ} (input : Fin count → Point)
    (injective : Function.Injective input) (miss : Fin count → Set Reply) :
    mass {table : Point → Reply | ∀ n, table (input n) ∈ miss n} = ∏ n, mass (miss n) := by
  have split : mass {table : Point → Reply | ∀ n, table (input n) ∈ miss n} =
      𝔼 table : Point → Reply,
        ∏ point, if ∀ n, input n = point → table point ∈ miss n then (1 : ℝ) else 0 := by
    unfold mass
    refine Finset.expect_congr rfl fun table _ => ?_
    by_cases all : table ∈ {table : Point → Reply | ∀ n, table (input n) ∈ miss n}
    · rw [ite_eq_left all]
      exact (Finset.prod_eq_one fun point _ =>
        ite_eq_left fun n same => same ▸ all n).symm
    · rw [ite_eq_right all]
      obtain ⟨n, outside⟩ := not_forall.mp (all : ¬ ∀ n, table (input n) ∈ miss n)
      exact (Finset.prod_eq_zero (Finset.mem_univ (input n))
        (ite_eq_right fun each => outside (each n rfl))).symm
  rw [split]
  unfold mass
  rw [expect_prod_points fun point reply =>
      if ∀ n, input n = point → reply ∈ miss n then (1 : ℝ) else 0,
    ← Finset.prod_subset (Finset.subset_univ (Finset.univ.image input)) fun point _ outside => by
      have none : ∀ n, input n ≠ point := fun n same =>
        outside (Finset.mem_image.mpr ⟨n, Finset.mem_univ n, same⟩)
      simp [none],
    Finset.prod_image fun a _ b _ same => injective same]
  refine Finset.prod_congr rfl fun n _ => Finset.expect_congr rfl fun reply _ => ?_
  by_cases inside : reply ∈ miss n
  · rw [ite_eq_left inside, ite_eq_left fun other same => injective same ▸ inside]
  · rw [ite_eq_right inside, ite_eq_right fun each => inside (each n rfl)]

/-- Akita Lemma 10.7 for one transcript prefix: the honest prover with `count` nonces finds no
nonce with probability at most `δ^count`, when the answer of one nonce is in the set `miss` of
the failing answers with probability at most `δ`. `input n` is the stage input with the nonce
`n`. -/
theorem mass_misses_le {count : ℕ} (input : Fin count → Point)
    (injective : Function.Injective input) (miss : Set Reply) {δ : ℝ} (bound : mass miss ≤ δ) :
    mass {table : Point → Reply | ∀ n, table (input n) ∈ miss} ≤ δ ^ count := by
  rw [mass_points input injective fun _ => miss, Finset.prod_const, Finset.card_univ,
    Fintype.card_fin]
  exact pow_le_pow_left₀ (mass_nonnegative miss) bound count

/-- The term of Akita (154) for one level: `E[1{Reach}·q(Tr)^N] ≤ δ^N`. The stage inputs `input`
of the nonces and the set `miss` of the failing answers are functions of the table. `stop` has
the stage inputs that these functions read: for a table with the same answers on `stop`, they
have the same values (`same`), and no stage input of a nonce is in `stop` (`fresh`). Then no
nonce is found with probability at most `δ^count`, when `miss` has the probability at most `δ`
for each table (Akita Definition 10.8). -/
theorem mass_search_le (stop : (Point → Reply) → Finset Point) (stopping : Stopping stop)
    {count : ℕ} (input : (Point → Reply) → Fin count → Point) (miss : (Point → Reply) → Set Reply)
    (same : ∀ table other : Point → Reply, (∀ point ∈ stop table, table point = other point) →
      input other = input table ∧ miss other = miss table)
    (fresh : ∀ table n, input table n ∉ stop table)
    (injective : ∀ table, Function.Injective (input table)) {δ : ℝ}
    (bound : ∀ table, mass (miss table) ≤ δ) :
    mass {table : Point → Reply | ∀ n, table (input table n) ∈ miss table} ≤ δ ^ count := by
  unfold mass
  rw [expect_resample stop stopping]
  refine (Finset.expect_le_expect fun table _ => ?_).trans
    (le_of_eq (Finset.expect_const Finset.univ_nonempty _))
  have inner : {other : Point → Reply | overlay (stop table) table other ∈
      {table : Point → Reply | ∀ n, table (input table n) ∈ miss table}} =
        {other : Point → Reply | ∀ n, other (input table n) ∈ miss table} := by
    ext other
    obtain ⟨inputs, misses⟩ := same table (overlay (stop table) table other)
      (overlay_agree (stop table) table other)
    have read : ∀ n, overlay (stop table) table other (input table n) = other (input table n) :=
      fun n => ite_eq_right (fresh table n)
    show (∀ n, overlay (stop table) table other (input (overlay (stop table) table other) n) ∈
      miss (overlay (stop table) table other)) ↔ ∀ n, other (input table n) ∈ miss table
    rw [inputs, misses]
    exact forall_congr' fun n => by rw [read n]
  show mass {other : Point → Reply | overlay (stop table) table other ∈
    {table : Point → Reply | ∀ n, table (input table n) ∈ miss table}} ≤ δ ^ count
  rw [inner]
  exact mass_misses_le (input table) (injective table) (miss table) (bound table)

end Points

section Views

variable {Point Coord Slot : Type} [Fintype Point] [DecidableEq Point] [Fintype Coord]
  [DecidableEq Coord] [Fintype Slot] [Nonempty Slot]

/-- A set of answers that does not read the work slot has the probability of its views. -/
theorem mass_view_only (bad : Set (Coord → Slot)) :
    mass {reply : Option Coord → Slot | view reply ∈ bad} = mass bad := by
  have whole := mass_view (Finset.univ : Finset Slot) bad
  have one : mass ((Finset.univ : Finset Slot) : Set Slot) = 1 := by
    unfold mass
    simp
  rw [one, one_mul] at whole
  rw [← whole]
  congr 1
  ext reply
  simp

/-- Akita Lemma 10.7 with the `responseTries` nonces of the Akita code: when the fold challenges
of one answer give a response that does not fit with probability at most `δ`, no nonce below
`2^12` gives a response that fits with probability at most `δ^4096`. `input n` is the stage
input with the nonce `n`, and `miss` has the views of the failing answers. -/
theorem mass_tries_le (input : Fin responseTries → Point)
    (injective : Function.Injective input) (miss : Set (Coord → Slot)) {δ : ℝ}
    (bound : mass miss ≤ δ) :
    mass {table : Point → Option Coord → Slot | ∀ n, view (table (input n)) ∈ miss} ≤
      δ ^ responseTries :=
  mass_misses_le input injective {reply | view reply ∈ miss}
    ((mass_view_only miss).le.trans bound)

end Views

end NightstreamFPrime.Spec.Compression.Akita.Extract
