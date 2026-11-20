import NightstreamFPrime.Spec.Compression.Lib.Density
import NightstreamFPrime.Spec.Compression.Akita.Extract.Segment

/-! The batching root of an opening of several claims, for independent weights (Akita,
ePrint 2026/1983, revision of 2026-10-15, Section 5.4 with (72), Lemma 10.33, and Theorem 10.46,
"Batching root").

The weights are one independent uniform element of `E` for each claim, which is the rule of the
Akita code (`row_coefficients`). The paper has the powers `ζ_batch^a` of one element (71) and the
error `(Λ − 1)/|E|`; for independent weights the error is `1/|E|`.

- `density_batched_le` is Akita Lemma 10.33 for independent weights: when one claim is false, the
  batched claim holds for a part of the weights of density at most `1/|E|`.
- `Root` has the data of the batching root at the stage `start`: the claimed values of the stage
  input, and the values of a candidate. `Root.segment` is its checkpoint segment: one draw of the
  weights, and the test "each claim is true for the candidate".
- `Root.weights` is the weights of a transcript, and `Root.game_eq` states that the game ends in
  them.
- `Root.suffix_before` puts the batching root before a suffix whose first certificates satisfy
  the batched claim. The first certificates of the result satisfy each claim. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Extract

open scoped BigOperators
open NightstreamFPrime.Spec.Checkpoint
open NightstreamFPrime.Spec.Compression (Game)
open NightstreamFPrime.Spec.Compression.Lib (density density_empty density_linear_le)

attribute [local instance low] Classical.propDecidable

variable {E : Type} [Field E] [Fintype E] [DecidableEq E]

/-- Akita Lemma 10.33 for independent weights. The values `actual` and `claimed` of `count` claims
are fixed before the draw of the weights. If one claim is false, then the batched claim
`Σ_i γ_i·actual_i = Σ_i γ_i·claimed_i` holds with probability at most `1/|E|`. -/
theorem density_batched_le {count : ℕ} {actual claimed : Fin count → E}
    (different : actual ≠ claimed) :
    density (Finset.univ.filter fun weights : Fin count → E =>
      ∑ i, weights i * actual i = ∑ i, weights i * claimed i) ≤ 1 / Fintype.card E := by
  obtain ⟨i, differs⟩ := Function.ne_iff.mp different
  have same : (Finset.univ.filter fun weights : Fin count → E =>
      ∑ i, weights i * actual i = ∑ i, weights i * claimed i) =
        Finset.univ.filter fun weights : Fin count → E =>
          ∑ i, weights i * (actual i - claimed i) = 0 := by
    ext weights
    simp only [Finset.mem_filter, Finset.mem_univ, true_and, mul_sub, Finset.sum_sub_distrib,
      sub_eq_zero]
  rw [same]
  exact density_linear_le count (fun i => actual i - claimed i) 0 ⟨i, sub_ne_zero.mpr differs⟩

/-- The batching root of `count` claims at the stage `start` (Akita Section 5.4, "Combining the
scalar claims"). `claimed anchor` has the claimed values `φ_ℓ·v_ℓ` of the stage input of the stage
`start`, and `actual anchor candidate` has the values `v̄_ℓ` of a candidate for the points of this
stage input (Akita Lemma 10.33). `read` decodes the messages of the proof. -/
structure Root (E Wire Coord Slot Cand : Type) (count total : ℕ) where
  start : ℕ
  read : Reader Wire
  claimed : Prefix Wire (Coord → Slot) total → Fin count → E
  actual : Prefix Wire (Coord → Slot) total → Cand → Fin count → E

namespace Root

variable {Wire Coord Slot Cand : Type} {count total : ℕ}
  (R : Root E Wire Coord Slot Cand count total)

/-- The bad set of the draw of the weights (Akita Theorem 10.46, "Bits and bad sets"): when a
claim is false for the candidate, the weights for which the batched claim holds. -/
noncomputable def bad (anchor : Prefix Wire (Coord → Slot) total) (candidate : Cand) :
    Finset (Fin count → E) :=
  if R.actual anchor candidate = R.claimed anchor then ∅
  else Finset.univ.filter fun weights =>
    ∑ i, weights i * R.actual anchor candidate i = ∑ i, weights i * R.claimed anchor i

theorem density_bad_le (anchor : Prefix Wire (Coord → Slot) total) (candidate : Cand) :
    density (R.bad anchor candidate) ≤ 1 / Fintype.card E := by
  unfold Root.bad
  split
  · rw [density_empty]
    positivity
  · rename_i different
    exact density_batched_le different

/-- The checkpoint segment of the batching root: one draw of one weight in `E` for each claim.
The result of the game is the weights. It is true for a candidate when the batched claim (72)
holds for the values of the candidate, and the test of a candidate is "each claim is true"
(Akita Theorem 10.46, "Batching root": the test `R_app`). -/
noncomputable def segment : Segment Wire Coord Slot (Fin count → E) Cand total where
  start := R.start
  costs := [1 / (Fintype.card E : ℝ)]
  read := R.read
  check anchor candidate := .draw (Fin count → E) (R.bad anchor candidate) .pure
  passes anchor candidate weights :=
    ∑ i, weights i * R.actual anchor candidate i = ∑ i, weights i * R.claimed anchor i
  test anchor candidate := R.actual anchor candidate = R.claimed anchor

variable [Fintype Slot] [Nonempty Slot] [Fintype Coord] [DecidableEq Coord]

omit [Nonempty Slot] in
/-- The batching root satisfies the conditions of `Segment.Valid`, with the bound `1/|E|` of its
draw (Akita Lemma 10.33 for independent weights). `fair`: an answer of the oracle gives a uniform
vector of `count` elements of `E`. -/
theorem segment_valid (fair : ∃ decode : (Coord → Slot) → Fin count → E, Fair decode) :
    R.segment.Valid where
  sound anchor candidate failed := fun weights outside passed => outside (by
    have failed : ¬ R.actual anchor candidate = R.claimed anchor := failed
    unfold Root.bad
    rw [ite_eq_right failed]
    exact Finset.mem_filter.mpr ⟨Finset.mem_univ _, passed⟩)
  staged anchor candidate := stagedTo_draw fair (R.density_bad_le anchor candidate) fun _ => trivial
  nonnegative cost member := by
    rw [List.mem_singleton.mp member]
    positivity
  draws := List.cons_ne_nil _ _

/-- The weights `ϑ_ℓ` of a transcript: the vector that the answer of the stage `start` gives. -/
noncomputable def weights (point : Prefix Wire (Coord → Slot) total) : Fin count → E :=
  ((point.reply R.start).map (sample (Coord → Slot) (Fin count → E))).getD 0

omit [Nonempty Slot] in
/-- The game of the batching root on a transcript with the stage `start` ends in the weights of
the transcript. -/
theorem game_eq (candidate : Cand) {point : Prefix Wire (Coord → Slot) total}
    (long : R.start < point.rest.length) :
    R.segment.game candidate point = .pure (R.weights point) := by
  obtain ⟨move, found⟩ : ∃ move, point.rest[R.start]? = some move :=
    ⟨_, List.getElem?_eq_getElem long⟩
  have moves : (point.rest.take R.segment.finish).drop R.segment.start = [move] := by
    show (point.rest.take (R.start + 1)).drop R.start = [move]
    rw [List.take_add_one, found, List.drop_append_of_le_length (by
        rw [List.length_take]
        omega),
      List.drop_eq_nil_of_le (by
        rw [List.length_take]
        omega)]
    rfl
  unfold Segment.game
  rw [moves]
  simp [position, settle, advance, Root.segment, Root.weights, Prefix.reply, found]

variable {Inner BreakT CandT KeyT Wit' : Type}
  {accept : Prefix Wire (Coord → Slot) total → Prop}
  {Opens' : Prefix Wire (Coord → Slot) total → Wit' → Prop}
  {later : Structure (Prefix Wire (Coord → Slot) total) Coord Slot (ℕ → Wire) Inner BreakT CandT
    KeyT Wit'}
  {error : ℝ} {loops tree : ℕ}

/-- The batching root before a suffix whose first certificates satisfy the batched claim is a
suffix whose first certificates satisfy each claim (Akita Theorem 10.46, "Batching root", for
independent weights). The stage errors add `1/|E|`, the retry loops add 1, and the tree size is
multiplied by 2. `source` maps a witness of the later stages to the candidate, for example to the
tables of its weak openings. `Kept` is the part of the relation of the later stages that reads
the transcript before the stage `start` only, for example the certified decommitments. `batched`:
a first certificate of the later stages has a transcript with the stage `start`, a witness of
`Kept`, and the batched claim (72) for the weights of the transcript (Akita (202)). `clashes`: two
witnesses with different candidates, for two accepting transcripts with one stage input of the
stage `start`, give a witness of `R_bind`. -/
theorem suffix_before (fair : ∃ decode : (Coord → Slot) → Fin count → E, Fair decode)
    (suffix : Tier.Suffix accept (R.start + 1) Opens' later error loops tree)
    (fits : R.start + 1 ≤ total) (source : Wit' → Cand) (clash : Wit' → Wit' → BreakT)
    (Kept : Prefix Wire (Coord → Slot) total → Wit' → Prop)
    (rooted : ∀ point witness, Kept point witness → Kept (point.take R.start) witness)
    (batched : ∀ point witness, accept point → Opens' point witness →
      R.start < point.rest.length ∧ Kept point witness ∧
        ∑ i, R.weights point i * R.actual (point.take R.start) (source witness) i =
          ∑ i, R.weights point i * R.claimed (point.take R.start) i)
    (clashes : ∀ point other witness witness', accept point → accept other →
      Opens' point witness → Opens' other witness' →
      point.take R.start = other.take R.start → source witness ≠ source witness' →
      later.breaks (clash witness witness')) :
    Tier.Suffix accept R.start
      (fun point witness => Kept point witness ∧
        R.actual (point.take R.start) (source witness) = R.claimed (point.take R.start))
      (R.segment.over accept later (fun inner => source (later.witness inner))
        (fun inner other => clash (later.witness inner) (later.witness other))
        fun point witness => Kept point witness ∧
          R.actual (point.take R.start) (source witness) = R.claimed (point.take R.start))
      (1 / Fintype.card E + error) (1 + loops) (2 * tree) := by
  have result := R.segment.suffix_before (R.segment_valid fair) suffix fits source clash
    (fun point witness => Kept point witness ∧
      R.actual (point.take R.start) (source witness) = R.claimed (point.take R.start))
    (fun point witness opens => ⟨rooted point witness opens.1, by
      show R.actual ((point.take R.start).take R.start) (source witness) =
        R.claimed ((point.take R.start).take R.start)
      rw [Prefix.take_take le_rfl]
      exact opens.2⟩)
    (fun point witness accepted opens holds => ⟨(batched point witness accepted opens).2.1, holds⟩)
    (fun point witness accepted opens => ⟨R.weights point,
      R.game_eq _ (batched point witness accepted opens).1,
      (batched point witness accepted opens).2.2⟩)
    clashes
  rw [show R.segment.costs.sum = 1 / (Fintype.card E : ℝ) from List.sum_singleton] at result
  exact result

end Root

end NightstreamFPrime.Spec.Compression.Akita.Extract
