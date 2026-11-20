import NightstreamFPrime.Spec.Compression.Akita.Extract.Staged
import NightstreamFPrime.Spec.Compression.Akita.Extract.Suffix
import NightstreamFPrime.Spec.Compression.Akita.Extract.Transcript

/-! A checkpoint segment of a Fiat–Shamir transcript as a part of a checkpoint structure (Akita,
ePrint 2026/1983, revision of 2026-10-15, Definition 10.16 (v) to (vii), and Theorem 10.45 for
one segment).

- `Tier.Suffix` is the invariant of the stages from one stage `start`: the conditions of
  Definition 10.16 from this stage, the relation `Opens` of each first certificate
  (Proposition 10.44 (i)), and the sums of the stage errors, of the retry loops and the tree size.
- `Segment` has the data of a checkpoint at the stage `start` and the plain stages after it. The
  stages have the draws of a public-coin game `check`, which reads the stage input of the stage
  `start` and a candidate. `test` is `Good_ℓ`, and `passes` is the truth of the result of the game
  for the candidate.
- `Segment.over` is the checkpoint structure of the segment over the structure `tail` of the later
  stages. The candidate of the checkpoint is `source` of a certificate of the later stages, and
  `clash` is `Coll_ℓ`.
- `Segment.Valid` has the conditions on the game: Definition 10.16 (vii)(a) and the bounds of the
  bad sets. `Segment.Fed` is the interface of the later stages: (vii)(c) and (vi).
- `Segment.over_lawfulFrom` states the conditions of Definition 10.16 from the stage `start`.
  `Segment.over_chargeFrom`, `Segment.over_loopsFrom` and `Segment.over_treeFrom` add the bounds of
  the draws, one retry loop and the tree factor 2.
- `Segment.suffix_before` puts a segment before a suffix of the later stages. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Extract

open scoped BigOperators
open NightstreamFPrime.Spec.Checkpoint
open NightstreamFPrime.Spec.Compression (Game)
open NightstreamFPrime.Spec.RandomOracle (mass)

attribute [local instance low] Classical.propDecidable

/-- The sum of the entries of a list by index. -/
theorem sum_range_getD (costs : List ℝ) :
    ∑ index ∈ Finset.range costs.length, costs.getD index 0 = costs.sum := by
  induction costs with
  | nil => simp
  | cons cost rest induction =>
      rw [List.length_cons, Finset.sum_range_succ', List.sum_cons, ← induction, add_comm]
      simp

theorem getD_nonnegative {costs : List ℝ} (nonnegative : ∀ cost ∈ costs, 0 ≤ cost) (index : ℕ) :
    0 ≤ costs.getD index 0 := by
  rw [List.getD_eq_getElem?_getD]
  rcases entry : costs[index]? with _ | cost
  · exact le_rfl
  · exact nonnegative cost (List.mem_of_getElem? entry)

namespace Tier

variable {Wire Coord Slot : Type} {total : ℕ} [Fintype Slot] [Nonempty Slot] [Fintype Coord]
  [DecidableEq Coord] {Inner BreakT CandT KeyT Wit : Type}

/-- The invariant of the stages from the stage `start` (Akita Proposition 10.44 for a suffix of
the levels). `body` is the checkpoint structure of these stages, and `Opens` is the relation of
their first certificates. `opened`: a first certificate has an accepting transcript and a witness
of `Opens` (Proposition 10.44 (i)). `rooted`: `Opens` reads the transcript before the stage
`start` only. `lawfulFrom`: the conditions of Definition 10.16 from the stage `start`. `related`:
`Wit` returns this witness. `errorSum`, `loopCount` and `treeSize`: the sum of the stage errors
(206), the number of the retry loops and the tree size of these stages. -/
structure Suffix (accept : Prefix Wire (Coord → Slot) total → Prop) (start : ℕ)
    (Opens : Prefix Wire (Coord → Slot) total → Wit → Prop)
    (body : Structure (Prefix Wire (Coord → Slot) total) Coord Slot (ℕ → Wire) Inner BreakT CandT
      KeyT Wit)
    (error : ℝ) (loops tree : ℕ) : Prop where
  opened : ∀ inner, body.holds start inner →
    accept (body.point inner) ∧ Opens (body.point inner) (body.witness inner)
  rooted : ∀ point witness, Opens point witness → Opens (point.take start) witness
  lawfulFrom : body.LawfulFrom (transcript total accept) start
  protocol : body.protocol = transcript total accept
  related : ∀ root witness, body.relation root witness ↔ Opens root witness
  errorSum : body.chargeFrom start = error
  loopCount : body.loopsFrom start = loops
  treeSize : body.treeFrom start = tree

end Tier

/-- A checkpoint at the stage `start` with the plain stages after it (Akita Definition 10.16 (v)
to (vii)). `check anchor candidate` is the public-coin game of these stages for the stage input
`anchor` of the stage `start` and a candidate: one stage for each draw, and `costs` has one bound
for each draw. `read` decodes the messages of the proof. `passes anchor candidate` is the truth of
the result of the game for the candidate, and `test anchor candidate` is `Good_ℓ`. -/
structure Segment (Wire Coord Slot Result Cand : Type) (total : ℕ) where
  start : ℕ
  costs : List ℝ
  read : Reader Wire
  check : Prefix Wire (Coord → Slot) total → Cand → Game Result
  passes : Prefix Wire (Coord → Slot) total → Cand → Result → Prop
  test : Prefix Wire (Coord → Slot) total → Cand → Prop

namespace Segment

variable {Wire Coord Slot Result Cand : Type} {total : ℕ}
  (G : Segment Wire Coord Slot Result Cand total)

/-- The first stage after the segment: one stage for each draw of the game. -/
def finish : ℕ := G.start + G.costs.length

/-- Definition 10.16 (i) for the stages of the segment. -/
def kind (level : ℕ) : Kind Coord :=
  if level = G.start then .checkpoint (G.costs.length - 1) else .plain

variable [Fintype Slot] [Nonempty Slot] [Fintype Coord] [DecidableEq Coord]

/-- The game for a candidate, after the answers and the messages of a stage input through the
last stage of the segment. -/
noncomputable def game (candidate : Cand) (point : Prefix Wire (Coord → Slot) total) :
    Game Result :=
  position G.read (G.check (point.take G.start) candidate) (point.wire G.start)
    ((point.rest.take G.finish).drop G.start)

/-- Akita Theorem 10.45 (c) on a transcript and a candidate: the game ends in a result that is
true for the candidate. -/
def Passed (candidate : Cand) (point : Prefix Wire (Coord → Slot) total) : Prop :=
  ∃ result, G.game candidate point = .pure result ∧
    G.passes (point.take G.start) candidate result

/-- The conditions on the game. `sound`: for a candidate that fails the test, each path of the
game that avoids the bad sets ends in a false result (Definition 10.16 (vii)(a) and (b)).
`staged` and `nonnegative`: the bounds `costs` of the draws. `draws`: the game has a draw. -/
structure Valid : Prop where
  sound : ∀ point candidate, ¬ G.test point candidate →
    (G.check point candidate).Avoids fun result => ¬ G.passes point candidate result
  staged : ∀ point candidate, Staged (Coord → Slot) (G.check point candidate) G.costs
  nonnegative : ∀ cost ∈ G.costs, 0 ≤ cost
  draws : G.costs ≠ []

variable {Inner BreakT CandT KeyT WitT : Type}
  (accept : Prefix Wire (Coord → Slot) total → Prop)
  (tail : Structure (Prefix Wire (Coord → Slot) total) Coord Slot (ℕ → Wire) Inner BreakT CandT
    KeyT WitT)
  (source : Inner → Cand) (clash : Inner → Inner → BreakT)
  (relation : Prefix Wire (Coord → Slot) total → WitT → Prop)

omit [Fintype Slot] [Nonempty Slot] [Fintype Coord] [DecidableEq Coord] in
/-- `Good_ℓ` of the checkpoint: the test of the candidate of a certificate. -/
def Good (inner : Inner) : Prop := G.test ((tail.point inner).take G.start) (source inner)

omit [Fintype Slot] [Nonempty Slot] [Fintype Coord] [DecidableEq Coord] in
/-- The certificate sets `Y_level`: through the checkpoint a first certificate of the later stages
whose candidate passes the test, at a later stage of the segment a first certificate of the later
stages, and after the segment a certificate of the later stages. -/
def holds (level : ℕ) (inner : Inner) : Prop :=
  if G.finish ≤ level then tail.holds level inner
  else tail.holds G.finish inner ∧ (level ≤ G.start → G.Good tail source inner)

/-- The checkpoint structure of the segment over the structure `tail` of the later stages (Akita
Definition 10.16 for the stages from `start`). The candidate of the checkpoint is `source` of a
certificate of the later stages, and `clash` is `Coll_ℓ`. A stage input is refuted for a candidate
when each path of the game from this stage input that avoids the bad sets ends in a false result.
`relation` is the relation of the witnesses. -/
noncomputable def over :
    Structure (Prefix Wire (Coord → Slot) total) Coord Slot (ℕ → Wire) Inner BreakT
      (Cand ⊕ CandT) KeyT WitT where
  protocol := transcript total accept
  kind level := if level < G.finish then G.kind level else tail.kind level
  key := tail.key
  weight := tail.weight
  error level := if level < G.finish then G.costs.getD (level - G.start) 0 else tail.error level
  leaf := tail.leaf
  point := tail.point
  holds := G.holds tail source
  breaks := tail.breaks
  combine := tail.combine
  candidate level inner :=
    if level < G.finish then .inl (source inner) else .inr (tail.candidate level inner)
  good level inner := if level < G.finish then G.Good tail source inner else tail.good level inner
  collide level inner other :=
    if level < G.finish then clash inner other else tail.collide level inner other
  refuted level input candidate :=
    match candidate with
    | .inl candidate => level < G.finish ∧
        (G.game candidate input).Avoids fun result =>
          ¬ G.passes (input.take G.start) candidate result
    | .inr candidate => G.finish ≤ level ∧ tail.refuted level input candidate
  bad level input candidate :=
    match candidate with
    | .inl candidate =>
        if level < G.finish then badReplies (Coord → Slot) (G.game candidate input) else ∅
    | .inr candidate => if level < G.finish then ∅ else tail.bad level input candidate
  witness := tail.witness
  relation := relation

/-- The interface of the later stages to the segment. `fits`: the stages of the segment are
stages of the transcript. `lawful`: the later stages satisfy the conditions of Definition 10.16.
`passed`: the game ends in a true result for the candidate of each first certificate of the later
stages (Definition 10.16 (vii)(c), Akita Theorem 10.45 (c)). `clash`: two first certificates of
the later stages with one stage input of the checkpoint and different candidates give a witness of
`R_bind` (Definition 10.16 (vi)). -/
structure Fed : Prop where
  fits : G.finish ≤ total
  lawful : tail.LawfulFrom (transcript total accept) G.finish
  passed : ∀ inner, tail.holds G.finish inner → G.Passed (source inner) (tail.point inner)
  clash : ∀ inner other anchor, tail.holds G.finish inner → tail.holds G.finish other →
    (transcript total accept).cut G.start (tail.point inner) = some anchor →
    (transcript total accept).cut G.start (tail.point other) = some anchor →
    source inner ≠ source other → tail.breaks (clash inner other)

variable {accept tail source clash relation}

theorem over_kind (level : ℕ) :
    (G.over accept tail source clash relation).kind level =
      if level < G.finish then G.kind level else tail.kind level :=
  rfl

theorem over_error (level : ℕ) :
    (G.over accept tail source clash relation).error level =
      if level < G.finish then G.costs.getD (level - G.start) 0 else tail.error level :=
  rfl

theorem over_candidate (level : ℕ) (inner : Inner) :
    (G.over accept tail source clash relation).candidate level inner =
      if level < G.finish then .inl (source inner) else .inr (tail.candidate level inner) :=
  rfl

theorem over_good (level : ℕ) (inner : Inner) :
    (G.over accept tail source clash relation).good level inner =
      if level < G.finish then G.Good tail source inner else tail.good level inner :=
  rfl

theorem over_collide (level : ℕ) (inner other : Inner) :
    (G.over accept tail source clash relation).collide level inner other =
      if level < G.finish then clash inner other else tail.collide level inner other :=
  rfl

theorem over_bad_inl (level : ℕ) (input : Prefix Wire (Coord → Slot) total) (candidate : Cand) :
    (G.over accept tail source clash relation).bad level input (.inl candidate) =
      if level < G.finish then badReplies (Coord → Slot) (G.game candidate input) else ∅ :=
  rfl

theorem over_bad_inr (level : ℕ) (input : Prefix Wire (Coord → Slot) total) (candidate : CandT) :
    (G.over accept tail source clash relation).bad level input (.inr candidate) =
      if level < G.finish then ∅ else tail.bad level input candidate :=
  rfl

theorem over_kind_indexed {level : ℕ} {coordinates : List Coord}
    (isIndexed : (G.over accept tail source clash relation).kind level = .indexed coordinates) :
    G.finish ≤ level ∧ tail.kind level = .indexed coordinates := by
  rw [over_kind] at isIndexed
  by_cases inside : level < G.finish
  · rw [ite_eq_left inside] at isIndexed
    unfold Segment.kind at isIndexed
    split at isIndexed <;> exact absurd isIndexed (by simp)
  · rw [ite_eq_right inside] at isIndexed
    exact ⟨Nat.le_of_not_lt inside, isIndexed⟩

theorem over_kind_checkpoint {level span : ℕ}
    (isCheckpoint : (G.over accept tail source clash relation).kind level = .checkpoint span) :
    (level = G.start ∧ span = G.costs.length - 1 ∧ level < G.finish) ∨
      (G.finish ≤ level ∧ tail.kind level = .checkpoint span) := by
  rw [over_kind] at isCheckpoint
  by_cases inside : level < G.finish
  · rw [ite_eq_left inside] at isCheckpoint
    unfold Segment.kind at isCheckpoint
    split at isCheckpoint
    · rename_i first
      exact Or.inl ⟨first, (Kind.checkpoint.inj isCheckpoint).symm, inside⟩
    · exact absurd isCheckpoint (by simp)
  · rw [ite_eq_right inside] at isCheckpoint
    exact Or.inr ⟨Nat.le_of_not_lt inside, isCheckpoint⟩

theorem over_kind_plain {level : ℕ}
    (isPlain : (G.over accept tail source clash relation).kind level = .plain) :
    (level ≠ G.start ∧ level < G.finish) ∨ (G.finish ≤ level ∧ tail.kind level = .plain) := by
  rw [over_kind] at isPlain
  by_cases inside : level < G.finish
  · rw [ite_eq_left inside] at isPlain
    unfold Segment.kind at isPlain
    split at isPlain
    · exact absurd isPlain (by simp)
    · rename_i other
      exact Or.inl ⟨other, inside⟩
  · rw [ite_eq_right inside] at isPlain
    exact Or.inr ⟨Nat.le_of_not_lt inside, isPlain⟩

omit [Fintype Slot] [Nonempty Slot] [Fintype Coord] [DecidableEq Coord] in
theorem holds_tail {level : ℕ} (outside : G.finish ≤ level) (inner : Inner) :
    G.holds tail source level inner ↔ tail.holds level inner :=
  Iff.of_eq (ite_eq_left outside)

omit [Fintype Slot] [Nonempty Slot] [Fintype Coord] [DecidableEq Coord] in
theorem holds_inside {level : ℕ} (inside : level < G.finish) (inner : Inner) :
    G.holds tail source level inner ↔
      tail.holds G.finish inner ∧ (level ≤ G.start → G.Good tail source inner) :=
  Iff.of_eq (ite_eq_right (Nat.not_le.mpr inside))

omit [Fintype Slot] [Nonempty Slot] [Fintype Coord] [DecidableEq Coord] in
theorem holds_finish {level : ℕ} (below : level ≤ G.finish) {inner : Inner}
    (held : G.holds tail source level inner) : tail.holds G.finish inner := by
  rcases Nat.lt_or_eq_of_le below with inside | rfl
  · exact ((G.holds_inside inside inner).mp held).1
  · exact (G.holds_tail le_rfl inner).mp held

omit [Nonempty Slot] in
/-- The game at the stage input of the checkpoint has no answer of a draw. -/
theorem game_take_start (candidate : Cand) (point : Prefix Wire (Coord → Slot) total) :
    G.game candidate (point.take G.start) =
      settle G.read (point.wire G.start) (G.check (point.take G.start) candidate) 0 := by
  have none : ((point.take G.start).rest.take G.finish).drop G.start = [] :=
    List.drop_eq_nil_of_le (by
      show ((point.rest.take G.start).take G.finish).length ≤ G.start
      rw [List.length_take, List.length_take]
      omega)
  unfold Segment.game
  rw [none, Prefix.take_take le_rfl, Prefix.wire_take le_rfl]
  rfl

omit [Nonempty Slot] in
/-- The game reads the stages of the segment only. -/
theorem game_take_finish (candidate : Cand) (point : Prefix Wire (Coord → Slot) total) :
    G.game candidate (point.take G.finish) = G.game candidate point := by
  have ends : G.start ≤ G.finish := Nat.le_add_right _ _
  have same : ((point.take G.finish).rest.take G.finish) = point.rest.take G.finish := by
    show (point.rest.take G.finish).take G.finish = _
    rw [List.take_take, Nat.min_self]
  unfold Segment.game
  rw [same, Prefix.take_take ends, Prefix.wire_take ends]

omit [Nonempty Slot] in
/-- The game after one more answer and one more message. -/
theorem game_snoc (candidate : Cand) {input : Prefix Wire (Coord → Slot) total}
    (move : (Coord → Slot) × Wire) (after : G.start ≤ input.rest.length)
    (before : input.rest.length < G.finish) (room : input.rest.length < total) :
    G.game candidate (input.snoc move room) =
      settle G.read move.2 (advance move.1 (G.game candidate input)) 0 := by
  have moves : ((input.snoc move room).rest.take G.finish).drop G.start =
      (input.rest.take G.finish).drop G.start ++ [move] := by
    show ((input.rest ++ [move]).take G.finish).drop G.start = _
    rw [List.take_of_length_le (by
        rw [List.length_append, List.length_singleton]
        omega),
      List.take_of_length_le (Nat.le_of_lt before), List.drop_append_of_le_length after]
  unfold Segment.game
  rw [moves, Prefix.take_snoc input move room after, Prefix.wire_snoc input move room after]
  exact position_append G.read _ _ _ _ move.1

/-- The segment over the later stages satisfies the conditions of Akita Definition 10.16 from its
checkpoint (Akita Theorem 10.45, the structure, for one segment). -/
theorem over_lawfulFrom (valid : G.Valid) (fed : G.Fed accept tail source clash) :
    (G.over accept tail source clash relation).LawfulFrom (transcript total accept) G.start := by
  have ends : G.finish = G.start + G.costs.length := rfl
  have long : 0 < G.costs.length := List.length_pos_iff.mpr valid.draws
  have fits := fed.fits
  refine {
    segment := ?_
    key_some := ?_
    key_mass := ?_
    weight_nonnegative := fed.lawful.weight_nonnegative
    leaf_point := fed.lawful.leaf_point
    leaf_holds := fun point accepted =>
      (G.holds_tail fits _).mpr (fed.lawful.leaf_holds point accepted)
    plain_holds := ?_
    combine_holds := ?_
    good_holds := ?_
    collide_breaks := ?_
    refuted_start := ?_
    refuted_step := ?_
    refuted_end := ?_
    bad_mass := ?_
    error_nonnegative := ?_ }
  · intro level span low isCheckpoint
    rcases G.over_kind_checkpoint isCheckpoint with ⟨rfl, rfl, _⟩ | ⟨outside, isTail⟩
    · refine ⟨by
        show G.start + (G.costs.length - 1) < total
        omega, fun later above below => ?_⟩
      rw [over_kind, ite_eq_left (by omega)]
      exact ite_eq_right (by omega)
    · obtain ⟨short, plains⟩ := fed.lawful.segment level span outside isTail
      refine ⟨short, fun later above below => ?_⟩
      rw [over_kind, ite_eq_right (by omega)]
      exact plains later above below
  · intro level coordinates coordinate point deeper low isIndexed member accepted found
    obtain ⟨outside, isTail⟩ := G.over_kind_indexed isIndexed
    exact fed.lawful.key_some level coordinates coordinate point deeper outside isTail member
      accepted found
  · intro level coordinates coordinate value low isIndexed member
    obtain ⟨outside, isTail⟩ := G.over_kind_indexed isIndexed
    exact fed.lawful.key_mass level coordinates coordinate value outside isTail member
  · intro level inner low isPlain held
    rcases G.over_kind_plain isPlain with ⟨other, inside⟩ | ⟨outside, isTail⟩
    · exact (G.holds_inside inside inner).mpr
        ⟨G.holds_finish (by omega) held, fun below => absurd (Nat.le_antisymm below low) other⟩
    · exact (G.holds_tail outside inner).mpr (fed.lawful.plain_holds level inner outside isTail
        ((G.holds_tail (by omega) inner).mp held))
  · intro level coordinates reference family anchor deeper low isIndexed held found foundDeeper
      each
    obtain ⟨outside, isTail⟩ := G.over_kind_indexed isIndexed
    have combined := fed.lawful.combine_holds level coordinates reference family anchor deeper
      outside isTail ((G.holds_tail (by omega) reference).mp held) found foundDeeper
      fun coordinate member => by
        obtain ⟨heldChild, rest⟩ := each coordinate member
        exact ⟨(G.holds_tail (by omega) (family coordinate)).mp heldChild, rest⟩
    show Sum.elim _ tail.breaks (tail.combine level reference family)
    generalize tail.combine level reference family = result at combined ⊢
    rcases result with inner | broken
    · exact ⟨(G.holds_tail outside inner).mpr combined.1, combined.2⟩
    · exact combined
  · intro level span inner low isCheckpoint held isGood
    rcases G.over_kind_checkpoint isCheckpoint with ⟨rfl, rfl, inside⟩ | ⟨outside, isTail⟩
    · rw [over_good, ite_eq_left inside] at isGood
      exact (G.holds_inside inside inner).mpr ⟨G.holds_finish (by omega) held, fun _ => isGood⟩
    · rw [over_good, ite_eq_right (by omega)] at isGood
      exact (G.holds_tail outside inner).mpr (fed.lawful.good_holds level span inner outside
        isTail ((G.holds_tail (by omega) inner).mp held) isGood)
  · intro level span inner other anchor low isCheckpoint held heldOther found foundOther
      different
    rcases G.over_kind_checkpoint isCheckpoint with ⟨rfl, rfl, inside⟩ | ⟨outside, isTail⟩
    · rw [over_candidate, over_candidate, ite_eq_left inside, ite_eq_left inside] at different
      rw [over_collide, ite_eq_left inside]
      exact fed.clash inner other anchor (G.holds_finish (by omega) held)
        (G.holds_finish (by omega) heldOther) found foundOther
        fun same => different (congrArg Sum.inl same)
    · rw [over_candidate, over_candidate, ite_eq_right (by omega), ite_eq_right (by omega)]
        at different
      rw [over_collide, ite_eq_right (by omega)]
      exact fed.lawful.collide_breaks level span inner other anchor outside isTail
        ((G.holds_tail (by omega) inner).mp held) ((G.holds_tail (by omega) other).mp heldOther)
        found foundOther fun same => different (congrArg Sum.inr same)
  · intro level span inner anchor low isCheckpoint held notGood found
    rcases G.over_kind_checkpoint isCheckpoint with ⟨rfl, rfl, inside⟩ | ⟨outside, isTail⟩
    · rw [over_good, ite_eq_left inside] at notGood
      rw [over_candidate, ite_eq_left inside]
      obtain ⟨_, rfl⟩ := transcript_cut found
      refine ⟨inside, ?_⟩
      rw [G.game_take_start, Prefix.take_take le_rfl]
      exact avoids_settle G.read _ (valid.sound _ _ notGood) 0
    · rw [over_good, ite_eq_right (by omega)] at notGood
      rw [over_candidate, ite_eq_right (by omega)]
      exact ⟨outside, fed.lawful.refuted_start level span inner anchor outside isTail
        ((G.holds_tail (by omega) inner).mp held) notGood found⟩
  · intro level span input candidate reply proof low isCheckpoint after before refuted outside
    rcases candidate with candidate | candidate
    · obtain ⟨inside, refuted⟩ := refuted
      rcases G.over_kind_checkpoint isCheckpoint with ⟨rfl, rfl, _⟩ | ⟨out, _⟩
      · have after : G.start ≤ input.rest.length := after
        have before : input.rest.length ≤ G.start + (G.costs.length - 1) := before
        have room : input.rest.length < total := by omega
        rw [over_bad_inl, ite_eq_left inside] at outside
        refine ⟨inside, ?_⟩
        rw [transcript_step proof input reply room, G.game_snoc candidate _ after (by omega) room,
          Prefix.take_snoc input _ room after]
        exact avoids_settle G.read _ (avoids_advance refuted outside) 0
      · exact absurd inside (by omega)
    · obtain ⟨out, refuted⟩ := refuted
      rcases G.over_kind_checkpoint isCheckpoint with ⟨rfl, rfl, inside⟩ | ⟨_, isTail⟩
      · exact absurd out (by omega)
      · rw [over_bad_inr, ite_eq_right (by omega)] at outside
        exact ⟨out, fed.lawful.refuted_step level span input candidate reply proof out isTail
          after before refuted outside⟩
  · intro level span inner input low isCheckpoint held found refuted
    rcases G.over_kind_checkpoint isCheckpoint with ⟨rfl, rfl, inside⟩ | ⟨outside, isTail⟩
    · rw [over_candidate, ite_eq_left inside] at refuted
      obtain ⟨_, avoided⟩ := refuted
      obtain ⟨result, ended, passed⟩ := fed.passed inner (G.holds_finish (by omega) held)
      obtain ⟨_, rfl⟩ := transcript_cut found
      have avoided : (G.game (source inner) ((tail.point inner).take
          (G.start + (G.costs.length - 1) + 1))).Avoids fun result =>
            ¬ G.passes (((tail.point inner).take (G.start + (G.costs.length - 1) + 1)).take
              G.start) (source inner) result := avoided
      have whole : G.start + (G.costs.length - 1) + 1 = G.finish := by omega
      rw [whole, G.game_take_finish, Prefix.take_take (Nat.le_of_lt inside), ended] at avoided
      exact avoided passed
    · rw [over_candidate, ite_eq_right (by omega)] at refuted
      exact fed.lawful.refuted_end level span inner input outside isTail
        ((G.holds_tail (by omega) inner).mp held) found refuted.2
  · intro level span input candidate low isCheckpoint after before
    rcases G.over_kind_checkpoint isCheckpoint with ⟨rfl, rfl, inside⟩ | ⟨outside, isTail⟩
    · have after : G.start ≤ input.rest.length := after
      have before : input.rest.length ≤ G.start + (G.costs.length - 1) := before
      show mass ((G.over accept tail source clash relation).bad _ input candidate) ≤
        (G.over accept tail source clash relation).error input.rest.length
      rw [over_error, ite_eq_left (by omega)]
      rcases candidate with candidate | candidate
      · rw [over_bad_inl, ite_eq_left inside]
        have staged := position_staged G.read (valid.staged (input.take G.start) candidate)
          (input.wire G.start) ((input.rest.take G.finish).drop G.start)
        have count : ((input.rest.take G.finish).drop G.start).length =
            input.rest.length - G.start := by
          rw [List.length_drop, List.length_take]
          omega
        have head : (G.costs.drop ((input.rest.take G.finish).drop G.start).length).headD 0 =
            G.costs.getD (input.rest.length - G.start) 0 := by
          rw [count, List.getD_eq_getElem?_getD, List.headD_eq_head?_getD, List.head?_drop]
        have bound := mass_badReplies_le (by
          rw [head]
          exact getD_nonnegative valid.nonnegative _) staged
        rw [head] at bound
        exact bound
      · rw [over_bad_inr, ite_eq_left inside]
        exact (le_of_eq (by simp [mass])).trans (getD_nonnegative valid.nonnegative _)
    · have after : level ≤ input.rest.length := after
      show mass ((G.over accept tail source clash relation).bad _ input candidate) ≤
        (G.over accept tail source clash relation).error input.rest.length
      rw [over_error, ite_eq_right (by omega)]
      rcases candidate with candidate | candidate
      · rw [over_bad_inl, ite_eq_right (by omega)]
        exact (le_of_eq (by simp [mass])).trans (fed.lawful.error_nonnegative _)
      · rw [over_bad_inr, ite_eq_right (by omega)]
        exact fed.lawful.bad_mass level span input candidate outside isTail after before
  · intro level
    rw [over_error]
    split
    · exact getD_nonnegative valid.nonnegative _
    · exact fed.lawful.error_nonnegative level

section Bound

theorem over_kind_at {index : ℕ} (inside : index < G.costs.length) :
    (G.over accept tail source clash relation).kind (G.start + index) =
      if index = 0 then .checkpoint (G.costs.length - 1) else .plain := by
  have ends : G.finish = G.start + G.costs.length := rfl
  rw [over_kind, ite_eq_left (by omega)]
  unfold Segment.kind
  by_cases first : index = 0
  · rw [ite_eq_left first, ite_eq_left (by omega)]
  · rw [ite_eq_right first, ite_eq_right (by omega)]

theorem over_kind_tail {level : ℕ} (outside : G.finish ≤ level) :
    (G.over accept tail source clash relation).kind level = tail.kind level := by
  rw [over_kind, ite_eq_right (by omega)]

theorem over_charge_at (fits : G.finish ≤ total) {index : ℕ} (inside : index < G.costs.length) :
    (G.over accept tail source clash relation).charge (G.start + index) =
      G.costs.getD index 0 := by
  have ends : G.finish = G.start + G.costs.length := rfl
  have error : (G.over accept tail source clash relation).error (G.start + index) =
      G.costs.getD index 0 := by
    rw [over_error, ite_eq_left (by omega), Nat.add_sub_cancel_left]
  unfold Structure.charge
  rw [ite_eq_left (show G.start + index <
      (G.over accept tail source clash relation).protocol.stages by
    show G.start + index < total
    omega), G.over_kind_at inside]
  by_cases first : index = 0
  · rw [ite_eq_left first]
    exact error
  · rw [ite_eq_right first]
    exact error

theorem over_charge_tail (stages : tail.protocol.stages = total) {level : ℕ}
    (outside : G.finish ≤ level) :
    (G.over accept tail source clash relation).charge level = tail.charge level := by
  have error : (G.over accept tail source clash relation).error level = tail.error level := by
    rw [over_error, ite_eq_right (by omega)]
  unfold Structure.charge
  rw [G.over_kind_tail outside, error, stages]
  rfl

theorem over_loops_tail {level : ℕ} (outside : G.finish ≤ level) :
    (G.over accept tail source clash relation).loops level = tail.loops level := by
  unfold Structure.loops
  rw [G.over_kind_tail outside]

theorem over_arity_tail {level : ℕ} (outside : G.finish ≤ level) :
    (G.over accept tail source clash relation).arity level = tail.arity level := by
  unfold Structure.arity
  rw [G.over_kind_tail outside]

/-- The stage errors from the checkpoint: the bounds of the draws of the game, and the stage
errors of the later stages. -/
theorem over_chargeFrom (fits : G.finish ≤ total) (stages : tail.protocol.stages = total) :
    (G.over accept tail source clash relation).chargeFrom G.start =
      G.costs.sum + tail.chargeFrom G.finish := by
  have ends : G.finish = G.start + G.costs.length := rfl
  unfold Structure.chargeFrom
  rw [stages]
  show ∑ later ∈ Finset.Ico G.start total,
    (G.over accept tail source clash relation).charge later = _
  rw [← Finset.sum_Ico_consecutive _ (show G.start ≤ G.finish by omega) fits]
  refine congrArg₂ (· + ·) ?_
    (Finset.sum_congr rfl fun level member =>
      G.over_charge_tail stages (Finset.mem_Ico.mp member).1)
  rw [Finset.sum_Ico_eq_sum_range, show G.finish - G.start = G.costs.length by omega,
    ← sum_range_getD]
  exact Finset.sum_congr rfl fun index member =>
    G.over_charge_at fits (Finset.mem_range.mp member)

/-- The extractor has one retry loop for the checkpoint, and the loops of the later stages. -/
theorem over_loopsFrom (draws : G.costs ≠ []) (fits : G.finish ≤ total)
    (stages : tail.protocol.stages = total) :
    (G.over accept tail source clash relation).loopsFrom G.start =
      1 + tail.loopsFrom G.finish := by
  have ends : G.finish = G.start + G.costs.length := rfl
  have long : 0 < G.costs.length := List.length_pos_iff.mpr draws
  have each : ∀ index ∈ Finset.range G.costs.length,
      (G.over accept tail source clash relation).loops (G.start + index) =
        if index = 0 then 1 else 0 := fun index member => by
    unfold Structure.loops
    rw [G.over_kind_at (Finset.mem_range.mp member)]
    by_cases first : index = 0
    · rw [ite_eq_left first, ite_eq_left first]
    · rw [ite_eq_right first, ite_eq_right first]
  unfold Structure.loopsFrom
  rw [stages]
  show ∑ later ∈ Finset.Ico G.start total,
    (G.over accept tail source clash relation).loops later = _
  rw [← Finset.sum_Ico_consecutive _ (show G.start ≤ G.finish by omega) fits]
  refine congrArg₂ (· + ·) ?_
    (Finset.sum_congr rfl fun level member => G.over_loops_tail (Finset.mem_Ico.mp member).1)
  rw [Finset.sum_Ico_eq_sum_range, show G.finish - G.start = G.costs.length by omega,
    Finset.sum_congr rfl each, Finset.sum_ite_eq', ite_eq_left (Finset.mem_range.mpr long)]

/-- The tree size from the checkpoint: two children at the checkpoint, and the tree size of the
later stages. -/
theorem over_treeFrom (draws : G.costs ≠ []) (fits : G.finish ≤ total)
    (stages : tail.protocol.stages = total) :
    (G.over accept tail source clash relation).treeFrom G.start = 2 * tail.treeFrom G.finish := by
  have ends : G.finish = G.start + G.costs.length := rfl
  have long : 0 < G.costs.length := List.length_pos_iff.mpr draws
  have each : ∀ index ∈ Finset.range G.costs.length,
      (G.over accept tail source clash relation).arity (G.start + index) =
        if index = 0 then 2 else 1 := fun index member => by
    unfold Structure.arity
    rw [G.over_kind_at (Finset.mem_range.mp member)]
    by_cases first : index = 0
    · rw [ite_eq_left first, ite_eq_left first]
    · rw [ite_eq_right first, ite_eq_right first]
  unfold Structure.treeFrom
  rw [stages]
  show ∏ later ∈ Finset.Ico G.start total,
    (G.over accept tail source clash relation).arity later = _
  rw [← Finset.prod_Ico_consecutive _ (show G.start ≤ G.finish by omega) fits]
  refine congrArg₂ (· * ·) ?_
    (Finset.prod_congr rfl fun level member => G.over_arity_tail (Finset.mem_Ico.mp member).1)
  rw [Finset.prod_Ico_eq_prod_range, show G.finish - G.start = G.costs.length by omega,
    Finset.prod_congr rfl each, Finset.prod_ite_eq', ite_eq_left (Finset.mem_range.mpr long)]

end Bound

variable {Wit' : Type} {Opens' : Prefix Wire (Coord → Slot) total → Wit' → Prop}
  {later : Structure (Prefix Wire (Coord → Slot) total) Coord Slot (ℕ → Wire) Inner BreakT CandT
    KeyT Wit'}
  {error : ℝ} {loops tree : ℕ}

/-- A segment before a suffix of the later stages is a suffix (Akita Theorem 10.45 for one
segment): the stage errors add the bounds of the draws of the game, the retry loops add 1, and the
tree size is multiplied by 2. `source` maps the witness of a first certificate of the later stages
to the candidate of the checkpoint. `Opens` is the relation of the first certificates of the
segment. `rooted`: it reads the transcript before the stage `start` only. `opens`: a witness of
the later stages whose candidate passes the test is a witness of `Opens`. `passed`: the game ends
in a true result for the candidate (Theorem 10.45 (c)). `clashes`: two witnesses with different
candidates, for two accepting transcripts with one stage input of the checkpoint, give a witness
of `R_bind` (Definition 10.16 (vi)). -/
theorem suffix_before (valid : G.Valid)
    (suffix : Tier.Suffix accept G.finish Opens' later error loops tree) (fits : G.finish ≤ total)
    (source : Wit' → Cand) (clash : Wit' → Wit' → BreakT)
    (Opens : Prefix Wire (Coord → Slot) total → Wit' → Prop)
    (rooted : ∀ point witness, Opens point witness → Opens (point.take G.start) witness)
    (opens : ∀ point witness, accept point → Opens' point witness →
      G.test (point.take G.start) (source witness) → Opens point witness)
    (passed : ∀ point witness, accept point → Opens' point witness →
      G.Passed (source witness) point)
    (clashes : ∀ point other witness witness', accept point → accept other →
      Opens' point witness → Opens' other witness' →
      point.take G.start = other.take G.start → source witness ≠ source witness' →
      later.breaks (clash witness witness')) :
    Tier.Suffix accept G.start Opens
      (G.over accept later (fun inner => source (later.witness inner))
        (fun inner other => clash (later.witness inner) (later.witness other)) Opens)
      (G.costs.sum + error) (1 + loops) (2 * tree) := by
  have stages : later.protocol.stages = total := by rw [suffix.protocol]; rfl
  have fed : G.Fed accept later (fun inner => source (later.witness inner))
      (fun inner other => clash (later.witness inner) (later.witness other)) :=
    { fits := fits
      lawful := suffix.lawfulFrom
      passed := fun inner held =>
        passed _ _ (suffix.opened inner held).1 (suffix.opened inner held).2
      clash := fun inner other anchor held heldOther found foundOther different => by
        obtain ⟨_, rfl⟩ := transcript_cut found
        obtain ⟨_, same⟩ := transcript_cut foundOther
        exact clashes _ _ _ _ (suffix.opened inner held).1 (suffix.opened other heldOther).1
          (suffix.opened inner held).2 (suffix.opened other heldOther).2 same different }
  exact
    { opened := fun inner held => by
        obtain ⟨heldTail, good⟩ := (G.holds_inside (by
          have long : 0 < G.costs.length := List.length_pos_iff.mpr valid.draws
          show G.start < G.start + G.costs.length
          omega) inner).mp held
        obtain ⟨accepted, opened⟩ := suffix.opened inner heldTail
        exact ⟨accepted, opens _ _ accepted opened (good le_rfl)⟩
      rooted := rooted
      lawfulFrom := G.over_lawfulFrom valid fed
      protocol := rfl
      related := fun _ _ => Iff.rfl
      errorSum := by rw [G.over_chargeFrom fits stages, suffix.errorSum]
      loopCount := by rw [G.over_loopsFrom valid.draws fits stages, suffix.loopCount]
      treeSize := by rw [G.over_treeFrom valid.draws fits stages, suffix.treeSize] }

end Segment

end NightstreamFPrime.Spec.Compression.Akita.Extract
