import NightstreamFPrime.Spec.Compression.Akita.LiftWitness
import NightstreamFPrime.Spec.Compression.Akita.Extract.BatchChain

/-! The checks of one fold of Akita on the quotient-lift route for the witness that the next
level commits to, for one group and for opening groups (ePrint 2026/1983, revision of 2026-11-23,
Section 5.5 with (76), Section 7.3 with (111) and (112), Lemma 7.1, Lemma 10.42 (i), and
Appendix L.2 with (348)).

- `OpeningShape.gaps` and `Batch.gaps` are the places of the quotient segments of Akita (76):
  `Q̂_main` with the quotient digits of the rows (102), (104), (105) and (106) after `t̂`, `Q̂_1`
  with those of `F_1` and `H_1` after the digits of `H_1`, and `Q̂_2` with those of `F_2` and
  `H_2` after the digits of `H_2`. This is the witness table of the pinned code
  (`crates/akita-params/src/witness.rs` and `witness/tail.rs` of the Akita code), with its
  alignment zeros, which the schedule files have only after the last segment
  (`tests/AkitaSchedules.lean`).
- `OpeningShape.quotientRoom` and `Batch.quotientRoom` are the end of the witness:
  `cells + δ_quot·Σ_r d_r` for the degrees `d_r` of the ring rows (`flatRelation_quotientEnd`).
  The condition `room` of a level is that this end is in the cube.
- `liftedReduction` and `Batch.liftedReduction` are the quotient-lift reductions on this witness.
  Their bad sets have at most `2·d_max − 1` points.
- `liftedFold_satisfies` and `Batch.liftedFold_satisfies` are `tableFold_satisfies` for these
  reductions: on a path of `foldCheck` that avoids the bad sets, a true final claim for the
  witness gives the exact branch relation of the digit vectors of its flat positions.
- `liftedFold_winsTo` and `Batch.liftedFold_winsTo` are the completeness of the checks, for the
  witness `FlatRelation.completion` with the digits of the quotients.
- `liftFold` and `liftBatch` are the levels `committedFold` and `committedBatch` with the
  quotient-lift reduction, and `liftFold_suffix` and `liftBatch_suffix` are their steps of the
  suffix. `liftRoot_suffix` is the batching root over `liftBatch` over a last packing fold.

The base of the quotient digits is the certified base `b⋆` of the range check, and `covers`
states that each element of `Z_q` has `digits` balanced digits. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita

open NightstreamFPrime.Spec.Compression.Lib NightstreamFPrime.Spec.Compression.Game

local instance : Fact (Nat.Prime fieldModulus) := ⟨FieldPrime.fieldModulus_natPrime⟩

section Level

variable {S : Shape} {O : OpeningShape S} {fold : Descriptor} {E : Type} [Field E] [Fintype E]
  [Algebra Zq E]

namespace OpeningShape

variable (O) (fold)

/-- The sum of the ring degrees of the rows of one fold before the first compression row: the
planes of the carrier row (102) and the rows (104), (105) and (106). -/
def mainDegrees : ℕ :=
  O.planes * O.carrierDegree + (S.innerRank * S.innerDegree +
    (S.slices * S.outerRank * S.outerDegree + 1 * O.openRank * O.openDegree))

/-- The sum of the ring degrees of the rows of the first compression maps `F_1` and `H_1`. -/
def firstDegrees : ℕ :=
  firstRank S.chain * firstDegree S.chain + firstRank O.chain * firstDegree O.chain

/-- The places of the quotient segments of one fold with `δ_quot = digits` (Akita (76)):
`Q̂_main` after `t̂`, `Q̂_1` after the digits of `H_1` and `Q̂_2` after the digits of `H_2`. -/
def gaps (digits : ℕ) : Gaps where
  main := O.chainStart fold
  middle := O.secondStart fold
  last := O.cells fold
  mainCells := digits * O.mainDegrees
  middleCells := digits * O.firstDegrees
  ordered := ⟨by unfold secondStart openFirstStart; omega, by unfold cells openSecondStart; omega⟩

/-- The end of the witness of one fold on the quotient-lift route: the flat witness and
`δ_quot·d_r` cells for each ring row `r` (Akita (76), (348)). -/
def quotientRoom (digits : ℕ) : ℕ := O.cells fold + digits * O.degreeSum

theorem gaps_counts (digits : ℕ) :
    (O.gaps fold digits).mainCells + (O.gaps fold digits).middleCells ≤ digits * O.degreeSum := by
  show digits * O.mainDegrees + digits * O.firstDegrees ≤ digits * O.degreeSum
  rw [← Nat.mul_add]
  exact Nat.mul_le_mul_left _ (by unfold mainDegrees firstDegrees degreeSum; omega)

theorem place_lt_room (digits : ℕ) {x : ℕ} (small : x < O.cells fold) :
    (O.gaps fold digits).place x < O.quotientRoom fold digits := by
  have counts := O.gaps_counts fold digits
  exact ((O.gaps fold digits).place_lt small).trans_le (by
    show O.cells fold + _ + _ ≤ _
    unfold quotientRoom
    omega)

theorem gaps_support_empty (digits : ℕ) (raw : O.compressed fold = false) (z : ℕ) :
    ¬ (O.gaps fold digits).support z :=
  (O.gaps fold digits).support_empty (Nat.le_of_not_lt (of_decide_eq_false raw)) z

end OpeningShape

variable (β : Module.Basis (Fin O.planes) Zq E) (st : Statement O E)
  (challenge : Fin S.blocks → Rq O.carrierDegree) (μ : ℕ) (fits : O.cells fold ≤ 2 ^ μ)

omit [Fintype E] in
theorem flatRelation_counts (digits : ℕ) :
    (O.gaps fold digits).mainCells + (O.gaps fold digits).middleCells ≤
      ∑ r, (flatRelation O fold β st challenge μ fits).quotientCells digits r := by
  unfold FlatRelation.quotientCells
  rw [← Finset.mul_sum]
  exact (O.gaps_counts fold digits).trans_eq
    (congrArg (digits * ·) (flatRows_degree_sum O fold β st challenge).symm)

omit [Fintype E] in
/-- The witness of `flatRelation` with the quotient digits ends at `quotientRoom`. -/
theorem flatRelation_quotientEnd (digits : ℕ) :
    (flatRelation O fold β st challenge μ fits).quotientEnd (O.gaps fold digits) digits =
      O.quotientRoom fold digits := by
  rw [FlatRelation.quotientEnd_eq]
  exact congrArg (fun total => O.cells fold + digits * total)
    (flatRows_degree_sum O fold β st challenge)

/-- The quotient-lift reduction of `flatRelation` (Akita Section 7.3 and Lemma 7.1) on the
witness of Akita (76), with the quotient digits of each ring row in the base `b⋆` with `digits`
digits, and `2^batch` places for the rows. Its bad set has at most `2·d_max − 1` points. -/
noncomputable def liftedReduction (digits : ℕ) (room : O.quotientRoom fold digits ≤ 2 ^ μ)
    (batch : ℕ) (rowsFit : O.rowCount + 1 ≤ 2 ^ batch) : Reduction E :=
  ((flatRelation O fold β st challenge μ fits).withQuotients (O.gaps fold digits) S.certifiedBase
    digits (flatRows_below O fold β st challenge) (flatRelation_counts β st challenge μ fits digits)
    ((flatRelation_quotientEnd β st challenge μ fits digits).trans_le room)).lift
    O.largestDegree batch (flatRows_degree_le O fold β st challenge) rowsFit

/-- A witness with the relation of `liftedReduction`, the digit range statement and the binary
statement on `I_bin` of the witness has the exact branch relation `Witness.Satisfies` for the
digit vectors `ofFlat` of the flat table at the positions `place`. The identities (111) give the
ring rows (110) (Akita Lemma 3.4). -/
theorem satisfies_of_lift (small : S.certifiedBase < fieldModulus) (digits : ℕ)
    (room : O.quotientRoom fold digits ≤ 2 ^ μ) (batch : ℕ) (rowsFit : O.rowCount + 1 ≤ 2 ^ batch)
    {table : Table}
    (relation : (liftedReduction (E := E) β st challenge μ fits digits room batch rowsFit).Holds
      table)
    (range : RangeHolds S.certifiedBase μ table)
    (binary : BinaryOn (O.gaps fold digits).support μ table) :
    (O.ofFlat fold fun x => table ((O.gaps fold digits).place x)).Satisfies β st challenge :=
  satisfies_of_table small
    (FlatRelation.holds_of_lift _ (fun _ large => scalarWeight_of_le O fold β st.point large)
      relation)
    (fun _ inside => range _ ((O.place_lt_room fold digits inside).trans_le room)) range.zero
    (fun _ member => binary _ ((O.place_lt_room fold digits member.2).trans_le room)
      ((O.gaps fold digits).support_place member.1 member.2))

/-- Akita Lemma 10.42 (i) on the quotient-lift route: on a path of `foldCheck` that avoids the
bad sets, a final claim that is true for the witness gives the exact branch relation
`Witness.Satisfies` of the digit vectors `ofFlat` of the flat table at the positions `place`. -/
theorem liftedFold_satisfies [DecidableEq E] (small : S.certifiedBase < fieldModulus)
    (digits : ℕ) (room : O.quotientRoom fold digits ≤ 2 ^ μ) (batch : ℕ)
    (rowsFit : O.rowCount + 1 ≤ 2 ^ batch) (table : Table) :
    (foldCheck (liftedReduction β st challenge μ fits digits room batch rowsFit) S.certifiedBase
      (O.gaps fold digits).support (O.compressed fold) table).Avoids fun result =>
      ∀ r2 v, result = some (r2, v) → v = mle r2 (lifted E table) →
        (O.ofFlat fold fun x => table ((O.gaps fold digits).place x)).Satisfies β st challenge :=
  (foldCheck_avoids (liftedReduction β st challenge μ fits digits room batch rowsFit)
    S.certifiedBase _ (O.compressed fold) (O.gaps_support_empty fold digits) table).mono
      fun _ checked r2 v equal opening =>
      satisfies_of_lift β st challenge μ fits small digits room batch rowsFit
        (checked r2 v equal opening).1 (checked r2 v equal opening).2.1
        (checked r2 v equal opening).2.2

variable {β st challenge μ fits}

/-- The completeness of the checks of a fold on the quotient-lift route (Akita Lemma 7.1,
item 1, with Lemma 10.7): for digit vectors with the exact branch relation, the prover has a
message for each draw sequence, and the checks end in the claim `(r_2, w̃(r_2))` for the witness
with the digits of the quotients. `covers`: each element of `Z_q` has `digits` balanced digits of
the base `b⋆`. -/
theorem liftedFold_winsTo [DecidableEq E] (admits : Type → Prop) (scalar : admits E)
    (range : ∀ n, n ≤ rangeWidth S.certifiedBase → admits (Fin n → E))
    (cubic : admits (Fin (3 + 1) → E)) (radix : Radix S.certifiedBase) {digits : ℕ}
    (covers : fieldModulus ≤ S.certifiedBase ^ digits)
    (room : O.quotientRoom fold digits ≤ 2 ^ μ) (batch : ℕ)
    (rowsFit : O.rowCount + 1 ≤ 2 ^ batch) {w : Witness O fold}
    (satisfies : w.Satisfies β st challenge) :
    WinsTo admits (foldCheck (liftedReduction β st challenge μ fits digits room batch rowsFit)
      S.certifiedBase (O.gaps fold digits).support (O.compressed fold)
      ((flatRelation O fold β st challenge μ fits).completion (O.gaps fold digits)
        S.certifiedBase digits w.flat)) (foldDraws μ S.certifiedBase (O.compressed fold)) true fun
          result _ _ =>
        ∃ r2 : List E, r2.length = μ ∧ result = some (r2, mle r2 (lifted E
          ((flatRelation O fold β st challenge μ fits).completion (O.gaps fold digits)
            S.certifiedBase digits w.flat))) :=
  foldCheck_winsTo admits (liftedReduction β st challenge μ fits digits room batch rowsFit)
    S.certifiedBase scalar range cubic _
    (FlatRelation.withQuotients_holds _ radix covers _ _ _
      (fun _ large => scalarWeight_of_le O fold β st.point large) (satisfies.flat μ fits))
    (FlatRelation.completion_range _ radix
      (satisfies.alphabets.rangeHolds radix.two radix.small μ))
    (fun z _ member => by
      obtain ⟨x, low, high, rfl⟩ := (O.gaps fold digits).support_cases member
      rw [FlatRelation.completion_place _ _ high]
      exact satisfies.alphabets.binaryOn radix.two μ x (high.trans_le fits) ⟨low, high⟩)

end Level

namespace Batch

variable {n : ℕ} {B : Batch (Fin n)} {E : Type} [Field E] [Fintype E] [Algebra Zq E]

variable (B) in
/-- The sum of the ring degrees of the rows of one fold with opening groups before the first
compression row: the rows `groupRows` of each group and the rows (106) of `D`. -/
def mainDegrees : ℕ := ∑ g, B.groupDegrees g + 1 * B.openRank * B.openDegree

variable (B) in
/-- The sum of the ring degrees of the rows of `F_1` of each group and of `H_1`. -/
def firstDegrees : ℕ :=
  ∑ g, firstRank (B.shape g).chain * firstDegree (B.shape g).chain +
    firstRank B.chain * firstDegree B.chain

variable (B) in
/-- The places of the quotient segments of one fold with opening groups (Akita (76)): `Q̂_main`
after the main segments of the groups, `Q̂_1` after the digits of `H_1` and `Q̂_2` after the
digits of `H_2`. -/
def gaps (digits : ℕ) : Gaps where
  main := B.chainStart
  middle := B.secondStart
  last := B.cells
  mainCells := digits * B.mainDegrees
  middleCells := digits * B.firstDegrees
  ordered := ⟨by unfold secondStart openFirstStart; omega, by unfold cells openSecondStart; omega⟩

variable (B) in
/-- The end of the witness of one fold with opening groups on the quotient-lift route: the flat
witness and `δ_quot·d_r` cells for each ring row `r` (Akita (76), (348)). -/
def quotientRoom (digits : ℕ) : ℕ := B.cells + digits * B.degreeSum

theorem gaps_counts (digits : ℕ) :
    (B.gaps digits).mainCells + (B.gaps digits).middleCells ≤ digits * B.degreeSum := by
  show digits * B.mainDegrees + digits * B.firstDegrees ≤ digits * B.degreeSum
  rw [← Nat.mul_add]
  exact Nat.mul_le_mul_left _ (by unfold mainDegrees firstDegrees degreeSum; omega)

theorem place_lt_room (digits : ℕ) {x : ℕ} (small : x < B.cells) :
    (B.gaps digits).place x < B.quotientRoom digits := by
  have counts := B.gaps_counts digits
  exact ((B.gaps digits).place_lt small).trans_le (by
    show B.cells + _ + _ ≤ _
    unfold quotientRoom
    omega)

theorem gaps_support_empty (digits : ℕ) (raw : B.compressed = false) (z : ℕ) :
    ¬ (B.gaps digits).support z :=
  (B.gaps digits).support_empty (Nat.le_of_not_lt (of_decide_eq_false raw)) z

variable (β : ∀ g, Module.Basis (Fin (B.planes g)) Zq E) (st : B.Statement E)
  (challenge : B.Challenge) (μ : ℕ) (fits : B.cells ≤ 2 ^ μ)

omit [Fintype E] in
theorem flatRelation_counts (digits : ℕ) :
    (B.gaps digits).mainCells + (B.gaps digits).middleCells ≤
      ∑ r, (B.flatRelation β st challenge μ fits).quotientCells digits r := by
  unfold FlatRelation.quotientCells
  rw [← Finset.mul_sum]
  exact (B.gaps_counts digits).trans_eq
    (congrArg (digits * ·) (B.flatRows_degree_sum β st challenge).symm)

omit [Fintype E] in
/-- The witness of `Batch.flatRelation` with the quotient digits ends at `Batch.quotientRoom`. -/
theorem flatRelation_quotientEnd (digits : ℕ) :
    (B.flatRelation β st challenge μ fits).quotientEnd (B.gaps digits) digits =
      B.quotientRoom digits := by
  rw [FlatRelation.quotientEnd_eq]
  exact congrArg (fun total => B.cells + digits * total) (B.flatRows_degree_sum β st challenge)

/-- The quotient-lift reduction of `Batch.flatRelation` (Akita Section 7.3 and Lemma 7.1) on the
witness of Akita (76), with the quotient digits of each ring row in the base `base` with `digits`
digits. Its bad set has at most `2·d_max − 1` points. -/
noncomputable def liftedReduction (base digits : ℕ) (room : B.quotientRoom digits ≤ 2 ^ μ)
    (batch : ℕ) (rowsFit : B.rowCount + 1 ≤ 2 ^ batch) : Reduction E :=
  ((B.flatRelation β st challenge μ fits).withQuotients (B.gaps digits) base digits
    (B.flatRows_below β st challenge) (flatRelation_counts β st challenge μ fits digits)
    ((flatRelation_quotientEnd β st challenge μ fits digits).trans_le room)).lift
    B.largestDegree batch (B.flatRows_degree_le β st challenge) rowsFit

variable {base : ℕ}

/-- A witness with the relation of `Batch.liftedReduction`, the digit range statement and the
binary statement on `I_bin` of the witness has the exact branch relation
`Batch.Witness.Satisfies` for the digit vectors `Batch.ofFlat` of the flat table at the positions
`place`. Each group has the certified base `base`. -/
theorem satisfies_of_lift (uniform : ∀ g, (B.shape g).certifiedBase = base)
    (small : base < fieldModulus) (digits : ℕ) (room : B.quotientRoom digits ≤ 2 ^ μ)
    (batch : ℕ) (rowsFit : B.rowCount + 1 ≤ 2 ^ batch) {table : Table}
    (relation : (liftedReduction (E := E) β st challenge μ fits base digits room batch
      rowsFit).Holds table)
    (range : RangeHolds base μ table) (binary : BinaryOn (B.gaps digits).support μ table) :
    (B.ofFlat fun x => table ((B.gaps digits).place x)).Satisfies β st challenge :=
  satisfies_of_table β st challenge μ fits uniform small
    (FlatRelation.holds_of_lift _ (fun _ large => B.scalarWeight_of_le β st μ large) relation)
    (fun _ inside => range _ ((B.place_lt_room digits inside).trans_le room)) range.zero
    (fun _ member => binary _ ((B.place_lt_room digits member.2).trans_le room)
      ((B.gaps digits).support_place member.1 member.2))

/-- Akita Lemma 10.42 (i) on the quotient-lift route for a fold with opening groups: on a path of
`foldCheck` that avoids the bad sets, a final claim that is true for the witness gives the exact
branch relation `Batch.Witness.Satisfies` of the digit vectors `Batch.ofFlat` of the flat table at
the positions `place`. -/
theorem liftedFold_satisfies [DecidableEq E] (uniform : ∀ g, (B.shape g).certifiedBase = base)
    (small : base < fieldModulus) (digits : ℕ) (room : B.quotientRoom digits ≤ 2 ^ μ)
    (batch : ℕ) (rowsFit : B.rowCount + 1 ≤ 2 ^ batch) (table : Table) :
    (foldCheck (liftedReduction β st challenge μ fits base digits room batch rowsFit) base
      (B.gaps digits).support B.compressed table).Avoids fun result =>
      ∀ r2 v, result = some (r2, v) → v = mle r2 (lifted E table) →
        (B.ofFlat fun x => table ((B.gaps digits).place x)).Satisfies β st challenge :=
  (foldCheck_avoids (liftedReduction β st challenge μ fits base digits room batch rowsFit) base
    _ B.compressed (B.gaps_support_empty digits) table).mono fun _ checked r2 v equal opening =>
      satisfies_of_lift β st challenge μ fits uniform small digits room batch rowsFit
        (checked r2 v equal opening).1 (checked r2 v equal opening).2.1
        (checked r2 v equal opening).2.2

variable {β st challenge μ fits}

/-- The completeness of the checks of a fold with opening groups on the quotient-lift route
(Akita Lemma 7.1, item 1, with Lemma 10.7): for digit vectors with the exact branch relation, the
checks end in the claim `(r_2, w̃(r_2))` for the witness with the digits of the quotients. -/
theorem liftedFold_winsTo [DecidableEq E] (admits : Type → Prop) (scalar : admits E)
    (range : ∀ n, n ≤ rangeWidth base → admits (Fin n → E)) (cubic : admits (Fin (3 + 1) → E))
    (uniform : ∀ g, (B.shape g).certifiedBase = base) (radix : Radix base) {digits : ℕ}
    (covers : fieldModulus ≤ base ^ digits)
    (room : B.quotientRoom digits ≤ 2 ^ μ) (batch : ℕ)
    (rowsFit : B.rowCount + 1 ≤ 2 ^ batch) {w : B.Witness}
    (satisfies : w.Satisfies β st challenge) :
    WinsTo admits (foldCheck (liftedReduction β st challenge μ fits base digits room batch
      rowsFit) base (B.gaps digits).support B.compressed
      ((B.flatRelation β st challenge μ fits).completion (B.gaps digits) base digits w.flat))
      (foldDraws μ base B.compressed) true fun result _ _ =>
        ∃ r2 : List E, r2.length = μ ∧ result = some (r2, mle r2 (lifted E
          ((B.flatRelation β st challenge μ fits).completion (B.gaps digits) base digits
            w.flat))) :=
  foldCheck_winsTo admits (liftedReduction β st challenge μ fits base digits room batch rowsFit)
    base scalar range cubic _
    (FlatRelation.withQuotients_holds _ radix covers _ _ _
      (fun _ large => B.scalarWeight_of_le β st μ large) (satisfies.flat μ fits))
    (FlatRelation.completion_range _ radix
      (satisfies.alphabets.rangeHolds uniform radix.two radix.small μ))
    (fun z _ member => by
      obtain ⟨x, low, high, rfl⟩ := (B.gaps digits).support_cases member
      rw [FlatRelation.completion_place _ _ high]
      exact satisfies.alphabets.binaryOn uniform radix.two μ x (high.trans_le fits) ⟨low, high⟩)

end Batch

namespace Extract

open scoped BigOperators
open NightstreamFPrime.Spec.Compression (Game)
open NightstreamFPrime.Spec.Checkpoint

attribute [local instance low] Classical.propDecidable

section Folds

variable {S : Shape} {O : OpeningShape S} {fold : Descriptor} {E : Type} [Field E] [Fintype E]
  [DecidableEq E] [Algebra Zq E] {Wire Coord : Type} [Fintype Coord] [DecidableEq Coord]

/-- The level `committedFold` on the quotient-lift route: the checks are `foldCheck` for
`liftedReduction`, the table of the next level is the witness of Akita (76) with the quotient
digits, and the digit vectors are those of the flat table at the positions `place`. -/
noncomputable def liftFold (β : Module.Basis (Fin O.planes) Zq E) (key : Key S)
    (openKey : OpeningKey O) (claim : Wire → Claim O E) (μ : ℕ) (fits : O.cells fold ≤ 2 ^ μ)
    (digits : ℕ) (room : O.quotientRoom fold digits ≤ 2 ^ μ) (batch : ℕ)
    (rowsFit : O.rowCount + 1 ≤ 2 ^ batch) (block : Fin S.blocks → Coord)
    (decode : E → Option (Rq O.carrierDegree)) (read : Reader Wire) (weight start : ℕ) :
    Tier (Fin S.blocks) (fun _ => Rq O.carrierDegree) (Claim O E)
      (LevelBreak key openKey (2 * (2 * weight) * fold.sourceBound S)) (Witness O fold) (Weak O)
      Wire Coord E (Option (List E × E)) Table where
  method := packing β key openKey weight
  claim := claim
  block b := some (block b)
  decode := fun _ => decode
  read := read
  digits table := O.ofFlat fold fun x => table ((O.gaps fold digits).place x)
  check wire challenge table :=
    foldCheck (liftedReduction β ((claim wire).statement key openKey) challenge μ fits digits
      room batch rowsFit) S.certifiedBase (O.gaps fold digits).support (O.compressed fold) table
  passes table result := ∃ r2 v, result = some (r2, v) ∧ v = mle r2 (lifted E table)
  costs := foldCosts E (2 * O.largestDegree - 1) μ batch S.certifiedBase (O.compressed fold)
  bits := (foldLosses (2 * S.innerDegree - 1) μ batch S.certifiedBase (O.compressed fold)).map
    workBits
  foldBits := foldWorkBits S.blocks (setBits false O.carrierDegree)
  start := start

variable {β : Module.Basis (Fin O.planes) Zq E} {key : Key S} {openKey : OpeningKey O}
  {claim : Wire → Claim O E} {μ : ℕ} {fits : O.cells fold ≤ 2 ^ μ} {digits : ℕ}
  {room : O.quotientRoom fold digits ≤ 2 ^ μ} {batch : ℕ}
  {rowsFit : O.rowCount + 1 ≤ 2 ^ batch} {block : Fin S.blocks → Coord}
  {decode : E → Option (Rq O.carrierDegree)} {read : Reader Wire} {weight start : ℕ}

/-- A level on the quotient-lift route satisfies the conditions of `Tier.Valid`, with the
hypotheses of `committedFold_valid`. -/
theorem liftFold_valid [Nonempty Coord] {peak : ℕ} (injective : Function.Injective block)
    (units : UnitDifferences {challenge | ∃ slot, decode slot = some challenge})
    (sparse : ∀ challenge, (∃ slot, decode slot = some challenge) → Sparse weight peak challenge)
    (base : 2 ≤ fold.responseBase) (small : S.certifiedBase < fieldModulus)
    (varsFit : μ ≤ Fintype.card Coord) (batchFit : batch ≤ Fintype.card Coord) :
    (liftFold β key openKey claim μ fits digits room batch rowsFit block decode read weight
      start).Valid where
  injective := (Option.some_injective _).comp injective
  avoid _ := Option.some_ne_none _
  merges := packing_sound units sparse base
  sound wire challenge table :=
    (liftedFold_satisfies β ((claim wire).statement key openKey) challenge μ fits small digits
      room batch rowsFit table).mono fun _ implies passed => by
        obtain ⟨r2, v, ended, opened⟩ := passed
        exact implies r2 v ended opened
  staged wire challenge table :=
    foldCheck_staged (liftedReduction β ((claim wire).statement key openKey) challenge μ fits
      digits room batch rowsFit) S.certifiedBase (O.gaps fold digits).support (O.compressed fold)
        table
        varsFit batchFit
  nonnegative := foldCosts_nonnegative _ _ _ _ _
  draws := by simp [liftFold, foldCosts]
  aligned := (List.length_map ..).trans (foldLosses_length ..)

/-- Akita Proposition 10.44 for one more level on the quotient-lift route: `committedFold_suffix`
with the checks for `liftedReduction`. The stage errors add `r·p + ε_fold` with `foldError` of
the quotient-lift reduction, which has `2·d_max − 1` in the place of `d_max − 1`. -/
theorem liftFold_suffix [Nonempty Coord] {total : ℕ}
    {accept : Prefix Wire (Option Coord → E) total → Prop} {Inner BreakT CandT KeyT : Type}
    {S' : Shape} {O' : OpeningShape S'} {fold' : Descriptor}
    {Result' Source' : Type} {β' : Module.Basis (Fin O'.planes) Zq E} {key' : Key S'}
    {openKey' : OpeningKey O'} {weight' : ℕ}
    {N : Tier (Fin S'.blocks) (fun _ => Rq O'.carrierDegree) (Claim O' E)
      (LevelBreak key' openKey' (2 * (2 * weight') * fold'.sourceBound S')) (Witness O' fold')
      (Weak O') Wire Coord E Result' Source'}
    {tail : Structure (Address Wire total) (Prefix Wire (Option Coord → E) total) (Option Coord) E (ℕ → Wire) Inner
      BreakT CandT KeyT
      (Option (Weak O'))}
    {error : ℝ} {loops tree : ℕ} {peak : ℕ} (injective : Function.Injective block)
    (units : UnitDifferences {challenge | ∃ slot, decode slot = some challenge})
    (sparse : ∀ challenge, (∃ slot, decode slot = some challenge) → Sparse weight peak challenge)
    (base : 2 ≤ fold.responseBase) (small : S.certifiedBase < fieldModulus)
    (varsFit : μ ≤ Fintype.card Coord) (batchFit : batch ≤ Fintype.card Coord)
    (packed : N.method = packing β' key' openKey' weight')
    (next : N.start =
      (liftFold β key openKey claim μ fits digits room batch rowsFit block decode read weight
        start).finish)
    (commit : Wire → S'.Payload)
    (inside : (liftFold β key openKey claim μ fits digits room batch rowsFit block decode read
      weight start).finish ≤ total)
    (linked : ∀ point, accept point →
      (liftFold β key openKey claim μ fits digits room batch rowsFit block decode read weight
        start).Linked N commit point)
    (embed : LevelBreak key' openKey' (2 * (2 * weight') * fold'.sourceBound S') → BreakT)
    (embedded : ∀ solution, solution.isSome → tail.breaks (embed solution))
    (later : Tier.Suffix accept N.start N.Opens tail error loops tree) :
    Tier.Suffix accept start
      (liftFold β key openKey claim μ fits digits room batch rowsFit block decode read weight
        start).Opens
      ((liftFold β key openKey claim μ fits digits room batch rowsFit block decode read weight
        start).over accept tail (fun inner => tableOf (tail.witness inner))
        (fun inner other => embed (Tier.collide key' openKey'
          (2 * (2 * weight') * fold'.sourceBound S') (tail.witness inner) (tail.witness other))))
      ((liftFold β key openKey claim μ fits digits room batch rowsFit block decode read
          weight start).levelError + error)
      (S.blocks + 1 + loops) (2 * (S.blocks + 1) * tree) := by
  exact Tier.suffix_push (Fintype.card_fin _)
    (liftFold_valid injective units sparse base small varsFit batchFit) packed next commit
    inside (fun _ _ table other => foldCheck_bare _ _ _ _ table other)
    (fun _ r2 v opened => ⟨r2, v, rfl, opened⟩) linked embed embedded later

end Folds

section Groups

variable {n : ℕ} {B : Batch (Fin n)} {E : Type} [Field E] [Fintype E] [DecidableEq E]
  [Algebra Zq E] {Wire Coord : Type} [Fintype Coord] [DecidableEq Coord]

/-- The level `committedBatch` on the quotient-lift route: the checks are `foldCheck` for
`Batch.liftedReduction`, the table of the next level is the witness of Akita (76) with the
quotient digits, and the digit vectors are those of the flat table at the positions `place`. -/
noncomputable def liftBatch (β : ∀ g, Module.Basis (Fin (B.planes g)) Zq E)
    (key : (g : Fin n) → Key (B.shape g)) (openKey : B.OpenKey) (claim : Wire → B.Claim E)
    (μ : ℕ) (fits : B.cells ≤ 2 ^ μ) (base digits : ℕ) (room : B.quotientRoom digits ≤ 2 ^ μ)
    (batch : ℕ) (rowsFit : B.rowCount + 1 ≤ 2 ^ batch) (block : B.Block → Coord)
    (decode : (g : Fin n) → E → Option (Rq (B.carrierDegree g))) (read : Reader Wire)
    (mass : Fin n → ℕ) (start : ℕ) :
    Tier B.Block (fun b => Rq (B.carrierDegree b.1)) (B.Claim E)
      (B.LevelBreak key openKey fun g => 2 * (2 * mass g) * (B.fold g).sourceBound (B.shape g))
      B.Witness B.Weak Wire Coord E (Option (List E × E)) Table where
  method := Batch.method β key openKey mass
  claim := claim
  block b := some (block b)
  decode := fun b => decode b.1
  read := read
  digits table := B.ofFlat fun x => table ((B.gaps digits).place x)
  check wire challenge table :=
    foldCheck (Batch.liftedReduction β ((claim wire).statement key openKey)
      (fun g i => challenge ⟨g, i⟩) μ fits base digits room batch rowsFit) base
        (B.gaps digits).support B.compressed table
  passes table result := ∃ r2 v, result = some (r2, v) ∧ v = mle r2 (lifted E table)
  costs := foldCosts E (2 * B.largestDegree - 1) μ batch base B.compressed
  bits := (foldLosses (2 * B.innerMax - 1) μ batch base B.compressed).map workBits
  foldBits := foldWorkBits (Fintype.card B.Block) B.familyBits
  start := start

variable {β : ∀ g, Module.Basis (Fin (B.planes g)) Zq E} {key : (g : Fin n) → Key (B.shape g)}
  {openKey : B.OpenKey} {claim : Wire → B.Claim E} {μ : ℕ} {fits : B.cells ≤ 2 ^ μ}
  {base digits : ℕ} {room : B.quotientRoom digits ≤ 2 ^ μ} {batch : ℕ}
  {rowsFit : B.rowCount + 1 ≤ 2 ^ batch} {block : B.Block → Coord}
  {decode : (g : Fin n) → E → Option (Rq (B.carrierDegree g))} {read : Reader Wire}
  {mass : Fin n → ℕ} {start : ℕ}

/-- A level with opening groups on the quotient-lift route satisfies the conditions of
`Tier.Valid`, with the hypotheses of `committedBatch_valid`. -/
theorem liftBatch_valid [Nonempty Coord] {peak : Fin n → ℕ}
    (injective : Function.Injective block)
    (units : ∀ g, UnitDifferences {challenge | ∃ slot, decode g slot = some challenge})
    (sparse : ∀ g challenge, (∃ slot, decode g slot = some challenge) →
      Sparse (mass g) (peak g) challenge)
    (response : ∀ g, 2 ≤ (B.fold g).responseBase)
    (uniform : ∀ g, (B.shape g).certifiedBase = base) (small : base < fieldModulus)
    (varsFit : μ ≤ Fintype.card Coord) (batchFit : batch ≤ Fintype.card Coord) :
    (liftBatch β key openKey claim μ fits base digits room batch rowsFit block decode read mass
      start).Valid where
  injective := (Option.some_injective _).comp injective
  avoid _ := Option.some_ne_none _
  merges := Batch.method_sound units sparse response
  sound wire challenge table :=
    (Batch.liftedFold_satisfies β ((claim wire).statement key openKey)
      (fun g i => challenge ⟨g, i⟩) μ fits uniform small digits room batch rowsFit table).mono
        fun _ implies passed => by
          obtain ⟨r2, v, ended, opened⟩ := passed
          exact implies r2 v ended opened
  staged wire challenge table :=
    foldCheck_staged (Batch.liftedReduction β ((claim wire).statement key openKey)
      (fun g i => challenge ⟨g, i⟩) μ fits base digits room batch rowsFit) base
        (B.gaps digits).support B.compressed table varsFit batchFit
  nonnegative := foldCosts_nonnegative _ _ _ _ _
  draws := by simp [liftBatch, foldCosts]
  aligned := (List.length_map ..).trans (foldLosses_length ..)

/-- Akita Proposition 10.44 for one more level with opening groups on the quotient-lift route:
`committedBatch_suffix` with the checks for `Batch.liftedReduction`. -/
theorem liftBatch_suffix [Nonempty Coord] {total : ℕ}
    {accept : Prefix Wire (Option Coord → E) total → Prop} {Inner BreakT CandT KeyT : Type}
    {S' : Shape} {O' : OpeningShape S'} {fold' : Descriptor}
    {Result' Source' : Type} {β' : Module.Basis (Fin O'.planes) Zq E} {key' : Key S'}
    {openKey' : OpeningKey O'} {weight' : ℕ}
    {N : Tier (Fin S'.blocks) (fun _ => Rq O'.carrierDegree) (Claim O' E)
      (LevelBreak key' openKey' (2 * (2 * weight') * fold'.sourceBound S')) (Witness O' fold')
      (Weak O') Wire Coord E Result' Source'}
    {tail : Structure (Address Wire total) (Prefix Wire (Option Coord → E) total) (Option Coord) E (ℕ → Wire) Inner
      BreakT CandT KeyT
      (Option (Weak O'))}
    {error : ℝ} {loops tree : ℕ} {peak : Fin n → ℕ} (injective : Function.Injective block)
    (units : ∀ g, UnitDifferences {challenge | ∃ slot, decode g slot = some challenge})
    (sparse : ∀ g challenge, (∃ slot, decode g slot = some challenge) →
      Sparse (mass g) (peak g) challenge)
    (response : ∀ g, 2 ≤ (B.fold g).responseBase)
    (uniform : ∀ g, (B.shape g).certifiedBase = base) (small : base < fieldModulus)
    (varsFit : μ ≤ Fintype.card Coord) (batchFit : batch ≤ Fintype.card Coord)
    (packed : N.method = packing β' key' openKey' weight')
    (next : N.start =
      (liftBatch β key openKey claim μ fits base digits room batch rowsFit block decode read mass
        start).finish)
    (commit : Wire → S'.Payload)
    (inside : (liftBatch β key openKey claim μ fits base digits room batch rowsFit block decode
      read mass start).finish ≤ total)
    (linked : ∀ point, accept point →
      (liftBatch β key openKey claim μ fits base digits room batch rowsFit block decode read mass
        start).Linked N commit point)
    (embed : LevelBreak key' openKey' (2 * (2 * weight') * fold'.sourceBound S') → BreakT)
    (embedded : ∀ solution, solution.isSome → tail.breaks (embed solution))
    (later : Tier.Suffix accept N.start N.Opens tail error loops tree) :
    Tier.Suffix accept start
      (liftBatch β key openKey claim μ fits base digits room batch rowsFit block decode read mass
        start).Opens
      ((liftBatch β key openKey claim μ fits base digits room batch rowsFit block decode read mass
        start).over accept tail (fun inner => tableOf (tail.witness inner))
        (fun inner other => embed (Tier.collide key' openKey'
          (2 * (2 * weight') * fold'.sourceBound S') (tail.witness inner) (tail.witness other))))
      ((liftBatch β key openKey claim μ fits base digits room batch rowsFit block decode
          read mass start).levelError + error)
      (∑ g, (B.shape g).blocks + 1 + loops) (2 * (∑ g, (B.shape g).blocks + 1) * tree) := by
  exact Tier.suffix_push B.card_block
    (liftBatch_valid injective units sparse response uniform small varsFit batchFit) packed next
    commit inside (fun _ _ table other => foldCheck_bare _ _ _ _ table other)
    (fun _ r2 v opened => ⟨r2, v, rfl, opened⟩) linked embed embedded later

/-- The batching root over a level with opening groups on the quotient-lift route over a last
packing fold is a suffix from the stage 0: `batchRootChain_suffix` with the checks for
`Batch.liftedReduction`. `Tier.Suffix.extract_sound`, `Tier.Suffix.extraction_le` and
`Tier.Suffix.runs_le` are its extraction theorem. -/
theorem liftRoot_suffix [Nonempty Coord] {S' : Shape} {O' : OpeningShape S'}
    {fold' : Descriptor} {β' : Module.Basis (Fin O'.planes) Zq E} {key' : Key S'}
    {openKey' : OpeningKey O'} {claim' : Wire → Claim O' E} {μ' : ℕ}
    {fits' : O'.cells fold' ≤ 2 ^ μ'} {batch' : ℕ} {rowsFit' : O'.rowCount + 1 ≤ 2 ^ batch'}
    {block' : Fin S'.blocks → Coord} {decode' : E → Option (Rq O'.carrierDegree)} {weight' : ℕ}
    {commit : Wire → (g : Fin n) → (B.shape g).Payload} {value : Wire → Fin n → E}
    {place : Wire → Fin n → List E} {next : Wire → S'.Payload}
    {witnessOf : Wire → Witness O' fold'} {peak : Fin n → ℕ} {peak' : ℕ}
    (injective : Function.Injective block)
    (units : ∀ g, UnitDifferences {challenge | ∃ slot, decode g slot = some challenge})
    (sparse : ∀ g challenge, (∃ slot, decode g slot = some challenge) →
      Sparse (mass g) (peak g) challenge)
    (response : ∀ g, 2 ≤ (B.fold g).responseBase)
    (uniform : ∀ g, (B.shape g).certifiedBase = base) (small : base < fieldModulus)
    (varsFit : μ ≤ Fintype.card Coord) (batchFit : batch ≤ Fintype.card Coord)
    (groupsFit : n ≤ Fintype.card Coord) (injective' : Function.Injective block')
    (units' : UnitDifferences {challenge | ∃ slot, decode' slot = some challenge})
    (sparse' : ∀ challenge, (∃ slot, decode' slot = some challenge) →
      Sparse weight' peak' challenge)
    (base' : 2 ≤ fold'.responseBase) (small' : S'.certifiedBase < fieldModulus)
    (varsFit' : μ' ≤ Fintype.card Coord) (batchFit' : batch' ≤ Fintype.card Coord) :
    Tier.Suffix
      (rootAccepts
        (liftBatch β key openKey claim μ fits base digits room batch rowsFit block decode read
          mass 1)
        (clearFold β' key' openKey' claim' μ' fits' batch' rowsFit' block' decode' read weight'
          (liftBatch β key openKey claim μ fits base digits room batch rowsFit block decode read
            mass 1).finish) commit value place read next witnessOf) 0
      (RootOpens commit value place key mass)
      (batchRoot
        (liftBatch β key openKey claim μ fits base digits room batch rowsFit block decode read
          mass 1)
        (clearFold β' key' openKey' claim' μ' fits' batch' rowsFit' block' decode' read weight'
          (liftBatch β key openKey claim μ fits base digits room batch rowsFit block decode read
            mass 1).finish) commit value place read next witnessOf)
      (1 / Fintype.card E / 2 ^ Root.rootBits n +
        ((liftBatch β key openKey claim μ fits base digits room batch rowsFit block decode
            read mass 1).levelError +
          (clearFold β' key' openKey' claim' μ' fits' batch' rowsFit' block' decode' read
              weight' (liftBatch β key openKey claim μ fits base digits room batch rowsFit block
                decode read mass 1).finish).levelError))
      (1 + (∑ g, (B.shape g).blocks + 1 + (S'.blocks + 1)))
      (2 * (2 * (∑ g, (B.shape g).blocks + 1) * (2 * (S'.blocks + 1)))) :=
  batchRoot_suffix B.card_block
    (liftBatch_valid injective units sparse response uniform small varsFit batchFit)
    (clearFold_valid injective' units' sparse' base' small' varsFit' batchFit')
    (exists_fair_slots groupsFit) rfl rfl rfl rfl
    (fun _ _ table other => foldCheck_bare _ _ _ _ table other)
    (fun _ r2 v opened => ⟨r2, v, rfl, opened⟩)

end Groups

end Extract

end NightstreamFPrime.Spec.Compression.Akita
