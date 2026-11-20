import NightstreamFPrime.Spec.Compression.Akita.LiftWitness
import NightstreamFPrime.Spec.Compression.Akita.Extract.BatchChain

/-! The checks of one fold of Akita on the quotient-lift route for a flat witness that the next
level commits to, for one group and for opening groups (ePrint 2026/1983, revision of 2026-10-15,
Section 5.5 with (76), Section 7.3 with (111) and (112), Lemma 7.1, Lemma 10.42 (i), and
Appendix L.2 with (348)).

- `liftedReduction` is the quotient-lift reduction of `lockedRelation` with the quotient digits
  of `FlatRelation.withQuotients` in the cells after `OpeningShape.cells`. Its bad set has at most
  `2·d_max − 1` points. `Batch.liftedReduction` is the reduction for opening groups.
- `liftedFold_satisfies` and `Batch.liftedFold_satisfies` are `lockedFold_satisfies` for this
  reduction: on a path of `foldCheck` that avoids the bad sets, a true final claim for the table
  gives the exact branch relation of the digit vectors of the table.
- `liftedFold_winsTo` and `Batch.liftedFold_winsTo` are the completeness of the checks, for the
  table `FlatRelation.completion` with the digits of the quotients.
- `liftFold` and `liftBatch` are the levels `committedFold` and `committedBatch` with the
  quotient-lift reduction, and `liftFold_suffix` and `liftBatch_suffix` are their steps of the
  suffix. `liftRoot_suffix` is the batching root over `liftBatch` over a last packing fold.

The quotient segments are after the zero cell `cells`, not in the places of Akita (76)
(`LiftWitness.lean`). The base of the quotient digits is the certified base `b⋆` of the range
check, and `covers` states that each element of `Z_q` has `digits` balanced digits. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita

open NightstreamFPrime.Spec.Compression.Lib NightstreamFPrime.Spec.Compression.Game

local instance : Fact (Nat.Prime fieldModulus) := ⟨FieldPrime.fieldModulus_natPrime⟩

section Level

variable {S : Shape} {O : OpeningShape S} {fold : Descriptor} {E : Type} [Field E] [Fintype E]
  [Algebra Zq E]

/-- A number of cells that has the flat witness, the zero cell and the quotient digits of one
fold: `δ_quot·d_max` cells for each ring row. -/
def OpeningShape.quotientRoom (O : OpeningShape S) (fold : Descriptor) (digits : ℕ) : ℕ :=
  O.cells fold + 1 + (O.rowCount + 1) * (digits * O.largestDegree)

variable (β : Module.Basis (Fin O.planes) Zq E) (st : Statement O E)
  (challenge : Fin S.blocks → Rq O.carrierDegree) (μ : ℕ) (fits : O.cells fold < 2 ^ μ)

/-- The quotient-lift reduction of `lockedRelation` (Akita Section 7.3 and Lemma 7.1), with the
quotient digits of each ring row in the cells from `cells + 1` (Akita (348)), in the base `b⋆`
with `digits` digits, and `2^batch` places for the rows. Its bad set has at most `2·d_max − 1`
points. -/
noncomputable def liftedReduction (digits : ℕ) (room : O.quotientRoom fold digits ≤ 2 ^ μ)
    (batch : ℕ) (rowsFit : O.rowCount + 1 + 1 ≤ 2 ^ batch) : Reduction E :=
  ((lockedRelation O fold β st challenge μ fits).withQuotients S.certifiedBase digits
    (O.cells fold + 1)
    (((lockedRelation O fold β st challenge μ fits).quotientEnd_le digits (O.cells fold + 1)
      (lockedRows_degree_le O fold β st challenge)).trans room)).lift
    (algebraMap Zq E).injective O.largestDegree batch (lockedRows_degree_le O fold β st challenge)
    rowsFit

/-- Akita Lemma 10.42 (i) on the quotient-lift route: on a path of `foldCheck` that avoids the
bad sets, a final claim that is true for the table gives the exact branch relation
`Witness.Satisfies` of the digit vectors `ofFlat` of the table. The identities (111) give the
ring rows (110) (Akita Lemma 3.4). -/
theorem liftedFold_satisfies [DecidableEq E] (small : S.certifiedBase < fieldModulus)
    (digits : ℕ) (room : O.quotientRoom fold digits ≤ 2 ^ μ) (batch : ℕ)
    (rowsFit : O.rowCount + 1 + 1 ≤ 2 ^ batch) (table : Table) :
    (foldCheck (liftedReduction β st challenge μ fits digits room batch rowsFit) S.certifiedBase
      (O.binarySupport fold) table).Avoids fun result =>
      ∀ r2 v, result = some (r2, v) → v = mle r2 (lifted E table) →
        (O.ofFlat fold table).Satisfies β st challenge :=
  (foldCheck_avoids (liftedReduction β st challenge μ fits digits room batch rowsFit)
    S.certifiedBase (O.binarySupport fold) table).mono fun _ checked r2 v equal opening =>
      satisfies_of_locked β st challenge μ fits small
        (LiftRelation.holds_flat _ (checked r2 v equal opening).1)
        (checked r2 v equal opening).2.1 (checked r2 v equal opening).2.2

variable {β st challenge μ fits}

/-- The completeness of the checks of a fold on the quotient-lift route (Akita Lemma 7.1,
item 1, with Lemma 10.7): for digit vectors with the exact branch relation, the prover has a
message for each draw sequence, and the checks end in the claim `(r_2, w̃(r_2))` for the flat
witness with the digits of the quotients. `covers`: each element of `Z_q` has `digits` balanced
digits of the base `b⋆`. -/
theorem liftedFold_winsTo [DecidableEq E] (admits : Type → Prop) (scalar : admits E)
    (range : admits (Fin (S.certifiedBase / 2 + 1 + 1) → E)) (cubic : admits (Fin (3 + 1) → E))
    (radix : Radix S.certifiedBase) {digits : ℕ}
    (covers : ∀ x : Zq, -((S.certifiedBase / 2 : ℕ) : ℤ) *
        ∑ u ∈ Finset.range digits, (S.certifiedBase : ℤ) ^ u ≤ x.valMinAbs ∧
      x.valMinAbs ≤ (((S.certifiedBase / 2 : ℕ) : ℤ) - 1) *
        ∑ u ∈ Finset.range digits, (S.certifiedBase : ℤ) ^ u)
    (room : O.quotientRoom fold digits ≤ 2 ^ μ) (batch : ℕ)
    (rowsFit : O.rowCount + 1 + 1 ≤ 2 ^ batch) {w : Witness O fold}
    (satisfies : w.Satisfies β st challenge) :
    WinsTo admits (foldCheck (liftedReduction β st challenge μ fits digits room batch rowsFit)
      S.certifiedBase (O.binarySupport fold)
      ((lockedRelation O fold β st challenge μ fits).completion S.certifiedBase digits
        (O.cells fold + 1) w.flat)) (0 + μ + 1 + 1 + μ + 1 + 1 + 1) true fun result _ _ =>
        ∃ r2 : List E, r2.length = μ ∧ result = some (r2, mle r2 (lifted E
          ((lockedRelation O fold β st challenge μ fits).completion S.certifiedBase digits
            (O.cells fold + 1) w.flat))) :=
  foldCheck_winsTo admits (liftedReduction β st challenge μ fits digits room batch rowsFit)
    S.certifiedBase scalar range cubic _
    (FlatRelation.withQuotients_holds _ radix covers _
      (fun r c j => Nat.lt_succ_of_le (lockedRows_below O fold β st challenge r c j))
      (fun _ large => scalarWeight_of_lt O fold β st.point (Nat.lt_of_succ_le large))
      (lockedRelation_of_flat (w.flat_of_le le_rfl) (satisfies.flat μ fits)))
    (FlatRelation.completion_range _ radix
      (satisfies.alphabets.rangeHolds radix.two radix.small μ))
    (fun x inside member => by
      rw [FlatRelation.completion_left _ _ (Nat.lt_succ_of_lt member.2)]
      exact satisfies.alphabets.binaryOn radix.two μ x inside member)

end Level

namespace Batch

variable {n : ℕ} {B : Batch (Fin n)} {E : Type} [Field E] [Fintype E] [Algebra Zq E]

/-- A number of cells that has the flat witness, the zero cell and the quotient digits of one
fold with opening groups: `δ_quot·d_max` cells for each ring row. -/
def quotientRoom (B : Batch (Fin n)) (digits : ℕ) : ℕ :=
  B.cells + 1 + (B.rowCount + 1) * (digits * B.largestDegree)

variable (β : ∀ g, Module.Basis (Fin (B.planes g)) Zq E) (st : B.Statement E)
  (challenge : B.Challenge) (μ : ℕ) (fits : B.cells < 2 ^ μ)

/-- The quotient-lift reduction of `Batch.lockedRelation` (Akita Section 7.3 and Lemma 7.1),
with the quotient digits of each ring row in the cells from `cells + 1` (Akita (348)), in the
base `base` with `digits` digits. Its bad set has at most `2·d_max − 1` points. -/
noncomputable def liftedReduction (base digits : ℕ) (room : B.quotientRoom digits ≤ 2 ^ μ)
    (batch : ℕ) (rowsFit : B.rowCount + 1 + 1 ≤ 2 ^ batch) : Reduction E :=
  ((B.lockedRelation β st challenge μ fits).withQuotients base digits (B.cells + 1)
    (((B.lockedRelation β st challenge μ fits).quotientEnd_le digits (B.cells + 1)
      (B.lockedRows_degree_le β st challenge)).trans room)).lift
    (algebraMap Zq E).injective B.largestDegree batch (B.lockedRows_degree_le β st challenge)
    rowsFit

variable {base : ℕ}

/-- Akita Lemma 10.42 (i) on the quotient-lift route for a fold with opening groups: on a path of
`foldCheck` that avoids the bad sets, a final claim that is true for the table gives the exact
branch relation `Batch.Witness.Satisfies` of the digit vectors `Batch.ofFlat` of the table. Each
group has the certified base `base`. -/
theorem liftedFold_satisfies [DecidableEq E] (uniform : ∀ g, (B.shape g).certifiedBase = base)
    (small : base < fieldModulus) (digits : ℕ) (room : B.quotientRoom digits ≤ 2 ^ μ)
    (batch : ℕ) (rowsFit : B.rowCount + 1 + 1 ≤ 2 ^ batch) (table : Table) :
    (foldCheck (liftedReduction β st challenge μ fits base digits room batch rowsFit) base
      B.binarySupport table).Avoids fun result =>
      ∀ r2 v, result = some (r2, v) → v = mle r2 (lifted E table) →
        (B.ofFlat table).Satisfies β st challenge :=
  (foldCheck_avoids (liftedReduction β st challenge μ fits base digits room batch rowsFit) base
    B.binarySupport table).mono fun _ checked r2 v equal opening =>
      satisfies_of_locked β st challenge μ fits uniform small
        (LiftRelation.holds_flat _ (checked r2 v equal opening).1)
        (checked r2 v equal opening).2.1 (checked r2 v equal opening).2.2

variable {β st challenge μ fits}

/-- The completeness of the checks of a fold with opening groups on the quotient-lift route
(Akita Lemma 7.1, item 1, with Lemma 10.7): for digit vectors with the exact branch relation, the
checks end in the claim `(r_2, w̃(r_2))` for the flat witness with the digits of the
quotients. -/
theorem liftedFold_winsTo [DecidableEq E] (admits : Type → Prop) (scalar : admits E)
    (range : admits (Fin (base / 2 + 1 + 1) → E)) (cubic : admits (Fin (3 + 1) → E))
    (uniform : ∀ g, (B.shape g).certifiedBase = base) (radix : Radix base) {digits : ℕ}
    (covers : ∀ x : Zq, -((base / 2 : ℕ) : ℤ) * ∑ u ∈ Finset.range digits, (base : ℤ) ^ u ≤
        x.valMinAbs ∧
      x.valMinAbs ≤ (((base / 2 : ℕ) : ℤ) - 1) * ∑ u ∈ Finset.range digits, (base : ℤ) ^ u)
    (room : B.quotientRoom digits ≤ 2 ^ μ) (batch : ℕ)
    (rowsFit : B.rowCount + 1 + 1 ≤ 2 ^ batch) {w : B.Witness}
    (satisfies : w.Satisfies β st challenge) :
    WinsTo admits (foldCheck (liftedReduction β st challenge μ fits base digits room batch
      rowsFit) base B.binarySupport
      ((B.lockedRelation β st challenge μ fits).completion base digits (B.cells + 1) w.flat))
      (0 + μ + 1 + 1 + μ + 1 + 1 + 1) true fun result _ _ =>
        ∃ r2 : List E, r2.length = μ ∧ result = some (r2, mle r2 (lifted E
          ((B.lockedRelation β st challenge μ fits).completion base digits (B.cells + 1)
            w.flat))) :=
  foldCheck_winsTo admits (liftedReduction β st challenge μ fits base digits room batch rowsFit)
    base scalar range cubic _
    (FlatRelation.withQuotients_holds _ radix covers _
      (fun r c j => Nat.lt_succ_of_le (B.lockedRows_below β st challenge r c j))
      (fun _ large => B.scalarWeight_of_lt β st μ (Nat.lt_of_succ_le large))
      (lockedRelation_of_flat (w.flat_of_le le_rfl) (satisfies.flat μ fits)))
    (FlatRelation.completion_range _ radix
      (satisfies.alphabets.rangeHolds uniform radix.two radix.small μ))
    (fun x inside member => by
      rw [FlatRelation.completion_left _ _ (Nat.lt_succ_of_lt member.2)]
      exact satisfies.alphabets.binaryOn uniform radix.two μ x inside member)

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
`liftedReduction`, and the table of the next level has the quotient digits after the cell
`cells`. -/
noncomputable def liftFold (β : Module.Basis (Fin O.planes) Zq E) (key : Key S)
    (openKey : OpeningKey O) (claim : Wire → Claim O E) (μ : ℕ) (fits : O.cells fold < 2 ^ μ)
    (digits : ℕ) (room : O.quotientRoom fold digits ≤ 2 ^ μ) (batch : ℕ)
    (rowsFit : O.rowCount + 1 + 1 ≤ 2 ^ batch) (block : Fin S.blocks → Coord)
    (decode : E → Option (Rq O.carrierDegree)) (read : Reader Wire) (weight start : ℕ) :
    Tier (Fin S.blocks) (fun _ => Rq O.carrierDegree) (Claim O E)
      (LevelBreak key openKey (2 * (2 * weight) * fold.sourceBound S)) (Witness O fold) (Weak O)
      Wire Coord E (Option (List E × E)) Table where
  method := packing β key openKey weight
  claim := claim
  block := block
  decode := fun _ => decode
  read := read
  digits := O.ofFlat fold
  check wire challenge table :=
    foldCheck (liftedReduction β ((claim wire).statement key openKey) challenge μ fits digits
      room batch rowsFit) S.certifiedBase (O.binarySupport fold) table
  passes table result := ∃ r2 v, result = some (r2, v) ∧ v = mle r2 (lifted E table)
  costs := foldCosts E (2 * O.largestDegree - 1) μ batch S.certifiedBase
  start := start

variable {β : Module.Basis (Fin O.planes) Zq E} {key : Key S} {openKey : OpeningKey O}
  {claim : Wire → Claim O E} {μ : ℕ} {fits : O.cells fold < 2 ^ μ} {digits : ℕ}
  {room : O.quotientRoom fold digits ≤ 2 ^ μ} {batch : ℕ}
  {rowsFit : O.rowCount + 1 + 1 ≤ 2 ^ batch} {block : Fin S.blocks → Coord}
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
  injective := injective
  merges := packing_sound units sparse base
  sound wire challenge table :=
    (liftedFold_satisfies β ((claim wire).statement key openKey) challenge μ fits small digits
      room batch rowsFit table).mono fun _ implies passed => by
        obtain ⟨r2, v, ended, opened⟩ := passed
        exact implies r2 v ended opened
  staged wire challenge table :=
    foldCheck_staged (liftedReduction β ((claim wire).statement key openKey) challenge μ fits
      digits room batch rowsFit) S.certifiedBase (O.binarySupport fold) table varsFit batchFit
  nonnegative := foldCosts_nonnegative _ _ _ _
  draws := by simp [liftFold, foldCosts]

/-- Akita Proposition 10.44 for one more level on the quotient-lift route: `committedFold_suffix`
with the checks for `liftedReduction`. The stage errors add `r·p + ε_fold` with `foldError` of
the quotient-lift reduction, which has `2·d_max − 1` in the place of `d_max − 1`. -/
theorem liftFold_suffix [Nonempty Coord] {total : ℕ}
    {accept : Prefix Wire (Coord → E) total → Prop} {Inner BreakT CandT KeyT : Type}
    {S' : Shape} {O' : OpeningShape S'} {fold' : Descriptor}
    {Result' Source' : Type} {β' : Module.Basis (Fin O'.planes) Zq E} {key' : Key S'}
    {openKey' : OpeningKey O'} {weight' : ℕ}
    {N : Tier (Fin S'.blocks) (fun _ => Rq O'.carrierDegree) (Claim O' E)
      (LevelBreak key' openKey' (2 * (2 * weight') * fold'.sourceBound S')) (Witness O' fold')
      (Weak O') Wire Coord E Result' Source'}
    {tail : Structure (Prefix Wire (Coord → E) total) Coord E (ℕ → Wire) Inner BreakT CandT KeyT
      (Option (Weak O'))}
    {error : ℝ} {loops tree : ℕ} {peak : ℕ} (injective : Function.Injective block)
    (units : UnitDifferences {challenge | ∃ slot, decode slot = some challenge})
    (sparse : ∀ challenge, (∃ slot, decode slot = some challenge) → Sparse weight peak challenge)
    (base : 2 ≤ fold.responseBase) (small : S.certifiedBase < fieldModulus)
    (varsFit : μ ≤ Fintype.card Coord) (batchFit : batch ≤ Fintype.card Coord)
    (statement : Statement O E) (challenge : Fin S.blocks → Rq O.carrierDegree)
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
      (∑ b, (liftFold β key openKey claim μ fits digits room batch rowsFit block decode read
          weight start).chance b +
        foldError (liftedReduction β statement challenge μ fits digits room batch rowsFit)
          S.certifiedBase + error)
      (S.blocks + 1 + loops) (2 * (S.blocks + 1) * tree) := by
  rw [← foldCosts_sum]
  exact Tier.suffix_push (Fintype.card_fin _)
    (liftFold_valid injective units sparse base small varsFit batchFit) packed next commit
    inside (fun _ _ table other => foldCheck_bare _ _ _ table other)
    (fun _ r2 v opened => ⟨r2, v, rfl, opened⟩) linked embed embedded later

end Folds

section Groups

variable {n : ℕ} {B : Batch (Fin n)} {E : Type} [Field E] [Fintype E] [DecidableEq E]
  [Algebra Zq E] {Wire Coord : Type} [Fintype Coord] [DecidableEq Coord]

/-- The level `committedBatch` on the quotient-lift route: the checks are `foldCheck` for
`Batch.liftedReduction`, and the table of the next level has the quotient digits after the cell
`cells`. -/
noncomputable def liftBatch (β : ∀ g, Module.Basis (Fin (B.planes g)) Zq E)
    (key : (g : Fin n) → Key (B.shape g)) (openKey : B.OpenKey) (claim : Wire → B.Claim E)
    (μ : ℕ) (fits : B.cells < 2 ^ μ) (base digits : ℕ) (room : B.quotientRoom digits ≤ 2 ^ μ)
    (batch : ℕ) (rowsFit : B.rowCount + 1 + 1 ≤ 2 ^ batch) (block : B.Block → Coord)
    (decode : (g : Fin n) → E → Option (Rq (B.carrierDegree g))) (read : Reader Wire)
    (mass : Fin n → ℕ) (start : ℕ) :
    Tier B.Block (fun b => Rq (B.carrierDegree b.1)) (B.Claim E)
      (B.LevelBreak key openKey fun g => 2 * (2 * mass g) * (B.fold g).sourceBound (B.shape g))
      B.Witness B.Weak Wire Coord E (Option (List E × E)) Table where
  method := Batch.method β key openKey mass
  claim := claim
  block := block
  decode := fun b => decode b.1
  read := read
  digits := B.ofFlat
  check wire challenge table :=
    foldCheck (Batch.liftedReduction β ((claim wire).statement key openKey)
      (fun g i => challenge ⟨g, i⟩) μ fits base digits room batch rowsFit) base B.binarySupport
      table
  passes table result := ∃ r2 v, result = some (r2, v) ∧ v = mle r2 (lifted E table)
  costs := foldCosts E (2 * B.largestDegree - 1) μ batch base
  start := start

variable {β : ∀ g, Module.Basis (Fin (B.planes g)) Zq E} {key : (g : Fin n) → Key (B.shape g)}
  {openKey : B.OpenKey} {claim : Wire → B.Claim E} {μ : ℕ} {fits : B.cells < 2 ^ μ}
  {base digits : ℕ} {room : B.quotientRoom digits ≤ 2 ^ μ} {batch : ℕ}
  {rowsFit : B.rowCount + 1 + 1 ≤ 2 ^ batch} {block : B.Block → Coord}
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
  injective := injective
  merges := Batch.method_sound units sparse response
  sound wire challenge table :=
    (Batch.liftedFold_satisfies β ((claim wire).statement key openKey)
      (fun g i => challenge ⟨g, i⟩) μ fits uniform small digits room batch rowsFit table).mono
        fun _ implies passed => by
          obtain ⟨r2, v, ended, opened⟩ := passed
          exact implies r2 v ended opened
  staged wire challenge table :=
    foldCheck_staged (Batch.liftedReduction β ((claim wire).statement key openKey)
      (fun g i => challenge ⟨g, i⟩) μ fits base digits room batch rowsFit) base B.binarySupport
      table varsFit batchFit
  nonnegative := foldCosts_nonnegative _ _ _ _
  draws := by simp [liftBatch, foldCosts]

/-- Akita Proposition 10.44 for one more level with opening groups on the quotient-lift route:
`committedBatch_suffix` with the checks for `Batch.liftedReduction`. -/
theorem liftBatch_suffix [Nonempty Coord] {total : ℕ}
    {accept : Prefix Wire (Coord → E) total → Prop} {Inner BreakT CandT KeyT : Type}
    {S' : Shape} {O' : OpeningShape S'} {fold' : Descriptor}
    {Result' Source' : Type} {β' : Module.Basis (Fin O'.planes) Zq E} {key' : Key S'}
    {openKey' : OpeningKey O'} {weight' : ℕ}
    {N : Tier (Fin S'.blocks) (fun _ => Rq O'.carrierDegree) (Claim O' E)
      (LevelBreak key' openKey' (2 * (2 * weight') * fold'.sourceBound S')) (Witness O' fold')
      (Weak O') Wire Coord E Result' Source'}
    {tail : Structure (Prefix Wire (Coord → E) total) Coord E (ℕ → Wire) Inner BreakT CandT KeyT
      (Option (Weak O'))}
    {error : ℝ} {loops tree : ℕ} {peak : Fin n → ℕ} (injective : Function.Injective block)
    (units : ∀ g, UnitDifferences {challenge | ∃ slot, decode g slot = some challenge})
    (sparse : ∀ g challenge, (∃ slot, decode g slot = some challenge) →
      Sparse (mass g) (peak g) challenge)
    (response : ∀ g, 2 ≤ (B.fold g).responseBase)
    (uniform : ∀ g, (B.shape g).certifiedBase = base) (small : base < fieldModulus)
    (varsFit : μ ≤ Fintype.card Coord) (batchFit : batch ≤ Fintype.card Coord)
    (statement : B.Statement E) (challenge : B.Challenge)
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
      (∑ b, (liftBatch β key openKey claim μ fits base digits room batch rowsFit block decode
          read mass start).chance b +
        foldError (Batch.liftedReduction β statement challenge μ fits base digits room batch
          rowsFit) base + error)
      (∑ g, (B.shape g).blocks + 1 + loops) (2 * (∑ g, (B.shape g).blocks + 1) * tree) := by
  rw [← foldCosts_sum]
  exact Tier.suffix_push B.card_block
    (liftBatch_valid injective units sparse response uniform small varsFit batchFit) packed next
    commit inside (fun _ _ table other => foldCheck_bare _ _ _ table other)
    (fun _ r2 v opened => ⟨r2, v, rfl, opened⟩) linked embed embedded later

/-- The batching root over a level with opening groups on the quotient-lift route over a last
packing fold is a suffix from the stage 0: `batchRootChain_suffix` with the checks for
`Batch.liftedReduction`. `Tier.Suffix.extract_sound`, `Tier.Suffix.extraction_le` and
`Tier.Suffix.runs_le` are its extraction theorem. -/
theorem liftRoot_suffix [Nonempty Coord] {S' : Shape} {O' : OpeningShape S'}
    {fold' : Descriptor} {β' : Module.Basis (Fin O'.planes) Zq E} {key' : Key S'}
    {openKey' : OpeningKey O'} {claim' : Wire → Claim O' E} {μ' : ℕ}
    {fits' : O'.cells fold' < 2 ^ μ'} {batch' : ℕ} {rowsFit' : O'.rowCount + 1 ≤ 2 ^ batch'}
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
      (1 / Fintype.card E +
        (∑ b, (liftBatch β key openKey claim μ fits base digits room batch rowsFit block decode
            read mass 1).chance b +
          (foldCosts E (2 * B.largestDegree - 1) μ batch base).sum +
          (∑ b, (clearFold β' key' openKey' claim' μ' fits' batch' rowsFit' block' decode' read
              weight' (liftBatch β key openKey claim μ fits base digits room batch rowsFit block
                decode read mass 1).finish).chance b +
            (foldCosts E (O'.largestDegree - 1) μ' batch' S'.certifiedBase).sum)))
      (1 + (∑ g, (B.shape g).blocks + 1 + (S'.blocks + 1)))
      (2 * (2 * (∑ g, (B.shape g).blocks + 1) * (2 * (S'.blocks + 1)))) :=
  batchRoot_suffix B.card_block
    (liftBatch_valid injective units sparse response uniform small varsFit batchFit)
    (clearFold_valid injective' units' sparse' base' small' varsFit' batchFit')
    (exists_fair_slots groupsFit) rfl rfl rfl rfl
    (fun _ _ table other => foldCheck_bare _ _ _ table other)
    (fun _ r2 v opened => ⟨r2, v, rfl, opened⟩)

end Groups

end Extract

end NightstreamFPrime.Spec.Compression.Akita
