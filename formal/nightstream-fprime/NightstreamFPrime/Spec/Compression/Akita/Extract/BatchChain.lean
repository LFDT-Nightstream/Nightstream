import NightstreamFPrime.Spec.Compression.Akita.Extract.BatchLevel
import NightstreamFPrime.Spec.Compression.Akita.Extract.BatchSource
import NightstreamFPrime.Spec.Compression.Akita.Extract.Chain

/-! A fold of Akita with opening groups as a level of the opening suffix with the checks
`foldCheck`, and the batching root over it (ePrint 2026/1983, revision of 2026-11-23,
Proposition 10.44, Theorem 10.46, "Batching root" and "Root claims", Theorem 10.17 and
Corollary 10.49 with (206)).

- `committedBatch` is the `Tier` of a fold with opening groups whose flat witness the next level
  commits to: its method is `Batch.method`, the source is a table, the digit vectors are
  `Batch.ofFlat`, and the checks are `foldCheck` for `Batch.flatReduction`. `clearBatch` is the
  `Tier` of a last fold with opening groups, with the checks for `Batch.flatReduction`.
- `committedBatch_suffix` is `Tier.suffix_push` for a fold with opening groups before a level
  with one group and coefficient packing: the next level commits to the flat witness
  `Batch.Witness.flat`, and its claim is the final claim `(r_2, v_W)` of the checks
  (`Tier.Linked`). `clearBatch_suffix` is `Tier.suffix_last`.
- `rootAccepts` is the verifier of the batching root, a fold with opening groups and a last
  packing fold, and `batchRoot` is their checkpoint structure. `RootOpens` is the relation of its
  first certificates: for each group a certified decommitment of the commitment of the group, and
  the claim of the group for its table (Akita (173) with `φ = 1`).
- `batchRoot_extract_sound`, `batchRoot_extraction_le` and `batchRoot_runs_le` are the extraction
  theorem of the batching root over `committedBatch` over `clearFold`, for each number of groups.
  The groups have their own shapes. The hypotheses are conditions on the parameters only.

Each group of a level has the certified base `base` (`BatchRows.lean`). The levels have the
quotient-free reduction, and the weights of the root are independent (`Root.lean`). -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita

theorem Batch.card_block {n : ℕ} (B : Batch (Fin n)) :
    Fintype.card B.Block = ∑ g, (B.shape g).blocks := by
  simp [Batch.Block]

namespace Extract

open scoped BigOperators
open NightstreamFPrime.Spec.Compression.Lib NightstreamFPrime.Spec.Compression.Game Polynomial
open NightstreamFPrime.Spec.Compression (Game)
open NightstreamFPrime.Spec.Checkpoint
open NightstreamFPrime.Spec.RandomOracle (OracleComp)

local instance : Fact (Nat.Prime fieldModulus) := ⟨FieldPrime.fieldModulus_natPrime⟩

attribute [local instance low] Classical.propDecidable

section Folds

variable {n : ℕ} {B : Batch (Fin n)} {E : Type} [Field E] [Fintype E] [DecidableEq E]
  [Algebra Zq E] {Wire Coord : Type} [Fintype Coord] [DecidableEq Coord]

/-- The largest ring degree `d_A` of the matrices `A` of the groups: the Akita code takes the
largest loss `2·d_A − 1` of `RingSwitchAlpha` over the groups. -/
def _root_.NightstreamFPrime.Spec.Compression.Akita.Batch.innerMax {n : ℕ} (B : Batch (Fin n)) :
    ℕ :=
  Finset.univ.sup fun g => (B.shape g).innerDegree

/-- `K` of the smallest fold challenge family of the groups, with the families without a filter
in the subrings of the carriers (`fold_challenge_runs` of the Akita code). The largest `K` of the
ladder is 132. -/
def _root_.NightstreamFPrime.Spec.Compression.Akita.Batch.familyBits {n : ℕ}
    (B : Batch (Fin n)) : ℕ :=
  ((List.finRange n).map fun g => setBits false (B.carrierDegree g)).foldr min 132

/-- A fold with opening groups and coefficient packing whose flat witness the next level commits
to, as a `Tier` with the first stage `start` (Akita Sections 5.4 and 5.5). Its method is
`Batch.method`. A slot of an answer of the oracle is one uniform element of `E`, and `decode g`
gives the fold challenge of a coordinate of the group `g`. The successor source is a table,
`Batch.ofFlat` gives its digit vectors, and the checks are `foldCheck` for the quotient-free
reduction of `Batch.flatRelation`, with the range base `base`. A final claim `(r_2, v_W)` is
true for a table `w` when `v_W = w̃(r_2)`. -/
noncomputable def committedBatch (β : ∀ g, Module.Basis (Fin (B.planes g)) Zq E)
    (key : (g : Fin n) → Key (B.shape g)) (openKey : B.OpenKey) (claim : Wire → B.Claim E)
    (μ : ℕ) (fits : B.cells ≤ 2 ^ μ) (batch : ℕ) (rowsFit : B.rowCount + 1 ≤ 2 ^ batch)
    (block : B.Block → Coord) (decode : (g : Fin n) → E → Option (Rq (B.carrierDegree g)))
    (read : Reader Wire) (mass : Fin n → ℕ) (base start : ℕ) :
    Tier B.Block (fun b => Rq (B.carrierDegree b.1)) (B.Claim E)
      (B.LevelBreak key openKey fun g => 2 * (2 * mass g) * (B.fold g).sourceBound (B.shape g))
      B.Witness B.Weak Wire Coord E (Option (List E × E)) Table where
  method := Batch.method β key openKey mass
  claim := claim
  block b := some (block b)
  decode := fun b => decode b.1
  read := read
  digits := B.ofFlat
  check wire challenge table :=
    foldCheck (Batch.flatReduction β ((claim wire).statement key openKey)
      (fun g i => challenge ⟨g, i⟩) μ fits batch rowsFit) base B.binarySupport B.compressed table
  passes table result := ∃ r2 v, result = some (r2, v) ∧ v = mle r2 (lifted E table)
  costs := foldCosts E (B.largestDegree - 1) μ batch base B.compressed
  bits := (foldLosses (2 * B.innerMax - 1) μ batch base B.compressed).map workBits
  foldBits := foldWorkBits (Fintype.card B.Block) B.familyBits
  start := start

/-- A last fold with opening groups and coefficient packing, as a `Tier` with the first stage
`start`. The successor source is the family of the digit vectors, which the proof has in the
clear, and the checks are `foldCheck` for the quotient-free reduction of `Batch.flatRelation` and
the flat witness of the digit vectors. -/
noncomputable def clearBatch (β : ∀ g, Module.Basis (Fin (B.planes g)) Zq E)
    (key : (g : Fin n) → Key (B.shape g)) (openKey : B.OpenKey) (claim : Wire → B.Claim E)
    (μ : ℕ) (fits : B.cells ≤ 2 ^ μ) (batch : ℕ) (rowsFit : B.rowCount + 1 ≤ 2 ^ batch)
    (block : B.Block → Coord) (decode : (g : Fin n) → E → Option (Rq (B.carrierDegree g)))
    (read : Reader Wire) (mass : Fin n → ℕ) (base start : ℕ) :
    Tier B.Block (fun b => Rq (B.carrierDegree b.1)) (B.Claim E)
      (B.LevelBreak key openKey fun g => 2 * (2 * mass g) * (B.fold g).sourceBound (B.shape g))
      B.Witness B.Weak Wire Coord E (Option (List E × E)) B.Witness where
  method := Batch.method β key openKey mass
  claim := claim
  block b := some (block b)
  decode := fun b => decode b.1
  read := read
  digits candidate := candidate
  check wire challenge candidate :=
    foldCheck (Batch.flatReduction β ((claim wire).statement key openKey)
      (fun g i => challenge ⟨g, i⟩) μ fits batch rowsFit) base B.binarySupport B.compressed
        candidate.flat
  passes candidate result :=
    ∃ r2 v, result = some (r2, v) ∧ v = mle r2 (lifted E candidate.flat)
  costs := foldCosts E (B.largestDegree - 1) μ batch base B.compressed
  bits := (foldLosses (2 * B.innerMax - 1) μ batch base B.compressed).map workBits
  foldBits := foldWorkBits (Fintype.card B.Block) B.familyBits
  start := start

variable {β : ∀ g, Module.Basis (Fin (B.planes g)) Zq E} {key : (g : Fin n) → Key (B.shape g)}
  {openKey : B.OpenKey} {claim : Wire → B.Claim E} {μ : ℕ} {fits : B.cells ≤ 2 ^ μ} {batch : ℕ}
  {block : B.Block → Coord} {decode : (g : Fin n) → E → Option (Rq (B.carrierDegree g))}
  {read : Reader Wire} {mass : Fin n → ℕ} {base start : ℕ}

/-- A committed fold with opening groups satisfies the conditions of `Tier.Valid`. The hypotheses
are: the fold coordinates have different coordinates of the answer, the challenge family of each
group has units as differences (Akita Definition 9.2, E1) and members with `‖c‖_1 = ω_g`,
`b_{z,g} ≥ 2`, each group has the certified base `base < q`, and an answer of the oracle has `μ'`
slots or more and `s_rel` slots or more. -/
theorem committedBatch_valid [Nonempty Coord] {rowsFit : B.rowCount + 1 ≤ 2 ^ batch}
    {peak : Fin n → ℕ} (injective : Function.Injective block)
    (units : ∀ g, UnitDifferences {challenge | ∃ slot, decode g slot = some challenge})
    (sparse : ∀ g challenge, (∃ slot, decode g slot = some challenge) →
      Sparse (mass g) (peak g) challenge)
    (response : ∀ g, 2 ≤ (B.fold g).responseBase)
    (uniform : ∀ g, (B.shape g).certifiedBase = base) (small : base < fieldModulus)
    (varsFit : μ ≤ Fintype.card Coord) (batchFit : batch ≤ Fintype.card Coord) :
    (committedBatch β key openKey claim μ fits batch rowsFit block decode read mass base
      start).Valid where
  injective := (Option.some_injective _).comp injective
  avoid _ := Option.some_ne_none _
  merges := Batch.method_sound units sparse response
  sound wire challenge table :=
    (Batch.tableFold_satisfies β ((claim wire).statement key openKey)
      (fun g i => challenge ⟨g, i⟩) μ fits uniform small batch rowsFit table).mono
        fun _ implies passed => by
          obtain ⟨r2, v, ended, opened⟩ := passed
          exact implies r2 v ended opened
  staged wire challenge table :=
    foldCheck_staged (Batch.flatReduction β ((claim wire).statement key openKey)
      (fun g i => challenge ⟨g, i⟩) μ fits batch rowsFit) base B.binarySupport B.compressed table
        varsFit
      batchFit
  nonnegative := foldCosts_nonnegative _ _ _ _ _
  draws := by simp [committedBatch, foldCosts]
  aligned := (List.length_map ..).trans (foldLosses_length ..)

/-- A last fold with opening groups satisfies the conditions of `Tier.Valid`, with the hypotheses
of `committedBatch_valid`. -/
theorem clearBatch_valid [Nonempty Coord] {rowsFit : B.rowCount + 1 ≤ 2 ^ batch}
    {peak : Fin n → ℕ} (injective : Function.Injective block)
    (units : ∀ g, UnitDifferences {challenge | ∃ slot, decode g slot = some challenge})
    (sparse : ∀ g challenge, (∃ slot, decode g slot = some challenge) →
      Sparse (mass g) (peak g) challenge)
    (response : ∀ g, 2 ≤ (B.fold g).responseBase)
    (uniform : ∀ g, (B.shape g).certifiedBase = base) (small : base < fieldModulus)
    (varsFit : μ ≤ Fintype.card Coord) (batchFit : batch ≤ Fintype.card Coord) :
    (clearBatch β key openKey claim μ fits batch rowsFit block decode read mass base
      start).Valid where
  injective := (Option.some_injective _).comp injective
  avoid _ := Option.some_ne_none _
  merges := Batch.method_sound units sparse response
  sound wire challenge candidate :=
    (Batch.flatFold_satisfies β ((claim wire).statement key openKey)
      (fun g i => challenge ⟨g, i⟩) μ fits uniform small batch rowsFit candidate).mono
        fun _ implies passed => by
          obtain ⟨r2, v, ended, opened⟩ := passed
          exact implies r2 v ended opened
  staged wire challenge candidate :=
    foldCheck_staged (Batch.flatReduction β ((claim wire).statement key openKey)
      (fun g i => challenge ⟨g, i⟩) μ fits batch rowsFit) base B.binarySupport B.compressed
        candidate.flat
      varsFit batchFit
  nonnegative := foldCosts_nonnegative _ _ _ _ _
  draws := by simp [clearBatch, foldCosts]
  aligned := (List.length_map ..).trans (foldLosses_length ..)

omit [Fintype Coord] [DecidableEq Coord] in
/-- The verifier of the checks of a committed fold with opening groups does not read the
table. -/
theorem committedBatch_uniform {rowsFit : B.rowCount + 1 ≤ 2 ^ batch} (wire : Wire)
    (challenge : (b : B.Block) → Rq (B.carrierDegree b.1)) (table other : Table) :
    bare ((committedBatch β key openKey claim μ fits batch rowsFit block decode read mass base
        start).check wire challenge table) =
      bare ((committedBatch β key openKey claim μ fits batch rowsFit block decode read mass base
        start).check wire challenge other) :=
  foldCheck_bare _ _ _ _ table other

section Suffixes

variable [Nonempty Coord] {total : ℕ} {accept : Prefix Wire (Option Coord → E) total → Prop}
  {Inner BreakT CandT KeyT : Type}

/-- Akita Proposition 10.44 for one more level, whose fold has opening groups: a committed fold
with opening groups over the suffix of the next level `N`, which has one group and commits to the
flat witness `Batch.Witness.flat`, is a suffix. Its stage errors add `Σ_b p_b + ε_fold`: `p_b` is
the largest probability of one fold challenge of the coordinate `b`, and `ε_fold` is `foldError`
of the quotient-free reduction, which is the same for each `statement` and each `challenge`. Its
retry loops add `r + 1` and its tree size is multiplied by `2·(r + 1)`, for `r = Σ_g B_g` fold
coordinates. `packed`: the method of `N` is `packing`. `linked`: the verifier accepts only with
the link `Tier.Linked`: the payload of the claim of `N` is the commitment that the proof has
after the fold challenges, and the point and the value of the claim of `N` are the final claim
`(r_2, v_W)` of the checks. `embedded`: `embed` maps a solution of Module-SIS for a matrix of `N`
to a witness of `R_bind` of the later stages. -/
theorem committedBatch_suffix {S' : Shape} {O' : OpeningShape S'} {fold' : Descriptor}
    {Result' Source' : Type} {β' : Module.Basis (Fin O'.planes) Zq E} {key' : Key S'}
    {openKey' : OpeningKey O'} {weight' : ℕ}
    {N : Tier (Fin S'.blocks) (fun _ => Rq O'.carrierDegree) (Claim O' E)
      (LevelBreak key' openKey' (2 * (2 * weight') * fold'.sourceBound S')) (Witness O' fold')
      (Weak O') Wire Coord E Result' Source'}
    {tail : Structure (Address Wire total) (Prefix Wire (Option Coord → E) total) (Option Coord) E (ℕ → Wire) Inner
      BreakT CandT KeyT
      (Option (Weak O'))}
    {error : ℝ} {loops tree : ℕ} {rowsFit : B.rowCount + 1 ≤ 2 ^ batch} {peak : Fin n → ℕ}
    (injective : Function.Injective block)
    (units : ∀ g, UnitDifferences {challenge | ∃ slot, decode g slot = some challenge})
    (sparse : ∀ g challenge, (∃ slot, decode g slot = some challenge) →
      Sparse (mass g) (peak g) challenge)
    (response : ∀ g, 2 ≤ (B.fold g).responseBase)
    (uniform : ∀ g, (B.shape g).certifiedBase = base) (small : base < fieldModulus)
    (varsFit : μ ≤ Fintype.card Coord) (batchFit : batch ≤ Fintype.card Coord)
    (packed : N.method = packing β' key' openKey' weight')
    (next : N.start =
      (committedBatch β key openKey claim μ fits batch rowsFit block decode read mass base
        start).finish)
    (commit : Wire → S'.Payload)
    (inside : (committedBatch β key openKey claim μ fits batch rowsFit block decode read mass
      base start).finish ≤ total)
    (linked : ∀ point, accept point →
      (committedBatch β key openKey claim μ fits batch rowsFit block decode read mass base
        start).Linked N commit point)
    (embed : LevelBreak key' openKey' (2 * (2 * weight') * fold'.sourceBound S') → BreakT)
    (embedded : ∀ solution, solution.isSome → tail.breaks (embed solution))
    (later : Tier.Suffix accept N.start N.Opens tail error loops tree) :
    Tier.Suffix accept start
      (committedBatch β key openKey claim μ fits batch rowsFit block decode read mass base
        start).Opens
      ((committedBatch β key openKey claim μ fits batch rowsFit block decode read mass base
        start).over accept tail (fun inner => tableOf (tail.witness inner))
        (fun inner other => embed (Tier.collide key' openKey'
          (2 * (2 * weight') * fold'.sourceBound S') (tail.witness inner) (tail.witness other))))
      ((committedBatch β key openKey claim μ fits batch rowsFit block decode read mass base
          start).levelError + error)
      (∑ g, (B.shape g).blocks + 1 + loops) (2 * (∑ g, (B.shape g).blocks + 1) * tree) := by
  exact Tier.suffix_push B.card_block
    (committedBatch_valid injective units sparse response uniform small varsFit batchFit) packed
    next commit inside committedBatch_uniform (fun _ r2 v opened => ⟨r2, v, rfl, opened⟩) linked
    embed embedded later

/-- The suffix of a last fold with opening groups: `Σ_b p_b + ε_fold` for the stage errors, with
`foldError` of the quotient-free reduction, `r + 1` retry loops and the tree size `2·(r + 1)`, for
`r = Σ_g B_g` fold coordinates. `passed`: the verifier accepts only when each fold challenge
decodes and the final claim of the checks is true for the flat witness of the digit vectors
`witnessOf` of the proof. -/
theorem clearBatch_suffix {rowsFit : B.rowCount + 1 ≤ 2 ^ batch} {peak : Fin n → ℕ}
    (injective : Function.Injective block)
    (units : ∀ g, UnitDifferences {challenge | ∃ slot, decode g slot = some challenge})
    (sparse : ∀ g challenge, (∃ slot, decode g slot = some challenge) →
      Sparse (mass g) (peak g) challenge)
    (response : ∀ g, 2 ≤ (B.fold g).responseBase)
    (uniform : ∀ g, (B.shape g).certifiedBase = base) (small : base < fieldModulus)
    (varsFit : μ ≤ Fintype.card Coord) (batchFit : batch ≤ Fintype.card Coord)
     (witnessOf : Wire → B.Witness)
    (inside : (clearBatch β key openKey claim μ fits batch rowsFit block decode read mass base
      start).finish ≤ total)
    (passed : ∀ point, accept point →
      (clearBatch β key openKey claim μ fits batch rowsFit block decode read mass base
        start).Passed (witnessOf (point.wire (start + 1))) point) :
    Tier.Suffix accept start
      (clearBatch β key openKey claim μ fits batch rowsFit block decode read mass base
        start).Opens
      ((clearBatch β key openKey claim μ fits batch rowsFit block decode read mass base
        start).over accept (finish (transcript total accept))
        (fun point => witnessOf (point.wire (start + 1))) (fun _ _ => ()))
      ((clearBatch β key openKey claim μ fits batch rowsFit block decode read mass base
          start).levelError)
      (∑ g, (B.shape g).blocks + 1) (2 * (∑ g, (B.shape g).blocks + 1)) := by
  exact Tier.suffix_last B.card_block
    (clearBatch_valid injective units sparse response uniform small varsFit batchFit) witnessOf
    inside passed

end Suffixes

end Folds

section Root

variable {count : ℕ} {B : Batch (Fin count)} {S' : Shape} {O' : OpeningShape S'}
  {fold' : Descriptor} {E Wire Coord Slot Result' Source' : Type} [Field E] [Fintype E]
  [DecidableEq E] [Algebra Zq E] [Fintype Slot] [Nonempty Slot] [Fintype Coord]
  [DecidableEq Coord] {β : ∀ g, Module.Basis (Fin (B.planes g)) Zq E}
  {key : (g : Fin count) → Key (B.shape g)} {openKey : B.OpenKey} {mass : Fin count → ℕ}
  {β' : Module.Basis (Fin O'.planes) Zq E} {key' : Key S'} {openKey' : OpeningKey O'}
  {weight' : ℕ}
  (T : Tier B.Block (fun b => Rq (B.carrierDegree b.1)) (B.Claim E)
    (B.LevelBreak key openKey fun g => 2 * (2 * mass g) * (B.fold g).sourceBound (B.shape g))
    B.Witness B.Weak Wire Coord Slot (Option (List E × E)) Table)
  (N : Tier (Fin S'.blocks) (fun _ => Rq O'.carrierDegree) (Claim O' E)
    (LevelBreak key' openKey' (2 * (2 * weight') * fold'.sourceBound S')) (Witness O' fold')
    (Weak O') Wire Coord Slot Result' Source')
  (commit : Wire → (g : Fin count) → (B.shape g).Payload) (value : Wire → Fin count → E)
  (place : Wire → Fin count → List E) (read : Reader Wire) (next : Wire → S'.Payload)
  (sourceOf : Wire → Source')

/-- The verifier of the batching root, the fold `T` with opening groups and the last level `N`
with one group (Akita Figure 4 with Section 5.4). The stage 0 has the weights `ϑ_g`. The verifier
makes the link `RootLinked` from the commitments `commit`, the points `place` and the values
`value` of the first message to the claim of `T` with the weights of the transcript, the link
`Tier.Linked` from `T` to the claim of `N`, and it accepts when the final claim of the checks of
`N` is true for the source `sourceOf` of the proof. -/
def rootAccepts (point : Prefix Wire (Option Coord → Slot) N.finish) : Prop :=
  RootLinked T 0 commit value place
      ((claimsRoot (Coord := Coord) (Slot := Slot) (total := N.finish) 0 read value place).weights
        point) point ∧
    T.Linked N next point ∧ N.Passed (sourceOf (point.wire (N.start + 1))) point

omit [Fintype E] [DecidableEq E] [Fintype Slot] [Nonempty Slot] [Fintype Coord]
  [DecidableEq Coord] in
/-- The relation of a first certificate of the batching root (Akita (173) with `φ = 1`): the
opening has, for each group, a certified decommitment of the commitment of the group in the first
message, with `κ̄_{1,g} = 2ω_g` and `β̄_{∞,g} = Δ^cert_{f,g}`, and the claim `f̃_g(r_g) = v_g` of
the first message holds for the tables that the source blocks decode to. -/
def RootOpens {total : ℕ} (commit : Wire → (g : Fin count) → (B.shape g).Payload)
    (value : Wire → Fin count → E) (place : Wire → Fin count → List E)
    (key : (g : Fin count) → Key (B.shape g)) (mass : Fin count → ℕ)
    (point : Prefix Wire (Option Coord → Slot) total) (opening : Option B.Weak) : Prop :=
  (∃ ω, opening = some ω ∧ ∀ g, (ω.certified g).Opens (2 * mass g)
    ((B.fold g).sourceBound (B.shape g)) (key g) (commit (point.wire 0) g)
    ((B.shape g).decode (ω.certified g).source)) ∧
  (fun g => mle (place ((point.take 0).wire 0) g) (lifted E (Batch.tablesOf opening g))) =
    value ((point.take 0).wire 0)

omit [Fintype E] [DecidableEq E] [Fintype Slot] [Nonempty Slot] [Fintype Coord]
  [DecidableEq Coord] in
/-- A witness of `RootOpens` has, for each group, a certified decommitment of its commitment to a
table with the claimed value at the point of the group. -/
theorem RootOpens.groups {total : ℕ} {point : Prefix Wire (Option Coord → Slot) total}
    {opening : Option B.Weak} (opens : RootOpens commit value place key mass point opening) :
    ∃ ω, opening = some ω ∧ ∀ g,
      (ω.certified g).Opens (2 * mass g) ((B.fold g).sourceBound (B.shape g)) (key g)
          (commit (point.wire 0) g) ((B.shape g).decode (ω.certified g).source) ∧
        mle (place (point.wire 0) g) (lifted E ((B.shape g).decode (ω.certified g).source)) =
          value (point.wire 0) g := by
  obtain ⟨⟨ω, rfl, certified⟩, values⟩ := opens
  refine ⟨ω, rfl, fun g => ⟨certified g, ?_⟩⟩
  have claim := congrFun values g
  rw [Prefix.wire_take le_rfl] at claim
  exact claim

/-- The checkpoint structure of the last level `N` under the verifier `rootAccepts`. -/
noncomputable def rootLast :=
  N.over (rootAccepts T N commit value place read next sourceOf)
    (finish (transcript N.finish (rootAccepts T N commit value place read next sourceOf)))
    (fun point => sourceOf (point.wire (N.start + 1))) (fun _ _ => ())

/-- The checkpoint structure of the fold `T` with opening groups over the last level `N`, whose
source is the flat witness of `T`. -/
noncomputable def rootFold :=
  T.over (rootAccepts T N commit value place read next sourceOf)
    (rootLast T N commit value place read next sourceOf)
    (fun inner => tableOf ((rootLast T N commit value place read next sourceOf).witness inner))
    (fun inner other => Sum.inl (Tier.collide key' openKey'
      (2 * (2 * weight') * fold'.sourceBound S')
      ((rootLast T N commit value place read next sourceOf).witness inner)
      ((rootLast T N commit value place read next sourceOf).witness other)))

/-- The checkpoint structure of the batching root at the stage 0 over the fold `T` with opening
groups and the last level `N` (Akita Theorem 10.46, "Batching root", and Proposition 10.44). The
candidate of the root is the tables of the groups. A witness of `R_bind` is a solution of
Module-SIS for a matrix of a group of `T`, for the shared matrices of `T`, or for a matrix of
`N`. -/
noncomputable def batchRoot :=
  (claimsRoot (Coord := Coord) (Slot := Slot) (total := N.finish) 0 read value place).segment.over
    (rootAccepts T N commit value place read next sourceOf)
    (rootFold T N commit value place read next sourceOf)
    (fun inner =>
      Batch.tablesOf ((rootFold T N commit value place read next sourceOf).witness inner))
    (fun inner other => Sum.inl (Batch.collide key openKey
      (fun g => 2 * (2 * mass g) * (B.fold g).sourceBound (B.shape g))
      ((rootFold T N commit value place read next sourceOf).witness inner)
      ((rootFold T N commit value place read next sourceOf).witness other)))
    (RootOpens commit value place key mass)

variable {T N commit value place read next sourceOf}

/-- The batching root over a fold with opening groups over a last packing fold is a suffix from
the stage 0 (Akita Theorem 10.46, "Batching root" and "Root claims", over Proposition 10.44).
The stage errors are `1/|E|` for the weights, and those of `Tier.suffix_push` over
`Tier.suffix_last`. `fair`: an answer of the oracle gives a uniform vector of one weight for each
group. `packed` and `packed'`: the methods are `Batch.method` and `packing`. `uniform` and
`passes`: the conditions of `Tier.suffix_push` on the checks of `T`. -/
theorem batchRoot_suffix {blocks : ℕ} (card : Fintype.card B.Block = blocks) (valid : T.Valid)
    (valid' : N.Valid) (fair : ∃ decode : (Coord → Slot) → Fin count → E, Fair decode)
    (packed : T.method = Batch.method β key openKey mass)
    (packed' : N.method = packing β' key' openKey' weight') (first : T.start = 0 + 1)
    (second : N.start = T.finish)
    (uniform : ∀ wire challenge table other,
      bare (T.check wire challenge table) = bare (T.check wire challenge other))
    (passes : ∀ (table : Table) r2 v, v = mle r2 (lifted E table) → T.passes table (some (r2, v))) :
    Tier.Suffix (rootAccepts T N commit value place read next sourceOf) 0
      (RootOpens commit value place key mass)
      (batchRoot T N commit value place read next sourceOf)
      (1 / Fintype.card E / 2 ^ Root.rootBits count + (T.levelError + N.levelError))
      (1 + (blocks + 1 + (S'.blocks + 1))) (2 * (2 * (blocks + 1) * (2 * (S'.blocks + 1)))) :=
  root_suffix 0 read value place fair packed first
    (by
      have ends := N.finish_eq
      omega)
    commit
    (fun point accepted => ⟨by
      have long := accepted.2.1.1
      have ends := T.finish_eq
      omega, accepted.1⟩)
    Sum.inl
    (fun solution found => by
      show T.method.broken solution
      rw [packed]
      exact found)
    (Tier.suffix_push card valid packed' second next
      (by
        have ends := N.finish_eq
        omega)
      uniform passes (fun _ accepted => accepted.2.1) Sum.inl
      (fun solution found => by
        show N.method.broken solution
        rw [packed']
        exact found)
      (Tier.suffix_last (Fintype.card_fin _) valid' sourceOf le_rfl
        fun _ accepted => accepted.2.2))

end Root

section Instance

variable {count : ℕ} {B : Batch (Fin count)} {S' : Shape} {O' : OpeningShape S'}
  {fold' : Descriptor} {E : Type} [Field E] [Fintype E] [DecidableEq E] [Algebra Zq E]
  {Wire Coord : Type} [Fintype Coord] [DecidableEq Coord] [Nonempty Coord]
  {β : ∀ g, Module.Basis (Fin (B.planes g)) Zq E} {key : (g : Fin count) → Key (B.shape g)}
  {openKey : B.OpenKey} {claim : Wire → B.Claim E} {μ : ℕ} {fits : B.cells ≤ 2 ^ μ} {batch : ℕ}
  {rowsFit : B.rowCount + 1 ≤ 2 ^ batch} {block : B.Block → Coord}
  {decode : (g : Fin count) → E → Option (Rq (B.carrierDegree g))} {read : Reader Wire}
  {mass : Fin count → ℕ} {base : ℕ}
  {β' : Module.Basis (Fin O'.planes) Zq E} {key' : Key S'} {openKey' : OpeningKey O'}
  {claim' : Wire → Claim O' E} {μ' : ℕ} {fits' : O'.cells fold' ≤ 2 ^ μ'} {batch' : ℕ}
  {rowsFit' : O'.rowCount + 1 ≤ 2 ^ batch'} {block' : Fin S'.blocks → Coord}
  {decode' : E → Option (Rq O'.carrierDegree)} {weight' : ℕ}
  {commit : Wire → (g : Fin count) → (B.shape g).Payload} {value : Wire → Fin count → E}
  {place : Wire → Fin count → List E} {next : Wire → S'.Payload}
  {witnessOf : Wire → Witness O' fold'}

local notation "first" =>
  committedBatch β key openKey claim μ fits batch rowsFit block decode read mass base 1

local notation "second" =>
  clearFold β' key' openKey' claim' μ' fits' batch' rowsFit' block' decode' read weight'
    (Tier.finish first)

local notation "accepts" => rootAccepts first second commit value place read next witnessOf

local notation "chain" => batchRoot first second commit value place read next witnessOf

/-- The batching root over a committed fold with opening groups over a last packing fold is a
suffix from the stage 0. `groupsFit`: an answer of the oracle has one slot or more for each
group. -/
theorem batchRootChain_suffix {peak : Fin count → ℕ} {peak' : ℕ}
    (injective : Function.Injective block)
    (units : ∀ g, UnitDifferences {challenge | ∃ slot, decode g slot = some challenge})
    (sparse : ∀ g challenge, (∃ slot, decode g slot = some challenge) →
      Sparse (mass g) (peak g) challenge)
    (response : ∀ g, 2 ≤ (B.fold g).responseBase)
    (uniform : ∀ g, (B.shape g).certifiedBase = base) (small : base < fieldModulus)
    (varsFit : μ ≤ Fintype.card Coord) (batchFit : batch ≤ Fintype.card Coord)
    (groupsFit : count ≤ Fintype.card Coord) (injective' : Function.Injective block')
    (units' : UnitDifferences {challenge | ∃ slot, decode' slot = some challenge})
    (sparse' : ∀ challenge, (∃ slot, decode' slot = some challenge) →
      Sparse weight' peak' challenge)
    (base' : 2 ≤ fold'.responseBase) (small' : S'.certifiedBase < fieldModulus)
    (varsFit' : μ' ≤ Fintype.card Coord) (batchFit' : batch' ≤ Fintype.card Coord) :
    Tier.Suffix accepts 0 (RootOpens commit value place key mass) chain
      (1 / Fintype.card E / 2 ^ Root.rootBits count + (Tier.levelError first +
        (Tier.levelError second)))
      (1 + (∑ g, (B.shape g).blocks + 1 + (S'.blocks + 1)))
      (2 * (2 * (∑ g, (B.shape g).blocks + 1) * (2 * (S'.blocks + 1)))) :=
  batchRoot_suffix B.card_block
    (committedBatch_valid injective units sparse response uniform small varsFit batchFit)
    (clearFold_valid injective' units' sparse' base' small' varsFit' batchFit')
    (exists_fair_slots groupsFit) rfl rfl rfl rfl committedBatch_uniform
    (fun _ r2 v opened => ⟨r2, v, rfl, opened⟩)

/-- An output of the extractor for the batching root over a committed fold with opening groups
over a last packing fold (Akita Theorem 10.46, "Batching root" and "Root claims"): the verifier
accepts the transcript of the prover, and the output has, for each group, a certified
decommitment of the commitment of the group in the first message to a table with the claimed
value `v_g` at the point `r_g` of the group, or it is a solution of Module-SIS for a matrix `A_g`,
`B_g` or a compression map of a group, for `D` or a map `H_a` of the first level, or for `A`, `B`,
`D` or a compression map of the last level. -/
theorem batchRoot_extract_sound {peak : Fin count → ℕ} {peak' : ℕ}
    (injective : Function.Injective block)
    (units : ∀ g, UnitDifferences {challenge | ∃ slot, decode g slot = some challenge})
    (sparse : ∀ g challenge, (∃ slot, decode g slot = some challenge) →
      Sparse (mass g) (peak g) challenge)
    (response : ∀ g, 2 ≤ (B.fold g).responseBase)
    (uniform : ∀ g, (B.shape g).certifiedBase = base) (small : base < fieldModulus)
    (varsFit : μ ≤ Fintype.card Coord) (batchFit : batch ≤ Fintype.card Coord)
    (groupsFit : count ≤ Fintype.card Coord) (injective' : Function.Injective block')
    (units' : UnitDifferences {challenge | ∃ slot, decode' slot = some challenge})
    (sparse' : ∀ challenge, (∃ slot, decode' slot = some challenge) →
      Sparse weight' peak' challenge)
    (base' : 2 ≤ fold'.responseBase) (small' : S'.certifiedBase < fieldModulus)
    (varsFit' : μ' ≤ Fintype.card Coord) (batchFit' : batch' ≤ Fintype.card Coord)
    (prover : OracleComp (Address Wire (Tier.finish second)) (Option Coord → E)
      (ℕ → Wire))
    (budget : ℕ) (table : Address Wire (Tier.finish second) → Option Coord → E)
    (tape : Tape (Address Wire (Tier.finish second)) (Option Coord) E budget
      (Structure.protocol chain).stages)
    (out : Opened B.Weak
        (Opened (Weak O') (Prefix Wire (Option Coord → E) (Tier.finish second))) ⊕
      (B.LevelBreak key openKey
          (fun g => 2 * (2 * mass g) * (B.fold g).sourceBound (B.shape g)) ⊕
        (LevelBreak key' openKey' (2 * (2 * weight') * fold'.sourceBound S') ⊕ Unit)))
    (result : Checkpoint.extract chain prover budget table tape = some out) :
    accepts ((Structure.protocol chain).final prover table) ∧
      Sum.elim
        (fun certificate => ∃ ω, certificate.opening = some ω ∧ ∀ g,
          (ω.certified g).Opens (2 * mass g) ((B.fold g).sourceBound (B.shape g)) (key g)
              (commit (((Structure.protocol chain).input prover table 0).wire 0) g)
              ((B.shape g).decode (ω.certified g).source) ∧
            mle (place (((Structure.protocol chain).input prover table 0).wire 0) g)
                (lifted E ((B.shape g).decode (ω.certified g).source)) =
              value (((Structure.protocol chain).input prover table 0).wire 0) g)
        (Sum.elim (fun solution => solution.isSome)
          (Sum.elim (fun solution => solution.isSome) fun _ => False))
        out := by
  obtain ⟨accepted, sound⟩ := (batchRootChain_suffix (commit := commit) (value := value)
    (place := place) (next := next) (witnessOf := witnessOf) injective units sparse response
    uniform small varsFit batchFit groupsFit injective' units' sparse' base' small' varsFit'
    batchFit').extract_sound rfl prover budget table tape out result
  refine ⟨accepted, ?_⟩
  rcases out with certificate | broken
  · exact RootOpens.groups commit value place sound
  · exact sound

variable [Finite Wire]

/-- Akita Theorem 10.17 for the batching root over a committed fold with opening groups over a
last packing fold, which is Akita Corollary 10.49 with (206) for these stages. For each prover
with at most `Q` queries to the oracle, the probability that the verifier accepts and the
extractor returns no output is at most
`(Q + 1)·(1/|E| + Σ_b p_b + ε_fold + r'·p' + ε'_fold) + (Q + 1)·(1 + r + 1 + r' + 1)/budget`.
`1/|E|` is the error of the weights of the root, `r = Σ_g B_g` and `r'` are the numbers of the
fold coordinates of the two levels, `p_b` and `p'` are the largest probabilities of one fold
challenge (`Tier.chance`), `ε_fold` and `ε'_fold` are `foldError` of the quotient-free
reductions, which are the same for each statement and each challenge, and `budget` is the limit
of the attempts of a retry loop of the extractor. By `batchRoot_extract_sound`, an output has a
certified decommitment with a true claim for each group, or it is a solution of Module-SIS. -/
theorem batchRoot_extraction_le {peak : Fin count → ℕ} {peak' : ℕ}
    (injective : Function.Injective block)
    (units : ∀ g, UnitDifferences {challenge | ∃ slot, decode g slot = some challenge})
    (sparse : ∀ g challenge, (∃ slot, decode g slot = some challenge) →
      Sparse (mass g) (peak g) challenge)
    (response : ∀ g, 2 ≤ (B.fold g).responseBase)
    (uniform : ∀ g, (B.shape g).certifiedBase = base) (small : base < fieldModulus)
    (varsFit : μ ≤ Fintype.card Coord) (batchFit : batch ≤ Fintype.card Coord)
    (groupsFit : count ≤ Fintype.card Coord) (injective' : Function.Injective block')
    (units' : UnitDifferences {challenge | ∃ slot, decode' slot = some challenge})
    (sparse' : ∀ challenge, (∃ slot, decode' slot = some challenge) →
      Sparse weight' peak' challenge)
    (base' : 2 ≤ fold'.responseBase) (small' : S'.certifiedBase < fieldModulus)
    (varsFit' : μ' ≤ Fintype.card Coord) (batchFit' : batch' ≤ Fintype.card Coord)
    (prover : OracleComp (Address Wire (Tier.finish second)) (Option Coord → E)
      (ℕ → Wire))
    {queries : ℕ} (bounded : prover.QueryBound queries) {budget : ℕ} (positive : 0 < budget) :
    𝔼 table, 𝔼 tape : Tape (Address Wire (Tier.finish second)) (Option Coord) E
      budget
          (Structure.protocol chain).stages,
        (if accepts ((Structure.protocol chain).final prover table) ∧
          Checkpoint.extract chain prover budget table tape = none then (1 : ℝ) else 0) ≤
      (queries + 1) * (1 / Fintype.card E / 2 ^ Root.rootBits count + (Tier.levelError first +
          (Tier.levelError second))) +
        (queries + 1) * (1 + (∑ g, (B.shape g).blocks + 1 + (S'.blocks + 1)) : ℕ) / budget := by
  exact (batchRootChain_suffix (commit := commit) (value := value) (place := place)
    (next := next) (witnessOf := witnessOf) injective units sparse response uniform small varsFit
    batchFit groupsFit injective' units' sparse' base' small' varsFit'
    batchFit').extraction_le rfl prover bounded positive

/-- The expected number of runs of the prover in the extraction of the batching root over a
committed fold with opening groups over a last packing fold is at most `K̂ + (Q + 1)·(K̂ − 1)`
with `K̂ = 2·2·(r + 1)·2·(r' + 1)` for `r = Σ_g B_g` and `r'` fold coordinates (Akita
Theorem 10.17, the cost). -/
theorem batchRoot_runs_le {peak : Fin count → ℕ} {peak' : ℕ}
    (injective : Function.Injective block)
    (units : ∀ g, UnitDifferences {challenge | ∃ slot, decode g slot = some challenge})
    (sparse : ∀ g challenge, (∃ slot, decode g slot = some challenge) →
      Sparse (mass g) (peak g) challenge)
    (response : ∀ g, 2 ≤ (B.fold g).responseBase)
    (uniform : ∀ g, (B.shape g).certifiedBase = base) (small : base < fieldModulus)
    (varsFit : μ ≤ Fintype.card Coord) (batchFit : batch ≤ Fintype.card Coord)
    (groupsFit : count ≤ Fintype.card Coord) (injective' : Function.Injective block')
    (units' : UnitDifferences {challenge | ∃ slot, decode' slot = some challenge})
    (sparse' : ∀ challenge, (∃ slot, decode' slot = some challenge) →
      Sparse weight' peak' challenge)
    (base' : 2 ≤ fold'.responseBase) (small' : S'.certifiedBase < fieldModulus)
    (varsFit' : μ' ≤ Fintype.card Coord) (batchFit' : batch' ≤ Fintype.card Coord)
    (prover : OracleComp (Address Wire (Tier.finish second)) (Option Coord → E)
      (ℕ → Wire))
    {queries : ℕ} (bounded : prover.QueryBound queries) (budget : ℕ) :
    𝔼 table, 𝔼 tape : Tape (Address Wire (Tier.finish second)) (Option Coord) E
      budget
          (Structure.protocol chain).stages,
        (extractRuns chain prover budget table tape : ℝ) ≤
      (2 * (2 * (∑ g, (B.shape g).blocks + 1) * (2 * (S'.blocks + 1))) : ℕ) +
        (queries + 1) * ((2 * (2 * (∑ g, (B.shape g).blocks + 1) * (2 * (S'.blocks + 1))) : ℕ) -
          1) :=
  (batchRootChain_suffix (commit := commit) (value := value) (place := place) (next := next)
    (witnessOf := witnessOf) injective units sparse response uniform small varsFit batchFit
    groupsFit injective' units' sparse' base' small' varsFit' batchFit').runs_le rfl prover
    bounded budget

end Instance

end Extract

end NightstreamFPrime.Spec.Compression.Akita
