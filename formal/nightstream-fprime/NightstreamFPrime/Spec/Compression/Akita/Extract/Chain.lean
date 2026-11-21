import NightstreamFPrime.Spec.Compression.Akita.Extract.FlatSource
import NightstreamFPrime.Spec.Compression.Akita.Extract.PackingFold
import NightstreamFPrime.Spec.Compression.Akita.Extract.Recursion

/-! The recursive opening suffix of Akita for packing folds with the checks `foldCheck`
(ePrint 2026/1983, revision of 2026-10-15, Proposition 10.44, Theorem 10.17 and Corollary 10.49
with (206)).

- `committedFold` is the `Tier` of a packing fold whose flat witness the next level commits to:
  the source is a table, the digit vectors are `OpeningShape.ofFlat`, and the checks are
  `foldCheck` for the quotient-free reduction of `flatRelation`.
- `clearFold` is the `Tier` of a last packing fold, whose proof has the digit vectors in the
  clear: the checks are `foldCheck` for the quotient-free reduction of `flatRelation`. With the
  first stage 0, `clearFold_suffix` and `Tier.Suffix.extraction_le` are the extraction theorem of
  one packing fold.
- `committedFold_suffix` and `clearFold_suffix` are `Tier.suffix_push` and `Tier.suffix_last`
  for these levels, with `foldError` for the sum of the bounds of the draws. A suffix of each
  number of levels is `committedFold_suffix` for each level but the last, over
  `clearFold_suffix`.
- `pair` is the checkpoint structure of two levels, and `pairAccepts` is its verifier.
  `packingPair_extract_sound`, `packingPair_extraction_le` and `packingPair_runs_le` are the
  extraction theorem of a committed packing fold over a last packing fold.

The trace levels, the quotient-lift reduction, the terminal of Akita Section 8.2, more than one
claim or group, and the setup offloading are not in this file. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Extract

open scoped BigOperators
open NightstreamFPrime.Spec.Compression.Lib NightstreamFPrime.Spec.Compression.Game Polynomial
open NightstreamFPrime.Spec.Compression (Game)
open NightstreamFPrime.Spec.Checkpoint
open NightstreamFPrime.Spec.RandomOracle (OracleComp)

local instance : Fact (Nat.Prime fieldModulus) := ⟨FieldPrime.fieldModulus_natPrime⟩

attribute [local instance low] Classical.propDecidable

section Folds

variable {S : Shape} {O : OpeningShape S} {fold : Descriptor} {E : Type} [Field E] [Fintype E]
  [DecidableEq E] [Algebra Zq E] {Wire Coord : Type} [Fintype Coord] [DecidableEq Coord]

/-- A packing fold for one claim whose flat witness the next level commits to, as a `Tier` with
the first stage `start`. A slot of an answer of the oracle is one uniform element of `E`. The
successor source is a table, `OpeningShape.ofFlat` gives its digit vectors, and the checks are
`foldCheck` for the quotient-free reduction of `flatRelation`. A final claim `(r_2, v_W)` is
true for a table `w` when `v_W = w̃(r_2)`. -/
noncomputable def committedFold (β : Module.Basis (Fin O.planes) Zq E) (key : Key S)
    (openKey : OpeningKey O) (claim : Wire → Claim O E) (μ : ℕ) (fits : O.cells fold ≤ 2 ^ μ)
    (batch : ℕ) (rowsFit : O.rowCount + 1 ≤ 2 ^ batch) (block : Fin S.blocks → Coord)
    (decode : E → Option (Rq O.carrierDegree)) (read : Reader Wire) (weight start : ℕ) :
    Tier (Fin S.blocks) (fun _ => Rq O.carrierDegree) (Claim O E)
      (LevelBreak key openKey (2 * (2 * weight) * fold.sourceBound S)) (Witness O fold) (Weak O)
        Wire Coord E
      (Option (List E × E)) Table where
  method := packing β key openKey weight
  claim := claim
  block b := some (block b)
  decode := fun _ => decode
  read := read
  digits := O.ofFlat fold
  check wire challenge table :=
    foldCheck (flatReduction β ((claim wire).statement key openKey) challenge μ fits batch
      rowsFit) S.certifiedBase (O.binarySupport fold) (O.compressed fold) table
  passes table result := ∃ r2 v, result = some (r2, v) ∧ v = mle r2 (lifted E table)
  costs := foldCosts E (O.largestDegree - 1) μ batch S.certifiedBase (O.compressed fold)
  bits := (foldLosses (2 * S.innerDegree - 1) μ batch S.certifiedBase (O.compressed fold)).map
    workBits
  foldBits := foldWorkBits S.blocks (setBits false O.carrierDegree)
  start := start

/-- A last packing fold for one claim, as a `Tier` with the first stage `start`. The successor
source is the family of the digit vectors, which the proof has in the clear, and the checks are
`foldCheck` for the quotient-free reduction of `flatRelation` and the flat witness of the digit
vectors. A final claim `(r_2, v_W)` is true when `v_W = w̃(r_2)` for this flat witness. -/
noncomputable def clearFold (β : Module.Basis (Fin O.planes) Zq E) (key : Key S)
    (openKey : OpeningKey O) (claim : Wire → Claim O E) (μ : ℕ) (fits : O.cells fold ≤ 2 ^ μ)
    (batch : ℕ) (rowsFit : O.rowCount + 1 ≤ 2 ^ batch) (block : Fin S.blocks → Coord)
    (decode : E → Option (Rq O.carrierDegree)) (read : Reader Wire) (weight start : ℕ) :
    Tier (Fin S.blocks) (fun _ => Rq O.carrierDegree) (Claim O E)
      (LevelBreak key openKey (2 * (2 * weight) * fold.sourceBound S)) (Witness O fold) (Weak O)
        Wire Coord E
      (Option (List E × E)) (Witness O fold) where
  method := packing β key openKey weight
  claim := claim
  block b := some (block b)
  decode := fun _ => decode
  read := read
  digits candidate := candidate
  check wire challenge candidate :=
    foldCheck (flatReduction β ((claim wire).statement key openKey) challenge μ fits batch
      rowsFit) S.certifiedBase (O.binarySupport fold) (O.compressed fold) candidate.flat
  passes candidate result :=
    ∃ r2 v, result = some (r2, v) ∧ v = mle r2 (lifted E candidate.flat)
  costs := foldCosts E (O.largestDegree - 1) μ batch S.certifiedBase (O.compressed fold)
  bits := (foldLosses (2 * S.innerDegree - 1) μ batch S.certifiedBase (O.compressed fold)).map
    workBits
  foldBits := foldWorkBits S.blocks (setBits false O.carrierDegree)
  start := start

variable {β : Module.Basis (Fin O.planes) Zq E} {key : Key S} {openKey : OpeningKey O}
  {claim : Wire → Claim O E} {μ : ℕ} {fits : O.cells fold ≤ 2 ^ μ} {batch : ℕ}
  {block : Fin S.blocks → Coord} {decode : E → Option (Rq O.carrierDegree)} {read : Reader Wire}
  {weight start : ℕ}

/-- A committed packing fold satisfies the conditions of `Tier.Valid`. The hypotheses are: the
blocks have different coordinates, the challenge family has units as differences (Akita
Definition 9.2, E1) and members with `‖c‖_1 = ω`, `b_z ≥ 2`, `b⋆ < q`, and an answer of the oracle
has `μ'` slots or more and `s_rel` slots or more. -/
theorem committedFold_valid [Nonempty Coord] {rowsFit : O.rowCount + 1 ≤ 2 ^ batch}
    {peak : ℕ} (injective : Function.Injective block)
    (units : UnitDifferences {challenge | ∃ slot, decode slot = some challenge})
    (sparse : ∀ challenge, (∃ slot, decode slot = some challenge) → Sparse weight peak challenge)
    (base : 2 ≤ fold.responseBase) (small : S.certifiedBase < fieldModulus)
    (varsFit : μ ≤ Fintype.card Coord) (batchFit : batch ≤ Fintype.card Coord) :
    (committedFold β key openKey claim μ fits batch rowsFit block decode read weight
      start).Valid where
  injective := (Option.some_injective _).comp injective
  avoid _ := Option.some_ne_none _
  merges := packing_sound units sparse base
  sound wire challenge table :=
    (tableFold_satisfies β ((claim wire).statement key openKey) challenge μ fits small batch
      rowsFit table).mono fun _ implies passed => by
        obtain ⟨r2, v, ended, opened⟩ := passed
        exact implies r2 v ended opened
  staged wire challenge table :=
    foldCheck_staged (flatReduction β ((claim wire).statement key openKey) challenge μ fits
      batch rowsFit) S.certifiedBase (O.binarySupport fold) (O.compressed fold) table varsFit
        batchFit
  nonnegative := foldCosts_nonnegative _ _ _ _ _
  draws := by simp [committedFold, foldCosts]
  aligned := (List.length_map ..).trans (foldLosses_length ..)

/-- A last packing fold satisfies the conditions of `Tier.Valid`, with the hypotheses of
`committedFold_valid`. -/
theorem clearFold_valid [Nonempty Coord] {rowsFit : O.rowCount + 1 ≤ 2 ^ batch} {peak : ℕ}
    (injective : Function.Injective block)
    (units : UnitDifferences {challenge | ∃ slot, decode slot = some challenge})
    (sparse : ∀ challenge, (∃ slot, decode slot = some challenge) → Sparse weight peak challenge)
    (base : 2 ≤ fold.responseBase) (small : S.certifiedBase < fieldModulus)
    (varsFit : μ ≤ Fintype.card Coord) (batchFit : batch ≤ Fintype.card Coord) :
    (clearFold β key openKey claim μ fits batch rowsFit block decode read weight
      start).Valid where
  injective := (Option.some_injective _).comp injective
  avoid _ := Option.some_ne_none _
  merges := packing_sound units sparse base
  sound wire challenge candidate :=
    (flatFold_satisfies β ((claim wire).statement key openKey) challenge μ fits small batch
      rowsFit candidate).mono fun _ implies passed => by
        obtain ⟨r2, v, ended, opened⟩ := passed
        exact implies r2 v ended opened
  staged wire challenge candidate :=
    foldCheck_staged (flatReduction β ((claim wire).statement key openKey) challenge μ fits
      batch rowsFit) S.certifiedBase (O.binarySupport fold) (O.compressed fold) candidate.flat
        varsFit batchFit
  nonnegative := foldCosts_nonnegative _ _ _ _ _
  draws := by simp [clearFold, foldCosts]
  aligned := (List.length_map ..).trans (foldLosses_length ..)

omit [Fintype Coord] [DecidableEq Coord] in
/-- The verifier of the checks of a committed packing fold does not read the table. -/
theorem committedFold_uniform {rowsFit : O.rowCount + 1 ≤ 2 ^ batch} (wire : Wire)
    (challenge : Fin S.blocks → Rq O.carrierDegree) (table other : Table) :
    bare ((committedFold β key openKey claim μ fits batch rowsFit block decode read weight
        start).check wire challenge table) =
      bare ((committedFold β key openKey claim μ fits batch rowsFit block decode read weight
        start).check wire challenge other) :=
  foldCheck_bare _ _ _ _ table other

section Suffixes

variable [Nonempty Coord] {total : ℕ} {accept : Prefix Wire (Option Coord → E) total → Prop}
  {Inner BreakT CandT KeyT : Type}

/-- Akita Proposition 10.44 for one more level: a committed packing fold over the suffix of the
next level `N`, which commits to its flat witness, is a suffix. Its stage errors add
`r·p + ε_fold`: `r` is the number of blocks, `p` is the largest probability of one fold
challenge, and `ε_fold` is `foldError` of the quotient-free reduction, which is the same for each
`statement` and each `challenge`. Its retry loops add `r + 1`, and its tree size is multiplied by
`2·(r + 1)`. `packed`: the method of `N` is `packing`. `linked`: the verifier accepts only with
the link `Tier.Linked` to the claim of `N`. `embedded`: `embed` maps a solution of Module-SIS for
a matrix of `N` to a witness of `R_bind` of the later stages. -/
theorem committedFold_suffix {S' : Shape} {O' : OpeningShape S'} {fold' : Descriptor}
    {Result' Source' : Type} {β' : Module.Basis (Fin O'.planes) Zq E} {key' : Key S'}
    {openKey' : OpeningKey O'} {weight' : ℕ}
    {N : Tier (Fin S'.blocks) (fun _ => Rq O'.carrierDegree) (Claim O' E)
      (LevelBreak key' openKey' (2 * (2 * weight') * fold'.sourceBound S')) (Witness O' fold')
        (Weak O') Wire Coord E Result'
      Source'}
    {tail : Structure (Prefix Wire (Option Coord → E) total) (Option Coord) E (ℕ → Wire) Inner
      BreakT CandT KeyT
      (Option (Weak O'))}
    {error : ℝ} {loops tree : ℕ}
    {rowsFit : O.rowCount + 1 ≤ 2 ^ batch} {peak : ℕ}
    (injective : Function.Injective block)
    (units : UnitDifferences {challenge | ∃ slot, decode slot = some challenge})
    (sparse : ∀ challenge, (∃ slot, decode slot = some challenge) → Sparse weight peak challenge)
    (base : 2 ≤ fold.responseBase) (small : S.certifiedBase < fieldModulus)
    (varsFit : μ ≤ Fintype.card Coord) (batchFit : batch ≤ Fintype.card Coord)
    (packed : N.method = packing β' key' openKey' weight')
    (next : N.start =
      (committedFold β key openKey claim μ fits batch rowsFit block decode read weight
        start).finish)
    (commit : Wire → S'.Payload)
    (inside : (committedFold β key openKey claim μ fits batch rowsFit block decode read weight
      start).finish ≤ total)
    (linked : ∀ point, accept point →
      (committedFold β key openKey claim μ fits batch rowsFit block decode read weight
        start).Linked N commit point)
    (embed : LevelBreak key' openKey' (2 * (2 * weight') * fold'.sourceBound S') → BreakT)
    (embedded : ∀ solution, solution.isSome → tail.breaks (embed solution))
    (later : Tier.Suffix accept N.start N.Opens tail error loops tree) :
    Tier.Suffix accept start
      (committedFold β key openKey claim μ fits batch rowsFit block decode read weight
        start).Opens
      ((committedFold β key openKey claim μ fits batch rowsFit block decode read weight
        start).over accept tail (fun inner => tableOf (tail.witness inner))
        (fun inner other => embed (Tier.collide key' openKey'
          (2 * (2 * weight') * fold'.sourceBound S') (tail.witness inner) (tail.witness other))))
      ((committedFold β key openKey claim μ fits batch rowsFit block decode read weight
          start).levelError +
        error)
      (S.blocks + 1 + loops) (2 * (S.blocks + 1) * tree) := by
  exact Tier.suffix_push (Fintype.card_fin _)
    (committedFold_valid injective units sparse base small varsFit batchFit) packed next commit
    inside committedFold_uniform (fun _ r2 v opened => ⟨r2, v, rfl, opened⟩) linked embed
    embedded later

/-- The suffix of a last packing fold: `r·p + ε_fold` for the stage errors, with `foldError` of
the quotient-free reduction, `r + 1` retry loops and the tree size `2·(r + 1)`. `passed`: the
verifier accepts only when each fold challenge decodes and the final claim of the checks is true
for the flat witness of the digit vectors `witnessOf` of the proof. -/
theorem clearFold_suffix {rowsFit : O.rowCount + 1 ≤ 2 ^ batch} {peak : ℕ}
    (injective : Function.Injective block)
    (units : UnitDifferences {challenge | ∃ slot, decode slot = some challenge})
    (sparse : ∀ challenge, (∃ slot, decode slot = some challenge) → Sparse weight peak challenge)
    (base : 2 ≤ fold.responseBase) (small : S.certifiedBase < fieldModulus)
    (varsFit : μ ≤ Fintype.card Coord) (batchFit : batch ≤ Fintype.card Coord)
    (witnessOf : Wire → Witness O fold)
    (inside : (clearFold β key openKey claim μ fits batch rowsFit block decode read weight
      start).finish ≤ total)
    (passed : ∀ point, accept point →
      (clearFold β key openKey claim μ fits batch rowsFit block decode read weight start).Passed
        (witnessOf (point.wire (start + 1))) point) :
    Tier.Suffix accept start
      (clearFold β key openKey claim μ fits batch rowsFit block decode read weight start).Opens
      ((clearFold β key openKey claim μ fits batch rowsFit block decode read weight start).over
        accept (finish (transcript total accept))
        (fun point => witnessOf (point.wire (start + 1))) (fun _ _ => ()))
      ((clearFold β key openKey claim μ fits batch rowsFit block decode read weight
          start).levelError)
      (S.blocks + 1) (2 * (S.blocks + 1)) := by
  exact Tier.suffix_last (Fintype.card_fin _) (clearFold_valid injective units sparse base small
    varsFit batchFit)
    witnessOf inside passed

end Suffixes

end Folds

section Pair

variable {S : Shape} {O : OpeningShape S} {fold : Descriptor} {S' : Shape} {O' : OpeningShape S'}
  {fold' : Descriptor} {E Ring Stmt Brk Wire Coord Slot Result' Source' : Type} [Zero Ring]
  [Field E] [Algebra Zq E] [Fintype Slot] [Nonempty Slot] [Fintype Coord] [DecidableEq Coord]
  {β' : Module.Basis (Fin O'.planes) Zq E} {key' : Key S'} {openKey' : OpeningKey O'}
  {weight' : ℕ}
  (T : Tier (Fin S.blocks) (fun _ => Ring) Stmt Brk (Witness O fold) (Weak O) Wire Coord Slot
    (Option (List E × E)) Table)
  (N : Tier (Fin S'.blocks) (fun _ => Rq O'.carrierDegree) (Claim O' E)
    (LevelBreak key' openKey' (2 * (2 * weight') * fold'.sourceBound S')) (Witness O' fold')
    (Weak O') Wire Coord Slot Result'
    Source')
  (commit : Wire → S'.Payload) (sourceOf : Wire → Source')

/-- The verifier of two levels: the level `T` with the link `Tier.Linked` to the claim of the
last level `N`, and the checks of `N` with a final claim that is true for the source `sourceOf`
of the proof. -/
def pairAccepts (point : Prefix Wire (Option Coord → Slot) N.finish) : Prop :=
  T.Linked N commit point ∧ N.Passed (sourceOf (point.wire (N.start + 1))) point

/-- The checkpoint structure of the last level `N` of two levels. -/
noncomputable def pairLast :
    Structure (Prefix Wire (Option Coord → Slot) N.finish) (Option Coord) Slot (ℕ → Wire)
      (Opened (Weak O') (Prefix Wire (Option Coord → Slot) N.finish))
      (LevelBreak key' openKey' (2 * (2 * weight') * fold'.sourceBound S') ⊕ Unit)
      (Source' ⊕ Unit) ((Σ _ : Fin S'.blocks, Rq O'.carrierDegree) ⊕ Unit) (Option (Weak O')) :=
  N.over (pairAccepts T N commit sourceOf)
    (finish (transcript N.finish (pairAccepts T N commit sourceOf)))
    (fun point => sourceOf (point.wire (N.start + 1))) (fun _ _ => ())

/-- The checkpoint structure of two levels (Akita Proposition 10.44 for the level `T`, whose
flat witness the last level `N` commits to). A witness of `R_bind` is a solution of Module-SIS
for `B`, `D` or a compression map of `T`, or for `A`, `B`, `D` or a compression map of `N`. -/
noncomputable def pair :
    Structure (Prefix Wire (Option Coord → Slot) N.finish) (Option Coord) Slot (ℕ → Wire)
      (Opened (Weak O) (Opened (Weak O') (Prefix Wire (Option Coord → Slot) N.finish)))
      (Brk ⊕ (LevelBreak key' openKey' (2 * (2 * weight') * fold'.sourceBound S') ⊕ Unit))
      (Table ⊕ (Source' ⊕ Unit))
      ((Σ _ : Fin S.blocks, Ring) ⊕ ((Σ _ : Fin S'.blocks, Rq O'.carrierDegree) ⊕ Unit))
      (Option (Weak O)) :=
  T.over (pairAccepts T N commit sourceOf) (pairLast T N commit sourceOf)
    (fun inner => tableOf ((pairLast T N commit sourceOf).witness inner))
    (fun inner other => Sum.inl (Tier.collide key' openKey'
      (2 * (2 * weight') * fold'.sourceBound S')
      ((pairLast T N commit sourceOf).witness inner)
      ((pairLast T N commit sourceOf).witness other)))

variable {T N commit sourceOf}

/-- Two levels are a suffix: the stage errors, the retry loops and the tree size are those of
`Tier.suffix_push` over `Tier.suffix_last`. -/
theorem pair_suffix (valid : T.Valid) (valid' : N.Valid)
    (packed : N.method = packing β' key' openKey' weight') (next : N.start = T.finish)
    (uniform : ∀ wire challenge table other,
      bare (T.check wire challenge table) = bare (T.check wire challenge other))
    (passes : ∀ (table : Table) r2 v, v = mle r2 (lifted E table) → T.passes table (some (r2, v))) :
    Tier.Suffix (pairAccepts T N commit sourceOf) T.start T.Opens (pair T N commit sourceOf)
      (T.levelError + N.levelError)
      (S.blocks + 1 + (S'.blocks + 1)) (2 * (S.blocks + 1) * (2 * (S'.blocks + 1))) :=
  Tier.suffix_push (Fintype.card_fin _) valid packed next commit
    (by
      have ends := N.finish_eq
      omega)
    uniform passes (fun _ accepted => accepted.1) Sum.inl
    (fun solution found => by
      show N.method.broken solution
      rw [packed]
      exact found)
    (Tier.suffix_last (Fintype.card_fin _) valid' sourceOf le_rfl fun _ accepted => accepted.2)

end Pair

section Packing

variable {S : Shape} {O : OpeningShape S} {fold : Descriptor} {S' : Shape} {O' : OpeningShape S'}
  {fold' : Descriptor} {E : Type} [Field E] [Fintype E] [DecidableEq E] [Algebra Zq E]
  {Wire Coord : Type} [Fintype Coord] [DecidableEq Coord] [Nonempty Coord]
  {β : Module.Basis (Fin O.planes) Zq E} {key : Key S} {openKey : OpeningKey O}
  {claim : Wire → Claim O E} {μ : ℕ} {fits : O.cells fold ≤ 2 ^ μ} {batch : ℕ}
  {rowsFit : O.rowCount + 1 ≤ 2 ^ batch} {block : Fin S.blocks → Coord}
  {decode : E → Option (Rq O.carrierDegree)} {read : Reader Wire} {weight : ℕ}
  {β' : Module.Basis (Fin O'.planes) Zq E} {key' : Key S'} {openKey' : OpeningKey O'}
  {claim' : Wire → Claim O' E} {μ' : ℕ} {fits' : O'.cells fold' ≤ 2 ^ μ'} {batch' : ℕ}
  {rowsFit' : O'.rowCount + 1 ≤ 2 ^ batch'} {block' : Fin S'.blocks → Coord}
  {decode' : E → Option (Rq O'.carrierDegree)} {weight' : ℕ} {commit : Wire → S'.Payload}
  {witnessOf : Wire → Witness O' fold'}

local notation "first" =>
  committedFold β key openKey claim μ fits batch rowsFit block decode read weight 0

local notation "second" =>
  clearFold β' key' openKey' claim' μ' fits' batch' rowsFit' block' decode' read weight'
    (Tier.finish first)

local notation "chain" => pair first second commit witnessOf

/-- A committed packing fold over a last packing fold is a suffix from the stage 0. -/
theorem packingPair_suffix {peak peak' : ℕ} (injective : Function.Injective block)
    (units : UnitDifferences {challenge | ∃ slot, decode slot = some challenge})
    (sparse : ∀ challenge, (∃ slot, decode slot = some challenge) → Sparse weight peak challenge)
    (base : 2 ≤ fold.responseBase) (small : S.certifiedBase < fieldModulus)
    (varsFit : μ ≤ Fintype.card Coord) (batchFit : batch ≤ Fintype.card Coord)
    (injective' : Function.Injective block')
    (units' : UnitDifferences {challenge | ∃ slot, decode' slot = some challenge})
    (sparse' : ∀ challenge, (∃ slot, decode' slot = some challenge) →
      Sparse weight' peak' challenge)
    (base' : 2 ≤ fold'.responseBase) (small' : S'.certifiedBase < fieldModulus)
    (varsFit' : μ' ≤ Fintype.card Coord) (batchFit' : batch' ≤ Fintype.card Coord) :
    Tier.Suffix (pairAccepts first second commit witnessOf) 0 (Tier.Opens first) chain
      (Tier.levelError first +
        (Tier.levelError second))
      (S.blocks + 1 + (S'.blocks + 1)) (2 * (S.blocks + 1) * (2 * (S'.blocks + 1))) :=
  pair_suffix (committedFold_valid injective units sparse base small varsFit batchFit)
    (clearFold_valid injective' units' sparse' base' small' varsFit' batchFit') rfl rfl
    committedFold_uniform (fun _ r2 v opened => ⟨r2, v, rfl, opened⟩)

/-- An output of the extractor for a committed packing fold over a last packing fold: the
verifier accepts the transcript of the prover, and the output is a root weak opening of the claim
of the first level with `κ̄_1 = 2ω` and `β̄_∞ = Δ^cert_f` (Akita Definition 10.22), or a solution
of Module-SIS for `B`, `D` or a compression map of the first level, or for `A`, `B`, `D` or a
compression map of the last level. By `Tier.opens_evaluation`, the value of the root claim is
the multilinear evaluation of the table of the root weak opening at the point of the claim. -/
theorem packingPair_extract_sound {peak peak' : ℕ} (injective : Function.Injective block)
    (units : UnitDifferences {challenge | ∃ slot, decode slot = some challenge})
    (sparse : ∀ challenge, (∃ slot, decode slot = some challenge) → Sparse weight peak challenge)
    (base : 2 ≤ fold.responseBase) (small : S.certifiedBase < fieldModulus)
    (varsFit : μ ≤ Fintype.card Coord) (batchFit : batch ≤ Fintype.card Coord)
    (injective' : Function.Injective block')
    (units' : UnitDifferences {challenge | ∃ slot, decode' slot = some challenge})
    (sparse' : ∀ challenge, (∃ slot, decode' slot = some challenge) →
      Sparse weight' peak' challenge)
    (base' : 2 ≤ fold'.responseBase) (small' : S'.certifiedBase < fieldModulus)
    (varsFit' : μ' ≤ Fintype.card Coord) (batchFit' : batch' ≤ Fintype.card Coord)
    (prover : OracleComp (Prefix Wire (Option Coord → E) (Tier.finish second)) (Option Coord → E)
      (ℕ → Wire))
    (budget : ℕ) (table : Prefix Wire (Option Coord → E) (Tier.finish second) → Option Coord → E)
    (tape : Tape (Prefix Wire (Option Coord → E) (Tier.finish second)) (Option Coord) E budget
      (Structure.protocol chain).stages)
    (out : Opened (Weak O) (Opened (Weak O') (Prefix Wire (Option Coord → E) (Tier.finish second)))
      ⊕
      (LevelBreak key openKey (2 * (2 * weight) * fold.sourceBound S) ⊕
        (LevelBreak key' openKey' (2 * (2 * weight') * fold'.sourceBound S') ⊕ Unit)))
    (result : Checkpoint.extract chain prover budget table tape = some out) :
    pairAccepts first second commit witnessOf ((Structure.protocol chain).final prover table) ∧
      Sum.elim
        (fun certificate => ∃ ω, certificate.opening = some ω ∧
          ω.Opens β (2 * weight) (fold.sourceBound S)
            ((claim (((Structure.protocol chain).input prover table 0).wire 0)).statement key
              openKey)
            (S.decode ω.source))
        (Sum.elim (fun solution => solution.isSome)
          (Sum.elim (fun solution => solution.isSome) fun _ => False))
        out :=
  (packingPair_suffix injective units sparse base small varsFit batchFit injective' units'
    sparse' base' small' varsFit' batchFit').extract_sound rfl prover budget table tape out result

variable [Finite Wire]

/-- Akita Theorem 10.17 for a committed packing fold over a last packing fold, which is Akita
Corollary 10.49 with (206) for two levels of the opening suffix. For each prover with at most `Q`
queries to the oracle, the probability that the verifier accepts and the extractor returns no
output is at most
`(Q + 1)·(r·p + ε_fold + r'·p' + ε'_fold) + (Q + 1)·(r + 1 + r' + 1)/budget`. `r` and `r'` are
the numbers of blocks of the two levels, `p` and `p'` are the largest probabilities of one fold
challenge (`Tier.chance`), `ε_fold` and `ε'_fold` are `foldError` of the quotient-free reductions,
which are the same for each statement and each challenge, and `budget` is the limit of the
attempts of a retry loop of the extractor. By `packingPair_extract_sound`, an output is a root
weak opening or a solution of Module-SIS. -/
theorem packingPair_extraction_le {peak peak' : ℕ} (injective : Function.Injective block)
    (units : UnitDifferences {challenge | ∃ slot, decode slot = some challenge})
    (sparse : ∀ challenge, (∃ slot, decode slot = some challenge) → Sparse weight peak challenge)
    (base : 2 ≤ fold.responseBase) (small : S.certifiedBase < fieldModulus)
    (varsFit : μ ≤ Fintype.card Coord) (batchFit : batch ≤ Fintype.card Coord)
    (injective' : Function.Injective block')
    (units' : UnitDifferences {challenge | ∃ slot, decode' slot = some challenge})
    (sparse' : ∀ challenge, (∃ slot, decode' slot = some challenge) →
      Sparse weight' peak' challenge)
    (base' : 2 ≤ fold'.responseBase) (small' : S'.certifiedBase < fieldModulus)
    (varsFit' : μ' ≤ Fintype.card Coord) (batchFit' : batch' ≤ Fintype.card Coord)


    (prover : OracleComp (Prefix Wire (Option Coord → E) (Tier.finish second)) (Option Coord → E)
      (ℕ → Wire))
    {queries : ℕ} (bounded : prover.QueryBound queries) {budget : ℕ} (positive : 0 < budget) :
    𝔼 table, 𝔼 tape : Tape (Prefix Wire (Option Coord → E) (Tier.finish second)) (Option Coord) E
      budget
          (Structure.protocol chain).stages,
        (if pairAccepts first second commit witnessOf
            ((Structure.protocol chain).final prover table) ∧
          Checkpoint.extract chain prover budget table tape = none then (1 : ℝ) else 0) ≤
      (queries + 1) * (Tier.levelError first +
          (Tier.levelError second)) +
        (queries + 1) * (S.blocks + 1 + (S'.blocks + 1) : ℕ) / budget := by
  exact (packingPair_suffix injective units sparse base small varsFit batchFit injective' units'
    sparse' base' small' varsFit' batchFit').extraction_le rfl prover bounded positive

/-- The expected number of runs of the prover in the extraction of a committed packing fold over
a last packing fold is at most `K̂ + (Q + 1)·(K̂ − 1)` with `K̂ = 2·(r + 1)·2·(r' + 1)` for `r`
and `r'` blocks (Akita Theorem 10.17, the cost). -/
theorem packingPair_runs_le {peak peak' : ℕ} (injective : Function.Injective block)
    (units : UnitDifferences {challenge | ∃ slot, decode slot = some challenge})
    (sparse : ∀ challenge, (∃ slot, decode slot = some challenge) → Sparse weight peak challenge)
    (base : 2 ≤ fold.responseBase) (small : S.certifiedBase < fieldModulus)
    (varsFit : μ ≤ Fintype.card Coord) (batchFit : batch ≤ Fintype.card Coord)
    (injective' : Function.Injective block')
    (units' : UnitDifferences {challenge | ∃ slot, decode' slot = some challenge})
    (sparse' : ∀ challenge, (∃ slot, decode' slot = some challenge) →
      Sparse weight' peak' challenge)
    (base' : 2 ≤ fold'.responseBase) (small' : S'.certifiedBase < fieldModulus)
    (varsFit' : μ' ≤ Fintype.card Coord) (batchFit' : batch' ≤ Fintype.card Coord)
    (prover : OracleComp (Prefix Wire (Option Coord → E) (Tier.finish second)) (Option Coord → E)
      (ℕ → Wire))
    {queries : ℕ} (bounded : prover.QueryBound queries) (budget : ℕ) :
    𝔼 table, 𝔼 tape : Tape (Prefix Wire (Option Coord → E) (Tier.finish second)) (Option Coord) E
      budget
          (Structure.protocol chain).stages,
        (extractRuns chain prover budget table tape : ℝ) ≤
      (2 * (S.blocks + 1) * (2 * (S'.blocks + 1)) : ℕ) +
        (queries + 1) * ((2 * (S.blocks + 1) * (2 * (S'.blocks + 1)) : ℕ) - 1) :=
  (packingPair_suffix injective units sparse base small varsFit batchFit injective' units'
    sparse' base' small' varsFit' batchFit').runs_le rfl prover bounded budget

end Packing

end NightstreamFPrime.Spec.Compression.Akita.Extract
