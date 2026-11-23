import NightstreamFPrime.Spec.Compression.Akita.Extract.TowerComplete

/-! The honest prover of a packing level of Akita on the quotient-lift route, and the verifier of
the levels from the packing level to the terminal (ePrint 2026/1983, revision of 2026-11-23,
Theorem 10.9 with Lemma 10.7, Lemma 7.1, item 1, Section 4.5, Figure 4 and Section 8.2).

- `sourceClaim` is the claim of the honest prover of a packing fold for given source blocks, and
  `sourceClaim_satisfies` is `honest_satisfies` for it and the state of `commitSource`. The source
  blocks of a schedule after the root are the bounded source `s_i := f_i` of Akita Section 4.5.
  `sourceClaim_value` states that the value of the claim for the bounded source of a table is the
  multilinear evaluation of the table.
- `liftNext` is the table that the next level commits to: the flat witness of the honest prover
  with the digits of the quotients (Akita (76)). `LiftFit` states that for each fold challenge
  vector of the family the response is admissible and a condition `after` holds for this table.
  `liftLevel_wins` gives the honest prover of the checks of the level `liftFold`, and the cells of
  `liftNext` after the quotient digits are 0.
- `packTower_complete` gives, for a transcript prefix on the path of the table of the oracle
  whose last message has the claim `sourceClaim` of the packing level, the later messages of a
  proof such that the complete transcript has the link `Tier.TraceLinked` from the packing level
  to the first trace level and the links and the checks of `traceTower_complete`.

The fit conditions `LiftFit` and `TraceFit` are hypotheses (Akita Definition 10.5). The file has
no bound for the probability `ξ_h` of Akita (155). -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Extract

open scoped BigOperators
open NightstreamFPrime.Spec.Compression.Lib (mle mle_congr eqAt)
open NightstreamFPrime.Spec.Checkpoint

local instance : Fact (Nat.Prime fieldModulus) := ⟨FieldPrime.fieldModulus_natPrime⟩

attribute [local instance low] Classical.propDecidable

section Claim

variable {S : Shape} {O : OpeningShape S} {fold : Descriptor} {E : Type} [Field E]
  [Algebra Zq E]

/-- The claim of the honest prover of a packing fold for the source blocks `source` and an opening
point (Akita Figure 4, step 1): the payload `u_pub` of `commitSource`, the payload `v_pub` of the
opening path for the digits `ê`, and the value (59) of the partials. -/
noncomputable def sourceClaim (β : Module.Basis (Fin O.planes) Zq E) (key : Key S)
    (openKey : OpeningKey O) (source : Fin S.blocks → S.Source) (point : Point O E) : Claim O E :=
  ⟨(commitSource key source).1, (honestOpening openKey (honestPartials β point source)).2, point,
    scalar β point fun i t => contract β point t (gadget S.sourceBase (source i))⟩

variable {β : Module.Basis (Fin O.planes) Zq E} {key : Key S} {openKey : OpeningKey O}

/-- `honest_satisfies` for the claim `sourceClaim` and the state of `commitSource` (Akita
Lemma 10.7 for a packing fold with a bounded source). -/
theorem sourceClaim_satisfies (source : Fin S.blocks → S.Source) (point : Point O E)
    (challenge : Fin S.blocks → Rq O.carrierDegree) (shape : S.Bounded)
    (radix : Radix fold.responseBase) (certified : fold.responseBase ≤ S.certifiedBase)
    (path : PathFits O.openDegree (1 * O.openRank) O.chain)
    (encodable : Encodable fold.responseBase fold.responseDigits (response O challenge source)) :
    (honest β openKey point fold challenge (commitSource key source).2).Satisfies β
      ((sourceClaim β key openKey source point).statement key openKey) challenge :=
  honest_satisfies (st := (sourceClaim β key openKey source point).statement key openKey)
    challenge shape.opening shape.certified radix certified path
    (commitSource_opens shape key source) encodable rfl rfl

/-- The value of the claim `sourceClaim` for the bounded source `s_i := f_i` of a table, at a
point with four segments, is the multilinear evaluation of the table (Akita Section 5.3), for a
table that is 0 after the `B·M·d_A` cells of the `B ≤ 2^{r_blk}` live blocks (Akita Section 4.5,
(40) to (42)). -/
theorem sourceClaim_value (shape : S.Bounded) (table : Table) (block position pack tail : List E)
    (blocks : S.blocks ≤ 2 ^ block.length) (length : S.blockLength = 2 ^ position.length)
    (stride : O.stride = 2 ^ pack.length) (degree : O.carrierDegree = 2 ^ tail.length)
    (live : ∀ cell, S.blocks * (S.blockLength * S.innerDegree) ≤ cell → table cell = 0) :
    (sourceClaim β key openKey (fun i => S.embed (S.pack table i))
      (Point.ofSegments O block position pack tail)).value =
      mle (pack ++ (tail ++ (position ++ block))) (lifted E table) := by
  show scalar β _ (fun i t => contract β _ t (gadget S.sourceBase (S.embed (S.pack table i)))) = _
  rw [scalar_contract_eq_mle β block position pack tail blocks length stride degree,
    decode_embed_pack shape table]
  refine mle_congr _ fun cell _ => ?_
  show algebraMap Zq E (S.pad table cell) = algebraMap Zq E (table cell)
  by_cases inside : cell / (S.blockLength * S.innerDegree) < S.blocks
  · rw [Shape.pad, ite_eq_left inside]
  · rw [Shape.pad, ite_eq_right inside,
      live cell (Nat.mul_le_of_le_div _ _ _ (Nat.le_of_not_lt inside))]

/-- The table that the next level commits to after a packing fold on the quotient-lift route: the
flat witness of the honest prover for the source blocks, the point and the fold challenges, with
the digits of the quotients of its ring rows at the places of Akita (76) (Figure 4, step 3). -/
noncomputable def liftNext (β : Module.Basis (Fin O.planes) Zq E) (key : Key S)
    (openKey : OpeningKey O) (fold : Descriptor) (μ : ℕ) (cells : O.cells fold ≤ 2 ^ μ)
    (digits : ℕ) (source : Fin S.blocks → S.Source) (point : Point O E)
    (challenge : Fin S.blocks → Rq O.carrierDegree) : Table :=
  (flatRelation O fold β ((sourceClaim β key openKey source point).statement key openKey)
    challenge μ cells).completion (O.gaps fold digits) S.certifiedBase digits
    (honest β openKey point fold challenge (commitSource key source).2).flat

/-- The fit conditions of the honest prover of a packing fold for the source blocks `source`
(Akita Definition 10.5 and Lemma 10.7): for each opening point and each fold challenge vector of
the family, the response `z = Σ_i ι(c_i)·s_i` satisfies `Enc`, and `after` holds for the table
`liftNext`. Akita Theorem 10.9 has the probability that a response does not fit in the term `ξ_h`
of (155). -/
def LiftFit (β : Module.Basis (Fin O.planes) Zq E) (key : Key S) (openKey : OpeningKey O)
    (fold : Descriptor) (μ : ℕ) (cells : O.cells fold ≤ 2 ^ μ) (digits : ℕ)
    (decode : E → Option (Rq O.carrierDegree)) (source : Fin S.blocks → S.Source)
    (after : Table → Prop) : Prop :=
  ∀ (point : Point O E) (challenge : Fin S.blocks → Rq O.carrierDegree),
    (∀ i, ∃ slot, decode slot = some (challenge i)) →
    Encodable fold.responseBase fold.responseDigits (response O challenge source) ∧
      after (liftNext β key openKey fold μ cells digits source point challenge)

variable [Fintype E] [DecidableEq E] {Wire Coord : Type} {claim : Wire → Claim O E} {μ : ℕ}
  {cells : O.cells fold ≤ 2 ^ μ} {digits : ℕ} {room : O.quotientRoom fold digits ≤ 2 ^ μ}
  {batch : ℕ} {rowsFit : O.rowCount + 1 ≤ 2 ^ batch} {block : Fin S.blocks → Coord}
  {decode : E → Option (Rq O.carrierDegree)} {read : Reader Wire} {weight start : ℕ}

/-- The honest prover of the checks of a packing level on the quotient-lift route (Akita
Lemma 10.7 and Lemma 7.1, item 1, with Theorem 10.9). The message `wire` has the claim
`sourceClaim`, and `fit` has the fit conditions. Then for each fold challenge vector of the
family, `after` holds for the table `liftNext`, this table is 0 from the end `quotientRoom` of the
quotient digits, and the prover has a message for each draw sequence of the checks for it. `shape`,
`radix`, `certified`, `path`: the conditions of `sourceClaim_satisfies`. `base` and `covers`: the
base `b⋆` of the quotient digits has the conditions `Radix`, and each element of `Z_q` has
`digits` balanced digits. `range`, `four`, `positive`: a message can have a round polynomial. -/
theorem liftLevel_wins {sendable : (M : Type) → M → Prop} {slots width cap : ℕ}
    (sends : Sends sendable E width cap) (range : rangeWidth S.certifiedBase ≤ width)
    (four : 4 ≤ width) (positive : 0 < slots) (shape : S.Bounded)
    (radix : Radix fold.responseBase) (certified : fold.responseBase ≤ S.certifiedBase)
    (path : PathFits O.openDegree (1 * O.openRank) O.chain) (base : Radix S.certifiedBase)
    (covers : fieldModulus ≤ S.certifiedBase ^ digits)
    {source : Fin S.blocks → S.Source} {point : Point O E} {wire : Wire}
    (ready : claim wire = sourceClaim β key openKey source point) {after : Table → Prop}
    (fit : LiftFit β key openKey fold μ cells digits decode source after)
    {c : Fin S.blocks → Rq O.carrierDegree} (member : ∀ i, ∃ slot, decode slot = some (c i)) :
    after (liftNext β key openKey fold μ cells digits source point c) ∧
      (∀ x, O.quotientRoom fold digits ≤ x →
        liftNext β key openKey fold μ cells digits source point c x = 0) ∧
      WinsAt sendable slots
        ((liftFold β key openKey claim μ cells digits room batch rowsFit block decode read weight
          start).check wire c (liftNext β key openKey fold μ cells digits source point c))
        (foldDraws μ S.certifiedBase (O.compressed fold)) 0 fun result _ place => place ≤ 1 ∧
          ∃ r2 : List E, r2.length = μ ∧ result = some (r2, mle r2
            (lifted E (liftNext β key openKey fold μ cells digits source point c))) := by
  obtain ⟨encodable, later⟩ := fit point c member
  have satisfies := sourceClaim_satisfies (β := β) (key := key) (openKey := openKey)
    (fold := fold) source point c shape radix certified path encodable
  rw [← ready] at satisfies
  have wins := liftFold_winsAt (cells := cells) (room := room) (batch := batch)
    (rowsFit := rowsFit) (block := block) (decode := decode) (read := read) (weight := weight)
    (start := start) sends range four base covers positive satisfies
  rw [ready] at wins
  exact ⟨later, fun x large => FlatRelation.completion_of_le _
    (flatRelation_counts β _ c μ cells digits) _
    ((flatRelation_quotientEnd β _ c μ cells digits).trans_le large), wins⟩

end Claim

section Tower

variable {E : Type} [Field E] [Fintype E] [DecidableEq E] [Algebra Zq E] [Degree2 E]
  {S₁ : Shape} {O₁ : OpeningShape S₁} {fold₁ : Descriptor} {S₂ : Shape} {O₂ : OpeningShape S₂}
  {fold₂ : Descriptor} {S₃ : Shape} {O₃ : OpeningShape S₃} {fold₃ : Descriptor} {S₄ : Shape}
  {O₄ : OpeningShape S₄} {fold₄ : Descriptor} {S₅ : Shape} {O₅ : OpeningShape S₅}
  {fold₅ : Descriptor} {S₆ : Shape}
  {Wire Coord : Type} [Fintype Coord] [DecidableEq Coord] {total : ℕ}
  {accept : Prefix Wire (Option Coord → E) total → Prop} {read : Reader Wire} {start : ℕ}
  {β₁ : Module.Basis (Fin O₁.planes) Zq E} {key₁ : Key S₁} {openKey₁ : OpeningKey O₁}
  {claim₁ : Wire → Claim O₁ E} {μ₁ : ℕ} {fits₁ : O₁.cells fold₁ ≤ 2 ^ μ₁} {digits₁ : ℕ}
  {room₁ : O₁.quotientRoom fold₁ digits₁ ≤ 2 ^ μ₁} {batch₁ : ℕ}
  {rowsFit₁ : O₁.rowCount + 1 ≤ 2 ^ batch₁} {block₁ : Fin S₁.blocks → Coord}
  {decode₁ : E → Option (Rq O₁.carrierDegree)} {weight₁ ν₁ : ℕ}
  {key₂ : Key S₂} {openKey₂ : OpeningKey O₂} {claim₂ : Wire → TraceClaim O₂ E}
  {μ₂ : ℕ} {fits₂ : O₂.cells fold₂ ≤ 2 ^ μ₂} {batch₂ : ℕ}
  {rowsFit₂ : O₂.traceRowCount + 1 ≤ 2 ^ batch₂} {block₂ : Fin S₂.blocks → Coord}
  {decode₂ : E → Option (Rq S₂.innerDegree)} {weight₂ ν₂ m₂ : ℕ}
  {key₃ : Key S₃} {openKey₃ : OpeningKey O₃} {claim₃ : Wire → TraceClaim O₃ E}
  {μ₃ : ℕ} {fits₃ : O₃.cells fold₃ ≤ 2 ^ μ₃} {batch₃ : ℕ}
  {rowsFit₃ : O₃.traceRowCount + 1 ≤ 2 ^ batch₃} {block₃ : Fin S₃.blocks → Coord}
  {decode₃ : E → Option (Rq S₃.innerDegree)} {weight₃ ν₃ m₃ : ℕ}
  {key₄ : Key S₄} {openKey₄ : OpeningKey O₄} {claim₄ : Wire → TraceClaim O₄ E}
  {μ₄ : ℕ} {fits₄ : O₄.cells fold₄ ≤ 2 ^ μ₄} {batch₄ : ℕ}
  {rowsFit₄ : O₄.traceRowCount + 1 ≤ 2 ^ batch₄} {block₄ : Fin S₄.blocks → Coord}
  {decode₄ : E → Option (Rq S₄.innerDegree)} {weight₄ ν₄ m₄ : ℕ}
  {key₅ : Key S₅} {openKey₅ : OpeningKey O₅} {claim₅ : Wire → TraceClaim O₅ E}
  {μ₅ : ℕ} {fits₅ : O₅.cells fold₅ ≤ 2 ^ μ₅} {batch₅ : ℕ}
  {rowsFit₅ : O₅.traceRowCount + 1 ≤ 2 ^ batch₅} {block₅ : Fin S₅.blocks → Coord}
  {decode₅ : E → Option (Rq S₅.innerDegree)} {weight₅ ν₅ m₅ : ℕ}
  {operator₃ limit₃ operator₄ limit₄ operator₅ limit₅ operatorL limitL : ℕ}
  {L : Terminal S₆ E Wire Coord E total (.euclidean operatorL limitL)} {m₆ : ℕ}
  {commit₂ : Wire → S₂.Payload} {commit₃ : Wire → S₃.Payload} {commit₄ : Wire → S₄.Payload}
  {commit₅ : Wire → S₅.Payload}
  {stateOf : Wire → Fin S₆.blocks → Fin S₆.innerRank → Rq S₆.innerDegree}
  {statementOf : Wire → TerminalStatement S₆ E}
  {partialsOf : Wire → Fin S₆.blocks → Rq S₆.innerDegree}

local notation "base₁" =>
  liftFold β₁ key₁ openKey₁ claim₁ μ₁ fits₁ digits₁ room₁ batch₁ rowsFit₁ block₁ decode₁ read
    weight₁ start

local notation "first" => Tier.tensored base₁ ν₁

local notation "second" =>
  Tier.tensored (committedTrace (fold := fold₂) key₂ openKey₂ claim₂ μ₂ fits₂ batch₂ rowsFit₂
    block₂ decode₂ read weight₂ (Tier.finish first)) ν₂

local notation "third" =>
  Tier.tensored (committedTraceSq (fold := fold₃) key₃ openKey₃ claim₃ μ₃ fits₃ batch₃ rowsFit₃
    block₃ decode₃ read weight₃ operator₃ limit₃ (Tier.finish second)) ν₃

local notation "fourth" =>
  Tier.tensored (committedTraceSq (fold := fold₄) key₄ openKey₄ claim₄ μ₄ fits₄ batch₄ rowsFit₄
    block₄ decode₄ read weight₄ operator₄ limit₄ (Tier.finish third)) ν₄

local notation "fifth" =>
  Tier.tensored (committedTraceSq (fold := fold₅) key₅ openKey₅ claim₅ μ₅ fits₅ batch₅ rowsFit₅
    block₅ decode₅ read weight₅ operator₅ limit₅ (Tier.finish fourth)) ν₅

/-- Akita Theorem 10.9 for the packing level, the trace levels and the terminal of the opening,
without the probability that a response does not fit and without the term of (155) for a zero tensor
factor, because this verifier has no rejection at a zero factor: the verifier of `mixedTower_suffix`
from the packing level accepts the honest prover. `early` is a transcript prefix on the path of the
table of the oracle whose last message has the claim `sourceClaim` of the level 1 for the source
blocks `source`, and `tried`: the nonce of the response search in this message is below
`responseTries`. Then later messages exist such that the complete transcript has the link
`Tier.TraceLinked` from the level 1 to the level 2, the links from the level 2 to the level 5, the
link `Terminal.Linked` and the checks `Terminal.Passed`. `shape₁`, `radix₁`, `certified₁`, `path₁`,
`digit₁`, `covers₁`, `range₁`: the conditions of `liftLevel_wins` on the level 1. `vars₁`, `point₁`,
`covered₁`: the tensor reduction of the level 1 has `μ_1 − 1` tail variables, its point has the
segments of the shape of the level 2, and the live blocks of the level 2 have each cell before the
end of the quotient digits of the level 1 (Akita Section 4.5, (40) to (42)). `carries₁`: a wire
message can have the commitment and the claim of the level 2. `decodes₁`: each slot of the table of
the oracle decodes to a fold challenge of the level 1, and `worked`: each work slot of the table
has 57 zero work bits. `fit`: the fit conditions `LiftFit` of the
level 1, `TraceFit` of the levels 2 to 5, and `‖z‖_2² ≤ S_max,term` at the terminal. The other
hypotheses are those of `traceTower_complete`. -/
theorem packTower_complete {sendable : (M : Type) → M → Prop} {slots width cap : ℕ}
    (sends : Sends sendable E width cap) (four : 4 ≤ width) (room : 3 < slots)
    (carries₁ : Carries sendable slots read fun wire => (commit₂ wire, claim₂ wire))
    (carries₂ : Carries sendable slots read fun wire => (commit₃ wire, claim₃ wire))
    (carries₃ : Carries sendable slots read fun wire => (commit₄ wire, claim₄ wire))
    (carries₄ : Carries sendable slots read fun wire => (commit₅ wire, claim₅ wire))
    (carries₅ : Carries sendable slots read fun wire =>
      (stateOf wire, statementOf wire, partialsOf wire))
    (carriesL : Carries sendable slots read L.response)
    {p₂ s₂ k₂ p₃ s₃ k₃ p₄ s₄ k₄ p₅ s₅ k₅ p₆ s₆ k₆ : ℕ}
    (shape₁ : S₁.Bounded) (radix₁ : Radix fold₁.responseBase)
    (certified₁ : fold₁.responseBase ≤ S₁.certifiedBase)
    (path₁ : PathFits O₁.openDegree (1 * O₁.openRank) O₁.chain)
    (digit₁ : Radix S₁.certifiedBase)
    (covers₁ : fieldModulus ≤ S₁.certifiedBase ^ digits₁)
    (range₁ : rangeWidth S₁.certifiedBase ≤ width)
    (quarter₂ : 4 * m₂ = S₂.innerDegree) (quarter₃ : 4 * m₃ = S₃.innerDegree)
    (quarter₄ : 4 * m₄ = S₄.innerDegree) (quarter₅ : 4 * m₅ = S₅.innerDegree)
    (quarter₆ : 4 * m₆ = S₆.innerDegree)
    (source₂ : SourceSized S₂ m₂ p₂ s₂ k₂) (source₃ : SourceSized S₃ m₃ p₃ s₃ k₃)
    (source₄ : SourceSized S₄ m₄ p₄ s₄ k₄) (source₅ : SourceSized S₅ m₅ p₅ s₅ k₅)
    (source₆ : SourceSized S₆ m₆ p₆ s₆ k₆)
    (sized₂ : FoldSized S₂ O₂ fold₂ width) (sized₃ : FoldSized S₃ O₃ fold₃ width)
    (sized₄ : FoldSized S₄ O₄ fold₄ width) (sized₅ : FoldSized S₅ O₅ fold₅ width)
    (vars₁ : μ₁ = ν₁ + 1) (vars₂ : μ₂ = ν₂ + 1) (vars₃ : μ₃ = ν₃ + 1) (vars₄ : μ₄ = ν₄ + 1)
    (vars₅ : μ₅ = ν₅ + 1)
    (point₁ : ν₁ = p₂ + (s₂ + k₂)) (point₂ : ν₂ = p₃ + (s₃ + k₃))
    (point₃ : ν₃ = p₄ + (s₄ + k₄)) (point₄ : ν₄ = p₅ + (s₅ + k₅))
    (point₅ : ν₅ = p₆ + (s₆ + k₆))
    (covered₁ : O₁.quotientRoom fold₁ digits₁ ≤ 2 * (S₂.blocks * (S₂.blockLength * (2 * m₂))))
    (covered₂ : O₂.cells fold₂ ≤ 2 * (S₃.blocks * (S₃.blockLength * (2 * m₃))))
    (covered₃ : O₃.cells fold₃ ≤ 2 * (S₄.blocks * (S₄.blockLength * (2 * m₄))))
    (covered₄ : O₄.cells fold₄ ≤ 2 * (S₅.blocks * (S₅.blockLength * (2 * m₅))))
    (covered₅ : O₅.cells fold₅ ≤ 2 * (S₆.blocks * (S₆.blockLength * (2 * m₆))))
    (normRoom₃ : S₃.blockLength * S₃.sourceDigits * S₃.innerDegree ≤ 2 ^ μ₃)
    (normRoom₄ : S₄.blockLength * S₄.sourceDigits * S₄.innerDegree ≤ 2 ^ μ₄)
    (normRoom₅ : S₅.blockLength * S₅.sourceDigits * S₅.innerDegree ≤ 2 ^ μ₅)
    (wide₃ : rangeWidthSq S₃.certifiedBase ≤ width) (wide₄ : rangeWidthSq S₄.certifiedBase ≤ width)
    (wide₅ : rangeWidthSq S₅.certifiedBase ≤ width)
    (capped₃ : limit₃ ≤ cap) (capped₄ : limit₄ ≤ cap) (capped₅ : limit₅ ≤ cap)
    (next : L.start = Tier.finish fifth) (inside : L.start < total)
    (reads : ∀ point, L.read point = statementOf (point.wire L.start))
    (clears : ∀ point, L.partials point = partialsOf (point.wire L.start))
    (counts : ∀ wire, L.nonce wire = read.nonce wire)
    (table : Prefix Wire (Option Coord → E) total → Option Coord → E)
    (decodes₁ : ∀ input i, (decode₁ (table input (some (block₁ i)))).isSome)
    (decodes₂ : ∀ input i, (decode₂ (table input (some (block₂ i)))).isSome)
    (decodes₃ : ∀ input i, (decode₃ (table input (some (block₃ i)))).isSome)
    (decodes₄ : ∀ input i, (decode₄ (table input (some (block₄ i)))).isSome)
    (decodes₅ : ∀ input i, (decode₅ (table input (some (block₅ i)))).isSome)
    (decodesL : ∀ input i, (L.decode (table input (L.slot i))).isSome)
    (worked : ∀ input, table input none ∈ workZeros E 57)
    {wire₀ : Wire} {early : List ((Option Coord → E) × Wire)} (on : OnPath accept table wire₀ early)
    (starts : early.length = start) {source : Fin S₁.blocks → S₁.Source} {point : Point O₁ E}
    (ready : claim₁ (lastWire wire₀ early) = sourceClaim β₁ key₁ openKey₁ source point)
    (tried : read.nonce (lastWire wire₀ early) < responseTries)
    (fit : LiftFit β₁ key₁ openKey₁ fold₁ μ₁ fits₁ digits₁ decode₁ source fun w₁ =>
      TraceFit quarter₂ key₂ openKey₂ fold₂ (fun _ => decode₂) (fun _ => True) (packed w₁)
        fun w₂ =>
      TraceFit quarter₃ key₃ openKey₃ fold₃ (fun _ => decode₃) (fun z => vecNormSq z ≤ limit₃)
        (packed w₂) fun w₃ =>
      TraceFit quarter₄ key₄ openKey₄ fold₄ (fun _ => decode₄) (fun z => vecNormSq z ≤ limit₄)
        (packed w₃) fun w₄ =>
      TraceFit quarter₅ key₅ openKey₅ fold₅ (fun _ => decode₅) (fun z => vecNormSq z ≤ limit₅)
        (packed w₄) fun w₅ =>
      ∀ c : Fin S₆.blocks → Rq S₆.innerDegree, (∀ i, ∃ slot, L.decode slot = some (c i)) →
        vecNormSq (traceResponse c (traceSource quarter₆ (packed w₅ : ℕ → E))) ≤ limitL) :
    ∃ later, OnPath accept table wire₀ (early ++ later) ∧ (early ++ later).length = total ∧
      ∀ stage : Prefix Wire (Option Coord → E) total, stage.first = wire₀ →
        stage.rest = early ++ later →
        Tier.TraceLinked first second quarter₂ commit₂ stage ∧
        Tier.TraceLinked second third quarter₃ commit₃ stage ∧
        Tier.TraceLinked third fourth quarter₄ commit₄ stage ∧
        Tier.TraceLinked fourth fifth quarter₅ commit₅ stage ∧
        Terminal.Linked fifth L quarter₆ stateOf stage ∧ L.Passed stage := by
  have ends₁ := Tier.finish_eq first
  have ends₂ := Tier.finish_eq second
  have ends₃ := Tier.finish_eq third
  have ends₄ := Tier.finish_eq fourth
  have ends₅ := Tier.finish_eq fifth
  have fin₁ : Tier.finish first ≤ total :=
    (show Tier.finish first ≤ Tier.finish second from
      (Nat.le_add_right _ _).trans_eq ends₂.symm).trans
    ((show Tier.finish second ≤ Tier.finish third from
      (Nat.le_add_right _ _).trans_eq ends₃.symm).trans
    ((show Tier.finish third ≤ Tier.finish fourth from
      (Nat.le_add_right _ _).trans_eq ends₄.symm).trans
    ((show Tier.finish fourth ≤ Tier.finish fifth from
      (Nat.le_add_right _ _).trans_eq ends₅.symm).trans (next ▸ inside.le))))
  have short₁ : early.length ≤ total :=
    (starts.le.trans ((Nat.le_add_right _ _).trans_eq ends₁.symm)).trans fin₁
  have member : ∀ i, ∃ slot, decode₁ slot = some (Tier.drawn base₁ table wire₀ early i) :=
    fun i => ⟨_, Tier.drawn_mem base₁ table short₁ decodes₁ i⟩
  obtain ⟨fit₂, zero₁, wins₁⟩ := liftLevel_wins (claim := claim₁) (room := room₁)
    (batch := batch₁) (rowsFit := rowsFit₁) (block := block₁) (read := read) (weight := weight₁)
    (start := start) sends range₁ four (Nat.zero_lt_of_lt room) shape₁ radix₁ certified₁ path₁
    digit₁ covers₁ ready fit member
  obtain ⟨later₁, blockSeg, positionSeg, packSeg, θ, packs, positions, long, on₁, len₁, fresh₁,
      ready₂, linked₁⟩ :=
    traceLink (B := base₁) (ν := ν₁) (Next := second) (commit' := commit₂) carries₁ sends
      (Nat.le_of_succ_le four) on starts fin₁ decodes₁ worked
      ⟨foldWorkBits_le _ _, map_workBits_le _⟩ tried
      (fun _ _ table other => foldCheck_bare _ _ _ _ table other) (foldCosts_length E _ _ _ _ _)
      (Nat.lt_of_succ_lt (Nat.lt_of_succ_lt room)) _
      (wins₁.mono fun _ _ _ kept => ⟨kept.1, kept.2.imp fun _ both => ⟨both.1.trans vars₁, both.2⟩⟩)
      rfl quarter₂ key₂ openKey₂ source₂.pack source₂.length source₂.blocks point₁
  have live₂ := fun cell (large : S₂.blocks * (S₂.blockLength * (2 * m₂)) ≤ cell) =>
    packed_of_le (E := E) zero₁ (covered₁.trans (Nat.mul_le_mul_left 2 large))
  obtain ⟨later, onL, whole, each⟩ := traceTower_complete sends four room carries₂ carries₃
    carries₄ carries₅ carriesL quarter₂ quarter₃ quarter₄ quarter₅ quarter₆ source₂ source₃
    source₄ source₅ source₆ sized₂ sized₃ sized₄ sized₅ vars₂ vars₃ vars₄ vars₅ point₂ point₃
    point₄ point₅ covered₂ covered₃ covered₄ covered₅ normRoom₃ normRoom₄ normRoom₅ wide₃ wide₄
    wide₅ capped₃ capped₄ capped₅ next inside reads clears counts table decodes₂ decodes₃ decodes₄
    decodes₅ decodesL worked on₁ len₁ packs positions long live₂ ready₂ fresh₁ fit₂
  refine ⟨later₁ ++ later, by rwa [← List.append_assoc], by rwa [← List.append_assoc],
    fun stage rooted shape => ⟨linked₁ stage rooted ⟨later, by rw [shape, List.append_assoc]⟩,
      each stage rooted (by rw [shape, List.append_assoc])⟩⟩

end Tower

end NightstreamFPrime.Spec.Compression.Akita.Extract
