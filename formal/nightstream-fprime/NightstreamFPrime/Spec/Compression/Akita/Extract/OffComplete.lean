import NightstreamFPrime.Spec.Compression.Akita.Extract.OffLink
import NightstreamFPrime.Spec.Compression.Akita.Extract.OffTower
import NightstreamFPrime.Spec.Compression.Akita.Extract.ShortComplete

/-! The verifier `offLinks` of the opening of Akita with an offloaded root accepts the honest
prover (ePrint 2026/1983, revision of 2026-10-15, Theorem 10.9 with Lemma 10.7, Section 9 with
(136), Section 5.4 with (69) to (72), Lemma 7.1, item 1, Section 4.5, Figure 4 and Section 8.2).
The theorem `offTower_suffix` bounds an event of this verifier.

- `segments_of_length` splits a point into the four segments `(r_pack, r_tail, r_pos, r_blk)`.
- `offState` has the commitment states of the level 1: the reference decommitment of the
  commitment `registry` of the setup prefix, and the state of `commitSource` for the bounded
  source of the table of the root fold.
- `stateTower_complete` gives, for a transcript prefix on the path of the table of the oracle whose
  last message has the commitments, the values and the points of the two claims of Akita (136),
  the later messages of a proof such that the complete transcript has the batching stage of these
  claims, the link of each level from the level 1 and the checks of the terminal.
- `offTower_complete` gives, for a transcript prefix whose last message has the commitments, the
  points and the values of the claims of the root groups, the later messages of a proof such that
  the complete transcript satisfies `offLinks`.

The fit conditions `BatchFit`, `StateFit`, `LiftFit` and `TraceFit` are hypotheses (Akita
Definition 10.5). The file has no bound for the probability `ξ_h` of Akita (155). -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Extract

open scoped BigOperators
open NightstreamFPrime.Spec.Compression.Lib (mle)
open NightstreamFPrime.Spec.Checkpoint

local instance : Fact (Nat.Prime fieldModulus) := ⟨FieldPrime.fieldModulus_natPrime⟩

attribute [local instance low] Classical.propDecidable

/-- A point of `p + (t + (s + k))` coordinates is the concatenation of four segments of `p`, `t`,
`s` and `k` coordinates. -/
theorem segments_of_length {E : Type} {r : List E} {p t s k : ℕ}
    (long : r.length = p + (t + (s + k))) :
    ∃ block position pack tail : List E, pack.length = p ∧ tail.length = t ∧
      position.length = s ∧ block.length = k ∧ r = pack ++ (tail ++ (position ++ block)) :=
  ⟨((r.drop p).drop t).drop s, ((r.drop p).drop t).take s, r.take p, (r.drop p).take t,
    by simp only [List.length_take]; omega,
    by simp only [List.length_take, List.length_drop]; omega,
    by simp only [List.length_take, List.length_drop]; omega,
    by simp only [List.length_drop]; omega, by simp⟩

/-- The commitment states of the level 1 of an opening with an offloaded root (Akita (136)): the
group 0 has the reference decommitment `reference` of the commitment `registry` of the setup
prefix, and the group 1 has the state of `commitSource` for the bounded source of the table
`witness`. -/
noncomputable def offState {B : Batch (Fin 2)} (key : Key (B.shape 1))
    (registry : (B.shape 0).Payload) (reference : Decommitment (B.shape 0)) (witness : Table) :
    (g : Fin 2) → (B.shape g).Payload × Decommitment (B.shape g) :=
  Batch.pairOf (F := fun g => (B.shape g).Payload × Decommitment (B.shape g)) (registry, reference)
    (commitSource key fun i => (B.shape 1).embed ((B.shape 1).pack witness i))

variable {E : Type} [Field E] [Fintype E] [DecidableEq E] [Algebra Zq E] [Degree2 E]
  {B₁ : Batch (Fin 2)} {S₂ : Shape} {O₂ : OpeningShape S₂} {fold₂ : Descriptor} {S₃ : Shape}
  {O₃ : OpeningShape S₃} {fold₃ : Descriptor} {S₄ : Shape} {O₄ : OpeningShape S₄}
  {fold₄ : Descriptor} {S₅ : Shape} {O₅ : OpeningShape S₅} {fold₅ : Descriptor} {S₆ : Shape}
  {O₆ : OpeningShape S₆} {fold₆ : Descriptor} {S₇ : Shape}
  {Wire Coord : Type} [Fintype Coord] [DecidableEq Coord] {total : ℕ}
  {accept : Prefix Wire (Option Coord → E) total → Prop} {read : Reader Wire}
  {β₁ : ∀ g, Module.Basis (Fin (B₁.planes g)) Zq E} {key₁ : (g : Fin 2) → Key (B₁.shape g)}
  {openKey₁ : B₁.OpenKey} {claim₁ : Wire → B₁.Claim E} {μ₁ : ℕ} {fits₁ : B₁.cells ≤ 2 ^ μ₁}
  {base₁ digits₁ : ℕ} {room₁ : B₁.quotientRoom digits₁ ≤ 2 ^ μ₁} {batch₁ : ℕ}
  {rowsFit₁ : B₁.rowCount + 1 ≤ 2 ^ batch₁} {block₁ : B₁.Block → Coord}
  {decode₁ : (g : Fin 2) → E → Option (Rq (B₁.carrierDegree g))} {mass₁ : Fin 2 → ℕ}
  {registry : (B₁.shape 0).Payload} {reference : Decommitment (B₁.shape 0)} {flat : Table}
  {fix : B₁.Weak → B₁.Claim E → B₁.LevelBreak key₁ openKey₁
    fun g => 2 * (2 * mass₁ g) * (B₁.fold g).sourceBound (B₁.shape g)}
  {β₂ : Module.Basis (Fin O₂.planes) Zq E} {key₂ : Key S₂} {openKey₂ : OpeningKey O₂}
  {claim₂ : Wire → Claim O₂ E} {μ₂ : ℕ} {fits₂ : O₂.cells fold₂ ≤ 2 ^ μ₂} {digits₂ : ℕ}
  {room₂ : O₂.quotientRoom fold₂ digits₂ ≤ 2 ^ μ₂} {batch₂ : ℕ}
  {rowsFit₂ : O₂.rowCount + 1 ≤ 2 ^ batch₂} {block₂ : Fin S₂.blocks → Coord}
  {decode₂ : E → Option (Rq O₂.carrierDegree)} {weight₂ : ℕ}
  {β₃ : Module.Basis (Fin O₃.planes) Zq E} {key₃ : Key S₃} {openKey₃ : OpeningKey O₃}
  {claim₃ : Wire → Claim O₃ E} {μ₃ : ℕ} {fits₃ : O₃.cells fold₃ ≤ 2 ^ μ₃} {digits₃ : ℕ}
  {room₃ : O₃.quotientRoom fold₃ digits₃ ≤ 2 ^ μ₃} {batch₃ : ℕ}
  {rowsFit₃ : O₃.rowCount + 1 ≤ 2 ^ batch₃} {block₃ : Fin S₃.blocks → Coord}
  {decode₃ : E → Option (Rq O₃.carrierDegree)} {weight₃ ν₃ : ℕ}
  {key₄ : Key S₄} {openKey₄ : OpeningKey O₄} {claim₄ : Wire → TraceClaim O₄ E}
  {μ₄ : ℕ} {fits₄ : O₄.cells fold₄ ≤ 2 ^ μ₄} {batch₄ : ℕ}
  {rowsFit₄ : O₄.traceRowCount + 1 ≤ 2 ^ batch₄} {block₄ : Fin S₄.blocks → Coord}
  {decode₄ : E → Option (Rq S₄.innerDegree)} {weight₄ ν₄ m₄ : ℕ}
  {key₅ : Key S₅} {openKey₅ : OpeningKey O₅} {claim₅ : Wire → TraceClaim O₅ E}
  {μ₅ : ℕ} {fits₅ : O₅.cells fold₅ ≤ 2 ^ μ₅} {batch₅ : ℕ}
  {rowsFit₅ : O₅.traceRowCount + 1 ≤ 2 ^ batch₅} {block₅ : Fin S₅.blocks → Coord}
  {decode₅ : E → Option (Rq S₅.innerDegree)} {weight₅ ν₅ m₅ : ℕ}
  {key₆ : Key S₆} {openKey₆ : OpeningKey O₆} {claim₆ : Wire → TraceClaim O₆ E}
  {μ₆ : ℕ} {fits₆ : O₆.cells fold₆ ≤ 2 ^ μ₆} {batch₆ : ℕ}
  {rowsFit₆ : O₆.traceRowCount + 1 ≤ 2 ^ batch₆} {block₆ : Fin S₆.blocks → Coord}
  {decode₆ : E → Option (Rq S₆.innerDegree)} {weight₆ ν₆ m₆ : ℕ}
  {operator₅ limit₅ operator₆ limit₆ operatorL limitL : ℕ}
  {L : Terminal S₇ E Wire Coord E total (.euclidean operatorL limitL)} {m₇ : ℕ}
  {commits : Wire → (g : Fin 2) → (B₁.shape g).Payload} {values : Wire → Fin 2 → E}
  {places : Wire → Fin 2 → List E} {commit₁ : Wire → (B₁.shape 1).Payload}
  {commit₂ : Wire → S₂.Payload} {commit₃ : Wire → S₃.Payload} {commit₄ : Wire → S₄.Payload}
  {commit₅ : Wire → S₅.Payload} {commit₆ : Wire → S₆.Payload}
  {stateOf : Wire → Fin S₇.blocks → Fin S₇.innerRank → Rq S₇.innerDegree}
  {statementOf : Wire → TerminalStatement S₇ E}
  {partialsOf : Wire → Fin S₇.blocks → Rq S₇.innerDegree}

section State

variable {start₁ : ℕ}

local notation "first" =>
  Tier.pinned (liftBatch β₁ key₁ openKey₁ claim₁ μ₁ fits₁ base₁ digits₁ room₁ batch₁ rowsFit₁
    block₁ decode₁ read mass₁ (start₁ + 1)) (Batch.Pinned 0 registry flat) fix

local notation "second" =>
  liftFold β₂ key₂ openKey₂ claim₂ μ₂ fits₂ digits₂ room₂ batch₂ rowsFit₂ block₂ decode₂ read
    weight₂ (Tier.finish first)

local notation "third" =>
  Tier.tensored (liftFold β₃ key₃ openKey₃ claim₃ μ₃ fits₃ digits₃ room₃ batch₃ rowsFit₃ block₃
    decode₃ read weight₃ (Tier.finish second)) ν₃

local notation "fourth" =>
  Tier.tensored (committedTrace (fold := fold₄) key₄ openKey₄ claim₄ μ₄ fits₄ batch₄ rowsFit₄
    block₄ decode₄ read weight₄ (Tier.finish third)) ν₄

local notation "fifth" =>
  Tier.tensored (committedTraceSq (fold := fold₅) key₅ openKey₅ claim₅ μ₅ fits₅ batch₅ rowsFit₅
    block₅ decode₅ read weight₅ operator₅ limit₅ (Tier.finish fourth)) ν₅

local notation "sixth" =>
  Tier.tensored (committedTraceSq (fold := fold₆) key₆ openKey₆ claim₆ μ₆ fits₆ batch₆ rowsFit₆
    block₆ decode₆ read weight₆ operator₆ limit₆ (Tier.finish fifth)) ν₆

/-- Akita Theorem 10.9 for the opening from the batching stage of the two claims of Akita (136),
without the probability that a response does not fit and without the term of (155) for a zero
tensor factor. `early` is a transcript prefix of `start₁` moves on the path of the table of the
oracle. Its last message has the commitments `(registry, u_pub)` of the two groups of the level 1,
where `u_pub` is the payload of `commitSource` for the bounded source of the table `witness`, the
values `(S̃^♭(ρ_S), w̃(r_2))` and the points `(ρ_S, r_2)` (`committed`, `valued`, `placed`). Then
later messages exist such that the complete transcript has the link `RootLinked` of the batching
stage at the stage `start₁` for its weights, the link `Tier.Linked` from the level 1 to the level
2 and from the level 2 to the level 3, and the links and the checks of `packTower₂_complete`.
`carriesR`: a wire message can have the claim of the level 1. `opening₁`, `bounded₁`, `radix₁`,
`certified₁`, `path₁`, `uniform₁`, `digit₁`, `covers₁`, `range₁`: the conditions of
`stateLevel_wins` on the level 1. `bounded₁₁`: the group 1 has the conditions `Shape.Bounded` of the
bounded source. `registered`: the premise of Akita Lemma 9.1 for the commitment `registry` of the
setup prefix `flat`. `stride₁₀` to `blocks₁₁`, `setup` and `pointed₁`: the points `ρ_S` and `r_2`
have the segments of the shapes of the groups 0 and 1. `live`: the table `witness` is 0 after the
cells of the live blocks of the group 1. `shape₂` to `range₃`: the conditions of `liftLevel_wins`
on the levels 2 and 3. `stride₂` to `covered₂`: the final point of the checks of the levels 1 and
2 has the segments of the shape of the next level, and the live blocks of the next level have each
cell before the end of the quotient digits. `decodes₁` to `decodes₃`: each slot of the table of the
oracle decodes to a fold challenge of the level. `fit`: the fit conditions `StateFit` of the level
1, `LiftFit` of the levels 2 and 3, `TraceFit` of the levels 4 to 6, and `‖z‖_2² ≤ S_max,term` at
the terminal. The other hypotheses are those of `packTower₂_complete`. -/
theorem stateTower_complete {sendable : (M : Type) → M → Prop} {slots width cap : ℕ}
    (sends : Sends sendable E width cap) (four : 4 ≤ width) (room : 3 < slots)
    (carriesR : Carries sendable slots read claim₁)
    (carries₁ : Carries sendable slots read fun wire => (commit₂ wire, claim₂ wire))
    (carries₂ : Carries sendable slots read fun wire => (commit₃ wire, claim₃ wire))
    (carries₃ : Carries sendable slots read fun wire => (commit₄ wire, claim₄ wire))
    (carries₄ : Carries sendable slots read fun wire => (commit₅ wire, claim₅ wire))
    (carries₅ : Carries sendable slots read fun wire => (commit₆ wire, claim₆ wire))
    (carries₆ : Carries sendable slots read fun wire =>
      (stateOf wire, statementOf wire, partialsOf wire))
    (carriesL : Carries sendable slots read L.response)
    {p₀ t₀ s₀ k₀ p₁ t₁ s₁ k₁ p₂ t₂ s₂ k₂ p₃ t₃ s₃ k₃ p₄ s₄ k₄ p₅ s₅ k₅ p₆ s₆ k₆ p₇ s₇ k₇ : ℕ}
    (opening₁ : ∀ g, Base (B₁.shape g).openingBase (B₁.shape g).openingDigits)
    (bounded₁ : ∀ g, (B₁.shape g).openingBase ≤ (B₁.shape g).certifiedBase)
    (radix₁ : ∀ g, Radix (B₁.fold g).responseBase)
    (certified₁ : ∀ g, (B₁.fold g).responseBase ≤ (B₁.shape g).certifiedBase)
    (path₁ : PathFits B₁.openDegree (1 * B₁.openRank) B₁.chain)
    (uniform₁ : ∀ g, (B₁.shape g).certifiedBase = base₁) (digit₁ : Radix base₁)
    (covers₁ : fieldModulus ≤ base₁ ^ digits₁) (range₁ : rangeWidth base₁ ≤ width)
    (bounded₁₁ : (B₁.shape 1).Bounded)
    (registered : Batch.Registered (key₁ 0) registry flat reference)
    (stride₁₀ : (B₁.opening 0).stride = 2 ^ p₀) (degree₁₀ : B₁.carrierDegree 0 = 2 ^ t₀)
    (length₁₀ : (B₁.shape 0).blockLength = 2 ^ s₀) (blocks₁₀ : (B₁.shape 0).blocks ≤ 2 ^ k₀)
    (stride₁₁ : (B₁.opening 1).stride = 2 ^ p₁) (degree₁₁ : B₁.carrierDegree 1 = 2 ^ t₁)
    (length₁₁ : (B₁.shape 1).blockLength = 2 ^ s₁) (blocks₁₁ : (B₁.shape 1).blocks ≤ 2 ^ k₁)
    (shape₂ : S₂.Bounded) (radix₂ : Radix fold₂.responseBase)
    (certified₂ : fold₂.responseBase ≤ S₂.certifiedBase)
    (path₂ : PathFits O₂.openDegree (1 * O₂.openRank) O₂.chain)
    (digit₂ : Radix S₂.certifiedBase) (covers₂ : fieldModulus ≤ S₂.certifiedBase ^ digits₂)
    (range₂ : rangeWidth S₂.certifiedBase ≤ width)
    (shape₃ : S₃.Bounded) (radix₃ : Radix fold₃.responseBase)
    (certified₃ : fold₃.responseBase ≤ S₃.certifiedBase)
    (path₃ : PathFits O₃.openDegree (1 * O₃.openRank) O₃.chain)
    (digit₃ : Radix S₃.certifiedBase) (covers₃ : fieldModulus ≤ S₃.certifiedBase ^ digits₃)
    (range₃ : rangeWidth S₃.certifiedBase ≤ width)
    (stride₂ : O₂.stride = 2 ^ p₂) (degree₂ : O₂.carrierDegree = 2 ^ t₂)
    (length₂ : S₂.blockLength = 2 ^ s₂) (blocks₂ : S₂.blocks ≤ 2 ^ k₂)
    (point₁ : μ₁ = p₂ + (t₂ + (s₂ + k₂)))
    (covered₁ : B₁.quotientRoom digits₁ ≤ S₂.blocks * (S₂.blockLength * S₂.innerDegree))
    (stride₃ : O₃.stride = 2 ^ p₃) (degree₃ : O₃.carrierDegree = 2 ^ t₃)
    (length₃ : S₃.blockLength = 2 ^ s₃) (blocks₃ : S₃.blocks ≤ 2 ^ k₃)
    (point₂ : μ₂ = p₃ + (t₃ + (s₃ + k₃)))
    (covered₂ : O₂.quotientRoom fold₂ digits₂ ≤ S₃.blocks * (S₃.blockLength * S₃.innerDegree))
    (quarter₄ : 4 * m₄ = S₄.innerDegree) (quarter₅ : 4 * m₅ = S₅.innerDegree)
    (quarter₆ : 4 * m₆ = S₆.innerDegree) (quarter₇ : 4 * m₇ = S₇.innerDegree)
    (source₄ : SourceSized S₄ m₄ p₄ s₄ k₄) (source₅ : SourceSized S₅ m₅ p₅ s₅ k₅)
    (source₆ : SourceSized S₆ m₆ p₆ s₆ k₆) (source₇ : SourceSized S₇ m₇ p₇ s₇ k₇)
    (sized₄ : FoldSized S₄ O₄ fold₄ width) (sized₅ : FoldSized S₅ O₅ fold₅ width)
    (sized₆ : FoldSized S₆ O₆ fold₆ width)
    (vars₃ : μ₃ = ν₃ + 1) (vars₄ : μ₄ = ν₄ + 1) (vars₅ : μ₅ = ν₅ + 1) (vars₆ : μ₆ = ν₆ + 1)
    (point₃ : ν₃ = p₄ + (s₄ + k₄)) (point₄ : ν₄ = p₅ + (s₅ + k₅))
    (point₅ : ν₅ = p₆ + (s₆ + k₆)) (point₆ : ν₆ = p₇ + (s₇ + k₇))
    (covered₃ : O₃.quotientRoom fold₃ digits₃ ≤ 2 * (S₄.blocks * (S₄.blockLength * (2 * m₄))))
    (covered₄ : O₄.cells fold₄ ≤ 2 * (S₅.blocks * (S₅.blockLength * (2 * m₅))))
    (covered₅ : O₅.cells fold₅ ≤ 2 * (S₆.blocks * (S₆.blockLength * (2 * m₆))))
    (covered₆ : O₆.cells fold₆ ≤ 2 * (S₇.blocks * (S₇.blockLength * (2 * m₇))))
    (normRoom₅ : S₅.blockLength * S₅.sourceDigits * S₅.innerDegree ≤ 2 ^ μ₅)
    (normRoom₆ : S₆.blockLength * S₆.sourceDigits * S₆.innerDegree ≤ 2 ^ μ₆)
    (wide₅ : rangeWidthSq S₅.certifiedBase ≤ width) (wide₆ : rangeWidthSq S₆.certifiedBase ≤ width)
    (capped₅ : limit₅ ≤ cap) (capped₆ : limit₆ ≤ cap)
    (next : L.start = Tier.finish sixth) (inside : L.start < total)
    (reads : ∀ point, L.read point = statementOf (point.wire L.start))
    (clears : ∀ point, L.partials point = partialsOf (point.wire L.start))
    (counts : ∀ wire, L.nonce wire = read.nonce wire)
    (table : Prefix Wire (Option Coord → E) total → Option Coord → E)
    (decodes₁ : ∀ input b, (decode₁ b.1 (table input (some (block₁ b)))).isSome)
    (decodes₂ : ∀ input i, (decode₂ (table input (some (block₂ i)))).isSome)
    (decodes₃ : ∀ input i, (decode₃ (table input (some (block₃ i)))).isSome)
    (decodes₄ : ∀ input i, (decode₄ (table input (some (block₄ i)))).isSome)
    (decodes₅ : ∀ input i, (decode₅ (table input (some (block₅ i)))).isSome)
    (decodes₆ : ∀ input i, (decode₆ (table input (some (block₆ i)))).isSome)
    (decodesL : ∀ input i, (L.decode (table input (L.slot i))).isSome)
    (worked : ∀ input, table input none ∈ workZeros E 57)
    {wire₀ : Wire} {early : List ((Option Coord → E) × Wire)} (on : OnPath accept table wire₀ early)
    (starts : early.length = start₁) {witness : Table} {r2 ρ : List E}
    (committed : commits (lastWire wire₀ early) =
      Batch.pairOf (F := fun g => (B₁.shape g).Payload) registry
        (commitSource (key₁ 1) fun i => (B₁.shape 1).embed ((B₁.shape 1).pack witness i)).1)
    (valued : values (lastWire wire₀ early) = ![mle ρ (lifted E flat), mle r2 (lifted E witness)])
    (placed : places (lastWire wire₀ early) = ![ρ, r2])
    (setup : ρ.length = p₀ + (t₀ + (s₀ + k₀))) (pointed₁ : r2.length = p₁ + (t₁ + (s₁ + k₁)))
    (live : ∀ cell, (B₁.shape 1).blocks * ((B₁.shape 1).blockLength * (B₁.shape 1).innerDegree) ≤
      cell → witness cell = 0)
    (fit : StateFit β₁ key₁ openKey₁ μ₁ fits₁ base₁ digits₁ decode₁
      (offState (key₁ 1) registry reference witness) fun w₁ =>
      LiftFit β₂ key₂ openKey₂ fold₂ μ₂ fits₂ digits₂ decode₂ (fun i => S₂.embed (S₂.pack w₁ i))
        fun w₂ =>
      LiftFit β₃ key₃ openKey₃ fold₃ μ₃ fits₃ digits₃ decode₃ (fun i => S₃.embed (S₃.pack w₂ i))
        fun w₃ =>
      TraceFit quarter₄ key₄ openKey₄ fold₄ (fun _ => decode₄) (fun _ => True) (packed w₃)
        fun w₄ =>
      TraceFit quarter₅ key₅ openKey₅ fold₅ (fun _ => decode₅) (fun z => vecNormSq z ≤ limit₅)
        (packed w₄) fun w₅ =>
      TraceFit quarter₆ key₆ openKey₆ fold₆ (fun _ => decode₆) (fun z => vecNormSq z ≤ limit₆)
        (packed w₅) fun w₆ =>
      ∀ c : Fin S₇.blocks → Rq S₇.innerDegree, (∀ i, ∃ slot, L.decode slot = some (c i)) →
        vecNormSq (traceResponse c (traceSource quarter₇ (packed w₆ : ℕ → E))) ≤ limitL) :
    ∃ later, OnPath accept table wire₀ (early ++ later) ∧ (early ++ later).length = total ∧
      ∀ stage : Prefix Wire (Option Coord → E) total, stage.first = wire₀ →
        stage.rest = early ++ later →
        (start₁ < stage.rest.length ∧
          RootLinked first start₁ commits values places
            ((claimsRoot (Coord := Coord) (Slot := E) (total := total) start₁ read values
              places).weights stage) stage) ∧
        Tier.Linked first second commit₂ stage ∧ Tier.Linked second third commit₃ stage ∧
        Tier.TraceLinked third fourth quarter₄ commit₄ stage ∧
        Tier.TraceLinked fourth fifth quarter₅ commit₅ stage ∧
        Tier.TraceLinked fifth sixth quarter₆ commit₆ stage ∧
        Terminal.Linked sixth L quarter₇ stateOf stage ∧ L.Passed stage := by
  have ends₁ := Tier.finish_eq first
  have ends₂ := Tier.finish_eq second
  have ends₃ := Tier.finish_eq third
  have ends₄ := Tier.finish_eq fourth
  have ends₅ := Tier.finish_eq fifth
  have ends₆ := Tier.finish_eq sixth
  have fin₂ : Tier.finish second ≤ total :=
    (show Tier.finish second ≤ Tier.finish third from
      (Nat.le_add_right _ _).trans_eq ends₃.symm).trans
    ((show Tier.finish third ≤ Tier.finish fourth from
      (Nat.le_add_right _ _).trans_eq ends₄.symm).trans
    ((show Tier.finish fourth ≤ Tier.finish fifth from
      (Nat.le_add_right _ _).trans_eq ends₅.symm).trans
    ((show Tier.finish fifth ≤ Tier.finish sixth from
      (Nat.le_add_right _ _).trans_eq ends₆.symm).trans (next ▸ inside.le))))
  have fin₁ : Tier.finish first ≤ total :=
    (show Tier.finish first ≤ Tier.finish second from
      (Nat.le_add_right _ _).trans_eq ends₂.symm).trans fin₂
  have rooted : start₁ + 1 ≤ total :=
    (show start₁ + 1 ≤ Tier.finish first from (Nat.le_add_right _ _).trans_eq ends₁.symm).trans
      fin₁
  obtain ⟨block₀', position₀, pack₀, tail₀, packs₀, tails₀, positions₀, blocks₀', rfl⟩ :=
    segments_of_length setup
  obtain ⟨block₁', position₁, pack₁, tail₁, packs₁, tails₁, positions₁, blocks₁', rfl⟩ :=
    segments_of_length pointed₁
  obtain ⟨points, pointed⟩ : ∃ points : (g : Fin 2) → Point (B₁.opening g) E,
      points = Batch.pairOf (F := fun g => Point (B₁.opening g) E)
        (Point.ofSegments (B₁.opening 0) block₀' position₀ pack₀ tail₀)
        (Point.ofSegments (B₁.opening 1) block₁' position₁ pack₁ tail₁) := ⟨_, rfl⟩
  have fits₁₀ : (B₁.shape 0).blocks ≤ 2 ^ block₀'.length := by rw [blocks₀']; exact blocks₁₀
  have fits₁₁ : (B₁.shape 1).blocks ≤ 2 ^ block₁'.length := by rw [blocks₁']; exact blocks₁₁
  have lengths₀ : (B₁.shape 0).blockLength = 2 ^ position₀.length := by
    rw [positions₀]; exact length₁₀
  have lengths₁ : (B₁.shape 1).blockLength = 2 ^ position₁.length := by
    rw [positions₁]; exact length₁₁
  have strides₀ : (B₁.opening 0).stride = 2 ^ pack₀.length := by rw [packs₀]; exact stride₁₀
  have strides₁ : (B₁.opening 1).stride = 2 ^ pack₁.length := by rw [packs₁]; exact stride₁₁
  have degrees₀ : B₁.carrierDegree 0 = 2 ^ tail₀.length := by rw [tails₀]; exact degree₁₀
  have degrees₁ : B₁.carrierDegree 1 = 2 ^ tail₁.length := by rw [tails₁]; exact degree₁₁
  have value₀ : Akita.scalar (O := B₁.opening 0) (β₁ 0)
      (Point.ofSegments (B₁.opening 0) block₀' position₀ pack₀ tail₀) (fun i t =>
        contract (β₁ 0) (Point.ofSegments (B₁.opening 0) block₀' position₀ pack₀ tail₀) t
          (gadget (B₁.shape 0).sourceBase (reference.source i))) =
      mle (pack₀ ++ (tail₀ ++ (position₀ ++ block₀'))) (lifted E flat) := by
    rw [scalar_contract_eq_mle (β₁ 0) block₀' position₀ pack₀ tail₀ fits₁₀ lengths₀ strides₀
      degrees₀, registered.opens.decodes]
    rfl
  have value₁ := sourceClaim_value (β := β₁ 1) (key := key₁ 1) (openKey := openKey₁.group 1)
    bounded₁₁ witness block₁' position₁ pack₁ tail₁ fits₁₁ lengths₁ strides₁ degrees₁ live
  obtain ⟨ϑ, wireR, claimed, triedR, onR, linkedR⟩ := rootLink (T := first) (commit := commits)
    (value := values) (place := places) carriesR rfl table worked on starts rooted
    (fun ϑ => Batch.stateClaim β₁ openKey₁
      (offState (key₁ 1) registry reference witness) points ϑ)
    (fun _ => by
      rw [committed]
      funext g
      fin_cases g <;> rfl)
    (fun _ => rfl)
    (fun ϑ => by
      dsimp only [Batch.stateClaim]
      rw [Fin.sum_univ_two, Fin.sum_univ_two, valued, pointed]
      exact congrArg₂ (· + ·) (congrArg (ϑ 0 * ·) value₀) (congrArg (ϑ 1 * ·) value₁))
    (fun _ g => by
      rw [pointed, placed]
      fin_cases g
      · exact ⟨block₀', position₀, pack₀, tail₀, fits₁₀, lengths₀, strides₀, degrees₀, rfl, rfl⟩
      · exact ⟨block₁', position₁, pack₁, tail₁, fits₁₁, lengths₁, strides₁, degrees₁, rfl, rfl⟩)
  have lenR : (early ++ [(answer table wire₀ early, wireR)]).length = start₁ + 1 := by
    rw [List.length_append, List.length_singleton, starts]
  have member₁ : ∀ b, ∃ slot, decode₁ b.1 slot =
      some (Tier.drawn first table wire₀ (early ++ [(answer table wire₀ early, wireR)]) b) :=
    fun b => ⟨_, Tier.drawn_mem first table (lenR.le.trans rooted) decodes₁ b⟩
  have opens : ∀ g, ∃ table, (offState (key₁ 1) registry reference witness g).2.Opens (key₁ g)
      (offState (key₁ 1) registry reference witness g).1 table := by
    intro g
    fin_cases g
    · exact ⟨flat, registered.opens⟩
    · exact ⟨_, commitSource_opens bounded₁₁ (key₁ 1) _⟩
  obtain ⟨fit₂, zero₁, wins₁⟩ := stateLevel_wins (claim := claim₁) (room := room₁)
    (batch := batch₁) (rowsFit := rowsFit₁) (block := block₁) (read := read) (mass := mass₁)
    (start := start₁ + 1) sends range₁ four (Nat.zero_lt_of_lt room) opening₁ bounded₁ radix₁
    certified₁ path₁ uniform₁ digit₁ covers₁ opens
    (wire := lastWire wire₀ (early ++ [(answer table wire₀ early, wireR)]))
    (by rw [lastWire_concat]; exact claimed) fit member₁
  obtain ⟨later₁, place₂, on₁, len₁, fresh₁, ready₂, linked₁⟩ :=
    packLink (T := first) (Next := second) (commit' := commit₂) carries₁ onR lenR fin₁ decodes₁
      worked ⟨foldWorkBits_le _ _, map_workBits_le _⟩ (by rw [lastWire_concat]; exact triedR)
      (fun _ _ table other => foldCheck_bare _ _ _ _ table other) (foldCosts_length E _ _ _ _ _) _
      (wins₁.mono fun _ _ _ kept => kept.2) rfl shape₂ β₂ stride₂ degree₂ length₂ blocks₂ point₁
      fun cell large => zero₁ cell (covered₁.trans large)
  have member₂ : ∀ i, ∃ slot, decode₂ slot = some (Tier.drawn second table wire₀
      (early ++ [(answer table wire₀ early, wireR)] ++ later₁) i) :=
    fun i => ⟨_, Tier.drawn_mem second table (len₁.le.trans fin₁) decodes₂ i⟩
  obtain ⟨fit₃, zero₂, wins₂⟩ := liftLevel_wins (claim := claim₂) (room := room₂)
    (batch := batch₂) (rowsFit := rowsFit₂) (block := block₂) (read := read) (weight := weight₂)
    (start := Tier.finish first) sends range₂ four (Nat.zero_lt_of_lt room) shape₂ radix₂
    certified₂ path₂ digit₂ covers₂ ready₂ fit₂ member₂
  obtain ⟨later₂, place₃, on₂, len₂, fresh₂, ready₃, linked₂⟩ :=
    packLink (T := second) (Next := third) (commit' := commit₃) carries₂ on₁ len₁ fin₂ decodes₂
      worked ⟨foldWorkBits_le _ _, map_workBits_le _⟩ fresh₁
      (fun _ _ table other => foldCheck_bare _ _ _ _ table other) (foldCosts_length E _ _ _ _ _) _
      (wins₂.mono fun _ _ _ kept => kept.2) rfl shape₃ β₃ stride₃ degree₃ length₃ blocks₃ point₂
      fun cell large => zero₂ cell (covered₂.trans large)
  obtain ⟨later, onL, whole, each⟩ := packTower₂_complete (start := Tier.finish second) sends four
    room carries₃ carries₄ carries₅ carries₆ carriesL shape₃ radix₃ certified₃ path₃ digit₃
    covers₃ range₃ quarter₄ quarter₅ quarter₆ quarter₇ source₄ source₅ source₆ source₇ sized₄
    sized₅ sized₆ vars₃ vars₄ vars₅ vars₆ point₃ point₄ point₅ point₆ covered₃ covered₄ covered₅
    covered₆ normRoom₅ normRoom₆ wide₅ wide₆ capped₅ capped₆ next inside reads clears counts table
    decodes₃ decodes₄ decodes₅ decodes₆ decodesL worked on₂ len₂ ready₃ fresh₂ fit₃
  have joined : early ++ (answer table wire₀ early, wireR) :: (later₁ ++ (later₂ ++ later)) =
      early ++ [(answer table wire₀ early, wireR)] ++ later₁ ++ later₂ ++ later := by
    simp only [List.append_assoc, List.cons_append, List.nil_append]
  refine ⟨(answer table wire₀ early, wireR) :: (later₁ ++ (later₂ ++ later)), joined ▸ onL,
    joined ▸ whole, fun stage rooted' shape =>
      ⟨linkedR stage rooted' ⟨_, shape⟩, linked₁ stage rooted' ⟨later₂ ++ later, ?_⟩,
        linked₂ stage rooted' ⟨later, ?_⟩, each stage rooted' (shape.trans joined)⟩⟩
  · rw [shape, joined]
    simp only [List.append_assoc]
  · rw [shape, joined]

end State

section Off

variable {count : ℕ} {B : Batch (Fin count)} {start : ℕ}
  {β : ∀ g, Module.Basis (Fin (B.planes g)) Zq E} {key : (g : Fin count) → Key (B.shape g)}
  {openKey : B.OpenKey} {claim : Wire → B.Claim E} {μ : ℕ} {fits : B.cells ≤ 2 ^ μ}
  {base digits : ℕ} {room₀ : B.quotientRoom digits ≤ 2 ^ μ} {batch : ℕ}
  {rowsFit : B.rowCount + 1 ≤ 2 ^ batch} {block : B.Block → Coord}
  {decode : (g : Fin count) → E → Option (Rq (B.carrierDegree g))} {mass : Fin count → ℕ} {ν : ℕ}
  {scan : ∀ (wire : Wire) (challenge : B.Challenge),
    SetupScan (Batch.liftedReduction β ((claim wire).statement key openKey) challenge μ fits base
      digits room₀ batch rowsFit) ν flat}
  {commit : Wire → (g : Fin count) → (B.shape g).Payload} {value : Wire → Fin count → E}
  {place : Wire → Fin count → List E}

local notation "zeroth" =>
  offBatch β key openKey claim μ fits base digits room₀ batch rowsFit block decode read mass
    (start + 1) ν flat scan

local notation "first" =>
  Tier.pinned (liftBatch β₁ key₁ openKey₁ claim₁ μ₁ fits₁ base₁ digits₁ room₁ batch₁ rowsFit₁
    block₁ decode₁ read mass₁ (Tier.finish zeroth + 1)) (Batch.Pinned 0 registry flat) fix

local notation "second" =>
  liftFold β₂ key₂ openKey₂ claim₂ μ₂ fits₂ digits₂ room₂ batch₂ rowsFit₂ block₂ decode₂ read
    weight₂ (Tier.finish first)

local notation "third" =>
  Tier.tensored (liftFold β₃ key₃ openKey₃ claim₃ μ₃ fits₃ digits₃ room₃ batch₃ rowsFit₃ block₃
    decode₃ read weight₃ (Tier.finish second)) ν₃

local notation "fourth" =>
  Tier.tensored (committedTrace (fold := fold₄) key₄ openKey₄ claim₄ μ₄ fits₄ batch₄ rowsFit₄
    block₄ decode₄ read weight₄ (Tier.finish third)) ν₄

local notation "fifth" =>
  Tier.tensored (committedTraceSq (fold := fold₅) key₅ openKey₅ claim₅ μ₅ fits₅ batch₅ rowsFit₅
    block₅ decode₅ read weight₅ operator₅ limit₅ (Tier.finish fourth)) ν₅

local notation "sixth" =>
  Tier.tensored (committedTraceSq (fold := fold₆) key₆ openKey₆ claim₆ μ₆ fits₆ batch₆ rowsFit₆
    block₆ decode₆ read weight₆ operator₆ limit₆ (Tier.finish fifth)) ν₆

/-- Akita Theorem 10.9 for the opening with a batching root, an offloaded root fold with opening
groups, the level 1 with the group of the setup prefix and the group of the witness, two packing
levels, three trace levels and the terminal, without the probability that a response does not fit
and without the term of (155) for a zero tensor factor: the verifier `offLinks` of
`offTower_suffix` accepts the honest prover. `early` is a transcript prefix of `start` moves on the
path of the table of the oracle. Its last message has the commitment `commit` of the table
`tables g` of each root group, a point `place g` with the segments of the shape of the group, and
the value of the table at this point (`committed`, `placed`, `valued`). Then later messages exist
such that the complete transcript satisfies `offLinks`: the link `RootLinked` of the batching
root, the link `Tier.Offloaded` from the root fold to the two claims of the level 1, and the links
and the checks of `stateTower_complete`. `carriesR`, `carriesO` and `carries₁R`: a wire message can
have the claim of the root fold, the commitment of the group 1 of the level 1 with the
commitments, the values and the points of the two claims, and the claim of the level 1.
`shape₀`, `radix₀`, `certified₀`, `path₀`, `uniform₀`, `digit₀`, `covers₀`, `range₀`: the
conditions of `offLevel_wins` on the root fold. `blocks₀`, `lengths₀`, `strides₀`, `degrees₀`,
`live₀`: the segments of the point of each root group have the lengths of its shape, and its table
is 0 after the cells of its blocks. `setup` and `point₀`: the points `ρ_S` of `ν` coordinates and
`r_2` of `μ` coordinates have the segments of the shapes of the groups 0 and 1 of the level 1.
`covered₀`: the live blocks of the group 1 have each cell before the end of the quotient digits of
the root fold. `decodes₀`: each slot of the table of the oracle decodes to a fold challenge of the
root fold. `fit`: the fit conditions `BatchFit` of the root fold over the conditions of
`stateTower_complete` for its table. The other hypotheses are those of `stateTower_complete`. -/
theorem offTower_complete {sendable : (M : Type) → M → Prop} {slots width cap : ℕ}
    (sends : Sends sendable E width cap) (four : 4 ≤ width) (room : 3 < slots)
    (carriesR : Carries sendable slots read claim)
    (carriesO : Carries sendable slots read fun wire =>
      (commit₁ wire, commits wire, values wire, places wire))
    (carries₁R : Carries sendable slots read claim₁)
    (carries₁ : Carries sendable slots read fun wire => (commit₂ wire, claim₂ wire))
    (carries₂ : Carries sendable slots read fun wire => (commit₃ wire, claim₃ wire))
    (carries₃ : Carries sendable slots read fun wire => (commit₄ wire, claim₄ wire))
    (carries₄ : Carries sendable slots read fun wire => (commit₅ wire, claim₅ wire))
    (carries₅ : Carries sendable slots read fun wire => (commit₆ wire, claim₆ wire))
    (carries₆ : Carries sendable slots read fun wire =>
      (stateOf wire, statementOf wire, partialsOf wire))
    (carriesL : Carries sendable slots read L.response)
    {p₀ t₀ s₀ k₀ p₁ t₁ s₁ k₁ p₂ t₂ s₂ k₂ p₃ t₃ s₃ k₃ p₄ s₄ k₄ p₅ s₅ k₅ p₆ s₆ k₆ p₇ s₇ k₇ : ℕ}
    (shape₀ : ∀ g, (B.shape g).Honest) (radix₀ : ∀ g, Radix (B.fold g).responseBase)
    (certified₀ : ∀ g, (B.fold g).responseBase ≤ (B.shape g).certifiedBase)
    (path₀ : PathFits B.openDegree (1 * B.openRank) B.chain)
    (uniform₀ : ∀ g, (B.shape g).certifiedBase = base) (digit₀ : Radix base)
    (covers₀ : fieldModulus ≤ base ^ digits) (range₀ : rangeWidth base ≤ width)
    (opening₁ : ∀ g, Base (B₁.shape g).openingBase (B₁.shape g).openingDigits)
    (bounded₁ : ∀ g, (B₁.shape g).openingBase ≤ (B₁.shape g).certifiedBase)
    (radix₁ : ∀ g, Radix (B₁.fold g).responseBase)
    (certified₁ : ∀ g, (B₁.fold g).responseBase ≤ (B₁.shape g).certifiedBase)
    (path₁ : PathFits B₁.openDegree (1 * B₁.openRank) B₁.chain)
    (uniform₁ : ∀ g, (B₁.shape g).certifiedBase = base₁) (digit₁ : Radix base₁)
    (covers₁ : fieldModulus ≤ base₁ ^ digits₁) (range₁ : rangeWidth base₁ ≤ width)
    (bounded₁₁ : (B₁.shape 1).Bounded)
    (registered : Batch.Registered (key₁ 0) registry flat reference)
    (stride₁₀ : (B₁.opening 0).stride = 2 ^ p₀) (degree₁₀ : B₁.carrierDegree 0 = 2 ^ t₀)
    (length₁₀ : (B₁.shape 0).blockLength = 2 ^ s₀) (blocks₁₀ : (B₁.shape 0).blocks ≤ 2 ^ k₀)
    (stride₁₁ : (B₁.opening 1).stride = 2 ^ p₁) (degree₁₁ : B₁.carrierDegree 1 = 2 ^ t₁)
    (length₁₁ : (B₁.shape 1).blockLength = 2 ^ s₁) (blocks₁₁ : (B₁.shape 1).blocks ≤ 2 ^ k₁)
    (setup : ν = p₀ + (t₀ + (s₀ + k₀))) (point₀ : μ = p₁ + (t₁ + (s₁ + k₁)))
    (covered₀ : B.quotientRoom digits ≤
      (B₁.shape 1).blocks * ((B₁.shape 1).blockLength * (B₁.shape 1).innerDegree))
    (shape₂ : S₂.Bounded) (radix₂ : Radix fold₂.responseBase)
    (certified₂ : fold₂.responseBase ≤ S₂.certifiedBase)
    (path₂ : PathFits O₂.openDegree (1 * O₂.openRank) O₂.chain)
    (digit₂ : Radix S₂.certifiedBase) (covers₂ : fieldModulus ≤ S₂.certifiedBase ^ digits₂)
    (range₂ : rangeWidth S₂.certifiedBase ≤ width)
    (shape₃ : S₃.Bounded) (radix₃ : Radix fold₃.responseBase)
    (certified₃ : fold₃.responseBase ≤ S₃.certifiedBase)
    (path₃ : PathFits O₃.openDegree (1 * O₃.openRank) O₃.chain)
    (digit₃ : Radix S₃.certifiedBase) (covers₃ : fieldModulus ≤ S₃.certifiedBase ^ digits₃)
    (range₃ : rangeWidth S₃.certifiedBase ≤ width)
    (stride₂ : O₂.stride = 2 ^ p₂) (degree₂ : O₂.carrierDegree = 2 ^ t₂)
    (length₂ : S₂.blockLength = 2 ^ s₂) (blocks₂ : S₂.blocks ≤ 2 ^ k₂)
    (point₁ : μ₁ = p₂ + (t₂ + (s₂ + k₂)))
    (covered₁ : B₁.quotientRoom digits₁ ≤ S₂.blocks * (S₂.blockLength * S₂.innerDegree))
    (stride₃ : O₃.stride = 2 ^ p₃) (degree₃ : O₃.carrierDegree = 2 ^ t₃)
    (length₃ : S₃.blockLength = 2 ^ s₃) (blocks₃ : S₃.blocks ≤ 2 ^ k₃)
    (point₂ : μ₂ = p₃ + (t₃ + (s₃ + k₃)))
    (covered₂ : O₂.quotientRoom fold₂ digits₂ ≤ S₃.blocks * (S₃.blockLength * S₃.innerDegree))
    (quarter₄ : 4 * m₄ = S₄.innerDegree) (quarter₅ : 4 * m₅ = S₅.innerDegree)
    (quarter₆ : 4 * m₆ = S₆.innerDegree) (quarter₇ : 4 * m₇ = S₇.innerDegree)
    (source₄ : SourceSized S₄ m₄ p₄ s₄ k₄) (source₅ : SourceSized S₅ m₅ p₅ s₅ k₅)
    (source₆ : SourceSized S₆ m₆ p₆ s₆ k₆) (source₇ : SourceSized S₇ m₇ p₇ s₇ k₇)
    (sized₄ : FoldSized S₄ O₄ fold₄ width) (sized₅ : FoldSized S₅ O₅ fold₅ width)
    (sized₆ : FoldSized S₆ O₆ fold₆ width)
    (vars₃ : μ₃ = ν₃ + 1) (vars₄ : μ₄ = ν₄ + 1) (vars₅ : μ₅ = ν₅ + 1) (vars₆ : μ₆ = ν₆ + 1)
    (point₃ : ν₃ = p₄ + (s₄ + k₄)) (point₄ : ν₄ = p₅ + (s₅ + k₅))
    (point₅ : ν₅ = p₆ + (s₆ + k₆)) (point₆ : ν₆ = p₇ + (s₇ + k₇))
    (covered₃ : O₃.quotientRoom fold₃ digits₃ ≤ 2 * (S₄.blocks * (S₄.blockLength * (2 * m₄))))
    (covered₄ : O₄.cells fold₄ ≤ 2 * (S₅.blocks * (S₅.blockLength * (2 * m₅))))
    (covered₅ : O₅.cells fold₅ ≤ 2 * (S₆.blocks * (S₆.blockLength * (2 * m₆))))
    (covered₆ : O₆.cells fold₆ ≤ 2 * (S₇.blocks * (S₇.blockLength * (2 * m₇))))
    (normRoom₅ : S₅.blockLength * S₅.sourceDigits * S₅.innerDegree ≤ 2 ^ μ₅)
    (normRoom₆ : S₆.blockLength * S₆.sourceDigits * S₆.innerDegree ≤ 2 ^ μ₆)
    (wide₅ : rangeWidthSq S₅.certifiedBase ≤ width) (wide₆ : rangeWidthSq S₆.certifiedBase ≤ width)
    (capped₅ : limit₅ ≤ cap) (capped₆ : limit₆ ≤ cap)
    (next : L.start = Tier.finish sixth) (inside : L.start < total)
    (reads : ∀ point, L.read point = statementOf (point.wire L.start))
    (clears : ∀ point, L.partials point = partialsOf (point.wire L.start))
    (counts : ∀ wire, L.nonce wire = read.nonce wire)
    (table : Prefix Wire (Option Coord → E) total → Option Coord → E)
    (decodes₀ : ∀ input b, (decode b.1 (table input (some (block b)))).isSome)
    (decodes₁ : ∀ input b, (decode₁ b.1 (table input (some (block₁ b)))).isSome)
    (decodes₂ : ∀ input i, (decode₂ (table input (some (block₂ i)))).isSome)
    (decodes₃ : ∀ input i, (decode₃ (table input (some (block₃ i)))).isSome)
    (decodes₄ : ∀ input i, (decode₄ (table input (some (block₄ i)))).isSome)
    (decodes₅ : ∀ input i, (decode₅ (table input (some (block₅ i)))).isSome)
    (decodes₆ : ∀ input i, (decode₆ (table input (some (block₆ i)))).isSome)
    (decodesL : ∀ input i, (L.decode (table input (L.slot i))).isSome)
    (worked : ∀ input, table input none ∈ workZeros E 57)
    {wire₀ : Wire} {early : List ((Option Coord → E) × Wire)} (on : OnPath accept table wire₀ early)
    (starts : early.length = start) {tables : Fin count → Table}
    {blockOf positionOf packOf tailOf : Fin count → List E}
    (blocks₀ : ∀ g, (B.shape g).blocks ≤ 2 ^ (blockOf g).length)
    (lengths₀ : ∀ g, (B.shape g).blockLength = 2 ^ (positionOf g).length)
    (strides₀ : ∀ g, (B.opening g).stride = 2 ^ (packOf g).length)
    (degrees₀ : ∀ g, B.carrierDegree g = 2 ^ (tailOf g).length)
    (live₀ : ∀ g cell, (B.shape g).blockLength * (B.shape g).innerDegree * (B.shape g).blocks ≤
      cell → tables g cell = 0)
    (committed : commit (lastWire wire₀ early) = fun g => (Akita.commit (key g) (tables g)).1)
    (placed : place (lastWire wire₀ early) =
      fun g => packOf g ++ (tailOf g ++ (positionOf g ++ blockOf g)))
    (valued : value (lastWire wire₀ early) =
      fun g => mle (place (lastWire wire₀ early) g) (lifted E (tables g)))
    (fit : BatchFit β key openKey μ fits base digits decode tables fun w₀ =>
      StateFit β₁ key₁ openKey₁ μ₁ fits₁ base₁ digits₁ decode₁
        (offState (key₁ 1) registry reference w₀) fun w₁ =>
      LiftFit β₂ key₂ openKey₂ fold₂ μ₂ fits₂ digits₂ decode₂ (fun i => S₂.embed (S₂.pack w₁ i))
        fun w₂ =>
      LiftFit β₃ key₃ openKey₃ fold₃ μ₃ fits₃ digits₃ decode₃ (fun i => S₃.embed (S₃.pack w₂ i))
        fun w₃ =>
      TraceFit quarter₄ key₄ openKey₄ fold₄ (fun _ => decode₄) (fun _ => True) (packed w₃)
        fun w₄ =>
      TraceFit quarter₅ key₅ openKey₅ fold₅ (fun _ => decode₅) (fun z => vecNormSq z ≤ limit₅)
        (packed w₄) fun w₅ =>
      TraceFit quarter₆ key₆ openKey₆ fold₆ (fun _ => decode₆) (fun z => vecNormSq z ≤ limit₆)
        (packed w₅) fun w₆ =>
      ∀ c : Fin S₇.blocks → Rq S₇.innerDegree, (∀ i, ∃ slot, L.decode slot = some (c i)) →
        vecNormSq (traceResponse c (traceSource quarter₇ (packed w₆ : ℕ → E))) ≤ limitL) :
    ∃ later, OnPath accept table wire₀ (early ++ later) ∧ (early ++ later).length = total ∧
      ∀ stage : Prefix Wire (Option Coord → E) total, stage.first = wire₀ →
        stage.rest = early ++ later →
        offLinks (weight₅ := weight₅) (weight₆ := weight₆) (route₄ := Route.coefficient)
          (route₅ := Route.euclidean operator₅ limit₅)
          (route₆ := Route.euclidean operator₆ limit₆) zeroth first second third fourth fifth
          sixth L start read commit value place commits values places registry commit₁ commit₂
          commit₃ quarter₄ quarter₅ quarter₆ quarter₇ commit₄ commit₅ commit₆ stateOf stage := by
  have ends₀ := Tier.finish_eq zeroth
  have ends₁ := Tier.finish_eq first
  have ends₂ := Tier.finish_eq second
  have ends₃ := Tier.finish_eq third
  have ends₄ := Tier.finish_eq fourth
  have ends₅ := Tier.finish_eq fifth
  have ends₆ := Tier.finish_eq sixth
  have fin₀ : Tier.finish zeroth ≤ total :=
    (show Tier.finish zeroth ≤ Tier.finish first from
      (Nat.le_succ _).trans ((Nat.le_add_right _ _).trans_eq ends₁.symm)).trans
    ((show Tier.finish first ≤ Tier.finish second from
      (Nat.le_add_right _ _).trans_eq ends₂.symm).trans
    ((show Tier.finish second ≤ Tier.finish third from
      (Nat.le_add_right _ _).trans_eq ends₃.symm).trans
    ((show Tier.finish third ≤ Tier.finish fourth from
      (Nat.le_add_right _ _).trans_eq ends₄.symm).trans
    ((show Tier.finish fourth ≤ Tier.finish fifth from
      (Nat.le_add_right _ _).trans_eq ends₅.symm).trans
    ((show Tier.finish fifth ≤ Tier.finish sixth from
      (Nat.le_add_right _ _).trans_eq ends₆.symm).trans (next ▸ inside.le))))))
  have rooted₀ : start + 1 ≤ total :=
    (show start + 1 ≤ Tier.finish zeroth from (Nat.le_add_right _ _).trans_eq ends₀.symm).trans
      fin₀
  obtain ⟨ϑ, wireR, claimedR, triedR, onR, linkedR⟩ := rootLink (T := zeroth) (commit := commit)
    (value := value) (place := place) carriesR rfl table worked on starts rooted₀
    (fun ϑ => Batch.honestClaim β key openKey tables (fun g => Point.ofSegments (B.opening g)
      (blockOf g) (positionOf g) (packOf g) (tailOf g)) ϑ)
    (fun _ => by
      rw [committed]
      rfl)
    (fun _ => rfl)
    (fun ϑ => by
      dsimp only [Batch.honestClaim]
      rw [valued, placed]
      exact Finset.sum_congr rfl fun g _ => congrArg (ϑ g * ·)
        (honestClaim_value (O := B.opening g) (β := β g) (key := key g)
          (openKey := openKey.group g) (shape₀ g) (tables g) (blockOf g) (positionOf g)
          (packOf g) (tailOf g) (blocks₀ g) (lengths₀ g) (strides₀ g) (degrees₀ g) (live₀ g)))
    (fun _ g => ⟨blockOf g, positionOf g, packOf g, tailOf g, blocks₀ g, lengths₀ g, strides₀ g,
      degrees₀ g, rfl, by rw [placed]⟩)
  have lenR : (early ++ [(answer table wire₀ early, wireR)]).length = start + 1 := by
    rw [List.length_append, List.length_singleton, starts]
  have member₀ : ∀ b, ∃ slot, decode b.1 slot =
      some (Tier.drawn zeroth table wire₀ (early ++ [(answer table wire₀ early, wireR)]) b) :=
    fun b => ⟨_, Tier.drawn_mem zeroth table (lenR.le.trans rooted₀) decodes₀ b⟩
  obtain ⟨fit₁, zero₀, wins₀⟩ := offLevel_wins (claim := claim) (room := room₀) (batch := batch)
    (rowsFit := rowsFit) (block := block) (read := read) (mass := mass) (start := start + 1)
    (ν := ν) (flat := flat) (scan := scan) sends range₀ four (Nat.lt_of_succ_lt room) shape₀
    radix₀ certified₀
    path₀ uniform₀ digit₀ covers₀
    (wire := lastWire wire₀ (early ++ [(answer table wire₀ early, wireR)]))
    (by rw [lastWire_concat]; exact claimedR) fit member₀
  generalize batchNext β key openKey μ fits base digits tables (fun g => Point.ofSegments
    (B.opening g) (blockOf g) (positionOf g) (packOf g) (tailOf g)) ϑ (fun g i =>
      Tier.drawn zeroth table wire₀ (early ++ [(answer table wire₀ early, wireR)]) ⟨g, i⟩) =
    witness at fit₁ zero₀ wins₀
  obtain ⟨later₀, r2, ρ, long₂, longρ, on₀, len₀, -, committed₁, valued₁, placed₁, linked₀⟩ :=
    offLink (T := zeroth) (B := B₁) (commits := commits) (values := values) (places := places)
      (commit := commit₁) carriesO onR lenR fin₀ decodes₀ worked
      ⟨foldWorkBits_le _ _, map_workBits_le _⟩ (by rw [lastWire_concat]; exact triedR)
      (fun wire challenge table other => offBatch_uniform wire challenge table other)
      offBatch_costs_length witness flat wins₀ registry
      (commitSource (key₁ 1) fun i => (B₁.shape 1).embed ((B₁.shape 1).pack witness i)).1
  obtain ⟨later, onL, whole, each⟩ := stateTower_complete (start₁ := Tier.finish zeroth) sends
    four room carries₁R carries₁ carries₂ carries₃ carries₄ carries₅ carries₆ carriesL opening₁
    bounded₁ radix₁ certified₁ path₁ uniform₁ digit₁ covers₁ range₁ bounded₁₁ registered stride₁₀
    degree₁₀ length₁₀ blocks₁₀ stride₁₁ degree₁₁ length₁₁ blocks₁₁ shape₂ radix₂ certified₂ path₂
    digit₂ covers₂ range₂ shape₃ radix₃ certified₃ path₃ digit₃ covers₃ range₃ stride₂ degree₂
    length₂ blocks₂ point₁ covered₁ stride₃ degree₃ length₃ blocks₃ point₂ covered₂ quarter₄
    quarter₅ quarter₆ quarter₇ source₄ source₅ source₆ source₇ sized₄ sized₅ sized₆ vars₃ vars₄
    vars₅ vars₆ point₃ point₄ point₅ point₆ covered₃ covered₄ covered₅ covered₆ normRoom₅
    normRoom₆ wide₅ wide₆ capped₅ capped₆ next inside reads clears counts table decodes₁ decodes₂
    decodes₃ decodes₄ decodes₅ decodes₆ decodesL worked on₀ len₀ committed₁ valued₁ placed₁
    (longρ.trans setup) (long₂.trans point₀) (fun cell large => zero₀ cell (covered₀.trans large))
    fit₁
  have joined : early ++ (answer table wire₀ early, wireR) :: (later₀ ++ later) =
      early ++ [(answer table wire₀ early, wireR)] ++ later₀ ++ later := by
    simp only [List.append_assoc, List.cons_append, List.nil_append]
  refine ⟨(answer table wire₀ early, wireR) :: (later₀ ++ later), joined ▸ onL, joined ▸ whole,
    fun stage rooted' shape => ⟨linkedR stage rooted' ⟨_, shape⟩,
      linked₀ stage rooted' ⟨later, ?_⟩, each stage rooted' (shape.trans joined)⟩⟩
  rw [shape, joined]

end Off

end NightstreamFPrime.Spec.Compression.Akita.Extract
