import NightstreamFPrime.Spec.Compression.Akita.Extract.Tower
import NightstreamFPrime.Spec.Compression.Akita.Extract.TraceComplete

/-! The verifier of the trace levels and of the terminal of the opening of Akita accepts the
honest prover (ePrint 2026/1983, revision of 2026-11-23, Theorem 10.9 with Lemma 10.7, Figure 4
and Section 8.2). The theorems `tower_suffix` and `mixedTower_suffix` bound an event of the
verifier `towerLinks`. This file shows that the part of this verifier from the first trace level
is not empty.

The levels are those of `mixedTower_suffix` after the packing level: a level `committedTrace`
on the coefficient route, three levels `committedTraceSq` on the Euclidean route, each with the
tensor reduction, and the terminal on the Euclidean route.

- `traceTower_complete` gives, for each table of the oracle whose slots decode to fold
  challenges and each transcript prefix on its path whose last message has the claim
  `honestTraceClaim` of the first trace level, the later messages of a proof such that the
  complete transcript has the links `Tier.TraceLinked` of the trace levels, the link
  `Terminal.Linked` and the checks `Terminal.Passed`.
- `traceTower_accepts` is this statement for a proof whose first message has the claim.

The fit conditions `TraceFit` are hypotheses: each response is admissible and has the norm bound
of its route for each fold challenge vector of the families (Akita Definition 10.5). The file has
no bound for the probability `ξ_h` of Akita (155). -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Extract

open scoped BigOperators
open NightstreamFPrime.Spec.Compression.Lib (mle eqAt)
open NightstreamFPrime.Spec.Checkpoint
open NightstreamFPrime.Spec.RandomOracle (OracleComp)

local instance : Fact (Nat.Prime fieldModulus) := ⟨FieldPrime.fieldModulus_natPrime⟩

attribute [local instance low] Classical.propDecidable

variable {E : Type} [Field E] [Fintype E]
  {Slot : Type} [Fintype Slot] [Nonempty Slot] [FieldSlot Slot E]
  [DecidableEq E] [Algebra Zq E] [Degree2 E]

section Trace

variable {S₂ : Shape} {O₂ : OpeningShape S₂} {fold₂ : Descriptor} {S₃ : Shape}
  {O₃ : OpeningShape S₃} {fold₃ : Descriptor} {S₄ : Shape} {O₄ : OpeningShape S₄}
  {fold₄ : Descriptor} {S₅ : Shape} {O₅ : OpeningShape S₅} {fold₅ : Descriptor} {S₆ : Shape}
  {Wire Coord : Type} [Fintype Coord] [DecidableEq Coord] {total : ℕ}
  {accept : Prefix Wire (Option Coord → Slot) total → Prop} {read : Reader Wire} {start : ℕ}
  {key₂ : Key S₂} {openKey₂ : OpeningKey O₂} {claim₂ : Wire → TraceClaim O₂ E}
  {μ₂ : ℕ} {fits₂ : O₂.cells fold₂ ≤ 2 ^ μ₂} {batch₂ : ℕ}
  {rowsFit₂ : O₂.traceRowCount + 1 ≤ 2 ^ batch₂} {block₂ : Fin S₂.blocks → Coord}
  {decode₂ : Slot → Option (Rq S₂.innerDegree)} {weight₂ ν₂ m₂ : ℕ}
  {key₃ : Key S₃} {openKey₃ : OpeningKey O₃} {claim₃ : Wire → TraceClaim O₃ E}
  {μ₃ : ℕ} {fits₃ : O₃.cells fold₃ ≤ 2 ^ μ₃} {batch₃ : ℕ}
  {rowsFit₃ : O₃.traceRowCount + 1 ≤ 2 ^ batch₃} {block₃ : Fin S₃.blocks → Coord}
  {decode₃ : Slot → Option (Rq S₃.innerDegree)} {weight₃ ν₃ m₃ : ℕ}
  {key₄ : Key S₄} {openKey₄ : OpeningKey O₄} {claim₄ : Wire → TraceClaim O₄ E}
  {μ₄ : ℕ} {fits₄ : O₄.cells fold₄ ≤ 2 ^ μ₄} {batch₄ : ℕ}
  {rowsFit₄ : O₄.traceRowCount + 1 ≤ 2 ^ batch₄} {block₄ : Fin S₄.blocks → Coord}
  {decode₄ : Slot → Option (Rq S₄.innerDegree)} {weight₄ ν₄ m₄ : ℕ}
  {key₅ : Key S₅} {openKey₅ : OpeningKey O₅} {claim₅ : Wire → TraceClaim O₅ E}
  {μ₅ : ℕ} {fits₅ : O₅.cells fold₅ ≤ 2 ^ μ₅} {batch₅ : ℕ}
  {rowsFit₅ : O₅.traceRowCount + 1 ≤ 2 ^ batch₅} {block₅ : Fin S₅.blocks → Coord}
  {decode₅ : Slot → Option (Rq S₅.innerDegree)} {weight₅ ν₅ m₅ : ℕ}
  {operator₃ limit₃ operator₄ limit₄ operator₅ limit₅ operatorL limitL : ℕ}
  {L : Terminal S₆ E Wire Coord Slot total (.euclidean operatorL limitL)} {m₆ : ℕ}
  {commit₃ : Wire → S₃.Payload} {commit₄ : Wire → S₄.Payload} {commit₅ : Wire → S₅.Payload}
  {stateOf : Wire → Fin S₆.blocks → Fin S₆.innerRank → Rq S₆.innerDegree}
  {statementOf : Wire → TerminalStatement S₆ E}
  {partialsOf : Wire → Fin S₆.blocks → Rq S₆.innerDegree}

local notation "base₂" =>
  committedTrace (fold := fold₂) key₂ openKey₂ claim₂ μ₂ fits₂ batch₂ rowsFit₂ block₂ decode₂
    read weight₂ start

local notation "second" => Tier.tensored base₂ ν₂

local notation "base₃" =>
  committedTraceSq (fold := fold₃) key₃ openKey₃ claim₃ μ₃ fits₃ batch₃ rowsFit₃ block₃
    decode₃ read weight₃ operator₃ limit₃ (Tier.finish second)

local notation "third" => Tier.tensored base₃ ν₃

local notation "base₄" =>
  committedTraceSq (fold := fold₄) key₄ openKey₄ claim₄ μ₄ fits₄ batch₄ rowsFit₄ block₄
    decode₄ read weight₄ operator₄ limit₄ (Tier.finish third)

local notation "fourth" => Tier.tensored base₄ ν₄

local notation "base₅" =>
  committedTraceSq (fold := fold₅) key₅ openKey₅ claim₅ μ₅ fits₅ batch₅ rowsFit₅ block₅
    decode₅ read weight₅ operator₅ limit₅ (Tier.finish fourth)

local notation "fifth" => Tier.tensored base₅ ν₅

omit [FieldSlot Slot E] in
/-- Akita Theorem 10.9 for the trace levels and the terminal of the opening, without the probability
that a response does not fit and without the term of (155) for a zero tensor factor, because this
verifier has no rejection at a zero factor: the verifier of `mixedTower_suffix` from the first trace
level accepts the honest prover. `early` is a transcript prefix on the path of the table of the
oracle whose last message has the claim `honestTraceClaim` of the level 2 for a packed polynomial
`g`. Then later messages exist such that the complete transcript has the links `Tier.TraceLinked`
from the level 2 to the level 5, the link `Terminal.Linked` and the checks `Terminal.Passed`. The
hypotheses are: `sends`, `four`, `room`, `carries₂` to `carriesL`: a wire message can have the data
of the prover with at most `slots` messages of the checks. `source₂` to `source₆` and `sized₂` to
`sized₅`: the conditions `SourceSized` and `FoldSized` of the shapes. `vars`, `point` and `covered`:
the tensor reduction of each level has `μ − 1` tail variables, its point has the segments of the
next shape, and the live blocks of the next shape have each cell of the flat witness (Akita Section
4.5, (40) to (42)). `normRoom`, `wide₃` to `wide₅` and `capped`: the cube of the witness has the coordinates of `z`,
and a message can have each message of the checks and the energy. `next`, `inside`, `reads`, `clears`: the fold challenge
of the terminal is the stage after the level 5, and the terminal reads its statement and its clear
partials from the message before it. `counts`: the terminal reads the nonce of the response
search with `read`. `tried`: the nonce of the response search in the last message of `early` is
below `responseTries`. `decodes`: each slot of the table of the oracle decodes to a
fold challenge, and `worked`: each work slot of the table has 57 zero work bits. `fit`: the fit
conditions `TraceFit` of the levels 2 to 5, and `‖z‖_2² ≤ S_max,term`
at the terminal, for each fold challenge vector of the families (Akita Definition 10.5). -/
theorem traceTower_complete {sendable : (M : Type) → M → Prop} {slots width cap : ℕ}
    (sends : Sends sendable E width cap) (four : 4 ≤ width) (room : 3 < slots)
    (carries₂ : Carries sendable slots read fun wire => (commit₃ wire, claim₃ wire))
    (carries₃ : Carries sendable slots read fun wire => (commit₄ wire, claim₄ wire))
    (carries₄ : Carries sendable slots read fun wire => (commit₅ wire, claim₅ wire))
    (carries₅ : Carries sendable slots read fun wire =>
      (stateOf wire, statementOf wire, partialsOf wire))
    (carriesL : Carries sendable slots read L.response)
    {p₂ s₂ k₂ p₃ s₃ k₃ p₄ s₄ k₄ p₅ s₅ k₅ p₆ s₆ k₆ : ℕ}
    (quarter₂ : 4 * m₂ = S₂.innerDegree) (quarter₃ : 4 * m₃ = S₃.innerDegree)
    (quarter₄ : 4 * m₄ = S₄.innerDegree) (quarter₅ : 4 * m₅ = S₅.innerDegree)
    (quarter₆ : 4 * m₆ = S₆.innerDegree)
    (source₂ : SourceSized S₂ m₂ p₂ s₂ k₂) (source₃ : SourceSized S₃ m₃ p₃ s₃ k₃)
    (source₄ : SourceSized S₄ m₄ p₄ s₄ k₄) (source₅ : SourceSized S₅ m₅ p₅ s₅ k₅)
    (source₆ : SourceSized S₆ m₆ p₆ s₆ k₆)
    (sized₂ : FoldSized S₂ O₂ fold₂ width) (sized₃ : FoldSized S₃ O₃ fold₃ width)
    (sized₄ : FoldSized S₄ O₄ fold₄ width) (sized₅ : FoldSized S₅ O₅ fold₅ width)
    (vars₂ : μ₂ = ν₂ + 1) (vars₃ : μ₃ = ν₃ + 1) (vars₄ : μ₄ = ν₄ + 1) (vars₅ : μ₅ = ν₅ + 1)
    (point₂ : ν₂ = p₃ + (s₃ + k₃)) (point₃ : ν₃ = p₄ + (s₄ + k₄))
    (point₄ : ν₄ = p₅ + (s₅ + k₅)) (point₅ : ν₅ = p₆ + (s₆ + k₆))
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
    (table : Prefix Wire (Option Coord → Slot) total → Option Coord → Slot)
    (decodes₂ : ∀ input i, (decode₂ (table input (some (block₂ i)))).isSome)
    (decodes₃ : ∀ input i, (decode₃ (table input (some (block₃ i)))).isSome)
    (decodes₄ : ∀ input i, (decode₄ (table input (some (block₄ i)))).isSome)
    (decodes₅ : ∀ input i, (decode₅ (table input (some (block₅ i)))).isSome)
    (decodesL : ∀ input i, (L.decode (table input (L.slot i))).isSome)
    (worked : ∀ input, table input none ∈ workZeros Slot 57)
    {first : Wire} {early : List ((Option Coord → Slot) × Wire)} (on : OnPath accept table first early)
    (starts : early.length = start) {g : ℕ → E} {block position pack : List E} {θ : E}
    (packs : pack.length = p₂) (positions : position.length = s₂) (blocks : k₂ ≤ block.length)
    (live : ∀ cell, S₂.blocks * (S₂.blockLength * (2 * m₂)) ≤ cell → g cell = 0)
    (ready : claim₂ (lastWire first early) =
      honestTraceClaim quarter₂ key₂ openKey₂ g block position pack θ)
    (tried : read.nonce (lastWire first early) < responseTries)
    (fit : TraceFit (E := E) quarter₂ key₂ openKey₂ fold₂ (fun _ => decode₂) (fun _ => True) g fun w₂ =>
      TraceFit (E := E) quarter₃ key₃ openKey₃ fold₃ (fun _ => decode₃) (fun z => vecNormSq z ≤ limit₃)
        (packed w₂) fun w₃ =>
      TraceFit (E := E) quarter₄ key₄ openKey₄ fold₄ (fun _ => decode₄) (fun z => vecNormSq z ≤ limit₄)
        (packed w₃) fun w₄ =>
      TraceFit (E := E) quarter₅ key₅ openKey₅ fold₅ (fun _ => decode₅) (fun z => vecNormSq z ≤ limit₅)
        (packed w₄) fun w₅ =>
      ∀ c : Fin S₆.blocks → Rq S₆.innerDegree, (∀ i, ∃ slot, L.decode slot = some (c i)) →
        vecNormSq (traceResponse c (traceSource quarter₆ (packed w₅ : ℕ → E))) ≤ limitL) :
    ∃ later, OnPath accept table first (early ++ later) ∧ (early ++ later).length = total ∧
      ∀ stage : Prefix Wire (Option Coord → Slot) total, stage.first = first →
        stage.rest = early ++ later →
        Tier.TraceLinked second third quarter₃ commit₃ stage ∧
        Tier.TraceLinked third fourth quarter₄ commit₄ stage ∧
        Tier.TraceLinked fourth fifth quarter₅ commit₅ stage ∧
        Terminal.Linked fifth L quarter₆ stateOf stage ∧ L.Passed stage := by
  subst packs positions
  have ends₂ := Tier.finish_eq second
  have ends₃ := Tier.finish_eq third
  have ends₄ := Tier.finish_eq fourth
  have ends₅ := Tier.finish_eq fifth
  have three : 3 ≤ width := by omega
  have positive : 0 < slots := by omega
  have several : 1 < slots := by omega
  have fin₅ : Tier.finish fifth ≤ total := by
    rw [← next]
    exact inside.le
  have fin₄ : Tier.finish fourth ≤ total :=
    (show Tier.finish fourth ≤ Tier.finish fifth from
      (Nat.le_add_right _ _).trans_eq ends₅.symm).trans fin₅
  have fin₃ : Tier.finish third ≤ total :=
    (show Tier.finish third ≤ Tier.finish fourth from
      (Nat.le_add_right _ _).trans_eq ends₄.symm).trans fin₄
  have fin₂ : Tier.finish second ≤ total :=
    (show Tier.finish second ≤ Tier.finish third from
      (Nat.le_add_right _ _).trans_eq ends₃.symm).trans fin₃
  have short₂ : early.length ≤ total :=
    (starts.le.trans ((Nat.le_add_right _ _).trans_eq ends₂.symm)).trans fin₂
  obtain ⟨fit₃, zero₂, wins₂⟩ := traceLevel_wins (B := base₂) (admissible := fun _ => True)
    (ν := ν₂)
    (fun wire c w satisfies _ => (committedTrace_winsAt sends sized₂.range four sized₂.two
      sized₂.small positive satisfies).mono fun _ _ _ kept =>
        ⟨kept.1, kept.2.imp fun _ both => ⟨both.1.trans vars₂, both.2⟩⟩)
    quarter₂ source₂.shape source₂.unit sized₂.radix sized₂.certified sized₂.path sized₂.wide
    (source₂.blocks.trans (Nat.pow_le_pow_right (by decide) blocks)) source₂.length source₂.pack
    live ready fit (c := Tier.drawn base₂ table first early)
    fun i => ⟨_, Tier.drawn_mem base₂ table short₂ decodes₂ i⟩
  obtain ⟨later₂, blockSeg₃, positionSeg₃, packSeg₃, θ₃, rfl, rfl, long₃, on₂, len₂, fresh₂, ready₃,
      linked₂⟩ :=
    traceLink (B := base₂) (ν := ν₂) (Next := third) (commit' := commit₃) carries₂ sends
      three on starts fin₂ decodes₂ worked ⟨foldWorkBits_le _ _, map_workBits_le _⟩ tried
      (fun wire challenge table other => committedTrace_uniform wire challenge table other)
      (foldCosts_length E _ _ _ _ _) several _ wins₂ rfl quarter₃ key₃ openKey₃
      source₃.pack source₃.length source₃.blocks point₂
  have live₃ := fun cell (large : S₃.blocks * (S₃.blockLength * (2 * m₃)) ≤ cell) =>
    packed_of_le (E := E) zero₂ (covered₂.trans (Nat.mul_le_mul_left 2 large))
  obtain ⟨fit₄, zero₃, wins₃⟩ := traceLevel_wins (B := base₃)
    (admissible := fun z => vecNormSq z ≤ limit₃) (ν := ν₃)
    (fun wire c w satisfies admitted => (committedTraceSq_winsAt sends wide₃ four
      sized₃.two sized₃.small (by omega) normRoom₃ satisfies admitted capped₃).mono
        fun _ _ _ kept => ⟨kept.1, kept.2.imp fun _ both => ⟨both.1.trans vars₃, both.2⟩⟩)
    quarter₃ source₃.shape source₃.unit sized₃.radix sized₃.certified sized₃.path sized₃.wide
    (source₃.blocks.trans (Nat.pow_le_pow_right (by decide) long₃)) source₃.length source₃.pack
    live₃ ready₃ fit₃ (c := Tier.drawn base₃ table first (early ++ later₂))
    fun i => ⟨_, Tier.drawn_mem base₃ table (len₂.le.trans fin₂) decodes₃ i⟩
  obtain ⟨later₃, blockSeg₄, positionSeg₄, packSeg₄, θ₄, rfl, rfl, long₄, on₃, len₃, fresh₃, ready₄,
      linked₃⟩ :=
    traceLink (B := base₃) (ν := ν₃) (Next := fourth) (commit' := commit₄) carries₃ sends
      three on₂ len₂ fin₃ decodes₃ worked ⟨foldWorkBits_le _ _, map_workBits_le _⟩ fresh₂
      (fun wire challenge table other => committedTraceSq_uniform wire challenge table other)
      committedTraceSq_costs_length several _ wins₃ rfl quarter₄ key₄ openKey₄
      source₄.pack source₄.length source₄.blocks point₃
  have live₄ := fun cell (large : S₄.blocks * (S₄.blockLength * (2 * m₄)) ≤ cell) =>
    packed_of_le (E := E) zero₃ (covered₃.trans (Nat.mul_le_mul_left 2 large))
  obtain ⟨fit₅, zero₄, wins₄⟩ := traceLevel_wins (B := base₄)
    (admissible := fun z => vecNormSq z ≤ limit₄) (ν := ν₄)
    (fun wire c w satisfies admitted => (committedTraceSq_winsAt sends wide₄ four
      sized₄.two sized₄.small (by omega) normRoom₄ satisfies admitted capped₄).mono
        fun _ _ _ kept => ⟨kept.1, kept.2.imp fun _ both => ⟨both.1.trans vars₄, both.2⟩⟩)
    quarter₄ source₄.shape source₄.unit sized₄.radix sized₄.certified sized₄.path sized₄.wide
    (source₄.blocks.trans (Nat.pow_le_pow_right (by decide) long₄)) source₄.length source₄.pack
    live₄ ready₄ fit₄ (c := Tier.drawn base₄ table first (early ++ later₂ ++ later₃))
    fun i => ⟨_, Tier.drawn_mem base₄ table (len₃.le.trans fin₃) decodes₄ i⟩
  obtain ⟨later₄, blockSeg₅, positionSeg₅, packSeg₅, θ₅, rfl, rfl, long₅, on₄, len₄, fresh₄, ready₅,
      linked₄⟩ :=
    traceLink (B := base₄) (ν := ν₄) (Next := fifth) (commit' := commit₅) carries₄ sends
      three on₃ len₃ fin₄ decodes₄ worked ⟨foldWorkBits_le _ _, map_workBits_le _⟩ fresh₃
      (fun wire challenge table other => committedTraceSq_uniform wire challenge table other)
      committedTraceSq_costs_length several _ wins₄ rfl quarter₅ key₅ openKey₅
      source₅.pack source₅.length source₅.blocks point₄
  have live₅ := fun cell (large : S₅.blocks * (S₅.blockLength * (2 * m₅)) ≤ cell) =>
    packed_of_le (E := E) zero₄ (covered₄.trans (Nat.mul_le_mul_left 2 large))
  obtain ⟨fitL, zero₅, wins₅⟩ := traceLevel_wins (B := base₅)
    (admissible := fun z => vecNormSq z ≤ limit₅) (ν := ν₅)
    (fun wire c w satisfies admitted => (committedTraceSq_winsAt sends wide₅ four
      sized₅.two sized₅.small (by omega) normRoom₅ satisfies admitted capped₅).mono
        fun _ _ _ kept => ⟨kept.1, kept.2.imp fun _ both => ⟨both.1.trans vars₅, both.2⟩⟩)
    quarter₅ source₅.shape source₅.unit sized₅.radix sized₅.certified sized₅.path sized₅.wide
    (source₅.blocks.trans (Nat.pow_le_pow_right (by decide) long₅)) source₅.length source₅.pack
    live₅ ready₅ fit₅ (c := Tier.drawn base₅ table first (early ++ later₂ ++ later₃ ++ later₄))
    fun i => ⟨_, Tier.drawn_mem base₅ table (len₄.le.trans fin₄) decodes₅ i⟩
  obtain ⟨later₅, blockSeg₆, positionSeg₆, packSeg₆, θ₆, rfl, rfl, long₆, on₅, len₅, fresh₅, stated,
      cleared, linked₅⟩ :=
    terminalLink (B := base₅) (ν := ν₅) (L := L) (stateOf := stateOf) carries₅ sends three
      on₄ len₄ fin₅ decodes₅ worked ⟨foldWorkBits_le _ _, map_workBits_le _⟩ fresh₄
      (fun wire challenge table other => committedTraceSq_uniform wire challenge table other)
      committedTraceSq_costs_length several _ wins₅ next reads quarter₆
      source₆.pack source₆.length source₆.blocks point₅
  have live₆ := fun cell (large : S₆.blocks * (S₆.blockLength * (2 * m₆)) ≤ cell) =>
    packed_of_le (E := E) zero₅ (covered₅.trans (Nat.mul_le_mul_left 2 large))
  obtain ⟨last, onL, whole, passed⟩ := terminal_passed carriesL on₅ len₅ inside reads clears
    decodesL worked ((counts _).trans_lt fresh₅) quarter₆ source₆.shape source₆.unit
    (source₆.blocks.trans (Nat.pow_le_pow_right (by decide) long₆)) source₆.length source₆.pack
    live₆ stated cleared fitL
  have joined : early ++ (later₂ ++ (later₃ ++ (later₄ ++ (later₅ ++ last)))) =
      early ++ later₂ ++ later₃ ++ later₄ ++ later₅ ++ last := by
    simp only [List.append_assoc]
  refine ⟨later₂ ++ (later₃ ++ (later₄ ++ (later₅ ++ last))), joined ▸ onL,
    joined ▸ whole, fun stage rooted shape =>
      ⟨linked₂ stage rooted ⟨later₃ ++ (later₄ ++ (later₅ ++ last)), ?_⟩,
        linked₃ stage rooted ⟨later₄ ++ (later₅ ++ last), ?_⟩,
        linked₄ stage rooted ⟨later₅ ++ last, ?_⟩, linked₅ stage rooted ⟨last, ?_⟩,
        passed stage rooted (shape.trans joined)⟩⟩
  · rw [shape]
    simp only [List.append_assoc]
  · rw [shape]
    simp only [List.append_assoc]
  · rw [shape]
    simp only [List.append_assoc]
  · rw [shape]
    simp only [List.append_assoc]

end Trace

end NightstreamFPrime.Spec.Compression.Akita.Extract
