import NightstreamFPrime.Spec.Compression.Akita.Extract.LiftComplete

/-! The verifier of a packing level, of three trace levels and of the terminal of the opening of
Akita accepts the honest prover (ePrint 2026/1983, revision of 2026-10-15, Theorem 10.9 with
Lemma 10.7, Lemma 7.1, item 1, Section 4.5, Figure 4 and Section 8.2). The levels are those of
the schedule `key30.aks` from the packing level 3: a level `committedTrace` on the coefficient
route, two levels `committedTraceSq` on the Euclidean route, each with the tensor reduction, and
the terminal on the Euclidean route.

- `traceTower₂_complete` is `traceTower_complete` with two levels `committedTraceSq` in place of
  three.
- `packTower₂_complete` is `packTower_complete` over the levels of `traceTower₂_complete`.

The fit conditions `LiftFit` and `TraceFit` are hypotheses (Akita Definition 10.5). The file has
no bound for the probability `ξ_h` of Akita (155). -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Extract

open scoped BigOperators
open NightstreamFPrime.Spec.Compression.Lib (mle eqAt)
open NightstreamFPrime.Spec.Checkpoint

local instance : Fact (Nat.Prime fieldModulus) := ⟨FieldPrime.fieldModulus_natPrime⟩

attribute [local instance low] Classical.propDecidable

variable {E : Type} [Field E] [Fintype E] [DecidableEq E] [Algebra Zq E] [Degree2 E]

section Trace

variable {S₂ : Shape} {O₂ : OpeningShape S₂} {fold₂ : Descriptor} {S₃ : Shape}
  {O₃ : OpeningShape S₃} {fold₃ : Descriptor} {S₄ : Shape} {O₄ : OpeningShape S₄}
  {fold₄ : Descriptor} {S₅ : Shape}
  {Wire Coord : Type} [Fintype Coord] [DecidableEq Coord] {total : ℕ}
  {accept : Prefix Wire (Option Coord → E) total → Prop} {read : Reader Wire} {start : ℕ}
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
  {operator₃ limit₃ operator₄ limit₄ operatorL limitL : ℕ}
  {L : Terminal S₅ E Wire Coord E total (.euclidean operatorL limitL)} {m₅ : ℕ}
  {commit₃ : Wire → S₃.Payload} {commit₄ : Wire → S₄.Payload}
  {stateOf : Wire → Fin S₅.blocks → Fin S₅.innerRank → Rq S₅.innerDegree}
  {statementOf : Wire → TerminalStatement S₅ E}
  {partialsOf : Wire → Fin S₅.blocks → Rq S₅.innerDegree}

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

/-- `traceTower_complete` for a level `committedTrace`, two levels `committedTraceSq` and the
terminal (Akita Theorem 10.9, without the probability that a response does not fit and without the
term of (155) for a zero tensor factor). `early` is a transcript prefix on the path of the table
of the oracle whose last message has the claim `honestTraceClaim` of the level 2 for a packed
polynomial `g`. Then later messages exist such that the complete transcript has the links
`Tier.TraceLinked` from the level 2 to the level 4, the link `Terminal.Linked` and the checks
`Terminal.Passed`. The hypotheses are those of `traceTower_complete` without the level 5, with the
terminal shape `S₅`. -/
theorem traceTower₂_complete {sendable : (M : Type) → M → Prop} {slots width cap : ℕ}
    (sends : Sends sendable E width cap) (four : 4 ≤ width) (room : 3 < slots)
    (carries₂ : Carries sendable slots read fun wire => (commit₃ wire, claim₃ wire))
    (carries₃ : Carries sendable slots read fun wire => (commit₄ wire, claim₄ wire))
    (carries₄ : Carries sendable slots read fun wire =>
      (stateOf wire, statementOf wire, partialsOf wire))
    (carriesL : Carries sendable slots read L.response)
    {p₂ s₂ k₂ p₃ s₃ k₃ p₄ s₄ k₄ p₅ s₅ k₅ : ℕ}
    (quarter₂ : 4 * m₂ = S₂.innerDegree) (quarter₃ : 4 * m₃ = S₃.innerDegree)
    (quarter₄ : 4 * m₄ = S₄.innerDegree) (quarter₅ : 4 * m₅ = S₅.innerDegree)
    (source₂ : SourceSized S₂ m₂ p₂ s₂ k₂) (source₃ : SourceSized S₃ m₃ p₃ s₃ k₃)
    (source₄ : SourceSized S₄ m₄ p₄ s₄ k₄) (source₅ : SourceSized S₅ m₅ p₅ s₅ k₅)
    (sized₂ : FoldSized S₂ O₂ fold₂ width) (sized₃ : FoldSized S₃ O₃ fold₃ width)
    (sized₄ : FoldSized S₄ O₄ fold₄ width)
    (vars₂ : μ₂ = ν₂ + 1) (vars₃ : μ₃ = ν₃ + 1) (vars₄ : μ₄ = ν₄ + 1)
    (point₂ : ν₂ = p₃ + (s₃ + k₃)) (point₃ : ν₃ = p₄ + (s₄ + k₄))
    (point₄ : ν₄ = p₅ + (s₅ + k₅))
    (covered₂ : O₂.cells fold₂ ≤ 2 * (S₃.blocks * (S₃.blockLength * (2 * m₃))))
    (covered₃ : O₃.cells fold₃ ≤ 2 * (S₄.blocks * (S₄.blockLength * (2 * m₄))))
    (covered₄ : O₄.cells fold₄ ≤ 2 * (S₅.blocks * (S₅.blockLength * (2 * m₅))))
    (normRoom₃ : S₃.blockLength * S₃.sourceDigits * S₃.innerDegree ≤ 2 ^ μ₃)
    (normRoom₄ : S₄.blockLength * S₄.sourceDigits * S₄.innerDegree ≤ 2 ^ μ₄)
    (wide₃ : rangeWidthSq S₃.certifiedBase ≤ width) (wide₄ : rangeWidthSq S₄.certifiedBase ≤ width)
    (capped₃ : limit₃ ≤ cap) (capped₄ : limit₄ ≤ cap)
    (next : L.start = Tier.finish fourth) (inside : L.start < total)
    (reads : ∀ point, L.read point = statementOf (point.wire L.start))
    (clears : ∀ point, L.partials point = partialsOf (point.wire L.start))
    (counts : ∀ wire, L.nonce wire = read.nonce wire)
    (table : Prefix Wire (Option Coord → E) total → Option Coord → E)
    (decodes₂ : ∀ input i, (decode₂ (table input (some (block₂ i)))).isSome)
    (decodes₃ : ∀ input i, (decode₃ (table input (some (block₃ i)))).isSome)
    (decodes₄ : ∀ input i, (decode₄ (table input (some (block₄ i)))).isSome)
    (decodesL : ∀ input i, (L.decode (table input (L.slot i))).isSome)
    (worked : ∀ input, table input none ∈ workZeros E 57)
    {first : Wire} {early : List ((Option Coord → E) × Wire)} (on : OnPath accept table first early)
    (starts : early.length = start) {g : ℕ → E} {block position pack : List E} {θ : E}
    (packs : pack.length = p₂) (positions : position.length = s₂) (blocks : k₂ ≤ block.length)
    (live : ∀ cell, S₂.blocks * (S₂.blockLength * (2 * m₂)) ≤ cell → g cell = 0)
    (ready : claim₂ (lastWire first early) =
      honestTraceClaim quarter₂ key₂ openKey₂ g block position pack θ)
    (tried : read.nonce (lastWire first early) < responseTries)
    (fit : TraceFit quarter₂ key₂ openKey₂ fold₂ (fun _ => decode₂) (fun _ => True) g fun w₂ =>
      TraceFit quarter₃ key₃ openKey₃ fold₃ (fun _ => decode₃) (fun z => vecNormSq z ≤ limit₃)
        (packed w₂) fun w₃ =>
      TraceFit quarter₄ key₄ openKey₄ fold₄ (fun _ => decode₄) (fun z => vecNormSq z ≤ limit₄)
        (packed w₃) fun w₄ =>
      ∀ c : Fin S₅.blocks → Rq S₅.innerDegree, (∀ i, ∃ slot, L.decode slot = some (c i)) →
        vecNormSq (traceResponse c (traceSource quarter₅ (packed w₄ : ℕ → E))) ≤ limitL) :
    ∃ later, OnPath accept table first (early ++ later) ∧ (early ++ later).length = total ∧
      ∀ stage : Prefix Wire (Option Coord → E) total, stage.first = first →
        stage.rest = early ++ later →
        Tier.TraceLinked second third quarter₃ commit₃ stage ∧
        Tier.TraceLinked third fourth quarter₄ commit₄ stage ∧
        Terminal.Linked fourth L quarter₅ stateOf stage ∧ L.Passed stage := by
  subst packs positions
  have ends₂ := Tier.finish_eq second
  have ends₃ := Tier.finish_eq third
  have ends₄ := Tier.finish_eq fourth
  have three : 3 ≤ width := by omega
  have positive : 0 < slots := by omega
  have several : 1 < slots := by omega
  have fin₄ : Tier.finish fourth ≤ total := by
    rw [← next]
    exact inside.le
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
  obtain ⟨later₂, blockSeg₃, positionSeg₃, packSeg₃, θ₃, rfl, rfl, long₃, on₂, len₂, fresh₂,
      ready₃, linked₂⟩ :=
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
  obtain ⟨later₃, blockSeg₄, positionSeg₄, packSeg₄, θ₄, rfl, rfl, long₄, on₃, len₃, fresh₃,
      ready₄, linked₃⟩ :=
    traceLink (B := base₃) (ν := ν₃) (Next := fourth) (commit' := commit₄) carries₃ sends
      three on₂ len₂ fin₃ decodes₃ worked ⟨foldWorkBits_le _ _, map_workBits_le _⟩ fresh₂
      (fun wire challenge table other => committedTraceSq_uniform wire challenge table other)
      committedTraceSq_costs_length several _ wins₃ rfl quarter₄ key₄ openKey₄
      source₄.pack source₄.length source₄.blocks point₃
  have live₄ := fun cell (large : S₄.blocks * (S₄.blockLength * (2 * m₄)) ≤ cell) =>
    packed_of_le (E := E) zero₃ (covered₃.trans (Nat.mul_le_mul_left 2 large))
  obtain ⟨fitL, zero₄, wins₄⟩ := traceLevel_wins (B := base₄)
    (admissible := fun z => vecNormSq z ≤ limit₄) (ν := ν₄)
    (fun wire c w satisfies admitted => (committedTraceSq_winsAt sends wide₄ four
      sized₄.two sized₄.small (by omega) normRoom₄ satisfies admitted capped₄).mono
        fun _ _ _ kept => ⟨kept.1, kept.2.imp fun _ both => ⟨both.1.trans vars₄, both.2⟩⟩)
    quarter₄ source₄.shape source₄.unit sized₄.radix sized₄.certified sized₄.path sized₄.wide
    (source₄.blocks.trans (Nat.pow_le_pow_right (by decide) long₄)) source₄.length source₄.pack
    live₄ ready₄ fit₄ (c := Tier.drawn base₄ table first (early ++ later₂ ++ later₃))
    fun i => ⟨_, Tier.drawn_mem base₄ table (len₃.le.trans fin₃) decodes₄ i⟩
  obtain ⟨later₄, blockSeg₅, positionSeg₅, packSeg₅, θ₅, rfl, rfl, long₅, on₄, len₄, fresh₄,
      stated, cleared, linked₄⟩ :=
    terminalLink (B := base₄) (ν := ν₄) (L := L) (stateOf := stateOf) carries₄ sends three
      on₃ len₃ fin₄ decodes₄ worked ⟨foldWorkBits_le _ _, map_workBits_le _⟩ fresh₃
      (fun wire challenge table other => committedTraceSq_uniform wire challenge table other)
      committedTraceSq_costs_length several _ wins₄ next reads quarter₅
      source₅.pack source₅.length source₅.blocks point₄
  have live₅ := fun cell (large : S₅.blocks * (S₅.blockLength * (2 * m₅)) ≤ cell) =>
    packed_of_le (E := E) zero₄ (covered₄.trans (Nat.mul_le_mul_left 2 large))
  obtain ⟨last, onL, whole, passed⟩ := terminal_passed carriesL on₄ len₄ inside reads clears
    decodesL worked ((counts _).trans_lt fresh₄) quarter₅ source₅.shape source₅.unit
    (source₅.blocks.trans (Nat.pow_le_pow_right (by decide) long₅)) source₅.length source₅.pack
    live₅ stated cleared fitL
  have joined : early ++ (later₂ ++ (later₃ ++ (later₄ ++ last))) =
      early ++ later₂ ++ later₃ ++ later₄ ++ last := by
    simp only [List.append_assoc]
  refine ⟨later₂ ++ (later₃ ++ (later₄ ++ last)), joined ▸ onL, joined ▸ whole,
    fun stage rooted shape =>
      ⟨linked₂ stage rooted ⟨later₃ ++ (later₄ ++ last), ?_⟩,
        linked₃ stage rooted ⟨later₄ ++ last, ?_⟩, linked₄ stage rooted ⟨last, ?_⟩,
        passed stage rooted (shape.trans joined)⟩⟩
  · rw [shape]
    simp only [List.append_assoc]
  · rw [shape]
    simp only [List.append_assoc]
  · rw [shape]
    simp only [List.append_assoc]

end Trace

section Tower

variable {S₁ : Shape} {O₁ : OpeningShape S₁} {fold₁ : Descriptor} {S₂ : Shape}
  {O₂ : OpeningShape S₂} {fold₂ : Descriptor} {S₃ : Shape} {O₃ : OpeningShape S₃}
  {fold₃ : Descriptor} {S₄ : Shape} {O₄ : OpeningShape S₄} {fold₄ : Descriptor} {S₅ : Shape}
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
  {operator₃ limit₃ operator₄ limit₄ operatorL limitL : ℕ}
  {L : Terminal S₅ E Wire Coord E total (.euclidean operatorL limitL)} {m₅ : ℕ}
  {commit₂ : Wire → S₂.Payload} {commit₃ : Wire → S₃.Payload} {commit₄ : Wire → S₄.Payload}
  {stateOf : Wire → Fin S₅.blocks → Fin S₅.innerRank → Rq S₅.innerDegree}
  {statementOf : Wire → TerminalStatement S₅ E}
  {partialsOf : Wire → Fin S₅.blocks → Rq S₅.innerDegree}

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

/-- `packTower_complete` over the levels of `traceTower₂_complete` (Akita Theorem 10.9, without
the probability that a response does not fit and without the term of (155) for a zero tensor
factor). `early` is a transcript prefix on the path of the table of the oracle whose last message
has the claim `sourceClaim` of the packing level 1 for the source blocks `source`. Then later
messages exist such that the complete transcript has the link `Tier.TraceLinked` from the level 1
to the level 2, the links from the level 2 to the level 4, the link `Terminal.Linked` and the
checks `Terminal.Passed`. The hypotheses are those of `packTower_complete` without the level 5,
with the terminal shape `S₅`. -/
theorem packTower₂_complete {sendable : (M : Type) → M → Prop} {slots width cap : ℕ}
    (sends : Sends sendable E width cap) (four : 4 ≤ width) (room : 3 < slots)
    (carries₁ : Carries sendable slots read fun wire => (commit₂ wire, claim₂ wire))
    (carries₂ : Carries sendable slots read fun wire => (commit₃ wire, claim₃ wire))
    (carries₃ : Carries sendable slots read fun wire => (commit₄ wire, claim₄ wire))
    (carries₄ : Carries sendable slots read fun wire =>
      (stateOf wire, statementOf wire, partialsOf wire))
    (carriesL : Carries sendable slots read L.response)
    {p₂ s₂ k₂ p₃ s₃ k₃ p₄ s₄ k₄ p₅ s₅ k₅ : ℕ}
    (shape₁ : S₁.Bounded) (radix₁ : Radix fold₁.responseBase)
    (certified₁ : fold₁.responseBase ≤ S₁.certifiedBase)
    (path₁ : PathFits O₁.openDegree (1 * O₁.openRank) O₁.chain)
    (digit₁ : Radix S₁.certifiedBase)
    (covers₁ : fieldModulus ≤ S₁.certifiedBase ^ digits₁)
    (range₁ : rangeWidth S₁.certifiedBase ≤ width)
    (quarter₂ : 4 * m₂ = S₂.innerDegree) (quarter₃ : 4 * m₃ = S₃.innerDegree)
    (quarter₄ : 4 * m₄ = S₄.innerDegree) (quarter₅ : 4 * m₅ = S₅.innerDegree)
    (source₂ : SourceSized S₂ m₂ p₂ s₂ k₂) (source₃ : SourceSized S₃ m₃ p₃ s₃ k₃)
    (source₄ : SourceSized S₄ m₄ p₄ s₄ k₄) (source₅ : SourceSized S₅ m₅ p₅ s₅ k₅)
    (sized₂ : FoldSized S₂ O₂ fold₂ width) (sized₃ : FoldSized S₃ O₃ fold₃ width)
    (sized₄ : FoldSized S₄ O₄ fold₄ width)
    (vars₁ : μ₁ = ν₁ + 1) (vars₂ : μ₂ = ν₂ + 1) (vars₃ : μ₃ = ν₃ + 1) (vars₄ : μ₄ = ν₄ + 1)
    (point₁ : ν₁ = p₂ + (s₂ + k₂)) (point₂ : ν₂ = p₃ + (s₃ + k₃))
    (point₃ : ν₃ = p₄ + (s₄ + k₄)) (point₄ : ν₄ = p₅ + (s₅ + k₅))
    (covered₁ : O₁.quotientRoom fold₁ digits₁ ≤ 2 * (S₂.blocks * (S₂.blockLength * (2 * m₂))))
    (covered₂ : O₂.cells fold₂ ≤ 2 * (S₃.blocks * (S₃.blockLength * (2 * m₃))))
    (covered₃ : O₃.cells fold₃ ≤ 2 * (S₄.blocks * (S₄.blockLength * (2 * m₄))))
    (covered₄ : O₄.cells fold₄ ≤ 2 * (S₅.blocks * (S₅.blockLength * (2 * m₅))))
    (normRoom₃ : S₃.blockLength * S₃.sourceDigits * S₃.innerDegree ≤ 2 ^ μ₃)
    (normRoom₄ : S₄.blockLength * S₄.sourceDigits * S₄.innerDegree ≤ 2 ^ μ₄)
    (wide₃ : rangeWidthSq S₃.certifiedBase ≤ width) (wide₄ : rangeWidthSq S₄.certifiedBase ≤ width)
    (capped₃ : limit₃ ≤ cap) (capped₄ : limit₄ ≤ cap)
    (next : L.start = Tier.finish fourth) (inside : L.start < total)
    (reads : ∀ point, L.read point = statementOf (point.wire L.start))
    (clears : ∀ point, L.partials point = partialsOf (point.wire L.start))
    (counts : ∀ wire, L.nonce wire = read.nonce wire)
    (table : Prefix Wire (Option Coord → E) total → Option Coord → E)
    (decodes₁ : ∀ input i, (decode₁ (table input (some (block₁ i)))).isSome)
    (decodes₂ : ∀ input i, (decode₂ (table input (some (block₂ i)))).isSome)
    (decodes₃ : ∀ input i, (decode₃ (table input (some (block₃ i)))).isSome)
    (decodes₄ : ∀ input i, (decode₄ (table input (some (block₄ i)))).isSome)
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
      ∀ c : Fin S₅.blocks → Rq S₅.innerDegree, (∀ i, ∃ slot, L.decode slot = some (c i)) →
        vecNormSq (traceResponse c (traceSource quarter₅ (packed w₄ : ℕ → E))) ≤ limitL) :
    ∃ later, OnPath accept table wire₀ (early ++ later) ∧ (early ++ later).length = total ∧
      ∀ stage : Prefix Wire (Option Coord → E) total, stage.first = wire₀ →
        stage.rest = early ++ later →
        Tier.TraceLinked first second quarter₂ commit₂ stage ∧
        Tier.TraceLinked second third quarter₃ commit₃ stage ∧
        Tier.TraceLinked third fourth quarter₄ commit₄ stage ∧
        Terminal.Linked fourth L quarter₅ stateOf stage ∧ L.Passed stage := by
  have ends₁ := Tier.finish_eq first
  have ends₂ := Tier.finish_eq second
  have ends₃ := Tier.finish_eq third
  have ends₄ := Tier.finish_eq fourth
  have fin₁ : Tier.finish first ≤ total :=
    (show Tier.finish first ≤ Tier.finish second from
      (Nat.le_add_right _ _).trans_eq ends₂.symm).trans
    ((show Tier.finish second ≤ Tier.finish third from
      (Nat.le_add_right _ _).trans_eq ends₃.symm).trans
    ((show Tier.finish third ≤ Tier.finish fourth from
      (Nat.le_add_right _ _).trans_eq ends₄.symm).trans (next ▸ inside.le)))
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
  obtain ⟨later, onL, whole, each⟩ := traceTower₂_complete sends four room carries₂ carries₃
    carries₄ carriesL quarter₂ quarter₃ quarter₄ quarter₅ source₂ source₃ source₄ source₅ sized₂
    sized₃ sized₄ vars₂ vars₃ vars₄ point₂ point₃ point₄ covered₂ covered₃ covered₄ normRoom₃
    normRoom₄ wide₃ wide₄ capped₃ capped₄ next inside reads clears counts table decodes₂ decodes₃
    decodes₄ decodesL worked on₁ len₁ packs positions long live₂ ready₂ fresh₁ fit₂
  refine ⟨later₁ ++ later, by rwa [← List.append_assoc], by rwa [← List.append_assoc],
    fun stage rooted shape => ⟨linked₁ stage rooted ⟨later, by rw [shape, List.append_assoc]⟩,
      each stage rooted (by rw [shape, List.append_assoc])⟩⟩

end Tower

end NightstreamFPrime.Spec.Compression.Akita.Extract
