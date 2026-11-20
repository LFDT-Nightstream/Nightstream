import NightstreamFPrime.Spec.Compression.Akita.Key20Tower
import NightstreamFPrime.Spec.Compression.Akita.Extract.TowerComplete

/-! The honest prover of the trace levels and of the terminal of the opening of Akita for the
schedule file `crates/ns-compress/artifacts/key20.aks` (Akita, ePrint 2026/1983, revision of
2026-10-15, Theorem 10.9 with Lemma 10.7, Figure 4 and Section 8.2).

- `not_honest`: the shapes of the levels 1 to 5 and of the terminal have one source digit of the
  base 8, 32 or 64, so they do not satisfy `Shape.Honest`: the schedule has the bounded source
  `s_i := f_i` of Akita Section 4.5 at these levels.
- `bounded`: the shapes of the levels 2 to 5 and of the terminal satisfy `Shape.Bounded`.
- `source₂` to `source₆` and `fold₂` to `fold₅` are the conditions `SourceSized` and `FoldSized`
  of `traceTower_complete`: the point of each receiver has the segments `(ρ_pack, ρ_pos, ρ_blk)`
  of the lengths (6, 9, 5), (6, 8, 4), (5, 9, 3), (5, 8, 3) and (5, 8, 3), and a message of the
  range tree has at most 8 field elements.
- `links`: the tensor reduction of each level has `μ − 1` tail variables, which is the count of
  the coordinates of the point of the next level, and the live blocks of the next level have each
  cell of the flat witness.
- `traceTower_complete` is `Extract.traceTower_complete` for these numbers, and
  `traceTower_accepts` is this statement for a proof whose first message has the claim of the
  level 2. The fit conditions `TraceFit` stay hypotheses on the responses of the honest
  prover. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Key20

open scoped BigOperators
open NightstreamFPrime.Spec.Compression.Akita.Extract
open NightstreamFPrime.Spec.Compression.Lib (mle eqAt)
open NightstreamFPrime.Spec.Checkpoint
open NightstreamFPrime.Spec.RandomOracle (OracleComp)

local instance : Fact (Nat.Prime fieldModulus) := ⟨FieldPrime.fieldModulus_natPrime⟩

attribute [local instance low] Classical.propDecidable

/-- The shapes of the levels 1 to 5 and of the terminal do not satisfy `Shape.Honest`, the
hypothesis of `commit_opens`, `honestClaim_satisfies` and `packingPair_complete`: they have one
source digit of the base 8, 32 or 64, and `b^{δ_com} < q`. The commitment algorithm `commit` with a
full-field source is not the algorithm of these levels. -/
theorem not_honest : ¬level₁.shape.Honest ∧ ¬level₂.shape.Honest ∧ ¬level₃.shape.Honest ∧
    ¬level₄.shape.Honest ∧ ¬level₅.shape.Honest ∧ ¬terminal.Honest :=
  ⟨fun honest => absurd honest.source.full (by decide),
    fun honest => absurd honest.source.full (by decide),
    fun honest => absurd honest.source.full (by decide),
    fun honest => absurd honest.source.full (by decide),
    fun honest => absurd honest.source.full (by decide),
    fun honest => absurd honest.source.full (by decide)⟩

/-- The shapes of the levels 2 to 5 and of the terminal satisfy the conditions `Shape.Bounded` of
the commitment algorithm with a bounded source. -/
theorem bounded : level₂.shape.Bounded ∧ level₃.shape.Bounded ∧ level₄.shape.Bounded ∧
    level₅.shape.Bounded ∧ terminal.Bounded :=
  ⟨⟨⟨by decide, by decide, by decide, by decide⟩, by decide, by decide, by decide,
      pathFits_of (by decide) (by decide)⟩,
    ⟨⟨by decide, by decide, by decide, by decide⟩, by decide, by decide, by decide,
      pathFits_of (by decide) (by decide)⟩,
    ⟨⟨by decide, by decide, by decide, by decide⟩, by decide, by decide, by decide, trivial⟩,
    ⟨⟨by decide, by decide, by decide, by decide⟩, by decide, by decide, by decide, trivial⟩,
    ⟨⟨by decide, by decide, by decide, by decide⟩, by decide, by decide, by decide, trivial⟩⟩

/-- `d_A/k` = 64 and 32 are units of `Z_q`. -/
theorem units : IsUnit ((2 * 32 : ℕ) : Zq) ∧ IsUnit ((2 * 16 : ℕ) : Zq) :=
  ⟨(ZMod.isUnit_iff_coprime _ _).mpr (by decide), (ZMod.isUnit_iff_coprime _ _).mpr (by decide)⟩

theorem source₂ : SourceSized level₂.shape 32 6 9 5 :=
  ⟨bounded.1, units.1, by decide, by decide, by decide⟩

theorem source₃ : SourceSized level₃.shape 32 6 8 4 :=
  ⟨bounded.2.1, units.1, by decide, by decide, by decide⟩

theorem source₄ : SourceSized level₄.shape 16 5 9 3 :=
  ⟨bounded.2.2.1, units.2, by decide, by decide, by decide⟩

theorem source₅ : SourceSized level₅.shape 16 5 8 3 :=
  ⟨bounded.2.2.2.1, units.2, by decide, by decide, by decide⟩

theorem source₆ : SourceSized terminal 16 5 8 3 :=
  ⟨bounded.2.2.2.2, units.2, by decide, by decide, by decide⟩

theorem fold₂ : FoldSized level₂.shape level₂.opening level₂.fold 8 :=
  ⟨⟨by decide, by decide, by decide⟩, by decide, pathFits_of (by decide) (by decide), by decide,
    by decide, by decide, by decide⟩

theorem fold₃ : FoldSized level₃.shape level₃.opening level₃.fold 8 :=
  ⟨⟨by decide, by decide, by decide⟩, by decide, pathFits_of (by decide) (by decide), by decide,
    by decide, by decide, by decide⟩

theorem fold₄ : FoldSized level₄.shape level₄.opening level₄.fold 8 :=
  ⟨⟨by decide, by decide, by decide⟩, by decide, trivial, by decide, by decide, by decide,
    by decide⟩

theorem fold₅ : FoldSized level₅.shape level₅.opening level₅.fold 8 :=
  ⟨⟨by decide, by decide, by decide⟩, by decide, trivial, by decide, by decide, by decide,
    by decide⟩

/-- The conditions of the links of `traceTower_complete`: `μ = ν + 1` at each level, the count
`ν` of the coordinates of the point of each receiver, and the cells of each flat witness in the
live blocks of its receiver, as `2·B·M·(d_A/k)` coordinates of `Z_q` (Akita Section 4.5, (40) to
(42)). -/
theorem links :
    (level₂.vars = level₂.tail + 1 ∧ level₃.vars = level₃.tail + 1 ∧
      level₄.vars = level₄.tail + 1 ∧ level₅.vars = level₅.tail + 1) ∧
    (level₂.tail = 6 + (8 + 4) ∧ level₃.tail = 5 + (9 + 3) ∧ level₄.tail = 5 + (8 + 3) ∧
      level₅.tail = 5 + (8 + 3)) ∧
    level₂.opening.cells level₂.fold ≤
        2 * (level₃.shape.blocks * (level₃.shape.blockLength * (2 * 32))) ∧
      level₃.opening.cells level₃.fold ≤
        2 * (level₄.shape.blocks * (level₄.shape.blockLength * (2 * 16))) ∧
      level₄.opening.cells level₄.fold ≤
        2 * (level₅.shape.blocks * (level₅.shape.blockLength * (2 * 16))) ∧
      level₅.opening.cells level₅.fold ≤
        2 * (terminal.blocks * (terminal.blockLength * (2 * 16))) := by decide

variable {E : Type} [Field E] [Fintype E] [DecidableEq E] [Algebra Zq E] [Degree2 E]
  {Wire Coord : Type} [Fintype Coord] [DecidableEq Coord] {total : ℕ}
  {accept : Prefix Wire (Coord → E) total → Prop} {read : Reader Wire} {start : ℕ}
  {key₂ : Key level₂.shape} {openKey₂ : OpeningKey level₂.opening}
  {claim₂ : Wire → TraceClaim level₂.opening E} {block₂ : Fin level₂.shape.blocks → Coord}
  {decode₂ : E → Option (Rq level₂.shape.innerDegree)} {weight₂ : ℕ}
  {key₃ : Key level₃.shape} {openKey₃ : OpeningKey level₃.opening}
  {claim₃ : Wire → TraceClaim level₃.opening E} {block₃ : Fin level₃.shape.blocks → Coord}
  {decode₃ : E → Option (Rq level₃.shape.innerDegree)} {weight₃ : ℕ}
  {key₄ : Key level₄.shape} {openKey₄ : OpeningKey level₄.opening}
  {claim₄ : Wire → TraceClaim level₄.opening E} {block₄ : Fin level₄.shape.blocks → Coord}
  {decode₄ : E → Option (Rq level₄.shape.innerDegree)} {weight₄ : ℕ}
  {key₅ : Key level₅.shape} {openKey₅ : OpeningKey level₅.opening}
  {claim₅ : Wire → TraceClaim level₅.opening E} {block₅ : Fin level₅.shape.blocks → Coord}
  {decode₅ : E → Option (Rq level₅.shape.innerDegree)} {weight₅ : ℕ}
  {L : Terminal terminal E Wire Coord E total
    (.euclidean normTerminal.operator normTerminal.limit)}
  {commit₃ : Wire → level₃.shape.Payload} {commit₄ : Wire → level₄.shape.Payload}
  {commit₅ : Wire → level₅.shape.Payload}
  {stateOf : Wire → Fin terminal.blocks → Fin terminal.innerRank → Rq terminal.innerDegree}
  {statementOf : Wire → TerminalStatement terminal E}
  {partialsOf : Wire → Fin terminal.blocks → Rq terminal.innerDegree}

local notation "second" =>
  Tier.tensored (committedTrace (fold := level₂.fold) key₂ openKey₂ claim₂ level₂.vars
    (And.left sized₂) level₂.batch (And.left (And.right sized₂)) block₂ decode₂ read weight₂
    start) level₂.tail

local notation "third" =>
  Tier.tensored (committedTraceSq (fold := level₃.fold) key₃ openKey₃ claim₃ level₃.vars
    (And.left sized₃) level₃.batch (And.left (And.right sized₃)) block₃ decode₃ read
    weight₃ norm₃.operator norm₃.limit (Tier.finish second)) level₃.tail

local notation "fourth" =>
  Tier.tensored (committedTraceSq (fold := level₄.fold) key₄ openKey₄ claim₄ level₄.vars
    (And.left sized₄) level₄.batch (And.left (And.right sized₄)) block₄ decode₄ read
    weight₄ norm₄.operator norm₄.limit (Tier.finish third)) level₄.tail

local notation "fifth" =>
  Tier.tensored (committedTraceSq (fold := level₅.fold) key₅ openKey₅ claim₅ level₅.vars
    (And.left sized₅) level₅.batch (And.left (And.right sized₅)) block₅ decode₅ read
    weight₅ norm₅.operator norm₅.limit (Tier.finish fourth)) level₅.tail

local notation "linked" =>
  fun stage : Prefix Wire (Coord → E) total =>
    Tier.TraceLinked second third (And.left (And.right quarters)) commit₃ stage ∧
    Tier.TraceLinked third fourth (And.left (And.right (And.right quarters))) commit₄ stage ∧
    Tier.TraceLinked fourth fifth (And.left (And.right (And.right (And.right quarters)))) commit₅
      stage ∧
    Terminal.Linked fifth L (And.right (And.right (And.right (And.right quarters)))) stateOf
      stage ∧ Terminal.Passed L stage

/-- Akita Theorem 10.9 for the trace levels 2 to 5 and the terminal of the schedule `key20.aks`,
without the probability that a response does not fit (`Extract.traceTower_complete`): for each
table of the oracle whose slots decode to fold challenges, and each transcript prefix on its path
whose last message has the claim `honestTraceClaim` of the level 2 for a packed polynomial `g`
and a point with the segments of the lengths 6, 9 and 5 or more, later messages exist such that
the complete transcript has the links `Tier.TraceLinked` of the levels 2 to 5, the link
`Terminal.Linked` and the checks `Terminal.Passed`: the part of the verifier `towerLinks` of
`Key20.tower_suffix` from the level 2. The conditions on the shapes are theorems of this file.
`sends`, `wide`, `capped`, `room`, `carries₂` to `carriesL`: a wire message can have the data of
the prover, with messages of 8 field elements, the energy `3029804899` and 4 messages of
the checks. `next`, `inside`, `reads`, `clears`: the stage of the terminal and its messages.
`fit`: the fit conditions `TraceFit` of the levels 2 to 5 with `S_max` of `norm₃` to `norm₅`, and
`‖z‖_2² ≤ S_max,term` at the terminal, for each fold challenge vector of the families. -/
theorem traceTower_complete {sendable : (M : Type) → M → Prop} {slots width cap : ℕ}
    (sends : Sends sendable E width cap) (wide : 8 ≤ width) (capped : 3029804899 ≤ cap)
    (room : 3 < slots)
    (carries₂ : Carries sendable slots read fun wire => (commit₃ wire, claim₃ wire))
    (carries₃ : Carries sendable slots read fun wire => (commit₄ wire, claim₄ wire))
    (carries₄ : Carries sendable slots read fun wire => (commit₅ wire, claim₅ wire))
    (carries₅ : Carries sendable slots read fun wire =>
      (stateOf wire, statementOf wire, partialsOf wire))
    (carriesL : Carries sendable slots read (Terminal.response L))
    (next : Terminal.start L = Tier.finish fifth) (inside : Terminal.start L < total)
    (reads : ∀ point, Terminal.read L point = statementOf (point.wire (Terminal.start L)))
    (clears : ∀ point, Terminal.partials L point = partialsOf (point.wire (Terminal.start L)))
    (table : Prefix Wire (Coord → E) total → Coord → E)
    (decodes₂ : ∀ input i, (decode₂ (table input (block₂ i))).isSome)
    (decodes₃ : ∀ input i, (decode₃ (table input (block₃ i))).isSome)
    (decodes₄ : ∀ input i, (decode₄ (table input (block₄ i))).isSome)
    (decodes₅ : ∀ input i, (decode₅ (table input (block₅ i))).isSome)
    (decodesL : ∀ input i, (Terminal.decode L (table input (Terminal.block L i))).isSome)
    {first : Wire} {early : List ((Coord → E) × Wire)} (on : OnPath accept table first early)
    (starts : early.length = start) {g : ℕ → E} {block position pack : List E} {θ : E}
    (packs : pack.length = 6) (positions : position.length = 9) (blocks : 5 ≤ block.length)
    (live : ∀ cell, level₂.shape.blocks * (level₂.shape.blockLength * (2 * 32)) ≤ cell →
      g cell = 0)
    (ready : claim₂ (lastWire first early) =
      honestTraceClaim (And.left quarters) key₂ openKey₂ g block position pack θ)
    (fit : TraceFit (And.left quarters) key₂ openKey₂ level₂.fold (fun _ => decode₂)
        (fun _ => True) g fun w₂ =>
      TraceFit (And.left (And.right quarters)) key₃ openKey₃ level₃.fold (fun _ => decode₃)
        (fun z => vecNormSq z ≤ norm₃.limit) (packed w₂) fun w₃ =>
      TraceFit (And.left (And.right (And.right quarters))) key₄ openKey₄ level₄.fold
        (fun _ => decode₄) (fun z => vecNormSq z ≤ norm₄.limit) (packed w₃) fun w₄ =>
      TraceFit (And.left (And.right (And.right (And.right quarters)))) key₅ openKey₅ level₅.fold
        (fun _ => decode₅) (fun z => vecNormSq z ≤ norm₅.limit) (packed w₄) fun w₅ =>
      ∀ c : Fin terminal.blocks → Rq terminal.innerDegree,
        (∀ i, ∃ slot, Terminal.decode L slot = some (c i)) →
        vecNormSq (traceResponse c (traceSource
          (And.right (And.right (And.right (And.right quarters)))) (packed w₅ : ℕ → E))) ≤
          normTerminal.limit) :
    ∃ later, OnPath accept table first (early ++ later) ∧ (early ++ later).length = total ∧
      ∀ stage : Prefix Wire (Coord → E) total, stage.first = first →
        stage.rest = early ++ later → linked stage :=
  Extract.traceTower_complete sends (by omega) room carries₂ carries₃ carries₄ carries₅ carriesL
    quarters.1 quarters.2.1 quarters.2.2.1 quarters.2.2.2.1 quarters.2.2.2.2 source₂ source₃
    source₄ source₅ source₆ (fold₂.mono wide) (fold₃.mono wide) (fold₄.mono wide)
    (fold₅.mono wide) links.1.1 links.1.2.1 links.1.2.2.1 links.1.2.2.2 links.2.1.1 links.2.1.2.1
    links.2.1.2.2.1 links.2.1.2.2.2 links.2.2.1 links.2.2.2.1 links.2.2.2.2.1 links.2.2.2.2.2
    (norms_sized (level₃, norm₃) (by simp)).1 (norms_sized (level₄, norm₄) (by simp)).1
    (norms_sized (level₅, norm₅) (by simp)).1
    ((by decide : rangeWidthSq level₃.shape.certifiedBase ≤ 8).trans wide)
    ((by decide : rangeWidthSq level₄.shape.certifiedBase ≤ 8).trans wide)
    ((by decide : rangeWidthSq level₅.shape.certifiedBase ≤ 8).trans wide)
    ((by decide : norm₃.limit ≤ 3029804899).trans capped)
    ((by decide : norm₄.limit ≤ 3029804899).trans capped)
    ((by decide : norm₅.limit ≤ 3029804899).trans capped) next inside reads clears table decodes₂
    decodes₃ decodes₄ decodes₅ decodesL on starts packs positions blocks live ready fit

/-- `traceTower_complete` for a proof whose first message has the claim of the level 2, which
has its fold challenges at the stage 0: for each table of the oracle whose slots decode to fold
challenges, a proof exists such that its first message has the claim `honestTraceClaim` and its
complete transcript has the links of the levels 2 to 5 and of the terminal and the checks
`Terminal.Passed`. `full`: a wire message can have this claim. The other hypotheses are those of
`traceTower_complete`. -/
theorem traceTower_accepts {sendable : (M : Type) → M → Prop} {slots width cap : ℕ}
    (sends : Sends sendable E width cap) (wide : 8 ≤ width) (capped : 3029804899 ≤ cap)
    (room : 3 < slots)
    (carries₂ : Carries sendable slots read fun wire => (commit₃ wire, claim₃ wire))
    (carries₃ : Carries sendable slots read fun wire => (commit₄ wire, claim₄ wire))
    (carries₄ : Carries sendable slots read fun wire => (commit₅ wire, claim₅ wire))
    (carries₅ : Carries sendable slots read fun wire =>
      (stateOf wire, statementOf wire, partialsOf wire))
    (carriesL : Carries sendable slots read (Terminal.response L)) (origin : start = 0)
    (next : Terminal.start L = Tier.finish fifth) (inside : Terminal.start L < total)
    (reads : ∀ point, Terminal.read L point = statementOf (point.wire (Terminal.start L)))
    (clears : ∀ point, Terminal.partials L point = partialsOf (point.wire (Terminal.start L)))
    (table : Prefix Wire (Coord → E) total → Coord → E)
    (decodes₂ : ∀ input i, (decode₂ (table input (block₂ i))).isSome)
    (decodes₃ : ∀ input i, (decode₃ (table input (block₃ i))).isSome)
    (decodes₄ : ∀ input i, (decode₄ (table input (block₄ i))).isSome)
    (decodes₅ : ∀ input i, (decode₅ (table input (block₅ i))).isSome)
    (decodesL : ∀ input i, (Terminal.decode L (table input (Terminal.block L i))).isSome)
    {g : ℕ → E} {block position pack : List E} {θ : E} (packs : pack.length = 6)
    (positions : position.length = 9) (blocks : 5 ≤ block.length)
    (live : ∀ cell, level₂.shape.blocks * (level₂.shape.blockLength * (2 * 32)) ≤ cell →
      g cell = 0)
    (full : ∃ wire, claim₂ wire =
      honestTraceClaim (And.left quarters) key₂ openKey₂ g block position pack θ)
    (fit : TraceFit (And.left quarters) key₂ openKey₂ level₂.fold (fun _ => decode₂)
        (fun _ => True) g fun w₂ =>
      TraceFit (And.left (And.right quarters)) key₃ openKey₃ level₃.fold (fun _ => decode₃)
        (fun z => vecNormSq z ≤ norm₃.limit) (packed w₂) fun w₃ =>
      TraceFit (And.left (And.right (And.right quarters))) key₄ openKey₄ level₄.fold
        (fun _ => decode₄) (fun z => vecNormSq z ≤ norm₄.limit) (packed w₃) fun w₄ =>
      TraceFit (And.left (And.right (And.right (And.right quarters)))) key₅ openKey₅ level₅.fold
        (fun _ => decode₅) (fun z => vecNormSq z ≤ norm₅.limit) (packed w₄) fun w₅ =>
      ∀ c : Fin terminal.blocks → Rq terminal.innerDegree,
        (∀ i, ∃ slot, Terminal.decode L slot = some (c i)) →
        vecNormSq (traceResponse c (traceSource
          (And.right (And.right (And.right (And.right quarters)))) (packed w₅ : ℕ → E))) ≤
          normTerminal.limit) :
    ∃ proof : ℕ → Wire,
      claim₂ (proof 0) =
        honestTraceClaim (And.left quarters) key₂ openKey₂ g block position pack θ ∧
      linked ((transcript total accept).final (.done proof) table) := by
  obtain ⟨wire, said⟩ := full
  obtain ⟨later, on, whole, each⟩ := traceTower_complete sends wide capped room carries₂ carries₃
    carries₄ carries₅ carriesL next inside reads clears table decodes₂ decodes₃ decodes₄ decodes₅
    decodesL (onPath_nil accept table wire) origin.symm packs positions blocks live said fit
  obtain ⟨proof, zero, holds⟩ := OnPath.accepts (holds := linked) on whole each
  exact ⟨proof, zero ▸ said, holds⟩

end NightstreamFPrime.Spec.Compression.Akita.Key20
