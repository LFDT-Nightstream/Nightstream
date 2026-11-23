import NightstreamFPrime.Spec.Compression.Akita.Extract.Groups
import NightstreamFPrime.Spec.Compression.Akita.Extract.Offload
import NightstreamFPrime.Spec.Compression.Akita.Extract.Tower

/-! The opening suffix of Akita for a schedule with an offloaded root (ePrint 2026/1983, revision
of 2026-11-23, Section 9 with (136) and Lemma 9.1, Proposition 10.44, Theorem 10.46 and
Corollary 10.49 with (206)).

The levels have the order of a row of `key30.aks`: the batching root, the offloaded root fold
with opening groups, the batching stage of the two claims of Akita (136), the level 1 with the
group of the setup prefix and the group of the witness, the packing levels 2 and 3, the trace
levels 4 to 6 and the terminal.

- `off₆` to `off₀` are the checkpoint structures of the levels over the later stages,
  `offBoundary` is the batching stage of the two claims of the level 1, and `offRoot` is the
  batching root over them.
- `offLinks` is the verifier of this opening: the links of the two batching stages, the link of
  each level and the checks of the terminal. `offAccepts` is the verifier of the compression
  argument over this opening: the conditions `Compose.Ran` and `offLinks`.
- `offTower_suffix` states that `offRoot` is a suffix from the stage of the batching root whose
  first certificates satisfy `OpensAt`: the hypothesis `later` of `Compose.suffix`. Its links are
  `root_suffix`, `Tier.suffix_push_offloaded`, `root_suffix_pinned`, `Tier.suffix_push` from the
  level 1 to the level 2 and from the level 2 to the level 3, `Tier.suffix_push_route` from each
  level to the next trace level, and `Terminal.suffix_linked`. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Extract

open scoped BigOperators
open NightstreamFPrime.Spec.Compression.Lib (mle)
open NightstreamFPrime.Spec.Checkpoint

local instance : Fact (Nat.Prime fieldModulus) := ⟨FieldPrime.fieldModulus_natPrime⟩

attribute [local instance low] Classical.propDecidable

variable {E : Type} [Field E] [Fintype E] [DecidableEq E] [Algebra Zq E] [Degree2 E]
  {count : ℕ} {B : Batch (Fin count)} {B₁ : Batch (Fin 2)} {S₂ : Shape} {O₂ : OpeningShape S₂}
  {fold₂ : Descriptor} {S₃ : Shape} {O₃ : OpeningShape S₃} {fold₃ : Descriptor} {S₄ : Shape}
  {O₄ : OpeningShape S₄} {fold₄ : Descriptor} {S₅ : Shape} {O₅ : OpeningShape S₅}
  {fold₅ : Descriptor} {S₆ : Shape} {O₆ : OpeningShape S₆} {fold₆ : Descriptor} {S₇ : Shape}
  {Wire Coord : Type} [Fintype Coord] [DecidableEq Coord] {total : ℕ}
  {β : ∀ g, Module.Basis (Fin (B.planes g)) Zq E}
  {key : (g : Fin count) → Key (B.shape g)} {openKey : B.OpenKey} {mass : Fin count → ℕ}
  {β₁ : ∀ g, Module.Basis (Fin (B₁.planes g)) Zq E} {key₁ : (g : Fin 2) → Key (B₁.shape g)}
  {openKey₁ : B₁.OpenKey} {mass₁ : Fin 2 → ℕ}
  {β₂ : Module.Basis (Fin O₂.planes) Zq E} {key₂ : Key S₂} {openKey₂ : OpeningKey O₂}
  {weight₂ : ℕ} {β₃ : Module.Basis (Fin O₃.planes) Zq E} {key₃ : Key S₃}
  {openKey₃ : OpeningKey O₃} {weight₃ : ℕ} {key₄ : Key S₄} {openKey₄ : OpeningKey O₄}
  {weight₄ : ℕ} {key₅ : Key S₅} {openKey₅ : OpeningKey O₅} {weight₅ : ℕ} {key₆ : Key S₆}
  {openKey₆ : OpeningKey O₆} {weight₆ : ℕ} {route₄ route₅ route₆ routeL : Route}
  (accept : Prefix Wire (Option Coord → E) total → Prop)
  (T₀ : Tier B.Block (fun b => Rq (B.carrierDegree b.1)) (B.Claim E)
    (B.LevelBreak key openKey fun g => 2 * (2 * mass g) * (B.fold g).sourceBound (B.shape g))
    B.Witness B.Weak Wire Coord E (Option ((List E × E) × (List E × E))) Table)
  (T₁ : Tier B₁.Block (fun b => Rq (B₁.carrierDegree b.1)) (B₁.Claim E)
    (B₁.LevelBreak key₁ openKey₁
      fun g => 2 * (2 * mass₁ g) * (B₁.fold g).sourceBound (B₁.shape g))
    B₁.Witness B₁.Weak Wire Coord E (Option (List E × E)) Table)
  (T₂ : Tier (Fin S₂.blocks) (fun _ => Rq O₂.carrierDegree) (Claim O₂ E)
    (LevelBreak key₂ openKey₂ (2 * (2 * weight₂) * fold₂.sourceBound S₂)) (Witness O₂ fold₂)
    (Weak O₂) Wire Coord E (Option (List E × E)) Table)
  (T₃ : Tier (Fin S₃.blocks) (fun _ => Rq O₃.carrierDegree) (Claim O₃ E)
    (LevelBreak key₃ openKey₃ (2 * (2 * weight₃) * fold₃.sourceBound S₃)) (Witness O₃ fold₃)
    (Weak O₃) Wire Coord E (Option (List E × E × E)) Table)
  (T₄ : Tier (Fin S₄.blocks) (fun _ => Rq S₄.innerDegree) (TraceClaim O₄ E)
    (route₄.Break key₄ openKey₄ weight₄ (fold₄.sourceBound S₄)) (Witness O₄ fold₄)
    (Weak O₄) Wire Coord E (Option (List E × E × E)) Table)
  (T₅ : Tier (Fin S₅.blocks) (fun _ => Rq S₅.innerDegree) (TraceClaim O₅ E)
    (route₅.Break key₅ openKey₅ weight₅ (fold₅.sourceBound S₅)) (Witness O₅ fold₅)
    (Weak O₅) Wire Coord E (Option (List E × E × E)) Table)
  (T₆ : Tier (Fin S₆.blocks) (fun _ => Rq S₆.innerDegree) (TraceClaim O₆ E)
    (route₆.Break key₆ openKey₆ weight₆ (fold₆.sourceBound S₆)) (Witness O₆ fold₆)
    (Weak O₆) Wire Coord E (Option (List E × E × E)) Table)
  (L : Terminal S₇ E Wire Coord E total routeL) (m₄ m₅ m₆ m₇ start : ℕ)
  (read : Reader Wire) (commit : Wire → (g : Fin count) → (B.shape g).Payload)
  (value : Wire → Fin count → E) (place : Wire → Fin count → List E)
  (commits : Wire → (g : Fin 2) → (B₁.shape g).Payload) (values : Wire → Fin 2 → E)
  (places : Wire → Fin 2 → List E) (flat : Table)

/-- The checkpoint structure of the trace level 6 over the terminal. -/
noncomputable def off₆ :=
  T₆.over accept (L.body accept)
    (fun inner => flatten (E := E) (unpackTable S₇.innerDegree m₇
      (S₇.decode (Terminal.blocksOf ((L.body accept).witness inner)))))
    (fun inner other => L.collide ((L.body accept).witness inner) ((L.body accept).witness other))

/-- The checkpoint structure of the trace level 5 over the level 6. -/
noncomputable def off₅ :=
  T₅.over accept (off₆ accept T₆ L m₇)
    (fun inner => flatten (E := E) (unpackTable S₆.innerDegree m₆
      (tableOf ((off₆ accept T₆ L m₇).witness inner))))
    (fun inner other => Sum.inl (route₆.collide key₆ openKey₆ weight₆ (fold₆.sourceBound S₆)
      ((off₆ accept T₆ L m₇).witness inner) ((off₆ accept T₆ L m₇).witness other)))

/-- The checkpoint structure of the trace level 4 over the level 5. -/
noncomputable def off₄ :=
  T₄.over accept (off₅ accept T₅ T₆ L m₆ m₇)
    (fun inner => flatten (E := E) (unpackTable S₅.innerDegree m₅
      (tableOf ((off₅ accept T₅ T₆ L m₆ m₇).witness inner))))
    (fun inner other => Sum.inl (route₅.collide key₅ openKey₅ weight₅ (fold₅.sourceBound S₅)
      ((off₅ accept T₅ T₆ L m₆ m₇).witness inner) ((off₅ accept T₅ T₆ L m₆ m₇).witness other)))

/-- The checkpoint structure of the packing level 3 over the trace level 4. -/
noncomputable def off₃ :=
  T₃.over accept (off₄ accept T₄ T₅ T₆ L m₅ m₆ m₇)
    (fun inner => flatten (E := E) (unpackTable S₄.innerDegree m₄
      (tableOf ((off₄ accept T₄ T₅ T₆ L m₅ m₆ m₇).witness inner))))
    (fun inner other => Sum.inl (route₄.collide key₄ openKey₄ weight₄ (fold₄.sourceBound S₄)
      ((off₄ accept T₄ T₅ T₆ L m₅ m₆ m₇).witness inner)
      ((off₄ accept T₄ T₅ T₆ L m₅ m₆ m₇).witness other)))

/-- The checkpoint structure of the packing level 2 over the packing level 3, which commits to
its flat witness. -/
noncomputable def off₂ :=
  T₂.over accept (off₃ accept T₃ T₄ T₅ T₆ L m₄ m₅ m₆ m₇)
    (fun inner => tableOf ((off₃ accept T₃ T₄ T₅ T₆ L m₄ m₅ m₆ m₇).witness inner))
    (fun inner other => Sum.inl (Tier.collide key₃ openKey₃
      (2 * (2 * weight₃) * fold₃.sourceBound S₃)
      ((off₃ accept T₃ T₄ T₅ T₆ L m₄ m₅ m₆ m₇).witness inner)
      ((off₃ accept T₃ T₄ T₅ T₆ L m₄ m₅ m₆ m₇).witness other)))

/-- The checkpoint structure of the level 1 with two groups over the packing level 2. -/
noncomputable def off₁ :=
  T₁.over accept (off₂ accept T₂ T₃ T₄ T₅ T₆ L m₄ m₅ m₆ m₇)
    (fun inner => tableOf ((off₂ accept T₂ T₃ T₄ T₅ T₆ L m₄ m₅ m₆ m₇).witness inner))
    (fun inner other => Sum.inl (Tier.collide key₂ openKey₂
      (2 * (2 * weight₂) * fold₂.sourceBound S₂)
      ((off₂ accept T₂ T₃ T₄ T₅ T₆ L m₄ m₅ m₆ m₇).witness inner)
      ((off₂ accept T₂ T₃ T₄ T₅ T₆ L m₄ m₅ m₆ m₇).witness other)))

/-- The checkpoint structure of the batching stage of the two claims of Akita (136), at the stage
after the checks of the root fold, over the level 1. A first certificate has a certified
decommitment of the commitment of each group, the table `flat` for the group of the setup prefix,
and the claim of each group for its table. -/
noncomputable def offBoundary :=
  (claimsRoot (Coord := Coord) (Slot := E) (total := total) T₀.finish read values
      places).segment.over accept (off₁ accept T₁ T₂ T₃ T₄ T₅ T₆ L m₄ m₅ m₆ m₇)
    (fun inner => Batch.tablesOf ((off₁ accept T₁ T₂ T₃ T₄ T₅ T₆ L m₄ m₅ m₆ m₇).witness inner))
    (fun inner other => Sum.inl (Batch.collide key₁ openKey₁
      (fun g => 2 * (2 * mass₁ g) * (B₁.fold g).sourceBound (B₁.shape g))
      ((off₁ accept T₁ T₂ T₃ T₄ T₅ T₆ L m₄ m₅ m₆ m₇).witness inner)
      ((off₁ accept T₁ T₂ T₃ T₄ T₅ T₆ L m₄ m₅ m₆ m₇).witness other)))
    fun point opening =>
      ((∃ ω, opening = some ω ∧ ∀ g, (ω.certified g).Opens (2 * mass₁ g)
        ((B₁.fold g).sourceBound (B₁.shape g)) (key₁ g) (commits (point.wire T₀.finish) g)
        ((B₁.shape g).decode (ω.certified g).source)) ∧ Batch.tablesOf opening 0 = flat) ∧
      (fun g => mle (places ((point.take T₀.finish).wire T₀.finish) g)
        (lifted E (Batch.tablesOf opening g))) = values ((point.take T₀.finish).wire T₀.finish)

/-- The checkpoint structure of the offloaded root fold over the batching stage of the level 1.
Its successor source is the table of the witness group of the level 1. -/
noncomputable def off₀ :=
  T₀.over accept (offBoundary accept T₀ T₁ T₂ T₃ T₄ T₅ T₆ L m₄ m₅ m₆ m₇ read commits values
      places flat)
    (fun inner => Batch.tablesOf ((offBoundary accept T₀ T₁ T₂ T₃ T₄ T₅ T₆ L m₄ m₅ m₆ m₇ read
      commits values places flat).witness inner) 1)
    (fun inner other => Sum.inl (Batch.groupBreak key₁ openKey₁
      (fun g => 2 * (2 * mass₁ g) * (B₁.fold g).sourceBound (B₁.shape g)) 1
      (groupCollide 1 (key₁ 1) _
        ((offBoundary accept T₀ T₁ T₂ T₃ T₄ T₅ T₆ L m₄ m₅ m₆ m₇ read commits values places
          flat).witness inner)
        ((offBoundary accept T₀ T₁ T₂ T₃ T₄ T₅ T₆ L m₄ m₅ m₆ m₇ read commits values places
          flat).witness other))))

/-- The checkpoint structure of the opening with an offloaded root: the batching root at the
stage `start` over the offloaded root fold, the level 1 with two groups, the packing levels 2 and
3, the trace levels 4 to 6 and the terminal. A witness of `R_bind` is a solution of Module-SIS
for a matrix of a group, of a level, or for `A_term`. -/
noncomputable def offRoot :=
  (claimsRoot (Coord := Coord) (Slot := E) (total := total) start read value
      place).segment.over accept
    (off₀ accept T₀ T₁ T₂ T₃ T₄ T₅ T₆ L m₄ m₅ m₆ m₇ read commits values places flat)
    (fun inner => Batch.tablesOf ((off₀ accept T₀ T₁ T₂ T₃ T₄ T₅ T₆ L m₄ m₅ m₆ m₇ read commits
      values places flat).witness inner))
    (fun inner other => Sum.inl (Batch.collide key openKey
      (fun g => 2 * (2 * mass g) * (B.fold g).sourceBound (B.shape g))
      ((off₀ accept T₀ T₁ T₂ T₃ T₄ T₅ T₆ L m₄ m₅ m₆ m₇ read commits values places flat).witness
        inner)
      ((off₀ accept T₀ T₁ T₂ T₃ T₄ T₅ T₆ L m₄ m₅ m₆ m₇ read commits values places flat).witness
        other)))
    (OpensAt start commit value place key mass)

variable {accept T₀ T₁ T₂ T₃ T₄ T₅ T₆ L m₄ m₅ m₆ m₇ start read commit value place commits values
  places flat}

variable (T₀ T₁ T₂ T₃ T₄ T₅ T₆ L start read commit value place commits values places) in
/-- The verifier of the opening with an offloaded root (Akita Figure 4, Sections 8.2 and 9). It
accepts with the stage and the link `RootLinked` of the batching root, the link `Tier.Offloaded`
from the root fold to the two claims of the level 1 with the commitment `registry` of the
verifier setup, the stage and the link `RootLinked` of the batching stage of these claims, the
link from each level to the next level or to the terminal, and the checks `Terminal.Passed` of
the terminal. -/
def offLinks (registry : (B₁.shape 0).Payload) (commit₁ : Wire → (B₁.shape 1).Payload)
    (commit₂ : Wire → S₂.Payload) (commit₃ : Wire → S₃.Payload)
    (quarter₄ : 4 * m₄ = S₄.innerDegree) (quarter₅ : 4 * m₅ = S₅.innerDegree)
    (quarter₆ : 4 * m₆ = S₆.innerDegree) (quarter₇ : 4 * m₇ = S₇.innerDegree)
    (commit₄ : Wire → S₄.Payload) (commit₅ : Wire → S₅.Payload) (commit₆ : Wire → S₆.Payload)
    (stateOf : Wire → Fin S₇.blocks → Fin S₇.innerRank → Rq S₇.innerDegree)
    (point : Prefix Wire (Option Coord → E) total) : Prop :=
  (start < point.rest.length ∧
    RootLinked T₀ start commit value place
      ((claimsRoot (Coord := Coord) (Slot := E) (total := total) start read value
        place).weights point) point) ∧
    T₀.Offloaded commits values places registry commit₁ point ∧
    (T₀.finish < point.rest.length ∧
      RootLinked T₁ T₀.finish commits values places
        ((claimsRoot (Coord := Coord) (Slot := E) (total := total) T₀.finish read values
          places).weights point) point) ∧
    T₁.Linked T₂ commit₂ point ∧ T₂.Linked T₃ commit₃ point ∧
    T₃.TraceLinked T₄ quarter₄ commit₄ point ∧ T₄.TraceLinked T₅ quarter₅ commit₅ point ∧
    T₅.TraceLinked T₆ quarter₆ commit₆ point ∧ Terminal.Linked T₆ L quarter₇ stateOf point ∧
    L.Passed point

/-- The batching root over an offloaded root fold, the level 1 with the group of the setup prefix
and the group of the witness, two packing levels, three trace levels and the terminal is a suffix
from the stage `start` (Akita Proposition 10.44, Theorem 10.46 and Lemma 9.1). The stage errors
are `1/|E|` for the weights of each batching stage, `Σ_b p_b + Σ_j ε_j` for each level and
`B_term·p_term` for the terminal. `valid`, `packed`, `pinned`, `traced`, `uniform` and `passes`
are the conditions of the step lemmas on each level: the method of the level 1 is `Batch.method`
with the condition `Batch.Pinned` of its group 0 for the commitment `registry` and the table
`flat`, and a final claim of the root fold is true for a table `w` when `v_W = w̃(r_2)` and
`s_{ρ_S}` is the value of `flat` at `ρ_S`. `links`: the verifier accepts only with `offLinks`. -/
theorem offTower_suffix {blocks blocks₁ : ℕ} (card : Fintype.card B.Block = blocks)
    (card₁ : Fintype.card B₁.Block = blocks₁)
    (valid₀ : T₀.Valid) (valid₁ : T₁.Valid) (valid₂ : T₂.Valid) (valid₃ : T₃.Valid)
    (valid₄ : T₄.Valid) (valid₅ : T₅.Valid) (valid₆ : T₆.Valid)
    (fair : ∃ decode : (Coord → E) → Fin count → E, Fair decode)
    (fair₁ : ∃ decode : (Coord → E) → Fin 2 → E, Fair decode)
    (packed₀ : T₀.method = Batch.method β key openKey mass)
    {registry : (B₁.shape 0).Payload}
    {fix : B₁.Weak → B₁.Claim E → B₁.LevelBreak key₁ openKey₁
      fun g => 2 * (2 * mass₁ g) * (B₁.fold g).sourceBound (B₁.shape g)}
    (pinned₁ : T₁.method =
      (Batch.method β₁ key₁ openKey₁ mass₁).pinned (Batch.Pinned 0 registry flat) fix)
    (packed₂ : T₂.method = packing β₂ key₂ openKey₂ weight₂)
    (packed₃ : T₃.method = packing β₃ key₃ openKey₃ weight₃)
    (traced₄ : T₄.method = traceOn key₄ openKey₄ weight₄ route₄)
    (traced₅ : T₅.method = traceOn key₅ openKey₅ weight₅ route₅)
    (traced₆ : T₆.method = traceOn key₆ openKey₆ weight₆ route₆)
    (first : T₀.start = start + 1) (next₁ : T₁.start = T₀.finish + 1)
    (next₂ : T₂.start = T₁.finish) (next₃ : T₃.start = T₂.finish) (next₄ : T₄.start = T₃.finish)
    (next₅ : T₅.start = T₄.finish) (next₆ : T₆.start = T₅.finish) (next₇ : L.start = T₆.finish)
    (fits : L.start < total)
    (quarter₄ : 4 * m₄ = S₄.innerDegree) (quarter₅ : 4 * m₅ = S₅.innerDegree)
    (quarter₆ : 4 * m₆ = S₆.innerDegree) (quarter₇ : 4 * m₇ = S₇.innerDegree)
    (uniform₀ : ∀ wire challenge table other,
      bare (T₀.check wire challenge table) = bare (T₀.check wire challenge other))
    (passes₀ : ∀ (table : Table) r2 v ρ s, v = mle r2 (lifted E table) →
      s = mle ρ (lifted E flat) → T₀.passes table (some ((r2, v), (ρ, s))))
    (uniform₁ : ∀ wire challenge table other,
      bare (T₁.check wire challenge table) = bare (T₁.check wire challenge other))
    (passes₁ : ∀ (table : Table) r2 v, v = mle r2 (lifted E table) →
      T₁.passes table (some (r2, v)))
    (uniform₂ : ∀ wire challenge table other,
      bare (T₂.check wire challenge table) = bare (T₂.check wire challenge other))
    (passes₂ : ∀ (table : Table) r2 v, v = mle r2 (lifted E table) →
      T₂.passes table (some (r2, v)))
    (uniform₃ : ∀ wire challenge table other,
      bare (T₃.check wire challenge table) = bare (T₃.check wire challenge other))
    (passes₃ : ∀ (table : Table) ρ θ v, v = θ * mle ρ (packed table) →
      T₃.passes table (some (ρ, θ, v)))
    (uniform₄ : ∀ wire challenge table other,
      bare (T₄.check wire challenge table) = bare (T₄.check wire challenge other))
    (passes₄ : ∀ (table : Table) ρ θ v, v = θ * mle ρ (packed table) →
      T₄.passes table (some (ρ, θ, v)))
    (uniform₅ : ∀ wire challenge table other,
      bare (T₅.check wire challenge table) = bare (T₅.check wire challenge other))
    (passes₅ : ∀ (table : Table) ρ θ v, v = θ * mle ρ (packed table) →
      T₅.passes table (some (ρ, θ, v)))
    (uniform₆ : ∀ wire challenge table other,
      bare (T₆.check wire challenge table) = bare (T₆.check wire challenge other))
    (passes₆ : ∀ (table : Table) ρ θ v, v = θ * mle ρ (packed table) →
      T₆.passes table (some (ρ, θ, v)))
    (injective : Function.Injective L.block) (units : UnitDifferences L.family)
    (challenges : routeL.Challenges L.weight L.family)
    (commit₁ : Wire → (B₁.shape 1).Payload) (commit₂ : Wire → S₂.Payload)
    (commit₃ : Wire → S₃.Payload) (commit₄ : Wire → S₄.Payload) (commit₅ : Wire → S₅.Payload)
    (commit₆ : Wire → S₆.Payload)
    (stateOf : Wire → Fin S₇.blocks → Fin S₇.innerRank → Rq S₇.innerDegree)
    (links : ∀ point, accept point →
      offLinks T₀ T₁ T₂ T₃ T₄ T₅ T₆ L start read commit value place commits values places registry
        commit₁ commit₂ commit₃ quarter₄ quarter₅ quarter₆ quarter₇ commit₄ commit₅ commit₆
        stateOf point) :
    Tier.Suffix accept start (OpensAt start commit value place key mass)
      (offRoot (route₄ := route₄) (route₅ := route₅) (route₆ := route₆) accept T₀ T₁ T₂ T₃ T₄ T₅
        T₆ L m₄ m₅ m₆ m₇ start read commit value place commits values places flat)
      (1 / Fintype.card E / 2 ^ Root.rootBits count + (T₀.levelError +
        (1 / Fintype.card E / 2 ^ Root.rootBits 2 +
        (T₁.levelError + (T₂.levelError +
          (T₃.levelError + (T₄.levelError +
            (T₅.levelError + (T₆.levelError +
              L.termError)))))))))
      (1 + (blocks + 1 + (1 + (blocks₁ + 1 + (S₂.blocks + 1 + (S₃.blocks + 1 + (S₄.blocks + 1 +
        (S₅.blocks + 1 + (S₆.blocks + 1 + S₇.blocks)))))))))
      (2 * (2 * (blocks + 1) * (2 * (2 * (blocks₁ + 1) * (2 * (S₂.blocks + 1) *
        (2 * (S₃.blocks + 1) * (2 * (S₄.blocks + 1) * (2 * (S₅.blocks + 1) *
          (2 * (S₆.blocks + 1) * (S₇.blocks + 1)))))))))) := by
  have ends₀ := T₀.finish_eq
  have ends₁ := T₁.finish_eq
  have ends₂ := T₂.finish_eq
  have ends₃ := T₃.finish_eq
  have ends₄ := T₄.finish_eq
  have ends₅ := T₅.finish_eq
  have ends₆ := T₆.finish_eq
  have level₆ := Terminal.suffix_linked valid₆ ⟨injective, fits, fun point accepted =>
    (links point accepted).2.2.2.2.2.2.2.2.2⟩ next₇ units challenges quarter₇ stateOf uniform₆
    passes₆ fun point accepted => (links point accepted).2.2.2.2.2.2.2.2.1
  have level₅ := Tier.suffix_push_route valid₅ traced₆ next₆ quarter₆ commit₆ (by omega)
    uniform₅ passes₅ (fun point accepted => (links point accepted).2.2.2.2.2.2.2.1) Sum.inl
    (fun _ found => found) level₆
  have level₄ := Tier.suffix_push_route valid₄ traced₅ next₅ quarter₅ commit₅ (by omega)
    uniform₄ passes₄ (fun point accepted => (links point accepted).2.2.2.2.2.2.1) Sum.inl
    (fun _ found => found) level₅
  have level₃ := Tier.suffix_push_route valid₃ traced₄ next₄ quarter₄ commit₄ (by omega)
    uniform₃ passes₃ (fun point accepted => (links point accepted).2.2.2.2.2.1) Sum.inl
    (fun _ found => found) level₄
  have level₂ := Tier.suffix_push (Fintype.card_fin _) valid₂ packed₃ next₃ commit₃ (by omega)
    uniform₂ passes₂ (fun point accepted => (links point accepted).2.2.2.2.1) Sum.inl
    (fun solution found => by
      show T₃.method.broken solution
      rw [packed₃]
      exact found)
    level₃
  have level₁ := Tier.suffix_push card₁ valid₁ packed₂ next₂ commit₂ (by omega) uniform₁ passes₁
    (fun point accepted => (links point accepted).2.2.2.1) Sum.inl
    (fun solution found => by
      show T₂.method.broken solution
      rw [packed₂]
      exact found)
    level₂
  have boundary := root_suffix_pinned (β := β₁) T₀.finish read values places fair₁ next₁
    (by omega) commits (fun opening => Batch.tablesOf opening 0 = flat)
    (fun point opening accepted opens => by
      obtain ⟨ω, rfl, opens⟩ := opens
      rw [pinned₁] at opens
      obtain ⟨_, _, _, _, _, _, _, _, _, registered, _⟩ := (links point accepted).2.1
      have payload := congrFun (links point accepted).2.2.1.2.1 0
      exact ⟨⟨ω, rfl, opens.1⟩, opens.2 (payload.trans registered)⟩)
    (fun point accepted => (links point accepted).2.2.1) Sum.inl
    (fun solution found => by
      show T₁.method.broken solution
      rw [pinned₁]
      exact found)
    level₁
  have level₀ := Tier.suffix_push_offloaded card valid₀ (by omega) flat uniform₀ passes₀ commits
    values places registry commit₁ (fun point accepted => (links point accepted).2.1) key₁ mass₁
    (fun solution => Sum.inl (Batch.groupBreak key₁ openKey₁
      (fun g => 2 * (2 * mass₁ g) * (B₁.fold g).sourceBound (B₁.shape g)) 1 solution))
    (fun solution found => by
      show T₁.method.broken (Batch.groupBreak key₁ openKey₁ _ 1 solution)
      rw [pinned₁]
      show (Option.map _ solution).isSome
      rwa [Option.isSome_map])
    boundary
  exact root_suffix start read value place fair packed₀ first (by omega) commit
    (fun point accepted => (links point accepted).1) Sum.inl
    (fun solution found => by
      show T₀.method.broken solution
      rw [packed₀]
      exact found)
    level₀

section Composed

variable {Lead : Type} {P : Verifier.Params Zq E} {B : Batch (Fin 3)}
  {key : (g : Fin 3) → Key (B.shape g)} {openKey : B.OpenKey} {mass : Fin 3 → ℕ}
  (C : Compose P B Wire Coord Lead total) (vf : Verifier.VerifierData P)
  (T₀ : Tier B.Block (fun b => Rq (B.carrierDegree b.1)) (B.Claim E)
    (B.LevelBreak key openKey fun g => 2 * (2 * mass g) * (B.fold g).sourceBound (B.shape g))
    B.Witness B.Weak Wire Coord E (Option ((List E × E) × (List E × E))) Table)
  (commit : Wire → (g : Fin 3) → (B.shape g).Payload) (value : Wire → Fin 3 → E)
  (place : Wire → Fin 3 → List E)

variable (T₁ T₂ T₃ T₄ T₅ T₆ L commits values places) in
/-- The verifier of the compression argument over the opening with an offloaded root (SPEC.md
Algorithm 10). It accepts with the conditions `Compose.Ran` of the stages before the opening and
the conditions `offLinks` of the opening with the batching root at the stage `Compose.root`. -/
def offAccepts (registry : (B₁.shape 0).Payload) (commit₁ : Wire → (B₁.shape 1).Payload)
    (commit₂ : Wire → S₂.Payload) (commit₃ : Wire → S₃.Payload)
    (quarter₄ : 4 * m₄ = S₄.innerDegree) (quarter₅ : 4 * m₅ = S₅.innerDegree)
    (quarter₆ : 4 * m₆ = S₆.innerDegree) (quarter₇ : 4 * m₇ = S₇.innerDegree)
    (commit₄ : Wire → S₄.Payload) (commit₅ : Wire → S₅.Payload) (commit₆ : Wire → S₆.Payload)
    (stateOf : Wire → Fin S₇.blocks → Fin S₇.innerRank → Rq S₇.innerDegree)
    (point : Prefix Wire (Option Coord → E) total) : Prop :=
  C.Ran vf commit value place point ∧
    offLinks T₀ T₁ T₂ T₃ T₄ T₅ T₆ L (C.root vf) C.read commit value place commits values places
      registry commit₁ commit₂ commit₃ quarter₄ quarter₅ quarter₆ quarter₇ commit₄ commit₅ commit₆
      stateOf point

end Composed

end NightstreamFPrime.Spec.Compression.Akita.Extract
