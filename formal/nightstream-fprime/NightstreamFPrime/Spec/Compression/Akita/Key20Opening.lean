import NightstreamFPrime.Spec.Compression.Akita.Key20Challenge

/-! The opening of Akita for the schedule file `crates/ns-compress/artifacts/key20.aks` with the
decoders `draw` of the sparse challenge families, as one statement (Akita, ePrint 2026/1983,
revision of 2026-10-15, Theorem 10.17, Proposition 10.44, Theorem 10.46 and (206)).

- `Tower` has the data of the opening: the matrices, the readers of the messages, the
  coordinates of the fold challenges, the filters `keep₃`, `keep₄`, `keep₅` of the levels 3 to 5,
  and the terminal. The batching root is at the stage 0.
- `Tower.accepts` is the verifier `towerLinks` for this data, and `Tower.body` is its checkpoint
  structure `towerRoot`. The decoder of each level is `draw` of its family.
- `Tower.Drawn` has the hypotheses of `tower_suffix` that are not theorems for these decoders.
- `Tower.suffix` is `tower_suffix` with `towerError_le` at the slot probability `1/q²`
  (`slotChance_draw`).
- `Tower.extract_sound` and `Tower.extraction_le` are `Tier.Suffix.extract_sound` and
  `Tier.Suffix.extraction_le` for this suffix. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Key20

open scoped BigOperators
open NightstreamFPrime.Spec.Compression.Akita.Extract
open NightstreamFPrime.Spec.Checkpoint
open NightstreamFPrime.Spec.RandomOracle (OracleComp)

local instance : Fact (Nat.Prime fieldModulus) := ⟨FieldPrime.fieldModulus_natPrime⟩

attribute [local instance low] Classical.propDecidable

/-- The data of the opening of `key20.aks` with the batching root at the stage 0: the matrices
and the bases of the root, of the levels 1 to 5 and of the terminal, the readers of the wire
messages, the coordinates of the slots of the fold challenges, and the filters of the challenge
families of the levels 3 to 5 (Akita (23)). -/
structure Tower (E : Type) [Field E] [Algebra Zq E] (Wire Coord : Type) (total : ℕ) where
  read : Reader Wire
  β : ∀ g, Module.Basis (Fin (root.planes g)) Zq E
  key : (g : Fin 3) → Key (root.shape g)
  openKey : root.OpenKey
  claim : Wire → root.Claim E
  block : root.Block → Coord
  β₁ : Module.Basis (Fin level₁.opening.planes) Zq E
  key₁ : Key level₁.shape
  openKey₁ : OpeningKey level₁.opening
  claim₁ : Wire → Claim level₁.opening E
  block₁ : Fin level₁.shape.blocks → Coord
  key₂ : Key level₂.shape
  openKey₂ : OpeningKey level₂.opening
  claim₂ : Wire → TraceClaim level₂.opening E
  block₂ : Fin level₂.shape.blocks → Coord
  key₃ : Key level₃.shape
  openKey₃ : OpeningKey level₃.opening
  claim₃ : Wire → TraceClaim level₃.opening E
  block₃ : Fin level₃.shape.blocks → Coord
  keep₃ : Rq level₃.shape.innerDegree → Prop
  key₄ : Key level₄.shape
  openKey₄ : OpeningKey level₄.opening
  claim₄ : Wire → TraceClaim level₄.opening E
  block₄ : Fin level₄.shape.blocks → Coord
  keep₄ : Rq level₄.shape.innerDegree → Prop
  key₅ : Key level₅.shape
  openKey₅ : OpeningKey level₅.opening
  claim₅ : Wire → TraceClaim level₅.opening E
  block₅ : Fin level₅.shape.blocks → Coord
  keep₅ : Rq level₅.shape.innerDegree → Prop
  terminal : Terminal Key20.terminal E Wire Coord E total
    (.euclidean normTerminal.operator normTerminal.limit)
  commit : Wire → (g : Fin 3) → (root.shape g).Payload
  value : Wire → Fin 3 → E
  place : Wire → Fin 3 → List E
  commit₁ : Wire → level₁.shape.Payload
  commit₂ : Wire → level₂.shape.Payload
  commit₃ : Wire → level₃.shape.Payload
  commit₄ : Wire → level₄.shape.Payload
  commit₅ : Wire → level₅.shape.Payload
  stateOf : Wire → Fin Key20.terminal.blocks → Fin Key20.terminal.innerRank →
    Rq Key20.terminal.innerDegree

variable {E : Type} [Field E] [Fintype E] [DecidableEq E] [Algebra Zq E] [Degree2 E]
  {Wire Coord : Type} [Fintype Coord] [DecidableEq Coord] [Nonempty Coord] {total : ℕ}
  (o : Tower E Wire Coord total)

local notation "draw₁" =>
  (draw (SparseFamily level₁.opening.carrierDegree 31 10) (fun _ => True) :
    E → Option (Rq level₁.opening.carrierDegree))

local notation "draw₂" =>
  (draw (SparseFamily level₂.shape.innerDegree 31 0) (fun _ => True) :
    E → Option (Rq level₂.shape.innerDegree))

local notation "draw₃" =>
  (draw (SparseFamily level₃.shape.innerDegree 31 0) o.keep₃ :
    E → Option (Rq level₃.shape.innerDegree))

local notation "draw₄" =>
  (draw (SparseFamily level₄.shape.innerDegree 31 11) o.keep₄ :
    E → Option (Rq level₄.shape.innerDegree))

local notation "draw₅" =>
  (draw (SparseFamily level₅.shape.innerDegree 31 11) o.keep₅ :
    E → Option (Rq level₅.shape.innerDegree))

local notation "zeroth" =>
  liftBatch o.β o.key o.openKey o.claim rootVars (And.left root_sized) 8 quotientDigits root_room
    rootBatch (And.left (And.right root_sized)) o.block (rootDecode E) o.read rootMass (0 + 1)

local notation "first" =>
  Tier.tensored (liftFold o.β₁ o.key₁ o.openKey₁ o.claim₁ level₁.vars (And.left level₁_sized)
    quotientDigits (And.left (And.right level₁_sized)) level₁.batch
    (And.left (And.right (And.right level₁_sized))) o.block₁ draw₁ o.read 51
    (Tier.finish zeroth)) level₁.tail

local notation "second" =>
  Tier.tensored (committedTrace (fold := level₂.fold) o.key₂ o.openKey₂ o.claim₂ level₂.vars
    (And.left sized₂) level₂.batch (And.left (And.right sized₂)) o.block₂ draw₂ o.read 31
    (Tier.finish first)) level₂.tail

local notation "third" =>
  Tier.tensored (committedTraceSq (fold := level₃.fold) o.key₃ o.openKey₃ o.claim₃ level₃.vars
    (And.left sized₃) level₃.batch (And.left (And.right sized₃)) o.block₃ draw₃
    o.read 31 norm₃.operator norm₃.limit (Tier.finish second)) level₃.tail

local notation "fourth" =>
  Tier.tensored (committedTraceSq (fold := level₄.fold) o.key₄ o.openKey₄ o.claim₄ level₄.vars
    (And.left sized₄) level₄.batch (And.left (And.right sized₄)) o.block₄ draw₄
    o.read 53 norm₄.operator norm₄.limit (Tier.finish third)) level₄.tail

local notation "fifth" =>
  Tier.tensored (committedTraceSq (fold := level₅.fold) o.key₅ o.openKey₅ o.claim₅ level₅.vars
    (And.left sized₅) level₅.batch (And.left (And.right sized₅)) o.block₅ draw₅
    o.read 53 norm₅.operator norm₅.limit (Tier.finish fourth)) level₅.tail

/-- The verifier of the opening of `key20.aks` (Akita Figure 4 and Section 8.2): `towerLinks` for
the data `o`, with the decoders `draw` of the families of `Key20Challenge.lean`. The families are
`(31, 10)` for the groups of `W` and `V` and for the level 1, `(31, 0)` for the key group and for
the levels 2 and 3, and `(31, 11)` for the levels 4 and 5. -/
def Tower.accepts : Prefix Wire (Option Coord → E) total → Prop :=
  towerLinks (weight₃ := 31) (weight₄ := 53) (weight₅ := 53)
    Route.coefficient (Route.euclidean norm₃.operator norm₃.limit)
    (Route.euclidean norm₄.operator norm₄.limit) (Route.euclidean norm₅.operator norm₅.limit)
    zeroth first second third fourth fifth o.terminal 0 o.read o.commit o.value o.place
    (And.left quarters) (And.left (And.right quarters))
    (And.left (And.right (And.right quarters)))
    (And.left (And.right (And.right (And.right quarters))))
    (And.right (And.right (And.right (And.right quarters)))) o.commit₁ o.commit₂ o.commit₃
    o.commit₄ o.commit₅ o.stateOf

/-- The checkpoint structure of the opening of `o`: `towerRoot` under the verifier
`Tower.accepts`. A first certificate has a weak opening of the three groups of the root, and a
witness of `R_bind` is a solution of Module-SIS for a matrix of a group, of a level, or for
`A_term`. -/
noncomputable def Tower.body :=
  towerRoot (weight₃ := 31) (weight₄ := 53) (weight₅ := 53) Route.coefficient
    (Route.euclidean norm₃.operator norm₃.limit) (Route.euclidean norm₄.operator norm₄.limit)
    (Route.euclidean norm₅.operator norm₅.limit) o.accepts zeroth first second third fourth fifth
    o.terminal 32 32 16 16 16 0 o.read o.commit o.value o.place

/-- The hypotheses of `tower_suffix` on the data `o` that are not theorems for the decoders
`draw`. `injective` to `injectiveL`: the fold coordinates of each level have different slots.
`bounded₃` to `boundedL`: a member of the base family that passes the filter has
`‖c‖_mul,2 ≤ Γ` for `Γ` = 13, 18, 18 and 18 (Akita (23) and Theorem H.1). `next` and `inside`:
the fold challenge of the terminal is the stage after the stages of the level 5, and it is a
stage of the transcript. `decodeL`: the decoder of the terminal is `draw` of the family
`(31, 11)` with the filter `keepL`. -/
structure Tower.Drawn (keepL : Rq Key20.terminal.innerDegree → Prop) : Prop where
  injective : Function.Injective o.block
  injective₁ : Function.Injective o.block₁
  injective₂ : Function.Injective o.block₂
  injective₃ : Function.Injective o.block₃
  bounded₃ : ∀ c ∈ SparseFamily level₃.shape.innerDegree 31 0, o.keep₃ c →
    MulBound norm₃.operator c
  injective₄ : Function.Injective o.block₄
  bounded₄ : ∀ c ∈ SparseFamily level₄.shape.innerDegree 31 11, o.keep₄ c →
    MulBound norm₄.operator c
  injective₅ : Function.Injective o.block₅
  bounded₅ : ∀ c ∈ SparseFamily level₅.shape.innerDegree 31 11, o.keep₅ c →
    MulBound norm₅.operator c
  next : Terminal.start o.terminal = Tier.finish fifth
  inside : Terminal.start o.terminal < total
  injectiveL : Function.Injective (Terminal.block o.terminal)
  decodeL : Terminal.decode o.terminal = draw (SparseFamily Key20.terminal.innerDegree 31 11) keepL
  boundedL : ∀ c ∈ SparseFamily Key20.terminal.innerDegree 31 11, keepL c →
    MulBound normTerminal.operator c

variable {o} {keepL : Rq Key20.terminal.innerDegree → Prop}

/-- Akita Proposition 10.44 and Theorem 10.46 for the opening of `key20.aks` with the decoders
`draw`: the stages of `Tower.body` are a suffix from the stage 0 under the verifier
`Tower.accepts`, with 142 retry loops, the tree size 21449539584, and a sum of the stage errors
of at most `317735/(1024·|E|) + 49/(8·q²)` (`tower_suffix` and `towerError_le` with
`slotChance_draw`).
`slots`: an answer of the oracle has 22 slots. -/
theorem Tower.suffix (drawn : o.Drawn keepL) (slots : 22 ≤ Fintype.card Coord) :
    ∃ error, Tier.Suffix o.accepts 0 (OpensAt 0 o.commit o.value o.place o.key rootMass) o.body
        error 142 21449539584 ∧
      error ≤ 317735 / (1024 * Fintype.card E) + 49 / 8 * (1 / (fieldModulus : ℝ) ^ 2) := by
  have unitsL : UnitDifferences (Terminal.family o.terminal) := by
    unfold Terminal.family
    rw [drawn.decodeL]
    exact draw_units 5 (by decide)
  have boundedL : ∀ challenge ∈ Terminal.family o.terminal,
      MulBound normTerminal.operator challenge := by
    intro challenge decoded
    unfold Terminal.family at decoded
    rw [drawn.decodeL] at decoded
    exact drawn.boundedL _ (draw_mem decoded.choose_spec).1 (draw_mem decoded.choose_spec).2
  have chanceL : slotChance (Terminal.decode o.terminal) ≤ 1 / (fieldModulus : ℝ) ^ 2 := by
    rw [drawn.decodeL]
    exact slotChance_draw _ _
  exact ⟨_,
    tower_suffix (β := o.β) (key := o.key) (openKey := o.openKey) (claim := o.claim)
      (block := o.block) (decode := rootDecode E) (mass := rootMass) (β₁ := o.β₁)
      (key₁ := o.key₁) (openKey₁ := o.openKey₁) (claim₁ := o.claim₁) (block₁ := o.block₁)
      (decode₁ := draw₁) (weight₁ := 51) (key₂ := o.key₂) (openKey₂ := o.openKey₂)
      (claim₂ := o.claim₂) (block₂ := o.block₂) (decode₂ := draw₂) (weight₂ := 31)
      (key₃ := o.key₃) (openKey₃ := o.openKey₃) (claim₃ := o.claim₃) (block₃ := o.block₃)
      (decode₃ := draw₃) (weight₃ := 31) (key₄ := o.key₄) (openKey₄ := o.openKey₄)
      (claim₄ := o.claim₄) (block₄ := o.block₄) (decode₄ := draw₄) (weight₄ := 53)
      (key₅ := o.key₅) (openKey₅ := o.openKey₅) (claim₅ := o.claim₅) (block₅ := o.block₅)
      (decode₅ := draw₅) (weight₅ := 53) (L := o.terminal) (read := o.read) (start := 0)
      (commit := o.commit) (value := o.value) (place := o.place) (commit₁ := o.commit₁)
      (commit₂ := o.commit₂) (commit₃ := o.commit₃) (commit₄ := o.commit₄)
      (commit₅ := o.commit₅) (stateOf := o.stateOf) (peak := rootPeak) slots drawn.injective
      rootDecode_units rootDecode_sparse
      drawn.injective₁ (draw_units 5 (by decide)) ⟨2, draw_sparse⟩
      drawn.injective₂ (draw_units 6 (by decide)) ⟨1, draw_sparse⟩
      drawn.injective₃ (draw_units 6 (by decide)) ⟨1, draw_sparse⟩
      (fun _ decoded =>
        drawn.bounded₃ _ (draw_mem decoded.choose_spec).1 (draw_mem decoded.choose_spec).2)
      drawn.injective₄ (draw_units 5 (by decide)) ⟨2, draw_sparse⟩
      (fun _ decoded =>
        drawn.bounded₄ _ (draw_mem decoded.choose_spec).1 (draw_mem decoded.choose_spec).2)
      drawn.injective₅ (draw_units 5 (by decide)) ⟨2, draw_sparse⟩
      (fun _ decoded =>
        drawn.bounded₅ _ (draw_mem decoded.choose_spec).1 (draw_mem decoded.choose_spec).2)
      drawn.next drawn.inside drawn.injectiveL unitsL boundedL,
    towerError_le (β := o.β) (key := o.key) (openKey := o.openKey) (claim := o.claim)
      (block := o.block) (decode := rootDecode E) (mass := rootMass) (β₁ := o.β₁)
      (key₁ := o.key₁) (openKey₁ := o.openKey₁) (claim₁ := o.claim₁) (block₁ := o.block₁)
      (decode₁ := draw₁) (weight₁ := 51) (key₂ := o.key₂) (openKey₂ := o.openKey₂)
      (claim₂ := o.claim₂) (block₂ := o.block₂) (decode₂ := draw₂) (weight₂ := 31)
      (key₃ := o.key₃) (openKey₃ := o.openKey₃) (claim₃ := o.claim₃) (block₃ := o.block₃)
      (decode₃ := draw₃) (weight₃ := 31) (key₄ := o.key₄) (openKey₄ := o.openKey₄)
      (claim₄ := o.claim₄) (block₄ := o.block₄) (decode₄ := draw₄) (weight₄ := 53)
      (key₅ := o.key₅) (openKey₅ := o.openKey₅) (claim₅ := o.claim₅) (block₅ := o.block₅)
      (decode₅ := draw₅) (weight₅ := 53) (L := o.terminal) (read := o.read) (start := 0)
      (fun _ => slotChance_draw _ _) (slotChance_draw _ _) (slotChance_draw _ _)
      (slotChance_draw _ _) (slotChance_draw _ _) (slotChance_draw _ _) chanceL⟩

/-- An output of the extractor for the opening of `key20.aks` is correct (Akita Definition 10.15
and Proposition 10.44 (i)). The verifier `Tower.accepts` accepts the transcript of the prover. A
first certificate has a certified decommitment of the commitment of each group of the root, with
`κ̄_1 = 2ω` and `β̄_∞ = Δ^cert_f`, to a table that has the claimed value at the claimed point
(`OpensAt`). Each other output is a solution of Module-SIS for a matrix of a group, of a level,
or for `A_term`. -/
theorem Tower.extract_sound (drawn : o.Drawn keepL) (slots : 22 ≤ Fintype.card Coord)
    (prover : OracleComp (Prefix Wire (Option Coord → E) total) (Option Coord → E) (ℕ → Wire))
      (budget : ℕ)
    (table : Prefix Wire (Option Coord → E) total → Option Coord → E)
    (tape : Tape (Prefix Wire (Option Coord → E) total) (Option Coord) E budget
      o.body.protocol.stages) :
    ∀ out ∈ Checkpoint.extract o.body prover budget table tape,
      o.accepts (o.body.protocol.final prover table) ∧
        Sum.elim
          (fun certificate => OpensAt 0 o.commit o.value o.place o.key rootMass
            (o.body.protocol.input prover table 0) (o.body.witness certificate))
          o.body.breaks out := by
  obtain ⟨_, suffix, -⟩ := Tower.suffix drawn slots
  intro out result
  have sound := suffix.extract_sound rfl prover budget table tape out result
  exact ⟨(congrFun (congrArg Protocol.accept suffix.protocol) _).mp sound.1, sound.2⟩

variable [Finite Wire]

/-- Akita Theorem 10.17 in the form (161), with Corollary 10.49 and (206), for the opening of
`key20.aks` with the decoders `draw`. For each prover with at most `Q` queries to the table of
the stage answers, the probability that `Tower.accepts` accepts and the extractor returns no
output is at most `(Q + 1)·(317735/(1024·|E|) + 49/(8·q²)) + (Q + 1)·142/budget`, with `|E| = q²`
(`card_field`). The probability is over the uniform table and the tape of the extractor. The
bound has no term for the probability that the output is a solution of Module-SIS. -/
theorem Tower.extraction_le (drawn : o.Drawn keepL) (slots : 22 ≤ Fintype.card Coord)
    (prover : OracleComp (Prefix Wire (Option Coord → E) total) (Option Coord → E) (ℕ → Wire))
      {queries : ℕ}
    (bounded : prover.QueryBound queries) {budget : ℕ} (positive : 0 < budget) :
    𝔼 table, 𝔼 tape : Tape (Prefix Wire (Option Coord → E) total) (Option Coord) E budget
          o.body.protocol.stages,
        (if o.accepts (o.body.protocol.final prover table) ∧
          Checkpoint.extract o.body prover budget table tape = none then (1 : ℝ) else 0) ≤
      (queries + 1) * (317735 / (1024 * Fintype.card E) + 49 / 8 * (1 / (fieldModulus : ℝ) ^ 2)) +
        (queries + 1) * (142 : ℕ) / budget := by
  obtain ⟨_, suffix, small⟩ := Tower.suffix drawn slots
  have bound := suffix.extraction_le rfl prover bounded positive
  rw [congrArg Protocol.accept suffix.protocol] at bound
  exact bound.trans (add_le_add (mul_le_mul_of_nonneg_left small (by positivity)) le_rfl)

end NightstreamFPrime.Spec.Compression.Akita.Key20
