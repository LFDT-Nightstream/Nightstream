import NightstreamFPrime.Spec.Compression.Akita.Key20Opening
import NightstreamFPrime.Spec.Compression.Akita.Key30Tower
import NightstreamFPrime.Spec.Compression.Akita.Extract.SetupFactor

/-! The opening of Akita for a row of the schedule file `crates/ns-compress/artifacts/key30.aks`
with the fold decoders of the pinned Akita code (revision `d50f700`,
`crates/akita-challenges/src/sampler/stream.rs`) on the slots `Replay.CodeSlot E`, as one
statement (Akita, ePrint 2026/1983, revision of 2026-11-23, Section 3.2, Section 9 with Lemma 9.1,
Theorem 10.17, Proposition 10.44, Theorem 10.46 and (206)).

The Akita code (revision `d50f700f`, `crates/akita-challenges/src/config.rs`) has the family
`(31, 10)` for the degree 64 and `(31, 0)` for the degree 128. The levels with the Euclidean
route have the filter `accept_strict_parts` of the code (`TrigTable.codeAccepts`) on the base
family `(31, 11)` for the degree 64.

- `mass`, `rootDecode`, `mass₁` and `decode₁` are the norms `ω = 31` and the decoders of the
  groups of the root and of the level 1: each group has the degree 128 and the family `(31, 0)`,
  and its decoder is `Replay.CodeSlot.open Key20.open128`.
- `Tower` has the data of the opening of a row: the matrices, the readers of the messages, the
  coordinates of the fold challenges, the setup prefix `flat`, the commitment `registry` of the
  setup prefix with its reference decommitment, and the terminal.
- `Tower.Shared` states that the matrices `A` and `B` of each group of the root and the matrix
  `D` of the root are the views (Akita (35)) of the setup prefix `flat`. `Tower.scan` is the
  split `SetupScan` of the public weight of the root (Akita (132)) that it gives, in the form of
  the pinned Akita code (`Batch.codeScan`, `root_blocked`). The definitions and the theorems have the hypothesis `shared`.
  `Tower.onSetup` puts these views in the data, and `Tower.onSetup_shared` states `Tower.Shared`
  for it.
- `Tower.links` is the verifier `offLinks` for this data with the batching root at the stage
  `start`, and `Tower.body` is its checkpoint structure `offRoot`. The decoder of each level is
  the decoder of the code of its family: `Replay.CodeSlot.open` of `(31, 0)` for the root and for
  the levels 1 and 4 and of `(31, 10)` for the levels 2 and 3, and `Replay.CodeSlot.kept64` of
  the filtered family `(64, 31, 11)` for the levels 5 and 6 and the terminal (`families_code`).
- `Tower.Drawn` has the hypotheses of `tower_suffix` that are not theorems for these decoders,
  and the count `Key20.Kept` of the filtered families. `Tower.Drawn.decodes`: each slot gives a
  fold challenge at each level.
- `Tower.suffix` is `tower_suffix` with `towerError_le` at the slot probability `1/q²`
  (`Replay.slotChance_open_field`, `Replay.slotChance_kept64_field`).
- `Tower.accepts`, `Tower.extract_sound` and `Tower.extraction_le` are the verifier with the
  batching root at the stage 0, and `Tier.Suffix.extract_sound` and `Tier.Suffix.extraction_le`
  for its suffix. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Key30

open scoped BigOperators
open NightstreamFPrime.Spec.Compression.Akita.Extract
open NightstreamFPrime.Spec.Checkpoint
open NightstreamFPrime.Spec.Compression.Lib (mle cubeSum)
open NightstreamFPrime.Spec.RandomOracle (OracleComp)
open Key20 (units_of_mem sparse_of_mem open64 open128 open64_fits open128_fits open64_bits
  open128_bits)

local instance : Fact (Nat.Prime fieldModulus) := ⟨FieldPrime.fieldModulus_natPrime⟩

attribute [local instance low] Classical.propDecidable

/-- `ω` of the family of each group of the root. -/
def mass (_ : Fin 3) : ℕ := 31

/-- `ω` of the family of each group of the level 1. -/
def mass₁ (_ : Fin 2) : ℕ := 31

/-- The decoder of the code of the fold challenges of each group of the root of the row `r`. -/
noncomputable def rootDecode (E : Type) (r : Row) (g : Fin 3) :
    Replay.CodeSlot E → Option (Rq ((root r).carrierDegree g)) :=
  Replay.CodeSlot.open open128 open128_fits

/-- The decoder of the code of the fold challenges of each group of the level 1 of the row `r`. -/
noncomputable def decode₁ (E : Type) (r : Row) (g : Fin 2) :
    Replay.CodeSlot E → Option (Rq ((level₁ r).carrierDegree g)) :=
  Replay.CodeSlot.open open128 open128_fits

/-- Each challenge of `rootDecode` is in the family `(31, 0)` of the degree 128. -/
theorem rootDecode_mem {E : Type} {r : Row} {g : Fin 3} {slot : Replay.CodeSlot E}
    {challenge : Rq ((root r).carrierDegree g)} (decoded : rootDecode E r g slot = some challenge) :
    challenge ∈ SparseFamily ((root r).carrierDegree g) 31 0 := by
  unfold rootDecode at decoded
  exact (Replay.open_mem decoded).1

/-- Each challenge of `decode₁` is in the family `(31, 0)` of the degree 128. -/
theorem decode₁_mem {E : Type} {r : Row} {g : Fin 2} {slot : Replay.CodeSlot E}
    {challenge : Rq ((level₁ r).carrierDegree g)} (decoded : decode₁ E r g slot = some challenge) :
    challenge ∈ SparseFamily ((level₁ r).carrierDegree g) 31 0 := by
  unfold decode₁ at decoded
  exact (Replay.open_mem decoded).1

/-- The conditions `Batch.Blocked` of the root for the block of `2^7 = 128` cells, which is
`relation_coefficient_block_len` of the pinned Akita code for the root of `key30.aks`. -/
theorem root_blocked : ∀ r ∈ rows, (root r).Blocked 7 := by decide

/-- The split of the public weight of the root of the row `r` for the setup prefix `flat` (Akita
(132)): a split `SetupScan` for the reduction of the statement of the claim of each wire and each
fold challenge. -/
def RootScan {r : Row} (member : r ∈ rows) {E : Type} [Field E] [Fintype E] [DecidableEq E]
    [Algebra Zq E] (β : ∀ g, Module.Basis (Fin ((root r).planes g)) Zq E)
    (key : (g : Fin 3) → Key ((root r).shape g)) (openKey : (root r).OpenKey) {Wire : Type}
    (claim : Wire → (root r).Claim E) (flat : Table) : Type :=
  ∀ (wire : Wire) (challenge : (root r).Challenge),
    SetupScan (Batch.liftedReduction β ((claim wire).statement key openKey) challenge rootVars
      (root_sized r member).1 8 quotientDigits₀ (root_sized r member).2.1 rootBatch
      (root_sized r member).2.2.1) setupVars flat

/-- The data of the opening of the row `r` of `key30.aks`: the matrices and the bases of the
root, of the levels 1 to 6 and of the terminal, the readers of the wire messages, the coordinates
of the slots of the fold challenges, the setup prefix `flat`, the commitment `registry` of the
setup prefix with its reference decommitment `reference` (Akita (162)), and the filters of the
challenge families of the levels 5 and 6 (Akita (23)). -/
structure Tower (r : Row) (member : r ∈ rows) (E : Type) [Field E] [Fintype E] [DecidableEq E]
    [Algebra Zq E] (Wire Coord : Type) (total : ℕ) where
  β : ∀ g, Module.Basis (Fin ((root r).planes g)) Zq E
  key : (g : Fin 3) → Key ((root r).shape g)
  openKey : (root r).OpenKey
  claim : Wire → (root r).Claim E
  block : (root r).Block → Coord
  flat : Table
  β₁ : ∀ g, Module.Basis (Fin ((level₁ r).planes g)) Zq E
  key₁ : (g : Fin 2) → Key ((level₁ r).shape g)
  openKey₁ : (level₁ r).OpenKey
  claim₁ : Wire → (level₁ r).Claim E
  block₁ : (level₁ r).Block → Coord
  registry : ((level₁ r).shape 0).Payload
  reference : Decommitment ((level₁ r).shape 0)
  β₂ : Module.Basis (Fin (level₂ r).opening.planes) Zq E
  key₂ : Key (level₂ r).shape
  openKey₂ : OpeningKey (level₂ r).opening
  claim₂ : Wire → Claim (level₂ r).opening E
  block₂ : Fin (level₂ r).shape.blocks → Coord
  β₃ : Module.Basis (Fin (level₃ r).opening.planes) Zq E
  key₃ : Key (level₃ r).shape
  openKey₃ : OpeningKey (level₃ r).opening
  claim₃ : Wire → Claim (level₃ r).opening E
  block₃ : Fin (level₃ r).shape.blocks → Coord
  key₄ : Key level₄.shape
  openKey₄ : OpeningKey level₄.opening
  claim₄ : Wire → TraceClaim level₄.opening E
  block₄ : Fin level₄.shape.blocks → Coord
  key₅ : Key level₅.shape
  openKey₅ : OpeningKey level₅.opening
  claim₅ : Wire → TraceClaim level₅.opening E
  block₅ : Fin level₅.shape.blocks → Coord
  key₆ : Key level₆.shape
  openKey₆ : OpeningKey level₆.opening
  claim₆ : Wire → TraceClaim level₆.opening E
  block₆ : Fin level₆.shape.blocks → Coord
  terminal : Terminal Key30.terminal E Wire Coord (Replay.CodeSlot E) total
    (.euclidean normTerminal.operator normTerminal.limit)
  commit : Wire → (g : Fin 3) → ((root r).shape g).Payload
  value : Wire → Fin 3 → E
  place : Wire → Fin 3 → List E
  commits : Wire → (g : Fin 2) → ((level₁ r).shape g).Payload
  values : Wire → Fin 2 → E
  places : Wire → Fin 2 → List E
  commit₁ : Wire → ((level₁ r).shape 1).Payload
  commit₂ : Wire → (level₂ r).shape.Payload
  commit₃ : Wire → (level₃ r).shape.Payload
  commit₄ : Wire → level₄.shape.Payload
  commit₅ : Wire → level₅.shape.Payload
  commit₆ : Wire → level₆.shape.Payload
  stateOf : Wire → Fin Key30.terminal.blocks → Fin Key30.terminal.innerRank →
    Rq Key30.terminal.innerDegree

variable {E : Type} [Field E] [Fintype E] [DecidableEq E] [Algebra Zq E] [Degree2 E]
  {Wire Coord : Type} [Fintype Coord] [DecidableEq Coord] [Nonempty Coord] {total : ℕ}
  {r : Row} {member : r ∈ rows} (o : Tower r member E Wire Coord total)

/-- The matrices of the root of the data `o` are the views of the shared setup vector with the
cells of the setup prefix `flat` (Akita Section 4.3 and (35)): the matrices `A` and `B` of each
group and the matrix `D` of the shared path, in the orders of the witness layout. -/
def Tower.Shared : Prop :=
  (∀ g, (o.key g).inner = Setup.shared (fun p : Fin (2 ^ setupVars) => o.flat p)
      ((root r).shape g).innerDegree ((root r).shape g).innerRank _
      ((root r).shape g).innerSetupCode ∧
    (o.key g).outer = Setup.shared (fun p : Fin (2 ^ setupVars) => o.flat p)
      ((root r).shape g).outerDegree ((root r).shape g).outerRank _
      ((root r).shape g).outerSetupCode) ∧
  o.openKey.matrix = Setup.shared (fun p : Fin (2 ^ setupVars) => o.flat p) (root r).openDegree
    (root r).openRank _ (root r).openSetupCode

/-- The data `o` with the matrices `A` and `B` of each group of the root and the matrix `D` of
the root replaced by the views of the setup prefix `flat`. The compression maps stay. -/
noncomputable def Tower.onSetup : Tower r member E Wire Coord total :=
  { o with
    key := fun g => ⟨Setup.shared (fun p : Fin (2 ^ setupVars) => o.flat p)
      ((root r).shape g).innerDegree ((root r).shape g).innerRank _
      ((root r).shape g).innerSetupCode, Setup.shared (fun p : Fin (2 ^ setupVars) => o.flat p)
      ((root r).shape g).outerDegree ((root r).shape g).outerRank _
      ((root r).shape g).outerSetupCode, (o.key g).chain⟩
    openKey := ⟨Setup.shared (fun p : Fin (2 ^ setupVars) => o.flat p) (root r).openDegree
      (root r).openRank _ (root r).openSetupCode, o.openKey.chain⟩ }

omit [Degree2 E] [Fintype Coord] [DecidableEq Coord] [Nonempty Coord] in
/-- The data `Tower.onSetup` satisfies `Tower.Shared`. -/
theorem Tower.onSetup_shared : o.onSetup.Shared :=
  ⟨fun _ => ⟨rfl, rfl⟩, rfl⟩

variable {o} in
/-- The split `SetupScan` of the public weight of the root for the data `o` with `Tower.Shared`,
in the form of the pinned Akita code: `Batch.codeScan` for the statement of the claim of each
wire, with the block of `2^7` cells. -/
noncomputable def Tower.scan (shared : o.Shared) :
    RootScan member o.β o.key o.openKey o.claim o.flat :=
  fun wire challenge =>
    Batch.Statement.over_eq ((o.claim wire).statement o.key o.openKey)
        (fun p : Fin (2 ^ setupVars) => o.flat p) (fun g => (shared.1 g).1)
        (fun g => (shared.1 g).2) shared.2 ▸
      Batch.codeScan o.β ((o.claim wire).statement o.key o.openKey) challenge rootVars
        (root_sized r member).1 8 quotientDigits₀ (root_sized r member).2.1 rootBatch
        (root_sized r member).2.2.1 setupVars o.flat 7 (root_blocked r member) (by decide)

variable (shared : o.Shared) (read : Reader Wire) (start : ℕ)

local notation "code₂" =>
  (Replay.CodeSlot.open open64 open64_fits :
    Replay.CodeSlot E → Option (Rq (OpeningShape.carrierDegree (Key20.Level.opening (level₂ r)))))

local notation "code₃" =>
  (Replay.CodeSlot.open open64 open64_fits :
    Replay.CodeSlot E → Option (Rq (OpeningShape.carrierDegree (Key20.Level.opening (level₃ r)))))

local notation "code₄" =>
  (Replay.CodeSlot.open open128 open128_fits :
    Replay.CodeSlot E → Option (Rq level₄.shape.innerDegree))

local notation "code₅" =>
  (Replay.CodeSlot.kept64 : Replay.CodeSlot E → Option (Rq level₅.shape.innerDegree))

local notation "code₆" =>
  (Replay.CodeSlot.kept64 : Replay.CodeSlot E → Option (Rq level₆.shape.innerDegree))

/-- The pair `(n₁, n₂)` of the challenge family of each group of each level of a row of
`key30.aks`, and then of the terminal: the fields `count_pm1` and `count_pm2` of the file. -/
def families : List (List (ℕ × ℕ)) :=
  [[(31, 0), (31, 0), (31, 0)], [(31, 0), (31, 0)], [(31, 10)], [(31, 10)], [(31, 0)],
    [(31, 11)], [(31, 11)], [(31, 11)]]

/-- The pair of `families` for a group of a level. -/
def family (level group : ℕ) : ℕ × ℕ := (families.getD level []).getD group (0, 0)

/-- The decoders of `Tower.accepts` are the decoders of the code for the families of `families`:
the groups of the root and of the levels 1 to 4 have no filter, and the levels 5 and 6 and the
terminal have the filtered family `(64, 31, 11)` of the code. -/
theorem families_code :
    (∀ g : Fin 3, (open128.pm1, open128.pm2) = family 0 g) ∧
    (∀ g : Fin 2, (open128.pm1, open128.pm2) = family 1 g) ∧
    (open64.pm1, open64.pm2) = family 2 0 ∧ (open64.pm1, open64.pm2) = family 3 0 ∧
    (open128.pm1, open128.pm2) = family 4 0 ∧
    (Replay.filtered64.pm1, Replay.filtered64.pm2) = family 5 0 ∧
    (Replay.filtered64.pm1, Replay.filtered64.pm2) = family 6 0 ∧
    (Replay.filtered64.pm1, Replay.filtered64.pm2) = family 7 0 :=
  ⟨fun g => by fin_cases g <;> rfl, fun g => by fin_cases g <;> rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩

local notation "zeroth" =>
  offBatch o.β o.key o.openKey o.claim rootVars (And.left (root_sized r member)) 8 quotientDigits₀
    (And.left (And.right (root_sized r member))) rootBatch
    (And.left (And.right (And.right (root_sized r member)))) o.block (rootDecode E r) read mass
    (start + 1) setupVars o.flat (Tower.scan shared)

local notation "first" =>
  Tier.pinned (liftBatch o.β₁ o.key₁ o.openKey₁ o.claim₁ vars₁ (And.left (level₁_sized r member))
    16 quotientDigits (And.left (And.right (level₁_sized r member))) batch₁
    (And.left (And.right (And.right (level₁_sized r member)))) o.block₁ (decode₁ E r) read mass₁
    (Tier.finish zeroth + 1)) (Batch.Pinned 0 o.registry o.flat)
    (fun ω _ => Batch.pinBreak o.key₁ o.openKey₁ _ 0 o.reference ω)

local notation "second" =>
  liftFold o.β₂ o.key₂ o.openKey₂ o.claim₂ (Key20.Level.vars (level₂ r))
    (And.left (level₂_sized r member)) quotientDigits
    (And.left (And.right (level₂_sized r member))) (Key20.Level.batch (level₂ r))
    (And.left (And.right (And.right (level₂_sized r member)))) o.block₂ code₂ read 51
    (Tier.finish first)

local notation "third" =>
  Tier.tensored (liftFold o.β₃ o.key₃ o.openKey₃ o.claim₃ (Key20.Level.vars (level₃ r)) (And.left (level₃_sized r member))
    quotientDigits (And.left (And.right (level₃_sized r member))) (Key20.Level.batch (level₃ r))
    (And.left (And.right (And.right (level₃_sized r member)))) o.block₃ code₃ read 51
    (Tier.finish second)) (Key20.Level.tail (level₃ r))

local notation "fourth" =>
  Tier.tensored (committedTrace (fold := level₄.fold) o.key₄ o.openKey₄ o.claim₄ level₄.vars
    (And.left level₄_sized) level₄.batch (And.left (And.right level₄_sized)) o.block₄ code₄ read
    31 (Tier.finish third)) level₄.tail

local notation "fifth" =>
  Tier.tensored (committedTraceSq (fold := level₅.fold) o.key₅ o.openKey₅ o.claim₅ level₅.vars
    (And.left (tails_sized level₅ (List.mem_cons_self ..))) level₅.batch
    (And.left (And.right (tails_sized level₅ (List.mem_cons_self ..)))) o.block₅ code₅
    read 53 (Key20.Squared.operator (norm₅ r)) (Key20.Squared.limit (norm₅ r)) (Tier.finish fourth)) level₅.tail

local notation "sixth" =>
  Tier.tensored (committedTraceSq (fold := level₆.fold) o.key₆ o.openKey₆ o.claim₆ level₆.vars
    (And.left (tails_sized level₆ (List.mem_cons_of_mem _ (List.mem_cons_self ..)))) level₆.batch
    (And.left (And.right (tails_sized level₆ (List.mem_cons_of_mem _ (List.mem_cons_self ..)))))
    o.block₆ code₆ read 53 norm₆.operator norm₆.limit (Tier.finish fifth))
    level₆.tail

/-- The verifier of the opening of a row of `key30.aks` with the batching root at the stage
`start` (Akita Figure 4, Sections 8.2 and 9): `offLinks` for the data `o`, with the decoders of
the code on the slots `Replay.CodeSlot E` (`families_code`). The families are `(31, 0)` for each group of the root
and of the level 1 and for the level 4, `(31, 10)` for the levels 2 and 3, and `(31, 11)` with
the filter `TrigTable.codeAccepts` of the code for `Γ` = 18 for the levels 5 and 6. -/
def Tower.links : Prefix Wire (Option Coord → Replay.CodeSlot E) total → Prop :=
  offLinks (weight₅ := 53) (weight₆ := 53) (route₄ := Route.coefficient)
    (route₅ := Route.euclidean (norm₅ r).operator (norm₅ r).limit)
    (route₆ := Route.euclidean norm₆.operator norm₆.limit) zeroth first second third fourth fifth
    sixth o.terminal start read o.commit o.value o.place o.commits o.values o.places o.registry
    o.commit₁ o.commit₂ o.commit₃ (And.left quarters) (And.left (And.right quarters))
    (And.left (And.right (And.right quarters))) (And.right (And.right (And.right quarters)))
    o.commit₄ o.commit₅ o.commit₆ o.stateOf

/-- The checkpoint structure of the opening of `o` under a verifier `accept`: `offRoot` with the
batching root at the stage `start`. A first certificate has a weak opening of the three groups
of the root, and a witness of `R_bind` is a solution of Module-SIS for a matrix of a group, of a
level, or for `A_term`. -/
noncomputable def Tower.body (accept : Prefix Wire (Option Coord → Replay.CodeSlot E) total →
    Prop) :=
  offRoot (weight₅ := 53) (weight₆ := 53) (route₄ := Route.coefficient)
    (route₅ := Route.euclidean (norm₅ r).operator (norm₅ r).limit)
    (route₆ := Route.euclidean norm₆.operator norm₆.limit) accept zeroth first second third
    fourth fifth sixth o.terminal 32 16 16 16 start read o.commit o.value o.place o.commits
    o.values o.places o.flat

/-- The hypotheses of `tower_suffix` on the data `o` that are not theorems for the decoders of
the code. `injective` to `injectiveL`: the fold coordinates of each level have different slots.
`registered`: the premise of Akita Lemma 9.1, "correct preprocessing": the commitment `registry`
of the verifier setup is the output of the commitment algorithm `commit` for a table, the setup
prefix `flat` is this table with zeros after its cells, and `reference` is the decommitment of
the algorithm (`Batch.registered_commit`). `next` and `inside`: the fold challenge of the
terminal is the stage after the stages of the level 6, and it is a stage of the transcript.
`decodeL`: the decoder of the terminal is `Replay.CodeSlot.kept64`, the filtered family
`(64, 31, 11)` of the code for `Γ` = 18. `kept`: `2^128` members or more of each filtered family pass the filter (Akita
Table 12). -/
structure Tower.Drawn : Prop where
  injective : Function.Injective o.block
  injective₁ : Function.Injective o.block₁
  registered : ∃ table, o.registry = (Akita.commit (o.key₁ 0) table).1 ∧
    o.flat = ((level₁ r).shape 0).pad table ∧ o.reference = (Akita.commit (o.key₁ 0) table).2
  injective₂ : Function.Injective o.block₂
  injective₃ : Function.Injective o.block₃
  injective₄ : Function.Injective o.block₄
  injective₅ : Function.Injective o.block₅
  injective₆ : Function.Injective o.block₆
  next : Terminal.start o.terminal = Tier.finish sixth
  inside : Terminal.start o.terminal < total
  injectiveL : Function.Injective (Terminal.block o.terminal)
  decodeL : Terminal.decode o.terminal = Replay.CodeSlot.kept64
  kept : Key20.Kept

variable {o shared read start}
  {accept : Prefix Wire (Option Coord → Replay.CodeSlot E) total → Prop}

omit [Fintype Coord] [DecidableEq Coord] [Nonempty Coord] in
/-- With `Tower.Drawn`, each slot gives a fold challenge of each group of the root and of the
level 1, of the levels 2 to 6 and of the terminal, so the conditions `decodes` of the verifier
hold for each table of the oracle on keys. -/
theorem Tower.Drawn.decodes (drawn : o.Drawn shared read start) (slot : Replay.CodeSlot E) :
    (∀ g, (rootDecode E r g slot).isSome) ∧ (∀ g, (decode₁ E r g slot).isSome) ∧
      (code₂ slot).isSome ∧ (code₃ slot).isSome ∧ (code₄ slot).isSome ∧ (code₅ slot).isSome ∧
      (code₆ slot).isSome ∧ (Terminal.decode o.terminal slot).isSome := by
  refine ⟨fun _ => Replay.open_isSome _ rfl (by decide) slot,
    fun _ => Replay.open_isSome _ rfl (by decide) slot, Replay.open_isSome _ rfl (by decide) slot,
    Replay.open_isSome _ rfl (by decide) slot, Replay.open_isSome _ rfl (by decide) slot,
    Replay.kept64_isSome drawn.kept slot, Replay.kept64_isSome drawn.kept slot, ?_⟩
  rw [drawn.decodeL]
  exact Replay.kept64_isSome drawn.kept slot

omit [Fintype Coord] [DecidableEq Coord] [Nonempty Coord] in
/-- The stage of the batching root is a stage of the transcript: the stages of the levels and the
stage of the terminal are after it. -/
theorem Tower.Drawn.fits (drawn : o.Drawn shared read start) : start ≤ total := by
  have ends₀ := Tier.finish_eq zeroth
  have ends₁ := Tier.finish_eq first
  have ends₂ := Tier.finish_eq second
  have ends₃ := Tier.finish_eq third
  have ends₄ := Tier.finish_eq fourth
  have ends₅ := Tier.finish_eq fifth
  have ends₆ := Tier.finish_eq sixth
  have starts₀ : Tier.start zeroth = start + 1 := rfl
  have starts₁ : Tier.start first = Tier.finish zeroth + 1 := rfl
  have starts₂ : Tier.start second = Tier.finish first := rfl
  have starts₃ : Tier.start third = Tier.finish second := rfl
  have starts₄ : Tier.start fourth = Tier.finish third := rfl
  have starts₅ : Tier.start fifth = Tier.finish fourth := rfl
  have starts₆ : Tier.start sixth = Tier.finish fifth := rfl
  have next := drawn.next
  have inside := drawn.inside
  omega

/-- Akita Proposition 10.44, Theorem 10.46 and Lemma 9.1 for the opening of a row of `key30.aks`
with the decoders of the code: the stages of `Tower.body` are a suffix from the stage `start` under
each verifier that accepts only with `Tower.links`, with `1188 + B_1 + B_2 + B_3` retry loops, the tree
size `1813708800·(129 + B_1)·(B_2 + 1)·(B_3 + 1)`, and a sum of the stage errors of at most
`1622923/(4096·|E|) + (1179 + B_1 + B_2 + B_3)/q²` (`tower_suffix` and `towerError_le`: the slot
probability of each decoder is at most `2^-128 ≤ 1/q²`, `Replay.slotChance_open_field`,
`Replay.slotChance_kept64_field`). `slots`: an answer of the oracle has 28 slots. -/
theorem Tower.suffix (drawn : o.Drawn shared read start) (slots : 28 ≤ Fintype.card Coord)
    (linked : ∀ point, accept point → o.links shared read start point) :
    ∃ error, Tier.Suffix accept start (OpensAt start o.commit o.value o.place o.key mass)
        (o.body shared read start accept) error (1188 + r.blocks₁ + r.blocks₂ + r.blocks₃)
        (1813708800 * ((129 + r.blocks₁) * (r.blocks₂ + 1) * (r.blocks₃ + 1))) ∧
      error ≤ 1622923 / (4096 * Fintype.card E) +
        ((1179 + r.blocks₁ + r.blocks₂ + r.blocks₃ : ℕ) : ℝ) * (1 / (fieldModulus : ℝ) ^ 2) := by
  have registered : Batch.Registered (o.key₁ 0) o.registry o.flat o.reference := by
    obtain ⟨table, payload, padded, decommitment⟩ := drawn.registered
    rw [payload, padded, decommitment]
    exact Batch.registered_commit honest.2 (o.key₁ 0) table
  have unitsL : UnitDifferences (Terminal.family o.terminal) := by
    unfold Terminal.family
    rw [drawn.decodeL]
    exact units_of_mem 5 (by decide) fun decoded => (Replay.kept64_mem decoded).1
  have boundedL : ∀ challenge ∈ Terminal.family o.terminal,
      MulBound normTerminal.operator challenge := by
    intro challenge decoded
    unfold Terminal.family at decoded
    rw [drawn.decodeL] at decoded
    exact (TrigTable.filtered_code (by decide) _).mulBound (by decide)
      (Replay.kept64_mem decoded.choose_spec).2
  have chanceL : slotChance (Terminal.decode o.terminal) ≤ 1 / (fieldModulus : ℝ) ^ 2 := by
    rw [drawn.decodeL]
    exact Replay.slotChance_kept64_field drawn.kept
  exact ⟨_,
    tower_suffix (member := member) (scan := Tower.scan shared) (reference := o.reference)
      (registry := o.registry) (β := o.β) (key := o.key) (openKey := o.openKey)
      (claim := o.claim) (block := o.block) (decode := rootDecode E r) (mass := mass)
      (β₁ := o.β₁) (key₁ := o.key₁) (openKey₁ := o.openKey₁) (claim₁ := o.claim₁)
      (block₁ := o.block₁) (decode₁ := decode₁ E r) (mass₁ := mass₁) (β₂ := o.β₂)
      (key₂ := o.key₂) (openKey₂ := o.openKey₂) (claim₂ := o.claim₂) (block₂ := o.block₂)
      (decode₂ := code₂) (weight₂ := 51) (β₃ := o.β₃) (key₃ := o.key₃) (openKey₃ := o.openKey₃)
      (claim₃ := o.claim₃) (block₃ := o.block₃) (decode₃ := code₃) (weight₃ := 51)
      (key₄ := o.key₄) (openKey₄ := o.openKey₄) (claim₄ := o.claim₄) (block₄ := o.block₄)
      (decode₄ := code₄) (weight₄ := 31) (key₅ := o.key₅) (openKey₅ := o.openKey₅)
      (claim₅ := o.claim₅) (block₅ := o.block₅) (decode₅ := code₅) (weight₅ := 53)
      (key₆ := o.key₆) (openKey₆ := o.openKey₆) (claim₆ := o.claim₆) (block₆ := o.block₆)
      (decode₆ := code₆) (weight₆ := 53) (L := o.terminal) (read := read) (start := start)
      (accept := accept) (commit := o.commit) (value := o.value) (place := o.place)
      (commits := o.commits) (values := o.values) (places := o.places) (commit₁ := o.commit₁)
      (commit₂ := o.commit₂) (commit₃ := o.commit₃) (commit₄ := o.commit₄)
      (commit₅ := o.commit₅) (commit₆ := o.commit₆) (stateOf := o.stateOf)
      (peak := fun _ => peak 31 0) (peak₁ := fun _ => peak 31 0) slots drawn.injective
      (fun _ => units_of_mem 6 rfl rootDecode_mem) (fun _ => sparse_of_mem rootDecode_mem)
      drawn.injective₁ (fun _ => units_of_mem 6 rfl decode₁_mem)
      (fun _ => sparse_of_mem decode₁_mem) registered (by decide)
      drawn.injective₂ (units_of_mem 5 rfl fun decoded => (Replay.open_mem decoded).1)
      ⟨2, sparse_of_mem (ones := 31) (twos := 10) fun decoded => (Replay.open_mem decoded).1⟩
      drawn.injective₃ (units_of_mem 5 rfl fun decoded => (Replay.open_mem decoded).1)
      ⟨2, sparse_of_mem (ones := 31) (twos := 10) fun decoded => (Replay.open_mem decoded).1⟩
      drawn.injective₄ (units_of_mem 6 (by decide) fun decoded => (Replay.open_mem decoded).1)
      ⟨1, sparse_of_mem (ones := 31) (twos := 0) fun decoded => (Replay.open_mem decoded).1⟩
      drawn.injective₅ (units_of_mem 5 (by decide) fun decoded => (Replay.kept64_mem decoded).1)
      ⟨2, sparse_of_mem (ones := 31) (twos := 11) fun decoded => (Replay.kept64_mem decoded).1⟩
      (fun _ decoded =>
        ((TrigTable.filtered_code (by decide) _).mulBound (by decide)
          (Replay.kept64_mem decoded.choose_spec).2))
      drawn.injective₆ (units_of_mem 5 (by decide) fun decoded => (Replay.kept64_mem decoded).1)
      ⟨2, sparse_of_mem (ones := 31) (twos := 11) fun decoded => (Replay.kept64_mem decoded).1⟩
      (fun _ decoded =>
        ((TrigTable.filtered_code (by decide) _).mulBound (by decide)
          (Replay.kept64_mem decoded.choose_spec).2))
      drawn.next drawn.inside drawn.injectiveL unitsL boundedL linked,
    towerError_le (member := member) (scan := Tower.scan shared) (reference := o.reference)
      (registry := o.registry) (β := o.β) (key := o.key) (openKey := o.openKey)
      (claim := o.claim) (block := o.block) (decode := rootDecode E r) (mass := mass)
      (β₁ := o.β₁) (key₁ := o.key₁) (openKey₁ := o.openKey₁) (claim₁ := o.claim₁)
      (block₁ := o.block₁) (decode₁ := decode₁ E r) (mass₁ := mass₁) (β₂ := o.β₂)
      (key₂ := o.key₂) (openKey₂ := o.openKey₂) (claim₂ := o.claim₂) (block₂ := o.block₂)
      (decode₂ := code₂) (weight₂ := 51) (β₃ := o.β₃) (key₃ := o.key₃) (openKey₃ := o.openKey₃)
      (claim₃ := o.claim₃) (block₃ := o.block₃) (decode₃ := code₃) (weight₃ := 51)
      (key₄ := o.key₄) (openKey₄ := o.openKey₄) (claim₄ := o.claim₄) (block₄ := o.block₄)
      (decode₄ := code₄) (weight₄ := 31) (key₅ := o.key₅) (openKey₅ := o.openKey₅)
      (claim₅ := o.claim₅) (block₅ := o.block₅) (decode₅ := code₅) (weight₅ := 53)
      (key₆ := o.key₆) (openKey₆ := o.openKey₆) (claim₆ := o.claim₆) (block₆ := o.block₆)
      (decode₆ := code₆) (weight₆ := 53) (L := o.terminal) (read := read) (start := start)
      (flat := o.flat)
      (fun _ => Replay.slotChance_open_field _ _ ((by decide : 128 ≤ 129).trans open128_bits.ge))
      (fun _ => Replay.slotChance_open_field _ _ ((by decide : 128 ≤ 129).trans open128_bits.ge))
      (Replay.slotChance_open_field _ _ open64_bits.ge)
      (Replay.slotChance_open_field _ _ open64_bits.ge)
      (Replay.slotChance_open_field _ _ ((by decide : 128 ≤ 129).trans open128_bits.ge))
      (Replay.slotChance_kept64_field drawn.kept) (Replay.slotChance_kept64_field drawn.kept)
      chanceL⟩

variable (o shared read) in
/-- The verifier of the opening of a row of `key30.aks` with the batching root at the stage 0:
`Tower.links` at the stage 0. -/
def Tower.accepts : Prefix Wire (Option Coord → Replay.CodeSlot E) total → Prop :=
  o.links shared read 0

local notation "whole" => Tower.body o shared read 0 (Tower.accepts o shared read)

/-- An output of the extractor for the opening of a row of `key30.aks` is correct (Akita
Definition 10.15, Proposition 10.44 (i) and Lemma 9.1). The verifier `Tower.accepts` accepts the
transcript of the prover. A first certificate has a certified decommitment of the commitment of
each group of the root, with `κ̄_1 = 2ω` and `β̄_∞ = Δ^cert_f`, to a table that has the claimed
value at the claimed point (`OpensAt`). Each other output is a solution of Module-SIS for a
matrix of a group, of a level, or for `A_term`. -/
theorem Tower.extract_sound (drawn : o.Drawn shared read 0) (slots : 28 ≤ Fintype.card Coord)
    (prover : OracleComp (Address Wire total) (Option Coord → Replay.CodeSlot E) (ℕ → Wire))
      (budget : ℕ)
    (table : Address Wire total → Option Coord → Replay.CodeSlot E)
    (tape : Tape (Address Wire total) (Option Coord) (Replay.CodeSlot E) budget
      (Structure.protocol whole).stages) :
    ∀ out ∈ Checkpoint.extract whole prover budget table tape,
      o.accepts shared read ((Structure.protocol whole).final prover table) ∧
        Sum.elim
          (fun certificate => OpensAt 0 o.commit o.value o.place o.key mass
            ((Structure.protocol whole).input prover table 0)
            (Structure.witness whole certificate))
          (Structure.breaks whole) out := by
  obtain ⟨_, suffix, -⟩ := Tower.suffix drawn slots fun _ accepted => accepted
  intro out result
  have sound := suffix.extract_sound rfl prover budget table tape out result
  exact ⟨(congrFun (congrArg Protocol.accept suffix.protocol) _).mp sound.1, sound.2⟩

variable [Finite Wire]

/-- Akita Theorem 10.17 in the form (161), with Corollary 10.49 and (206), for the opening of a
row of `key30.aks` with the decoders of the code. For each prover with at most `Q` different
queried keys of each stage (`Protocol.Spread`), the probability that `Tower.accepts` accepts and
the extractor returns no output is at most
`(Q + 1)·(1622923/(4096·|E|) + (1179 + B_1 + B_2 + B_3)/q²) + (Q + 1)·(1188 + B_1 + B_2 + B_3)/budget`,
with
`|E| = q²` (`Key20.card_field`). The probability is over the uniform table and the tape of the
extractor. The bound has no term for the probability that the output is a solution of
Module-SIS. -/
theorem Tower.extraction_spread_le (drawn : o.Drawn shared read 0)
    (slots : 28 ≤ Fintype.card Coord)
    (prover : OracleComp (Address Wire total) (Option Coord → Replay.CodeSlot E) (ℕ → Wire))
      {queries : ℕ}
    (spread : (Structure.protocol whole).Spread prover queries) {budget : ℕ}
    (positive : 0 < budget) :
    𝔼 table, 𝔼 tape : Tape (Address Wire total) (Option Coord) (Replay.CodeSlot E) budget
          (Structure.protocol whole).stages,
        (if o.accepts shared read ((Structure.protocol whole).final prover table) ∧
          Checkpoint.extract whole prover budget table tape = none
            then (1 : ℝ) else 0) ≤
      (queries + 1) * (1622923 / (4096 * Fintype.card E) +
          ((1179 + r.blocks₁ + r.blocks₂ + r.blocks₃ : ℕ) : ℝ) * (1 / (fieldModulus : ℝ) ^ 2)) +
        (queries + 1) * (1188 + r.blocks₁ + r.blocks₂ + r.blocks₃ : ℕ) / budget := by
  obtain ⟨_, suffix, small⟩ := Tower.suffix drawn slots fun _ accepted => accepted
  have bound := suffix.extraction_le rfl prover spread positive
  rw [congrArg Protocol.accept suffix.protocol] at bound
  exact bound.trans (add_le_add (mul_le_mul_of_nonneg_left small (by positivity)) le_rfl)

/-- `Tower.extraction_spread_le` for a prover with at most `Q` queries to the table of the oracle
on keys (Akita Theorem 10.17 in the form (161)). -/
theorem Tower.extraction_le (drawn : o.Drawn shared read 0) (slots : 28 ≤ Fintype.card Coord)
    (prover : OracleComp (Address Wire total) (Option Coord → Replay.CodeSlot E) (ℕ → Wire))
      {queries : ℕ}
    (bounded : prover.QueryBound queries) {budget : ℕ} (positive : 0 < budget) :
    𝔼 table, 𝔼 tape : Tape (Address Wire total) (Option Coord) (Replay.CodeSlot E) budget
          (Structure.protocol whole).stages,
        (if o.accepts shared read ((Structure.protocol whole).final prover table) ∧
          Checkpoint.extract whole prover budget table tape = none
            then (1 : ℝ) else 0) ≤
      (queries + 1) * (1622923 / (4096 * Fintype.card E) +
          ((1179 + r.blocks₁ + r.blocks₂ + r.blocks₃ : ℕ) : ℝ) * (1 / (fieldModulus : ℝ) ^ 2)) +
        (queries + 1) * (1188 + r.blocks₁ + r.blocks₂ + r.blocks₃ : ℕ) / budget :=
  Tower.extraction_spread_le drawn slots prover (bounded.spread _) positive

end NightstreamFPrime.Spec.Compression.Akita.Key30
