import NightstreamFPrime.Spec.Compression.Akita.Key20Challenge
import NightstreamFPrime.Spec.Compression.Akita.Replay.CodeSlot

/-! The opening of Akita for the schedule file `crates/ns-compress/artifacts/key20.aks` with the
fold decoders of the pinned Akita code (revision `d50f700`,
`crates/akita-challenges/src/sampler/stream.rs`) on the slots `Replay.CodeSlot E`, as one
statement (Akita, ePrint 2026/1983, revision of 2026-11-23, Theorem 10.17, Proposition 10.44,
Theorem 10.46 and (206)).

- `rootGroup`, `open64` and `open128` are the groups of the code without a filter: the root
  groups, the family `(31, 10)` of the degree 64 and the family `(31, 0)` of the degree 128.
  Their words have 128 and 129 bits (`candidateBits_open`). `rootDecode` is
  `Replay.CodeSlot.open` of the groups of the root: each slot decodes (`rootDecode_isSome`), and
  the sign flips act on the slots (`rootDecode_signFair`).
- `Tower` has the data of the opening: the matrices, the readers of the messages, the
  coordinates of the fold challenges and the terminal. The batching root is at the stage 0.
- `Tower.accepts` is the verifier `towerLinks` for this data, and `Tower.body` is its checkpoint
  structure `towerRoot`. The decoder of the root and of the levels 1 and 2 is
  `Replay.CodeSlot.open` of its group, and the decoder of the levels 3 to 5 and of the terminal
  is `Replay.CodeSlot.kept128` or `Replay.CodeSlot.kept64` of the filtered family of the code
  (`families_code`).
- `Tower.Drawn` has the hypotheses of `tower_suffix` that are not theorems for these decoders,
  and the count `Kept` of the filtered families. `Tower.Drawn.decodes`: each slot gives a fold
  challenge at each level.
- `Tower.suffix` is `tower_suffix` with `towerError_le` at the slot probability `1/q²`
  (`Replay.slotChance_open_field`, `Replay.slotChance_kept64_field`,
  `Replay.slotChance_kept128_field`).
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

section Groups

/-- The group of the code of each group of the root, without a filter: the degree of the group
and the family `(31, n₂)` of `rootTwos`. -/
abbrev rootGroup (g : Fin 3) : Replay.FoldGroup :=
  ⟨0, root.carrierDegree g, root.carrierDegree g, 31, rootTwos g, false⟩

/-- The group of the code of the family `(31, 10)` of the degree 64, without a filter. -/
abbrev open64 : Replay.FoldGroup := ⟨0, 64, 64, 31, 10, false⟩

/-- The group of the code of the family `(31, 0)` of the degree 128, without a filter. -/
abbrev open128 : Replay.FoldGroup := ⟨0, 128, 128, 31, 0, false⟩

/-- Without a filter a word has `⌊log₂ rankSpace⌋` rank bits and one sign bit for each nonzero
coefficient. -/
theorem candidateBits_open {G : Replay.FoldGroup} (unfiltered : G.filtered = false) {m : ℕ}
    (low : 2 ^ m ≤ G.rankSpace) (high : G.rankSpace < 2 ^ (m + 1)) :
    G.candidateBits = m + G.pm1 + G.pm2 := by
  unfold Replay.FoldGroup.candidateBits
  rw [ite_eq_right (by simp [unfiltered]), Nat.log_eq_of_pow_le_of_lt_pow low high]

/-- The words of `(31, 10)` of the degree 64 have 128 bits. -/
theorem open64_bits : open64.candidateBits = 128 :=
  candidateBits_open (m := 87) rfl (by decide) (by decide)

/-- The words of `(31, 0)` of the degree 128 have 129 bits. -/
theorem open128_bits : open128.candidateBits = 129 :=
  candidateBits_open (m := 98) rfl (by decide) (by decide)

/-- The words of the groups of the root have 129, 128 and 128 bits. -/
theorem rootGroup_bits : ∀ g, (rootGroup g).candidateBits = ![129, 128, 128] g := by
  intro g
  fin_cases g
  · exact open128_bits
  · exact open64_bits
  · exact open64_bits

theorem open64_fits : open64.candidateBits ≤ 131 := open64_bits.trans_le (by decide)

theorem open128_fits : open128.candidateBits ≤ 131 := open128_bits.trans_le (by decide)

theorem rootGroup_fits (g : Fin 3) : (rootGroup g).candidateBits ≤ 131 := by
  rw [rootGroup_bits]
  fin_cases g <;> decide

theorem rootGroup_wide (g : Fin 3) : 128 ≤ (rootGroup g).candidateBits := by
  rw [rootGroup_bits]
  fin_cases g <;> decide

theorem rootGroup_positive (g : Fin 3) : 0 < (rootGroup g).rankSpace := by
  fin_cases g <;> decide

variable (E : Type) in
/-- The decoder of the code of the fold challenges of each group of the root. -/
noncomputable def rootDecode (g : Fin 3) :
    Replay.CodeSlot E → Option (Rq (root.carrierDegree g)) :=
  Replay.CodeSlot.open (rootGroup g) (rootGroup_fits g)

variable {E : Type}

/-- Each challenge of `rootDecode` is in the family `(31, n₂)` of its group. -/
theorem rootDecode_mem {g : Fin 3} {slot : Replay.CodeSlot E} {challenge : Rq (root.carrierDegree g)}
    (decoded : rootDecode E g slot = some challenge) :
    challenge ∈ SparseFamily (root.carrierDegree g) 31 (rootTwos g) := by
  unfold rootDecode at decoded
  exact (Replay.open_mem decoded).1

theorem rootDecode_units (g : Fin 3) :
    UnitDifferences {challenge | ∃ slot, rootDecode E g slot = some challenge} :=
  units_of_mem (![6, 5, 5] g) (by revert g; decide) rootDecode_mem

theorem rootDecode_sparse (g : Fin 3) (challenge : Rq (root.carrierDegree g))
    (decoded : ∃ slot, rootDecode E g slot = some challenge) :
    Sparse (rootMass g) (rootPeak g) challenge :=
  sparse_of_mem rootDecode_mem challenge decoded

/-- Each slot gives a fold challenge of each group of the root. -/
theorem rootDecode_isSome (g : Fin 3) (slot : Replay.CodeSlot E) : (rootDecode E g slot).isSome :=
  Replay.open_isSome _ rfl (rootGroup_positive g) slot

/-- The sign flips of a fold challenge act on the slots of each decoder of the root
(`Replay.signFair_open`): the condition `fair` of `root_search`. -/
theorem rootDecode_signFair (g : Fin 3) : SignFair (rootDecode E g) := by
  unfold rootDecode
  exact Replay.signFair_open (rootGroup_fits g) rfl

/-- A slot has `2^131` values or more. -/
theorem card_codeSlot [Fintype E] [Nonempty E] : 2 ^ 131 ≤ Fintype.card (Replay.CodeSlot E) := by
  simp only [Replay.CodeSlot, Fintype.card_prod, Fintype.card_fin]
  exact (Nat.le_mul_of_pos_right _ (Nat.mul_pos Fintype.card_pos Fintype.card_pos)).trans
    (Nat.le_mul_of_pos_left _ Fintype.card_pos)

end Groups

/-- The data of the opening of `key20.aks` with the batching root at the stage 0: the matrices
and the bases of the root, of the levels 1 to 5 and of the terminal, the readers of the wire
messages, and the coordinates of the slots of the fold challenges. -/
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
  key₄ : Key level₄.shape
  openKey₄ : OpeningKey level₄.opening
  claim₄ : Wire → TraceClaim level₄.opening E
  block₄ : Fin level₄.shape.blocks → Coord
  key₅ : Key level₅.shape
  openKey₅ : OpeningKey level₅.opening
  claim₅ : Wire → TraceClaim level₅.opening E
  block₅ : Fin level₅.shape.blocks → Coord
  terminal : Terminal Key20.terminal E Wire Coord (Replay.CodeSlot E) total
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

local notation "code₁" =>
  (Replay.CodeSlot.open open64 open64_fits :
    Replay.CodeSlot E → Option (Rq level₁.opening.carrierDegree))

local notation "code₂" =>
  (Replay.CodeSlot.open open128 open128_fits :
    Replay.CodeSlot E → Option (Rq level₂.shape.innerDegree))

local notation "code₃" =>
  (Replay.CodeSlot.kept128 : Replay.CodeSlot E → Option (Rq level₃.shape.innerDegree))

local notation "code₄" =>
  (Replay.CodeSlot.kept64 : Replay.CodeSlot E → Option (Rq level₄.shape.innerDegree))

local notation "code₅" =>
  (Replay.CodeSlot.kept64 : Replay.CodeSlot E → Option (Rq level₅.shape.innerDegree))

/-- The pair `(n₁, n₂)` of the challenge family of each group of each level of `key20.aks`, and
then of the terminal: the fields `count_pm1` and `count_pm2` of the file. -/
def families : List (List (ℕ × ℕ)) :=
  [[(31, 0), (31, 10), (31, 10)], [(31, 10)], [(31, 0)], [(31, 0)], [(31, 11)], [(31, 11)],
    [(31, 11)]]

/-- The pair of `families` for a group of a level. -/
def family (level group : ℕ) : ℕ × ℕ := (families.getD level []).getD group (0, 0)

/-- The decoders of `Tower.accepts` are the decoders of the code for the families of `families`:
the groups of the root and of the levels 1 and 2 have no filter, and the levels 3 to 5 and the
terminal have the filtered families `(128, 31, 0)` and `(64, 31, 11)` of the code. -/
theorem families_code :
    (∀ g, ((rootGroup g).pm1, (rootGroup g).pm2) = family 0 g ∧ (rootGroup g).filtered = false) ∧
    (open64.pm1, open64.pm2) = family 1 0 ∧ (open128.pm1, open128.pm2) = family 2 0 ∧
    (Replay.filtered128.pm1, Replay.filtered128.pm2) = family 3 0 ∧
    (Replay.filtered64.pm1, Replay.filtered64.pm2) = family 4 0 ∧
    (Replay.filtered64.pm1, Replay.filtered64.pm2) = family 5 0 ∧
    (Replay.filtered64.pm1, Replay.filtered64.pm2) = family 6 0 :=
  ⟨fun g => by fin_cases g <;> exact ⟨rfl, rfl⟩, rfl, rfl, rfl, rfl, rfl, rfl⟩

local notation "zeroth" =>
  liftBatch o.β o.key o.openKey o.claim rootVars (And.left root_sized) 8 quotientDigits root_room
    rootBatch (And.left (And.right root_sized)) o.block (rootDecode E) o.read rootMass (0 + 1)

local notation "first" =>
  Tier.tensored (liftFold o.β₁ o.key₁ o.openKey₁ o.claim₁ level₁.vars (And.left level₁_sized)
    quotientDigits (And.left (And.right level₁_sized)) level₁.batch
    (And.left (And.right (And.right level₁_sized))) o.block₁ code₁ o.read 51
    (Tier.finish zeroth)) level₁.tail

local notation "second" =>
  Tier.tensored (committedTrace (fold := level₂.fold) o.key₂ o.openKey₂ o.claim₂ level₂.vars
    (And.left sized₂) level₂.batch (And.left (And.right sized₂)) o.block₂ code₂ o.read 31
    (Tier.finish first)) level₂.tail

local notation "third" =>
  Tier.tensored (committedTraceSq (fold := level₃.fold) o.key₃ o.openKey₃ o.claim₃ level₃.vars
    (And.left sized₃) level₃.batch (And.left (And.right sized₃)) o.block₃ code₃
    o.read 31 norm₃.operator norm₃.limit (Tier.finish second)) level₃.tail

local notation "fourth" =>
  Tier.tensored (committedTraceSq (fold := level₄.fold) o.key₄ o.openKey₄ o.claim₄ level₄.vars
    (And.left sized₄) level₄.batch (And.left (And.right sized₄)) o.block₄ code₄
    o.read 53 norm₄.operator norm₄.limit (Tier.finish third)) level₄.tail

local notation "fifth" =>
  Tier.tensored (committedTraceSq (fold := level₅.fold) o.key₅ o.openKey₅ o.claim₅ level₅.vars
    (And.left sized₅) level₅.batch (And.left (And.right sized₅)) o.block₅ code₅
    o.read 53 norm₅.operator norm₅.limit (Tier.finish fourth)) level₅.tail

/-- The verifier of the opening of `key20.aks` (Akita Figure 4 and Section 8.2): `towerLinks` for
the data `o`, with the decoders of the code on the slots `Replay.CodeSlot E`. The families are
`(31, 10)` for the groups of `W` and `V` and for the level 1, `(31, 0)` for the key group and for
the levels 2 and 3, and `(31, 11)` for the levels 4 and 5, with the filter of the code at the
levels 3 to 5 (`families_code`). -/
def Tower.accepts : Prefix Wire (Option Coord → Replay.CodeSlot E) total → Prop :=
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

/-- The hypotheses of `tower_suffix` on the data `o` that are not theorems for the decoders of
the code. `injective` to `injectiveL`: the fold coordinates of each level have different slots.
`next` and `inside`: the fold challenge of the terminal is the stage after the stages of the
level 5, and it is a stage of the transcript. `decodeL`: the decoder of the terminal is
`Replay.CodeSlot.kept64`, the filtered family `(64, 31, 11)` of the code for `Γ` = 18. `kept`:
`2^128` members or more of each filtered family pass the filter (Akita Table 12). -/
structure Tower.Drawn : Prop where
  injective : Function.Injective o.block
  injective₁ : Function.Injective o.block₁
  injective₂ : Function.Injective o.block₂
  injective₃ : Function.Injective o.block₃
  injective₄ : Function.Injective o.block₄
  injective₅ : Function.Injective o.block₅
  next : Terminal.start o.terminal = Tier.finish fifth
  inside : Terminal.start o.terminal < total
  injectiveL : Function.Injective (Terminal.block o.terminal)
  decodeL : Terminal.decode o.terminal = Replay.CodeSlot.kept64
  kept : Kept

variable {o}

omit [Fintype Coord] [DecidableEq Coord] [Nonempty Coord] in
/-- With `Tower.Drawn`, each slot gives a fold challenge of each group of the root, of the
levels 1 to 5 and of the terminal, so the conditions `decodes` of the verifier hold for each
table of the oracle on keys. -/
theorem Tower.Drawn.decodes (drawn : o.Drawn) (slot : Replay.CodeSlot E) :
    (∀ g, (rootDecode E g slot).isSome) ∧ (code₁ slot).isSome ∧ (code₂ slot).isSome ∧
      (code₃ slot).isSome ∧ (code₄ slot).isSome ∧ (code₅ slot).isSome ∧
      (Terminal.decode o.terminal slot).isSome := by
  refine ⟨fun g => rootDecode_isSome g slot, Replay.open_isSome _ rfl (by decide) slot,
    Replay.open_isSome _ rfl (by decide) slot, Replay.kept128_isSome drawn.kept slot,
    Replay.kept64_isSome drawn.kept slot, Replay.kept64_isSome drawn.kept slot, ?_⟩
  rw [drawn.decodeL]
  exact Replay.kept64_isSome drawn.kept slot

/-- Akita Proposition 10.44 and Theorem 10.46 for the opening of `key20.aks` with the decoders of
the code: the stages of `Tower.body` are a suffix from the stage 0 under the verifier
`Tower.accepts`, with 142 retry loops, the tree size 21449539584, and a sum of the stage errors
of at most `317479/(1024·|E|) + 135/q²` (`tower_suffix` and `towerError_le`: the slot
probability of each decoder is at most `2^-128 ≤ 1/q²`, `Replay.slotChance_open_field`,
`Replay.slotChance_kept64_field`, `Replay.slotChance_kept128_field`).
`slots`: an answer of the oracle has 22 slots. -/
theorem Tower.suffix (drawn : o.Drawn) (slots : 22 ≤ Fintype.card Coord) :
    ∃ error, Tier.Suffix o.accepts 0 (OpensAt 0 o.commit o.value o.place o.key rootMass) o.body
        error 142 21449539584 ∧
      error ≤ 317479 / (1024 * Fintype.card E) + 135 * (1 / (fieldModulus : ℝ) ^ 2) := by
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
    tower_suffix (β := o.β) (key := o.key) (openKey := o.openKey) (claim := o.claim)
      (block := o.block) (decode := rootDecode E) (mass := rootMass) (β₁ := o.β₁)
      (key₁ := o.key₁) (openKey₁ := o.openKey₁) (claim₁ := o.claim₁) (block₁ := o.block₁)
      (decode₁ := code₁) (weight₁ := 51) (key₂ := o.key₂) (openKey₂ := o.openKey₂)
      (claim₂ := o.claim₂) (block₂ := o.block₂) (decode₂ := code₂) (weight₂ := 31)
      (key₃ := o.key₃) (openKey₃ := o.openKey₃) (claim₃ := o.claim₃) (block₃ := o.block₃)
      (decode₃ := code₃) (weight₃ := 31) (key₄ := o.key₄) (openKey₄ := o.openKey₄)
      (claim₄ := o.claim₄) (block₄ := o.block₄) (decode₄ := code₄) (weight₄ := 53)
      (key₅ := o.key₅) (openKey₅ := o.openKey₅) (claim₅ := o.claim₅) (block₅ := o.block₅)
      (decode₅ := code₅) (weight₅ := 53) (L := o.terminal) (read := o.read) (start := 0)
      (commit := o.commit) (value := o.value) (place := o.place) (commit₁ := o.commit₁)
      (commit₂ := o.commit₂) (commit₃ := o.commit₃) (commit₄ := o.commit₄)
      (commit₅ := o.commit₅) (stateOf := o.stateOf) (peak := rootPeak) slots drawn.injective
      rootDecode_units rootDecode_sparse
      drawn.injective₁ (units_of_mem 5 (by decide) fun decoded => (Replay.open_mem decoded).1)
      ⟨2, sparse_of_mem (ones := 31) (twos := 10) fun decoded => (Replay.open_mem decoded).1⟩
      drawn.injective₂ (units_of_mem 6 (by decide) fun decoded => (Replay.open_mem decoded).1)
      ⟨1, sparse_of_mem (ones := 31) (twos := 0) fun decoded => (Replay.open_mem decoded).1⟩
      drawn.injective₃ (units_of_mem 6 (by decide) fun decoded => (Replay.kept128_mem decoded).1)
      ⟨1, sparse_of_mem (ones := 31) (twos := 0) fun decoded => (Replay.kept128_mem decoded).1⟩
      (fun _ decoded =>
        ((TrigTable.filtered_code (by decide) _).mulBound (by decide)
          (Replay.kept128_mem decoded.choose_spec).2))
      drawn.injective₄ (units_of_mem 5 (by decide) fun decoded => (Replay.kept64_mem decoded).1)
      ⟨2, sparse_of_mem (ones := 31) (twos := 11) fun decoded => (Replay.kept64_mem decoded).1⟩
      (fun _ decoded =>
        ((TrigTable.filtered_code (by decide) _).mulBound (by decide)
          (Replay.kept64_mem decoded.choose_spec).2))
      drawn.injective₅ (units_of_mem 5 (by decide) fun decoded => (Replay.kept64_mem decoded).1)
      ⟨2, sparse_of_mem (ones := 31) (twos := 11) fun decoded => (Replay.kept64_mem decoded).1⟩
      (fun _ decoded =>
        ((TrigTable.filtered_code (by decide) _).mulBound (by decide)
          (Replay.kept64_mem decoded.choose_spec).2))
      drawn.next drawn.inside drawn.injectiveL unitsL boundedL,
    towerError_le (β := o.β) (key := o.key) (openKey := o.openKey) (claim := o.claim)
      (block := o.block) (decode := rootDecode E) (mass := rootMass) (β₁ := o.β₁)
      (key₁ := o.key₁) (openKey₁ := o.openKey₁) (claim₁ := o.claim₁) (block₁ := o.block₁)
      (decode₁ := code₁) (weight₁ := 51) (key₂ := o.key₂) (openKey₂ := o.openKey₂)
      (claim₂ := o.claim₂) (block₂ := o.block₂) (decode₂ := code₂) (weight₂ := 31)
      (key₃ := o.key₃) (openKey₃ := o.openKey₃) (claim₃ := o.claim₃) (block₃ := o.block₃)
      (decode₃ := code₃) (weight₃ := 31) (key₄ := o.key₄) (openKey₄ := o.openKey₄)
      (claim₄ := o.claim₄) (block₄ := o.block₄) (decode₄ := code₄) (weight₄ := 53)
      (key₅ := o.key₅) (openKey₅ := o.openKey₅) (claim₅ := o.claim₅) (block₅ := o.block₅)
      (decode₅ := code₅) (weight₅ := 53) (L := o.terminal) (read := o.read) (start := 0)
      (fun g => Replay.slotChance_open_field _ _ (rootGroup_wide g))
      (Replay.slotChance_open_field _ _ open64_bits.ge)
      (Replay.slotChance_open_field _ _ (open128_bits.ge.trans' (by decide)))
      (Replay.slotChance_kept128_field drawn.kept) (Replay.slotChance_kept64_field drawn.kept)
      (Replay.slotChance_kept64_field drawn.kept) chanceL⟩

/-- An output of the extractor for the opening of `key20.aks` is correct (Akita Definition 10.15
and Proposition 10.44 (i)). The verifier `Tower.accepts` accepts the transcript of the prover. A
first certificate has a certified decommitment of the commitment of each group of the root, with
`κ̄_1 = 2ω` and `β̄_∞ = Δ^cert_f`, to a table that has the claimed value at the claimed point
(`OpensAt`). Each other output is a solution of Module-SIS for a matrix of a group, of a level,
or for `A_term`. -/
theorem Tower.extract_sound (drawn : o.Drawn) (slots : 22 ≤ Fintype.card Coord)
    (prover : OracleComp (Address Wire total) (Option Coord → Replay.CodeSlot E) (ℕ → Wire))
      (budget : ℕ)
    (table : Address Wire total → Option Coord → Replay.CodeSlot E)
    (tape : Tape (Address Wire total) (Option Coord) (Replay.CodeSlot E) budget
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
`key20.aks` with the decoders of the code. For each prover with at most `Q` different queried keys
of each stage (`Protocol.Spread`), the probability that `Tower.accepts` accepts and the extractor
returns no output is at most `(Q + 1)·(317479/(1024·|E|) + 135/q²) + (Q + 1)·142/budget`, with `|E| = q²`
(`card_field`). The fold term has the slot probability `1/q²` of each fold coordinate, an upper
bound of the probability `2^-K` or `1/|Kept G|` of the decoder of the code. The probability is over the uniform table and the tape of the extractor. The
bound has no term for the probability that the output is a solution of Module-SIS. -/
theorem Tower.extraction_spread_le (drawn : o.Drawn) (slots : 22 ≤ Fintype.card Coord)
    (prover : OracleComp (Address Wire total) (Option Coord → Replay.CodeSlot E) (ℕ → Wire))
      {queries : ℕ}
    (spread : o.body.protocol.Spread prover queries) {budget : ℕ} (positive : 0 < budget) :
    𝔼 table, 𝔼 tape : Tape (Address Wire total) (Option Coord) (Replay.CodeSlot E) budget
          o.body.protocol.stages,
        (if o.accepts (o.body.protocol.final prover table) ∧
          Checkpoint.extract o.body prover budget table tape = none then (1 : ℝ) else 0) ≤
      (queries + 1) * (317479 / (1024 * Fintype.card E) + 135 * (1 / (fieldModulus : ℝ) ^ 2)) +
        (queries + 1) * (142 : ℕ) / budget := by
  obtain ⟨_, suffix, small⟩ := Tower.suffix drawn slots
  have bound := suffix.extraction_le rfl prover spread positive
  rw [congrArg Protocol.accept suffix.protocol] at bound
  exact bound.trans (add_le_add (mul_le_mul_of_nonneg_left small (by positivity)) le_rfl)

/-- `Tower.extraction_spread_le` for a prover with at most `Q` queries to the table of the oracle
on keys (Akita Theorem 10.17 in the form (161)). -/
theorem Tower.extraction_le (drawn : o.Drawn) (slots : 22 ≤ Fintype.card Coord)
    (prover : OracleComp (Address Wire total) (Option Coord → Replay.CodeSlot E) (ℕ → Wire))
      {queries : ℕ}
    (bounded : prover.QueryBound queries) {budget : ℕ} (positive : 0 < budget) :
    𝔼 table, 𝔼 tape : Tape (Address Wire total) (Option Coord) (Replay.CodeSlot E) budget
          o.body.protocol.stages,
        (if o.accepts (o.body.protocol.final prover table) ∧
          Checkpoint.extract o.body prover budget table tape = none then (1 : ℝ) else 0) ≤
      (queries + 1) * (317479 / (1024 * Fintype.card E) + 135 * (1 / (fieldModulus : ℝ) ^ 2)) +
        (queries + 1) * (142 : ℕ) / budget :=
  Tower.extraction_spread_le drawn slots prover (bounded.spread _) positive

end NightstreamFPrime.Spec.Compression.Akita.Key20
