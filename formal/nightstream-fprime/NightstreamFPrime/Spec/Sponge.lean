import Mathlib.Basic.Real.Basic
import Mathlib.Data.Set.Finite.List
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Positivity
import NightstreamFPrime.Spec.Poseidon2
import NightstreamFPrime.Spec.RandomOracle

/-! The sponge layer of `SECURITY_MODEL.md`, section Assumptions.

`[P2-PERM]` takes `Poseidon2.permute` as a uniform permutation, `[SPONGE]` is the sponge
indifferentiability theorem with the error `spongeError`, and `[DSFS]` is the duplex-sponge
Fiat–Shamir lemma with the error `duplexError`. This file states the conditions of `[SPONGE]`
as Lean predicates.

- `Padding`: a padding that is injective, keeps its input as a prefix of the padded words and
  gives full rate blocks. The `10*` padding of `Poseidon2.hash` is one (`padding`), and `hash` is
  the sponge digest (`digest`) of its blocks (`hash_eq_digest`).
- So a collision of `hash` is a sponge collision on two distinct block lists
  (`collision_blocks`).
- Domain separation: two block lists whose words start with inputs that differ at a position of
  both are distinct (`blocks_ne_of_differ`), so two uses with such tags never absorb one block
  list (`padded_ne_of_differ`).
- `Transfer` applies `[SPONGE]` to one use: under the use's conditions, each adversary of the
  ideal-permutation model has a counterpart in the use's random-oracle model that wins at most
  `spongeError calls` less often.
- `DuplexTransfer` applies `[DSFS]` to one Fiat–Shamir use: under the use's conditions, each
  prover of the ideal-permutation model with `calls` permutation calls has a counterpart with at
  most `calls` queries in the use's random-oracle model that wins at most `duplexError calls`
  less often. The event can include an extractor that reruns the counterpart.
- The collision use (`CollisionConditions`, proved by `collisionConditions`): a collision of
  `hash` is a collision of a uniform 4-lane function on block lists. A model adversary with `Q`
  queries finds one with chance at most `C(Q + 2, 2)/q⁴` (`collisionChance_le`, the birthday
  bound `RandomOracle.pair_collision_le`), so a finder with `calls` permutation calls succeeds
  with chance at most that bound plus `spongeError calls` (`collision_le`). -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Sponge

universe uWorld uModel

open scoped BigOperators
open NightstreamFPrime.Spec.Poseidon2

/-- The sponge state after absorbing the blocks from the zero state. -/
def run (blocks : List (List F)) : State := blocks.foldl absorbBlock zeroState

/-- The sponge digest of a block list: the first `digestLen` lanes of the state. -/
def digest (blocks : List (List F)) : List F := (run blocks).take digestLen

/-- A padding of `[SPONGE]`: injective, with full rate blocks, at least one block, and the
input as a prefix of the padded words. -/
structure Padding where
  pad : List F → List (List F)
  injective : Function.Injective pad
  full : ∀ input, ∀ block ∈ pad input, block.length = rate
  nonempty : ∀ input, pad input ≠ []
  keeps : ∀ input, input <+: (pad input).flatten

/-- The `10*` padding of `Poseidon2.hash` (`Poseidon2.blocks`) meets the conditions of
`[SPONGE]`. -/
def padding : Padding where
  pad := Poseidon2.blocks
  injective _ _ equal := Poseidon2.blocks_injective equal
  full _ _ := Poseidon2.blocks_length
  nonempty := Poseidon2.blocks_ne_nil
  keeps input := by
    rw [Poseidon2.blocks_flatten]
    exact List.prefix_append _ _

/-- `Poseidon2.hash` is the sponge digest of its padded blocks. -/
theorem hash_eq_digest (input : List F) : Poseidon2.hash input = digest (padding.pad input) := rfl

/-- The capacity term of `[SPONGE]` for `calls` permutation calls: `calls² / q^c`, with the
capacity `c = width − rate = 4` lanes. -/
noncomputable def spongeError (calls : ℕ) : ℝ :=
  (calls : ℝ) ^ 2 / (fieldModulus : ℝ) ^ (width - rate)

theorem spongeError_nonneg (calls : ℕ) : 0 ≤ spongeError calls := by
  unfold spongeError
  positivity

/-- Two distinct inputs with one hash give two distinct block lists with one sponge digest. -/
theorem collision_blocks {left right : List F} (distinct : left ≠ right)
    (same : Poseidon2.hash left = Poseidon2.hash right) :
    padding.pad left ≠ padding.pad right ∧ digest (padding.pad left) = digest (padding.pad right) :=
  ⟨fun equal => distinct (padding.injective equal), same⟩

/-- Two block lists whose words start with inputs that differ at a position of both are
distinct. -/
theorem blocks_ne_of_differ {blocks blocks' : List (List F)} {left right : List F}
    (leftPrefix : left <+: blocks.flatten) (rightPrefix : right <+: blocks'.flatten)
    {position : ℕ} (inLeft : position < left.length) (inRight : position < right.length)
    (differ : left[position] ≠ right[position]) : blocks ≠ blocks' := by
  intro equal
  subst equal
  apply differ
  rw [leftPrefix.getElem inLeft, rightPrefix.getElem inRight]

/-- Two inputs that differ at a position of both have distinct padded block lists. -/
theorem padded_ne_of_differ {left right : List F} {position : ℕ}
    (inLeft : position < left.length) (inRight : position < right.length)
    (differ : left[position] ≠ right[position]) : padding.pad left ≠ padding.pad right :=
  blocks_ne_of_differ (padding.keeps left) (padding.keeps right) inLeft inRight differ

/-- `[SPONGE]` for one use. Under `conditions`, the use's conditions of `[SPONGE]`, each
adversary `world` of the ideal-permutation model (`[P2-PERM]`) with `calls` permutation calls has
a counterpart `model` in the use's random-oracle model whose wins `modelWins model` are at most
`spongeError calls` below the wins `winsOf world`. A use states its conditions as a structure
and proves them, so the premise applies only through that proof. -/
def Transfer {World : Type uWorld} {Model : Type uModel} (conditions : Prop) (winsOf : World → ℝ)
    (modelWins : Model → ℝ) (calls : ℕ) : Prop :=
  conditions → ∀ world, ∃ model, winsOf world ≤ modelWins model + spongeError calls

/-- A bound of the random-oracle model holds in the ideal-permutation model, plus
`spongeError calls`, once the use's conditions are proved. -/
theorem transfer_le {World : Type uWorld} {Model : Type uModel} {conditions : Prop} {winsOf : World → ℝ}
    {modelWins : Model → ℝ} {calls : ℕ} {bound : ℝ}
    (premise : Transfer conditions winsOf modelWins calls) (proved : conditions)
    (modelBound : ∀ model, modelWins model ≤ bound) (world : World) :
    winsOf world ≤ bound + spongeError calls := by
  obtain ⟨model, wins⟩ := premise proved world
  linarith [modelBound model]

/-- The error of `[DSFS]` for `calls` permutation calls: `25 · calls² / q^c`, with the capacity
`c = width − rate = 4` lanes. It is `η⋆` of ePrint 2025/536 (Chiesa and Orrù), Theorem 6.2, at
the codec bias 0. -/
noncomputable def duplexError (calls : ℕ) : ℝ :=
  25 * (calls : ℝ) ^ 2 / (fieldModulus : ℝ) ^ (width - rate)

/-- `[DSFS]` for one use, in the shape of ePrint 2025/536 (Chiesa and Orrù), Lemma 5.1. Under
`conditions`, each prover `world` of the ideal-permutation model (`[P2-PERM]`) with `calls`
permutation calls has a counterpart `model` with the query bound `calls` in the use's
random-oracle model, and the wins `winsOf world` are at most `modelWins calls model` plus
`duplexError calls`. `Model queries` is the type of the model provers with at most `queries`
queries. -/
def DuplexTransfer {World : Type uWorld} {Model : ℕ → Type uModel} (conditions : Prop)
    (winsOf : World → ℝ) (modelWins : ∀ queries, Model queries → ℝ) (calls : ℕ) : Prop :=
  conditions → ∀ world, ∃ model : Model calls,
    winsOf world ≤ modelWins calls model + duplexError calls

/-- A point of the collision model: a list of at most `blocks` rate blocks. -/
abbrev BlockList (blocks : ℕ) := {list : List (Fin rate → F) // list.length ≤ blocks}

noncomputable instance (blocks : ℕ) : Fintype (BlockList blocks) :=
  (List.finite_length_le _ blocks).fintype

/-- A digest of the collision model: the first `digestLen` lanes. -/
abbrev Digest := Fin digestLen → F

/-- An adversary of the collision model: at most `queries` queries to a uniform function from
block lists to digests, and two block lists as output. -/
abbrev CollisionAdversary (blocks queries : ℕ) :=
  {adversary : RandomOracle.OracleComp (BlockList blocks) Digest
    (BlockList blocks × BlockList blocks) // adversary.QueryBound queries}

/-- The chance that the adversary outputs two different block lists with one digest. -/
noncomputable def collisionChance {blocks queries : ℕ}
    (adversary : CollisionAdversary blocks queries) : ℝ :=
  𝔼 oracle : BlockList blocks → Digest,
    if (adversary.1.run oracle).1 ≠ (adversary.1.run oracle).2 ∧
      oracle (adversary.1.run oracle).1 = oracle (adversary.1.run oracle).2 then 1 else 0

/-- The birthday bound for `queries` queries and the two output points: `C(queries + 2, 2)/q⁴`. -/
noncomputable def birthdayBound (queries : ℕ) : ℝ :=
  ((queries + 2).choose 2 : ℝ) / (fieldModulus : ℝ) ^ digestLen

theorem collisionChance_le {blocks queries : ℕ} (adversary : CollisionAdversary blocks queries) :
    collisionChance adversary ≤ birthdayBound queries := by
  have card : (Fintype.card Digest : ℝ) = (fieldModulus : ℝ) ^ digestLen := by
    simp [Digest, Fintype.card_fin]
  unfold collisionChance birthdayBound
  rw [← card]
  convert RandomOracle.pair_collision_le adversary.1 adversary.2 using 4

/-- The conditions of `[SPONGE]` for a collision of `Poseidon2.hash`: the hash is the sponge
digest of its `10*` blocks, and two different inputs have two different nonempty lists of full
rate blocks. -/
structure CollisionConditions : Prop where
  digest : ∀ input, Poseidon2.hash input = Sponge.digest (padding.pad input)
  distinct : ∀ {left right : List F}, left ≠ right → padding.pad left ≠ padding.pad right
  full : ∀ input, ∀ block ∈ padding.pad input, block.length = rate
  nonempty : ∀ input, padding.pad input ≠ []

theorem collisionConditions : CollisionConditions :=
  ⟨hash_eq_digest, fun distinct equal => distinct (padding.injective equal), padding.full,
    padding.nonempty⟩

/-- `[SPONGE]` for a Poseidon2 collision finder: under `Transfer` of the collision event with
`CollisionConditions`, a finder `world` with `calls` permutation calls, whose counterpart has
`calls` queries on block lists of at most `blocks` blocks, succeeds with chance at most
`C(calls + 2, 2)/q⁴ + spongeError calls`. -/
theorem collision_le {World : Type uWorld} {blocks calls : ℕ} {winsOf : World → ℝ}
    (premise : Transfer CollisionConditions winsOf
      (fun adversary : CollisionAdversary blocks calls => collisionChance adversary) calls)
    (world : World) :
    winsOf world ≤ birthdayBound calls + spongeError calls :=
  transfer_le premise collisionConditions collisionChance_le world

end NightstreamFPrime.Spec.Sponge
