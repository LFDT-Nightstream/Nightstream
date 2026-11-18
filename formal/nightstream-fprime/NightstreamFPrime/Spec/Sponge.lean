import Mathlib.Basic.Real.Basic
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Positivity
import NightstreamFPrime.Spec.Poseidon2

/-! The sponge layer of `SECURITY_MODEL.md`, section Assumptions.

`[P2-PERM]` takes `Poseidon2.permute` as a uniform permutation, and `[SPONGE]` is the sponge
indifferentiability theorem with the error `spongeError`. This file states the conditions of
`[SPONGE]` as Lean predicates.

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
  `spongeError calls` less often. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Sponge

universe uWorld uModel

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

end NightstreamFPrime.Spec.Sponge
