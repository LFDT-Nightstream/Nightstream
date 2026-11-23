import NightstreamFPrime.Spec.Compression.Game.Game
import NightstreamFPrime.Spec.RandomOracle

/-! A public-coin game as the stages of a Fiat–Shamir transcript (Akita, ePrint 2026/1983,
revision of 2026-11-23, Definition 10.16 (vii): the bad sets `Bad_j(ξ; w)` of the stages of a
checkpoint segment).

- `Fair decode` states that `decode` maps a uniform answer of the oracle to a challenge with the
  uniform law. `sample` is one such function for each challenge type that has one. `fair_slot` and
  `fair_slots` give one for a field element and for a vector of field elements.
- `settle` gives the messages of the proof to a game up to its next draw, `advance` gives one
  answer of the oracle to this draw, and `position` is the game after a transcript prefix.
- `badReplies` is the set of the answers that the draw of a game decodes to its bad set.
- `StagedTo game costs` states that draw `j` of each path of the game has a bad set of
  probability at most entry `j` of `costs`. `Staged` is this statement with no condition at the
  end of the game. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Extract

open scoped BigOperators
open NightstreamFPrime.Spec.RandomOracle (mass)
open NightstreamFPrime.Spec.Compression (Game)

attribute [local instance low] Classical.propDecidable

section Sampler

/-- The expectation of the indicator of a set is its density. -/
theorem expect_mem {C : Type} [Fintype C] [DecidableEq C] (bad : Finset C) :
    (𝔼 c, (if c ∈ bad then (1 : ℝ) else 0)) = (bad.card : ℝ) / Fintype.card C := by
  rw [Finset.expect_eq_sum_div_card, Finset.sum_boole, Finset.card_univ]
  congr 2
  simp

/-- The entries of a uniform tuple at the coordinates `place i`, for an injective `place`, are a
uniform tuple. -/
theorem expect_restrict {Coord Slot : Type} [Fintype Coord] [DecidableEq Coord] [Fintype Slot]
    [Nonempty Slot] {count : ℕ} (place : Fin count → Coord)
    (injective : Function.Injective place) (value : (Fin count → Slot) → ℝ) :
    𝔼 reply : Coord → Slot, value (fun index => reply (place index)) = 𝔼 entries, value entries := by
  let split := Equiv.piEquivPiSubtypeProd (fun coordinate => coordinate ∈ Set.range place)
    (fun _ => Slot)
  let name : Fin count ≃ {coordinate // coordinate ∈ Set.range place} :=
    Equiv.ofInjective place injective
  calc
    _ = 𝔼 pair : ({coordinate // coordinate ∈ Set.range place} → Slot) ×
          ({coordinate // coordinate ∉ Set.range place} → Slot),
          value (fun index => pair.1 (name index)) :=
      Fintype.expect_equiv split _ _ fun _ => rfl
    _ = 𝔼 inside : {coordinate // coordinate ∈ Set.range place} → Slot,
          value (fun index => inside (name index)) := by
      rw [← Finset.univ_product_univ, Finset.expect_product]
      exact Finset.expect_congr rfl fun inside _ =>
        Fintype.expect_const (value fun index => inside (name index))
    _ = _ := Fintype.expect_equiv (Equiv.arrowCongr name.symm (Equiv.refl Slot)) _ _ fun inside =>
      congrArg value (funext fun index => by simp [Equiv.arrowCongr])

variable {Reply : Type} [Fintype Reply] {C : Type} [Fintype C]

/-- `decode` maps a uniform answer to a value in a set `bad` with probability at most the density
of `bad`. -/
def Fair (decode : Reply → C) : Prop :=
  ∀ bad : Finset C, mass {reply | decode reply ∈ bad} ≤ (bad.card : ℝ) / Fintype.card C

/-- The challenge of type `C` that an answer of the oracle gives: a fixed function with the
property `Fair`, when one exists. -/
noncomputable def sample (Reply : Type) [Fintype Reply] (C : Type) [Fintype C] [Nonempty C] :
    Reply → C :=
  if fair : ∃ decode : Reply → C, Fair decode then fair.choose
  else fun _ => Classical.arbitrary C

theorem sample_fair [Nonempty C] (fair : ∃ decode : Reply → C, Fair decode) :
    Fair (sample Reply C) := by
  unfold sample
  rw [dite_eq_left fair]
  exact fair.choose_spec

/-- One slot of an answer is a uniform field element. -/
theorem fair_slot {Coord Slot : Type} [Fintype Coord] [DecidableEq Coord] [Fintype Slot]
    [Nonempty Slot] (coordinate : Coord) :
    Fair fun reply : Coord → Slot => reply coordinate := by
  intro bad
  have uniform := RandomOracle.expect_eval coordinate fun slot : Slot =>
    if slot ∈ bad then (1 : ℝ) else 0
  rw [expect_mem bad] at uniform
  refine le_of_eq ?_
  unfold mass
  convert uniform
  exact Iff.rfl

/-- The slots of an answer at `count` different coordinates are a uniform vector. -/
theorem fair_slots {Coord Slot : Type} [Fintype Coord] [DecidableEq Coord] [Fintype Slot]
    [Nonempty Slot] {count : ℕ} (place : Fin count → Coord)
    (injective : Function.Injective place) :
    Fair fun (reply : Coord → Slot) (index : Fin count) => reply (place index) := by
  intro bad
  have uniform := expect_restrict place injective fun entries =>
    if entries ∈ bad then (1 : ℝ) else 0
  rw [expect_mem bad] at uniform
  refine le_of_eq ?_
  unfold mass
  convert uniform
  exact Iff.rfl

/-- An answer with a slot gives a uniform field element. -/
theorem exists_fair_slot {Coord Slot : Type} [Fintype Coord] [DecidableEq Coord] [Nonempty Coord]
    [Fintype Slot] [Nonempty Slot] : ∃ decode : (Coord → Slot) → Slot, Fair decode :=
  ⟨_, fair_slot (Classical.arbitrary Coord)⟩

/-- An answer with `count` slots or more gives a uniform vector of `count` field elements. -/
theorem exists_fair_slots {Coord Slot : Type} [Fintype Coord] [DecidableEq Coord] [Fintype Slot]
    [Nonempty Slot] {count : ℕ} (enough : count ≤ Fintype.card Coord) :
    ∃ decode : (Coord → Slot) → Fin count → Slot, Fair decode :=
  ⟨_, fair_slots (fun index => (Fintype.equivFin Coord).symm (Fin.castLE enough index))
    ((Fintype.equivFin Coord).symm.injective.comp (Fin.castLE_injective enough))⟩

end Sampler

/-- The decoder of the messages of a proof. `message` gives message `index` of the type `M` that
a game receives after one draw, from the wire message that the proof has after this draw. `nonce`
gives the nonce of the response search of the prover from the wire message before the fold
challenges of a level (Akita Figure 4, step 2; `FoldResponse` of the Akita code). -/
structure Reader (Wire : Type) : Type 1 where
  message : ∀ (M : Type) [Nonempty M], Wire → ℕ → M
  nonce : Wire → ℕ

/-- The number of the nonces of the response search of a fold: the verifier accepts a nonce
below it (Akita Section 5.4, `N_try,h`; `FOLD_RESPONSE_ATTEMPTS = 2^12` of the Akita code,
`crates/akita-params/src/transcript_grinding.rs`). -/
def responseTries : ℕ := 2 ^ 12

variable {Wire Reply α β : Type} [Fintype Reply]

/-- The game after the messages that it receives before its next draw, read from one wire
message. -/
def settle (read : Reader Wire) (wire : Wire) : Game α → ℕ → Game α
  | .pure result, _ => .pure result
  | @Game.draw _ C finite nonempty bad next, _ => @Game.draw _ C finite nonempty bad next
  | @Game.recv _ M nonempty next, index =>
      settle read wire (next (@Reader.message _ read M nonempty wire index)) (index + 1)

/-- The game after one answer of the oracle to its draw. A game that has ended does not change. -/
noncomputable def advance (reply : Reply) : Game α → Game α
  | @Game.draw _ C finite nonempty _ next => next (@sample Reply _ C finite nonempty reply)
  | game => game

/-- The answers of the oracle that the draw of a game decodes to a challenge of its bad set. -/
noncomputable def badReplies (Reply : Type) [Fintype Reply] : Game α → Set Reply
  | @Game.draw _ C finite nonempty bad _ => {reply | @sample Reply _ C finite nonempty reply ∈ bad}
  | _ => ∅

/-- The game is not at a message of the prover. -/
def Settled : Game α → Prop
  | @Game.recv _ _ _ _ => False
  | _ => True

/-- Draw `j` of each path of the game has a bad set that an answer of the oracle reaches with
probability at most entry `j` of `costs`, and at an end with the result `a` the remaining entries
`rest` satisfy `tail a rest`. -/
def StagedTo (Reply : Type) [Fintype Reply] : Game α → List ℝ → (α → List ℝ → Prop) → Prop
  | .pure result, costs, tail => tail result costs
  | @Game.draw _ C finite nonempty bad next, costs, tail =>
      mass {reply : Reply | @sample Reply _ C finite nonempty reply ∈ bad} ≤ costs.headD 0 ∧
        ∀ challenge, StagedTo Reply (next challenge) costs.tail tail
  | @Game.recv _ _ _ next, costs, tail => ∀ message, StagedTo Reply (next message) costs tail

/-- `StagedTo` with no condition at the end. -/
def Staged (Reply : Type) [Fintype Reply] (game : Game α) (costs : List ℝ) : Prop :=
  StagedTo Reply game costs fun _ _ => True

theorem StagedTo.mono {game : Game α} {costs : List ℝ} {tail other : α → List ℝ → Prop}
    (staged : StagedTo Reply game costs tail) (weaker : ∀ result rest, tail result rest →
      other result rest) : StagedTo Reply game costs other := by
  induction game generalizing costs with
  | pure result => exact weaker result costs staged
  | draw C bad next induction => exact ⟨staged.1, fun challenge => induction challenge (staged.2 challenge)⟩
  | recv M next induction => exact fun message => induction message (staged message)

theorem stagedTo_bind (game : Game α) (after : α → Game β) (costs : List ℝ)
    (tail : β → List ℝ → Prop) :
    StagedTo Reply (game.bind after) costs tail ↔
      StagedTo Reply game costs fun result rest => StagedTo Reply (after result) rest tail := by
  induction game generalizing costs with
  | pure result => exact Iff.rfl
  | draw C bad next induction =>
      exact and_congr_right fun _ => forall_congr' fun challenge => induction challenge _
  | recv M next induction => exact forall_congr' fun message => induction message _

/-- A draw of a challenge type with a fair decoder, with a bad set of density at most `cost`. -/
theorem stagedTo_draw {C : Type} [Fintype C] [Nonempty C]
    (fair : ∃ decode : Reply → C, Fair decode) {bad : Finset C} {next : C → Game α} {cost : ℝ}
    {costs : List ℝ} {tail : α → List ℝ → Prop}
    (density : (bad.card : ℝ) / Fintype.card C ≤ cost)
    (nexts : ∀ challenge, StagedTo Reply (next challenge) costs tail) :
    StagedTo Reply (.draw C bad next) (cost :: costs) tail :=
  ⟨(sample_fair fair bad).trans density, nexts⟩

variable (read : Reader Wire) (wire : Wire)

theorem settle_settled (game : Game α) (index : ℕ) : Settled (settle read wire game index) := by
  induction game generalizing index with
  | pure result => trivial
  | draw C bad next _ => trivial
  | recv M next induction => exact induction _ _

theorem avoids_settle {game : Game α} {goal : α → Prop} (avoids : game.Avoids goal) (index : ℕ) :
    (settle read wire game index).Avoids goal := by
  induction game generalizing index with
  | pure result => exact avoids
  | draw C bad next _ => exact avoids
  | recv M next induction => exact induction _ (avoids _) _

theorem staged_settle {game : Game α} {costs : List ℝ} (staged : Staged Reply game costs)
    (index : ℕ) : Staged Reply (settle read wire game index) costs := by
  induction game generalizing index with
  | pure result => exact staged
  | draw C bad next _ => exact staged
  | recv M next induction => exact induction _ (staged _) _

theorem avoids_advance {game : Game α} {goal : α → Prop} (avoids : game.Avoids goal)
    {reply : Reply} (outside : reply ∉ badReplies Reply game) :
    (advance reply game).Avoids goal := by
  cases game with
  | pure result => exact avoids
  | draw C bad next => exact avoids _ outside
  | recv M next => exact avoids

theorem staged_advance {game : Game α} {costs : List ℝ} (settled : Settled game)
    (staged : Staged Reply game costs) (reply : Reply) :
    Staged Reply (advance reply game) costs.tail := by
  cases game with
  | pure result => trivial
  | draw C bad next => exact staged.2 _
  | recv M next => exact settled.elim

theorem mass_badReplies_le {game : Game α} {costs : List ℝ} (nonnegative : 0 ≤ costs.headD 0)
    (staged : Staged Reply game costs) : mass (badReplies Reply game) ≤ costs.headD 0 := by
  have empty : mass (∅ : Set Reply) = 0 := by simp [mass]
  cases game with
  | pure result => exact empty.trans_le nonnegative
  | draw C bad next => exact staged.1
  | recv M next => exact empty.trans_le nonnegative

/-- The game after a transcript prefix: the message `first`, and for each stage the answer of the
oracle and the next message. -/
noncomputable def position (start : Game α) (first : Wire) (moves : List (Reply × Wire)) :
    Game α :=
  moves.foldl (fun game move => settle read move.2 (advance move.1 game) 0)
    (settle read first start 0)

theorem position_append (start : Game α) (first : Wire) (moves : List (Reply × Wire))
    (reply : Reply) :
    position read start first (moves ++ [(reply, wire)]) =
      settle read wire (advance reply (position read start first moves)) 0 := by
  simp [position, List.foldl_append]

/-- `position_append` for a transcript whose answers the game reads through `look`. -/
theorem position_map_append {Answer : Type} (look : Answer → Reply) (start : Game α)
    (first : Wire) (moves : List (Answer × Wire)) (answer : Answer) :
    position read start first ((moves ++ [(answer, wire)]).map fun move => (look move.1, move.2)) =
      settle read wire (advance (look answer)
        (position read start first (moves.map fun move => (look move.1, move.2)))) 0 := by
  rw [List.map_append, List.map_singleton]
  exact position_append read wire start first _ (look answer)

theorem position_staged {start : Game α} {costs : List ℝ} (staged : Staged Reply start costs)
    (first : Wire) (moves : List (Reply × Wire)) :
    Staged Reply (position read start first moves) (costs.drop moves.length) := by
  have step : ∀ (moves : List (Reply × Wire)) (game : Game α) (costs : List ℝ), Settled game →
      Staged Reply game costs →
      Staged Reply (moves.foldl (fun game move => settle read move.2 (advance move.1 game) 0)
        game) (costs.drop moves.length) := by
    intro moves
    induction moves with
    | nil => exact fun game costs _ staged => by simpa using staged
    | cons move rest induction =>
        intro game costs settled staged
        have next := induction _ costs.tail (settle_settled read move.2 _ 0)
          (staged_settle read move.2 (staged_advance settled staged move.1) 0)
        simpa using next
  exact step moves _ costs (settle_settled read first start 0) (staged_settle read first staged 0)

end NightstreamFPrime.Spec.Compression.Akita.Extract
