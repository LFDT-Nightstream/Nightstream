import NightstreamFPrime.Spec.Compression.Akita.Extract.CheckComplete
import NightstreamFPrime.Spec.Compression.Akita.Extract.PresentPage

/-! The number of the draws of a path on which a staged game runs (Akita, ePrint 2026/1983,
revision of 2026-11-23, (189): the draws of one fold in their order).

- `StagedTo.runs_bind`: when a game with `StagedTo` runs with a continuation on a path, the game
  ends in a result whose end condition holds for the entries of the costs after its draws, and
  the continuation runs on the rest of the path.
- `StagedTo.draws_le`: a game whose accepted ends have no entry of the costs left, before a
  continuation that ends in `none` after `none`, runs to `some` only on a path with one draw or
  more for each entry.
- `foldCheckSq_drawn`: `foldCheckSq` before such a continuation runs to `some` only on a path
  with `foldDrawsSq` draws or more. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Extract

open NightstreamFPrime.Spec.Compression.Lib NightstreamFPrime.Spec.Compression.Game
open NightstreamFPrime.Spec.Compression (Game)

section Draws

variable {α β : Type} {Reply : Type} [Fintype Reply]

/-- A run of a game with `StagedTo` and a continuation: the game ends in a result `a`, the
continuation of `a` runs on the rest of the path, and the end condition of `a` holds for the
entries of the costs after the draws of the game. -/
theorem StagedTo.runs_bind {game : Game α} {after : α → Game β} {costs : List ℝ}
    {tail : α → List ℝ → Prop} (staged : StagedTo Reply game costs tail) {path : List Move}
    {b : β} (runs : (game.bind after).Runs path b) :
    ∃ (a : α) (rest : List Move), (after a).Runs rest b ∧ countDraws rest ≤ countDraws path ∧
      tail a (costs.drop (countDraws path - countDraws rest)) := by
  induction game generalizing costs path with
  | pure a => exact ⟨a, path, runs, le_rfl, by rw [Nat.sub_self]; exact staged⟩
  | draw C bad next step =>
      cases path with
      | nil => exact False.elim runs
      | cons m path =>
          cases m with
          | draw c =>
              obtain ⟨same, runs⟩ := runs
              obtain ⟨a, rest, ran, le, ended⟩ := step _ (staged.2 _) runs
              refine ⟨a, rest, ran, Nat.le_succ_of_le le, ?_⟩
              have count : countDraws (Move.draw c :: path) - countDraws rest =
                  countDraws path - countDraws rest + 1 := by
                show countDraws path + 1 - countDraws rest = _
                omega
              have dropped : costs.tail.drop (countDraws path - countDraws rest) =
                  costs.drop (countDraws path - countDraws rest + 1) := by
                cases costs <;> simp
              rw [count, ← dropped]
              exact ended
          | recv m => exact False.elim runs
  | recv M next step =>
      cases path with
      | nil => exact False.elim runs
      | cons m path =>
          cases m with
          | draw c => exact False.elim runs
          | recv m =>
              obtain ⟨same, runs⟩ := runs
              have ended := step _ (staged _) runs
              exact ended

/-- A game whose accepted ends have no entry of `costs` left, before a continuation that ends in
`none` after `none`, runs to `some` only on a path with `costs.length` draws or more. -/
theorem StagedTo.draws_le {game : Game (Option α)} {after : Option α → Game (Option β)}
    {costs : List ℝ}
    (staged : StagedTo Reply game costs fun result rest => result = none ∨ rest = [])
    {path : List Move} {b : β} (runs : (game.bind after).Runs path (some b))
    (ends : after none = .pure none) : costs.length ≤ countDraws path := by
  obtain ⟨a, rest, ran, le, ended⟩ := staged.runs_bind runs
  rcases ended with rfl | dropped
  · rw [ends] at ran
    cases rest with
    | nil =>
        have equal : (none : Option β) = some b := ran
        cases equal
    | cons m rest => cases m <;> exact False.elim ran
  · have short := List.drop_eq_nil_iff.mp dropped
    omega

end Draws

section Fold

variable {E : Type} [Field E] [Fintype E] [DecidableEq E] [Algebra Zq E]

/-- `foldCheckSq`, before a continuation that ends in `none` after `none`, runs to `some` only on
a path with `foldDrawsSq` draws or more: the draws `α`, `τ_0`, `τ_1`, the draws of the range tree
with `ζ_norm`, `η_norm`, `ζ_bin`, `γ` and the rounds of the fused sum-check. -/
theorem foldCheckSq_drawn {R : Reduction E} {base : ℕ} {support : ℕ → Prop}
    {decidable : DecidablePred support} {bin : Bool} {limit : ℕ} {kernel : ℕ → ℕ → Zq}
    {w : Table} {β : Type} {after : Option (List E × E) → Game (Option β)} {path : List Move}
    {b : β}
    (runs : ((foldCheckSq R base support bin limit kernel w).bind after).Runs path (some b))
    (ends : after none = .pure none) : foldDrawsSq R.vars base bin ≤ countDraws path := by
  have staged := foldCheckSq_stagedTo (Slot := E) (Coord := Fin (R.vars + R.batch + 1)) R base
    support bin limit kernel w (by rw [Fintype.card_fin]; omega) (by rw [Fintype.card_fin]; omega)
    [] (fun result rest => result = none ∨ rest = []) (fun _ => Or.inl rfl) (fun _ => Or.inr rfl)
  rw [List.append_nil] at staged
  have drawn := staged.draws_le runs ends
  rwa [foldCostsSq_length] at drawn

end Fold

end NightstreamFPrime.Spec.Compression.Akita.Extract
