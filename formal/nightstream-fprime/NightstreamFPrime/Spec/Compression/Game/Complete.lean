import NightstreamFPrime.Spec.Compression.Game.SumCheck

/-! The honest prover of a public-coin game (Akita, ePrint 2026/1983, revision of 2026-10-15,
Theorem 10.9: "All honest algebraic checks are exact identities").

- `WinsTo fits game draws fresh tail` states that a prover has a message for each draw sequence
  such that the game ends with `tail`. The game has at most `draws` draws on these paths. `fresh`
  states that the prover can send a message now: the game receives at most one message between
  two draws, and the type of each message satisfies `fits`.
- `honest` is a strategy of these messages, and `prob_honest` states that it reaches the end with
  probability 1.
- `sumcheck_winsTo` is the completeness of the sum-check (Akita Theorem 3.3): the true round
  polynomials pass each round of a true claim, and the final claim is the true value. -/

namespace NightstreamFPrime.Spec.Compression.Game

open scoped BigOperators
open NightstreamFPrime.Spec.Compression.Lib Polynomial

variable {α β : Type} (fits : Type → Prop)

/-- A prover has a message for each draw sequence such that the game ends in a result, a number
of remaining draws and a state `fresh` that satisfy `tail`. A draw uses one of the `draws` draws
and makes `fresh` true. A message needs `fresh` and a type that satisfies `fits`, and makes
`fresh` false. -/
def WinsTo : Game α → ℕ → Bool → (α → ℕ → Bool → Prop) → Prop
  | pure result, draws, fresh, tail => tail result draws fresh
  | draw _ _ _, 0, _, _ => False
  | draw _ _ next, rest + 1, _, tail => ∀ challenge, WinsTo (next challenge) rest true tail
  | recv M next, draws, fresh, tail =>
      fresh = true ∧ fits M ∧ ∃ message, WinsTo (next message) draws false tail

theorem winsTo_bind (game : Game α) (after : α → Game β) (draws : ℕ) (fresh : Bool)
    (tail : β → ℕ → Bool → Prop) :
    WinsTo fits (game.bind after) draws fresh tail ↔
      WinsTo fits game draws fresh fun result rest ready =>
        WinsTo fits (after result) rest ready tail := by
  induction game generalizing draws fresh with
  | pure result => exact Iff.rfl
  | draw C bad next induction =>
      cases draws with
      | zero => exact Iff.rfl
      | succ rest => exact forall_congr' fun challenge => induction challenge rest true
  | recv M next induction =>
      exact and_congr_right fun _ => and_congr_right fun _ =>
        exists_congr fun message => induction message draws false

open Classical in
/-- The strategy that sends a message after which the prover has a message for each draw
sequence, when one exists. -/
noncomputable def honest (tail : α → ℕ → Bool → Prop) : Strategy α := fun M _ next =>
  if found : ∃ message draws, WinsTo fits (next message) draws false tail then found.choose
  else Classical.arbitrary M

/-- Perfect completeness of a game with `WinsTo`: the strategy `honest` reaches an end of `tail`
with probability 1. -/
theorem prob_honest {game : Game α} {draws : ℕ} {fresh : Bool} {tail : α → ℕ → Bool → Prop}
    {goal : α → Prop} (wins : WinsTo fits game draws fresh tail)
    (ends : ∀ result rest ready, tail result rest ready → goal result) :
    prob (honest fits tail) game goal = 1 := by
  classical
  induction game generalizing draws fresh with
  | pure result =>
      simp only [prob]
      rw [ite_eq_left (ends result draws fresh wins)]
  | draw C bad next induction =>
      cases draws with
      | zero => exact wins.elim
      | succ rest =>
          show (𝔼 challenge, prob (honest fits tail) (next challenge) goal) = 1
          rw [Finset.expect_congr rfl fun challenge _ => induction challenge (wins challenge)]
          exact Finset.expect_const Finset.univ_nonempty 1
  | recv M next induction =>
      obtain ⟨_, _, message, won⟩ := wins
      have found : ∃ message draws, WinsTo fits (next message) draws false tail :=
        ⟨message, draws, won⟩
      show prob (honest fits tail) (next (honest fits tail M next)) goal = 1
      have chosen : honest fits tail M next = found.choose := dite_eq_left found
      rw [chosen]
      exact induction _ found.choose_spec.choose_spec

variable {K : Type} [Field K] [Fintype K] [DecidableEq K]

/-- Akita Theorem 3.3, completeness: for a true claim, the coefficients of the true round
polynomials pass each round, and the final claim is the true value at the final point. `rest` is
the number of the draws after the sum-check. -/
theorem sumcheck_winsTo {d bound : ℕ} {truth : List K → K} {round : List K → K[X]}
    (true_rounds : RoundTruth d bound truth round) (message : fits (Fin (d + 1) → K)) (rest : ℕ)
    (tail : Option (List K × K) → ℕ → Bool → Prop) :
    ∀ (rounds : ℕ) (point : List K) (claim : K), point.length + rounds ≤ bound →
      claim = truth point →
      (∀ final : List K, final.length = point.length + rounds →
        tail (some (final, truth final)) rest true) →
      WinsTo fits (sumcheck d round rounds point claim) (rest + rounds) true tail := by
  intro rounds
  induction rounds with
  | zero =>
      intro point claim _ true_claim ended
      rw [true_claim]
      exact ended point rfl
  | succ rounds step =>
      intro point claim inside true_claim ended
      have within : point.length < bound := by omega
      refine ⟨rfl, message, fun index => (round point).coeff index, ?_⟩
      dsimp only
      rw [coeffPoly_coeff (round point) (true_rounds.degree point within),
        ite_eq_left (true_claim ▸ true_rounds.halves point within)]
      intro challenge
      show WinsTo fits (sumcheck d round rounds (point ++ [challenge])
        ((round point).eval challenge)) (rest + rounds) true tail
      rw [true_rounds.binds point challenge within]
      exact step (point ++ [challenge]) _ (by simp; omega) rfl fun final length =>
        ended final (by simp at length; omega)

end NightstreamFPrime.Spec.Compression.Game
