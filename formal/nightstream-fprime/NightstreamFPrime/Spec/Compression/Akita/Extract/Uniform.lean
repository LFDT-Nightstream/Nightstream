import NightstreamFPrime.Spec.Compression.Akita.FoldCheck
import NightstreamFPrime.Spec.Compression.Akita.Extract.Staged

/-! The verifier of the checks of one fold does not read the witness (Akita, ePrint 2026/1983,
revision of 2026-11-23, Figure 6: the verifier reads the messages of the proof and its
challenges; the witness is in the bad sets of Lemma 10.42 only).

- `bare` is a game with each bad set empty: the messages, the draws and the result of the
  verifier.
- `ended_of_bare` states that two games with one `bare` end in the same result after the same
  transcript.
- `foldCheck_bare` states that `foldCheck` has the same `bare` for each witness. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Extract

open NightstreamFPrime.Spec.Compression.Lib NightstreamFPrime.Spec.Compression.Game Polynomial
open NightstreamFPrime.Spec.Compression (Game)

variable {α β : Type}

/-- The game with each bad set empty. -/
def bare : Game α → Game α
  | .pure result => .pure result
  | @Game.draw _ C finite nonempty _ next =>
      @Game.draw _ C finite nonempty ∅ fun challenge => bare (next challenge)
  | @Game.recv _ M nonempty next => @Game.recv _ M nonempty fun message => bare (next message)

theorem bare_bind (game : Game α) (after : α → Game β) :
    bare (game.bind after) = (bare game).bind fun result => bare (after result) := by
  induction game with
  | pure result => rfl
  | draw C bad next induction =>
      exact congrArg (Game.draw C ∅) (funext fun challenge => induction challenge)
  | recv M next induction =>
      exact congrArg (Game.recv M) (funext fun message => induction message)

theorem bare_draw_congr {C : Type} [Fintype C] [Nonempty C] {bad other : Finset C}
    {next after : C → Game α} (same : ∀ challenge, bare (next challenge) = bare (after challenge)) :
    bare (.draw C bad next) = bare (.draw C other after) :=
  congrArg (Game.draw C ∅) (funext same)

theorem bare_recv_congr {M : Type} [Nonempty M] {next after : M → Game α}
    (same : ∀ message, bare (next message) = bare (after message)) :
    bare (.recv M next) = bare (.recv M after) :=
  congrArg (Game.recv M) (funext same)

theorem bare_bind_congr {game other : Game α} {after later : α → Game β}
    (same : bare game = bare other) (each : ∀ result, bare (after result) = bare (later result)) :
    bare (game.bind after) = bare (other.bind later) := by
  rw [bare_bind, bare_bind, same]
  exact congrArg _ (funext each)

theorem eq_pure_of_bare {game : Game α} {result : α} (ended : bare game = .pure result) :
    game = .pure result := by
  cases game with
  | pure other => exact ended
  | draw C bad next =>
      exact nomatch (ended : Game.draw C ∅ (fun challenge => bare (next challenge)) = .pure result)
  | recv M next =>
      exact nomatch (ended : Game.recv M (fun message => bare (next message)) = .pure result)

section Position

variable {Wire Reply : Type} [Fintype Reply] (read : Reader Wire)

theorem settle_bare (wire : Wire) (game : Game α) (index : ℕ) :
    settle read wire (bare game) index = bare (settle read wire game index) := by
  induction game generalizing index with
  | pure result => rfl
  | draw C bad next _ => rfl
  | recv M next induction => exact induction _ _

theorem advance_bare (reply : Reply) (game : Game α) :
    advance reply (bare game) = bare (advance reply game) := by
  cases game with
  | pure result => rfl
  | draw C bad next => rfl
  | recv M next => rfl

theorem position_bare (start : Game α) (first : Wire) (moves : List (Reply × Wire)) :
    position read (bare start) first moves = bare (position read start first moves) := by
  have step : ∀ (moves : List (Reply × Wire)) (game : Game α),
      moves.foldl (fun game move => settle read move.2 (advance move.1 game) 0) (bare game) =
        bare (moves.foldl (fun game move => settle read move.2 (advance move.1 game) 0)
          game) := by
    intro moves
    induction moves with
    | nil => exact fun _ => rfl
    | cons move rest induction =>
        intro game
        rw [List.foldl_cons, List.foldl_cons, advance_bare, settle_bare]
        exact induction _
  unfold position
  rw [settle_bare]
  exact step moves _

/-- Two games with one `bare` end in the same result after the same transcript. -/
theorem ended_of_bare {start other : Game α} (same : bare start = bare other) {first : Wire}
    {moves : List (Reply × Wire)} {result : α}
    (ended : position read start first moves = .pure result) :
    position read other first moves = .pure result := by
  refine eq_pure_of_bare ?_
  rw [← position_bare, ← same, position_bare, ended]
  rfl

end Position

section Checks

variable {K : Type} [Field K] [Fintype K] [DecidableEq K]

theorem sumcheck_bare (d : ℕ) (round other : List K → K[X]) :
    ∀ (rounds : ℕ) (point : List K) (claim : K),
      bare (sumcheck d round rounds point claim) = bare (sumcheck d other rounds point claim) := by
  intro rounds
  induction rounds with
  | zero => exact fun _ _ => rfl
  | succ rounds induction =>
      intro point claim
      refine bare_recv_congr fun message => ?_
      by_cases passed : (coeffPoly message).eval 0 + (coeffPoly message).eval 1 = claim
      · rw [ite_eq_left passed, ite_eq_left passed]
        exact bare_draw_congr fun challenge => induction _ _
      · rw [ite_eq_right passed, ite_eq_right passed]

theorem sumcheckWith_bare (d : ℕ) (round other : List K → K[X])
    (extra more : List K → Finset K) :
    ∀ (rounds : ℕ) (point : List K) (claim : K),
      bare (sumcheckWith d round extra rounds point claim) =
        bare (sumcheckWith d other more rounds point claim) := by
  intro rounds
  induction rounds with
  | zero => exact fun _ _ => rfl
  | succ rounds induction =>
      intro point claim
      refine bare_recv_congr fun message => ?_
      by_cases passed : (coeffPoly message).eval 0 + (coeffPoly message).eval 1 = claim
      · rw [ite_eq_left passed, ite_eq_left passed]
        exact bare_draw_congr fun challenge => induction _ _
      · rw [ite_eq_right passed, ite_eq_right passed]

end Checks

variable {E : Type} [Field E] [Fintype E] [DecidableEq E] [Algebra Zq E]

theorem fusedCheck_bare (μ : ℕ) (support : ℕ → Prop) [DecidablePred support] (w other : Table)
    (weight : ℕ → E) (target : E) (r : List E) (sClaim ζ γ : E) :
    bare (fusedCheck μ support w weight target r sClaim ζ γ) =
      bare (fusedCheck μ support other weight target r sClaim ζ γ) :=
  bare_bind_congr (sumcheck_bare 3 _ _ μ [] _) fun _ => rfl

/-- The verifier of the range tree is the same for each witness, when the verifier of its last
stage is. -/
theorem rangeTree_bare {β : Type} (leaf μ : ℕ) (w other : Table)
    (last more : ℕ → (ℕ → E) → List E → E → Game (Option β))
    (same : ∀ nodes weights point claim,
      bare (last nodes weights point claim) = bare (more nodes weights point claim)) :
    ∀ (arities : List ℕ) (nodes : ℕ) (weights : ℕ → E) (point : List E) (claim : E),
      bare (rangeTree leaf μ w last arities nodes weights point claim) =
        bare (rangeTree leaf μ other more arities nodes weights point claim) := by
  intro arities
  induction arities with
  | nil => exact same
  | cons a others step =>
      intro nodes weights point claim
      refine bare_bind_congr (sumcheck_bare _ _ _ μ [] claim) fun result => ?_
      rcases result with _ | ⟨r, value⟩
      · rfl
      · refine bare_recv_congr fun child => ?_
        by_cases checked : value = mle r (eqAt point) * ∑ m ∈ Finset.range nodes,
            weights m * ∏ t ∈ Finset.range a, entry child (m * a + t)
        · rw [ite_eq_left checked, ite_eq_left checked]
          exact bare_draw_congr fun γ => step _ _ _ _
        · rw [ite_eq_right checked, ite_eq_right checked]

/-- The verifier of `rangeCheck` is the same for each witness. -/
theorem rangeCheck_bare (b μ : ℕ) (support : ℕ → Prop) [DecidablePred support]
    (w other : Table) (τ0 : List E) :
    bare (rangeCheck b μ support w τ0) = bare (rangeCheck b μ support other τ0) := by
  refine rangeTree_bare (rangeLeaf b) μ w other _ _ (fun nodes weights point claim =>
    bare_bind_congr (sumcheckWith_bare _ _ _ _ _ μ [] claim) fun result => ?_) _ _ _ _ _
  rcases result with _ | ⟨r, value⟩ <;> rfl

omit [DecidableEq E] [Algebra Zq E] in
theorem drawBin_bare_congr {β : Type} (bin : Bool) {bad other : Finset E}
    {next after : E → Game β} (same : ∀ ζ, bare (next ζ) = bare (after ζ)) :
    bare (drawBin bin bad next) = bare (drawBin bin other after) := by
  cases bin
  · exact same 0
  · exact bare_draw_congr same

/-- The verifier of `foldCheck` is the same for each witness: the witness is in the bad sets
only. -/
theorem foldCheck_bare (R : Reduction E) (b : ℕ) (support : ℕ → Prop) [DecidablePred support]
    (bin : Bool) (w other : Table) :
    bare (foldCheck R b support bin w) = bare (foldCheck R b support bin other) := by
  unfold foldCheck
  refine bare_draw_congr fun α => bare_draw_congr fun τ0 => bare_draw_congr fun τ1 =>
    bare_bind_congr (rangeCheck_bare b R.vars support w other _) fun result => ?_
  rcases result with _ | ⟨r, sClaim⟩
  · rfl
  · exact drawBin_bare_congr bin fun ζ => bare_draw_congr fun γ =>
      fusedCheck_bare R.vars support w other _ _ r sClaim ζ γ

end NightstreamFPrime.Spec.Compression.Akita.Extract
