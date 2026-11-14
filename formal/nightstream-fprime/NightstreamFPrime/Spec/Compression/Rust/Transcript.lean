import NightstreamFPrime.Spec.Compression.Rust.Layout

/-! The order of the messages and draws of Rust `verify` (`crates/neo-compress/src/verifier.rs`,
SPEC.md section 5.4, rows 4 to 17), and its match with `Verifier.game`.

`labels n k` lists what `verify` absorbs or draws, in order: the statement, `W`, the lookup
challenges `τ`, `V`, the challenges `(ω, δ, γ, μ, ψ, ξ, λ, β)`, then `n` rounds of the first
sum-check with their coordinates of `ρ`, the values at `ρ`, the column point, then `k` rounds of
the key sum-check with their coordinates, and the key value. Each path of `game` receives and draws
exactly these, with these types, until it rejects or accepts at the end (`game_follows`). `calls`
gives the Poseidon2 transcript operations of each label. A Rust test replays a real proof in this
order and checks each challenge of `verify` (`crates/neo-compress/tests/unit/transcript_order.rs`).
Lean does not model the bytes of a message. -/

namespace NightstreamFPrime.Spec.Compression.Rust

open NightstreamFPrime.Spec.Compression NightstreamFPrime.Spec.Compression.Game
open NightstreamFPrime.Spec.Compression.Verifier

/-- A message of the prover or a draw of the verifier. -/
inductive Move where
  | recv
  | draw
  deriving DecidableEq

/-- What `verify` absorbs or draws. -/
inductive Label where
  | statement
  | witness
  | tau
  | inverse
  | challenges
  | round
  | rho
  | values
  | column
  | keyRound
  | zeta
  | keyValue
  deriving DecidableEq

def Label.move : Label → Move
  | .tau | .challenges | .rho | .column | .zeta => .draw
  | _ => .recv

def Label.name : Label → String
  | .statement => "statement"
  | .witness => "witness"
  | .tau => "tau"
  | .inverse => "inverse"
  | .challenges => "challenges"
  | .round => "round"
  | .rho => "rho"
  | .values => "values"
  | .column => "column"
  | .keyRound => "key_round"
  | .zeta => "zeta"
  | .keyValue => "key_value"

/-- The labels of `verify` with `n` rounds of the first sum-check and `k` of the key sum-check. -/
def labels (n k : ℕ) : List Label :=
  [.statement, .witness, .tau, .inverse, .challenges] ++
    (List.replicate n [Label.round, .rho]).flatten ++ [.values, .column] ++
    (List.replicate k [Label.keyRound, .zeta]).flatten ++ [.keyValue]

section Game

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K]
  [Algebra F K] {P : Params F K}

/-- The type of what `game` receives or draws at each label. -/
def labelType (vf : VerifierData P) : Label → Type
  | .statement => Statement P
  | .witness => ℕ → F
  | .tau => Fin P.lookups → F
  | .inverse => ℕ → F
  | .challenges => Challenges P
  | .round => Fin (3 + 1) → K
  | .rho => K
  | .values => Values K
  | .column => Fin vf.columnVars → K
  | .keyRound => Fin (2 + 1) → K
  | .zeta => K
  | .keyValue => K

end Game

/-- `g` receives and draws the types of `steps` in order; at its end with `steps` left, `k` holds. -/
def FollowsK {α : Type} : Game α → List (Move × Type) → (α → List (Move × Type) → Prop) → Prop
  | .pure a, steps, k => k a steps
  | @Game.recv _ M _ next, (.recv, M') :: steps, k => M = M' ∧ ∀ m, FollowsK (next m) steps k
  | @Game.draw _ C _ _ _ next, (.draw, C') :: steps, k => C = C' ∧ ∀ c, FollowsK (next c) steps k
  | _, _, _ => False

theorem followsK_bind {α β : Type} (g : Game α) (f : α → Game β)
    (steps : List (Move × Type)) (k : β → List (Move × Type) → Prop) :
    FollowsK (Game.bind g f) steps k ↔ FollowsK g steps (fun a rest => FollowsK (f a) rest k) := by
  induction g generalizing steps with
  | pure a => exact Iff.rfl
  | draw C bad next ih =>
      match steps with
      | [] => exact Iff.rfl
      | (.recv, _) :: _ => exact Iff.rfl
      | (.draw, C') :: rest =>
          exact and_congr Iff.rfl (forall_congr' fun c => ih c rest)
  | recv M next ih =>
      match steps with
      | [] => exact Iff.rfl
      | (.draw, _) :: _ => exact Iff.rfl
      | (.recv, M') :: rest =>
          exact and_congr Iff.rfl (forall_congr' fun m => ih m rest)

/-- A sum-check of `rounds` rounds receives a round and draws a coordinate in each round. It ends
early only in `none`, and after the last round in a point and a claim. -/
theorem sumcheck_followsK {K : Type} [Field K] [Fintype K] [DecidableEq K] (d : ℕ)
    (round : List K → Polynomial K)
    (rest : List (Move × Type)) (k : Option (List K × K) → List (Move × Type) → Prop)
    (early : ∀ steps, k none steps) (done : ∀ point claim, k (some (point, claim)) rest) :
    ∀ (rounds : ℕ) (point : List K) (claim : K),
      FollowsK (sumcheck d round rounds point claim)
        ((List.replicate rounds [((Move.recv, Fin (d + 1) → K) : Move × Type), (.draw, K)]).flatten ++
          rest) k
  | 0, point, claim => done point claim
  | rounds + 1, point, claim => by
      refine ⟨rfl, fun message => ?_⟩
      dsimp only
      split
      · exact ⟨rfl, fun c => sumcheck_followsK d round rest k early done rounds _ _⟩
      · exact early _

section Game

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K]
  [Algebra F K] {P : Params F K}

/-- Each path of `game` receives and draws the labels of `verify` in order, with their types,
until it rejects, and it accepts only after the last label. -/
theorem game_follows (vf : VerifierData P) :
    FollowsK (game vf)
      ((labels P.witnessVars (P.laneVars + P.keyBlockVars)).map fun l => (l.move, labelType vf l))
      (fun r rest => r.isSome → rest = []) := by
  simp only [labels, List.map_append, List.map_cons, List.map_nil, List.map_flatten,
    List.map_replicate, List.append_assoc, List.cons_append, List.nil_append]
  refine ⟨rfl, fun st => ⟨rfl, fun W => ⟨rfl, fun τ => ⟨rfl, fun V => ⟨rfl, fun ch => ?_⟩⟩⟩⟩⟩
  refine (followsK_bind _ _ _ _).mpr (sumcheck_followsK 3 _ _ _ (fun steps => ?_)
    (fun ρ value => ?_) _ _ _)
  · intro accepted
    cases accepted
  refine ⟨rfl, fun vals => ⟨rfl, fun col => ?_⟩⟩
  refine (followsK_bind _ _ _ _).mpr (sumcheck_followsK 2 _ _ _ (fun steps => ?_)
    (fun kp kfinal => ?_) _ _ _)
  · intro accepted
    cases accepted
  exact ⟨rfl, fun kv => fun _ => rfl⟩

end Game

/-- A Poseidon2 transcript operation of `verify`: absorb `words` words in one call, absorb a byte
string (its length, then 7 bytes a word) in one call, or draw `count` challenges of `K`. -/
inductive Call where
  | absorb (words : ℕ)
  | bytes
  | draw (count : ℕ)

def Call.isDraw : Call → Bool
  | .draw _ => true
  | _ => false

/-- The sizes of a statement that the transcript reads: the public cells, the point variables and
the matrices of each claim, the claims with a table, and the variables of the span, of `W` and of
the blocks of the key polynomial. -/
structure Shape where
  claims : List (ℕ × ℕ × ℕ)
  tables : ℕ
  spanVars : ℕ
  witnessVars : ℕ
  keyBlockVars : ℕ

/-- The transcript operations of each label (SPEC.md section 5.4), with `κ = 18`, `D = 64`, `L = 4`
and two column variables: the domain, the key and each claim; the commitments; `L/2` draws for
`τ`; `4 + L + 2·tables + span` draws; 4 coefficients a round; `W(ρ)`, the four columns and the key
part; 2 draws; 3 coefficients a key round; the key value. -/
def calls (shape : Shape) : Label → List Call
  | .statement => .absorb 12 :: .bytes :: shape.claims.map fun claim =>
      .absorb (5 + 18 * 64 + claim.1 + 2 * (claim.2.1 + 64 + claim.2.2 * 64))
  | .witness => [.bytes]
  | .tau => [.draw 2]
  | .inverse => [.bytes]
  | .challenges => [.draw (4 + 4 + 2 * shape.tables + shape.spanVars)]
  | .round => [.absorb 8]
  | .rho => [.draw 1]
  | .values => [.absorb 2, .absorb 8, .absorb 2]
  | .column => [.draw 2]
  | .keyRound => [.absorb 6]
  | .zeta => [.draw 1]
  | .keyValue => [.absorb 2]

/-- The calls of a label draw exactly when the game draws at that label. -/
theorem calls_draw (shape : Shape) (l : Label) :
    (calls shape l).all Call.isDraw = (l.move == .draw) ∧
      (calls shape l).all (fun c => !c.isDraw) = (l.move == .recv) := by
  cases l <;> simp [calls, Call.isDraw, Label.move]

/-- The schedule of `verify` for `shape`: each label with its calls. -/
def schedule (shape : Shape) : List (Label × List Call) :=
  (labels shape.witnessVars (laneVars + shape.keyBlockVars)).map fun l => (l, calls shape l)

end NightstreamFPrime.Spec.Compression.Rust
