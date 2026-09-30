import NightstreamFPrime.Gadgets.Polynomial.Horner
import NightstreamFPrime.Spec.SumCheck.FixedPhase

/-!
Owns the fixed-width SumCheck claimed-chain gadget over the production
quadratic extension. Each extension value is represented by two Goldilocks
expressions in `c0`, `c1` order. The gadget checks only the round recurrence
and final equality; transcript replay and the PiCCS terminal expression have
separate owners.
-/

namespace NightstreamFPrime.Gadgets.SumCheck.FixedChain

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier

/-- One prover polynomial paired with its verifier-owned challenge. -/
structure Round (degree : Nat) where
  coefficient : Fin (degree + 1) → KExpr
  challenge : KExpr

def Round.VarsBelow {degree : Nat} (round : Round degree)
    (bound : Nat) : Prop :=
  (∀ coefficient, (round.coefficient coefficient).VarsBelow bound) ∧
    round.challenge.VarsBelow bound

theorem Round.varsBelow_mono {degree : Nat} (round : Round degree)
    {lower upper : Nat} (below : round.VarsBelow lower)
    (le : lower ≤ upper) : round.VarsBelow upper :=
  ⟨fun coefficient => KExpr.varsBelow_mono _ (below.1 coefficient) le,
    KExpr.varsBelow_mono _ below.2 le⟩

def Round.coefficients {degree : Nat} (round : Round degree) : List KExpr :=
  List.ofFn round.coefficient

def Round.semanticPolynomial {degree : Nat} (env : Env)
    (round : Round degree) :
    NightstreamFPrime.Spec.SumCheck.Finite.FixedPolynomial K degree where
  coefficients := round.coefficients.map (KExpr.eval env)
  coefficients_length := by simp [Round.coefficients]

/-- Constant-first Horner evaluation, identical to the semantic verifier. -/
def evaluateCoefficients (point : KExpr) : List KExpr → KExpr
  | [] => KExpr.zero
  | coefficient :: rest =>
      KExpr.add coefficient (KExpr.mul point (evaluateCoefficients point rest))

theorem eval_evaluateCoefficients (env : Env) (point : KExpr)
    (coefficients : List KExpr) :
    (evaluateCoefficients point coefficients).eval env =
      NightstreamFPrime.Spec.SumCheck.Finite.Message.evaluateCoefficients
        extensionOps.toOps (point.eval env)
        (coefficients.map (KExpr.eval env)) := by
  induction coefficients with
  | nil => rfl
  | cons coefficient coefficients inductionHypothesis =>
      simp [evaluateCoefficients,
        NightstreamFPrime.Spec.SumCheck.Finite.Message.evaluateCoefficients,
        inductionHypothesis, extensionOps]

theorem evaluateCoefficients_varsBelow (point : KExpr)
    (coefficients : List KExpr) (bound : Nat)
    (pointBelow : point.VarsBelow bound)
    (coefficientsBelow : ∀ coefficient ∈ coefficients,
      coefficient.VarsBelow bound) :
    (evaluateCoefficients point coefficients).VarsBelow bound := by
  induction coefficients with
  | nil =>
      exact ⟨trivial, trivial⟩
  | cons coefficient coefficients inductionHypothesis =>
      apply KExpr.add_varsBelow
      · exact coefficientsBelow coefficient (by simp)
      · apply KExpr.mul_varsBelow point
          (evaluateCoefficients point coefficients) bound pointBelow
        apply inductionHypothesis
        intro current member
        exact coefficientsBelow current (by simp [member])

def evaluateRound {degree : Nat} (round : Round degree)
    (point : KExpr) : KExpr :=
  evaluateCoefficients point round.coefficients

theorem eval_evaluateRound {degree : Nat} (env : Env)
    (round : Round degree) (point : KExpr) :
    (evaluateRound round point).eval env =
      (round.semanticPolynomial env).evaluate extensionOps.toOps
        (point.eval env) := by
  exact eval_evaluateCoefficients env point round.coefficients

theorem evaluateRound_varsBelow {degree : Nat} (round : Round degree)
    (point : KExpr) (bound : Nat) (roundBelow : round.VarsBelow bound)
    (pointBelow : point.VarsBelow bound) :
    (evaluateRound round point).VarsBelow bound := by
  apply evaluateCoefficients_varsBelow point round.coefficients bound pointBelow
  intro coefficient member
  rw [Round.coefficients, List.mem_ofFn'] at member
  rcases member with ⟨index, rfl⟩
  exact roundBelow.1 index

/-- Flat equations for the exact claimed chain. -/
def chainConstraints {degree : Nat} (current : KExpr) :
    List (Round degree) → KExpr → List Expr
  | [], terminal => KExpr.equalities current terminal
  | round :: rounds, terminal =>
      KExpr.equalities current
        (KExpr.add (evaluateRound round KExpr.zero)
          (evaluateRound round KExpr.one)) ++
      chainConstraints (evaluateRound round round.challenge) rounds terminal

theorem chainConstraints_length {degree : Nat} (current terminal : KExpr)
    (rounds : List (Round degree)) :
    (chainConstraints current rounds terminal).length =
      2 * (rounds.length + 1) := by
  induction rounds generalizing current with
  | nil => simp [chainConstraints, KExpr.equalities]
  | cons round rounds inductionHypothesis =>
      simp [chainConstraints, KExpr.equalities, inductionHypothesis]
      omega

theorem chainConstraints_varsBelow {degree : Nat}
    (current terminal : KExpr) (rounds : List (Round degree)) (bound : Nat)
    (currentBelow : current.VarsBelow bound)
    (roundsBelow : ∀ round ∈ rounds, round.VarsBelow bound)
    (terminalBelow : terminal.VarsBelow bound) :
    ∀ expression ∈ chainConstraints current rounds terminal,
      expression.VarsBelow bound := by
  induction rounds generalizing current with
  | nil =>
      simpa [chainConstraints] using
        KExpr.equalities_varsBelow current terminal bound currentBelow
          terminalBelow
  | cons round rounds inductionHypothesis =>
      intro expression member
      rw [chainConstraints] at member
      rcases List.mem_append.mp member with headMember | tailMember
      · have roundBelow := roundsBelow round (by simp)
        have rightBelow :
            (KExpr.add (evaluateRound round KExpr.zero)
              (evaluateRound round KExpr.one)).VarsBelow bound :=
          KExpr.add_varsBelow _ _ bound
            (evaluateRound_varsBelow round KExpr.zero bound roundBelow
              ⟨trivial, trivial⟩)
            (evaluateRound_varsBelow round KExpr.one bound roundBelow
              ⟨trivial, trivial⟩)
        exact KExpr.equalities_varsBelow current
          (KExpr.add (evaluateRound round KExpr.zero)
            (evaluateRound round KExpr.one)) bound currentBelow rightBelow
          expression headMember
      · have roundBelow := roundsBelow round (by simp)
        have nextBelow := evaluateRound_varsBelow round round.challenge bound
          roundBelow roundBelow.2
        exact inductionHypothesis (evaluateRound round round.challenge)
          nextBelow
          (fun current currentMember =>
            roundsBelow current (by simp [currentMember]))
          expression tailMember

theorem constraintsHold_append (env : Env) (first second : List Expr) :
    ConstraintsHold env (first ++ second) ↔
      ConstraintsHold env first ∧ ConstraintsHold env second := by
  constructor
  · intro holds
    exact ⟨
      fun expression member => holds expression
        (List.mem_append_left second member),
      fun expression member => holds expression
        (List.mem_append_right first member)⟩
  · rintro ⟨firstHolds, secondHolds⟩ expression member
    rcases List.mem_append.mp member with member | member
    · exact firstHolds expression member
    · exact secondHolds expression member

theorem chainConstraints_hold_iff {degree : Nat} (env : Env)
    (current terminal : KExpr) (rounds : List (Round degree)) :
    ConstraintsHold env (chainConstraints current rounds terminal) ↔
      NightstreamFPrime.Spec.SumCheck.Finite.FixedPhase.Chain
        extensionOps.toOps (current.eval env)
        (rounds.map (Round.semanticPolynomial env))
        (rounds.map fun round => round.challenge.eval env)
        (terminal.eval env) := by
  induction rounds generalizing current with
  | nil =>
      simpa [chainConstraints,
        NightstreamFPrime.Spec.SumCheck.Finite.FixedPhase.Chain] using
        KExpr.equalities_hold_iff env current terminal
  | cons round rounds inductionHypothesis =>
      rw [chainConstraints, constraintsHold_append,
        KExpr.equalities_hold_iff, inductionHypothesis]
      simp only [List.map_cons,
        NightstreamFPrime.Spec.SumCheck.Finite.FixedPhase.Chain]
      simp only [KExpr.eval_add, eval_evaluateRound, KExpr.eval_zero,
        KExpr.eval_one]
      simp [extensionOps]

/-- Fixed finite interface. All expressions are supplied by the parent ABI;
the gadget allocates no challenge or prover-message authority. -/
structure Interface (degree roundCount : Nat) where
  initial : KExpr
  round : Fin roundCount → Round degree
  terminal : KExpr

def Interface.VarsBelow {degree roundCount : Nat}
    (interface : Interface degree roundCount) (bound : Nat) : Prop :=
  interface.initial.VarsBelow bound ∧
    (∀ round, (interface.round round).VarsBelow bound) ∧
    interface.terminal.VarsBelow bound

theorem Interface.varsBelow_mono {degree roundCount : Nat}
    (interface : Interface degree roundCount) {lower upper : Nat}
    (below : interface.VarsBelow lower) (le : lower ≤ upper) :
    interface.VarsBelow upper :=
  ⟨KExpr.varsBelow_mono _ below.1 le,
    fun round => (interface.round round).varsBelow_mono (below.2.1 round) le,
    KExpr.varsBelow_mono _ below.2.2 le⟩

def Interface.rounds {degree roundCount : Nat}
    (interface : Interface degree roundCount) : List (Round degree) :=
  List.ofFn interface.round

def SpecHolds {degree roundCount : Nat}
    (interface : Interface degree roundCount) (env : Env) : Prop :=
  NightstreamFPrime.Spec.SumCheck.Finite.FixedPhase.Chain extensionOps.toOps
    (interface.initial.eval env)
    (interface.rounds.map (Round.semanticPolynomial env))
    (interface.rounds.map fun round => round.challenge.eval env)
    (interface.terminal.eval env)

def Assumptions {degree roundCount : Nat}
    (interface : Interface degree roundCount) (offset : Nat) (_env : Env) :
    Prop :=
  interface.VarsBelow offset

theorem Round.semanticPolynomial_eq_of_agree_below {degree : Nat}
    (round : Round degree) (bound : Nat) (left right : Env)
    (below : round.VarsBelow bound)
    (agrees : ∀ index, index < bound → left index = right index) :
    round.semanticPolynomial left = round.semanticPolynomial right := by
  unfold Round.semanticPolynomial Round.coefficients
  congr 1
  apply List.map_congr_left
  intro value member
  rw [List.mem_ofFn'] at member
  rcases member with ⟨coefficient, rfl⟩
  exact (round.coefficient coefficient).eval_eq_of_agree_below
    bound left right (below.1 coefficient) agrees

theorem specHolds_eq_of_agree_below {degree roundCount : Nat}
    (interface : Interface degree roundCount) (bound : Nat)
    (left right : Env) (below : interface.VarsBelow bound)
    (agrees : ∀ index, index < bound → left index = right index) :
    SpecHolds interface left ↔ SpecHolds interface right := by
  have initial := interface.initial.eval_eq_of_agree_below bound left right
    below.1 agrees
  have terminal := interface.terminal.eval_eq_of_agree_below bound left right
    below.2.2 agrees
  have rounds :
      interface.rounds.map (Round.semanticPolynomial left) =
        interface.rounds.map (Round.semanticPolynomial right) := by
    apply List.map_congr_left
    intro round member
    rw [Interface.rounds, List.mem_ofFn'] at member
    rcases member with ⟨index, rfl⟩
    exact (interface.round index).semanticPolynomial_eq_of_agree_below
      bound left right (below.2.1 index) agrees
  have challenges :
      interface.rounds.map (fun round => round.challenge.eval left) =
        interface.rounds.map (fun round => round.challenge.eval right) := by
    apply List.map_congr_left
    intro round member
    rw [Interface.rounds, List.mem_ofFn'] at member
    rcases member with ⟨index, rfl⟩
    exact (interface.round index).challenge.eval_eq_of_agree_below
      bound left right (below.2.1 index).2 agrees
  unfold SpecHolds
  rw [initial, terminal, rounds, challenges]

def constraints {degree roundCount : Nat}
    (interface : Interface degree roundCount) : List Expr :=
  chainConstraints interface.initial interface.rounds interface.terminal

theorem constraints_length {degree roundCount : Nat}
    (interface : Interface degree roundCount) :
    (constraints interface).length = 2 * (roundCount + 1) := by
  rw [constraints, chainConstraints_length]
  simp [Interface.rounds]

theorem constraints_varsBelow {degree roundCount : Nat}
    (interface : Interface degree roundCount) (bound : Nat)
    (below : interface.VarsBelow bound) :
    ∀ expression ∈ constraints interface, expression.VarsBelow bound := by
  apply chainConstraints_varsBelow interface.initial interface.terminal
      interface.rounds bound below.1
  · intro round member
    rw [Interface.rounds, List.mem_ofFn'] at member
    rcases member with ⟨index, rfl⟩
    exact below.2.1 index
  · exact below.2.2

def main {degree roundCount : Nat}
    (interface : Interface degree roundCount) : Circuit Unit :=
  fun offset => ((), offset, (constraints interface).map Op.assertZero)

theorem circuit_localLength {degree roundCount : Nat}
    (interface : Interface degree roundCount) (offset : Nat) :
    localLength (Circuit.ops (main interface) offset) = 0 := by
  change (List.map Op.localLength
    ((constraints interface).map Op.assertZero)).sum = 0
  rw [List.map_map]
  simp [Function.comp_def, Op.localLength]

theorem operations_length {degree roundCount : Nat}
    (interface : Interface degree roundCount) (offset : Nat) :
    (Circuit.ops (main interface) offset).length =
      2 * (roundCount + 1) := by
  change ((constraints interface).map Op.assertZero).length = _
  simp [constraints_length]

theorem flatConstraints_assertions_eq (expressions : List Expr) :
    flatConstraints (expressions.map Op.assertZero) = expressions := by
  induction expressions with
  | nil => rfl
  | cons expression expressions inductionHypothesis =>
      change expression :: flatConstraints (expressions.map Op.assertZero) =
        expression :: expressions
      rw [inductionHypothesis]

theorem flatConstraints_length {degree roundCount : Nat}
    (interface : Interface degree roundCount) (offset : Nat) :
    (flatConstraints (Circuit.ops (main interface) offset)).length =
      2 * (roundCount + 1) := by
  change (flatConstraints ((constraints interface).map Op.assertZero)).length = _
  rw [flatConstraints_assertions_eq]
  exact constraints_length interface

theorem flatConstraints_varsBelow {degree roundCount : Nat}
    (interface : Interface degree roundCount) (offset : Nat)
    (below : interface.VarsBelow offset) :
    ∀ expression ∈ flatConstraints (Circuit.ops (main interface) offset),
      expression.VarsBelow offset := by
  change ∀ expression ∈
    flatConstraints ((constraints interface).map Op.assertZero),
      expression.VarsBelow offset
  rw [flatConstraints_assertions_eq]
  exact constraints_varsBelow interface offset below

theorem holds_assertions_iff (env : Env) (expressions : List Expr) :
    holds env (expressions.map Op.assertZero) ↔
      ConstraintsHold env expressions := by
  induction expressions with
  | nil => simp [ConstraintsHold]
  | cons expression expressions inductionHypothesis =>
      simp only [List.map_cons, holds_cons, Op.holds_assertZero,
        inductionHypothesis]
      constructor
      · rintro ⟨head, tail⟩ current member
        rcases List.mem_cons.mp member with rfl | member
        · exact head
        · exact tail current member
      · intro all
        exact ⟨all expression (by simp), fun current member =>
          all current (by simp [member])⟩

theorem holdsFlat_assertions_iff (env : Env) (expressions : List Expr) :
    holdsFlat env (expressions.map Op.assertZero) ↔
      ConstraintsHold env expressions := by
  have flatten :
      flatConstraints (expressions.map Op.assertZero) = expressions := by
    induction expressions with
    | nil => rfl
    | cons expression expressions inductionHypothesis =>
        change expression ::
          flatConstraints (expressions.map Op.assertZero) =
            expression :: expressions
        rw [inductionHypothesis]
  unfold holdsFlat
  rw [flatten]

/-- The one opaque fixed-chain circuit. -/
def circuit {degree roundCount : Nat}
    (interface : Interface degree roundCount) : FormalCircuit where
  main := main interface
  assumptions := Assumptions interface
  spec := fun _ env => SpecHolds interface env
  soundness := by
    intro env offset assumptions rows
    have constraintsHold : ConstraintsHold env (constraints interface) :=
      (holds_assertions_iff env (constraints interface)).mp rows
    exact (chainConstraints_hold_iff env interface.initial
      interface.terminal interface.rounds).mp constraintsHold
  completeness := by
    intro env offset assumptions specification
    refine ⟨env, ?_, ?_⟩
    · intro index outside
      rfl
    · apply (holdsFlat_assertions_iff env (constraints interface)).mpr
      exact (chainConstraints_hold_iff env interface.initial
        interface.terminal interface.rounds).mpr specification

/-! ## Child-owned terminal variant -/

namespace Owned

/-!
Obligation: Enforce every SumCheck round equation and export the final
`p_i(r_i)` value directly to the terminal-check owner.

Inputs:
- one initial claim;
- indexed prover polynomials and verifier-owned challenges.

Output:
- the final claimed value after all indexed rounds.

Constraint groups:
- C1: the stored Horner products of every `p_i(r_i)`;
- C2: two base-field rows per round for `claim_i = p_i(0) + p_i(1)`.

`p_i(0)` is the constant coefficient and `p_i(1)` is the coefficient sum, so
C2 has no multiplication. Each later round and the terminal owner read a
stored `p_i(r_i)`, never a repeated expression. The final equality belongs to
the protocol terminal gadget; this circuit has no terminal copy row.
-/

open NightstreamFPrime.Gadgets.Polynomial

/-- `p(1)`: the sum of all coefficients, in the semantic evaluation order. -/
def coefficientSum : List KExpr → KExpr
  | [] => KExpr.zero
  | coefficient :: rest => KExpr.add coefficient (coefficientSum rest)

/-- `p(0) + p(1)`: the constant coefficient plus the coefficient sum. -/
def boundarySum : List KExpr → KExpr
  | [] => KExpr.add KExpr.zero KExpr.zero
  | coefficient :: rest =>
      KExpr.add coefficient (coefficientSum (coefficient :: rest))

theorem eval_coefficientSum (env : Env) (coefficients : List KExpr) :
    (coefficientSum coefficients).eval env =
      NightstreamFPrime.Spec.SumCheck.Finite.Message.evaluateCoefficients
        extensionOps.toOps extensionOps.one
        (coefficients.map (KExpr.eval env)) := by
  induction coefficients with
  | nil => rfl
  | cons coefficient rest inductionHypothesis =>
      change K.add (coefficient.eval env) ((coefficientSum rest).eval env) =
        extensionOps.add (coefficient.eval env)
          (extensionOps.mul extensionOps.one
            (NightstreamFPrime.Spec.SumCheck.Finite.Message.evaluateCoefficients
              extensionOps.toOps extensionOps.one
              (rest.map (KExpr.eval env))))
      rw [extensionLaws.one_mul, inductionHypothesis]
      rfl

theorem eval_boundarySum (env : Env) (coefficients : List KExpr) :
    (boundarySum coefficients).eval env =
      extensionOps.add
        (NightstreamFPrime.Spec.SumCheck.Finite.Message.evaluateCoefficients
          extensionOps.toOps extensionOps.zero
          (coefficients.map (KExpr.eval env)))
        (NightstreamFPrime.Spec.SumCheck.Finite.Message.evaluateCoefficients
          extensionOps.toOps extensionOps.one
          (coefficients.map (KExpr.eval env))) := by
  cases coefficients with
  | nil => rfl
  | cons coefficient rest =>
      change K.add (coefficient.eval env)
          ((coefficientSum (coefficient :: rest)).eval env) =
        extensionOps.add
          (extensionOps.add (coefficient.eval env)
            (extensionOps.mul extensionOps.zero
              (NightstreamFPrime.Spec.SumCheck.Finite.Message.evaluateCoefficients
                extensionOps.toOps extensionOps.zero
                (rest.map (KExpr.eval env)))))
          (NightstreamFPrime.Spec.SumCheck.Finite.Message.evaluateCoefficients
            extensionOps.toOps extensionOps.one
            ((coefficient :: rest).map (KExpr.eval env)))
      rw [extensionLaws.mul_comm, extensionLaws.mul_zero,
        extensionLaws.add_zero, eval_coefficientSum]
      rfl

theorem coefficientSum_varsBelow (coefficients : List KExpr) (bound : Nat)
    (below : ∀ coefficient ∈ coefficients, coefficient.VarsBelow bound) :
    (coefficientSum coefficients).VarsBelow bound := by
  induction coefficients with
  | nil => exact ⟨trivial, trivial⟩
  | cons coefficient rest inductionHypothesis =>
      exact KExpr.add_varsBelow _ _ bound (below coefficient (by simp))
        (inductionHypothesis fun current member =>
          below current (by simp [member]))

theorem boundarySum_varsBelow (coefficients : List KExpr) (bound : Nat)
    (below : ∀ coefficient ∈ coefficients, coefficient.VarsBelow bound) :
    (boundarySum coefficients).VarsBelow bound := by
  cases coefficients with
  | nil => exact ⟨⟨trivial, trivial⟩, ⟨trivial, trivial⟩⟩
  | cons coefficient rest =>
      exact KExpr.add_varsBelow _ _ bound (below coefficient (by simp))
        (coefficientSum_varsBelow (coefficient :: rest) bound below)

/-- `p(0) + p(1)` of one round. -/
def roundBoundary {degree : Nat} (round : Round degree) : KExpr :=
  boundarySum round.coefficients

theorem eval_roundBoundary {degree : Nat} (env : Env) (round : Round degree) :
    (roundBoundary round).eval env =
      extensionOps.add
        ((round.semanticPolynomial env).evaluate extensionOps.toOps
          extensionOps.zero)
        ((round.semanticPolynomial env).evaluate extensionOps.toOps
          extensionOps.one) :=
  eval_boundarySum env round.coefficients

private theorem Round.coefficients_below {degree : Nat} (round : Round degree)
    (bound : Nat) (below : round.VarsBelow bound) :
    ∀ coefficient ∈ round.coefficients, coefficient.VarsBelow bound := by
  intro coefficient member
  rw [Round.coefficients, List.mem_ofFn'] at member
  rcases member with ⟨index, rfl⟩
  exact below.1 index

/-- The stored Horner evaluation of one round at its challenge. -/
def roundProgram {degree : Nat} (start : Nat) (round : Round degree) :
    Horner.Program :=
  Horner.compile start round.challenge round.coefficients

theorem roundProgram_recipes_length {degree : Nat} (start : Nat)
    (round : Round degree) :
    (roundProgram start round).recipes.length = 2 * degree := by
  rw [roundProgram, Horner.compile_recipes_length]
  simp [Round.coefficients]

theorem roundProgram_output_eval {degree : Nat} (env : Env) (start : Nat)
    (round : Round degree)
    (rows : ConstraintsHold env
      (recipeConstraints start (roundProgram start round).recipes)) :
    (roundProgram start round).output.eval env =
      (round.semanticPolynomial env).evaluate extensionOps.toOps
        (round.challenge.eval env) :=
  (Horner.compile_output_sound env start round.challenge round.coefficients
    rows).trans (Horner.evaluate_eq_messageEvaluate _ _)

/-- Stored products of every round, in round order. -/
def recipesFrom {degree : Nat} (start : Nat) : List (Round degree) → List Expr
  | [] => []
  | round :: rounds =>
      (roundProgram start round).recipes ++
        recipesFrom (start + 2 * degree) rounds

/-- The final stored claim; an empty chain returns its input claim. -/
def outputFrom {degree : Nat} (start : Nat) :
    KExpr → List (Round degree) → KExpr
  | current, [] => current
  | _, round :: rounds =>
      outputFrom (start + 2 * degree) (roundProgram start round).output rounds

/-- The round equations `claim_i = p_i(0) + p_i(1)`. -/
def equalitiesFrom {degree : Nat} (start : Nat) :
    KExpr → List (Round degree) → List Expr
  | _, [] => []
  | current, round :: rounds =>
      KExpr.equalities current (roundBoundary round) ++
        equalitiesFrom (start + 2 * degree)
          (roundProgram start round).output rounds

theorem recipesFrom_length {degree : Nat} (start : Nat)
    (rounds : List (Round degree)) :
    (recipesFrom start rounds).length = 2 * degree * rounds.length := by
  induction rounds generalizing start with
  | nil => rfl
  | cons round rounds inductionHypothesis =>
      simp only [recipesFrom, List.length_append, List.length_cons,
        roundProgram_recipes_length, inductionHypothesis]
      rw [Nat.mul_succ]
      omega

theorem equalitiesFrom_length {degree : Nat} (start : Nat) (current : KExpr)
    (rounds : List (Round degree)) :
    (equalitiesFrom start current rounds).length = 2 * rounds.length := by
  induction rounds generalizing start current with
  | nil => rfl
  | cons round rounds inductionHypothesis =>
      simp only [equalitiesFrom, KExpr.equalities, List.length_append,
        List.length_cons, List.length_nil, inductionHypothesis]
      omega

private theorem recipesCausal_concat (start : Nat) (first second : List Expr)
    (firstCausal : RecipesCausal start first)
    (secondCausal : RecipesCausal (start + first.length) second) :
    RecipesCausal start (first ++ second) := by
  induction first generalizing start with
  | nil => simpa using secondCausal
  | cons recipe rest inductionHypothesis =>
      refine ⟨firstCausal.1, inductionHypothesis (start + 1) firstCausal.2 ?_⟩
      have shifted : start + 1 + rest.length =
          start + (recipe :: rest).length := by
        simp only [List.length_cons]
        omega
      rw [shifted]
      exact secondCausal

theorem recipesFrom_causal {degree : Nat} (start : Nat)
    (rounds : List (Round degree))
    (roundsBelow : ∀ round ∈ rounds, round.VarsBelow start) :
    RecipesCausal start (recipesFrom start rounds) := by
  induction rounds generalizing start with
  | nil => trivial
  | cons round rounds inductionHypothesis =>
      have roundBelow := roundsBelow round (by simp)
      have headCausal : RecipesCausal start (roundProgram start round).recipes :=
        Horner.compile_causal start round.challenge round.coefficients
          roundBelow.2 (Round.coefficients_below round start roundBelow)
      have tailCausal := inductionHypothesis (start + 2 * degree)
        (fun later member =>
          later.varsBelow_mono (roundsBelow later (by simp [member]))
            (by omega))
      apply recipesCausal_concat start _ _ headCausal
      rw [roundProgram_recipes_length]
      exact tailCausal

private theorem roundProgram_output_below {degree : Nat} (start : Nat)
    (round : Round degree) (roundBelow : round.VarsBelow start) :
    (roundProgram start round).output.VarsBelow (start + 2 * degree) := by
  have below := (Horner.compile_causal_and_output_below start round.challenge
    round.coefficients roundBelow.2
    (Round.coefficients_below round start roundBelow)).2
  rw [← roundProgram_recipes_length start round]
  exact below

theorem outputFrom_varsBelow {degree : Nat} (start : Nat) (current : KExpr)
    (rounds : List (Round degree))
    (currentBelow : current.VarsBelow start)
    (roundsBelow : ∀ round ∈ rounds, round.VarsBelow start) :
    (outputFrom start current rounds).VarsBelow
      (start + 2 * degree * rounds.length) := by
  induction rounds generalizing start current with
  | nil => simpa [outputFrom] using currentBelow
  | cons round rounds inductionHypothesis =>
      have step := inductionHypothesis (start + 2 * degree)
        (roundProgram start round).output
        (roundProgram_output_below start round (roundsBelow round (by simp)))
        (fun later member =>
          later.varsBelow_mono (roundsBelow later (by simp [member]))
            (by omega))
      have boundEq : start + 2 * degree + 2 * degree * rounds.length =
          start + 2 * degree * (round :: rounds).length := by
        simp only [List.length_cons]
        rw [Nat.mul_succ]
        omega
      rw [← boundEq]
      exact step

theorem equalitiesFrom_varsBelow {degree : Nat} (start : Nat) (current : KExpr)
    (rounds : List (Round degree))
    (currentBelow : current.VarsBelow start)
    (roundsBelow : ∀ round ∈ rounds, round.VarsBelow start) :
    ∀ expression ∈ equalitiesFrom start current rounds,
      expression.VarsBelow (start + 2 * degree * rounds.length) := by
  induction rounds generalizing start current with
  | nil =>
      intro expression member
      simp [equalitiesFrom] at member
  | cons round rounds inductionHypothesis =>
      intro expression member
      have boundEq : start + 2 * degree + 2 * degree * rounds.length =
          start + 2 * degree * (round :: rounds).length := by
        simp only [List.length_cons]
        rw [Nat.mul_succ]
        omega
      have roundBelow := roundsBelow round (by simp)
      rw [equalitiesFrom] at member
      rcases List.mem_append.mp member with headMember | tailMember
      · have headBelow := KExpr.equalities_varsBelow current
          (roundBoundary round) start currentBelow
          (boundarySum_varsBelow round.coefficients start
            (Round.coefficients_below round start roundBelow))
          expression headMember
        exact Expr.VarsBelow.mono expression headBelow (by omega)
      · have step := inductionHypothesis (start + 2 * degree)
          (roundProgram start round).output
          (roundProgram_output_below start round roundBelow)
          (fun later laterMember =>
            later.varsBelow_mono (roundsBelow later (by simp [laterMember]))
              (by omega))
          expression tailMember
        rw [← boundEq]
        exact step

/-- Soundness core: stored products and round equations give the exact
claimed chain ending at the stored final claim. -/
theorem chain_of_rows {degree : Nat} (env : Env) (start : Nat)
    (current : KExpr) (rounds : List (Round degree))
    (recipeRows : ConstraintsHold env
      (recipeConstraints start (recipesFrom start rounds)))
    (equalityRows : ConstraintsHold env
      (equalitiesFrom start current rounds)) :
    NightstreamFPrime.Spec.SumCheck.Finite.FixedPhase.Chain
      extensionOps.toOps (current.eval env)
      (rounds.map (Round.semanticPolynomial env))
      (rounds.map fun round => round.challenge.eval env)
      ((outputFrom start current rounds).eval env) := by
  induction rounds generalizing start current with
  | nil =>
      simp [outputFrom,
        NightstreamFPrime.Spec.SumCheck.Finite.FixedPhase.Chain]
  | cons round rounds inductionHypothesis =>
      rw [recipesFrom, recipeConstraints_append,
        roundProgram_recipes_length] at recipeRows
      have recipeSplit :=
        (constraintsHold_append env _ _).mp recipeRows
      rw [equalitiesFrom] at equalityRows
      have equalitySplit :=
        (constraintsHold_append env _ _).mp equalityRows
      have headEq := (KExpr.equalities_hold_iff env current
        (roundBoundary round)).mp equalitySplit.1
      have outputEq := roundProgram_output_eval env start round recipeSplit.1
      have tail := inductionHypothesis (start + 2 * degree)
        (roundProgram start round).output recipeSplit.2 equalitySplit.2
      simp only [List.map_cons,
        NightstreamFPrime.Spec.SumCheck.Finite.FixedPhase.Chain, outputFrom]
      refine ⟨headEq.trans (eval_roundBoundary env round), ?_⟩
      rw [← outputEq]
      exact tail

/-- Completeness core: with the stored products in place, an accepted chain
satisfies every round equation and its terminal is the stored final claim. -/
theorem rows_of_chain {degree : Nat} (env : Env) (start : Nat)
    (current : KExpr) (rounds : List (Round degree))
    (recipeRows : ConstraintsHold env
      (recipeConstraints start (recipesFrom start rounds)))
    (terminal : K)
    (chain : NightstreamFPrime.Spec.SumCheck.Finite.FixedPhase.Chain
      extensionOps.toOps (current.eval env)
      (rounds.map (Round.semanticPolynomial env))
      (rounds.map fun round => round.challenge.eval env) terminal) :
    ConstraintsHold env (equalitiesFrom start current rounds) ∧
      (outputFrom start current rounds).eval env = terminal := by
  induction rounds generalizing start current with
  | nil =>
      refine ⟨?_, ?_⟩
      · intro expression member
        simp [equalitiesFrom] at member
      · simpa [outputFrom,
          NightstreamFPrime.Spec.SumCheck.Finite.FixedPhase.Chain] using chain
  | cons round rounds inductionHypothesis =>
      rw [recipesFrom, recipeConstraints_append,
        roundProgram_recipes_length] at recipeRows
      have recipeSplit :=
        (constraintsHold_append env _ _).mp recipeRows
      simp only [List.map_cons,
        NightstreamFPrime.Spec.SumCheck.Finite.FixedPhase.Chain] at chain
      have outputEq := roundProgram_output_eval env start round recipeSplit.1
      have tail := inductionHypothesis (start + 2 * degree)
        (roundProgram start round).output recipeSplit.2
        (by rw [outputEq]; exact chain.2)
      refine ⟨?_, by simpa [outputFrom] using tail.2⟩
      rw [equalitiesFrom]
      apply (constraintsHold_append env _ _).mpr
      refine ⟨?_, tail.1⟩
      apply (KExpr.equalities_hold_iff env current (roundBoundary round)).mpr
      exact chain.1.trans (eval_roundBoundary env round).symm

structure Interface (degree roundCount : Nat) where
  initial : KExpr
  round : Fin roundCount → Round degree

def Interface.rounds {degree roundCount : Nat}
    (interface : Interface degree roundCount) : List (Round degree) :=
  List.ofFn interface.round

def Interface.VarsBelow {degree roundCount : Nat}
    (interface : Interface degree roundCount) (bound : Nat) : Prop :=
  interface.initial.VarsBelow bound ∧
    ∀ round, (interface.round round).VarsBelow bound

private theorem Interface.rounds_below {degree roundCount : Nat}
    (interface : Interface degree roundCount) (bound : Nat)
    (below : interface.VarsBelow bound) :
    ∀ round ∈ interface.rounds, round.VarsBelow bound := by
  intro round member
  rw [Interface.rounds, List.mem_ofFn'] at member
  rcases member with ⟨index, rfl⟩
  exact below.2 index

/-- Number of stored base-field values: two for each Horner product. -/
def privateCount (degree roundCount : Nat) : Nat := 2 * degree * roundCount

def recipes {degree roundCount : Nat}
    (interface : Interface degree roundCount) (offset : Nat) : List Expr :=
  recipesFrom offset interface.rounds

def assertions {degree roundCount : Nat}
    (interface : Interface degree roundCount) (offset : Nat) : List Expr :=
  equalitiesFrom offset interface.initial interface.rounds

/-- The stored final claim `p_n(r_n)`. -/
def output {degree roundCount : Nat}
    (interface : Interface degree roundCount) (offset : Nat) : KExpr :=
  outputFrom offset interface.initial interface.rounds

theorem recipes_length {degree roundCount : Nat}
    (interface : Interface degree roundCount) (offset : Nat) :
    (recipes interface offset).length = privateCount degree roundCount := by
  rw [recipes, recipesFrom_length]
  simp [Interface.rounds, privateCount]

theorem assertions_length {degree roundCount : Nat}
    (interface : Interface degree roundCount) (offset : Nat) :
    (assertions interface offset).length = 2 * roundCount := by
  rw [assertions, equalitiesFrom_length]
  simp [Interface.rounds]

def SpecHolds {degree roundCount : Nat}
    (interface : Interface degree roundCount) (offset : Nat) (env : Env) :
    Prop :=
  NightstreamFPrime.Spec.SumCheck.Finite.FixedPhase.Chain extensionOps.toOps
    (interface.initial.eval env)
    (interface.rounds.map (Round.semanticPolynomial env))
    (interface.rounds.map fun round => round.challenge.eval env)
    ((output interface offset).eval env)

def Assumptions {degree roundCount : Nat}
    (interface : Interface degree roundCount) (offset : Nat) (_env : Env) :
    Prop :=
  interface.VarsBelow offset

def opsAt {degree roundCount : Nat}
    (interface : Interface degree roundCount) (offset : Nat) : List Op :=
  Op.witness (WitnessBatch.arithmetic offset (recipes interface offset)) ::
    (assertions interface offset).map Op.assertZero

def main {degree roundCount : Nat}
    (interface : Interface degree roundCount) : Circuit Unit :=
  fun offset =>
    ((), offset + (recipes interface offset).length, opsAt interface offset)

@[simp] theorem main_ops {degree roundCount : Nat}
    (interface : Interface degree roundCount) (offset : Nat) :
    Circuit.ops (main interface) offset = opsAt interface offset := by
  rfl

theorem flatConstraints_opsAt {degree roundCount : Nat}
    (interface : Interface degree roundCount) (offset : Nat) :
    flatConstraints (opsAt interface offset) =
      recipeConstraints offset (recipes interface offset) ++
        assertions interface offset := by
  change recipeConstraints offset (recipes interface offset) ++
      flatConstraints ((assertions interface offset).map Op.assertZero) = _
  rw [flatConstraints_assertions_eq]

theorem recipes_causal {degree roundCount : Nat}
    (interface : Interface degree roundCount) (offset : Nat)
    (below : interface.VarsBelow offset) :
    RecipesCausal offset (recipes interface offset) :=
  recipesFrom_causal offset interface.rounds
    (Interface.rounds_below interface offset below)

theorem soundness {degree roundCount : Nat}
    (interface : Interface degree roundCount) (env : Env) (offset : Nat)
    (_assumptions : Assumptions interface offset env)
    (rows : holds env (Circuit.ops (main interface) offset)) :
    SpecHolds interface offset env := by
  have recipeRows : ConstraintsHold env
      (recipeConstraints offset (recipes interface offset)) :=
    rows (Op.witness (WitnessBatch.arithmetic offset
      (recipes interface offset))) (by simp [main_ops, opsAt])
  have equalityRows : ConstraintsHold env (assertions interface offset) := by
    intro expression member
    exact rows (Op.assertZero expression) (by
      simp [main_ops, opsAt, member])
  exact chain_of_rows env offset interface.initial interface.rounds
    recipeRows equalityRows

/-- Honest execution from an accepted chain over the input wires. It stores
every product and returns the chain terminal as the owned output. -/
theorem build {degree roundCount : Nat}
    (interface : Interface degree roundCount) (env : Env) (offset : Nat)
    (assumptions : Assumptions interface offset env) (terminal : K)
    (chain : NightstreamFPrime.Spec.SumCheck.Finite.FixedPhase.Chain
      extensionOps.toOps (interface.initial.eval env)
      (interface.rounds.map (Round.semanticPolynomial env))
      (interface.rounds.map fun round => round.challenge.eval env) terminal) :
    ∃ completed,
      AgreesOutside env completed offset
        (localLength (Circuit.ops (main interface) offset)) ∧
      holdsFlat completed (Circuit.ops (main interface) offset) ∧
      (output interface offset).eval completed = terminal := by
  let stored := recipes interface offset
  let completed := executeRecipes env offset stored
  have causal : RecipesCausal offset stored :=
    recipes_causal interface offset assumptions
  have recipeRows : ConstraintsHold completed
      (recipeConstraints offset stored) :=
    executeRecipes_holds_recipeConstraints env offset stored causal
  have agreesBelow : ∀ index, index < offset → completed index = env index :=
    executeRecipes_agrees_below env offset stored
  have initialEq : interface.initial.eval completed =
      interface.initial.eval env :=
    interface.initial.eval_eq_of_agree_below offset completed env
      assumptions.1 agreesBelow
  have roundsEq :
      interface.rounds.map (Round.semanticPolynomial completed) =
        interface.rounds.map (Round.semanticPolynomial env) := by
    apply List.map_congr_left
    intro round member
    exact round.semanticPolynomial_eq_of_agree_below offset completed env
      (Interface.rounds_below interface offset assumptions round member)
      agreesBelow
  have challengesEq :
      interface.rounds.map (fun round => round.challenge.eval completed) =
        interface.rounds.map (fun round => round.challenge.eval env) := by
    apply List.map_congr_left
    intro round member
    exact round.challenge.eval_eq_of_agree_below offset completed env
      (Interface.rounds_below interface offset assumptions round member).2
      agreesBelow
  have completedChain :
      NightstreamFPrime.Spec.SumCheck.Finite.FixedPhase.Chain
        extensionOps.toOps (interface.initial.eval completed)
        (interface.rounds.map (Round.semanticPolynomial completed))
        (interface.rounds.map fun round => round.challenge.eval completed)
        terminal := by
    rw [initialEq, roundsEq, challengesEq]
    exact chain
  have result := rows_of_chain completed offset interface.initial
    interface.rounds recipeRows terminal completedChain
  refine ⟨completed, ?_, ?_, result.2⟩
  · change AgreesOutside env completed offset
      (localLength (opsAt interface offset))
    have localEq : localLength (opsAt interface offset) = stored.length := by
      change (List.map Op.localLength (opsAt interface offset)).sum = _
      simp [opsAt, Op.localLength, Function.comp_def, stored]
    rw [localEq]
    exact executeRecipes_agreesOutside env offset stored
  · change ConstraintsHold completed (flatConstraints (opsAt interface offset))
    rw [flatConstraints_opsAt]
    exact (constraintsHold_append completed _ _).mpr ⟨recipeRows, result.1⟩

theorem completeness {degree roundCount : Nat}
    (interface : Interface degree roundCount) (env : Env) (offset : Nat)
    (assumptions : Assumptions interface offset env)
    (specification : SpecHolds interface offset env) :
    ∃ completed,
      AgreesOutside env completed offset
        (localLength (Circuit.ops (main interface) offset)) ∧
      holdsFlat completed (Circuit.ops (main interface) offset) := by
  rcases build interface env offset assumptions
      ((output interface offset).eval env) specification with
    ⟨completed, agrees, rows, _⟩
  exact ⟨completed, agrees, rows⟩

def circuit {degree roundCount : Nat}
    (interface : Interface degree roundCount) : FormalCircuit where
  main := main interface
  assumptions := Assumptions interface
  spec := SpecHolds interface
  soundness := soundness interface
  completeness := completeness interface

@[simp] theorem circuit_ops {degree roundCount : Nat}
    (interface : Interface degree roundCount) (offset : Nat) :
    Circuit.ops (circuit interface).main offset = opsAt interface offset := by
  rfl

theorem flatConstraints_eq {degree roundCount : Nat}
    (interface : Interface degree roundCount) (offset : Nat) :
    flatConstraints (Circuit.ops (circuit interface).main offset) =
      recipeConstraints offset (recipes interface offset) ++
        assertions interface offset := by
  rw [circuit_ops, flatConstraints_opsAt]

theorem localLength_eq {degree roundCount : Nat}
    (interface : Interface degree roundCount) (offset : Nat) :
    localLength (Circuit.ops (circuit interface).main offset) =
      privateCount degree roundCount := by
  change (List.map Op.localLength (opsAt interface offset)).sum = _
  simp [opsAt, Op.localLength, Function.comp_def, recipes_length]

theorem operations_length {degree roundCount : Nat}
    (interface : Interface degree roundCount) (offset : Nat) :
    (Circuit.ops (circuit interface).main offset).length =
      2 * roundCount + 1 := by
  change (opsAt interface offset).length = _
  simp [opsAt, assertions_length]

theorem flatConstraints_length {degree roundCount : Nat}
    (interface : Interface degree roundCount) (offset : Nat) :
    (flatConstraints (Circuit.ops (circuit interface).main offset)).length =
      privateCount degree roundCount + 2 * roundCount := by
  rw [flatConstraints_eq, List.length_append, recipeConstraints_length,
    recipes_length, assertions_length]

/-- Every row reads only input wires or this child's stored interval. -/
theorem flatConstraints_varsBelow {degree roundCount : Nat}
    (interface : Interface degree roundCount) (offset : Nat)
    (below : interface.VarsBelow offset) :
    ∀ expression ∈ flatConstraints (Circuit.ops (circuit interface).main offset),
      expression.VarsBelow (offset + privateCount degree roundCount) := by
  rw [flatConstraints_eq]
  intro expression member
  rcases List.mem_append.mp member with recipeMember | assertionMember
  · have scope := recipeConstraints_varsBelow_of_causal offset
      (recipes interface offset) (recipes_causal interface offset below)
      expression recipeMember
    rwa [recipes_length] at scope
  · have scope := equalitiesFrom_varsBelow offset interface.initial
      interface.rounds below.1
      (Interface.rounds_below interface offset below) expression
      assertionMember
    simpa [Interface.rounds, privateCount] using scope

/-- The owned result lies inside the child's stored interval. -/
theorem output_varsBelow {degree roundCount : Nat}
    (interface : Interface degree roundCount) (offset : Nat)
    (below : interface.VarsBelow offset) :
    (output interface offset).VarsBelow
      (offset + privateCount degree roundCount) := by
  have scope := outputFrom_varsBelow offset interface.initial interface.rounds
    below.1 (Interface.rounds_below interface offset below)
  simpa [output, Interface.rounds, privateCount] using scope

/-- The specification is stable when the input wires and the stored interval
are unchanged. -/
theorem specHolds_eq_of_agree_below {degree roundCount : Nat}
    (interface : Interface degree roundCount) (offset : Nat)
    (left right : Env) (below : interface.VarsBelow offset)
    (agrees : ∀ index, index < offset + privateCount degree roundCount →
      left index = right index) :
    SpecHolds interface offset left ↔ SpecHolds interface offset right := by
  have agreesInputs : ∀ index, index < offset → left index = right index :=
    fun index indexBelow => agrees index (by omega)
  have initial := interface.initial.eval_eq_of_agree_below offset left right
    below.1 agreesInputs
  have rounds :
      interface.rounds.map (Round.semanticPolynomial left) =
        interface.rounds.map (Round.semanticPolynomial right) := by
    apply List.map_congr_left
    intro round member
    exact round.semanticPolynomial_eq_of_agree_below offset left right
      (Interface.rounds_below interface offset below round member)
      agreesInputs
  have challenges :
      interface.rounds.map (fun round => round.challenge.eval left) =
        interface.rounds.map (fun round => round.challenge.eval right) := by
    apply List.map_congr_left
    intro round member
    exact round.challenge.eval_eq_of_agree_below offset left right
      (Interface.rounds_below interface offset below round member).2
      agreesInputs
  have outputEq := (output interface offset).eval_eq_of_agree_below
    (offset + privateCount degree roundCount) left right
    (output_varsBelow interface offset below) agrees
  unfold SpecHolds
  rw [initial, rounds, challenges, outputEq]

end Owned

end NightstreamFPrime.Gadgets.SumCheck.FixedChain
