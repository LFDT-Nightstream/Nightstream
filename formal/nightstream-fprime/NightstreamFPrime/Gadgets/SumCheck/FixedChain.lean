import NightstreamFPrime.Circuit.Quadratic
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier.Algebra
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
`p_i(r_i)` expression directly to the terminal-check owner.

Inputs:
- one initial claim;
- indexed prover polynomials and verifier-owned challenges.

Output:
- the final claimed value after all indexed rounds.

The final equality belongs to the protocol terminal gadget. This circuit
therefore has exactly two base-field rows per SumCheck round and no terminal
copy row.
-/

def outputFrom {degree : Nat} : KExpr → List (Round degree) → KExpr
  | current, [] => current
  | _, round :: rounds =>
      outputFrom (evaluateRound round round.challenge) rounds

def constraintsFrom {degree : Nat} : KExpr → List (Round degree) → List Expr
  | _, [] => []
  | current, round :: rounds =>
      KExpr.equalities current
        (KExpr.add (evaluateRound round KExpr.zero)
          (evaluateRound round KExpr.one)) ++
      constraintsFrom (evaluateRound round round.challenge) rounds

theorem constraintsFrom_length {degree : Nat} (current : KExpr)
    (rounds : List (Round degree)) :
    (constraintsFrom current rounds).length = 2 * rounds.length := by
  induction rounds generalizing current with
  | nil => rfl
  | cons round rounds inductionHypothesis =>
      simp [constraintsFrom, KExpr.equalities, inductionHypothesis]
      omega

theorem outputFrom_varsBelow {degree : Nat} (current : KExpr)
    (rounds : List (Round degree)) (bound : Nat)
    (currentBelow : current.VarsBelow bound)
    (roundsBelow : ∀ round ∈ rounds, round.VarsBelow bound) :
    (outputFrom current rounds).VarsBelow bound := by
  induction rounds generalizing current with
  | nil => exact currentBelow
  | cons round rounds inductionHypothesis =>
      apply inductionHypothesis
      · exact evaluateRound_varsBelow round round.challenge bound
          (roundsBelow round (by simp)) (roundsBelow round (by simp)).2
      · intro later member
        exact roundsBelow later (by simp [member])

theorem constraintsFrom_varsBelow {degree : Nat} (current : KExpr)
    (rounds : List (Round degree)) (bound : Nat)
    (currentBelow : current.VarsBelow bound)
    (roundsBelow : ∀ round ∈ rounds, round.VarsBelow bound) :
    ∀ expression ∈ constraintsFrom current rounds,
      expression.VarsBelow bound := by
  induction rounds generalizing current with
  | nil =>
      intro expression member
      simp [constraintsFrom] at member
  | cons round rounds inductionHypothesis =>
      intro expression member
      rw [constraintsFrom] at member
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
        exact KExpr.equalities_varsBelow current _ bound currentBelow
          rightBelow expression headMember
      · exact inductionHypothesis
          (evaluateRound round round.challenge)
          (evaluateRound_varsBelow round round.challenge bound
            (roundsBelow round (by simp)) (roundsBelow round (by simp)).2)
          (fun later laterMember =>
            roundsBelow later (by simp [laterMember]))
          expression tailMember

theorem constraintsFrom_hold_iff {degree : Nat} (env : Env)
    (current : KExpr) (rounds : List (Round degree)) :
    ConstraintsHold env (constraintsFrom current rounds) ↔
      NightstreamFPrime.Spec.SumCheck.Finite.FixedPhase.Chain
        extensionOps.toOps (current.eval env)
        (rounds.map (Round.semanticPolynomial env))
        (rounds.map fun round => round.challenge.eval env)
        ((outputFrom current rounds).eval env) := by
  induction rounds generalizing current with
  | nil =>
      simp [constraintsFrom, outputFrom, ConstraintsHold,
        NightstreamFPrime.Spec.SumCheck.Finite.FixedPhase.Chain]
  | cons round rounds inductionHypothesis =>
      rw [constraintsFrom, constraintsHold_append,
        KExpr.equalities_hold_iff, inductionHypothesis]
      simp only [List.map_cons,
        NightstreamFPrime.Spec.SumCheck.Finite.FixedPhase.Chain, outputFrom]
      simp only [KExpr.eval_add, eval_evaluateRound, KExpr.eval_zero,
        KExpr.eval_one]
      simp [extensionOps]

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

def output {degree roundCount : Nat}
    (interface : Interface degree roundCount) : KExpr :=
  outputFrom interface.initial interface.rounds

def SpecHolds {degree roundCount : Nat}
    (interface : Interface degree roundCount) (env : Env) : Prop :=
  NightstreamFPrime.Spec.SumCheck.Finite.FixedPhase.Chain extensionOps.toOps
    (interface.initial.eval env)
    (interface.rounds.map (Round.semanticPolynomial env))
    (interface.rounds.map fun round => round.challenge.eval env)
    ((output interface).eval env)

/-- A verifier chain is exactly the owned round constraints plus the final
claimed value. This separates the SumCheck leaf from the protocol terminal
leaf without changing either formula. -/
theorem chain_iff_specHolds_and_output_eq
    {degree roundCount : Nat}
    (interface : Interface degree roundCount) (env : Env) (terminal : K) :
    NightstreamFPrime.Spec.SumCheck.Finite.FixedPhase.Chain extensionOps.toOps
      (interface.initial.eval env)
      (interface.rounds.map (Round.semanticPolynomial env))
      (interface.rounds.map fun round => round.challenge.eval env)
      terminal ↔
    SpecHolds interface env ∧ (output interface).eval env = terminal := by
  have split : ∀ (current : KExpr) (rounds : List (Round degree)),
      NightstreamFPrime.Spec.SumCheck.Finite.FixedPhase.Chain
        extensionOps.toOps (current.eval env)
        (rounds.map (Round.semanticPolynomial env))
        (rounds.map fun round => round.challenge.eval env) terminal ↔
      NightstreamFPrime.Spec.SumCheck.Finite.FixedPhase.Chain
        extensionOps.toOps (current.eval env)
        (rounds.map (Round.semanticPolynomial env))
        (rounds.map fun round => round.challenge.eval env)
        ((outputFrom current rounds).eval env) ∧
      (outputFrom current rounds).eval env = terminal := by
    intro current rounds
    induction rounds generalizing current with
    | nil =>
        simp [NightstreamFPrime.Spec.SumCheck.Finite.FixedPhase.Chain,
          outputFrom]
    | cons round rounds inductionHypothesis =>
        simp only [List.map_cons,
          NightstreamFPrime.Spec.SumCheck.Finite.FixedPhase.Chain, outputFrom]
        rw [show
          (round.semanticPolynomial env).evaluate extensionOps.toOps
              (round.challenge.eval env) =
            (evaluateRound round round.challenge).eval env by
          symm
          exact eval_evaluateRound env round round.challenge]
        rw [inductionHypothesis]
        tauto
  simpa only [SpecHolds, output] using
    split interface.initial interface.rounds

def Assumptions {degree roundCount : Nat}
    (interface : Interface degree roundCount) (offset : Nat) (_env : Env) :
    Prop :=
  interface.VarsBelow offset

def constraints {degree roundCount : Nat}
    (interface : Interface degree roundCount) : List Expr :=
  constraintsFrom interface.initial interface.rounds

theorem constraints_length {degree roundCount : Nat}
    (interface : Interface degree roundCount) :
    (constraints interface).length = 2 * roundCount := by
  rw [constraints, constraintsFrom_length]
  simp [Interface.rounds]

theorem constraints_varsBelow {degree roundCount : Nat}
    (interface : Interface degree roundCount) (bound : Nat)
    (below : interface.VarsBelow bound) :
    ∀ expression ∈ constraints interface, expression.VarsBelow bound := by
  apply constraintsFrom_varsBelow interface.initial interface.rounds bound
    below.1
  intro round member
  rw [Interface.rounds, List.mem_ofFn'] at member
  rcases member with ⟨index, rfl⟩
  exact below.2 index

theorem specHolds_eq_of_agree_below {degree roundCount : Nat}
    (interface : Interface degree roundCount) (bound : Nat)
    (left right : Env) (below : interface.VarsBelow bound)
    (agrees : ∀ index, index < bound → left index = right index) :
    SpecHolds interface left ↔ SpecHolds interface right := by
  have initial := interface.initial.eval_eq_of_agree_below bound left right
    below.1 agrees
  have rounds :
      interface.rounds.map (Round.semanticPolynomial left) =
        interface.rounds.map (Round.semanticPolynomial right) := by
    apply List.map_congr_left
    intro round member
    rw [Interface.rounds, List.mem_ofFn'] at member
    rcases member with ⟨index, rfl⟩
    exact (interface.round index).semanticPolynomial_eq_of_agree_below
      bound left right (below.2 index) agrees
  have challenges :
      interface.rounds.map (fun round => round.challenge.eval left) =
        interface.rounds.map (fun round => round.challenge.eval right) := by
    apply List.map_congr_left
    intro round member
    rw [Interface.rounds, List.mem_ofFn'] at member
    rcases member with ⟨index, rfl⟩
    exact (interface.round index).challenge.eval_eq_of_agree_below
      bound left right (below.2 index).2 agrees
  have outputBelow : (output interface).VarsBelow bound :=
    outputFrom_varsBelow interface.initial interface.rounds bound below.1
      (by
        intro round member
        rw [Interface.rounds, List.mem_ofFn'] at member
        rcases member with ⟨index, rfl⟩
        exact below.2 index)
  have outputEq := (output interface).eval_eq_of_agree_below bound left right
    outputBelow agrees
  unfold SpecHolds
  rw [initial, rounds, challenges, outputEq]

def main {degree roundCount : Nat}
    (interface : Interface degree roundCount) : Circuit Unit :=
  fun offset => ((), offset, (constraints interface).map Op.assertZero)

def circuit {degree roundCount : Nat}
    (interface : Interface degree roundCount) : FormalCircuit where
  main := main interface
  assumptions := Assumptions interface
  spec := fun _ env => SpecHolds interface env
  soundness := by
    intro env offset assumptions rows
    have constraintsHold : ConstraintsHold env (constraints interface) :=
      (holds_assertions_iff env (constraints interface)).mp rows
    exact (constraintsFrom_hold_iff env interface.initial
      interface.rounds).mp constraintsHold
  completeness := by
    intro env offset assumptions specification
    refine ⟨env, ?_, ?_⟩
    · intro index outside
      rfl
    · apply (holdsFlat_assertions_iff env (constraints interface)).mpr
      exact (constraintsFrom_hold_iff env interface.initial
        interface.rounds).mpr specification

@[simp] theorem circuit_ops {degree roundCount : Nat}
    (interface : Interface degree roundCount) (offset : Nat) :
    Circuit.ops (circuit interface).main offset =
      (constraints interface).map Op.assertZero := by
  rfl

theorem flatConstraints_eq {degree roundCount : Nat}
    (interface : Interface degree roundCount) (offset : Nat) :
    flatConstraints (Circuit.ops (circuit interface).main offset) =
      constraints interface := by
  rw [circuit_ops, flatConstraints_assertions_eq]

theorem soundness {degree roundCount : Nat}
    (interface : Interface degree roundCount) (env : Env) (offset : Nat)
    (assumptions : Assumptions interface offset env)
    (rows : holds env (Circuit.ops (circuit interface).main offset)) :
    SpecHolds interface env :=
  (circuit interface).soundness env offset assumptions rows

theorem completeness {degree roundCount : Nat}
    (interface : Interface degree roundCount) (env : Env) (offset : Nat)
    (assumptions : Assumptions interface offset env)
    (specification : SpecHolds interface env) :
    ∃ completed,
      AgreesOutside env completed offset
        (localLength (Circuit.ops (circuit interface).main offset)) ∧
      holdsFlat completed (Circuit.ops (circuit interface).main offset) :=
  (circuit interface).completeness env offset assumptions specification

theorem localLength_eq {degree roundCount : Nat}
    (interface : Interface degree roundCount) (offset : Nat) :
    localLength (Circuit.ops (circuit interface).main offset) = 0 := by
  change (List.map Op.localLength
    ((constraints interface).map Op.assertZero)).sum = 0
  rw [List.map_map]
  simp [Function.comp_def, Op.localLength]

theorem operations_length {degree roundCount : Nat}
    (interface : Interface degree roundCount) (offset : Nat) :
    (Circuit.ops (circuit interface).main offset).length = 2 * roundCount := by
  change ((constraints interface).map Op.assertZero).length = _
  simp [constraints_length]

theorem flatConstraints_length {degree roundCount : Nat}
    (interface : Interface degree roundCount) (offset : Nat) :
    (flatConstraints (Circuit.ops (circuit interface).main offset)).length =
      2 * roundCount := by
  change (flatConstraints ((constraints interface).map Op.assertZero)).length = _
  rw [flatConstraints_assertions_eq]
  exact constraints_length interface

theorem flatConstraints_varsBelow {degree roundCount : Nat}
    (interface : Interface degree roundCount) (offset : Nat)
    (below : interface.VarsBelow offset) :
    ∀ expression ∈ flatConstraints (Circuit.ops (circuit interface).main offset),
      expression.VarsBelow offset := by
  change ∀ expression ∈
    flatConstraints ((constraints interface).map Op.assertZero),
      expression.VarsBelow offset
  rw [flatConstraints_assertions_eq]
  exact constraints_varsBelow interface offset below

end Owned

end NightstreamFPrime.Gadgets.SumCheck.FixedChain
