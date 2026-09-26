import NightstreamFPrime.Gadgets.SumCheck.FixedChain
import NightstreamFPrime.Gadgets.Polynomial.Horner

/-!
Checks the existing SumCheck chain with one materialized Horner evaluation
per round. Evaluation at zero and one uses only affine expressions. The
final output is shared with the terminal check through the owned output.
-/

namespace NightstreamFPrime.Gadgets.SumCheck.CompactChain

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier
open NightstreamFPrime.Gadgets.Polynomial
open FixedChain

def coefficientSum : List KExpr → KExpr
  | [] => KExpr.zero
  | coefficient :: rest => KExpr.add coefficient (coefficientSum rest)

def booleanSum (coefficients : List KExpr) : KExpr :=
  KExpr.add (coefficients.headD KExpr.zero) (coefficientSum coefficients)

theorem evaluate_zero (env : Env) (coefficients : List KExpr) :
    (evaluateCoefficients KExpr.zero coefficients).eval env =
      (coefficients.headD KExpr.zero).eval env := by
  cases coefficients with
  | nil => rfl
  | cons coefficient rest =>
      simp only [evaluateCoefficients, KExpr.eval_add, KExpr.eval_mul,
        KExpr.eval_zero, List.headD_cons]
      change extensionOps.add _ (extensionOps.mul extensionOps.zero _) = _
      rw [extensionLaws.mul_comm, extensionLaws.mul_zero, extensionLaws.add_zero]

theorem evaluate_one (env : Env) (coefficients : List KExpr) :
    (evaluateCoefficients KExpr.one coefficients).eval env =
      (coefficientSum coefficients).eval env := by
  induction coefficients with
  | nil => rfl
  | cons coefficient rest ih =>
      simp only [evaluateCoefficients, coefficientSum, KExpr.eval_add,
        KExpr.eval_mul, KExpr.eval_one]
      change extensionOps.add _ (extensionOps.mul extensionOps.one _) = _
      rw [extensionLaws.one_mul, ih]
      rfl

theorem booleanSum_eval (env : Env) (coefficients : List KExpr) :
    (booleanSum coefficients).eval env =
      K.add ((evaluateCoefficients KExpr.zero coefficients).eval env)
        ((evaluateCoefficients KExpr.one coefficients).eval env) := by
  rw [evaluate_zero, evaluate_one]
  rfl

theorem coefficientSum_varsBelow (coefficients : List KExpr) (bound : Nat)
    (below : ∀ coefficient ∈ coefficients, coefficient.VarsBelow bound) :
    (coefficientSum coefficients).VarsBelow bound := by
  induction coefficients with
  | nil => exact ⟨trivial, trivial⟩
  | cons coefficient rest ih =>
      exact KExpr.add_varsBelow _ _ bound (below _ (by simp))
        (ih (fun value member => below value (by simp [member])))

theorem booleanSum_varsBelow (coefficients : List KExpr) (bound : Nat)
    (below : ∀ coefficient ∈ coefficients, coefficient.VarsBelow bound) :
    (booleanSum coefficients).VarsBelow bound := by
  apply KExpr.add_varsBelow
  · cases coefficients with
    | nil => exact ⟨trivial, trivial⟩
    | cons coefficient rest => exact below coefficient (by simp)
  · exact coefficientSum_varsBelow coefficients bound below

structure Program where
  recipes : List Expr
  checks : List Expr
  output : KExpr

def compile {degree : Nat} (start : Nat) (current : KExpr) :
    List (Round degree) → Program
  | [] => ⟨[], [], current⟩
  | round :: rounds =>
      let evaluation := Horner.compile start round.challenge round.coefficients
      let tail := compile (start + 2 * degree) evaluation.output rounds
      ⟨evaluation.recipes ++ tail.recipes,
        KExpr.equalities current (booleanSum round.coefficients) ++ tail.checks,
        tail.output⟩

/-- Use the known degree for offsets; the interface proof does not execute Horner. -/
theorem compile_cons {degree : Nat} (start : Nat) (current : KExpr)
    (round : Round degree) (rounds : List (Round degree)) :
    compile start current (round :: rounds) =
      let evaluation := Horner.compile start round.challenge round.coefficients
      let tail := compile (start + evaluation.recipes.length) evaluation.output rounds
      ⟨evaluation.recipes ++ tail.recipes,
        KExpr.equalities current (booleanSum round.coefficients) ++ tail.checks,
        tail.output⟩ := by
  simp only [compile, Horner.compile_recipes_length, Round.coefficients,
    List.length_ofFn, Nat.add_sub_cancel]

theorem compile_recipes_length {degree : Nat} (start : Nat) (current : KExpr)
    (rounds : List (Round degree)) :
    (compile start current rounds).recipes.length = 2 * degree * rounds.length := by
  induction rounds generalizing start current with
  | nil => simp [compile]
  | cons round rounds ih =>
      simp only [compile_cons, List.length_append, ih, Horner.compile_recipes_length,
        Round.coefficients, List.length_ofFn, Nat.add_sub_cancel,
        List.length_cons]
      simp [Nat.mul_add, Nat.add_comm]

theorem compile_checks_length {degree : Nat} (start : Nat) (current : KExpr)
    (rounds : List (Round degree)) :
    (compile start current rounds).checks.length = 2 * rounds.length := by
  induction rounds generalizing start current with
  | nil => simp [compile]
  | cons round rounds ih => simp [compile_cons, KExpr.equalities, ih]; omega

private theorem coefficients_below {degree : Nat} (round : Round degree)
    (bound : Nat) (below : round.VarsBelow bound) :
    ∀ coefficient ∈ round.coefficients, coefficient.VarsBelow bound := by
  intro coefficient member
  rw [Round.coefficients, List.mem_ofFn'] at member
  obtain ⟨index, rfl⟩ := member
  exact below.1 index

private theorem append_causal (start : Nat) (first second : List Expr)
    (firstCausal : RecipesCausal start first)
    (secondCausal : RecipesCausal (start + first.length) second) :
    RecipesCausal start (first ++ second) := by
  induction first generalizing start with
  | nil => simpa using secondCausal
  | cons recipe rest ih =>
      exact ⟨firstCausal.1, ih (start + 1) firstCausal.2 (by
        simpa [Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using secondCausal)⟩

theorem compile_scope {degree : Nat} (start : Nat) (current : KExpr)
    (rounds : List (Round degree)) (currentBelow : current.VarsBelow start)
    (roundsBelow : ∀ round ∈ rounds, round.VarsBelow start) :
    RecipesCausal start (compile start current rounds).recipes ∧
      (compile start current rounds).output.VarsBelow
        (start + (compile start current rounds).recipes.length) ∧
      ∀ expression ∈ (compile start current rounds).checks,
        expression.VarsBelow (start + (compile start current rounds).recipes.length) := by
  induction rounds generalizing start current with
  | nil => simpa [compile, RecipesCausal] using currentBelow
  | cons round rounds ih =>
      let evaluation := Horner.compile start round.challenge round.coefficients
      let next := start + evaluation.recipes.length
      have roundBelow := roundsBelow round (by simp)
      have evalScope := Horner.compile_causal_and_output_below start round.challenge
        round.coefficients roundBelow.2 (coefficients_below round start roundBelow)
      have tailScope := ih next evaluation.output evalScope.2 (by
        intro later member
        exact later.varsBelow_mono (roundsBelow later (by simp [member]))
          (Nat.le_add_right _ _))
      rw [compile_cons]
      refine ⟨?_, ?_, ?_⟩
      · exact append_causal start evaluation.recipes _ evalScope.1 tailScope.1
      · simpa [compile_cons, evaluation, next, List.length_append, Nat.add_assoc]
          using tailScope.2.1
      · intro expression member
        have endEq : start + (compile start current (round :: rounds)).recipes.length =
            next + (compile next evaluation.output rounds).recipes.length := by
          simp [compile_cons, evaluation, next, List.length_append, Nat.add_assoc]
        rw [compile_cons] at endEq
        rw [endEq]
        simp only [compile_cons, List.mem_append] at member
        rcases member with head | tail
        · have headBelow := KExpr.equalities_varsBelow current
            (booleanSum round.coefficients) start currentBelow
            (booleanSum_varsBelow round.coefficients start
              (coefficients_below round start roundBelow)) expression head
          exact Expr.VarsBelow.mono expression headBelow
            (Nat.le_trans (Nat.le_add_right start evaluation.recipes.length)
              (Nat.le_add_right next _))
        · exact tailScope.2.2 expression tail

/-- Equivalence for arbitrary assignments satisfying every Horner recipe.
The terminal equality is explicit, so a missing final link cannot pass. -/
theorem compile_chain_iff {degree : Nat} (env : Env) (start : Nat)
    (current : KExpr) (rounds : List (Round degree)) (terminal : K)
    (recipeRows : ConstraintsHold env
      (recipeConstraints start (compile start current rounds).recipes)) :
    (ConstraintsHold env (compile start current rounds).checks ∧
        (compile start current rounds).output.eval env = terminal) ↔
      Spec.SumCheck.Finite.FixedPhase.Chain extensionOps.toOps (current.eval env)
        (rounds.map (Round.semanticPolynomial env))
        (rounds.map fun round => round.challenge.eval env) terminal := by
  induction rounds generalizing start current with
  | nil => simp [compile, ConstraintsHold, Spec.SumCheck.Finite.FixedPhase.Chain]
  | cons round rounds ih =>
      let evaluation := Horner.compile start round.challenge round.coefficients
      let next := start + evaluation.recipes.length
      have split : ConstraintsHold env (recipeConstraints start evaluation.recipes) ∧
          ConstraintsHold env
            (recipeConstraints next (compile next evaluation.output rounds).recipes) := by
        apply (Circuit.constraintsHold_append env _ _).mp
        rw [← recipeConstraints_append]
        simpa only [compile_cons] using recipeRows
      have evaluationEq := (Horner.compile_output_sound env start round.challenge
        round.coefficients split.1).trans (Horner.evaluate_eq_messageEvaluate _ _)
      have evaluationRound : evaluation.output.eval env =
          (round.semanticPolynomial env).evaluate extensionOps.toOps
            (round.challenge.eval env) := evaluationEq
      simp only [compile_cons, Circuit.constraintsHold_append, KExpr.equalities_hold_iff,
        and_assoc, List.map_cons, Spec.SumCheck.Finite.FixedPhase.Chain]
      rw [ih next evaluation.output split.2, evaluationRound,
        booleanSum_eval, eval_evaluateCoefficients, eval_evaluateCoefficients]
      rfl

abbrev Interface := FixedChain.Owned.Interface

def program {degree roundCount : Nat} (interface : Interface degree roundCount)
    (offset : Nat) : Program := compile offset interface.initial interface.rounds

def output {degree roundCount : Nat} (interface : Interface degree roundCount)
    (offset : Nat) : KExpr := (program interface offset).output

def opsAt {degree roundCount : Nat} (interface : Interface degree roundCount)
    (offset : Nat) : List Op :=
  Op.witness (WitnessBatch.arithmetic offset (program interface offset).recipes) ::
    (program interface offset).checks.map Op.assertZero

def main {degree roundCount : Nat} (interface : Interface degree roundCount) : Circuit Unit :=
  fun offset => ((), offset + (program interface offset).recipes.length, opsAt interface offset)

def SpecHolds {degree roundCount : Nat} (interface : Interface degree roundCount)
    (offset : Nat) (env : Env) : Prop :=
  FixedChain.Owned.SpecHolds interface env ∧
    (output interface offset).eval env = (FixedChain.Owned.output interface).eval env

theorem flatConstraints_opsAt {degree roundCount : Nat}
    (interface : Interface degree roundCount) (offset : Nat) :
    flatConstraints (opsAt interface offset) =
      recipeConstraints offset (program interface offset).recipes ++
        (program interface offset).checks := by
  change recipeConstraints offset (program interface offset).recipes ++
      flatConstraints ((program interface offset).checks.map Op.assertZero) = _
  rw [flatConstraints_assertions_eq]

theorem soundness {degree roundCount : Nat} (interface : Interface degree roundCount)
    (env : Env) (offset : Nat)
    (rows : holds env (Circuit.ops (main interface) offset)) :
    SpecHolds interface offset env := by
  have recipeRows : ConstraintsHold env
      (recipeConstraints offset (program interface offset).recipes) :=
    rows (Op.witness (WitnessBatch.arithmetic offset
      (program interface offset).recipes)) (by simp [main, Circuit.ops, opsAt])
  have checkRows : ConstraintsHold env (program interface offset).checks := by
    intro expression member
    exact rows (Op.assertZero expression) (by simp [main, Circuit.ops, opsAt, member])
  have chain := (compile_chain_iff env offset interface.initial interface.rounds
    ((output interface offset).eval env) recipeRows).mp ⟨checkRows, rfl⟩
  have exactChain := (FixedChain.Owned.chain_iff_specHolds_and_output_eq
    interface env _).mp chain
  exact ⟨exactChain.1, exactChain.2.symm⟩

theorem build {degree roundCount : Nat} (interface : Interface degree roundCount)
    (env : Env) (offset : Nat) (below : interface.VarsBelow offset)
    (specification : FixedChain.Owned.SpecHolds interface env) :
    ∃ completed,
      AgreesOutside env completed offset (program interface offset).recipes.length ∧
      holdsFlat completed (Circuit.ops (main interface) offset) := by
  let completed := executeRecipes env offset (program interface offset).recipes
  have scope := compile_scope offset interface.initial interface.rounds below.1 (by
    intro round member
    rw [FixedChain.Owned.Interface.rounds, List.mem_ofFn'] at member
    obtain ⟨index, rfl⟩ := member
    exact below.2 index)
  have recipeRows := executeRecipes_holds_recipeConstraints env offset
    (program interface offset).recipes scope.1
  have previousSpec : FixedChain.Owned.SpecHolds interface completed :=
    (FixedChain.Owned.specHolds_eq_of_agree_below interface offset env completed below
      (fun index bound => (executeRecipes_agrees_below env offset
        (program interface offset).recipes index bound).symm)).mp specification
  have checks := (compile_chain_iff completed offset interface.initial interface.rounds
    ((FixedChain.Owned.output interface).eval completed) recipeRows).mpr previousSpec
  refine ⟨completed, executeRecipes_agreesOutside env offset _, ?_⟩
  change ConstraintsHold completed (flatConstraints (opsAt interface offset))
  rw [flatConstraints_opsAt, Circuit.constraintsHold_append]
  exact ⟨recipeRows, checks.1⟩

theorem localLength_eq {degree roundCount : Nat}
    (interface : Interface degree roundCount) (offset : Nat) :
    localLength (Circuit.ops (main interface) offset) =
      (program interface offset).recipes.length := by
  simp [main, Circuit.ops, opsAt, localLength, Op.localLength,
    WitnessBatch.outputLength, WitnessBatch.arithmetic, List.map_map, Function.comp_def]

def circuit {degree roundCount : Nat} (interface : Interface degree roundCount) :
    FormalCircuit where
  main := main interface
  assumptions := FixedChain.Owned.Assumptions interface
  spec := SpecHolds interface
  soundness := fun env offset _ rows => soundness interface env offset rows
  completeness := by
    intro env offset assumptions specification
    rw [localLength_eq]
    exact build interface env offset assumptions specification.1

end NightstreamFPrime.Gadgets.SumCheck.CompactChain
