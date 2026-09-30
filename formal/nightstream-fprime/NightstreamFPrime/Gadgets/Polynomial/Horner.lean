import Mathlib.Tactic.Ring
import NightstreamFPrime.Circuit.Quadratic
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier.Algebra
import NightstreamFPrime.Spec.SumCheck.Polynomial

/-!
Obligation: Evaluate one constant-first polynomial over the production
quadratic extension with a causal, reusable accumulator.

Inputs:
- one extension-field point;
- a non-authoritative finite coefficient list;
- one external expected-output wire.

Outputs:
- the expected output, constrained to canonical Horner evaluation.

Constraint groups:
- C1: three rank-one witness recipes for each non-final extension
  multiplication;
- C2: two final component equalities.

The singleton branch returns its coefficient directly, so the circuit does
not multiply by the semantic trailing zero. The correctness theorem proves
equality to `SumCheck.Finite.Message.evaluateCoefficients`.
-/

namespace NightstreamFPrime.Gadgets.Polynomial.Horner

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier

/-- Three rank-one recipes for one extension-field multiplication whose
cells start at `start`:
`t = l₁·r₁`, `c₀ = l₀·r₀ + 7t`, `c₁ = (l₀ + l₁)·(r₀ + r₁) − c₀ + 6t`.
With `X² = 7`, the last two cells are the two product components. -/
def mulRecipes (start : Nat) (left right : KExpr) : List Expr :=
  [left.c1 * right.c1,
    left.c0 * right.c0 + 7 * Expr.var start,
    (left.c0 + left.c1) * (right.c0 + right.c1) +
      (-Expr.var (start + 1) + 6 * Expr.var start)]

/-- The two product components of the multiplication stored at `start`. -/
def productAt (start : Nat) : KExpr :=
  ⟨Expr.var (start + 1), Expr.var (start + 2)⟩

@[simp] theorem mulRecipes_length (start : Nat) (left right : KExpr) :
    (mulRecipes start left right).length = 3 := by
  rfl

structure Program where
  recipes : List Expr
  output : KExpr

/-- High coefficients are evaluated first. Every reused accumulator is
materialized before the next lower coefficient reads it. -/
def compile (start : Nat) (point : KExpr) : List KExpr → Program
  | [] => ⟨[], KExpr.zero⟩
  | [coefficient] => ⟨[], coefficient⟩
  | coefficient :: next :: rest =>
      let tail := compile start point (next :: rest)
      let productStart := start + tail.recipes.length
      let product := productAt productStart
      ⟨tail.recipes ++ mulRecipes productStart point tail.output,
        KExpr.add coefficient product⟩

/-- Linear executable builder for `compile`. The difference list preserves
the structural compiler's tail-first recipe order, while `recipeCount` avoids
rescanning the growing recipe prefix at each Horner step. -/
structure ProgramBuilder where
  recipes : List Expr → List Expr
  recipeCount : Nat
  output : KExpr

@[inline] def compileBuilder (start : Nat) (point : KExpr) :
    List KExpr → ProgramBuilder
  | [] => ⟨fun suffix => suffix, 0, KExpr.zero⟩
  | [coefficient] => ⟨fun suffix => suffix, 0, coefficient⟩
  | coefficient :: next :: rest =>
      let tail := compileBuilder start point (next :: rest)
      let productStart := start + tail.recipeCount
      let product := productAt productStart
      { recipes := fun suffix =>
          tail.recipes
            (mulRecipes productStart point tail.output ++ suffix)
        recipeCount := tail.recipeCount + 3
        output := KExpr.add coefficient product }

@[inline] def compileFast (start : Nat) (point : KExpr)
    (coefficients : List KExpr) : Program :=
  let builder := compileBuilder start point coefficients
  ⟨builder.recipes [], builder.output⟩

private theorem compileBuilder_recipeCount (start : Nat) (point : KExpr) :
    ∀ coefficients : List KExpr,
      (compileBuilder start point coefficients).recipeCount =
        (compile start point coefficients).recipes.length
  | [] => rfl
  | [_] => rfl
  | coefficient :: next :: rest => by
      simp only [compileBuilder, compile, List.length_append,
        mulRecipes_length]
      rw [compileBuilder_recipeCount start point (next :: rest)]

private theorem compileBuilder_output (start : Nat) (point : KExpr) :
    ∀ coefficients : List KExpr,
      (compileBuilder start point coefficients).output =
        (compile start point coefficients).output
  | [] => rfl
  | [_] => rfl
  | coefficient :: next :: rest => by
      simp only [compileBuilder, compile]
      rw [compileBuilder_recipeCount start point (next :: rest)]

private theorem compileBuilder_recipes (start : Nat) (point : KExpr) :
    ∀ (coefficients : List KExpr) (suffix : List Expr),
      (compileBuilder start point coefficients).recipes suffix =
        (compile start point coefficients).recipes ++ suffix
  | [], _ => rfl
  | [_], _ => rfl
  | coefficient :: next :: rest, suffix => by
      simp only [compileBuilder, compile]
      rw [compileBuilder_recipes start point (next :: rest)
        (mulRecipes
          (start + (compileBuilder start point (next :: rest)).recipeCount)
          point (compileBuilder start point (next :: rest)).output ++ suffix)]
      rw [compileBuilder_output start point (next :: rest),
        compileBuilder_recipeCount start point (next :: rest)]
      simp only [List.append_assoc]

theorem compileFast_eq_compile (start : Nat) (point : KExpr)
    (coefficients : List KExpr) :
    compileFast start point coefficients = compile start point coefficients := by
  apply congrArg₂ Program.mk
  · simpa [compileFast] using
      compileBuilder_recipes start point coefficients []
  · simpa [compileFast] using
      compileBuilder_output start point coefficients

@[csimp] theorem compile_eq_compileFast : @compile = @compileFast := by
  funext start point coefficients
  exact (compileFast_eq_compile start point coefficients).symm

/-- Optimized semantic Horner form with no trailing multiplication by zero. -/
def evaluate (point : K) : List K → K
  | [] => K.zero
  | [coefficient] => coefficient
  | coefficient :: next :: rest =>
      K.add coefficient (K.mul point (evaluate point (next :: rest)))

theorem evaluate_eq_messageEvaluate (point : K) : ∀ coefficients : List K,
    evaluate point coefficients =
      SumCheck.Finite.Message.evaluateCoefficients
        extensionOps.toOps point coefficients
  | [] => rfl
  | [coefficient] => by
      change coefficient = extensionOps.add coefficient
        (extensionOps.mul point extensionOps.zero)
      rw [extensionLaws.mul_zero, extensionLaws.add_zero]
  | coefficient :: next :: rest => by
      simp only [evaluate, SumCheck.Finite.Message.evaluateCoefficients]
      rw [evaluate_eq_messageEvaluate point (next :: rest)]
      rfl

private theorem add_varsBelow (left right : KExpr) (bound : Nat)
    (leftBelow : left.VarsBelow bound)
    (rightBelow : right.VarsBelow bound) :
    (KExpr.add left right).VarsBelow bound := by
  exact ⟨⟨leftBelow.1, rightBelow.1⟩,
    ⟨leftBelow.2, rightBelow.2⟩⟩

/-- Causality composes across two recipe lists at the exact boundary. -/
theorem recipesCausal_concat (start : Nat) (first second : List Expr)
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

/-- Each recipe reads its operands and only earlier cells of its own
product. -/
theorem mulRecipes_causal (start : Nat) (left right : KExpr)
    (leftBelow : left.VarsBelow start)
    (rightBelow : right.VarsBelow start) :
    RecipesCausal start (mulRecipes start left right) := by
  have left1 : left.VarsBelow (start + 1) :=
    left.varsBelow_mono leftBelow (by omega)
  have right1 : right.VarsBelow (start + 1) :=
    right.varsBelow_mono rightBelow (by omega)
  have left2 : left.VarsBelow (start + 1 + 1) :=
    left.varsBelow_mono leftBelow (by omega)
  have right2 : right.VarsBelow (start + 1 + 1) :=
    right.varsBelow_mono rightBelow (by omega)
  refine ⟨⟨leftBelow.2, rightBelow.2⟩, ⟨⟨left1.1, right1.1⟩, trivial, ?_⟩,
    ⟨⟨⟨left2.1, left2.2⟩, ⟨right2.1, right2.2⟩⟩, ⟨trivial, ?_⟩, trivial, ?_⟩,
    trivial⟩
  all_goals
    simp only [Expr.VarsBelow]
    omega

private theorem productAt_varsBelow (start : Nat) :
    (productAt start).VarsBelow (start + 3) := by
  unfold productAt KExpr.VarsBelow Expr.VarsBelow
  omega

theorem compile_recipes_length (start : Nat) (point : KExpr) :
    ∀ coefficients : List KExpr,
    (compile start point coefficients).recipes.length =
      3 * (coefficients.length - 1)
  | [] => rfl
  | [coefficient] => rfl
  | coefficient :: next :: rest => by
      simp only [compile, List.length_append, mulRecipes_length,
        List.length_cons]
      rw [compile_recipes_length start point (next :: rest)]
      simp only [List.length_cons]
      omega

/-- Causality and output scope are proved together so no concrete coefficient
count is reduced in the kernel. -/
theorem compile_causal_and_output_below
    (start : Nat) (point : KExpr) (coefficients : List KExpr)
    (pointBelow : point.VarsBelow start)
    (coefficientsBelow : ∀ coefficient ∈ coefficients,
      coefficient.VarsBelow start) :
    RecipesCausal start (compile start point coefficients).recipes ∧
      (compile start point coefficients).output.VarsBelow
        (start + (compile start point coefficients).recipes.length) := by
  induction coefficients generalizing start with
  | nil =>
      exact ⟨trivial, by simp [compile, KExpr.zero, KExpr.VarsBelow,
        Expr.VarsBelow]⟩
  | cons coefficient rest inductionHypothesis =>
      cases rest with
      | nil =>
          refine ⟨trivial, ?_⟩
          simpa [compile] using
            coefficientsBelow coefficient (by simp)
      | cons next rest =>
          let tail := compile start point (next :: rest)
          let productStart := start + tail.recipes.length
          have tailCoefficientsBelow : ∀ current ∈ next :: rest,
              current.VarsBelow start := by
            intro current member
            exact coefficientsBelow current (by simp [member])
          have tailProof := inductionHypothesis
            (start := start) pointBelow tailCoefficientsBelow
          have pointAtProduct : point.VarsBelow productStart :=
            point.varsBelow_mono pointBelow (by unfold productStart; omega)
          have causal : RecipesCausal start
              (tail.recipes ++ mulRecipes productStart point tail.output) :=
            recipesCausal_concat start tail.recipes
              (mulRecipes productStart point tail.output) tailProof.1
              (mulRecipes_causal productStart point tail.output
                pointAtProduct tailProof.2)
          have coefficientAtFinal : coefficient.VarsBelow
              (productStart + 3) :=
            coefficient.varsBelow_mono
              (coefficientsBelow coefficient (by simp)) (by
                unfold productStart
                omega)
          have productBelow := productAt_varsBelow productStart
          refine ⟨?_, ?_⟩
          · simpa [compile, tail, productStart] using causal
          · have outputBelow := add_varsBelow coefficient
              (productAt productStart) (productStart + 3)
              coefficientAtFinal productBelow
            simpa [compile, tail, productStart] using! outputBelow

theorem compile_causal
    (start : Nat) (point : KExpr) (coefficients : List KExpr)
    (pointBelow : point.VarsBelow start)
    (coefficientsBelow : ∀ coefficient ∈ coefficients,
      coefficient.VarsBelow start) :
    RecipesCausal start (compile start point coefficients).recipes :=
  (compile_causal_and_output_below start point coefficients pointBelow
    coefficientsBelow).1

open Fin.CommRing in
/-- The three recipe rows force the stored pair to the extension product. -/
theorem productAt_sound (env : Env) (start : Nat)
    (left right : KExpr)
    (rows : ConstraintsHold env
      (recipeConstraints start (mulRecipes start left right))) :
    (productAt start).eval env = K.mul (left.eval env) (right.eval env) := by
  have highRow : (Expr.var start - left.c1 * right.c1).eval env = 0 :=
    rows _ (by simp [mulRecipes, recipeConstraints])
  have c0Row : (Expr.var (start + 1) -
      (left.c0 * right.c0 + 7 * Expr.var start)).eval env = 0 :=
    rows _ (by simp [mulRecipes, recipeConstraints])
  have c1Row : (Expr.var (start + 1 + 1) -
      ((left.c0 + left.c1) * (right.c0 + right.c1) +
        (-Expr.var (start + 1) + 6 * Expr.var start))).eval env = 0 :=
    rows _ (by simp [mulRecipes, recipeConstraints])
  have seven : (7 : Expr).eval env = (7 : F) := rfl
  have six : (6 : Expr).eval env = (6 : F) := rfl
  have high : env start = left.c1.eval env * right.c1.eval env :=
    sub_eq_zero.mp (by simpa using highRow)
  have c0 : env (start + 1) =
      left.c0.eval env * right.c0.eval env + 7 * env start :=
    sub_eq_zero.mp (by simpa [seven] using c0Row)
  have c1 : env (start + 1 + 1) =
      (left.c0.eval env + left.c1.eval env) *
          (right.c0.eval env + right.c1.eval env) +
        (-env (start + 1) + 6 * env start) :=
    sub_eq_zero.mp (by simpa [six] using c1Row)
  apply congrArg₂ K.mk
  · change env (start + 1) =
      left.c0.eval env * right.c0.eval env +
        7 * left.c1.eval env * right.c1.eval env
    rw [c0, high]
    ring
  · change env (start + 1 + 1) =
      left.c0.eval env * right.c1.eval env +
        left.c1.eval env * right.c0.eval env
    rw [c1, c0, high]
    ring

theorem compile_output_sound (env : Env) (start : Nat) (point : KExpr) :
    ∀ coefficients : List KExpr,
    ConstraintsHold env
      (recipeConstraints start (compile start point coefficients).recipes) →
    (compile start point coefficients).output.eval env =
      evaluate (point.eval env) (coefficients.map (KExpr.eval env))
  | [], _ => rfl
  | [coefficient], _ => rfl
  | coefficient :: next :: rest, rows => by
      let tail := compile start point (next :: rest)
      let productStart := start + tail.recipes.length
      have split :
          ConstraintsHold env (recipeConstraints start tail.recipes) ∧
          ConstraintsHold env
            (recipeConstraints productStart
              (mulRecipes productStart point tail.output)) := by
        apply (constraintsHold_append env _ _).mp
        rw [← recipeConstraints_append]
        simpa [compile, tail, productStart] using rows
      have tailSound := compile_output_sound env start point
        (next :: rest) split.1
      have productSound := productAt_sound env productStart point
        tail.output split.2
      simp only [compile, KExpr.eval_add, List.map_cons, evaluate]
      rw [productSound, tailSound]
      rfl

structure Interface where
  point : Nat → KExpr
  coefficients : Nat → List KExpr
  expected : Nat → KExpr

def Assumptions (interface : Interface) (offset : Nat) (_env : Env) : Prop :=
  (interface.point offset).VarsBelow offset ∧
    (∀ coefficient ∈ interface.coefficients offset,
      coefficient.VarsBelow offset) ∧
    (interface.expected offset).VarsBelow offset

def SpecHolds (interface : Interface) (offset : Nat) (env : Env) : Prop :=
  (interface.expected offset).eval env =
    SumCheck.Finite.Message.evaluateCoefficients extensionOps.toOps
      ((interface.point offset).eval env)
      ((interface.coefficients offset).map (KExpr.eval env))

def program (interface : Interface) (offset : Nat) : Program :=
  compile offset (interface.point offset) (interface.coefficients offset)

def allAssertions (interface : Interface) (offset : Nat) : List Expr :=
  KExpr.equalities (program interface offset).output
    (interface.expected offset)

def opsAt (interface : Interface) (offset : Nat) : List Op :=
  [Op.witness (WitnessBatch.arithmetic offset
    (program interface offset).recipes)] ++
    (allAssertions interface offset).map Op.assertZero

def main (interface : Interface) : Circuit Unit := fun offset =>
  ((), offset + (program interface offset).recipes.length,
    opsAt interface offset)

@[simp] theorem main_ops (interface : Interface) (offset : Nat) :
    Circuit.ops (main interface) offset = opsAt interface offset := by
  rfl

private theorem flatConstraints_opsAt (interface : Interface) (offset : Nat) :
    flatConstraints (opsAt interface offset) =
      recipeConstraints offset (program interface offset).recipes ++
        allAssertions interface offset := by
  simp [flatConstraints, opsAt, Op.flatConstraints, allAssertions,
    KExpr.equalities]

theorem soundness (interface : Interface) (env : Env) (offset : Nat)
    (_assumptions : Assumptions interface offset env)
    (rows : holds env (Circuit.ops (main interface) offset)) :
    SpecHolds interface offset env := by
  have recipeRows : ConstraintsHold env
      (recipeConstraints offset (program interface offset).recipes) :=
    rows (Op.witness (WitnessBatch.arithmetic offset
      (program interface offset).recipes))
      (by simp [main_ops, opsAt])
  have assertionRows : ConstraintsHold env
      (allAssertions interface offset) := by
    intro expression member
    exact rows (Op.assertZero expression) (by
      simp [main_ops, opsAt, member])
  have outputExpected := (KExpr.equalities_hold_iff env
    (program interface offset).output (interface.expected offset)).mp
      (by simpa [allAssertions] using assertionRows)
  have outputSemantic := compile_output_sound env offset
    (interface.point offset) (interface.coefficients offset) recipeRows
  unfold SpecHolds
  exact outputExpected.symm.trans <|
    outputSemantic.trans <|
      evaluate_eq_messageEvaluate _ _

theorem completeness (interface : Interface) (env : Env) (offset : Nat)
    (assumptions : Assumptions interface offset env)
    (specification : SpecHolds interface offset env) :
    ∃ completed,
      AgreesOutside env completed offset
        (localLength (Circuit.ops (main interface) offset)) ∧
      holdsFlat completed (Circuit.ops (main interface) offset) := by
  let recipes := (program interface offset).recipes
  let completed := executeRecipes env offset recipes
  have causal : RecipesCausal offset recipes :=
    compile_causal offset (interface.point offset)
      (interface.coefficients offset) assumptions.1 assumptions.2.1
  have recipeRows : ConstraintsHold completed
      (recipeConstraints offset recipes) :=
    executeRecipes_holds_recipeConstraints env offset recipes causal
  have agreesBelow : ∀ index, index < offset → completed index = env index :=
    executeRecipes_agrees_below env offset recipes
  have pointEval : (interface.point offset).eval completed =
      (interface.point offset).eval env :=
    (interface.point offset).eval_eq_of_agree_below offset completed env
      assumptions.1 agreesBelow
  have coefficientsEval :
      (interface.coefficients offset).map (KExpr.eval completed) =
        (interface.coefficients offset).map (KExpr.eval env) := by
    apply List.map_congr_left
    intro coefficient member
    exact coefficient.eval_eq_of_agree_below offset completed env
      (assumptions.2.1 coefficient member) agreesBelow
  have expectedEval : (interface.expected offset).eval completed =
      (interface.expected offset).eval env :=
    (interface.expected offset).eval_eq_of_agree_below offset completed env
      assumptions.2.2 agreesBelow
  have outputSemantic := compile_output_sound completed offset
    (interface.point offset) (interface.coefficients offset) recipeRows
  have outputExpected : (program interface offset).output.eval completed =
      (interface.expected offset).eval completed := by
    calc
      (program interface offset).output.eval completed =
          evaluate ((interface.point offset).eval completed)
            ((interface.coefficients offset).map (KExpr.eval completed)) :=
        outputSemantic
      _ = SumCheck.Finite.Message.evaluateCoefficients extensionOps.toOps
          ((interface.point offset).eval completed)
          ((interface.coefficients offset).map (KExpr.eval completed)) :=
        evaluate_eq_messageEvaluate _ _
      _ = SumCheck.Finite.Message.evaluateCoefficients extensionOps.toOps
          ((interface.point offset).eval env)
          ((interface.coefficients offset).map (KExpr.eval env)) := by
        rw [pointEval, coefficientsEval]
      _ = (interface.expected offset).eval env := specification.symm
      _ = (interface.expected offset).eval completed := expectedEval.symm
  have assertionRows : ConstraintsHold completed
      (allAssertions interface offset) :=
    (KExpr.equalities_hold_iff completed
      (program interface offset).output (interface.expected offset)).mpr
        outputExpected
  refine ⟨completed, ?_, ?_⟩
  · rw [main_ops]
    change AgreesOutside env completed offset recipes.length
    exact executeRecipes_agreesOutside env offset recipes
  · change ConstraintsHold completed
      (flatConstraints (opsAt interface offset))
    rw [flatConstraints_opsAt]
    exact (constraintsHold_append completed _ _).mpr
      ⟨recipeRows, assertionRows⟩

def circuit (interface : Interface) : FormalCircuit where
  main := main interface
  assumptions := Assumptions interface
  spec := SpecHolds interface
  soundness := soundness interface
  completeness := completeness interface

theorem localLength_eq (interface : Interface) (offset : Nat) :
    localLength (Circuit.ops (circuit interface).main offset) =
      3 * ((interface.coefficients offset).length - 1) := by
  calc
    localLength (Circuit.ops (circuit interface).main offset) =
        (program interface offset).recipes.length := by
      change localLength (opsAt interface offset) = _
      simp [opsAt, localLength, allAssertions, KExpr.equalities,
        Op.localLength]
    _ = 3 * ((interface.coefficients offset).length - 1) := by
      exact compile_recipes_length offset (interface.point offset)
        (interface.coefficients offset)

theorem operations_length (interface : Interface) (offset : Nat) :
    (Circuit.ops (circuit interface).main offset).length = 3 := by
  change (opsAt interface offset).length = 3
  simp [opsAt, allAssertions, KExpr.equalities]

theorem flatConstraints_length (interface : Interface) (offset : Nat) :
    (flatConstraints (Circuit.ops (circuit interface).main offset)).length =
      3 * ((interface.coefficients offset).length - 1) + 2 := by
  change (flatConstraints (opsAt interface offset)).length = _
  rw [flatConstraints_opsAt, List.length_append,
    recipeConstraints_length]
  simp [allAssertions, KExpr.equalities, program, compile_recipes_length]

/-- The specification is stable when every external input wire is unchanged. -/
theorem specHolds_of_agree_below (interface : Interface) (offset : Nat)
    (before after : Env) (assumptions : Assumptions interface offset before)
    (agrees : ∀ index, index < offset → after index = before index)
    (specification : SpecHolds interface offset before) :
    SpecHolds interface offset after := by
  have pointEq : (interface.point offset).eval after =
      (interface.point offset).eval before :=
    (interface.point offset).eval_eq_of_agree_below offset after before
      assumptions.1 agrees
  have coefficientsEq :
      (interface.coefficients offset).map (KExpr.eval after) =
        (interface.coefficients offset).map (KExpr.eval before) := by
    apply List.map_congr_left
    intro coefficient member
    exact coefficient.eval_eq_of_agree_below offset after before
      (assumptions.2.1 coefficient member) agrees
  have expectedEq : (interface.expected offset).eval after =
      (interface.expected offset).eval before :=
    (interface.expected offset).eval_eq_of_agree_below offset after before
      assumptions.2.2 agrees
  unfold SpecHolds at specification ⊢
  rw [expectedEq, pointEq, coefficientsEq]
  exact specification

/-- Every flattened child row reads only external wires or this child's
completed private interval. -/
theorem flatConstraints_varsBelow (interface : Interface) (offset : Nat)
    (env : Env) (assumptions : Assumptions interface offset env) :
    ∀ expression ∈ flatConstraints (Circuit.ops (circuit interface).main offset),
      expression.VarsBelow
        (offset + (program interface offset).recipes.length) := by
  change ∀ expression ∈ flatConstraints (opsAt interface offset), _
  rw [flatConstraints_opsAt]
  intro expression member
  rcases List.mem_append.mp member with recipeMember | assertionMember
  · exact recipeConstraints_varsBelow_of_causal offset
      (program interface offset).recipes
      (compile_causal offset (interface.point offset)
        (interface.coefficients offset) assumptions.1 assumptions.2.1)
      expression recipeMember
  · have outputBelow := (compile_causal_and_output_below offset
      (interface.point offset) (interface.coefficients offset)
      assumptions.1 assumptions.2.1).2
    have expectedBelow : (interface.expected offset).VarsBelow
        (offset + (program interface offset).recipes.length) :=
      (interface.expected offset).varsBelow_mono assumptions.2.2 (by omega)
    exact KExpr.equalities_varsBelow
      (program interface offset).output (interface.expected offset)
      (offset + (program interface offset).recipes.length)
      outputBelow expectedBelow expression (by
        simpa [allAssertions] using assertionMember)

/-! ## Child-owned output variant -/

namespace Owned

/-!
Obligation: Evaluate one constant-first polynomial and expose the compiler
output directly to a parent circuit.

Unlike `Horner.Interface`, this interface has no external expected-output
wire. The child owns its accumulator interval. A parent can use `output`
without adding copy rows.
-/

structure Interface where
  point : Nat → KExpr
  coefficients : Nat → List KExpr

def program (interface : Interface) (offset : Nat) : Program :=
  compile offset (interface.point offset) (interface.coefficients offset)

def output (interface : Interface) (offset : Nat) : KExpr :=
  (program interface offset).output

def Assumptions (interface : Interface) (offset : Nat) (_env : Env) : Prop :=
  (interface.point offset).VarsBelow offset ∧
    ∀ coefficient ∈ interface.coefficients offset,
      coefficient.VarsBelow offset

def SpecHolds (interface : Interface) (offset : Nat) (env : Env) : Prop :=
  (output interface offset).eval env =
    SumCheck.Finite.Message.evaluateCoefficients extensionOps.toOps
      ((interface.point offset).eval env)
      ((interface.coefficients offset).map (KExpr.eval env))

def opsAt (interface : Interface) (offset : Nat) : List Op :=
  [Op.witness (WitnessBatch.arithmetic offset
    (program interface offset).recipes)]

def main (interface : Interface) : Circuit Unit := fun offset =>
  ((), offset + (program interface offset).recipes.length,
    opsAt interface offset)

@[simp] theorem main_ops (interface : Interface) (offset : Nat) :
    Circuit.ops (main interface) offset = opsAt interface offset := by
  rfl

theorem flatConstraints_opsAt (interface : Interface) (offset : Nat) :
    flatConstraints (opsAt interface offset) =
      recipeConstraints offset (program interface offset).recipes := by
  simp [flatConstraints, opsAt, Op.flatConstraints]

theorem soundness (interface : Interface) (env : Env) (offset : Nat)
    (_assumptions : Assumptions interface offset env)
    (rows : holds env (Circuit.ops (main interface) offset)) :
    SpecHolds interface offset env := by
  have recipeRows : ConstraintsHold env
      (recipeConstraints offset (program interface offset).recipes) :=
    rows (Op.witness (WitnessBatch.arithmetic offset
      (program interface offset).recipes))
      (by simp [main_ops, opsAt])
  exact (compile_output_sound env offset (interface.point offset)
    (interface.coefficients offset) recipeRows).trans
      (evaluate_eq_messageEvaluate _ _)

/-- Honest execution constructs the owned result with no semantic premise. -/
theorem completeness (interface : Interface) (env : Env) (offset : Nat)
    (assumptions : Assumptions interface offset env) :
    ∃ completed,
      AgreesOutside env completed offset
        (localLength (Circuit.ops (main interface) offset)) ∧
      holdsFlat completed (Circuit.ops (main interface) offset) := by
  let recipes := (program interface offset).recipes
  let completed := executeRecipes env offset recipes
  have causal : RecipesCausal offset recipes :=
    compile_causal offset (interface.point offset)
      (interface.coefficients offset) assumptions.1 assumptions.2
  have recipeRows : ConstraintsHold completed
      (recipeConstraints offset recipes) :=
    executeRecipes_holds_recipeConstraints env offset recipes causal
  refine ⟨completed, ?_, ?_⟩
  · change AgreesOutside env completed offset recipes.length
    exact executeRecipes_agreesOutside env offset recipes
  · change ConstraintsHold completed (flatConstraints (opsAt interface offset))
    rw [flatConstraints_opsAt]
    exact recipeRows

def circuit (interface : Interface) : FormalCircuit where
  main := main interface
  assumptions := Assumptions interface
  spec := SpecHolds interface
  soundness := soundness interface
  completeness := fun env offset assumptions _specification =>
    completeness interface env offset assumptions

@[simp] theorem circuit_ops (interface : Interface) (offset : Nat) :
    Circuit.ops (circuit interface).main offset = opsAt interface offset := by
  rfl

theorem build (interface : Interface) (env : Env) (offset : Nat)
    (assumptions : Assumptions interface offset env) :
    ∃ completed,
      AgreesOutside env completed offset
        (localLength (Circuit.ops (circuit interface).main offset)) ∧
      holdsFlat completed (Circuit.ops (circuit interface).main offset) :=
  completeness interface env offset assumptions

theorem localLength_eq (interface : Interface) (offset : Nat) :
    localLength (Circuit.ops (circuit interface).main offset) =
      3 * ((interface.coefficients offset).length - 1) := by
  change (program interface offset).recipes.length = _
  exact compile_recipes_length offset (interface.point offset)
    (interface.coefficients offset)

theorem operations_length (interface : Interface) (offset : Nat) :
    (Circuit.ops (circuit interface).main offset).length = 1 := by
  rfl

theorem flatConstraints_length (interface : Interface) (offset : Nat) :
    (flatConstraints (Circuit.ops (circuit interface).main offset)).length =
      3 * ((interface.coefficients offset).length - 1) := by
  change (flatConstraints (Circuit.ops (main interface) offset)).length = _
  rw [main_ops, flatConstraints_opsAt, recipeConstraints_length]
  exact compile_recipes_length offset (interface.point offset)
    (interface.coefficients offset)

theorem flatConstraints_varsBelow (interface : Interface) (offset : Nat)
    (_env : Env) (assumptions : Assumptions interface offset _env) :
    ∀ expression ∈ flatConstraints
      (Circuit.ops (circuit interface).main offset),
      expression.VarsBelow
        (offset + localLength (Circuit.ops (circuit interface).main offset)) := by
  have causal : RecipesCausal offset (program interface offset).recipes :=
    compile_causal offset (interface.point offset)
      (interface.coefficients offset) assumptions.1 assumptions.2
  have scope := recipeConstraints_varsBelow_of_causal offset
    (program interface offset).recipes causal
  change ∀ expression ∈ flatConstraints (Circuit.ops (main interface) offset),
    expression.VarsBelow
      (offset + localLength (Circuit.ops (main interface) offset))
  rw [main_ops, flatConstraints_opsAt]
  simpa [opsAt, localLength, Op.localLength] using scope

/-- The owned result lies inside the child's declared symbolic interval. -/
theorem output_varsBelow (interface : Interface) (offset : Nat)
    (env : Env) (assumptions : Assumptions interface offset env) :
    (output interface offset).VarsBelow
      (offset + localLength
        (Circuit.ops (circuit interface).main offset)) := by
  have below := (compile_causal_and_output_below offset
    (interface.point offset) (interface.coefficients offset)
      assumptions.1 assumptions.2).2
  change (program interface offset).output.VarsBelow
    (offset + localLength (Circuit.ops (main interface) offset))
  simpa [main, opsAt, localLength, Op.localLength] using! below

end Owned

end NightstreamFPrime.Gadgets.Polynomial.Horner
