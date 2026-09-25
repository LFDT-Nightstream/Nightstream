import NightstreamFPrime.Gadgets.Polynomial.Horner
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.TargetPolynomial

/-! Shared 16-multiplication construction of gamma^864 and gamma^12960. -/

namespace NightstreamFPrime.Lifecycle.PiCCS.v1_1.GammaPowers

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Gadgets.Polynomial
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier

def arguments (step : Fin 16) : Nat × Nat :=
  ([(0, 0), (1, 0), (2, 2), (3, 3), (4, 4), (5, 2), (6, 6), (7, 7),
    (8, 8), (9, 9), (10, 10), (11, 11), (12, 11), (13, 13), (14, 14),
    (15, 13)] : List (Nat × Nat)).getD step.val (0, 0)

def exponent (index : Nat) : Nat :=
  [1, 2, 3, 6, 12, 24, 27, 54, 108, 216, 432, 864, 1728, 2592,
    5184, 10368, 12960].getD index 0

theorem schedule_valid : ∀ step : Fin 16,
    (arguments step).1 ≤ step.val ∧ (arguments step).2 ≤ step.val ∧
      exponent (step.val + 1) =
        exponent (arguments step).1 + exponent (arguments step).2 := by decide

def wire (gamma : KExpr) (start : Nat) : Nat → KExpr
  | 0 => gamma
  | index + 1 => Horner.productAt (start + 2 * index)

def product (gamma : KExpr) (start : Nat) (step : Fin 16) : KExpr :=
  KExpr.mul (wire gamma start (arguments step).1)
    (wire gamma start (arguments step).2)

def recipe (gamma : KExpr) (start : Nat) (index : Fin 32) : Expr :=
  let step : Fin 16 := ⟨index.val / 2, by omega⟩
  if index.val % 2 = 0 then (product gamma start step).c0
  else (product gamma start step).c1

def recipes (gamma : KExpr) (start : Nat) : List Expr :=
  List.ofFn (recipe gamma start)

theorem recipes_length (gamma : KExpr) (start : Nat) :
    (recipes gamma start).length = 32 := by simp [recipes]

def matrixOutput (gamma : KExpr) (start : Nat) : KExpr := wire gamma start 11
def constraintOutput (gamma : KExpr) (start : Nat) : KExpr := wire gamma start 16

theorem wire_varsBelow (gamma : KExpr) (start index count : Nat)
    (gammaBelow : gamma.VarsBelow start) (bounded : index ≤ count) :
    (wire gamma start index).VarsBelow (start + 2 * count) := by
  cases index with
  | zero => exact KExpr.varsBelow_mono gamma gammaBelow (Nat.le_add_right _ _)
  | succ index =>
      change start + 2 * index < start + 2 * count ∧
        start + 2 * index + 1 < start + 2 * count
      omega

theorem recipe_varsBelow (gamma : KExpr) (start : Nat) (index : Fin 32)
    (gammaBelow : gamma.VarsBelow start) :
    (recipe gamma start index).VarsBelow (start + index.val) := by
  let step : Fin 16 := ⟨index.val / 2, by omega⟩
  have valid := schedule_valid step
  have left := wire_varsBelow gamma start (arguments step).1 step.val gammaBelow valid.1
  have right := wire_varsBelow gamma start (arguments step).2 step.val gammaBelow valid.2.1
  have both := KExpr.mul_varsBelow _ _ (start + 2 * step.val) left right
  have scope : start + 2 * step.val ≤ start + index.val := by
    dsimp [step]
    omega
  unfold recipe
  split
  · exact Expr.VarsBelow.mono _ both.1 scope
  · exact Expr.VarsBelow.mono _ both.2 scope

private theorem causal_of_get (start : Nat) (values : List Expr)
    (scope : ∀ index : Fin values.length,
      (values.get index).VarsBelow (start + index.val)) :
    RecipesCausal start values := by
  induction values generalizing start with
  | nil => trivial
  | cons value values ih =>
      refine ⟨?_, ih (start + 1) ?_⟩
      · simpa using scope ⟨0, by simp⟩
      · intro index
        have bound : index.val + 1 < (value :: values).length := by
          have := index.isLt
          simp only [List.length_cons]
          omega
        simpa [Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using
          scope ⟨index.val + 1, bound⟩

theorem recipes_causal (gamma : KExpr) (start : Nat)
    (gammaBelow : gamma.VarsBelow start) : RecipesCausal start (recipes gamma start) := by
  apply causal_of_get
  intro index
  have bound : index.val < 32 := by simpa [recipes] using index.isLt
  simpa only [recipes, List.get_ofFn] using!
    recipe_varsBelow gamma start ⟨index.val, bound⟩ gammaBelow

theorem product_sound (gamma : KExpr) (start : Nat) (env : Env)
    (rows : ConstraintsHold env (recipeConstraints start (recipes gamma start)))
    (step : Fin 16) :
    (wire gamma start (step.val + 1)).eval env =
      K.mul ((wire gamma start (arguments step).1).eval env)
        ((wire gamma start (arguments step).2).eval env) := by
  have firstBound : 2 * step.val < (recipes gamma start).length := by
    rw [recipes_length]
    have := step.isLt
    omega
  have secondBound : 2 * step.val + 1 < (recipes gamma start).length := by
    rw [recipes_length]
    have := step.isLt
    omega
  have first := recipeConstraints_value env start (recipes gamma start) rows
    (2 * step.val) firstBound
  have second := recipeConstraints_value env start (recipes gamma start) rows
    (2 * step.val + 1) secondBound
  have firstRecipe : (recipes gamma start).get ⟨2 * step.val, firstBound⟩ =
      (product gamma start step).c0 := by
    unfold recipes
    rw [List.get_ofFn]
    simp only [recipe, Fin.val_cast]
    have same : (⟨2 * step.val / 2, by omega⟩ : Fin 16) = step := by
      apply Fin.ext
      change 2 * step.val / 2 = step.val
      omega
    rw [same, if_pos (by omega)]
  have secondRecipe : (recipes gamma start).get ⟨2 * step.val + 1, secondBound⟩ =
      (product gamma start step).c1 := by
    unfold recipes
    rw [List.get_ofFn]
    simp only [recipe, Fin.val_cast]
    have same : (⟨(2 * step.val + 1) / 2, by omega⟩ : Fin 16) = step := by
      apply Fin.ext
      change (2 * step.val + 1) / 2 = step.val
      omega
    rw [same, if_neg (by omega)]
  rw [firstRecipe] at first
  rw [secondRecipe] at second
  have pair : (wire gamma start (step.val + 1)).eval env =
      (product gamma start step).eval env := by
    apply congrArg₂ K.mk
    · exact first
    · simpa [Nat.add_assoc] using! second
  exact pair.trans (KExpr.eval_mul env _ _)

private def powerLaws : TargetPolynomial.ShiftLaws extensionOps.toOps where
  one_mul := extensionLaws.one_mul
  mul_assoc := extensionLaws.mul_assoc
  mul_zero := extensionLaws.mul_zero
  mul_add := extensionLaws.left_distrib

theorem wire_sound (gamma : KExpr) (start : Nat) (env : Env)
    (rows : ConstraintsHold env (recipeConstraints start (recipes gamma start)))
    (index : Nat) (bounded : index ≤ 16) :
    (wire gamma start index).eval env =
      TargetPolynomial.power extensionOps.toOps (gamma.eval env) (exponent index) := by
  induction index using Nat.strong_induction_on with
  | h index ih =>
      cases index with
      | zero =>
          change gamma.eval env = extensionOps.mul (gamma.eval env) extensionOps.one
          exact (extensionLaws.mul_one _).symm
      | succ index =>
          let step : Fin 16 := ⟨index, by omega⟩
          have valid := schedule_valid step
          have leftLe : (arguments step).1 ≤ index := valid.1
          have rightLe : (arguments step).2 ≤ index := valid.2.1
          have left := ih (arguments step).1 (by omega) (by omega)
          have right := ih (arguments step).2 (by omega) (by omega)
          have value := product_sound gamma start env rows step
          rw [left, right] at value
          change _ = extensionOps.mul _ _ at value
          rw [← TargetPolynomial.power_add extensionOps.toOps powerLaws] at value
          rw [← valid.2.2] at value
          exact value

def SpecHolds (gamma : KExpr) (start : Nat) (env : Env) : Prop :=
  (matrixOutput gamma start).eval env =
      TargetPolynomial.power extensionOps.toOps (gamma.eval env) 864 ∧
    (constraintOutput gamma start).eval env =
      TargetPolynomial.power extensionOps.toOps (gamma.eval env) 12960

def main (gamma : KExpr) : Circuit Unit := fun start =>
  ((), start + 32, [Op.witness (WitnessBatch.arithmetic start (recipes gamma start))])

theorem soundness (gamma : KExpr) (start : Nat) (env : Env)
    (rows : holds env (Circuit.ops (main gamma) start)) : SpecHolds gamma start env := by
  have recipeRows : ConstraintsHold env (recipeConstraints start (recipes gamma start)) :=
    rows (Op.witness (WitnessBatch.arithmetic start (recipes gamma start))) (by simp [main, Circuit.ops])
  exact ⟨wire_sound gamma start env recipeRows 11 (by decide),
    wire_sound gamma start env recipeRows 16 (by decide)⟩

theorem build (gamma : KExpr) (start : Nat) (env : Env)
    (gammaBelow : gamma.VarsBelow start) :
    ∃ completed, AgreesOutside env completed start 32 ∧
      holdsFlat completed (Circuit.ops (main gamma) start) := by
  let completed := executeRecipes env start (recipes gamma start)
  refine ⟨completed, ?_, ?_⟩
  · simpa [recipes_length] using executeRecipes_agreesOutside env start (recipes gamma start)
  · change ConstraintsHold completed (recipeConstraints start (recipes gamma start) ++ [])
    simp only [List.append_nil]
    exact executeRecipes_holds_recipeConstraints env start (recipes gamma start)
      (recipes_causal gamma start gammaBelow)

def circuit (gamma : KExpr) : FormalCircuit where
  main := main gamma
  assumptions := fun start _ => gamma.VarsBelow start
  spec := SpecHolds gamma
  soundness := fun env start _ rows => soundness gamma start env rows
  completeness := by
    intro env start assumptions _
    have result := build gamma start env assumptions
    simpa [main, Circuit.ops, localLength, Op.localLength, WitnessBatch.outputLength,
      WitnessBatch.arithmetic, recipes_length] using result

theorem localLength_eq (gamma : KExpr) (start : Nat) :
    localLength (Circuit.ops (circuit gamma).main start) = 32 := by
  change (recipes gamma start).length = 32
  exact recipes_length gamma start

theorem flatConstraints_eq (gamma : KExpr) (start : Nat) :
    flatConstraints (Circuit.ops (circuit gamma).main start) =
      recipeConstraints start (recipes gamma start) := by
  change recipeConstraints start (recipes gamma start) ++ [] = _
  rw [List.append_nil]

theorem flatConstraints_length (gamma : KExpr) (start : Nat) :
    (flatConstraints (Circuit.ops (circuit gamma).main start)).length = 32 := by
  rw [flatConstraints_eq, recipeConstraints_length, recipes_length]

theorem flatConstraints_varsBelow (gamma : KExpr) (start : Nat)
    (gammaBelow : gamma.VarsBelow start) :
    ∀ expression ∈ flatConstraints (Circuit.ops (circuit gamma).main start),
      expression.VarsBelow (start + 32) := by
  rw [flatConstraints_eq]
  have scope := recipeConstraints_varsBelow_of_causal start (recipes gamma start)
    (recipes_causal gamma start gammaBelow)
  simpa only [recipes_length] using scope

end NightstreamFPrime.Lifecycle.PiCCS.v1_1.GammaPowers
