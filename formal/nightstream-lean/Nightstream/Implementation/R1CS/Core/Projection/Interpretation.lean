import Nightstream.Implementation.R1CS.Core.Projection.Polynomial

/-! R1CS interpretation theorems for projection traces and polynomial identities. -/

namespace Nightstream.Implementation.R1CS.ProjectionProgram

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program

set_option maxRecDepth 4096

local instance : Std.Associative (fun (a b : F) => a + b) :=
  ⟨fadd_assoc⟩
local instance : Std.Commutative (fun (a b : F) => a + b) :=
  ⟨fadd_comm⟩
local instance : Std.Associative (fun (a b : F) => a * b) :=
  ⟨fmul_assoc⟩
local instance : Std.Commutative (fun (a b : F) => a * b) :=
  ⟨fmul_comm⟩
local instance : Std.Associative K.add := ⟨K.add_assoc⟩
local instance : Std.Commutative K.add := ⟨K.add_comm⟩
local instance : Std.Associative K.mul := ⟨K.mul_assoc⟩
local instance : Std.Commutative K.mul := ⟨K.mul_comm⟩
/-! ## Exact-definition interpretation -/
def DefinitionsHold (assignment : Nat → Nat)
    (definitions : List Definition) : Prop :=
  ∀ definition ∈ definitions, definition.Holds assignment
private theorem rawLcEval_append (assignment : Nat → Nat)
    (left right : List (Nat × Nat)) :
    rawLcEval assignment (left ++ right) =
      rawLcEval assignment left + rawLcEval assignment right := by
  induction left with
  | nil => simp [rawLcEval]
  | cons head tail inductionHypothesis =>
      simp [rawLcEval, inductionHypothesis, Nat.add_assoc]
private theorem rawLcEval_perm (assignment : Nat → Nat)
    {left right : List (Nat × Nat)} (permutation : left.Perm right) :
    rawLcEval assignment left = rawLcEval assignment right := by
  induction permutation with
  | nil => rfl
  | cons _ _ inductionHypothesis => simp [rawLcEval, inductionHypothesis]
  | swap _ _ _ => simp [rawLcEval]; omega
  | trans _ _ leftHypothesis rightHypothesis =>
      exact leftHypothesis.trans rightHypothesis
private theorem termsValue_perm (assignment : Nat → Nat)
    {left right : List (Nat × Nat)} (permutation : left.Perm right) :
    residue (lcEval assignment left) = residue (lcEval assignment right) := by
  apply Fin.ext
  simp only [residue]
  rw [lcEval_eq_raw_mod, lcEval_eq_raw_mod,
    rawLcEval_perm assignment permutation]
private theorem termsValue_append (assignment : Nat → Nat)
    (left right : List (Nat × Nat)) :
    residue (lcEval assignment (left ++ right)) =
      residue (lcEval assignment left) + residue (lcEval assignment right) := by
  apply Fin.ext
  simp only [residue, Fin.val_add]
  rw [lcEval_eq_raw_mod, lcEval_eq_raw_mod, lcEval_eq_raw_mod,
    rawLcEval_append]
  simp [Nat.add_mod]
private theorem karatsuba_cross (a0 a1 b0 b1 : F) :
    ((a0 + a1) * (b0 + b1) +
        residue (goldilocksP - 1) * (a0 * b0)) +
      residue (goldilocksP - 1) * (a1 * b1) =
        a0 * b1 + a1 * b0 := by
  let rawLeft := (a0.val + a1.val) * (b0.val + b1.val) +
    (goldilocksP - 1) * (a0.val * b0.val) +
    (goldilocksP - 1) * (a1.val * b1.val)
  let rawRight := a0.val * b1.val + a1.val * b0.val
  have rawEquality : rawLeft = rawRight + goldilocksP *
      ((a0.val * b0.val) + (a1.val * b1.val)) := by
    dsimp [rawLeft, rawRight]
    simp only [Nat.add_mul, Nat.mul_add]
    unfold goldilocksP
    omega
  apply Fin.ext
  simp only [Fin.val_add, Fin.val_mul, residue]
  have modularEquality : rawLeft % goldilocksP =
      rawRight % goldilocksP := by
    rw [rawEquality, Nat.add_mul_mod_self_left]
  dsimp [rawLeft, rawRight] at modularEquality
  simpa only [Nat.add_mod, Nat.mul_mod, Nat.mod_mod] using modularEquality
private theorem negOne_mul_add_self (value : F) :
    residue (goldilocksP - 1) * value + value = 0 := by
  apply Fin.ext
  simp only [Fin.val_add, Fin.val_mul, residue]
  have raw : (goldilocksP - 1) * value.val + value.val =
      goldilocksP * value.val := by
    unfold goldilocksP
    omega
  have modular : ((goldilocksP - 1) * value.val + value.val) %
      goldilocksP = 0 := by
    rw [raw, Nat.mul_mod_right]
  simpa only [Nat.add_mod, Nat.mul_mod, Nat.mod_mod,
    Nat.mod_eq_of_lt value.isLt] using modular
private theorem solve_two_negatives (left right0 right1 : F)
    (zero : (left + residue (goldilocksP - 1) * right0) +
      residue (goldilocksP - 1) * right1 = 0) :
    left = right0 + right1 := by
  have added := congrArg (fun value => (value + right0) + right1) zero
  dsimp at added
  have rearrange :
      ((((left + residue (goldilocksP - 1) * right0) +
          residue (goldilocksP - 1) * right1) + right0) + right1) =
        (left + (residue (goldilocksP - 1) * right0 + right0)) +
          (residue (goldilocksP - 1) * right1 + right1) := by
    ac_rfl
  rw [rearrange, negOne_mul_add_self, negOne_mul_add_self,
    Fin.add_zero, Fin.zero_add] at added
  simpa only [Fin.zero_add, Fin.add_zero] using added
private theorem productDefinition_value (assignment : Nat → Nat)
    (output : Nat) (left right : List (Nat × Nat))
    (holds : assignment output =
      lcEval assignment left * lcEval assignment right % goldilocksP) :
    baseAt assignment output =
      residue (lcEval assignment left) * residue (lcEval assignment right) := by
  apply Fin.ext
  simp only [baseAt, residue, Fin.val_mul]
  rw [holds]
  simp [Nat.mul_mod]
private theorem evalProducts_sound (assignment : Nat → Nat) :
    ∀ (coefficients : List Nat) (powers products : List KColumns),
      coefficients.length = powers.length →
      coefficients.length = products.length →
      DefinitionsHold assignment
        (EvalTrace.productDefinitionsFor coefficients powers products) →
      products.map (fun product => product.value assignment) =
        (List.zip (List.zip coefficients powers) products).map fun entry =>
          K.mul (K.ofBase (baseAt assignment entry.1.1))
            (entry.1.2.value assignment) := by
  intro coefficients
  induction coefficients with
  | nil =>
      intro powers products powersLength productsLength _
      cases powers with
      | cons _ _ => simp at powersLength
      | nil =>
          cases products with
          | cons _ _ => simp at productsLength
          | nil => rfl
  | cons coefficient coefficientTail inductionHypothesis =>
      intro powers products powersLength productsLength definitionsHold
      cases powers with
      | nil => simp at powersLength
      | cons power powerTail =>
          cases products with
          | nil => simp at productsLength
          | cons product productTail =>
              have tailPowersLength : coefficientTail.length = powerTail.length := by
                simpa using powersLength
              have tailProductsLength : coefficientTail.length = productTail.length := by
                simpa using productsLength
              have productC0Holds : assignment product.c0 =
                  lcEval assignment [(coefficient, 1)] *
                    lcEval assignment [(power.c0, 1)] % goldilocksP := by
                simpa [Definition.Holds, Rhs.eval] using
                  definitionsHold
                    ⟨product.c0,
                      .product [(coefficient, 1)] [(power.c0, 1)]⟩
                    (by simp [EvalTrace.productDefinitionsFor])
              have productC1Holds : assignment product.c1 =
                  lcEval assignment [(coefficient, 1)] *
                    lcEval assignment [(power.c1, 1)] % goldilocksP := by
                simpa [Definition.Holds, Rhs.eval] using
                  definitionsHold
                    ⟨product.c1,
                      .product [(coefficient, 1)] [(power.c1, 1)]⟩
                    (by simp [EvalTrace.productDefinitionsFor])
              have productC0Value := productDefinition_value assignment product.c0
                [(coefficient, 1)] [(power.c0, 1)] productC0Holds
              have productC1Value := productDefinition_value assignment product.c1
                [(coefficient, 1)] [(power.c1, 1)] productC1Holds
              have headValue : product.value assignment =
                  K.mul (K.ofBase (baseAt assignment coefficient))
                    (power.value assignment) := by
                rcases power with ⟨powerC0, powerC1⟩
                simp only [KColumns.value, K.ofBase, K.mul, K.mk.injEq,
                  Fin.zero_mul, Fin.mul_zero, Fin.add_zero]
                constructor
                · simpa [lcEval, residue, baseAt] using productC0Value
                · simpa [lcEval, residue, baseAt] using productC1Value
              have tailDefinitionsHold : DefinitionsHold assignment
                  (EvalTrace.productDefinitionsFor coefficientTail powerTail
                    productTail) := by
                intro definition member
                apply definitionsHold definition
                simp only [EvalTrace.productDefinitionsFor, List.zip_cons_cons,
                  List.flatMap_cons, List.mem_append]
                exact Or.inr member
              simp only [List.map_cons, List.zip_cons_cons]
              rw [headValue]
              congr 1
              exact inductionHypothesis powerTail productTail
                tailPowersLength tailProductsLength tailDefinitionsHold
private theorem expectedProducts_fold_eq_dot (assignment : Nat → Nat) :
    ∀ (coefficients : List Nat) (powers products : List KColumns),
      coefficients.length = powers.length →
      coefficients.length = products.length →
      ((List.zip (List.zip coefficients powers) products).map fun entry =>
        K.mul (K.ofBase (baseAt assignment entry.1.1))
          (entry.1.2.value assignment)).foldr K.add K.zero =
        Polynomial.dot assignment coefficients
          (powers.map fun power => power.value assignment) := by
  intro coefficients
  induction coefficients with
  | nil =>
      intro powers products powersLength productsLength
      cases powers with
      | cons _ _ => simp at powersLength
      | nil =>
          cases products with
          | cons _ _ => simp at productsLength
          | nil => rfl
  | cons coefficient coefficientTail inductionHypothesis =>
      intro powers products powersLength productsLength
      cases powers with
      | nil => simp at powersLength
      | cons power powerTail =>
          cases products with
          | nil => simp at productsLength
          | cons product productTail =>
              have tailPowersLength : coefficientTail.length = powerTail.length := by
                simpa using powersLength
              have tailProductsLength : coefficientTail.length = productTail.length := by
                simpa using productsLength
              simp only [List.zip_cons_cons, List.map_cons, List.foldr_cons,
                Polynomial.dot]
              rw [inductionHypothesis powerTail productTail
                tailPowersLength tailProductsLength]
              rfl
private theorem linearDefinition2_value (assignment : Nat → Nat)
    (output left right leftCoefficient rightCoefficient : Nat)
    (holds : assignment output = lcEval assignment
      [(left, leftCoefficient), (right, rightCoefficient)]) :
    baseAt assignment output =
      residue leftCoefficient * baseAt assignment left +
      residue rightCoefficient * baseAt assignment right := by
  apply Fin.ext
  simp [baseAt, residue, lcEval, Fin.val_add, Fin.val_mul, holds]
private theorem linearDefinition3_value (assignment : Nat → Nat)
    (output first second third firstCoefficient secondCoefficient
      thirdCoefficient : Nat)
    (holds : assignment output = lcEval assignment
      [(first, firstCoefficient), (second, secondCoefficient),
       (third, thirdCoefficient)]) :
    baseAt assignment output =
      (residue firstCoefficient * baseAt assignment first +
       residue secondCoefficient * baseAt assignment second) +
      residue thirdCoefficient * baseAt assignment third := by
  apply Fin.ext
  simp [baseAt, residue, lcEval, Fin.val_add, Fin.val_mul, holds]
private theorem termsValue_columns (assignment : Nat → Nat)
    (columns : List Nat) :
    residue (lcEval assignment (columns.map fun column => (column, 1))) =
      columns.foldr (fun column suffix =>
        baseAt assignment column + suffix) 0 := by
  induction columns with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      apply Fin.ext
      simp only [List.map_cons, List.foldr_cons, Fin.val_add, residue]
      rw [lcEval_eq_raw_mod]
      simp only [rawLcEval, Nat.one_mul, Nat.mod_mod]
      have valueHypothesis := congrArg Fin.val inductionHypothesis
      simp only [residue] at valueHypothesis
      rw [lcEval_eq_raw_mod] at valueHypothesis
      simp only [Nat.mod_mod] at valueHypothesis
      rw [← valueHypothesis]
      simp only [baseAt, residue]
      rw [← Nat.add_mod]
private theorem linearColumns_value (assignment : Nat → Nat)
    (output : Nat) (columns : List Nat)
    (holds : assignment output =
      lcEval assignment (columns.map fun column => (column, 1))) :
    baseAt assignment output =
      columns.foldr (fun column suffix =>
        baseAt assignment column + suffix) 0 := by
  calc
    baseAt assignment output =
        residue (lcEval assignment (columns.map fun column => (column, 1))) := by
      apply Fin.ext
      simp only [baseAt, residue]
      rw [holds]
    _ = _ := termsValue_columns assignment columns
private theorem projectionCheckLimb_sound (assignment : Nat → Nat)
    (constantOne : assignment 0 = 1) (outputs : List Nat)
    (quotientPhi output : Nat)
    (holds : RowHolds assignment
      ⟨outputs.map (fun column => (column, 1)) ++
          [(quotientPhi, goldilocksP - 1),
           (output, goldilocksP - 1)],
       [(0, 1)], []⟩) :
    outputs.foldr (fun column suffix =>
      baseAt assignment column + suffix) 0 =
      baseAt assignment quotientPhi + baseAt assignment output := by
  have linearZero : lcEval assignment
      (outputs.map (fun column => (column, 1)) ++
        [(quotientPhi, goldilocksP - 1),
         (output, goldilocksP - 1)]) = 0 := by
    simpa [RowHolds, lcEval, constantOne] using holds
  have split := termsValue_append assignment
    (outputs.map fun column => (column, 1))
    [(quotientPhi, goldilocksP - 1),
     (output, goldilocksP - 1)]
  rw [linearZero] at split
  have outputTerms := termsValue_columns assignment outputs
  have negativeTerms : residue (lcEval assignment
      [(quotientPhi, goldilocksP - 1),
       (output, goldilocksP - 1)]) =
      residue (goldilocksP - 1) * baseAt assignment quotientPhi +
      residue (goldilocksP - 1) * baseAt assignment output := by
    apply Fin.ext
    simp [baseAt, residue, lcEval, Fin.val_add, Fin.val_mul]
  rw [outputTerms, negativeTerms] at split
  apply solve_two_negatives
  rw [fadd_assoc]
  simpa only [residue_zero] using split.symm
private theorem foldKValues (values : List K) :
    values.foldr K.add K.zero =
      ⟨(values.map K.c0).foldr (· + ·) 0,
       (values.map K.c1).foldr (· + ·) 0⟩ := by
  induction values with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      simp only [List.foldr_cons, List.map_cons]
      rw [inductionHypothesis]
      rfl
theorem ProjectionTrace.checks_sound (trace : ProjectionTrace)
    (assignment : Nat → Nat) (constantOne : assignment 0 = 1)
    (checksHold : Satisfies trace.checks assignment) :
    (trace.pairProductValues assignment).foldr K.add K.zero =
      K.add (trace.quotientPhiProduct.output.value assignment)
        (trace.outputEvaluation.output.value assignment) := by
  let outputC0 := trace.pairs.map fun pair => pair.product.output.c0
  let outputC1 := trace.pairs.map fun pair => pair.product.output.c1
  have rowC0 : RowHolds assignment
      ⟨outputC0.map (fun column => (column, 1)) ++
          [(trace.quotientPhiProduct.output.c0, goldilocksP - 1),
           (trace.outputEvaluation.output.c0, goldilocksP - 1)],
       [(0, 1)], []⟩ := by
    apply checksHold
    simp [ProjectionTrace.checks, negatedColumns, outputC0]
  have rowC1 : RowHolds assignment
      ⟨outputC1.map (fun column => (column, 1)) ++
          [(trace.quotientPhiProduct.output.c1, goldilocksP - 1),
           (trace.outputEvaluation.output.c1, goldilocksP - 1)],
       [(0, 1)], []⟩ := by
    apply checksHold
    simp [ProjectionTrace.checks, negatedColumns, outputC1]
  have c0 := projectionCheckLimb_sound assignment constantOne outputC0
    trace.quotientPhiProduct.output.c0 trace.outputEvaluation.output.c0 rowC0
  have c1 := projectionCheckLimb_sound assignment constantOne outputC1
    trace.quotientPhiProduct.output.c1 trace.outputEvaluation.output.c1 rowC1
  dsimp [outputC0] at c0
  dsimp [outputC1] at c1
  rw [List.foldr_map] at c0 c1
  unfold ProjectionTrace.pairProductValues
  rw [foldKValues]
  simp only [KColumns.value, K.add, K.mk.injEq, List.map_map]
  constructor
  · simpa only [List.foldr_map, Function.comp_apply] using c0
  · simpa only [List.foldr_map, Function.comp_apply] using c1
theorem EvalTrace.products_sound (trace : EvalTrace)
    (assignment : Nat → Nat) (layout : trace.LayoutValid)
    (definitionsHold : DefinitionsHold assignment trace.definitions) :
    trace.products.map (fun product => product.value assignment) =
      trace.ExpectedProducts assignment := by
  rcases trace with ⟨coefficients, powers, products, output⟩
  cases coefficients with
  | nil => exact False.elim (layout.1 rfl)
  | cons coefficient coefficientTail =>
      cases powers with
      | nil => simp [EvalTrace.LayoutValid] at layout
      | cons power powerTail =>
          have powersLength : coefficientTail.length = powerTail.length := by
            simpa [EvalTrace.LayoutValid] using layout.2.1
          have productsLength : coefficientTail.length = products.length := by
            exact (Nat.add_right_cancel layout.2.2).symm
          have productDefinitionsHold : DefinitionsHold assignment
              (EvalTrace.productDefinitionsFor coefficientTail powerTail products) := by
            intro definition member
            apply definitionsHold definition
            apply List.mem_append_left
            simpa [EvalTrace.definitions, EvalTrace.productDefinitions] using member
          exact evalProducts_sound assignment coefficientTail powerTail products
            powersLength productsLength productDefinitionsHold
theorem EvalTrace.output_value (trace : EvalTrace)
    (assignment : Nat → Nat) (layout : trace.LayoutValid)
    (definitionsHold : DefinitionsHold assignment trace.definitions) :
    trace.output.value assignment =
      K.add (K.ofBase
        (baseAt assignment (trace.coefficients.head layout.1)))
        ((trace.products.map fun product => product.value assignment).foldr
          K.add K.zero) := by
  rcases trace with ⟨coefficients, powers, products, output⟩
  cases coefficients with
  | nil => exact False.elim (layout.1 rfl)
  | cons coefficient coefficientTail =>
      have outputC0Holds : assignment output.c0 = lcEval assignment
          ((coefficient :: products.map KColumns.c0).map fun column =>
            (column, 1)) := by
        simpa [Definition.Holds, Rhs.eval, EvalTrace.definitions] using
          definitionsHold
            ⟨output.c0,
              .linear ((coefficient, 1) ::
                (products.map fun product => (product.c0, 1)))⟩
            (by simp [EvalTrace.definitions])
      have outputC1Holds : assignment output.c1 = lcEval assignment
          ((products.map KColumns.c1).map fun column => (column, 1)) := by
        simpa [Definition.Holds, Rhs.eval, EvalTrace.definitions] using
          definitionsHold
            ⟨output.c1,
              .linear (products.map fun product => (product.c1, 1))⟩
            (by simp [EvalTrace.definitions])
      have outputC0Value := linearColumns_value assignment output.c0
        (coefficient :: products.map KColumns.c0) outputC0Holds
      have outputC1Value := linearColumns_value assignment output.c1
        (products.map KColumns.c1) outputC1Holds
      change output.value assignment =
        K.add (K.ofBase (baseAt assignment coefficient))
          ((products.map fun product => product.value assignment).foldr
            K.add K.zero)
      rw [foldKValues]
      simp only [KColumns.value, K.ofBase, K.add, K.mk.injEq,
        List.map_map]
      constructor
      · simpa only [List.foldr_cons, List.foldr_map,
          Function.comp_apply] using outputC0Value
      · simpa only [List.foldr_map, Function.comp_apply,
          Fin.zero_add] using outputC1Value

/-- Exact evaluation rows compute the Horner evaluation of every committed
coefficient at the supplied power ladder. -/
theorem EvalTrace.sound (trace : EvalTrace) (assignment : Nat → Nat)
    (point : K) (layout : trace.LayoutValid)
    (powersValid : trace.PowersValid assignment point)
    (definitionsHold : DefinitionsHold assignment trace.definitions) :
    trace.output.value assignment =
      Polynomial.eval (basePolynomial assignment trace.coefficients) point := by
  have productValues := trace.products_sound assignment layout definitionsHold
  have outputValue := trace.output_value assignment layout definitionsHold
  rcases trace with ⟨coefficients, powers, products, output⟩
  cases coefficients with
  | nil => exact False.elim (layout.1 rfl)
  | cons coefficient coefficientTail =>
      cases powers with
      | nil => simp [EvalTrace.LayoutValid] at layout
      | cons power powerTail =>
          have tailPowersLength : coefficientTail.length = powerTail.length := by
            simpa using layout.2.1
          have tailProductsLength : coefficientTail.length = products.length := by
            exact (Nat.add_right_cancel layout.2.2).symm
          have powerSequence := powersValid
          simp only [EvalTrace.PowersValid, List.map_cons, List.length_cons,
            K.powersFrom, K.one_mul, List.cons.injEq] at powerSequence
          have expectedFold := expectedProducts_fold_eq_dot assignment
            coefficientTail powerTail products tailPowersLength
            tailProductsLength
          rw [outputValue, productValues]
          change K.add (K.ofBase (baseAt assignment coefficient))
              (((List.zip (List.zip coefficientTail powerTail) products).map
                fun entry =>
                  K.mul (K.ofBase (baseAt assignment entry.1.1))
                    (entry.1.2.value assignment)).foldr K.add K.zero) =
            Polynomial.eval
              (basePolynomial assignment (coefficient :: coefficientTail)) point
          rw [expectedFold, powerSequence.2, Polynomial.dot_powersFrom]
          rfl
theorem EvalTrace.powersValid_of_ladderPrefix (trace : EvalTrace)
    (assignment : Nat → Nat) (point : K) (ladder : List KColumns)
    (prefixShape : trace.powers = ladder.take trace.coefficients.length)
    (within : trace.coefficients.length ≤ ladder.length)
    (ladderValues : ladder.map (fun power => power.value assignment) =
      K.powersFrom point K.one ladder.length) :
    trace.PowersValid assignment point := by
  unfold EvalTrace.PowersValid
  rw [prefixShape, List.map_take]
  calc
    (ladder.map fun power => power.value assignment).take
        trace.coefficients.length =
        (K.powersFrom point K.one ladder.length).take
          trace.coefficients.length := by rw [ladderValues]
    _ = K.powersFrom point K.one trace.coefficients.length :=
      K.take_powersFrom point K.one within
theorem EvalTrace.coefficientLength_le_ladder (trace : EvalTrace)
    (ladder : List KColumns) (layout : trace.LayoutValid)
    (prefixShape : trace.powers = ladder.take trace.coefficients.length) :
    trace.coefficients.length ≤ ladder.length := by
  have lengths := layout.2.1.trans (congrArg List.length prefixShape)
  rw [List.length_take] at lengths
  omega

/-- The five exact Karatsuba definitions determine extension-field
multiplication.  The only layout premise says that the two sum rows contain
the same LC terms as their components; sparse ordering may differ. -/
theorem KMulTrace.sound (trace : KMulTrace) (assignment : Nat → Nat)
    (layout : trace.SumLayoutValid)
    (definitionsHold : DefinitionsHold assignment trace.definitions) :
    trace.output.value assignment =
      K.mul (trace.left.value assignment) (trace.right.value assignment) := by
  have productC0Holds : assignment trace.productC0 =
      lcEval assignment trace.left.c0 * lcEval assignment trace.right.c0 %
        goldilocksP := by
    simpa [Definition.Holds, Rhs.eval] using
      definitionsHold
        ⟨trace.productC0, .product trace.left.c0 trace.right.c0⟩
        (by simp [KMulTrace.definitions])
  have productC1Holds : assignment trace.productC1 =
      lcEval assignment trace.left.c1 * lcEval assignment trace.right.c1 %
        goldilocksP := by
    simpa [Definition.Holds, Rhs.eval] using
      definitionsHold
        ⟨trace.productC1, .product trace.left.c1 trace.right.c1⟩
        (by simp [KMulTrace.definitions])
  have productSumHolds : assignment trace.productSum =
      lcEval assignment trace.sumLeft * lcEval assignment trace.sumRight %
        goldilocksP := by
    simpa [Definition.Holds, Rhs.eval] using
      definitionsHold
        ⟨trace.productSum, .product trace.sumLeft trace.sumRight⟩
        (by simp [KMulTrace.definitions])
  have outputC0Holds : assignment trace.output.c0 = lcEval assignment
      [(trace.productC0, 1), (trace.productC1, 7)] := by
    simpa [Definition.Holds, Rhs.eval] using
      definitionsHold
        ⟨trace.output.c0,
          .linear [(trace.productC0, 1), (trace.productC1, 7)]⟩
        (by simp [KMulTrace.definitions])
  have outputC1Holds : assignment trace.output.c1 = lcEval assignment
      [(trace.productSum, 1),
       (trace.productC0, goldilocksP - 1),
       (trace.productC1, goldilocksP - 1)] := by
    simpa [Definition.Holds, Rhs.eval] using
      definitionsHold
        ⟨trace.output.c1,
          .linear [(trace.productSum, 1),
            (trace.productC0, goldilocksP - 1),
            (trace.productC1, goldilocksP - 1)]⟩
        (by simp [KMulTrace.definitions])
  have productC0Value := productDefinition_value assignment
    trace.productC0 trace.left.c0 trace.right.c0 productC0Holds
  have productC1Value := productDefinition_value assignment
    trace.productC1 trace.left.c1 trace.right.c1 productC1Holds
  have productSumValue := productDefinition_value assignment
    trace.productSum trace.sumLeft trace.sumRight productSumHolds
  have sumLeftValue : residue (lcEval assignment trace.sumLeft) =
      residue (lcEval assignment trace.left.c0) +
      residue (lcEval assignment trace.left.c1) := by
    rw [termsValue_perm assignment layout.1, termsValue_append]
  have sumRightValue : residue (lcEval assignment trace.sumRight) =
      residue (lcEval assignment trace.right.c0) +
      residue (lcEval assignment trace.right.c1) := by
    rw [termsValue_perm assignment layout.2, termsValue_append]
  have outputC0Value := linearDefinition2_value assignment
    trace.output.c0 trace.productC0 trace.productC1 1 7 outputC0Holds
  have outputC1Value := linearDefinition3_value assignment
    trace.output.c1 trace.productSum trace.productC0 trace.productC1
    1 (goldilocksP - 1) (goldilocksP - 1) outputC1Holds
  simp only [KColumns.value, KTerms.value, K.mul, K.mk.injEq]
  constructor
  · rw [productC0Value, productC1Value] at outputC0Value
    simpa only [residue_one, residue_seven, Fin.one_mul] using outputC0Value
  · rw [productSumValue, productC0Value, productC1Value,
      sumLeftValue, sumRightValue] at outputC1Value
    simp only [residue_one, Fin.one_mul] at outputC1Value
    rw [karatsuba_cross] at outputC1Value
    exact outputC1Value
theorem PairTrace.sound (trace : PairTrace) (assignment : Nat → Nat)
    (point : K) (ladder : List KColumns)
    (ladderValues : ladder.map (fun power => power.value assignment) =
      K.powersFrom point K.one ladder.length)
    (layout : trace.LayoutValid ladder)
    (definitionsHold : DefinitionsHold assignment trace.definitions) :
    trace.product.output.value assignment =
      K.mul
        (Polynomial.eval (basePolynomial assignment trace.rhoColumns) point)
        (Polynomial.eval (basePolynomial assignment trace.inputColumns) point) := by
  rcases layout with
    ⟨rhoLayout, inputLayout, rhoCoefficients, inputCoefficients,
     rhoPrefix, inputPrefix, productLeft, productRight, productLayout⟩
  have rhoDefinitionsHold : DefinitionsHold assignment
      trace.rhoEvaluation.definitions := by
    intro definition member
    apply definitionsHold definition
    simp [PairTrace.definitions, member]
  have inputDefinitionsHold : DefinitionsHold assignment
      trace.inputEvaluation.definitions := by
    intro definition member
    apply definitionsHold definition
    simp [PairTrace.definitions, member]
  have productDefinitionsHold : DefinitionsHold assignment
      trace.product.definitions := by
    intro definition member
    apply definitionsHold definition
    simp [PairTrace.definitions, member]
  have rhoPrefix' :
      trace.rhoEvaluation.powers =
        ladder.take trace.rhoEvaluation.coefficients.length := by
    rw [rhoCoefficients]
    exact rhoPrefix
  have inputPrefix' :
      trace.inputEvaluation.powers =
        ladder.take trace.inputEvaluation.coefficients.length := by
    rw [inputCoefficients]
    exact inputPrefix
  have rhoWithin := trace.rhoEvaluation.coefficientLength_le_ladder
    ladder rhoLayout rhoPrefix'
  have inputWithin := trace.inputEvaluation.coefficientLength_le_ladder
    ladder inputLayout inputPrefix'
  have rhoPowers := trace.rhoEvaluation.powersValid_of_ladderPrefix
    assignment point ladder rhoPrefix' rhoWithin ladderValues
  have inputPowers := trace.inputEvaluation.powersValid_of_ladderPrefix
    assignment point ladder inputPrefix' inputWithin ladderValues
  have rhoValue := trace.rhoEvaluation.sound assignment point rhoLayout
    rhoPowers rhoDefinitionsHold
  have inputValue := trace.inputEvaluation.sound assignment point inputLayout
    inputPowers inputDefinitionsHold
  have productValue := trace.product.sound assignment productLayout
    productDefinitionsHold
  rw [productLeft, productRight, KTerms.ofColumns_value,
    KTerms.ofColumns_value, rhoValue, inputValue,
    rhoCoefficients, inputCoefficients] at productValue
  exact productValue
private theorem ladderLinked_values (assignment : Nat → Nat)
    (beta : KColumns) :
    ∀ (current : KColumns) (rest : List KColumns)
      (multiplications : List KMulTrace) (expected : K),
      LadderLinked beta (current :: rest) multiplications →
      current.value assignment = expected →
      DefinitionsHold assignment
        (multiplications.flatMap KMulTrace.definitions) →
      (current :: rest).map (fun power => power.value assignment) =
        K.powersFrom (beta.value assignment) expected
          (current :: rest).length := by
  intro current rest
  induction rest generalizing current with
  | nil =>
      intro multiplications expected linked currentValue _
      cases multiplications with
      | nil =>
          simp only [List.map_cons, List.map_nil, List.length_cons,
            List.length_nil, K.powersFrom]
          exact congrArg (fun value => [value]) currentValue
      | cons multiplication multiplications =>
          simp [LadderLinked] at linked
  | cons next rest inductionHypothesis =>
      intro multiplications expected linked currentValue definitionsHold
      cases multiplications with
      | nil => simp [LadderLinked] at linked
      | cons multiplication multiplications =>
          simp only [LadderLinked] at linked
          rcases linked with
            ⟨leftShape, rightShape, outputShape, sumLayout, tailLinked⟩
          have multiplicationDefinitionsHold : DefinitionsHold assignment
              multiplication.definitions := by
            intro definition member
            apply definitionsHold definition
            simp [member]
          have multiplicationValue := multiplication.sound assignment sumLayout
            multiplicationDefinitionsHold
          rw [leftShape, rightShape, outputShape,
            KTerms.ofColumns_value, KTerms.ofColumns_value] at multiplicationValue
          have nextValue : next.value assignment =
              K.mul expected (beta.value assignment) := by
            rw [← currentValue]
            exact multiplicationValue
          have tailDefinitionsHold : DefinitionsHold assignment
              (multiplications.flatMap KMulTrace.definitions) := by
            intro definition member
            apply definitionsHold definition
            simp [member]
          have tailValues := inductionHypothesis next multiplications
            (K.mul expected (beta.value assignment)) tailLinked nextValue
            tailDefinitionsHold
          simp only [List.map_cons, List.length_cons, K.powersFrom]
          rw [currentValue]
          exact congrArg (List.cons expected) tailValues

/-- The exact base rows and linked K-multiplication blocks force the shared
ladder to be `1, beta, ..., beta^D`. -/
theorem LadderTrace.sound (trace : LadderTrace) (assignment : Nat → Nat)
    (constantOne : assignment 0 = 1) (layout : trace.LayoutValid)
    (definitionsHold : DefinitionsHold assignment trace.definitions) :
    trace.powers.map (fun power => power.value assignment) =
      K.powersFrom (trace.beta.value assignment) K.one trace.powers.length := by
  rcases trace with ⟨beta, powers, multiplications⟩
  cases powers with
  | nil => simp [LadderTrace.LayoutValid, LadderLinked] at layout
  | cons base rest =>
      have baseC0Holds : assignment base.c0 = lcEval assignment [(0, 1)] := by
        simpa [Definition.Holds, Rhs.eval] using
          definitionsHold ⟨base.c0, .linear [(0, 1)]⟩
            (by simp [LadderTrace.definitions])
      have baseC1Holds : assignment base.c1 = lcEval assignment [] := by
        simpa [Definition.Holds, Rhs.eval] using
          definitionsHold ⟨base.c1, .linear []⟩
            (by simp [LadderTrace.definitions])
      have baseValue : base.value assignment = K.one := by
        simp only [KColumns.value, K.one, K.mk.injEq]
        constructor
        · apply Fin.ext
          simp [baseAt, residue, lcEval, baseC0Holds, constantOne]
          decide
        · apply Fin.ext
          simp [baseAt, residue, lcEval, baseC1Holds]
      have multiplicationDefinitionsHold : DefinitionsHold assignment
          (multiplications.flatMap KMulTrace.definitions) := by
        intro definition member
        apply definitionsHold definition
        simp [LadderTrace.definitions, member]
      exact ladderLinked_values assignment beta base rest multiplications K.one
        layout baseValue multiplicationDefinitionsHold
private theorem ladderPower27 (assignment : Nat → Nat)
    (powers : List KColumns) (length : powers.length = 55) (point : K)
    (values : powers.map (fun power => power.value assignment) =
      K.powersFrom point K.one 55) :
    (powers.getD 27 default).value assignment = K.pow point 27 := by
  have bound : 27 < powers.length := by omega
  have selected := congrArg (fun values : List K => values[27]?) values
  dsimp at selected
  rw [List.getElem?_map, List.getElem?_eq_getElem bound] at selected
  have expected : (K.powersFrom point K.one 55)[27]? =
      some (K.pow point 27) := by
    simp [K.powersFrom, K.pow, K.one_mul]
  rw [expected] at selected
  have selectedValue : powers[27].value assignment = K.pow point 27 :=
    Option.some.inj selected
  rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem bound]
  exact selectedValue
private theorem ladderPower54 (assignment : Nat → Nat)
    (powers : List KColumns) (length : powers.length = 55) (point : K)
    (values : powers.map (fun power => power.value assignment) =
      K.powersFrom point K.one 55) :
    (powers.getD 54 default).value assignment = K.pow point 54 := by
  have bound : 54 < powers.length := by omega
  have selected := congrArg (fun values : List K => values[54]?) values
  dsimp at selected
  rw [List.getElem?_map, List.getElem?_eq_getElem bound] at selected
  have expected : (K.powersFrom point K.one 55)[54]? =
      some (K.pow point 54) := by
    simp [K.powersFrom, K.pow, K.one_mul]
  rw [expected] at selected
  have selectedValue : powers[54].value assignment = K.pow point 54 :=
    Option.some.inj selected
  rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem bound]
  exact selectedValue
theorem phiTerms_value (assignment : Nat → Nat) (constantOne : assignment 0 = 1)
    (powers : List KColumns) (length : powers.length = 55) (point : K)
    (values : powers.map (fun power => power.value assignment) =
      K.powersFrom point K.one 55) :
    (phiTerms powers).value assignment = Polynomial.eval Polynomial.phi81 point := by
  let power54 := powers.getD 54 default
  let power27 := powers.getD 27 default
  have termsValue : (phiTerms powers).value assignment =
      K.add (power54.value assignment)
        (K.add (power27.value assignment) K.one) := by
    change (⟨[(power54.c0, 1), (power27.c0, 1), (0, 1)],
      [(power54.c1, 1), (power27.c1, 1)]⟩ : KTerms).value assignment = _
    simp only [KTerms.value, KColumns.value, K.add, K.one, K.mk.injEq]
    constructor
    · have coefficientValues := termsValue_columns assignment
        [power54.c0, power27.c0, 0]
      simp only [List.map_cons, List.map_nil, List.foldr_cons,
        List.foldr_nil, Fin.add_zero] at coefficientValues
      rw [coefficientValues]
      have oneValue : baseAt assignment 0 = (1 : F) := by
        apply Fin.ext
        simp [baseAt, residue, constantOne]
        decide
      rw [oneValue]
    · have coefficientValues := termsValue_columns assignment
        [power54.c1, power27.c1]
      simpa only [List.map_cons, List.map_nil, List.foldr_cons,
        List.foldr_nil, Fin.add_zero] using coefficientValues
  rw [termsValue, ladderPower54 assignment powers length point values,
    ladderPower27 assignment powers length point values,
    Polynomial.phi81_eval]
  ac_rfl

end Nightstream.Implementation.R1CS.ProjectionProgram
