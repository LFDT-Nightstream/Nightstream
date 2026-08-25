import NightstreamFPrime.Circuit.Quadratic
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.SumCheckTruthPath
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier.Algebra

/-!
Paper authority: SuperNeo v1.1, Section 7.3, Step 4, `eq(r', r)`;
`SumCheckTruthPath.equalityFactor_eq_affine`.
Obligation: Compute the multilinear point-equality polynomial in canonical
coordinate order.

Inputs:
- two dimension-checked symbolic extension-field points;
- one external expected-output wire.

Outputs:
- the expected output, constrained to
  `product_i ((1-r_i) + r'_i * (r_i-(1-r_i)))`.

Constraint groups:
- C1: materialize one affine equality factor per coordinate;
- C2: multiply the factors through one causal witness batch;
- C3: two output-component equalities.

The empty product is one. No challenge is a witness output.
-/

namespace NightstreamFPrime.Gadgets.Multilinear.PointEquality

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier

structure CoordinateExpr where
  left : KExpr
  right : KExpr

/-- The proved affine form of one paper equality factor. -/
def factorExpr (coordinate : CoordinateExpr) : KExpr :=
  let oneMinusRight := KExpr.sub KExpr.one coordinate.right
  KExpr.add oneMinusRight
    (KExpr.mul coordinate.left
      (KExpr.sub coordinate.right oneMinusRight))

theorem factorExpr_eval (env : Env) (coordinate : CoordinateExpr) :
    (factorExpr coordinate).eval env =
      SumCheckTruthPath.equalityFactor extensionOps
        (coordinate.left.eval env) (coordinate.right.eval env) := by
  rw [SumCheckTruthPath.equalityFactor_eq_affine extensionLaws]
  simp only [factorExpr, KExpr.eval_add, KExpr.eval_mul, KExpr.eval_sub]
  simp only [derived_sub_eq_concrete_sub]
  rfl

def factorRecipes (coordinate : CoordinateExpr) : List Expr :=
  [(factorExpr coordinate).c0, (factorExpr coordinate).c1]

def mulRecipes (left right : KExpr) : List Expr :=
  [(KExpr.mul left right).c0, (KExpr.mul left right).c1]

def materializedAt (start : Nat) : KExpr :=
  ⟨Expr.var start, Expr.var (start + 1)⟩

@[simp] theorem factorRecipes_length (coordinate : CoordinateExpr) :
    (factorRecipes coordinate).length = 2 := by
  rfl

@[simp] theorem mulRecipes_length (left right : KExpr) :
    (mulRecipes left right).length = 2 := by
  rfl

structure Program where
  recipes : List Expr
  output : KExpr

/-- Compile from the tail so every product reads only materialized factors
and a completed tail product. -/
def compile (start : Nat) : List CoordinateExpr → Program
  | [] => ⟨[], KExpr.one⟩
  | [coordinate] =>
      ⟨factorRecipes coordinate, materializedAt start⟩
  | coordinate :: next :: rest =>
      let tail := compile start (next :: rest)
      let factorStart := start + tail.recipes.length
      let factor := materializedAt factorStart
      let productStart := factorStart + 2
      ⟨tail.recipes ++ factorRecipes coordinate ++
          mulRecipes factor tail.output,
        materializedAt productStart⟩

/-- Semantic product over the same ordered coordinate list. -/
def evaluate (env : Env) : List CoordinateExpr → K
  | [] => K.one
  | coordinate :: rest =>
      K.mul
        (SumCheckTruthPath.equalityFactor extensionOps
          (coordinate.left.eval env) (coordinate.right.eval env))
        (evaluate env rest)

theorem evaluate_eq_pointEqualityCoordinates (env : Env) :
    ∀ coordinates : List CoordinateExpr,
    evaluate env coordinates =
      SumCheckTruthPath.pointEqualityCoordinates extensionOps
        (coordinates.map fun coordinate => coordinate.left.eval env)
        (coordinates.map fun coordinate => coordinate.right.eval env)
  | [] => rfl
  | coordinate :: rest => by
      simp only [evaluate, List.map_cons,
        SumCheckTruthPath.pointEqualityCoordinates]
      rw [evaluate_eq_pointEqualityCoordinates env rest]
      rfl

private theorem add_varsBelow (left right : KExpr) (bound : Nat)
    (leftBelow : left.VarsBelow bound)
    (rightBelow : right.VarsBelow bound) :
    (KExpr.add left right).VarsBelow bound :=
  ⟨⟨leftBelow.1, rightBelow.1⟩, ⟨leftBelow.2, rightBelow.2⟩⟩

private theorem sub_varsBelow (left right : KExpr) (bound : Nat)
    (leftBelow : left.VarsBelow bound)
    (rightBelow : right.VarsBelow bound) :
    (KExpr.sub left right).VarsBelow bound := by
  change
    (left.c0.VarsBelow bound ∧
      ((Expr.const (-1)).VarsBelow bound ∧ right.c0.VarsBelow bound)) ∧
    (left.c1.VarsBelow bound ∧
      ((Expr.const (-1)).VarsBelow bound ∧ right.c1.VarsBelow bound))
  exact ⟨⟨leftBelow.1, ⟨trivial, rightBelow.1⟩⟩,
    ⟨leftBelow.2, ⟨trivial, rightBelow.2⟩⟩⟩

private theorem mul_varsBelow (left right : KExpr) (bound : Nat)
    (leftBelow : left.VarsBelow bound)
    (rightBelow : right.VarsBelow bound) :
    (KExpr.mul left right).VarsBelow bound := by
  unfold KExpr.mul KExpr.VarsBelow Expr.VarsBelow
  exact ⟨
    ⟨⟨leftBelow.1, rightBelow.1⟩,
      ⟨⟨trivial, leftBelow.2⟩, rightBelow.2⟩⟩,
    ⟨⟨leftBelow.1, rightBelow.2⟩,
      ⟨leftBelow.2, rightBelow.1⟩⟩⟩

private theorem factorExpr_varsBelow (coordinate : CoordinateExpr)
    (bound : Nat) (leftBelow : coordinate.left.VarsBelow bound)
    (rightBelow : coordinate.right.VarsBelow bound) :
    (factorExpr coordinate).VarsBelow bound := by
  let oneMinusRight := KExpr.sub KExpr.one coordinate.right
  have oneBelow : KExpr.one.VarsBelow bound := by
    simp [KExpr.one, KExpr.VarsBelow, Expr.VarsBelow]
  have oneMinusBelow := sub_varsBelow KExpr.one coordinate.right bound
    oneBelow rightBelow
  exact add_varsBelow oneMinusRight
    (KExpr.mul coordinate.left
      (KExpr.sub coordinate.right oneMinusRight)) bound
    oneMinusBelow
    (mul_varsBelow coordinate.left
      (KExpr.sub coordinate.right oneMinusRight) bound leftBelow
      (sub_varsBelow coordinate.right oneMinusRight bound rightBelow
        oneMinusBelow))

private theorem factorRecipes_below (coordinate : CoordinateExpr)
    (bound : Nat) (leftBelow : coordinate.left.VarsBelow bound)
    (rightBelow : coordinate.right.VarsBelow bound) :
    ∀ expression ∈ factorRecipes coordinate,
      expression.VarsBelow bound := by
  intro expression member
  simp only [factorRecipes, List.mem_cons, List.not_mem_nil,
    or_false] at member
  rcases member with rfl | rfl
  · exact (factorExpr_varsBelow coordinate bound leftBelow rightBelow).1
  · exact (factorExpr_varsBelow coordinate bound leftBelow rightBelow).2

private theorem mulRecipes_below (left right : KExpr) (bound : Nat)
    (leftBelow : left.VarsBelow bound)
    (rightBelow : right.VarsBelow bound) :
    ∀ expression ∈ mulRecipes left right,
      expression.VarsBelow bound := by
  intro expression member
  simp only [mulRecipes, List.mem_cons, List.not_mem_nil,
    or_false] at member
  rcases member with rfl | rfl
  · exact (mul_varsBelow left right bound leftBelow rightBelow).1
  · exact (mul_varsBelow left right bound leftBelow rightBelow).2

private theorem materializedAt_varsBelow (start : Nat) :
    (materializedAt start).VarsBelow (start + 2) := by
  unfold materializedAt KExpr.VarsBelow Expr.VarsBelow
  omega

theorem compile_recipes_length (start : Nat) :
    ∀ coordinates : List CoordinateExpr,
    (compile start coordinates).recipes.length =
      match coordinates with
      | [] => 0
      | _ => 4 * coordinates.length - 2
  | [] => rfl
  | [coordinate] => rfl
  | coordinate :: next :: rest => by
      simp only [compile, List.length_append, factorRecipes_length,
        mulRecipes_length, List.length_cons]
      rw [compile_recipes_length start (next :: rest)]
      simp only [List.length_cons]
      omega

/-- Causality and output scope are structural in the coordinate list. -/
theorem compile_causal_and_output_below
    (start : Nat) (coordinates : List CoordinateExpr)
    (coordinatesBelow : ∀ coordinate ∈ coordinates,
      coordinate.left.VarsBelow start ∧ coordinate.right.VarsBelow start) :
    RecipesCausal start (compile start coordinates).recipes ∧
      (compile start coordinates).output.VarsBelow
        (start + (compile start coordinates).recipes.length) := by
  induction coordinates with
  | nil =>
      exact ⟨trivial, by simp [compile, KExpr.one, KExpr.VarsBelow,
        Expr.VarsBelow]⟩
  | cons coordinate rest inductionHypothesis =>
      cases rest with
      | nil =>
          have currentBelow := coordinatesBelow coordinate (by simp)
          refine ⟨?_, ?_⟩
          · exact recipesCausal_of_all_below start
              (factorRecipes coordinate)
              (factorRecipes_below coordinate start currentBelow.1
                currentBelow.2)
          · simpa [compile] using materializedAt_varsBelow start
      | cons next rest =>
          let tail := compile start (next :: rest)
          let factorStart := start + tail.recipes.length
          let factor := materializedAt factorStart
          let productStart := factorStart + 2
          have tailBelow : ∀ current ∈ next :: rest,
              current.left.VarsBelow start ∧ current.right.VarsBelow start := by
            intro current member
            exact coordinatesBelow current (by simp [member])
          have tailProof := inductionHypothesis tailBelow
          have currentBelow := coordinatesBelow coordinate (by simp)
          have currentAtFactor : coordinate.left.VarsBelow factorStart ∧
              coordinate.right.VarsBelow factorStart :=
            ⟨coordinate.left.varsBelow_mono currentBelow.1 (by
                unfold factorStart
                omega),
              coordinate.right.varsBelow_mono currentBelow.2 (by
                unfold factorStart
                omega)⟩
          have causalWithFactor : RecipesCausal start
              (tail.recipes ++ factorRecipes coordinate) :=
            recipesCausal_append start tail.recipes
              (factorRecipes coordinate) tailProof.1
              (factorRecipes_below coordinate factorStart
                currentAtFactor.1 currentAtFactor.2)
          have factorBelow : factor.VarsBelow productStart := by
            simpa [factor, productStart] using
              materializedAt_varsBelow factorStart
          have tailAtFactor : tail.output.VarsBelow factorStart := by
            simpa [tail, factorStart] using tailProof.2
          have tailAtProduct : tail.output.VarsBelow productStart :=
            tail.output.varsBelow_mono tailAtFactor (by
              unfold productStart
              omega)
          have productAddedBelow : ∀ expression ∈
              mulRecipes factor tail.output,
              expression.VarsBelow productStart :=
            mulRecipes_below factor tail.output productStart factorBelow
              tailAtProduct
          have causal : RecipesCausal start
              (tail.recipes ++ factorRecipes coordinate ++
                mulRecipes factor tail.output) := by
            apply recipesCausal_append start
              (tail.recipes ++ factorRecipes coordinate)
              (mulRecipes factor tail.output) causalWithFactor
            intro expression member
            have below := productAddedBelow expression member
            simpa [productStart, factorStart] using below
          refine ⟨?_, ?_⟩
          · simpa [compile, tail, factorStart, factor, productStart] using causal
          · have outputBelow := materializedAt_varsBelow productStart
            simpa [compile, tail, factorStart, factor, productStart] using
              outputBelow

private theorem materializedFactor_sound (env : Env) (start : Nat)
    (coordinate : CoordinateExpr)
    (rows : ConstraintsHold env
      (recipeConstraints start (factorRecipes coordinate))) :
    (materializedAt start).eval env =
      SumCheckTruthPath.equalityFactor extensionOps
        (coordinate.left.eval env) (coordinate.right.eval env) := by
  have equality : (materializedAt start).eval env =
      (factorExpr coordinate).eval env := by
    apply (KExpr.equalities_hold_iff env (materializedAt start)
      (factorExpr coordinate)).mp
    simpa [materializedAt, factorRecipes, KExpr.equalities,
      recipeConstraints, Nat.add_assoc] using rows
  exact equality.trans (factorExpr_eval env coordinate)

private theorem materializedProduct_sound (env : Env) (start : Nat)
    (left right : KExpr)
    (rows : ConstraintsHold env
      (recipeConstraints start (mulRecipes left right))) :
    (materializedAt start).eval env =
      K.mul (left.eval env) (right.eval env) := by
  have equality : (materializedAt start).eval env =
      (KExpr.mul left right).eval env := by
    apply (KExpr.equalities_hold_iff env (materializedAt start)
      (KExpr.mul left right)).mp
    simpa [materializedAt, mulRecipes, KExpr.equalities,
      recipeConstraints, Nat.add_assoc] using rows
  exact equality.trans (KExpr.eval_mul env left right)

theorem compile_output_sound (env : Env) (start : Nat) :
    ∀ coordinates : List CoordinateExpr,
    ConstraintsHold env
      (recipeConstraints start (compile start coordinates).recipes) →
    (compile start coordinates).output.eval env = evaluate env coordinates
  | [], _ => rfl
  | [coordinate], rows => by
      calc
        (compile start [coordinate]).output.eval env =
            SumCheckTruthPath.equalityFactor extensionOps
              (coordinate.left.eval env) (coordinate.right.eval env) :=
          materializedFactor_sound env start coordinate rows
        _ = evaluate env [coordinate] := by
          change _ = K.mul _ K.one
          exact (extensionLaws.mul_one _).symm
  | coordinate :: next :: rest, rows => by
      let tail := compile start (next :: rest)
      let factorStart := start + tail.recipes.length
      let factor := materializedAt factorStart
      let productStart := factorStart + 2
      have productStart_eq : productStart =
          start + (tail.recipes ++ factorRecipes coordinate).length := by
        simp [productStart, factorStart, Nat.add_assoc]
      have splitProduct :
          ConstraintsHold env (recipeConstraints start
            (tail.recipes ++ factorRecipes coordinate)) ∧
          ConstraintsHold env (recipeConstraints productStart
            (mulRecipes factor tail.output)) := by
        apply (constraintsHold_append env _ _).mp
        rw [productStart_eq]
        rw [← recipeConstraints_append]
        simpa [compile, tail, factorStart, factor, productStart] using rows
      have factorStart_eq : factorStart = start + tail.recipes.length := rfl
      have splitPrefix :
          ConstraintsHold env (recipeConstraints start tail.recipes) ∧
          ConstraintsHold env (recipeConstraints factorStart
            (factorRecipes coordinate)) := by
        apply (constraintsHold_append env _ _).mp
        rw [factorStart_eq]
        rw [← recipeConstraints_append]
        simpa [factorStart] using splitProduct.1
      have tailSound := compile_output_sound env start (next :: rest)
        splitPrefix.1
      have factorSound := materializedFactor_sound env factorStart coordinate
        splitPrefix.2
      have productSound := materializedProduct_sound env productStart factor
        tail.output splitProduct.2
      simp only [compile, evaluate]
      rw [productSound, factorSound, tailSound]
      rfl

structure Interface (variableCount : Nat) where
  left : Nat → Fin variableCount → KExpr
  right : Nat → Fin variableCount → KExpr
  expected : Nat → KExpr

def coordinateExprs {variableCount : Nat}
    (interface : Interface variableCount) (offset : Nat) : List CoordinateExpr :=
  (canonicalFinIndices variableCount).map fun coordinate =>
    ⟨interface.left offset coordinate, interface.right offset coordinate⟩

def evalLeftPoint {variableCount : Nat} (interface : Interface variableCount)
    (offset : Nat) (env : Env) : CubePoint K variableCount where
  coordinates := (canonicalFinIndices variableCount).map fun coordinate =>
    (interface.left offset coordinate).eval env
  dimension := by rw [List.length_map, canonicalFinIndices_length]

def evalRightPoint {variableCount : Nat} (interface : Interface variableCount)
    (offset : Nat) (env : Env) : CubePoint K variableCount where
  coordinates := (canonicalFinIndices variableCount).map fun coordinate =>
    (interface.right offset coordinate).eval env
  dimension := by rw [List.length_map, canonicalFinIndices_length]

def Assumptions {variableCount : Nat} (interface : Interface variableCount)
    (offset : Nat) (_env : Env) : Prop :=
  (∀ coordinate,
    (interface.left offset coordinate).VarsBelow offset ∧
      (interface.right offset coordinate).VarsBelow offset) ∧
    (interface.expected offset).VarsBelow offset

def SpecHolds {variableCount : Nat} (interface : Interface variableCount)
    (offset : Nat) (env : Env) : Prop :=
  (interface.expected offset).eval env =
    SumCheckTruthPath.pointEquality extensionOps
      (evalLeftPoint interface offset env) (evalRightPoint interface offset env)

private theorem cubePoint_eq_of_coordinates
    {Field : Type} {variableCount : Nat}
    (left right : CubePoint Field variableCount)
    (coordinates : left.coordinates = right.coordinates) : left = right := by
  cases left
  cases right
  simp_all

theorem specHolds_iff_of_fields_eq {variableCount : Nat}
    (leftInterface rightInterface : Interface variableCount)
    (leftOffset rightOffset : Nat) (env : Env)
    (leftEq : leftInterface.left leftOffset =
      rightInterface.left rightOffset)
    (rightEq : leftInterface.right leftOffset =
      rightInterface.right rightOffset)
    (expectedEq : leftInterface.expected leftOffset =
      rightInterface.expected rightOffset) :
    SpecHolds leftInterface leftOffset env ↔
      SpecHolds rightInterface rightOffset env := by
  have leftPointEq : evalLeftPoint leftInterface leftOffset env =
      evalLeftPoint rightInterface rightOffset env := by
    apply cubePoint_eq_of_coordinates
    unfold evalLeftPoint
    apply List.map_congr_left
    intro coordinate _
    exact congrArg (KExpr.eval env) (congrFun leftEq coordinate)
  have rightPointEq : evalRightPoint leftInterface leftOffset env =
      evalRightPoint rightInterface rightOffset env := by
    apply cubePoint_eq_of_coordinates
    unfold evalRightPoint
    apply List.map_congr_left
    intro coordinate _
    exact congrArg (KExpr.eval env) (congrFun rightEq coordinate)
  have expectedEvalEq : (leftInterface.expected leftOffset).eval env =
      (rightInterface.expected rightOffset).eval env :=
    congrArg (KExpr.eval env) expectedEq
  unfold SpecHolds
  rw [expectedEvalEq, leftPointEq, rightPointEq]

theorem specHolds_at_iff_of_fields_eq {variableCount : Nat}
    (interface : Interface variableCount) (leftOffset rightOffset : Nat)
    (env : Env)
    (leftEq : interface.left leftOffset = interface.left rightOffset)
    (rightEq : interface.right leftOffset = interface.right rightOffset)
    (expectedEq : interface.expected leftOffset =
      interface.expected rightOffset) :
    SpecHolds interface leftOffset env ↔
      SpecHolds interface rightOffset env :=
  specHolds_iff_of_fields_eq interface interface leftOffset rightOffset env
    leftEq rightEq expectedEq

def program {variableCount : Nat} (interface : Interface variableCount)
    (offset : Nat) : Program :=
  compile offset (coordinateExprs interface offset)

def allAssertions {variableCount : Nat} (interface : Interface variableCount)
    (offset : Nat) : List Expr :=
  KExpr.equalities (program interface offset).output
    (interface.expected offset)

def opsAt {variableCount : Nat} (interface : Interface variableCount)
    (offset : Nat) : List Op :=
  [Op.witness (WitnessBatch.arithmetic offset
    (program interface offset).recipes)] ++
    (allAssertions interface offset).map Op.assertZero

def main {variableCount : Nat} (interface : Interface variableCount) : Circuit Unit :=
  fun offset =>
    ((), offset + (program interface offset).recipes.length,
      opsAt interface offset)

@[simp] theorem main_ops {variableCount : Nat} (interface : Interface variableCount)
    (offset : Nat) :
    Circuit.ops (main interface) offset = opsAt interface offset := by
  rfl

private theorem flatConstraints_opsAt {variableCount : Nat}
    (interface : Interface variableCount) (offset : Nat) :
    flatConstraints (opsAt interface offset) =
      recipeConstraints offset (program interface offset).recipes ++
        allAssertions interface offset := by
  simp [flatConstraints, opsAt, Op.flatConstraints, allAssertions,
    KExpr.equalities]

private theorem coordinateExprs_below {variableCount : Nat}
    (interface : Interface variableCount) (offset : Nat) (env : Env)
    (assumptions : Assumptions interface offset env) :
    ∀ coordinate ∈ coordinateExprs interface offset,
      coordinate.left.VarsBelow offset ∧ coordinate.right.VarsBelow offset := by
  intro coordinate member
  simp only [coordinateExprs, List.mem_map] at member
  rcases member with ⟨index, _, rfl⟩
  exact assumptions.1 index

theorem evaluate_eq_reference {variableCount : Nat}
    (interface : Interface variableCount) (offset : Nat) (env : Env) :
    evaluate env (coordinateExprs interface offset) =
      SumCheckTruthPath.pointEquality extensionOps
        (evalLeftPoint interface offset env)
        (evalRightPoint interface offset env) := by
  rw [evaluate_eq_pointEqualityCoordinates]
  simp [SumCheckTruthPath.pointEquality, coordinateExprs,
    evalLeftPoint, evalRightPoint, List.map_map, Function.comp_def]

/-- The semantic point-equality check is stable when every external input
wire is unchanged. -/
theorem specHolds_of_agree_below {variableCount : Nat}
    (interface : Interface variableCount) (offset : Nat)
    (before after : Env) (assumptions : Assumptions interface offset before)
    (agrees : ∀ index, index < offset → after index = before index)
    (specification : SpecHolds interface offset before) :
    SpecHolds interface offset after := by
  have expectedEq : (interface.expected offset).eval after =
      (interface.expected offset).eval before :=
    (interface.expected offset).eval_eq_of_agree_below offset after before
      assumptions.2 agrees
  have leftCoordinates :
      (evalLeftPoint interface offset after).coordinates =
        (evalLeftPoint interface offset before).coordinates := by
    apply List.map_congr_left
    intro coordinate _
    exact (interface.left offset coordinate).eval_eq_of_agree_below offset
      after before (assumptions.1 coordinate).1 agrees
  have rightCoordinates :
      (evalRightPoint interface offset after).coordinates =
        (evalRightPoint interface offset before).coordinates := by
    apply List.map_congr_left
    intro coordinate _
    exact (interface.right offset coordinate).eval_eq_of_agree_below offset
      after before (assumptions.1 coordinate).2 agrees
  unfold SpecHolds at specification ⊢
  rw [expectedEq]
  unfold SumCheckTruthPath.pointEquality
  rw [leftCoordinates, rightCoordinates]
  exact specification

theorem soundness {variableCount : Nat} (interface : Interface variableCount)
    (env : Env) (offset : Nat)
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
    (coordinateExprs interface offset) recipeRows
  unfold SpecHolds
  exact outputExpected.symm.trans <|
    outputSemantic.trans <| evaluate_eq_reference interface offset env

theorem completeness {variableCount : Nat} (interface : Interface variableCount)
    (env : Env) (offset : Nat)
    (assumptions : Assumptions interface offset env)
    (specification : SpecHolds interface offset env) :
    ∃ completed,
      AgreesOutside env completed offset
        (localLength (Circuit.ops (main interface) offset)) ∧
      holdsFlat completed (Circuit.ops (main interface) offset) := by
  let recipes := (program interface offset).recipes
  let completed := executeRecipes env offset recipes
  have causal : RecipesCausal offset recipes :=
    (compile_causal_and_output_below offset
      (coordinateExprs interface offset)
      (coordinateExprs_below interface offset env assumptions)).1
  have recipeRows : ConstraintsHold completed
      (recipeConstraints offset recipes) :=
    executeRecipes_holds_recipeConstraints env offset recipes causal
  have agreesBelow : ∀ index, index < offset → completed index = env index :=
    executeRecipes_agrees_below env offset recipes
  have leftCoordinates :
      (evalLeftPoint interface offset completed).coordinates =
        (evalLeftPoint interface offset env).coordinates := by
    apply List.map_congr_left
    intro coordinate _
    exact (interface.left offset coordinate).eval_eq_of_agree_below
      offset completed env (assumptions.1 coordinate).1 agreesBelow
  have rightCoordinates :
      (evalRightPoint interface offset completed).coordinates =
        (evalRightPoint interface offset env).coordinates := by
    apply List.map_congr_left
    intro coordinate _
    exact (interface.right offset coordinate).eval_eq_of_agree_below
      offset completed env (assumptions.1 coordinate).2 agreesBelow
  have referenceEq :
      SumCheckTruthPath.pointEquality extensionOps
          (evalLeftPoint interface offset completed)
          (evalRightPoint interface offset completed) =
        SumCheckTruthPath.pointEquality extensionOps
          (evalLeftPoint interface offset env)
          (evalRightPoint interface offset env) := by
    unfold SumCheckTruthPath.pointEquality
    rw [leftCoordinates, rightCoordinates]
  have expectedEval : (interface.expected offset).eval completed =
      (interface.expected offset).eval env :=
    (interface.expected offset).eval_eq_of_agree_below offset completed env
      assumptions.2 agreesBelow
  have outputSemantic := compile_output_sound completed offset
    (coordinateExprs interface offset) recipeRows
  have outputExpected : (program interface offset).output.eval completed =
      (interface.expected offset).eval completed := by
    calc
      (program interface offset).output.eval completed =
          evaluate completed (coordinateExprs interface offset) :=
        outputSemantic
      _ = SumCheckTruthPath.pointEquality extensionOps
          (evalLeftPoint interface offset completed)
          (evalRightPoint interface offset completed) :=
        evaluate_eq_reference interface offset completed
      _ = SumCheckTruthPath.pointEquality extensionOps
          (evalLeftPoint interface offset env)
          (evalRightPoint interface offset env) := referenceEq
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
  · change ConstraintsHold completed (flatConstraints (opsAt interface offset))
    rw [flatConstraints_opsAt]
    exact (constraintsHold_append completed _ _).mpr
      ⟨recipeRows, assertionRows⟩

def circuit {variableCount : Nat} (interface : Interface variableCount) : FormalCircuit where
  main := main interface
  assumptions := Assumptions interface
  spec := SpecHolds interface
  soundness := soundness interface
  completeness := completeness interface

theorem coordinateExprs_length {variableCount : Nat}
    (interface : Interface variableCount) (offset : Nat) :
    (coordinateExprs interface offset).length = variableCount := by
  simp [coordinateExprs, canonicalFinIndices_length]

theorem program_recipes_length_of_positive {variableCount : Nat}
    (interface : Interface variableCount) (offset : Nat)
    (positive : 0 < variableCount) :
    (program interface offset).recipes.length = 4 * variableCount - 2 := by
  unfold program
  rw [compile_recipes_length]
  have length := coordinateExprs_length interface offset
  cases coordinates : coordinateExprs interface offset with
  | nil =>
      rw [coordinates] at length
      simp at length
      omega
  | cons coordinate rest =>
      rw [coordinates] at length
      simp only [List.length_cons] at length ⊢
      omega

theorem localLength_eq_of_positive {variableCount : Nat}
    (interface : Interface variableCount) (offset : Nat)
    (positive : 0 < variableCount) :
    localLength (Circuit.ops (circuit interface).main offset) =
      4 * variableCount - 2 := by
  calc
    localLength (Circuit.ops (circuit interface).main offset) =
        (program interface offset).recipes.length := by
      change localLength (opsAt interface offset) = _
      simp [opsAt, localLength, allAssertions, KExpr.equalities,
        Op.localLength]
    _ = 4 * variableCount - 2 :=
      program_recipes_length_of_positive interface offset positive

theorem operations_length {variableCount : Nat}
    (interface : Interface variableCount)
    (offset : Nat) :
    (Circuit.ops (circuit interface).main offset).length = 3 := by
  change (opsAt interface offset).length = 3
  simp [opsAt, allAssertions, KExpr.equalities]

theorem flatConstraints_length_of_positive {variableCount : Nat}
    (interface : Interface variableCount) (offset : Nat)
    (positive : 0 < variableCount) :
    (flatConstraints (Circuit.ops (circuit interface).main offset)).length =
      4 * variableCount := by
  change (flatConstraints (opsAt interface offset)).length = _
  rw [flatConstraints_opsAt, List.length_append,
    recipeConstraints_length]
  rw [program_recipes_length_of_positive interface offset positive]
  simp [allAssertions, KExpr.equalities]
  omega

/-- Every flattened child row reads only external wires or this child's
completed private interval. -/
theorem flatConstraints_varsBelow {variableCount : Nat}
    (interface : Interface variableCount) (offset : Nat) (env : Env)
    (assumptions : Assumptions interface offset env) :
    ∀ expression ∈ flatConstraints (Circuit.ops (circuit interface).main offset),
      expression.VarsBelow
        (offset + (program interface offset).recipes.length) := by
  change ∀ expression ∈ flatConstraints (opsAt interface offset), _
  rw [flatConstraints_opsAt]
  intro expression member
  let coordinates := coordinateExprs interface offset
  have coordinatesBelow := coordinateExprs_below interface offset env assumptions
  rcases List.mem_append.mp member with recipeMember | assertionMember
  · exact recipeConstraints_varsBelow_of_causal offset
      (program interface offset).recipes
      (compile_causal_and_output_below offset coordinates coordinatesBelow).1
      expression recipeMember
  · have outputBelow := (compile_causal_and_output_below offset
      coordinates coordinatesBelow).2
    have expectedBelow : (interface.expected offset).VarsBelow
        (offset + (program interface offset).recipes.length) :=
      (interface.expected offset).varsBelow_mono assumptions.2 (by omega)
    exact KExpr.equalities_varsBelow
      (program interface offset).output (interface.expected offset)
      (offset + (program interface offset).recipes.length)
      outputBelow expectedBelow expression (by
        simpa [allAssertions] using assertionMember)

/-! ## Child-owned output variant -/

namespace Owned

/-!
Obligation: Compute the multilinear point-equality polynomial and expose the
compiler output directly to a parent circuit.

This interface has no external expected-output wire. It owns the same causal
factor/product recipes as `PointEquality.circuit` and removes the two final
copy rows.
-/

structure Interface (variableCount : Nat) where
  left : Nat → Fin variableCount → KExpr
  right : Nat → Fin variableCount → KExpr

def coordinateExprs {variableCount : Nat}
    (interface : Interface variableCount) (offset : Nat) :
    List CoordinateExpr :=
  (canonicalFinIndices variableCount).map fun coordinate =>
    ⟨interface.left offset coordinate, interface.right offset coordinate⟩

def evalLeftPoint {variableCount : Nat}
    (interface : Interface variableCount) (offset : Nat) (env : Env) :
    CubePoint K variableCount where
  coordinates := (canonicalFinIndices variableCount).map fun coordinate =>
    (interface.left offset coordinate).eval env
  dimension := by rw [List.length_map, canonicalFinIndices_length]

def evalRightPoint {variableCount : Nat}
    (interface : Interface variableCount) (offset : Nat) (env : Env) :
    CubePoint K variableCount where
  coordinates := (canonicalFinIndices variableCount).map fun coordinate =>
    (interface.right offset coordinate).eval env
  dimension := by rw [List.length_map, canonicalFinIndices_length]

def program {variableCount : Nat} (interface : Interface variableCount)
    (offset : Nat) : Program :=
  compile offset (coordinateExprs interface offset)

def output {variableCount : Nat} (interface : Interface variableCount)
    (offset : Nat) : KExpr :=
  (program interface offset).output

def Assumptions {variableCount : Nat} (interface : Interface variableCount)
    (offset : Nat) (_env : Env) : Prop :=
  ∀ coordinate,
    (interface.left offset coordinate).VarsBelow offset ∧
      (interface.right offset coordinate).VarsBelow offset

def SpecHolds {variableCount : Nat} (interface : Interface variableCount)
    (offset : Nat) (env : Env) : Prop :=
  (output interface offset).eval env =
    SumCheckTruthPath.pointEquality extensionOps
      (evalLeftPoint interface offset env) (evalRightPoint interface offset env)

def opsAt {variableCount : Nat} (interface : Interface variableCount)
    (offset : Nat) : List Op :=
  [Op.witness (WitnessBatch.arithmetic offset
    (program interface offset).recipes)]

def main {variableCount : Nat} (interface : Interface variableCount) :
    Circuit Unit := fun offset =>
  ((), offset + (program interface offset).recipes.length,
    opsAt interface offset)

@[simp] theorem main_ops {variableCount : Nat}
    (interface : Interface variableCount) (offset : Nat) :
    Circuit.ops (main interface) offset = opsAt interface offset := by
  rfl

theorem flatConstraints_opsAt {variableCount : Nat}
    (interface : Interface variableCount) (offset : Nat) :
    flatConstraints (opsAt interface offset) =
      recipeConstraints offset (program interface offset).recipes := by
  simp [flatConstraints, opsAt, Op.flatConstraints]

private theorem coordinateExprs_below {variableCount : Nat}
    (interface : Interface variableCount) (offset : Nat) (env : Env)
    (assumptions : Assumptions interface offset env) :
    ∀ coordinate ∈ coordinateExprs interface offset,
      coordinate.left.VarsBelow offset ∧ coordinate.right.VarsBelow offset := by
  intro coordinate member
  simp only [coordinateExprs, List.mem_map] at member
  rcases member with ⟨index, _, rfl⟩
  exact assumptions index

theorem evaluate_eq_reference {variableCount : Nat}
    (interface : Interface variableCount) (offset : Nat) (env : Env) :
    evaluate env (coordinateExprs interface offset) =
      SumCheckTruthPath.pointEquality extensionOps
        (evalLeftPoint interface offset env)
        (evalRightPoint interface offset env) := by
  rw [evaluate_eq_pointEqualityCoordinates]
  simp [SumCheckTruthPath.pointEquality, coordinateExprs,
    evalLeftPoint, evalRightPoint, List.map_map, Function.comp_def]

theorem soundness {variableCount : Nat} (interface : Interface variableCount)
    (env : Env) (offset : Nat)
    (_assumptions : Assumptions interface offset env)
    (rows : holds env (Circuit.ops (main interface) offset)) :
    SpecHolds interface offset env := by
  have recipeRows : ConstraintsHold env
      (recipeConstraints offset (program interface offset).recipes) :=
    rows (Op.witness (WitnessBatch.arithmetic offset
      (program interface offset).recipes))
      (by simp [main_ops, opsAt])
  exact (compile_output_sound env offset
    (coordinateExprs interface offset) recipeRows).trans
      (evaluate_eq_reference interface offset env)

/-- Honest execution constructs the owned result with no semantic premise. -/
theorem completeness {variableCount : Nat}
    (interface : Interface variableCount) (env : Env) (offset : Nat)
    (assumptions : Assumptions interface offset env) :
    ∃ completed,
      AgreesOutside env completed offset
        (localLength (Circuit.ops (main interface) offset)) ∧
      holdsFlat completed (Circuit.ops (main interface) offset) := by
  let recipes := (program interface offset).recipes
  let completed := executeRecipes env offset recipes
  have causal : RecipesCausal offset recipes :=
    (compile_causal_and_output_below offset
      (coordinateExprs interface offset)
      (coordinateExprs_below interface offset env assumptions)).1
  have recipeRows : ConstraintsHold completed
      (recipeConstraints offset recipes) :=
    executeRecipes_holds_recipeConstraints env offset recipes causal
  refine ⟨completed, ?_, ?_⟩
  · change AgreesOutside env completed offset recipes.length
    exact executeRecipes_agreesOutside env offset recipes
  · change ConstraintsHold completed (flatConstraints (opsAt interface offset))
    rw [flatConstraints_opsAt]
    exact recipeRows

def circuit {variableCount : Nat}
    (interface : Interface variableCount) : FormalCircuit where
  main := main interface
  assumptions := Assumptions interface
  spec := SpecHolds interface
  soundness := soundness interface
  completeness := fun env offset assumptions _specification =>
    completeness interface env offset assumptions

theorem build {variableCount : Nat} (interface : Interface variableCount)
    (env : Env) (offset : Nat)
    (assumptions : Assumptions interface offset env) :
    ∃ completed,
      AgreesOutside env completed offset
        (localLength (Circuit.ops (circuit interface).main offset)) ∧
      holdsFlat completed (Circuit.ops (circuit interface).main offset) :=
  completeness interface env offset assumptions

theorem coordinateExprs_length {variableCount : Nat}
    (interface : Interface variableCount) (offset : Nat) :
    (coordinateExprs interface offset).length = variableCount := by
  simp [coordinateExprs, canonicalFinIndices_length]

theorem program_recipes_length_of_positive {variableCount : Nat}
    (interface : Interface variableCount) (offset : Nat)
    (positive : 0 < variableCount) :
    (program interface offset).recipes.length = 4 * variableCount - 2 := by
  unfold program
  rw [compile_recipes_length]
  have length := coordinateExprs_length interface offset
  cases coordinates : coordinateExprs interface offset with
  | nil =>
      rw [coordinates] at length
      simp at length
      omega
  | cons coordinate rest =>
      rw [coordinates] at length
      simp only [List.length_cons] at length ⊢
      omega

theorem localLength_eq_of_positive {variableCount : Nat}
    (interface : Interface variableCount) (offset : Nat)
    (positive : 0 < variableCount) :
    localLength (Circuit.ops (circuit interface).main offset) =
      4 * variableCount - 2 := by
  change (program interface offset).recipes.length = _
  exact program_recipes_length_of_positive interface offset positive

theorem operations_length {variableCount : Nat}
    (interface : Interface variableCount) (offset : Nat) :
    (Circuit.ops (circuit interface).main offset).length = 1 := by
  rfl

theorem flatConstraints_length_of_positive {variableCount : Nat}
    (interface : Interface variableCount) (offset : Nat)
    (positive : 0 < variableCount) :
    (flatConstraints (Circuit.ops (circuit interface).main offset)).length =
      4 * variableCount - 2 := by
  change (flatConstraints (Circuit.ops (main interface) offset)).length = _
  rw [main_ops, flatConstraints_opsAt, recipeConstraints_length]
  exact program_recipes_length_of_positive interface offset positive

theorem flatConstraints_varsBelow {variableCount : Nat}
    (interface : Interface variableCount) (offset : Nat) (env : Env)
    (assumptions : Assumptions interface offset env) :
    ∀ expression ∈ flatConstraints
      (Circuit.ops (circuit interface).main offset),
      expression.VarsBelow
        (offset + (program interface offset).recipes.length) := by
  have causal : RecipesCausal offset (program interface offset).recipes :=
    (compile_causal_and_output_below offset
      (coordinateExprs interface offset)
      (coordinateExprs_below interface offset env assumptions)).1
  change ∀ expression ∈ flatConstraints (Circuit.ops (main interface) offset), _
  rw [main_ops, flatConstraints_opsAt]
  exact recipeConstraints_varsBelow_of_causal offset
    (program interface offset).recipes causal

/-- The owned point-equality result lies inside the child's declared
symbolic interval. -/
theorem output_varsBelow {variableCount : Nat}
    (interface : Interface variableCount) (offset : Nat) (env : Env)
    (assumptions : Assumptions interface offset env) :
    (output interface offset).VarsBelow
      (offset + localLength
        (Circuit.ops (circuit interface).main offset)) := by
  have coordinatesBelow := coordinateExprs_below interface offset env
    assumptions
  have below := (compile_causal_and_output_below offset
    (coordinateExprs interface offset) coordinatesBelow).2
  change (program interface offset).output.VarsBelow
    (offset + localLength (Circuit.ops (main interface) offset))
  simpa [main, opsAt, localLength, Op.localLength] using below

end Owned

end NightstreamFPrime.Gadgets.Multilinear.PointEquality
