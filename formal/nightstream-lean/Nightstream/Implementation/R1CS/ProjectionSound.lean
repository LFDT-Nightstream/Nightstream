import Nightstream.Implementation.R1CS.ProjectionProgram

/-!
Contract: semantic soundness for a complete PiRLC projection trace.

This module composes the row-family lemmas from `ProjectionProgram`.  Its main
theorem starts from exact definition and assertion-row satisfaction and ends at
the one-point polynomial identity consumed by `ProjectionCheck`.
-/

namespace Nightstream.Implementation.R1CS.ProjectionProgram

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program

private theorem projection_pair_values (trace : ProjectionTrace)
    (assignment : Nat → Nat) (point : K)
    (ladderValues : trace.ladder.powers.map
      (fun power => power.value assignment) =
        K.powersFrom point K.one trace.ladder.powers.length)
    (pairLayouts : ∀ pair ∈ trace.pairs,
      pair.LayoutValid trace.ladder.powers)
    (definitionsHold : DefinitionsHold assignment trace.definitions) :
    trace.pairProductValues assignment =
      trace.pairs.map fun pair =>
        Polynomial.eval (pair.productPolynomial assignment) point := by
  unfold ProjectionTrace.pairProductValues
  apply List.map_congr_left
  intro pair member
  have pairDefinitionsHold : DefinitionsHold assignment pair.definitions := by
    intro definition definitionMember
    apply definitionsHold definition
    unfold PairTrace.definitions at definitionMember
    rw [List.mem_append] at definitionMember
    rcases definitionMember with pairEvaluationMember | productMember
    · rw [List.mem_append] at pairEvaluationMember
      rcases pairEvaluationMember with rhoMember | inputMember
      · unfold ProjectionTrace.definitions
        apply List.mem_append_left
        apply List.mem_append_left
        apply List.mem_append_left
        apply List.mem_append_left
        apply List.mem_append_right trace.ladder.definitions
        exact List.mem_flatMap.mpr ⟨pair, member, rhoMember⟩
      · unfold ProjectionTrace.definitions
        apply List.mem_append_left
        apply List.mem_append_left
        apply List.mem_append_left
        apply List.mem_append_right
        exact List.mem_flatMap.mpr
          ⟨pair, member, List.mem_append_left _ inputMember⟩
    · unfold ProjectionTrace.definitions
      apply List.mem_append_left
      apply List.mem_append_left
      apply List.mem_append_left
      apply List.mem_append_right
      exact List.mem_flatMap.mpr
        ⟨pair, member, List.mem_append_right _ productMember⟩
  calc
    pair.product.output.value assignment =
        K.mul
          (Polynomial.eval (basePolynomial assignment pair.rhoColumns) point)
          (Polynomial.eval
            (basePolynomial assignment pair.inputColumns) point) :=
      pair.sound assignment point trace.ladder.powers ladderValues
        (pairLayouts pair member) pairDefinitionsHold
    _ = Polynomial.eval (pair.productPolynomial assignment) point := by
      exact (Polynomial.eval_mul
        (basePolynomial assignment pair.rhoColumns)
        (basePolynomial assignment pair.inputColumns) point).symm

private theorem projection_output_value (trace : ProjectionTrace)
    (assignment : Nat → Nat) (point : K)
    (ladderValues : trace.ladder.powers.map
      (fun power => power.value assignment) =
        K.powersFrom point K.one trace.ladder.powers.length)
    (layout : trace.outputEvaluation.LayoutValid)
    (coefficientShape : trace.outputEvaluation.coefficients =
      trace.outputColumns)
    (prefixShape : trace.outputEvaluation.powers =
      trace.ladder.powers.take trace.outputColumns.length)
    (definitionsHold : DefinitionsHold assignment trace.definitions) :
    trace.outputEvaluation.output.value assignment =
      Polynomial.eval (basePolynomial assignment trace.outputColumns) point := by
  have prefixShape' : trace.outputEvaluation.powers =
      trace.ladder.powers.take
        trace.outputEvaluation.coefficients.length := by
    rw [coefficientShape]
    exact prefixShape
  have outputDefinitionsHold : DefinitionsHold assignment
      trace.outputEvaluation.definitions := by
    intro definition member
    apply definitionsHold definition
    simp [ProjectionTrace.definitions, member]
  have within := trace.outputEvaluation.coefficientLength_le_ladder
    trace.ladder.powers layout prefixShape'
  have powersValid := trace.outputEvaluation.powersValid_of_ladderPrefix
    assignment point trace.ladder.powers prefixShape' within ladderValues
  have value := trace.outputEvaluation.sound assignment point layout
    powersValid outputDefinitionsHold
  rwa [coefficientShape] at value

private theorem projection_quotient_value (trace : ProjectionTrace)
    (assignment : Nat → Nat) (point : K)
    (ladderValues : trace.ladder.powers.map
      (fun power => power.value assignment) =
        K.powersFrom point K.one trace.ladder.powers.length)
    (layout : trace.quotientEvaluation.LayoutValid)
    (coefficientShape : trace.quotientEvaluation.coefficients =
      trace.quotientColumns)
    (prefixShape : trace.quotientEvaluation.powers =
      trace.ladder.powers.take trace.quotientColumns.length)
    (definitionsHold : DefinitionsHold assignment trace.definitions) :
    trace.quotientEvaluation.output.value assignment =
      Polynomial.eval
        (basePolynomial assignment trace.quotientColumns) point := by
  have prefixShape' : trace.quotientEvaluation.powers =
      trace.ladder.powers.take
        trace.quotientEvaluation.coefficients.length := by
    rw [coefficientShape]
    exact prefixShape
  have quotientDefinitionsHold : DefinitionsHold assignment
      trace.quotientEvaluation.definitions := by
    intro definition member
    apply definitionsHold definition
    simp [ProjectionTrace.definitions, member]
  have within := trace.quotientEvaluation.coefficientLength_le_ladder
    trace.ladder.powers layout prefixShape'
  have powersValid := trace.quotientEvaluation.powersValid_of_ladderPrefix
    assignment point trace.ladder.powers prefixShape' within ladderValues
  have value := trace.quotientEvaluation.sound assignment point layout
    powersValid quotientDefinitionsHold
  rwa [coefficientShape] at value

/-- Exact satisfaction of all named projection definitions and the final two
assertion rows forces equality of the two complete polynomial evaluations.
The theorem is quantified over the assignment; golden witnesses are not a
premise. -/
theorem ProjectionTrace.evaluation_sound (trace : ProjectionTrace)
    (assignment : Nat → Nat) (constantOne : assignment 0 = 1)
    (layout : trace.LayoutValid)
    (definitionsHold : DefinitionsHold assignment trace.definitions)
    (checksHold : Satisfies trace.checks assignment) :
    Nightstream.SuperNeo.ProjectionCheck.eval K.ops
        (trace.identity assignment).lhs
        (trace.identity assignment).beta =
      Nightstream.SuperNeo.ProjectionCheck.eval K.ops
        (trace.identity assignment).rhs
        (trace.identity assignment).beta := by
  rcases layout with
    ⟨ladderLayout, ladderLength, pairLayouts, outputLayout,
     quotientLayout, outputCoefficientShape, quotientCoefficientShape,
     outputPrefixShape, quotientPrefixShape, quotientPhiLeft,
     quotientPhiRight, quotientPhiLayout, outputLength, quotientLength,
     maxDegree⟩
  let point := trace.ladder.beta.value assignment
  have ladderDefinitionsHold : DefinitionsHold assignment
      trace.ladder.definitions := by
    intro definition member
    apply definitionsHold definition
    simp [ProjectionTrace.definitions, member]
  have ladderValues := trace.ladder.sound assignment constantOne ladderLayout
    ladderDefinitionsHold
  have pairValues := projection_pair_values trace assignment point ladderValues
    pairLayouts definitionsHold
  have outputValue := projection_output_value trace assignment point
    ladderValues outputLayout outputCoefficientShape outputPrefixShape
    definitionsHold
  have quotientValue := projection_quotient_value trace assignment point
    ladderValues quotientLayout quotientCoefficientShape quotientPrefixShape
    definitionsHold
  have quotientPhiDefinitionsHold : DefinitionsHold assignment
      trace.quotientPhiProduct.definitions := by
    intro definition member
    apply definitionsHold definition
    simp [ProjectionTrace.definitions, member]
  have quotientPhiValue := trace.quotientPhiProduct.sound assignment
    quotientPhiLayout quotientPhiDefinitionsHold
  have ladderValues55 : trace.ladder.powers.map
      (fun power => power.value assignment) =
        K.powersFrom point K.one 55 := by
    rw [← ladderLength]
    simpa [point] using ladderValues
  have phiValue := phiTerms_value assignment constantOne trace.ladder.powers
    ladderLength point ladderValues55
  rw [quotientPhiLeft, quotientPhiRight, KTerms.ofColumns_value,
    quotientValue, phiValue] at quotientPhiValue
  have checksValue := trace.checks_sound assignment constantOne checksHold
  have pairFold := congrArg
    (fun values : List K => values.foldr K.add K.zero) pairValues
  change (trace.pairProductValues assignment).foldr K.add K.zero =
    (trace.pairs.map fun pair =>
      Polynomial.eval (pair.productPolynomial assignment) point).foldr
        K.add K.zero at pairFold
  rw [List.foldr_map] at pairFold
  change Polynomial.eval
      (Polynomial.sum (trace.pairs.map fun pair =>
        pair.productPolynomial assignment)) point =
    Polynomial.eval
      (Polynomial.add
        (Polynomial.mul
          (basePolynomial assignment trace.quotientColumns)
          Polynomial.phi81)
        (Polynomial.padRight (trace.maxDegree + 1)
          (basePolynomial assignment trace.outputColumns))) point
  rw [Polynomial.eval_sum, List.foldr_map, Polynomial.eval_add,
    Polynomial.eval_mul, Polynomial.eval_padRight]
  rw [← pairFold, checksValue, quotientPhiValue, outputValue]

end Nightstream.Implementation.R1CS.ProjectionProgram
