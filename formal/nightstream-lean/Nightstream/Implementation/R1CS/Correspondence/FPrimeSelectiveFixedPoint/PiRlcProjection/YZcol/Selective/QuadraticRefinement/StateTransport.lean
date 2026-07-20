import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.QuadraticRefinement.FieldEvaluation

/-!
Symbolic-state transport for the focused compact `y_zcol` quadratic
refinement.

Owns: the invariant relating compact rewrite state to decoded source values,
and preservation of that invariant by one step and a complete group.

Does not own: certificate aggregation, terminal matching, selected-row
materialization, protocol authority, or security events.

Emits constraints: no.

| State leaf | Mathematical obligation | Authority class |
|---|---|---|
| empty state | the initial symbolic state has no unsupported derived value or terminal | derived |
| rewrite step | a satisfied source recurrence preserves the decoded-value invariant | direct dataflow |
| complete group | executing satisfied steps leaves every emitted terminal equal to its source value | derived |
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.QuadraticRefinement

open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.ProjectionProgram
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.SourceDecode
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.RewriteBridge

private theorem factorFold_eq_factorSum (assignment : Nat → Nat)
    (factors : List DecodedProductFactor) (capacity : factors.length ≤ 5) :
    factors.foldr
        (fun factor suffix => factorValue assignment factor + suffix) 0 =
      factorSum assignment factors := by
  cases factors with
  | nil => rfl
  | cons first rest =>
      cases rest with
      | nil => simp [factorSum, factorValueAt]
      | cons second rest =>
          cases rest with
          | nil =>
              simp [factorSum, factorValueAt]
          | cons third rest =>
              cases rest with
              | nil =>
                  simp [factorSum, factorValueAt, Lean.Grind.Fin.add_assoc]
              | cons fourth rest =>
                  cases rest with
                  | nil =>
                      simp [factorSum, factorValueAt, Lean.Grind.Fin.add_assoc]
                  | cons fifth rest =>
                      cases rest with
                      | nil =>
                          simp [factorSum, factorValueAt, Lean.Grind.Fin.add_assoc]
                      | cons sixth rest => simp at capacity

structure StateValid (assignment : Nat → Nat)
    (state : SymbolicState) : Prop where
  lastDerived : ∀ slot expression,
    state.lastDerived = some (slot, expression) →
      derivedValue assignment slot =
        Materialized.QuadraticForm.eval (sourceFieldAssignment assignment) expression
  terminals : ∀ terminal ∈ state.terminals,
    sourceValue assignment terminal.output =
      Materialized.QuadraticForm.eval (sourceFieldAssignment assignment) terminal.expression

theorem emptyStateValid (assignment : Nat → Nat) :
    StateValid assignment emptyState := by
  constructor
  · intro slot expression impossible
    simp [emptyState] at impossible
  · intro terminal impossible
    simp [emptyState] at impossible

private theorem evalPreviousExpression {assignment : Nat → Nat}
    {state : SymbolicState} (valid : StateValid assignment state)
    (previous : Option DecodedDerivedSlot) (expression : Form)
    (decoded : previousExpression? state previous = some expression) :
    Materialized.QuadraticForm.eval (sourceFieldAssignment assignment) expression =
      previousValue assignment previous := by
  cases previous with
  | none =>
      simp [previousExpression?] at decoded
      subst expression
      rfl
  | some slot =>
      cases priorEq : state.lastDerived with
      | none =>
          simp [previousExpression?, priorEq] at decoded
      | some prior =>
          rcases prior with ⟨priorSlot, priorExpression⟩
          by_cases same : SameDerivedSlot slot priorSlot
          · have expressionEq : expression = priorExpression := by
              simpa [previousExpression?, priorEq, same] using decoded.symm
            subst expression
            have slotEq := sameDerivedSlot_eq same
            subst priorSlot
            exact (valid.lastDerived slot priorExpression priorEq).symm
          · simp [previousExpression?, priorEq, same] at decoded

private theorem evalRewriteExpression {assignment : Nat → Nat}
    {state : SymbolicState} (valid : StateValid assignment state)
    (step : DecodedRewriteStep) (expression : Form)
    (decoded : rewriteExpression? state step = some expression) :
    Materialized.QuadraticForm.eval (sourceFieldAssignment assignment) expression =
      sourceValue assignment step.base +
        previousValue assignment step.previous +
        factorSum assignment step.factors := by
  cases previousEq : previousExpression? state step.previous with
  | none =>
      simp [rewriteExpression?, previousEq] at decoded
  | some previousExpression =>
      have expressionEq : expression =
          linearExpression step.base ++ previousExpression ++
            factorsExpression step.factors := by
        simpa [rewriteExpression?, previousEq] using decoded.symm
      subst expression
      rw [Materialized.QuadraticForm.eval_append, Materialized.QuadraticForm.eval_append,
        evalLinearExpression,
        evalPreviousExpression valid step.previous previousExpression
          previousEq,
        evalFactorsExpression,
        factorFold_eq_factorSum assignment step.factors step.factorCapacity]

private theorem executeStep_valid {assignment : Nat → Nat}
    {state next : SymbolicState} {step : DecodedRewriteStep}
    (valid : StateValid assignment state)
    (holds : StepHolds assignment step)
    (executed : executeStep state step = some next) :
    StateValid assignment next := by
  cases expressionEq : rewriteExpression? state step with
  | none =>
      simp [executeStep, expressionEq] at executed
  | some expression =>
      have expressionValue := evalRewriteExpression valid step expression
        expressionEq
      cases outputEq : step.output with
      | source output =>
          have nextEq : next =
              { state with
                terminals := state.terminals ++
                  [{ output := output, expression }] } := by
            simpa [executeStep, expressionEq, outputEq] using executed.symm
          subst next
          constructor
          · intro slot priorExpression priorEq
            exact valid.lastDerived slot priorExpression priorEq
          · intro terminal member
            rw [List.mem_append] at member
            rcases member with old | added
            · exact valid.terminals terminal old
            · simp only [List.mem_singleton] at added
              subst terminal
              change sourceValue assignment output = _
              rw [expressionValue]
              simpa [StepHolds, outputEq] using holds
      | derivedProductSum slot =>
          have nextEq : next =
              { state with lastDerived := some (slot, expression) } := by
            simpa [executeStep, expressionEq, outputEq] using executed.symm
          subst next
          constructor
          · intro candidate candidateExpression selected
            simp only [Option.some.injEq, Prod.mk.injEq] at selected
            rcases selected with ⟨slotEq, expressionEq'⟩
            subst candidate
            subst candidateExpression
            change derivedValue assignment slot = _
            rw [expressionValue]
            simpa [StepHolds, outputEq] using holds
          · intro terminal member
            exact valid.terminals terminal member

def StepsHold (assignment : Nat → Nat)
    (steps : List DecodedRewriteStep) : Prop :=
  ∀ step ∈ steps, StepHolds assignment step

private theorem executeSteps_valid {assignment : Nat → Nat}
    {initial final : SymbolicState} {steps : List DecodedRewriteStep}
    (valid : StateValid assignment initial)
    (holds : StepsHold assignment steps)
    (executed : executeSteps initial steps = some final) :
    StateValid assignment final := by
  induction steps generalizing initial with
  | nil =>
      simp [executeSteps] at executed
      subst final
      exact valid
  | cons step rest inductionHypothesis =>
      cases nextEq : executeStep initial step with
      | none =>
          simp [executeSteps, nextEq] at executed
      | some next =>
          have stepHolds : StepHolds assignment step :=
            holds step (by simp)
          have nextValid := executeStep_valid valid stepHolds nextEq
          apply inductionHypothesis nextValid
          · intro candidate member
            exact holds candidate (by simp [member])
          · simpa [executeSteps, nextEq] using executed

theorem executeGroup_terminalsValid {assignment : Nat → Nat}
    {steps : List DecodedRewriteStep} {terminals : List TerminalExpression}
    (holds : StepsHold assignment steps)
    (executed : executeGroup steps = some terminals) :
    ∀ terminal ∈ terminals,
      sourceValue assignment terminal.output =
        Materialized.QuadraticForm.eval (sourceFieldAssignment assignment) terminal.expression := by
  unfold executeGroup at executed
  cases finalEq : executeSteps emptyState steps with
  | none => simp [finalEq] at executed
  | some final =>
      have terminalsEq : terminals = final.terminals := by
        simpa [finalEq] using executed.symm
      subst terminals
      exact (executeSteps_valid (emptyStateValid assignment) holds finalEq).terminals

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.QuadraticRefinement
