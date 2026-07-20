import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.HonestAssignment
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.QuadraticRefinement

/-!
Direct-family justification of the honest intermediate-program terminals.

Owns: lockstep execution of the deterministic derived-value program and the
independently normalized quadratic expressions, transport of direct
`EvalTrace`/`KMulTrace` equations to terminal expressions, and exact grouping
of the rewrite steps.

Does not own: centered-word packing, retained final checks, selected-row
coefficient equality, producer authority, security events, or row removal.

Emits constraints: no.

| Leaf | Mathematical obligation | Authority class |
|---|---|---|
| `honest.terminal_lockstep` | derived execution evaluates the checked symbolic recurrence | derived |
| `honest.evaluation_terminals` | terminal pairs equal direct polynomial evaluations | artifact-checked + derived |
| `honest.product_terminals` | terminal pairs equal direct extension products | artifact-checked + derived |
| `honest.rewrite_partition` | grouped steps concatenate to every rewrite | artifact-checked |
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.HonestAssignment.TerminalSemantics

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.ProjectionProgram
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.SourceDecode
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.RewriteBridge
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.DirectSemantics
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.QuadraticRefinement
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.HonestAssignment


def sourceField (source : Nat → Nat) : Nat → F :=
  fun column => Materialized.Semantics.fieldResidue (source column)

theorem evalLinearExpression (source : Nat → Nat)
    (linear : DecodedSourceLinearCombination) :
    Materialized.QuadraticForm.eval (sourceField source) (QuadraticRefinement.linearExpression linear) =
      abstractSourceValue source linear := by
  unfold QuadraticRefinement.linearExpression
  rw [Materialized.QuadraticForm.eval_ofLinear]
  exact evalNatTermsLinearForm source linear.programTerms

theorem evalFactorExpression (source : Nat → Nat)
    (factor : DecodedProductFactor) :
    Materialized.QuadraticForm.eval (sourceField source) (QuadraticRefinement.factorExpression factor) =
      abstractFactorValue source factor := by
  unfold QuadraticRefinement.factorExpression abstractFactorValue
  rw [Materialized.QuadraticForm.eval_mulLinear]
  change Materialized.Semantics.fieldResidue factor.coefficient *
      evalLinearForm source (natTermsLinearForm factor.left.programTerms) *
      evalLinearForm source (natTermsLinearForm factor.right.programTerms) = _
  rw [
    evalNatTermsLinearForm source factor.left.programTerms,
    evalNatTermsLinearForm source factor.right.programTerms]
  rfl

private theorem evalFactorsExpression (source : Nat → Nat) :
    ∀ factors,
      Materialized.QuadraticForm.eval (sourceField source) (QuadraticRefinement.factorsExpression factors) =
        factors.foldr
          (fun factor suffix => abstractFactorValue source factor + suffix) 0 := by
  intro factors
  induction factors with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      simp only [QuadraticRefinement.factorsExpression, List.flatMap_cons, Materialized.QuadraticForm.eval_append,
        List.foldr_cons]
      unfold QuadraticRefinement.factorsExpression at inductionHypothesis
      rw [evalFactorExpression, inductionHypothesis]

private theorem factorFold_eq_sum (source : Nat → Nat)
    (factors : List DecodedProductFactor) (capacity : factors.length ≤ 5) :
    factors.foldr
        (fun factor suffix => abstractFactorValue source factor + suffix) 0 =
      abstractFactorSum source factors := by
  cases factors with
  | nil => rfl
  | cons first rest =>
      cases rest with
      | nil => simp [abstractFactorSum, abstractFactorValueAt]
      | cons second rest =>
          cases rest with
          | nil =>
              simp [abstractFactorSum, abstractFactorValueAt]
          | cons third rest =>
              cases rest with
              | nil =>
                  simp [abstractFactorSum, abstractFactorValueAt]
                  simp only [Lean.Grind.Fin.add_assoc]
              | cons fourth rest =>
                  cases rest with
                  | nil =>
                      simp [abstractFactorSum, abstractFactorValueAt]
                      simp only [Lean.Grind.Fin.add_assoc]
                  | cons fifth rest =>
                      cases rest with
                      | nil =>
                          simp [abstractFactorSum, abstractFactorValueAt]
                          simp only [Lean.Grind.Fin.add_assoc]
                      | cons sixth rest => simp at capacity

structure StateConsistent (source : Nat → Nat) (derived : Nat → F)
    (symbolic : QuadraticRefinement.SymbolicState) : Prop where
  lastDerived : ∀ slot expression,
    symbolic.lastDerived = some (slot, expression) →
      derived slot.compilerIndex = Materialized.QuadraticForm.eval (sourceField source) expression

theorem emptyConsistent (source : Nat → Nat) (derived : Nat → F) :
    StateConsistent source derived QuadraticRefinement.emptyState := by
  constructor
  intro slot expression impossible
  simp [QuadraticRefinement.emptyState] at impossible

private theorem evalPreviousExpression
    {source : Nat → Nat} {derived : Nat → F}
    {symbolic : QuadraticRefinement.SymbolicState}
    (consistent : StateConsistent source derived symbolic)
    (previous : Option DecodedDerivedSlot) (expression : Materialized.QuadraticForm.Form)
    (decoded : QuadraticRefinement.previousExpression? symbolic previous = some expression) :
    Materialized.QuadraticForm.eval (sourceField source) expression =
      derivedPreviousValue derived previous := by
  cases previous with
  | none =>
      simp [QuadraticRefinement.previousExpression?] at decoded
      subst expression
      rfl
  | some slot =>
      cases priorEq : symbolic.lastDerived with
      | none => simp [QuadraticRefinement.previousExpression?, priorEq] at decoded
      | some prior =>
          rcases prior with ⟨priorSlot, priorExpression⟩
          by_cases same : QuadraticRefinement.SameDerivedSlot slot priorSlot
          · have expressionEq : expression = priorExpression := by
              simpa [QuadraticRefinement.previousExpression?, priorEq, same] using decoded.symm
            subst expression
            have slotEq := QuadraticRefinement.sameDerivedSlot_eq same
            subst priorSlot
            exact (consistent.lastDerived slot priorExpression priorEq).symm
          · simp [QuadraticRefinement.previousExpression?, priorEq, same] at decoded

private theorem evalRewriteExpression
    {source : Nat → Nat} {derived : Nat → F}
    {symbolic : QuadraticRefinement.SymbolicState}
    (consistent : StateConsistent source derived symbolic)
    (step : DecodedRewriteStep) (expression : Materialized.QuadraticForm.Form)
    (decoded : QuadraticRefinement.rewriteExpression? symbolic step = some expression) :
    Materialized.QuadraticForm.eval (sourceField source) expression = derivedRhs source derived step := by
  cases previousEq : QuadraticRefinement.previousExpression? symbolic step.previous with
  | none => simp [QuadraticRefinement.rewriteExpression?, previousEq] at decoded
  | some previousExpression =>
      have expressionEq : expression =
          QuadraticRefinement.linearExpression step.base ++ previousExpression ++
            QuadraticRefinement.factorsExpression step.factors := by
        simpa [QuadraticRefinement.rewriteExpression?, previousEq] using decoded.symm
      subst expression
      rw [Materialized.QuadraticForm.eval_append, Materialized.QuadraticForm.eval_append, evalLinearExpression,
        evalPreviousExpression consistent step.previous previousExpression
          previousEq,
        evalFactorsExpression,
        factorFold_eq_sum source step.factors step.factorCapacity]
      rfl

private theorem executeStep_consistent
    {source : Nat → Nat} {derived : Nat → F}
    {symbolic next : QuadraticRefinement.SymbolicState} {step : DecodedRewriteStep}
    (consistent : StateConsistent source derived symbolic)
    (executed : QuadraticRefinement.executeStep symbolic step = some next) :
    StateConsistent source (executeDerived source derived step) next := by
  cases expressionEq : QuadraticRefinement.rewriteExpression? symbolic step with
  | none => simp [QuadraticRefinement.executeStep, expressionEq] at executed
  | some expression =>
      have expressionValue := evalRewriteExpression consistent step expression
        expressionEq
      cases outputEq : step.output with
      | source output =>
          have nextEq : next =
              { symbolic with terminals := symbolic.terminals ++
                [{ output := output, expression }] } := by
            simpa [QuadraticRefinement.executeStep, expressionEq, outputEq] using executed.symm
          subst next
          constructor
          intro slot candidateExpression selected
          simpa only [executeDerived, outputEq] using
            consistent.lastDerived slot candidateExpression selected
      | derivedProductSum slot =>
          have nextEq : next =
              { symbolic with lastDerived := some (slot, expression) } := by
            simpa [QuadraticRefinement.executeStep, expressionEq, outputEq] using executed.symm
          subst next
          constructor
          intro candidate candidateExpression selected
          simp only [Option.some.injEq, Prod.mk.injEq] at selected
          rcases selected with ⟨slotEq, expressionEq'⟩
          subst candidate
          subst candidateExpression
          simp only [executeDerived, outputEq, setDerived, if_pos]
          exact expressionValue.symm

private theorem executeStep_terminals_mono
    {symbolic next : QuadraticRefinement.SymbolicState} {step : DecodedRewriteStep}
    (executed : QuadraticRefinement.executeStep symbolic step = some next) :
    ∀ terminal ∈ symbolic.terminals, terminal ∈ next.terminals := by
  cases expressionEq : QuadraticRefinement.rewriteExpression? symbolic step with
  | none => simp [QuadraticRefinement.executeStep, expressionEq] at executed
  | some expression =>
      cases outputEq : step.output with
      | source output =>
          have nextEq : next =
              { symbolic with terminals := symbolic.terminals ++
                [{ output := output, expression }] } := by
            simpa [QuadraticRefinement.executeStep, expressionEq, outputEq] using executed.symm
          subst next
          intro terminal member
          simp [member]
      | derivedProductSum slot =>
          have nextEq : next =
              { symbolic with lastDerived := some (slot, expression) } := by
            simpa [QuadraticRefinement.executeStep, expressionEq, outputEq] using executed.symm
          subst next
          intro terminal member
          exact member

private theorem executeSteps_terminals_mono
    {initial final : QuadraticRefinement.SymbolicState} {steps : List DecodedRewriteStep}
    (executed : QuadraticRefinement.executeSteps initial steps = some final) :
    ∀ terminal ∈ initial.terminals, terminal ∈ final.terminals := by
  induction steps generalizing initial with
  | nil =>
      simp [QuadraticRefinement.executeSteps] at executed
      subst final
      intro terminal member
      exact member
  | cons step rest inductionHypothesis =>
      cases nextEq : QuadraticRefinement.executeStep initial step with
      | none => simp [QuadraticRefinement.executeSteps, nextEq] at executed
      | some next =>
          have suffix : QuadraticRefinement.executeSteps next rest = some final := by
            simpa [QuadraticRefinement.executeSteps, nextEq] using executed
          intro terminal member
          exact inductionHypothesis suffix terminal
            (executeStep_terminals_mono nextEq terminal member)

def TerminalValid (source : Nat → Nat)
    (terminal : QuadraticRefinement.TerminalExpression) : Prop :=
  abstractSourceValue source terminal.output =
    Materialized.QuadraticForm.eval (sourceField source) terminal.expression

private theorem sourceStep_terminal
    {source : Nat → Nat} {derived : Nat → F}
    {symbolic next final : QuadraticRefinement.SymbolicState}
    {step : DecodedRewriteStep} {rest : List DecodedRewriteStep}
    {output : DecodedSourceLinearCombination}
    (outputEq : step.output = .source output)
    (executed : QuadraticRefinement.executeStep symbolic step = some next)
    (suffix : QuadraticRefinement.executeSteps next rest = some final)
    (finalValid : ∀ terminal ∈ final.terminals,
      TerminalValid source terminal)
    (consistent : StateConsistent source derived symbolic) :
    abstractSourceValue source output = derivedRhs source derived step := by
  cases expressionEq : QuadraticRefinement.rewriteExpression? symbolic step with
  | none => simp [QuadraticRefinement.executeStep, expressionEq] at executed
  | some expression =>
      have nextEq : next =
          { symbolic with terminals := symbolic.terminals ++
            [{ output := output, expression }] } := by
        simpa [QuadraticRefinement.executeStep, expressionEq, outputEq] using executed.symm
      have inNext :
          ({ output := output, expression := expression } :
            QuadraticRefinement.TerminalExpression) ∈ next.terminals := by
        subst next
        simp
      have inFinal := executeSteps_terminals_mono suffix _ inNext
      exact (finalValid _ inFinal).trans
        (evalRewriteExpression consistent step expression expressionEq)

private theorem terminalsHoldFrom_of_execute
    {source : Nat → Nat} {derived : Nat → F}
    {symbolic final : QuadraticRefinement.SymbolicState} {steps : List DecodedRewriteStep}
    (consistent : StateConsistent source derived symbolic)
    (executed : QuadraticRefinement.executeSteps symbolic steps = some final)
    (finalValid : ∀ terminal ∈ final.terminals,
      TerminalValid source terminal) :
    TerminalsHoldFrom source derived steps := by
  induction steps generalizing derived symbolic with
  | nil => trivial
  | cons step rest inductionHypothesis =>
      cases nextEq : QuadraticRefinement.executeStep symbolic step with
      | none => simp [QuadraticRefinement.executeSteps, nextEq] at executed
      | some next =>
          have suffix : QuadraticRefinement.executeSteps next rest = some final := by
            simpa [QuadraticRefinement.executeSteps, nextEq] using executed
          constructor
          · cases outputEq : step.output with
            | source output =>
                exact sourceStep_terminal outputEq nextEq suffix finalValid
                  consistent
            | derivedProductSum slot => trivial
          · exact inductionHypothesis
              (executeStep_consistent consistent nextEq) suffix

def ExpectedSourceHolds (source : Nat → Nat)
    (expected : List QuadraticRefinement.ExpectedTerminal) : Prop :=
  ∀ terminal ∈ expected,
    sourceField source terminal.outputColumn =
      Materialized.QuadraticForm.eval (sourceField source) terminal.expression

private theorem abstractSourceValue_of_unit
    (source : Nat → Nat) (output : DecodedSourceLinearCombination)
    (column : Nat) (unit : output.programTerms = [(column, 1)]) :
    abstractSourceValue source output = sourceField source column := by
  unfold abstractSourceValue sourceField
  rw [unit]
  exact QuadraticRefinement.fieldResidue_lcEval_unit source column

private theorem terminalValid_of_match
    {source : Nat → Nat} {actual : QuadraticRefinement.TerminalExpression}
    {expected : QuadraticRefinement.ExpectedTerminal}
    (expectedHolds : sourceField source expected.outputColumn =
      Materialized.QuadraticForm.eval (sourceField source) expected.expression)
    (matching : QuadraticRefinement.TerminalMatches actual expected) :
    TerminalValid source actual := by
  rcases matching with ⟨unit, equivalent⟩
  unfold TerminalValid
  rw [abstractSourceValue_of_unit source actual.output
    expected.outputColumn unit, expectedHolds]
  exact (Materialized.QuadraticForm.eval_eq_of_equivalent equivalent _).symm

private theorem terminalsValid_of_match
    {source : Nat → Nat}
    {actual : List QuadraticRefinement.TerminalExpression}
    {expected : List QuadraticRefinement.ExpectedTerminal}
    (expectedHolds : ExpectedSourceHolds source expected)
    (matching : QuadraticRefinement.TerminalsMatch actual expected) :
    ∀ terminal ∈ actual, TerminalValid source terminal := by
  induction matching with
  | nil => intro terminal member; simp at member
  | @cons actualHead expectedHead actualTail expectedTail headMatch tailMatch
      inductionHypothesis =>
      intro terminal member
      simp only [List.mem_cons] at member
      rcases member with rfl | inTail
      · apply terminalValid_of_match
        · exact expectedHolds expectedHead (by simp)
        · exact headMatch
      · apply inductionHypothesis
        · intro candidate candidateMember
          exact expectedHolds candidate (by simp [candidateMember])
        · exact inTail

theorem groupTerminalsHold
    (source : Nat → Nat) (derived : Nat → F)
    {steps : List DecodedRewriteStep}
    {expected : List QuadraticRefinement.ExpectedTerminal}
    (matching : QuadraticRefinement.GroupMatches steps expected)
    (expectedHolds : ExpectedSourceHolds source expected) :
    TerminalsHoldFrom source derived steps := by
  rcases matching with ⟨actual, executed, pairwise⟩
  unfold QuadraticRefinement.executeGroup at executed
  cases finalEq : QuadraticRefinement.executeSteps QuadraticRefinement.emptyState steps with
  | none => simp [finalEq] at executed
  | some final =>
      have actualEq : actual = final.terminals := by
        simpa [finalEq] using executed.symm
      subst actual
      exact terminalsHoldFrom_of_execute
        (emptyConsistent source derived) finalEq
        (terminalsValid_of_match expectedHolds pairwise)

/-! ## Direct family equations imply expected quadratic terminals -/

private theorem evalTermsExpression (source : Nat → Nat)
    (terms : List (Nat × Nat)) :
    Materialized.QuadraticForm.eval (sourceField source) (QuadraticRefinement.termsExpression terms) =
      Materialized.Semantics.fieldResidue (lcEval source terms) := by
  unfold QuadraticRefinement.termsExpression
  rw [Materialized.QuadraticForm.eval_ofLinear]
  exact evalNatTermsLinearForm source terms

private theorem evalProductExpression (source : Nat → Nat)
    (coefficient : Nat) (left right : List (Nat × Nat)) :
    Materialized.QuadraticForm.eval (sourceField source)
        (QuadraticRefinement.productExpression coefficient left right) =
      Materialized.Semantics.fieldResidue coefficient *
        Materialized.Semantics.fieldResidue (lcEval source left) *
        Materialized.Semantics.fieldResidue (lcEval source right) := by
  unfold QuadraticRefinement.productExpression
  rw [Materialized.QuadraticForm.eval_mulLinear]
  unfold QuadraticRefinement.termsLinearForm
  change Materialized.Semantics.fieldResidue coefficient *
      evalLinearForm source (natTermsLinearForm left) *
      evalLinearForm source (natTermsLinearForm right) = _
  rw [evalNatTermsLinearForm, evalNatTermsLinearForm]

private theorem evalProductExpressionProjection (source : Nat → Nat)
    (coefficient : Nat) (left right : List (Nat × Nat)) :
    Materialized.QuadraticForm.eval (sourceField source)
        (QuadraticRefinement.productExpression coefficient left right) =
      residue coefficient * residue (lcEval source left) *
        residue (lcEval source right) := by
  rw [evalProductExpression,
    QuadraticRefinement.fieldResidue_eq_residue,
    QuadraticRefinement.fieldResidue_eq_residue,
    QuadraticRefinement.fieldResidue_eq_residue]
  rfl

private theorem evalProductExpressionOne (source : Nat → Nat)
    (left right : List (Nat × Nat)) :
    Materialized.QuadraticForm.eval (sourceField source)
        (QuadraticRefinement.productExpression 1 left right) =
      residue (lcEval source left) * residue (lcEval source right) := by
  rw [evalProductExpressionProjection, residue_one, Fin.one_mul]

private theorem evalProductExpressionSeven (source : Nat → Nat)
    (left right : List (Nat × Nat)) :
    Materialized.QuadraticForm.eval (sourceField source)
        (QuadraticRefinement.productExpression 7 left right) =
      residue 7 *
        (residue (lcEval source left) * residue (lcEval source right)) := by
  rw [evalProductExpressionProjection, Fin.mul_assoc]

private theorem evalSingletonProduct (source : Nat → Nat)
    (coefficient left right : Nat) :
    Materialized.QuadraticForm.eval (sourceField source)
        (QuadraticRefinement.productExpression coefficient [(left, 1)] [(right, 1)]) =
      residue coefficient *
        baseAt source left * baseAt source right := by
  rw [evalProductExpressionProjection,
    QuadraticRefinement.residue_lcEval_unit,
    QuadraticRefinement.residue_lcEval_unit]
  rfl

private theorem evalEntryC0 (source : Nat → Nat)
    (entry : (Nat × KColumns) × KColumns) :
    Materialized.QuadraticForm.eval (sourceField source)
        (QuadraticRefinement.productExpression 1 [(entry.1.1, 1)] [(entry.1.2.c0, 1)]) =
      (K.mul (K.ofBase (baseAt source entry.1.1))
        (entry.1.2.value source)).c0 := by
  rw [evalSingletonProduct]
  simp only [K.mul, K.ofBase, KColumns.value, residue_one,
    Fin.one_mul, Fin.zero_mul, Fin.mul_zero, Fin.add_zero]

private theorem evalEntryC1 (source : Nat → Nat)
    (entry : (Nat × KColumns) × KColumns) :
    Materialized.QuadraticForm.eval (sourceField source)
        (QuadraticRefinement.productExpression 1 [(entry.1.1, 1)] [(entry.1.2.c1, 1)]) =
      (K.mul (K.ofBase (baseAt source entry.1.1))
        (entry.1.2.value source)).c1 := by
  rw [evalSingletonProduct]
  simp only [K.mul, K.ofBase, KColumns.value, residue_one,
    Fin.one_mul, Fin.zero_mul, Fin.mul_zero, Fin.add_zero]

private theorem evalEntriesC0 (source : Nat → Nat) :
    ∀ entries : List ((Nat × KColumns) × KColumns),
      Materialized.QuadraticForm.eval (sourceField source)
          (entries.flatMap fun entry =>
            QuadraticRefinement.productExpression 1 [(entry.1.1, 1)] [(entry.1.2.c0, 1)]) =
        (entries.map fun entry =>
          (K.mul (K.ofBase (baseAt source entry.1.1))
            (entry.1.2.value source)).c0).foldr
              (fun left right : ProjectionProgram.F => left + right)
              (0 : ProjectionProgram.F) := by
  intro entries
  induction entries with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      simp only [List.flatMap_cons, Materialized.QuadraticForm.eval_append, List.map_cons,
        List.foldr_cons]
      rw [evalEntryC0, inductionHypothesis]
      rfl

private theorem evalEntriesC1 (source : Nat → Nat) :
    ∀ entries : List ((Nat × KColumns) × KColumns),
      Materialized.QuadraticForm.eval (sourceField source)
          (entries.flatMap fun entry =>
            QuadraticRefinement.productExpression 1 [(entry.1.1, 1)] [(entry.1.2.c1, 1)]) =
        (entries.map fun entry =>
          (K.mul (K.ofBase (baseAt source entry.1.1))
            (entry.1.2.value source)).c1).foldr
              (fun left right : ProjectionProgram.F => left + right)
              (0 : ProjectionProgram.F) := by
  intro entries
  induction entries with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      simp only [List.flatMap_cons, Materialized.QuadraticForm.eval_append, List.map_cons,
        List.foldr_cons]
      rw [evalEntryC1, inductionHypothesis]
      rfl

private theorem foldK_c0 : ∀ values : List K,
    (values.foldr K.add K.zero).c0 =
      (values.map K.c0).foldr (· + ·) 0 := by
  intro values
  induction values with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      simp only [List.foldr_cons, K.add, List.map_cons]
      rw [inductionHypothesis]

private theorem foldK_c1 : ∀ values : List K,
    (values.foldr K.add K.zero).c1 =
      (values.map K.c1).foldr (· + ·) 0 := by
  intro values
  induction values with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      simp only [List.foldr_cons, K.add, List.map_cons]
      rw [inductionHypothesis]

private theorem evalEvaluationC0 (source : Nat → Nat)
    (trace : EvalTrace) :
    Materialized.QuadraticForm.eval (sourceField source) (QuadraticRefinement.evaluationC0Expression trace) =
      (K.add
        (K.ofBase (baseAt source (trace.coefficients.headD 0)))
        ((trace.ExpectedProducts source).foldr K.add K.zero)).c0 := by
  rw [QuadraticRefinement.evaluationC0Expression, Materialized.QuadraticForm.eval_append,
    evalTermsExpression, QuadraticRefinement.fieldResidue_lcEval_unit,
    evalEntriesC0]
  simp only [K.add, K.ofBase]
  rw [foldK_c0]
  simp only [QuadraticRefinement.fieldResidue_eq_residue, baseAt,
    QuadraticRefinement.evaluationProducts, EvalTrace.ExpectedProducts,
    List.map_map, Function.comp_def, K.ofBase]
  rfl

private theorem evalEvaluationC1 (source : Nat → Nat)
    (trace : EvalTrace) :
    Materialized.QuadraticForm.eval (sourceField source) (QuadraticRefinement.evaluationC1Expression trace) =
      (K.add
        (K.ofBase (baseAt source (trace.coefficients.headD 0)))
        ((trace.ExpectedProducts source).foldr K.add K.zero)).c1 := by
  rw [QuadraticRefinement.evaluationC1Expression, evalEntriesC1]
  simp only [K.add, K.ofBase, Fin.zero_add]
  rw [foldK_c1]
  simp only [QuadraticRefinement.evaluationProducts,
    EvalTrace.ExpectedProducts, List.map_map, Function.comp_def, K.ofBase]

private theorem evalProductC0 (source : Nat → Nat)
    (trace : KMulTrace) :
    Materialized.QuadraticForm.eval (sourceField source) (QuadraticRefinement.productC0Expression trace) =
      (K.mul (trace.left.value source) (trace.right.value source)).c0 := by
  rw [QuadraticRefinement.productC0Expression, Materialized.QuadraticForm.eval_append,
    evalProductExpressionOne, evalProductExpressionSeven]
  rfl

private theorem evalProductC1 (source : Nat → Nat)
    (trace : KMulTrace) :
    Materialized.QuadraticForm.eval (sourceField source) (QuadraticRefinement.productC1Expression trace) =
      (K.mul (trace.left.value source) (trace.right.value source)).c1 := by
  rw [QuadraticRefinement.productC1Expression, Materialized.QuadraticForm.eval_append,
    evalProductExpressionOne, evalProductExpressionOne]
  rfl

theorem expectedEvaluationHolds_of_direct
    (source : Nat → Nat) (trace : EvalTrace)
    (direct : EvaluationDirect trace source) :
    ExpectedSourceHolds source (QuadraticRefinement.evaluationExpected trace) := by
  intro terminal member
  simp only [QuadraticRefinement.evaluationExpected, List.mem_cons,
    List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl
  · change sourceField source trace.output.c0 = _
    rw [evalEvaluationC0]
    have component := congrArg K.c0 direct
    simpa [sourceField, KColumns.value, baseAt, residue] using component
  · change sourceField source trace.output.c1 = _
    rw [evalEvaluationC1]
    have component := congrArg K.c1 direct
    simpa [sourceField, KColumns.value, baseAt, residue] using component

theorem expectedProductHolds_of_direct
    (source : Nat → Nat) (trace : KMulTrace)
    (direct : ProductDirect trace source) :
    ExpectedSourceHolds source (QuadraticRefinement.productExpected trace) := by
  intro terminal member
  simp only [QuadraticRefinement.productExpected, List.mem_cons,
    List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl
  · change sourceField source trace.output.c0 = _
    rw [evalProductC0]
    have component := congrArg K.c0 direct
    simpa [sourceField, KColumns.value, baseAt, residue] using component
  · change sourceField source trace.output.c1 = _
    rw [evalProductC1]
    have component := congrArg K.c1 direct
    simpa [sourceField, KColumns.value, baseAt, residue] using component

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.HonestAssignment.TerminalSemantics
