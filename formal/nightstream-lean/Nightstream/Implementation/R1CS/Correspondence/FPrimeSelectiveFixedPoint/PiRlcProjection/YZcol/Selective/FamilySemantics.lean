import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.DirectSemantics
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.QuadraticRefinement

/-!
Semantic interpretation of the checked quadratic rewrite families.

Owns: evaluation of the independently normalized family expressions, and
transport from all selected rewrite-step recurrences to the direct
`EvalTrace` and `KMulTrace` equations under the compiler assignment.

Does not own: source/selected output agreement, retained final checks,
assignment construction, protocol authority, security events, or permission
to remove rows.

Emits constraints: no.

| Rewrite family | Mathematical obligation | Authority class |
|---|---|---|
| polynomial evaluation | normalized recurrence equals the direct trace equation | derived |
| product sum | normalized recurrence equals extension multiplication | derived |
| schedule coverage | every scheduled source trace occurs once | checked |
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.FamilySemantics

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.ProjectionProgram
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.SourceDecode
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.RewriteBridge
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.DirectSemantics
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.QuadraticRefinement

private theorem sourceField_eq_baseAt (assignment : Nat → Nat)
    (column : Nat) :
    sourceFieldAssignment assignment column =
      baseAt (compilerAssignment assignment) column := by
  rfl

private theorem evalSingletonProduct (assignment : Nat → Nat)
    (coefficient left right : Nat) :
    Materialized.QuadraticForm.eval (sourceFieldAssignment assignment)
        (productExpression coefficient [(left, 1)] [(right, 1)]) =
      residue coefficient *
        baseAt (compilerAssignment assignment) left *
        baseAt (compilerAssignment assignment) right := by
  rw [evalProductExpression_projection,
    residue_lcEval_unit, residue_lcEval_unit]
  rfl

private theorem evalEntryC0 (assignment : Nat → Nat)
    (entry : (Nat × KColumns) × KColumns) :
    Materialized.QuadraticForm.eval (sourceFieldAssignment assignment)
        (productExpression 1 [(entry.1.1, 1)] [(entry.1.2.c0, 1)]) =
      (K.mul
        (K.ofBase (baseAt (compilerAssignment assignment) entry.1.1))
        (entry.1.2.value (compilerAssignment assignment))).c0 := by
  rw [evalSingletonProduct]
  simp only [K.mul, K.ofBase, KColumns.value, residue_one,
    Fin.one_mul, Fin.zero_mul, Fin.mul_zero, Fin.add_zero]

private theorem evalEntryC1 (assignment : Nat → Nat)
    (entry : (Nat × KColumns) × KColumns) :
    Materialized.QuadraticForm.eval (sourceFieldAssignment assignment)
        (productExpression 1 [(entry.1.1, 1)] [(entry.1.2.c1, 1)]) =
      (K.mul
        (K.ofBase (baseAt (compilerAssignment assignment) entry.1.1))
        (entry.1.2.value (compilerAssignment assignment))).c1 := by
  rw [evalSingletonProduct]
  simp only [K.mul, K.ofBase, KColumns.value, residue_one,
    Fin.one_mul, Fin.zero_mul, Fin.mul_zero, Fin.add_zero]

private theorem evalEntriesC0 (assignment : Nat → Nat) :
    ∀ entries : List ((Nat × KColumns) × KColumns),
      Materialized.QuadraticForm.eval (sourceFieldAssignment assignment)
          (entries.flatMap fun entry =>
            productExpression 1 [(entry.1.1, 1)] [(entry.1.2.c0, 1)]) =
        (entries.map fun entry =>
          (K.mul
            (K.ofBase (baseAt (compilerAssignment assignment) entry.1.1))
            (entry.1.2.value (compilerAssignment assignment))).c0).foldr
          (fun left right => left + right) K.zero.c0 := by
  intro entries
  induction entries with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      simp only [List.flatMap_cons, Materialized.QuadraticForm.eval_append, List.map_cons,
        List.foldr_cons]
      rw [evalEntryC0, inductionHypothesis]
      rfl

private theorem evalEntriesC1 (assignment : Nat → Nat) :
    ∀ entries : List ((Nat × KColumns) × KColumns),
      Materialized.QuadraticForm.eval (sourceFieldAssignment assignment)
          (entries.flatMap fun entry =>
            productExpression 1 [(entry.1.1, 1)] [(entry.1.2.c1, 1)]) =
        (entries.map fun entry =>
          (K.mul
            (K.ofBase (baseAt (compilerAssignment assignment) entry.1.1))
            (entry.1.2.value (compilerAssignment assignment))).c1).foldr
          (fun left right => left + right) K.zero.c1 := by
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
      (values.map K.c0).foldr (fun left right => left + right) K.zero.c0 := by
  intro values
  induction values with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      simp only [List.foldr_cons, K.add, List.map_cons]
      rw [inductionHypothesis]

private theorem foldK_c1 : ∀ values : List K,
    (values.foldr K.add K.zero).c1 =
      (values.map K.c1).foldr (fun left right => left + right) K.zero.c1 := by
  intro values
  induction values with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      simp only [List.foldr_cons, K.add, List.map_cons]
      rw [inductionHypothesis]

private theorem evalEvaluationC0 (assignment : Nat → Nat)
    (trace : EvalTrace) :
    Materialized.QuadraticForm.eval (sourceFieldAssignment assignment)
        (evaluationC0Expression trace) =
      (K.add
        (K.ofBase (baseAt (compilerAssignment assignment)
          (trace.coefficients.headD 0)))
        ((trace.ExpectedProducts (compilerAssignment assignment)).foldr
          K.add K.zero)).c0 := by
  rw [evaluationC0Expression, Materialized.QuadraticForm.eval_append,
    evalTermsExpression_sourceField, fieldResidue_lcEval_unit,
    evalEntriesC0]
  simp only [K.add, K.ofBase]
  rw [foldK_c0]
  simp only [fieldResidue_eq_residue, baseAt, evaluationProducts,
    EvalTrace.ExpectedProducts, List.map_map, Function.comp_def, K.ofBase]
  rfl

private theorem evalEvaluationC1 (assignment : Nat → Nat)
    (trace : EvalTrace) :
    Materialized.QuadraticForm.eval (sourceFieldAssignment assignment)
        (evaluationC1Expression trace) =
      (K.add
        (K.ofBase (baseAt (compilerAssignment assignment)
          (trace.coefficients.headD 0)))
        ((trace.ExpectedProducts (compilerAssignment assignment)).foldr
          K.add K.zero)).c1 := by
  rw [evaluationC1Expression, evalEntriesC1]
  simp only [K.add, K.ofBase, Fin.zero_add]
  rw [foldK_c1]
  simp only [evaluationProducts, EvalTrace.ExpectedProducts,
    List.map_map, Function.comp_def, K.ofBase]

private theorem evalProductC0 (assignment : Nat → Nat)
    (trace : KMulTrace) :
    Materialized.QuadraticForm.eval (sourceFieldAssignment assignment) (productC0Expression trace) =
      (K.mul (trace.left.value (compilerAssignment assignment))
        (trace.right.value (compilerAssignment assignment))).c0 := by
  rw [productC0Expression, Materialized.QuadraticForm.eval_append,
    evalProductExpression_one, evalProductExpression_seven]
  rfl

private theorem evalProductC1 (assignment : Nat → Nat)
    (trace : KMulTrace) :
    Materialized.QuadraticForm.eval (sourceFieldAssignment assignment) (productC1Expression trace) =
      (K.mul (trace.left.value (compilerAssignment assignment))
        (trace.right.value (compilerAssignment assignment))).c1 := by
  rw [productC1Expression, Materialized.QuadraticForm.eval_append,
    evalProductExpression_one, evalProductExpression_one]
  rfl

theorem evaluationDirect_of_expectedHolds (assignment : Nat → Nat)
    (trace : EvalTrace)
    (holds : ExpectedHolds assignment (evaluationExpected trace)) :
    EvaluationDirect trace (compilerAssignment assignment) := by
  have c0 : sourceFieldAssignment assignment trace.output.c0 =
      Materialized.QuadraticForm.eval (sourceFieldAssignment assignment)
        (evaluationC0Expression trace) := by
    exact holds
      { outputColumn := trace.output.c0,
        expression := evaluationC0Expression trace }
      (by
        simp only [evaluationExpected, List.mem_cons]
        exact Or.inl trivial)
  have c1 : sourceFieldAssignment assignment trace.output.c1 =
      Materialized.QuadraticForm.eval (sourceFieldAssignment assignment)
        (evaluationC1Expression trace) := by
    exact holds
      { outputColumn := trace.output.c1,
        expression := evaluationC1Expression trace }
      (by
        simp only [evaluationExpected, List.mem_cons]
        exact Or.inr (Or.inl trivial))
  unfold EvaluationDirect
  simp only [KColumns.value, K.add, K.mk.injEq]
  constructor
  · exact (sourceField_eq_baseAt assignment trace.output.c0).symm.trans
      (c0.trans (evalEvaluationC0 assignment trace))
  · exact (sourceField_eq_baseAt assignment trace.output.c1).symm.trans
      (c1.trans (evalEvaluationC1 assignment trace))

theorem productDirect_of_expectedHolds (assignment : Nat → Nat)
    (trace : KMulTrace)
    (holds : ExpectedHolds assignment (productExpected trace)) :
    ProductDirect trace (compilerAssignment assignment) := by
  have c0 : sourceFieldAssignment assignment trace.output.c0 =
      Materialized.QuadraticForm.eval (sourceFieldAssignment assignment)
        (productC0Expression trace) := by
    exact holds
      { outputColumn := trace.output.c0,
        expression := productC0Expression trace }
      (by
        simp only [productExpected, List.mem_cons]
        exact Or.inl trivial)
  have c1 : sourceFieldAssignment assignment trace.output.c1 =
      Materialized.QuadraticForm.eval (sourceFieldAssignment assignment)
        (productC1Expression trace) := by
    exact holds
      { outputColumn := trace.output.c1,
        expression := productC1Expression trace }
      (by
        simp only [productExpected, List.mem_cons]
        exact Or.inr (Or.inl trivial))
  unfold ProductDirect
  simp only [KColumns.value, KTerms.value, K.mul, K.mk.injEq]
  constructor
  · exact (sourceField_eq_baseAt assignment trace.output.c0).symm.trans
      (c0.trans (evalProductC0 assignment trace))
  · exact (sourceField_eq_baseAt assignment trace.output.c1).symm.trans
      (c1.trans (evalProductC1 assignment trace))

private theorem evaluationPairStepMember
    {pair : List DecodedRewriteStep × EvalTrace}
    (pairMember : pair ∈ evaluationPairs) {step : DecodedRewriteStep}
    (stepMember : step ∈ pair.1) : step ∈ decodedRewriteSteps := by
  have groupMember : pair.1 ∈ evaluationGroups :=
    (List.of_mem_zip pairMember).1
  rcases List.mem_map.mp groupMember with ⟨index, _, groupEq⟩
  have sliced : step ∈
      (decodedRewriteSteps.drop (22 * index)).take 22 := by
    rw [groupEq]
    exact stepMember
  exact List.mem_of_mem_drop (List.mem_of_mem_take sliced)

private theorem productPairStepMember
    {pair : List DecodedRewriteStep × KMulTrace}
    (pairMember : pair ∈ productPairs) {step : DecodedRewriteStep}
    (stepMember : step ∈ pair.1) : step ∈ decodedRewriteSteps := by
  have groupMember : pair.1 ∈ productGroups :=
    (List.of_mem_zip pairMember).1
  rcases List.mem_map.mp groupMember with ⟨index, _, groupEq⟩
  have sliced : step ∈
      (decodedRewriteSteps.drop (1078 + 2 * index)).take 2 := by
    rw [groupEq]
    exact stepMember
  exact List.mem_of_mem_drop (List.mem_of_mem_take sliced)

theorem evaluationPairsDirect {assignment : Nat → Nat}
    (allSteps : ∀ step ∈ decodedRewriteSteps,
      StepHolds assignment step) :
    ∀ pair ∈ evaluationPairs,
      EvaluationDirect pair.2 (compilerAssignment assignment) := by
  intro pair pairMember
  apply evaluationDirect_of_expectedHolds
  apply groupExpectedHolds
  · intro step stepMember
    exact allSteps step (evaluationPairStepMember pairMember stepMember)
  · exact evaluationGroupsExact.2 pair pairMember

theorem productPairsDirect {assignment : Nat → Nat}
    (allSteps : ∀ step ∈ decodedRewriteSteps,
      StepHolds assignment step) :
    ∀ pair ∈ productPairs,
      ProductDirect pair.2 (compilerAssignment assignment) := by
  intro pair pairMember
  apply productDirect_of_expectedHolds
  apply groupExpectedHolds
  · intro step stepMember
    exact allSteps step (productPairStepMember pairMember stepMember)
  · exact productGroupsExact.2 pair pairMember

set_option maxRecDepth 100000 in
theorem evaluationPairTracesExact :
    evaluationPairs.map Prod.snd = SourceSchedule.evaluationTraces := by
  native_decide

set_option maxRecDepth 100000 in
theorem productPairTracesExact :
    productPairs.map Prod.snd = SourceSchedule.productTraces := by
  native_decide

theorem compilerEvaluationsDirect {assignment : Nat → Nat}
    (allSteps : ∀ step ∈ decodedRewriteSteps,
      StepHolds assignment step) :
    ∀ trace ∈ SourceSchedule.evaluationTraces,
      EvaluationDirect trace (compilerAssignment assignment) := by
  intro trace traceMember
  have mapped : trace ∈ evaluationPairs.map Prod.snd := by
    rw [evaluationPairTracesExact]
    exact traceMember
  rcases List.mem_map.mp mapped with ⟨pair, pairMember, traceEq⟩
  subst trace
  exact evaluationPairsDirect allSteps pair pairMember

theorem compilerProductsDirect {assignment : Nat → Nat}
    (allSteps : ∀ step ∈ decodedRewriteSteps,
      StepHolds assignment step) :
    ∀ trace ∈ SourceSchedule.productTraces,
      ProductDirect trace (compilerAssignment assignment) := by
  intro trace traceMember
  have mapped : trace ∈ productPairs.map Prod.snd := by
    rw [productPairTracesExact]
    exact traceMember
  rcases List.mem_map.mp mapped with ⟨pair, pairMember, traceEq⟩
  subst trace
  exact productPairsDirect allSteps pair pairMember

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.FamilySemantics
