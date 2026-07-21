import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.FamilySemantics

/-!
Topological source/compiler agreement for the focused `y_zcol` program.

Owns: the explicit dependency schedule (the initial product prefix, the full
evaluation schedule, then the dependent product suffix), its checked input
closure, and propagation of canonical assignment equality through the direct
family equations.

Does not own: retained final checks, selected-row decoding, assignment
construction, protocol authority, security events, or permission to remove
rows.

Emits constraints: no.

| Agreement leaf | Mathematical obligation | Authority class |
|---|---|---|
| dependency schedule | every operation reads only known or earlier outputs | checked |
| direct equations | source and compiler assignments satisfy the same operation | derived |
| propagation | both assignments agree on every scheduled output | derived |
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.Agreement

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.ProjectionProgram
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.DirectSemantics
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.SourceDecode
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.SourceProgram

inductive Operation where
  | evaluation (trace : EvalTrace)
  | product (trace : KMulTrace)

def evaluationInputs (trace : EvalTrace) : List Nat :=
  trace.coefficients.headD 0 ::
    trace.entries.flatMap fun entry =>
      [entry.1.1, entry.1.2.c0, entry.1.2.c1]

def productInputs (trace : KMulTrace) : List Nat :=
  (trace.left.c0 ++ trace.left.c1 ++
    trace.right.c0 ++ trace.right.c1).map Prod.fst

def Operation.inputs : Operation → List Nat
  | .evaluation trace => evaluationInputs trace
  | .product trace => productInputs trace

def Operation.outputs : Operation → List Nat
  | .evaluation trace => [trace.output.c0, trace.output.c1]
  | .product trace => [trace.output.c0, trace.output.c1]

def Operation.Direct (operation : Operation)
    (assignment : Nat → Nat) : Prop :=
  match operation with
  | .evaluation trace => EvaluationDirect trace assignment
  | .product trace => ProductDirect trace assignment

def operations : List Operation :=
  (SourceSchedule.productTraces.take 54).map Operation.product ++
    SourceSchedule.evaluationTraces.map Operation.evaluation ++
    (SourceSchedule.productTraces.drop 54).map Operation.product

def initialKnown : List Nat :=
  [7692949, 7692950] ++ sourceKnownColumns

def extendKnown (known : List Nat) (operation : Operation) : List Nat :=
  operation.outputs ++ known

def finalKnown : List Nat :=
  operations.foldl extendKnown initialKnown

def scheduleClosed : List Nat → List Operation → Bool
  | _, [] => true
  | known, operation :: rest =>
      operation.inputs.all known.contains &&
        scheduleClosed (extendKnown known operation) rest

/- Exact dependency fact. It checks source column numbers, not stage labels:
every direct equation reads only the initial boundary or earlier outputs. -/
set_option maxRecDepth 100000 in
theorem scheduleClosedChecked : scheduleClosed initialKnown operations = true := by
  native_decide

private theorem rawLcEval_eq_of_agree
    {left right : Nat → Nat} : ∀ terms : List (Nat × Nat),
    (∀ column ∈ terms.map Prod.fst, left column = right column) →
      rawLcEval left terms = rawLcEval right terms := by
  intro terms
  induction terms with
  | nil => intro _; rfl
  | cons head tail inductionHypothesis =>
      intro agree
      simp only [rawLcEval]
      rw [agree head.1 (by simp)]
      rw [inductionHypothesis]
      intro column member
      exact agree column (by simp [member])

private theorem lcEval_eq_of_agree
    {left right : Nat → Nat} (terms : List (Nat × Nat))
    (agree : ∀ column ∈ terms.map Prod.fst,
      left column = right column) :
    lcEval left terms = lcEval right terms := by
  rw [lcEval_eq_raw_mod, lcEval_eq_raw_mod,
    rawLcEval_eq_of_agree terms agree]

private theorem kColumnsValue_eq_of_agree
    {left right : Nat → Nat} (columns : KColumns)
    (c0 : left columns.c0 = right columns.c0)
    (c1 : left columns.c1 = right columns.c1) :
    columns.value left = columns.value right := by
  simp [KColumns.value, baseAt, c0, c1]

private theorem kTermsValue_eq_of_agree
    {left right : Nat → Nat} (terms : KTerms)
    (agree : AgreeOn left right
      ((terms.c0 ++ terms.c1).map Prod.fst)) :
    terms.value left = terms.value right := by
  have c0Terms : ∀ column ∈ terms.c0.map Prod.fst,
      left column = right column := by
    intro column member
    exact agree column (by simp [member])
  have c1Terms : ∀ column ∈ terms.c1.map Prod.fst,
      left column = right column := by
    intro column member
    exact agree column (by simp [member])
  simp only [KTerms.value, K.mk.injEq]
  exact ⟨congrArg residue (lcEval_eq_of_agree terms.c0 c0Terms),
    congrArg residue (lcEval_eq_of_agree terms.c1 c1Terms)⟩

private theorem evaluationRhs_eq_of_agree
    {left right : Nat → Nat} (trace : EvalTrace)
    (agree : AgreeOn left right (evaluationInputs trace)) :
    K.add
        (K.ofBase (baseAt left (trace.coefficients.headD 0)))
        ((trace.ExpectedProducts left).foldr K.add K.zero) =
      K.add
        (K.ofBase (baseAt right (trace.coefficients.headD 0)))
        ((trace.ExpectedProducts right).foldr K.add K.zero) := by
  have headEq : left (trace.coefficients.headD 0) =
      right (trace.coefficients.headD 0) :=
    agree _ (by simp [evaluationInputs])
  have productsEq : trace.ExpectedProducts left =
      trace.ExpectedProducts right := by
    unfold EvalTrace.ExpectedProducts
    apply List.map_congr_left
    intro entry member
    have coefficientEq : left entry.1.1 = right entry.1.1 := by
      apply agree
      simp only [evaluationInputs, List.mem_cons]
      exact Or.inr (List.mem_flatMap.mpr
        ⟨entry, member, by simp⟩)
    have powerC0Eq : left entry.1.2.c0 = right entry.1.2.c0 := by
      apply agree
      simp only [evaluationInputs, List.mem_cons]
      exact Or.inr (List.mem_flatMap.mpr
        ⟨entry, member, by simp⟩)
    have powerC1Eq : left entry.1.2.c1 = right entry.1.2.c1 := by
      apply agree
      simp only [evaluationInputs, List.mem_cons]
      exact Or.inr (List.mem_flatMap.mpr
        ⟨entry, member, by simp⟩)
    rw [show baseAt left entry.1.1 = baseAt right entry.1.1 by
      simp [baseAt, coefficientEq]]
    rw [kColumnsValue_eq_of_agree entry.1.2 powerC0Eq powerC1Eq]
  rw [productsEq]
  unfold baseAt
  rw [headEq]

private theorem productRhs_eq_of_agree
    {left right : Nat → Nat} (trace : KMulTrace)
    (agree : AgreeOn left right (productInputs trace)) :
    K.mul (trace.left.value left) (trace.right.value left) =
      K.mul (trace.left.value right) (trace.right.value right) := by
  have leftAgree : AgreeOn left right
      ((trace.left.c0 ++ trace.left.c1).map Prod.fst) := by
    intro column member
    apply agree
    rcases List.mem_map.mp member with ⟨term, termMember, rfl⟩
    apply List.mem_map.mpr
    refine ⟨term, ?_, rfl⟩
    simp only [List.mem_append] at termMember ⊢
    rcases termMember with inC0 | inC1
    · exact Or.inl (Or.inl (Or.inl inC0))
    · exact Or.inl (Or.inl (Or.inr inC1))
  have rightAgree : AgreeOn left right
      ((trace.right.c0 ++ trace.right.c1).map Prod.fst) := by
    intro column member
    apply agree
    rcases List.mem_map.mp member with ⟨term, termMember, rfl⟩
    apply List.mem_map.mpr
    refine ⟨term, ?_, rfl⟩
    simp only [List.mem_append] at termMember ⊢
    rcases termMember with inC0 | inC1
    · exact Or.inl (Or.inr inC0)
    · exact Or.inr inC1
  rw [kTermsValue_eq_of_agree trace.left leftAgree,
    kTermsValue_eq_of_agree trace.right rightAgree]

private theorem outputNatAgreement
    {left right : Nat → Nat} (columns : KColumns)
    (leftCanonical : ∀ column, left column < goldilocksP)
    (rightCanonical : ∀ column, right column < goldilocksP)
    (valuesEqual : columns.value left = columns.value right) :
    AgreeOn left right [columns.c0, columns.c1] := by
  intro column member
  simp only [List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl
  · have equal := congrArg K.c0 valuesEqual
    simpa [KColumns.value, baseAt, residue,
      Nat.mod_eq_of_lt (leftCanonical columns.c0),
      Nat.mod_eq_of_lt (rightCanonical columns.c0)] using equal
  · have equal := congrArg K.c1 valuesEqual
    simpa [KColumns.value, baseAt, residue,
      Nat.mod_eq_of_lt (leftCanonical columns.c1),
      Nat.mod_eq_of_lt (rightCanonical columns.c1)] using equal

private theorem operationOutputsAgree
    {left right : Nat → Nat} (operation : Operation)
    (leftCanonical : ∀ column, left column < goldilocksP)
    (rightCanonical : ∀ column, right column < goldilocksP)
    (inputsAgree : AgreeOn left right operation.inputs)
    (leftDirect : operation.Direct left)
    (rightDirect : operation.Direct right) :
    AgreeOn left right operation.outputs := by
  cases operation with
  | evaluation trace =>
      have rhs := evaluationRhs_eq_of_agree trace inputsAgree
      have values : trace.output.value left = trace.output.value right :=
        leftDirect.trans (rhs.trans rightDirect.symm)
      exact outputNatAgreement trace.output leftCanonical rightCanonical values
  | product trace =>
      have rhs := productRhs_eq_of_agree trace inputsAgree
      have values : trace.output.value left = trace.output.value right :=
        leftDirect.trans (rhs.trans rightDirect.symm)
      exact outputNatAgreement trace.output leftCanonical rightCanonical values

private theorem sourceEvaluationTracesDirect (assignment : Nat → Nat) :
    ∀ trace ∈ SourceSchedule.evaluationTraces,
      EvaluationDirect trace (sourceAssignment assignment) := by
  intro trace traceMember
  rcases List.mem_map.mp traceMember with ⟨owner, ownerMember, traceEq⟩
  subst trace
  exact DirectSemantics.sourceEvaluationsDirect assignment owner ownerMember

private theorem sourceProductTracesDirect (assignment : Nat → Nat) :
    ∀ trace ∈ SourceSchedule.productTraces,
      ProductDirect trace (sourceAssignment assignment) := by
  intro trace traceMember
  rcases List.mem_map.mp traceMember with ⟨owner, ownerMember, traceEq⟩
  subst trace
  exact DirectSemantics.sourceProductsDirect assignment owner ownerMember

theorem sourceOperationsDirect (assignment : Nat → Nat) :
    ∀ operation ∈ operations,
      operation.Direct (sourceAssignment assignment) := by
  intro operation member
  simp only [operations, List.mem_append, List.mem_map] at member
  rcases member with productPrefixOrEvaluation | productSuffix
  · rcases productPrefixOrEvaluation with productPrefix | evaluation
    · rcases productPrefix with ⟨trace, traceMember, rfl⟩
      exact sourceProductTracesDirect assignment trace
        (List.mem_of_mem_take traceMember)
    · rcases evaluation with ⟨trace, traceMember, rfl⟩
      exact sourceEvaluationTracesDirect assignment trace traceMember
  · rcases productSuffix with ⟨trace, traceMember, rfl⟩
    exact sourceProductTracesDirect assignment trace
      (List.mem_of_mem_drop traceMember)

theorem compilerOperationsDirect {assignment : Nat → Nat}
    (allSteps : ∀ step ∈ RewriteBridge.decodedRewriteSteps,
      RewriteBridge.StepHolds assignment step) :
    ∀ operation ∈ operations,
      operation.Direct (compilerAssignment assignment) := by
  intro operation member
  simp only [operations, List.mem_append, List.mem_map] at member
  rcases member with productPrefixOrEvaluation | productSuffix
  · rcases productPrefixOrEvaluation with productPrefix | evaluation
    · rcases productPrefix with ⟨trace, traceMember, rfl⟩
      exact FamilySemantics.compilerProductsDirect allSteps trace
        (List.mem_of_mem_take traceMember)
    · rcases evaluation with ⟨trace, traceMember, rfl⟩
      exact FamilySemantics.compilerEvaluationsDirect allSteps trace traceMember
  · rcases productSuffix with ⟨trace, traceMember, rfl⟩
    exact FamilySemantics.compilerProductsDirect allSteps trace
      (List.mem_of_mem_drop traceMember)

private theorem initialAgreement {assignment : Nat → Nat}
    (constantOne : assignment 0 = 1) :
    AgreeOn (sourceAssignment assignment) (compilerAssignment assignment)
      initialKnown := by
  intro column member
  simp only [initialKnown, List.mem_append, List.mem_cons,
    List.not_mem_nil, or_false] at member
  rcases member with base | known
  · rcases base with rfl | rfl
    · exact (ladderBaseAgrees constantOne).1
    · exact (ladderBaseAgrees constantOne).2
  · exact sourceAssignmentPreservesKnown assignment column known

private theorem propagateAgreement
    {left right : Nat → Nat}
    (leftCanonical : ∀ column, left column < goldilocksP)
    (rightCanonical : ∀ column, right column < goldilocksP)
    (leftDirect : ∀ operation ∈ operations, operation.Direct left)
    (rightDirect : ∀ operation ∈ operations, operation.Direct right) :
    ∀ (known : List Nat) (remaining : List Operation),
      (∀ operation ∈ remaining, operation ∈ operations) →
      scheduleClosed known remaining = true →
      AgreeOn left right known →
      AgreeOn left right (remaining.foldl extendKnown known) := by
  intro known remaining contained closed agree
  induction remaining generalizing known with
  | nil => simpa using agree
  | cons operation rest inductionHypothesis =>
      simp only [scheduleClosed, Bool.and_eq_true] at closed
      have operationMember : operation ∈ operations :=
        contained operation (by simp)
      have inputsAgree : AgreeOn left right operation.inputs := by
        intro column member
        apply agree
        have containedColumn :=
          (List.all_eq_true.mp closed.1) column member
        simpa using containedColumn
      have outputsAgree := operationOutputsAgree operation
        leftCanonical rightCanonical inputsAgree
        (leftDirect operation operationMember)
        (rightDirect operation operationMember)
      have extended : AgreeOn left right (extendKnown known operation) := by
        intro column member
        simp only [extendKnown, List.mem_append] at member
        rcases member with output | prior
        · exact outputsAgree column output
        · exact agree column prior
      apply inductionHypothesis (known := extendKnown known operation)
      · intro candidate candidateMember
        exact contained candidate (by simp [candidateMember])
      · exact closed.2
      · exact extended

/-- Every source column read or produced by the direct topological schedule
has the same canonical value in the independently recomputed source
assignment and the compiler assignment forced by the selected rows. -/
theorem sourceCompilerAgreeOnFinalKnown {assignment : Nat → Nat}
    (constantOne : assignment 0 = 1)
    (allSteps : ∀ step ∈ RewriteBridge.decodedRewriteSteps,
      RewriteBridge.StepHolds assignment step) :
    AgreeOn (sourceAssignment assignment) (compilerAssignment assignment)
      finalKnown := by
  unfold finalKnown
  apply propagateAgreement
    (sourceAssignmentCanonical assignment)
    (compilerAssignmentCanonical assignment)
    (sourceOperationsDirect assignment)
    (compilerOperationsDirect allSteps)
    initialKnown operations
  · intro operation member
    exact member
  · exact scheduleClosedChecked
  · exact initialAgreement constantOne

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.Agreement
