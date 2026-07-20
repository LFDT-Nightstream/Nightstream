import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.Materialized.QuadraticForm
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.RewriteBridge
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.SourceSchedule

/-!
Shared symbolic core for the focused compact `y_zcol` quadratic refinement.

Owns: independent degree-two source expressions, deterministic symbolic
execution, length-preserving terminal correspondence, and the exact
evaluation/product group lists paired with the source schedule.

Does not own: native group certificates, selected-row satisfaction,
source-program execution, protocol authority, security events, or permission
to remove rows.

Emits constraints: no.

| Core leaf | Mathematical obligation | Authority class |
|---|---|---|
| symbolic recurrence | execute base, predecessor, and product factors | computed |
| terminal relation | pair executed terminals with independent targets | derived |
| group schedule | pair checked rewrite slices with source traces | direct dataflow |
| compact certificate bridge | lift proof-free normalized terminal data to `GroupMatches` | derived |
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.QuadraticRefinement

open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.ProjectionProgram
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.SourceDecode
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.RewriteBridge

abbrev Form := Materialized.QuadraticForm.Form

def linearExpression (linear : DecodedSourceLinearCombination) : Form :=
  Materialized.QuadraticForm.ofLinear (natTermsLinearForm linear.programTerms)

def termsLinearForm (terms : List (Nat × Nat)) : LinearForm :=
  natTermsLinearForm terms

def termsExpression (terms : List (Nat × Nat)) : Form :=
  Materialized.QuadraticForm.ofLinear (termsLinearForm terms)

def productExpression (coefficient : Nat)
    (left right : List (Nat × Nat)) : Form :=
  Materialized.QuadraticForm.mulLinear (Materialized.Semantics.fieldResidue coefficient)
    (termsLinearForm left) (termsLinearForm right)

def factorExpression (factor : DecodedProductFactor) : Form :=
  Materialized.QuadraticForm.mulLinear (Materialized.Semantics.fieldResidue factor.coefficient)
    (natTermsLinearForm factor.left.programTerms)
    (natTermsLinearForm factor.right.programTerms)

def factorsExpression (factors : List DecodedProductFactor) : Form :=
  factors.flatMap factorExpression

structure TerminalExpression where
  output : DecodedSourceLinearCombination
  expression : Form

structure SymbolicState where
  lastDerived : Option (DecodedDerivedSlot × Form)
  terminals : List TerminalExpression

def emptyState : SymbolicState :=
  { lastDerived := none, terminals := [] }

def SameDerivedSlot (left right : DecodedDerivedSlot) : Prop :=
  left.compilerIndex = right.compilerIndex ∧
    left.start = right.start ∧ left.width = right.width

instance (left right : DecodedDerivedSlot) :
    Decidable (SameDerivedSlot left right) := by
  unfold SameDerivedSlot
  infer_instance

def previousExpression? (state : SymbolicState) :
    Option DecodedDerivedSlot → Option Form
  | none => some []
  | some slot =>
      match state.lastDerived with
      | none => none
      | some prior =>
          if SameDerivedSlot slot prior.1 then some prior.2 else none

def rewriteExpression? (state : SymbolicState)
    (step : DecodedRewriteStep) : Option Form := do
  let previous ← previousExpression? state step.previous
  pure (linearExpression step.base ++ previous ++
    factorsExpression step.factors)

def executeStep (state : SymbolicState)
    (step : DecodedRewriteStep) : Option SymbolicState := do
  let expression ← rewriteExpression? state step
  match step.output with
  | .source linear =>
      pure
        { state with
          terminals := state.terminals ++
            [{ output := linear, expression }] }
  | .derivedProductSum slot =>
      pure { state with lastDerived := some (slot, expression) }

def executeSteps : SymbolicState → List DecodedRewriteStep →
    Option SymbolicState
  | state, [] => some state
  | state, step :: rest => do
      executeSteps (← executeStep state step) rest

def executeGroup (steps : List DecodedRewriteStep) :
    Option (List TerminalExpression) := do
  pure (← executeSteps emptyState steps).terminals

structure ExpectedTerminal where
  outputColumn : Nat
  expression : Form

def evaluationProducts (trace : EvalTrace) :
    List ((Nat × KColumns) × KColumns) :=
  trace.entries

def evaluationC0Expression (trace : EvalTrace) : Form :=
  termsExpression [(trace.coefficients.headD 0, 1)] ++
    (evaluationProducts trace).flatMap fun entry =>
      productExpression 1 [(entry.1.1, 1)] [(entry.1.2.c0, 1)]

def evaluationC1Expression (trace : EvalTrace) : Form :=
  (evaluationProducts trace).flatMap fun entry =>
    productExpression 1 [(entry.1.1, 1)] [(entry.1.2.c1, 1)]

def evaluationExpected (trace : EvalTrace) : List ExpectedTerminal :=
  [ { outputColumn := trace.output.c0,
      expression := evaluationC0Expression trace },
    { outputColumn := trace.output.c1,
      expression := evaluationC1Expression trace } ]

def productC0Expression (trace : KMulTrace) : Form :=
  productExpression 1 trace.left.c0 trace.right.c0 ++
    productExpression 7 trace.left.c1 trace.right.c1

def productC1Expression (trace : KMulTrace) : Form :=
  productExpression 1 trace.left.c0 trace.right.c1 ++
    productExpression 1 trace.left.c1 trace.right.c0

def productExpected (trace : KMulTrace) : List ExpectedTerminal :=
  [ { outputColumn := trace.output.c0,
      expression := productC0Expression trace },
    { outputColumn := trace.output.c1,
      expression := productC1Expression trace } ]

def TerminalMatches (actual : TerminalExpression)
    (expected : ExpectedTerminal) : Prop :=
  actual.output.programTerms = [(expected.outputColumn, 1)] ∧
    Materialized.QuadraticForm.Equivalent actual.expression expected.expression

instance (actual : TerminalExpression) (expected : ExpectedTerminal) :
    Decidable (TerminalMatches actual expected) := by
  unfold TerminalMatches
  infer_instance

/-- Length-preserving correspondence between executed and independently
expected terminals. Kept local to this refinement so no generic list
relation becomes protocol authority. -/
inductive TerminalsMatch :
    List TerminalExpression → List ExpectedTerminal → Prop
  | nil : TerminalsMatch [] []
  | cons {actual expected actualRest expectedRest}
      (head : TerminalMatches actual expected)
      (tail : TerminalsMatch actualRest expectedRest) :
      TerminalsMatch (actual :: actualRest) (expected :: expectedRest)

private def terminalsMatchDecidable :
    (actual : List TerminalExpression) →
      (expected : List ExpectedTerminal) →
        Decidable (TerminalsMatch actual expected)
  | [], [] => isTrue .nil
  | [], _ :: _ => isFalse fun matching => by cases matching
  | _ :: _, [] => isFalse fun matching => by cases matching
  | actual :: actualRest, expected :: expectedRest =>
      if head : TerminalMatches actual expected then
        match terminalsMatchDecidable actualRest expectedRest with
        | isTrue tail => isTrue (.cons head tail)
        | isFalse notTail => isFalse fun matching => by
            cases matching with
            | cons _ tail => exact notTail tail
      else
        isFalse fun matching => by
          cases matching with
          | cons actualHead _ => exact head actualHead

instance (actual : List TerminalExpression)
    (expected : List ExpectedTerminal) :
    Decidable (TerminalsMatch actual expected) :=
  terminalsMatchDecidable actual expected

def GroupMatches (steps : List DecodedRewriteStep)
    (expected : List ExpectedTerminal) : Prop :=
  ∃ actual,
    executeGroup steps = some actual ∧
      TerminalsMatch actual expected

instance (steps : List DecodedRewriteStep)
    (expected : List ExpectedTerminal) : Decidable (GroupMatches steps expected) := by
  cases executed : executeGroup steps with
  | none =>
      exact isFalse fun witness => by
        rcases witness with ⟨actual, actualExecuted, _⟩
        rw [executed] at actualExecuted
        contradiction
  | some actual =>
      if matching : TerminalsMatch actual expected then
        exact isTrue ⟨actual, executed, matching⟩
      else
        exact isFalse fun witness => by
          rcases witness with ⟨candidate, candidateExecuted, candidateMatch⟩
          rw [executed] at candidateExecuted
          have same : actual = candidate := Option.some.inj candidateExecuted
          subst candidate
          exact matching candidateMatch

/-! ## Proof-free group-certificate representation -/

/-- One normalized terminal with all proof fields and bounded-field witnesses
erased. Field coefficients are represented by their canonical natural
residues. -/
structure GroupTerminalShape where
  outputTerms : List (Nat × Nat)
  normalizedExpression :
    List (Materialized.QuadraticForm.Monomial × Nat)
deriving DecidableEq, Repr

def compactQuadraticTerm
    (term : Materialized.QuadraticForm.Term) :
    Materialized.QuadraticForm.Monomial × Nat :=
  (term.1, term.2.val)

def normalizedExpressionShape (expression : Form) :
    List (Materialized.QuadraticForm.Monomial × Nat) :=
  (Materialized.QuadraticForm.normalize expression).map compactQuadraticTerm

def actualTerminalShape (terminal : TerminalExpression) :
    GroupTerminalShape :=
  { outputTerms := terminal.output.programTerms
    normalizedExpression := normalizedExpressionShape terminal.expression }

def expectedTerminalShape (terminal : ExpectedTerminal) :
    GroupTerminalShape :=
  { outputTerms := [(terminal.outputColumn, 1)]
    normalizedExpression := normalizedExpressionShape terminal.expression }

/-- Compact input to native artifact checking. The expensive symbolic
execution happens while projecting the decoded group; the checked value
contains only proof-free natural-number data. -/
structure GroupMatchShape where
  actual : Option (List GroupTerminalShape)
  expected : List GroupTerminalShape
deriving DecidableEq, Repr

def groupMatchShape (steps : List DecodedRewriteStep)
    (expected : List ExpectedTerminal) : GroupMatchShape :=
  { actual := (executeGroup steps).map (List.map actualTerminalShape)
    expected := expected.map expectedTerminalShape }

def groupMatchShapeCheck (shape : GroupMatchShape) : Bool :=
  match shape.actual with
  | none => false
  | some actual => decide (actual = shape.expected)

def groupMatchShapesCheck (shapes : List GroupMatchShape) : Bool :=
  shapes.all groupMatchShapeCheck

private theorem compactQuadraticTerm_injective :
    Function.Injective compactQuadraticTerm := by
  intro left right equal
  apply Prod.ext
  · exact congrArg
      (fun value : Materialized.QuadraticForm.Monomial × Nat => value.1)
      equal
  · apply Fin.ext
    exact congrArg Prod.snd equal

private theorem compactQuadraticTerms_injective :
    Function.Injective (List.map compactQuadraticTerm) := by
  intro left right equal
  induction left generalizing right with
  | nil =>
      cases right with
      | nil => rfl
      | cons head tail => simp at equal
  | cons leftHead leftTail inductionHypothesis =>
      cases right with
      | nil => simp at equal
      | cons rightHead rightTail =>
          simp only [List.map_cons, List.cons.injEq] at equal
          have headEqual := compactQuadraticTerm_injective equal.1
          have tailEqual := inductionHypothesis equal.2
          subst rightHead
          subst rightTail
          rfl

private theorem terminalMatches_of_shape_eq
    {actual : TerminalExpression} {expected : ExpectedTerminal}
    (equal : actualTerminalShape actual = expectedTerminalShape expected) :
    TerminalMatches actual expected := by
  constructor
  · have outputEqual := congrArg GroupTerminalShape.outputTerms equal
    simpa only [actualTerminalShape, expectedTerminalShape] using outputEqual
  · unfold Materialized.QuadraticForm.Equivalent
    apply compactQuadraticTerms_injective
    have expressionEqual :=
      congrArg GroupTerminalShape.normalizedExpression equal
    simpa only [actualTerminalShape, expectedTerminalShape,
      normalizedExpressionShape] using expressionEqual

private theorem terminalsMatch_of_shapes_eq :
    ∀ {actual : List TerminalExpression}
      {expected : List ExpectedTerminal},
      actual.map actualTerminalShape = expected.map expectedTerminalShape →
        TerminalsMatch actual expected
  | [], [], _ => .nil
  | [], _ :: _, equal => by simp at equal
  | _ :: _, [], equal => by simp at equal
  | actualHead :: actualTail, expectedHead :: expectedTail, equal => by
      simp only [List.map_cons, List.cons.injEq] at equal
      exact .cons (terminalMatches_of_shape_eq equal.1)
        (terminalsMatch_of_shapes_eq equal.2)

/-- Generic kernel bridge from a compact, proof-free group certificate to the
original typed symbolic recurrence. -/
theorem groupMatches_of_shape_check_true
    {steps : List DecodedRewriteStep}
    {expected : List ExpectedTerminal}
    (checked :
      groupMatchShapeCheck (groupMatchShape steps expected) = true) :
    GroupMatches steps expected := by
  cases executed : executeGroup steps with
  | none =>
      simp [groupMatchShapeCheck, groupMatchShape, executed] at checked
  | some actual =>
      have shapesEqual :
          actual.map actualTerminalShape =
            expected.map expectedTerminalShape := by
        apply of_decide_eq_true
        simpa [groupMatchShapeCheck, groupMatchShape, executed] using checked
      exact ⟨actual, executed, terminalsMatch_of_shapes_eq shapesEqual⟩

def evaluationGroups : List (List DecodedRewriteStep) :=
  (List.range 49).map fun index =>
    (decodedRewriteSteps.drop (22 * index)).take 22

def productGroups : List (List DecodedRewriteStep) :=
  (List.range 86).map fun index =>
    (decodedRewriteSteps.drop (1078 + 2 * index)).take 2

abbrev EvaluationPair := List DecodedRewriteStep × EvalTrace

abbrev ProductPair := List DecodedRewriteStep × KMulTrace

def evaluationPairs : List EvaluationPair :=
  evaluationGroups.zip SourceSchedule.evaluationTraces

def productPairs : List ProductPair :=
  productGroups.zip SourceSchedule.productTraces

def evaluationPairShape (pair : EvaluationPair) : GroupMatchShape :=
  groupMatchShape pair.1 (evaluationExpected pair.2)

def productPairShape (pair : ProductPair) : GroupMatchShape :=
  groupMatchShape pair.1 (productExpected pair.2)

theorem evaluationPairsLengthExact : evaluationPairs.length = 49 := by
  simp [evaluationPairs, evaluationGroups,
    SourceSchedule.evaluation_trace_count]

theorem productPairsLengthExact : productPairs.length = 86 := by
  simp [productPairs, productGroups,
    SourceSchedule.product_trace_count]

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.QuadraticRefinement
