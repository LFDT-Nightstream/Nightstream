import Nightstream.Implementation.R1CS.Core.Program
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.SelectiveCompilerBridge

/-!
Kernel semantics for one exact source-definition block eliminated by a
combined-NC product-sum rewrite.

Owns: symbolic substitution of an ordered list of actual `Program.Definition`
values, evaluation preservation under `Definition.Holds`, coefficient-level
comparison with one actual terminal `.source` `DecodedRewriteStep`, and the
resulting `RewriteStepHolds` theorem.

Does not own: generated block membership, generated certificate truth,
derived-accumulator chain folding, source-row satisfaction, selected-row
satisfaction, transcript order, parent or child authority, commitment
binding, costs, or row removal.

The source interpreter is intentionally partial at multiplication: a product
is accepted only when symbolic substitution leaves both operands linear.
This is the exact degree-two lowering vocabulary used by product-sum
rewrites. A later bounded artifact leaf must show that each of the 1,493
concrete blocks takes this branch and that its normalized coefficients match.
No source-row satisfaction proposition is an input to this module.

Assurance tier: model-level.
-/

/-!
Emits constraints: none; this module states model-level source/rewrite semantics.

| Stable stage path | Obligation | Authority class |
|---|---|---|
| `f_prime.pi_ccs_nc.delayed.combined.rewrite_source.core` | Define the independent source-expression and rewrite-program relations. | computed semantics |

-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.RewriteSourceSemantics.Core

open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.Decoder
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.Semantics
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.SelectiveCompilerBridge

/-! ## Small normalized degree-two kernel -/

namespace Symbolic

abbrev LinearTerm := Nat × F
abbrev LinearForm := List LinearTerm

inductive Monomial where
  | linear (column : Nat)
  | quadratic (left right : Nat)
deriving DecidableEq, Repr

/-- Products have one canonical commutative orientation. -/
def Monomial.product (left right : Nat) : Monomial :=
  if left ≤ right then .quadratic left right else .quadratic right left

def Monomial.key : Monomial → Nat × Nat × Nat
  | .linear column => (0, column, 0)
  | .quadratic left right => (1, left, right)

def Monomial.Before (left right : Monomial) : Prop :=
  left.key.1 < right.key.1 ∨
    (left.key.1 = right.key.1 ∧
      (left.key.2.1 < right.key.2.1 ∨
        (left.key.2.1 = right.key.2.1 ∧
          left.key.2.2 < right.key.2.2)))

instance (left right : Monomial) : Decidable (left.Before right) := by
  unfold Monomial.Before
  infer_instance

abbrev Term := Monomial × F
abbrev Form := List Term

def linearEval (assignment : Nat → F) : LinearForm → F
  | [] => 0
  | term :: rest =>
      term.2 * assignment term.1 + linearEval assignment rest

def monomialValue (assignment : Nat → F) : Monomial → F
  | .linear column => assignment column
  | .quadratic left right => assignment left * assignment right

def termValue (assignment : Nat → F) (term : Term) : F :=
  term.2 * monomialValue assignment term.1

def eval (assignment : Nat → F) : Form → F
  | [] => 0
  | term :: rest => termValue assignment term + eval assignment rest

private theorem fadd_assoc (a b c : F) :
    (a + b) + c = a + (b + c) :=
  Lean.Grind.Fin.add_assoc _ _ _

private theorem fadd_comm (a b : F) : a + b = b + a :=
  Lean.Grind.Fin.add_comm _ _

private theorem fadd_left_comm (a b c : F) :
    a + (b + c) = b + (a + c) := by
  rw [← fadd_assoc, fadd_comm a b, fadd_assoc]

private theorem fadd_mul (a b c : F) :
    (a + b) * c = a * c + b * c := by
  calc
    (a + b) * c = c * (a + b) := Fin.mul_comm _ _
    _ = c * a + c * b := Lean.Grind.Fin.left_distrib _ _ _
    _ = a * c + b * c := by
      rw [Fin.mul_comm c a, Fin.mul_comm c b]

private theorem fmul_add (a b c : F) :
    a * (b + c) = a * b + a * c :=
  Lean.Grind.Fin.left_distrib _ _ _

def insert (term : Term) : Form → Form
  | [] => if term.2 = 0 then [] else [term]
  | head :: rest =>
      if term.2 = 0 then
        head :: rest
      else if term.1.Before head.1 then
        term :: head :: rest
      else if term.1 = head.1 then
        let coefficient := term.2 + head.2
        if coefficient = 0 then rest else (head.1, coefficient) :: rest
      else
        head :: insert term rest

theorem eval_insert (assignment : Nat → F) (term : Term) :
    ∀ terms, eval assignment (insert term terms) =
      termValue assignment term + eval assignment terms := by
  rcases term with ⟨termMonomial, termCoefficient⟩
  intro terms
  induction terms with
  | nil =>
      by_cases coefficientZero : termCoefficient = 0
      · simp [insert, coefficientZero, termValue, eval, Fin.zero_mul]
      · simp [insert, coefficientZero, termValue, eval]
  | cons head rest inductionHypothesis =>
      rcases head with ⟨headMonomial, headCoefficient⟩
      by_cases coefficientZero : termCoefficient = 0
      · simp [insert, coefficientZero, termValue, eval, Fin.zero_mul]
      · simp only [insert, coefficientZero, ↓reduceIte]
        by_cases before : termMonomial.Before headMonomial
        · simp only [before, ↓reduceIte, eval, termValue]
        · simp only [before, ↓reduceIte]
          by_cases same : termMonomial = headMonomial
          · subst headMonomial
            simp only [↓reduceIte, eval, termValue]
            by_cases sumZero : termCoefficient + headCoefficient = 0
            · simp only [sumZero, ↓reduceIte]
              have multiplied := congrArg
                (fun value : F =>
                  value * monomialValue assignment termMonomial) sumZero
              change
                (termCoefficient + headCoefficient) *
                    monomialValue assignment termMonomial =
                  0 * monomialValue assignment termMonomial at multiplied
              rw [fadd_mul, Fin.zero_mul] at multiplied
              rw [← fadd_assoc, multiplied, Fin.zero_add]
            · simp only [sumZero, ↓reduceIte, eval, termValue]
              rw [fadd_mul, fadd_assoc]
          · simp only [same, ↓reduceIte, eval, inductionHypothesis,
              termValue]
            exact fadd_left_comm _ _ _

def normalize (form : Form) : Form :=
  form.foldr insert []

theorem eval_normalize (assignment : Nat → F) (form : Form) :
    eval assignment (normalize form) = eval assignment form := by
  induction form with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      simp only [normalize, List.foldr_cons, eval_insert, eval]
      unfold normalize at inductionHypothesis
      rw [inductionHypothesis]

def Equivalent (left right : Form) : Prop :=
  normalize left = normalize right

instance (left right : Form) : Decidable (Equivalent left right) := by
  unfold Equivalent
  infer_instance

theorem eval_eq_of_equivalent {left right : Form}
    (equivalent : Equivalent left right) (assignment : Nat → F) :
    eval assignment left = eval assignment right := by
  rw [← eval_normalize assignment left,
    ← eval_normalize assignment right, equivalent]

def ofLinear (form : LinearForm) : Form :=
  form.map fun term => (.linear term.1, term.2)

def scale (coefficient : F) (form : Form) : Form :=
  form.map fun term => (term.1, coefficient * term.2)

def mulLinear (coefficient : F)
    (left right : LinearForm) : Form :=
  left.flatMap fun leftTerm =>
    right.map fun rightTerm =>
      (Monomial.product leftTerm.1 rightTerm.1,
        coefficient * leftTerm.2 * rightTerm.2)

theorem eval_append (assignment : Nat → F) (left right : Form) :
    eval assignment (left ++ right) =
      eval assignment left + eval assignment right := by
  induction left with
  | nil => simp [eval]
  | cons head tail inductionHypothesis =>
      simp only [List.cons_append, eval, inductionHypothesis]
      exact (fadd_assoc _ _ _).symm

theorem eval_scale (assignment : Nat → F) (coefficient : F)
    (form : Form) :
    eval assignment (scale coefficient form) =
      coefficient * eval assignment form := by
  induction form with
  | nil => exact (Fin.mul_zero coefficient).symm
  | cons head tail inductionHypothesis =>
      simp only [scale, List.map_cons, eval, termValue]
      unfold scale at inductionHypothesis
      rw [inductionHypothesis, Fin.mul_assoc, fmul_add]

theorem eval_ofLinear (assignment : Nat → F) (form : LinearForm) :
    eval assignment (ofLinear form) = linearEval assignment form := by
  induction form with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      simp only [ofLinear, List.map_cons, eval, termValue, monomialValue,
        linearEval]
      unfold ofLinear at inductionHypothesis
      rw [inductionHypothesis]

private theorem monomialValue_product (assignment : Nat → F)
    (left right : Nat) :
    monomialValue assignment (Monomial.product left right) =
      assignment left * assignment right := by
  unfold Monomial.product
  split
  · rfl
  · simp only [monomialValue]
    exact Fin.mul_comm _ _

private theorem eval_mulLinear_one (assignment : Nat → F)
    (coefficient : F) (leftTerm : LinearTerm) :
    ∀ right,
      eval assignment
          (right.map fun rightTerm =>
            (Monomial.product leftTerm.1 rightTerm.1,
              coefficient * leftTerm.2 * rightTerm.2)) =
        coefficient * (leftTerm.2 * assignment leftTerm.1) *
          linearEval assignment right := by
  intro right
  induction right with
  | nil => simp [eval, linearEval, Fin.mul_zero]
  | cons head tail inductionHypothesis =>
      simp only [List.map_cons, eval, termValue, monomialValue_product,
        linearEval]
      rw [inductionHypothesis]
      calc
        (coefficient * leftTerm.2 * head.2) *
              (assignment leftTerm.1 * assignment head.1) +
            coefficient * (leftTerm.2 * assignment leftTerm.1) *
              linearEval assignment tail =
            coefficient * (leftTerm.2 * assignment leftTerm.1) *
              (head.2 * assignment head.1) +
            coefficient * (leftTerm.2 * assignment leftTerm.1) *
              linearEval assignment tail := by
          congr 1
          ac_rfl
        _ = coefficient * (leftTerm.2 * assignment leftTerm.1) *
              ((head.2 * assignment head.1) +
                linearEval assignment tail) := by
          rw [← fmul_add]

theorem eval_mulLinear (assignment : Nat → F) (coefficient : F)
    (left right : LinearForm) :
    eval assignment (mulLinear coefficient left right) =
      coefficient * linearEval assignment left *
        linearEval assignment right := by
  induction left with
  | nil =>
      simp [mulLinear, eval, linearEval, Fin.zero_mul, Fin.mul_zero]
  | cons head tail inductionHypothesis =>
      simp only [mulLinear, List.flatMap_cons, eval_append, linearEval]
      unfold mulLinear at inductionHypothesis
      rw [eval_mulLinear_one, inductionHypothesis]
      calc
        coefficient * (head.2 * assignment head.1) *
              linearEval assignment right +
            coefficient * linearEval assignment tail *
              linearEval assignment right =
            coefficient *
              ((head.2 * assignment head.1) +
                linearEval assignment tail) *
              linearEval assignment right := by
          rw [fmul_add, fadd_mul]
        _ = _ := rfl

/-- Fail closed if a substituted product operand already has degree two. -/
def asLinear : Form → Option LinearForm
  | [] => some []
  | (.linear column, coefficient) :: rest => do
      pure ((column, coefficient) :: (← asLinear rest))
  | (.quadratic _ _, _) :: _ => none

theorem eval_asLinear {form : Form} {linear : LinearForm}
    (decoded : asLinear form = some linear) (assignment : Nat → F) :
    eval assignment form = linearEval assignment linear := by
  induction form generalizing linear with
  | nil =>
      simp [asLinear] at decoded
      subst linear
      rfl
  | cons head tail inductionHypothesis =>
      rcases head with ⟨monomial, coefficient⟩
      cases monomial with
      | quadratic left right => simp [asLinear] at decoded
      | linear column =>
          cases tailDecoded : asLinear tail with
          | none => simp [asLinear, tailDecoded] at decoded
          | some tailLinear =>
              simp [asLinear, tailDecoded] at decoded
              subst linear
              simp only [eval, termValue, monomialValue, linearEval]
              rw [inductionHypothesis tailDecoded]

end Symbolic

/-! ## Exact symbolic execution of `Program.Definition` -/

abbrev SymbolicState := Nat → Symbolic.Form

def fieldAssignment (assignment : Nat → Nat) : Nat → F :=
  fun column => fieldResidue (assignment column)

private theorem fieldResidue_add (left right : Nat) :
    fieldResidue (left + right) = fieldResidue left + fieldResidue right := by
  apply Fin.ext
  simp [fieldResidue, Fin.val_add, Nat.add_mod]

private theorem fieldResidue_mul (left right : Nat) :
    fieldResidue (left * right) = fieldResidue left * fieldResidue right := by
  apply Fin.ext
  simp [fieldResidue, Fin.val_mul, Nat.mul_mod]

def natLinearForm (terms : List (Nat × Nat)) : Symbolic.LinearForm :=
  terms.map fun term => (term.1, fieldResidue term.2)

private theorem evalNatLinearForm_raw (assignment : Nat → Nat)
    (terms : List (Nat × Nat)) :
    Symbolic.linearEval (fieldAssignment assignment) (natLinearForm terms) =
      fieldResidue (Program.rawLcEval assignment terms) := by
  induction terms with
  | nil => simp [Symbolic.linearEval, natLinearForm,
      Program.rawLcEval, fieldResidue]
  | cons head tail inductionHypothesis =>
      simp only [natLinearForm, List.map_cons, Symbolic.linearEval,
        Program.rawLcEval, fieldAssignment]
      change
        fieldResidue head.2 * fieldResidue (assignment head.1) +
            Symbolic.linearEval (fieldAssignment assignment)
              (natLinearForm tail) =
          fieldResidue
            (head.2 * assignment head.1 + Program.rawLcEval assignment tail)
      rw [inductionHypothesis, ← fieldResidue_mul,
        ← fieldResidue_add]

theorem evalNatLinearForm (assignment : Nat → Nat)
    (terms : List (Nat × Nat)) :
    Symbolic.linearEval (fieldAssignment assignment) (natLinearForm terms) =
      fieldResidue (lcEval assignment terms) := by
  have modulusEq : goldilocksP = goldilocksModulus := by rfl
  rw [evalNatLinearForm_raw, Program.lcEval_eq_raw_mod]
  apply Fin.ext
  simp [fieldResidue, modulusEq, Nat.mod_mod]

def variableState : SymbolicState :=
  fun column => Symbolic.ofLinear [(column, 1)]

def setExpression (state : SymbolicState) (column : Nat)
    (expression : Symbolic.Form) : SymbolicState :=
  fun candidate => if candidate = column then expression else state candidate

def substituteTerms (state : SymbolicState)
    (terms : List (Nat × Nat)) : Symbolic.Form :=
  terms.flatMap fun term =>
    Symbolic.scale (fieldResidue term.2) (state term.1)

def StateRepresents (assignment : Nat → Nat)
    (state : SymbolicState) : Prop :=
  ∀ column,
    Symbolic.eval (fieldAssignment assignment) (state column) =
      fieldResidue (assignment column)

theorem variableState_represents (assignment : Nat → Nat) :
    StateRepresents assignment variableState := by
  intro column
  simp [variableState, Symbolic.ofLinear, Symbolic.eval,
    Symbolic.termValue, Symbolic.monomialValue, fieldAssignment,
    Fin.one_mul]

private theorem evalSubstituteTerms_eq_natLinear
    (assignment : Nat → Nat) (state : SymbolicState)
    (represents : StateRepresents assignment state) :
    ∀ terms,
      Symbolic.eval (fieldAssignment assignment)
          (substituteTerms state terms) =
        Symbolic.linearEval (fieldAssignment assignment)
          (natLinearForm terms) := by
  intro terms
  induction terms with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      simp only [substituteTerms, List.flatMap_cons, natLinearForm,
        List.map_cons, Symbolic.linearEval]
      rw [Symbolic.eval_append, Symbolic.eval_scale]
      change
        fieldResidue head.2 *
              Symbolic.eval (fieldAssignment assignment) (state head.1) +
            Symbolic.eval (fieldAssignment assignment)
              (substituteTerms state tail) =
          fieldResidue head.2 * fieldResidue (assignment head.1) +
            Symbolic.linearEval (fieldAssignment assignment)
              (natLinearForm tail)
      rw [represents head.1, inductionHypothesis]

theorem evalSubstituteTerms
    (assignment : Nat → Nat) (state : SymbolicState)
    (represents : StateRepresents assignment state)
    (terms : List (Nat × Nat)) :
    Symbolic.eval (fieldAssignment assignment)
        (substituteTerms state terms) =
      fieldResidue (lcEval assignment terms) := by
  rw [evalSubstituteTerms_eq_natLinear assignment state represents terms]
  exact evalNatLinearForm assignment terms

def rhsExpression? (state : SymbolicState) :
    Program.Rhs → Option Symbolic.Form
  | .linear terms => some (substituteTerms state terms)
  | .product left right => do
      let leftLinear ← Symbolic.asLinear (substituteTerms state left)
      let rightLinear ← Symbolic.asLinear (substituteTerms state right)
      pure (Symbolic.mulLinear 1 leftLinear rightLinear)

private theorem fieldResidue_productEval
    (assignment : Nat → Nat) (left right : List (Nat × Nat)) :
    fieldResidue
        (Program.Rhs.eval assignment (.product left right)) =
      fieldResidue (lcEval assignment left) *
        fieldResidue (lcEval assignment right) := by
  have modulusEq : goldilocksP = goldilocksModulus := by rfl
  unfold Program.Rhs.eval
  rw [← fieldResidue_mul]
  apply Fin.ext
  simp [fieldResidue, modulusEq, Nat.mod_mod]

theorem eval_rhsExpression
    {assignment : Nat → Nat} {state : SymbolicState}
    (represents : StateRepresents assignment state)
    {rhs : Program.Rhs} {expression : Symbolic.Form}
    (decoded : rhsExpression? state rhs = some expression) :
    Symbolic.eval (fieldAssignment assignment) expression =
      fieldResidue (rhs.eval assignment) := by
  cases rhs with
  | linear terms =>
      simp only [rhsExpression?, Option.some.injEq] at decoded
      subst expression
      exact evalSubstituteTerms assignment state represents terms
  | product left right =>
      cases leftDecoded : Symbolic.asLinear
          (substituteTerms state left) with
      | none => simp [rhsExpression?, leftDecoded] at decoded
      | some leftLinear =>
          cases rightDecoded : Symbolic.asLinear
              (substituteTerms state right) with
          | none =>
              simp [rhsExpression?, leftDecoded, rightDecoded] at decoded
          | some rightLinear =>
              simp [rhsExpression?, leftDecoded, rightDecoded] at decoded
              subst expression
              rw [Symbolic.eval_mulLinear, Fin.one_mul,
                ← Symbolic.eval_asLinear leftDecoded,
                ← Symbolic.eval_asLinear rightDecoded,
                evalSubstituteTerms assignment state represents left,
                evalSubstituteTerms assignment state represents right]
              exact (fieldResidue_productEval assignment left right).symm

def executeDefinition? (state : SymbolicState)
    (definition : Program.Definition) : Option SymbolicState := do
  let expression ← rhsExpression? state definition.rhs
  pure (setExpression state definition.output expression)

def runDefinitions? :
    SymbolicState → List Program.Definition → Option SymbolicState
  | state, [] => some state
  | state, definition :: rest => do
      runDefinitions? (← executeDefinition? state definition) rest

private theorem executeDefinition_represents
    {assignment : Nat → Nat} {state next : SymbolicState}
    {definition : Program.Definition}
    (represents : StateRepresents assignment state)
    (holds : definition.Holds assignment)
    (executed : executeDefinition? state definition = some next) :
    StateRepresents assignment next := by
  cases expressionEq : rhsExpression? state definition.rhs with
  | none => simp [executeDefinition?, expressionEq] at executed
  | some expression =>
      have nextEq : next =
          setExpression state definition.output expression := by
        simpa [executeDefinition?, expressionEq] using executed.symm
      subst next
      intro column
      by_cases isOutput : column = definition.output
      · subst column
        simp only [setExpression, ↓reduceIte]
        have rhsValue := eval_rhsExpression represents expressionEq
        have holdsField := congrArg fieldResidue holds
        exact rhsValue.trans holdsField.symm
      · simp only [setExpression, isOutput, ↓reduceIte]
        exact represents column

/-- Symbolic execution agrees with any assignment satisfying exactly the
definitions in the supplied ordered block. This theorem assumes no R1CS row
or verifier acceptance proposition. -/
theorem runDefinitions_represents_of_holds
    {assignment : Nat → Nat} {initial final : SymbolicState}
    {definitions : List Program.Definition}
    (initialRepresents : StateRepresents assignment initial)
    (holds : ∀ definition ∈ definitions, definition.Holds assignment)
    (executed : runDefinitions? initial definitions = some final) :
    StateRepresents assignment final := by
  induction definitions generalizing initial with
  | nil =>
      simp [runDefinitions?] at executed
      subst final
      exact initialRepresents
  | cons definition rest inductionHypothesis =>
      cases nextEq : executeDefinition? initial definition with
      | none => simp [runDefinitions?, nextEq] at executed
      | some next =>
          apply inductionHypothesis
          · exact executeDefinition_represents initialRepresents
              (holds definition (by simp)) nextEq
          · intro candidate member
            exact holds candidate (by simp [member])
          · simpa [runDefinitions?, nextEq] using executed

/-! ## One terminal decoded product-sum recurrence -/

def decodedLinearExpression {columns : Nat}
    (state : SymbolicState)
    (linear : DecodedLinearCombination columns) : Symbolic.Form :=
  substituteTerms state (linearCombinationTerms linear)

theorem eval_decodedLinearExpression
    {columns : Nat} {assignment : Nat → Nat} {state : SymbolicState}
    (represents : StateRepresents assignment state)
    (linear : DecodedLinearCombination columns) :
    Symbolic.eval (fieldAssignment assignment)
        (decodedLinearExpression state linear) =
      linearCombinationValue linear assignment := by
  exact evalSubstituteTerms assignment state represents
    (linearCombinationTerms linear)

def factorExpression? {columns : Nat} (state : SymbolicState)
    (factor : DecodedProductFactor columns) : Option Symbolic.Form := do
  let left ← Symbolic.asLinear
    (decodedLinearExpression state factor.left)
  let right ← Symbolic.asLinear
    (decodedLinearExpression state factor.right)
  pure (Symbolic.mulLinear factor.coefficient left right)

theorem eval_factorExpression
    {columns : Nat} {assignment : Nat → Nat} {state : SymbolicState}
    (represents : StateRepresents assignment state)
    {factor : DecodedProductFactor columns}
    {expression : Symbolic.Form}
    (decoded : factorExpression? state factor = some expression) :
    Symbolic.eval (fieldAssignment assignment) expression =
      productFactorValue factor assignment := by
  cases leftEq : Symbolic.asLinear
      (decodedLinearExpression state factor.left) with
  | none => simp [factorExpression?, leftEq] at decoded
  | some left =>
      cases rightEq : Symbolic.asLinear
          (decodedLinearExpression state factor.right) with
      | none => simp [factorExpression?, leftEq, rightEq] at decoded
      | some right =>
          simp [factorExpression?, leftEq, rightEq] at decoded
          subst expression
          rw [Symbolic.eval_mulLinear,
            ← Symbolic.eval_asLinear leftEq,
            ← Symbolic.eval_asLinear rightEq,
            eval_decodedLinearExpression represents,
            eval_decodedLinearExpression represents]
          rfl

def factorsExpression? {columns : Nat} (state : SymbolicState) :
    List (DecodedProductFactor columns) → Option Symbolic.Form
  | [] => some []
  | factor :: rest => do
      let factorExpression ← factorExpression? state factor
      let restExpression ← factorsExpression? state rest
      pure (factorExpression ++ restExpression)

theorem eval_factorsExpression
    {columns : Nat} {assignment : Nat → Nat} {state : SymbolicState}
    (represents : StateRepresents assignment state) :
    ∀ {factors : List (DecodedProductFactor columns)}
      {expression : Symbolic.Form},
      factorsExpression? state factors = some expression →
      Symbolic.eval (fieldAssignment assignment) expression =
        factors.foldr
          (fun factor suffix => productFactorValue factor assignment + suffix)
          0 := by
  intro factors
  induction factors with
  | nil =>
      intro expression decoded
      simp [factorsExpression?] at decoded
      subst expression
      rfl
  | cons factor rest inductionHypothesis =>
      intro expression decoded
      cases factorEq : factorExpression? state factor with
      | none => simp [factorsExpression?, factorEq] at decoded
      | some factorValue =>
          cases restEq : factorsExpression? state rest with
          | none =>
              simp [factorsExpression?, factorEq, restEq] at decoded
          | some restValue =>
              simp [factorsExpression?, factorEq, restEq] at decoded
              subst expression
              change
                Symbolic.eval (fieldAssignment assignment)
                    (factorValue ++ restValue) =
                  productFactorValue factor assignment +
                    rest.foldr
                      (fun candidate suffix =>
                        productFactorValue candidate assignment + suffix) 0
              rw [Symbolic.eval_append,
                eval_factorExpression represents factorEq,
                inductionHypothesis restEq]

private theorem factorFold_eq_factorSum
    {columns : Nat} (assignment : Nat → Nat)
    (factors : List (DecodedProductFactor columns))
    (capacity : factors.length ≤ 5) :
    factors.foldr
        (fun factor suffix => productFactorValue factor assignment + suffix) 0 =
      factorSum assignment factors := by
  cases factors with
  | nil => rfl
  | cons first rest =>
      cases rest with
      | nil => simp [factorSum, factorValueAt]
      | cons second rest =>
          cases rest with
          | nil => simp [factorSum, factorValueAt]
          | cons third rest =>
              cases rest with
              | nil =>
                  simp [factorSum, factorValueAt,
                    Lean.Grind.Fin.add_assoc]
              | cons fourth rest =>
                  cases rest with
                  | nil =>
                      simp [factorSum, factorValueAt,
                        Lean.Grind.Fin.add_assoc]
                  | cons fifth rest =>
                      cases rest with
                      | nil =>
                          simp [factorSum, factorValueAt,
                            Lean.Grind.Fin.add_assoc]
                      | cons sixth rest => simp at capacity

/-- The predecessor expression is supplied by the separately owned decoded
rewrite-chain fold. This leaf only checks the terminal source block. -/
def recurrenceExpression? {columns : Nat} (state : SymbolicState)
    (previous : Symbolic.Form) (step : DecodedRewriteStep columns) :
    Option Symbolic.Form := do
  let factors ← factorsExpression? state step.factors
  pure (decodedLinearExpression state step.base ++ previous ++ factors)

theorem eval_recurrenceExpression
    {columns : Nat} {assignment : Nat → Nat} {state : SymbolicState}
    (represents : StateRepresents assignment state)
    (previous : Symbolic.Form) (step : DecodedRewriteStep columns)
    (capacity : step.factors.length ≤ 5)
    {expression : Symbolic.Form}
    (decoded : recurrenceExpression? state previous step = some expression) :
    Symbolic.eval (fieldAssignment assignment) expression =
      linearCombinationValue step.base assignment +
        Symbolic.eval (fieldAssignment assignment) previous +
        factorSum assignment step.factors := by
  cases factorsEq : factorsExpression? state step.factors with
  | none => simp [recurrenceExpression?, factorsEq] at decoded
  | some factors =>
      simp [recurrenceExpression?, factorsEq] at decoded
      subst expression
      rw [Symbolic.eval_append, Symbolic.eval_append,
        eval_decodedLinearExpression represents,
        eval_factorsExpression represents factorsEq,
        factorFold_eq_factorSum assignment step.factors capacity]
      exact (Lean.Grind.Fin.add_assoc _ _ _).symm

/-- Executable coefficient certificate for one exact ordered source block.
The block itself is the supplied list of actual `Program.Definition` values.
No label, row range, stage name, or caller-provided semantic proposition is
part of the match. -/
def ExactBlockMatch {columns : Nat}
    (definitions : List Program.Definition)
    (step : DecodedRewriteStep columns)
    (previous : Symbolic.Form) : Prop :=
  match runDefinitions? variableState definitions with
  | none => False
  | some state =>
      match step.output with
      | .derivedProductSum _ => False
      | .source output =>
          match recurrenceExpression? state previous step with
          | none => False
          | some recurrence =>
              step.factors.length ≤ 5 ∧
                Symbolic.Equivalent
                  (decodedLinearExpression state output) recurrence

instance {columns : Nat} (definitions : List Program.Definition)
    (step : DecodedRewriteStep columns) (previous : Symbolic.Form) :
    Decidable (ExactBlockMatch definitions step previous) := by
  unfold ExactBlockMatch
  split <;> try infer_instance
  split <;> try infer_instance
  split <;> infer_instance

/-- Headline source-to-rewrite theorem. Exact coefficient matching and truth
of the exact decoded definitions force the terminal `.source` recurrence.
The only external semantic input is the value of the predecessor expression,
which is produced by the separately owned derived-accumulator chain fold. -/
theorem exactBlockMatch_implies_terminalSourceRecurrence
    {columns : Nat} {definitions : List Program.Definition}
    {step : DecodedRewriteStep columns} {previous : Symbolic.Form}
    {assignment : Nat → Nat} {derivedValue : Nat → F}
    (matching : ExactBlockMatch definitions step previous)
    (definitionsHold : ∀ definition ∈ definitions,
      definition.Holds assignment)
    (previousValue :
      Symbolic.eval (fieldAssignment assignment) previous =
        rewritePreviousValue derivedValue step.previous) :
    RewriteStepHolds assignment derivedValue step := by
  cases stateEq : runDefinitions? variableState definitions with
  | none => simp [ExactBlockMatch, stateEq] at matching
  | some state =>
      have represents : StateRepresents assignment state :=
        runDefinitions_represents_of_holds
          (variableState_represents assignment) definitionsHold stateEq
      cases outputEq : step.output with
      | derivedProductSum compilerIndex =>
          simp [ExactBlockMatch, stateEq, outputEq] at matching
      | source output =>
          cases recurrenceEq : recurrenceExpression? state previous step with
          | none =>
              simp [ExactBlockMatch, stateEq, outputEq, recurrenceEq]
                at matching
          | some recurrence =>
              have matched :
                  step.factors.length ≤ 5 ∧
                    Symbolic.Equivalent
                      (decodedLinearExpression state output) recurrence := by
                simpa [ExactBlockMatch, stateEq, outputEq, recurrenceEq]
                  using matching
              constructor
              · exact matched.1
              · simp only [rewriteOutputValue, outputEq]
                calc
                  linearCombinationValue output assignment =
                      Symbolic.eval (fieldAssignment assignment)
                        (decodedLinearExpression state output) :=
                    (eval_decodedLinearExpression represents output).symm
                  _ = Symbolic.eval (fieldAssignment assignment)
                        recurrence :=
                    Symbolic.eval_eq_of_equivalent matched.2 _
                  _ = linearCombinationValue step.base assignment +
                        Symbolic.eval (fieldAssignment assignment) previous +
                        factorSum assignment step.factors :=
                    eval_recurrenceExpression represents previous step
                      matched.1 recurrenceEq
                  _ = linearCombinationValue step.base assignment +
                        rewritePreviousValue derivedValue step.previous +
                        factorSum assignment step.factors := by
                    rw [previousValue]

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.RewriteSourceSemantics.Core
