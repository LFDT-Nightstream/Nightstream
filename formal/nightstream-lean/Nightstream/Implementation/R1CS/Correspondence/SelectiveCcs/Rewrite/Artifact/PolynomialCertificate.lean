import Mathlib.Data.ZMod.Basic
import Mathlib.Tactic.Ring
import Nightstream.Implementation.R1CS.Correspondence.SelectiveCcs.Rewrite.Artifact.SourceRelation

/-!
Contract: executable degree-two polynomial certificates for grouped-product
artifact refinement.

Assurance tier: model-level certificate kernel.

Owns: a proof-free canonical polynomial list, executable normalization, and
evaluation soundness for decoded source rows and grouped-product identities.

Does not own: a generated fixture, Rust conformance, production-family
coverage, row removal, or construction of eliminated source witnesses.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.Artifact.PolynomialCertificate

open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.Artifact.Decoder
open Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.Artifact.SourceImage
open Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.Artifact.SourceRelation

/-- Standard ring used only to state and prove evaluation soundness. The
certificate data and normalization remain executable `F` values. -/
abbrev CertificateField := ZMod goldilocksModulus

def certificateValue (value : F) : CertificateField :=
  ZMod.finEquiv goldilocksModulus value

@[simp] theorem certificateValue_zero : certificateValue 0 = 0 :=
  (ZMod.finEquiv goldilocksModulus).map_zero

@[simp] theorem certificateValue_one : certificateValue 1 = 1 := by
  rfl

@[simp] theorem certificateValue_add (left right : F) :
    certificateValue (left + right) =
      certificateValue left + certificateValue right :=
  (ZMod.finEquiv goldilocksModulus).map_add left right

@[simp] theorem certificateValue_mul (left right : F) :
    certificateValue (left * right) =
      certificateValue left * certificateValue right :=
  (ZMod.finEquiv goldilocksModulus).map_mul left right

@[simp] theorem certificateValue_neg (value : F) :
    certificateValue (-value) = -certificateValue value :=
  (ZMod.finEquiv goldilocksModulus).map_neg value

@[simp] theorem certificateValue_sub (left right : F) :
    certificateValue (left - right) =
      certificateValue left - certificateValue right :=
  (ZMod.finEquiv goldilocksModulus).map_sub left right

/-- Degree-at-most-two monomial. Product coordinates are stored in ascending
order by `productMonomial`. -/
inductive Monomial (columns : Nat) where
  | one
  | variable (column : Fin columns)
  | product (left right : Fin columns)
deriving DecidableEq, Repr

structure Term (columns : Nat) where
  monomial : Monomial columns
  coefficient : F
deriving DecidableEq, Repr

abbrev Polynomial (columns : Nat) := List (Term columns)

namespace Monomial

/-- Deterministic strict order used by the executable insertion normalizer. -/
def less {columns : Nat} : Monomial columns → Monomial columns → Bool
  | .one, .one => false
  | .one, _ => true
  | .variable _, .one => false
  | .variable left, .variable right => left.val < right.val
  | .variable _, .product _ _ => true
  | .product _ _, .one => false
  | .product _ _, .variable _ => false
  | .product left0 left1, .product right0 right1 =>
      left0.val < right0.val ||
        (left0.val = right0.val && left1.val < right1.val)

/-- Canonical commutative product of two source variables. -/
def productMonomial {columns : Nat}
    (left right : Fin columns) : Monomial columns :=
  if left.val ≤ right.val then .product left right else .product right left

end Monomial

/-- Insert one term, combine an equal monomial, and remove zero coefficients. -/
def insertTerm {columns : Nat} (term : Term columns) :
    Polynomial columns → Polynomial columns
  | [] => if term.coefficient = 0 then [] else [term]
  | head :: tail =>
      if term.coefficient = 0 then
        head :: tail
      else if term.monomial = head.monomial then
        let coefficient := term.coefficient + head.coefficient
        if coefficient = 0 then tail
        else { monomial := head.monomial, coefficient } :: tail
      else if term.monomial.less head.monomial then
        term :: head :: tail
      else
        head :: insertTerm term tail

/-- Canonicalize an arbitrary term list. -/
def normalize {columns : Nat} (terms : List (Term columns)) :
    Polynomial columns :=
  terms.foldl (fun polynomial term => insertTerm term polynomial) []

def add {columns : Nat}
    (left right : Polynomial columns) : Polynomial columns :=
  normalize (left ++ right)

def scale {columns : Nat}
    (coefficient : F) (polynomial : Polynomial columns) : Polynomial columns :=
  normalize (polynomial.map fun term =>
    { term with coefficient := coefficient * term.coefficient })

def sub {columns : Nat}
    (left right : Polynomial columns) : Polynomial columns :=
  add left (scale (-1) right)

def monomialValue {columns : Nat}
    (assignment : Fin columns → F) : Monomial columns → CertificateField
  | .one => 1
  | .variable column => certificateValue (assignment column)
  | .product left right =>
      certificateValue (assignment left) * certificateValue (assignment right)

def termValue {columns : Nat}
    (assignment : Fin columns → F) (term : Term columns) : CertificateField :=
  certificateValue term.coefficient * monomialValue assignment term.monomial

def evaluate {columns : Nat}
    (assignment : Fin columns → F) : Polynomial columns → CertificateField
  | [] => 0
  | term :: tail => termValue assignment term + evaluate assignment tail

theorem evaluate_insertTerm {columns : Nat}
    (term : Term columns) (polynomial : Polynomial columns)
    (assignment : Fin columns → F) :
    evaluate assignment (insertTerm term polynomial) =
      termValue assignment term + evaluate assignment polynomial := by
  induction polynomial with
  | nil =>
      simp only [insertTerm]
      split
      · rename_i zero
        simp [evaluate, termValue, zero]
      · simp [evaluate]
  | cons head tail inductionHypothesis =>
      simp only [insertTerm]
      by_cases termZero : term.coefficient = 0
      · simp [termZero, evaluate, termValue]
      · simp only [termZero, if_false]
        by_cases same : term.monomial = head.monomial
        · simp only [same, if_true]
          by_cases cancel : term.coefficient + head.coefficient = 0
          · simp only [cancel, if_true, evaluate]
            have valuesEqual :
                monomialValue assignment term.monomial =
                  monomialValue assignment head.monomial := by
              rw [same]
            have mappedCancel :
                certificateValue term.coefficient +
                    certificateValue head.coefficient = 0 := by
              have mapped := congrArg certificateValue cancel
              simpa only [certificateValue_add, certificateValue_zero] using mapped
            simp only [termValue, valuesEqual]
            calc
              evaluate assignment tail =
                  (certificateValue term.coefficient +
                      certificateValue head.coefficient) *
                      monomialValue assignment head.monomial +
                    evaluate assignment tail := by rw [mappedCancel]; ring
              _ = certificateValue term.coefficient *
                      monomialValue assignment head.monomial +
                    (certificateValue head.coefficient *
                        monomialValue assignment head.monomial +
                      evaluate assignment tail) := by ring
          · simp only [cancel, if_false, evaluate]
            have valuesEqual :
                monomialValue assignment term.monomial =
                  monomialValue assignment head.monomial := by
              rw [same]
            simp only [termValue, certificateValue_add, valuesEqual]
            ring
        · simp only [same, if_false]
          by_cases lower : term.monomial.less head.monomial
          · simp [lower, evaluate]
          · simp only [lower, Bool.false_eq_true, if_false, evaluate,
              inductionHypothesis]
            ring

private theorem evaluate_foldl_insert {columns : Nat}
    (terms : List (Term columns)) (initial : Polynomial columns)
    (assignment : Fin columns → F) :
    evaluate assignment
        (terms.foldl (fun polynomial term => insertTerm term polynomial) initial) =
      evaluate assignment initial + evaluate assignment terms := by
  induction terms generalizing initial with
  | nil => simp [evaluate]
  | cons head tail inductionHypothesis =>
      rw [List.foldl_cons, inductionHypothesis]
      rw [evaluate_insertTerm]
      simp only [evaluate]
      ring

theorem evaluate_normalize {columns : Nat}
    (terms : List (Term columns)) (assignment : Fin columns → F) :
    evaluate assignment (normalize terms) = evaluate assignment terms := by
  unfold normalize
  rw [evaluate_foldl_insert]
  simp [evaluate]

private theorem evaluate_append {columns : Nat}
    (left right : Polynomial columns) (assignment : Fin columns → F) :
    evaluate assignment (left ++ right) =
      evaluate assignment left + evaluate assignment right := by
  induction left with
  | nil => simp [evaluate]
  | cons head tail inductionHypothesis =>
      simp only [List.cons_append, evaluate, inductionHypothesis]
      ring

theorem evaluate_add {columns : Nat}
    (left right : Polynomial columns) (assignment : Fin columns → F) :
    evaluate assignment (add left right) =
      evaluate assignment left + evaluate assignment right := by
  rw [add, evaluate_normalize, evaluate_append]

private theorem evaluate_map_scale {columns : Nat}
    (coefficient : F) (polynomial : Polynomial columns)
    (assignment : Fin columns → F) :
    evaluate assignment
        (polynomial.map fun term =>
          { term with coefficient := coefficient * term.coefficient }) =
      certificateValue coefficient * evaluate assignment polynomial := by
  induction polynomial with
  | nil => simp [evaluate]
  | cons head tail inductionHypothesis =>
      simp only [List.map_cons, evaluate, termValue, certificateValue_mul,
        inductionHypothesis]
      ring

theorem evaluate_scale {columns : Nat}
    (coefficient : F) (polynomial : Polynomial columns)
    (assignment : Fin columns → F) :
    evaluate assignment (scale coefficient polynomial) =
      certificateValue coefficient * evaluate assignment polynomial := by
  rw [scale, evaluate_normalize, evaluate_map_scale]

theorem evaluate_sub {columns : Nat}
    (left right : Polynomial columns) (assignment : Fin columns → F) :
    evaluate assignment (sub left right) =
      evaluate assignment left - evaluate assignment right := by
  rw [sub, evaluate_add, evaluate_scale, certificateValue_neg,
    certificateValue_one]
  ring

/-- Degree-at-most-one syntax. Its separate type makes quadratic overflow
impossible when the kernel forms one R1CS product. -/
inductive AffineMonomial (columns : Nat) where
  | one
  | variable (column : Fin columns)
deriving DecidableEq, Repr

structure AffineTerm (columns : Nat) where
  monomial : AffineMonomial columns
  coefficient : F
deriving DecidableEq, Repr

def affineMonomialValue {columns : Nat}
    (assignment : Fin columns → F) :
    AffineMonomial columns → CertificateField
  | .one => 1
  | .variable column => certificateValue (assignment column)

def affineTermValue {columns : Nat}
    (assignment : Fin columns → F) (term : AffineTerm columns) :
    CertificateField :=
  certificateValue term.coefficient *
    affineMonomialValue assignment term.monomial

def evaluateAffine {columns : Nat}
    (assignment : Fin columns → F) :
    List (AffineTerm columns) → CertificateField
  | [] => 0
  | term :: tail =>
      affineTermValue assignment term + evaluateAffine assignment tail

def sourceAffineTerm {columns : Nat}
    (term : DecodedSourceTerm columns) : AffineTerm columns :=
  { monomial := .variable term.column
    coefficient := term.coefficient }

def affineTerms {columns : Nat}
    (value : DecodedSourceLinearCombination columns) :
    List (AffineTerm columns) :=
  { monomial := .one, coefficient := value.constant } ::
    value.terms.map sourceAffineTerm

def AffineMonomial.toMonomial {columns : Nat} :
    AffineMonomial columns → Monomial columns
  | .one => .one
  | .variable column => .variable column

def AffineTerm.toTerm {columns : Nat}
    (term : AffineTerm columns) : Term columns :=
  { monomial := term.monomial.toMonomial
    coefficient := term.coefficient }

@[simp] theorem monomialValue_toMonomial {columns : Nat}
    (assignment : Fin columns → F) (monomial : AffineMonomial columns) :
    monomialValue assignment monomial.toMonomial =
      affineMonomialValue assignment monomial := by
  cases monomial <;> rfl

@[simp] theorem termValue_toTerm {columns : Nat}
    (assignment : Fin columns → F) (term : AffineTerm columns) :
    termValue assignment term.toTerm = affineTermValue assignment term := by
  simp [termValue, affineTermValue, AffineTerm.toTerm]

private theorem evaluate_map_toTerm {columns : Nat}
    (terms : List (AffineTerm columns)) (assignment : Fin columns → F) :
    evaluate assignment (terms.map AffineTerm.toTerm) =
      evaluateAffine assignment terms := by
  induction terms with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      simp only [List.map_cons, evaluate, evaluateAffine, termValue_toTerm,
        inductionHypothesis]

private theorem certificate_foldl_sourceTerms {columns : Nat}
    (terms : List (DecodedSourceTerm columns))
    (assignment : Fin columns → F) (initial : F) :
    certificateValue
        (terms.foldl
          (fun total term =>
            total + term.coefficient * assignment term.column)
          initial) =
      certificateValue initial +
        evaluateAffine assignment (terms.map sourceAffineTerm) := by
  induction terms generalizing initial with
  | nil => simp [evaluateAffine]
  | cons head tail inductionHypothesis =>
      rw [List.foldl_cons, inductionHypothesis]
      simp only [List.map_cons, evaluateAffine, affineTermValue,
        sourceAffineTerm, affineMonomialValue, certificateValue_add,
        certificateValue_mul]
      ring

theorem evaluateAffine_affineTerms {columns : Nat}
    (value : DecodedSourceLinearCombination columns)
    (assignment : Fin columns → F) :
    evaluateAffine assignment (affineTerms value) =
      certificateValue (linearValue value assignment 1) := by
  unfold affineTerms linearValue directSourceLinearValue
  rw [certificate_foldl_sourceTerms]
  simp only [evaluateAffine, affineTermValue, affineMonomialValue,
    certificateValue_mul, certificateValue_one]

/-- Canonical polynomial for one decoded affine expression. -/
def linearPolynomial {columns : Nat}
    (value : DecodedSourceLinearCombination columns) : Polynomial columns :=
  normalize ((affineTerms value).map AffineTerm.toTerm)

theorem evaluate_linearPolynomial {columns : Nat}
    (value : DecodedSourceLinearCombination columns)
    (assignment : Fin columns → F) :
    evaluate assignment (linearPolynomial value) =
      certificateValue (linearValue value assignment 1) := by
  rw [linearPolynomial, evaluate_normalize, evaluate_map_toTerm,
    evaluateAffine_affineTerms]

/-- Total multiplication from two affine monomials into the degree-two
normal form. -/
def multiplyAffineMonomial {columns : Nat} :
    AffineMonomial columns → AffineMonomial columns → Monomial columns
  | .one, .one => .one
  | .one, .variable column => .variable column
  | .variable column, .one => .variable column
  | .variable left, .variable right => Monomial.productMonomial left right

def multiplyAffineTerm {columns : Nat}
    (left right : AffineTerm columns) : Term columns :=
  { monomial := multiplyAffineMonomial left.monomial right.monomial
    coefficient := left.coefficient * right.coefficient }

def productTerms {columns : Nat}
    (left right : List (AffineTerm columns)) : List (Term columns) :=
  left.flatMap fun leftTerm => right.map (multiplyAffineTerm leftTerm)

theorem monomialValue_multiplyAffineMonomial {columns : Nat}
    (left right : AffineMonomial columns)
    (assignment : Fin columns → F) :
    monomialValue assignment (multiplyAffineMonomial left right) =
      affineMonomialValue assignment left *
        affineMonomialValue assignment right := by
  cases left <;> cases right
  · simp [multiplyAffineMonomial, monomialValue, affineMonomialValue]
  · simp [multiplyAffineMonomial, monomialValue, affineMonomialValue]
  · simp [multiplyAffineMonomial, monomialValue, affineMonomialValue]
  · simp only [multiplyAffineMonomial, Monomial.productMonomial]
    split
    · simp [monomialValue, affineMonomialValue]
    · simp [monomialValue, affineMonomialValue]
      ring

theorem termValue_multiplyAffineTerm {columns : Nat}
    (left right : AffineTerm columns)
    (assignment : Fin columns → F) :
    termValue assignment (multiplyAffineTerm left right) =
      affineTermValue assignment left * affineTermValue assignment right := by
  simp only [termValue, multiplyAffineTerm, certificateValue_mul,
    monomialValue_multiplyAffineMonomial, affineTermValue]
  ring

private theorem evaluate_map_multiplyAffineTerm {columns : Nat}
    (left : AffineTerm columns) (right : List (AffineTerm columns))
    (assignment : Fin columns → F) :
    evaluate assignment (right.map (multiplyAffineTerm left)) =
      affineTermValue assignment left * evaluateAffine assignment right := by
  induction right with
  | nil => simp [evaluate, evaluateAffine]
  | cons head tail inductionHypothesis =>
      simp only [List.map_cons, evaluate, evaluateAffine,
        termValue_multiplyAffineTerm, inductionHypothesis]
      ring

private theorem evaluate_productTerms {columns : Nat}
    (left right : List (AffineTerm columns))
    (assignment : Fin columns → F) :
    evaluate assignment (productTerms left right) =
      evaluateAffine assignment left * evaluateAffine assignment right := by
  induction left with
  | nil => simp [productTerms, evaluate, evaluateAffine]
  | cons head tail inductionHypothesis =>
      change evaluate assignment
          (right.map (multiplyAffineTerm head) ++ productTerms tail right) = _
      rw [evaluate_append, evaluate_map_multiplyAffineTerm,
        inductionHypothesis]
      simp only [evaluateAffine]
      ring

/-- Canonical product of two decoded affine expressions. -/
def productPolynomial {columns : Nat}
    (left right : DecodedSourceLinearCombination columns) :
    Polynomial columns :=
  normalize (productTerms (affineTerms left) (affineTerms right))

theorem evaluate_productPolynomial {columns : Nat}
    (left right : DecodedSourceLinearCombination columns)
    (assignment : Fin columns → F) :
    evaluate assignment (productPolynomial left right) =
      certificateValue
        (linearValue left assignment 1 * linearValue right assignment 1) := by
  rw [productPolynomial, evaluate_normalize, evaluate_productTerms,
    evaluateAffine_affineTerms, evaluateAffine_affineTerms,
    certificateValue_mul]

/-- Canonical residual polynomial of one decoded source R1CS row. -/
def rowPolynomial {rows columns : Nat}
    (row : DecodedSourceR1csRow rows columns) : Polynomial columns :=
  sub (productPolynomial row.a row.b) (linearPolynomial row.c)

theorem evaluate_rowPolynomial {rows columns : Nat}
    (row : DecodedSourceR1csRow rows columns)
    (assignment : Fin columns → F) :
    evaluate assignment (rowPolynomial row) =
      certificateValue (rowResidual row assignment 1) := by
  simp only [rowPolynomial, rowResidual, evaluate_sub,
    evaluate_productPolynomial, evaluate_linearPolynomial,
    certificateValue_sub]

/-- Canonical residual polynomial for one scaled grouped product. -/
def factorPolynomial {columns : Nat}
    (factor : DecodedFactor columns) : Polynomial columns :=
  scale factor.coefficient
    (productPolynomial factor.left factor.right)

theorem evaluate_factorPolynomial {columns : Nat}
    (factor : DecodedFactor columns)
    (assignment : Fin columns → F) :
    evaluate assignment (factorPolynomial factor) =
      certificateValue (factorValue factor assignment 1) := by
  simp only [factorPolynomial, factorValue, evaluate_scale,
    evaluate_productPolynomial, certificateValue_mul]
  ring

def factorSumPolynomial {columns : Nat} :
    List (DecodedFactor columns) → Polynomial columns
  | [] => []
  | factor :: tail =>
      add (factorPolynomial factor) (factorSumPolynomial tail)

theorem evaluate_factorSumPolynomial {columns : Nat}
    (factors : List (DecodedFactor columns))
    (assignment : Fin columns → F) :
    evaluate assignment (factorSumPolynomial factors) =
      certificateValue (factorSum assignment 1 factors) := by
  induction factors with
  | nil => simp [factorSumPolynomial, factorSum, evaluate]
  | cons head tail inductionHypothesis =>
      simp only [factorSumPolynomial, factorSum, evaluate_add,
        evaluate_factorPolynomial, inductionHypothesis, certificateValue_add]

def sourceOutputPolynomial {columns : Nat} :
    DecodedOutput columns → Polynomial columns
  | .source value => linearPolynomial value
  | .derivedProductSum _ => []

theorem evaluate_sourceOutputPolynomial {columns : Nat}
    (output : DecodedOutput columns) (assignment : Fin columns → F) :
    evaluate assignment (sourceOutputPolynomial output) =
      certificateValue (outputValue assignment 1 (fun _ => 0) output) := by
  cases output <;>
    simp [sourceOutputPolynomial, outputValue, evaluate,
      evaluate_linearPolynomial]

def contributionPolynomial {rows columns : Nat}
    (step : DecodedStep rows columns) : Polynomial columns :=
  add (linearPolynomial step.base) (factorSumPolynomial step.factors)

theorem evaluate_contributionPolynomial {rows columns : Nat}
    (step : DecodedStep rows columns) (assignment : Fin columns → F) :
    evaluate assignment (contributionPolynomial step) =
      certificateValue
        (linearValue step.base assignment 1 +
          factorSum assignment 1 step.factors) := by
  simp only [contributionPolynomial, evaluate_add,
    evaluate_linearPolynomial, evaluate_factorSumPolynomial,
    certificateValue_add]

/-- Canonical residual after the private carry between two steps is
eliminated. -/
def identityPolynomial {rows columns : Nat}
    (first second : DecodedStep rows columns) : Polynomial columns :=
  sub (sourceOutputPolynomial second.output)
    (add (contributionPolynomial first) (contributionPolynomial second))

theorem evaluate_identityPolynomial {rows columns : Nat}
    (first second : DecodedStep rows columns)
    (assignment : Fin columns → F) :
    evaluate assignment (identityPolynomial first second) =
      certificateValue
        (outputValue assignment 1 (fun _ => 0) second.output -
          ((linearValue first.base assignment 1 +
              factorSum assignment 1 first.factors) +
            (linearValue second.base assignment 1 +
              factorSum assignment 1 second.factors))) := by
  simp only [identityPolynomial, evaluate_sub, evaluate_add,
    evaluate_sourceOutputPolynomial, evaluate_contributionPolynomial,
    certificateValue_add, certificateValue_sub]

end Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.Artifact.PolynomialCertificate
