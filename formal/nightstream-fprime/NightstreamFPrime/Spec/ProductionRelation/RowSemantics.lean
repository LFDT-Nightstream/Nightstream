import NightstreamFPrime.Spec.ProductionRelation

/-!
Owns the named 7-port image interface used by the production selective
compiler. Each row constructor is interpreted by the sole fixed 8-term
constraint polynomial.

This module does not assign columns or construct sparse matrices.
-/

namespace NightstreamFPrime.Spec.ProductionRelation.RowSemantics

open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.CCSResidualTable
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier

/-- Named values of the 7 selective matrix images. -/
structure PortValues where
  bit : F := 0
  generalSelector : F := 0
  a : F := 0
  b : F := 0
  c : F := 0
  sboxInput : F := 0
  evalSelector : F := 0
deriving Repr, DecidableEq

/-- Convert the named interface to the exact 7-slot matrix-image order. -/
def PortValues.get (values : PortValues)
    (port : Fin matrixCount) : F :=
  match port.val with
  | 0 => values.bit
  | 1 => values.generalSelector
  | 2 => values.a
  | 3 => values.b
  | 4 => values.c
  | 5 => values.sboxInput
  | 6 => values.evalSelector
  | _ => 0

/-- Fixed seventh power used by the Poseidon2 S-box row. -/
def seventhPower (value : F) : F := pow baseOps value 7

/-- Complete selector-gated base row. Individual row families set unused
ports to zero. -/
def general (selector bitValue left right output sboxValue : F) :
    PortValues :=
  { bit := bitValue
    generalSelector := selector
    a := left
    b := right
    c := output
    sboxInput := sboxValue }

/-- Exact residual of the general row family before specialization. -/
theorem evaluate_general
    (selector bitValue left right output sboxValue : F) :
    evaluatePolynomial baseOps polynomial
        (general selector bitValue left right output sboxValue).get =
      selector *
        ((bitValue * bitValue - bitValue) +
          (left * right - output) + seventhPower sboxValue) := by
  simp [polynomial, SelectivePolynomial.polynomial,
    SelectivePolynomial.terms, SelectivePolynomial.termData,
    SelectivePolynomial.monomial,
    SelectivePolynomial.Term.toMonomial, SelectivePolynomial.sboxTermData,
    SelectivePolynomial.powers, SelectivePolynomial.PortExponents.get,
    evaluatePolynomial, evaluateMonomial, canonicalFinIndices, List.foldl,
    Fin.val_cast, pow, seventhPower, general, PortValues.get, baseOps]
  simp only [mul_add, mul_neg, sub_eq_add_neg, mul_comm, mul_left_comm]
  abel

/-- One selector-gated multiplication row `left * right = output`. -/
def multiplication (selector left right output : F) : PortValues :=
  general selector 0 left right output 0

/-- Exact residual selected by a multiplication row. -/
theorem evaluate_multiplication (selector left right output : F) :
    evaluatePolynomial baseOps polynomial
        (multiplication selector left right output).get =
      selector * (left * right - output) := by
  rw [multiplication, evaluate_general]
  simp [seventhPower, pow, baseOps]

/-- One selector-gated Boolean row `value * value = value`. -/
def boolean (selector value : F) : PortValues :=
  general selector value 0 0 0 0

theorem evaluate_boolean (selector value : F) :
    evaluatePolynomial baseOps polynomial (boolean selector value).get =
      selector * (value * value - value) := by
  rw [boolean, evaluate_general]
  simp [seventhPower, pow, baseOps]

/-- One selector-gated S-box row `input^7 = output`. -/
def sbox (selector input output : F) : PortValues :=
  general selector 0 0 0 output input

theorem evaluate_sbox (selector input output : F) :
    evaluatePolynomial baseOps polynomial (sbox selector input output).get =
      selector * (seventhPower input - output) := by
  rw [sbox, evaluate_general]
  simp [seventhPower, pow, baseOps, sub_eq_add_neg]
  rw [add_comm]

/-- One selector-gated zero pin. -/
def pin (selector value : F) : PortValues :=
  multiplication selector 0 0 value

theorem evaluate_pin (selector value : F) :
    evaluatePolynomial baseOps polynomial (pin selector value).get =
      -(selector * value) := by
  rw [pin, evaluate_multiplication]
  simp [sub_eq_add_neg]

theorem multiplication_zero_of_equal (selector left right output : F)
    (equal : left * right = output) :
    evaluatePolynomial baseOps polynomial
      (multiplication selector left right output).get = 0 := by
  rw [evaluate_multiplication, equal, sub_self, mul_zero]

/-- Exact sum selected by one two-product evaluation row. -/
def productTotal (left right : Fin 2 → F) : F :=
  left 0 * right 0 + left 1 * right 1

/-- One evaluation-selector row. The general selector stays zero, so the two
pair ports are independent multiplication factors. -/
def productSum (selector : F) (left right : Fin 2 → F)
    (output : F) : PortValues :=
  { bit := left 0
    a := right 0
    b := left 1
    c := output
    sboxInput := right 1
    evalSelector := selector }

/-- Exact residual selected by a two-product row. -/
theorem evaluate_productSum (selector : F) (left right : Fin 2 → F)
    (output : F) :
    evaluatePolynomial baseOps polynomial
        (productSum selector left right output).get =
      selector * (productTotal left right - output) := by
  simp [polynomial, SelectivePolynomial.polynomial,
    SelectivePolynomial.terms, SelectivePolynomial.termData,
    SelectivePolynomial.monomial,
    SelectivePolynomial.Term.toMonomial, SelectivePolynomial.sboxTermData,
    SelectivePolynomial.powers, SelectivePolynomial.PortExponents.get,
    evaluatePolynomial, evaluateMonomial, canonicalFinIndices, List.foldl,
    Fin.val_cast, pow, productSum, productTotal, PortValues.get, baseOps]
  simp only [mul_add, mul_neg, sub_eq_add_neg, mul_comm]
  abel

theorem productSum_zero_of_equal (selector : F) (left right : Fin 2 → F)
    (output : F) (equal : productTotal left right = output) :
    evaluatePolynomial baseOps polynomial
      (productSum selector left right output).get = 0 := by
  rw [evaluate_productSum, equal, sub_self, mul_zero]

end NightstreamFPrime.Spec.ProductionRelation.RowSemantics
