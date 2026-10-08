import NightstreamFPrime.Spec.ProductionRelation

/-!
Owns the named 4-port image interface used by the production gate compiler.
Each row constructor is interpreted by the sole fixed 3-term gate polynomial.

This module does not assign columns or construct sparse matrices.
-/

namespace NightstreamFPrime.Spec.ProductionRelation.RowSemantics

open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.CCSResidualTable
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier

/-- Named values of the 4 gate matrix images. -/
structure PortValues where
  a : F := 0
  b : F := 0
  c : F := 0
  sboxInput : F := 0
deriving Repr, DecidableEq

/-- Convert the named interface to the exact 4-slot matrix-image order. -/
def PortValues.get (values : PortValues)
    (port : Fin matrixCount) : F :=
  match port.val with
  | 0 => values.a
  | 1 => values.b
  | 2 => values.c
  | 3 => values.sboxInput
  | _ => 0

/-- Fixed seventh power used by the Poseidon2 S-box row. -/
def seventhPower (value : F) : F := pow baseOps value 7

/-- Exact gate value on named port images. -/
theorem evaluate (values : PortValues) :
    evaluatePolynomial baseOps polynomial values.get =
      values.a * values.b - values.c + seventhPower values.sboxInput := by
  simp [polynomial, GatePolynomial.polynomial,
    GatePolynomial.terms, GatePolynomial.termData, GatePolynomial.monomial,
    GatePolynomial.Term.toMonomial, GatePolynomial.sboxTermData,
    GatePolynomial.powers, GatePolynomial.PortExponents.get,
    evaluatePolynomial, evaluateMonomial, canonicalFinIndices, List.foldl,
    Fin.val_cast, pow, seventhPower, PortValues.get, baseOps]
  rw [sub_eq_add_neg]

/-- One multiplication row `left * right = output`. -/
def multiplication (left right output : F) : PortValues :=
  { a := left, b := right, c := output }

/-- Exact residual of a multiplication row. -/
theorem evaluate_multiplication (left right output : F) :
    evaluatePolynomial baseOps polynomial
        (multiplication left right output).get =
      left * right - output := by
  rw [evaluate]
  simp [multiplication, seventhPower, pow, baseOps]

/-- One S-box row `input^7 = output`. -/
def sbox (input output : F) : PortValues :=
  { c := output, sboxInput := input }

theorem evaluate_sbox (input output : F) :
    evaluatePolynomial baseOps polynomial (sbox input output).get =
      seventhPower input - output := by
  rw [evaluate]
  simp [sbox, sub_eq_add_neg, add_comm]

/-- One zero pin. -/
def pin (value : F) : PortValues :=
  multiplication 0 0 value

theorem evaluate_pin (value : F) :
    evaluatePolynomial baseOps polynomial (pin value).get = -value := by
  rw [pin, evaluate_multiplication]
  simp

theorem multiplication_zero_of_equal (left right output : F)
    (equal : left * right = output) :
    evaluatePolynomial baseOps polynomial
      (multiplication left right output).get = 0 := by
  rw [evaluate_multiplication, equal, sub_self]

end NightstreamFPrime.Spec.ProductionRelation.RowSemantics
