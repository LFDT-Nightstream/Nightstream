import NightstreamFPrime.Spec.ProductionRelation

/-!
Owns the named 13-port image interface used by the production selective
compiler. Each row constructor is interpreted by the sole fixed 74-term
constraint polynomial. Matrix slot 13 is zero by construction.

This module does not assign columns or construct sparse matrices.

Canonical-class semantic alphabet adapted from
`formal/nightstream-lean/Nightstream/Implementation/R1CS/Correspondence/
FPrimeFullHistory/SelectiveCcs/CanonicalOpeningArtifactRows.lean` at commit
`8f3c3489ca4d73429069520264c2be320044d622`.
-/

namespace NightstreamFPrime.Spec.ProductionRelation.RowSemantics

open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.CCSResidualTable
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier

/-- Named values of the 13 meaningful selective matrix images. -/
structure PortValues where
  bit : F := 0
  generalSelector : F := 0
  a : F := 0
  b : F := 0
  c : F := 0
  sboxInput : F := 0
  centeredUnit : F := 0
  evalSelector : F := 0
  class0 : F := 0
  class1 : F := 0
  class2 : F := 0
  class3 : F := 0
  class4 : F := 0
deriving Repr, DecidableEq

/-- Convert the named interface to the exact 14-slot matrix-image order. -/
def PortValues.get (values : PortValues)
    (port : Fin matrixCount) : F :=
  match port.val with
  | 0 => values.bit
  | 1 => values.generalSelector
  | 2 => values.a
  | 3 => values.b
  | 4 => values.c
  | 5 => values.sboxInput
  | 6 => values.centeredUnit
  | 7 => values.evalSelector
  | 8 => values.class0
  | 9 => values.class1
  | 10 => values.class2
  | 11 => values.class3
  | 12 => values.class4
  | _ => 0

@[simp] theorem PortValues.get_zeroPort (values : PortValues) :
    values.get zeroPort = 0 := by
  rfl

/-- Fixed seventh power used by the Poseidon2 S-box row. -/
def seventhPower (value : F) : F := pow baseOps value 7

/-- Complete selector-gated base row. Individual row families set unused
ports to zero. -/
def general (selector bitValue left right output sboxValue centeredValue : F) :
    PortValues :=
  { bit := bitValue
    generalSelector := selector
    a := left
    b := right
    c := output
    sboxInput := sboxValue
    centeredUnit := centeredValue }

/-- Exact residual of the general row family before specialization. -/
theorem evaluate_general
    (selector bitValue left right output sboxValue centeredValue : F) :
    evaluatePolynomial baseOps polynomial
        (general selector bitValue left right output sboxValue
          centeredValue).get =
      selector *
        ((bitValue * bitValue - bitValue) +
          (left * right - output) + seventhPower sboxValue +
          (centeredValue * centeredValue * centeredValue - centeredValue)) := by
  simp [polynomial, SelectivePolynomial.polynomial,
    SelectivePolynomial.terms, SelectivePolynomial.baseTerms,
    SelectivePolynomial.borrowTerms0, SelectivePolynomial.borrowTerms1,
    SelectivePolynomial.borrowTerms2, SelectivePolynomial.borrowTerms3,
    SelectivePolynomial.borrowTerms4, SelectivePolynomial.borrowTerm,
    SelectivePolynomial.borrowPowers, SelectivePolynomial.classExponent,
    SelectivePolynomial.sboxTerm, SelectivePolynomial.monomial,
    SelectivePolynomial.powers, SelectivePolynomial.PortExponents.get,
    evaluatePolynomial, evaluateMonomial, canonicalFinIndices, List.foldl,
    Fin.val_cast, pow, seventhPower, general, PortValues.get, baseOps]
  simp only [mul_add, mul_neg, sub_eq_add_neg, mul_comm, mul_left_comm]
  abel

/-- One selector-gated multiplication row `left * right = output`. -/
def multiplication (selector left right output : F) : PortValues :=
  general selector 0 left right output 0 0

/-- Exact residual selected by a multiplication row. -/
theorem evaluate_multiplication (selector left right output : F) :
    evaluatePolynomial baseOps polynomial
        (multiplication selector left right output).get =
      selector * (left * right - output) := by
  rw [multiplication, evaluate_general]
  simp [seventhPower, pow, baseOps]

/-- One selector-gated Boolean row `value * value = value`. -/
def boolean (selector value : F) : PortValues :=
  general selector value 0 0 0 0 0

theorem evaluate_boolean (selector value : F) :
    evaluatePolynomial baseOps polynomial (boolean selector value).get =
      selector * (value * value - value) := by
  rw [boolean, evaluate_general]
  simp [seventhPower, pow, baseOps]

/-- One selector-gated S-box row `input^7 = output`. -/
def sbox (selector input output : F) : PortValues :=
  general selector 0 0 0 output input 0

theorem evaluate_sbox (selector input output : F) :
    evaluatePolynomial baseOps polynomial (sbox selector input output).get =
      selector * (seventhPower input - output) := by
  rw [sbox, evaluate_general]
  simp [seventhPower, pow, baseOps, sub_eq_add_neg]
  rw [add_comm]

/-- One selector-gated centered-unit row `value^3 = value`. -/
def centered (selector value : F) : PortValues :=
  general selector 0 0 0 0 0 value

theorem evaluate_centered (selector value : F) :
    evaluatePolynomial baseOps polynomial (centered selector value).get =
      selector * (value * value * value - value) := by
  rw [centered, evaluate_general]
  simp [seventhPower, pow, baseOps]

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

/-- Exact sum selected by one five-product evaluation row. -/
def productTotal (left right : Fin 5 → F) : F :=
  left 0 * right 0 + left 1 * right 1 + left 2 * right 2 +
    left 3 * right 3 + left 4 * right 4

/-- One evaluation-selector row. The general selector stays zero, so the five
pair ports are independent multiplication factors. -/
def productSum (selector : F) (left right : Fin 5 → F)
    (output : F) : PortValues :=
  { bit := left 0
    a := right 0
    b := left 1
    c := output
    sboxInput := right 1
    centeredUnit := left 2
    evalSelector := selector
    class0 := right 2
    class1 := left 3
    class2 := right 3
    class3 := left 4
    class4 := right 4 }

/-- Exact residual selected by a five-product row. -/
theorem evaluate_productSum (selector : F) (left right : Fin 5 → F)
    (output : F) :
    evaluatePolynomial baseOps polynomial
        (productSum selector left right output).get =
      selector * (productTotal left right - output) := by
  simp [polynomial, SelectivePolynomial.polynomial,
    SelectivePolynomial.terms, SelectivePolynomial.baseTerms,
    SelectivePolynomial.borrowTerms0, SelectivePolynomial.borrowTerms1,
    SelectivePolynomial.borrowTerms2, SelectivePolynomial.borrowTerms3,
    SelectivePolynomial.borrowTerms4, SelectivePolynomial.borrowTerm,
    SelectivePolynomial.borrowPowers, SelectivePolynomial.classExponent,
    SelectivePolynomial.sboxTerm, SelectivePolynomial.monomial,
    SelectivePolynomial.powers, SelectivePolynomial.PortExponents.get,
    evaluatePolynomial, evaluateMonomial, canonicalFinIndices, List.foldl,
    Fin.val_cast, pow, productSum, productTotal, PortValues.get, baseOps]
  simp only [mul_add, mul_neg, sub_eq_add_neg, mul_comm, mul_left_comm]
  abel

theorem productSum_zero_of_equal (selector : F) (left right : Fin 5 → F)
    (output : F) (equal : productTotal left right = output) :
    evaluatePolynomial baseOps polynomial
      (productSum selector left right output).get = 0 := by
  rw [evaluate_productSum, equal, sub_self, mul_zero]

/-- Centered digit alphabet in the order `-1, 0, 1`. -/
def centeredValue (value : Fin 3) : F :=
  if value.val = 0 then -1 else if value.val = 1 then 0 else 1

/-- Canonical natural representative of `centeredValue`. -/
def centeredNat (value : Fin 3) : Nat :=
  if value.val = 0 then goldilocksModulus - 1
  else if value.val = 1 then 0 else 1

def scaledDigit (complemented : Bool) (value : F) : F :=
  if complemented then -value else value

def scaledBorrow (complemented : Bool) (value : F) : F :=
  if complemented then 1 - value else value

def binaryValue (value : Fin 2) : F :=
  if value.val = 0 then 0 else 1

def originalClassBound (complemented : Bool) (boundClass : Fin 5) : Nat :=
  if complemented then 8 - boundClass.val else boundClass.val

/-- Ordinary comparison step on one base-three digit. -/
def scalarStep (bound trit borrow : Nat) : Nat :=
  if bound < trit + borrow then 1 else 0

def scalarTwo (boundZero boundOne tritZero tritOne borrow : Nat) : Nat :=
  scalarStep boundOne tritOne (scalarStep boundZero tritZero borrow)

/-- One active one-hot canonical class before the complement transform is
specialized. -/
def canonicalValues (boundClass : Fin 5)
    (input second output first : F) : PortValues :=
  { bit := input
    generalSelector := 1
    a := second
    c := output
    sboxInput := output
    centeredUnit := first
    class0 := if boundClass.val = 0 then 1 else 0
    class1 := if boundClass.val = 1 then 1 else 0
    class2 := if boundClass.val = 2 then 1 else 0
    class3 := if boundClass.val = 3 then 1 else 0
    class4 := if boundClass.val = 4 then 1 else 0 }

/-- One active canonical-class row after the emitter's complement transform. -/
def canonicalClass (complemented : Bool) (boundClass : Fin 5)
    (first second output : Fin 3) (input : Fin 2) : PortValues :=
  canonicalValues boundClass
    (scaledBorrow complemented (binaryValue input))
    (scaledDigit complemented (centeredValue second))
    (scaledBorrow complemented (centeredValue output))
    (scaledDigit complemented (centeredValue first))

private def canonicalHalf : F := 9223372034707292161
private def canonicalQuarter : F := 13835058052060938241

/-- Exact selected residual of the five canonical interpolation tables. -/
def canonicalClassResidual : Fin 5 → F → F → F → F → F
  | ⟨0, _⟩, input, second, output, first =>
      -1 + (output +
        (canonicalQuarter * second * first +
        ((-canonicalQuarter) * input * second * first +
        ((-canonicalQuarter) * (second * second) * first +
        (canonicalQuarter * input * (second * second) * first +
        ((-canonicalQuarter) * second * (first * first) +
        (canonicalQuarter * input * second * (first * first) +
        (canonicalQuarter * (second * second) * (first * first) +
          (-canonicalQuarter) * input * (second * second) *
            (first * first)))))))))
  | ⟨1, _⟩, input, second, output, first =>
      -1 + (output +
        ((-canonicalHalf) * second +
        (canonicalHalf * input * second +
        (canonicalHalf * (second * second) +
        ((-canonicalHalf) * input * (second * second) +
        (canonicalQuarter * second * first +
        ((-canonicalQuarter) * (second * second) * first +
        (canonicalQuarter * second * (first * first) +
        ((-canonicalHalf) * input * second * (first * first) +
        ((-canonicalQuarter) * (second * second) * (first * first) +
          canonicalHalf * input * (second * second) *
            (first * first)))))))))))
  | ⟨2, _⟩, input, second, output, first =>
      -1 + (output +
        ((-canonicalHalf) * second +
        (canonicalHalf * (second * second) +
        (canonicalQuarter * input * second * first +
        ((-canonicalQuarter) * input * (second * second) * first +
        (canonicalQuarter * input * second * (first * first) +
          (-canonicalQuarter) * input * (second * second) *
            (first * first)))))))
  | ⟨3, _⟩, input, second, output, first =>
      -1 + (output +
        ((-canonicalHalf) * second +
        (canonicalHalf * (second * second) +
        ((-canonicalHalf) * first +
        (canonicalHalf * input * first +
        (canonicalHalf * (second * second) * first +
        ((-canonicalHalf) * input * (second * second) * first +
        (canonicalHalf * (first * first) +
        ((-canonicalHalf) * input * (first * first) +
        ((-canonicalHalf) * (second * second) * (first * first) +
          canonicalHalf * input * (second * second) *
            (first * first)))))))))))
  | ⟨4, _⟩, input, second, output, first =>
      output +
        ((-1) * input +
        ((-canonicalHalf) * second +
        ((-canonicalHalf) * (second * second) +
        (input * (second * second) +
        ((-canonicalHalf) * first +
        (canonicalHalf * (second * second) * first +
        ((-canonicalHalf) * (first * first) +
        (input * (first * first) +
        (canonicalHalf * (second * second) * (first * first) +
          (-1) * input * (second * second) * (first * first))))))))))

/-- Base-domain terms plus the selected canonical interpolation table. -/
def canonicalTotalResidual
    (boundClass : Fin 5) (input second output first : F) : F :=
  (input * input + -input) +
    (-output) +
    (output * output * output * output * output * output * output) +
    (first * first * first + -first) +
    canonicalClassResidual boundClass input second output first

/-- Compact normal form of the complete 74-term polynomial on one canonical
class row. -/
theorem evaluate_canonicalValues
    (boundClass : Fin 5) (input second output first : F) :
    evaluatePolynomial baseOps polynomial
        (canonicalValues boundClass input second output first).get =
      canonicalTotalResidual boundClass input second output first := by
  rcases boundClass with ⟨boundClass, boundClassLt⟩
  have boundClassCases :
      boundClass = 0 ∨ boundClass = 1 ∨ boundClass = 2 ∨
        boundClass = 3 ∨ boundClass = 4 := by
    omega
  rcases boundClassCases with rfl | rfl | rfl | rfl | rfl <;>
    simp [polynomial, SelectivePolynomial.polynomial,
      SelectivePolynomial.terms, SelectivePolynomial.baseTerms,
      SelectivePolynomial.borrowTerms0, SelectivePolynomial.borrowTerms1,
      SelectivePolynomial.borrowTerms2, SelectivePolynomial.borrowTerms3,
      SelectivePolynomial.borrowTerms4, SelectivePolynomial.borrowTerm,
      SelectivePolynomial.borrowPowers, SelectivePolynomial.classExponent,
      SelectivePolynomial.sboxTerm, SelectivePolynomial.monomial,
      SelectivePolynomial.powers, SelectivePolynomial.PortExponents.get,
      evaluatePolynomial, evaluateMonomial, canonicalFinIndices, List.foldl,
      Fin.val_cast, pow, canonicalValues, canonicalTotalResidual,
      canonicalClassResidual, canonicalHalf, canonicalQuarter,
      SelectivePolynomial.half, SelectivePolynomial.quarter,
      PortValues.get, baseOps]
  all_goals simp only [add_assoc]

/-- Intended two-trit transition before the complement transform. The finite
indices of `first` and `second` are exactly ordinary trits `0, 1, 2`. -/
def CanonicalAccepted (complemented : Bool) (boundClass : Fin 5)
    (first second output : Fin 3) (input : Fin 2) : Prop :=
  let bound := originalClassBound complemented boundClass
  centeredNat output =
    scalarTwo (bound % 3) ((bound / 3) % 3)
      first.val second.val input.val

private instance canonicalAcceptedDecidable
    (complemented : Bool) (boundClass : Fin 5)
    (first second output : Fin 3) (input : Fin 2) :
    Decidable
      (CanonicalAccepted complemented boundClass first second output input) := by
  unfold CanonicalAccepted
  infer_instance

/-- The five polynomial tables are exact on the complete centered/Boolean
semantic alphabet. -/
theorem evaluate_canonicalClass_eq_zero_iff
    (complemented : Bool) (boundClass : Fin 5)
    (first second output : Fin 3) (input : Fin 2) :
    evaluatePolynomial baseOps polynomial
        (canonicalClass complemented boundClass first second output input).get = 0 ↔
      CanonicalAccepted complemented boundClass first second output input := by
  rw [canonicalClass, evaluate_canonicalValues]
  cases complemented <;>
    fin_cases boundClass <;>
    fin_cases first <;>
    fin_cases second <;>
    fin_cases output <;>
    fin_cases input <;>
    decide

end NightstreamFPrime.Spec.ProductionRelation.RowSemantics
