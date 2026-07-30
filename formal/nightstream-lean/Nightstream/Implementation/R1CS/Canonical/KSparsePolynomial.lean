import Nightstream.Implementation.R1CS.Canonical.KPointEquality

/-!
Contract: the paper's explicit sparse CCS polynomial as canonical rows.

Owns: expansion of each monomial's typed exponent vector, one multiplication
frame per actual degree, row-free summation of term outputs, exact cost, and
soundness to `CCSResidualTable.evaluatePolynomial`.

Declared degree metadata never controls allocation. The emitted count is the
sum of the explicit monomials' derived total degrees.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Canonical.KSparsePolynomial

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal
open Nightstream.Implementation.R1CS.Canonical.KMul
open Nightstream.Implementation.R1CS.Canonical.KHorner
open Nightstream.Implementation.R1CS.Canonical.KConcreteFixedPhaseBridge

abbrev ConcreteK := Nightstream.SuperNeo.Concrete.K

/-- Flatten `x_i ^ e_i` into `e_i` copies of `x_i`. -/
def expandedFactors
    {matrixCount : Nat}
    (point : Fin matrixCount → Carried)
    (monomial : CCSResidualTable.Monomial ConcreteK matrixCount) :
    List Carried :=
  (canonicalFinIndices matrixCount).flatMap fun index =>
    List.replicate (monomial.exponents index) (point index)

theorem expandedFactors_length
    {matrixCount : Nat}
    (point : Fin matrixCount → Carried)
    (monomial : CCSResidualTable.Monomial ConcreteK matrixCount) :
    (expandedFactors point monomial).length = monomial.totalDegree := by
  unfold expandedFactors CCSResidualTable.Monomial.totalDegree
  rw [List.length_flatMap]
  congr 1
  apply List.map_congr_left
  intro index _
  exact List.length_replicate

def termRows
    {matrixCount : Nat}
    (point : Fin matrixCount → Carried)
    (monomial : CCSResidualTable.Monomial ConcreteK matrixCount)
    (base : Nat) : List Row :=
  KMulChain.rows (KLinear.constantCarried monomial.coefficient)
    (KFrames.frameAt base) (expandedFactors point monomial) 0

def termOutput
    {matrixCount : Nat}
    (point : Fin matrixCount → Carried)
    (monomial : CCSResidualTable.Monomial ConcreteK matrixCount)
    (base : Nat) : Carried :=
  KMulChain.productCarried (KLinear.constantCarried monomial.coefficient)
    (KFrames.frameAt base) (expandedFactors point monomial) 0

theorem termRows_length
    {matrixCount : Nat}
    (point : Fin matrixCount → Carried)
    (monomial : CCSResidualTable.Monomial ConcreteK matrixCount)
    (base : Nat) :
    (termRows point monomial base).length = 3 * monomial.totalDegree := by
  unfold termRows
  rw [KMulChain.rows_length, expandedFactors_length]

def totalDegreeSum
    {matrixCount : Nat}
    (terms : List (CCSResidualTable.Monomial ConcreteK matrixCount)) : Nat :=
  (terms.map CCSResidualTable.Monomial.totalDegree).sum

/-- Allocate each term immediately after the prior term's exact degree block. -/
def termsRows
    {matrixCount : Nat}
    (point : Fin matrixCount → Carried) :
    List (CCSResidualTable.Monomial ConcreteK matrixCount) → Nat → Nat → List Row
  | [], _, _ => []
  | monomial :: rest, base, offset =>
      termRows point monomial (base + 3 * offset) ++
        termsRows point rest base (offset + monomial.totalDegree)

def termOutputs
    {matrixCount : Nat}
    (point : Fin matrixCount → Carried) :
    List (CCSResidualTable.Monomial ConcreteK matrixCount) → Nat → Nat →
      List Carried
  | [], _, _ => []
  | monomial :: rest, base, offset =>
      termOutput point monomial (base + 3 * offset) ::
        termOutputs point rest base (offset + monomial.totalDegree)

theorem termsRows_length
    {matrixCount : Nat}
    (point : Fin matrixCount → Carried) :
    ∀ (terms : List (CCSResidualTable.Monomial ConcreteK matrixCount))
      (base offset : Nat),
      (termsRows point terms base offset).length = 3 * totalDegreeSum terms
  | [], _, _ => rfl
  | monomial :: rest, base, offset => by
      rw [termsRows, List.length_append, termRows_length,
        termsRows_length point rest base (offset + monomial.totalDegree)]
      unfold totalDegreeSum
      simp only [List.map_cons, List.sum_cons]
      omega

theorem termOutputs_length
    {matrixCount : Nat}
    (point : Fin matrixCount → Carried) :
    ∀ (terms : List (CCSResidualTable.Monomial ConcreteK matrixCount))
      (base offset : Nat),
      (termOutputs point terms base offset).length = terms.length
  | [], _, _ => rfl
  | monomial :: rest, base, offset => by
      simp only [termOutputs, List.length_cons,
        termOutputs_length point rest base
          (offset + monomial.totalDegree)]

def sumCarried : List Carried → Carried
  | [] => KLinear.zeroCarried
  | value :: rest => KLinear.addCarried value (sumCarried rest)

structure Input (matrixCount : Nat) where
  polynomial : CCSResidualTable.ConstraintPolynomial ConcreteK matrixCount
  point : Fin matrixCount → Carried
  frameBase : Nat

def rows {matrixCount : Nat} (input : Input matrixCount) : List Row :=
  termsRows input.point input.polynomial.terms input.frameBase 0

def output {matrixCount : Nat} (input : Input matrixCount) : Carried :=
  sumCarried
    (termOutputs input.point input.polynomial.terms input.frameBase 0)

def columns {matrixCount : Nat} (input : Input matrixCount) : List Nat :=
  KFrames.frameColumns input.frameBase
    (totalDegreeSum input.polynomial.terms)

theorem rows_length {matrixCount : Nat} (input : Input matrixCount) :
    (rows input).length = 3 * totalDegreeSum input.polynomial.terms :=
  termsRows_length input.point input.polynomial.terms input.frameBase 0

theorem columns_length {matrixCount : Nat} (input : Input matrixCount) :
    (columns input).length = 3 * totalDegreeSum input.polynomial.terms :=
  KFrames.frameColumns_length _ _

theorem columns_nodup {matrixCount : Nat} (input : Input matrixCount) :
    (columns input).Nodup :=
  KFrames.frameColumns_nodup _ _

/-! ## Semantic expansion -/

def decoded (assignment : Nat → Nat) (value : Carried) : ConcreteK :=
  KPointEquality.decoded assignment value

theorem ofConcrete_decoded (assignment : Nat → Nat) (value : Carried) :
    KConcreteBridge.ofConcrete (decoded assignment value) =
      carriedValue assignment value :=
  KPointEquality.ofConcrete_decoded assignment value

def semanticExpandedFactors
    {matrixCount : Nat}
    (point : Fin matrixCount → ConcreteK)
    (monomial : CCSResidualTable.Monomial ConcreteK matrixCount) :
    List ConcreteK :=
  (canonicalFinIndices matrixCount).flatMap fun index =>
    List.replicate (monomial.exponents index) (point index)

def concreteProduct (initial : ConcreteK) : List ConcreteK → ConcreteK
  | [] => initial
  | factor :: rest =>
      concreteProduct
        (Nightstream.SuperNeo.Concrete.K.mul initial factor) rest

theorem ofConcrete_concreteProduct (initial : ConcreteK) :
    ∀ factors : List ConcreteK,
      KConcreteBridge.ofConcrete (concreteProduct initial factors) =
        KMulChain.productValue (KConcreteBridge.ofConcrete initial)
          (factors.map KConcreteBridge.ofConcrete)
  | [] => rfl
  | factor :: rest => by
      rw [concreteProduct, List.map_cons, KMulChain.productValue,
        ofConcrete_concreteProduct
          (Nightstream.SuperNeo.Concrete.K.mul initial factor),
        KConcreteBridge.ofConcrete_mul]

theorem expandedFactors_bridge
    {matrixCount : Nat}
    (point : Fin matrixCount → Carried)
    (monomial : CCSResidualTable.Monomial ConcreteK matrixCount)
    (assignment : Nat → Nat) :
    (expandedFactors point monomial).map (carriedValue assignment) =
      (semanticExpandedFactors (fun index => decoded assignment (point index))
        monomial).map KConcreteBridge.ofConcrete := by
  unfold expandedFactors semanticExpandedFactors
  rw [List.map_flatMap, List.map_flatMap]
  generalize canonicalFinIndices matrixCount = indices
  induction indices with
  | nil => rfl
  | cons index rest inductionHypothesis =>
      rw [List.flatMap_cons, List.flatMap_cons, inductionHypothesis,
        List.map_replicate, List.map_replicate,
        ofConcrete_decoded assignment (point index)]

theorem concreteProduct_append (initial : ConcreteK) :
    ∀ left right : List ConcreteK,
      concreteProduct initial (left ++ right) =
        concreteProduct (concreteProduct initial left) right
  | [], _ => rfl
  | factor :: rest, right => by
      change
        concreteProduct
            (Nightstream.SuperNeo.Concrete.K.mul initial factor)
            (rest ++ right) =
          concreteProduct
            (concreteProduct
              (Nightstream.SuperNeo.Concrete.K.mul initial factor) rest)
            right
      exact concreteProduct_append
        (Nightstream.SuperNeo.Concrete.K.mul initial factor) rest right

theorem concreteProduct_replicate (initial value : ConcreteK) :
    ∀ exponent,
      concreteProduct initial (List.replicate exponent value) =
        Nightstream.SuperNeo.Concrete.K.mul initial
          (CCSResidualTable.pow ConcreteCarrier.extensionOps value exponent)
  | 0 => by
      rw [List.replicate_zero, concreteProduct]
      exact (ConcreteCarrier.extensionLaws.mul_one initial).symm
  | exponent + 1 => by
      rw [List.replicate_succ, concreteProduct,
        concreteProduct_replicate
          (Nightstream.SuperNeo.Concrete.K.mul initial value) value exponent]
      change
        Nightstream.SuperNeo.Concrete.K.mul
            (Nightstream.SuperNeo.Concrete.K.mul initial value)
            (CCSResidualTable.pow ConcreteCarrier.extensionOps value exponent) =
          Nightstream.SuperNeo.Concrete.K.mul initial
            (Nightstream.SuperNeo.Concrete.K.mul
              (CCSResidualTable.pow ConcreteCarrier.extensionOps value exponent)
              value)
      calc
        Nightstream.SuperNeo.Concrete.K.mul
            (Nightstream.SuperNeo.Concrete.K.mul initial value)
            (CCSResidualTable.pow ConcreteCarrier.extensionOps value exponent) =
          Nightstream.SuperNeo.Concrete.K.mul initial
            (Nightstream.SuperNeo.Concrete.K.mul value
              (CCSResidualTable.pow ConcreteCarrier.extensionOps value exponent)) :=
            ConcreteCarrier.extensionLaws.mul_assoc _ _ _
        _ = Nightstream.SuperNeo.Concrete.K.mul initial
            (Nightstream.SuperNeo.Concrete.K.mul
              (CCSResidualTable.pow ConcreteCarrier.extensionOps value exponent)
              value) := by
            apply congrArg (Nightstream.SuperNeo.Concrete.K.mul initial)
            exact ConcreteCarrier.extensionLaws.mul_comm value _

theorem concreteProduct_expanded
    {matrixCount : Nat}
    (point : Fin matrixCount → ConcreteK)
    (monomial : CCSResidualTable.Monomial ConcreteK matrixCount)
    (initial : ConcreteK) :
    concreteProduct initial (semanticExpandedFactors point monomial) =
      (canonicalFinIndices matrixCount).foldl
        (fun accumulated index =>
          Nightstream.SuperNeo.Concrete.K.mul accumulated
            (CCSResidualTable.pow ConcreteCarrier.extensionOps
              (point index) (monomial.exponents index)))
        initial := by
  unfold semanticExpandedFactors
  generalize canonicalFinIndices matrixCount = coordinateIndices
  induction coordinateIndices generalizing initial with
  | nil => rfl
  | cons index rest inductionHypothesis =>
      rw [List.flatMap_cons, concreteProduct_append,
        concreteProduct_replicate, List.foldl_cons,
        inductionHypothesis]

theorem concreteProduct_eq_evaluateMonomial
    {matrixCount : Nat}
    (point : Fin matrixCount → ConcreteK)
    (monomial : CCSResidualTable.Monomial ConcreteK matrixCount) :
    concreteProduct monomial.coefficient
        (semanticExpandedFactors point monomial) =
      CCSResidualTable.evaluateMonomial ConcreteCarrier.extensionOps
        monomial point := by
  rw [concreteProduct_expanded]
  rfl

/-! ## Term and polynomial soundness -/

theorem termRows_sound
    {matrixCount : Nat}
    (point : Fin matrixCount → Carried)
    (monomial : CCSResidualTable.Monomial ConcreteK matrixCount)
    (base : Nat) (assignment : Nat → Nat)
    (constantWire : assignment 0 = 1)
    (satisfied : Satisfies (termRows point monomial base) assignment) :
    decoded assignment (termOutput point monomial base) =
      CCSResidualTable.evaluateMonomial ConcreteCarrier.extensionOps monomial
        (fun index => decoded assignment (point index)) := by
  apply KConcreteBridge.ofConcrete_injective
  rw [ofConcrete_decoded]
  have chain :=
    KMulChain.rows_sound assignment (KFrames.frameAt base)
      (KLinear.constantCarried monomial.coefficient)
      (expandedFactors point monomial) 0 satisfied
  unfold termOutput
  rw [chain, KLinear.carriedValue_constant assignment monomial.coefficient
    constantWire, expandedFactors_bridge point monomial assignment,
    ← ofConcrete_concreteProduct,
    concreteProduct_eq_evaluateMonomial]

theorem termsRows_sound
    {matrixCount : Nat}
    (point : Fin matrixCount → Carried) (assignment : Nat → Nat)
    (constantWire : assignment 0 = 1) :
    ∀ (terms : List (CCSResidualTable.Monomial ConcreteK matrixCount))
      (base offset : Nat),
      Satisfies (termsRows point terms base offset) assignment →
      (termOutputs point terms base offset).map (decoded assignment) =
        terms.map fun monomial =>
          CCSResidualTable.evaluateMonomial ConcreteCarrier.extensionOps
            monomial (fun index => decoded assignment (point index))
  | [], _, _, _ => rfl
  | monomial :: rest, base, offset, satisfied => by
      have headSatisfied :
          Satisfies (termRows point monomial (base + 3 * offset)) assignment :=
        fun row member =>
          satisfied row (List.mem_append_left _ member)
      have tailSatisfied :
          Satisfies
            (termsRows point rest base (offset + monomial.totalDegree))
            assignment :=
        fun row member =>
          satisfied row (List.mem_append_right _ member)
      rw [termOutputs, List.map_cons, List.map_cons,
        termRows_sound point monomial (base + 3 * offset) assignment
          constantWire headSatisfied,
        termsRows_sound point assignment constantWire rest base
          (offset + monomial.totalDegree) tailSatisfied]

theorem decoded_zero (assignment : Nat → Nat) :
    decoded assignment KLinear.zeroCarried =
      Nightstream.SuperNeo.Concrete.K.zero := by
  apply KConcreteBridge.ofConcrete_injective
  rw [ofConcrete_decoded, KLinear.carriedValue_zero,
    KConcreteBridge.ofConcrete_zero]

theorem decoded_add (assignment : Nat → Nat) (left right : Carried) :
    decoded assignment (KLinear.addCarried left right) =
      Nightstream.SuperNeo.Concrete.K.add
        (decoded assignment left) (decoded assignment right) := by
  apply KConcreteBridge.ofConcrete_injective
  rw [ofConcrete_decoded, KLinear.carriedValue_add,
    KConcreteBridge.ofConcrete_add, ofConcrete_decoded, ofConcrete_decoded]

theorem decoded_sum (assignment : Nat → Nat) :
    ∀ values : List Carried,
      decoded assignment (sumCarried values) =
        BooleanTable.finiteSum ConcreteCarrier.extensionOps
          (values.map (decoded assignment))
  | [] => decoded_zero assignment
  | value :: rest => by
      rw [sumCarried, decoded_add, List.map_cons,
        BooleanTable.finiteSum, decoded_sum assignment rest]
      rfl

/-- Satisfying rows compute the exact sparse polynomial selected by the
verifier-owned explicit term list. -/
theorem rows_sound
    {matrixCount : Nat} (input : Input matrixCount)
    (assignment : Nat → Nat) (constantWire : assignment 0 = 1)
    (satisfied : Satisfies (rows input) assignment) :
    decoded assignment (output input) =
      CCSResidualTable.evaluatePolynomial ConcreteCarrier.extensionOps
        input.polynomial
        (fun index => decoded assignment (input.point index)) := by
  have outputs :=
    termsRows_sound input.point assignment constantWire input.polynomial.terms
      input.frameBase 0 satisfied
  unfold output
  rw [decoded_sum, outputs,
    CCSResidualTable.evaluatePolynomial_eq_sumMap
      ConcreteCarrier.extensionOps ConcreteCarrier.extensionLaws]
  rfl

end Nightstream.Implementation.R1CS.Canonical.KSparsePolynomial
