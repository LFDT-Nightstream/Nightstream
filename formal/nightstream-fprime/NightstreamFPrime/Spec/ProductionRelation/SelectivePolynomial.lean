import Mathlib.Tactic
import NightstreamFPrime.Spec.Algebra
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.CCSResidualTable

/-!
Paper authority: SuperNeo v1.1, Definitions 18--20 and Section 7.3.
Compiler obligation: one low-norm CCS gate for the fixed Nightstream circuit.

Inputs:
- 13 meaningful matrix images selected by the row family;
- one final canonical-zero matrix image;
- Pad is not an input here and remains the separate `Eval_K` family.

Constraint groups:
- C1: Boolean, multiplication, and S-box trace checks;
- C2: centered-unit and canonical field-representation checks;
- C3: five packed evaluation products;
- C4: one indexed two-trit borrow relation for each bound class `0..4`.

Parent coverage:
- `ProductionRelation.polynomial`;
- the `F` term in SuperNeo v1.1 PiCCS;
- the production CCS matrix family, without Pad-as-matrix-zero compression.
-/

namespace NightstreamFPrime.Spec.ProductionRelation.SelectivePolynomial

open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.CCSResidualTable

/-- The fixed SuperNeo v1.1 `Eval_A` arity. The final slot is zero. -/
def matrixCount : Nat := 14

/-- Number of matrix slots that carry selective compiler ports. -/
def meaningfulPortCount : Nat := 13

/-- Exponents for the 13 meaningful selective ports. Slot 13 is absent by
construction, so every monomial is independent of the canonical-zero slot. -/
structure PortExponents where
  bit : Nat := 0
  generalSelector : Nat := 0
  a : Nat := 0
  b : Nat := 0
  c : Nat := 0
  sboxInput : Nat := 0
  centeredUnit : Nat := 0
  evalSelector : Nat := 0
  class0 : Nat := 0
  class1 : Nat := 0
  class2 : Nat := 0
  class3 : Nat := 0
  class4 : Nat := 0

namespace PortExponents

/-- Convert the named audit interface to the fixed matrix-slot order. -/
def get (powers : PortExponents) (index : Fin matrixCount) : Nat :=
  match index.val with
  | 0 => powers.bit
  | 1 => powers.generalSelector
  | 2 => powers.a
  | 3 => powers.b
  | 4 => powers.c
  | 5 => powers.sboxInput
  | 6 => powers.centeredUnit
  | 7 => powers.evalSelector
  | 8 => powers.class0
  | 9 => powers.class1
  | 10 => powers.class2
  | 11 => powers.class3
  | 12 => powers.class4
  | _ => 0

/-- Total degree in the 13 meaningful ports. -/
def totalDegree (powers : PortExponents) : Nat :=
  [powers.bit, powers.generalSelector, powers.a, powers.b, powers.c,
    powers.sboxInput, powers.centeredUnit, powers.evalSelector,
    powers.class0, powers.class1, powers.class2, powers.class3,
    powers.class4].sum

end PortExponents

/-- Compact constructor used by every explicit selective monomial. -/
def powers
    (bit : Nat := 0)
    (generalSelector : Nat := 0)
    (a : Nat := 0)
    (b : Nat := 0)
    (c : Nat := 0)
    (sboxInput : Nat := 0)
    (centeredUnit : Nat := 0)
    (evalSelector : Nat := 0)
    (class0 : Nat := 0)
    (class1 : Nat := 0)
    (class2 : Nat := 0)
    (class3 : Nat := 0)
    (class4 : Nat := 0) : PortExponents :=
  { bit, generalSelector, a, b, c, sboxInput, centeredUnit,
    evalSelector, class0, class1, class2, class3, class4 }

/-- A compiler term retains its named exponent record before semantic erasure. -/
structure Term where
  coefficient : F
  powers : PortExponents

/-- One sparse monomial over the fixed 14-slot production relation. -/
def monomial (coefficient : F) (portPowers : PortExponents) :
    Monomial F matrixCount where
  coefficient := coefficient
  exponents := portPowers.get

def Term.toMonomial (term : Term) : Monomial F matrixCount :=
  monomial term.coefficient term.powers

@[simp] theorem monomial_totalDegree
    (coefficient : F) (portPowers : PortExponents) :
    (monomial coefficient portPowers).totalDegree =
      portPowers.totalDegree := by
  unfold Monomial.totalDegree monomial PortExponents.totalDegree
    canonicalFinIndices matrixCount PortExponents.get
  rfl

/-- Matrix slot 13 is the canonical zero matrix. -/
def zeroPort : Fin matrixCount := ⟨13, by norm_num [matrixCount]⟩

@[simp] theorem monomial_zeroPort
    (coefficient : F) (portPowers : PortExponents) :
    (monomial coefficient portPowers).exponents zeroPort = 0 := by
  rfl

/-- The degree-eight S-box trace term. It fixes the maximum CCS degree. -/
def sboxTermData : Term :=
  ⟨1, powers (generalSelector := 1) (sboxInput := 7)⟩

def sboxTerm : Monomial F matrixCount := sboxTermData.toMonomial

@[simp] theorem sboxTerm_totalDegree : sboxTerm.totalDegree = 8 := by
  simp [sboxTerm, sboxTermData, Term.toMonomial, powers, PortExponents.totalDegree]

/-- Boolean, multiplication, S-box, centered-unit, canonical-representation,
and five evaluation-product terms. -/
def baseTermData : List Term :=
  [ Term.mk 1 (powers (bit := 2) (generalSelector := 1)),
    Term.mk (-1) (powers (bit := 1) (generalSelector := 1)),
    Term.mk 1 (powers (generalSelector := 1) (a := 1) (b := 1)),
    Term.mk (-1) (powers (generalSelector := 1) (c := 1)),
    sboxTermData,
    Term.mk 1 (powers (generalSelector := 1) (centeredUnit := 3)),
    Term.mk (-1) (powers (generalSelector := 1) (centeredUnit := 1)),
    Term.mk (-1)
      (powers (generalSelector := 1) (centeredUnit := 3)
        (evalSelector := 1)),
    Term.mk 1
      (powers (generalSelector := 1) (centeredUnit := 1)
        (evalSelector := 1)),
    Term.mk 1
      (powers (generalSelector := 1) (centeredUnit := 6)
        (evalSelector := 1)),
    Term.mk (-2)
      (powers (generalSelector := 1) (centeredUnit := 4)
        (evalSelector := 1)),
    Term.mk 1
      (powers (generalSelector := 1) (centeredUnit := 2)
        (evalSelector := 1)),
    Term.mk (-7)
      (powers (generalSelector := 1) (a := 6) (evalSelector := 1)),
    Term.mk 14
      (powers (generalSelector := 1) (a := 4) (evalSelector := 1)),
    Term.mk (-7)
      (powers (generalSelector := 1) (a := 2) (evalSelector := 1)),
    Term.mk (-1) (powers (c := 1) (evalSelector := 1)),
    Term.mk 1 (powers (bit := 1) (a := 1) (evalSelector := 1)),
    Term.mk 1
      (powers (b := 1) (sboxInput := 1) (evalSelector := 1)),
    Term.mk 1
      (powers (centeredUnit := 1) (evalSelector := 1) (class0 := 1)),
    Term.mk 1
      (powers (evalSelector := 1) (class1 := 1) (class2 := 1)),
    Term.mk 1
      (powers (evalSelector := 1) (class3 := 1) (class4 := 1)) ]

def baseTerms : List (Monomial F matrixCount) :=
  baseTermData.map Term.toMonomial

def classExponent (classIndex : Fin 5) (slot : Nat) : Nat :=
  if classIndex.val = slot then 1 else 0

/-- Port assignment for one two-trit borrow-recurrence monomial. The variables
are `(digit0, digit1, borrowIn, borrowOut) = (centeredUnit, a, bit, c)`. -/
def borrowPowers
    (classIndex : Fin 5)
    (digit0 digit1 borrowIn borrowOut : Nat) : PortExponents :=
  powers
    (bit := borrowIn)
    (generalSelector := 1)
    (a := digit1)
    (c := borrowOut)
    (centeredUnit := digit0)
    (class0 := classExponent classIndex 0)
    (class1 := classExponent classIndex 1)
    (class2 := classExponent classIndex 2)
    (class3 := classExponent classIndex 3)
    (class4 := classExponent classIndex 4)

def borrowTermData
    (classIndex : Fin 5)
    (coefficient : F)
    (digit0 digit1 borrowIn borrowOut : Nat) :
    Term :=
  Term.mk coefficient
    (borrowPowers classIndex digit0 digit1 borrowIn borrowOut)

def borrowTerm
    (classIndex : Fin 5) (coefficient : F)
    (digit0 digit1 borrowIn borrowOut : Nat) : Monomial F matrixCount :=
  (borrowTermData classIndex coefficient digit0 digit1 borrowIn borrowOut).toMonomial

/-- Half-field interpolation coefficient used by the canonical borrow tables. -/
def half : F := 9223372034707292161

/-- Quarter-field interpolation coefficient used by the canonical borrow tables. -/
def quarter : F := 13835058052060938241

/-- Expanded relation for the two-trit bound class 0. -/
def borrowTermData0 : List Term :=
  [ borrowTermData 0 (-1) 0 0 0 0,
    borrowTermData 0 1 0 0 0 1,
    borrowTermData 0 quarter 1 1 0 0,
    borrowTermData 0 (-quarter) 1 1 1 0,
    borrowTermData 0 (-quarter) 1 2 0 0,
    borrowTermData 0 quarter 1 2 1 0,
    borrowTermData 0 (-quarter) 2 1 0 0,
    borrowTermData 0 quarter 2 1 1 0,
    borrowTermData 0 quarter 2 2 0 0,
    borrowTermData 0 (-quarter) 2 2 1 0 ]

def borrowTerms0 : List (Monomial F matrixCount) :=
  borrowTermData0.map Term.toMonomial

/-- Expanded relation for the two-trit bound class 1. -/
def borrowTermData1 : List Term :=
  [ borrowTermData 1 (-1) 0 0 0 0,
    borrowTermData 1 1 0 0 0 1,
    borrowTermData 1 (-half) 0 1 0 0,
    borrowTermData 1 half 0 1 1 0,
    borrowTermData 1 half 0 2 0 0,
    borrowTermData 1 (-half) 0 2 1 0,
    borrowTermData 1 quarter 1 1 0 0,
    borrowTermData 1 (-quarter) 1 2 0 0,
    borrowTermData 1 quarter 2 1 0 0,
    borrowTermData 1 (-half) 2 1 1 0,
    borrowTermData 1 (-quarter) 2 2 0 0,
    borrowTermData 1 half 2 2 1 0 ]

def borrowTerms1 : List (Monomial F matrixCount) :=
  borrowTermData1.map Term.toMonomial

/-- Expanded relation for the two-trit bound class 2. -/
def borrowTermData2 : List Term :=
  [ borrowTermData 2 (-1) 0 0 0 0,
    borrowTermData 2 1 0 0 0 1,
    borrowTermData 2 (-half) 0 1 0 0,
    borrowTermData 2 half 0 2 0 0,
    borrowTermData 2 quarter 1 1 1 0,
    borrowTermData 2 (-quarter) 1 2 1 0,
    borrowTermData 2 quarter 2 1 1 0,
    borrowTermData 2 (-quarter) 2 2 1 0 ]

def borrowTerms2 : List (Monomial F matrixCount) :=
  borrowTermData2.map Term.toMonomial

/-- Expanded relation for the two-trit bound class 3. -/
def borrowTermData3 : List Term :=
  [ borrowTermData 3 (-1) 0 0 0 0,
    borrowTermData 3 1 0 0 0 1,
    borrowTermData 3 (-half) 0 1 0 0,
    borrowTermData 3 half 0 2 0 0,
    borrowTermData 3 (-half) 1 0 0 0,
    borrowTermData 3 half 1 0 1 0,
    borrowTermData 3 half 1 2 0 0,
    borrowTermData 3 (-half) 1 2 1 0,
    borrowTermData 3 half 2 0 0 0,
    borrowTermData 3 (-half) 2 0 1 0,
    borrowTermData 3 (-half) 2 2 0 0,
    borrowTermData 3 half 2 2 1 0 ]

def borrowTerms3 : List (Monomial F matrixCount) :=
  borrowTermData3.map Term.toMonomial

/-- Expanded relation for the two-trit bound class 4. -/
def borrowTermData4 : List Term :=
  [ borrowTermData 4 1 0 0 0 1,
    borrowTermData 4 (-1) 0 0 1 0,
    borrowTermData 4 (-half) 0 1 0 0,
    borrowTermData 4 (-half) 0 2 0 0,
    borrowTermData 4 1 0 2 1 0,
    borrowTermData 4 (-half) 1 0 0 0,
    borrowTermData 4 half 1 2 0 0,
    borrowTermData 4 (-half) 2 0 0 0,
    borrowTermData 4 1 2 0 1 0,
    borrowTermData 4 half 2 2 0 0,
    borrowTermData 4 (-1) 2 2 1 0 ]

def borrowTerms4 : List (Monomial F matrixCount) :=
  borrowTermData4.map Term.toMonomial

/-- The sole stored term table, before function-valued exponent erasure. -/
def termData : List Term :=
  baseTermData ++ borrowTermData0 ++ borrowTermData1 ++ borrowTermData2 ++
    borrowTermData3 ++ borrowTermData4

/-- Exact sparse term order used by the Lean-owned selective compiler. -/
def terms : List (Monomial F matrixCount) :=
  baseTerms ++ borrowTerms0 ++ borrowTerms1 ++ borrowTerms2 ++
    borrowTerms3 ++ borrowTerms4

theorem termData_toMonomial : termData.map Term.toMonomial = terms := by
  simp only [termData, List.map_append, terms, baseTerms, borrowTerms0,
    borrowTerms1, borrowTerms2, borrowTerms3, borrowTerms4]

@[simp] theorem baseTerms_length : baseTerms.length = 21 := by
  rfl

@[simp] theorem terms_length : terms.length = 74 := by
  rfl

theorem termData_length : termData.length = 74 := by
  have sameLength := congrArg List.length termData_toMonomial
  simpa only [List.length_map, terms_length] using sameLength

theorem term_totalDegree_le_eight
    (candidate : Monomial F matrixCount)
    (member : candidate ∈ terms) :
    candidate.totalDegree ≤ 8 := by
  have degreeMember :
      candidate.totalDegree ∈ terms.map Monomial.totalDegree :=
    List.mem_map_of_mem member
  simp [terms, baseTerms, borrowTerms0, borrowTerms1, borrowTerms2,
    borrowTerms3, borrowTerms4, borrowTerm, borrowPowers, classExponent,
    powers, PortExponents.totalDegree, Term.toMonomial, sboxTermData, baseTermData,
    borrowTermData, borrowTermData0, borrowTermData1, borrowTermData2,
    borrowTermData3, borrowTermData4] at degreeMember
  omega

/-- Every production monomial has positive degree, so the fixed polynomial
has no constant term. -/
theorem term_totalDegree_pos
    (candidate : Monomial F matrixCount)
    (member : candidate ∈ terms) :
    0 < candidate.totalDegree := by
  have degreeMember :
      candidate.totalDegree ∈ terms.map Monomial.totalDegree :=
    List.mem_map_of_mem member
  simp [terms, baseTerms, borrowTerms0, borrowTerms1, borrowTerms2,
    borrowTerms3, borrowTerms4, borrowTerm, borrowPowers, classExponent,
    powers, PortExponents.totalDegree, Term.toMonomial, sboxTermData, baseTermData,
    borrowTermData, borrowTermData0, borrowTermData1, borrowTermData2,
    borrowTermData3, borrowTermData4] at degreeMember
  omega

theorem term_zeroPort
    (candidate : Monomial F matrixCount)
    (member : candidate ∈ terms) :
    candidate.exponents zeroPort = 0 := by
  have valueMember :
      candidate.exponents zeroPort ∈
        terms.map (fun current => current.exponents zeroPort) :=
    List.mem_map_of_mem member
  simpa [terms, baseTerms, borrowTerms0, borrowTerms1, borrowTerms2,
    borrowTerms3, borrowTerms4, borrowTerm, sboxTerm, Term.toMonomial, sboxTermData,
    baseTermData, borrowTermData, borrowTermData0, borrowTermData1, borrowTermData2,
    borrowTermData3, borrowTermData4] using valueMember

/-- The sole production CCS polynomial after selective low-norm lowering. -/
def polynomial : ConstraintPolynomial F matrixCount where
  degreeBound := 9
  terms := terms
  termsBelowDegree := by
    intro candidate member
    exact Nat.lt_succ_of_le (term_totalDegree_le_eight candidate member)

@[simp] theorem polynomial_terms : polynomial.terms = terms := by
  rfl

@[simp] theorem polynomial_degreeBound : polynomial.degreeBound = 9 := by
  rfl

theorem sboxTerm_mem : sboxTerm ∈ polynomial.terms := by
  simp [polynomial, terms, baseTerms, baseTermData, Term.toMonomial, sboxTerm, sboxTermData]

/-- The explicit syntax, not metadata, fixes the degree-nine PiCCS ceiling. -/
theorem polynomial_canonicalEqualityGatedDegreeBound :
    polynomial.canonicalEqualityGatedDegreeBound = 9 := by
  apply Nat.le_antisymm
  · simpa [polynomial] using
      ConstraintPolynomial.canonicalEqualityGatedDegreeBound_le_degreeBound
        polynomial
  · have lower :=
      ConstraintPolynomial.term_totalDegree_succ_le_canonicalEqualityGatedDegreeBound
        polynomial sboxTerm sboxTerm_mem
    simpa using lower

/-- No term can read the final canonical-zero `Eval_A` slot. -/
theorem polynomial_zeroPort
    (candidate : Monomial F matrixCount)
    (member : candidate ∈ polynomial.terms) :
    candidate.exponents zeroPort = 0 := by
  exact term_zeroPort candidate member

end NightstreamFPrime.Spec.ProductionRelation.SelectivePolynomial
