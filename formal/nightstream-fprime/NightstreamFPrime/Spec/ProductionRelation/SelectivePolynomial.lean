import Mathlib.Tactic
import NightstreamFPrime.Spec.Algebra
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.CCSResidualTable
import NightstreamFPrime.Spec.Profile

/-!
Paper authority: SuperNeo v1.2, Definitions 19--21 and Section 7.3.
Compiler obligation: one low-norm CCS gate for the fixed Nightstream circuit.

Inputs:
- 7 matrix images, one for each selective port;
- Pad is not an input here and remains the separate `Eval_K` family.

Constraint groups:
- C1: Boolean, multiplication, and S-box trace checks, gated by the general
  selector;
- C2: two packed evaluation products, gated by the evaluation selector.

Parent coverage:
- `ProductionRelation.polynomial`;
- the `F` term in SuperNeo v1.2 PiCCS;
- the production CCS matrix family, without Pad-as-matrix-zero compression.
-/

namespace NightstreamFPrime.Spec.ProductionRelation.SelectivePolynomial

open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.CCSResidualTable

/-- The `Eval_A` arity, read from the production profile. -/
def matrixCount : Nat := productionProfile.ccsMatrices

/-- Number of matrix slots that carry selective compiler ports. -/
def meaningfulPortCount : Nat := 7

/-- Exponents for the 7 selective ports. -/
structure PortExponents where
  bit : Nat := 0
  generalSelector : Nat := 0
  a : Nat := 0
  b : Nat := 0
  c : Nat := 0
  sboxInput : Nat := 0
  evalSelector : Nat := 0

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
  | 6 => powers.evalSelector
  | _ => 0

/-- Total degree in the 7 ports. -/
def totalDegree (powers : PortExponents) : Nat :=
  [powers.bit, powers.generalSelector, powers.a, powers.b, powers.c,
    powers.sboxInput, powers.evalSelector].sum

end PortExponents

/-- Compact constructor used by every explicit selective monomial. -/
def powers
    (bit : Nat := 0)
    (generalSelector : Nat := 0)
    (a : Nat := 0)
    (b : Nat := 0)
    (c : Nat := 0)
    (sboxInput : Nat := 0)
    (evalSelector : Nat := 0) : PortExponents :=
  { bit, generalSelector, a, b, c, sboxInput, evalSelector }

/-- A compiler term retains its named exponent record before semantic erasure. -/
structure Term where
  coefficient : F
  powers : PortExponents

/-- One sparse monomial over the fixed 7-slot production relation. -/
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

/-- The degree-eight S-box trace term. It fixes the maximum CCS degree. -/
def sboxTermData : Term :=
  ⟨1, powers (generalSelector := 1) (sboxInput := 7)⟩

def sboxTerm : Monomial F matrixCount := sboxTermData.toMonomial

@[simp] theorem sboxTerm_totalDegree : sboxTerm.totalDegree = 8 := by
  simp [sboxTerm, sboxTermData, Term.toMonomial, powers, PortExponents.totalDegree]

/-- Boolean, multiplication, and S-box terms under the general selector, then
two evaluation products under the evaluation selector. No row sets both
selectors. -/
def termData : List Term :=
  [ Term.mk 1 (powers (bit := 2) (generalSelector := 1)),
    Term.mk (-1) (powers (bit := 1) (generalSelector := 1)),
    Term.mk 1 (powers (generalSelector := 1) (a := 1) (b := 1)),
    Term.mk (-1) (powers (generalSelector := 1) (c := 1)),
    sboxTermData,
    Term.mk (-1) (powers (c := 1) (evalSelector := 1)),
    Term.mk 1 (powers (bit := 1) (a := 1) (evalSelector := 1)),
    Term.mk 1 (powers (b := 1) (sboxInput := 1) (evalSelector := 1)) ]

/-- Exact sparse term order used by the Lean-owned selective compiler. -/
def terms : List (Monomial F matrixCount) :=
  termData.map Term.toMonomial

theorem termData_toMonomial : termData.map Term.toMonomial = terms := by
  rfl

@[simp] theorem terms_length : terms.length = 8 := by
  rfl

theorem termData_length : termData.length = 8 := by
  rfl

theorem term_totalDegree_le_eight
    (candidate : Monomial F matrixCount)
    (member : candidate ∈ terms) :
    candidate.totalDegree ≤ 8 := by
  have degreeMember :
      candidate.totalDegree ∈ terms.map Monomial.totalDegree :=
    List.mem_map_of_mem member
  simp [terms, powers, PortExponents.totalDegree, Term.toMonomial,
    sboxTermData, termData] at degreeMember
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
  simp [terms, powers, PortExponents.totalDegree, Term.toMonomial,
    sboxTermData, termData] at degreeMember
  omega

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
  simp [polynomial, terms, termData, Term.toMonomial, sboxTerm, sboxTermData]

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

end NightstreamFPrime.Spec.ProductionRelation.SelectivePolynomial
