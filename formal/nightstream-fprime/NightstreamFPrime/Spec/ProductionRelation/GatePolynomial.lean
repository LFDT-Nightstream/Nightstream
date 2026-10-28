import Mathlib.Tactic
import NightstreamFPrime.Spec.Algebra
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.CCSResidualTable
import NightstreamFPrime.Spec.Profile

/-!
Paper authority: SuperNeo v1.2, Definitions 19--20 and Section 7.3.
Compiler obligation: one low-norm CCS gate for the fixed Nightstream circuit.

Inputs:
- 4 matrix images, one for each port `a`, `b`, `c`, and the S-box input;
- Pad is not an input here and remains the separate `Eval_K` family.

Gate: `f = a * b - c + sboxInput ^ 7`. A row sets the product pair or the
S-box input, never both. The gate has no constant term, so zero padding rows
pass.

Parent coverage:
- `ProductionRelation.polynomial`;
- the `F` term in SuperNeo v1.2 PiCCS;
- the production CCS matrix family, without Pad-as-matrix-zero compression.
-/

namespace NightstreamFPrime.Spec.ProductionRelation.GatePolynomial

open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.CCSResidualTable

def matrixCount : Nat := productionProfile.ccsMatrices

def meaningfulPortCount : Nat := 4

structure PortExponents where
  a : Nat := 0
  b : Nat := 0
  c : Nat := 0
  sboxInput : Nat := 0

namespace PortExponents

def get (powers : PortExponents) (index : Fin matrixCount) : Nat :=
  match index.val with
  | 0 => powers.a
  | 1 => powers.b
  | 2 => powers.c
  | 3 => powers.sboxInput
  | _ => 0

def totalDegree (powers : PortExponents) : Nat :=
  [powers.a, powers.b, powers.c, powers.sboxInput].sum

end PortExponents

def powers
    (a : Nat := 0)
    (b : Nat := 0)
    (c : Nat := 0)
    (sboxInput : Nat := 0) : PortExponents :=
  { a, b, c, sboxInput }

structure Term where
  coefficient : F
  powers : PortExponents

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

def sboxTermData : Term :=
  ⟨1, powers (sboxInput := 7)⟩

def sboxTerm : Monomial F matrixCount := sboxTermData.toMonomial

@[simp] theorem sboxTerm_totalDegree : sboxTerm.totalDegree = 7 := by
  simp [sboxTerm, sboxTermData, Term.toMonomial, powers, PortExponents.totalDegree]

def termData : List Term :=
  [ Term.mk 1 (powers (a := 1) (b := 1)),
    Term.mk (-1) (powers (c := 1)),
    sboxTermData ]

def terms : List (Monomial F matrixCount) :=
  termData.map Term.toMonomial

theorem termData_toMonomial : termData.map Term.toMonomial = terms := by
  rfl

@[simp] theorem terms_length : terms.length = 3 := by
  rfl

theorem termData_length : termData.length = 3 := by
  rfl

theorem term_totalDegree_le_seven
    (candidate : Monomial F matrixCount)
    (member : candidate ∈ terms) :
    candidate.totalDegree ≤ 7 := by
  have degreeMember :
      candidate.totalDegree ∈ terms.map Monomial.totalDegree :=
    List.mem_map_of_mem member
  simp [terms, powers, PortExponents.totalDegree, Term.toMonomial,
    sboxTermData, termData] at degreeMember
  omega

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

def polynomial : ConstraintPolynomial F matrixCount where
  degreeBound := 8
  terms := terms
  termsBelowDegree := by
    intro candidate member
    exact Nat.lt_succ_of_le (term_totalDegree_le_seven candidate member)

@[simp] theorem polynomial_terms : polynomial.terms = terms := by
  rfl

@[simp] theorem polynomial_degreeBound : polynomial.degreeBound = 8 := by
  rfl

theorem sboxTerm_mem : sboxTerm ∈ polynomial.terms := by
  simp [polynomial, terms, termData, Term.toMonomial, sboxTerm, sboxTermData]

theorem polynomial_canonicalEqualityGatedDegreeBound :
    polynomial.canonicalEqualityGatedDegreeBound = 8 := by
  apply Nat.le_antisymm
  · simpa [polynomial] using
      ConstraintPolynomial.canonicalEqualityGatedDegreeBound_le_degreeBound
        polynomial
  · have lower :=
      ConstraintPolynomial.term_totalDegree_succ_le_canonicalEqualityGatedDegreeBound
        polynomial sboxTerm sboxTerm_mem
    simpa using lower

end NightstreamFPrime.Spec.ProductionRelation.GatePolynomial
