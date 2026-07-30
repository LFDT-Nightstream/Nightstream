import Nightstream.Implementation.R1CS.Canonical.KCompositeRowSupport
import Nightstream.Implementation.R1CS.Canonical.KMulChainHonest

/-!
Contract: honest completeness for the canonical explicit sparse-polynomial
program.

Owns: the sequential term witness and its exact composition across the
degree-sized term allocations.  The only caller facts are positivity of the
allocation base and placement of every point coordinate before it.

Does not own: the polynomial or its exponents.  They remain typed verifier
data and determine the emitted multiplication chains definitionally.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Canonical.KSparsePolynomialHonest

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal
open Nightstream.Implementation.R1CS.Canonical.KMul
open Nightstream.Implementation.R1CS.Canonical.KHornerHonest
open Nightstream.Implementation.R1CS.Canonical.KHornerSupport
open Nightstream.Implementation.R1CS.Canonical.KCompositeRowSupport
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

private abbrev ConcreteK := Nightstream.SuperNeo.Concrete.K

private theorem expandedFactors_below
    {matrixCount base : Nat}
    (point : Fin matrixCount → Carried)
    (pointBelow : ∀ index, CarriedBelow (point index) base)
    (monomial : CCSResidualTable.Monomial ConcreteK matrixCount) :
    ∀ factor ∈ KSparsePolynomial.expandedFactors point monomial,
      CarriedBelow factor base := by
  intro factor member
  rcases List.mem_flatMap.1 member with ⟨index, _, inReplicate⟩
  have same : factor = point index := List.eq_of_mem_replicate inReplicate
  subst factor
  exact pointBelow index

/-- Apply one term chain and continue at the next exact degree offset. -/
def termsWitness
    {matrixCount : Nat}
    (assignment : Nat → Nat)
    (point : Fin matrixCount → Carried) :
    List (CCSResidualTable.Monomial ConcreteK matrixCount) →
      Nat → Nat → (Nat → Nat)
  | [], _, _ => assignment
  | monomial :: rest, base, offset =>
      let head :=
        KMulChainHonest.witness assignment
          (KLinear.constantCarried monomial.coefficient)
          (KSparsePolynomial.expandedFactors point monomial)
          (base + 3 * offset) 0
      termsWitness head point rest base
        (offset + monomial.totalDegree)

theorem termsWitness_off_before
    {matrixCount : Nat}
    (assignment : Nat → Nat)
    (point : Fin matrixCount → Carried) :
    ∀ (terms : List (CCSResidualTable.Monomial ConcreteK matrixCount))
      (base offset column : Nat),
      column < base + 3 * offset →
      termsWitness assignment point terms base offset column =
        assignment column
  | [], _, _, _, _ => rfl
  | monomial :: rest, base, offset, column, below => by
      rw [termsWitness,
        termsWitness_off_before
          (KMulChainHonest.witness assignment
            (KLinear.constantCarried monomial.coefficient)
            (KSparsePolynomial.expandedFactors point monomial)
            (base + 3 * offset) 0)
          point rest base (offset + monomial.totalDegree) column (by omega),
        KMulChainHonest.witness_off_before assignment
          (KLinear.constantCarried monomial.coefficient)
          (base + 3 * offset)
          (KSparsePolynomial.expandedFactors point monomial)
          0 column (by simpa using below)]

private theorem termRows_below_next
    {matrixCount : Nat}
    (point : Fin matrixCount → Carried)
    (monomial : CCSResidualTable.Monomial ConcreteK matrixCount)
    (base offset : Nat)
    (positive : 0 < base)
    (pointBelow : ∀ index, CarriedBelow (point index) base) :
    RowsBelow
      (KSparsePolynomial.termRows point monomial (base + 3 * offset))
      (base + 3 * (offset + monomial.totalDegree)) := by
  unfold KSparsePolynomial.termRows
  apply mulChain_rows_below
  · exact constant_below monomial.coefficient _
      (by omega)
  · intro factor member
    exact carried_mono
      (expandedFactors_below point pointBelow monomial factor member)
      (by omega)
  · rw [KSparsePolynomial.expandedFactors_length]
    omega

private theorem satisfies_append
    {left right : List Row} {assignment : Nat → Nat}
    (leftSatisfied : Satisfies left assignment)
    (rightSatisfied : Satisfies right assignment) :
    Satisfies (left ++ right) assignment := by
  intro row member
  exact (List.mem_append.1 member).elim
    (leftSatisfied row) (rightSatisfied row)

theorem termsRows_honest
    {matrixCount base : Nat}
    (assignment : Nat → Nat)
    (point : Fin matrixCount → Carried)
    (positive : 0 < base)
    (pointBelow : ∀ index, CarriedBelow (point index) base) :
    ∀ (terms : List (CCSResidualTable.Monomial ConcreteK matrixCount))
      (offset : Nat),
      Satisfies (KSparsePolynomial.termsRows point terms base offset)
        (termsWitness assignment point terms base offset)
  | [], _ => by
      intro row member
      simp [KSparsePolynomial.termsRows] at member
  | monomial :: rest, offset => by
      let headWitness :=
        KMulChainHonest.witness assignment
          (KLinear.constantCarried monomial.coefficient)
          (KSparsePolynomial.expandedFactors point monomial)
          (base + 3 * offset) 0
      let finalWitness :=
        termsWitness headWitness point rest base
          (offset + monomial.totalDegree)
      have headSatisfied :
          Satisfies
            (KSparsePolynomial.termRows point monomial
              (base + 3 * offset))
            headWitness := by
        unfold KSparsePolynomial.termRows
        exact KMulChainHonest.witness_satisfies_from_base assignment
          (KLinear.constantCarried monomial.coefficient)
          (KSparsePolynomial.expandedFactors point monomial)
          (base + 3 * offset)
          (constant_below monomial.coefficient _ (by omega)).1
          (constant_below monomial.coefficient _ (by omega)).2
          (by
            intro factor member
            exact carried_mono
              (expandedFactors_below point pointBelow monomial factor member)
              (by omega))
      have headPreserved :
          Satisfies
            (KSparsePolynomial.termRows point monomial
              (base + 3 * offset))
            finalWitness := by
        apply satisfies_extend _ headWitness finalWitness
        · intro row member column mentioned
          symm
          apply termsWitness_off_before
          exact termRows_below_next point monomial base offset positive
            pointBelow row member column mentioned
        · exact headSatisfied
      have tailSatisfied :
          Satisfies
            (KSparsePolynomial.termsRows point rest base
              (offset + monomial.totalDegree))
            finalWitness :=
        termsRows_honest headWitness point positive pointBelow rest
          (offset + monomial.totalDegree)
      simpa [KSparsePolynomial.termsRows, termsWitness, headWitness,
        finalWitness] using satisfies_append headPreserved tailSatisfied

def witness {matrixCount : Nat}
    (input : KSparsePolynomial.Input matrixCount)
    (assignment : Nat → Nat) : Nat → Nat :=
  termsWitness assignment input.point input.polynomial.terms
    input.frameBase 0

theorem witness_off_block
    {matrixCount : Nat}
    (input : KSparsePolynomial.Input matrixCount)
    (assignment : Nat → Nat)
    (column : Nat) (below : column < input.frameBase) :
    witness input assignment column = assignment column :=
  termsWitness_off_before assignment input.point input.polynomial.terms
    input.frameBase 0 column (by simpa using below)

/-- Every authoritative point has one satisfying sparse-polynomial execution.
No polynomial evaluation result is supplied. -/
theorem rows_honest
    {matrixCount : Nat}
    (input : KSparsePolynomial.Input matrixCount)
    (assignment : Nat → Nat)
    (positive : 0 < input.frameBase)
    (pointBelow :
      ∀ index, CarriedBelow (input.point index) input.frameBase) :
    Satisfies (KSparsePolynomial.rows input) (witness input assignment) := by
  exact termsRows_honest assignment input.point positive pointBelow
    input.polynomial.terms 0

end Nightstream.Implementation.R1CS.Canonical.KSparsePolynomialHonest
