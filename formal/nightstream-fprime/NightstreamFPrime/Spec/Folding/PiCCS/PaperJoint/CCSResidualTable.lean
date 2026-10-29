import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.FiniteSumAlgebra

/-! SuperNeo v1.2 Section 7.3, Appendix B.2. -/

namespace NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.CCSResidualTable

universe uField

open NightstreamFPrime.Spec.SumCheck
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open PaperLinearAlgebra

structure Monomial (Field : Type uField) (matrixCount : Nat) where
  coefficient : Field
  exponents : Fin matrixCount -> Nat

namespace Monomial

def totalDegree
    {Field : Type uField}
    {matrixCount : Nat}
    (monomial : Monomial Field matrixCount) : Nat :=
  ((canonicalFinIndices matrixCount).map monomial.exponents).sum

end Monomial

/-- SuperNeo v1.2 Definition 19. -/
structure ConstraintPolynomial (Field : Type uField) (matrixCount : Nat) where
  degreeBound : Nat
  terms : List (Monomial Field matrixCount)
  termsBelowDegree :
    ∀ term, term ∈ terms -> term.totalDegree < degreeBound

namespace ConstraintPolynomial

universe uItem

private theorem initial_le_foldl_max
    {Item : Type uItem}
    (value : Item -> Nat)
    (items : List Item)
    (initial : Nat) :
    initial <= items.foldl (fun current item => Nat.max current (value item)) initial := by
  induction items generalizing initial with
  | nil => exact Nat.le_refl initial
  | cons item items inductionHypothesis =>
      exact Nat.le_trans (Nat.le_max_left initial (value item))
        (inductionHypothesis (Nat.max initial (value item)))

private theorem value_le_foldl_max_of_mem
    {Item : Type uItem}
    (value : Item -> Nat)
    (items : List Item)
    (initial : Nat)
    (item : Item)
    (member : item ∈ items) :
    value item <=
      items.foldl (fun current next => Nat.max current (value next)) initial := by
  induction items generalizing initial with
  | nil => simp at member
  | cons head tail inductionHypothesis =>
      simp only [List.mem_cons] at member
      rcases member with rfl | member
      · exact Nat.le_trans (Nat.le_max_right initial (value item))
          (initial_le_foldl_max value tail
            (Nat.max initial (value item)))
      · exact inductionHypothesis (Nat.max initial (value head)) member

def canonicalEqualityGatedDegreeBound
    {Field : Type uField}
    {matrixCount : Nat}
    (polynomial : ConstraintPolynomial Field matrixCount) : Nat :=
  polynomial.terms.foldl
    (fun current term => Nat.max current (term.totalDegree + 1)) 0

theorem canonicalEqualityGatedDegreeBound_eq_of_terms_eq
    {Field : Type uField}
    {matrixCount : Nat}
    (left right : ConstraintPolynomial Field matrixCount)
    (terms : left.terms = right.terms) :
    left.canonicalEqualityGatedDegreeBound =
      right.canonicalEqualityGatedDegreeBound := by
  simp [canonicalEqualityGatedDegreeBound, terms]

theorem term_totalDegree_succ_le_canonicalEqualityGatedDegreeBound
    {Field : Type uField}
    {matrixCount : Nat}
    (polynomial : ConstraintPolynomial Field matrixCount)
    (term : Monomial Field matrixCount)
    (member : term ∈ polynomial.terms) :
    term.totalDegree + 1 <=
      polynomial.canonicalEqualityGatedDegreeBound := by
  exact value_le_foldl_max_of_mem
    (fun current : Monomial Field matrixCount => current.totalDegree + 1)
    polynomial.terms 0 term member

private theorem foldl_max_le_of_forall_le
    {Item : Type uItem}
    (value : Item -> Nat)
    (bound : Nat) :
    forall (items : List Item) (initial : Nat),
      initial <= bound ->
      (forall item, item ∈ items -> value item <= bound) ->
      items.foldl (fun current item => Nat.max current (value item)) initial <=
        bound
  | [], _, initialLe, _ => initialLe
  | item :: items, initial, initialLe, valuesLe => by
      apply foldl_max_le_of_forall_le value bound items
        (Nat.max initial (value item))
      · exact (Nat.max_le).2
          ⟨initialLe, valuesLe item (by simp)⟩
      · intro tail tailMember
        exact valuesLe tail (by simp [tailMember])

theorem canonicalEqualityGatedDegreeBound_le_degreeBound
    {Field : Type uField}
    {matrixCount : Nat}
    (polynomial : ConstraintPolynomial Field matrixCount) :
    polynomial.canonicalEqualityGatedDegreeBound <= polynomial.degreeBound := by
  unfold canonicalEqualityGatedDegreeBound
  apply foldl_max_le_of_forall_le
  · exact Nat.zero_le _
  · intro term termMember
    exact Nat.succ_le_of_lt
      (polynomial.termsBelowDegree term termMember)

end ConstraintPolynomial

/-- SuperNeo v1.2 Definition 19. -/
structure Structure
    (Field : Type uField)
    (shape : Shape)
    (columns : Nat) where
  matrices : Fin shape.matrixCount ->
    BooleanMatrix Field shape.cubeVariables columns
  constraintPolynomial : ConstraintPolynomial Field shape.matrixCount

def pow
    {Field : Type uField}
    (ops : InterpolationOps Field)
    (value : Field) : Nat -> Field
  | 0 => ops.one
  | exponent + 1 => ops.mul (pow ops value exponent) value

def evaluateMonomial
    {Field : Type uField}
    (ops : InterpolationOps Field)
    {matrixCount : Nat}
    (monomial : Monomial Field matrixCount)
    (point : Fin matrixCount -> Field) : Field :=
  (canonicalFinIndices matrixCount).foldl
    (fun accumulated index =>
      ops.mul accumulated
        (pow ops (point index) (monomial.exponents index)))
    monomial.coefficient

def evaluatePolynomial
    {Field : Type uField}
    (ops : InterpolationOps Field)
    {matrixCount : Nat}
    (polynomial : ConstraintPolynomial Field matrixCount)
    (point : Fin matrixCount -> Field) : Field :=
  polynomial.terms.foldl
    (fun accumulated monomial =>
      ops.add accumulated (evaluateMonomial ops monomial point))
    ops.zero

private theorem foldl_add_eq_add_finiteSum
    {Field : Type uField}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (values : List Field)
    (initial : Field) :
    values.foldl ops.add initial =
      ops.add initial (BooleanTable.finiteSum ops values) := by
  induction values generalizing initial with
  | nil => exact (laws.add_zero initial).symm
  | cons value values inductionHypothesis =>
      rw [List.foldl_cons, inductionHypothesis]
      exact laws.add_assoc initial value _

theorem evaluatePolynomial_eq_sumMap
    {Field : Type uField}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    {matrixCount : Nat}
    (polynomial : ConstraintPolynomial Field matrixCount)
    (point : Fin matrixCount -> Field) :
    evaluatePolynomial ops polynomial point =
      FiniteSumAlgebra.sumMap ops polynomial.terms fun monomial =>
        evaluateMonomial ops monomial point := by
  unfold evaluatePolynomial FiniteSumAlgebra.sumMap
  rw [← List.foldl_map]
  rw [foldl_add_eq_add_finiteSum ops laws]
  rw [laws.zero_add]

def matrixImagesAt
    {Field : Type uField}
    (ops : InterpolationOps Field)
    {shape : Shape}
    {columns : Nat}
    (system : Structure Field shape columns)
    (assignment : Assignment Field columns)
    (vertex : BooleanVertex shape.cubeVariables) :
    Fin shape.matrixCount -> Field :=
  fun matrix =>
    matrixVectorAt ops (system.matrices matrix) assignment vertex

def residualAt
    {Field : Type uField}
    (ops : InterpolationOps Field)
    {shape : Shape}
    {columns : Nat}
    (system : Structure Field shape columns)
    (assignment : Assignment Field columns)
    (vertex : BooleanVertex shape.cubeVariables) : Field :=
  evaluatePolynomial ops system.constraintPolynomial
    (matrixImagesAt ops system assignment vertex)

/-- SuperNeo Definition 20, Lemma 9 Item 1. -/
def ConstraintSatisfied
    {Field : Type uField}
    (ops : InterpolationOps Field)
    {shape : Shape}
    {columns : Nat}
    (system : Structure Field shape columns)
    (assignment : Assignment Field columns) : Prop :=
  ∀ vertex, residualAt ops system assignment vertex = ops.zero

def residualTable
    {Field : Type uField}
    (ops : InterpolationOps Field)
    {shape : Shape}
    {columns : Nat}
    (system : Structure Field shape columns)
    (assignment : Assignment Field columns) :
    BooleanTable Field shape.cubeVariables :=
  BooleanTable.tabulate (residualAt ops system assignment)

theorem residualTable_entries_eq
    {Field : Type uField}
    (ops : InterpolationOps Field)
    {shape : Shape}
    {columns : Nat}
    (system : Structure Field shape columns)
    (assignment : Assignment Field columns) :
    (residualTable ops system assignment).entries =
      (BooleanVertex.all shape.cubeVariables).map
        (residualAt ops system assignment) := by
  exact BooleanTable.entries_tabulate _

theorem residualTable_allEntriesZero_iff_constraintSatisfied
    {Field : Type uField}
    (ops : InterpolationOps Field)
    {shape : Shape}
    {columns : Nat}
    (system : Structure Field shape columns)
    (assignment : Assignment Field columns) :
    (residualTable ops system assignment).AllEntriesZero ops ↔
      ConstraintSatisfied ops system assignment := by
  exact BooleanTable.tabulate_allEntriesZero_iff ops _

theorem residualPolynomial_coefficientZero_iff_constraintSatisfied
    {Field : Type uField}
    (ops : InterpolationOps Field)
    (laws : InterpolationZeroLaws ops)
    {shape : Shape}
    {columns : Nat}
    (system : Structure Field shape columns)
    (assignment : Assignment Field columns) :
    AlphaPolynomial.CoefficientZero ops.toOps
        ((residualTable ops system assignment).toAlphaPolynomial ops) ↔
      ConstraintSatisfied ops system assignment := by
  exact Iff.trans
    (NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.BooleanTable.toAlphaPolynomial_coefficientZero_iff_allEntriesZero
        ops laws (residualTable ops system assignment))
    (residualTable_allEntriesZero_iff_constraintSatisfied
      ops system assignment)

structure FreshBatch
    (Field : Type uField)
    (shape : Shape)
    (columns : Nat) where
  system : Structure Field shape columns
  assignments : Fin shape.freshCount -> Assignment Field columns

namespace FreshBatch

def residualTables
    {Field : Type uField}
    {shape : Shape}
    {columns : Nat}
    (ops : InterpolationOps Field)
    (batch : FreshBatch Field shape columns) :
    Fin shape.freshCount -> BooleanTable Field shape.cubeVariables :=
  fun source => residualTable ops batch.system (batch.assignments source)

def AllConstraintsSatisfied
    {Field : Type uField}
    {shape : Shape}
    {columns : Nat}
    (ops : InterpolationOps Field)
    (batch : FreshBatch Field shape columns) : Prop :=
  ∀ source, ConstraintSatisfied ops batch.system (batch.assignments source)

theorem allResidualTablesZero_iff_allConstraintsSatisfied
    {Field : Type uField}
    {shape : Shape}
    {columns : Nat}
    (ops : InterpolationOps Field)
    (batch : FreshBatch Field shape columns) :
    (∀ source, (batch.residualTables ops source).AllEntriesZero ops) ↔
      batch.AllConstraintsSatisfied ops := by
  constructor
  · intro allZero source
    exact (residualTable_allEntriesZero_iff_constraintSatisfied
      ops batch.system (batch.assignments source)).mp (allZero source)
  · intro allSatisfied source
    exact (residualTable_allEntriesZero_iff_constraintSatisfied
      ops batch.system (batch.assignments source)).mpr
        (allSatisfied source)

end FreshBatch

end NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.CCSResidualTable
