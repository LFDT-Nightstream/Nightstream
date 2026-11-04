import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.BooleanDomain
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.NormRange

/-! SuperNeo v1.2 Section 7.3, Appendix B.2. -/

namespace NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.NormResidualTable

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

abbrev BooleanAssignment (variables : Nat) := BooleanVertex variables -> F

structure BaseZeroAgreement (ops : InterpolationOps F) : Prop where
  zero_eq : ops.zero = (0 : F)

def StrictNormBounded
    {variables : Nat}
    (assignment : BooleanAssignment variables) : Prop :=
  ∀ vertex, centeredMagnitude (assignment vertex) < 2

def orderedValues
    {variables : Nat}
    (assignment : BooleanAssignment variables) : List F :=
  (BooleanVertex.all variables).map assignment

theorem strictNormBounded_iff_orderedValues_normBounded
    {variables : Nat}
    (assignment : BooleanAssignment variables) :
    StrictNormBounded assignment ↔ normBounded 2 (orderedValues assignment) := by
  constructor
  · intro bounded value member
    rw [orderedValues] at member
    rcases List.mem_map.mp member with ⟨vertex, _, rfl⟩
    exact bounded vertex
  · intro bounded vertex
    exact bounded (assignment vertex) (by
      rw [orderedValues]
      exact List.mem_map.mpr
        ⟨vertex, BooleanVertex.mem_all vertex, rfl⟩)

def residualTable
    {variables : Nat}
    (assignment : BooleanAssignment variables) : BooleanTable F variables :=
  BooleanTable.tabulate fun vertex =>
    NormRange.cubicResidual (assignment vertex)

theorem residualTable_entries_eq
    {variables : Nat}
    (assignment : BooleanAssignment variables) :
    (residualTable assignment).entries =
      (BooleanVertex.all variables).map fun vertex =>
        NormRange.cubicResidual (assignment vertex) := by
  exact BooleanTable.entries_tabulate _

theorem residualTable_allEntriesZero_iff_strictNormBounded
    (ops : InterpolationOps F)
    (zeroAgreement : BaseZeroAgreement ops)
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    {variables : Nat}
    (assignment : BooleanAssignment variables) :
    (residualTable assignment).AllEntriesZero ops ↔
      StrictNormBounded assignment := by
  unfold residualTable
  rw [BooleanTable.tabulate_allEntriesZero_iff]
  constructor
  · intro allZero vertex
    apply (NormRange.cubicResidual_eq_zero_iff_strictNormTwo
      noZeroDivisors (assignment vertex)).mp
    rw [← zeroAgreement.zero_eq]
    exact allZero vertex
  · intro bounded vertex
    rw [zeroAgreement.zero_eq]
    exact (NormRange.cubicResidual_eq_zero_iff_strictNormTwo
      noZeroDivisors (assignment vertex)).mpr (bounded vertex)

theorem residualTable_allEntriesZero_iff_normBounded
    (ops : InterpolationOps F)
    (zeroAgreement : BaseZeroAgreement ops)
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    {variables : Nat}
    (assignment : BooleanAssignment variables) :
    (residualTable assignment).AllEntriesZero ops ↔
      normBounded 2 (orderedValues assignment) := by
  exact Iff.trans
    (residualTable_allEntriesZero_iff_strictNormBounded
      ops zeroAgreement noZeroDivisors assignment)
    (strictNormBounded_iff_orderedValues_normBounded assignment)

theorem residualPolynomial_coefficientZero_iff_strictNormBounded
    (ops : InterpolationOps F)
    (zeroLaws : InterpolationZeroLaws ops)
    (zeroAgreement : BaseZeroAgreement ops)
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    {shape : Shape}
    (assignment : BooleanAssignment shape.cubeVariables) :
    AlphaPolynomial.CoefficientZero ops.toOps
        ((residualTable assignment).toAlphaPolynomial ops) ↔
      StrictNormBounded assignment := by
  exact Iff.trans
    (BooleanTable.toAlphaPolynomial_coefficientZero_iff_allEntriesZero
      ops zeroLaws (residualTable assignment))
    (residualTable_allEntriesZero_iff_strictNormBounded
      ops zeroAgreement noZeroDivisors assignment)

structure SourceBatch (shape : Shape) where
  assignments : Fin shape.sourceCount ->
    BooleanAssignment shape.cubeVariables

namespace SourceBatch

def residualTables
    {shape : Shape}
    (batch : SourceBatch shape) :
    Fin shape.sourceCount -> BooleanTable F shape.cubeVariables :=
  fun source => residualTable (batch.assignments source)

def AllStrictNormBounded
    {shape : Shape}
    (batch : SourceBatch shape) : Prop :=
  ∀ source, StrictNormBounded (batch.assignments source)

theorem allResidualTablesZero_iff_allStrictNormBounded
    (ops : InterpolationOps F)
    (zeroAgreement : BaseZeroAgreement ops)
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    {shape : Shape}
    (batch : SourceBatch shape) :
    (∀ source, (batch.residualTables source).AllEntriesZero ops) ↔
      batch.AllStrictNormBounded := by
  constructor
  · intro allZero source
    exact (residualTable_allEntriesZero_iff_strictNormBounded
      ops zeroAgreement noZeroDivisors (batch.assignments source)).mp
        (allZero source)
  · intro allBounded source
    exact (residualTable_allEntriesZero_iff_strictNormBounded
      ops zeroAgreement noZeroDivisors (batch.assignments source)).mpr
        (allBounded source)

end SourceBatch

end NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.NormResidualTable
