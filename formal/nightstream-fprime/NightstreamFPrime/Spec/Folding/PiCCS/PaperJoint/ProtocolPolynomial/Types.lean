import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.CCSResidualTable

/-! SuperNeo v1.2 Section 7.3, Appendix B.2. -/

namespace NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ProtocolPolynomial

universe uField

structure VerifierInput (Field : Type uField) (shape : Shape) where
  constraintPolynomial :
    CCSResidualTable.ConstraintPolynomial Field shape.matrixCount
  priorPoint : CubePoint Field shape.cubeVariables
  claimedPadCoefficient : PadCoordinate shape -> Field
  claimedMatrixCoefficient : MatrixCoordinate shape -> Field

namespace VerifierInput

@[implicit_reducible]
def sumcheckDegreeBound
    {Field : Type uField}
    {shape : Shape}
    (input : VerifierInput Field shape) : Nat :=
  Nat.max
    input.constraintPolynomial.canonicalEqualityGatedDegreeBound 4

/-- Corrected SuperNeo v1.2 Appendix B.2 per-round degree bound. -/
def paperRoundDegreeCeiling
    {Field : Type uField}
    {shape : Shape}
    (input : VerifierInput Field shape)
    (b : Nat) : Nat :=
  Nat.max input.constraintPolynomial.canonicalEqualityGatedDegreeBound
    (Nat.max (2 * b) 2)

theorem sumcheckDegreeBound_le_paperRoundDegreeCeiling_of_b_eq_two
    {Field : Type uField}
    {shape : Shape}
    (input : VerifierInput Field shape)
    {b : Nat}
    (b_eq : b = 2) :
    input.sumcheckDegreeBound <= input.paperRoundDegreeCeiling b := by
  subst b
  unfold sumcheckDegreeBound paperRoundDegreeCeiling
  exact (Nat.max_le).2 ⟨
    Nat.le_max_left _ _,
    Nat.le_trans (by decide : 4 <= Nat.max (2 * 2) 2)
      (Nat.le_max_right _ _)⟩

theorem sumcheckDegreeBound_eq_of_terms_eq
    {Field : Type uField}
    {shape : Shape}
    (left right : VerifierInput Field shape)
    (terms : left.constraintPolynomial.terms =
      right.constraintPolynomial.terms) :
    left.sumcheckDegreeBound = right.sumcheckDegreeBound := by
  unfold sumcheckDegreeBound
  exact congrArg (fun degree => Nat.max degree 4)
    (CCSResidualTable.ConstraintPolynomial.canonicalEqualityGatedDegreeBound_eq_of_terms_eq
        left.constraintPolynomial right.constraintPolynomial terms)

@[ext] theorem ext
    {Field : Type uField}
    {shape : Shape}
    (left right : VerifierInput Field shape)
    (constraintPolynomial :
      left.constraintPolynomial = right.constraintPolynomial)
    (priorPoint : left.priorPoint = right.priorPoint)
    (claimedPadCoefficient :
      left.claimedPadCoefficient = right.claimedPadCoefficient)
    (claimedMatrixCoefficient :
      left.claimedMatrixCoefficient = right.claimedMatrixCoefficient) :
    left = right := by
  cases left
  cases right
  simp_all

end VerifierInput

structure OutputMessage (Field : Type uField) (shape : Shape) where
  freshMatrixImage : Fin shape.freshCount -> Fin shape.matrixCount -> Field
  sourceAssignment : Fin shape.sourceCount -> Field
  padImage : PadCoordinate shape -> Field
  matrixImage : MatrixCoordinate shape -> Field

namespace OutputMessage

@[ext] theorem ext
    {Field : Type uField}
    {shape : Shape}
    (left right : OutputMessage Field shape)
    (freshMatrixImage : left.freshMatrixImage = right.freshMatrixImage)
    (sourceAssignment : left.sourceAssignment = right.sourceAssignment)
    (padImage : left.padImage = right.padImage)
    (matrixImage : left.matrixImage = right.matrixImage) :
    left = right := by
  cases left
  cases right
  simp_all

end OutputMessage

end NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ProtocolPolynomial
