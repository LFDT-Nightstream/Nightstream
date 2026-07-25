import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.InputAuthority

/-!
Batch-count erasure for the concrete Split-NC matrix owner.

Assurance tier: model-level.

Owns: the two definitional projections needed to compare a fold-shaped fresh
CCS source with the batch-invariant Phi81 relation. Fresh and running batch
counts are not matrix or constraint-polynomial dimensions.

Does not own: source validity, transcript acceptance, commitments, extraction,
probability, Rust, R1CS, costs, or rows.

Emits constraints: no.

| Stage path | Owned equation | Authority |
|---|---|---|
| `piccs.split_nc.product_truth.source` | fold-shaped matrices/polynomial equal the batch-invariant relation owners | definitional |
-/

set_option autoImplicit false

namespace Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.ProductTruth.SourceBridge

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources

private def rawResidualAt
    {cubeVariables matrixCount columns : Nat}
    (matrices : Fin matrixCount ->
      PaperLinearAlgebra.BooleanMatrix F cubeVariables columns)
    (polynomial :
      CCSResidualTable.ConstraintPolynomial F matrixCount)
    (assignment : PaperLinearAlgebra.Assignment F columns)
    (vertex : BooleanVertex cubeVariables) : F :=
  CCSResidualTable.evaluatePolynomial ConcreteCarrier.baseOps polynomial
    (fun matrix =>
      PaperLinearAlgebra.matrixVectorAt ConcreteCarrier.baseOps
        (matrices matrix) assignment vertex)

/-- Forgetting batch counts preserves the sole completed matrix family. -/
theorem freshMatrices_eq_relation
    {shape : SemanticShape}
    (publicRingColumns : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (data : Data shape) :
    data.freshBatch.system.matrices =
      (Phi81Relation.Structure.ofSourceData
        publicRingColumns publicFits data).matrixSource.system.matrices := by
  rfl

/-- Forgetting batch counts preserves the sole constraint polynomial. -/
theorem freshPolynomial_eq_relation
    {shape : SemanticShape}
    (publicRingColumns : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (data : Data shape) :
    data.freshBatch.system.constraintPolynomial =
      (Phi81Relation.Structure.ofSourceData
        publicRingColumns publicFits data).matrixSource.system.constraintPolynomial := by
  rfl

/-- A batch-invariant relation proof is the fresh-batch CCS proof for the
same authoritative assignment. -/
theorem freshConstraintSatisfied_of_relation
    {shape : SemanticShape}
    (publicRingColumns : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (data : Data shape)
    (fresh : Fin shape.freshCount)
    (relationTruth :
      Phi81Relation.ccsSatisfied
        (Phi81Relation.Structure.ofSourceData
          publicRingColumns publicFits data)
        (data.freshAssignment fresh)) :
    CCSResidualTable.ConstraintSatisfied ConcreteCarrier.baseOps
      data.freshBatch.system (data.freshBatch.assignments fresh) := by
  simp only [Phi81Relation.ccsSatisfied, CCSResidualTable.ConstraintSatisfied]
    at relationTruth
  intro vertex
  have relationResidual := relationTruth vertex
  change
    rawResidualAt data.freshBatch.system.matrices
      data.freshBatch.system.constraintPolynomial
      (data.freshAssignment fresh) vertex =
      ConcreteCarrier.baseOps.zero
  change
    rawResidualAt
        (Phi81Relation.Structure.ofSourceData
          publicRingColumns publicFits data).matrixSource.system.matrices
        (Phi81Relation.Structure.ofSourceData
          publicRingColumns publicFits data).matrixSource.system.constraintPolynomial
        (data.freshAssignment fresh) vertex =
      ConcreteCarrier.baseOps.zero at relationResidual
  rw [freshMatrices_eq_relation publicRingColumns publicFits data,
    freshPolynomial_eq_relation publicRingColumns publicFits data]
  exact relationResidual

end Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.ProductTruth.SourceBridge
