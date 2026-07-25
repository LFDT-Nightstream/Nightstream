import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.ProductTruth.SourceBridge
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.ProductTruth.NormBridge

/-!
Assignment-indexed truth transport for the concrete Split-NC source product.

Assurance tier: model-level.

Owns: derivation of independent FE truth from every concrete `PiCCS`
payload obligation, and derivation of independent full-carrier NC truth from
every concrete norm obligation, at the exact partition-preserving assignment
ordering.

Does not own: commitments, extraction, transcript acceptance, SumCheck
messages, challenges, output authority, Fiat--Shamir, Rust, R1CS, costs, or
rows.

Emits constraints: no.

| Stage path | Owned equation | Authority |
|---|---|---|
| `piccs.split_nc.product_truth` | product payloads/norms imply independent FE/NC truth | derived |
-/

set_option autoImplicit false

namespace Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.ProductTruth

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier

universe uCommitment

/-- Every public product source payload at the exact authoritative assignment
ordering. -/
def PayloadsHold
    {shape : SemanticShape}
    {params : GlobalParams}
    {arity : BatchArity params}
    {Commitment : Type uCommitment}
    (publicRingColumns : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (commit : SourceAssignment shape -> Commitment)
    (data : Data shape)
    (alignment : SourceAlignment shape params arity)
    (input :
      SourceProduct shape publicRingColumns publicFits Commitment params arity) :
    Prop :=
  ∀ source,
    PiCCS.Source.PayloadTruth
      (productSemantics publicRingColumns publicFits commit)
      (input.source source)
      (InputAuthority.productAssignments data alignment source)

/-- One public fresh payload is the batch-invariant relation proof for the
same authoritative fresh assignment. -/
theorem freshRelationTruth_of_payloads
    {shape : SemanticShape}
    {params : GlobalParams}
    {arity : BatchArity params}
    {Commitment : Type uCommitment}
    (publicRingColumns : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (commit : SourceAssignment shape -> Commitment)
    (data : Data shape)
    (alignment : SourceAlignment shape params arity)
    (input :
      SourceProduct shape publicRingColumns publicFits Commitment params arity)
    (inputAuthority :
      InputAuthority.BoundToSources publicRingColumns publicFits commit data
        alignment input)
    (payloads :
      PayloadsHold publicRingColumns publicFits commit data alignment input)
    (fresh : Fin shape.freshCount) :
    Phi81Relation.ccsSatisfied
      (Phi81Relation.Structure.ofSourceData
        publicRingColumns publicFits data)
      (data.freshAssignment fresh) := by
  let productFresh := alignment.productFreshIndex fresh
  have payload :=
    payloads (Fin.castAdd (arity.mode.count params) productFresh)
  simp only [PiCCS.InputProduct.source, PiCCS.Source.PayloadTruth,
    InputAuthority.productAssignments_fresh] at payload
  simp [productFresh] at payload
  have payload' :
      Phi81Relation.ccsSatisfied
        (input.fresh productFresh).constraintSystem
        (data.freshAssignment fresh) := by
    change
      Phi81Relation.ccsSatisfied
        (input.fresh productFresh).constraintSystem
        (data.freshAssignment
          (alignment.semanticFreshIndex productFresh)) at payload
    simpa only [SourceAlignment.semanticFreshIndex_productFreshIndex,
      productFresh] using payload
  simpa only [(inputAuthority.fresh productFresh).constraintSystem] using
    payload'

/-- Concrete fresh-source payloads imply the independent fresh CCS
statement. -/
theorem freshTruth_of_payloads
    {shape : SemanticShape}
    {params : GlobalParams}
    {arity : BatchArity params}
    {Commitment : Type uCommitment}
    (publicRingColumns : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (commit : SourceAssignment shape -> Commitment)
    (data : Data shape)
    (alignment : SourceAlignment shape params arity)
    (input :
      SourceProduct shape publicRingColumns publicFits Commitment params arity)
    (inputAuthority :
      InputAuthority.BoundToSources publicRingColumns publicFits commit data
        alignment input)
    (payloads :
      PayloadsHold publicRingColumns publicFits commit data alignment input) :
    ∀ fresh,
      CCSResidualTable.ConstraintSatisfied ConcreteCarrier.baseOps
        data.freshBatch.system (data.freshBatch.assignments fresh) := by
  intro fresh
  exact
    @SourceBridge.freshConstraintSatisfied_of_relation shape
      publicRingColumns publicFits data fresh
      (@freshRelationTruth_of_payloads shape params arity Commitment
        publicRingColumns publicFits commit data alignment input inputAuthority
        payloads fresh)

/-- Every concrete generic norm obligation at `b = 2` is exactly the
independent full-carrier NC truth. -/
theorem ncTruth_of_norms
    {shape : SemanticShape}
    {params : GlobalParams}
    {arity : BatchArity params}
    {Commitment : Type uCommitment}
    (publicRingColumns : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (commit : SourceAssignment shape -> Commitment)
    (data : Data shape)
    (alignment : SourceAlignment shape params arity)
    (freshBound : params.b = 2)
    (norms :
      ∀ source,
        (productSemantics publicRingColumns publicFits commit).normBounded
          params.b (InputAuthority.productAssignments data alignment source)) :
    ∀ source column,
      centeredMagnitude (data.assignment source column) < 2 := by
  intro source column
  have bounded := NormBridge.atProductIndex publicRingColumns publicFits commit
    data alignment norms source
  rw [freshBound] at bounded
  exact bounded column

end Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.ProductTruth
