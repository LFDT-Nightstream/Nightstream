import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources

/-!
Independent FE obligations for the Phi81 SplitNc `Pi_CCS` model.

Protocol: SuperNeo `Pi_CCS`.
Phase: FE semantic truth before challenge compression or SumCheck.
Constraint family: fresh CCS residuals and running CE evaluation residuals.

Owns: the conjunction of all fresh CCS zero-set obligations and all running
carried-evaluation equations, plus exact equivalence with the already-audited
uncompressed residual families.

Does not own: FE challenge mixing, the two-dimensional FE polynomial,
SumCheck rounds, transcript derivation, Rust, R1CS, or constraint counts.

Emits constraints: no.

Authority boundary: every residual is derived from `Sources.Data`. The caller
cannot provide a separate FE truth callback, coefficient matrix, or running
assignment view.

| Protocol | Phase | Family | Mathematical result |
|---|---|---|---|
| `Pi_CCS` | FE semantics | fresh CCS | every explicit matrix/polynomial row is zero |
| `Pi_CCS` | FE semantics | running CE | every claimed Phi81 coefficient equals the derived matrix-image evaluation |
| `Pi_CCS` | FE residualization | CCS / carried | uncompressed residual zero iff independent FE truth |
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Semantics.Fe

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources

/-- Every fresh source satisfies the explicit CCS relation. -/
def FreshTruth
    {shape : SemanticShape}
    (data : Data shape) : Prop :=
  data.freshBatch.AllConstraintsSatisfied ConcreteCarrier.baseOps

/-- Every running source satisfies every derived coefficient-evaluation
claim at the prior row point. -/
def CarriedTruth
    {shape : SemanticShape}
    (data : Data shape) : Prop :=
  CarriedEvaluationResidual.AllClaimsHold
    ConcreteCarrier.baseOps ConcreteCarrier.extensionOps K.embed
    data.carriedData

/-- Complete FE semantic truth before any random compression. -/
def Truth
    {shape : SemanticShape}
    (data : Data shape) : Prop :=
  FreshTruth data /\ CarriedTruth data

/-- Uncompressed FE residual families. This is a proof-facing view, not an
R1CS row set. -/
def ResidualsZero
    {shape : SemanticShape}
    (data : Data shape) : Prop :=
  (forall source,
      (data.freshBatch.residualTables ConcreteCarrier.baseOps source).AllEntriesZero
        ConcreteCarrier.baseOps) /\
    (forall coordinate,
      CarriedEvaluationResidual.residual
          ConcreteCarrier.baseOps ConcreteCarrier.extensionOps K.embed
          data.carriedData coordinate = K.zero)

/-- The uncompressed FE residuals are zero exactly when the independently
stated fresh CCS and running CE obligations hold. -/
theorem residualsZero_iff_truth
    {shape : SemanticShape}
    (data : Data shape) :
    ResidualsZero data <-> Truth data := by
  constructor
  · rintro ⟨freshZero, carriedZero⟩
    exact ⟨
      (CCSResidualTable.FreshBatch.allResidualTablesZero_iff_allConstraintsSatisfied
        ConcreteCarrier.baseOps data.freshBatch).mp freshZero,
      (CarriedEvaluationResidual.allResidualsZero_iff_allClaimsHold
        ConcreteCarrier.baseOps ConcreteCarrier.extensionOps
        ConcreteCarrier.extensionLaws K.embed data.carriedData).mp carriedZero⟩
  · rintro ⟨freshTruth, carriedTruth⟩
    exact ⟨
      (CCSResidualTable.FreshBatch.allResidualTablesZero_iff_allConstraintsSatisfied
        ConcreteCarrier.baseOps data.freshBatch).mpr freshTruth,
      (CarriedEvaluationResidual.allResidualsZero_iff_allClaimsHold
        ConcreteCarrier.baseOps ConcreteCarrier.extensionOps
        ConcreteCarrier.extensionLaws K.embed data.carriedData).mpr carriedTruth⟩

end Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Semantics.Fe
