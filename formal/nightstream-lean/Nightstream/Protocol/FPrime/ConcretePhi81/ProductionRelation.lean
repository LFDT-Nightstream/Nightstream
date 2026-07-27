import Nightstream.Protocol.FPrime.ConcretePhi81.Outer
import Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.CcsRefinement
import Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.LogicalCarrier
import Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.PiCcsSources

/-!
Exact production relation profile for the five-ring F-prime public carrier.

Assurance tier: model-level.

Protocol: SuperNeo Definitions 12--14, specialized to the fixed-active
HyperNova Construction-2 source arity.

Owns: the one-fresh/fourteen-running semantic shape; its definitional
identification with the independent 270-coordinate relation; the exact
fresh public projection; full-carrier running ownership; concrete CCS
preservation; definitional instantiation of the existing exact CCS/CE
membership expansions; and closure under the selected `Pi_RLC` and `Pi_DEC`
public-input operations.

Does not own: Rust decoding, Ajtai key bytes, Fiat--Shamir, R1CS, generated
rows, or physical-column ownership.

Authority boundary: only fresh inputs have thirteen fixed zero coordinates.
Running inputs are values of the complete 270-coordinate `L_in` and no
257-coordinate projection is used as authority.

Emits constraints: no.

| Boundary | Exact equation or ownership rule | Lean owner |
|---|---|---|
| source arity | `fresh = 1`, `running = 14`, `total = 15` | `sourceArity_exact` |
| public carrier | `n_F,in = 54 * 5 = 270` and `ProductionPublicInput = LIn` | `publicWidth_eq`, `publicInput_eq_lIn` |
| fresh source | legacy coordinates followed by thirteen fixed zeros | `freshPublicInput_exact` |
| running source | every source retains the complete 270-coordinate assignment | `runningAssignment_exact` |
| PiDEC | splitting then recomposing the complete public input is identity | `piDecPublicInput_roundTrip` |
-/

set_option autoImplicit false

namespace Nightstream.Protocol.FPrime.ConcretePhi81.ProductionRelation

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270
open Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.LogicalCarrier
open Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.PiCcsSources
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier

/-- Fixed-active production source shape: one fresh CCS source and fourteen
running CE sources. -/
abbrev Shape (dimensions : Dimensions) : SemanticShape :=
  semanticShape dimensions 1 14

/-- The exact five-ring public-width proof for the production source shape. -/
def publicFits (dimensions : Dimensions) :
    ringDegree * publicRingColumns <= (Shape dimensions).carrierWidth :=
  dimensions.alignedPublicFitsCarrier

/-- The batch-shaped production relation is definitionally the independent
five-ring relation. No cast, truncation, or proof-supplied isomorphism occurs. -/
@[simp] theorem relationShape_eq (dimensions : Dimensions) :
    RelationShape (Shape dimensions) publicRingColumns
        (publicFits dimensions) =
      dimensions.shape := by
  rfl

/-- The production relation has the paper-required exact public width
`d * n_R,in = 54 * 5 = 270`. -/
@[simp] theorem publicWidth_eq (dimensions : Dimensions) :
    (RelationShape (Shape dimensions) publicRingColumns
      (publicFits dimensions)).publicWidth = 270 := by
  rfl

/-- The production relation's public carrier is exactly the paper-owned
five-ring `L_in`. -/
abbrev ProductionPublicInput (dimensions : Dimensions) :=
  Phi81Relation.PublicInput dimensions.shape

@[simp] theorem publicInput_eq_lIn (dimensions : Dimensions) :
    ProductionPublicInput dimensions = LIn dimensions := by
  rfl

/-- The fixed-active source partition is exactly `1 + 14`. -/
theorem sourceArity_exact (dimensions : Dimensions) :
    (Shape dimensions).freshCount = 1 /\
      (Shape dimensions).runningCount = 14 /\
      (Shape dimensions).sourceCount = 15 := by
  exact ⟨rfl, rfl, rfl⟩

/-- Fresh source projection is the legacy 257-coordinate input followed by
the thirteen verifier-owned zeros. -/
theorem freshPublicInput_exact
    (dimensions : Dimensions)
    (legacy : LegacyAssignment dimensions) :
    sourcePublicInput publicRingColumns (publicFits dimensions)
        (assignment dimensions legacy) =
      expectedPublicInput dimensions legacy := by
  exact projectPublicInput_exact dimensions legacy

/-- A running source is retained as a complete carrier value. In particular,
the thirteen coordinates initialized to zero for fresh inputs are not erased
after folding. -/
theorem runningAssignment_exact
    (dimensions : Dimensions)
    (inputs : Inputs dimensions 1 14)
    (source : Fin 14) :
    inputs.data.assignment (Data.runningIndex source) =
      inputs.runningAssignments source :=
  Inputs.data_assignment_runningIndex_eq inputs source

/-- Concrete CCS preservation for the exact production relation carrier. -/
theorem liftedCcsSatisfied_iff
    (dimensions : Dimensions)
    (legacy : CcsRefinement.LegacyStructure dimensions)
    (legacyAssignment : LegacyAssignment dimensions) :
    CCSResidualTable.ConstraintSatisfied ConcreteCarrier.baseOps
        (CcsRefinement.liftStructure dimensions legacy).matrixSource.system
        (assignment dimensions legacyAssignment) <->
      CCSResidualTable.ConstraintSatisfied ConcreteCarrier.baseOps legacy
        legacyAssignment :=
  CcsRefinement.constraintSatisfied_iff dimensions legacy legacyAssignment

/-- The selected PiDEC public split and recomposition are inverse on the
complete production `L_in`. -/
theorem piDecPublicInput_roundTrip
    (dimensions : Dimensions) (input : ProductionPublicInput dimensions) :
    piDecRecompose dimensions (piDecSplit dimensions input) = input :=
  piDecSplit_recompose dimensions input

/-- One compact headline collecting the exact production profile facts
required before the NIFS/F-prime bridge can be instantiated. -/
theorem exactProfile (dimensions : Dimensions) :
    (Shape dimensions).freshCount = 1 /\
      (Shape dimensions).runningCount = 14 /\
      (Shape dimensions).sourceCount = 15 /\
      (RelationShape (Shape dimensions) publicRingColumns
        (publicFits dimensions)).publicWidth = ringDegree * 5 /\
      (RelationShape (Shape dimensions) publicRingColumns
        (publicFits dimensions)).publicWidth = 270 /\
      (forall legacy : LegacyAssignment dimensions,
        sourcePublicInput publicRingColumns (publicFits dimensions)
            (assignment dimensions legacy) =
          expectedPublicInput dimensions legacy) := by
  refine ⟨rfl, rfl, rfl, rfl, rfl, ?_⟩
  exact freshPublicInput_exact dimensions

end Nightstream.Protocol.FPrime.ConcretePhi81.ProductionRelation
