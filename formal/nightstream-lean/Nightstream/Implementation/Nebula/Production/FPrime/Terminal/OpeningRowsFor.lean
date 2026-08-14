import Nightstream.Implementation.Lowering.Goldilocks.NumericRowBridge
import Nightstream.Implementation.Nebula.Production.FPrime.Terminal.InvocationRowsSoundFor
import Nightstream.Implementation.Nebula.Production.FPrime.Terminal.CoreRowsFor
import Nightstream.Implementation.Nebula.Production.FPrime.Terminal.RunningPlacementFor
import Nightstream.Implementation.Nebula.Commitment.Terminal.ProductCommitmentBridge

/-!
Contract: one-assignment, row-derived terminal openings for the exact paper
F-prime relation.

The numeric terminal fold is interpreted through the fixed injective
`numericColumn` embedding into one typed Goldilocks assignment. For each of
the fourteen PiDEC children, static column aliases connect the four public
commitment components and all terminal-core inputs directly to the verified
trailing NIFS output carrier. Typed norm, Ajtai, public-projection, and Phi81
rows then derive one bounded complete witness for that child.

No commitment equality, `ProductOpening`, terminal relation result, or
acceptance Boolean is an input. This module does not own the generated WASM
relation, byte parsing, a compact terminal backend, Rust refinement, or a
cryptographic binding reduction.

Assurance tier: exponent-indexed terminal row implementation.
-/

set_option autoImplicit false
set_option maxHeartbeats 2000000
set_option maxRecDepth 100000

namespace Nightstream.Implementation.Nebula.ProductionPaperTerminalOpeningRowsFor

open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.Goldilocks.NumericRowBridge
open Nightstream.Implementation.Nebula
open Nightstream.Implementation.Nebula.ProductNifsRunningCoordinatesFor
open Nightstream.Implementation.Nebula.ProductionPaperTerminalInvocationRowsSoundFor
open Nightstream.Implementation.R1CS
open Nightstream.Protocol.Nebula
open Nightstream.Protocol.Nebula.ProductionProfileCandidates
open Nightstream.Protocol.Nebula.Terminal
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

abbrev FullShape
    (rowVariables logicalWidth : Nat)
    (publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth) :=
  ProductionPaperTerminalInvocationRowsSoundFor.FullShape rowVariables
    logicalWidth publicFits

/-- Canonical numeric view of the one typed terminal assignment. This is a
definition, not a prover-supplied agreement relation. -/
def pulledAssignment (assignment : ColumnId -> F) : Nat -> Nat :=
  NumericRowBridge.numericAssignment
    TerminalBundleOpeningRows.Layout.numericColumn assignment

theorem pulledAssignment_canonical (assignment : ColumnId -> F) :
    forall column, pulledAssignment assignment column < goldilocksP :=
  NumericRowBridge.numericAssignment_canonical
    TerminalBundleOpeningRows.Layout.numericColumn assignment

/-- The final NIFS carrier inside the same typed assignment. -/
def outputCarrier {rowVariables : Nat}
    (layout : ProductionProductNifsOutputRowsFor.Layout rowVariables) :
    ProductionPaperTerminalRunningPlacementFor.Carrier rowVariables where
  column := fun index => TerminalBundleOpeningRows.Layout.numericColumn
    (layout.carrierColumn index)

/-- Rows and static aliases for one terminal child. All children share the
same `assignment`; only their complete-witness column families differ. -/
structure ChildRows
    {manifest : SeedSchedule.Manifest}
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape)
    (outputLayout : ProductionProductNifsOutputRowsFor.Layout rowVariables)
    (assignment : ColumnId -> F)
    (child : FoldedChild) where
  layout : TerminalBundleOpeningRows.Layout manifest
    (FullShape rowVariables logicalWidth publicFits) operationsShape snapshotShape
  typedRows : Nightstream.Implementation.Lowering.Goldilocks.Satisfies
    (TerminalBundleOpeningRows.rows layout) assignment
  configExact : TerminalProductCommitmentBridge.config layout = config
  oneAlias : layout.one = TerminalBundleOpeningRows.Layout.numericColumn 0
  commitmentAlias : forall component : Fin 4, forall row lane,
    layout.commitmentColumn (componentAt component) row lane =
      TerminalBundleOpeningRows.Layout.numericColumn
        (outputLayout.carrierColumn
          (ProductionProductNifsOutputRowsFor.commitmentIndex child component
            row lane))

namespace ChildRows

theorem typedOne
    {manifest : SeedSchedule.Manifest}
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    {config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape}
    {outputLayout : ProductionProductNifsOutputRowsFor.Layout rowVariables}
    {assignment : ColumnId -> F} {child : FoldedChild}
    (rows : ChildRows (manifest := manifest) config outputLayout assignment child)
    (numericOne : pulledAssignment assignment 0 = 1) :
    assignment rows.layout.one = 1 := by
  rw [rows.oneAlias]
  apply Fin.ext
  simpa [pulledAssignment, NumericRowBridge.numericAssignment] using numericOne

theorem commitmentCoordinateExact
    {manifest : SeedSchedule.Manifest}
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    {config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape}
    {outputLayout : ProductionProductNifsOutputRowsFor.Layout rowVariables}
    {assignment : ColumnId -> F} {child : FoldedChild}
    {running : ProductNifsCodec.RunningFor rowVariables
      (FullShape rowVariables logicalWidth publicFits)}
    (rows : ChildRows (manifest := manifest) config outputLayout assignment child)
    (placed : ProductionProductNifsOutputRowsFor.Placed outputLayout
      (pulledAssignment assignment) (pulledAssignment_canonical assignment)
      running)
    (component : Fin 4) (row : Fin ProductCommitmentAlgebra.Rank)
    (lane : Fin ringDegree) :
    assignment (rows.layout.commitmentColumn (componentAt component) row lane) =
      running.commitments child (componentAt component) row lane := by
  rw [rows.commitmentAlias component row lane]
  apply Fin.ext
  change
    pulledAssignment assignment
        (outputLayout.carrierColumn
          (ProductionProductNifsOutputRowsFor.commitmentIndex child component
            row lane)) =
      (running.commitments child (componentAt component) row lane).val
  rw [placed.assignment_coordinate]
  exact congrArg Fin.val
    (ProductNifsRunningCoordinatesFor.runningCodecFor_commitment_getD running
      child (componentAt component) row lane)

theorem publicBundleExact
    {manifest : SeedSchedule.Manifest}
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    {config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape}
    {outputLayout : ProductionProductNifsOutputRowsFor.Layout rowVariables}
    {assignment : ColumnId -> F} {child : FoldedChild}
    {running : ProductNifsCodec.RunningFor rowVariables
      (FullShape rowVariables logicalWidth publicFits)}
    (rows : ChildRows (manifest := manifest) config outputLayout assignment child)
    (placed : ProductionProductNifsOutputRowsFor.Placed outputLayout
      (pulledAssignment assignment) (pulledAssignment_canonical assignment)
      running) :
    rows.layout.publicBundle assignment = running.commitments child := by
  funext component row lane
  cases component with
  | full =>
      simpa [componentAt] using
        rows.commitmentCoordinateExact placed (0 : Fin 4) row lane
  | operations =>
      simpa [componentAt] using
        rows.commitmentCoordinateExact placed (1 : Fin 4) row lane
  | initialSnapshot =>
      simpa [componentAt] using
        rows.commitmentCoordinateExact placed (2 : Fin 4) row lane
  | finalSnapshot =>
      simpa [componentAt] using
        rows.commitmentCoordinateExact placed (3 : Fin 4) row lane

theorem bounded
    {manifest : SeedSchedule.Manifest}
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    {config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape}
    {outputLayout : ProductionProductNifsOutputRowsFor.Layout rowVariables}
    {assignment : ColumnId -> F} {child : FoldedChild}
    (rows : ChildRows (manifest := manifest) config outputLayout assignment child)
    (numericOne : pulledAssignment assignment 0 = 1) :
    assignmentNormBounded 2 (rows.layout.fullAssignment assignment) :=
  (TerminalBundleOpeningRows.sound rows.layout assignment
    (rows.typedOne numericOne) rows.typedRows).bounded

theorem opens
    {manifest : SeedSchedule.Manifest}
    {candidate : ProductionProfileCandidates.Id}
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    {statementId : ProductConcreteNifsFor.StatementId}
    {config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape}
    {artifact : ProductConcreteNifsFor.RelationArtifact rowVariables
      logicalWidth publicFits}
    {value : ProductionFieldNativeFullClaim.Value candidate
      (FullShape rowVariables logicalWidth publicFits)}
    {proof : ProductionProductPiCcsTypedBridgeFor.ExactProof rowVariables}
    {outputLayout : ProductionProductNifsOutputRowsFor.Layout rowVariables}
    {assignment : ColumnId -> F} {child : FoldedChild}
    (rows : ChildRows (manifest := manifest) config outputLayout assignment child)
    (numericOne : pulledAssignment assignment 0 = 1)
    (placed : ProductionProductNifsOutputRowsFor.Placed outputLayout
      (pulledAssignment assignment) (pulledAssignment_canonical assignment)
      (ProductionPaperTerminalInvocationRowsSoundFor.finalRunning candidate
        statementId config artifact value proof)) :
    ProductCommitmentAlgebra.commit config
        (rows.layout.fullAssignment assignment) =
      (ProductionPaperTerminalInvocationRowsSoundFor.children candidate
        statementId config artifact value proof child).commitment := by
  have opened := TerminalBundleOpeningRows.sound rows.layout assignment
    (rows.typedOne numericOne) rows.typedRows
  calc
    ProductCommitmentAlgebra.commit config
        (rows.layout.fullAssignment assignment) =
        ProductCommitmentAlgebra.commit
          (TerminalProductCommitmentBridge.config rows.layout)
          (rows.layout.fullAssignment assignment) := by
      rw [rows.configExact]
    _ = TerminalBundleOpeningRows.exactBundle rows.layout assignment :=
      TerminalProductCommitmentBridge.commit_eq_exactBundle _ _
    _ = rows.layout.publicBundle assignment := opened.opensAll
    _ = (ProductionPaperTerminalInvocationRowsSoundFor.finalRunning candidate
          statementId config artifact value proof).commitments child :=
      rows.publicBundleExact placed
    _ = _ := rfl

end ChildRows

/-- Exact terminal row family for all fourteen children. The recursive result
is indexed by the numeric view of this same typed assignment. -/
structure Family
    {manifest : SeedSchedule.Manifest}
    {candidate : ProductionProfileCandidates.Id}
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (statementId : ProductConcreteNifsFor.StatementId)
    (config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape)
    (artifact : ProductConcreteNifsFor.RelationArtifact rowVariables
      logicalWidth publicFits)
    (priorAuthority : ProductionPaperPriorStateAuthorityRowsFor.Layout candidate
      rowVariables)
    (headers : FPrime.ChainHeaders Digest.Value)
    (priorPrefix : ProductionPaperPriorStateAuthorityRowsFor.Prefix candidate
      (FullShape rowVariables logicalWidth publicFits))
    (value : ProductionFieldNativeFullClaim.Value candidate
      (FullShape rowVariables logicalWidth publicFits))
    (proof : ProductionProductPiCcsTypedBridgeFor.ExactProof rowVariables)
    (assignment : ColumnId -> F)
    (recursive : ProductionPaperRecursiveRelationRowsSoundFor.Result candidate
      statementId config artifact priorAuthority (pulledAssignment assignment)
      headers priorPrefix value proof) where
  outputLayout : ProductionProductNifsOutputRowsFor.Layout rowVariables
  outputLayoutExact : outputLayout = recursive.nifsOutputLayout
  child : (index : FoldedChild) -> ChildRows (manifest := manifest) config
    outputLayout assignment index
  coreLayout : (index : FoldedChild) ->
    ProductionPaperTerminalCoreRowsFor.Layout (child index).layout
  coreAliases : (index : FoldedChild) ->
    ProductionPaperTerminalRunningPlacementFor.Aliases
      (outputCarrier outputLayout) (coreLayout index) index
  coreRows : (index : FoldedChild) ->
    ProductionPaperTerminalCoreRowsFor.RowsEvidence (coreLayout index)
      artifact.system assignment

namespace Family

theorem runningPlaced
    {manifest : SeedSchedule.Manifest}
    {candidate : ProductionProfileCandidates.Id}
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    {statementId : ProductConcreteNifsFor.StatementId}
    {config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape}
    {artifact : ProductConcreteNifsFor.RelationArtifact rowVariables
      logicalWidth publicFits}
    {priorAuthority : ProductionPaperPriorStateAuthorityRowsFor.Layout candidate
      rowVariables}
    {headers : FPrime.ChainHeaders Digest.Value}
    {priorPrefix : ProductionPaperPriorStateAuthorityRowsFor.Prefix candidate
      (FullShape rowVariables logicalWidth publicFits)}
    {value : ProductionFieldNativeFullClaim.Value candidate
      (FullShape rowVariables logicalWidth publicFits)}
    {proof : ProductionProductPiCcsTypedBridgeFor.ExactProof rowVariables}
    {assignment : ColumnId -> F}
    {recursive : ProductionPaperRecursiveRelationRowsSoundFor.Result candidate
      statementId config artifact priorAuthority (pulledAssignment assignment)
      headers priorPrefix value proof}
    (family : Family (manifest := manifest) statementId config artifact
      priorAuthority headers priorPrefix value proof assignment recursive) :
    ProductionPaperTerminalRunningPlacementFor.Placed
      (outputCarrier family.outputLayout) assignment
      (ProductionPaperTerminalInvocationRowsSoundFor.finalRunning candidate
        statementId config artifact value proof) := by
  have placed : ProductionProductNifsOutputRowsFor.Placed family.outputLayout
      (pulledAssignment assignment) (pulledAssignment_canonical assignment)
      (ProductionPaperTerminalInvocationRowsSoundFor.finalRunning candidate
        statementId config artifact value proof) := by
    rw [family.outputLayoutExact]
    exact recursive.nifsOutputPlaced
  constructor
  intro index
  apply Fin.ext
  change
    pulledAssignment assignment
        (family.outputLayout.carrierColumn index) =
      (((ProductNifsCodec.runningCodecFor rowVariables
          (FullShape rowVariables logicalWidth publicFits)).encode
        (ProductionPaperTerminalInvocationRowsSoundFor.finalRunning candidate
          statementId config artifact value proof)).getD index.val 0).val
  simpa [ProductionPaperTerminalInvocationRowsSoundFor.finalRunning,
    ProductionRecursiveSuccessorFor.nextRunning] using
      placed.assignment_coordinate index

def coreEvidence
    {manifest : SeedSchedule.Manifest}
    {candidate : ProductionProfileCandidates.Id}
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    {statementId : ProductConcreteNifsFor.StatementId}
    {config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape}
    {artifact : ProductConcreteNifsFor.RelationArtifact rowVariables
      logicalWidth publicFits}
    {priorAuthority : ProductionPaperPriorStateAuthorityRowsFor.Layout candidate
      rowVariables}
    {headers : FPrime.ChainHeaders Digest.Value}
    {priorPrefix : ProductionPaperPriorStateAuthorityRowsFor.Prefix candidate
      (FullShape rowVariables logicalWidth publicFits)}
    {value : ProductionFieldNativeFullClaim.Value candidate
      (FullShape rowVariables logicalWidth publicFits)}
    {proof : ProductionProductPiCcsTypedBridgeFor.ExactProof rowVariables}
    {assignment : ColumnId -> F}
    {recursive : ProductionPaperRecursiveRelationRowsSoundFor.Result candidate
      statementId config artifact priorAuthority (pulledAssignment assignment)
      headers priorPrefix value proof}
    (family : Family (manifest := manifest) statementId config artifact
      priorAuthority headers priorPrefix value proof assignment recursive)
    (index : FoldedChild) :
    ProductionPaperTerminalCoreRowsFor.Evidence (family.coreLayout index)
      artifact.system assignment
      (ProductionPaperTerminalInvocationRowsSoundFor.finalRunning candidate
        statementId config artifact value proof) index :=
  ProductionPaperTerminalCoreRowsFor.Evidence.ofRows
    (ProductionPaperTerminalRunningPlacementFor.toVerifierInputPlacement
      family.runningPlaced (family.coreAliases index))
    (family.coreRows index)

def assignments
    {manifest : SeedSchedule.Manifest}
    {candidate : ProductionProfileCandidates.Id}
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    {statementId : ProductConcreteNifsFor.StatementId}
    {config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape}
    {artifact : ProductConcreteNifsFor.RelationArtifact rowVariables
      logicalWidth publicFits}
    {priorAuthority : ProductionPaperPriorStateAuthorityRowsFor.Layout candidate
      rowVariables}
    {headers : FPrime.ChainHeaders Digest.Value}
    {priorPrefix : ProductionPaperPriorStateAuthorityRowsFor.Prefix candidate
      (FullShape rowVariables logicalWidth publicFits)}
    {value : ProductionFieldNativeFullClaim.Value candidate
      (FullShape rowVariables logicalWidth publicFits)}
    {proof : ProductionProductPiCcsTypedBridgeFor.ExactProof rowVariables}
    {assignment : ColumnId -> F}
    {recursive : ProductionPaperRecursiveRelationRowsSoundFor.Result candidate
      statementId config artifact priorAuthority (pulledAssignment assignment)
      headers priorPrefix value proof}
    (family : Family (manifest := manifest) statementId config artifact
      priorAuthority headers priorPrefix value proof assignment recursive) :
    ProductTerminalRelation.Assignments
      (FullShape rowVariables logicalWidth publicFits) :=
  fun index => (family.child index).layout.fullAssignment assignment

theorem coreHolds
    {manifest : SeedSchedule.Manifest}
    {candidate : ProductionProfileCandidates.Id}
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    {statementId : ProductConcreteNifsFor.StatementId}
    {config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape}
    {artifact : ProductConcreteNifsFor.RelationArtifact rowVariables
      logicalWidth publicFits}
    {priorAuthority : ProductionPaperPriorStateAuthorityRowsFor.Layout candidate
      rowVariables}
    {headers : FPrime.ChainHeaders Digest.Value}
    {priorPrefix : ProductionPaperPriorStateAuthorityRowsFor.Prefix candidate
      (FullShape rowVariables logicalWidth publicFits)}
    {value : ProductionFieldNativeFullClaim.Value candidate
      (FullShape rowVariables logicalWidth publicFits)}
    {proof : ProductionProductPiCcsTypedBridgeFor.ExactProof rowVariables}
    {assignment : ColumnId -> F}
    {recursive : ProductionPaperRecursiveRelationRowsSoundFor.Result candidate
      statementId config artifact priorAuthority (pulledAssignment assignment)
      headers priorPrefix value proof}
    (family : Family (manifest := manifest) statementId config artifact
      priorAuthority headers priorPrefix value proof assignment recursive)
    (numericOne : pulledAssignment assignment 0 = 1) :
    ProductTerminalRelation.CoreHolds
      (ProductionPaperTerminalInvocationRowsSoundFor.children candidate
        statementId config artifact value proof)
      family.assignments := by
  intro child
  refine ⟨ProductionPaperTerminalInvocationRowsSoundFor.children_stage
    candidate statementId config artifact value proof child, ?_, ?_⟩
  · simpa [assignments,
      ProductionPaperTerminalInvocationRowsSoundFor.children] using
      ProductionPaperTerminalCoreRowsFor.public_exact
        ((family.child child).typedOne numericOne)
        (family.coreEvidence child)
  · simpa [assignments,
      ProductionPaperTerminalInvocationRowsSoundFor.children] using
      ProductionPaperTerminalCoreRowsFor.evaluations_exact
        ((family.child child).typedOne numericOne)
        (family.coreEvidence child)

theorem holds
    {manifest : SeedSchedule.Manifest}
    {candidate : ProductionProfileCandidates.Id}
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    {statementId : ProductConcreteNifsFor.StatementId}
    {config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape}
    {artifact : ProductConcreteNifsFor.RelationArtifact rowVariables
      logicalWidth publicFits}
    {priorAuthority : ProductionPaperPriorStateAuthorityRowsFor.Layout candidate
      rowVariables}
    {headers : FPrime.ChainHeaders Digest.Value}
    {priorPrefix : ProductionPaperPriorStateAuthorityRowsFor.Prefix candidate
      (FullShape rowVariables logicalWidth publicFits)}
    {value : ProductionFieldNativeFullClaim.Value candidate
      (FullShape rowVariables logicalWidth publicFits)}
    {proof : ProductionProductPiCcsTypedBridgeFor.ExactProof rowVariables}
    {assignment : ColumnId -> F}
    {recursive : ProductionPaperRecursiveRelationRowsSoundFor.Result candidate
      statementId config artifact priorAuthority (pulledAssignment assignment)
      headers priorPrefix value proof}
    (family : Family (manifest := manifest) statementId config artifact
      priorAuthority headers priorPrefix value proof assignment recursive)
    (numericOne : pulledAssignment assignment 0 = 1) :
    ProductTerminalRelation.Holds config
      (ProductionPaperTerminalInvocationRowsSoundFor.children candidate
        statementId config artifact value proof)
      family.assignments := by
  apply ProductTerminalRelation.holds_of_common_openings config _ _
  · intro child
    exact (family.child child).bounded numericOne
  · intro child
    have placed : ProductionProductNifsOutputRowsFor.Placed family.outputLayout
        (pulledAssignment assignment) (pulledAssignment_canonical assignment)
        (ProductionPaperTerminalInvocationRowsSoundFor.finalRunning candidate
          statementId config artifact value proof) := by
      rw [family.outputLayoutExact]
      exact recursive.nifsOutputPlaced
    exact (family.child child).opens numericOne placed
  · exact family.coreHolds numericOne

noncomputable def opening
    {manifest : SeedSchedule.Manifest}
    {candidate : ProductionProfileCandidates.Id}
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    {statementId : ProductConcreteNifsFor.StatementId}
    {config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape}
    {artifact : ProductConcreteNifsFor.RelationArtifact rowVariables
      logicalWidth publicFits}
    {priorAuthority : ProductionPaperPriorStateAuthorityRowsFor.Layout candidate
      rowVariables}
    {headers : FPrime.ChainHeaders Digest.Value}
    {priorPrefix : ProductionPaperPriorStateAuthorityRowsFor.Prefix candidate
      (FullShape rowVariables logicalWidth publicFits)}
    {value : ProductionFieldNativeFullClaim.Value candidate
      (FullShape rowVariables logicalWidth publicFits)}
    {proof : ProductionProductPiCcsTypedBridgeFor.ExactProof rowVariables}
    {assignment : ColumnId -> F}
    {recursive : ProductionPaperRecursiveRelationRowsSoundFor.Result candidate
      statementId config artifact priorAuthority (pulledAssignment assignment)
      headers priorPrefix value proof}
    (family : Family (manifest := manifest) statementId config artifact
      priorAuthority headers priorPrefix value proof assignment recursive)
    (numericOne : pulledAssignment assignment 0 = 1) :
    ProductionPaperTerminalInvocationRowsSoundFor.ProductOpening candidate
      statementId config artifact value proof :=
  ProductionPaperTerminalInvocationRowsSoundFor.ProductOpening.ofHolds
    family.assignments (family.holds numericOne)

end Family

end Nightstream.Implementation.Nebula.ProductionPaperTerminalOpeningRowsFor
