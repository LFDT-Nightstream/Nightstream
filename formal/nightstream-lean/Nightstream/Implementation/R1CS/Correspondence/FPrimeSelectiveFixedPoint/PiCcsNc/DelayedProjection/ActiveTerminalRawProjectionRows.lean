import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.ActiveTerminalRawProjection
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.ActiveTerminalRawProjectionRows.RawScalarKernel
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.PackedWitnessOldBlockProjection
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.ProductionMessageAcceptance
import Nightstream.Implementation.R1CS.Correspondence.Gadgets.TerminalRawOldBlockProjectionFinalScaleCompiler
import Nightstream.Implementation.R1CS.Correspondence.Gadgets.TerminalRawOldBlockProjectionTensorPrefix
import Nightstream.Implementation.R1CS.Correspondence.PiCcsNc.Authority.DelayedResidual.CombinedNc.ProductionTerminal.TerminalCEBridge

/-!
Internal row-to-terminal seam for the direct raw-old-block projection.

Assurance tier: model-level generic composition.  The active generated-layout
leaf must discharge every column-map field below before this seam can support
an artifact-checked or Rust-conformant claim.

The contract has one ordered `finalWitnesses` family.  The projection rows
read that family through exact `rawWitnessColumn` equalities, while terminal
CE opens definitionally the same `unpack (finalWitnesses child)` values.
There is no independent raw-child family, child `y_zcol` sidecar, digest
authority, desired projection premise, or implementation-refinement event.

Owns: the generic row-to-semantics proof that exact indexed projection rows,
canonical column decoding, padding, and terminal CE over one shared packed
witness family imply `ProjectionOpeningAccepted`.

Does not own: the fixed production emitter layout, Rust column allocation,
physical row placement, production assignment construction, transcript
sampling, or the independent paper/`y_ring` track.

Emits constraints: no; consumes the indexed physical projection rows.

| Stable stage path | Mathematical obligation | Authority class |
|---|---|---|
| `f_prime.pi_ccs_nc.delayed.terminal_rows.columns` | decoded old-block, parent, and raw-child columns equal their typed sources | direct dataflow premise / derived |
| `f_prime.pi_ccs_nc.delayed.terminal_rows.projection` | tensor, product, and terminal rows imply the packed old-point projection equality | derived |
| `f_prime.pi_ccs_nc.delayed.terminal_rows.opening` | the row-derived projection and same-family terminal CE imply `ProjectionOpeningAccepted` | derived |
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.ActiveTerminalRawProjectionRows

open Nightstream.Protocol
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Authority
open Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.ProjectionProgram
open Nightstream.Implementation.R1CS.TerminalRawOldBlockProjectionCompiler
open Nightstream.Implementation.R1CS.TerminalRawOldBlockProjectionTensorPrefix
open Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual
open Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual.CombinedNc
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.ProductionMessageAcceptance
open PackedWitness

private abbrev productionShape := ProductionDomain.semanticShape
private abbrev productionDomain := PiCcsDomains.production.nc
private abbrev productionCovers := ProductionDomain.blockLaneDomain_covers

universe uState

section Context

variable
  {State : Type uState}
  {publicRingColumns verifierRows : Nat}
  {publicFits : ringDegree * publicRingColumns <=
    productionShape.carrierWidth}

private theorem productionChildCount : productionGlobalParams.k = 14 := by
  rfl

private theorem productionBlockVariables :
    productionDomain.blockVariables = 19 := by
  rfl

private theorem productionActiveLanes : ringDegree = 54 := by
  rfl

/-- Reindex a point without changing its ordered coordinate list. -/
private def castPoint {Value : Type} {left right : Nat}
    (equal : left = right) (point : CubePoint Value right) :
    CubePoint Value left where
  coordinates := point.coordinates
  dimension := point.dimension.trans equal.symm

private theorem cubePoint_ext {Value : Type} {variables : Nat}
    (left right : CubePoint Value variables)
    (coordinates : left.coordinates = right.coordinates) : left = right := by
  rcases left with ⟨leftCoordinates, leftDimension⟩
  rcases right with ⟨rightCoordinates, rightDimension⟩
  simp only at coordinates
  cases coordinates
  rfl

private def childCountEq {layout : Layout}
    {artifactRow : Fin (rowCount layout) -> Row}
    (contract : ArtifactContract layout artifactRow) :
    layout.childCount = productionGlobalParams.k :=
  contract.profileChildren.trans productionChildCount.symm

private def activeLaneCountEq {layout : Layout}
    {artifactRow : Fin (rowCount layout) -> Row}
    (contract : ArtifactContract layout artifactRow) :
    layout.activeLanes = ringDegree :=
  contract.profileActiveLanes.trans productionActiveLanes.symm

private def logicalWidthEq {layout : Layout}
    {artifactRow : Fin (rowCount layout) -> Row}
    (contract : ArtifactContract layout artifactRow) :
    layout.logicalWidth = productionShape.carrierWidth :=
  contract.profileLogicalWidth.trans
    ProductionDomain.semanticShape_carrierWidth.symm

private def blockVariablesEq {layout : Layout}
    {artifactRow : Fin (rowCount layout) -> Row}
    (contract : ArtifactContract layout artifactRow) :
    layout.blockVariables = productionDomain.blockVariables :=
  contract.profileBlockVariables.trans productionBlockVariables.symm

private def blockCountEq {layout : Layout}
    {artifactRow : Fin (rowCount layout) -> Row}
    (contract : ArtifactContract layout artifactRow) :
    blockCount layout =
      Phi81ColumnLayout.blockCount productionShape.carrierWidth :=
  contract.profileBlockCount.trans
    ProductionDomain.semanticShape_blockCount.symm

private def productionChild {layout : Layout}
    {artifactRow : Fin (rowCount layout) -> Row}
    (contract : ArtifactContract layout artifactRow)
    (child : Fin layout.childCount) : Fin productionGlobalParams.k :=
  Fin.cast (childCountEq contract) child

private def productionLane {layout : Layout}
    {artifactRow : Fin (rowCount layout) -> Row}
    (contract : ArtifactContract layout artifactRow)
    (lane : Fin layout.activeLanes) : Fin ringDegree :=
  Fin.cast (activeLaneCountEq contract) lane

private def productionCoordinate {layout : Layout}
    {artifactRow : Fin (rowCount layout) -> Row}
    (contract : ArtifactContract layout artifactRow)
    (coordinate : Fin layout.logicalWidth) :
    Fin productionShape.carrierWidth :=
  Fin.cast (logicalWidthEq contract) coordinate

private def factoredChildCountEq
    {layout : TerminalRawOldBlockProjectionFinalScaleCompiler.Layout}
    {artifactRow : Fin
      (TerminalRawOldBlockProjectionFinalScaleCompiler.rowCount layout) -> Row}
    (contract :
      TerminalRawOldBlockProjectionFinalScaleCompiler.ArtifactContract
        layout artifactRow) :
    layout.base.childCount = productionGlobalParams.k :=
  contract.profileChildren.trans productionChildCount.symm

private def factoredActiveLaneCountEq
    {layout : TerminalRawOldBlockProjectionFinalScaleCompiler.Layout}
    {artifactRow : Fin
      (TerminalRawOldBlockProjectionFinalScaleCompiler.rowCount layout) -> Row}
    (contract :
      TerminalRawOldBlockProjectionFinalScaleCompiler.ArtifactContract
        layout artifactRow) :
    layout.base.activeLanes = ringDegree :=
  contract.profileActiveLanes.trans productionActiveLanes.symm

private def factoredLogicalWidthEq
    {layout : TerminalRawOldBlockProjectionFinalScaleCompiler.Layout}
    {artifactRow : Fin
      (TerminalRawOldBlockProjectionFinalScaleCompiler.rowCount layout) -> Row}
    (contract :
      TerminalRawOldBlockProjectionFinalScaleCompiler.ArtifactContract
        layout artifactRow) :
    layout.base.logicalWidth = productionShape.carrierWidth :=
  contract.profileLogicalWidth.trans
    ProductionDomain.semanticShape_carrierWidth.symm

private theorem factoredPrefixVariables
    {layout : TerminalRawOldBlockProjectionFinalScaleCompiler.Layout}
    {artifactRow : Fin
      (TerminalRawOldBlockProjectionFinalScaleCompiler.rowCount layout) -> Row}
    (contract :
      TerminalRawOldBlockProjectionFinalScaleCompiler.ArtifactContract
        layout artifactRow) :
    layout.base.blockVariables = 18 := by
  rw [← contract.shape.tensorVariables,
    contract.profileTensorVariables]

private def factoredBlockVariablesEq
    {layout : TerminalRawOldBlockProjectionFinalScaleCompiler.Layout}
    {artifactRow : Fin
      (TerminalRawOldBlockProjectionFinalScaleCompiler.rowCount layout) -> Row}
    (contract :
      TerminalRawOldBlockProjectionFinalScaleCompiler.ArtifactContract
        layout artifactRow) :
    layout.base.blockVariables + 1 = productionDomain.blockVariables := by
  rw [factoredPrefixVariables contract, productionBlockVariables]

private def factoredBlockCountEq
    {layout : TerminalRawOldBlockProjectionFinalScaleCompiler.Layout}
    {artifactRow : Fin
      (TerminalRawOldBlockProjectionFinalScaleCompiler.rowCount layout) -> Row}
    (contract :
      TerminalRawOldBlockProjectionFinalScaleCompiler.ArtifactContract
        layout artifactRow) :
    blockCount layout.base =
      Phi81ColumnLayout.blockCount productionShape.carrierWidth :=
  contract.profileBlockCount.trans
    ProductionDomain.semanticShape_blockCount.symm

private def factoredProductionChild
    {layout : TerminalRawOldBlockProjectionFinalScaleCompiler.Layout}
    {artifactRow : Fin
      (TerminalRawOldBlockProjectionFinalScaleCompiler.rowCount layout) -> Row}
    (contract :
      TerminalRawOldBlockProjectionFinalScaleCompiler.ArtifactContract
        layout artifactRow)
    (child : Fin layout.base.childCount) : Fin productionGlobalParams.k :=
  Fin.cast (factoredChildCountEq contract) child

private def factoredProductionLane
    {layout : TerminalRawOldBlockProjectionFinalScaleCompiler.Layout}
    {artifactRow : Fin
      (TerminalRawOldBlockProjectionFinalScaleCompiler.rowCount layout) -> Row}
    (contract :
      TerminalRawOldBlockProjectionFinalScaleCompiler.ArtifactContract
        layout artifactRow)
    (lane : Fin layout.base.activeLanes) : Fin ringDegree :=
  Fin.cast (factoredActiveLaneCountEq contract) lane

private def factoredProductionCoordinate
    {layout : TerminalRawOldBlockProjectionFinalScaleCompiler.Layout}
    {artifactRow : Fin
      (TerminalRawOldBlockProjectionFinalScaleCompiler.rowCount layout) -> Row}
    (contract :
      TerminalRawOldBlockProjectionFinalScaleCompiler.ArtifactContract
        layout artifactRow)
    (coordinate : Fin layout.base.logicalWidth) :
    Fin productionShape.carrierWidth :=
  Fin.cast (factoredLogicalWidthEq contract) coordinate

/-- Exact internal execution-audit contract.  The generated active
specialization must prove the mapping fields from its fixed layout; callers
must not retain this structure as an opaque production-authority premise. -/
structure TerminalExecutionAudit
    (context : FixedActive.Context productionShape State publicRingColumns
      publicFits verifierRows)
    (certificate : FixedActive.Certificate context)
    (layout : Layout)
    (artifactRow : Fin (rowCount layout) -> Row)
    (assignment : Nat -> Nat)
    (finalWitnesses : Fin productionGlobalParams.k -> Matrix productionShape) where
  artifact : ArtifactContract layout artifactRow
  rows : ArtifactRowsSatisfied artifact assignment
  canonical : forall column, assignment column < goldilocksP
  constantOne : assignment 0 = 1
  oldBlockColumns :
    (oldBlockValues layout assignment).map toConcreteK =
      (castPoint (blockVariablesEq artifact)
        (DelayedProduction.outgoingPending context certificate).oldBlock).coordinates
  parentColumns : forall lane : Fin layout.activeLanes,
    toConcreteK ((layout.parent lane).value assignment) =
      (DelayedProduction.outgoingPending context certificate).parentYZcol
        (productionLane artifact lane)
  rawWitnessColumns : forall
      (child : Fin layout.childCount)
      (coordinate : Fin layout.logicalWidth),
    assignment (rawWitnessColumn layout child coordinate) =
      (unpack (finalWitnesses (productionChild artifact child))
        (productionCoordinate artifact coordinate)).val
  terminalCE : TerminalCE.Holds
    (ProductionTerminal.TerminalCEBridge.semantics context)
    (ProductionTerminal.TerminalCEBridge.terminalInstance context certificate
      (fun child => unpack (finalWitnesses child)))

/-- Execution-audit contract for the final-round-factorized emitter.  Its
old-block field names all nineteen authoritative columns directly: the
eighteen tensor-prefix columns followed by the emitted common-factor column.
The structure contains no projection conclusion or generic refinement event. -/
structure FactoredTerminalExecutionAudit
    (context : FixedActive.Context productionShape State publicRingColumns
      publicFits verifierRows)
    (certificate : FixedActive.Certificate context)
    (layout : TerminalRawOldBlockProjectionFinalScaleCompiler.Layout)
    (artifactRow : Fin
      (TerminalRawOldBlockProjectionFinalScaleCompiler.rowCount layout) -> Row)
    (assignment : Nat -> Nat)
    (finalWitnesses : Fin productionGlobalParams.k -> Matrix productionShape) where
  artifact :
    TerminalRawOldBlockProjectionFinalScaleCompiler.ArtifactContract
      layout artifactRow
  rows : TerminalRawOldBlockProjectionFinalScaleCompiler.ArtifactRowsSatisfied
    artifact assignment
  canonical : forall column, assignment column < goldilocksP
  constantOne : assignment 0 = 1
  oldBlockColumns :
    ((oldBlockValues layout.base assignment ++
      [layout.factor.finalPoint.value assignment]).map toConcreteK) =
      (castPoint (factoredBlockVariablesEq artifact)
        (DelayedProduction.outgoingPending context certificate).oldBlock).coordinates
  parentColumns : forall lane : Fin layout.base.activeLanes,
    toConcreteK ((layout.base.parent lane).value assignment) =
      (DelayedProduction.outgoingPending context certificate).parentYZcol
        (factoredProductionLane artifact lane)
  rawWitnessColumns : forall
      (child : Fin layout.base.childCount)
      (coordinate : Fin layout.base.logicalWidth),
    assignment (rawWitnessColumn layout.base child coordinate) =
      (unpack (finalWitnesses (factoredProductionChild artifact child))
        (factoredProductionCoordinate artifact coordinate)).val
  terminalCE : TerminalCE.Holds
    (ProductionTerminal.TerminalCEBridge.semantics context)
    (ProductionTerminal.TerminalCEBridge.terminalInstance context certificate
      (fun child => unpack (finalWitnesses child)))

/-! ## Carrier conversion -/

@[simp] private theorem toConcreteK_ofBase (value : ProjectionProgram.F) :
    toConcreteK (ProjectionProgram.K.ofBase value) =
      Concrete.K.embed (toConcreteField value) := by
  rfl

@[simp] private theorem toConcreteK_sub
    (left right : ProjectionProgram.K) :
    toConcreteK
        (TerminalRawOldBlockProjectionCompiler.K.sub left right) =
      Concrete.K.sub (toConcreteK left) (toConcreteK right) := by
  rcases left with ⟨left0, left1⟩
  rcases right with ⟨right0, right1⟩
  simp only [TerminalRawOldBlockProjectionCompiler.K.sub, toConcreteK,
    Concrete.K.sub, Concrete.K.mk.injEq]
  constructor <;> apply Fin.ext <;> rfl

private theorem projectionTensorOps_sub
    (left right : ProjectionProgram.K) :
    InterpolationOps.sub projectionTensorOps left right =
      TerminalRawOldBlockProjectionCompiler.K.sub left right := by
  rcases left with ⟨left0, left1⟩
  rcases right with ⟨right0, right1⟩
  simp only [InterpolationOps.sub, projectionTensorOps,
    ProjectionProgram.K.add, TerminalRawOldBlockProjectionCompiler.K.sub,
    ProjectionProgram.K.zero, ProjectionProgram.K.mk.injEq,
    Fin.sub_eq_add_neg]
  constructor <;> simp [ProjectionProgram.K.add]

private theorem concreteTensorOps_sub
    (left right : Concrete.K) :
    InterpolationOps.sub ConcreteCarrier.extensionOps left right =
      Concrete.K.sub left right := by
  rcases left with ⟨left0, left1⟩
  rcases right with ⟨right0, right1⟩
  simp only [InterpolationOps.sub, ConcreteCarrier.extensionOps,
    Concrete.K.add, Concrete.K.sub, Concrete.K.zero, Concrete.K.mk.injEq,
    Fin.sub_eq_add_neg]
  constructor <;> simp [Concrete.K.add]

private def mapPoint {variables : Nat}
    (point : CubePoint ProjectionProgram.K variables) :
    CubePoint Concrete.K variables where
  coordinates := point.coordinates.map toConcreteK
  dimension := by simp [point.dimension]

private theorem mappedGetD (values : List ProjectionProgram.K)
    (index : Nat) :
    toConcreteK (values.getD index ProjectionProgram.K.zero) =
      (values.map toConcreteK).getD index Concrete.K.zero := by
  by_cases within : index < values.length
  · rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem within,
      List.getD_eq_getElem?_getD]
    have mappedWithin : index < (values.map toConcreteK).length := by
      simpa using within
    rw [List.getElem?_eq_getElem mappedWithin, List.getElem_map]
    rfl
  · have outside : values.length <= index := Nat.le_of_not_gt within
    have mappedOutside : (values.map toConcreteK).length <= index := by
      simpa using outside
    rw [List.getD_eq_getElem?_getD, List.getElem?_eq_none outside,
      List.getD_eq_getElem?_getD, List.getElem?_eq_none mappedOutside]
    rfl

private theorem testBitWeight_map
    {variables : Nat}
    (point : CubePoint ProjectionProgram.K variables)
    (index : Fin (2 ^ variables)) :
    toConcreteK
        (NumericBooleanDomain.testBitWeight
          projectionTensorOps point index) =
      NumericBooleanDomain.testBitWeight ConcreteCarrier.extensionOps
        (mapPoint point) index := by
  unfold NumericBooleanDomain.testBitWeight
  simp only [projectionTensorOps_sub, concreteTensorOps_sub]
  let bits := canonicalFinIndices variables
  have mappedFold : forall (indices : List (Fin variables))
      (initial : ProjectionProgram.K),
      toConcreteK
          (indices.foldl
            (fun accumulated bit =>
              let coordinate :=
                point.coordinates.getD bit.val ProjectionProgram.K.zero
              let factor := if Nat.testBit index.val bit.val then
                coordinate
              else
                TerminalRawOldBlockProjectionCompiler.K.sub
                  ProjectionProgram.K.one coordinate
              ProjectionProgram.K.mul accumulated factor)
            initial) =
        indices.foldl
          (fun accumulated bit =>
            let coordinate :=
              (mapPoint point).coordinates.getD bit.val Concrete.K.zero
            let factor := if Nat.testBit index.val bit.val then
              coordinate
            else Concrete.K.sub Concrete.K.one coordinate
            Concrete.K.mul accumulated factor)
          (toConcreteK initial) := by
    intro indices initial
    induction indices generalizing initial with
    | nil => rfl
    | cons bit indices inductionHypothesis =>
        simp only [List.foldl_cons]
        rw [inductionHypothesis, toConcreteK_mul]
        congr 1
        apply congrArg (fun factor =>
          Concrete.K.mul (toConcreteK initial) factor)
        by_cases selected : Nat.testBit index.val bit.val
        · simp only [selected, if_true]
          change toConcreteK
              (point.coordinates.getD bit.val ProjectionProgram.K.zero) =
            (point.coordinates.map toConcreteK).getD bit.val Concrete.K.zero
          exact mappedGetD point.coordinates bit.val
        · have bitFalse : Nat.testBit index.val bit.val = false :=
            Bool.eq_false_iff.mpr selected
          simp only [bitFalse, Bool.false_eq_true, if_false, toConcreteK_sub,
            toConcreteK_one]
          apply congrArg (Concrete.K.sub Concrete.K.one)
          change toConcreteK
              (point.coordinates.getD bit.val ProjectionProgram.K.zero) =
            (point.coordinates.map toConcreteK).getD bit.val Concrete.K.zero
          exact mappedGetD point.coordinates bit.val
  simpa [bits, mapPoint, projectionTensorOps,
    ConcreteCarrier.extensionOps, toConcreteK_one] using
    mappedFold (canonicalFinIndices variables) ProjectionProgram.K.one

private theorem testBitWeight_cast
    {left right : Nat} (equal : left = right)
    (point : CubePoint Concrete.K right)
    (index : Fin (2 ^ left)) :
    NumericBooleanDomain.testBitWeight ConcreteCarrier.extensionOps
        (castPoint equal point) index =
      NumericBooleanDomain.testBitWeight ConcreteCarrier.extensionOps point
        (Fin.cast (congrArg (fun variables => 2 ^ variables) equal) index) := by
  cases equal
  rfl

private theorem mappedOldBlockPoint
    {context : FixedActive.Context productionShape State publicRingColumns
      publicFits verifierRows}
    {certificate : FixedActive.Certificate context}
    {layout : Layout}
    {artifactRow : Fin (rowCount layout) -> Row}
    {assignment : Nat -> Nat}
    {finalWitnesses : Fin productionGlobalParams.k -> Matrix productionShape}
    (audit : TerminalExecutionAudit context certificate layout artifactRow
      assignment finalWitnesses) :
    mapPoint (oldBlockPoint layout assignment) =
      castPoint (blockVariablesEq audit.artifact)
        (DelayedProduction.outgoingPending context certificate).oldBlock := by
  apply cubePoint_ext
  exact audit.oldBlockColumns

private theorem mappedFactoredOldBlockPoint
    {context : FixedActive.Context productionShape State publicRingColumns
      publicFits verifierRows}
    {certificate : FixedActive.Certificate context}
    {layout : TerminalRawOldBlockProjectionFinalScaleCompiler.Layout}
    {artifactRow : Fin
      (TerminalRawOldBlockProjectionFinalScaleCompiler.rowCount layout) -> Row}
    {assignment : Nat -> Nat}
    {finalWitnesses : Fin productionGlobalParams.k -> Matrix productionShape}
    (audit : FactoredTerminalExecutionAudit context certificate layout
      artifactRow assignment finalWitnesses) :
    mapPoint
        (appendPoint (oldBlockPoint layout.base assignment)
          (layout.factor.finalPoint.value assignment)) =
      castPoint (factoredBlockVariablesEq audit.artifact)
        (DelayedProduction.outgoingPending context certificate).oldBlock := by
  apply cubePoint_ext
  simpa [mapPoint, appendPoint] using audit.oldBlockColumns

end Context

/-! ## Raw-column semantics -/

private theorem productionWidthProduct {layout : Layout}
    {artifactRow : Fin (rowCount layout) -> Row}
    (contract : ArtifactContract layout artifactRow) :
    layout.logicalWidth = blockCount layout * layout.activeLanes := by
  rw [contract.profileLogicalWidth, contract.profileBlockCount,
    contract.profileActiveLanes]

private def laneCoordinate {layout : Layout}
    {artifactRow : Fin (rowCount layout) -> Row}
    (contract : ArtifactContract layout artifactRow)
    (lane : Fin layout.activeLanes) (block : Fin (blockCount layout)) :
    Fin layout.logicalWidth :=
  ⟨block.val * layout.activeLanes + lane.val, by
    have width := productionWidthProduct contract
    have blockBound := block.isLt
    have laneBound := lane.isLt
    have scaled :
        (block.val + 1) * layout.activeLanes <=
          blockCount layout * layout.activeLanes :=
      Nat.mul_le_mul_right layout.activeLanes
        (Nat.succ_le_iff.mpr blockBound)
    rw [width]
    simp only [Nat.add_mul, Nat.one_mul] at scaled
    omega⟩

private def laneCoordinateNat {layout : Layout}
    {artifactRow : Fin (rowCount layout) -> Row}
    (contract : ArtifactContract layout artifactRow)
    (lane : Fin layout.activeLanes) (block : Nat) :
    Fin layout.logicalWidth :=
  if within : block < blockCount layout then
    laneCoordinate contract lane ⟨block, within⟩
  else
    ⟨0, by rw [contract.profileLogicalWidth]; decide⟩

private theorem filterMap_eq_map_on
    {Source Target : Type} (values : List Source)
    (filter : Source -> Option Target) (map : Source -> Target)
    (aligns : forall value, value ∈ values -> filter value = some (map value)) :
    values.filterMap filter = values.map map := by
  induction values with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      have headAligned : filter head = some (map head) :=
        aligns head (by simp)
      have tailAligned : forall value, value ∈ tail ->
          filter value = some (map value) := by
        intro value member
        exact aligns value (by simp [member])
      have tailEq : tail.filterMap filter = tail.map map :=
        inductionHypothesis tailAligned
      simp [headAligned, tailEq]

private theorem laneCoordinates_eq_mapRange {layout : Layout}
    {artifactRow : Fin (rowCount layout) -> Row}
    (contract : ArtifactContract layout artifactRow)
    (lane : Fin layout.activeLanes) :
    laneCoordinates layout lane =
      (List.range (blockCount layout)).map
        (laneCoordinateNat contract lane) := by
  unfold laneCoordinates
  apply filterMap_eq_map_on
  intro block member
  have blockBound : block < blockCount layout := List.mem_range.mp member
  have coordinateBound :
      block * layout.activeLanes + lane.val < layout.logicalWidth := by
    have width := productionWidthProduct contract
    have laneBound := lane.isLt
    have scaled :
        (block + 1) * layout.activeLanes <=
          blockCount layout * layout.activeLanes :=
      Nat.mul_le_mul_right layout.activeLanes
        (Nat.succ_le_iff.mpr blockBound)
    rw [width]
    simp only [Nat.add_mul, Nat.one_mul] at scaled
    omega
  simp only [coordinateBound, dif_pos, laneCoordinateNat, blockBound]
  apply congrArg some
  apply Fin.ext
  rfl

private theorem semanticLaneCoordinate {layout : Layout}
    {artifactRow : Fin (rowCount layout) -> Row}
    (contract : ArtifactContract layout artifactRow)
    (lane : Fin layout.activeLanes) (block : Fin (blockCount layout)) :
    productionCoordinate contract (laneCoordinate contract lane block) =
      Phi81CarrierLayout.carrierColumn
        (Fin.cast (blockCountEq contract) block)
        (productionLane contract lane) := by
  unfold productionCoordinate laneCoordinate productionLane
  unfold Phi81CarrierLayout.carrierColumn Phi81ColumnLayout.flatIndex
  exact Fin.ext (by
    change block.val * layout.activeLanes + lane.val =
      block.val * ringDegree + lane.val
    exact congrArg (fun lanes => block.val * lanes + lane.val)
      (activeLaneCountEq contract))

private theorem factoredProductionWidthProduct
    {layout : TerminalRawOldBlockProjectionFinalScaleCompiler.Layout}
    {artifactRow : Fin
      (TerminalRawOldBlockProjectionFinalScaleCompiler.rowCount layout) -> Row}
    (contract :
      TerminalRawOldBlockProjectionFinalScaleCompiler.ArtifactContract
        layout artifactRow) :
    layout.base.logicalWidth =
      blockCount layout.base * layout.base.activeLanes := by
  rw [contract.profileLogicalWidth, contract.profileBlockCount,
    contract.profileActiveLanes]

private def factoredLaneCoordinate
    {layout : TerminalRawOldBlockProjectionFinalScaleCompiler.Layout}
    {artifactRow : Fin
      (TerminalRawOldBlockProjectionFinalScaleCompiler.rowCount layout) -> Row}
    (contract :
      TerminalRawOldBlockProjectionFinalScaleCompiler.ArtifactContract
        layout artifactRow)
    (lane : Fin layout.base.activeLanes)
    (block : Fin (blockCount layout.base)) :
    Fin layout.base.logicalWidth :=
  ⟨block.val * layout.base.activeLanes + lane.val, by
    have width := factoredProductionWidthProduct contract
    have scaled :
        (block.val + 1) * layout.base.activeLanes <=
          blockCount layout.base * layout.base.activeLanes :=
      Nat.mul_le_mul_right layout.base.activeLanes
        (Nat.succ_le_iff.mpr block.isLt)
    rw [width]
    simp only [Nat.add_mul, Nat.one_mul] at scaled
    omega⟩

private def factoredLaneCoordinateNat
    {layout : TerminalRawOldBlockProjectionFinalScaleCompiler.Layout}
    {artifactRow : Fin
      (TerminalRawOldBlockProjectionFinalScaleCompiler.rowCount layout) -> Row}
    (contract :
      TerminalRawOldBlockProjectionFinalScaleCompiler.ArtifactContract
        layout artifactRow)
    (lane : Fin layout.base.activeLanes) (block : Nat) :
    Fin layout.base.logicalWidth :=
  if within : block < blockCount layout.base then
    factoredLaneCoordinate contract lane ⟨block, within⟩
  else
    ⟨0, by rw [contract.profileLogicalWidth]; decide⟩

private theorem factoredLaneCoordinates_eq_mapRange
    {layout : TerminalRawOldBlockProjectionFinalScaleCompiler.Layout}
    {artifactRow : Fin
      (TerminalRawOldBlockProjectionFinalScaleCompiler.rowCount layout) -> Row}
    (contract :
      TerminalRawOldBlockProjectionFinalScaleCompiler.ArtifactContract
        layout artifactRow)
    (lane : Fin layout.base.activeLanes) :
    laneCoordinates layout.base lane =
      (List.range (blockCount layout.base)).map
        (factoredLaneCoordinateNat contract lane) := by
  unfold laneCoordinates
  apply filterMap_eq_map_on
  intro block member
  have blockBound : block < blockCount layout.base := List.mem_range.mp member
  have coordinateBound :
      block * layout.base.activeLanes + lane.val <
        layout.base.logicalWidth := by
    have width := factoredProductionWidthProduct contract
    have scaled :
        (block + 1) * layout.base.activeLanes <=
          blockCount layout.base * layout.base.activeLanes :=
      Nat.mul_le_mul_right layout.base.activeLanes
        (Nat.succ_le_iff.mpr blockBound)
    rw [width]
    simp only [Nat.add_mul, Nat.one_mul] at scaled
    omega
  simp only [coordinateBound, dif_pos, factoredLaneCoordinateNat, blockBound]
  apply congrArg some
  apply Fin.ext
  rfl

private theorem factoredSemanticLaneCoordinate
    {layout : TerminalRawOldBlockProjectionFinalScaleCompiler.Layout}
    {artifactRow : Fin
      (TerminalRawOldBlockProjectionFinalScaleCompiler.rowCount layout) -> Row}
    (contract :
      TerminalRawOldBlockProjectionFinalScaleCompiler.ArtifactContract
        layout artifactRow)
    (lane : Fin layout.base.activeLanes)
    (block : Fin (blockCount layout.base)) :
    factoredProductionCoordinate contract
        (factoredLaneCoordinate contract lane block) =
      Phi81CarrierLayout.carrierColumn
        (Fin.cast (factoredBlockCountEq contract) block)
        (factoredProductionLane contract lane) := by
  unfold factoredProductionCoordinate factoredLaneCoordinate
  unfold factoredProductionLane
  unfold Phi81CarrierLayout.carrierColumn Phi81ColumnLayout.flatIndex
  exact Fin.ext (by
    change block.val * layout.base.activeLanes + lane.val =
      block.val * ringDegree + lane.val
    exact congrArg (fun lanes => block.val * lanes + lane.val)
      (factoredActiveLaneCountEq contract))

section Context

variable
  {State : Type uState}
  {publicRingColumns verifierRows : Nat}
  {publicFits : ringDegree * publicRingColumns <=
    productionShape.carrierWidth}

private theorem rawScalar_eq_recomposed
    {context : FixedActive.Context productionShape State publicRingColumns
      publicFits verifierRows}
    {certificate : FixedActive.Certificate context}
    {layout : Layout}
    {artifactRow : Fin (rowCount layout) -> Row}
    {assignment : Nat -> Nat}
    {finalWitnesses : Fin productionGlobalParams.k -> Matrix productionShape}
    (audit : TerminalExecutionAudit context certificate layout artifactRow
      assignment finalWitnesses)
    (coordinate : Fin layout.logicalWidth) :
    toConcreteField
        (ProjectionProgram.residue
          (lcEval assignment (rawTerms layout coordinate))) =
      PiDEC.Raw.recomposeAssignment
        (fun child => unpack (finalWitnesses child))
        (productionCoordinate audit.artifact coordinate) := by
  let columns : Fin productionGlobalParams.k -> Nat := fun child =>
    rawWitnessColumn layout
      (Fin.cast (childCountEq audit.artifact).symm child) coordinate
  have termsExact : rawTerms layout coordinate =
      List.ofFn fun child : Fin productionGlobalParams.k =>
        (columns child,
          productionGlobalParams.b ^ child.val % goldilocksP) := by
    apply List.ext_get
    · simp [rawTerms, audit.artifact.profileChildren,
        productionGlobalParams]
    · intro index leftLt rightLt
      have leftBound : index < layout.childCount := by
        simpa [rawTerms] using leftLt
      have rightBound : index < productionGlobalParams.k := by
        simpa using rightLt
      have childExact : (⟨index, leftBound⟩ : Fin layout.childCount) =
          Fin.cast (childCountEq audit.artifact).symm
            (⟨index, rightBound⟩ : Fin productionGlobalParams.k) := by
        apply Fin.ext
        rfl
      simp only [rawTerms, List.get_eq_getElem, List.getElem_ofFn]
      rw [childExact]
      simp [columns, radixCoefficient, audit.artifact.profileRadix,
        productionGlobalParams]
  have columnsExact : forall child : Fin productionGlobalParams.k,
      assignment (columns child) =
        (unpack (finalWitnesses child)
          (productionCoordinate audit.artifact coordinate)).val := by
    intro child
    have mapped := audit.rawWitnessColumns
      (Fin.cast (childCountEq audit.artifact).symm child) coordinate
    have childExact : productionChild audit.artifact
        (Fin.cast (childCountEq audit.artifact).symm child) = child := by
      apply Fin.ext
      rfl
    rw [childExact] at mapped
    exact mapped
  rw [termsExact]
  exact RawScalarKernel.lcEval_radixTerms_eq_recomposeScalar
    assignment columns (fun child => unpack (finalWitnesses child))
    (productionCoordinate audit.artifact coordinate) columnsExact

private theorem factoredRawScalar_eq_recomposed
    {context : FixedActive.Context productionShape State publicRingColumns
      publicFits verifierRows}
    {certificate : FixedActive.Certificate context}
    {layout : TerminalRawOldBlockProjectionFinalScaleCompiler.Layout}
    {artifactRow : Fin
      (TerminalRawOldBlockProjectionFinalScaleCompiler.rowCount layout) -> Row}
    {assignment : Nat -> Nat}
    {finalWitnesses : Fin productionGlobalParams.k -> Matrix productionShape}
    (audit : FactoredTerminalExecutionAudit context certificate layout
      artifactRow assignment finalWitnesses)
    (coordinate : Fin layout.base.logicalWidth) :
    toConcreteField
        (ProjectionProgram.residue
          (lcEval assignment (rawTerms layout.base coordinate))) =
      PiDEC.Raw.recomposeAssignment
        (fun child => unpack (finalWitnesses child))
        (factoredProductionCoordinate audit.artifact coordinate) := by
  let columns : Fin productionGlobalParams.k -> Nat := fun child =>
    rawWitnessColumn layout.base
      (Fin.cast (factoredChildCountEq audit.artifact).symm child) coordinate
  have termsExact : rawTerms layout.base coordinate =
      List.ofFn fun child : Fin productionGlobalParams.k =>
        (columns child,
          productionGlobalParams.b ^ child.val % goldilocksP) := by
    apply List.ext_get
    · simp [rawTerms, audit.artifact.profileChildren,
        productionGlobalParams]
    · intro index leftLt rightLt
      have leftBound : index < layout.base.childCount := by
        simpa [rawTerms] using leftLt
      have rightBound : index < productionGlobalParams.k := by
        simpa using rightLt
      have childExact :
          (⟨index, leftBound⟩ : Fin layout.base.childCount) =
            Fin.cast (factoredChildCountEq audit.artifact).symm
              (⟨index, rightBound⟩ : Fin productionGlobalParams.k) := by
        apply Fin.ext
        rfl
      simp only [rawTerms, List.get_eq_getElem, List.getElem_ofFn]
      rw [childExact]
      simp [columns, radixCoefficient, audit.artifact.profileRadix,
        productionGlobalParams]
  have columnsExact : forall child : Fin productionGlobalParams.k,
      assignment (columns child) =
        (unpack (finalWitnesses child)
          (factoredProductionCoordinate audit.artifact coordinate)).val := by
    intro child
    have mapped := audit.rawWitnessColumns
      (Fin.cast (factoredChildCountEq audit.artifact).symm child) coordinate
    have childExact : factoredProductionChild audit.artifact
        (Fin.cast (factoredChildCountEq audit.artifact).symm child) = child := by
      apply Fin.ext
      rfl
    rw [childExact] at mapped
    exact mapped
  rw [termsExact]
  exact RawScalarKernel.lcEval_radixTerms_eq_recomposeScalar
    assignment columns (fun child => unpack (finalWitnesses child))
    (factoredProductionCoordinate audit.artifact coordinate) columnsExact

private theorem coordinateChi_eq_semanticWeight
    {context : FixedActive.Context productionShape State publicRingColumns
      publicFits verifierRows}
    {certificate : FixedActive.Certificate context}
    {layout : Layout}
    {artifactRow : Fin (rowCount layout) -> Row}
    {assignment : Nat -> Nat}
    {finalWitnesses : Fin productionGlobalParams.k -> Matrix productionShape}
    (audit : TerminalExecutionAudit context certificate layout artifactRow
      assignment finalWitnesses)
    (coordinate : Fin layout.logicalWidth) :
    toConcreteK
        ((coordinateChiTerms layout coordinate).value assignment) =
      NumericBooleanDomain.testBitWeight ConcreteCarrier.extensionOps
        (DelayedProduction.outgoingPending context certificate).oldBlock
        (Fin.cast
          (congrArg (fun variables => 2 ^ variables)
            (blockVariablesEq audit.artifact))
          ⟨coordinateBlock layout coordinate, by
            have coordinateBound := coordinate.isLt
            have within : coordinateBlock layout coordinate <
                blockCount layout := by
              unfold coordinateBlock
              rw [audit.artifact.profileActiveLanes,
                audit.artifact.profileBlockCount]
              have exactBound : coordinate.val < 11437038 := by
                simpa [audit.artifact.profileLogicalWidth] using coordinateBound
              omega
            exact Nat.lt_of_lt_of_le within (by
              rw [audit.artifact.profileBlockCount,
                audit.artifact.profileBlockVariables]
              decide)⟩) := by
  have within : coordinateBlock layout coordinate < blockCount layout := by
    unfold coordinateBlock
    rw [audit.artifact.profileActiveLanes,
      audit.artifact.profileBlockCount]
    have exactBound : coordinate.val < 11437038 := by
      simpa [audit.artifact.profileLogicalWidth] using coordinate.isLt
    omega
  have blocksFit : blockCount layout <= 2 ^ layout.blockVariables := by
    rw [audit.artifact.profileBlockCount,
      audit.artifact.profileBlockVariables]
    decide
  have compilerWeight := coordinateChiTerms_value_eq_testBitWeight
    audit.artifact.shape audit.canonical audit.constantOne
    (audit.artifact.rowsSatisfied audit.rows) coordinate within blocksFit
  change (coordinateChiTerms layout coordinate).value assignment =
    NumericBooleanDomain.testBitWeight projectionTensorOps
      (oldBlockPoint layout assignment)
      ⟨coordinateBlock layout coordinate,
        Nat.lt_of_lt_of_le within blocksFit⟩ at compilerWeight
  apply Eq.trans (congrArg toConcreteK compilerWeight)
  rw [testBitWeight_map, mappedOldBlockPoint audit,
    testBitWeight_cast]

private theorem factoredCoordinateChi_eq_prefixWeight
    {context : FixedActive.Context productionShape State publicRingColumns
      publicFits verifierRows}
    {certificate : FixedActive.Certificate context}
    {layout : TerminalRawOldBlockProjectionFinalScaleCompiler.Layout}
    {artifactRow : Fin
      (TerminalRawOldBlockProjectionFinalScaleCompiler.rowCount layout) -> Row}
    {assignment : Nat -> Nat}
    {finalWitnesses : Fin productionGlobalParams.k -> Matrix productionShape}
    (audit : FactoredTerminalExecutionAudit context certificate layout
      artifactRow assignment finalWitnesses)
    (coordinate : Fin layout.base.logicalWidth) :
    toConcreteK
        ((coordinateChiTerms layout.base coordinate).value assignment) =
      NumericBooleanDomain.testBitWeight ConcreteCarrier.extensionOps
        (mapPoint (oldBlockPoint layout.base assignment))
        ⟨coordinateBlock layout.base coordinate, by
          have within : coordinateBlock layout.base coordinate <
              blockCount layout.base := by
            unfold coordinateBlock
            rw [audit.artifact.profileActiveLanes,
              audit.artifact.profileBlockCount]
            have exactBound : coordinate.val < 11437038 := by
              simpa [audit.artifact.profileLogicalWidth] using coordinate.isLt
            omega
          exact Nat.lt_of_lt_of_le within
            audit.artifact.shape.blocksFitPrefix⟩ := by
  have within : coordinateBlock layout.base coordinate <
      blockCount layout.base := by
    unfold coordinateBlock
    rw [audit.artifact.profileActiveLanes,
      audit.artifact.profileBlockCount]
    have exactBound : coordinate.val < 11437038 := by
      simpa [audit.artifact.profileLogicalWidth] using coordinate.isLt
    omega
  let blockFin : Fin (2 ^ layout.base.blockVariables) :=
    ⟨coordinateBlock layout.base coordinate,
      Nat.lt_of_lt_of_le within audit.artifact.shape.blocksFitPrefix⟩
  have compilerWeight :
      (coordinateChiTerms layout.base coordinate).value assignment =
        NumericBooleanDomain.testBitWeight projectionTensorOps
          (oldBlockPoint layout.base assignment) blockFin := by
    calc
      (coordinateChiTerms layout.base coordinate).value assignment =
          (tensorValues layout.base assignment).getD
            (coordinateBlock layout.base coordinate) ProjectionProgram.K.one :=
        TerminalRawOldBlockProjectionFinalScaleCompiler.coordinateChiTerms_value_eq_tensorValue
          audit.artifact.shape audit.canonical audit.constantOne
          (audit.artifact.rowsSatisfied audit.rows) coordinate
      _ = NumericBooleanDomain.testBitWeight projectionTensorOps
          (oldBlockPoint layout.base assignment) blockFin := by
        rw [tensorValues_eq_expectedPrefix layout.base assignment
          (Nat.lt_of_le_of_lt (Nat.zero_le _) within)
          audit.artifact.shape.baseShape.levelCount]
        exact expectedPrefixGetD_eq_testBitWeight layout.base assignment blockFin
          (by simpa [blockFin] using within)
          audit.artifact.shape.blocksFitPrefix
  apply Eq.trans (congrArg toConcreteK compilerWeight)
  exact testBitWeight_map (oldBlockPoint layout.base assignment) blockFin

private theorem factoredScaledChi_eq_semanticWeight
    {context : FixedActive.Context productionShape State publicRingColumns
      publicFits verifierRows}
    {certificate : FixedActive.Certificate context}
    {layout : TerminalRawOldBlockProjectionFinalScaleCompiler.Layout}
    {artifactRow : Fin
      (TerminalRawOldBlockProjectionFinalScaleCompiler.rowCount layout) -> Row}
    {assignment : Nat -> Nat}
    {finalWitnesses : Fin productionGlobalParams.k -> Matrix productionShape}
    (audit : FactoredTerminalExecutionAudit context certificate layout
      artifactRow assignment finalWitnesses)
    (coordinate : Fin layout.base.logicalWidth) :
    toConcreteK
        (ProjectionProgram.K.mul
          ((coordinateChiTerms layout.base coordinate).value assignment)
          (TerminalRawOldBlockProjectionCompiler.K.sub
            ProjectionProgram.K.one
            (layout.factor.finalPoint.value assignment))) =
      NumericBooleanDomain.testBitWeight ConcreteCarrier.extensionOps
        (DelayedProduction.outgoingPending context certificate).oldBlock
        (Fin.cast
          (congrArg (fun variables => 2 ^ variables)
            (factoredBlockVariablesEq audit.artifact))
          ⟨coordinateBlock layout.base coordinate, by
            have within : coordinateBlock layout.base coordinate <
                blockCount layout.base := by
              unfold coordinateBlock
              rw [audit.artifact.profileActiveLanes,
                audit.artifact.profileBlockCount]
              have exactBound : coordinate.val < 11437038 := by
                simpa [audit.artifact.profileLogicalWidth] using coordinate.isLt
              omega
            have low := Nat.lt_of_lt_of_le within
              audit.artifact.shape.blocksFitPrefix
            change coordinateBlock layout.base coordinate <
              2 ^ (layout.base.blockVariables + 1)
            rw [Nat.pow_succ]
            omega⟩) := by
  have within : coordinateBlock layout.base coordinate <
      blockCount layout.base := by
    unfold coordinateBlock
    rw [audit.artifact.profileActiveLanes,
      audit.artifact.profileBlockCount]
    have exactBound : coordinate.val < 11437038 := by
      simpa [audit.artifact.profileLogicalWidth] using coordinate.isLt
    omega
  have low : coordinateBlock layout.base coordinate <
      2 ^ layout.base.blockVariables :=
    Nat.lt_of_lt_of_le within audit.artifact.shape.blocksFitPrefix
  let fullIndex : Fin (2 ^ (layout.base.blockVariables + 1)) :=
    ⟨coordinateBlock layout.base coordinate, by
      rw [Nat.pow_succ]
      omega⟩
  have appended := testBitWeight_appendPoint_low
    (oldBlockPoint layout.base assignment)
    (layout.factor.finalPoint.value assignment) fullIndex low
  have mapped := congrArg toConcreteK appended
  rw [toConcreteK_mul, toConcreteK_sub, toConcreteK_one,
    testBitWeight_map, testBitWeight_map,
    mappedFactoredOldBlockPoint audit, testBitWeight_cast] at mapped
  rw [toConcreteK_mul,
    factoredCoordinateChi_eq_prefixWeight audit coordinate,
    toConcreteK_sub, toConcreteK_one]
  exact mapped.symm

private theorem toConcreteK_foldrAdd
    {Index : Type} (indices : List Index)
    (value : Index -> ProjectionProgram.K) :
    toConcreteK
        (indices.foldr
          (fun index suffix => ProjectionProgram.K.add (value index) suffix)
          ProjectionProgram.K.zero) =
      indices.foldr
        (fun index suffix => Concrete.K.add (toConcreteK (value index)) suffix)
        Concrete.K.zero := by
  induction indices with
  | nil => exact toConcreteK_zero
  | cons index tail inductionHypothesis =>
      simp only [List.foldr_cons, toConcreteK_add, inductionHypothesis]

private theorem foldrAdd_mul
    {Index : Type} (indices : List Index)
    (value : Index -> ProjectionProgram.K) (factor : ProjectionProgram.K) :
    ProjectionProgram.K.mul
        (indices.foldr
          (fun index suffix => ProjectionProgram.K.add (value index) suffix)
          ProjectionProgram.K.zero)
        factor =
      indices.foldr
        (fun index suffix => ProjectionProgram.K.add
          (ProjectionProgram.K.mul (value index) factor) suffix)
        ProjectionProgram.K.zero := by
  induction indices with
  | nil => simp
  | cons index tail inductionHypothesis =>
      simp only [List.foldr_cons]
      rw [ProjectionProgram.K.add_mul, inductionHypothesis]

private theorem foldrAdd_eq_foldlAdd
    {Index : Type} (indices : List Index) (value : Index -> Concrete.K) :
    indices.foldr (fun index suffix => Concrete.K.add (value index) suffix)
        Concrete.K.zero =
      indices.foldl (fun accumulated index =>
        Concrete.K.add accumulated (value index)) Concrete.K.zero := by
  have withInitial : forall initial,
      indices.foldl (fun accumulated index =>
          Concrete.K.add accumulated (value index)) initial =
        Concrete.K.add initial
          (indices.foldr
            (fun index suffix => Concrete.K.add (value index) suffix)
            Concrete.K.zero) := by
    intro initial
    induction indices generalizing initial with
    | nil => exact (ConcreteCarrier.extensionLaws.add_zero initial).symm
    | cons index tail inductionHypothesis =>
        rw [List.foldl_cons, inductionHypothesis]
        simp only [List.foldr_cons]
        exact ConcreteCarrier.extensionLaws.add_assoc initial (value index)
          (tail.foldr
            (fun current suffix => Concrete.K.add (value current) suffix)
            Concrete.K.zero)
  symm
  calc
    indices.foldl (fun accumulated index =>
        Concrete.K.add accumulated (value index)) Concrete.K.zero =
      Concrete.K.add Concrete.K.zero
        (indices.foldr
          (fun index suffix => Concrete.K.add (value index) suffix)
          Concrete.K.zero) := withInitial Concrete.K.zero
    _ = indices.foldr
        (fun index suffix => Concrete.K.add (value index) suffix)
        Concrete.K.zero := ConcreteCarrier.extensionLaws.zero_add _

private theorem rows_projection
    {context : FixedActive.Context productionShape State publicRingColumns
      publicFits verifierRows}
    {certificate : FixedActive.Certificate context}
    {layout : Layout}
    {artifactRow : Fin (rowCount layout) -> Row}
    {assignment : Nat -> Nat}
    {finalWitnesses : Fin productionGlobalParams.k -> Matrix productionShape}
    (audit : TerminalExecutionAudit context certificate layout artifactRow
      assignment finalWitnesses) :
    (DelayedProduction.outgoingPending context certificate).parentYZcol =
      PackedBlockAction.packedYZcol productionCovers
        (PiDEC.Raw.recomposeAssignment fun child =>
          unpack (finalWitnesses child))
        (DelayedProduction.outgoingPending context certificate).oldBlock := by
  funext semanticLane
  let compilerLane : Fin layout.activeLanes :=
    Fin.cast (activeLaneCountEq audit.artifact).symm semanticLane
  have parentRow := artifact_rows_sound audit.artifact audit.canonical
    audit.constantOne audit.rows compilerLane
  have parentDecoded := audit.parentColumns compilerLane
  have parentLane : productionLane audit.artifact compilerLane = semanticLane := by
    apply Fin.ext
    rfl
  rw [parentLane] at parentDecoded
  rw [← parentDecoded]
  apply Eq.trans (congrArg toConcreteK parentRow)
  unfold decodedRawProjection
  rw [laneCoordinates_eq_mapRange audit.artifact compilerLane,
    List.foldr_map]
  let recomposed := PiDEC.Raw.recomposeAssignment fun child =>
    unpack (finalWitnesses child)
  let packed : Matrix productionShape := pack recomposed
  have termEq : forall block : Fin (blockCount layout),
      toConcreteK
          (ProjectionProgram.K.mul
            (ProjectionProgram.K.ofBase
              (ProjectionProgram.residue
                (lcEval assignment
                  (rawTerms layout
                    (laneCoordinate audit.artifact compilerLane block)))))
            ((coordinateChiTerms layout
              (laneCoordinate audit.artifact compilerLane block)).value
                assignment)) =
        PackedWitnessOldBlockProjection.nativeProjectionTerm packed
          (DelayedProduction.outgoingPending context certificate).oldBlock
          semanticLane (Fin.cast (blockCountEq audit.artifact) block) := by
    intro block
    rw [toConcreteK_mul, toConcreteK_ofBase,
      rawScalar_eq_recomposed audit,
      coordinateChi_eq_semanticWeight audit,
      semanticLaneCoordinate audit.artifact compilerLane block]
    simp only [packed, PackedWitnessOldBlockProjection.nativeProjectionTerm,
      pack, semanticBlockOfRust_rustBlockOfSemantic]
    congr 2
    apply Fin.ext
    change
      (block.val * layout.activeLanes + compilerLane.val) /
          layout.activeLanes = block.val
    rw [Nat.mul_comm block.val layout.activeLanes,
      Nat.mul_add_div (Nat.zero_lt_of_lt compilerLane.isLt),
      Nat.div_eq_of_lt compilerLane.isLt, Nat.add_zero]
  rw [toConcreteK_foldrAdd]
  calc
    (List.range (blockCount layout)).foldr
        (fun block suffix =>
          Concrete.K.add
            (toConcreteK
              (ProjectionProgram.K.mul
                (ProjectionProgram.K.ofBase
                  (ProjectionProgram.residue
                    (lcEval assignment
                      (rawTerms layout
                        (laneCoordinateNat audit.artifact compilerLane
                          block)))))
                ((coordinateChiTerms layout
                  (laneCoordinateNat audit.artifact compilerLane
                    block)).value
                      assignment)))
            suffix)
        Concrete.K.zero =
      (canonicalFinIndices (blockCount layout)).foldl
        (fun accumulated block =>
          Concrete.K.add accumulated
            (PackedWitnessOldBlockProjection.nativeProjectionTerm packed
              (DelayedProduction.outgoingPending context certificate).oldBlock
              semanticLane (Fin.cast (blockCountEq audit.artifact) block)))
        Concrete.K.zero := by
      rw [foldrAdd_eq_foldlAdd]
      have rangeEq := canonicalFinIndices_values (blockCount layout)
      rw [← rangeEq, List.foldl_map]
      apply congrArg (fun operation =>
        (canonicalFinIndices (blockCount layout)).foldl operation
          Concrete.K.zero)
      funext accumulated block
      simp only [laneCoordinateNat, dif_pos block.isLt]
      rw [termEq]
    _ = PackedWitnessOldBlockProjection.nativeProjectedLane packed
        (DelayedProduction.outgoingPending context certificate).oldBlock
        semanticLane := by
      unfold PackedWitnessOldBlockProjection.nativeProjectedLane
      have indicesExact :
          (canonicalFinIndices (blockCount layout)).map
              (Fin.cast (blockCountEq audit.artifact)) =
            canonicalFinIndices
              (Phi81ColumnLayout.blockCount productionShape.carrierWidth) := by
        apply List.ext_get
        · simp [canonicalFinIndices, blockCountEq audit.artifact]
        · intro index leftLt rightLt
          apply Fin.ext
          simp [canonicalFinIndices]
      rw [← indicesExact, List.foldl_map]
    _ = PackedBlockAction.packedYZcol productionCovers
        (unpack packed)
        (DelayedProduction.outgoingPending context certificate).oldBlock
        semanticLane :=
      PackedWitnessOldBlockProjection.nativeProjectedLane_eq_packedYZcol
        packed
        (DelayedProduction.outgoingPending context certificate).oldBlock
        semanticLane
    _ = PackedBlockAction.packedYZcol productionCovers recomposed
        (DelayedProduction.outgoingPending context certificate).oldBlock
        semanticLane := by simp [packed]

private theorem factoredRows_projection
    {context : FixedActive.Context productionShape State publicRingColumns
      publicFits verifierRows}
    {certificate : FixedActive.Certificate context}
    {layout : TerminalRawOldBlockProjectionFinalScaleCompiler.Layout}
    {artifactRow : Fin
      (TerminalRawOldBlockProjectionFinalScaleCompiler.rowCount layout) -> Row}
    {assignment : Nat -> Nat}
    {finalWitnesses : Fin productionGlobalParams.k -> Matrix productionShape}
    (audit : FactoredTerminalExecutionAudit context certificate layout
      artifactRow assignment finalWitnesses) :
    (DelayedProduction.outgoingPending context certificate).parentYZcol =
      PackedBlockAction.packedYZcol productionCovers
        (PiDEC.Raw.recomposeAssignment fun child =>
          unpack (finalWitnesses child))
        (DelayedProduction.outgoingPending context certificate).oldBlock := by
  funext semanticLane
  let compilerLane : Fin layout.base.activeLanes :=
    Fin.cast (factoredActiveLaneCountEq audit.artifact).symm semanticLane
  have parentRow :=
    TerminalRawOldBlockProjectionFinalScaleCompiler.artifact_rows_sound
      audit.artifact audit.canonical audit.constantOne audit.rows compilerLane
  have parentDecoded := audit.parentColumns compilerLane
  have parentLane :
      factoredProductionLane audit.artifact compilerLane = semanticLane := by
    apply Fin.ext
    rfl
  rw [parentLane] at parentDecoded
  rw [← parentDecoded]
  apply Eq.trans (congrArg toConcreteK parentRow)
  unfold decodedRawProjection
  rw [factoredLaneCoordinates_eq_mapRange audit.artifact compilerLane,
    List.foldr_map]
  let factor := TerminalRawOldBlockProjectionCompiler.K.sub
    ProjectionProgram.K.one (layout.factor.finalPoint.value assignment)
  rw [foldrAdd_mul]
  rw [toConcreteK_foldrAdd]
  let recomposed := PiDEC.Raw.recomposeAssignment fun child =>
    unpack (finalWitnesses child)
  let packed : Matrix productionShape := pack recomposed
  have termEq : forall block : Fin (blockCount layout.base),
      toConcreteK
          (ProjectionProgram.K.mul
            (ProjectionProgram.K.mul
              (ProjectionProgram.K.ofBase
                (ProjectionProgram.residue
                  (lcEval assignment
                    (rawTerms layout.base
                      (factoredLaneCoordinate audit.artifact compilerLane
                        block)))))
              ((coordinateChiTerms layout.base
                (factoredLaneCoordinate audit.artifact compilerLane block)).value
                  assignment))
            factor) =
        PackedWitnessOldBlockProjection.nativeProjectionTerm packed
          (DelayedProduction.outgoingPending context certificate).oldBlock
          semanticLane (Fin.cast (factoredBlockCountEq audit.artifact) block) := by
    intro block
    rw [ProjectionProgram.K.mul_assoc, toConcreteK_mul, toConcreteK_ofBase,
      factoredRawScalar_eq_recomposed audit,
      factoredScaledChi_eq_semanticWeight audit,
      factoredSemanticLaneCoordinate audit.artifact compilerLane block]
    simp only [factor, packed,
      PackedWitnessOldBlockProjection.nativeProjectionTerm, pack,
      semanticBlockOfRust_rustBlockOfSemantic]
    congr 2
    apply Fin.ext
    change
      (block.val * layout.base.activeLanes + compilerLane.val) /
          layout.base.activeLanes = block.val
    rw [Nat.mul_comm block.val layout.base.activeLanes,
      Nat.mul_add_div (Nat.zero_lt_of_lt compilerLane.isLt),
      Nat.div_eq_of_lt compilerLane.isLt, Nat.add_zero]
  calc
    (List.range (blockCount layout.base)).foldr
        (fun block suffix =>
          Concrete.K.add
            (toConcreteK
              (ProjectionProgram.K.mul
                (ProjectionProgram.K.mul
                  (ProjectionProgram.K.ofBase
                    (ProjectionProgram.residue
                      (lcEval assignment
                        (rawTerms layout.base
                          (factoredLaneCoordinateNat audit.artifact compilerLane
                            block)))))
                  ((coordinateChiTerms layout.base
                    (factoredLaneCoordinateNat audit.artifact compilerLane
                      block)).value assignment))
                factor))
            suffix)
        Concrete.K.zero =
      (canonicalFinIndices (blockCount layout.base)).foldl
        (fun accumulated block =>
          Concrete.K.add accumulated
            (PackedWitnessOldBlockProjection.nativeProjectionTerm packed
              (DelayedProduction.outgoingPending context certificate).oldBlock
              semanticLane
              (Fin.cast (factoredBlockCountEq audit.artifact) block)))
        Concrete.K.zero := by
      rw [foldrAdd_eq_foldlAdd]
      have rangeEq := canonicalFinIndices_values (blockCount layout.base)
      rw [← rangeEq, List.foldl_map]
      apply congrArg (fun operation =>
        (canonicalFinIndices (blockCount layout.base)).foldl operation
          Concrete.K.zero)
      funext accumulated block
      simp only [factoredLaneCoordinateNat, dif_pos block.isLt]
      rw [termEq]
    _ = PackedWitnessOldBlockProjection.nativeProjectedLane packed
        (DelayedProduction.outgoingPending context certificate).oldBlock
        semanticLane := by
      unfold PackedWitnessOldBlockProjection.nativeProjectedLane
      have indicesExact :
          (canonicalFinIndices (blockCount layout.base)).map
              (Fin.cast (factoredBlockCountEq audit.artifact)) =
            canonicalFinIndices
              (Phi81ColumnLayout.blockCount productionShape.carrierWidth) := by
        apply List.ext_get
        · simp [canonicalFinIndices, factoredBlockCountEq audit.artifact]
        · intro index leftLt rightLt
          apply Fin.ext
          simp [canonicalFinIndices]
      rw [← indicesExact, List.foldl_map]
    _ = PackedBlockAction.packedYZcol productionCovers
        (unpack packed)
        (DelayedProduction.outgoingPending context certificate).oldBlock
        semanticLane :=
      PackedWitnessOldBlockProjection.nativeProjectedLane_eq_packedYZcol
        packed
        (DelayedProduction.outgoingPending context certificate).oldBlock
        semanticLane
    _ = PackedBlockAction.packedYZcol productionCovers recomposed
        (DelayedProduction.outgoingPending context certificate).oldBlock
        semanticLane := by simp [packed]

/-- The exact indexed projection rows plus the selected terminal-CE boundary
derive the existing minimal terminal authority object over definitionally the
same raw `WitnessMat` family. -/
theorem TerminalExecutionAudit.projectionOpeningAccepted
    {context : FixedActive.Context productionShape State publicRingColumns
      publicFits verifierRows}
    {certificate : FixedActive.Certificate context}
    {layout : Layout}
    {artifactRow : Fin (rowCount layout) -> Row}
    {assignment : Nat -> Nat}
    {finalWitnesses : Fin productionGlobalParams.k -> Matrix productionShape}
    (audit : TerminalExecutionAudit context certificate layout artifactRow
      assignment finalWitnesses) :
    ProductionTerminal.ProjectionOpeningAccepted context certificate
      (fun child => unpack (finalWitnesses child)) := by
  have projection : ProductionTerminal.projectionCheck context certificate
      (fun child => unpack (finalWitnesses child)) = true :=
    (ProductionTerminal.projectionCheck_eq_true_iff context certificate
      (fun child => unpack (finalWitnesses child))).2 (rows_projection audit)
  exact (ProductionTerminal.TerminalCEBridge.accepted_of_terminalCE_and_projectionCheck
    context certificate (fun child => unpack (finalWitnesses child))
    audit.terminalCE projection).projectionOpeningAccepted

/-- The optimized four-family artifact derives the same terminal authority as
the direct emitter.  The common bit-18 low factor is established from the
compact-prefix rows and its verifier-owned nineteenth old-block column; no
implementation/refinement failure remains in the conclusion. -/
theorem FactoredTerminalExecutionAudit.projectionOpeningAccepted
    {context : FixedActive.Context productionShape State publicRingColumns
      publicFits verifierRows}
    {certificate : FixedActive.Certificate context}
    {layout : TerminalRawOldBlockProjectionFinalScaleCompiler.Layout}
    {artifactRow : Fin
      (TerminalRawOldBlockProjectionFinalScaleCompiler.rowCount layout) -> Row}
    {assignment : Nat -> Nat}
    {finalWitnesses : Fin productionGlobalParams.k -> Matrix productionShape}
    (audit : FactoredTerminalExecutionAudit context certificate layout
      artifactRow assignment finalWitnesses) :
    ProductionTerminal.ProjectionOpeningAccepted context certificate
      (fun child => unpack (finalWitnesses child)) := by
  have projection : ProductionTerminal.projectionCheck context certificate
      (fun child => unpack (finalWitnesses child)) = true :=
    (ProductionTerminal.projectionCheck_eq_true_iff context certificate
      (fun child => unpack (finalWitnesses child))).2
        (factoredRows_projection audit)
  exact (ProductionTerminal.TerminalCEBridge.accepted_of_terminalCE_and_projectionCheck
    context certificate (fun child => unpack (finalWitnesses child))
    audit.terminalCE projection).projectionOpeningAccepted

end Context

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.ActiveTerminalRawProjectionRows
