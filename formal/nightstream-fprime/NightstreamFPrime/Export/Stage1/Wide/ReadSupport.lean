import NightstreamFPrime.Export.Stage1.Wide.PlanSupport
import NightstreamFPrime.Export.Stage1.Wide.InputSupport
import NightstreamFPrime.Export.Stage1.Wide.CoordinateRecovery

/-! The supported source regions of the reused Stage 1 phases. Every result
concerns stored sparse entries, before cancellation or matrix evaluation. -/

namespace NightstreamFPrime.Export.Stage1.Wide.ReadSupport

open NightstreamFPrime.Spec NightstreamFPrime.Layout
open ProductionRelation FormSupport
open Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Lifecycle.PaperAlgebra

abbrev Program := RetainedLayout.Program
def Live (program : Program) (column : Fin (PerApplicationFixedPoint.logicalWidth program)) : Prop :=
  RetainedLayout.Live program column.val
abbrev Form (program : Program) := Supported (Live program)
abbrev Plans (program : Program) := PlanSupported (Live program)
abbrev CommonForm (program : Program) := Supported (Common program)
abbrev CommonPlans (program : Program) := PlanSupported (Common program)
abbrev Copied (program : Program) (column : Fin (PerApplicationFixedPoint.logicalWidth program)) :=
  CoordinateRecovery.CommonSource program column.val
abbrev CopiedForm (program : Program) := Supported (Copied program)
abbrev CopiedPlans (program : Program) := PlanSupported (Copied program)

theorem common_copied (program : Program) (form : SparseForm (PerApplicationFixedPoint.logicalWidth program))
    (supported : CommonForm program form) : CopiedForm program form := by
  refine mono supported (fun column inside => ?_)
  rcases inside with hash | shared
  · exact Or.inl hash
  · exact Or.inr (Or.inl shared)

theorem copied_plan (program : Program)
    (plan : ProductionRelation.Plan (PerApplicationFixedPoint.logicalWidth program))
    (supported : CopiedPlans program plan) : Plans program plan := by
  intro row port
  exact mono (supported row port) (fun column live => CoordinateRecovery.commonSource_live program column.val live)

theorem common (program : Program) (column : Fin (PerApplicationFixedPoint.logicalWidth program))
    (inside : FormSupport.Common program column) : Live program column := by
  rcases inside with hash | shared
  · exact Or.inl hash
  · exact Or.inr (Or.inl shared)

theorem common_form (program : Program) (form : SparseForm (PerApplicationFixedPoint.logicalWidth program))
    (supported : Supported (Common program) form) : Form program form :=
  mono supported (common program)

theorem common_plan (program : Program)
    (plan : ProductionRelation.Plan (PerApplicationFixedPoint.logicalWidth program))
    (supported : CommonPlans program plan) : Plans program plan := by
  intro row port
  exact common_form program _ (supported row port)

theorem one_common (program : Program) (column : Fin (PerApplicationFixedPoint.logicalWidth program))
    (zero : column.val = 0) : Common program column := by
  left
  rw [zero, (RetainedLayout.boundaries program).1]
  decide

theorem one (program : Program) (column : Fin (PerApplicationFixedPoint.logicalWidth program))
    (zero : column.val = 0) : Live program column := by
  left
  rw [zero, (RetainedLayout.boundaries program).1]
  decide

theorem shared_block_common (program : Program) {sourceWidth : Nat} (retained : LowNormBlock.Block sourceWidth)
    (start : Nat) (fits : start + retained.coordinateCount ≤ PerApplicationFixedPoint.logicalWidth program)
    (slot : Fin retained.slotCount) (lower : RetainedLayout.sharedStart program ≤ start)
    (upper : start + retained.coordinateCount ≤ RetainedLayout.sharedEnd program) :
    CommonForm program (retained.form start fits slot) := by
  apply block
  intro column low high
  exact Or.inr ⟨by omega, by omega⟩

theorem shared_block (program : Program) {sourceWidth : Nat} (retained : LowNormBlock.Block sourceWidth)
    (start : Nat) (fits : start + retained.coordinateCount ≤ PerApplicationFixedPoint.logicalWidth program)
    (slot : Fin retained.slotCount) (lower : RetainedLayout.sharedStart program ≤ start)
    (upper : start + retained.coordinateCount ≤ RetainedLayout.sharedEnd program) :
    Form program (retained.form start fits slot) :=
  common_form program _ (shared_block_common program retained start fits slot lower upper)

theorem piCcs_endpoint_value (program : Program)
    (geometry : PiCCSOrdinaryRetainedGeometry.Geometry program (PerApplicationFixedPoint.logicalWidth program))
    (slot : Fin 8) : CommonForm program ((PiCCSOrdinaryRetainedBlocks.outputEndpointBlock program).form
      (PiCCSOrdinaryRetainedGeometry.outputEndpointStart program)
      (PiCCSOrdinaryRetainedGeometry.outputEndpointFits geometry) slot) := by
  apply shared_block_common
  · have bound : RetainedLayout.sharedStart program ≤ PiCCSOrdinaryRetainedGeometry.proofLogicalStart program := by
      rw [(RetainedLayout.boundaries program).2.1]
      unfold PiCCSOrdinaryRetainedGeometry.proofLogicalStart PiCCSOrdinaryRetainedGeometry.expectedContextStart
        PiCCSOrdinaryRetainedGeometry.outputLastStart PiCCSOrdinaryRetainedGeometry.priorLastStart
        PiCCSOrdinaryRetainedGeometry.freshPublicInputStart PiCCSOrdinaryRetainedGeometry.prefixLogicalWidth
      rw [RunningTransitionReducedRetainedBlocks.nextStart_eq]
      omega
    unfold PiCCSOrdinaryRetainedGeometry.outputEndpointStart
    omega
  · let within : PiCCSOrdinaryRetainedGeometry.Geometry program (RetainedLayout.sharedEnd program) :=
      ⟨by rw [PiCCSOrdinaryRetainedGeometry.completeLogicalWidth_eq,
        (RetainedLayout.boundaries program).2.2.1]; decide⟩
    exact PiCCSOrdinaryRetainedGeometry.outputEndpointFits within

theorem piCcs_endpoint (program : Program)
    (geometry : PiCCSOrdinaryRetainedGeometry.Geometry program (PerApplicationFixedPoint.logicalWidth program))
    (column : Fin Layout.Stage1.Spartan.spartanColumnCount) :
    CommonForm program (PiCCSOrdinaryDirectPlan.endpointForm geometry column) := by
  unfold PiCCSOrdinaryDirectPlan.endpointForm
  split
  · exact empty _
  · split
    · exact piCcs_endpoint_value program geometry _
    · exact empty _

theorem piCcs_source (program : Program)
    (geometry : PiCCSOrdinaryRetainedGeometry.Geometry program (PerApplicationFixedPoint.logicalWidth program))
    (column : Fin Layout.Stage1.Spartan.spartanColumnCount) :
    CommonForm program ((PiCCSOrdinaryDirectPlan.sourceMap geometry).form column) := by
  dsimp only [PiCCSOrdinaryDirectPlan.sourceMap]
  split
  · exact piCcs_endpoint program geometry column
  · exact InputSupport.location program geometry _

theorem piCcs_ordinary (program : Program)
    {width : Nat} {publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth width}
    (relation : Lifecycle.ProductionKey.LogicalRelation width publicFits)
    (geometry : PiCCSOrdinaryRetainedGeometry.Geometry program (PerApplicationFixedPoint.logicalWidth program)) :
    CommonPlans program (PiCCSOrdinaryDirectPlan.plan relation geometry) := by
  apply ordinary_plan
  intro row
  exact source_row _ _ (piCcs_source program geometry) (one_common program _ rfl) _ _

theorem nextPreimage (program : Program)
    (geometry : PiCCSOrdinaryRetainedGeometry.Geometry program (PerApplicationFixedPoint.logicalWidth program)) :
    CommonPlans program (NextPreimageDirectPlan.plan geometry) := by
  apply source_plan
  · exact one_common program _ rfl
  · intro row
    exact piCcs_source program geometry

end NightstreamFPrime.Export.Stage1.Wide.ReadSupport
