import NightstreamFPrime.Export.Stage1.Wide.ReadSupport

namespace NightstreamFPrime.Export.Stage1.Wide.ReadSupport

open NightstreamFPrime.Spec NightstreamFPrime.Layout
open ProductionRelation FormSupport
open Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Lifecycle.PaperAlgebra

theorem product_block (program : Program) {sourceWidth : Nat} (retained : LowNormBlock.Block sourceWidth)
    (start : Nat) (fits : start + retained.coordinateCount ≤ PerApplicationFixedPoint.logicalWidth program)
    (slot : Fin retained.slotCount) (lower : PiRLCRetainedGeometry.productOutputStart program ≤ start)
    (upper : start + retained.coordinateCount ≤ PiRLCRetainedGeometry.productOutputStart program + 52326 * 41) :
    Form program (retained.form start fits slot) := by
  have total := PiRLCRetainedGeometry.prefixLogicalWidth_eq program
  change PiRLCRetainedGeometry.productOutputStart program +
    (PiRLCProductSourceBlocks.outputBlock program).coordinateCount = 121293360 at total
  rw [PiRLCProductSourceBlocks.outputBlock_coordinateCount] at total
  apply block
  intro column low high
  exact Or.inr (Or.inr (Or.inr (Or.inl ⟨by omega, by omega⟩)))

theorem piDec_location (program : Program)
    (geometry : PiDECRetainedGeometry.Geometry program (PerApplicationFixedPoint.logicalWidth program))
    (value : PiDECDirectPlan.Location) : Form program (value.form geometry) := by
  let within : PiDECRetainedGeometry.Geometry program (RetainedLayout.sharedEnd program) := ⟨Nat.le_refl _⟩
  cases value <;> dsimp only [PiDECDirectPlan.Location.form]
  case parentCommitment index =>
    apply product_block
    · change PiRLCRetainedGeometry.productOutputStart program ≤ PiRLCRetainedGeometry.productOutputStart program + 19008 * 41
      omega
    · change PiRLCRetainedGeometry.productOutputStart program + 19008 * 41 + 1188 * 41 ≤ _
      omega
  case parentPublicInput index =>
    apply product_block
    · change PiRLCRetainedGeometry.productOutputStart program ≤ PiRLCRetainedGeometry.productOutputStart program + 24516 * 41
      omega
    · change PiRLCRetainedGeometry.productOutputStart program + 24516 * 41 + 270 * 41 ≤ _
      omega
  case parentEvalK index =>
    apply product_block
    · change PiRLCRetainedGeometry.productOutputStart program ≤ PiRLCRetainedGeometry.productOutputStart program + 26514 * 41
      omega
    · change PiRLCRetainedGeometry.productOutputStart program + 26514 * 41 + 108 * 41 ≤ _
      omega
  case parentEvalA index =>
    apply product_block
    · change PiRLCRetainedGeometry.productOutputStart program ≤ PiRLCRetainedGeometry.productOutputStart program + 50814 * 41
      omega
    · change PiRLCRetainedGeometry.productOutputStart program + 50814 * 41 + 1512 * 41 ≤ _
      omega
  case proof index =>
    apply shared_block
    · change RetainedLayout.sharedStart program ≤ RetainedLayout.sharedStart program + _ + _
      omega
    · exact PiDECRetainedGeometry.proofFits within
  case logical index =>
    apply shared_block
    · rw [(RetainedLayout.boundaries program).2.1]
      unfold PiDECRetainedGeometry.logicalStart PiDECRetainedGeometry.prefixLogicalWidth
      rw [PilotOrdinaryRetainedGeometry.completeLogicalWidth_eq]
      decide
    · exact PiDECRetainedGeometry.logicalFits within
  case fresh index =>
    apply shared_block
    · rw [(RetainedLayout.boundaries program).2.1]
      unfold PiDECRetainedGeometry.freshStart PiDECRetainedGeometry.logicalStart PiDECRetainedGeometry.prefixLogicalWidth
      rw [PilotOrdinaryRetainedGeometry.completeLogicalWidth_eq]
      omega
    · exact PiDECRetainedGeometry.freshFits within

theorem piDec_source (program : Program)
    (geometry : PiDECRetainedGeometry.Geometry program (PerApplicationFixedPoint.logicalWidth program))
    (column : Fin Layout.Stage1.Spartan.spartanColumnCount) :
    Form program ((PiDECDirectPlan.sourceMap geometry).form column) := by
  dsimp only [PiDECDirectPlan.sourceMap]
  split
  · exact empty _
  · exact piDec_location program geometry _

theorem piDec (program : Program)
    {width : Nat} {publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth width}
    (relation : Lifecycle.ProductionKey.LogicalRelation width publicFits)
    (geometry : PiDECRetainedGeometry.Geometry program (PerApplicationFixedPoint.logicalWidth program)) :
    Plans program (PiDECDirectPlan.plan relation geometry) := by
  unfold PiDECDirectPlan.plan PiDECDirectPlan.recompositionPlan PiDECDirectPlan.evaluationPlan
  apply append
  · apply source_plan
    · exact one program _ rfl
    · intro row; exact piDec_source program geometry
  · apply append
    · apply source_plan
      · exact one program _ rfl
      · intro row; exact piDec_source program geometry
    · apply append <;> apply source_plan
      · exact one program _ rfl
      · intro row; exact piDec_source program geometry
      · exact one program _ rfl
      · intro row; exact piDec_source program geometry

end NightstreamFPrime.Export.Stage1.Wide.ReadSupport
