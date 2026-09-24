import NightstreamFPrime.Export.Stage1.Wide.PilotPoseidonSupport

namespace NightstreamFPrime.Export.Stage1.Wide.ReadSupport

open NightstreamFPrime.Spec NightstreamFPrime.Layout
open ProductionRelation FormSupport

theorem pilot_location (program : Program)
    (geometry : PilotOrdinaryRetainedGeometry.Geometry program (PerApplicationFixedPoint.logicalWidth program))
    (value : PilotOrdinaryDirectPlan.Location) : CommonForm program (value.form geometry) := by
  let within : PilotOrdinaryRetainedGeometry.Geometry program (RetainedLayout.sharedEnd program) :=
    ⟨by rw [PilotOrdinaryRetainedGeometry.completeLogicalWidth_eq, (RetainedLayout.boundaries program).2.2.1]; decide⟩
  have startLower : RetainedLayout.sharedStart program ≤ PiCCSOrdinaryRetainedGeometry.freshPublicInputStart program := by
    rw [(RetainedLayout.boundaries program).2.1]
    unfold PiCCSOrdinaryRetainedGeometry.freshPublicInputStart PiCCSOrdinaryRetainedGeometry.prefixLogicalWidth
    rw [RunningTransitionReducedRetainedBlocks.nextStart_eq]
    decide
  have localLower : RetainedLayout.sharedStart program ≤ PilotOrdinaryRetainedGeometry.canonicalLocalStart program := by
    rw [(RetainedLayout.boundaries program).2.1]
    unfold PilotOrdinaryRetainedGeometry.canonicalLocalStart PilotOrdinaryRetainedGeometry.prefixLogicalWidth
    rw [PiCCSOrdinaryRetainedGeometry.completeLogicalWidth_eq]
    decide
  cases value <;> dsimp only [PilotOrdinaryDirectPlan.Location.form]
  case priorDigest lane =>
    apply shared_block_common
    · unfold PiCCSOrdinaryRetainedGeometry.priorLastStart; omega
    · exact PiCCSOrdinaryRetainedGeometry.priorLastFits (PilotOrdinaryRetainedGeometry.prefixGeometry within)
  case priorPublic index =>
    exact shared_block_common program _ _ _ _ startLower
      (PiCCSOrdinaryRetainedGeometry.freshPublicInputFits (PilotOrdinaryRetainedGeometry.prefixGeometry within))
  case outputState lane =>
    apply shared_block_common
    · unfold PiCCSOrdinaryRetainedGeometry.outputLastStart PiCCSOrdinaryRetainedGeometry.priorLastStart; omega
    · exact PiCCSOrdinaryRetainedGeometry.outputLastFits (PilotOrdinaryRetainedGeometry.prefixGeometry within)
  case canonicalLocal index =>
    exact shared_block_common program _ _ _ _ localLower (PilotOrdinaryRetainedGeometry.canonicalLocalFits within)
  case canonicalFresh index =>
    apply shared_block_common
    · unfold PilotOrdinaryRetainedGeometry.canonicalFreshStart; omega
    · exact PilotOrdinaryRetainedGeometry.canonicalFreshFits within
  case outputDigest lane =>
    apply shared_block_common
    · unfold PilotOrdinaryRetainedGeometry.outputDigestStart PilotOrdinaryRetainedGeometry.canonicalFreshStart; omega
    · exact PilotOrdinaryRetainedGeometry.outputDigestFits within

theorem pilot_source (program : Program)
    (geometry : PilotOrdinaryRetainedGeometry.Geometry program (PerApplicationFixedPoint.logicalWidth program))
    (column : Fin Layout.PilotSpartan.spartanColumnCount) :
    CommonForm program ((PilotOrdinaryDirectPlan.sourceMap geometry).form column) := by
  dsimp only [PilotOrdinaryDirectPlan.sourceMap]
  split
  · exact empty _
  · exact pilot_location program geometry _

theorem pilot_ordinary (program : Program)
    (geometry : PilotOrdinaryRetainedGeometry.Geometry program (PerApplicationFixedPoint.logicalWidth program)) :
    CommonPlans program (PilotOrdinaryDirectPlan.plan geometry) := by
  apply ordinary_plan
  intro row
  exact source_row _ _ (pilot_source program geometry) (one_common program _ rfl) _ _

theorem pilot_binding (program : Program)
    (geometry : PilotOrdinaryRetainedGeometry.Geometry program (PerApplicationFixedPoint.logicalWidth program)) :
    CommonPlans program (PilotDigestBindingPlan.plan geometry) := by
  apply pin_plan
  · exact one_common program _ rfl
  · intro row
    apply add
    · unfold PilotDigestBindingPlan.legacyForm
      dsimp only
      split
      · exact pilot_location program geometry (.priorDigest _)
      · exact pilot_location program geometry (.outputState _)
    · apply scale
      unfold PilotDigestBindingPlan.derivedForm
      dsimp only
      split
      · exact external _ (fun _ => prior_sbox program (PilotDigestBindingPlan.poseidonGeometry geometry) _) _
      · exact external _ (fun _ => output_sbox program (PilotDigestBindingPlan.poseidonGeometry geometry) _) _

theorem public_output (program : Program)
    (geometry : ApplicationRetainedGeometry.Geometry program (PerApplicationFixedPoint.logicalWidth program)) :
    CommonPlans program (RecursivePublicOutputPlan.plan geometry) := by
  have bit (word : Fin 4) (index : Nat) : CommonForm program (RecursivePublicOutputPlan.bitForm geometry word index) := by
    apply FormSupport.singleton
    left
    change (RecursivePublicOutputPlan.publicBitIndex word index).val < RetainedLayout.hashEnd program
    have bound := (RecursivePublicOutputPlan.publicBitIndex word index).isLt
    change (RecursivePublicOutputPlan.publicBitIndex word index).val < 270 at bound
    rw [(RetainedLayout.boundaries program).1]
    omega
  have bits (word : Fin 4) (indices : List Nat) (base : SparseForm (PerApplicationFixedPoint.logicalWidth program))
      (supported : CommonForm program base) : CommonForm program
      (indices.foldl (fun form index => SparseForm.add form (RecursivePublicOutputPlan.bitForm geometry word index)) base) := by
    induction indices generalizing base with
    | nil => exact supported
    | cons index rest ih => exact ih _ (add supported (bit word index))
  apply pin_plan
  · exact one_common program _ rfl
  · intro row
    exact add (pilot_location program (RecursivePublicOutputPlan.pilotOrdinaryGeometry geometry) (.outputDigest row))
      (scale (-1) (bits row _ .empty (empty _)))

end NightstreamFPrime.Export.Stage1.Wide.ReadSupport
