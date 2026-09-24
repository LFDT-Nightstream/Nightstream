import NightstreamFPrime.Export.Stage1.Wide.PilotPoseidonSupport

namespace NightstreamFPrime.Export.Stage1.Wide.ReadSupport

open NightstreamFPrime.Spec NightstreamFPrime.Layout
open ProductionRelation FormSupport

theorem piCcs_payload (program : Program)
    (geometry : PiCCSOrdinaryRetainedGeometry.Geometry program (PerApplicationFixedPoint.logicalWidth program))
    (index : Fin PiCCSActionPayloadBlock.payloadCount) : Form program (PiCCSPayloadWiring.form geometry index) := by
  have loaded := PiCCSPayloadWiring.form?_eq_some_form geometry index
  unfold PiCCSPayloadWiring.form? at loaded
  rw [PiCCSPayloadWiring.lowering?_eq_some_lowering] at loaded
  have same := Option.some.inj loaded
  rw [← same]
  exact source_combination (PiCCSPayloadWiring.sourceMap geometry)
    (PiCCSOrdinaryRetainedGeometry.oneColumn geometry) (fun column => piCcs_source program geometry _)
    (one program _ rfl) _ _

theorem piCcs_sbox (program : Program)
    (geometry : PiCCSPoseidonPlan.Geometry program (PerApplicationFixedPoint.logicalWidth program))
    (slot : Fin (PiCCSPoseidonPlan.retainedBlock program).slotCount) :
    Form program ((PiCCSPoseidonPlan.retainedBlock program).form
      (PiCCSPoseidonPlan.retainedStart program) (PiCCSPoseidonPlan.retainedFits geometry) slot) := by
  apply hash_block
  rw [PiCCSPoseidonPlan.retainedBlock_coordinateCount]
  exact le_of_eq (LaterPoseidonRetainedBlocks.samplerStart_eq program).symm

theorem piCcs_previous (program : Program)
    (geometry : PiCCSPoseidonPlan.Geometry program (PerApplicationFixedPoint.logicalWidth program))
    (invocation : Fin PiCCSPoseidonPlan.invocationCount) (lane : Fin 8) :
    Form program (PiCCSPoseidonPlan.previousOutput geometry invocation lane) := by
  unfold PiCCSPoseidonPlan.previousOutput
  split
  · exact empty _
  · exact common_form program _ (InputSupport.piCcsOutput program geometry _ lane)

theorem piCcs_poseidon (program : Program)
    (ordinary : PiCCSOrdinaryRetainedGeometry.Geometry program (PerApplicationFixedPoint.logicalWidth program))
    (geometry : PiCCSPoseidonPlan.Geometry program (PerApplicationFixedPoint.logicalWidth program)) :
    Plans program (PiCCSPoseidonPlan.plan (PiCCSPayloadWiring.form ordinary) geometry) := by
  have payload (invocation : Fin PiCCSPoseidonPlan.invocationCount) (lane : Fin 8) :
      Form program (PiCCSPoseidonPlan.payloadForm (PiCCSPayloadWiring.form ordinary) invocation lane) := by
    unfold PiCCSPoseidonPlan.payloadForm
    split
    · exact piCcs_payload program ordinary _
    · exact empty _
  apply append
  · apply poseidon_family
    · exact one program _ rfl
    · intro invocation lane
      dsimp only [PiCCSPoseidonPlan.interface, PoseidonRetainedFamily.familyInterface]
      unfold PiCCSPoseidonPlan.inputState
      cases PiCCSActionPayloadBlock.kindAt invocation
      · exact add (piCcs_previous program geometry invocation lane) (payload invocation lane)
      · exact piCcs_previous program geometry invocation lane
      · exact piCcs_previous program geometry invocation lane
    · intro invocation slot; exact piCcs_sbox program geometry _
  · apply pin_plan
    · exact one program _ rfl
    · intro row
      dsimp only [PiCCSPoseidonPlan.bindingInterface]
      unfold PiCCSPoseidonPlan.bindingForm
      split
      · apply add (payload _ _)
        apply scale
        unfold PiCCSPoseidonPlan.bindingActual
        split
        · exact piCcs_previous program geometry _ _
        · exact common_form program _ (InputSupport.piCcsOutput program geometry _ _)
      · exact empty _
      · exact empty _

theorem piCcs_pins (program : Program)
    (ordinary : PiCCSOrdinaryRetainedGeometry.Geometry program (PerApplicationFixedPoint.logicalWidth program))
    (geometry : PiCCSPoseidonPlan.Geometry program (PerApplicationFixedPoint.logicalWidth program)) :
    Plans program (PiCCSTranscriptEndpointPlan.plan geometry ordinary) := by
  apply pin_plan
  · exact one program _ rfl
  · intro row
    apply add (common_form program _ (InputSupport.piCcsOutput program geometry _ _))
    apply scale
    unfold PiCCSTranscriptEndpointPlan.sourceForm
    split
    · exact piCcs_endpoint_value program ordinary _
    · exact common_form program _ (InputSupport.location program ordinary _)

end NightstreamFPrime.Export.Stage1.Wide.ReadSupport
