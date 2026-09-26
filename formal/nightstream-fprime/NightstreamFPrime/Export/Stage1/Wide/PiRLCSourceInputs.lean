import NightstreamFPrime.Export.Stage1.Wide.SourceAssignment
import NightstreamFPrime.Export.Stage1.PiCCSEndpointCompleteness

/-! Connect the direct PiRLC constructor's inputs to the wide physical source.
The right operands are copied fields; the initial state is the checked final
PiCCS permutation output. No old sampler execution is needed. -/

namespace NightstreamFPrime.Export.Stage1.Wide.PiRLCSourceInputs

open NightstreamFPrime.Circuit NightstreamFPrime.Spec NightstreamFPrime.Layout
open NightstreamFPrime.Lifecycle NightstreamFPrime.Lifecycle.PaperAlgebra
open ProductionRelation
open Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra
open Spec.Folding.PiCCS.PaperJoint

abbrev Program := RetainedLayout.Program

variable (program : Program) (env : Env)
  (application : Fin (PerApplicationPackage.addedPrivateColumnCount program) → F)


theorem input_value (ring : PiRLCGeometry.RingIndex) (lane : Fin ringDegree) :
    PiRLCWitness.inputValues (Stage1Plan.piRlcInterface program) (AssignmentProjection.seed program (SourceAssignment.raw program env application).assignment) ring lane =
      env ((PiRLCProductSchedule.descriptor (PiRLCProductRingSchedule.laneInvocation ring lane)).valueColumn lane) := by
  let invocation := PiRLCProductRingSchedule.laneInvocation ring lane
  have scope := InputSupport.location program (Stage1Plan.piCcsGeometry program)
    (PiRLCValueWiring.located invocation).location
  dsimp only [PiRLCWitness.inputValues, Phi81ProductPlan.evalState, Stage1Plan.piRlcInterface, Stage1Plan.value]
  refine (AssignmentProjection.seed_form program (SourceAssignment.raw program env application).assignment
    (PiRLCValueWiring.form (Stage1Plan.piCcsGeometry program) invocation) scope _).trans ?_
  have encoded := (PerApplicationCanonicalEncodes.samplerPrefixEncodes (SourceAssignment.raw program env application)).prior.pilotOrdinary.prior
  rw [PiRLCValueWiring.form_eval_source _ _ _ (SourceAssignment.raw program env application).base (SourceAssignment.raw program env application).groupValue (SourceAssignment.raw program env application).products encoded]
  have descriptorLane : (PiRLCProductSchedule.descriptor invocation).lane = lane := by
    rw [PiRLCProductRingSchedule.descriptor_laneInvocation]
    rfl
  have bounded := Layout.Stage1.PiCCSOrdinarySourceSupport.source_lt_sourceColumnCount
    (PiRLCValueWiring.valueSource_support (PiRLCProductSchedule.descriptor invocation))
  have copied := SourceAssignment.raw_source program env application _ bounded
  change PiRLCProductPlan.baseEnv program (SourceAssignment.raw program env application).base _ = _ at copied
  rw [copied, SourceAssignment.sourceEnv_prefix]
  · rw [descriptorLane]
  · exact lt_of_lt_of_le (PiRLCValueWiring.valueSource_beforePhase _)
      (by change 14751804 ≤ 19513117; decide)

variable {width : Nat}
  {fits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth width}

/-- The source view preserves all accepted PiCCS rows, using their proved scope. -/
theorem piCcs_physical (relation : ProductionKey.LogicalRelation width fits)
    (ajtai : AjtaiKey (logicalWidth := width) (publicFits := fits)) (template : Proof 9)
    (assumptions : PiCCS.v1_1.Formal.Assumptions relation
      (Layout.Stage1.PiCCSInputs.interface width fits) Layout.Stage1.PiCCSInputs.phaseOffset env)
    (physical : Layout.PiCCS.v1_1.PhysicalHolds relation
      (Layout.Stage1.PiCCSInputs.interface width fits) Layout.Stage1.PiCCSInputs.phaseOffset env) :
    Layout.PiCCS.v1_1.PhysicalHolds relation
      (Layout.Stage1.PiCCSInputs.interface width fits) Layout.Stage1.PiCCSInputs.phaseOffset
      (Layout.Stage1.Spartan.pullback (SourceAssignment.targetEnv env)) := by
  have phase := Layout.PiCCS.v1_1.physical_implies_phaseHolds relation ajtai
    (Layout.Stage1.PiCCSInputs.interface width fits) template Layout.Stage1.PiCCSInputs.phaseOffset
    env assumptions physical
  have scope := Layout.PiCCS.v1_1.physicalRows_varsBelow_of_phase relation ajtai
    (Layout.Stage1.PiCCSInputs.interface width fits) template Layout.Stage1.PiCCSInputs.phaseOffset
    env assumptions phase
  have endpoint : Layout.PiCCS.v1_1.physicalColumnCount relation
      (Layout.Stage1.PiCCSInputs.interface width fits) Layout.Stage1.PiCCSInputs.phaseOffset ≤
      SourceAssignment.prefixEnd := by
    rw [SourceAssignment.prefixEnd, Layout.Stage1.PiRLCInputs.phaseOffset_matches_piCcs relation]
    exact Nat.le_max_right _ _
  apply R1CS.rowsHold_of_agree_below _ _ env _ scope _ physical
  intro source below
  have prefixBound := lt_of_lt_of_le below endpoint
  change SourceAssignment.targetEnv env (Layout.Stage1.Spartan.sourceToSpartan source) = env source
  rw [SourceAssignment.targetEnv_source env source (by
    have total := Layout.Stage1.Spartan.sourceColumnCount_eq
    change source < 19513117 at prefixBound
    rw [total]; omega)]
  exact SourceAssignment.sourceEnv_prefix env source prefixBound

/-- The direct sampler starts from the checked physical PiCCS transcript. -/
theorem initial_value (relation : ProductionKey.LogicalRelation width fits)
    (ajtai : AjtaiKey (logicalWidth := width) (publicFits := fits)) (template : Proof 9)
    (assumptions : PiCCS.v1_1.Formal.Assumptions relation
      (Layout.Stage1.PiCCSInputs.interface width fits) Layout.Stage1.PiCCSInputs.phaseOffset env)
    (physical : Layout.PiCCS.v1_1.PhysicalHolds relation
      (Layout.Stage1.PiCCSInputs.interface width fits) Layout.Stage1.PiCCSInputs.phaseOffset env)
    (lane : Fin 8) :
    PiRLCWitness.initial (Stage1Plan.piRlcInterface program) (AssignmentProjection.seed program (SourceAssignment.raw program env application).assignment) lane =
      ((Layout.Stage1.Wide.PiRLCInputs.piCcsOutputState (logicalWidth := width) (publicFits := fits)) lane).eval env := by
  dsimp only [PiRLCWitness.initial, SparseLayer.evalState, Stage1Plan.piRlcInterface, Stage1Plan.initialState]
  refine (AssignmentProjection.seed_form program (SourceAssignment.raw program env application).assignment
    (PiRLCSamplerPoseidonPlan.piCcsFinalOutput (Stage1Plan.poseidonGeometry program) lane)
    (InputSupport.piCcsOutput program (Stage1Plan.poseidonGeometry program) _ lane) _).trans ?_
  have physicalView := piCcs_physical env relation ajtai template assumptions physical
  have endpoint := congrFun (PiCCSCompletedReadout.outputValue_of_base program relation
    (SourceAssignment.targetEnv env) application physicalView (SourceAssignment.raw program env application) rfl
    (PiCCSTranscriptEndpointPlan.endpointInvocation PiCCSTranscriptEndpointPlan.outputFamily)) lane
  rw [PiCCSEndpointCompleteness.physicalEndpoint_column] at endpoint
  have stateColumn :
      (Layout.Stage1.Wide.PiRLCInputs.piCcsOutputState (logicalWidth := width) (publicFits := fits)) lane =
        .var (PiCCSTranscriptEndpointPlan.endpointColumn PiCCSTranscriptEndpointPlan.outputFamily lane) :=
    PiCCSTranscriptEndpointPlan.outputFinalState_endpoint_of_shape width fits lane
  rw [stateColumn]
  change PiCCSPoseidonPreservation.outputValue
    (PerApplicationCanonicalEncodes.poseidonGeometry program) (SourceAssignment.raw program env application).assignment
    (PiCCSTranscriptEndpointPlan.endpointInvocation PiCCSTranscriptEndpointPlan.outputFamily) lane = _
  rw [endpoint, SourceAssignment.targetEnv_source env _
    (PiCCSTranscriptEndpointPlan.endpointColumn_lt_source _ lane)]
  apply SourceAssignment.sourceEnv_prefix
  have bound : lane.val < 8 := lane.isLt
  change 19306098 + lane.val < 19513117
  omega

end NightstreamFPrime.Export.Stage1.Wide.PiRLCSourceInputs
