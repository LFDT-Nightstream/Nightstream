import NightstreamFPrime.Export.Stage1.PiCCSDecodedTranscript
import NightstreamFPrime.Export.Stage1.PiCCSTranscriptEndpointPlan

/-!
Owns the connection from the existing endpoint pin rows to the four PiCCS
transcript states in the arbitrary decoded environment. The output state
uses its existing owned block; the other states use retained output forms.
-/

namespace NightstreamFPrime.Export.Stage1.PiCCSDecodedEndpoints

open NightstreamFPrime.Circuit
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Layout.ProductionRelation
open PiCCSTranscriptEndpointPlan

variable {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}

private theorem output_not_source (lane : Fin laneCount) :
    ¬ PiCCSOrdinarySourceSupport.Source
      (endpointColumn outputFamily lane) := by
  have laneBound : lane.val < 8 := lane.isLt
  intro support
  rcases support with (external | transcript | ordinary) | fresh
  · norm_num [PiCCSOrdinarySourceSupport.External,
      PiCCSOrdinarySourceSupport.InRange, endpointColumn, endpointStart,
      outputFamily, PiCCSStarts.logicalFreshBase, PiCCSInputs.phaseOffset_eq,
      PilotProduction.priorPreimageStart, PilotProduction.priorPublicInputStart,
      PilotProduction.outputPreimageStart, PilotProduction.stateHashWords_eq,
      Lifecycle.PriorStateHash.publicWidth,
      Lifecycle.PaperAlgebra.publicRingColumns, ringDegree,
      PiCCSInputs.expectedContextStart_eq, PiCCSInputs.expectedContextWords,
      PiCCSInputs.proofInputStart_eq] at external
    omega
  · rcases transcript with ⟨invocation, outputLane, equality⟩
    have invocationBound : invocation.val < 718 := by
      simpa only [PiCCSOrdinarySourceSupport.transcriptInvocationCount_eq]
        using invocation.isLt
    have outputLaneBound : outputLane.val < 8 := outputLane.isLt
    norm_num [endpointColumn, endpointStart, outputFamily,
      PiCCSStarts.logicalFreshBase, PiCCSInputs.phaseOffset_eq] at equality
    omega
  · norm_num [PiCCSOrdinarySourceSupport.OrdinaryLogical,
      PiCCSOrdinarySourceSupport.InRange,
      PiCCSOrdinarySourceSupport.ordinaryLogicalCount_eq,
      PiCCSStarts.initialClaimLogicalStart,
      PiCCSStarts.roundTranscriptWitnessStart_eq, endpointColumn,
      endpointStart, outputFamily, PiCCSStarts.logicalFreshBase,
      PiCCSInputs.phaseOffset_eq] at ordinary
    omega
  · norm_num [PiCCSStarts.initialClaimFreshStart,
      PiCCSStarts.roundTranscriptFreshStart, PiCCSStarts.challengeFreshStart,
      PiCCSStarts.statementAbsorptionFreshStart,
      PiCCSStarts.statementBindingFreshStart, PiCCSStarts.logicalFreshBase,
      PiCCSInputs.phaseOffset_eq, endpointColumn, endpointStart,
      outputFamily] at fresh
    omega

private theorem output_classify_none (lane : Fin laneCount) :
    PiCCSOrdinaryDirectPlan.classifySource
      (endpointColumn outputFamily lane) = none := by
  cases found : PiCCSOrdinaryDirectPlan.classifySource
      (endpointColumn outputFamily lane) with
  | none => rfl
  | some located =>
      exact False.elim (output_not_source lane (by
        rw [← located.owns]
        exact located.location.sourceSupport))

private theorem decoded_output
    (geometry : PiCCSOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth) (lane : Fin laneCount) :
    PiCCSAssignmentSoundness.decodedEnv geometry assignment
        (Spartan.sourceToSpartan (endpointColumn outputFamily lane)) =
      (sourceForm geometry outputFamily lane).eval assignment := by
  let column : Fin Spartan.spartanColumnCount :=
    ⟨Spartan.sourceToSpartan (endpointColumn outputFamily lane),
      Spartan.sourceToSpartan_lt _ (endpointColumn_lt_source outputFamily lane)⟩
  have inverse := Spartan.spartanToSource_sourceToSpartan
    (endpointColumn outputFamily lane) (endpointColumn_lt_source outputFamily lane)
  have inside : PiCCSOrdinarySourceSupport.InRange
      PiCCSOrdinaryRetainedBlocks.outputEndpointStart 8
        (endpointColumn outputFamily lane) := by
    change PiCCSOrdinaryRetainedBlocks.outputEndpointStart ≤
        PiCCSOrdinaryRetainedBlocks.outputEndpointStart + lane.val ∧
      PiCCSOrdinaryRetainedBlocks.outputEndpointStart + lane.val <
        PiCCSOrdinaryRetainedBlocks.outputEndpointStart + 8
    have bound : lane.val < 8 := lane.isLt
    omega
  have sameIndex : PiCCSOrdinaryDirectPlan.rangeIndex inside = lane := by
    apply Fin.ext
    change (PiCCSOrdinaryRetainedBlocks.outputEndpointStart + lane.val) -
        PiCCSOrdinaryRetainedBlocks.outputEndpointStart = lane.val
    omega
  have selected : (PiCCSOrdinaryDirectPlan.sourceMap geometry).form column =
      sourceForm geometry outputFamily lane := by
    simp only [PiCCSOrdinaryDirectPlan.sourceMap,
      PiCCSOrdinaryDirectPlan.classifyTarget, column, inverse,
      output_classify_none, PiCCSOrdinaryDirectPlan.endpointForm,
      dif_pos inside, sourceForm,
      dif_pos (show outputFamily.val = 3 from rfl)]
    exact congrArg _ sameIndex
  change SourceCompiler.sourceEnv
      (fun c => ((PiCCSOrdinaryDirectPlan.sourceMap geometry).form c).eval
        assignment) column.val = _
  rw [SourceCompiler.sourceEnv_at, selected]

/-- Each endpoint form is read from its actual owned coordinates in the
same decoded environment used by the arithmetic and transcript contracts. -/
theorem sourceForm_eval
    (geometry : PiCCSOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (family : Fin familyCount) (lane : Fin laneCount) :
    (PiCCSTranscriptEndpointPlan.sourceForm geometry family lane).eval assignment =
      Spartan.pullback (PiCCSAssignmentSoundness.decodedEnv geometry assignment)
        (endpointColumn family lane) := by
  by_cases output : family.val = 3
  · have same : family = outputFamily := Fin.ext output
    subst family
    exact (decoded_output geometry assignment lane).symm
  · have mapped := PiCCSAssignmentSoundness.decodedEnv_location geometry assignment
      (.proofLogical (proofLogicalIndex family output lane))
    rw [PiCCSOrdinaryDirectPlan.Location.sourceColumn,
      proofLogicalIndex_source] at mapped
    simpa only [PiCCSTranscriptEndpointPlan.sourceForm, dif_neg output,
      Spartan.pullback] using mapped.symm

/-- The existing 32 endpoint rows force all four direct states to equal
the lifecycle values decoded from the same arbitrary assignment. -/
theorem rowsZero_implies_endpointStates
    (ordinary : PiCCSOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (poseidon : PiCCSPoseidonPlan.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (one : assignment (PiCCSOrdinaryRetainedGeometry.oneColumn ordinary) = 1)
    (rows : (PiCCSTranscriptEndpointPlan.plan poseidon ordinary).RowsZero assignment)
    (family : Fin familyCount) :
    PiCCSPoseidonPreservation.valueState poseidon assignment
        (endpointInvocation family) =
      List.ofFn fun lane : Fin laneCount =>
        Spartan.pullback (PiCCSAssignmentSoundness.decodedEnv ordinary assignment)
          (endpointColumn family lane) := by
  unfold PiCCSPoseidonPreservation.valueState
  apply congrArg List.ofFn
  funext lane
  have rowZero := (PiCCSTranscriptEndpointPlan.rowsZero_iff poseidon ordinary
    assignment one).mp rows (row family lane)
  rw [bindingForm, descriptor_row, SparseForm.add_eval,
    SparseForm.scale_eval] at rowZero
  have formsEq : (directForm poseidon family lane).eval assignment =
      (PiCCSTranscriptEndpointPlan.sourceForm ordinary family lane).eval assignment := by
    apply Lean.Grind.AddCommGroup.sub_eq_zero_iff.mp
    simpa [sub_eq_add_neg] using rowZero
  exact formsEq.trans (sourceForm_eval ordinary assignment family lane)

/-- Accepted direct transcript and endpoint rows imply all four transcript
leaf contracts in the arbitrary decoded environment, without an encoding
or representation premise. -/
theorem rowsZero_implies_transcriptSpecs
    {relationLogicalWidth : Nat}
    {relationPublicFits : ringDegree * Lifecycle.PaperAlgebra.publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth relationLogicalWidth}
    (ordinary : PiCCSOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (poseidon : PiCCSPoseidonPlan.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (one : assignment (PiCCSOrdinaryRetainedGeometry.oneColumn ordinary) = 1)
    (transcriptRows : (PiCCSPoseidonPlan.plan (PiCCSPayloadWiring.form ordinary)
      poseidon).RowsZero assignment)
    (endpointRows : (PiCCSTranscriptEndpointPlan.plan poseidon ordinary).RowsZero
      assignment) :
    PiCCSInvocations.TranscriptSpecs relationLogicalWidth relationPublicFits
      (PiCCSAssignmentSoundness.decodedEnv ordinary assignment) :=
  PiCCSTranscriptEndpointPlan.traces_and_endpoints_imply_transcriptSpecs
    poseidon assignment (PiCCSAssignmentSoundness.decodedEnv ordinary assignment)
    (PiCCSDecodedTranscript.rowsZero_implies_traces ordinary poseidon assignment
      one transcriptRows)
    (rowsZero_implies_endpointStates ordinary poseidon assignment one endpointRows)

end NightstreamFPrime.Export.Stage1.PiCCSDecodedEndpoints
