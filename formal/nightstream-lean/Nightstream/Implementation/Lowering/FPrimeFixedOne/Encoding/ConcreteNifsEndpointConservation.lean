import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsOperationalConservation
import Nightstream.Implementation.R1CS.Canonical.KSplitNcEndpointsSupport

/-!
Contract: selected call-frame conservation for the complete Split-NC endpoint
program.

This module connects the generic endpoint support theorem to the one global
fixed-one `nifsVerify` column map. Core and round challenges come from the
proved transcript allocation; claimed values come from exact proof-codec
views; the five chain values are the declared pre-transcript temporaries.
No Rust layout, measured row range, digest, or verifier conclusion is used.
-/

set_option autoImplicit false
set_option maxRecDepth 100000
set_option maxHeartbeats 1200000

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsEndpointConservation

open Nightstream.HyperNova.Construction2.Paper
open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.DirectCalls
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsCallFrame
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsCallColumnMap
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsGlobalColumnMap
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsParameters
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsOperationalProfile
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.Implementation.R1CS.Canonical.KMul
open Nightstream.Implementation.R1CS.Canonical.KCompositeRowSupport
open Nightstream.Implementation.R1CS.Canonical.KSplitNcEndpointsSupport
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc

private abbrev TranscriptState := Poseidon2Duplex.State
private abbrev KColumns :=
  Nightstream.Implementation.R1CS.ProjectionProgram.KColumns

private theorem carried_below_of_columns
    (columns : KColumns) (boundary : Nat)
    (below : columns.c0 < boundary ∧ columns.c1 < boundary) :
    CarriedBelow (KFixedPhaseSemanticOccurrence.carried columns) boundary := by
  constructor <;> intro column mentioned
  · have same : column = columns.c0 := by
      simpa [KFixedPhaseSemanticOccurrence.carried, LinCombNormal.Mentions]
        using mentioned
    omega
  · have same : column = columns.c1 := by
      simpa [KFixedPhaseSemanticOccurrence.carried, LinCombNormal.Mentions]
        using mentioned
    omega

private theorem carriedAt_below
    {count boundary : Nat}
    (columns : List KColumns) (length : columns.length = count)
    (allBelow :
      ∀ value ∈ columns,
        value.c0 < boundary ∧ value.c1 < boundary)
    (index : Fin count) :
    CarriedBelow (KSplitNcEndpoints.carriedAt columns length index) boundary := by
  unfold KSplitNcEndpoints.carriedAt
  apply carried_below_of_columns
  exact allBelow _ (List.get_mem columns _)

section SelectedFrame

variable {shape : SemanticShape}
variable {Digest AppState Witness Encoded RunningWitness FreshWitness : Type}
variable [DecidableEq AppState] [DecidableEq Encoded]
variable {publicRingColumns verifierRows : Nat}
variable {publicFits :
  ringDegree * publicRingColumns <= shape.carrierWidth}
variable {keys : Fin 1 →
  SelectedKey shape TranscriptState publicRingColumns publicFits verifierRows}
variable {defaultRunning :
  SelectedRunning shape publicRingColumns publicFits verifierRows}
variable {machine :
  Machine
    (SelectedKey shape TranscriptState publicRingColumns publicFits
      verifierRows)
    Digest AppState Witness
    (SelectedRunning shape publicRingColumns publicFits verifierRows)
    (SelectedFresh shape publicRingColumns publicFits verifierRows)
    Encoded 1}
variable {terminalRelations :
  TerminalRelations
    (SelectedKey shape TranscriptState publicRingColumns publicFits
      verifierRows)
    (SelectedRunning shape publicRingColumns publicFits verifierRows)
    RunningWitness
    (SelectedFresh shape publicRingColumns publicFits verifierRows)
    FreshWitness 1}
variable {terminalChecks :
  Nightstream.Protocol.FPrime.CanonicalTerminalVerifier.RelationChecks
    terminalRelations}
variable {widths : Widths} {footprints : Footprints}

local notation "Selected" =>
  ConcreteNifsParameters.selected keys defaultRunning machine
    terminalRelations terminalChecks widths footprints

private abbrev FamilyFor
    (application : Poseidon23ApplicationProfile Selected) :=
  application.family Selected

private abbrev FrameFor
    (application : Poseidon23ApplicationProfile Selected)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)} :=
  CallFrame (signature := signature Selected)
    (FamilyFor application) Call.nifsVerify
    (Refs.cons runningRef
      (Refs.cons freshRef (Refs.cons proofRef .nil)))

private theorem temporaryBase_le_numericBase
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef)) :
    temporaryBase frame ≤
      KSplitNcOperationalRows.numericBase
        (ConcreteNifsOperationalOccurrence.input application profile frame) := by
  change temporaryBase frame ≤
    temporarySource frame 10 +
      (KSplitNcTranscript.replay
        (ConcreteNifsOperationalOccurrence.transcriptInput
          application profile frame)).afterOutput.entries.length *
        SymbolicDuplex.stride
  unfold temporarySource
  omega

private theorem numericBase_le_endpointBase
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef)) :
    KSplitNcOperationalRows.numericBase
        (ConcreteNifsOperationalOccurrence.input application profile frame) ≤
      KSplitNcOperationalRows.endpointBase
        (ConcreteNifsOperationalOccurrence.input application profile frame) := by
  unfold KSplitNcOperationalRows.endpointBase
  omega

private theorem proofView_below_endpointBase
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    {value :
      SelectedProof shape TranscriptState publicRingColumns publicFits
          verifierRows →
        Nightstream.SuperNeo.Concrete.K}
    (view :
      PaperNifsCodecProjection.KView
        ((FamilyFor application).codecFor (.data .nifsProof)) value) :
    CarriedBelow
      (ConcreteNifsOperationalFrame.proofLocation
        (FamilyFor application) frame view).carried
      (KSplitNcOperationalRows.endpointBase
        (ConcreteNifsOperationalOccurrence.input application profile frame)) := by
  apply carried_below_of_columns
  have visible :=
    ConcreteNifsOperationalFrame.proofLocation_numeric_lt
      (FamilyFor application) frame view
  have visibleToNumeric :=
    temporaryBase_le_numericBase application profile frame
  have numericToEndpoint :=
    numericBase_le_endpointBase application profile frame
  exact
    ⟨Nat.lt_of_lt_of_le visible.1
        (Nat.le_trans visibleToNumeric numericToEndpoint),
      Nat.lt_of_lt_of_le visible.2
        (Nat.le_trans visibleToNumeric numericToEndpoint)⟩

/-- Every selected proof-codec authority view is below the transcript replay
allocation.  Numeric and endpoint witnesses therefore preserve its decoded
value. -/
theorem proofView_below_transcriptBase
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    {value :
      SelectedProof shape TranscriptState publicRingColumns publicFits
          verifierRows →
        Nightstream.SuperNeo.Concrete.K}
    (view :
      PaperNifsCodecProjection.KView
        ((FamilyFor application).codecFor (.data .nifsProof)) value) :
    CarriedBelow
      (ConcreteNifsOperationalFrame.proofLocation
        (FamilyFor application) frame view).carried
      (ConcreteNifsOperationalOccurrence.transcriptInput
        application profile frame).transcriptBase := by
  apply carried_below_of_columns
  have visible :=
    ConcreteNifsOperationalFrame.proofLocation_numeric_lt
      (FamilyFor application) frame view
  constructor
  · change
      (ConcreteNifsOperationalFrame.proofLocation
          (FamilyFor application) frame view).numeric.c0 <
        temporarySource frame 10
    unfold temporarySource
    omega
  · change
      (ConcreteNifsOperationalFrame.proofLocation
          (FamilyFor application) frame view).numeric.c1 <
        temporarySource frame 10
    unfold temporarySource
    omega

private theorem numeric_below_endpoint
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    {value : Carried}
    (below :
      CarriedBelow value
        (KSplitNcOperationalRows.numericBase
          (ConcreteNifsOperationalOccurrence.input
            application profile frame))) :
    CarriedBelow value
      (KSplitNcOperationalRows.endpointBase
        (ConcreteNifsOperationalOccurrence.input application profile frame)) :=
  carried_mono below
    (numericBase_le_endpointBase application profile frame)

/-- Every source consumed by the selected endpoint rows is physically bound
to the transcript allocation or the exact proof/output codec. -/
theorem endpointInputs_below
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef)) :
    KSplitNcEndpointsSupport.InputsBelow
      (KSplitNcOperationalRows.endpointInput
        (ConcreteNifsOperationalOccurrence.input
          application profile frame)) := by
  let input :=
    ConcreteNifsOperationalOccurrence.input application profile frame
  let transcript := input.transcript
  let endpoint := KSplitNcOperationalRows.endpointInput input
  have core :=
    ConcreteNifsOperationalConservation.coreColumns_below_numericBase
      application profile frame
  have feRows :=
    ConcreteNifsOperationalConservation.feRowChallenges_below_numericBase
      application profile frame
  have feLanes :=
    ConcreteNifsOperationalConservation.feLaneChallenges_below_numericBase
      application profile frame
  have ncBlocks :=
    ConcreteNifsOperationalConservation.ncBlockChallenges_below_numericBase
      application profile frame
  have ncLanes :=
    ConcreteNifsOperationalConservation.ncLaneChallenges_below_numericBase
      application profile frame
  have temp (index : Nat) (bounded : index < 5) :
      CarriedBelow
        (KFixedPhaseSemanticOccurrence.carried
          (ConcreteNifsOperationalOccurrence.temporaryK
            (FamilyFor application) frame index))
        (KSplitNcOperationalRows.endpointBase input) := by
    apply numeric_below_endpoint application profile frame
    apply carried_below_of_columns
    exact ConcreteNifsOperationalConservation.temporaryK_below_numericBase
      application profile frame index bounded
  have coreAt
      {count : Nat}
      (columns : List KColumns) (length : columns.length = count)
      (allBelow :
        ∀ value ∈ columns,
          value.c0 < KSplitNcOperationalRows.numericBase input ∧
            value.c1 < KSplitNcOperationalRows.numericBase input)
      (index : Fin count) :
      CarriedBelow (KSplitNcEndpoints.carriedAt columns length index)
        (KSplitNcOperationalRows.endpointBase input) :=
    numeric_below_endpoint application profile frame
      (carriedAt_below columns length allBelow index)
  refine {
    feInitialGamma := ?_
    feInitialAlpha := ?_
    feInitialClaims := ?_
    feInitialEndpoint := ?_
    feTerminalGamma := ?_
    feTerminalAlpha := ?_
    feTerminalBetaA := ?_
    feTerminalBetaR := ?_
    feTerminalPointLane := ?_
    feTerminalPointRow := ?_
    feTerminalPriorPoint := ?_
    feTerminalMessage := ?_
    feTerminalEndpoint := ?_
    ncGamma := ?_
    ncBetaBlock := ?_
    ncBetaA := ?_
    ncPointBlock := ?_
    ncPointLane := ?_
    ncMessage := ?_
    ncInitialEndpoint := ?_
    ncTerminalEndpoint := ?_
  }
  · apply numeric_below_endpoint application profile frame
    exact carried_below_of_columns _ _ core.gamma
  · intro coordinate
    exact coreAt _
      (KSplitNcTranscript.Core.alpha_length _) core.alpha coordinate
  · intro running matrix lane
    exact proofView_below_endpointBase application profile frame
      (profile.endpointViews.claimedYRing running matrix lane)
  · exact temp 0 (by omega)
  · apply numeric_below_endpoint application profile frame
    exact carried_below_of_columns _ _ core.gamma
  · intro coordinate
    exact coreAt _
      (KSplitNcTranscript.Core.alpha_length _) core.alpha coordinate
  · intro coordinate
    exact coreAt _
      (KSplitNcTranscript.Core.betaA_length _) core.betaA coordinate
  · intro coordinate
    exact coreAt _
      (KSplitNcTranscript.Core.betaR_length _) core.betaR coordinate
  · intro coordinate
    exact coreAt _
      (KSplitNcEndpoints.feLaneChallenges_length transcript)
      feLanes coordinate
  · intro coordinate
    exact coreAt _
      (KSplitNcEndpoints.feRowChallenges_length transcript)
      feRows coordinate
  · intro coordinate
    exact proofView_below_endpointBase application profile frame
      (profile.endpointViews.priorPoint coordinate)
  · intro source matrix lane
    exact proofView_below_endpointBase application profile frame
      (profile.endpointViews.outputYRing source matrix lane)
  · exact temp 2 (by omega)
  · apply numeric_below_endpoint application profile frame
    exact carried_below_of_columns _ _ core.gamma
  · intro coordinate
    exact coreAt _
      (KSplitNcTranscript.Core.betaBlock_length _) core.betaBlock coordinate
  · intro coordinate
    exact coreAt _
      (KSplitNcTranscript.Core.betaA_length _) core.betaA coordinate
  · intro coordinate
    exact coreAt _
      (KSplitNcEndpoints.ncBlockChallenges_length transcript)
      ncBlocks coordinate
  · intro coordinate
    exact coreAt _
      (KSplitNcEndpoints.ncLaneChallenges_length transcript)
      ncLanes coordinate
  · intro source lane
    exact proofView_below_endpointBase application profile frame
      (profile.endpointViews.outputYZcol source lane)
  · exact temp 3 (by omega)
  · exact temp 4 (by omega)

/-- Every selected endpoint row lies below the exact end of the operational
Split-NC allocation. -/
theorem endpointRows_below_afterAllocation
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef)) :
    RowsBelow
      (KSplitNcOperationalRows.endpointRows
        (ConcreteNifsOperationalOccurrence.input application profile frame))
      (KSplitNcOperationalRows.afterAllocation
        (ConcreteNifsOperationalOccurrence.input application profile frame)) := by
  let input :=
    ConcreteNifsOperationalOccurrence.input application profile frame
  apply KSplitNcEndpointsSupport.rows_below
      (KSplitNcOperationalRows.endpointInput input)
      (KSplitNcOperationalRows.afterAllocation input)
  · have positive : 0 < input.transcript.transcriptBase := by
      dsimp only [input, ConcreteNifsOperationalOccurrence.input,
        ConcreteNifsOperationalOccurrence.transcriptInput]
      unfold temporarySource temporaryBase visibleIds
      simp
    unfold KSplitNcOperationalRows.afterAllocation
    omega
  · exact endpointInputs_below application profile frame
  · rw [KSplitNcOperationalRows.afterAllocation_eq_endpoint_end]
    change KSplitNcOperationalRows.endpointBase input +
        KSplitNcEndpoints.allocationWidth
          (KSplitNcOperationalRows.endpointInput input) ≤
      KSplitNcOperationalRows.endpointBase input +
        KSplitNcEndpoints.allocationWidth
          (KSplitNcOperationalRows.endpointInput input)
    exact Nat.le_refl _

/-- The complete selected operational ΠCCS program—transcript, numeric
claimed-chain rows, and endpoint rows—mentions no column at or beyond its
exact compact allocation end. -/
theorem operationalRows_below_afterAllocation
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef)) :
    RowsBelow
      (ConcreteNifsOperationalOccurrence.rows application profile frame)
      (KSplitNcOperationalRows.afterAllocation
        (ConcreteNifsOperationalOccurrence.input application profile frame)) := by
  let input :=
    ConcreteNifsOperationalOccurrence.input application profile frame
  have numericToEnd :
      KSplitNcOperationalRows.numericBase input ≤
        KSplitNcOperationalRows.afterAllocation input := by
    unfold KSplitNcOperationalRows.afterAllocation
      KSplitNcOperationalRows.allocationWidth
      KSplitNcOperationalRows.numericBase
    omega
  have endpointToEnd :
      KSplitNcOperationalRows.endpointBase input ≤
        KSplitNcOperationalRows.afterAllocation input := by
    rw [KSplitNcOperationalRows.afterAllocation_eq_endpoint_end]
    omega
  have endpoints :=
    endpointRows_below_afterAllocation application profile frame
  intro row member column mentioned
  change row ∈ KSplitNcOperationalRows.rows profile.constants input at member
  rcases List.mem_flatten.mp member with
    ⟨group, groupMember, rowMember⟩
  simp only [KSplitNcOperationalRows.rowGroups, List.mem_cons,
    List.not_mem_nil, or_false] at groupMember
  rcases groupMember with rfl | rfl | rfl
  · exact Nat.lt_of_lt_of_le
      (ConcreteNifsOperationalConservation.transcriptRows_below_numericBase
        application profile frame row rowMember column mentioned)
      numericToEnd
  · exact Nat.lt_of_lt_of_le
      (ConcreteNifsOperationalConservation.numericRows_below_endpointBase
        application profile frame row rowMember column mentioned)
      endpointToEnd
  · exact endpoints row rowMember column mentioned

end SelectedFrame

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsEndpointConservation
