import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsPiRlcPointRows
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsSelectedCallFrame
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsSelectedFePoint

/-!
Contract: exact semantic refinement and honest completeness for the selected
PiRLC parent-point binding rows.

Satisfaction of the operational transcript rows determines the frozen
ConcretePhi81 FE point.  Satisfaction of this slice then forces the decoded
running output's parent point to that exact row point.  No point, acceptance
bit, or source-authority proposition is supplied by the caller.
-/

set_option autoImplicit false
set_option maxRecDepth 100000
set_option maxHeartbeats 2000000

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsPiRlcPointSemantics

open Nightstream.HyperNova.Construction2.Paper
open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.Goldilocks.NumericRowBridge
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.DirectCalls
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsCallFrame
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsGlobalColumnMap
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsParameters
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsOperationalProfile
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.Implementation.R1CS.Canonical.KMul
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc

private abbrev TranscriptState := Poseidon2Duplex.State

private def pointAt
    {Value : Type} {variables : Nat}
    (point : CubePoint Value variables) (coordinate : Fin variables) : Value :=
  point.coordinates.get
    ⟨coordinate.val, by
      rw [point.dimension]
      exact coordinate.isLt⟩

/-- Coordinate equality determines two dimension-checked cube points. -/
private theorem cubePoint_eq_of_coordinates
    {Value : Type} {variables : Nat}
    (left right : CubePoint Value variables)
    (coordinates :
      ∀ coordinate : Fin variables,
        pointAt left coordinate = pointAt right coordinate) :
    left = right := by
  cases left with
  | mk leftCoordinates leftDimension =>
      cases right with
      | mk rightCoordinates rightDimension =>
          congr 1
          apply List.ext_get
          · omega
          · intro index leftLt rightLt
            let coordinate : Fin variables :=
              ⟨index, by omega⟩
            exact coordinates coordinate

/-- Two physical coordinate-equality rows imply equality of the decoded
quadratic-extension values. -/
private theorem decoded_eq_of_rows
    (assignment : Nat → Nat)
    (left right : Carried)
    (constantWire : assignment 0 = 1)
    (satisfied : Satisfies (KEquality.rows left right) assignment) :
    KPointEquality.decoded assignment left =
      KPointEquality.decoded assignment right := by
  rcases KEquality.rows_sound assignment left right constantWire satisfied
    with ⟨lowEqual, highEqual⟩
  apply KConcreteBridge.ofConcrete_injective
  rw [KPointEquality.ofConcrete_decoded,
    KPointEquality.ofConcrete_decoded]
  unfold KHorner.carriedValue
  rw [lowEqual, highEqual]

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

private theorem coordinate_satisfied
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame :
      CallFrame (signature := signature Selected)
        (FamilyFor application) Call.nifsVerify
        (Refs.cons runningRef
          (Refs.cons freshRef (Refs.cons proofRef .nil))))
    (assignment : Nat → Nat)
    (satisfied :
      Satisfies
        (ConcreteNifsPiRlcPointRows.rows application profile frame)
        assignment)
    (coordinate : Fin shape.rowVariables) :
    Satisfies
      (ConcreteNifsPiRlcPointRows.coordinateRows
        application profile frame coordinate)
      assignment := by
  intro row member
  exact satisfied row
    (List.mem_flatMap.2
      ⟨coordinate, List.mem_ofFn.2 ⟨coordinate, rfl⟩, member⟩)

/-- Every physical output-parent point coordinate is forced to the
verifier-derived PiRLC point before assuming that the output codec decodes. -/
theorem physical_coordinate_eq_derived
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame :
      CallFrame (signature := signature Selected)
        (FamilyFor application) Call.nifsVerify
        (Refs.cons runningRef
          (Refs.cons freshRef (Refs.cons proofRef .nil))))
    (assignment : ColumnId → Field)
    (running :
      SelectedRunning shape publicRingColumns publicFits verifierRows)
    (fresh :
      SelectedFresh shape publicRingColumns publicFits verifierRows)
    (proof :
      SelectedProof shape TranscriptState publicRingColumns publicFits
        verifierRows)
    (constantWire : assignment frame.one = 1)
    (decodedInputs :
      frame.operands.Decodes (FamilyFor application) assignment
        (.cons running (.cons fresh (.cons proof .nil))))
    (operationalSatisfied :
      Satisfies
        (ConcreteNifsOperationalOccurrence.rows application profile frame)
        (numericAssignment (columnMap frame) assignment))
    (pointSatisfied :
      Satisfies
        (ConcreteNifsPiRlcPointRows.rows application profile frame)
        (numericAssignment (columnMap frame) assignment))
    (coordinate : Fin shape.rowVariables) :
    KPointEquality.decoded
        (numericAssignment (columnMap frame) assignment)
        (ConcreteNifsPiRlcPointRows.outputCoordinate
          application profile frame coordinate) =
      pointAt
        (Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.derive
          (ConcreteNifsParameters.context
            (keys
              Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected)
            running fresh proof).materialize
          proof.certificate).piCcs.fePoint.row
        coordinate := by
  let numeric := numericAssignment (columnMap frame) assignment
  have numericOne :
      numeric 0 = 1 :=
    ConcreteNifsOperationalOccurrenceSemantics.numericConstantWire
      application frame assignment constantWire
  have carriedEqual :=
    decoded_eq_of_rows numeric
      (ConcreteNifsPiRlcPointRows.transcriptCoordinate
        application profile frame coordinate)
      (ConcreteNifsPiRlcPointRows.outputCoordinate
        application profile frame coordinate)
      numericOne
      (coordinate_satisfied application profile frame numeric
        pointSatisfied coordinate)
  have transcriptPoint :=
    KSplitNcEndpoints.feRowPoint_eq_decoded
      (ConcreteNifsPiRlcPointRows.endpointInput application profile frame)
      numeric
  have transcriptCoordinateExact :
      KPointEquality.decoded numeric
          (ConcreteNifsPiRlcPointRows.transcriptCoordinate
            application profile frame coordinate) =
        pointAt
          (KSplitNcTranscriptPhases.decodedFePoint numeric
            (ConcreteNifsOperationalOccurrence.transcriptInput
              application profile frame)).row coordinate := by
    have exact :=
      congrArg (fun point => pointAt point coordinate) transcriptPoint
    simpa [ConcreteNifsPiRlcPointRows.transcriptCoordinate,
      ConcreteNifsPiRlcPointRows.endpointInput,
      KSplitNcEndpoints.decodedPointOf, pointAt] using exact
  have selectedPoint :=
    ConcreteNifsSelectedFePoint.selectedFePoint_eq
      application profile frame assignment running fresh proof
      constantWire decodedInputs operationalSatisfied
  have retargetUnchanged :
      KSplitNcTranscriptPhases.decodedFePoint numeric
          (ConcreteNifsOperationalOccurrence.transcriptInput
            application profile frame) =
        KSplitNcTranscriptPhases.decodedFePoint numeric
          (KSplitNcStaticInput.retargetTranscript proof.piCcsInput
            (ConcreteNifsOperationalOccurrence.transcriptInput
              application profile frame)) := by
    rfl
  have selectedRow :
      (KSplitNcTranscriptPhases.decodedFePoint numeric
        (ConcreteNifsOperationalOccurrence.transcriptInput
          application profile frame)).row =
        (Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.derive
          (ConcreteNifsParameters.context
            (keys
              Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected)
            running fresh proof).materialize
          proof.certificate).piCcs.fePoint.row := by
    exact congrArg (fun point => point.row)
      (retargetUnchanged.trans selectedPoint)
  calc
    KPointEquality.decoded numeric
          (ConcreteNifsPiRlcPointRows.outputCoordinate
            application profile frame coordinate) =
        KPointEquality.decoded numeric
          (ConcreteNifsPiRlcPointRows.transcriptCoordinate
            application profile frame coordinate) :=
      carriedEqual.symm
    _ = pointAt
          (KSplitNcTranscriptPhases.decodedFePoint numeric
            (ConcreteNifsOperationalOccurrence.transcriptInput
              application profile frame)).row coordinate :=
      transcriptCoordinateExact
    _ = pointAt
          (Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.derive
            (ConcreteNifsParameters.context
              (keys
                Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected)
              running fresh proof).materialize
            proof.certificate).piCcs.fePoint.row coordinate :=
      congrArg (fun point => pointAt point coordinate) selectedRow

/-- The same physical coordinate stated directly against the parent point of
the frozen verifier result.  This is the public bridge needed by output-codec
reconstruction; the intermediate FE-point projection remains private here. -/
theorem physical_coordinate_eq_parent
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame :
      CallFrame (signature := signature Selected)
        (FamilyFor application) Call.nifsVerify
        (Refs.cons runningRef
          (Refs.cons freshRef (Refs.cons proofRef .nil))))
    (assignment : ColumnId → Field)
    (running :
      SelectedRunning shape publicRingColumns publicFits verifierRows)
    (fresh :
      SelectedFresh shape publicRingColumns publicFits verifierRows)
    (proof :
      SelectedProof shape TranscriptState publicRingColumns publicFits
        verifierRows)
    (constantWire : assignment frame.one = 1)
    (decodedInputs :
      frame.operands.Decodes (FamilyFor application) assignment
        (.cons running (.cons fresh (.cons proof .nil))))
    (operationalSatisfied :
      Satisfies
        (ConcreteNifsOperationalOccurrence.rows application profile frame)
        (numericAssignment (columnMap frame) assignment))
    (pointSatisfied :
      Satisfies
        (ConcreteNifsPiRlcPointRows.rows application profile frame)
        (numericAssignment (columnMap frame) assignment))
    (coordinate : Fin shape.rowVariables) :
    KPointEquality.decoded
        (numericAssignment (columnMap frame) assignment)
        (ConcreteNifsPiRlcPointRows.outputCoordinate
          application profile frame coordinate) =
      ConcreteNifsCarrierViews.pointCoordinate coordinate
        (Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.derive
          (ConcreteNifsParameters.context
            (keys
              Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected)
            running fresh proof).materialize
          proof.certificate).piRlcOutput.point := by
  simpa [Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.derive,
    Nightstream.SuperNeo.Folding.PiRLC.combinedOutput,
    ConcreteNifsCarrierViews.pointCoordinate, pointAt] using
    physical_coordinate_eq_derived
      application profile frame assignment running fresh proof
      constantWire decodedInputs operationalSatisfied pointSatisfied
      coordinate

/-- Every point-binding coordinate is forced to the corresponding coordinate
of the frozen verifier-derived PiRLC parent point. -/
theorem coordinate_eq_derived
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame :
      CallFrame (signature := signature Selected)
        (FamilyFor application) Call.nifsVerify
        (Refs.cons runningRef
          (Refs.cons freshRef (Refs.cons proofRef .nil))))
    (assignment : ColumnId → Field)
    (running :
      SelectedRunning shape publicRingColumns publicFits verifierRows)
    (fresh :
      SelectedFresh shape publicRingColumns publicFits verifierRows)
    (proof :
      SelectedProof shape TranscriptState publicRingColumns publicFits
        verifierRows)
    (output :
      SelectedRunning shape publicRingColumns publicFits verifierRows)
    (constantWire : assignment frame.one = 1)
    (decodedInputs :
      frame.operands.Decodes (FamilyFor application) assignment
        (.cons running (.cons fresh (.cons proof .nil))))
    (decodedOutput :
      frame.outputs.Decodes (FamilyFor application) assignment
        (.cons output .nil))
    (operationalSatisfied :
      Satisfies
        (ConcreteNifsOperationalOccurrence.rows application profile frame)
        (numericAssignment (columnMap frame) assignment))
    (pointSatisfied :
      Satisfies
        (ConcreteNifsPiRlcPointRows.rows application profile frame)
        (numericAssignment (columnMap frame) assignment))
    (coordinate : Fin shape.rowVariables) :
    pointAt output.parent.point coordinate =
      pointAt
        (Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.derive
          (ConcreteNifsParameters.context
            (keys
              Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected)
            running fresh proof).materialize
          proof.certificate).piCcs.fePoint.row
        coordinate := by
  let numeric := numericAssignment (columnMap frame) assignment
  have numericOne :
      numeric 0 = 1 :=
    ConcreteNifsOperationalOccurrenceSemantics.numericConstantWire
      application frame assignment constantWire
  have carriedEqual :=
    decoded_eq_of_rows numeric
      (ConcreteNifsPiRlcPointRows.transcriptCoordinate
        application profile frame coordinate)
      (ConcreteNifsPiRlcPointRows.outputCoordinate
        application profile frame coordinate)
      numericOne
      (coordinate_satisfied application profile frame numeric
        pointSatisfied coordinate)
  have outputDecoded :=
    ConcreteNifsSelectedCallFrame.output_decodes_of_frame_decodes
      (FamilyFor application) frame assignment output decodedOutput
  have outputExact :=
    ConcreteNifsCarrierFrame.outputK_decoded
      (FamilyFor application) frame
      (profile.runningViews.parentPoint coordinate)
      assignment output outputDecoded
  have transcriptPoint :=
    KSplitNcEndpoints.feRowPoint_eq_decoded
      (ConcreteNifsPiRlcPointRows.endpointInput application profile frame)
      numeric
  have transcriptCoordinateExact :
      KPointEquality.decoded numeric
          (ConcreteNifsPiRlcPointRows.transcriptCoordinate
            application profile frame coordinate) =
        pointAt
          (KSplitNcTranscriptPhases.decodedFePoint numeric
            (ConcreteNifsOperationalOccurrence.transcriptInput
              application profile frame)).row coordinate := by
    have exact :=
      congrArg (fun point => pointAt point coordinate) transcriptPoint
    simpa [ConcreteNifsPiRlcPointRows.transcriptCoordinate,
      ConcreteNifsPiRlcPointRows.endpointInput,
      KSplitNcEndpoints.decodedPointOf, pointAt] using exact
  have selectedPoint :=
    ConcreteNifsSelectedFePoint.selectedFePoint_eq
      application profile frame assignment running fresh proof
      constantWire decodedInputs operationalSatisfied
  have retargetUnchanged :
      KSplitNcTranscriptPhases.decodedFePoint numeric
          (ConcreteNifsOperationalOccurrence.transcriptInput
            application profile frame) =
        KSplitNcTranscriptPhases.decodedFePoint numeric
          (KSplitNcStaticInput.retargetTranscript proof.piCcsInput
            (ConcreteNifsOperationalOccurrence.transcriptInput
              application profile frame)) := by
    rfl
  have selectedRow :
      (KSplitNcTranscriptPhases.decodedFePoint numeric
        (ConcreteNifsOperationalOccurrence.transcriptInput
          application profile frame)).row =
        (Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.derive
          (ConcreteNifsParameters.context
            (keys
              Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected)
            running fresh proof).materialize
          proof.certificate).piCcs.fePoint.row := by
    exact congrArg (fun point => point.row)
      (retargetUnchanged.trans selectedPoint)
  calc
    pointAt output.parent.point coordinate =
        KPointEquality.decoded numeric
          (ConcreteNifsPiRlcPointRows.outputCoordinate
            application profile frame coordinate) := by
      simpa [ConcreteNifsPiRlcPointRows.outputCoordinate,
        ConcreteNifsCarrierViews.parentPointCoordinate, pointAt] using
        outputExact.symm
    _ = KPointEquality.decoded numeric
          (ConcreteNifsPiRlcPointRows.transcriptCoordinate
            application profile frame coordinate) := carriedEqual.symm
    _ = pointAt
          (KSplitNcTranscriptPhases.decodedFePoint numeric
            (ConcreteNifsOperationalOccurrence.transcriptInput
              application profile frame)).row coordinate :=
      transcriptCoordinateExact
    _ = pointAt
          (Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.derive
            (ConcreteNifsParameters.context
              (keys
                Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected)
              running fresh proof).materialize
            proof.certificate).piCcs.fePoint.row coordinate :=
      congrArg (fun point => pointAt point coordinate) selectedRow

/-- **Headline point refinement.** The complete decoded output-parent point is
the exact row point selected by the frozen verifier execution. -/
theorem point_eq_derived
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame :
      CallFrame (signature := signature Selected)
        (FamilyFor application) Call.nifsVerify
        (Refs.cons runningRef
          (Refs.cons freshRef (Refs.cons proofRef .nil))))
    (assignment : ColumnId → Field)
    (running :
      SelectedRunning shape publicRingColumns publicFits verifierRows)
    (fresh :
      SelectedFresh shape publicRingColumns publicFits verifierRows)
    (proof :
      SelectedProof shape TranscriptState publicRingColumns publicFits
        verifierRows)
    (output :
      SelectedRunning shape publicRingColumns publicFits verifierRows)
    (constantWire : assignment frame.one = 1)
    (decodedInputs :
      frame.operands.Decodes (FamilyFor application) assignment
        (.cons running (.cons fresh (.cons proof .nil))))
    (decodedOutput :
      frame.outputs.Decodes (FamilyFor application) assignment
        (.cons output .nil))
    (operationalSatisfied :
      Satisfies
        (ConcreteNifsOperationalOccurrence.rows application profile frame)
        (numericAssignment (columnMap frame) assignment))
    (pointSatisfied :
      Satisfies
        (ConcreteNifsPiRlcPointRows.rows application profile frame)
        (numericAssignment (columnMap frame) assignment)) :
    output.parent.point =
      (Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.derive
        (ConcreteNifsParameters.context
          (keys
            Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected)
          running fresh proof).materialize
        proof.certificate).piCcs.fePoint.row := by
  apply cubePoint_eq_of_coordinates
  intro coordinate
  exact coordinate_eq_derived application profile frame assignment
    running fresh proof output constantWire decodedInputs decodedOutput
    operationalSatisfied pointSatisfied coordinate

/-- Honest completeness for the point slice.  Since the slice allocates
nothing, the original assignment itself satisfies every row. -/
theorem rows_honest
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame :
      CallFrame (signature := signature Selected)
        (FamilyFor application) Call.nifsVerify
        (Refs.cons runningRef
          (Refs.cons freshRef (Refs.cons proofRef .nil))))
    (assignment : ColumnId → Field)
    (running :
      SelectedRunning shape publicRingColumns publicFits verifierRows)
    (fresh :
      SelectedFresh shape publicRingColumns publicFits verifierRows)
    (proof :
      SelectedProof shape TranscriptState publicRingColumns publicFits
        verifierRows)
    (output :
      SelectedRunning shape publicRingColumns publicFits verifierRows)
    (constantWire : assignment frame.one = 1)
    (decodedInputs :
      frame.operands.Decodes (FamilyFor application) assignment
        (.cons running (.cons fresh (.cons proof .nil))))
    (decodedOutput :
      frame.outputs.Decodes (FamilyFor application) assignment
        (.cons output .nil))
    (operationalSatisfied :
      Satisfies
        (ConcreteNifsOperationalOccurrence.rows application profile frame)
        (numericAssignment (columnMap frame) assignment))
    (outputExact :
      output.parent.point =
        (Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.derive
          (ConcreteNifsParameters.context
            (keys
              Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected)
            running fresh proof).materialize
          proof.certificate).piCcs.fePoint.row) :
    Satisfies
      (ConcreteNifsPiRlcPointRows.rows application profile frame)
      (numericAssignment (columnMap frame) assignment) := by
  let numeric := numericAssignment (columnMap frame) assignment
  have numericOne :
      numeric 0 = 1 :=
    ConcreteNifsOperationalOccurrenceSemantics.numericConstantWire
      application frame assignment constantWire
  have outputDecoded :=
    ConcreteNifsSelectedCallFrame.output_decodes_of_frame_decodes
      (FamilyFor application) frame assignment output decodedOutput
  have selectedPoint :=
    ConcreteNifsSelectedFePoint.selectedFePoint_eq
      application profile frame assignment running fresh proof
      constantWire decodedInputs operationalSatisfied
  have retargetUnchanged :
      KSplitNcTranscriptPhases.decodedFePoint numeric
          (ConcreteNifsOperationalOccurrence.transcriptInput
            application profile frame) =
        KSplitNcTranscriptPhases.decodedFePoint numeric
          (KSplitNcStaticInput.retargetTranscript proof.piCcsInput
            (ConcreteNifsOperationalOccurrence.transcriptInput
              application profile frame)) := by
    rfl
  have transcriptPoint :
      KSplitNcEndpoints.decodedPointOf
          (KSplitNcEndpoints.feRowPoint
            (ConcreteNifsPiRlcPointRows.endpointInput
              application profile frame))
          numeric =
        (KSplitNcTranscriptPhases.decodedFePoint numeric
          (ConcreteNifsOperationalOccurrence.transcriptInput
            application profile frame)).row :=
    KSplitNcEndpoints.feRowPoint_eq_decoded
      (ConcreteNifsPiRlcPointRows.endpointInput application profile frame)
      numeric
  intro row member
  rcases List.mem_flatMap.1 member with
    ⟨coordinate, coordinateMember, rowMember⟩
  have transcriptCoordinateExact :
      KPointEquality.decoded numeric
          (ConcreteNifsPiRlcPointRows.transcriptCoordinate
            application profile frame coordinate) =
        pointAt
          (KSplitNcTranscriptPhases.decodedFePoint numeric
            (ConcreteNifsOperationalOccurrence.transcriptInput
              application profile frame)).row coordinate := by
    have exact :=
      congrArg (fun point => pointAt point coordinate) transcriptPoint
    simpa [ConcreteNifsPiRlcPointRows.transcriptCoordinate,
      ConcreteNifsPiRlcPointRows.endpointInput,
      KSplitNcEndpoints.decodedPointOf, pointAt] using exact
  have outputCoordinateExact :
      KPointEquality.decoded numeric
          (ConcreteNifsPiRlcPointRows.outputCoordinate
            application profile frame coordinate) =
        pointAt output.parent.point coordinate := by
    simpa [ConcreteNifsPiRlcPointRows.outputCoordinate,
      ConcreteNifsCarrierViews.parentPointCoordinate, pointAt] using
      (ConcreteNifsCarrierFrame.outputK_decoded
        (FamilyFor application) frame
        (profile.runningViews.parentPoint coordinate)
        assignment output outputDecoded)
  have selectedRow :
      (KSplitNcTranscriptPhases.decodedFePoint numeric
        (ConcreteNifsOperationalOccurrence.transcriptInput
          application profile frame)).row =
        (Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.derive
          (ConcreteNifsParameters.context
            (keys
              Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected)
            running fresh proof).materialize
          proof.certificate).piCcs.fePoint.row :=
    congrArg (fun point => point.row) (retargetUnchanged.trans selectedPoint)
  have decodedEqual :
      KPointEquality.decoded numeric
          (ConcreteNifsPiRlcPointRows.transcriptCoordinate
            application profile frame coordinate) =
        KPointEquality.decoded numeric
          (ConcreteNifsPiRlcPointRows.outputCoordinate
            application profile frame coordinate) := by
    rw [transcriptCoordinateExact, outputCoordinateExact,
      congrArg (fun point => pointAt point coordinate) selectedRow,
      outputExact]
    rfl
  have pairEqual :=
    congrArg KConcreteBridge.ofConcrete decodedEqual
  rw [KPointEquality.ofConcrete_decoded,
    KPointEquality.ofConcrete_decoded] at pairEqual
  have lowEqual := congrArg KHorner.Pair.low pairEqual
  have highEqual := congrArg KHorner.Pair.high pairEqual
  exact KEquality.rows_complete numeric
    (ConcreteNifsPiRlcPointRows.transcriptCoordinate
      application profile frame coordinate)
    (ConcreteNifsPiRlcPointRows.outputCoordinate
      application profile frame coordinate)
    numericOne lowEqual highEqual row rowMember

end SelectedFrame

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsPiRlcPointSemantics
