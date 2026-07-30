import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsOutputSemantics
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsPiRlcActionBridge
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsPiRlcPointSemantics

/-!
Contract: derive the selected NIFS output codec from the physical output
coordinates forced by the Lean-owned row program.

This module closes the representation direction that cannot be assumed by a
sound `CallRecipe`: satisfied action, point, and child-materialization rows
determine every coordinate of the running codec, and the codec therefore
decodes to the unique frozen verifier result.

It imports no Rust or generated-row artifact and accepts no caller-selected
output value.
-/

set_option autoImplicit false
set_option maxRecDepth 100000
set_option maxHeartbeats 2000000

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsOutputDecode

open Nightstream.HyperNova.Construction2.Paper
open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.Goldilocks.NumericRowBridge
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.DirectCalls
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsCallFrame
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsGlobalColumnMap
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsParameters
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCarrierViews
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.Nifs
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc

private abbrev TranscriptState := Poseidon2Duplex.State

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

private def selectedResult
    (running :
      SelectedRunning shape publicRingColumns publicFits verifierRows)
    (fresh :
      SelectedFresh shape publicRingColumns publicFits verifierRows)
    (proof :
      SelectedProof shape TranscriptState publicRingColumns publicFits
        verifierRows) :
    FixedActive.FoldResult
      shape publicRingColumns publicFits verifierRows :=
  FixedActive.resultOf
    (ConcreteNifsParameters.context
      (keys
        Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected)
      running fresh proof).materialize
    proof.certificate

private def selectedOutput
    (running :
      SelectedRunning shape publicRingColumns publicFits verifierRows)
    (fresh :
      SelectedFresh shape publicRingColumns publicFits verifierRows)
    (proof :
      SelectedProof shape TranscriptState publicRingColumns publicFits
        verifierRows) :
    SelectedRunning shape publicRingColumns publicFits verifierRows :=
  SelectedRunning.ofResult (selectedResult (keys := keys) running fresh proof)

/-- A base-field output view reads the same physical value as its canonical
singleton carried expression, without assuming that the bundle decodes. -/
private theorem outputF_getD_eq_carried
    (application : Poseidon23ApplicationProfile Selected)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (assignment : ColumnId → Field)
    {value :
      SelectedRunning shape publicRingColumns publicFits verifierRows → Field}
    (view :
      PaperNifsCodecProjection.FView
        ((FamilyFor application).codecFor (.data .running)) value) :
    ((unaryOutput frame.outputs).values assignment).getD view.index.val 0 =
      residue
        (Nightstream.Implementation.R1CS.lcEval
          (NumericRowBridge.numericAssignment (columnMap frame) assignment)
          (ConcreteNifsCarrierFrame.outputFLocation
            (FamilyFor application) frame view).carried) := by
  calc
    ((unaryOutput frame.outputs).values assignment).getD
          view.index.val 0 =
        (view.column (unaryOutput frame.outputs)
          (ConcreteNifsCarrierFrame.output_widthsAgree
            (FamilyFor application) frame)).value assignment :=
      view.bundle_getD_eq_value
        (unaryOutput frame.outputs)
        (ConcreteNifsCarrierFrame.output_widthsAgree
          (FamilyFor application) frame)
        assignment
    _ = residue
          (Nightstream.Implementation.R1CS.lcEval
            (NumericRowBridge.numericAssignment
              (columnMap frame) assignment)
            (ConcreteNifsCarrierFrame.outputFLocation
              (FamilyFor application) frame view).carried) :=
      (ConcreteNifsCarrierFrame.outputFLocation
        (FamilyFor application) frame view).carried_value_eq assignment |>.symm

/-- One component of a quadratic-extension output view is the corresponding
component of the physical carried pair, again without a decoder premise. -/
private theorem outputK_getD_eq_carried
    (application : Poseidon23ApplicationProfile Selected)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (assignment : ColumnId → Field)
    {value :
      SelectedRunning shape publicRingColumns publicFits verifierRows → K}
    (view :
      PaperNifsCodecProjection.KView
        ((FamilyFor application).codecFor (.data .running)) value)
    (component : KComponent) :
    ((unaryOutput frame.outputs).values assignment).getD
        ((component.view view).index.val) 0 =
      component.value
        (KPointEquality.decoded
          (NumericRowBridge.numericAssignment (columnMap frame) assignment)
          (ConcreteNifsCarrierFrame.outputKLocation
            (FamilyFor application) frame view).carried) := by
  cases component with
  | c0 =>
      calc
        ((unaryOutput frame.outputs).values assignment).getD
              ((KComponent.c0.view view).index.val) 0 =
            ((KComponent.c0.view view).column
              (unaryOutput frame.outputs)
              (ConcreteNifsCarrierFrame.output_widthsAgree
                (FamilyFor application) frame)).value assignment :=
          (KComponent.c0.view view).bundle_getD_eq_value
            (unaryOutput frame.outputs)
            (ConcreteNifsCarrierFrame.output_widthsAgree
              (FamilyFor application) frame)
            assignment
        _ = ((view.columns (unaryOutput frame.outputs)
              (ConcreteNifsCarrierFrame.output_widthsAgree
                (FamilyFor application) frame)).value assignment).c0 := rfl
        _ = (KPointEquality.decoded
              (NumericRowBridge.numericAssignment
                (columnMap frame) assignment)
              (ConcreteNifsCarrierFrame.outputKLocation
                (FamilyFor application) frame view).carried).c0 :=
          congrArg K.c0
            ((ConcreteNifsCarrierFrame.outputKLocation
              (FamilyFor application) frame view).decodeCarried_eq
                assignment).symm
  | c1 =>
      calc
        ((unaryOutput frame.outputs).values assignment).getD
              ((KComponent.c1.view view).index.val) 0 =
            ((KComponent.c1.view view).column
              (unaryOutput frame.outputs)
              (ConcreteNifsCarrierFrame.output_widthsAgree
                (FamilyFor application) frame)).value assignment :=
          (KComponent.c1.view view).bundle_getD_eq_value
            (unaryOutput frame.outputs)
            (ConcreteNifsCarrierFrame.output_widthsAgree
              (FamilyFor application) frame)
            assignment
        _ = ((view.columns (unaryOutput frame.outputs)
              (ConcreteNifsCarrierFrame.output_widthsAgree
                (FamilyFor application) frame)).value assignment).c1 := rfl
        _ = (KPointEquality.decoded
              (NumericRowBridge.numericAssignment
                (columnMap frame) assignment)
              (ConcreteNifsCarrierFrame.outputKLocation
                (FamilyFor application) frame view).carried).c1 :=
          congrArg K.c1
            ((ConcreteNifsCarrierFrame.outputKLocation
              (FamilyFor application) frame view).decodeCarried_eq
                assignment).symm

/-- **Headline output-decoder refinement.** The physical parent/action,
point-binding, and child-materialization rows force the complete output bundle
to decode to the unique frozen fixed-active result.  No semantic output is a
premise. -/
theorem output_decodes_selectedResult_of_rows
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (assignment : ColumnId → Field)
    (running :
      SelectedRunning shape publicRingColumns publicFits verifierRows)
    (fresh :
      SelectedFresh shape publicRingColumns publicFits verifierRows)
    (proof :
      SelectedProof shape TranscriptState publicRingColumns publicFits
        verifierRows)
    (productBase : Nat)
    (constantWire : assignment frame.one = 1)
    (decodedInputs :
      frame.operands.Decodes (FamilyFor application) assignment
        (.cons running (.cons fresh (.cons proof .nil))))
    (operationalSatisfied :
      Satisfies
        (ConcreteNifsOperationalOccurrence.rows application profile frame)
        (NumericRowBridge.numericAssignment
          (columnMap frame) assignment))
    (actionSatisfied :
      RawSatisfies
        (ConcreteNifsPiRlcActionRows.rows
          application profile frame productBase)
        assignment)
    (pointSatisfied :
      Satisfies
        (ConcreteNifsPiRlcPointRows.rows application profile frame)
        (NumericRowBridge.numericAssignment
          (columnMap frame) assignment))
    (outputSatisfied :
      Satisfies
        (ConcreteNifsOutputRows.rows application profile frame)
        (NumericRowBridge.numericAssignment
          (columnMap frame) assignment)) :
    frame.outputs.Decodes (FamilyFor application) assignment
      (.cons (selectedOutput (keys := keys) running fresh proof) .nil) := by
  let numeric :=
    NumericRowBridge.numericAssignment (columnMap frame) assignment
  let result := selectedResult (keys := keys) running fresh proof
  let output := SelectedRunning.ofResult result
  have actionEquations :=
    ConcreteNifsPiRlcActionSemantics.physical_equations_of_rows
      application profile frame assignment running fresh proof productBase
      constantWire decodedInputs actionSatisfied
  have childEquations :=
    ConcreteNifsOutputSemantics.physical_child_equations_of_rows
      application profile frame assignment running fresh proof constantWire
      decodedInputs outputSatisfied
  have valuesExact :
      ∀ coordinate :
        RunningCoordinate shape publicRingColumns verifierRows,
        ((unaryOutput frame.outputs).values assignment).getD
            ((coordinate.view profile.runningViews).index.val) 0 =
          coordinate.value output := by
    intro coordinate
    cases coordinate with
    | parentCommitment row lane =>
        have carried :=
          outputF_getD_eq_carried application frame assignment
            (profile.runningViews.parentCommitment row lane)
        have carriedAction :
            ((unaryOutput frame.outputs).values assignment).getD
                ((profile.runningViews.parentCommitment row lane).index.val)
                0 =
              Phi81RingAction.decoded assignment
                (ConcreteNifsPiRlcActionRows.commitmentOutput
                  application profile frame row) lane := by
          calc
            _ = residue
                  (Nightstream.Implementation.R1CS.lcEval numeric
                    (ConcreteNifsCarrierFrame.outputFLocation
                      (FamilyFor application) frame
                      (profile.runningViews.parentCommitment
                        row lane)).carried) := carried
            _ = Phi81RingAction.decoded assignment
                  (ConcreteNifsPiRlcActionRows.commitmentOutput
                    application profile frame row) lane := by
              symm
              exact
                terms_eval_eq_residue_lcEval
                  (columnMap frame) assignment
                  (ConcreteNifsCarrierFrame.outputFLocation
                    (FamilyFor application) frame
                    (profile.runningViews.parentCommitment row lane)).carried
        exact carriedAction.trans
          (by
            simpa [output, result, selectedResult, SelectedRunning.ofResult,
              FixedActive.resultOf, Result.resultOf,
              RunningCoordinate.value, parentCommitmentCoordinate] using
              ConcreteNifsPiRlcActionBridge.physical_commitment_eq_derived
                (keys := keys) application profile frame assignment
                running fresh proof actionEquations row lane)
    | childCommitment child row lane =>
        exact
          (outputF_getD_eq_carried application frame assignment
              (profile.runningViews.childCommitment child row lane)).trans
            (by
              simpa [output, result, selectedResult, SelectedRunning.ofResult,
                FixedActive.resultOf, Result.resultOf,
                ConcretePhi81.outputChildren, Execution.piDecChildren,
                PiDecChildPayload.materialize,
                RunningCoordinate.value, childCommitmentCoordinate,
                ConcreteNifsOutputRows.outputChildCommitment] using
                childEquations.commitment child row lane)
    | parentPublic column =>
        let relationShape :=
          PiCCS.SplitNc.Verifier.RelationShape
            shape publicRingColumns publicFits
        let block :=
          PiRLCAlgebra.PublicInput.publicBlockIndex relationShape column
        let lane :=
          PiRLCAlgebra.PublicInput.publicLaneIndex
            (shape := relationShape) column
        have coordinateEq :
            ConcreteNifsPiRlcActionRows.publicCoordinate block lane =
              column := by
          apply Fin.ext
          dsimp only [block, lane,
            PiRLCAlgebra.PublicInput.publicBlockIndex,
            PiRLCAlgebra.PublicInput.publicLaneIndex,
            ConcreteNifsPiRlcActionRows.publicCoordinate]
          rw [Nat.mul_comm]
          exact Nat.div_add_mod column.val ringDegree
        rw [← coordinateEq]
        have carried :=
          outputF_getD_eq_carried application frame assignment
            (profile.runningViews.parentPublic
              (ConcreteNifsPiRlcActionRows.publicCoordinate block lane))
        have carriedAction :
            ((unaryOutput frame.outputs).values assignment).getD
                ((profile.runningViews.parentPublic
                  (ConcreteNifsPiRlcActionRows.publicCoordinate
                    block lane)).index.val) 0 =
              Phi81RingAction.decoded assignment
                (ConcreteNifsPiRlcActionRows.publicOutput
                  application profile frame block) lane := by
          calc
            _ = residue
                  (Nightstream.Implementation.R1CS.lcEval numeric
                    (ConcreteNifsCarrierFrame.outputFLocation
                      (FamilyFor application) frame
                      (profile.runningViews.parentPublic
                        (ConcreteNifsPiRlcActionRows.publicCoordinate
                          block lane))).carried) := carried
            _ = Phi81RingAction.decoded assignment
                  (ConcreteNifsPiRlcActionRows.publicOutput
                    application profile frame block) lane := by
              symm
              exact
                terms_eval_eq_residue_lcEval
                  (columnMap frame) assignment
                  (ConcreteNifsCarrierFrame.outputFLocation
                    (FamilyFor application) frame
                    (profile.runningViews.parentPublic
                      (ConcreteNifsPiRlcActionRows.publicCoordinate
                        block lane))).carried
        exact carriedAction.trans
          (by
            simpa [output, result, selectedResult, SelectedRunning.ofResult,
              FixedActive.resultOf, Result.resultOf,
              RunningCoordinate.value, parentPublicCoordinate] using
              ConcreteNifsPiRlcActionBridge.physical_public_eq_derived
                (keys := keys) application profile frame assignment
                running fresh proof actionEquations block lane)
    | childPublic child column =>
        exact
          (outputF_getD_eq_carried application frame assignment
              (profile.runningViews.childPublic child column)).trans
            (by
              simpa [output, result, selectedResult, SelectedRunning.ofResult,
                FixedActive.resultOf, Result.resultOf,
                ConcretePhi81.outputChildren, Execution.piDecChildren,
                PiDecChildPayload.materialize,
                RunningCoordinate.value, childPublicCoordinate,
                ConcreteNifsOutputRows.outputChildPublic] using
                childEquations.publicInput child column)
    | parentPoint coordinate component =>
        have physicalPoint :=
          ConcreteNifsPiRlcPointSemantics.physical_coordinate_eq_parent
            application profile frame assignment running fresh proof
            constantWire decodedInputs operationalSatisfied pointSatisfied
            coordinate
        calc
          ((unaryOutput frame.outputs).values assignment).getD
                (((RunningCoordinate.parentPoint
                  coordinate component).view
                    profile.runningViews).index.val) 0 =
              component.value
                (KPointEquality.decoded numeric
                  (ConcreteNifsPiRlcPointRows.outputCoordinate
                    application profile frame coordinate)) := by
            simpa [RunningCoordinate.view,
              ConcreteNifsPiRlcPointRows.outputCoordinate] using
              outputK_getD_eq_carried application frame assignment
                (profile.runningViews.parentPoint coordinate) component
          _ = component.value
                (ConcreteNifsCarrierViews.pointCoordinate coordinate
                  (ConcretePhi81.derive
                    (ConcreteNifsParameters.context
                      (keys
                        Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected)
                      running fresh proof).materialize
                    proof.certificate).piRlcOutput.point) :=
            congrArg component.value physicalPoint
          _ = (RunningCoordinate.parentPoint
                coordinate component).value output := by
            simp [output, result, selectedResult, SelectedRunning.ofResult,
              FixedActive.resultOf, Result.resultOf,
              RunningCoordinate.value, parentPointCoordinate]
    | childPoint child coordinate component =>
        have physicalPoint :=
          ConcreteNifsPiRlcPointSemantics.physical_coordinate_eq_parent
            application profile frame assignment running fresh proof
            constantWire decodedInputs operationalSatisfied pointSatisfied
            coordinate
        calc
          ((unaryOutput frame.outputs).values assignment).getD
                (((RunningCoordinate.childPoint
                  child coordinate component).view
                    profile.runningViews).index.val) 0 =
              component.value
                (KPointEquality.decoded numeric
                  (ConcreteNifsOutputRows.outputChildPoint
                    application profile frame child coordinate)) := by
            simpa [RunningCoordinate.view,
              ConcreteNifsOutputRows.outputChildPoint] using
              outputK_getD_eq_carried application frame assignment
                (profile.runningViews.childPoint child coordinate) component
          _ = component.value
                (KPointEquality.decoded numeric
                  (ConcreteNifsOutputRows.outputParentPoint
                    application profile frame coordinate)) :=
            congrArg component.value
              (childEquations.point child coordinate)
          _ = component.value
                (ConcreteNifsCarrierViews.pointCoordinate coordinate
                  (ConcretePhi81.derive
                    (ConcreteNifsParameters.context
                      (keys
                        Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected)
                      running fresh proof).materialize
                    proof.certificate).piRlcOutput.point) := by
            exact congrArg component.value
              (by
                simpa [ConcreteNifsOutputRows.outputParentPoint,
                  ConcreteNifsPiRlcPointRows.outputCoordinate] using
                    physicalPoint)
          _ = (RunningCoordinate.childPoint
                child coordinate component).value output := by
            simp [output, result, selectedResult, SelectedRunning.ofResult,
              FixedActive.resultOf, Result.resultOf,
              ConcretePhi81.outputChildren, Execution.piDecChildren,
              PiDecChildPayload.materialize, RunningCoordinate.value,
              childPointCoordinate]
    | parentEvaluation matrix lane component =>
        cases component with
        | c0 =>
            have carried :=
              outputF_getD_eq_carried application frame assignment
                (KComponent.c0.view
                  (profile.runningViews.parentEvaluation matrix lane))
            have carriedAction :
                ((unaryOutput frame.outputs).values assignment).getD
                    (((RunningCoordinate.parentEvaluation
                      matrix lane KComponent.c0).view
                        profile.runningViews).index.val) 0 =
                  Phi81RingAction.decoded assignment
                    (ConcreteNifsPiRlcActionRows.evaluationOutputLow
                      application profile frame matrix) lane := by
              calc
                _ = residue
                      (Nightstream.Implementation.R1CS.lcEval numeric
                        (ConcreteNifsCarrierFrame.outputFLocation
                          (FamilyFor application) frame
                          (KComponent.c0.view
                            (profile.runningViews.parentEvaluation
                              matrix lane))).carried) := by
                    simpa [RunningCoordinate.view] using carried
                _ = Phi81RingAction.decoded assignment
                      (ConcreteNifsPiRlcActionRows.evaluationOutputLow
                        application profile frame matrix) lane := by
                  symm
                  exact
                    terms_eval_eq_residue_lcEval
                      (columnMap frame) assignment
                      (ConcreteNifsCarrierFrame.outputFLocation
                        (FamilyFor application) frame
                        (KComponent.c0.view
                          (profile.runningViews.parentEvaluation
                            matrix lane))).carried
            calc
              ((unaryOutput frame.outputs).values assignment).getD
                    (((RunningCoordinate.parentEvaluation
                      matrix lane KComponent.c0).view
                        profile.runningViews).index.val) 0 =
                  Phi81RingAction.decoded assignment
                    (ConcreteNifsPiRlcActionRows.evaluationOutputLow
                      application profile frame matrix) lane :=
                carriedAction
              _ = (RunningCoordinate.parentEvaluation
                    matrix lane KComponent.c0).value output := by
                simpa [output, result, selectedResult,
                  SelectedRunning.ofResult, FixedActive.resultOf,
                  Result.resultOf, RunningCoordinate.value,
                  parentEvaluationCoordinate,
                  ConcreteNifsPiRlcActionRows.evaluationOutputLow,
                  Phi81RingAction.decoded] using
                    ConcreteNifsPiRlcActionBridge.physical_evaluation_low_eq_derived
                      (keys := keys) application profile frame assignment
                      running fresh proof actionEquations matrix lane
        | c1 =>
            have carried :=
              outputF_getD_eq_carried application frame assignment
                (KComponent.c1.view
                  (profile.runningViews.parentEvaluation matrix lane))
            have carriedAction :
                ((unaryOutput frame.outputs).values assignment).getD
                    (((RunningCoordinate.parentEvaluation
                      matrix lane KComponent.c1).view
                        profile.runningViews).index.val) 0 =
                  Phi81RingAction.decoded assignment
                    (ConcreteNifsPiRlcActionRows.evaluationOutputHigh
                      application profile frame matrix) lane := by
              calc
                _ = residue
                      (Nightstream.Implementation.R1CS.lcEval numeric
                        (ConcreteNifsCarrierFrame.outputFLocation
                          (FamilyFor application) frame
                          (KComponent.c1.view
                            (profile.runningViews.parentEvaluation
                              matrix lane))).carried) := by
                    simpa [RunningCoordinate.view] using carried
                _ = Phi81RingAction.decoded assignment
                      (ConcreteNifsPiRlcActionRows.evaluationOutputHigh
                        application profile frame matrix) lane := by
                  symm
                  exact
                    terms_eval_eq_residue_lcEval
                      (columnMap frame) assignment
                      (ConcreteNifsCarrierFrame.outputFLocation
                        (FamilyFor application) frame
                        (KComponent.c1.view
                          (profile.runningViews.parentEvaluation
                            matrix lane))).carried
            calc
              ((unaryOutput frame.outputs).values assignment).getD
                    (((RunningCoordinate.parentEvaluation
                      matrix lane KComponent.c1).view
                        profile.runningViews).index.val) 0 =
                  Phi81RingAction.decoded assignment
                    (ConcreteNifsPiRlcActionRows.evaluationOutputHigh
                      application profile frame matrix) lane :=
                carriedAction
              _ = (RunningCoordinate.parentEvaluation
                    matrix lane KComponent.c1).value output := by
                simpa [output, result, selectedResult,
                  SelectedRunning.ofResult, FixedActive.resultOf,
                  Result.resultOf, RunningCoordinate.value,
                  parentEvaluationCoordinate,
                  ConcreteNifsPiRlcActionRows.evaluationOutputHigh,
                  Phi81RingAction.decoded] using
                    ConcreteNifsPiRlcActionBridge.physical_evaluation_high_eq_derived
                      (keys := keys) application profile frame assignment
                      running fresh proof actionEquations matrix lane
    | childEvaluation child matrix lane component =>
        exact
          (outputK_getD_eq_carried application frame assignment
              (profile.runningViews.childEvaluation
                child matrix lane) component).trans
            (by
              have exact :=
                congrArg component.value
                  (childEquations.evaluations child matrix lane)
              simpa [output, result, selectedResult, SelectedRunning.ofResult,
                FixedActive.resultOf, Result.resultOf,
                ConcretePhi81.outputChildren, Execution.piDecChildren,
                PiDecChildPayload.materialize,
                RunningCoordinate.value, childEvaluationCoordinate,
                ConcreteNifsOutputRows.outputChildEvaluation] using exact)
  have outputLength :
      ((unaryOutput frame.outputs).values assignment).length =
        ((FamilyFor application).codecFor (.data .running)).width := by
    rw [ColumnBundle.values_length]
    exact
      (ConcreteNifsCarrierFrame.output_widthsAgree
        (FamilyFor application) frame).symm
  have encoded :
      (unaryOutput frame.outputs).values assignment =
        ((FamilyFor application).codecFor
          (.data .running)).encode output :=
    profile.runningCoverage.coordinates_eq_encode
      ((unaryOutput frame.outputs).values assignment)
      output outputLength valuesExact
  have proofDecoded :=
    ConcreteNifsSelectedCallFrame.proof_decodes_of_frame_decodes
      (FamilyFor application) frame assignment running fresh proof
      decodedInputs
  have proofAdmissible :
      ((FamilyFor application).codecFor (.data .nifsProof)).Admissible proof :=
    Codec.admissible_of_decode
      ((FamilyFor application).codecFor (.data .nifsProof)) proofDecoded
  have parentEvaluationsSize :
      result.parent.evaluations.size = shape.matrixCount := by
    simp [result, selectedResult, FixedActive.resultOf, Result.resultOf,
      ConcretePhi81.derive, PiRLC.combinedOutput,
      ConcretePhi81.rlcAlgebra, PiRLCAlgebra.Algebra.concrete,
      Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism.PiRLCFinite.combineEvaluations]
    rfl
  have childEvaluationsSize :
      ∀ child,
        (result.children child).evaluations.size = shape.matrixCount := by
    intro child
    simpa [result, selectedResult, FixedActive.resultOf, Result.resultOf,
      ConcretePhi81.outputChildren, Execution.piDecChildren,
      PiDecChildPayload.materialize] using
        profile.payloadViews.evaluationsSize proof proofAdmissible child
  have decoded :
      (unaryOutput frame.outputs).Decodes
          (FamilyFor application) (.data .running) assignment output := by
    unfold ColumnBundle.Decodes
    rw [encoded]
    exact
      ((FamilyFor application).codecFor
        (.data .running)).decode_encode output
        (by
          exact profile.runningCoverage.resultAdmissible result
            parentEvaluationsSize childEvaluationsSize)
  apply
    (unaryOutput_decodes_iff
      (FamilyFor application) assignment frame.outputs output).2
  simpa [output, result, selectedOutput, selectedResult] using decoded

end SelectedFrame

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsOutputDecode
