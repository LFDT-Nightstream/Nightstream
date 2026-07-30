import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsOperationalOccurrence

/-!
Contract: prove that the frame-static operational Split-NC occurrence reads
exactly the values decoded from the selected `nifsVerify` call frame.

No equation, challenge, certificate, acceptance result, or paper event is a
premise.  The only semantic input is one successful whole-frame decode.
-/

set_option autoImplicit false
set_option maxRecDepth 100000
set_option maxHeartbeats 1200000

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsOperationalOccurrenceSemantics

open Nightstream.HyperNova.Construction2.Paper
open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.Goldilocks.NumericRowBridge
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.DirectCalls
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsCallFrame
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsCallColumnMap
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsGlobalColumnMap
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsCodecProjection
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsParameters
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsOperationalProfile
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsOperationalOccurrence
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.Implementation.R1CS.Canonical.KConcreteFixedPhaseBridge
open Nightstream.Implementation.R1CS.Canonical.SymbolicDuplexSemantics
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier

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

private abbrev FamilyFor (application : Poseidon23ApplicationProfile Selected) :=
  application.family Selected

private theorem duplexState_eq
    (left right : Poseidon2Duplex.State)
    (lanes : left.lanes = right.lanes)
    (absorbed : left.absorbed = right.absorbed) :
    left = right := by
  cases left
  cases right
  simp only at lanes absorbed
  cases lanes
  cases absorbed
  rfl

theorem numericConstantWire
    (application : Poseidon23ApplicationProfile Selected)
    {context : Schema (typeSystem Selected)}
    {runningRef :
      Ref (typeSystem Selected) context (.data .running)}
    {freshRef :
      Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef :
      Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame :
      CallFrame (signature := signature Selected)
        (FamilyFor application) Call.nifsVerify
        (Refs.cons runningRef
          (Refs.cons freshRef (Refs.cons proofRef .nil))))
    (assignment : ColumnId → Field)
    (constantWire : assignment frame.one = 1) :
    numericAssignment (columnMap frame) assignment 0 = 1 := by
  change (assignment frame.one).val = 1
  exact congrArg Fin.val constantWire

private theorem locatedField_eval
    {α : Type}
    {codec : Codec α}
    {value : α → Field}
    (view : FView codec value)
    {layout : Layout}
    (bundle : ColumnBundle layout)
    (widthsAgree : codec.width = layout.owners.length)
    {columnMap : Nat → ColumnId}
    (location :
      FLocation columnMap (view.column bundle widthsAgree))
    (assignment : ColumnId → Field)
    (input : α)
    (decoded : codec.decode (bundle.values assignment) = some input) :
    lcEval (numericAssignment columnMap assignment) location.carried =
      (value input).val := by
  have physical := location.carried_value_eq assignment
  have semantic :=
    view.value_eq_of_decodes bundle widthsAgree assignment input decoded
  have equal :
      residue
          (lcEval (numericAssignment columnMap assignment)
            location.carried) =
        value input :=
    physical.trans semantic
  have representatives := congrArg Fin.val equal
  have below :
      lcEval (numericAssignment columnMap assignment) location.carried <
        Nightstream.SuperNeo.Concrete.goldilocksModulus := by
    simpa [Nightstream.Implementation.R1CS.goldilocksP,
      Nightstream.SuperNeo.Concrete.goldilocksModulus] using
        (lcEval_lt (numericAssignment columnMap assignment)
          location.carried)
  simpa only [residue, Nat.mod_eq_of_lt below] using representatives

/-- Every serialized field expression evaluates to the exact semantic word
selected by the successful whole-frame decode. -/
theorem sourceExpression_eval
    (application : Poseidon23ApplicationProfile Selected)
    {context : Schema (typeSystem Selected)}
    {runningRef :
      Ref (typeSystem Selected) context (.data .running)}
    {freshRef :
      Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef :
      Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame :
      CallFrame (signature := signature Selected)
        (FamilyFor application) Call.nifsVerify
        (Refs.cons runningRef
          (Refs.cons freshRef (Refs.cons proofRef .nil))))
    (source :
      FieldSource
        (shape := shape) (publicRingColumns := publicRingColumns)
        (verifierRows := verifierRows) (publicFits := publicFits)
        ((FamilyFor application).codecFor (.data .running))
        ((FamilyFor application).codecFor (.data .fresh))
        ((FamilyFor application).codecFor (.data .nifsProof)))
    (assignment : ColumnId → Field)
    (running :
      SelectedRunning shape publicRingColumns publicFits verifierRows)
    (fresh :
      SelectedFresh shape publicRingColumns publicFits verifierRows)
    (proof :
      SelectedProof shape TranscriptState publicRingColumns publicFits
        verifierRows)
    (constantWire : assignment frame.one = 1)
    (decoded :
      frame.operands.Decodes (FamilyFor application) assignment
        (.cons running (.cons fresh (.cons proof .nil)))) :
    lcEval
        (numericAssignment (columnMap frame) assignment)
        (sourceExpression (FamilyFor application) frame source) =
      (source.value running fresh proof).val := by
  cases source with
  | constant value =>
      simp only [sourceExpression, FieldSource.value]
      rw [KSplitNcTranscriptSemantics.word_eval
        _ (numericConstantWire application frame assignment constantWire)]
      exact Nat.mod_eq_of_lt (by
        simpa [Nightstream.Implementation.R1CS.goldilocksP,
          Nightstream.SuperNeo.Concrete.goldilocksModulus] using value.isLt)
  | running value view =>
      have runningDecoded :=
        ConcreteNifsSelectedCallFrame.running_decodes_of_frame_decodes
          (FamilyFor application) frame assignment running fresh proof decoded
      exact locatedField_eval view (runningOperand frame.operands)
        (running_widthsAgree frame)
        (fLocation frame
          (view.column (runningOperand frame.operands)
            (running_widthsAgree frame))
          (runningOperand_mem frame
            (view.column_mem (runningOperand frame.operands)
              (running_widthsAgree frame))))
        assignment running runningDecoded
  | fresh value view =>
      have freshDecoded :=
        ConcreteNifsSelectedCallFrame.fresh_decodes_of_frame_decodes
          (FamilyFor application) frame assignment running fresh proof decoded
      exact locatedField_eval view (freshOperand frame.operands)
        (fresh_widthsAgree frame)
        (fLocation frame
          (view.column (freshOperand frame.operands)
            (fresh_widthsAgree frame))
          (freshOperand_mem frame
            (view.column_mem (freshOperand frame.operands)
              (fresh_widthsAgree frame))))
        assignment fresh freshDecoded
  | proof value view =>
      have proofDecoded :=
        ConcreteNifsSelectedCallFrame.proof_decodes_of_frame_decodes
          (FamilyFor application) frame assignment running fresh proof decoded
      exact locatedField_eval view (proofOperand frame.operands)
        (proof_widthsAgree frame)
        (proofFieldLocation (FamilyFor application) frame view)
        assignment proof proofDecoded

/-- Every quadratic-extension proof projection evaluates to the exact
semantic component selected by the successful whole-proof decode. -/
theorem proofColumns_value
    (application : Poseidon23ApplicationProfile Selected)
    {context : Schema (typeSystem Selected)}
    {runningRef :
      Ref (typeSystem Selected) context (.data .running)}
    {freshRef :
      Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef :
      Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame :
      CallFrame (signature := signature Selected)
        (FamilyFor application) Call.nifsVerify
        (Refs.cons runningRef
          (Refs.cons freshRef (Refs.cons proofRef .nil))))
    {value :
      SelectedProof shape TranscriptState publicRingColumns publicFits
          verifierRows →
        Nightstream.SuperNeo.Concrete.K}
    (view :
      KView
        ((FamilyFor application).codecFor (.data .nifsProof)) value)
    (assignment : ColumnId → Field)
    (proof :
      SelectedProof shape TranscriptState publicRingColumns publicFits
        verifierRows)
    (decoded :
      (proofOperand frame.operands).Decodes (FamilyFor application)
        (.data .nifsProof) assignment proof) :
    ofProjection
        ((proofColumns (FamilyFor application) frame view).value
          (numericAssignment (columnMap frame) assignment)) =
      value proof := by
  calc
    ofProjection
          ((proofColumns (FamilyFor application) frame view).value
            (numericAssignment (columnMap frame) assignment)) =
        (view.columns (proofOperand frame.operands)
          (proof_widthsAgree frame)).value assignment := by
      exact
        (ConcreteNifsOperationalFrame.proofLocation
          (FamilyFor application) frame view).numeric_value_eq assignment
    _ = value proof :=
      view.value_eq_of_bundle_decodes
        (FamilyFor application) (.data .nifsProof)
        (proofOperand frame.operands) (proof_widthsAgree frame)
        assignment proof decoded

private theorem ofFn_getD_eq_self
    {α : Type} {count : Nat}
    (values : List α) (default : α)
    (lengthEq : values.length = count) :
    (List.ofFn fun index : Fin count =>
      values.getD index.val default) = values := by
  apply List.ext_get
  · simp [lengthEq]
  · intro index leftLt rightLt
    simp only [List.get_eq_getElem, List.getElem_ofFn]
    rw [List.getD_eq_getElem?_getD,
      List.getElem?_eq_getElem rightLt]
    rfl

/-- A complete fixed-width family of proof views reconstructs the exact
semantic coefficient list, in order and without an independently supplied
message. -/
theorem proofCoefficientList_eq
    (application : Poseidon23ApplicationProfile Selected)
    {context : Schema (typeSystem Selected)}
    {runningRef :
      Ref (typeSystem Selected) context (.data .running)}
    {freshRef :
      Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef :
      Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame :
      CallFrame (signature := signature Selected)
        (FamilyFor application) Call.nifsVerify
        (Refs.cons runningRef
          (Refs.cons freshRef (Refs.cons proofRef .nil))))
    {count : Nat}
    (coefficients :
      SelectedProof shape TranscriptState publicRingColumns publicFits
          verifierRows →
        List Nightstream.SuperNeo.Concrete.K)
    (views :
      ∀ slot : Fin count,
        KView
          ((FamilyFor application).codecFor (.data .nifsProof))
          (fun proof => (coefficients proof).getD slot.val K.zero))
    (assignment : ColumnId → Field)
    (proof :
      SelectedProof shape TranscriptState publicRingColumns publicFits
        verifierRows)
    (decoded :
      (proofOperand frame.operands).Decodes (FamilyFor application)
        (.data .nifsProof) assignment proof)
    (lengthEq : (coefficients proof).length = count) :
    (List.ofFn fun slot =>
      ofProjection
        ((proofColumns (FamilyFor application) frame
          (views slot)).value
            (numericAssignment (columnMap frame) assignment))) =
      coefficients proof := by
  calc
    (List.ofFn fun slot =>
        ofProjection
          ((proofColumns (FamilyFor application) frame
            (views slot)).value
              (numericAssignment (columnMap frame) assignment))) =
        (List.ofFn fun slot =>
          (coefficients proof).getD slot.val K.zero) := by
      apply congrArg List.ofFn
      funext slot
      exact proofColumns_value application frame (views slot)
        assignment proof decoded
    _ = coefficients proof :=
      ofFn_getD_eq_self (coefficients proof) K.zero lengthEq

private theorem fixedPolynomial_eq_of_coefficients_eq
    {α : Type} {degree : Nat}
    {left right :
      Nightstream.SuperNeo.SumCheck.Finite.FixedPolynomial α degree}
    (equal : left.coefficients = right.coefficients) :
    left = right := by
  cases left with
  | mk leftCoefficients leftLength =>
      cases right with
      | mk rightCoefficients rightLength =>
          simp only at equal
          subst rightCoefficients
          rfl

private theorem fixedPolynomial_heq_of_degree_eq
    {α : Type} {leftDegree rightDegree : Nat}
    (left :
      Nightstream.SuperNeo.SumCheck.Finite.FixedPolynomial α leftDegree)
    (right :
      Nightstream.SuperNeo.SumCheck.Finite.FixedPolynomial α rightDegree)
    (degreeEqual : leftDegree = rightDegree)
    (coefficientsEqual : left.coefficients = right.coefficients) :
    HEq left right := by
  subst rightDegree
  exact heq_of_eq
    (fixedPolynomial_eq_of_coefficients_eq coefficientsEqual)

private theorem feCertificate_heq_of_input_eq
    {shape : SemanticShape}
    {leftInput rightInput : PublicInput shape}
    {domain : FlatNcDomain}
    (left : SumCheck.Fe.Certificate leftInput domain)
    (right : SumCheck.Fe.Certificate rightInput domain)
    (inputEqual : leftInput = rightInput)
    (rowRounds :
      ∀ round, HEq (left.rowRounds round) (right.rowRounds round))
    (laneRounds :
      ∀ round, left.laneRounds round = right.laneRounds round) :
    HEq left right := by
  subst rightInput
  apply heq_of_eq
  cases left with
  | mk leftRows leftLanes =>
      cases right with
      | mk rightRows rightLanes =>
          simp only at rowRounds laneRounds
          congr
          · funext round
            exact eq_of_heq (rowRounds round)
          · funext round
            exact laneRounds round

private theorem ncCertificate_eq_of_rounds
    {domain : BlockNcDomain}
    (left right : Transcript.Nc.BlockLane.Certificate domain)
    (rounds : left.rounds = right.rounds) :
    left = right := by
  cases left
  cases right
  simp_all

/-- The polynomial decoded from a complete family of projected proof
coordinates is the exact semantic polynomial carried by the proof. -/
theorem proofPolynomial_eq
    (application : Poseidon23ApplicationProfile Selected)
    {context : Schema (typeSystem Selected)}
    {runningRef :
      Ref (typeSystem Selected) context (.data .running)}
    {freshRef :
      Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef :
      Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame :
      CallFrame (signature := signature Selected)
        (FamilyFor application) Call.nifsVerify
        (Refs.cons runningRef
          (Refs.cons freshRef (Refs.cons proofRef .nil))))
    {degree : Nat}
    (polynomial :
      SelectedProof shape TranscriptState publicRingColumns publicFits
          verifierRows →
        Nightstream.SuperNeo.SumCheck.Finite.FixedPolynomial
          Nightstream.SuperNeo.Concrete.K degree)
    (views :
      ∀ slot : Fin (degree + 1),
        KView
          ((FamilyFor application).codecFor (.data .nifsProof))
          (fun proof =>
            (polynomial proof).coefficients.getD slot.val
              Nightstream.SuperNeo.Concrete.K.zero))
    (assignment : ColumnId → Field)
    (proof :
      SelectedProof shape TranscriptState publicRingColumns publicFits
        verifierRows)
    (decoded :
      (proofOperand frame.operands).Decodes (FamilyFor application)
        (.data .nifsProof) assignment proof) :
    ({
      coefficients :=
        List.ofFn fun slot =>
          ofProjection
            ((proofColumns (FamilyFor application) frame
              (views slot)).value
                (numericAssignment (columnMap frame) assignment))
      coefficients_length := by simp
    } :
      Nightstream.SuperNeo.SumCheck.Finite.FixedPolynomial
        Nightstream.SuperNeo.Concrete.K degree) =
      polynomial proof := by
  apply fixedPolynomial_eq_of_coefficients_eq
  exact proofCoefficientList_eq application frame
    (fun proof => (polynomial proof).coefficients) views
    assignment proof decoded (polynomial proof).coefficients_length

/-- Evaluating the physical statement expressions reconstructs the exact
selected PiCCS statement serialization. -/
theorem statementFields_eq
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {schema : Schema (typeSystem Selected)}
    {runningRef :
      Ref (typeSystem Selected) schema (.data .running)}
    {freshRef :
      Ref (typeSystem Selected) schema (.data .fresh)}
    {proofRef :
      Ref (typeSystem Selected) schema (.data .nifsProof)}
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
    (decoded :
      frame.operands.Decodes (FamilyFor application) assignment
        (.cons running (.cons fresh (.cons proof .nil)))) :
    KSplitNcTranscriptSemantics.fieldValues
        (numericAssignment (columnMap frame) assignment)
        (transcriptInput application profile frame).statementFields =
      profile.serialization.statementFields
        (ConcreteNifsParameters.context
          (keys
            Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected)
          running fresh proof).materialize.piCcsStatement := by
  calc
    KSplitNcTranscriptSemantics.fieldValues
          (numericAssignment (columnMap frame) assignment)
          (transcriptInput application profile frame).statementFields =
        profile.statementSources.map
          (fun source => (source.value running fresh proof).val) := by
      unfold KSplitNcTranscriptSemantics.fieldValues
      simp only [transcriptInput, List.map_map, Function.comp_apply]
      apply List.map_congr_left
      intro source _
      exact sourceExpression_eval application frame source assignment
        running fresh proof constantWire decoded
    _ = _ := profile.statementExact running fresh proof

/-- Evaluating the physical output expressions reconstructs the exact raw
PiCCS output serialization carried by the selected certificate. -/
theorem outputFields_eq
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {schema : Schema (typeSystem Selected)}
    {runningRef :
      Ref (typeSystem Selected) schema (.data .running)}
    {freshRef :
      Ref (typeSystem Selected) schema (.data .fresh)}
    {proofRef :
      Ref (typeSystem Selected) schema (.data .nifsProof)}
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
    (decoded :
      frame.operands.Decodes (FamilyFor application) assignment
        (.cons running (.cons fresh (.cons proof .nil)))) :
    KSplitNcTranscriptSemantics.fieldValues
        (numericAssignment (columnMap frame) assignment)
        (transcriptInput application profile frame).outputFields =
      profile.serialization.outputFields proof.certificate.piCcs.output := by
  calc
    KSplitNcTranscriptSemantics.fieldValues
          (numericAssignment (columnMap frame) assignment)
          (transcriptInput application profile frame).outputFields =
        profile.outputSources.map
          (fun source => (source.value running fresh proof).val) := by
      unfold KSplitNcTranscriptSemantics.fieldValues
      simp only [transcriptInput, List.map_map, Function.comp_apply]
      apply List.map_congr_left
      intro source _
      exact sourceExpression_eval application frame source assignment
        running fresh proof constantWire decoded
    _ = _ := profile.outputExact running fresh proof

/-- The symbolic prior transcript state is exactly the selected proof's
decoded prior state, including all eight lanes and the absorb cursor. -/
theorem priorState_eq
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {schema : Schema (typeSystem Selected)}
    {runningRef :
      Ref (typeSystem Selected) schema (.data .running)}
    {freshRef :
      Ref (typeSystem Selected) schema (.data .fresh)}
    {proofRef :
      Ref (typeSystem Selected) schema (.data .nifsProof)}
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
    (decoded :
      frame.operands.Decodes (FamilyFor application) assignment
        (.cons running (.cons fresh (.cons proof .nil)))) :
    KSplitNcTranscriptSemantics.priorState
        (numericAssignment (columnMap frame) assignment)
        (transcriptInput application profile frame) =
      proof.priorState := by
  have proofDecoded :=
    ConcreteNifsSelectedCallFrame.proof_decodes_of_frame_decodes
      (FamilyFor application) frame assignment running fresh proof decoded
  have admissible :
      ((FamilyFor application).codecFor (.data .nifsProof)).Admissible
        proof :=
    Codec.admissible_of_decode
      ((FamilyFor application).codecFor (.data .nifsProof)) proofDecoded
  apply duplexState_eq
  · funext lane
    have read := locatedField_eval (profile.priorLane lane)
      (proofOperand frame.operands) (proof_widthsAgree frame)
      (proofFieldLocation (FamilyFor application) frame
        (profile.priorLane lane))
      assignment proof proofDecoded
    have laneBound := profile.proofAdmissibleLanes proof admissible lane
    have laneBoundConcrete :
        proof.priorState.lanes lane <
          Nightstream.SuperNeo.Concrete.goldilocksModulus := by
      simpa only [
        Nightstream.Implementation.R1CS.goldilocksP,
        Nightstream.SuperNeo.Concrete.goldilocksModulus] using laneBound
    simpa only [
      KSplitNcTranscriptSemantics.priorState,
      KSplitNcTranscript.initialBuilder,
      SymbolicDuplex.start,
      SymbolicDuplexSemantics.decodedBuilder,
      SymbolicDuplexSemantics.evalState,
      transcriptInput,
      NumericRowBridge.residue,
      Nat.mod_eq_of_lt laneBoundConcrete] using read
  · exact (profile.proofAdmissibleCursor proof admissible).symm

/-- Each decoded FE row-phase message is the exact message carried by the
selected proof.  The heterogeneous equality records that the proof's public
input is codec-bound to the verifier-selected static polynomial. -/
theorem feRowPolynomial_heq
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {schema : Schema (typeSystem Selected)}
    {runningRef :
      Ref (typeSystem Selected) schema (.data .running)}
    {freshRef :
      Ref (typeSystem Selected) schema (.data .fresh)}
    {proofRef :
      Ref (typeSystem Selected) schema (.data .nifsProof)}
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
    (decoded :
      frame.operands.Decodes (FamilyFor application) assignment
        (.cons running (.cons fresh (.cons proof .nil))))
    (round : Fin shape.rowVariables) :
    HEq
      ((transcriptInput application profile frame).fe.rowRounds round
        |>.paperPolynomial
          (numericAssignment (columnMap frame) assignment))
      (proof.certificate.piCcs.fe.rowRounds round) := by
  have proofDecoded :=
    ConcreteNifsSelectedCallFrame.proof_decodes_of_frame_decodes
      (FamilyFor application) frame assignment running fresh proof decoded
  have admissible :
      ((FamilyFor application).codecFor (.data .nifsProof)).Admissible
        proof :=
    Codec.admissible_of_decode
      ((FamilyFor application).codecFor (.data .nifsProof)) proofDecoded
  have selected :=
    profile.proofAdmissiblePolynomial proof admissible
  have restored :=
    KSplitNcStaticInput.withDynamicClaims_eq
      profile.constraintPolynomial proof.piCcsInput selected
  have degreeEq :
      SumCheck.Fe.Drow proof.piCcsInput =
        SumCheck.Fe.Drow
          (KSplitNcStaticInput.layoutInput
            profile.constraintPolynomial) := by
    rw [← restored]
    exact KSplitNcStaticInput.drow_withDynamicClaims
      profile.constraintPolynomial proof.piCcsInput
  have coefficientsLength :
      (proof.certificate.piCcs.fe.rowRounds round).coefficients.length =
        SumCheck.Fe.Drow
            (KSplitNcStaticInput.layoutInput
              profile.constraintPolynomial) + 1 := by
    rw [
      (proof.certificate.piCcs.fe.rowRounds round).coefficients_length,
      degreeEq]
  have coefficientsEq :=
    proofCoefficientList_eq application frame
      (fun proof =>
        (proof.certificate.piCcs.fe.rowRounds round).coefficients)
      (profile.messageViews.feRow round)
      assignment proof proofDecoded coefficientsLength
  apply fixedPolynomial_heq_of_degree_eq _ _ degreeEq.symm
  simpa only [
    KFixedPhaseSemanticOccurrence.RoundColumns.paperPolynomial,
    transcriptInput,
    List.map_ofFn] using coefficientsEq

/-- Each fixed-width FE lane-phase message is exactly the selected proof
message. -/
theorem feLanePolynomial_eq
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {schema : Schema (typeSystem Selected)}
    {runningRef :
      Ref (typeSystem Selected) schema (.data .running)}
    {freshRef :
      Ref (typeSystem Selected) schema (.data .fresh)}
    {proofRef :
      Ref (typeSystem Selected) schema (.data .nifsProof)}
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
    (decoded :
      frame.operands.Decodes (FamilyFor application) assignment
        (.cons running (.cons fresh (.cons proof .nil))))
    (round :
      Fin
        Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.PiCcsDomains.production.fe.laneVariables) :
    ((transcriptInput application profile frame).fe.laneRounds round
      |>.paperPolynomial
        (numericAssignment (columnMap frame) assignment)) =
      proof.certificate.piCcs.fe.laneRounds round := by
  have proofDecoded :=
    ConcreteNifsSelectedCallFrame.proof_decodes_of_frame_decodes
      (FamilyFor application) frame assignment running fresh proof decoded
  have coefficientsEq :=
    proofCoefficientList_eq application frame
      (fun proof =>
        (proof.certificate.piCcs.fe.laneRounds round).coefficients)
      (profile.messageViews.feLane round)
      assignment proof proofDecoded
      (proof.certificate.piCcs.fe.laneRounds round).coefficients_length
  apply fixedPolynomial_eq_of_coefficients_eq
  simpa only [
    KFixedPhaseSemanticOccurrence.RoundColumns.paperPolynomial,
    transcriptInput,
    List.map_ofFn] using coefficientsEq

/-- Each block-prefix NC message is exactly the corresponding selected proof
message. -/
theorem ncBlockPolynomial_eq
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {schema : Schema (typeSystem Selected)}
    {runningRef :
      Ref (typeSystem Selected) schema (.data .running)}
    {freshRef :
      Ref (typeSystem Selected) schema (.data .fresh)}
    {proofRef :
      Ref (typeSystem Selected) schema (.data .nifsProof)}
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
    (decoded :
      frame.operands.Decodes (FamilyFor application) assignment
        (.cons running (.cons fresh (.cons proof .nil))))
    (round :
      Fin
        Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.PiCcsDomains.production.nc.blockVariables) :
    ((transcriptInput application profile frame).nc.blockRounds round
      |>.paperPolynomial
        (numericAssignment (columnMap frame) assignment)) =
      proof.certificate.piCcs.nc.rounds
        (Fin.castAdd
          Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.PiCcsDomains.production.nc.laneVariables
          round) := by
  have proofDecoded :=
    ConcreteNifsSelectedCallFrame.proof_decodes_of_frame_decodes
      (FamilyFor application) frame assignment running fresh proof decoded
  have coefficientsEq :=
    proofCoefficientList_eq application frame
      (fun proof =>
        (proof.certificate.piCcs.nc.rounds
          (Fin.castAdd
            Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.PiCcsDomains.production.nc.laneVariables
            round)).coefficients)
      (profile.messageViews.nc
        (Fin.castAdd
          Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.PiCcsDomains.production.nc.laneVariables
          round))
      assignment proof proofDecoded
      (proof.certificate.piCcs.nc.rounds
        (Fin.castAdd
          Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.PiCcsDomains.production.nc.laneVariables
          round)).coefficients_length
  apply fixedPolynomial_eq_of_coefficients_eq
  simpa only [
    KFixedPhaseSemanticOccurrence.RoundColumns.paperPolynomial,
    transcriptInput,
    List.map_ofFn] using coefficientsEq

/-- Each lane-suffix NC message is exactly the corresponding selected proof
message. -/
theorem ncLanePolynomial_eq
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {schema : Schema (typeSystem Selected)}
    {runningRef :
      Ref (typeSystem Selected) schema (.data .running)}
    {freshRef :
      Ref (typeSystem Selected) schema (.data .fresh)}
    {proofRef :
      Ref (typeSystem Selected) schema (.data .nifsProof)}
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
    (decoded :
      frame.operands.Decodes (FamilyFor application) assignment
        (.cons running (.cons fresh (.cons proof .nil))))
    (round :
      Fin
        Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.PiCcsDomains.production.nc.laneVariables) :
    ((transcriptInput application profile frame).nc.laneRounds round
      |>.paperPolynomial
        (numericAssignment (columnMap frame) assignment)) =
      proof.certificate.piCcs.nc.rounds
        (Fin.natAdd
          Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.PiCcsDomains.production.nc.blockVariables
          round) := by
  have proofDecoded :=
    ConcreteNifsSelectedCallFrame.proof_decodes_of_frame_decodes
      (FamilyFor application) frame assignment running fresh proof decoded
  have coefficientsEq :=
    proofCoefficientList_eq application frame
      (fun proof =>
        (proof.certificate.piCcs.nc.rounds
          (Fin.natAdd
            Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.PiCcsDomains.production.nc.blockVariables
            round)).coefficients)
      (profile.messageViews.nc
        (Fin.natAdd
          Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.PiCcsDomains.production.nc.blockVariables
          round))
      assignment proof proofDecoded
      (proof.certificate.piCcs.nc.rounds
        (Fin.natAdd
          Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.PiCcsDomains.production.nc.blockVariables
          round)).coefficients_length
  apply fixedPolynomial_eq_of_coefficients_eq
  simpa only [
    KFixedPhaseSemanticOccurrence.RoundColumns.paperPolynomial,
    transcriptInput,
    List.map_ofFn] using coefficientsEq

/-- The complete decoded FE certificate is the exact selected proof
certificate after the codec-bound public input is restored. -/
theorem feCertificate_heq
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {schema : Schema (typeSystem Selected)}
    {runningRef :
      Ref (typeSystem Selected) schema (.data .running)}
    {freshRef :
      Ref (typeSystem Selected) schema (.data .fresh)}
    {proofRef :
      Ref (typeSystem Selected) schema (.data .nifsProof)}
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
    (decoded :
      frame.operands.Decodes (FamilyFor application) assignment
        (.cons running (.cons fresh (.cons proof .nil)))) :
    HEq
      (KSplitNcTranscriptPhases.feCertificate
        (numericAssignment (columnMap frame) assignment)
        (KSplitNcStaticInput.retargetTranscript proof.piCcsInput
          (transcriptInput application profile frame)))
      proof.certificate.piCcs.fe := by
  have proofDecoded :=
    ConcreteNifsSelectedCallFrame.proof_decodes_of_frame_decodes
      (FamilyFor application) frame assignment running fresh proof decoded
  have admissible :
      ((FamilyFor application).codecFor (.data .nifsProof)).Admissible
        proof :=
    Codec.admissible_of_decode
      ((FamilyFor application).codecFor (.data .nifsProof)) proofDecoded
  have selected :=
    profile.proofAdmissiblePolynomial proof admissible
  have restored :=
    KSplitNcStaticInput.withDynamicClaims_eq
      profile.constraintPolynomial proof.piCcsInput selected
  apply feCertificate_heq_of_input_eq _ _ restored
  · intro round
    simpa only [
      KSplitNcTranscriptPhases.feCertificate,
      KSplitNcStaticInput.retargetTranscript] using
        feRowPolynomial_heq application profile frame assignment
          running fresh proof decoded round
  · intro round
    simpa only [
      KSplitNcTranscriptPhases.feCertificate,
      KSplitNcStaticInput.retargetTranscript] using
        feLanePolynomial_eq application profile frame assignment
          running fresh proof decoded round

/-- The complete decoded block×lane NC certificate is exactly the selected
proof certificate, including the physical block-prefix/lane-suffix order. -/
theorem ncCertificate_eq
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {schema : Schema (typeSystem Selected)}
    {runningRef :
      Ref (typeSystem Selected) schema (.data .running)}
    {freshRef :
      Ref (typeSystem Selected) schema (.data .fresh)}
    {proofRef :
      Ref (typeSystem Selected) schema (.data .nifsProof)}
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
    (decoded :
      frame.operands.Decodes (FamilyFor application) assignment
        (.cons running (.cons fresh (.cons proof .nil)))) :
    KSplitNcTranscriptPhases.ncCertificate
        (numericAssignment (columnMap frame) assignment)
        (KSplitNcStaticInput.retargetTranscript proof.piCcsInput
          (transcriptInput application profile frame)) =
      proof.certificate.piCcs.nc := by
  apply ncCertificate_eq_of_rounds
  funext round
  refine Fin.addCases (motive := fun index =>
    (KSplitNcTranscriptPhases.ncCertificate
      (numericAssignment (columnMap frame) assignment)
      (KSplitNcStaticInput.retargetTranscript proof.piCcsInput
        (transcriptInput application profile frame))).rounds index =
      proof.certificate.piCcs.nc.rounds index)
    (fun blockRound => ?_) (fun laneRound => ?_) round
  · simpa only [
      KSplitNcTranscriptPhases.ncCertificate,
      KSplitNcStaticInput.retargetTranscript,
      Fin.addCases_left] using
        ncBlockPolynomial_eq application profile frame assignment
          running fresh proof decoded blockRound
  · simpa only [
      KSplitNcTranscriptPhases.ncCertificate,
      KSplitNcStaticInput.retargetTranscript,
      Fin.addCases_right] using
        ncLanePolynomial_eq application profile frame assignment
          running fresh proof decoded laneRound

end SelectedFrame

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsOperationalOccurrenceSemantics
