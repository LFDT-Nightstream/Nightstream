import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsOutputRows
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsPiRlcParentSemantics
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsSelectedCallFrame

/-!
Contract: exact semantic refinement and honest completeness of selected NIFS
output-child materialization.

Whole-frame decoding is the sole source of proof and output values.
Satisfaction forces every output child to be the exact
`PiDecChildPayload.materialize` view: payload commitment, payload public input,
the computed parent point, and payload evaluations. Composed with the
independently row-derived parent theorem, this yields exact equality with
`SelectedRunning.ofResult`.
-/

set_option autoImplicit false
set_option maxRecDepth 100000
set_option maxHeartbeats 2400000

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsOutputSemantics

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
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCarrierViews
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal
open Nightstream.Implementation.R1CS.Canonical.KMul
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.Nifs
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
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
            let coordinate : Fin variables := ⟨index, by omega⟩
            exact coordinates coordinate

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
  simp only [KHorner.Pair.mk.injEq]
  exact ⟨lowEqual, highEqual⟩

private theorem fEquality_complete_of_decoded
    (assignment : Nat → Nat)
    (left right : LinComb)
    (leftValue rightValue : F)
    (constantWire : assignment 0 = 1)
    (leftExact : residue (lcEval assignment left) = leftValue)
    (rightExact : residue (lcEval assignment right) = rightValue)
    (equal : leftValue = rightValue) :
    RowHolds assignment (KEquality.equalityRow left right) := by
  apply (KEquality.equalityRow_iff assignment left right constantWire).2
  apply residue_injective_of_lt
  · unfold lcEval
    exact Nat.mod_lt _ (by decide)
  · unfold lcEval
    exact Nat.mod_lt _ (by decide)
  · exact leftExact.trans (equal.trans rightExact.symm)

private theorem kEquality_complete_of_decoded
    (assignment : Nat → Nat)
    (left right : Carried)
    (leftValue rightValue : K)
    (constantWire : assignment 0 = 1)
    (leftExact : KPointEquality.decoded assignment left = leftValue)
    (rightExact : KPointEquality.decoded assignment right = rightValue)
    (equal : leftValue = rightValue) :
    Satisfies (KEquality.rows left right) assignment := by
  have decodedEqual : KPointEquality.decoded assignment left =
      KPointEquality.decoded assignment right :=
    leftExact.trans (equal.trans rightExact.symm)
  have pairs := congrArg KConcreteBridge.ofConcrete decodedEqual
  rw [KPointEquality.ofConcrete_decoded,
    KPointEquality.ofConcrete_decoded] at pairs
  unfold KHorner.carriedValue at pairs
  simp only [KHorner.Pair.mk.injEq] at pairs
  exact KEquality.rows_complete assignment left right constantWire
    pairs.1 pairs.2

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

/-- Exact child-field equations expressed through decoded public carriers. -/
structure ChildEquations
    (output :
      SelectedRunning shape publicRingColumns publicFits verifierRows)
    (proof :
      SelectedProof shape TranscriptState publicRingColumns publicFits
        verifierRows) : Prop where
  commitment :
    ∀ child,
      (output.children child).commitment =
        (proof.certificate.piDecPayloads child).commitment
  publicInput :
    ∀ child,
      (output.children child).publicInput =
        (proof.certificate.piDecPayloads child).publicInput
  point :
    ∀ child, (output.children child).point = output.parent.point
  evaluations :
    ∀ child,
      (output.children child).evaluations =
        (proof.certificate.piDecPayloads child).evaluations

/-- Output-child equations before assuming that the output codec decodes.
Every left side is read directly from the physical output bundle. -/
structure PhysicalChildEquations
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
    (proof :
      SelectedProof shape TranscriptState publicRingColumns publicFits
        verifierRows) : Prop where
  commitment :
    ∀ child row lane,
      residue
          (lcEval
            (numericAssignment (columnMap frame) assignment)
            (ConcreteNifsOutputRows.outputChildCommitment
              application profile frame child row lane)) =
        (proof.certificate.piDecPayloads child).commitment row lane
  publicInput :
    ∀ child column,
      residue
          (lcEval
            (numericAssignment (columnMap frame) assignment)
            (ConcreteNifsOutputRows.outputChildPublic
              application profile frame child column)) =
        (proof.certificate.piDecPayloads child).publicInput column
  point :
    ∀ child coordinate,
      KPointEquality.decoded
          (numericAssignment (columnMap frame) assignment)
          (ConcreteNifsOutputRows.outputChildPoint
            application profile frame child coordinate) =
        KPointEquality.decoded
          (numericAssignment (columnMap frame) assignment)
          (ConcreteNifsOutputRows.outputParentPoint
            application profile frame coordinate)
  evaluations :
    ∀ child matrix lane,
      KPointEquality.decoded
          (numericAssignment (columnMap frame) assignment)
          (ConcreteNifsOutputRows.outputChildEvaluation
            application profile frame child matrix lane) =
        (proof.certificate.piDecPayloads child).evaluations.getD
          matrix.val ringKZero lane

private theorem runningPayload_eq
    (left right :
      FixedActive.Canonical.RunningPayload
        shape publicRingColumns publicFits verifierRows)
    (commitment : left.commitment = right.commitment)
    (publicInput : left.publicInput = right.publicInput)
    (point : left.point = right.point)
    (evaluations : left.evaluations = right.evaluations) :
    left = right := by
  cases left
  cases right
  cases commitment
  cases publicInput
  cases point
  cases evaluations
  rfl

private theorem selectedRunning_eq
    (left right :
      SelectedRunning shape publicRingColumns publicFits verifierRows)
    (parent : left.parent = right.parent)
    (children : ∀ child, left.children child = right.children child) :
    left = right := by
  cases left
  cases right
  cases parent
  congr
  funext child
  exact children child

/-- Satisfaction of the output-child slice determines every physical child
coordinate without an output-decoding premise. -/
theorem physical_child_equations_of_rows
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
    (satisfied :
      Satisfies
        (ConcreteNifsOutputRows.rows application profile frame)
        (numericAssignment (columnMap frame) assignment)) :
    PhysicalChildEquations application profile frame assignment proof := by
  let numeric := numericAssignment (columnMap frame) assignment
  have numericOne : numeric 0 = 1 :=
    ConcreteNifsOperationalOccurrenceSemantics.numericConstantWire
      application frame assignment constantWire
  have proofDecoded :=
    ConcreteNifsSelectedCallFrame.proof_decodes_of_frame_decodes
      (FamilyFor application) frame assignment running fresh proof decodedInputs
  refine {
    commitment := ?_
    publicInput := ?_
    point := ?_
    evaluations := ?_
  }
  · intro child row lane
    have rowHolds :=
      satisfied _
        (ConcreteNifsOutputRows.commitmentRow_mem
          application profile frame child row lane)
    have rawEqual :=
      (KEquality.equalityRow_iff numeric _ _ numericOne).1 rowHolds
    have proofExact :=
      ConcreteNifsCarrierFrame.proofF_decoded
        (FamilyFor application) frame
        (profile.payloadViews.commitment child row lane)
        assignment proof proofDecoded
    exact
      (congrArg residue rawEqual).trans
        (by
          simpa [ConcreteNifsOutputRows.proofChildCommitment,
            payloadCommitmentCoordinate] using proofExact)
  · intro child column
    have rowHolds :=
      satisfied _
        (ConcreteNifsOutputRows.publicRow_mem
          application profile frame child column)
    have rawEqual :=
      (KEquality.equalityRow_iff numeric _ _ numericOne).1 rowHolds
    have proofExact :=
      ConcreteNifsCarrierFrame.proofF_decoded
        (FamilyFor application) frame
        (profile.payloadViews.publicInput child column)
        assignment proof proofDecoded
    exact
      (congrArg residue rawEqual).trans
        (by
          simpa [ConcreteNifsOutputRows.proofChildPublic,
            payloadPublicCoordinate] using proofExact)
  · intro child coordinate
    have coordinateSatisfied :
        Satisfies
          (KEquality.rows
            (ConcreteNifsOutputRows.outputChildPoint
              application profile frame child coordinate)
            (ConcreteNifsOutputRows.outputParentPoint
              application profile frame coordinate))
          numeric := by
      intro row member
      exact satisfied row
        (ConcreteNifsOutputRows.pointRow_mem
          application profile frame child coordinate row member)
    exact decoded_eq_of_rows numeric
      (ConcreteNifsOutputRows.outputChildPoint
        application profile frame child coordinate)
      (ConcreteNifsOutputRows.outputParentPoint
        application profile frame coordinate)
      numericOne coordinateSatisfied
  · intro child matrix lane
    have coordinateSatisfied :
        Satisfies
          (KEquality.rows
            (ConcreteNifsOutputRows.outputChildEvaluation
              application profile frame child matrix lane)
            (ConcreteNifsOutputRows.proofChildEvaluation
              application profile frame child matrix lane))
          numeric := by
      intro row member
      exact satisfied row
        (ConcreteNifsOutputRows.evaluationRow_mem
          application profile frame child matrix lane row member)
    have decodedEqual :=
      decoded_eq_of_rows numeric
        (ConcreteNifsOutputRows.outputChildEvaluation
          application profile frame child matrix lane)
        (ConcreteNifsOutputRows.proofChildEvaluation
          application profile frame child matrix lane)
        numericOne coordinateSatisfied
    have proofExact :=
      ConcreteNifsCarrierFrame.proofK_decoded
        (FamilyFor application) frame
        (profile.payloadViews.evaluation child matrix lane)
        assignment proof proofDecoded
    exact decodedEqual.trans
      (by
        simpa [ConcreteNifsOutputRows.proofChildEvaluation,
          payloadEvaluationCoordinate] using proofExact)

/-- The output-child rows derive every field of `ChildEquations`; none is a
caller premise. -/
theorem child_equations_of_rows
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
    (satisfied :
      Satisfies
        (ConcreteNifsOutputRows.rows application profile frame)
        (numericAssignment (columnMap frame) assignment)) :
    ChildEquations output proof := by
  let numeric := numericAssignment (columnMap frame) assignment
  have numericOne : numeric 0 = 1 :=
    ConcreteNifsOperationalOccurrenceSemantics.numericConstantWire
      application frame assignment constantWire
  have proofDecoded :=
    ConcreteNifsSelectedCallFrame.proof_decodes_of_frame_decodes
      (FamilyFor application) frame assignment running fresh proof decodedInputs
  have outputDecoded :=
    ConcreteNifsSelectedCallFrame.output_decodes_of_frame_decodes
      (FamilyFor application) frame assignment output decodedOutput
  have proofAdmissible :
      ((FamilyFor application).codecFor (.data .nifsProof)).Admissible proof :=
    Codec.admissible_of_decode
      ((FamilyFor application).codecFor (.data .nifsProof)) proofDecoded
  have outputAdmissible :
      ((FamilyFor application).codecFor (.data .running)).Admissible output :=
    Codec.admissible_of_decode
      ((FamilyFor application).codecFor (.data .running)) outputDecoded
  refine {
    commitment := ?_
    publicInput := ?_
    point := ?_
    evaluations := ?_
  }
  · intro child
    funext row lane
    have rowHolds :=
      satisfied _
        (ConcreteNifsOutputRows.commitmentRow_mem
          application profile frame child row lane)
    have rawEqual :=
      (KEquality.equalityRow_iff numeric _ _ numericOne).1 rowHolds
    have outputExact :=
      ConcreteNifsCarrierFrame.outputF_decoded
        (FamilyFor application) frame
        (profile.runningViews.childCommitment child row lane)
        assignment output outputDecoded
    have proofExact :=
      ConcreteNifsCarrierFrame.proofF_decoded
        (FamilyFor application) frame
        (profile.payloadViews.commitment child row lane)
        assignment proof proofDecoded
    calc
      (output.children child).commitment row lane =
          residue
            (lcEval numeric
              (ConcreteNifsOutputRows.outputChildCommitment
                application profile frame child row lane)) := by
        simpa [ConcreteNifsOutputRows.outputChildCommitment,
          childCommitmentCoordinate] using outputExact.symm
      _ = residue
            (lcEval numeric
              (ConcreteNifsOutputRows.proofChildCommitment
                application profile frame child row lane)) :=
        congrArg residue rawEqual
      _ = (proof.certificate.piDecPayloads child).commitment row lane := by
        simpa [ConcreteNifsOutputRows.proofChildCommitment,
          payloadCommitmentCoordinate] using proofExact
  · intro child
    funext column
    have rowHolds :=
      satisfied _
        (ConcreteNifsOutputRows.publicRow_mem
          application profile frame child column)
    have rawEqual :=
      (KEquality.equalityRow_iff numeric _ _ numericOne).1 rowHolds
    have outputExact :=
      ConcreteNifsCarrierFrame.outputF_decoded
        (FamilyFor application) frame
        (profile.runningViews.childPublic child column)
        assignment output outputDecoded
    have proofExact :=
      ConcreteNifsCarrierFrame.proofF_decoded
        (FamilyFor application) frame
        (profile.payloadViews.publicInput child column)
        assignment proof proofDecoded
    calc
      (output.children child).publicInput column =
          residue
            (lcEval numeric
              (ConcreteNifsOutputRows.outputChildPublic
                application profile frame child column)) := by
        simpa [ConcreteNifsOutputRows.outputChildPublic,
          childPublicCoordinate] using outputExact.symm
      _ = residue
            (lcEval numeric
              (ConcreteNifsOutputRows.proofChildPublic
                application profile frame child column)) :=
        congrArg residue rawEqual
      _ = (proof.certificate.piDecPayloads child).publicInput column := by
        simpa [ConcreteNifsOutputRows.proofChildPublic,
          payloadPublicCoordinate] using proofExact
  · intro child
    apply cubePoint_eq_of_coordinates
    intro coordinate
    have coordinateSatisfied :
        Satisfies
          (KEquality.rows
            (ConcreteNifsOutputRows.outputChildPoint
              application profile frame child coordinate)
            (ConcreteNifsOutputRows.outputParentPoint
              application profile frame coordinate))
          numeric := by
      intro row member
      exact satisfied row
        (ConcreteNifsOutputRows.pointRow_mem
          application profile frame child coordinate row member)
    have decodedEqual :=
      decoded_eq_of_rows numeric
        (ConcreteNifsOutputRows.outputChildPoint
          application profile frame child coordinate)
        (ConcreteNifsOutputRows.outputParentPoint
          application profile frame coordinate)
        numericOne coordinateSatisfied
    have childExact :=
      ConcreteNifsCarrierFrame.outputK_decoded
        (FamilyFor application) frame
        (profile.runningViews.childPoint child coordinate)
        assignment output outputDecoded
    have parentExact :=
      ConcreteNifsCarrierFrame.outputK_decoded
        (FamilyFor application) frame
        (profile.runningViews.parentPoint coordinate)
        assignment output outputDecoded
    calc
      pointAt (output.children child).point coordinate =
          KPointEquality.decoded numeric
            (ConcreteNifsOutputRows.outputChildPoint
              application profile frame child coordinate) := by
        simpa [ConcreteNifsOutputRows.outputChildPoint,
          childPointCoordinate, pointAt] using childExact.symm
      _ = KPointEquality.decoded numeric
            (ConcreteNifsOutputRows.outputParentPoint
              application profile frame coordinate) :=
        decodedEqual
      _ = pointAt output.parent.point coordinate := by
        simpa [ConcreteNifsOutputRows.outputParentPoint,
          parentPointCoordinate, pointAt] using parentExact
  · intro child
    have outputSize :
        (output.children child).evaluations.size = shape.matrixCount :=
      profile.runningViews.childEvaluationsSize output outputAdmissible child
    have proofSize :
        (proof.certificate.piDecPayloads child).evaluations.size =
          shape.matrixCount :=
      profile.payloadViews.evaluationsSize proof proofAdmissible child
    apply Array.ext
    · rw [outputSize, proofSize]
    · intro index outputLt proofLt
      let matrix : Fin shape.matrixCount :=
        ⟨index, by
          rw [outputSize] at outputLt
          exact outputLt⟩
      funext lane
      have coordinateSatisfied :
          Satisfies
            (KEquality.rows
              (ConcreteNifsOutputRows.outputChildEvaluation
                application profile frame child matrix lane)
              (ConcreteNifsOutputRows.proofChildEvaluation
                application profile frame child matrix lane))
            numeric := by
        intro row member
        exact satisfied row
          (ConcreteNifsOutputRows.evaluationRow_mem
            application profile frame child matrix lane row member)
      have decodedEqual :=
        decoded_eq_of_rows numeric
          (ConcreteNifsOutputRows.outputChildEvaluation
            application profile frame child matrix lane)
          (ConcreteNifsOutputRows.proofChildEvaluation
            application profile frame child matrix lane)
          numericOne coordinateSatisfied
      have outputExact :=
        ConcreteNifsCarrierFrame.outputK_decoded
          (FamilyFor application) frame
          (profile.runningViews.childEvaluation child matrix lane)
          assignment output outputDecoded
      have proofExact :=
        ConcreteNifsCarrierFrame.proofK_decoded
          (FamilyFor application) frame
          (profile.payloadViews.evaluation child matrix lane)
          assignment proof proofDecoded
      calc
        (output.children child).evaluations[index] lane =
            (output.children child).evaluations.getD
              index ringKZero lane := by
          rw [Array.getD_eq_getD_getElem?,
            Array.getElem?_eq_getElem outputLt]
          simp
        _ = KPointEquality.decoded numeric
              (ConcreteNifsOutputRows.outputChildEvaluation
                application profile frame child matrix lane) := by
          simpa [ConcreteNifsOutputRows.outputChildEvaluation,
            childEvaluationCoordinate] using outputExact.symm
        _ = KPointEquality.decoded numeric
              (ConcreteNifsOutputRows.proofChildEvaluation
                application profile frame child matrix lane) :=
          decodedEqual
        _ = (proof.certificate.piDecPayloads child).evaluations.getD
              index ringKZero lane := by
          simpa [ConcreteNifsOutputRows.proofChildEvaluation,
            payloadEvaluationCoordinate] using proofExact
        _ = (proof.certificate.piDecPayloads child).evaluations[index] lane := by
          rw [Array.getD_eq_getD_getElem?,
            Array.getElem?_eq_getElem proofLt]
          simp

/-- Honest child materialization equations satisfy every emitted output row
without allocating or extending the assignment. -/
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
    (equations : ChildEquations output proof) :
    Satisfies
      (ConcreteNifsOutputRows.rows application profile frame)
      (numericAssignment (columnMap frame) assignment) := by
  let numeric := numericAssignment (columnMap frame) assignment
  have numericOne : numeric 0 = 1 :=
    ConcreteNifsOperationalOccurrenceSemantics.numericConstantWire
      application frame assignment constantWire
  have proofDecoded :=
    ConcreteNifsSelectedCallFrame.proof_decodes_of_frame_decodes
      (FamilyFor application) frame assignment running fresh proof decodedInputs
  have outputDecoded :=
    ConcreteNifsSelectedCallFrame.output_decodes_of_frame_decodes
      (FamilyFor application) frame assignment output decodedOutput
  intro physical member
  rcases List.mem_flatMap.1 member with
    ⟨child, childMember, physicalMember⟩
  simp only [ConcreteNifsOutputRows.childRows, List.mem_append] at physicalMember
  rcases physicalMember with
    ((commitmentMember | publicMember) | pointMember) | evaluationMember
  · rcases List.mem_flatMap.1 commitmentMember with
      ⟨row, rowMember, laneMember⟩
    rcases List.mem_map.1 laneMember with
      ⟨lane, laneMember, rfl⟩
    have outputExact :=
      ConcreteNifsCarrierFrame.outputF_decoded
        (FamilyFor application) frame
        (profile.runningViews.childCommitment child row lane)
        assignment output outputDecoded
    have proofExact :=
      ConcreteNifsCarrierFrame.proofF_decoded
        (FamilyFor application) frame
        (profile.payloadViews.commitment child row lane)
        assignment proof proofDecoded
    apply fEquality_complete_of_decoded
      numeric
      (ConcreteNifsOutputRows.outputChildCommitment
        application profile frame child row lane)
      (ConcreteNifsOutputRows.proofChildCommitment
        application profile frame child row lane)
      ((output.children child).commitment row lane)
      ((proof.certificate.piDecPayloads child).commitment row lane)
      numericOne
    · simpa [ConcreteNifsOutputRows.outputChildCommitment,
        childCommitmentCoordinate] using outputExact
    · simpa [ConcreteNifsOutputRows.proofChildCommitment,
        payloadCommitmentCoordinate] using proofExact
    · exact congrFun (congrFun (equations.commitment child) row) lane
  · rcases List.mem_map.1 publicMember with
      ⟨column, columnMember, rfl⟩
    have outputExact :=
      ConcreteNifsCarrierFrame.outputF_decoded
        (FamilyFor application) frame
        (profile.runningViews.childPublic child column)
        assignment output outputDecoded
    have proofExact :=
      ConcreteNifsCarrierFrame.proofF_decoded
        (FamilyFor application) frame
        (profile.payloadViews.publicInput child column)
        assignment proof proofDecoded
    apply fEquality_complete_of_decoded
      numeric
      (ConcreteNifsOutputRows.outputChildPublic
        application profile frame child column)
      (ConcreteNifsOutputRows.proofChildPublic
        application profile frame child column)
      ((output.children child).publicInput column)
      ((proof.certificate.piDecPayloads child).publicInput column)
      numericOne
    · simpa [ConcreteNifsOutputRows.outputChildPublic,
        childPublicCoordinate] using outputExact
    · simpa [ConcreteNifsOutputRows.proofChildPublic,
        payloadPublicCoordinate] using proofExact
    · exact congrFun (equations.publicInput child) column
  · rcases List.mem_flatMap.1 pointMember with
      ⟨coordinate, coordinateMember, rowMember⟩
    have outputExact :=
      ConcreteNifsCarrierFrame.outputK_decoded
        (FamilyFor application) frame
        (profile.runningViews.childPoint child coordinate)
        assignment output outputDecoded
    have parentExact :=
      ConcreteNifsCarrierFrame.outputK_decoded
        (FamilyFor application) frame
        (profile.runningViews.parentPoint coordinate)
        assignment output outputDecoded
    have semanticEqual :
        pointAt (output.children child).point coordinate =
          pointAt output.parent.point coordinate :=
      congrArg (fun point => pointAt point coordinate) (equations.point child)
    exact
      kEquality_complete_of_decoded
        numeric
        (ConcreteNifsOutputRows.outputChildPoint
          application profile frame child coordinate)
        (ConcreteNifsOutputRows.outputParentPoint
          application profile frame coordinate)
        (pointAt (output.children child).point coordinate)
        (pointAt output.parent.point coordinate)
        numericOne
        (by simpa [ConcreteNifsOutputRows.outputChildPoint,
          childPointCoordinate, pointAt] using outputExact)
        (by simpa [ConcreteNifsOutputRows.outputParentPoint,
          parentPointCoordinate, pointAt] using parentExact)
        semanticEqual physical rowMember
  · rcases List.mem_flatMap.1 evaluationMember with
      ⟨matrix, matrixMember, laneRowsMember⟩
    rcases List.mem_flatMap.1 laneRowsMember with
      ⟨lane, laneMember, rowMember⟩
    have outputExact :=
      ConcreteNifsCarrierFrame.outputK_decoded
        (FamilyFor application) frame
        (profile.runningViews.childEvaluation child matrix lane)
        assignment output outputDecoded
    have proofExact :=
      ConcreteNifsCarrierFrame.proofK_decoded
        (FamilyFor application) frame
        (profile.payloadViews.evaluation child matrix lane)
        assignment proof proofDecoded
    have semanticEqual :
        (output.children child).evaluations.getD
            matrix.val ringKZero lane =
          (proof.certificate.piDecPayloads child).evaluations.getD
            matrix.val ringKZero lane := by
      exact congrFun
        (congrArg
          (fun values => values.getD matrix.val ringKZero)
          (equations.evaluations child)) lane
    exact
      kEquality_complete_of_decoded
        numeric
        (ConcreteNifsOutputRows.outputChildEvaluation
          application profile frame child matrix lane)
        (ConcreteNifsOutputRows.proofChildEvaluation
          application profile frame child matrix lane)
        ((output.children child).evaluations.getD
          matrix.val ringKZero lane)
        ((proof.certificate.piDecPayloads child).evaluations.getD
          matrix.val ringKZero lane)
        numericOne
        (by simpa [ConcreteNifsOutputRows.outputChildEvaluation,
          childEvaluationCoordinate] using outputExact)
        (by simpa [ConcreteNifsOutputRows.proofChildEvaluation,
          payloadEvaluationCoordinate] using proofExact)
        semanticEqual physical rowMember

/-- **Headline output refinement.** The parent/action/point rows and all child
materialization rows determine exactly the selected frozen NIFS result. -/
theorem output_eq_selectedResult_of_rows
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
    (productBase : Nat)
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
    (actionSatisfied :
      RawSatisfies
        (ConcreteNifsPiRlcActionRows.rows
          application profile frame productBase)
        assignment)
    (pointSatisfied :
      Satisfies
        (ConcreteNifsPiRlcPointRows.rows application profile frame)
        (numericAssignment (columnMap frame) assignment))
    (outputSatisfied :
      Satisfies
        (ConcreteNifsOutputRows.rows application profile frame)
        (numericAssignment (columnMap frame) assignment)) :
    SelectedRunning.ofResult
        (FixedActive.resultOf
          (ConcreteNifsParameters.context
            (keys
              Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected)
            running fresh proof).materialize
          proof.certificate) =
      output := by
  have parent :=
    ConcreteNifsPiRlcParentSemantics.parent_eq_derived
      application profile frame assignment running fresh proof output
      productBase constantWire decodedInputs decodedOutput
      operationalSatisfied actionSatisfied pointSatisfied
  have children :=
    child_equations_of_rows
      application profile frame assignment running fresh proof output
      constantWire decodedInputs decodedOutput outputSatisfied
  apply selectedRunning_eq
  · simpa [SelectedRunning.ofResult,
      FixedActive.resultOf, Result.resultOf,
      ConcreteNifsPiRlcParentSemantics.derivedParentPayload] using parent.symm
  · intro child
    apply runningPayload_eq
    · simpa [SelectedRunning.ofResult,
        FixedActive.resultOf, Result.resultOf,
        ConcretePhi81.outputChildren, Execution.piDecChildren,
        PiDecChildPayload.materialize] using
          (children.commitment child).symm
    · simpa [SelectedRunning.ofResult,
        FixedActive.resultOf, Result.resultOf,
        ConcretePhi81.outputChildren, Execution.piDecChildren,
        PiDecChildPayload.materialize] using
          (children.publicInput child).symm
    · have childPoint := children.point child
      rw [parent] at childPoint
      simpa [SelectedRunning.ofResult,
        FixedActive.resultOf, Result.resultOf,
        ConcretePhi81.outputChildren, Execution.piDecChildren,
        PiDecChildPayload.materialize,
        ConcreteNifsPiRlcParentSemantics.derivedParentPayload] using
          childPoint.symm
    · simpa [SelectedRunning.ofResult,
        FixedActive.resultOf, Result.resultOf,
        ConcretePhi81.outputChildren, Execution.piDecChildren,
        PiDecChildPayload.materialize] using
          (children.evaluations child).symm

end SelectedFrame

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsOutputSemantics
