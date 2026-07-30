import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsParameters
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsPiCcsFramePlacement
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsPiCcsEventBinding
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsPiRlcSourceBinding

/-!
Contract: specialize the authoritative `nifsVerify` call frame to the exact
paper NIFS selected by `PaperNifsParameters`.

This is the semantic-type bridge absent from the generic column-placement
modules.  Whole-frame decoding now yields the selected paper running, fresh,
and proof values directly; no separately decoded operand can be substituted.

Emits constraints: none.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsSelectedCallFrame

open Nightstream.HyperNova.Construction2.Paper
open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.Goldilocks.NumericRowBridge
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsCallFrame
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsGlobalColumnMap
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsParameters
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsPiCcsSourceBinding
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsPiCcsFramePlacement
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsPiCcsEventBinding
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsPiRlcSourceBinding
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.Nifs

abbrev K := Nightstream.SuperNeo.Concrete.K
abbrev Constants := Poseidon2Schedule.Constants

/-- The exact vocabulary parameters whose `nifsVerify` branch is the selected
one-message paper verifier over the canonical quadratic extension. -/
abbrev SelectedParameters
    {Commitment PublicInput Scalar TranscriptState : Type}
    {Digest AppState Witness Encoded RunningWitness FreshWitness : Type}
    [DecidableEq AppState]
    [DecidableEq Encoded]
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    (oneFresh : shape.freshCount = 1)
    (keys : Fin 1 ->
      SelectedKey K Commitment PublicInput Scalar TranscriptState
        shape columns blockCount degreeBound)
    (defaultRunning : SelectedRunning K Commitment PublicInput shape)
    (machine :
      Machine
        (SelectedKey K Commitment PublicInput Scalar TranscriptState
          shape columns blockCount degreeBound)
        Digest AppState Witness
        (SelectedRunning K Commitment PublicInput shape)
        (SelectedFresh Commitment PublicInput shape)
        Encoded 1)
    (terminalRelations :
      TerminalRelations
        (SelectedKey K Commitment PublicInput Scalar TranscriptState
          shape columns blockCount degreeBound)
        (SelectedRunning K Commitment PublicInput shape)
        RunningWitness
        (SelectedFresh Commitment PublicInput shape)
        FreshWitness 1)
    (terminalChecks :
      Nightstream.Protocol.FPrime.CanonicalTerminalVerifier.RelationChecks
        terminalRelations)
    (widths : Widths)
    (footprints : Footprints) : Parameters :=
  selected oneFresh keys defaultRunning machine terminalRelations
    terminalChecks widths footprints

section SelectedFrame

variable {Commitment PublicInput Scalar TranscriptState : Type}
variable {Digest AppState Witness Encoded RunningWitness FreshWitness : Type}
variable [DecidableEq AppState] [DecidableEq Encoded]
variable {shape : Shape} {columns blockCount degreeBound : Nat}
variable {oneFresh : shape.freshCount = 1}
variable {keys : Fin 1 ->
  SelectedKey K Commitment PublicInput Scalar TranscriptState
    shape columns blockCount degreeBound}
variable {defaultRunning : SelectedRunning K Commitment PublicInput shape}
variable {machine :
  Machine
    (SelectedKey K Commitment PublicInput Scalar TranscriptState
      shape columns blockCount degreeBound)
    Digest AppState Witness
    (SelectedRunning K Commitment PublicInput shape)
    (SelectedFresh Commitment PublicInput shape)
    Encoded 1}
variable {terminalRelations :
  TerminalRelations
    (SelectedKey K Commitment PublicInput Scalar TranscriptState
      shape columns blockCount degreeBound)
    (SelectedRunning K Commitment PublicInput shape)
    RunningWitness
    (SelectedFresh Commitment PublicInput shape)
    FreshWitness 1}
variable {terminalChecks :
  Nightstream.Protocol.FPrime.CanonicalTerminalVerifier.RelationChecks
    terminalRelations}
variable {widths : Widths} {footprints : Footprints}

local notation "Selected" =>
  SelectedParameters oneFresh keys defaultRunning machine terminalRelations
    terminalChecks widths footprints

theorem running_decodes_of_frame_decodes
    (family : Family (typeSystem Selected))
    {context : Schema (typeSystem Selected)}
    {runningRef :
      Ref (typeSystem Selected) context (.data .running)}
    {freshRef :
      Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef :
      Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame :
      CallFrame (signature := signature Selected) family Call.nifsVerify
        (Refs.cons runningRef
          (Refs.cons freshRef (Refs.cons proofRef .nil))))
    (assignment : ColumnId → Field)
    (running : SelectedRunning K Commitment PublicInput shape)
    (fresh : SelectedFresh Commitment PublicInput shape)
    (proof : SelectedProof K Commitment shape degreeBound)
    (decoded :
      frame.operands.Decodes family assignment
        (.cons running (.cons fresh (.cons proof .nil)))) :
    (family.codecFor (.data .running)).decode
        ((runningOperand frame.operands).values assignment) =
      some running :=
  (decodes_iff family assignment frame.operands running fresh proof).mp
    decoded |>.1

theorem fresh_decodes_of_frame_decodes
    (family : Family (typeSystem Selected))
    {context : Schema (typeSystem Selected)}
    {runningRef :
      Ref (typeSystem Selected) context (.data .running)}
    {freshRef :
      Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef :
      Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame :
      CallFrame (signature := signature Selected) family Call.nifsVerify
        (Refs.cons runningRef
          (Refs.cons freshRef (Refs.cons proofRef .nil))))
    (assignment : ColumnId → Field)
    (running : SelectedRunning K Commitment PublicInput shape)
    (fresh : SelectedFresh Commitment PublicInput shape)
    (proof : SelectedProof K Commitment shape degreeBound)
    (decoded :
      frame.operands.Decodes family assignment
        (.cons running (.cons fresh (.cons proof .nil)))) :
    (family.codecFor (.data .fresh)).decode
        ((freshOperand frame.operands).values assignment) =
      some fresh :=
  (decodes_iff family assignment frame.operands running fresh proof).mp
    decoded |>.2.1

theorem proof_decodes_of_frame_decodes
    (family : Family (typeSystem Selected))
    {context : Schema (typeSystem Selected)}
    {runningRef :
      Ref (typeSystem Selected) context (.data .running)}
    {freshRef :
      Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef :
      Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame :
      CallFrame (signature := signature Selected) family Call.nifsVerify
        (Refs.cons runningRef
          (Refs.cons freshRef (Refs.cons proofRef .nil))))
    (assignment : ColumnId → Field)
    (running : SelectedRunning K Commitment PublicInput shape)
    (fresh : SelectedFresh Commitment PublicInput shape)
    (proof : SelectedProof K Commitment shape degreeBound)
    (decoded :
      frame.operands.Decodes family assignment
        (.cons running (.cons fresh (.cons proof .nil)))) :
    (family.codecFor (.data .nifsProof)).decode
        ((proofOperand frame.operands).values assignment) =
      some proof :=
  (decodes_iff family assignment frame.operands running fresh proof).mp
    decoded |>.2.2

/-- **Selected-frame public `Pi_RLC` source binding.**

Every physical public-input column of the quotient occurrence is projected
from the one decoded running/fresh/proof frame and agrees with the exact
`K+k` semantic source order.  Per-operand decoding equations are derived
here; none is exposed to the caller. -/
theorem decodedPiRlcInput_eq
    (family : Family (typeSystem Selected))
    {context : Schema (typeSystem Selected)}
    {runningRef :
      Ref (typeSystem Selected) context (.data .running)}
    {freshRef :
      Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef :
      Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame :
      CallFrame (signature := signature Selected) family Call.nifsVerify
        (Refs.cons runningRef
          (Refs.cons freshRef (Refs.cons proofRef .nil))))
    (coordinates : CarrierCoordinates Commitment PublicInput)
    (views :
      Views shape degreeBound coordinates
        (family.codecFor (.data .running))
        (family.codecFor (.data .fresh))
        (family.codecFor (.data .nifsProof)))
    (coefficientWidth :
      shape.coefficientCount = Nightstream.SuperNeo.Concrete.ringDegree)
    (assignment : ColumnId → Field)
    (running : SelectedRunning K Commitment PublicInput shape)
    (fresh : SelectedFresh Commitment PublicInput shape)
    (proof : SelectedProof K Commitment shape degreeBound)
    (decoded :
      frame.operands.Decodes family assignment
        (.cons running (.cons fresh (.cons proof .nil))))
    (index :
      Fin
        (keys
          Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected).arity.total) :
    let profile :
        FrameViews frame shape coordinates degreeBound := {
      runningCodec := family.codecFor (.data .running)
      freshCodec := family.codecFor (.data .fresh)
      proofCodec := family.codecFor (.data .nifsProof)
      views := views
      runningWidthsAgree := running_widthsAgree frame
      freshWidthsAgree := fresh_widthsAgree frame
      proofWidthsAgree := proof_widthsAgree frame
      coefficientWidthAgree := coefficientWidth
    }
    let placement :=
      PaperNifsPiRlcSourceBinding.fromFrame frame shape profile
    let key :=
      keys Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected
    Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.decodeOpening
        (numericAssignment (columnMap frame) assignment)
        (placement.inputColumns key index) =
      piCcsOutputProjection coordinates key running fresh proof index := by
  dsimp only
  let profile :
      FrameViews frame shape coordinates degreeBound := {
    runningCodec := family.codecFor (.data .running)
    freshCodec := family.codecFor (.data .fresh)
    proofCodec := family.codecFor (.data .nifsProof)
    views := views
    runningWidthsAgree := running_widthsAgree frame
    freshWidthsAgree := fresh_widthsAgree frame
    proofWidthsAgree := proof_widthsAgree frame
    coefficientWidthAgree := coefficientWidth
  }
  let placement :=
    PaperNifsPiRlcSourceBinding.fromFrame frame shape profile
  let key :=
    keys Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected
  calc
    Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.decodeOpening
          (numericAssignment (columnMap frame) assignment)
          (placement.inputColumns key index) =
        profile.inputOpening key running fresh proof index :=
      placement.decoded_inputColumns_eq
        key assignment running fresh proof
        (running_decodes_of_frame_decodes family frame assignment
          running fresh proof decoded)
        (fresh_decodes_of_frame_decodes family frame assignment
          running fresh proof decoded)
        (proof_decodes_of_frame_decodes family frame assignment
          running fresh proof decoded)
        index
    _ = piCcsOutputProjection coordinates key running fresh proof index :=
      profile.inputOpening_eq_piCcsOutputProjection
        key running fresh proof index

/-- The selected paper occurrence's public verifier input is fixed by whole
call-frame decoding.  Neither a decoded running value nor any numeric
placement is supplied independently. -/
theorem decodedVerifierInput_eq_statement
    (family : Family (typeSystem Selected))
    {context : Schema (typeSystem Selected)}
    {runningRef :
      Ref (typeSystem Selected) context (.data .running)}
    {freshRef :
      Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef :
      Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame :
      CallFrame (signature := signature Selected) family Call.nifsVerify
        (Refs.cons runningRef
          (Refs.cons freshRef (Refs.cons proofRef .nil))))
    (views :
      RunningViews (family.codecFor (.data .running)))
    (template : KPiCcsTranscript.Input shape degreeBound)
    (assignment : ColumnId → Field)
    (running : SelectedRunning K Commitment PublicInput shape)
    (fresh : SelectedFresh Commitment PublicInput shape)
    (proof : SelectedProof K Commitment shape degreeBound)
    (decoded :
      frame.operands.Decodes family assignment
        (.cons running (.cons fresh (.cons proof .nil)))) :
    KPiCcsOccurrence.decodedVerifierInput
        (KPiCcsTranscript.occurrenceInput
          ((fromFrame frame views (running_widthsAgree frame)).bindInput
            (keys
              Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected)
            template))
        (numericAssignment (columnMap frame) assignment) =
      ((keys
          Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected).statement
        running fresh).verifierInput
          (keys
            Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected).lift := by
  exact
    PaperNifsPiCcsFramePlacement.decodedVerifierInput_eq_statement
      frame
      (keys Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected)
      views (running_widthsAgree frame) template assignment running fresh
      (running_decodes_of_frame_decodes family frame assignment
        running fresh proof decoded)

/-- The selected frame-level PiCCS occurrence reaches the paper event
theorem without a caller-supplied running decoder, numeric placement,
verifier input, or source-binding equality. -/
theorem rows_imply_tableTruth_or_paperBadEvent
    (family : Family (typeSystem Selected))
    {context : Schema (typeSystem Selected)}
    {runningRef :
      Ref (typeSystem Selected) context (.data .running)}
    {freshRef :
      Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef :
      Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame :
      CallFrame (signature := signature Selected) family Call.nifsVerify
        (Refs.cons runningRef
          (Refs.cons freshRef (Refs.cons proofRef .nil))))
    (views :
      RunningViews (family.codecFor (.data .running)))
    (constants : Constants)
    (template : KPiCcsTranscript.Input shape degreeBound)
    (assignment : ColumnId → Field)
    (running : SelectedRunning K Commitment PublicInput shape)
    (fresh : SelectedFresh Commitment PublicInput shape)
    (proof : SelectedProof K Commitment shape degreeBound)
    (witness : StrongReduction.OutputWitness shape columns)
    (decoded :
      frame.operands.Decodes family assignment
        (.cons running (.cons fresh (.cons proof .nil))))
    (constantWire : assignment frame.one = 1)
    (satisfied :
      Satisfies
        (KPiCcsTranscript.rows constants
          ((fromFrame frame views (running_widthsAgree frame)).bindInput
            (keys
              Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected)
            template))
        (numericAssignment (columnMap frame) assignment)) :
    let key :=
      keys Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected
    let placement := fromFrame frame views (running_widthsAgree frame)
    let input := placement.bindInput key template
    let pulled := numericAssignment (columnMap frame) assignment
    let data :=
      (key.statement running fresh).sourceProtocolData key.lift witness
    (TableResidualData.toTableObligations
        ConcreteCarrier.extensionOps
        (SignedCoefficientObject.toTableResidualData
          ConcreteCarrier.extensionOps
          (data.toJointData ConcreteCarrier.extensionOps))).AllHold ∨
      SignedCoefficientObject.MixingRoot ConcreteCarrier.extensionOps
        (data.toJointData ConcreteCarrier.extensionOps)
        (KPiCcsEventBinding.paperAlpha constants pulled input)
        (KPiCcsEventBinding.paperGamma constants pulled input) ∨
      ProtocolPolynomial.FixedWidth.SumCheckCollision
        ConcreteCarrier.extensionOps data
        (KPiCcsEventBinding.paperAlpha constants pulled input)
        (KPiCcsEventBinding.paperGamma constants pulled input)
        degreeBound key.challengeSetSize
        (KPiCcsEventBinding.paperPoint constants pulled input)
        (KPiCcsOccurrence.decodedCertificate
          (KPiCcsTranscript.occurrenceInput input) pulled) ∨
      ProtocolPolynomial.OutputMismatch ConcreteCarrier.extensionOps data
        (KPiCcsEventBinding.paperAlpha constants pulled input)
        (KPiCcsEventBinding.paperGamma constants pulled input)
        (KPiCcsEventBinding.paperPoint constants pulled input)
        (KPiCcsOccurrence.decodedMessage
          (KPiCcsTranscript.occurrenceInput input) pulled) := by
  dsimp only
  have decodedRunning :=
    running_decodes_of_frame_decodes family frame assignment
      running fresh proof decoded
  have mappedConstant :
      assignment (columnMap frame 0) = 1 := by
    simpa using constantWire
  exact
    PaperNifsPiCcsEventBinding.rows_imply_tableTruth_or_paperBadEvent
      (keys Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected)
      (fromFrame frame views (running_widthsAgree frame))
      constants template assignment running fresh witness decodedRunning
      mappedConstant satisfied

end SelectedFrame

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsSelectedCallFrame
