import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsSelectedCallFrame
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsGlobalColumnMap
import Nightstream.Implementation.R1CS.Canonical.KSplitNcEndpoints

/-!
Contract: bind the four public/message values consumed by the canonical
Split-NC endpoint rows to one decoded concrete `nifsVerify` proof operand.

The proof codec owns the exact coordinate order.  `ProofViews` names the two
codec coordinates of each semantic `K` value.  The resulting canonical
carried expressions use the sole global call-column map; they are reads of
the proof operand, never copied witness values.

`decodedAuthority_of_frame_decodes` constructs the endpoint decoder required
by `KSplitNcOperationalRows.accepted_of_rows` from whole-frame decoding.  It
does not accept endpoint equations, verifier acceptance, source authority, or
an independently decoded proof.

Emits constraints: none.
-/

set_option autoImplicit false
set_option maxHeartbeats 800000

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsOperationalFrame

open Nightstream.HyperNova.Construction2.Paper
open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.Goldilocks.NumericRowBridge
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.DirectCalls
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsCallFrame
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsCallColumnMap
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsGlobalColumnMap
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsParameters
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsSelectedCallFrame
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.OutputClaims
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.TranscriptAuthority.BlockLane

/-- Exact semantic prior-point coordinate stored in the selected proof. -/
def priorPointCoordinate
    {shape : SemanticShape}
    {TranscriptState : Type}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (coordinate : Fin shape.rowVariables)
    (proof :
      SelectedProof shape TranscriptState publicRingColumns publicFits
        verifierRows) : Nightstream.SuperNeo.Concrete.K :=
  proof.piCcsInput.priorPoint.coordinates.get
    ⟨coordinate.val, by
      rw [proof.piCcsInput.priorPoint.dimension]
      exact coordinate.isLt⟩

/-- Exact semantic carried evaluation stored in the selected public input. -/
def claimedYRingCoordinate
    {shape : SemanticShape}
    {TranscriptState : Type}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (running : Fin shape.runningCount)
    (matrix : Fin shape.matrixCount)
    (lane : Fin ringDegree)
    (proof :
      SelectedProof shape TranscriptState publicRingColumns publicFits
        verifierRows) : Nightstream.SuperNeo.Concrete.K :=
  proof.piCcsInput.claimedYRing running matrix lane

/-- Exact output-ring coordinate stored in the selected PiCCS certificate. -/
def outputYRingCoordinate
    {shape : SemanticShape}
    {TranscriptState : Type}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (source : Fin shape.sourceCount)
    (matrix : Fin shape.matrixCount)
    (lane : Fin ringDegree)
    (proof :
      SelectedProof shape TranscriptState publicRingColumns publicFits
        verifierRows) : Nightstream.SuperNeo.Concrete.K :=
  proof.certificate.piCcs.output.yRing source matrix lane

/-- Exact output old-point coordinate stored in the selected PiCCS
certificate. -/
def outputYZcolCoordinate
    {shape : SemanticShape}
    {TranscriptState : Type}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (source : Fin shape.sourceCount)
    (lane : Fin ringDegree)
    (proof :
      SelectedProof shape TranscriptState publicRingColumns publicFits
        verifierRows) : Nightstream.SuperNeo.Concrete.K :=
  proof.certificate.piCcs.output.yZcol source lane

/-- Codec-owned locations of every semantic value read by the endpoint
program.  These are representation laws only. -/
structure ProofViews
    {shape : SemanticShape}
    {TranscriptState : Type}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (codec :
      Codec
        (SelectedProof shape TranscriptState publicRingColumns publicFits
          verifierRows)) where
  priorPoint :
    ∀ coordinate : Fin shape.rowVariables,
      PaperNifsCodecProjection.KView codec (priorPointCoordinate coordinate)
  claimedYRing :
    ∀ running matrix lane,
      PaperNifsCodecProjection.KView codec
        (claimedYRingCoordinate running matrix lane)
  outputYRing :
    ∀ source matrix lane,
      PaperNifsCodecProjection.KView codec
        (outputYRingCoordinate source matrix lane)
  outputYZcol :
    ∀ source lane,
      PaperNifsCodecProjection.KView codec
        (outputYZcolCoordinate source lane)

section SelectedFrame

variable {shape : SemanticShape}
variable {TranscriptState : Type}
variable {Digest AppState Witness Encoded RunningWitness FreshWitness : Type}
variable [DecidableEq AppState] [DecidableEq Encoded]
variable {publicRingColumns verifierRows : Nat}
variable {publicFits :
  ringDegree * publicRingColumns <= shape.carrierWidth}
variable {keys : Fin 1 ->
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

/-- Locate one proof-codec `K` view in the sole numeric namespace. -/
def proofLocation
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
    {value :
      SelectedProof shape TranscriptState publicRingColumns publicFits
        verifierRows → Nightstream.SuperNeo.Concrete.K}
    (view :
      PaperNifsCodecProjection.KView
        (family.codecFor (.data .nifsProof)) value) :
    KLocation (columnMap frame)
      (view.columns (proofOperand frame.operands)
        (proof_widthsAgree frame)) := by
  apply kLocation frame
  · exact proofOperand_mem frame
      (view.c0_mem (proofOperand frame.operands)
        (proof_widthsAgree frame))
  · exact proofOperand_mem frame
      (view.c1_mem (proofOperand frame.operands)
        (proof_widthsAgree frame))

/-- Every endpoint proof view remains in the selected call frame's visible
prefix. -/
theorem proofLocation_numeric_lt
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
    {value :
      SelectedProof shape TranscriptState publicRingColumns publicFits
        verifierRows → Nightstream.SuperNeo.Concrete.K}
    (view :
      PaperNifsCodecProjection.KView
        (family.codecFor (.data .nifsProof)) value) :
    (proofLocation family frame view).numeric.c0 < temporaryBase frame ∧
      (proofLocation family frame view).numeric.c1 < temporaryBase frame := by
  unfold proofLocation
  apply kLocation_numeric_lt_temporaryBase
  · exact proofOperand_mem_visible frame
      (view.c0_mem (proofOperand frame.operands)
        (proof_widthsAgree frame))
  · exact proofOperand_mem_visible frame
      (view.c1_mem (proofOperand frame.operands)
        (proof_widthsAgree frame))

/-- Canonical carried expressions for all four endpoint authorities. -/
def authorityColumns
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
      ProofViews (family.codecFor (.data .nifsProof))) :
    KSplitNcEndpoints.AuthorityColumns shape where
  priorPoint coordinate :=
    (proofLocation family frame (views.priorPoint coordinate)).carried
  claimedYRing running matrix lane :=
    (proofLocation family frame
      (views.claimedYRing running matrix lane)).carried
  outputYRing source matrix lane :=
    (proofLocation family frame
      (views.outputYRing source matrix lane)).carried
  outputYZcol source lane :=
    (proofLocation family frame
      (views.outputYZcol source lane)).carried

theorem decodedView
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
    {value :
      SelectedProof shape TranscriptState publicRingColumns publicFits
        verifierRows → Nightstream.SuperNeo.Concrete.K}
    (view :
      PaperNifsCodecProjection.KView
        (family.codecFor (.data .nifsProof)) value)
    (assignment : ColumnId → Field)
    (proof :
      SelectedProof shape TranscriptState publicRingColumns publicFits
        verifierRows)
    (decoded :
      (proofOperand frame.operands).Decodes family (.data .nifsProof)
        assignment proof) :
    Nightstream.Implementation.R1CS.Canonical.KPointEquality.decoded
        (numericAssignment (columnMap frame) assignment)
        (proofLocation family frame view).carried =
      value proof := by
  calc
    Nightstream.Implementation.R1CS.Canonical.KPointEquality.decoded
          (numericAssignment (columnMap frame) assignment)
          (proofLocation family frame view).carried =
        (view.columns (proofOperand frame.operands)
          (proof_widthsAgree frame)).value assignment := by
      exact (proofLocation family frame view).decodeCarried_eq assignment
    _ = value proof :=
      view.value_eq_of_bundle_decodes family (.data .nifsProof)
        (proofOperand frame.operands) (proof_widthsAgree frame)
        assignment proof decoded

/-- Whole call-frame decoding constructs the exact authority relation
consumed by the endpoint rows.  No component decoding or semantic equation
is supplied independently. -/
theorem decodedAuthority_of_frame_decodes
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
      ProofViews (family.codecFor (.data .nifsProof)))
    (domains : Domains)
    (assignment : ColumnId → Field)
    (running :
      SelectedRunning shape publicRingColumns publicFits verifierRows)
    (fresh :
      SelectedFresh shape publicRingColumns publicFits verifierRows)
    (proof :
      SelectedProof shape TranscriptState publicRingColumns publicFits
        verifierRows)
    (transcript :
      KSplitNcTranscript.Input proof.piCcsInput domains)
    (frameBase : Nat)
    (decoded :
      frame.operands.Decodes family assignment
        (.cons running (.cons fresh (.cons proof .nil)))) :
    let endpointInput :
      KSplitNcEndpoints.Input proof.piCcsInput domains := {
        transcript := transcript
        authority := authorityColumns family frame views
        frameBase := frameBase
      }
    KSplitNcEndpoints.DecodedAuthority endpointInput
      (numericAssignment (columnMap frame) assignment)
      proof.certificate.piCcs.output := by
  dsimp only
  have proofDecoded :=
    ConcreteNifsSelectedCallFrame.proof_decodes_of_frame_decodes
      family frame assignment running fresh proof decoded
  constructor
  · intro coordinate
    exact decodedView family frame (views.priorPoint coordinate)
      assignment proof proofDecoded
  · intro runningIndex matrix lane
    exact decodedView family frame
      (views.claimedYRing runningIndex matrix lane)
      assignment proof proofDecoded
  · intro source matrix lane
    exact decodedView family frame
      (views.outputYRing source matrix lane)
      assignment proof proofDecoded
  · intro source lane
    exact decodedView family frame
      (views.outputYZcol source lane)
      assignment proof proofDecoded

end SelectedFrame

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsOperationalFrame
