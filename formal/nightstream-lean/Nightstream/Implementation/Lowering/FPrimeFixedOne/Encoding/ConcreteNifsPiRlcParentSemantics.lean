import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsPiRlcActionBridge
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsPiRlcPointSemantics

/-!
Contract: compose the selected PiRLC action and point-binding rows into one
exact outgoing-parent theorem.

The theorem starts from the whole typed call-frame decoder and physical row
satisfaction.  It derives all four fields of the verifier-owned
`ParentPayload` and then its materialized combined CE statement.  No parent,
point, source-authority proposition, or accepted result is supplied as a
semantic premise.
-/

set_option autoImplicit false
set_option maxRecDepth 100000
set_option maxHeartbeats 2000000

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsPiRlcParentSemantics

open Nightstream.HyperNova.Construction2.Paper
open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.DirectCalls
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsCallFrame
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsGlobalColumnMap
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsParameters
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsOperationalProfile
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
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

private def selectedKey :
    SelectedKey shape TranscriptState publicRingColumns publicFits
      verifierRows :=
  keys Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected

private def selectedContext
    (running :
      SelectedRunning shape publicRingColumns publicFits verifierRows)
    (fresh :
      SelectedFresh shape publicRingColumns publicFits verifierRows)
    (proof :
      SelectedProof shape TranscriptState publicRingColumns publicFits
        verifierRows) :=
  (ConcreteNifsParameters.context
    (selectedKey (keys := keys)) running fresh proof).materialize

/-- The payload view of the uniquely verifier-derived PiRLC parent. -/
def derivedParentPayload
    (running :
      SelectedRunning shape publicRingColumns publicFits verifierRows)
    (fresh :
      SelectedFresh shape publicRingColumns publicFits verifierRows)
    (proof :
      SelectedProof shape TranscriptState publicRingColumns publicFits
        verifierRows) :
    FixedActive.Canonical.ParentPayload
      shape publicRingColumns publicFits verifierRows :=
  let parent :=
    (ConcretePhi81.derive
      (selectedContext (keys := keys) running fresh proof)
      proof.certificate).piRlcOutput
  {
    commitment := parent.commitment
    publicInput := parent.publicInput
    point := parent.point
    evaluations := parent.evaluations
  }

private theorem parentPayload_eq
    (left right :
      FixedActive.Canonical.ParentPayload
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

/-- **Headline PiRLC parent refinement.** The action rows and point rows,
under the one authoritative call-frame decoder, determine the complete
outgoing parent payload. -/
theorem parent_eq_derived
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
        (NumericRowBridge.numericAssignment (columnMap frame) assignment))
    (actionSatisfied :
      RawSatisfies
        (ConcreteNifsPiRlcActionRows.rows
          application profile frame productBase)
        assignment)
    (pointSatisfied :
      Satisfies
        (ConcreteNifsPiRlcPointRows.rows application profile frame)
        (NumericRowBridge.numericAssignment (columnMap frame) assignment)) :
    output.parent =
      derivedParentPayload (keys := keys) running fresh proof := by
  have outputDecoded :=
    ConcreteNifsSelectedCallFrame.output_decodes_of_frame_decodes
      (FamilyFor application) frame assignment output decodedOutput
  have outputAdmissible :
      ((FamilyFor application).codecFor (.data .running)).Admissible output :=
    Codec.admissible_of_decode
      ((FamilyFor application).codecFor (.data .running)) outputDecoded
  have equations :=
    ConcreteNifsPiRlcActionSemantics.equations_of_rows
      application profile frame assignment running fresh proof output
      productBase constantWire decodedInputs decodedOutput actionSatisfied
  have commitment :=
    ConcreteNifsPiRlcActionBridge.commitment_eq_derived
      (keys := keys) running fresh proof output equations
  have publicInput :=
    ConcreteNifsPiRlcActionBridge.publicInput_eq_derived
      (keys := keys) running fresh proof output equations
  have point :=
    ConcreteNifsPiRlcPointSemantics.point_eq_derived
      application profile frame assignment running fresh proof output
      constantWire decodedInputs decodedOutput operationalSatisfied
      pointSatisfied
  have evaluations :=
    ConcreteNifsPiRlcActionBridge.evaluations_eq_derived
      (keys := keys) application profile running fresh proof output
      outputAdmissible equations
  apply parentPayload_eq
  · change
      output.parent.commitment =
        (ConcretePhi81.derive
          (selectedContext (keys := keys) running fresh proof)
          proof.certificate).piRlcOutput.commitment
    exact commitment
  · change
      output.parent.publicInput =
        (ConcretePhi81.derive
          (selectedContext (keys := keys) running fresh proof)
          proof.certificate).piRlcOutput.publicInput
    exact publicInput
  · change
      output.parent.point =
        (ConcretePhi81.derive
          (selectedContext (keys := keys) running fresh proof)
          proof.certificate).piRlcOutput.point
    exact point
  · change
      output.parent.evaluations =
        (ConcretePhi81.derive
          (selectedContext (keys := keys) running fresh proof)
          proof.certificate).piRlcOutput.evaluations
    exact evaluations

/-- Materializing the payload with the selected relation structure yields the
exact combined CE statement computed by frozen `ConcretePhi81.derive`. -/
theorem materialized_parent_eq_derived
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
        (NumericRowBridge.numericAssignment (columnMap frame) assignment))
    (actionSatisfied :
      RawSatisfies
        (ConcreteNifsPiRlcActionRows.rows
          application profile frame productBase)
        assignment)
    (pointSatisfied :
      Satisfies
        (ConcreteNifsPiRlcPointRows.rows application profile frame)
        (NumericRowBridge.numericAssignment (columnMap frame) assignment)) :
    output.parent.materialize (selectedKey (keys := keys)).system =
      (ConcretePhi81.derive
        (selectedContext (keys := keys) running fresh proof)
        proof.certificate).piRlcOutput := by
  rw [parent_eq_derived application profile frame assignment
    running fresh proof output productBase constantWire decodedInputs
    decodedOutput operationalSatisfied actionSatisfied pointSatisfied]
  rfl

end SelectedFrame

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsPiRlcParentSemantics
