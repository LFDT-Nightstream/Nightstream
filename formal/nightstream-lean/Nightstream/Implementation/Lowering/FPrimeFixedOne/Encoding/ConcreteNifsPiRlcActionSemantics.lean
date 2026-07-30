import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsPiRlcActionRows
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsSelectedCallFrame
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.Phi81RadixRows
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.Phi81RingActionSemantics

/-!
Contract: exact semantic decoding of the selected fixed-active `Pi_RLC`
parent-action rows.

Whole-frame decoding supplies the sole running, fresh, proof, and output
values.  Row satisfaction then fixes every parent commitment ring, public
ring, and both coordinates of every extension-valued evaluation ring to the
same fifteen-source Phi81 action.

This module does not own activation, challenge sampling, `Pi_CCS` acceptance,
the point/output-child bindings, outgoing radix recomposition, or the final
`PiRLC.Equations` assembly.
-/

set_option autoImplicit false
set_option maxRecDepth 100000
set_option maxHeartbeats 2000000

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsPiRlcActionSemantics

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
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsPiRlcActionRows
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81

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

/-! ## Semantic source order -/

/-- Commitment rings in the canonical fresh-then-running source order. -/
def commitmentSource
    (running :
      SelectedRunning shape publicRingColumns publicFits verifierRows)
    (fresh :
      SelectedFresh shape publicRingColumns publicFits verifierRows)
    (row : Fin verifierRows) :
    Fin FixedActive.arity.total → RingF :=
  Fin.addCases
    (fun _ => fresh.commitment row)
    (fun child => (running.children child).commitment row)

/-- Public-input rings in the canonical fresh-then-running source order. -/
def publicSource
    (running :
      SelectedRunning shape publicRingColumns publicFits verifierRows)
    (fresh :
      SelectedFresh shape publicRingColumns publicFits verifierRows)
    (block : Fin publicRingColumns) :
    Fin FixedActive.arity.total → RingF :=
  Fin.addCases
    (fun _ lane =>
      fresh.publicInput
        (ConcreteNifsPiRlcActionRows.publicCoordinate block lane))
    (fun child lane =>
      (running.children child).publicInput
        (ConcreteNifsPiRlcActionRows.publicCoordinate block lane))

/-- One exact `Pi_CCS` output evaluation ring in canonical source order. -/
def evaluationSource
    (proof :
      SelectedProof shape TranscriptState publicRingColumns publicFits
        verifierRows)
    (matrix : Fin shape.matrixCount)
    (source : Fin FixedActive.arity.total) : RingK :=
  fun lane =>
    proof.certificate.piCcs.output.yRing
      ((keys
        Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected).template.alignment.semanticIndex
          source)
      matrix lane

/-- Exact coordinate equations enforced by the action slice. -/
structure Equations
    (running :
      SelectedRunning shape publicRingColumns publicFits verifierRows)
    (fresh :
      SelectedFresh shape publicRingColumns publicFits verifierRows)
    (proof :
      SelectedProof shape TranscriptState publicRingColumns publicFits
        verifierRows)
    (output :
      SelectedRunning shape publicRingColumns publicFits verifierRows) :
    Prop where
  commitment :
    ∀ row,
      output.parent.commitment row =
        Phi81RingAction.combine proof.certificate.piRlcChallenges
          (commitmentSource running fresh row)
  publicInput :
    ∀ block,
      (fun lane =>
        output.parent.publicInput
          (ConcreteNifsPiRlcActionRows.publicCoordinate block lane)) =
        Phi81RingAction.combine proof.certificate.piRlcChallenges
          (publicSource running fresh block)
  evaluationLow :
    ∀ matrix lane,
      (output.parent.evaluations.getD
        matrix.val ringKZero lane).c0 =
        Phi81RingAction.combine proof.certificate.piRlcChallenges
          (fun source position =>
            (evaluationSource (keys := keys) proof matrix source position).c0)
          lane
  evaluationHigh :
    ∀ matrix lane,
      (output.parent.evaluations.getD
        matrix.val ringKZero lane).c1 =
        Phi81RingAction.combine proof.certificate.piRlcChallenges
          (fun source position =>
            (evaluationSource (keys := keys) proof matrix source position).c1)
          lane

/-- The same four parent equations before assuming that the output bundle
decodes.  The left sides are the physical output columns read directly by the
action rows.  This is the form needed to prove output decoding rather than
assuming it. -/
structure PhysicalEquations
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
        verifierRows) : Prop where
  commitment :
    ∀ row,
      Phi81RingAction.decoded assignment
          (ConcreteNifsPiRlcActionRows.commitmentOutput
            application profile frame row) =
        Phi81RingAction.combine proof.certificate.piRlcChallenges
          (commitmentSource running fresh row)
  publicInput :
    ∀ block,
      Phi81RingAction.decoded assignment
          (ConcreteNifsPiRlcActionRows.publicOutput
            application profile frame block) =
        Phi81RingAction.combine proof.certificate.piRlcChallenges
          (publicSource running fresh block)
  evaluationLow :
    ∀ matrix,
      Phi81RingAction.decoded assignment
          (ConcreteNifsPiRlcActionRows.evaluationOutputLow
            application profile frame matrix) =
        fun lane =>
          Phi81RingAction.combine proof.certificate.piRlcChallenges
            (fun source position =>
              (evaluationSource
                (keys := keys) proof matrix source position).c0)
            lane
  evaluationHigh :
    ∀ matrix,
      Phi81RingAction.decoded assignment
          (ConcreteNifsPiRlcActionRows.evaluationOutputHigh
            application profile frame matrix) =
        fun lane =>
          Phi81RingAction.combine proof.certificate.piRlcChallenges
            (fun source position =>
              (evaluationSource
                (keys := keys) proof matrix source position).c1)
            lane

/-! ## Decoding lemmas -/

private theorem carriedF_eval
    (application : Poseidon23ApplicationProfile Selected)
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
    (source : List (Nat × Nat)) :
    (ConcreteNifsPiRlcActionRows.carriedF application frame source).eval
        assignment =
      residue
        (Nightstream.Implementation.R1CS.lcEval
          (numericAssignment (columnMap frame) assignment) source) := by
  exact terms_eval_eq_residue_lcEval (columnMap frame) assignment source

private theorem decoded_challenge
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
        verifierRows)
    (proofDecoded :
      (proofOperand frame.operands).Decodes
        (FamilyFor application) (.data .nifsProof) assignment proof)
    (source : Fin FixedActive.arity.total) :
    Phi81RingAction.decoded assignment
        (ConcreteNifsPiRlcActionRows.challenge
          application profile frame source) =
      proof.certificate.piRlcChallenges source := by
  funext lane
  unfold Phi81RingAction.decoded
  unfold ConcreteNifsPiRlcActionRows.challenge
  rw [carriedF_eval]
  exact ConcreteNifsCarrierFrame.proofF_decoded
    (FamilyFor application) frame
    (profile.samplerViews.challenge source lane)
    assignment proof proofDecoded

private theorem decoded_commitmentValue
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
    (runningDecoded :
      (runningOperand frame.operands).Decodes
        (FamilyFor application) (.data .running) assignment running)
    (freshDecoded :
      (freshOperand frame.operands).Decodes
        (FamilyFor application) (.data .fresh) assignment fresh)
    (row : Fin verifierRows)
    (source : Fin FixedActive.arity.total) :
    Phi81RingAction.decoded assignment
        (ConcreteNifsPiRlcActionRows.commitmentValue
          application profile frame row source) =
      commitmentSource running fresh row source := by
  refine Fin.addCases ?_ ?_ source
  · intro freshIndex
    funext lane
    unfold Phi81RingAction.decoded
    simp only [ConcreteNifsPiRlcActionRows.commitmentValue,
      commitmentSource, Fin.addCases_left]
    rw [carriedF_eval]
    exact ConcreteNifsCarrierFrame.freshF_decoded
      (FamilyFor application) frame
      (profile.freshViews.commitment row lane)
      assignment fresh freshDecoded
  · intro child
    funext lane
    unfold Phi81RingAction.decoded
    simp only [ConcreteNifsPiRlcActionRows.commitmentValue,
      commitmentSource, Fin.addCases_right]
    rw [carriedF_eval]
    exact ConcreteNifsCarrierFrame.runningF_decoded
      (FamilyFor application) frame
      (profile.runningViews.childCommitment child row lane)
      assignment running runningDecoded

private theorem decoded_publicValue
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
    (runningDecoded :
      (runningOperand frame.operands).Decodes
        (FamilyFor application) (.data .running) assignment running)
    (freshDecoded :
      (freshOperand frame.operands).Decodes
        (FamilyFor application) (.data .fresh) assignment fresh)
    (block : Fin publicRingColumns)
    (source : Fin FixedActive.arity.total) :
    Phi81RingAction.decoded assignment
        (ConcreteNifsPiRlcActionRows.publicValue
          application profile frame block source) =
      publicSource running fresh block source := by
  refine Fin.addCases ?_ ?_ source
  · intro freshIndex
    funext lane
    unfold Phi81RingAction.decoded
    simp only [ConcreteNifsPiRlcActionRows.publicValue,
      publicSource, Fin.addCases_left]
    rw [carriedF_eval]
    exact ConcreteNifsCarrierFrame.freshF_decoded
      (FamilyFor application) frame
      (profile.freshViews.publicInput
        (ConcreteNifsPiRlcActionRows.publicCoordinate block lane))
      assignment fresh freshDecoded
  · intro child
    funext lane
    unfold Phi81RingAction.decoded
    simp only [ConcreteNifsPiRlcActionRows.publicValue,
      publicSource, Fin.addCases_right]
    rw [carriedF_eval]
    exact ConcreteNifsCarrierFrame.runningF_decoded
      (FamilyFor application) frame
      (profile.runningViews.childPublic child
        (ConcreteNifsPiRlcActionRows.publicCoordinate block lane))
      assignment running runningDecoded

private theorem decoded_evaluationValueLow
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
        verifierRows)
    (proofDecoded :
      (proofOperand frame.operands).Decodes
        (FamilyFor application) (.data .nifsProof) assignment proof)
    (matrix : Fin shape.matrixCount)
    (source : Fin FixedActive.arity.total) :
    Phi81RingAction.decoded assignment
        (ConcreteNifsPiRlcActionRows.evaluationValueLow
          application profile frame matrix source) =
      fun lane =>
        (evaluationSource (keys := keys) proof matrix source lane).c0 := by
  funext lane
  unfold Phi81RingAction.decoded
  unfold ConcreteNifsPiRlcActionRows.evaluationValueLow
  rw [carriedF_eval,
    Phi81RadixRows.residue_lcEval_eq_decoded_c0]
  exact congrArg K.c0
    (ConcreteNifsCarrierFrame.proofK_decoded
      (FamilyFor application) frame
      (profile.endpointViews.outputYRing
        ((keys
          Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected).template.alignment.semanticIndex
            source)
        matrix lane)
      assignment proof proofDecoded)

private theorem decoded_evaluationValueHigh
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
        verifierRows)
    (proofDecoded :
      (proofOperand frame.operands).Decodes
        (FamilyFor application) (.data .nifsProof) assignment proof)
    (matrix : Fin shape.matrixCount)
    (source : Fin FixedActive.arity.total) :
    Phi81RingAction.decoded assignment
        (ConcreteNifsPiRlcActionRows.evaluationValueHigh
          application profile frame matrix source) =
      fun lane =>
        (evaluationSource (keys := keys) proof matrix source lane).c1 := by
  funext lane
  unfold Phi81RingAction.decoded
  unfold ConcreteNifsPiRlcActionRows.evaluationValueHigh
  rw [carriedF_eval,
    Phi81RadixRows.residue_lcEval_eq_decoded_c1]
  exact congrArg K.c1
    (ConcreteNifsCarrierFrame.proofK_decoded
      (FamilyFor application) frame
      (profile.endpointViews.outputYRing
        ((keys
          Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected).template.alignment.semanticIndex
            source)
        matrix lane)
      assignment proof proofDecoded)

private theorem decoded_commitmentOutput
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
    (output :
      SelectedRunning shape publicRingColumns publicFits verifierRows)
    (outputDecoded :
      (unaryOutput frame.outputs).Decodes
        (FamilyFor application) (.data .running) assignment output)
    (row : Fin verifierRows) :
    Phi81RingAction.decoded assignment
        (ConcreteNifsPiRlcActionRows.commitmentOutput
          application profile frame row) =
      output.parent.commitment row := by
  funext lane
  unfold Phi81RingAction.decoded
  unfold ConcreteNifsPiRlcActionRows.commitmentOutput
  rw [carriedF_eval]
  exact ConcreteNifsCarrierFrame.outputF_decoded
    (FamilyFor application) frame
    (profile.runningViews.parentCommitment row lane)
    assignment output outputDecoded

private theorem decoded_publicOutput
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
    (output :
      SelectedRunning shape publicRingColumns publicFits verifierRows)
    (outputDecoded :
      (unaryOutput frame.outputs).Decodes
        (FamilyFor application) (.data .running) assignment output)
    (block : Fin publicRingColumns) :
    Phi81RingAction.decoded assignment
        (ConcreteNifsPiRlcActionRows.publicOutput
          application profile frame block) =
      fun lane =>
        output.parent.publicInput
          (ConcreteNifsPiRlcActionRows.publicCoordinate block lane) := by
  funext lane
  unfold Phi81RingAction.decoded
  unfold ConcreteNifsPiRlcActionRows.publicOutput
  rw [carriedF_eval]
  exact ConcreteNifsCarrierFrame.outputF_decoded
    (FamilyFor application) frame
    (profile.runningViews.parentPublic
      (ConcreteNifsPiRlcActionRows.publicCoordinate block lane))
    assignment output outputDecoded

private theorem decoded_evaluationOutputLow
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
    (output :
      SelectedRunning shape publicRingColumns publicFits verifierRows)
    (outputDecoded :
      (unaryOutput frame.outputs).Decodes
        (FamilyFor application) (.data .running) assignment output)
    (matrix : Fin shape.matrixCount) :
    Phi81RingAction.decoded assignment
        (ConcreteNifsPiRlcActionRows.evaluationOutputLow
          application profile frame matrix) =
      fun lane =>
        (output.parent.evaluations.getD
          matrix.val ringKZero lane).c0 := by
  funext lane
  unfold Phi81RingAction.decoded
  unfold ConcreteNifsPiRlcActionRows.evaluationOutputLow
  rw [carriedF_eval,
    Phi81RadixRows.residue_lcEval_eq_decoded_c0]
  exact congrArg K.c0
    (ConcreteNifsCarrierFrame.outputK_decoded
      (FamilyFor application) frame
      (profile.runningViews.parentEvaluation matrix lane)
      assignment output outputDecoded)

private theorem decoded_evaluationOutputHigh
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
    (output :
      SelectedRunning shape publicRingColumns publicFits verifierRows)
    (outputDecoded :
      (unaryOutput frame.outputs).Decodes
        (FamilyFor application) (.data .running) assignment output)
    (matrix : Fin shape.matrixCount) :
    Phi81RingAction.decoded assignment
        (ConcreteNifsPiRlcActionRows.evaluationOutputHigh
          application profile frame matrix) =
      fun lane =>
        (output.parent.evaluations.getD
          matrix.val ringKZero lane).c1 := by
  funext lane
  unfold Phi81RingAction.decoded
  unfold ConcreteNifsPiRlcActionRows.evaluationOutputHigh
  rw [carriedF_eval,
    Phi81RadixRows.residue_lcEval_eq_decoded_c1]
  exact congrArg K.c1
    (ConcreteNifsCarrierFrame.outputK_decoded
      (FamilyFor application) frame
      (profile.runningViews.parentEvaluation matrix lane)
      assignment output outputDecoded)

/-! ## Honest frame equations -/

/-- Every selected action frame inherits its exact visible semantic equation
from the decoded call inputs and the selected output-parent equations.  This
is the semantic half of whole-slice honest completion; product-cell placement
and witness construction remain separate. -/
theorem frame_semantic_of_equations
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
    (decodedInputs :
      frame.operands.Decodes (FamilyFor application) assignment
        (.cons running (.cons fresh (.cons proof .nil))))
    (decodedOutput :
      frame.outputs.Decodes (FamilyFor application) assignment
        (.cons output .nil))
    (equations :
      Equations (keys := keys) running fresh proof output)
    (target : Phi81RingAction.Frame FixedActive.arity.total)
    (member :
      target ∈ ConcreteNifsPiRlcActionRows.frames
        application profile frame productBase) :
    Phi81RingAction.decoded assignment target.output =
      Phi81RingAction.combine
        (fun source =>
          Phi81RingAction.decoded assignment
            (target.challenges source))
        (fun source =>
          Phi81RingAction.decoded assignment (target.values source)) := by
  have runningDecoded :=
    ConcreteNifsSelectedCallFrame.running_decodes_of_frame_decodes
      (FamilyFor application) frame assignment running fresh proof
      decodedInputs
  have freshDecoded :=
    ConcreteNifsSelectedCallFrame.fresh_decodes_of_frame_decodes
      (FamilyFor application) frame assignment running fresh proof
      decodedInputs
  have proofDecoded :=
    ConcreteNifsSelectedCallFrame.proof_decodes_of_frame_decodes
      (FamilyFor application) frame assignment running fresh proof
      decodedInputs
  have outputDecoded :=
    ConcreteNifsSelectedCallFrame.output_decodes_of_frame_decodes
      (FamilyFor application) frame assignment output decodedOutput
  unfold ConcreteNifsPiRlcActionRows.frames at member
  rcases List.mem_append.1 member with firstThree | inHigh
  rcases List.mem_append.1 firstThree with firstTwo | inLow
  rcases List.mem_append.1 firstTwo with inCommitment | inPublic
  · rcases List.mem_ofFn.1 inCommitment with ⟨row, rfl⟩
    change
      Phi81RingAction.decoded assignment
          (ConcreteNifsPiRlcActionRows.commitmentOutput
            application profile frame row) =
        Phi81RingAction.combine
          (fun source =>
            Phi81RingAction.decoded assignment
              (ConcreteNifsPiRlcActionRows.challenge
                application profile frame source))
          (fun source =>
            Phi81RingAction.decoded assignment
              (ConcreteNifsPiRlcActionRows.commitmentValue
                application profile frame row source))
    calc
      _ = output.parent.commitment row :=
        decoded_commitmentOutput application profile frame assignment
          output outputDecoded row
      _ = Phi81RingAction.combine proof.certificate.piRlcChallenges
          (commitmentSource running fresh row) :=
        equations.commitment row
      _ = _ := by
        have challengesExact :
            (fun source =>
              Phi81RingAction.decoded assignment
                (ConcreteNifsPiRlcActionRows.challenge
                  application profile frame source)) =
              proof.certificate.piRlcChallenges := by
          funext source
          exact decoded_challenge application profile frame assignment proof
            proofDecoded source
        have valuesExact :
            (fun source =>
              Phi81RingAction.decoded assignment
                (ConcreteNifsPiRlcActionRows.commitmentValue
                  application profile frame row source)) =
              commitmentSource running fresh row := by
          funext source
          exact decoded_commitmentValue application profile frame assignment
            running fresh runningDecoded freshDecoded row source
        rw [challengesExact, valuesExact]
  · rcases List.mem_ofFn.1 inPublic with ⟨block, rfl⟩
    change
      Phi81RingAction.decoded assignment
          (ConcreteNifsPiRlcActionRows.publicOutput
            application profile frame block) =
        Phi81RingAction.combine
          (fun source =>
            Phi81RingAction.decoded assignment
              (ConcreteNifsPiRlcActionRows.challenge
                application profile frame source))
          (fun source =>
            Phi81RingAction.decoded assignment
              (ConcreteNifsPiRlcActionRows.publicValue
                application profile frame block source))
    calc
      _ = fun lane =>
          output.parent.publicInput
            (ConcreteNifsPiRlcActionRows.publicCoordinate block lane) :=
        decoded_publicOutput application profile frame assignment
          output outputDecoded block
      _ = Phi81RingAction.combine proof.certificate.piRlcChallenges
          (publicSource running fresh block) :=
        equations.publicInput block
      _ = _ := by
        have challengesExact :
            (fun source =>
              Phi81RingAction.decoded assignment
                (ConcreteNifsPiRlcActionRows.challenge
                  application profile frame source)) =
              proof.certificate.piRlcChallenges := by
          funext source
          exact decoded_challenge application profile frame assignment proof
            proofDecoded source
        have valuesExact :
            (fun source =>
              Phi81RingAction.decoded assignment
                (ConcreteNifsPiRlcActionRows.publicValue
                  application profile frame block source)) =
              publicSource running fresh block := by
          funext source
          exact decoded_publicValue application profile frame assignment
            running fresh runningDecoded freshDecoded block source
        rw [challengesExact, valuesExact]
  · rcases List.mem_ofFn.1 inLow with ⟨matrix, rfl⟩
    funext lane
    change
      Phi81RingAction.decoded assignment
          (ConcreteNifsPiRlcActionRows.evaluationOutputLow
            application profile frame matrix) lane =
        Phi81RingAction.combine
          (fun source =>
            Phi81RingAction.decoded assignment
              (ConcreteNifsPiRlcActionRows.challenge
                application profile frame source))
          (fun source =>
            Phi81RingAction.decoded assignment
              (ConcreteNifsPiRlcActionRows.evaluationValueLow
                application profile frame matrix source))
          lane
    calc
      _ = (output.parent.evaluations.getD
          matrix.val ringKZero lane).c0 :=
        congrFun
          (decoded_evaluationOutputLow application profile frame assignment
            output outputDecoded matrix) lane
      _ = Phi81RingAction.combine proof.certificate.piRlcChallenges
          (fun source position =>
            (evaluationSource
              (keys := keys) proof matrix source position).c0)
          lane :=
        equations.evaluationLow matrix lane
      _ = _ := by
        have challengesExact :
            (fun source =>
              Phi81RingAction.decoded assignment
                (ConcreteNifsPiRlcActionRows.challenge
                  application profile frame source)) =
              proof.certificate.piRlcChallenges := by
          funext source
          exact decoded_challenge application profile frame assignment proof
            proofDecoded source
        have valuesExact :
            (fun source =>
              Phi81RingAction.decoded assignment
                (ConcreteNifsPiRlcActionRows.evaluationValueLow
                  application profile frame matrix source)) =
              (fun source position =>
                (evaluationSource
                  (keys := keys) proof matrix source position).c0) := by
          funext source
          exact decoded_evaluationValueLow application profile frame
            assignment proof proofDecoded matrix source
        rw [challengesExact, valuesExact]
  · rcases List.mem_ofFn.1 inHigh with ⟨matrix, rfl⟩
    funext lane
    change
      Phi81RingAction.decoded assignment
          (ConcreteNifsPiRlcActionRows.evaluationOutputHigh
            application profile frame matrix) lane =
        Phi81RingAction.combine
          (fun source =>
            Phi81RingAction.decoded assignment
              (ConcreteNifsPiRlcActionRows.challenge
                application profile frame source))
          (fun source =>
            Phi81RingAction.decoded assignment
              (ConcreteNifsPiRlcActionRows.evaluationValueHigh
                application profile frame matrix source))
          lane
    calc
      _ = (output.parent.evaluations.getD
          matrix.val ringKZero lane).c1 :=
        congrFun
          (decoded_evaluationOutputHigh application profile frame assignment
            output outputDecoded matrix) lane
      _ = Phi81RingAction.combine proof.certificate.piRlcChallenges
          (fun source position =>
            (evaluationSource
              (keys := keys) proof matrix source position).c1)
          lane :=
        equations.evaluationHigh matrix lane
      _ = _ := by
        have challengesExact :
            (fun source =>
              Phi81RingAction.decoded assignment
                (ConcreteNifsPiRlcActionRows.challenge
                  application profile frame source)) =
              proof.certificate.piRlcChallenges := by
          funext source
          exact decoded_challenge application profile frame assignment proof
            proofDecoded source
        have valuesExact :
            (fun source =>
              Phi81RingAction.decoded assignment
                (ConcreteNifsPiRlcActionRows.evaluationValueHigh
                  application profile frame matrix source)) =
              (fun source position =>
                (evaluationSource
                  (keys := keys) proof matrix source position).c1) := by
          funext source
          exact decoded_evaluationValueHigh application profile frame
            assignment proof proofDecoded matrix source
        rw [challengesExact, valuesExact]

/-! ## Whole-slice soundness -/

private theorem rawSatisfies_member
    {source : List Row}
    {assignment : ColumnId → Field}
    (satisfied : RawSatisfies source assignment)
    {row : Row}
    (member : row ∈ source) :
    row.Holds assignment := by
  induction source with
  | nil =>
      simp at member
  | cons head tail inductionHypothesis =>
      rcases List.mem_cons.mp member with rfl | tailMember
      · exact satisfied.1
      · exact inductionHypothesis satisfied.2 tailMember

private theorem rawSatisfies_of_forall
    (source : List Row)
    (assignment : ColumnId → Field)
    (holds : ∀ row ∈ source, row.Holds assignment) :
    RawSatisfies source assignment := by
  induction source with
  | nil =>
      exact True.intro
  | cons head tail inductionHypothesis =>
      exact ⟨holds head (by simp),
        inductionHypothesis fun row member => holds row (by simp [member])⟩

private theorem frame_satisfied
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
    (productBase : Nat)
    (satisfied :
      RawSatisfies
        (ConcreteNifsPiRlcActionRows.rows
          application profile frame productBase)
        assignment)
    (target : Phi81RingAction.Frame FixedActive.arity.total)
    (member :
      target ∈
        ConcreteNifsPiRlcActionRows.frames
          application profile frame productBase) :
    Satisfies (Phi81RingAction.rows target) assignment := by
  apply (Phi81RingAction.satisfies_rows_iff target assignment).2
  apply rawSatisfies_of_forall
  intro row rowMember
  exact rawSatisfies_member satisfied
    (List.mem_flatMap.2 ⟨target, member, rowMember⟩)

private theorem commitmentFrame_mem
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
    (productBase : Nat)
    (row : Fin verifierRows) :
    ConcreteNifsPiRlcActionRows.commitmentFrame
        application profile frame productBase row ∈
      ConcreteNifsPiRlcActionRows.frames
        application profile frame productBase := by
  unfold ConcreteNifsPiRlcActionRows.frames
  apply List.mem_append_left
  apply List.mem_append_left
  apply List.mem_append_left
  exact List.mem_ofFn.2 ⟨row, rfl⟩

private theorem publicFrame_mem
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
    (productBase : Nat)
    (block : Fin publicRingColumns) :
    ConcreteNifsPiRlcActionRows.publicFrame
        application profile frame productBase block ∈
      ConcreteNifsPiRlcActionRows.frames
        application profile frame productBase := by
  unfold ConcreteNifsPiRlcActionRows.frames
  apply List.mem_append_left
  apply List.mem_append_left
  apply List.mem_append_right
  exact List.mem_ofFn.2 ⟨block, rfl⟩

private theorem evaluationLowFrame_mem
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
    (productBase : Nat)
    (matrix : Fin shape.matrixCount) :
    ConcreteNifsPiRlcActionRows.evaluationLowFrame
        application profile frame productBase matrix ∈
      ConcreteNifsPiRlcActionRows.frames
        application profile frame productBase := by
  unfold ConcreteNifsPiRlcActionRows.frames
  apply List.mem_append_left
  apply List.mem_append_right
  exact List.mem_ofFn.2 ⟨matrix, rfl⟩

private theorem evaluationHighFrame_mem
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
    (productBase : Nat)
    (matrix : Fin shape.matrixCount) :
    ConcreteNifsPiRlcActionRows.evaluationHighFrame
        application profile frame productBase matrix ∈
      ConcreteNifsPiRlcActionRows.frames
        application profile frame productBase := by
  unfold ConcreteNifsPiRlcActionRows.frames
  apply List.mem_append_right
  exact List.mem_ofFn.2 ⟨matrix, rfl⟩

/-- Satisfaction of the action slice determines every physical output-parent
coordinate before the output codec is decoded. -/
theorem physical_equations_of_rows
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
    (productBase : Nat)
    (constantWire : assignment frame.one = 1)
    (decodedInputs :
      frame.operands.Decodes (FamilyFor application) assignment
        (.cons running (.cons fresh (.cons proof .nil))))
    (satisfied :
      RawSatisfies
        (ConcreteNifsPiRlcActionRows.rows
          application profile frame productBase)
        assignment) :
    PhysicalEquations application profile frame assignment
      running fresh proof := by
  have runningDecoded :=
    ConcreteNifsSelectedCallFrame.running_decodes_of_frame_decodes
      (FamilyFor application) frame assignment running fresh proof
      decodedInputs
  have freshDecoded :=
    ConcreteNifsSelectedCallFrame.fresh_decodes_of_frame_decodes
      (FamilyFor application) frame assignment running fresh proof
      decodedInputs
  have proofDecoded :=
    ConcreteNifsSelectedCallFrame.proof_decodes_of_frame_decodes
      (FamilyFor application) frame assignment running fresh proof
      decodedInputs
  refine {
    commitment := ?_
    publicInput := ?_
    evaluationLow := ?_
    evaluationHigh := ?_
  }
  · intro row
    let target :=
      ConcreteNifsPiRlcActionRows.commitmentFrame
        application profile frame productBase row
    have targetSatisfied :=
      frame_satisfied application profile frame assignment productBase
        satisfied target
        (commitmentFrame_mem application profile frame productBase row)
    have exact :=
      Phi81RingAction.rows_sound target assignment constantWire
        targetSatisfied
    dsimp [target, ConcreteNifsPiRlcActionRows.commitmentFrame,
      ConcreteNifsPiRlcActionRows.actionFrame] at exact
    have challengesExact :
        (fun source =>
          Phi81RingAction.decoded assignment
            (ConcreteNifsPiRlcActionRows.challenge
              application profile frame source)) =
          proof.certificate.piRlcChallenges := by
      funext source
      exact decoded_challenge application profile frame assignment proof
        proofDecoded source
    have valuesExact :
        (fun source =>
          Phi81RingAction.decoded assignment
            (ConcreteNifsPiRlcActionRows.commitmentValue
              application profile frame row source)) =
          commitmentSource running fresh row := by
      funext source
      exact decoded_commitmentValue application profile frame assignment
        running fresh runningDecoded freshDecoded row source
    rw [← challengesExact, ← valuesExact]
    exact exact
  · intro block
    let target :=
      ConcreteNifsPiRlcActionRows.publicFrame
        application profile frame productBase block
    have targetSatisfied :=
      frame_satisfied application profile frame assignment productBase
        satisfied target
        (publicFrame_mem application profile frame productBase block)
    have exact :=
      Phi81RingAction.rows_sound target assignment constantWire
        targetSatisfied
    dsimp [target, ConcreteNifsPiRlcActionRows.publicFrame,
      ConcreteNifsPiRlcActionRows.actionFrame] at exact
    have challengesExact :
        (fun source =>
          Phi81RingAction.decoded assignment
            (ConcreteNifsPiRlcActionRows.challenge
              application profile frame source)) =
          proof.certificate.piRlcChallenges := by
      funext source
      exact decoded_challenge application profile frame assignment proof
        proofDecoded source
    have valuesExact :
        (fun source =>
          Phi81RingAction.decoded assignment
            (ConcreteNifsPiRlcActionRows.publicValue
              application profile frame block source)) =
          publicSource running fresh block := by
      funext source
      exact decoded_publicValue application profile frame assignment
        running fresh runningDecoded freshDecoded block source
    rw [← challengesExact, ← valuesExact]
    exact exact
  · intro matrix
    let target :=
      ConcreteNifsPiRlcActionRows.evaluationLowFrame
        application profile frame productBase matrix
    have targetSatisfied :=
      frame_satisfied application profile frame assignment productBase
        satisfied target
        (evaluationLowFrame_mem
          application profile frame productBase matrix)
    have exact :=
      Phi81RingAction.rows_sound target assignment constantWire
        targetSatisfied
    dsimp [target, ConcreteNifsPiRlcActionRows.evaluationLowFrame,
      ConcreteNifsPiRlcActionRows.actionFrame] at exact
    have challengesExact :
        (fun source =>
          Phi81RingAction.decoded assignment
            (ConcreteNifsPiRlcActionRows.challenge
              application profile frame source)) =
          proof.certificate.piRlcChallenges := by
      funext source
      exact decoded_challenge application profile frame assignment proof
        proofDecoded source
    have valuesExact :
        (fun source =>
          Phi81RingAction.decoded assignment
            (ConcreteNifsPiRlcActionRows.evaluationValueLow
              application profile frame matrix source)) =
          (fun source position =>
            (evaluationSource
              (keys := keys) proof matrix source position).c0) := by
      funext source
      exact decoded_evaluationValueLow application profile frame assignment
        proof proofDecoded matrix source
    rw [← challengesExact, ← valuesExact]
    exact exact
  · intro matrix
    let target :=
      ConcreteNifsPiRlcActionRows.evaluationHighFrame
        application profile frame productBase matrix
    have targetSatisfied :=
      frame_satisfied application profile frame assignment productBase
        satisfied target
        (evaluationHighFrame_mem
          application profile frame productBase matrix)
    have exact :=
      Phi81RingAction.rows_sound target assignment constantWire
        targetSatisfied
    dsimp [target, ConcreteNifsPiRlcActionRows.evaluationHighFrame,
      ConcreteNifsPiRlcActionRows.actionFrame] at exact
    have challengesExact :
        (fun source =>
          Phi81RingAction.decoded assignment
            (ConcreteNifsPiRlcActionRows.challenge
              application profile frame source)) =
          proof.certificate.piRlcChallenges := by
      funext source
      exact decoded_challenge application profile frame assignment proof
        proofDecoded source
    have valuesExact :
        (fun source =>
          Phi81RingAction.decoded assignment
            (ConcreteNifsPiRlcActionRows.evaluationValueHigh
              application profile frame matrix source)) =
          (fun source position =>
            (evaluationSource
              (keys := keys) proof matrix source position).c1) := by
      funext source
      exact decoded_evaluationValueHigh application profile frame assignment
        proof proofDecoded matrix source
    rw [← challengesExact, ← valuesExact]
    exact exact

/-- **Headline soundness.** Satisfaction of the complete Phi81-action slice,
together with whole-frame decoding, yields every visible parent coordinate
equation without a caller-supplied semantic conclusion. -/
theorem equations_of_rows
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
    (satisfied :
      RawSatisfies
        (ConcreteNifsPiRlcActionRows.rows
          application profile frame productBase)
        assignment) :
    Equations (keys := keys) running fresh proof output := by
  have runningDecoded :=
    ConcreteNifsSelectedCallFrame.running_decodes_of_frame_decodes
      (FamilyFor application) frame assignment running fresh proof
      decodedInputs
  have freshDecoded :=
    ConcreteNifsSelectedCallFrame.fresh_decodes_of_frame_decodes
      (FamilyFor application) frame assignment running fresh proof
      decodedInputs
  have proofDecoded :=
    ConcreteNifsSelectedCallFrame.proof_decodes_of_frame_decodes
      (FamilyFor application) frame assignment running fresh proof
      decodedInputs
  have outputDecoded :=
    ConcreteNifsSelectedCallFrame.output_decodes_of_frame_decodes
      (FamilyFor application) frame assignment output decodedOutput
  refine {
    commitment := ?_
    publicInput := ?_
    evaluationLow := ?_
    evaluationHigh := ?_
  }
  · intro row
    let target :=
      ConcreteNifsPiRlcActionRows.commitmentFrame
        application profile frame productBase row
    have targetSatisfied :=
      frame_satisfied application profile frame assignment productBase
        satisfied target
        (commitmentFrame_mem application profile frame productBase row)
    have exact :=
      Phi81RingAction.rows_sound target assignment constantWire
        targetSatisfied
    dsimp [target, ConcreteNifsPiRlcActionRows.commitmentFrame,
      ConcreteNifsPiRlcActionRows.actionFrame] at exact
    have challengesExact :
        (fun source =>
          Phi81RingAction.decoded assignment
            (ConcreteNifsPiRlcActionRows.challenge
              application profile frame source)) =
          proof.certificate.piRlcChallenges := by
      funext source
      exact decoded_challenge application profile frame assignment proof
        proofDecoded source
    have valuesExact :
        (fun source =>
          Phi81RingAction.decoded assignment
            (ConcreteNifsPiRlcActionRows.commitmentValue
              application profile frame row source)) =
          commitmentSource running fresh row := by
      funext source
      exact decoded_commitmentValue application profile frame assignment
        running fresh runningDecoded freshDecoded row source
    rw [decoded_commitmentOutput application profile frame assignment
        output outputDecoded row] at exact
    rw [← challengesExact, ← valuesExact]
    exact exact
  · intro block
    let target :=
      ConcreteNifsPiRlcActionRows.publicFrame
        application profile frame productBase block
    have targetSatisfied :=
      frame_satisfied application profile frame assignment productBase
        satisfied target
        (publicFrame_mem application profile frame productBase block)
    have exact :=
      Phi81RingAction.rows_sound target assignment constantWire
        targetSatisfied
    dsimp [target, ConcreteNifsPiRlcActionRows.publicFrame,
      ConcreteNifsPiRlcActionRows.actionFrame] at exact
    have challengesExact :
        (fun source =>
          Phi81RingAction.decoded assignment
            (ConcreteNifsPiRlcActionRows.challenge
              application profile frame source)) =
          proof.certificate.piRlcChallenges := by
      funext source
      exact decoded_challenge application profile frame assignment proof
        proofDecoded source
    have valuesExact :
        (fun source =>
          Phi81RingAction.decoded assignment
            (ConcreteNifsPiRlcActionRows.publicValue
              application profile frame block source)) =
          publicSource running fresh block := by
      funext source
      exact decoded_publicValue application profile frame assignment
        running fresh runningDecoded freshDecoded block source
    rw [decoded_publicOutput application profile frame assignment
        output outputDecoded block] at exact
    rw [← challengesExact, ← valuesExact]
    exact exact
  · intro matrix lane
    let target :=
      ConcreteNifsPiRlcActionRows.evaluationLowFrame
        application profile frame productBase matrix
    have targetSatisfied :=
      frame_satisfied application profile frame assignment productBase
        satisfied target
        (evaluationLowFrame_mem
          application profile frame productBase matrix)
    have exact :=
      Phi81RingAction.rows_sound target assignment constantWire
        targetSatisfied
    dsimp [target, ConcreteNifsPiRlcActionRows.evaluationLowFrame,
      ConcreteNifsPiRlcActionRows.actionFrame] at exact
    have challengesExact :
        (fun source =>
          Phi81RingAction.decoded assignment
            (ConcreteNifsPiRlcActionRows.challenge
              application profile frame source)) =
          proof.certificate.piRlcChallenges := by
      funext source
      exact decoded_challenge application profile frame assignment proof
        proofDecoded source
    have valuesExact :
        (fun source =>
          Phi81RingAction.decoded assignment
            (ConcreteNifsPiRlcActionRows.evaluationValueLow
              application profile frame matrix source)) =
          (fun source position =>
            (evaluationSource (keys := keys) proof matrix source position).c0) := by
      funext source
      exact decoded_evaluationValueLow application profile frame assignment
        proof proofDecoded matrix source
    rw [decoded_evaluationOutputLow application profile frame assignment
        output outputDecoded matrix] at exact
    rw [← challengesExact, ← valuesExact]
    exact congrFun exact lane
  · intro matrix lane
    let target :=
      ConcreteNifsPiRlcActionRows.evaluationHighFrame
        application profile frame productBase matrix
    have targetSatisfied :=
      frame_satisfied application profile frame assignment productBase
        satisfied target
        (evaluationHighFrame_mem
          application profile frame productBase matrix)
    have exact :=
      Phi81RingAction.rows_sound target assignment constantWire
        targetSatisfied
    dsimp [target, ConcreteNifsPiRlcActionRows.evaluationHighFrame,
      ConcreteNifsPiRlcActionRows.actionFrame] at exact
    have challengesExact :
        (fun source =>
          Phi81RingAction.decoded assignment
            (ConcreteNifsPiRlcActionRows.challenge
              application profile frame source)) =
          proof.certificate.piRlcChallenges := by
      funext source
      exact decoded_challenge application profile frame assignment proof
        proofDecoded source
    have valuesExact :
        (fun source =>
          Phi81RingAction.decoded assignment
            (ConcreteNifsPiRlcActionRows.evaluationValueHigh
              application profile frame matrix source)) =
          (fun source position =>
            (evaluationSource (keys := keys) proof matrix source position).c1) := by
      funext source
      exact decoded_evaluationValueHigh application profile frame assignment
        proof proofDecoded matrix source
    rw [decoded_evaluationOutputHigh application profile frame assignment
        output outputDecoded matrix] at exact
    rw [← challengesExact, ← valuesExact]
    exact congrFun exact lane

end SelectedFrame

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsPiRlcActionSemantics
