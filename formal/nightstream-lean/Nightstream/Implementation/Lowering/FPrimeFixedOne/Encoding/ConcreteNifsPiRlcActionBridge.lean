import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsPiRlcActionSemantics
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.RingKBaseActionCoordinates

/-!
Contract: refine the coordinate equations emitted by the selected
`Pi_RLC` action slice to the independently defined ConcretePhi81 parent.

This module joins only Lean-owned definitions: the typed call operands, the
canonical `Pi_CCS` output product, the concrete `PiRLC` algebra, and the
physical action equations.  It imports no Rust or generated-row artifact.
-/

set_option autoImplicit false
set_option maxRecDepth 100000
set_option maxHeartbeats 2000000

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsPiRlcActionBridge

open Nightstream.HyperNova.Construction2.Paper
open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.DirectCalls
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsParameters
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsPiRlcActionSemantics
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism
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

private theorem combineCommitments_apply
    {count verifierRows : Nat}
    (challenges : Fin count → RingF)
    (values : Fin count → PiRLCAlgebra.Commitment.Value verifierRows)
    (row : Fin verifierRows) :
    (PiRLCAlgebra.Commitment.combineCommitments challenges values) row =
      Phi81RingAction.combine challenges (fun source => values source row) := by
  induction count with
  | zero =>
      rfl
  | succ count inductionHypothesis =>
      simp only [PiRLCAlgebra.Commitment.combineCommitments,
        PiRLCAlgebra.Commitment.commitmentAdd,
        PiRLCAlgebra.Commitment.commitmentAct,
        Phi81RingAction.combine]
      rw [inductionHypothesis
        (fun source => challenges source.succ)
        (fun source => values source.succ)]

private theorem combinePublicInputs_apply
    {relationShape : Phi81Relation.Shape}
    {count : Nat}
    (challenges : Fin count → RingF)
    (inputs : Fin count → Phi81Relation.PublicInput relationShape)
    (column : Fin relationShape.publicWidth) :
    (PiRLCAlgebra.PublicInput.combinePublicInputs challenges inputs) column =
      Phi81RingAction.combine challenges
        (fun source =>
          PiRLCAlgebra.PublicInput.publicBlock (inputs source)
            (PiRLCAlgebra.PublicInput.publicBlockIndex relationShape column))
        (PiRLCAlgebra.PublicInput.publicLaneIndex column) := by
  induction count with
  | zero =>
      rfl
  | succ count inductionHypothesis =>
      simp only [PiRLCAlgebra.PublicInput.combinePublicInputs,
        PiRLCAlgebra.PublicInput.publicAdd,
        PiRLCAlgebra.PublicInput.publicAct,
        Phi81RingAction.combine]
      rw [inductionHypothesis
        (fun source => challenges source.succ)
        (fun source => inputs source.succ)]
      rfl

private theorem k_eq_of_coordinates
    (left right : K)
    (lowEqual : left.c0 = right.c0)
    (highEqual : left.c1 = right.c1) :
    left = right := by
  cases left
  cases right
  cases lowEqual
  cases highEqual
  rfl

/-- One physical output-parent commitment coordinate is exactly the
verifier-computed parent coordinate, without an output-decoding premise. -/
theorem physical_commitment_eq_derived
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
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
    (running :
      SelectedRunning shape publicRingColumns publicFits verifierRows)
    (fresh :
      SelectedFresh shape publicRingColumns publicFits verifierRows)
    (proof :
      SelectedProof shape TranscriptState publicRingColumns publicFits
        verifierRows)
    (equations :
      PhysicalEquations application profile frame assignment
        running fresh proof)
    (row : Fin verifierRows)
    (lane : Fin ringDegree) :
    Phi81RingAction.decoded assignment
        (ConcreteNifsPiRlcActionRows.commitmentOutput
          application profile frame row) lane =
      (ConcretePhi81.derive
        (selectedContext (keys := keys) running fresh proof)
        proof.certificate).piRlcOutput.commitment row lane := by
  have emitted := congrFun (equations.commitment row) lane
  rw [emitted]
  change
    Phi81RingAction.combine proof.certificate.piRlcChallenges
        (commitmentSource running fresh row) lane =
      (PiRLCAlgebra.Commitment.combineCommitments
        proof.certificate.piRlcChallenges
        (fun source =>
          ((ConcretePhi81.derive
            (selectedContext (keys := keys) running fresh proof)
            proof.certificate).piCcsOutputs source).commitment)) row lane
  rw [combineCommitments_apply]
  rfl

/-- One physical output-parent public-input coordinate is exactly the
verifier-computed parent coordinate, without an output-decoding premise. -/
theorem physical_public_eq_derived
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
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
    (running :
      SelectedRunning shape publicRingColumns publicFits verifierRows)
    (fresh :
      SelectedFresh shape publicRingColumns publicFits verifierRows)
    (proof :
      SelectedProof shape TranscriptState publicRingColumns publicFits
        verifierRows)
    (equations :
      PhysicalEquations application profile frame assignment
        running fresh proof)
    (block : Fin publicRingColumns)
    (lane : Fin ringDegree) :
    Phi81RingAction.decoded assignment
        (ConcreteNifsPiRlcActionRows.publicOutput
          application profile frame block) lane =
      (ConcretePhi81.derive
        (selectedContext (keys := keys) running fresh proof)
        proof.certificate).piRlcOutput.publicInput
          (ConcreteNifsPiRlcActionRows.publicCoordinate block lane) := by
  have emitted := congrFun (equations.publicInput block) lane
  rw [emitted]
  change
    Phi81RingAction.combine proof.certificate.piRlcChallenges
        (publicSource running fresh block) lane =
      (PiRLCAlgebra.PublicInput.combinePublicInputs
        proof.certificate.piRlcChallenges
        (fun source =>
          ((ConcretePhi81.derive
            (selectedContext (keys := keys) running fresh proof)
            proof.certificate).piCcsOutputs source).publicInput))
        (ConcreteNifsPiRlcActionRows.publicCoordinate block lane)
  rw [combinePublicInputs_apply]
  have blockIndex :
      PiRLCAlgebra.PublicInput.publicBlockIndex
          (PiCCS.SplitNc.Verifier.RelationShape
            shape publicRingColumns publicFits)
          (ConcreteNifsPiRlcActionRows.publicCoordinate block lane) =
        block := by
    apply Fin.ext
    simp only [
      PiRLCAlgebra.PublicInput.publicBlockIndex,
      ConcreteNifsPiRlcActionRows.publicCoordinate]
    rw [Nat.mul_comm block.val ringDegree,
      Nat.mul_add_div (by simp [ringDegree])]
    simp [Nat.div_eq_of_lt lane.isLt]
  have laneIndex :
      PiRLCAlgebra.PublicInput.publicLaneIndex
          (shape :=
            PiCCS.SplitNc.Verifier.RelationShape
              shape publicRingColumns publicFits)
          (ConcreteNifsPiRlcActionRows.publicCoordinate block lane) =
        lane := by
    apply Fin.ext
    simp only [
      PiRLCAlgebra.PublicInput.publicLaneIndex,
      ConcreteNifsPiRlcActionRows.publicCoordinate]
    exact Nat.mul_add_mod_of_lt lane.isLt
  rw [blockIndex, laneIndex]
  rfl

/-- Commitment coordinates emitted by the action slice are exactly the
commitment of the verifier-computed `PiRLC` parent. -/
theorem commitment_eq_derived
    (running :
      SelectedRunning shape publicRingColumns publicFits verifierRows)
    (fresh :
      SelectedFresh shape publicRingColumns publicFits verifierRows)
    (proof :
      SelectedProof shape TranscriptState publicRingColumns publicFits
        verifierRows)
    (output :
      SelectedRunning shape publicRingColumns publicFits verifierRows)
    (equations :
      Equations (keys := keys) running fresh proof output) :
    output.parent.commitment =
      (ConcretePhi81.derive
        (selectedContext (keys := keys) running fresh proof)
        proof.certificate).piRlcOutput.commitment := by
  funext row
  rw [equations.commitment row]
  change
    Phi81RingAction.combine proof.certificate.piRlcChallenges
        (commitmentSource running fresh row) =
      (PiRLCAlgebra.Commitment.combineCommitments
        proof.certificate.piRlcChallenges
        (fun source =>
          ((ConcretePhi81.derive
            (selectedContext (keys := keys) running fresh proof)
            proof.certificate).piCcsOutputs source).commitment)) row
  rw [combineCommitments_apply]
  rfl

/-- Public-input coordinates emitted by the action slice are exactly the
public input of the verifier-computed `PiRLC` parent. -/
theorem publicInput_eq_derived
    (running :
      SelectedRunning shape publicRingColumns publicFits verifierRows)
    (fresh :
      SelectedFresh shape publicRingColumns publicFits verifierRows)
    (proof :
      SelectedProof shape TranscriptState publicRingColumns publicFits
        verifierRows)
    (output :
      SelectedRunning shape publicRingColumns publicFits verifierRows)
    (equations :
      Equations (keys := keys) running fresh proof output) :
    output.parent.publicInput =
      (ConcretePhi81.derive
        (selectedContext (keys := keys) running fresh proof)
        proof.certificate).piRlcOutput.publicInput := by
  funext column
  let relationShape :=
    PiCCS.SplitNc.Verifier.RelationShape shape publicRingColumns publicFits
  let block :=
    PiRLCAlgebra.PublicInput.publicBlockIndex relationShape column
  let lane :=
    PiRLCAlgebra.PublicInput.publicLaneIndex column
  have coordinateEq :
      ConcreteNifsPiRlcActionRows.publicCoordinate block lane = column := by
    apply Fin.ext
    dsimp only [block, lane,
      PiRLCAlgebra.PublicInput.publicBlockIndex,
      PiRLCAlgebra.PublicInput.publicLaneIndex,
      ConcreteNifsPiRlcActionRows.publicCoordinate]
    rw [Nat.mul_comm]
    exact Nat.div_add_mod column.val ringDegree
  have emitted := congrFun (equations.publicInput block) lane
  rw [coordinateEq] at emitted
  rw [emitted]
  change
    Phi81RingAction.combine proof.certificate.piRlcChallenges
        (publicSource running fresh block) lane =
      (PiRLCAlgebra.PublicInput.combinePublicInputs
        proof.certificate.piRlcChallenges
        (fun source =>
          ((ConcretePhi81.derive
            (selectedContext (keys := keys) running fresh proof)
            proof.certificate).piCcsOutputs source).publicInput)) column
  rw [combinePublicInputs_apply]
  rfl

private theorem derived_source_evaluation_getD
    (running :
      SelectedRunning shape publicRingColumns publicFits verifierRows)
    (fresh :
      SelectedFresh shape publicRingColumns publicFits verifierRows)
    (proof :
      SelectedProof shape TranscriptState publicRingColumns publicFits
        verifierRows)
    (source : Fin FixedActive.arity.total)
    (matrix : Fin shape.matrixCount) :
    ((ConcretePhi81.derive
      (selectedContext (keys := keys) running fresh proof)
      proof.certificate).piCcsOutputs source).evaluations.getD
        matrix.val BaseLinear.evaluationZero =
      evaluationSource (keys := keys) proof matrix source := by
  change
    (PiCCS.SplitNc.Verifier.OutputProduct.claimedEvaluations
      proof.certificate.piCcs.output
      ((keys
        Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected).template.alignment.semanticIndex
          source)).getD matrix.val BaseLinear.evaluationZero =
      evaluationSource (keys := keys) proof matrix source
  funext lane
  have matrixLt :
      matrix.val <
        (PiCCS.SplitNc.Verifier.OutputProduct.claimedEvaluations
          proof.certificate.piCcs.output
          ((keys
            Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected).template.alignment.semanticIndex
              source)).size := by
    simp
  rw [Array.getD_eq_getD_getElem?,
    Array.getElem?_eq_getElem matrixLt]
  simp only [
    PiCCS.SplitNc.Verifier.OutputProduct.claimedEvaluations_get,
    Option.getD_some]
  rfl

/-- The low physical coordinate of one output-parent evaluation is exactly
the verifier-computed parent coordinate. -/
theorem physical_evaluation_low_eq_derived
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
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
    (running :
      SelectedRunning shape publicRingColumns publicFits verifierRows)
    (fresh :
      SelectedFresh shape publicRingColumns publicFits verifierRows)
    (proof :
      SelectedProof shape TranscriptState publicRingColumns publicFits
        verifierRows)
    (equations :
      PhysicalEquations application profile frame assignment
        running fresh proof)
    (matrix : Fin shape.matrixCount)
    (lane : Fin ringDegree) :
    Phi81RingAction.decoded assignment
        (ConcreteNifsPiRlcActionRows.evaluationOutputLow
          application profile frame matrix) lane =
      ((ConcretePhi81.derive
        (selectedContext (keys := keys) running fresh proof)
        proof.certificate).piRlcOutput.evaluations.getD
          matrix.val ringKZero lane).c0 := by
  have emitted := congrFun (equations.evaluationLow matrix) lane
  rw [emitted]
  change
    Phi81RingAction.combine proof.certificate.piRlcChallenges
        (fun source position =>
          (evaluationSource (keys := keys) proof matrix source position).c0)
        lane =
      ((PiRLCFinite.combineEvaluations
        proof.certificate.piRlcChallenges
        (fun source =>
          ((ConcretePhi81.derive
            (selectedContext (keys := keys) running fresh proof)
            proof.certificate).piCcsOutputs source).evaluations)).getD
          matrix.val BaseLinear.evaluationZero lane).c0
  rw [RingKBaseActionCoordinates.combineEvaluations_getD_low]
  apply congrArg (fun values =>
    Phi81RingAction.combine proof.certificate.piRlcChallenges values lane)
  funext source position
  rw [derived_source_evaluation_getD
    (keys := keys) running fresh proof source matrix]
  rfl

/-- The high physical coordinate of one output-parent evaluation is exactly
the verifier-computed parent coordinate. -/
theorem physical_evaluation_high_eq_derived
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
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
    (running :
      SelectedRunning shape publicRingColumns publicFits verifierRows)
    (fresh :
      SelectedFresh shape publicRingColumns publicFits verifierRows)
    (proof :
      SelectedProof shape TranscriptState publicRingColumns publicFits
        verifierRows)
    (equations :
      PhysicalEquations application profile frame assignment
        running fresh proof)
    (matrix : Fin shape.matrixCount)
    (lane : Fin ringDegree) :
    Phi81RingAction.decoded assignment
        (ConcreteNifsPiRlcActionRows.evaluationOutputHigh
          application profile frame matrix) lane =
      ((ConcretePhi81.derive
        (selectedContext (keys := keys) running fresh proof)
        proof.certificate).piRlcOutput.evaluations.getD
          matrix.val ringKZero lane).c1 := by
  have emitted := congrFun (equations.evaluationHigh matrix) lane
  rw [emitted]
  change
    Phi81RingAction.combine proof.certificate.piRlcChallenges
        (fun source position =>
          (evaluationSource (keys := keys) proof matrix source position).c1)
        lane =
      ((PiRLCFinite.combineEvaluations
        proof.certificate.piRlcChallenges
        (fun source =>
          ((ConcretePhi81.derive
            (selectedContext (keys := keys) running fresh proof)
            proof.certificate).piCcsOutputs source).evaluations)).getD
          matrix.val BaseLinear.evaluationZero lane).c1
  rw [RingKBaseActionCoordinates.combineEvaluations_getD_high]
  apply congrArg (fun values =>
    Phi81RingAction.combine proof.certificate.piRlcChallenges values lane)
  funext source position
  rw [derived_source_evaluation_getD
    (keys := keys) running fresh proof source matrix]
  rfl

/-- Each matrix/lane coordinate emitted by the two base-ring action branches
is exactly the corresponding quadratic-extension coordinate of the
verifier-computed `PiRLC` parent. -/
theorem evaluation_getD_eq_derived
    (running :
      SelectedRunning shape publicRingColumns publicFits verifierRows)
    (fresh :
      SelectedFresh shape publicRingColumns publicFits verifierRows)
    (proof :
      SelectedProof shape TranscriptState publicRingColumns publicFits
        verifierRows)
    (output :
      SelectedRunning shape publicRingColumns publicFits verifierRows)
    (equations :
      Equations (keys := keys) running fresh proof output)
    (matrix : Fin shape.matrixCount)
    (lane : Fin ringDegree) :
    output.parent.evaluations.getD matrix.val ringKZero lane =
      (ConcretePhi81.derive
        (selectedContext (keys := keys) running fresh proof)
        proof.certificate).piRlcOutput.evaluations.getD
          matrix.val ringKZero lane := by
  change
    output.parent.evaluations.getD matrix.val ringKZero lane =
      (PiRLCFinite.combineEvaluations
        proof.certificate.piRlcChallenges
        (fun source =>
          ((ConcretePhi81.derive
            (selectedContext (keys := keys) running fresh proof)
            proof.certificate).piCcsOutputs source).evaluations)).getD
          matrix.val ringKZero lane
  apply k_eq_of_coordinates
  · rw [equations.evaluationLow matrix lane]
    change
      Phi81RingAction.combine proof.certificate.piRlcChallenges
          (fun source position =>
            (evaluationSource (keys := keys) proof matrix source position).c0)
          lane =
        ((PiRLCFinite.combineEvaluations
          proof.certificate.piRlcChallenges
          (fun source =>
            ((ConcretePhi81.derive
              (selectedContext (keys := keys) running fresh proof)
              proof.certificate).piCcsOutputs source).evaluations)).getD
            matrix.val BaseLinear.evaluationZero lane).c0
    rw [RingKBaseActionCoordinates.combineEvaluations_getD_low]
    apply congrArg (fun values =>
      Phi81RingAction.combine proof.certificate.piRlcChallenges values lane)
    funext source position
    rw [derived_source_evaluation_getD
      (keys := keys) running fresh proof source matrix]
    rfl
  · rw [equations.evaluationHigh matrix lane]
    change
      Phi81RingAction.combine proof.certificate.piRlcChallenges
          (fun source position =>
            (evaluationSource (keys := keys) proof matrix source position).c1)
          lane =
        ((PiRLCFinite.combineEvaluations
          proof.certificate.piRlcChallenges
          (fun source =>
            ((ConcretePhi81.derive
              (selectedContext (keys := keys) running fresh proof)
              proof.certificate).piCcsOutputs source).evaluations)).getD
            matrix.val BaseLinear.evaluationZero lane).c1
    rw [RingKBaseActionCoordinates.combineEvaluations_getD_high]
    apply congrArg (fun values =>
      Phi81RingAction.combine proof.certificate.piRlcChallenges values lane)
    funext source position
    rw [derived_source_evaluation_getD
      (keys := keys) running fresh proof source matrix]
    rfl

/-- Successful output decoding fixes the array shape, so the coordinate
equations refine the complete evaluation array rather than only a prefix. -/
theorem evaluations_eq_derived
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    (running :
      SelectedRunning shape publicRingColumns publicFits verifierRows)
    (fresh :
      SelectedFresh shape publicRingColumns publicFits verifierRows)
    (proof :
      SelectedProof shape TranscriptState publicRingColumns publicFits
        verifierRows)
    (output :
      SelectedRunning shape publicRingColumns publicFits verifierRows)
    (outputAdmissible :
      ((FamilyFor application).codecFor (.data .running)).Admissible output)
    (equations :
      Equations (keys := keys) running fresh proof output) :
    output.parent.evaluations =
      (ConcretePhi81.derive
        (selectedContext (keys := keys) running fresh proof)
        proof.certificate).piRlcOutput.evaluations := by
  have outputSize :
      output.parent.evaluations.size = shape.matrixCount :=
    profile.runningViews.parentEvaluationsSize output outputAdmissible
  apply Array.ext
  · change
      output.parent.evaluations.size =
        (PiRLCFinite.combineEvaluations
          proof.certificate.piRlcChallenges
          (fun source =>
            ((ConcretePhi81.derive
              (selectedContext (keys := keys) running fresh proof)
              proof.certificate).piCcsOutputs source).evaluations)).size
    rw [outputSize]
    simp [PiRLCFinite.combineEvaluations]
    rfl
  · intro index outputLt derivedLt
    have matrixLt : index < shape.matrixCount := by
      rw [← outputSize]
      exact outputLt
    let matrix : Fin shape.matrixCount := ⟨index, matrixLt⟩
    funext lane
    have coordinate :=
      evaluation_getD_eq_derived (keys := keys)
        running fresh proof output equations matrix lane
    dsimp only [matrix] at coordinate
    rw [Array.getD_eq_getD_getElem?,
      Array.getElem?_eq_getElem outputLt,
      Option.getD_some] at coordinate
    rw [Array.getD_eq_getD_getElem?,
      Array.getElem?_eq_getElem derivedLt,
      Option.getD_some] at coordinate
    exact coordinate

/-! ## Constructive equations for the verifier-computed result -/

/-- The deterministic fixed-active result satisfies exactly the four semantic
equation families emitted by the selected Phi81 action slice.  This is the
honest direction needed by the physical call recipe; no output equation is
supplied by a caller. -/
theorem equations_of_result
    (running :
      SelectedRunning shape publicRingColumns publicFits verifierRows)
    (fresh :
      SelectedFresh shape publicRingColumns publicFits verifierRows)
    (proof :
      SelectedProof shape TranscriptState publicRingColumns publicFits
        verifierRows) :
    Equations (keys := keys) running fresh proof
      (SelectedRunning.ofResult
        (FixedActive.resultOf
          (selectedContext (keys := keys) running fresh proof)
          proof.certificate)) := by
  refine {
    commitment := ?_
    publicInput := ?_
    evaluationLow := ?_
    evaluationHigh := ?_
  }
  · intro row
    change
      (ConcretePhi81.derive
        (selectedContext (keys := keys) running fresh proof)
        proof.certificate).piRlcOutput.commitment row =
        Phi81RingAction.combine proof.certificate.piRlcChallenges
          (commitmentSource running fresh row)
    change
      (PiRLCAlgebra.Commitment.combineCommitments
        proof.certificate.piRlcChallenges
        (fun source =>
          ((ConcretePhi81.derive
            (selectedContext (keys := keys) running fresh proof)
            proof.certificate).piCcsOutputs source).commitment)) row =
        Phi81RingAction.combine proof.certificate.piRlcChallenges
          (commitmentSource running fresh row)
    rw [combineCommitments_apply]
    rfl
  · intro block
    funext lane
    change
      (ConcretePhi81.derive
        (selectedContext (keys := keys) running fresh proof)
        proof.certificate).piRlcOutput.publicInput
          (ConcreteNifsPiRlcActionRows.publicCoordinate block lane) =
        Phi81RingAction.combine proof.certificate.piRlcChallenges
          (publicSource running fresh block) lane
    change
      (PiRLCAlgebra.PublicInput.combinePublicInputs
        proof.certificate.piRlcChallenges
        (fun source =>
          ((ConcretePhi81.derive
            (selectedContext (keys := keys) running fresh proof)
            proof.certificate).piCcsOutputs source).publicInput))
        (ConcreteNifsPiRlcActionRows.publicCoordinate block lane) =
          Phi81RingAction.combine proof.certificate.piRlcChallenges
            (publicSource running fresh block) lane
    rw [combinePublicInputs_apply]
    have blockIndex :
        PiRLCAlgebra.PublicInput.publicBlockIndex
            (PiCCS.SplitNc.Verifier.RelationShape
              shape publicRingColumns publicFits)
            (ConcreteNifsPiRlcActionRows.publicCoordinate block lane) =
          block := by
      apply Fin.ext
      simp only [
        PiRLCAlgebra.PublicInput.publicBlockIndex,
        ConcreteNifsPiRlcActionRows.publicCoordinate]
      rw [Nat.mul_comm block.val ringDegree,
        Nat.mul_add_div (by simp [ringDegree])]
      simp [Nat.div_eq_of_lt lane.isLt]
    have laneIndex :
        PiRLCAlgebra.PublicInput.publicLaneIndex
            (shape :=
              PiCCS.SplitNc.Verifier.RelationShape
                shape publicRingColumns publicFits)
            (ConcreteNifsPiRlcActionRows.publicCoordinate block lane) =
          lane := by
      apply Fin.ext
      simp only [
        PiRLCAlgebra.PublicInput.publicLaneIndex,
        ConcreteNifsPiRlcActionRows.publicCoordinate]
      exact Nat.mul_add_mod_of_lt lane.isLt
    rw [blockIndex, laneIndex]
    rfl
  · intro matrix lane
    change
      ((ConcretePhi81.derive
        (selectedContext (keys := keys) running fresh proof)
        proof.certificate).piRlcOutput.evaluations.getD
          matrix.val ringKZero lane).c0 =
        Phi81RingAction.combine proof.certificate.piRlcChallenges
          (fun source position =>
            (evaluationSource
              (keys := keys) proof matrix source position).c0)
          lane
    change
      ((PiRLCFinite.combineEvaluations
        proof.certificate.piRlcChallenges
        (fun source =>
          ((ConcretePhi81.derive
            (selectedContext (keys := keys) running fresh proof)
            proof.certificate).piCcsOutputs source).evaluations)).getD
          matrix.val BaseLinear.evaluationZero lane).c0 =
        Phi81RingAction.combine proof.certificate.piRlcChallenges
          (fun source position =>
            (evaluationSource
              (keys := keys) proof matrix source position).c0)
          lane
    rw [RingKBaseActionCoordinates.combineEvaluations_getD_low]
    apply congrArg (fun values =>
      Phi81RingAction.combine proof.certificate.piRlcChallenges values lane)
    funext source position
    rw [derived_source_evaluation_getD
      (keys := keys) running fresh proof source matrix]
    rfl
  · intro matrix lane
    change
      ((ConcretePhi81.derive
        (selectedContext (keys := keys) running fresh proof)
        proof.certificate).piRlcOutput.evaluations.getD
          matrix.val ringKZero lane).c1 =
        Phi81RingAction.combine proof.certificate.piRlcChallenges
          (fun source position =>
            (evaluationSource
              (keys := keys) proof matrix source position).c1)
          lane
    change
      ((PiRLCFinite.combineEvaluations
        proof.certificate.piRlcChallenges
        (fun source =>
          ((ConcretePhi81.derive
            (selectedContext (keys := keys) running fresh proof)
            proof.certificate).piCcsOutputs source).evaluations)).getD
          matrix.val BaseLinear.evaluationZero lane).c1 =
        Phi81RingAction.combine proof.certificate.piRlcChallenges
          (fun source position =>
            (evaluationSource
              (keys := keys) proof matrix source position).c1)
          lane
    rw [RingKBaseActionCoordinates.combineEvaluations_getD_high]
    apply congrArg (fun values =>
      Phi81RingAction.combine proof.certificate.piRlcChallenges values lane)
    funext source position
    rw [derived_source_evaluation_getD
      (keys := keys) running fresh proof source matrix]
    rfl

end SelectedFrame

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsPiRlcActionBridge
