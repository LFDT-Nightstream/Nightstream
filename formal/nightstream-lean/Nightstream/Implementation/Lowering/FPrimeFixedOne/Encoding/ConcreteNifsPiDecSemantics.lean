import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsPiDecRows
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsPiRlcParentSemantics
import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.DerivedPiDec

/-!
Contract: exact semantic refinement and honest completeness of the selected
outgoing `Pi_DEC` rows.

Whole-frame decoding is the sole source of child and parent values.  The
headline theorem composes these rows with the independently row-derived
`Pi_RLC` parent theorem and concludes the frozen
`DerivedPiDec.RecompositionEquations`; no parent or recomposition equation is
supplied as a premise on that path.
-/

set_option autoImplicit false
set_option maxRecDepth 100000
set_option maxHeartbeats 2000000

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsPiDecSemantics

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
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsPiDecRows
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.Implementation.R1CS.Canonical.KHorner
open Nightstream.Implementation.R1CS.Canonical.KMul
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier
open Nightstream.SuperNeo.Folding.Nifs
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
    (SelectedKey shape TranscriptState publicRingColumns publicFits verifierRows)
    Digest AppState Witness
    (SelectedRunning shape publicRingColumns publicFits verifierRows)
    (SelectedFresh shape publicRingColumns publicFits verifierRows)
    Encoded 1}
variable {terminalRelations :
  TerminalRelations
    (SelectedKey shape TranscriptState publicRingColumns publicFits verifierRows)
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

private def selectedContext
    (running :
      SelectedRunning shape publicRingColumns publicFits verifierRows)
    (fresh :
      SelectedFresh shape publicRingColumns publicFits verifierRows)
    (proof :
      SelectedProof shape TranscriptState publicRingColumns publicFits
        verifierRows) :=
  (ConcreteNifsParameters.context
    (keys Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected)
    running fresh proof).materialize

/-- The exact three equations expressed through the decoded output carrier.
This intermediate statement is derived from rows below and is never accepted
as a premise by the headline frozen-refinement theorem. -/
structure OutputEquations
    (output :
      SelectedRunning shape publicRingColumns publicFits verifierRows)
    (proof :
      SelectedProof shape TranscriptState publicRingColumns publicFits
        verifierRows) : Prop where
  commitment :
    output.parent.commitment =
      PiDECAlgebra.Commitment.recomposeCommitment
        (fun child => (proof.certificate.piDecPayloads child).commitment)
  publicInput :
    output.parent.publicInput =
      PiDECAlgebra.PublicInput.recomposePublicInput
        (fun child => (proof.certificate.piDecPayloads child).publicInput)
  evaluations :
    output.parent.evaluations =
      PiDEC.recomposeEvaluations
        (shape := RelationShape shape publicRingColumns publicFits)
        (fun child => (proof.certificate.piDecPayloads child).evaluations)

/-- Satisfaction of the selected `Pi_DEC` slice determines all three
recomposition equations on the decoded output parent. -/
theorem output_equations_of_rows
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
        (rows application profile frame)
        (numericAssignment (columnMap frame) assignment)) :
    OutputEquations output proof := by
  let numeric := numericAssignment (columnMap frame) assignment
  have wire : numeric 0 = 1 :=
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
    evaluations := ?_
  }
  · funext row lane
    have equation :=
      Phi81RadixRows.rows_sound_f
        (fCoordinates application profile frame)
        (evaluationCoordinates application profile frame)
        numeric wire satisfied
        (commitmentCoordinate application profile frame row lane)
        (commitmentCoordinate_mem application profile frame row lane)
    simp only [commitmentCoordinate] at equation
    have childrenDecoded :
        (fun child =>
          residue
            (lcEval numeric
              (ConcreteNifsCarrierFrame.proofFLocation
                (FamilyFor application) frame
                (profile.payloadViews.commitment child row lane)).carried)) =
          (fun child =>
            payloadCommitmentCoordinate child row lane proof) := by
      funext child
      exact ConcreteNifsCarrierFrame.proofF_decoded
        (FamilyFor application) frame
        (profile.payloadViews.commitment child row lane)
        assignment proof proofDecoded
    have parentDecoded :=
      ConcreteNifsCarrierFrame.outputF_decoded
        (FamilyFor application) frame
        (profile.runningViews.parentCommitment row lane)
        assignment output outputDecoded
    rw [childrenDecoded, parentDecoded] at equation
    rw [Phi81RadixRows.recomposeCommitment_apply]
    simpa only [payloadCommitmentCoordinate, parentCommitmentCoordinate] using
      equation.symm
  · funext column
    have equation :=
      Phi81RadixRows.rows_sound_f
        (fCoordinates application profile frame)
        (evaluationCoordinates application profile frame)
        numeric wire satisfied
        (publicCoordinate application profile frame column)
        (publicCoordinate_mem application profile frame column)
    simp only [publicCoordinate] at equation
    have childrenDecoded :
        (fun child =>
          residue
            (lcEval numeric
              (ConcreteNifsCarrierFrame.proofFLocation
                (FamilyFor application) frame
                (profile.payloadViews.publicInput child column)).carried)) =
          (fun child => payloadPublicCoordinate child column proof) := by
      funext child
      exact ConcreteNifsCarrierFrame.proofF_decoded
        (FamilyFor application) frame
        (profile.payloadViews.publicInput child column)
        assignment proof proofDecoded
    have parentDecoded :=
      ConcreteNifsCarrierFrame.outputF_decoded
        (FamilyFor application) frame
        (profile.runningViews.parentPublic column)
        assignment output outputDecoded
    rw [childrenDecoded, parentDecoded] at equation
    rw [PiDECAlgebra.PublicInput.recomposePublicInput_apply]
    simpa only [payloadPublicCoordinate, parentPublicCoordinate] using
      equation.symm
  · have parentSize :
        output.parent.evaluations.size = shape.matrixCount :=
      profile.runningViews.parentEvaluationsSize output outputAdmissible
    have childSize :
        ∀ child,
          (proof.certificate.piDecPayloads child).evaluations.size =
            shape.matrixCount :=
      profile.payloadViews.evaluationsSize proof proofAdmissible
    apply Array.ext
    · rw [parentSize]
      simp [PiDEC.recomposeEvaluations, RelationShape,
        Phi81Relation.Shape.ofSemantic]
    · intro index leftLt rightLt
      let matrix : Fin shape.matrixCount :=
        ⟨index, by
          rw [parentSize] at leftLt
          exact leftLt⟩
      funext lane
      have equation :=
        Phi81RadixRows.rows_sound_k
          (fCoordinates application profile frame)
          (evaluationCoordinates application profile frame)
          numeric wire satisfied
          (evaluationCoordinate application profile frame matrix lane)
          (evaluationCoordinate_mem application profile frame matrix lane)
      simp only [evaluationCoordinate] at equation
      have childrenDecoded :
          (fun child =>
            KPointEquality.decoded numeric
              (ConcreteNifsCarrierFrame.proofKLocation
                (FamilyFor application) frame
                (profile.payloadViews.evaluation child matrix lane)).carried) =
            (fun child =>
              payloadEvaluationCoordinate child matrix lane proof) := by
        funext child
        exact ConcreteNifsCarrierFrame.proofK_decoded
          (FamilyFor application) frame
          (profile.payloadViews.evaluation child matrix lane)
          assignment proof proofDecoded
      have parentDecoded :=
        ConcreteNifsCarrierFrame.outputK_decoded
          (FamilyFor application) frame
          (profile.runningViews.parentEvaluation matrix lane)
          assignment output outputDecoded
      rw [childrenDecoded, parentDecoded] at equation
      simp only [payloadEvaluationCoordinate,
        parentEvaluationCoordinate] at equation
      have recomposed :=
        Phi81RadixRows.recomposeEvaluations_get
          (shape := RelationShape shape publicRingColumns publicFits)
          (fun child => (proof.certificate.piDecPayloads child).evaluations)
          matrix lane
      calc
        output.parent.evaluations[index] lane =
            output.parent.evaluations.getD index ringKZero lane := by
          rw [Array.getD_eq_getD_getElem?,
            Array.getElem?_eq_getElem leftLt]
          simp
        _ = Phi81RadixRows.combineK PiDEC.radixWeight
              (fun child =>
                (proof.certificate.piDecPayloads child).evaluations.getD
                  index BaseLinear.evaluationZero lane) :=
          equation.symm
        _ =
            (PiDEC.recomposeEvaluations
              (shape := RelationShape shape publicRingColumns publicFits)
              (fun child =>
                (proof.certificate.piDecPayloads child).evaluations)).getD
                  index BaseLinear.evaluationZero lane :=
          recomposed.symm
        _ =
            (PiDEC.recomposeEvaluations
              (shape := RelationShape shape publicRingColumns publicFits)
              (fun child =>
                (proof.certificate.piDecPayloads child).evaluations))[index]
                lane := by
          rw [Array.getD_eq_getD_getElem?,
            Array.getElem?_eq_getElem rightLt]
          simp
          rfl

/-- **Headline outgoing `Pi_DEC` refinement.** The row-derived parent and
selected radix rows imply the unchanged frozen recomposition predicate. -/
theorem recomposition_of_rows
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
    (piDecSatisfied :
      Satisfies
        (rows application profile frame)
        (numericAssignment (columnMap frame) assignment)) :
    DerivedPiDec.RecompositionEquations
      (selectedContext (keys := keys) running fresh proof)
      proof.certificate := by
  have outputEquations :=
    output_equations_of_rows application profile frame assignment running fresh
      proof output constantWire decodedInputs decodedOutput piDecSatisfied
  have parent :=
    ConcreteNifsPiRlcParentSemantics.materialized_parent_eq_derived
      application profile frame assignment running fresh proof output
      productBase constantWire decodedInputs decodedOutput operationalSatisfied
      actionSatisfied pointSatisfied
  refine {
    commitment := ?_
    publicInput := ?_
    evaluations := ?_
  }
  · calc
      (ConcretePhi81.derive
          (selectedContext (keys := keys) running fresh proof)
          proof.certificate).piRlcOutput.commitment =
          output.parent.commitment :=
        (congrArg (fun statement => statement.commitment) parent).symm
      _ = PiDECAlgebra.Commitment.recomposeCommitment
          (fun child =>
            (proof.certificate.piDecPayloads child).commitment) :=
        outputEquations.commitment
  · calc
      (ConcretePhi81.derive
          (selectedContext (keys := keys) running fresh proof)
          proof.certificate).piRlcOutput.publicInput =
          output.parent.publicInput :=
        (congrArg (fun statement => statement.publicInput) parent).symm
      _ = PiDECAlgebra.PublicInput.recomposePublicInput
          (fun child =>
            (proof.certificate.piDecPayloads child).publicInput) :=
        outputEquations.publicInput
  · calc
      (ConcretePhi81.derive
          (selectedContext (keys := keys) running fresh proof)
          proof.certificate).piRlcOutput.evaluations =
          output.parent.evaluations :=
        (congrArg (fun statement => statement.evaluations) parent).symm
      _ = PiDEC.recomposeEvaluations
          (shape := RelationShape shape publicRingColumns publicFits)
          (fun child =>
            (proof.certificate.piDecPayloads child).evaluations) :=
        outputEquations.evaluations
      _ = (ConcretePhi81.decAlgebra
          (selectedContext (keys := keys) running fresh proof).key
        ).recomposeEvaluations
          (fun child =>
            (proof.certificate.piDecPayloads child).evaluations) := by
        rfl

/-- Honest decoded outgoing recomposition equations satisfy exactly the
selected `Pi_DEC` slice without extending the assignment. -/
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
    (equations : OutputEquations output proof) :
    Satisfies
      (rows application profile frame)
      (numericAssignment (columnMap frame) assignment) := by
  let numeric := numericAssignment (columnMap frame) assignment
  have wire : numeric 0 = 1 :=
    ConcreteNifsOperationalOccurrenceSemantics.numericConstantWire
      application frame assignment constantWire
  have proofDecoded :=
    ConcreteNifsSelectedCallFrame.proof_decodes_of_frame_decodes
      (FamilyFor application) frame assignment running fresh proof decodedInputs
  have outputDecoded :=
    ConcreteNifsSelectedCallFrame.output_decodes_of_frame_decodes
      (FamilyFor application) frame assignment output decodedOutput
  have fHonest :
      ∀ coordinate ∈ fCoordinates application profile frame,
        PiDECAlgebra.Radix.recomposeScalar
            (fun child =>
              residue (lcEval numeric (coordinate.children child))) =
          residue (lcEval numeric coordinate.parent) := by
    intro coordinate member
    rcases List.mem_append.1 member with inCommitment | inPublic
    · unfold commitmentCoordinates at inCommitment
      rcases List.mem_flatMap.1 inCommitment with
        ⟨row, rowMember, laneMember⟩
      rcases List.mem_ofFn.1 rowMember with ⟨rowIndex, rfl⟩
      rcases List.mem_ofFn.1 laneMember with ⟨lane, rfl⟩
      have childrenDecoded :
          (fun child =>
            residue
              (lcEval numeric
                ((commitmentCoordinate application profile frame
                  rowIndex lane).children child))) =
            (fun child =>
              payloadCommitmentCoordinate child rowIndex lane proof) := by
        funext child
        exact ConcreteNifsCarrierFrame.proofF_decoded
          (FamilyFor application) frame
          (profile.payloadViews.commitment child rowIndex lane)
          assignment proof proofDecoded
      have parentDecoded :=
        ConcreteNifsCarrierFrame.outputF_decoded
          (FamilyFor application) frame
          (profile.runningViews.parentCommitment rowIndex lane)
          assignment output outputDecoded
      simp only [commitmentCoordinate] at childrenDecoded ⊢
      rw [childrenDecoded, parentDecoded]
      simpa only [payloadCommitmentCoordinate, parentCommitmentCoordinate,
        Phi81RadixRows.recomposeCommitment_apply] using
        congrArg
          (fun commitment => commitment rowIndex lane)
          equations.commitment.symm
    · unfold publicCoordinates at inPublic
      rcases List.mem_ofFn.1 inPublic with ⟨column, rfl⟩
      have childrenDecoded :
          (fun child =>
            residue
              (lcEval numeric
                ((publicCoordinate application profile frame column).children
                  child))) =
            (fun child => payloadPublicCoordinate child column proof) := by
        funext child
        exact ConcreteNifsCarrierFrame.proofF_decoded
          (FamilyFor application) frame
          (profile.payloadViews.publicInput child column)
          assignment proof proofDecoded
      have parentDecoded :=
        ConcreteNifsCarrierFrame.outputF_decoded
          (FamilyFor application) frame
          (profile.runningViews.parentPublic column)
          assignment output outputDecoded
      simp only [publicCoordinate] at childrenDecoded ⊢
      rw [childrenDecoded, parentDecoded]
      simpa only [payloadPublicCoordinate, parentPublicCoordinate,
        PiDECAlgebra.PublicInput.recomposePublicInput_apply] using
        congrFun equations.publicInput.symm column
  have kHonest :
      ∀ coordinate ∈ evaluationCoordinates application profile frame,
        Phi81RadixRows.combineK PiDEC.radixWeight
            (fun child =>
              KPointEquality.decoded numeric (coordinate.children child)) =
          KPointEquality.decoded numeric coordinate.parent := by
    intro coordinate member
    unfold evaluationCoordinates at member
    rcases List.mem_flatMap.1 member with
      ⟨matrix, matrixMember, laneMember⟩
    rcases List.mem_ofFn.1 matrixMember with ⟨matrixIndex, rfl⟩
    rcases List.mem_ofFn.1 laneMember with ⟨lane, rfl⟩
    have childrenDecoded :
        (fun child =>
          KPointEquality.decoded numeric
            ((evaluationCoordinate application profile frame
              matrixIndex lane).children child)) =
          (fun child =>
            payloadEvaluationCoordinate child matrixIndex lane proof) := by
      funext child
      exact ConcreteNifsCarrierFrame.proofK_decoded
        (FamilyFor application) frame
        (profile.payloadViews.evaluation child matrixIndex lane)
        assignment proof proofDecoded
    have parentDecoded :=
      ConcreteNifsCarrierFrame.outputK_decoded
        (FamilyFor application) frame
        (profile.runningViews.parentEvaluation matrixIndex lane)
        assignment output outputDecoded
    simp only [evaluationCoordinate] at childrenDecoded ⊢
    rw [childrenDecoded, parentDecoded]
    simp only [payloadEvaluationCoordinate, parentEvaluationCoordinate]
    have equal :=
      congrArg
        (fun values => values.getD matrixIndex.val ringKZero lane)
        equations.evaluations.symm
    change
      (PiDEC.recomposeEvaluations
          (shape := RelationShape shape publicRingColumns publicFits)
          (fun child =>
            (proof.certificate.piDecPayloads child).evaluations)).getD
            matrixIndex.val BaseLinear.evaluationZero lane =
        output.parent.evaluations.getD
          matrixIndex.val BaseLinear.evaluationZero lane at equal
    rw [Phi81RadixRows.recomposeEvaluations_get] at equal
    exact equal
  exact
    Phi81RadixRows.rows_honest
      (fCoordinates application profile frame)
      (evaluationCoordinates application profile frame)
      numeric wire fHonest kHonest

end SelectedFrame

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsPiDecSemantics
