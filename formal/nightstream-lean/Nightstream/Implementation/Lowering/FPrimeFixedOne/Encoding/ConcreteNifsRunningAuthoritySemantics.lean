import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsRunningAuthorityRows

/-!
Contract: exact semantic refinement of the Lean-owned incoming
running-authority rows.

Whole-frame decoding is the sole source of carrier values.  Row satisfaction
proves all four fields of the frozen `RunningAuthority.Equations`; no
recomposition equation, accepted result, or abstract terminal fact is a
premise.
-/

set_option autoImplicit false
set_option maxRecDepth 100000
set_option maxHeartbeats 1600000

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsRunningAuthoritySemantics

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
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsRunningAuthorityRows
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
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81

private abbrev TranscriptState := Poseidon2Duplex.State

/-- Coordinate equality determines two dimension-checked cube points. -/
private theorem cubePoint_eq_of_coordinates
    {Value : Type} {variables : Nat}
    (left right : CubePoint Value variables)
    (coordinates :
      ∀ coordinate : Fin variables,
        left.coordinates.get
            ⟨coordinate.val, by
              rw [left.dimension]
              exact coordinate.isLt⟩ =
          right.coordinates.get
            ⟨coordinate.val, by
              rw [right.dimension]
              exact coordinate.isLt⟩) :
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

/-- Equality rows on a carried pair imply equality of the decoded extension
values. -/
private theorem decoded_eq_of_consistency
    (assignment : Nat → Nat)
    (pairs : List (Carried × Carried))
    (constantWire : assignment 0 = 1)
    (satisfied : Satisfies (KConsistency.consistencyRows pairs) assignment)
    (pair : Carried × Carried) (member : pair ∈ pairs) :
    KPointEquality.decoded assignment pair.1 =
      KPointEquality.decoded assignment pair.2 := by
  have pairEqual :=
    KConsistency.consistencyRows_sound assignment pairs constantWire
      satisfied pair member
  apply KConcreteBridge.ofConcrete_injective
  rw [KPointEquality.ofConcrete_decoded,
    KPointEquality.ofConcrete_decoded]
  exact pairEqual

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

/-- Satisfaction of the complete running-authority slice implies the exact
frozen incoming equations. -/
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
    (constantWire : assignment frame.one = 1)
    (decoded :
      frame.operands.Decodes (FamilyFor application) assignment
        (.cons running (.cons fresh (.cons proof .nil))))
    (satisfied :
      Satisfies
        (rows application profile frame)
        (numericAssignment (columnMap frame) assignment)) :
    FixedActive.Canonical.RunningAuthority.Equations
      (ConcreteNifsParameters.context
        (keys Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected)
        running fresh proof) := by
  let numeric := numericAssignment (columnMap frame) assignment
  have wire : numeric 0 = 1 :=
    ConcreteNifsOperationalOccurrenceSemantics.numericConstantWire
      application frame assignment constantWire
  have runningDecoded :=
    ConcreteNifsSelectedCallFrame.running_decodes_of_frame_decodes
      (FamilyFor application) frame assignment running fresh proof decoded
  have admissible :
      ((FamilyFor application).codecFor (.data .running)).Admissible running :=
    Codec.admissible_of_decode
      ((FamilyFor application).codecFor (.data .running)) runningDecoded
  have pointSatisfied :
      Satisfies
        (KConsistency.consistencyRows
          (pointPairs application profile frame)) numeric := by
    intro row member
    exact satisfied row (List.mem_append_left _ member)
  have radixSatisfied :
      Satisfies
        (Phi81RadixRows.rows
          (fCoordinates application profile frame)
          (evaluationCoordinates application profile frame)) numeric := by
    intro row member
    exact satisfied row (List.mem_append_right _ member)
  refine {
    points := ?_
    commitment := ?_
    publicInput := ?_
    evaluations := ?_
  }
  · change
      (fun child => (running.children child).point) =
        (fun _ => running.parent.point)
    funext child
    apply cubePoint_eq_of_coordinates
    intro coordinate
    have equal :=
      decoded_eq_of_consistency numeric
        (pointPairs application profile frame) wire pointSatisfied
        (pointPair application profile frame child coordinate)
        (pointPair_mem application profile frame child coordinate)
    simp only [pointPair] at equal
    have childDecoded :=
      ConcreteNifsCarrierFrame.runningK_decoded
        (FamilyFor application) frame
        (profile.runningViews.childPoint child coordinate)
        assignment running runningDecoded
    have parentDecoded :=
      ConcreteNifsCarrierFrame.runningK_decoded
        (FamilyFor application) frame
        (profile.runningViews.parentPoint coordinate)
        assignment running runningDecoded
    rw [childDecoded, parentDecoded] at equal
    simpa only [childPointCoordinate, parentPointCoordinate] using equal
  · change
      running.parent.commitment =
        PiDECAlgebra.Commitment.recomposeCommitment
          (fun child => (running.children child).commitment)
    funext row lane
    have equation :=
      Phi81RadixRows.rows_sound_f
        (fCoordinates application profile frame)
        (evaluationCoordinates application profile frame)
        numeric wire radixSatisfied
        (commitmentCoordinate application profile frame row lane)
        (commitmentCoordinate_mem application profile frame row lane)
    simp only [commitmentCoordinate] at equation
    have childrenDecoded :
        (fun child =>
          residue
            (lcEval numeric
              (ConcreteNifsCarrierFrame.runningFLocation
                (FamilyFor application) frame
                (profile.runningViews.childCommitment child row lane)).carried)) =
          (fun child =>
            childCommitmentCoordinate child row lane running) := by
      funext child
      exact ConcreteNifsCarrierFrame.runningF_decoded
        (FamilyFor application) frame
        (profile.runningViews.childCommitment child row lane)
        assignment running runningDecoded
    have parentDecoded :=
      ConcreteNifsCarrierFrame.runningF_decoded
        (FamilyFor application) frame
        (profile.runningViews.parentCommitment row lane)
        assignment running runningDecoded
    rw [childrenDecoded, parentDecoded] at equation
    rw [Phi81RadixRows.recomposeCommitment_apply]
    simpa only [childCommitmentCoordinate, parentCommitmentCoordinate] using
      equation.symm
  · change
      running.parent.publicInput =
        PiDECAlgebra.PublicInput.recomposePublicInput
          (fun child => (running.children child).publicInput)
    funext column
    have equation :=
      Phi81RadixRows.rows_sound_f
        (fCoordinates application profile frame)
        (evaluationCoordinates application profile frame)
        numeric wire radixSatisfied
        (publicCoordinate application profile frame column)
        (publicCoordinate_mem application profile frame column)
    simp only [publicCoordinate] at equation
    have childrenDecoded :
        (fun child =>
          residue
            (lcEval numeric
              (ConcreteNifsCarrierFrame.runningFLocation
                (FamilyFor application) frame
                (profile.runningViews.childPublic child column)).carried)) =
          (fun child => childPublicCoordinate child column running) := by
      funext child
      exact ConcreteNifsCarrierFrame.runningF_decoded
        (FamilyFor application) frame
        (profile.runningViews.childPublic child column)
        assignment running runningDecoded
    have parentDecoded :=
      ConcreteNifsCarrierFrame.runningF_decoded
        (FamilyFor application) frame
        (profile.runningViews.parentPublic column)
        assignment running runningDecoded
    rw [childrenDecoded, parentDecoded] at equation
    rw [PiDECAlgebra.PublicInput.recomposePublicInput_apply]
    simpa only [childPublicCoordinate, parentPublicCoordinate] using
      equation.symm
  · change
      running.parent.evaluations =
        PiDEC.recomposeEvaluations
          (shape := RelationShape shape publicRingColumns publicFits)
          (fun child => (running.children child).evaluations)
    have parentSize :
        running.parent.evaluations.size = shape.matrixCount :=
      profile.runningViews.parentEvaluationsSize running admissible
    have childSize :
        ∀ child,
          (running.children child).evaluations.size = shape.matrixCount :=
      profile.runningViews.childEvaluationsSize running admissible
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
          numeric wire radixSatisfied
          (evaluationCoordinate application profile frame matrix lane)
          (evaluationCoordinate_mem application profile frame matrix lane)
      simp only [evaluationCoordinate] at equation
      have childrenDecoded :
          (fun child =>
            KPointEquality.decoded numeric
              (ConcreteNifsCarrierFrame.runningKLocation
                (FamilyFor application) frame
                (profile.runningViews.childEvaluation child matrix lane)).carried) =
            (fun child =>
              childEvaluationCoordinate child matrix lane running) := by
        funext child
        exact ConcreteNifsCarrierFrame.runningK_decoded
          (FamilyFor application) frame
          (profile.runningViews.childEvaluation child matrix lane)
          assignment running runningDecoded
      have parentDecoded :=
        ConcreteNifsCarrierFrame.runningK_decoded
          (FamilyFor application) frame
          (profile.runningViews.parentEvaluation matrix lane)
          assignment running runningDecoded
      rw [childrenDecoded, parentDecoded] at equation
      simp only [childEvaluationCoordinate, parentEvaluationCoordinate] at equation
      have recomposed :=
        Phi81RadixRows.recomposeEvaluations_get
          (shape :=
            RelationShape shape publicRingColumns publicFits)
          (fun child => (running.children child).evaluations)
          matrix lane
      calc
        running.parent.evaluations[index] lane =
            running.parent.evaluations.getD index ringKZero lane := by
          rw [Array.getD_eq_getD_getElem?,
            Array.getElem?_eq_getElem leftLt]
          simp
        _ = Phi81RadixRows.combineK PiDEC.radixWeight
              (fun child =>
                (running.children child).evaluations.getD
                  index BaseLinear.evaluationZero lane) :=
          equation.symm
        _ =
            (PiDEC.recomposeEvaluations
              (shape := RelationShape shape publicRingColumns publicFits)
              (fun child =>
                (running.children child).evaluations)).getD
                  index BaseLinear.evaluationZero lane :=
          recomposed.symm
        _ =
            (PiDEC.recomposeEvaluations
              (shape := RelationShape shape publicRingColumns publicFits)
              (fun child =>
                (running.children child).evaluations))[index] lane := by
          rw [Array.getD_eq_getD_getElem?,
            Array.getElem?_eq_getElem rightLt]
          simp
          rfl

/-- Honest incoming running equations satisfy exactly the emitted slice under
the caller's decoded assignment.  The slice allocates no witness columns. -/
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
    (constantWire : assignment frame.one = 1)
    (decoded :
      frame.operands.Decodes (FamilyFor application) assignment
        (.cons running (.cons fresh (.cons proof .nil))))
    (equations :
      FixedActive.Canonical.RunningAuthority.Equations
        (ConcreteNifsParameters.context
          (keys Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected)
          running fresh proof)) :
    Satisfies
      (rows application profile frame)
      (numericAssignment (columnMap frame) assignment) := by
  let numeric := numericAssignment (columnMap frame) assignment
  have wire : numeric 0 = 1 :=
    ConcreteNifsOperationalOccurrenceSemantics.numericConstantWire
      application frame assignment constantWire
  have runningDecoded :=
    ConcreteNifsSelectedCallFrame.running_decodes_of_frame_decodes
      (FamilyFor application) frame assignment running fresh proof decoded
  have pointsEquation :
      (fun child => (running.children child).point) =
        (fun _ => running.parent.point) := by
    exact equations.points
  have commitmentEquation :
      running.parent.commitment =
        PiDECAlgebra.Commitment.recomposeCommitment
          (fun child => (running.children child).commitment) := by
    exact equations.commitment
  have publicEquation :
      running.parent.publicInput =
        PiDECAlgebra.PublicInput.recomposePublicInput
          (fun child => (running.children child).publicInput) := by
    exact equations.publicInput
  have evaluationEquation :
      running.parent.evaluations =
        PiDEC.recomposeEvaluations
          (shape := RelationShape shape publicRingColumns publicFits)
          (fun child => (running.children child).evaluations) := by
    exact equations.evaluations
  have pointsHonest :
      Satisfies
        (KConsistency.consistencyRows
          (pointPairs application profile frame)) numeric := by
    apply KConsistency.consistencyRows_honest
      numeric (pointPairs application profile frame) wire
    intro pair member
    unfold pointPairs at member
    rcases List.mem_flatMap.1 member with
      ⟨child, childMember, coordinateMember⟩
    rcases List.mem_ofFn.1 childMember with ⟨childIndex, rfl⟩
    rcases List.mem_ofFn.1 coordinateMember with ⟨coordinate, rfl⟩
    change
      carriedValue numeric
          (ConcreteNifsCarrierFrame.runningKLocation
            (FamilyFor application) frame
            (profile.runningViews.childPoint childIndex coordinate)).carried =
        carriedValue numeric
          (ConcreteNifsCarrierFrame.runningKLocation
            (FamilyFor application) frame
            (profile.runningViews.parentPoint coordinate)).carried
    have childDecoded :=
      ConcreteNifsCarrierFrame.runningK_decoded
        (FamilyFor application) frame
        (profile.runningViews.childPoint childIndex coordinate)
        assignment running runningDecoded
    have parentDecoded :=
      ConcreteNifsCarrierFrame.runningK_decoded
        (FamilyFor application) frame
        (profile.runningViews.parentPoint coordinate)
        assignment running runningDecoded
    have semanticEqual :
        childPointCoordinate childIndex coordinate running =
          parentPointCoordinate coordinate running := by
      unfold childPointCoordinate parentPointCoordinate
      rw [congrFun pointsEquation childIndex]
    have decodedEqual :
        KPointEquality.decoded numeric
            (ConcreteNifsCarrierFrame.runningKLocation
              (FamilyFor application) frame
              (profile.runningViews.childPoint childIndex coordinate)).carried =
          KPointEquality.decoded numeric
            (ConcreteNifsCarrierFrame.runningKLocation
              (FamilyFor application) frame
              (profile.runningViews.parentPoint coordinate)).carried := by
      rw [childDecoded, parentDecoded]
      exact semanticEqual
    have represented := congrArg KConcreteBridge.ofConcrete decodedEqual
    simpa only [KPointEquality.ofConcrete_decoded] using represented
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
              childCommitmentCoordinate child rowIndex lane running) := by
        funext child
        exact ConcreteNifsCarrierFrame.runningF_decoded
          (FamilyFor application) frame
          (profile.runningViews.childCommitment child rowIndex lane)
          assignment running runningDecoded
      have parentDecoded :=
        ConcreteNifsCarrierFrame.runningF_decoded
          (FamilyFor application) frame
          (profile.runningViews.parentCommitment rowIndex lane)
          assignment running runningDecoded
      simp only [commitmentCoordinate] at childrenDecoded ⊢
      rw [childrenDecoded, parentDecoded]
      simpa only [childCommitmentCoordinate, parentCommitmentCoordinate,
        Phi81RadixRows.recomposeCommitment_apply] using
        congrArg
          (fun commitment => commitment rowIndex lane)
          commitmentEquation.symm
    · unfold publicCoordinates at inPublic
      rcases List.mem_ofFn.1 inPublic with ⟨column, rfl⟩
      have childrenDecoded :
          (fun child =>
            residue
              (lcEval numeric
                ((publicCoordinate application profile frame column).children
                  child))) =
            (fun child => childPublicCoordinate child column running) := by
        funext child
        exact ConcreteNifsCarrierFrame.runningF_decoded
          (FamilyFor application) frame
          (profile.runningViews.childPublic child column)
          assignment running runningDecoded
      have parentDecoded :=
        ConcreteNifsCarrierFrame.runningF_decoded
          (FamilyFor application) frame
          (profile.runningViews.parentPublic column)
          assignment running runningDecoded
      simp only [publicCoordinate] at childrenDecoded ⊢
      rw [childrenDecoded, parentDecoded]
      simpa only [childPublicCoordinate, parentPublicCoordinate,
        PiDECAlgebra.PublicInput.recomposePublicInput_apply] using
        congrFun publicEquation.symm column
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
            childEvaluationCoordinate child matrixIndex lane running) := by
      funext child
      exact ConcreteNifsCarrierFrame.runningK_decoded
        (FamilyFor application) frame
        (profile.runningViews.childEvaluation child matrixIndex lane)
        assignment running runningDecoded
    have parentDecoded :=
      ConcreteNifsCarrierFrame.runningK_decoded
        (FamilyFor application) frame
        (profile.runningViews.parentEvaluation matrixIndex lane)
        assignment running runningDecoded
    simp only [evaluationCoordinate] at childrenDecoded ⊢
    rw [childrenDecoded, parentDecoded]
    simp only [childEvaluationCoordinate, parentEvaluationCoordinate]
    have equal :=
      congrArg
        (fun values =>
          values.getD matrixIndex.val ringKZero lane)
        evaluationEquation.symm
    change
      (PiDEC.recomposeEvaluations
          (shape := RelationShape shape publicRingColumns publicFits)
          (fun child => (running.children child).evaluations)).getD
            matrixIndex.val BaseLinear.evaluationZero lane =
        running.parent.evaluations.getD
          matrixIndex.val BaseLinear.evaluationZero lane at equal
    rw [Phi81RadixRows.recomposeEvaluations_get] at equal
    exact equal
  intro row member
  rcases List.mem_append.1 member with inPoints | inRadix
  · exact pointsHonest row inPoints
  · exact
      Phi81RadixRows.rows_honest
        (fCoordinates application profile frame)
        (evaluationCoordinates application profile frame)
        numeric wire fHonest kHonest row inRadix

end SelectedFrame

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsRunningAuthoritySemantics
