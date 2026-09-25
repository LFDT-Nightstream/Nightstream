import NightstreamFPrime.Export.Stage1.CheckedReplayStep
import NightstreamFPrime.Export.Stage1.PiCCSOriginalPadPreservation
import NightstreamFPrime.Export.Stage1.PiCCSOriginalMatrixRange
import NightstreamFPrime.Export.Stage1.PiRLCWitnessHonestResponse

/-! Derive the actual computed R parent opening from the original seventeen
openings and the existing C/R source kernels. Custody names exact computed
field/block outputs; no semantic parent-correctness callback is supplied. -/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.CheckedReplayParent

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.Nifs.StoredAssignmentArithmetic (StoredAssignment view)
open NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Layout.Stage1
open PiRLCNonzero (SourceCount sourceIndex)
open PiRLCPartialTrace (MaterializedRingF)
open Poseidon2HashChainV1Package (application fits)
open Poseidon2HashChainV1Setup (productionSetup productionAjtaiKey)

private abbrev shape := PiCCSSourceImages.shape

def assignments (masks : Array (Array (Nat × Nat))) :
    Fin SourceCount → Phi81Relation.Assignment shape :=
  fun source => PiCCSOriginalReads.assignment masks (sourceIndex source)

/-- The same complete original-source C evaluation kernels used in the replay. -/
def evaluations (masks : Array (Array (Nat × Nat))) (point : PaperAlgebra.Point)
    (source : Fin SourceCount) : PaperAlgebra.Evaluation where
  pad := ((PiCCSOriginalPad.range 0 PiCCSSourceImages.blockCount point masks).get
    (sourceIndex source)).toRing
  matrix := fun port =>
    ((PiCCSOriginalMatrixRange.originalRange masks point 0
      (PerApplicationMatrixProgram.matrixProgram application).rowCount
      (sourceIndex source)).get port).toRing

private theorem evaluation_ext (left right : PaperAlgebra.Evaluation)
    (pad : left.pad = right.pad) (matrix : left.matrix = right.matrix) : left = right := by
  cases left
  cases right
  cases pad
  cases matrix
  rfl

theorem evaluations_eq_family (masks : Array (Array (Nat × Nat)))
    (point : PaperAlgebra.Point) (source : Fin SourceCount) :
    evaluations masks point source =
      PaperAlgebra.evaluationFamily
        (Lifecycle.PiRLC.v1_1.InputBinding.relationSource PiDECInputCheck.relation)
        (assignments masks source) point := by
  apply evaluation_ext
  · exact PiCCSOriginalPad.complete_eq_evaluationFamily point masks (sourceIndex source)
  · funext port
    have value := (PiCCSOriginalMatrixRange.range_eq_matrix masks point
      (sourceIndex source) port).trans
      (PiCCSOriginalMatrixPreservation.matrix_eq_evaluationFamily masks point
        (sourceIndex source) port)
    change ((PiCCSOriginalMatrixRange.originalRange masks point 0
      (PerApplicationMatrixProgram.matrixProgram application).rowCount
      (sourceIndex source)).get port).toRing =
      (PaperAlgebra.evaluationFamily
        (Lifecycle.PiRLC.v1_1.InputBinding.relationSource
          (PerApplicationFixedPoint.relation application fits))
        (assignments masks source) point).matrix port at value
    rw [← PiDECInputCheck.relation_eq_selected] at value
    exact value

private theorem opening_family
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth}
    (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (running : Running (logicalWidth := logicalWidth) (publicFits := publicFits))
    (fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
    (runningWitness : Stage1.Terminal.RunningWitness
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (freshWitness : Stage1.Terminal.FreshWitness
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (runningOpenings : ∀ child, Opening.Holds (semantics ajtai) productionGlobalParams.b
      (running.commitments child) (running.publicInputs child) (runningWitness child))
    (freshOpening : Opening.Holds (semantics ajtai) productionGlobalParams.b
      (fresh.commitments ⟨0, by decide⟩) (fresh.publicInputs ⟨0, by decide⟩) freshWitness) :
    ∀ source : Fin productionShape.sourceCount,
      Opening.Holds (semantics ajtai) productionGlobalParams.b
        (Fin.addCases fresh.commitments running.commitments source)
        (Fin.addCases fresh.publicInputs running.publicInputs source)
        (Fin.addCases (fun _ : Fin productionShape.freshCount => freshWitness)
          runningWitness source) := by
  intro source
  refine Fin.addCases (m := productionShape.freshCount) (n := productionShape.runningCount)
    ?_ ?_ source
  · intro index
    have zero : index = ⟨0, by decide⟩ := by
      apply Fin.ext
      change index.val = 0
      have bound : index.val < 1 := index.isLt
      omega
    subst index
    simpa only [Fin.addCases_left] using freshOpening
  · intro index
    simpa only [Fin.addCases_right] using runningOpenings index

private theorem block_ext {carrier : Phi81Relation.Shape}
    (left right : Phi81Relation.Assignment carrier)
    (same : ∀ block, CarrierAction.assignmentBlock left block =
      CarrierAction.assignmentBlock right block) : left = right := by
  funext column
  let pair := Phi81ColumnLayout.decode column
  have recovered : CarrierAction.carrierColumn pair.1 pair.2 = column := by
    apply Fin.ext
    exact Phi81ColumnLayout.flatIndex_decode column
  have lane := congrFun (same pair.1) pair.2
  change left (CarrierAction.carrierColumn pair.1 pair.2) =
    right (CarrierAction.carrierColumn pair.1 pair.2) at lane
  simpa only [recovered] using lane

/-- A returned prepared R block is the block of the actual complete-carrier
combination. Every block is covered; no stored output is taken as authoritative. -/
theorem returned_witness
    (masks : Array (Array (Nat × Nat))) (batch : PiRLCParent.Batch)
    (parentWitness : StoredAssignment shape.carrierWidth)
    (blocks : ∀ block,
      ((PiRLCWitnessBlock.preparedWitnessBlockPartials batch.challenges
        (PiRLCWitnessBlock.prepareWitnessActions batch.challenges)
        (fun source => MaterializedRingF.ofRing
          (CarrierAction.assignmentBlock (assignments masks source) block))).map
        MaterializedRingF.toRing).getLast? =
        some (CarrierAction.assignmentBlock (view parentWitness) block)) :
    view parentWitness = PiRLCFinite.combineAssignments batch.challenges (assignments masks) := by
  apply block_ext
  intro block
  exact Option.some.inj ((blocks block).symm.trans
    (PiRLCWitnessBlock.preparedWitnessBlockPartials_getLast? batch.challenges
      (assignments masks) block))

/-- This is the exact ParentValid premise required by computed D children.
Initial acceptance supplies old openings. New evaluation and R-block custody
refer to concrete source kernels at the actual C point and sampled rho. -/
theorem parent_opening
    (statement : PerApplicationTerminal.Statement) (input : PiCCSInputCheck.Input)
    (runningWitness : Stage1.Terminal.RunningWitness
      (logicalWidth := PerApplicationFixedPoint.logicalWidth application)
      (publicFits := PerApplicationFixedPoint.publicFits application))
    (freshWitness : Stage1.Terminal.FreshWitness
      (logicalWidth := PerApplicationFixedPoint.logicalWidth application)
      (publicFits := PerApplicationFixedPoint.publicFits application))
    (masks : Array (Array (Nat × Nat))) (batch : PiRLCParent.Batch)
    (parent : PiRLCParent.Values) (parentWitness : StoredAssignment shape.carrierWidth)
    (accepted : PerApplicationTerminal.Holds application fits productionSetup statement
      (.recursive (CheckedReplayStep.prior input runningWitness freshWitness)))
    (sourceCustody : ∀ source, assignments masks source =
      Fin.addCases (fun _ : Fin productionShape.freshCount => freshWitness)
        runningWitness (sourceIndex source))
    (evaluationCustody : ∀ source, PiRLCInputCheck.evaluations input source =
      evaluations masks (PiCCSInputCheck.execute input).point source)
    (sampled : PiRLCInputCheck.sampled input = some batch)
    (returned : PiRLCParent.computedParent input batch = some parent)
    (blocks : ∀ block,
      ((PiRLCWitnessBlock.preparedWitnessBlockPartials batch.challenges
        (PiRLCWitnessBlock.prepareWitnessActions batch.challenges)
        (fun source => MaterializedRingF.ofRing
          (CarrierAction.assignmentBlock (assignments masks source) block))).map
        MaterializedRingF.toRing).getLast? =
        some (CarrierAction.assignmentBlock (view parentWitness) block)) :
    CE.Holds (semantics productionAjtaiKey) productionGlobalParams
      (PiDECInputCheck.parent parent) (view parentWitness) := by
  obtain ⟨_, _, _, _, runningValid, freshValid⟩ :=
    (PerApplicationTerminal.holds_recursive_iff application fits productionSetup statement
      (CheckedReplayStep.prior input runningWitness freshWitness)).mp accepted
  have old := opening_family productionAjtaiKey (PiCCSInputCheck.running input)
    (PiCCSInputCheck.fresh input) runningWitness freshWitness
    (fun child => (runningValid functionIndex child).1) freshValid.1
  have inputValid : ∀ source, CE.Holds (semantics productionAjtaiKey) productionGlobalParams
      (PiRLCParent.sourceClaim input source) (assignments masks source) := by
    intro source
    refine ⟨?_, trivial, ?_⟩
    · rw [sourceCustody source]
      exact old (sourceIndex source)
    · change #[PaperAlgebra.evaluationFamily
          (Lifecycle.PiRLC.v1_1.InputBinding.relationSource PiDECInputCheck.relation)
          (assignments masks source) (PiCCSInputCheck.execute input).point] =
        #[PiRLCInputCheck.evaluations input source]
      exact congrArg (fun value : PaperAlgebra.Evaluation => #[value])
        ((evaluationCustody source).trans
          (evaluations_eq_family masks (PiCCSInputCheck.execute input).point source)).symm
  have challengeValid := ProductionKey.piRlcResponse_valid
    (PiCCSInputCheck.execute input).outgoing batch.challenges
    (PiRLCInputCheck.sampled_response input batch sampled).2
  have combined := PiRLC.combinedOutput_holds (semantics productionAjtaiKey) productionGlobalParams
    (piRlcAlgebra productionAjtaiKey) Nifs.PaperProfile.arity
    (PiRLCParent.inputBatch input).system (PiRLCParent.inputBatch input).point
    (PiRLCParent.inputBatch input).inputs batch.challenges (assignments masks)
    (fun _ => rfl) (PiRLCParent.inputBatch input).sameSystem
    (PiRLCParent.inputBatch input).samePoint challengeValid inputValid trivial
  have parentEq := PiRLCParent.computedParent_eq_combined input batch parent returned
  have witnessEq := returned_witness masks batch parentWitness blocks
  exact Eq.mpr (congrArg₂ (CE.Holds (semantics productionAjtaiKey) productionGlobalParams)
    parentEq witnessEq) combined

end NightstreamFPrime.Export.Stage1.CheckedReplayParent
