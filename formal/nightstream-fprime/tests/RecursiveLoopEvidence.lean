import tests.CheckedReplayEvidence
import tests.EvidenceMetadata

/-! Exact accepted-successor and next-prior target for the fresh recursive
replay. Source, evaluation, returned-block, carrier, input and context custody
remain explicit. Full execution and byte comparisons are separate evidence. -/

set_option autoImplicit false

namespace LeanGraph.Targets

open NightstreamFPrime
open Export.Stage1
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.Nifs.StoredAssignmentArithmetic (StoredAssignment view)
open NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism
open NightstreamFPrime.Spec.Phi81Relation.PiDECAlgebra
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Layout.Stage1
open PerApplicationCanonicalAssignment
open PiRLCNonzero (sourceIndex)
open PiRLCPartialTrace (MaterializedRingF)
open Poseidon2HashChainV1Package (application fits)
open Poseidon2HashChainV1Setup (productionSetup)

private abbrev RunningWitness := Stage1.Terminal.RunningWitness
  (logicalWidth := PerApplicationFixedPoint.logicalWidth application)
  (publicFits := PerApplicationFixedPoint.publicFits application)
private abbrev FreshWitness := Stage1.Terminal.FreshWitness
  (logicalWidth := PerApplicationFixedPoint.logicalWidth application)
  (publicFits := PerApplicationFixedPoint.publicFits application)

def CheckedRecursiveReplay : Prop :=
  ∀ (statement : PerApplicationTerminal.Statement) (input : PiCCSInputCheck.Input)
    (runningWitness : RunningWitness) (freshWitness : FreshWitness)
    (advice : AppWitness) (masks : Array (Array (Nat × Nat)))
    (batch : PiRLCParent.Batch) (parent : PiRLCParent.Values)
    (parentWitness : StoredAssignment PiCCSSourceImages.shape.carrierWidth)
    (children : PiDECComputedChildren.Children)
    (raw : RawValues application) (bytes : ByteArray)
    (accepted : PerApplicationTerminal.Holds application fits productionSetup statement
      (.recursive (CheckedReplayStep.prior input runningWitness freshWitness)))
    (nonwrap : statement.iteration + 1 < goldilocksModulus)
    (sourceCustody : ∀ source, CheckedReplayParent.assignments masks source =
      Fin.addCases (fun _ : Fin productionShape.freshCount => freshWitness)
        runningWitness (sourceIndex source))
    (evaluationCustody : ∀ source, PiRLCInputCheck.evaluations input source =
      CheckedReplayParent.evaluations masks (PiCCSInputCheck.execute input).point source)
    (sampled : PiRLCInputCheck.sampled input = some batch)
    (returned : PiRLCParent.computedParent input batch = some parent)
    (parentBlocks : ∀ block,
      ((PiRLCWitnessBlock.preparedWitnessBlockPartials batch.challenges
        (PiRLCWitnessBlock.prepareWitnessActions batch.challenges)
        (fun source => MaterializedRingF.ofRing
          (CarrierAction.assignmentBlock (CheckedReplayParent.assignments masks source) block))).map
        MaterializedRingF.toRing).getLast? =
        some (CarrierAction.assignmentBlock (view parentWitness) block))
    (split : StoredSplit.splitChecked parentWitness = some children)
    (checked : PiDECInputCheck.accepted parent (PiDECComputedChildren.messages parent children) = true)
    (carrierCustody : FreshRowsCheck.carrierRead bytes = raw.completeAssignment)
    (rowCheck : FreshRowsCheck.checkProgram (PerApplicationMatrixProgram.matrixProgram application)
      (fun source => (PiDECCanonicalSourceCache.stored application)[source]?)
      (FreshRowsCheck.logicalRead bytes) = true)
    (inputCustody : PerApplicationDecodedIO.input application fits raw =
      HyperNovaStepData.input statement (CheckedReplayStep.prior input runningWitness freshWitness)
        advice (CheckedReplayNifs.proof input (PiDECComputedChildren.messages parent children)))
    (contextCustody : PerApplicationDecodedIO.contextKey raw =
      PerApplicationCanonicalPackage.verifierContextDigest fits productionSetup),
    let result := PiCCSInputCheck.runningFromInput (PiDECComputedChildren.messages parent children)
    let openings := PiDECComputedChildren.witnesses children
    let successor := CheckedReplaySuccessor.payload result openings raw
    let nextStatement := CheckedReplaySuccessor.nextStatement statement advice
    PerApplicationTerminal.Holds application fits productionSetup nextStatement (.recursive successor) ∧
      ∀ nextProof : Lifecycle.Proof 9,
        PerApplicationTerminal.Holds application fits productionSetup nextStatement
          (.recursive (CheckedReplayStep.prior
            (HyperNovaInput.ofClaims result successor.fresh nextProof)
            openings raw.completeAssignment))

theorem checkedRecursiveReplay : CheckedRecursiveReplay :=
  CheckedReplayComposition.accepted_and_handoff

#audit_axioms checkedRecursiveReplay

end LeanGraph.Targets
