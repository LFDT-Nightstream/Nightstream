import NightstreamFPrime.Export.Stage1.CheckedReplayParent
import NightstreamFPrime.Export.Stage1.PiDECComputedChildren
import NightstreamFPrime.Export.Stage1.FreshRowsCheck
import NightstreamFPrime.Export.Stage1.CheckedReplayHandoff

/-! Exact accepted successor and literal second-prior handoff for the supplied
C/R/D replay, source files and checked fresh carrier. The only semantic starting
premise is acceptance of the actual original prior with its original openings. -/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.CheckedReplayComposition

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

/-- The actual computed D fields and fresh carrier form an accepted successor.
The second conclusion can be used literally as the next call's prior.
Custody premises connect supplied bytes to the exact source/field constructors;
all output, parent opening, child openings, row validity and norm conclusions
are derived by the source and check theorems. -/
theorem accepted_and_handoff
    (statement : PerApplicationTerminal.Statement) (input : PiCCSInputCheck.Input)
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
      PerApplicationCanonicalPackage.verifierContextDigest fits productionSetup) :
    let result := PiCCSInputCheck.runningFromInput (PiDECComputedChildren.messages parent children)
    let openings := PiDECComputedChildren.witnesses children
    let successor := CheckedReplaySuccessor.payload result openings raw
    let nextStatement := CheckedReplaySuccessor.nextStatement statement advice
    PerApplicationTerminal.Holds application fits productionSetup nextStatement (.recursive successor) ∧
      ∀ nextProof : Lifecycle.Proof 9,
        PerApplicationTerminal.Holds application fits productionSetup nextStatement
          (.recursive (CheckedReplayStep.prior
            (HyperNovaInput.ofClaims result successor.fresh nextProof)
            openings raw.completeAssignment)) := by
  dsimp only
  have parentValid := CheckedReplayParent.parent_opening statement input runningWitness freshWitness
    masks batch parent parentWitness accepted sourceCustody evaluationCustody sampled returned parentBlocks
  have childOpenings := PiDECComputedChildren.child_openings
    parent parentWitness children split parentValid
  have canonical := FreshRowsCheck.checked_raw bytes raw carrierCustody rowCheck
  have successor := CheckedReplaySuccessor.accepted_of_checked_rows statement input batch parent
    (PiDECComputedChildren.messages parent children) runningWitness freshWitness advice raw
    (PiDECComputedChildren.witnesses children) sampled returned checked accepted nonwrap
    canonical.1 canonical.2 inputCustody contextCustody childOpenings
  refine ⟨successor, ?_⟩
  intro nextProof
  rw [CheckedReplayHandoff.prior_eq_payload]
  exact successor

end NightstreamFPrime.Export.Stage1.CheckedReplayComposition
