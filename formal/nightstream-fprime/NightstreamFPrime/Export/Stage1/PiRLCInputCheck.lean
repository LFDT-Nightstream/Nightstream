import NightstreamFPrime.Export.Stage1.PiCCSInputCheck
import NightstreamFPrime.Export.Stage1.PiRLCParity
import NightstreamFPrime.Spec.Folding.Nifs.PaperStrongInterface

/-!
Continue the actual checked PiCCS input through the selected PiRLC sampler
and all 17 indexed combinations. The input is the existing PiCCS schema;
the point, transcript state and source claims come from that execution.
The result preserves the complete PiCCS fields, then appends the PiRLC input
and result in the existing parity encoding. Opening validity remains with
the source-opening owner.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiRLCInputCheck

open NightstreamFPrime.Export.Codec
open NightstreamFPrime.Spec
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.StrongReduction
open PiRLCNonzero PiRLCPartialTrace

abbrev Input := PiCCSInputCheck.Input

def commitments (input : Input) (source : Fin SourceCount) : PaperAlgebra.Commitment :=
  PiCCSInputCheck.outputCommitments input (sourceIndex source)

def publicInputs (input : Input) (source : Fin SourceCount) : Fin 270 → F :=
  PiCCSInputCheck.outputPublicInputs input (sourceIndex source)

def evaluations (input : Input) (source : Fin SourceCount) : PaperAlgebra.Evaluation where
  pad := (input.evalK.get source).get
  matrix := fun matrix => ((input.evalA.get source).get matrix).get

variable
  (relation : ProductionKey.LogicalRelation
    (PerApplicationFixedPoint.logicalWidth Poseidon2HashChainV1Package.application)
    (PerApplicationFixedPoint.publicFits Poseidon2HashChainV1Package.application))
  (ajtai : PaperAlgebra.AjtaiKey
    (logicalWidth := PerApplicationFixedPoint.logicalWidth Poseidon2HashChainV1Package.application)
    (publicFits := PerApplicationFixedPoint.publicFits Poseidon2HashChainV1Package.application))

/-- The emitted commitment input is the selected NIFS batch's exact Phi. -/
theorem commitments_eq_batch (input : Input) (probe : Probe K productionShape)
    (source : Fin SourceCount) :
    commitments input source =
      ((Folding.Nifs.PaperStrongInterface.piRlcBatchForProbe (ProductionKey.key relation ajtai)
        (PiCCSInputCheck.running input) (PiCCSInputCheck.fresh input) probe).inputs source).commitment := rfl

theorem publicInputs_eq_batch (input : Input) (probe : Probe K productionShape)
    (source : Fin SourceCount) :
    publicInputs input source =
      ((Folding.Nifs.PaperStrongInterface.piRlcBatchForProbe (ProductionKey.key relation ajtai)
        (PiCCSInputCheck.running input) (PiCCSInputCheck.fresh input) probe).inputs source).publicInput := rfl

theorem evaluations_eq_batch (input : Input) (probe : Probe K productionShape)
    (sameOutput : probe.response.fullOutput =
      Layout.Stage1.PiCCSProofInputs.output (PiCCSInputCheck.proofValues input))
    (source : Fin SourceCount) :
    #[evaluations input source] =
      ((Folding.Nifs.PaperStrongInterface.piRlcBatchForProbe (ProductionKey.key relation ajtai)
        (PiCCSInputCheck.running input) (PiCCSInputCheck.fresh input) probe).inputs source).evaluations := by
  change #[evaluations input source] = #[_]
  simp only [sameOutput]
  rfl

/-- Rejected PiCCS inputs do not reach the sampler. -/
def sampled (input : Input) : Option (Transcript.PiRlcSampler.Batch SourceCount) :=
  let ccsResult := PiCCSInputCheck.execute input
  if ccsResult.accepted then
    Transcript.PiRlcSampler.piRlcChallengesWithState ccsResult.outgoing SourceCount
  else none

theorem sampled_on_rejection (input : Input)
    (rejected : (PiCCSInputCheck.execute input).accepted = false) : sampled input = none := by
  simp only [sampled, rejected, Bool.false_eq_true, ↓reduceIte]

/-- The actual endpoint gives the production key response, including its
fail-closed sampler behavior. This is an execution equality, not a coin law. -/
theorem sampled_response (input : Input) (batch : Transcript.PiRlcSampler.Batch SourceCount)
    (returned : sampled input = some batch) :
    (PiCCSInputCheck.execute input).accepted = true ∧
      ProductionKey.piRlcResponse (PiCCSInputCheck.execute input).outgoing = some batch.challenges := by
  dsimp only [sampled] at returned
  split at returned
  · rename_i accepted
    refine ⟨accepted, ?_⟩
    simp only [ProductionKey.piRlcResponse, Transcript.PiRlcSampler.piRlcChallenges,
      returned, Option.map_some]
  · cases returned

def inputFamilies (input : Input) : List Value :=
  [.array ((List.finRange SourceCount).map fun source =>
      PiCCSParity.fieldWordsValue (serializeCommitment (commitments input source))),
    .array ((List.finRange SourceCount).map fun source =>
      PiCCSParity.fieldWordsValue (serializePublicInput
        (logicalWidth := VerifierContext.candidateLogicalWidth)
        (publicFits := VerifierContext.candidatePublicFits) (publicInputs input source))),
    .array (input.evalK.toList.map fun family => PiCCSParity.extensionWordsValue family.toList),
    .array (input.evalA.toList.map fun family =>
      .array (family.toList.map fun matrix => PiCCSParity.extensionWordsValue matrix.toList))]

def inputValue (input : Input) (ccsResult : PiCCSInputCheck.Execution)
    (packageIdentity : VerifierContext.Digest4) : Value :=
  .array ([PiCCSParity.stateValue ccsResult.outgoing,
    PiRLCParity.pointValue ccsResult.point] ++ inputFamilies input ++
      [PiCCSParity.fieldWordsValue packageIdentity.toList])

def inputsNonzero (input : Input) : Bool :=
  (List.finRange SourceCount).all fun source =>
    PiRLCParity.commitmentHasNonzero (commitments input source) &&
      PiRLCParity.publicInputHasNonzero (publicInputs input source) &&
      PiRLCParity.evaluationHasNonzero (evaluations input source)

/-- The complete C prefix remains in fields 0..5. Fields 6 and 7 are the
actual R input and result. Parallel work only materializes independent
families; all prefix values remain in their original source order. -/
def checkValueIO (input : Input) (packageIdentity : VerifierContext.Digest4) : IO Value := do
  let ccsResult := PiCCSInputCheck.execute input
  let fields := PiCCSInputCheck.checkFields input ccsResult
  let rlcInput := inputValue input ccsResult packageIdentity
  if !ccsResult.accepted then
    return .array (fields ++ [rlcInput, .array [.atom 0]])
  match Transcript.PiRlcSampler.piRlcChallengesWithState ccsResult.outgoing SourceCount with
  | none => return .array (fields ++ [rlcInput, .array [.atom 0]])
  | some batch =>
      let commitmentTask ← IO.asTask (PiRLCParity.prepare fun _ =>
        commitmentPartials batch.challenges (commitments input))
      let publicTask ← IO.asTask (PiRLCParity.prepare fun _ =>
        publicInputPartials batch.challenges (publicInputs input))
      let padTask ← IO.asTask (PiRLCParity.prepare fun _ =>
        evaluationPartials batch.challenges fun source => (evaluations input source).pad)
      let matrixTasks ← (List.finRange productionShape.matrixCount).mapM fun matrix =>
        IO.asTask (PiRLCParity.prepare fun _ =>
          evaluationPartials batch.challenges fun source => (evaluations input source).matrix matrix)
      let commitmentValues ← PiRLCParity.prepared commitmentTask
      let publicValues ← PiRLCParity.prepared publicTask
      let padValues ← PiRLCParity.prepared padTask
      let matrixValues ← matrixTasks.mapM PiRLCParity.prepared
      match PiRLCParity.resultValueFromPartials ccsResult.point batch (inputsNonzero input)
          commitmentValues publicValues padValues matrixValues with
      | some result => return .array (fields ++ [rlcInput, result])
      | none => throw (IO.userError "incomplete actual PiRLC indexed trace")

end NightstreamFPrime.Export.Stage1.PiRLCInputCheck
