import NightstreamFPrime.Export.Stage1.PiCCSNonzero
import NightstreamFPrime.Lifecycle.PiRLC.v1_1.Semantics

/-!
Owns one deterministic nonzero PiRLC value fixture that starts at the exact
PiCCS fixture output state.

The fixture uses the production 17-input sampler and the four public
combination operations from the authoritative Lean algebra. Its acceptance
theorem is parametric in the final logical relation and Ajtai key. Neither a
temporary relation nor a temporary key becomes fixture authority.
-/

namespace NightstreamFPrime.Export.Stage1.PiRLCNonzero

open NightstreamFPrime.Export.Stage1.PiCCSNonzero
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

abbrev SourceCount : Nat := Nifs.PaperProfile.arity.total

theorem sourceCount_eq : SourceCount = productionShape.sourceCount := by
  rfl

def sourceIndex (source : Fin SourceCount) :
    Fin productionShape.sourceCount :=
  Fin.cast sourceCount_eq source

def initialState (_ : Unit) : Transcript.State :=
  (PiCCSNonzero.compute ()).outgoingState

/-- The exact fixed-size sampler result. `none` is an explicit phase
rejection; no challenge fallback exists. -/
def sampled (_ : Unit) : Option (Transcript.PiRlcSampler.Batch SourceCount) :=
  Transcript.PiRlcSampler.piRlcChallengesWithState (initialState ()) SourceCount

def inputCommitment (source : Fin SourceCount) : PaperAlgebra.Commitment :=
  Fin.addCases fresh.commitments running.commitments (sourceIndex source)

def inputPublicInput (source : Fin SourceCount) :
    PublicInput (logicalWidth := VerifierContext.candidateLogicalWidth)
      (publicFits := VerifierContext.candidatePublicFits) :=
  Fin.addCases fresh.publicInputs running.publicInputs (sourceIndex source)

def inputEvaluation (source : Fin SourceCount) : Evaluation where
  pad := fun coefficient => output.padCoordinate (sourceIndex source) coefficient
  matrix := fun matrix coefficient =>
    output.matrixCoordinate (sourceIndex source) matrix coefficient

def point (_ : Unit) : Point :=
  (PiCCSNonzero.compute ()).verifierRoundPoint

def combinedCommitment (challenges : Fin SourceCount → RingF) :
    PaperAlgebra.Commitment :=
  NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.Commitment.combineCommitments
    challenges inputCommitment

def combinedPublicInput (challenges : Fin SourceCount → RingF) :
    PublicInput (logicalWidth := VerifierContext.candidateLogicalWidth)
      (publicFits := VerifierContext.candidatePublicFits) :=
  NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.PublicInput.combinePublicInputs
    challenges inputPublicInput

def combinedEvaluation (challenges : Fin SourceCount → RingF) : Evaluation :=
  PaperAlgebra.combineEvaluationFamily challenges inputEvaluation

def inputInstance
    (relation : ProductionKey.LogicalRelation
      VerifierContext.candidateLogicalWidth VerifierContext.candidatePublicFits)
    (source : Fin SourceCount) :
    NightstreamFPrime.Lifecycle.PiRLC.v1_1.InputBinding.InputInstance
      VerifierContext.candidateLogicalWidth
        VerifierContext.candidatePublicFits where
  constraintSystem :=
    NightstreamFPrime.Lifecycle.PiRLC.v1_1.InputBinding.relationSource relation
  commitment := inputCommitment source
  publicInput := inputPublicInput source
  point := point ()
  evaluations := #[inputEvaluation source]
  stage := .fresh

def attempt
    (relation : ProductionKey.LogicalRelation
      VerifierContext.candidateLogicalWidth VerifierContext.candidatePublicFits)
    (challenges : Fin SourceCount → RingF) :
    PiRLC.Attempt
      (PaperAlgebra.Structure VerifierContext.candidateLogicalWidth)
      (PaperAlgebra.PublicInput
        (logicalWidth := VerifierContext.candidateLogicalWidth)
        (publicFits := VerifierContext.candidatePublicFits))
      PaperAlgebra.Point PaperAlgebra.Evaluation PaperAlgebra.Commitment RingF
      productionGlobalParams
      Nifs.PaperProfile.arity where
  inputs := inputInstance relation
  challenges := challenges
  output := {
    constraintSystem :=
      NightstreamFPrime.Lifecycle.PiRLC.v1_1.InputBinding.relationSource relation
    commitment := combinedCommitment challenges
    publicInput := combinedPublicInput challenges
    point := point ()
    evaluations := #[combinedEvaluation challenges]
    stage := .combined }

theorem attempt_output_commitment
    (relation : ProductionKey.LogicalRelation
      VerifierContext.candidateLogicalWidth VerifierContext.candidatePublicFits)
    (challenges : Fin SourceCount → RingF) :
    (attempt relation challenges).output.commitment =
      combinedCommitment challenges := by
  rfl

theorem attempt_output_publicInput
    (relation : ProductionKey.LogicalRelation
      VerifierContext.candidateLogicalWidth VerifierContext.candidatePublicFits)
    (challenges : Fin SourceCount → RingF) :
    (attempt relation challenges).output.publicInput =
      combinedPublicInput challenges := by
  rfl

theorem attempt_output_evaluations
    (relation : ProductionKey.LogicalRelation
      VerifierContext.candidateLogicalWidth VerifierContext.candidatePublicFits)
    (challenges : Fin SourceCount → RingF) :
    (attempt relation challenges).output.evaluations =
      #[combinedEvaluation challenges] := by
  rfl

/-- The concrete sampler result plus the verifier-computed combined claim
satisfy the exact model-level PiRLC acceptance predicate for any final
relation and Ajtai key. -/
theorem accepted
    (relation : ProductionKey.LogicalRelation
      VerifierContext.candidateLogicalWidth VerifierContext.candidatePublicFits)
    (key : AjtaiKey
      (logicalWidth := VerifierContext.candidateLogicalWidth)
      (publicFits := VerifierContext.candidatePublicFits))
    (batch : Transcript.PiRlcSampler.Batch SourceCount)
    (success : sampled () = some batch) :
    PiRLC.Accepted (PaperAlgebra.piRlcAlgebra key)
      (attempt relation batch.challenges) := by
  refine {
    inputFresh := fun _ => rfl
    sameStructure := fun _ => rfl
    samePoint := fun _ => rfl
    outputCombined := rfl
    commitmentEquation := rfl
    publicInputEquation := rfl
    evaluationEquation := ?_
    challengesValid := ?_ }
  · exact (PaperAlgebra.combineEvaluations_singletons (by decide)
      batch.challenges inputEvaluation).symm
  intro source
  have member := Transcript.PiRlcSampler.piRlcChallenges_member success source
  simpa [PaperAlgebra.piRlcAlgebra,
    NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.Challenge.challengeValid]
    using member

end NightstreamFPrime.Export.Stage1.PiRLCNonzero
