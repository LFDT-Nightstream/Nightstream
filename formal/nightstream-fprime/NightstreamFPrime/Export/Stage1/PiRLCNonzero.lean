import NightstreamFPrime.Export.Stage1.PiCCSNonzero
import NightstreamFPrime.Lifecycle.PiRLC.v1_2.Semantics

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

/-- The total fixed-size production sampler result. -/
def sampled (_ : Unit) : Transcript.PiRlcSampler.Batch SourceCount :=
  Transcript.PiRlcSampler.piRlcChallengesWithState (initialState ()) SourceCount

def inputCommitment (source : Fin SourceCount) : PaperAlgebra.Commitment :=
  Fin.addCases (fun _ => freshCommitment) running.commitments (sourceIndex source)

/-- The downstream fixture still uses the fixed 270-coordinate candidate
carrier. This view evaluates the same key-bound PiCCS digest in that carrier;
the logical-width proof does not change any serialized coordinate. -/
def candidateFreshPublicInput :
    PublicInput (logicalWidth := PhaseReference.logicalWidth)
      (publicFits := PhaseReference.publicFits) :=
  fun column => encHash (publicFits := PhaseReference.publicFits)
    (stateDigest () (stateVerifierKey ())) column

def inputPublicInput (source : Fin SourceCount) :
    PublicInput (logicalWidth := PhaseReference.logicalWidth)
      (publicFits := PhaseReference.publicFits) :=
  Fin.addCases (fun _ => candidateFreshPublicInput)
    (fun runningSource column => runningPublicDigit runningSource.val column.val)
    (sourceIndex source)

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
    PublicInput (logicalWidth := PhaseReference.logicalWidth)
      (publicFits := PhaseReference.publicFits) :=
  let digest := stateDigest () (stateVerifierKey ())
  let inputs := fun source =>
    Fin.addCases
      (fun _ column =>
        encHash (publicFits := PhaseReference.publicFits)
          digest column)
      (fun runningSource column => runningPublicDigit runningSource.val column.val)
      (sourceIndex source)
  NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.PublicInput.combinePublicInputs
    challenges inputs

def combinedEvaluation (challenges : Fin SourceCount → RingF) : Evaluation :=
  PaperAlgebra.combineEvaluationFamily challenges inputEvaluation

def inputInstance
    (relation : ProductionKey.LogicalRelation
      PhaseReference.logicalWidth PhaseReference.publicFits)
    (source : Fin SourceCount) :
    NightstreamFPrime.Lifecycle.PiRLC.v1_2.InputBinding.InputInstance
      PhaseReference.logicalWidth
        PhaseReference.publicFits where
  constraintSystem :=
    NightstreamFPrime.Lifecycle.PiRLC.v1_2.InputBinding.relationSource relation
  commitment := inputCommitment source
  publicInput := inputPublicInput source
  point := point ()
  evaluations := #[inputEvaluation source]
  stage := .fresh

def attempt
    (relation : ProductionKey.LogicalRelation
      PhaseReference.logicalWidth PhaseReference.publicFits)
    (challenges : Fin SourceCount → RingF) :
    PiRLC.Attempt
      (PaperAlgebra.Structure PhaseReference.logicalWidth)
      (PaperAlgebra.PublicInput
        (logicalWidth := PhaseReference.logicalWidth)
        (publicFits := PhaseReference.publicFits))
      PaperAlgebra.Point PaperAlgebra.Evaluation PaperAlgebra.Commitment RingF
      productionGlobalParams
      Nifs.PaperProfile.arity where
  inputs := inputInstance relation
  challenges := challenges
  output := {
    constraintSystem :=
      NightstreamFPrime.Lifecycle.PiRLC.v1_2.InputBinding.relationSource relation
    commitment := combinedCommitment challenges
    publicInput := combinedPublicInput challenges
    point := point ()
    evaluations := #[combinedEvaluation challenges]
    stage := .combined }

theorem attempt_output_commitment
    (relation : ProductionKey.LogicalRelation
      PhaseReference.logicalWidth PhaseReference.publicFits)
    (challenges : Fin SourceCount → RingF) :
    (attempt relation challenges).output.commitment =
      combinedCommitment challenges := by
  rfl

theorem attempt_output_publicInput
    (relation : ProductionKey.LogicalRelation
      PhaseReference.logicalWidth PhaseReference.publicFits)
    (challenges : Fin SourceCount → RingF) :
    (attempt relation challenges).output.publicInput =
      combinedPublicInput challenges := by
  rfl

theorem attempt_output_evaluations
    (relation : ProductionKey.LogicalRelation
      PhaseReference.logicalWidth PhaseReference.publicFits)
    (challenges : Fin SourceCount → RingF) :
    (attempt relation challenges).output.evaluations =
      #[combinedEvaluation challenges] := by
  rfl

/-- The concrete sampler result plus the verifier-computed combined claim
satisfy the exact model-level PiRLC acceptance predicate for any final
relation and Ajtai key. -/
theorem accepted
    (relation : ProductionKey.LogicalRelation
      PhaseReference.logicalWidth PhaseReference.publicFits)
    (key : AjtaiKey
      (logicalWidth := PhaseReference.logicalWidth)
      (publicFits := PhaseReference.publicFits)) :
    PiRLC.Accepted (PaperAlgebra.piRlcAlgebra key)
      (attempt relation (sampled ()).challenges) := by
  let batch := sampled ()
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
  have member := Transcript.PiRlcSampler.piRlcChallenges_member (initialState ()) SourceCount source
  simpa [PaperAlgebra.piRlcAlgebra,
    NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.Challenge.challengeValid]
    using! member

end NightstreamFPrime.Export.Stage1.PiRLCNonzero
