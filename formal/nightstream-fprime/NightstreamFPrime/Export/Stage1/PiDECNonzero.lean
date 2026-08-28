import NightstreamFPrime.Export.Stage1.PiRLCNonzero
import NightstreamFPrime.Export.Stage1.PiRLCPartialTrace
import NightstreamFPrime.Spec.Folding.PiDEC.PaperVerifier

/-!
Owns one deterministic nonzero PiDEC v1.1 conformance fixture. It consumes
the exact accepted PiRLC output, computes the verifier-owned signed public
digits, and solves child zero after choosing nonzero prover messages for
children one through fifteen. No transcript state is changed by PiDEC.
-/

namespace NightstreamFPrime.Export.Stage1.PiDECNonzero

open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Phi81Relation.PiDECAlgebra

abbrev Batch := Transcript.PiRlcSampler.Batch PiRLCNonzero.SourceCount

structure Fixture where
  batch : Batch
  point : PaperAlgebra.Point
  commitment : PiRLCPartialTrace.MaterializedCommitment
  publicInput : PiRLCPartialTrace.MaterializedPublicInput
  evalK : PiRLCPartialTrace.MaterializedRingK
  evalA : PiRLCPartialTrace.FixedArray PiRLCPartialTrace.MaterializedRingK
    productionShape.matrixCount

/-- A computable relation value used only to type the fixture claims. PiDEC
does not inspect its matrices; final integration replaces this value with the
package-selected logical relation. -/
def fixtureRelation : ProductionKey.LogicalRelation
    VerifierContext.candidateLogicalWidth VerifierContext.candidatePublicFits where
  matrices := fun _ _ _ => 0
  cubeFits := by
    norm_num [VerifierContext.candidateLogicalWidth,
      Phi81CarrierLayout.carrierWidth, Phi81ColumnLayout.blockCount,
      cubeVariables, ringDegree]

/-- A deterministic key value used only to instantiate the concrete PiDEC
algebra. The public PiDEC equations do not inspect key entries. -/
def fixtureAjtaiKey : AjtaiKey
    (logicalWidth := VerifierContext.candidateLogicalWidth)
    (publicFits := VerifierContext.candidatePublicFits) :=
  fun _ _ _ => 0

def makeFixture (computed : PiCCSNonzero.Computed) (batch : Batch) : Fixture :=
  let evaluation := PiRLCNonzero.combinedEvaluation batch.challenges
  {
    batch := batch
    point := computed.verifierRoundPoint
    commitment := PiRLCPartialTrace.MaterializedCommitment.ofCommitment
      (PiRLCNonzero.combinedCommitment batch.challenges)
    publicInput := PiRLCPartialTrace.MaterializedPublicInput.ofPublicInput
      (PiRLCNonzero.combinedPublicInput batch.challenges)
    evalK := PiRLCPartialTrace.MaterializedRingK.ofRing evaluation.pad
    evalA := PiRLCPartialTrace.FixedArray.ofFn fun matrix =>
      PiRLCPartialTrace.MaterializedRingK.ofRing (evaluation.matrix matrix) }

def Fixture.evaluation (fixture : Fixture) : PaperAlgebra.Evaluation where
  pad := fixture.evalK.toRing
  matrix := fun matrix => (fixture.evalA.get matrix).toRing

def parent (fixture : Fixture) : CE.Instance
    (PaperAlgebra.Structure VerifierContext.candidateLogicalWidth)
    (PaperAlgebra.PublicInput
      (logicalWidth := VerifierContext.candidateLogicalWidth)
      (publicFits := VerifierContext.candidatePublicFits))
    PaperAlgebra.Point PaperAlgebra.Evaluation PaperAlgebra.Commitment where
  constraintSystem :=
    NightstreamFPrime.Lifecycle.PiRLC.v1_1.InputBinding.relationSource
      fixtureRelation
  commitment := fixture.commitment.toCommitment
  publicInput := fixture.publicInput.toPublicInput
  point := fixture.point
  evaluations := #[fixture.evaluation]
  stage := .combined

def rawChildCommitment (child : Radix.ChildIndex) : PaperAlgebra.Commitment :=
  if child.val = 0 then fun _ _ => 0
  else fun row lane => PiCCSNonzero.field
    (100_000_000 + child.val * 100_000 + row.val * ringDegree + lane.val)

def rawChildEvaluation (child : Radix.ChildIndex) : PaperAlgebra.Evaluation :=
  if child.val = 0 then evaluationZero
  else {
    pad := fun coefficient => PiCCSNonzero.extension
      (110_000_000 + child.val * 100_000 + coefficient.val)
      (120_000_000 + child.val * 100_000 + coefficient.val)
    matrix := fun matrix coefficient => PiCCSNonzero.extension
      (130_000_000 + child.val * 1_000_000 +
        matrix.val * 10_000 + coefficient.val)
      (150_000_000 + child.val * 1_000_000 +
        matrix.val * 10_000 + coefficient.val) }

def commitmentTail : PaperAlgebra.Commitment :=
  Commitment.recomposeCommitment rawChildCommitment

def evaluationTail : PaperAlgebra.Evaluation :=
  recomposeEvaluationFamily rawChildEvaluation

def childCommitment (fixture : Fixture) (child : Radix.ChildIndex) :
    PaperAlgebra.Commitment :=
  if child.val = 0 then fun row lane =>
    (parent fixture).commitment row lane - commitmentTail row lane
  else rawChildCommitment child

def childEvaluation (fixture : Fixture) (child : Radix.ChildIndex) :
    PaperAlgebra.Evaluation :=
  if child.val = 0 then {
    pad := fun coefficient =>
      K.sub (((parent fixture).evaluations.getD 0 evaluationZero).pad coefficient)
        (evaluationTail.pad coefficient)
    matrix := fun matrix coefficient =>
      K.sub (((parent fixture).evaluations.getD 0 evaluationZero).matrix
          matrix coefficient)
        (evaluationTail.matrix matrix coefficient) }
  else rawChildEvaluation child

def message (fixture : Fixture) (child : Radix.ChildIndex) :
    PiDEC.PaperVerifier.ChildMessage PaperAlgebra.Evaluation
      PaperAlgebra.Commitment where
  commitment := childCommitment fixture child
  evaluations := #[childEvaluation fixture child]

def attempt (fixture : Fixture) : PiDEC.PaperVerifier.Attempt
    (PaperAlgebra.Structure VerifierContext.candidateLogicalWidth)
    (PaperAlgebra.PublicInput
      (logicalWidth := VerifierContext.candidateLogicalWidth)
      (publicFits := VerifierContext.candidatePublicFits))
    PaperAlgebra.Point PaperAlgebra.Evaluation PaperAlgebra.Commitment
      productionGlobalParams where
  parent := parent fixture
  messages := message fixture

def childPublicInput (fixture : Fixture) (child : Radix.ChildIndex) :
    PaperAlgebra.PublicInput
      (logicalWidth := VerifierContext.candidateLogicalWidth)
      (publicFits := VerifierContext.candidatePublicFits) :=
  PublicInput.splitPublicInput (parent fixture).publicInput child

def children (fixture : Fixture) :=
  PiDEC.PaperVerifier.children (publicInputSplit fixtureAjtaiKey)
    (attempt fixture)

def recomposedCommitment (fixture : Fixture) : PaperAlgebra.Commitment :=
  Commitment.recomposeCommitment (childCommitment fixture)

def recomposedPublicInput (fixture : Fixture) :
    PaperAlgebra.PublicInput
      (logicalWidth := VerifierContext.candidateLogicalWidth)
      (publicFits := VerifierContext.candidatePublicFits) :=
  PublicInput.recomposePublicInput (childPublicInput fixture)

def recomposedEvaluation (fixture : Fixture) : PaperAlgebra.Evaluation :=
  recomposeEvaluationFamily (childEvaluation fixture)

def parentBounded (fixture : Fixture) : Bool :=
  let split := publicInputSplit fixtureAjtaiKey
  letI := split.parentBounded_decidable (parent fixture).publicInput
  decide (split.parentBounded (parent fixture).publicInput)

def accepted (fixture : Fixture) : Bool :=
  letI := piDecDecision fixtureAjtaiKey (attempt fixture)
  decide (PiDEC.PaperVerifier.Accepted (piDecAlgebra fixtureAjtaiKey)
    (publicInputSplit fixtureAjtaiKey) (evaluationArity fixtureAjtaiKey)
    (attempt fixture))

def unboundedPublicInput (fixture : Fixture) :
    PaperAlgebra.PublicInput
      (logicalWidth := VerifierContext.candidateLogicalWidth)
      (publicFits := VerifierContext.candidatePublicFits) :=
  fun column => if column.val = 0 then Radix.fieldOfNat Radix.combinedBound
    else (parent fixture).publicInput column

def unboundedAttempt (fixture : Fixture) : PiDEC.PaperVerifier.Attempt
    (PaperAlgebra.Structure VerifierContext.candidateLogicalWidth)
    (PaperAlgebra.PublicInput
      (logicalWidth := VerifierContext.candidateLogicalWidth)
      (publicFits := VerifierContext.candidatePublicFits))
    PaperAlgebra.Point PaperAlgebra.Evaluation PaperAlgebra.Commitment
      productionGlobalParams :=
  { attempt fixture with
    parent := { parent fixture with
      publicInput := unboundedPublicInput fixture } }

def unboundedRejected (fixture : Fixture) : Bool :=
  letI := piDecDecision fixtureAjtaiKey (unboundedAttempt fixture)
  !decide (PiDEC.PaperVerifier.Accepted (piDecAlgebra fixtureAjtaiKey)
    (publicInputSplit fixtureAjtaiKey) (evaluationArity fixtureAjtaiKey)
    (unboundedAttempt fixture))

def digitInRange (fixture : Fixture) (child : Radix.ChildIndex)
    (coordinate : Fin 270) : Bool :=
  decide (centeredMagnitude (childPublicInput fixture child coordinate) < 2)

def outgoingState (fixture : Fixture) : Transcript.State :=
  fixture.batch.finalState

def allChildrenNonzero (fixture : Fixture) : Bool :=
  (List.finRange productionGlobalParams.k).all fun child =>
    (List.finRange productionProfile.commitmentWidth).any fun row =>
      (List.finRange ringDegree).any fun lane =>
        decide (childCommitment fixture child row lane != 0)

end NightstreamFPrime.Export.Stage1.PiDECNonzero
