import NightstreamFPrime.Export.Codec
import NightstreamFPrime.Export.Stage1.PiCCSParity
import NightstreamFPrime.Export.Stage1.PiDECNonzero
import NightstreamFPrime.Export.Stage1.PiRLCParity

/-!
Owns the complete deterministic PiDEC v1.1 Lean parity artifact. The input
tuple preserves the four physical PiDEC caller-input segments. The result
contains every verifier-computed digit, range result, recomposition family,
child claim, the unchanged transcript state, and the transition-ready output
state preimage and digest.
-/

namespace NightstreamFPrime.Export.Stage1.PiDECParity

open NightstreamFPrime.Export.Codec
open NightstreamFPrime.Export.Stage1
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Phi81Relation.PiDECAlgebra

def commitmentValue (value : PaperAlgebra.Commitment) : Value :=
  PiCCSParity.fieldWordsValue (serializeCommitment value)

def publicInputValue
    (value : PaperAlgebra.PublicInput
      (logicalWidth := VerifierContext.candidateLogicalWidth)
      (publicFits := VerifierContext.candidatePublicFits)) : Value :=
  PiCCSParity.fieldWordsValue (serializePublicInput value)

def evalKValue (value : PaperAlgebra.Evaluation) : Value :=
  PiCCSParity.extensionWordsValue
    ((List.finRange productionShape.coefficientCount).map value.pad)

def evalAValue (value : PaperAlgebra.Evaluation) : Value :=
  .array ((List.finRange productionShape.matrixCount).map fun matrix =>
    PiCCSParity.extensionWordsValue
      ((List.finRange productionShape.coefficientCount).map
        (value.matrix matrix)))

def pointValue (value : PaperAlgebra.Point) : Value :=
  PiCCSParity.extensionWordsValue value.coordinates

def parentEvaluation (fixture : PiDECNonzero.Fixture) :
    PaperAlgebra.Evaluation :=
  (PiDECNonzero.parent fixture).evaluations.getD 0 evaluationZero

def parentValue (fixture : PiDECNonzero.Fixture) : Value :=
  let parent := PiDECNonzero.parent fixture
  .array [commitmentValue parent.commitment,
    publicInputValue parent.publicInput,
    pointValue parent.point,
    evalKValue (parentEvaluation fixture),
    evalAValue (parentEvaluation fixture),
    .atom 1]

def messageCommitmentsValue (fixture : PiDECNonzero.Fixture) : Value :=
  .array ((List.finRange productionGlobalParams.k).map fun child =>
    commitmentValue (PiDECNonzero.childCommitment fixture child))

def messageEvalKValue (fixture : PiDECNonzero.Fixture) : Value :=
  .array ((List.finRange productionGlobalParams.k).map fun child =>
    evalKValue (PiDECNonzero.childEvaluation fixture child))

def messageEvalAValue (fixture : PiDECNonzero.Fixture) : Value :=
  .array ((List.finRange productionGlobalParams.k).map fun child =>
    evalAValue (PiDECNonzero.childEvaluation fixture child))

def childPublicInputsValue (fixture : PiDECNonzero.Fixture) : Value :=
  .array ((List.finRange productionGlobalParams.k).map fun child =>
    publicInputValue (PiDECNonzero.childPublicInput fixture child))

/-- Input order after the PiRLC parent: 16 commitments, 16 `Eval_K`
families, 16 separate 14-matrix `Eval_A` families, then 16 public digit
vectors. This is the exact `PiDECInputs` physical segment order. -/
def inputValue (fixture : PiDECNonzero.Fixture) : Value :=
  .array [parentValue fixture,
    messageCommitmentsValue fixture,
    messageEvalKValue fixture,
    messageEvalAValue fixture,
    childPublicInputsValue fixture,
    PiCCSParity.stateValue fixture.batch.finalState,
    PiCCSParity.fieldWordsValue VerifierContext.productionPackageIdentityWords]

def parentBoundResultsValue (fixture : PiDECNonzero.Fixture) : Value :=
  .array ((List.finRange 270).map fun coordinate =>
    PiCCSParity.boolValue
      (decide (centeredMagnitude
        ((PiDECNonzero.parent fixture).publicInput coordinate) <
          Radix.combinedBound)))

def digitRangeResultsValue (fixture : PiDECNonzero.Fixture) : Value :=
  .array ((List.finRange productionGlobalParams.k).map fun child =>
    .array ((List.finRange 270).map fun coordinate =>
      PiCCSParity.boolValue
        (PiDECNonzero.digitInRange fixture child coordinate)))

def commitmentEquation (fixture : PiDECNonzero.Fixture) : Bool :=
  decide (serializeCommitment (PiDECNonzero.recomposedCommitment fixture) =
    serializeCommitment (PiDECNonzero.parent fixture).commitment)

def publicInputEquation (fixture : PiDECNonzero.Fixture) : Bool :=
  decide (serializePublicInput (PiDECNonzero.recomposedPublicInput fixture) =
    serializePublicInput (PiDECNonzero.parent fixture).publicInput)

def evalKEquation (fixture : PiDECNonzero.Fixture) : Bool :=
  decide (((List.finRange productionShape.coefficientCount).map
      (PiDECNonzero.recomposedEvaluation fixture).pad) =
    (List.finRange productionShape.coefficientCount).map
      (parentEvaluation fixture).pad)

def evalAEquation (fixture : PiDECNonzero.Fixture) : Bool :=
  decide (((List.finRange productionShape.matrixCount).map fun matrix =>
      (List.finRange productionShape.coefficientCount).map
        ((PiDECNonzero.recomposedEvaluation fixture).matrix matrix)) =
    (List.finRange productionShape.matrixCount).map fun matrix =>
      (List.finRange productionShape.coefficientCount).map
        ((parentEvaluation fixture).matrix matrix))

def childValue (fixture : PiDECNonzero.Fixture)
    (child : Radix.ChildIndex) : Value :=
  let value := PiDECNonzero.children fixture child
  let evaluation := value.evaluations.getD 0 evaluationZero
  .array [commitmentValue value.commitment,
    publicInputValue value.publicInput,
    pointValue value.point,
    evalKValue evaluation,
    evalAValue evaluation,
    .atom 0]

def childrenValue (fixture : PiDECNonzero.Fixture) : Value :=
  .array ((List.finRange productionGlobalParams.k).map
    (childValue fixture))

def transitionRunning (fixture : PiDECNonzero.Fixture) : Running
    (logicalWidth := VerifierContext.candidateLogicalWidth)
    (publicFits := VerifierContext.candidatePublicFits) where
  point := (PiDECNonzero.parent fixture).point
  commitments := PiDECNonzero.childCommitment fixture
  publicInputs := PiDECNonzero.childPublicInput fixture
  evaluations := PiDECNonzero.childEvaluation fixture

def transitionOutputPreimage (fixture : PiDECNonzero.Fixture) : HashPreimage
    (logicalWidth := VerifierContext.candidateLogicalWidth)
    (publicFits := VerifierContext.candidatePublicFits) where
  verifierKeys := fun _ => PiCCSNonzero.stateVerifierKey ()
  iteration := 7
  z0 := PiCCSNonzero.stateZ0
  current := PiCCSNonzero.stateCurrent
  running := fun _ => transitionRunning fixture
  pc := 1

def transitionOutputPreimageWords (fixture : PiDECNonzero.Fixture) : List F :=
  serializePreimage (publicFits := VerifierContext.candidatePublicFits)
    (transitionOutputPreimage fixture)

def transitionOutputDigest (fixture : PiDECNonzero.Fixture) : Digest :=
  stateHash (publicFits := VerifierContext.candidatePublicFits)
    (transitionOutputPreimage fixture)

def allMessageEvaluationsNonzero (fixture : PiDECNonzero.Fixture) : Bool :=
  (List.finRange productionGlobalParams.k).all fun child =>
    PiRLCParity.evaluationHasNonzero
      (PiDECNonzero.childEvaluation fixture child)

def digitsHaveNonzero (fixture : PiDECNonzero.Fixture) : Bool :=
  (List.finRange productionGlobalParams.k).any fun child =>
    PiRLCParity.publicInputHasNonzero
      (PiDECNonzero.childPublicInput fixture child)

def assuranceValue (fixture : PiDECNonzero.Fixture) : Value :=
  .array [PiCCSParity.boolValue
      (PiRLCParity.commitmentHasNonzero
        (PiDECNonzero.parent fixture).commitment),
    PiCCSParity.boolValue
      (PiRLCParity.publicInputHasNonzero
        (PiDECNonzero.parent fixture).publicInput),
    PiCCSParity.boolValue
      (PiRLCParity.evaluationHasNonzero (parentEvaluation fixture)),
    PiCCSParity.boolValue (PiDECNonzero.allChildrenNonzero fixture),
    PiCCSParity.boolValue (allMessageEvaluationsNonzero fixture),
    PiCCSParity.boolValue (digitsHaveNonzero fixture)]

def resultValue (fixture : PiDECNonzero.Fixture) : Value :=
  .array [PiCCSParity.boolValue (PiDECNonzero.accepted fixture),
    PiCCSParity.boolValue (PiDECNonzero.parentBounded fixture),
    childPublicInputsValue fixture,
    parentBoundResultsValue fixture,
    digitRangeResultsValue fixture,
    commitmentValue (PiDECNonzero.recomposedCommitment fixture),
    PiCCSParity.boolValue (commitmentEquation fixture),
    publicInputValue (PiDECNonzero.recomposedPublicInput fixture),
    PiCCSParity.boolValue (publicInputEquation fixture),
    evalKValue (PiDECNonzero.recomposedEvaluation fixture),
    PiCCSParity.boolValue (evalKEquation fixture),
    evalAValue (PiDECNonzero.recomposedEvaluation fixture),
    PiCCSParity.boolValue (evalAEquation fixture),
    childrenValue fixture,
    PiCCSParity.stateValue (PiDECNonzero.outgoingState fixture),
    PiCCSParity.boolValue (PiDECNonzero.unboundedRejected fixture),
    assuranceValue fixture,
    PiCCSParity.fieldWordsValue (transitionOutputPreimageWords fixture),
    PiCCSParity.fieldWordsValue (transitionOutputDigest fixture)]

def rejectedValue : Value :=
  .array [PiCCSParity.boolValue false]

def parityValueForFixture (fixture : PiDECNonzero.Fixture) : Value :=
  .array [.atom 2, inputValue fixture, resultValue fixture]

def parityValue (_ : Unit) : Value :=
  let computed := PiCCSNonzero.compute ()
  match Transcript.PiRlcSampler.piRlcChallengesWithState
      computed.outgoingState PiRLCNonzero.SourceCount with
  | some batch =>
      parityValueForFixture (PiDECNonzero.makeFixture computed batch)
  | none => .array [.atom 2, .array [], rejectedValue]

def parityValueIO : IO Value := do
  let computed ← PiCCSNonzero.computeIO
  match Transcript.PiRlcSampler.piRlcChallengesWithState
      computed.outgoingState PiRLCNonzero.SourceCount with
  | some batch =>
      pure (parityValueForFixture (PiDECNonzero.makeFixture computed batch))
  | none => throw (IO.userError "PiRLC sampler shortfall before PiDEC fixture")

end NightstreamFPrime.Export.Stage1.PiDECParity
