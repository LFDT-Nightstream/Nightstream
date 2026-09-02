import NightstreamFPrime.Export.Codec
import NightstreamFPrime.Export.Stage1.PiCCSParity
import NightstreamFPrime.Export.Stage1.PiRLCNonzero
import NightstreamFPrime.Export.Stage1.PiRLCPartialTrace

/-!
Paper authority: SuperNeo v1.1, Section 7.4, steps 1--3.
Obligation: emit one complete nonzero PiRLC input and verifier-computed result
for Lean--Rust conformance. The result ends at the paper PiRLC output. It does
not include Rust's separate projection-proof transcript plumbing.
-/

namespace NightstreamFPrime.Export.Stage1.PiRLCParity

open NightstreamFPrime.Export.Codec
open NightstreamFPrime.Export.Stage1
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open PiRLCNonzero
open PiRLCPartialTrace

def challengeValue (challenge : RingF) : Value :=
  PiCCSParity.fieldWordsValue (List.ofFn challenge)

def challengesValue (challenges : Fin SourceCount → RingF) : Value :=
  .array ((List.finRange SourceCount).map fun source =>
    challengeValue (challenges source))

/-- A successful typed sampler batch carries a theorem that every indexed
challenge is in the production strong set. Rust checks each result again from
the serialized coefficients. -/
def membershipValue : Value :=
  .array (List.replicate SourceCount (PiCCSParity.boolValue true))

def combinedCommitmentValue (challenges : Fin SourceCount → RingF) : Value :=
  PiCCSParity.fieldWordsValue
    (serializeCommitment (combinedCommitment challenges))

def combinedPublicInputValue (challenges : Fin SourceCount → RingF) : Value :=
  PiCCSParity.fieldWordsValue
    (serializePublicInput (combinedPublicInput challenges))

def combinedEvalKValue (challenges : Fin SourceCount → RingF) : Value :=
  PiCCSParity.extensionWordsValue
    ((List.finRange productionShape.coefficientCount).map fun coefficient =>
      (combinedEvaluation challenges).pad coefficient)

def combinedEvalAValue (challenges : Fin SourceCount → RingF) : Value :=
  .array ((List.finRange productionShape.matrixCount).map fun matrix =>
    PiCCSParity.extensionWordsValue
      ((List.finRange productionShape.coefficientCount).map fun coefficient =>
        (combinedEvaluation challenges).matrix matrix coefficient))

def pointValue (value : Point) : Value :=
  PiCCSParity.extensionWordsValue value.coordinates

def inputFamilyValues : List Value :=
  [PiCCSParity.outputCommitmentsValue,
    PiCCSParity.outputPublicInputsValue,
    PiCCSParity.outputEval_KValue,
    PiCCSParity.outputEval_AValue]

def inputValueWithFamilies (computed : PiCCSNonzero.Computed)
    (families : List Value) : Value :=
  .array ([PiCCSParity.stateValue computed.outgoingState,
    pointValue computed.verifierRoundPoint] ++ families ++
      [PiCCSParity.fieldWordsValue
        VerifierContext.productionPackageIdentityWords])

def inputValue (computed : PiCCSNonzero.Computed) : Value :=
  inputValueWithFamilies computed inputFamilyValues

def ringHasNonzero (value : RingF) : Bool :=
  (List.finRange ringDegree).any fun coefficient =>
    decide (value coefficient ≠ 0)

def commitmentHasNonzero (value : PaperAlgebra.Commitment) : Bool :=
  (List.finRange productionProfile.commitmentWidth).any fun row =>
    ringHasNonzero (value row)

def publicInputHasNonzero
    (value : PublicInput
      (logicalWidth := VerifierContext.candidateLogicalWidth)
      (publicFits := VerifierContext.candidatePublicFits)) : Bool :=
  (List.finRange
    (FullShape VerifierContext.candidateLogicalWidth
      VerifierContext.candidatePublicFits).publicWidth).any
    fun column => decide (value column ≠ 0)

def evaluationHasNonzero (value : Evaluation) : Bool :=
  ((List.finRange productionShape.coefficientCount).any fun coefficient =>
      decide (value.pad coefficient ≠ K.zero)) ||
    ((List.finRange productionShape.matrixCount).any fun matrix =>
      (List.finRange productionShape.coefficientCount).any fun coefficient =>
        decide (value.matrix matrix coefficient ≠ K.zero))

def inputsNonzero : Bool :=
  (List.finRange SourceCount).all fun source =>
    commitmentHasNonzero (inputCommitment source) &&
      publicInputHasNonzero (inputPublicInput source) &&
      evaluationHasNonzero (inputEvaluation source)

def assuranceValue (challenges : Fin SourceCount → RingF) : Value :=
  .array [PiCCSParity.boolValue inputsNonzero,
    PiCCSParity.boolValue (commitmentHasNonzero
      (combinedCommitment challenges)),
    PiCCSParity.boolValue (publicInputHasNonzero
      (combinedPublicInput challenges)),
    PiCCSParity.boolValue (evaluationHasNonzero
      (combinedEvaluation challenges))]

def materializedCommitmentValue (value : MaterializedCommitment) : Value :=
  PiCCSParity.fieldWordsValue
    (serializeCommitment value.toCommitment)

def materializedPublicInputValue (value : MaterializedPublicInput) : Value :=
  PiCCSParity.fieldWordsValue
    (serializePublicInput value.toPublicInput)

def materializedRingKValue (value : MaterializedRingK) : Value :=
  PiCCSParity.extensionWordsValue value.toList

def materializedEvalAValue (values : List MaterializedRingK) : Value :=
  .array (values.map materializedRingKValue)

def materializedCommitmentHasNonzero
    (value : MaterializedCommitment) : Bool :=
  value.toList.any fun row =>
    row.toList.any fun coefficient => decide (coefficient ≠ 0)

def materializedPublicInputHasNonzero
    (value : MaterializedPublicInput) : Bool :=
  value.toList.any fun column => decide (column ≠ 0)

def materializedRingKHasNonzero (value : MaterializedRingK) : Bool :=
  value.toList.any fun coefficient => decide (coefficient ≠ K.zero)

def materializedEvaluationHasNonzero (evalK : MaterializedRingK)
    (evalA : List MaterializedRingK) : Bool :=
  materializedRingKHasNonzero evalK ||
    evalA.any materializedRingKHasNonzero

def assemblePartialClaims :
    List MaterializedCommitment →
      List MaterializedPublicInput →
      List MaterializedRingK →
      List (List MaterializedRingK) → Option (List Value)
  | [], [], [], [] => some []
  | commitment :: commitments, publicInput :: publicInputs,
      evalK :: evalKs, evalA :: evalAs =>
      if evalA.length = productionShape.matrixCount then
        return .array [materializedCommitmentValue commitment,
          materializedPublicInputValue publicInput,
          materializedRingKValue evalK,
          materializedEvalAValue evalA] ::
            (← assemblePartialClaims commitments publicInputs evalKs evalAs)
      else none
  | _, _, _, _ => none

def resultValueFromPartials (computed : PiCCSNonzero.Computed)
    (batch : Transcript.PiRlcSampler.Batch SourceCount)
    (inputsAreNonzero : Bool)
    (commitments : List MaterializedCommitment)
    (publicInputs : List MaterializedPublicInput)
    (evalKs : List MaterializedRingK)
    (evalAsByMatrix : List (List MaterializedRingK)) : Option Value := do
  if evalAsByMatrix.length ≠ productionShape.matrixCount then
    none
  let evalAsBySource := evalAsByMatrix.transpose
  let partialClaims ←
    assemblePartialClaims commitments publicInputs evalKs evalAsBySource
  if partialClaims.length ≠ SourceCount then
    none
  let finalCommitment ← commitments.getLast?
  let finalPublicInput ← publicInputs.getLast?
  let finalEvalK ← evalKs.getLast?
  let finalEvalA ← evalAsBySource.getLast?
  if finalEvalA.length ≠ productionShape.matrixCount then
    none
  pure <| .array [PiCCSParity.boolValue true,
    challengesValue batch.challenges,
    membershipValue,
    materializedCommitmentValue finalCommitment,
    materializedPublicInputValue finalPublicInput,
    pointValue computed.verifierRoundPoint,
    materializedRingKValue finalEvalK,
    materializedEvalAValue finalEvalA,
    .array partialClaims,
    PiCCSParity.stateValue batch.finalState,
    .array [PiCCSParity.boolValue inputsAreNonzero,
      PiCCSParity.boolValue
        (materializedCommitmentHasNonzero finalCommitment),
      PiCCSParity.boolValue
        (materializedPublicInputHasNonzero finalPublicInput),
      PiCCSParity.boolValue
        (materializedEvaluationHasNonzero finalEvalK finalEvalA)]]

def resultValue (computed : PiCCSNonzero.Computed) : Value :=
  match Transcript.PiRlcSampler.piRlcChallengesWithState
      computed.outgoingState SourceCount with
  | none => .array [PiCCSParity.boolValue false]
  | some batch =>
      (resultValueFromPartials computed batch inputsNonzero
        (commitmentPartials batch.challenges)
        (publicInputPartials batch.challenges)
        (evalKPartials batch.challenges)
        ((List.finRange productionShape.matrixCount).map fun matrix =>
          evalAPartials batch.challenges matrix)).getD
            (.array [PiCCSParity.boolValue false])

abbrev PreparedTask (Alpha : Type) := Task (Except IO.Error Alpha)

def prepare {Alpha : Type} (build : Unit → Alpha) : IO Alpha := do
  pure (build ())

def prepared {Alpha : Type} (task : PreparedTask Alpha) : IO Alpha :=
  match task.get with
  | .ok value => pure value
  | .error error => throw error

def parityValueIO : IO Value := do
  let computedTask ← IO.asTask (prio := Task.Priority.dedicated)
    PiCCSNonzero.computeIO
  let inputFamiliesTask ← IO.asTask (prepare fun _ => inputFamilyValues)
  let inputsNonzeroTask ← IO.asTask (prepare fun _ => inputsNonzero)
  let computed ← prepared computedTask
  let inputFamilies ← prepared inputFamiliesTask
  let input := inputValueWithFamilies computed inputFamilies
  match Transcript.PiRlcSampler.piRlcChallengesWithState
      computed.outgoingState SourceCount with
  | none =>
      pure <| .array [.atom 3, input,
        .array [PiCCSParity.boolValue false]]
  | some batch =>
      let commitmentTask ← IO.asTask (prio := Task.Priority.dedicated)
        (prepare fun _ => commitmentPartials batch.challenges)
      let publicInputTask ← IO.asTask (prio := Task.Priority.dedicated)
        (prepare fun _ => publicInputPartials batch.challenges)
      let evalKTask ← IO.asTask (prio := Task.Priority.dedicated)
        (prepare fun _ => evalKPartials batch.challenges)
      let evalATasks ←
        (List.finRange productionShape.matrixCount).mapM fun matrix =>
          IO.asTask (prio := Task.Priority.dedicated) (prepare fun _ =>
            evalAPartials batch.challenges matrix)
      let commitments ← prepared commitmentTask
      let publicInputs ← prepared publicInputTask
      let evalKs ← prepared evalKTask
      let evalAsByMatrix ← evalATasks.mapM prepared
      let inputsAreNonzero ← prepared inputsNonzeroTask
      match resultValueFromPartials computed batch inputsAreNonzero
          commitments publicInputs evalKs evalAsByMatrix with
      | some result =>
          pure <| Value.array [Value.atom 3, input, result]
      | none =>
          throw (IO.userError "incomplete PiRLC indexed partial grid")

/-- Schema 3 adds the verifier-owned production package identity. -/
def parityValue (_ : Unit) : Value :=
  let computed := PiCCSNonzero.compute ()
  .array [.atom 3, inputValue computed, resultValue computed]

def render (_ : Unit) : String := (parityValue ()).render

end NightstreamFPrime.Export.Stage1.PiRLCParity
