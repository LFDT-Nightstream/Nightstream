import NightstreamFPrime.Export.Codec
import NightstreamFPrime.Export.Stage1.PiCCSNonzero

/-!
Paper authority: SuperNeo v1.1, Section 7.3, steps 1--5.
Obligation: Emit the complete synthetic PiCCS input, proof, and verifier
result for Lean--Rust result comparison. This fixture does not establish
valid bounded openings. Lean uses the production transcript schedule and
fixed-width SumCheck checker.
-/

namespace NightstreamFPrime.Export.Stage1.PiCCSParity

open NightstreamFPrime.Export.Codec
open NightstreamFPrime.Export.Stage1.PiCCSNonzero
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

def boolValue (value : Bool) : Value :=
  .atom (if value then 1 else 0)

def fieldValue (value : F) : Value := .atom value.val

def extensionValue (value : K) : Value :=
  .array [fieldValue value.c0, fieldValue value.c1]

def fieldWordsValue (values : List F) : Value :=
  .array (values.map fieldValue)

def extensionWordsValue (values : List K) : Value :=
  .array (values.map extensionValue)

def fieldBlocksValue (blocks : List (List F)) : Value :=
  .array (blocks.map fieldWordsValue)

def stateValue (state : Transcript.State) : Value :=
  fieldWordsValue state

def roundMessagesValue (computed : Computed) : Value :=
  .array ((List.finRange productionShape.cubeVariables).map fun roundIndex =>
    .array ((List.finRange (9 + 1)).map fun coefficient =>
      extensionValue
        (computed.roundTrace.roundCoefficient roundIndex coefficient)))

def outputEval_KValue (computed : Computed := compute ()) : Value :=
  .array ((List.finRange productionShape.sourceCount).map fun source =>
    .array ((List.finRange productionShape.coefficientCount).map fun coefficient =>
      extensionValue (computed.output.padCoordinate source coefficient)))

def outputEval_AValue (computed : Computed := compute ()) : Value :=
  .array ((List.finRange productionShape.sourceCount).map fun source =>
    .array ((List.finRange productionShape.matrixCount).map fun matrix =>
      .array ((List.finRange productionShape.coefficientCount).map fun coefficient =>
        extensionValue
          (computed.output.matrixCoordinate source matrix coefficient))))

def outputCommitment :
    Fin (productionShape.freshCount + productionShape.runningCount) →
      PaperAlgebra.Commitment :=
  Fin.addCases (fun _ => freshCommitment) running.commitments

def outputPublicInput (vk : KeyDigest) :
    Fin (productionShape.freshCount + productionShape.runningCount) →
      PaperAlgebra.PublicInput
        (logicalWidth := fixtureLogicalWidth)
        (publicFits := fixturePublicFits) :=
  Fin.addCases (fresh vk).publicInputs running.publicInputs

def outputCommitmentsValue : Value :=
  .array ((List.finRange productionShape.sourceCount).map fun source =>
    fieldWordsValue (serializeCommitment (outputCommitment source)))

def outputPublicInputsValue (vk : KeyDigest := stateVerifierKey ()) : Value :=
  .array ((List.finRange productionShape.sourceCount).map fun source =>
    fieldWordsValue (serializePublicInput (outputPublicInput vk source)))

def outputPublicInputFromFresh (freshValue : FixtureFresh) :
    Fin (productionShape.freshCount + productionShape.runningCount) →
      FixturePublicInput :=
  Fin.addCases freshValue.publicInputs running.publicInputs

def outputPublicInputsValueFromFresh (freshValue : FixtureFresh) : Value :=
  .array ((List.finRange productionShape.sourceCount).map fun source =>
    fieldWordsValue
      (serializePublicInput (outputPublicInputFromFresh freshValue source)))

theorem outputPublicInputsValueFromFresh_eq
    (statement : FixtureStatement) :
    outputPublicInputsValueFromFresh statement.freshValue =
      outputPublicInputsValue statement.stateKey := by
  rw [statement.freshValue_eq]
  rfl

def roundStatesValue (computed : Computed) : Value :=
  .array (computed.roundTrace.states.map stateValue)

def terminalComponentsValue (computed : Computed) : Value :=
  .array [extensionValue computed.padTerminal,
    extensionValue computed.matrixTerminal,
    extensionValue computed.ccsTerminal,
    extensionValue computed.normTerminal,
    extensionValue computed.verifierTerminal,
    extensionValue computed.roundTrace.claim]

def assuranceValue (computed : Computed) : Value :=
  .array [boolValue freshCommitmentNonzero,
    boolValue computed.proofMessagesNonzero,
    boolValue (outputEval_KNonzero computed.output),
    boolValue (outputEval_ANonzero computed.output)]

/-- Caller-owned input and proof tuple.

Order: prior preimage, output preimage, prior public input, output digest,
verifier-context digest, fresh commitment, round messages, output `Eval_K`, output
`Eval_A`, complete digest-only transcript blocks, semantic verifier-input
blocks. The production verifier recomputes the key digest from the canonical
sealed package and fixed setup; the fixture does not duplicate that authority. -/
def inputValue (computed : Computed) : Value :=
  .array [fieldWordsValue computed.statement.preimageWords,
    fieldWordsValue computed.statement.preimageWords,
    fieldWordsValue (List.ofFn computed.statement.publicInput),
    fieldWordsValue computed.statement.digest,
    fieldWordsValue computed.statement.stateKey,
    fieldWordsValue (serializeCommitment freshCommitment),
    roundMessagesValue computed,
    outputEval_KValue computed,
    outputEval_AValue computed,
    fieldBlocksValue
      (ProductionKey.publicInputBlocks running computed.statement.freshValue),
    fieldBlocksValue (Transcript.verifierInputBlocks verifierInput)]

/-- Complete verifier result tuple.

Order: acceptance, alpha, gamma, pre-SumCheck state, every round challenge,
every intermediate round state, `r'`, initial claim, every post-round claim,
terminal components, all output commitments, all output public inputs, all
separate output `Eval_K`, all separate output `Eval_A`, outgoing state, and
nonzero assurance flags. -/
def resultValue (computed : Computed) : Value :=
  .array [boolValue computed.accepted,
    extensionWordsValue computed.preSumcheck.alpha.coordinates,
    extensionValue computed.preSumcheck.gamma,
    stateValue computed.preSumcheck.state,
    extensionWordsValue computed.verifierRoundResult.1,
    roundStatesValue computed,
    extensionWordsValue computed.verifierRoundPoint.coordinates,
    extensionValue computed.initialClaim,
    extensionWordsValue computed.roundTrace.claims,
    terminalComponentsValue computed,
    outputCommitmentsValue,
    outputPublicInputsValueFromFresh computed.statement.freshValue,
    outputEval_KValue computed,
    outputEval_AValue computed,
    stateValue computed.outgoingState,
    assuranceValue computed]

/-- Schema 8 binds the context through the separately loaded canonical
production package and removes the old fixture-owned authority duplicate. -/
private def parityValueFrom (computed : Computed) : Value :=
  .array [.atom 8, inputValue computed, resultValue computed]

def parityValue (_ : Unit)
    (vk : KeyDigest := stateVerifierKey ()) : Value :=
  parityValueFrom (compute () vk)

/-- Schedule independent fixture calculations before deterministic emission. -/
def parityValueIO (vk : KeyDigest := stateVerifierKey ()) : IO Value := do
  pure (parityValueFrom (← computeIO vk))

def render (_ : Unit)
    (vk : KeyDigest := stateVerifierKey ()) : String :=
  (parityValue () vk).render

end NightstreamFPrime.Export.Stage1.PiCCSParity
