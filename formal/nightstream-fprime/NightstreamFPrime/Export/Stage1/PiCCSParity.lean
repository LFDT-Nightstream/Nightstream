import NightstreamFPrime.Export.Codec
import NightstreamFPrime.Export.Stage1.PiCCSNonzero

/-!
Paper authority: SuperNeo v1.1, Section 7.3, steps 1--5.
Obligation: Emit one complete valid nonzero PiCCS input, proof, and verifier
result for Lean--Rust conformance. Lean computes the result through the exact
production transcript schedule and fixed-width SumCheck checker.
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

def outputEval_KValue : Value :=
  .array ((List.finRange productionShape.sourceCount).map fun source =>
    .array ((List.finRange productionShape.coefficientCount).map fun coefficient =>
      extensionValue (output.padCoordinate source coefficient)))

def outputEval_AValue : Value :=
  .array ((List.finRange productionShape.sourceCount).map fun source =>
    .array ((List.finRange productionShape.matrixCount).map fun matrix =>
      .array ((List.finRange productionShape.coefficientCount).map fun coefficient =>
        extensionValue (output.matrixCoordinate source matrix coefficient))))

def outputCommitment :
    Fin (productionShape.freshCount + productionShape.runningCount) →
      PaperAlgebra.Commitment :=
  Fin.addCases fresh.commitments running.commitments

def outputPublicInput :
    Fin (productionShape.freshCount + productionShape.runningCount) →
      PaperAlgebra.PublicInput
        (logicalWidth := VerifierContext.candidateLogicalWidth)
        (publicFits := VerifierContext.candidatePublicFits) :=
  Fin.addCases fresh.publicInputs running.publicInputs

def outputCommitmentsValue : Value :=
  .array ((List.finRange productionShape.sourceCount).map fun source =>
    fieldWordsValue (serializeCommitment (outputCommitment source)))

def outputPublicInputsValue : Value :=
  .array ((List.finRange productionShape.sourceCount).map fun source =>
    fieldWordsValue (serializePublicInput (outputPublicInput source)))

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
    boolValue outputEval_KNonzero,
    boolValue outputEval_ANonzero]

/-- Caller-owned input and proof tuple.

Order: prior preimage, output preimage, prior public input, output digest,
verifier context, fresh commitment, round messages, output `Eval_K`, output
`Eval_A`, complete digest-only transcript blocks, semantic verifier-input
blocks, and the four complete verifier-context authority word lists. -/
def inputValue (computed : Computed) : Value :=
  .array [fieldWordsValue (statePreimageWords ()),
    fieldWordsValue (statePreimageWords ()),
    fieldWordsValue (statePublicInputWords ()),
    fieldWordsValue (stateDigest ()),
    fieldWordsValue stateVerifierKey,
    fieldWordsValue (serializeCommitment freshCommitment),
    roundMessagesValue computed,
    outputEval_KValue,
    outputEval_AValue,
    fieldBlocksValue (ProductionKey.publicInputBlocks running fresh),
    fieldBlocksValue (Transcript.verifierInputBlocks verifierInput),
    fieldBlocksValue [fixtureContextAuthority.relationWords,
      fixtureContextAuthority.applicationWords,
      fixtureContextAuthority.nifsKeyWords,
      fixtureContextAuthority.commitmentKeyWords]]

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
    outputPublicInputsValue,
    outputEval_KValue,
    outputEval_AValue,
    stateValue computed.outgoingState,
    assuranceValue computed]

/-- Schema 7 adds the complete authority preimage used to derive the separate
verifier-owned context input. -/
def parityValue (_ : Unit) : Value :=
  let computed := compute ()
  .array [.atom 7, inputValue computed, resultValue computed]

def render (_ : Unit) : String := (parityValue ()).render

end NightstreamFPrime.Export.Stage1.PiCCSParity
