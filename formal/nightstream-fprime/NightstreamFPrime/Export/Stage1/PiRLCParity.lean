import NightstreamFPrime.Export.Codec
import NightstreamFPrime.Export.Stage1.PiCCSParity
import NightstreamFPrime.Export.Stage1.PiRLCNonzero

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

def pointValue : Value :=
  PiCCSParity.extensionWordsValue point.coordinates

def inputValue : Value :=
  .array [PiCCSParity.stateValue initialState,
    pointValue,
    PiCCSParity.outputCommitmentsValue,
    PiCCSParity.outputPublicInputsValue,
    PiCCSParity.outputEval_KValue,
    PiCCSParity.outputEval_AValue]

def ringHasNonzero (value : RingF) : Bool :=
  (List.finRange ringDegree).any fun coefficient =>
    decide (value coefficient ≠ 0)

def commitmentHasNonzero (value : PaperAlgebra.Commitment) : Bool :=
  (List.finRange productionProfile.commitmentWidth).any fun row =>
    ringHasNonzero (value row)

def publicInputHasNonzero
    (value : PublicInput (logicalWidth := Data.logicalWidth)
      (publicFits := Data.publicFits)) : Bool :=
  (List.finRange (FullShape Data.logicalWidth Data.publicFits).publicWidth).any
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

def resultValue : Value :=
  match sampled with
  | none => .array [PiCCSParity.boolValue false]
  | some batch =>
      .array [PiCCSParity.boolValue true,
        challengesValue batch.challenges,
        membershipValue,
        combinedCommitmentValue batch.challenges,
        combinedPublicInputValue batch.challenges,
        pointValue,
        combinedEvalKValue batch.challenges,
        combinedEvalAValue batch.challenges,
        PiCCSParity.stateValue batch.finalState,
        assuranceValue batch.challenges]

/-- Schema 1 is the complete paper-level PiRLC result. -/
def parityValue : Value := .array [.atom 1, inputValue, resultValue]

def render : String := parityValue.render

end NightstreamFPrime.Export.Stage1.PiRLCParity
