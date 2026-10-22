import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.FullOutputCoordinates
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ProtocolPolynomialFixedWidth
import NightstreamFPrime.Spec.Folding.PiRLC.PaperCorrections

/-!
Operational public-coin core of the paper's strong `Pi_CCS` reduction.

Protocol: SuperNeo v1.2 `Pi_CCS` (Section 7.3 and Appendix B.2).
Reduction class: strong, SuperNeo v1.2 Definition 18, as v1.2 Lemma 7
states.
Phase: one explicit public-coin probe and extraction from an ambient output
witness.
Constraint family: paper semantics only; this file emits no rows.

Owns: a public statement without source assignments; explicit public coins;
one prover response containing finite SumCheck messages and the complete
paper `y'` family; verifier construction of the output CE product; the
definitional commitment projection `phi`; an adapter from the output witness
to `ConnectedInputs`/`UnifiedInputs`; and the deterministic extraction step
from acceptance plus corrected-ambient output validity to source truth or a
typed algebraic/SumCheck event.

Does not own: an adversary language, an extractor loop, expected runtime,
probability, Fiat--Shamir, Poseidon2, commitment hardness, Rust, R1CS, row
removal, or constraint counts.

Emits constraints: no.
-/

namespace NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.StrongReduction

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.SumCheck
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open FullOutputCoordinates
open MatrixCoefficientSource
open PaperLinearAlgebra
open UnifiedSources

universe uExtension uCommitment uPublicInput

structure EvaluationFamily (Extension : Type uExtension) (shape : Shape) where
  pad : Fin shape.coefficientCount -> Extension
  matrix : Fin shape.matrixCount -> Fin shape.coefficientCount -> Extension

structure RelationSource (shape : Shape) (columns blockCount : Nat) where
  cubeLayout : ColumnLayout shape.cubeVariables columns
  matrixSource : MatrixSource F shape columns blockCount

structure OpeningMaps
    (Commitment : Type uCommitment)
    (PublicInput : Type uPublicInput)
    (columns : Nat) where
  commit : Assignment F columns -> Commitment
  projectPublicInput : Assignment F columns -> PublicInput

def paperRelationSemantics
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {shape : Shape}
    {columns blockCount : Nat}
    (baseOps : InterpolationOps F)
    (extensionOps : InterpolationOps Extension)
    (lift : F -> Extension)
    (openingMaps : OpeningMaps Commitment PublicInput columns) :
    RelationSemantics
      (RelationSource shape columns blockCount)
      (Assignment F columns)
      PublicInput
      (CubePoint Extension shape.cubeVariables)
      (EvaluationFamily Extension shape)
      Commitment where
  commit := openingMaps.commit
  projectPublicInput := openingMaps.projectPublicInput
  normBounded := fun bound assignment =>
    forall column, centeredMagnitude (assignment column) < bound
  ccsSatisfied := fun source assignment =>
    CCSResidualTable.ConstraintSatisfied baseOps
      source.matrixSource.system assignment
  evaluationPointValid := fun _ _ => True
  evaluations := fun source assignment point => #[{
    pad := fun coefficient =>
      (BooleanTable.tabulate fun vertex =>
        lift (matrixVectorAt baseOps
          (source.matrixSource.coefficientMatrixOf baseOps
            (fun row column =>
              source.cubeLayout.paddedIdentityEntry
                baseOps.zero baseOps.one row column)
            coefficient)
          assignment vertex)).evaluate extensionOps point
    matrix := fun matrix coefficient =>
      (BooleanTable.tabulate fun vertex =>
        lift (matrixVectorAt baseOps
          (source.matrixSource.coefficientMatrix baseOps matrix coefficient)
          assignment vertex)).evaluate extensionOps point
  }]

structure Statement
    (Extension : Type uExtension)
    (Commitment : Type uCommitment)
    (PublicInput : Type uPublicInput)
    (shape : Shape)
    (columns blockCount : Nat)
    (baseOps : InterpolationOps F) where
  cubeLayout : ColumnLayout shape.cubeVariables columns
  matrixSource : MatrixSource F shape columns blockCount
  commitments : Fin shape.sourceCount -> Commitment
  publicInputs : Fin shape.sourceCount -> PublicInput
  priorPoint : CubePoint Extension shape.cubeVariables
  claimedPadCoefficient : PadCoordinate shape -> Extension
  claimedMatrixCoefficient : MatrixCoordinate shape -> Extension

structure OutputWitness (shape : Shape) (columns : Nat) where
  assignments : Fin shape.sourceCount -> Assignment F columns

namespace Statement

def sourceConnectedInputs
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {shape : Shape}
    {columns blockCount : Nat}
    {baseOps : InterpolationOps F}
    (statement : Statement Extension Commitment PublicInput shape
      columns blockCount baseOps)
    (witness : OutputWitness shape columns) :
    ConnectedInputs Extension shape columns blockCount where
  cubeLayout := statement.cubeLayout
  matrixSource := statement.matrixSource
  assignments := witness.assignments
  priorPoint := statement.priorPoint
  claimedPadCoefficient := statement.claimedPadCoefficient
  claimedMatrixCoefficient := statement.claimedMatrixCoefficient

def relationSource
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {shape : Shape}
    {columns blockCount : Nat}
    {baseOps : InterpolationOps F}
    (statement : Statement Extension Commitment PublicInput shape
      columns blockCount baseOps) : RelationSource shape columns blockCount where
  cubeLayout := statement.cubeLayout
  matrixSource := statement.matrixSource

def verifierInput
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {shape : Shape}
    {columns blockCount : Nat}
    {baseOps : InterpolationOps F}
    (lift : F -> Extension)
    (statement : Statement Extension Commitment PublicInput shape
      columns blockCount baseOps) :
    ProtocolPolynomial.VerifierInput Extension shape where
  constraintPolynomial :=
    ConstraintPolynomialLift.liftConstraintPolynomial lift
      statement.matrixSource.constraintPolynomial
  priorPoint := statement.priorPoint
  claimedPadCoefficient := statement.claimedPadCoefficient
  claimedMatrixCoefficient := statement.claimedMatrixCoefficient

def sourceProtocolData
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {shape : Shape}
    {columns blockCount : Nat}
    {baseOps : InterpolationOps F}
    (lift : F -> Extension)
    (statement : Statement Extension Commitment PublicInput shape
      columns blockCount baseOps)
    (witness : OutputWitness shape columns) :
    ProtocolPolynomial.Data Extension shape :=
  ProtocolDataRefinement.toProtocolData baseOps lift
    ((statement.sourceConnectedInputs witness).toUnifiedInputs baseOps)

theorem sourceProtocolData_toVerifierInput
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {shape : Shape}
    {columns blockCount : Nat}
    {baseOps : InterpolationOps F}
    (lift : F -> Extension)
    (statement : Statement Extension Commitment PublicInput shape
      columns blockCount baseOps)
    (witness : OutputWitness shape columns) :
    (statement.sourceProtocolData lift witness).toVerifierInput =
      statement.verifierInput lift := by
  rfl

def projectOutput
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {shape : Shape}
    {columns blockCount : Nat}
    {baseOps : InterpolationOps F}
    (statement : Statement Extension Commitment PublicInput shape
      columns blockCount baseOps)
    (output : FullOutput Extension shape) :
    ProtocolPolynomial.OutputMessage Extension shape where
  freshMatrixImage := fun source matrix =>
    output.matrixCoordinate (freshSourceIndex source) matrix
      statement.matrixSource.kernel.constant
  sourceAssignment := fun source =>
    output.padCoordinate source
      statement.matrixSource.kernel.constant
  padImage := fun coordinate =>
    output.padCoordinate (runningSourceIndex coordinate.running)
      coordinate.coefficient
  matrixImage := fun coordinate =>
    output.matrixCoordinate (runningSourceIndex coordinate.running)
      coordinate.matrix coordinate.coefficient

theorem projectOutput_eq_toOutputMessage
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {shape : Shape}
    {columns blockCount : Nat}
    {baseOps : InterpolationOps F}
    (statement : Statement Extension Commitment PublicInput shape
      columns blockCount baseOps)
    (witness : OutputWitness shape columns)
    (output : FullOutput Extension shape) :
    statement.projectOutput output =
      output.toOutputMessage (statement.sourceConnectedInputs witness) := by
  rfl

end Statement

structure PublicCoins (Extension : Type uExtension) (shape : Shape) where
  alpha : CubePoint Extension shape.cubeVariables
  gamma : Extension
  roundPoint : CubePoint Extension shape.cubeVariables

structure Response (Extension : Type uExtension) (shape : Shape) where
  rounds : SumCheck.Finite.Certificate Extension
  fullOutput : FullOutput Extension shape

structure Probe (Extension : Type uExtension) (shape : Shape) where
  coins : PublicCoins Extension shape
  response : Response Extension shape

namespace Probe

def Accepted
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    [DecidableEq Extension]
    {shape : Shape}
    {columns blockCount : Nat}
    {baseOps : InterpolationOps F}
    (extensionOps : InterpolationOps Extension)
    (lift : F -> Extension)
    (statement : Statement Extension Commitment PublicInput shape
      columns blockCount baseOps)
    (probe : Probe Extension shape) : Prop :=
  ProtocolPolynomial.check extensionOps (statement.verifierInput lift)
    probe.coins.alpha probe.coins.gamma probe.coins.roundPoint
    (statement.projectOutput probe.response.fullOutput)
    probe.response.rounds = true

def FixedWidthAccepted
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    [DecidableEq Extension]
    {shape : Shape}
    {columns blockCount : Nat}
    {baseOps : InterpolationOps F}
    (extensionOps : InterpolationOps Extension)
    (lift : F -> Extension)
    (statement : Statement Extension Commitment PublicInput shape
      columns blockCount baseOps)
    (width : Nat)
    (probe : Probe Extension shape) : Prop :=
  ProtocolPolynomial.FixedWidth.check extensionOps width
    (statement.verifierInput lift)
    probe.coins.alpha probe.coins.gamma probe.coins.roundPoint
    (statement.projectOutput probe.response.fullOutput)
    probe.response.rounds = true

end Probe

def FixedWidthSumCheckFailure
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {shape : Shape}
    {columns blockCount : Nat}
    {baseOps : InterpolationOps F}
    (extensionOps : InterpolationOps Extension)
    (lift : F -> Extension)
    (statement : Statement Extension Commitment PublicInput shape
      columns blockCount baseOps)
    (width challengeSetSize : Nat)
    (probe : Probe Extension shape)
    (witness : OutputWitness shape columns) : Prop :=
  exists certificate :
      SumCheck.Finite.FixedPhase.Certificate Extension width,
    SumCheck.Finite.FixedPhase.RawCertificate.decode width
        probe.response.rounds = some certificate /\
      ProtocolPolynomial.FixedWidth.SumCheckCollision extensionOps
        (statement.sourceProtocolData lift witness)
        probe.coins.alpha probe.coins.gamma width challengeSetSize
        probe.coins.roundPoint certificate

abbrev PublicOutput
    (Extension : Type uExtension)
    (Commitment : Type uCommitment)
    (PublicInput : Type uPublicInput)
    (shape : Shape)
    (columns blockCount : Nat) :=
  Fin shape.sourceCount ->
    CE.Instance
      (RelationSource shape columns blockCount)
      PublicInput
      (CubePoint Extension shape.cubeVariables)
      (EvaluationFamily Extension shape)
      Commitment

namespace Statement

def publicOutput
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {shape : Shape}
    {columns blockCount : Nat}
    {baseOps : InterpolationOps F}
    (statement : Statement Extension Commitment PublicInput shape
      columns blockCount baseOps)
    (probe : Probe Extension shape) :
    PublicOutput Extension Commitment PublicInput shape columns blockCount :=
  fun source => {
    constraintSystem := statement.relationSource
    commitment := statement.commitments source
    publicInput := statement.publicInputs source
    point := probe.coins.roundPoint
    evaluations := #[{
      pad := probe.response.fullOutput.padCoordinate source
      matrix := probe.response.fullOutput.matrixCoordinate source
    }]
    /- The protocol output is an instance of the honest target `CE(b)`.  The
    strong reduction's relaxed target `CE(⌊q/2⌋+1)` is a second relation over this
    same public instance and is stated by `CorrectedAmbientHolds`, which
    deliberately ignores this tag.  Marking the instance itself `.ambient`
    would make it unusable as the literal input of `Pi_RLC` and would conflate
    `R₂` with `R₂'` in SuperNeo v1.2 Theorem 12. -/
    stage := .fresh
  }

end Statement

def outputPhi
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {shape : Shape}
    {columns blockCount : Nat}
    (output : PublicOutput Extension Commitment PublicInput shape
      columns blockCount) :
    Fin shape.sourceCount -> Commitment :=
  fun source => (output source).commitment

theorem repeatedPublicOutputs_same_phi
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {shape : Shape}
    {columns blockCount : Nat}
    {baseOps : InterpolationOps F}
    (statement : Statement Extension Commitment PublicInput shape
      columns blockCount baseOps)
    (left right : Probe Extension shape) :
    outputPhi (statement.publicOutput left) =
      outputPhi (statement.publicOutput right) := by
  rfl

def AmbientOutputHolds
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {shape : Shape}
    {columns blockCount : Nat}
    {baseOps : InterpolationOps F}
    (extensionOps : InterpolationOps Extension)
    (lift : F -> Extension)
    (openingMaps : OpeningMaps Commitment PublicInput columns)
    (params : GlobalParams)
    (statement : Statement Extension Commitment PublicInput shape
      columns blockCount baseOps)
    (probe : Probe Extension shape)
    (witness : OutputWitness shape columns) : Prop :=
  forall source,
    NightstreamFPrime.Spec.Folding.PiRLC.PaperCorrections.CorrectedAmbientHolds
      (paperRelationSemantics baseOps extensionOps lift openingMaps)
      params (statement.publicOutput probe source)
      (witness.assignments source)

def SourceHolds
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {shape : Shape}
    {columns blockCount : Nat}
    {baseOps : InterpolationOps F}
    (extensionOps : InterpolationOps Extension)
    (lift : F -> Extension)
    (openingMaps : OpeningMaps Commitment PublicInput columns)
    (params : GlobalParams)
    (statement : Statement Extension Commitment PublicInput shape
      columns blockCount baseOps)
    (witness : OutputWitness shape columns) : Prop :=
  (forall source,
    Opening.Holds
      (paperRelationSemantics (shape := shape) (blockCount := blockCount)
        baseOps extensionOps lift openingMaps)
      params.b (statement.commitments source) (statement.publicInputs source)
      (witness.assignments source)) /\
  (statement.sourceConnectedInputs witness).SemanticTruth
    baseOps extensionOps lift

def ambientShape (shape : Shape) : Shape where
  cubeVariables := shape.cubeVariables
  freshCount := 0
  runningCount := shape.sourceCount
  matrixCount := shape.matrixCount
  coefficientCount := shape.coefficientCount

@[simp] theorem ambientShape_sourceCount (shape : Shape) :
    (ambientShape shape).sourceCount = shape.sourceCount := by
  simp [ambientShape, Shape.sourceCount]

private def ambientMatrixSource
    {shape : Shape}
    {columns blockCount : Nat}
    (source : MatrixSource F shape columns blockCount) :
    MatrixSource F (ambientShape shape) columns blockCount where
  columnLayout := source.columnLayout
  matrices := source.matrices
  constraintPolynomial := source.constraintPolynomial
  kernel := source.kernel

namespace Statement

def ambientConnectedInputs
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {shape : Shape}
    {columns blockCount : Nat}
    {baseOps : InterpolationOps F}
    (statement : Statement Extension Commitment PublicInput shape
      columns blockCount baseOps)
    (probe : Probe Extension shape)
    (witness : OutputWitness shape columns) :
    ConnectedInputs Extension (ambientShape shape) columns blockCount where
  cubeLayout := statement.cubeLayout
  matrixSource := ambientMatrixSource statement.matrixSource
  assignments := fun source =>
    witness.assignments (Fin.cast (ambientShape_sourceCount shape) source)
  priorPoint := probe.coins.roundPoint
  claimedPadCoefficient := fun coordinate =>
    probe.response.fullOutput.padCoordinate coordinate.running
      coordinate.coefficient
  claimedMatrixCoefficient := fun coordinate =>
    probe.response.fullOutput.matrixCoordinate coordinate.running
      coordinate.matrix coordinate.coefficient

def ambientUnifiedInputs
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {shape : Shape}
    {columns blockCount : Nat}
    {baseOps : InterpolationOps F}
    (statement : Statement Extension Commitment PublicInput shape
      columns blockCount baseOps)
    (probe : Probe Extension shape)
    (witness : OutputWitness shape columns) :
    UnifiedInputs Extension (ambientShape shape) columns :=
  (statement.ambientConnectedInputs probe witness).toUnifiedInputs baseOps

end Statement

private theorem fullOutput_ext
    {Extension : Type uExtension}
    {shape : Shape}
    (left right : FullOutput Extension shape)
    (padEqual : forall source coefficient,
      left.padCoordinate source coefficient =
        right.padCoordinate source coefficient)
    (matrixEqual : forall source matrix coefficient,
      left.matrixCoordinate source matrix coefficient =
        right.matrixCoordinate source matrix coefficient) :
    left = right := by
  cases left with
  | mk leftPad leftMatrix =>
      cases right with
      | mk rightPad rightMatrix =>
          congr
          · funext source coefficient
            exact padEqual source coefficient
          · funext source matrix coefficient
            exact matrixEqual source matrix coefficient

theorem fullOutput_eq_honestAt_of_ambientOutputHolds
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {shape : Shape}
    {columns blockCount : Nat}
    {baseOps : InterpolationOps F}
    (extensionOps : InterpolationOps Extension)
    (lift : F -> Extension)
    (openingMaps : OpeningMaps Commitment PublicInput columns)
    (params : GlobalParams)
    (statement : Statement Extension Commitment PublicInput shape
      columns blockCount baseOps)
    (probe : Probe Extension shape)
    (witness : OutputWitness shape columns)
    (ambient : AmbientOutputHolds extensionOps lift openingMaps params
      statement probe witness) :
    probe.response.fullOutput =
      FullOutput.honestAt baseOps extensionOps lift
        (statement.sourceConnectedInputs witness) probe.coins.roundPoint := by
  apply fullOutput_ext
  · intro source coefficient
    have evaluationsEqual := (ambient source).2.2
    change
      #[({
        pad := fun currentCoefficient =>
          (BooleanTable.tabulate fun vertex =>
            lift (matrixVectorAt baseOps
              (statement.matrixSource.coefficientMatrixOf baseOps
                (fun row column =>
                  statement.cubeLayout.paddedIdentityEntry
                    baseOps.zero baseOps.one row column)
                currentCoefficient)
              (witness.assignments source) vertex)).evaluate
                extensionOps probe.coins.roundPoint
        matrix := fun currentMatrix currentCoefficient =>
          (BooleanTable.tabulate fun vertex =>
            lift (matrixVectorAt baseOps
              (statement.matrixSource.coefficientMatrix baseOps
                currentMatrix currentCoefficient)
              (witness.assignments source) vertex)).evaluate
                extensionOps probe.coins.roundPoint
      } : EvaluationFamily Extension shape)] = #[({
        pad := probe.response.fullOutput.padCoordinate source
        matrix := probe.response.fullOutput.matrixCoordinate source
      } : EvaluationFamily Extension shape)] at evaluationsEqual
    have familyEqual :
        ({
          pad := fun currentCoefficient =>
            (BooleanTable.tabulate fun vertex =>
              lift (matrixVectorAt baseOps
                (statement.matrixSource.coefficientMatrixOf baseOps
                  (fun row column =>
                    statement.cubeLayout.paddedIdentityEntry
                      baseOps.zero baseOps.one row column)
                  currentCoefficient)
                (witness.assignments source) vertex)).evaluate
                  extensionOps probe.coins.roundPoint
          matrix := fun currentMatrix currentCoefficient =>
            (BooleanTable.tabulate fun vertex =>
              lift (matrixVectorAt baseOps
                (statement.matrixSource.coefficientMatrix baseOps
                  currentMatrix currentCoefficient)
                (witness.assignments source) vertex)).evaluate
                  extensionOps probe.coins.roundPoint
        } : EvaluationFamily Extension shape) = ({
          pad := probe.response.fullOutput.padCoordinate source
          matrix := probe.response.fullOutput.matrixCoordinate source
        } : EvaluationFamily Extension shape) := by
      have listEqual := congrArg Array.toList evaluationsEqual
      simpa using listEqual
    exact (congrFun (congrArg EvaluationFamily.pad familyEqual)
      coefficient).symm
  · intro source matrix coefficient
    have evaluationsEqual := (ambient source).2.2
    change
      #[({
        pad := fun currentCoefficient =>
          (BooleanTable.tabulate fun vertex =>
            lift (matrixVectorAt baseOps
              (statement.matrixSource.coefficientMatrixOf baseOps
                (fun row column =>
                  statement.cubeLayout.paddedIdentityEntry
                    baseOps.zero baseOps.one row column)
                currentCoefficient)
              (witness.assignments source) vertex)).evaluate
                extensionOps probe.coins.roundPoint
        matrix := fun currentMatrix currentCoefficient =>
          (BooleanTable.tabulate fun vertex =>
            lift (matrixVectorAt baseOps
              (statement.matrixSource.coefficientMatrix baseOps
                currentMatrix currentCoefficient)
              (witness.assignments source) vertex)).evaluate
                extensionOps probe.coins.roundPoint
      } : EvaluationFamily Extension shape)] = #[({
        pad := probe.response.fullOutput.padCoordinate source
        matrix := probe.response.fullOutput.matrixCoordinate source
      } : EvaluationFamily Extension shape)] at evaluationsEqual
    have familyEqual :
        ({
          pad := fun currentCoefficient =>
            (BooleanTable.tabulate fun vertex =>
              lift (matrixVectorAt baseOps
                (statement.matrixSource.coefficientMatrixOf baseOps
                  (fun row column =>
                    statement.cubeLayout.paddedIdentityEntry
                      baseOps.zero baseOps.one row column)
                  currentCoefficient)
                (witness.assignments source) vertex)).evaluate
                  extensionOps probe.coins.roundPoint
          matrix := fun currentMatrix currentCoefficient =>
            (BooleanTable.tabulate fun vertex =>
              lift (matrixVectorAt baseOps
                (statement.matrixSource.coefficientMatrix baseOps
                  currentMatrix currentCoefficient)
                (witness.assignments source) vertex)).evaluate
                  extensionOps probe.coins.roundPoint
        } : EvaluationFamily Extension shape) = ({
          pad := probe.response.fullOutput.padCoordinate source
          matrix := probe.response.fullOutput.matrixCoordinate source
        } : EvaluationFamily Extension shape) := by
      have listEqual := congrArg Array.toList evaluationsEqual
      simpa using listEqual
    exact (congrFun (congrFun
      (congrArg EvaluationFamily.matrix familyEqual) matrix) coefficient).symm

theorem projectedOutput_eq_messageAt_of_ambientOutputHolds
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {shape : Shape}
    {columns blockCount : Nat}
    {baseOps : InterpolationOps F}
    (baseLaws : InterpolationEvaluationLaws baseOps)
    (baseZero : NormResidualTable.BaseZeroAgreement baseOps)
    (extensionOps : InterpolationOps Extension)
    (lift : F -> Extension)
    (openingMaps : OpeningMaps Commitment PublicInput columns)
    (params : GlobalParams)
    (statement : Statement Extension Commitment PublicInput shape
      columns blockCount baseOps)
    (constantLaw : ConstantTermLaw baseOps statement.matrixSource.kernel)
    (probe : Probe Extension shape)
    (witness : OutputWitness shape columns)
    (ambient : AmbientOutputHolds extensionOps lift openingMaps params
      statement probe witness) :
    statement.projectOutput probe.response.fullOutput =
      ProtocolPolynomial.messageAt extensionOps
        (statement.sourceProtocolData lift witness) probe.coins.roundPoint := by
  have fullOutputEqual := fullOutput_eq_honestAt_of_ambientOutputHolds
    extensionOps lift openingMaps params statement probe witness ambient
  calc
    statement.projectOutput probe.response.fullOutput =
        probe.response.fullOutput.toOutputMessage
          (statement.sourceConnectedInputs witness) :=
      statement.projectOutput_eq_toOutputMessage witness _
    _ = (FullOutput.honestAt baseOps extensionOps lift
          (statement.sourceConnectedInputs witness) probe.coins.roundPoint).toOutputMessage
        (statement.sourceConnectedInputs witness) := by
      rw [fullOutputEqual]
    _ = ProtocolPolynomial.messageAt extensionOps
        (statement.sourceProtocolData lift witness) probe.coins.roundPoint :=
      FullOutput.honestAt_toOutputMessage_eq_messageAt
        baseOps baseLaws baseZero extensionOps lift
        (statement.sourceConnectedInputs witness) constantLaw
        probe.coins.roundPoint

theorem acceptedProbe_extracts_source_or_badEvent
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    [DecidableEq Extension]
    {shape : Shape}
    {columns blockCount : Nat}
    {baseOps : InterpolationOps F}
    (baseLaws : InterpolationEvaluationLaws baseOps)
    (baseZero : NormResidualTable.BaseZeroAgreement baseOps)
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    (extensionOps : InterpolationOps Extension)
    (extensionLaws : InterpolationEvaluationLaws extensionOps)
    (extensionZeroLaws : InterpolationZeroLaws extensionOps)
    (lift : F -> Extension)
    (liftLaws : ProtocolDataRefinement.ProtocolLift
      baseOps extensionOps lift)
    (openingMaps : OpeningMaps Commitment PublicInput columns)
    (params : GlobalParams)
    (freshBound : params.b = 2)
    (statement : Statement Extension Commitment PublicInput shape
      columns blockCount baseOps)
    (constantLaw : ConstantTermLaw baseOps statement.matrixSource.kernel)
    (challengeSetSize : Nat)
    (probe : Probe Extension shape)
    (witness : OutputWitness shape columns)
    (ambient : AmbientOutputHolds extensionOps lift openingMaps params
      statement probe witness)
    (accepted : probe.Accepted extensionOps lift statement) :
    SourceHolds extensionOps lift openingMaps params statement witness \/
      SignedCoefficientObject.MixingRoot extensionOps
        ((statement.sourceProtocolData lift witness).toJointData extensionOps)
        probe.coins.alpha probe.coins.gamma \/
      (exists round,
        SumCheck.BadChallenge
          (SumCheckInitial.symbolicInstance extensionOps
            ((statement.sourceProtocolData lift witness).toJointData
              extensionOps)
            probe.coins.alpha probe.coins.gamma
            (statement.verifierInput lift).sumcheckDegreeBound
            challengeSetSize probe.coins.roundPoint.coordinates
            (ProtocolPolynomial.terminalFromMessage extensionOps
              (statement.verifierInput lift)
              probe.coins.alpha probe.coins.gamma probe.coins.roundPoint
              (statement.projectOutput probe.response.fullOutput))
            probe.response.rounds
            (ProtocolPolynomial.canonicalExpected extensionOps
              (statement.sourceProtocolData lift witness)
              probe.coins.alpha probe.coins.gamma
              probe.coins.roundPoint.coordinates))
          round) := by
  have checked :
      ProtocolPolynomial.check extensionOps
          (statement.sourceProtocolData lift witness).toVerifierInput
          probe.coins.alpha probe.coins.gamma probe.coins.roundPoint
          (statement.projectOutput probe.response.fullOutput)
          probe.response.rounds = true := by
    rw [statement.sourceProtocolData_toVerifierInput lift witness]
    exact accepted
  rcases ProtocolPolynomial.check_implies_tableTruth_or_badEvent
      extensionOps extensionLaws extensionZeroLaws
      (statement.sourceProtocolData lift witness)
      probe.coins.alpha probe.coins.gamma challengeSetSize
      probe.coins.roundPoint (statement.projectOutput probe.response.fullOutput)
      probe.response.rounds checked with
    tableTruth | mixingRoot | badChallenge | outputMismatch
  · left
    let unifiedData :=
      (statement.sourceConnectedInputs witness).toUnifiedInputs baseOps
    have independentTableTruth :
        (TableResidualData.toTableObligations extensionOps
          (SignedCoefficientObject.toTableResidualData extensionOps
            (unifiedData.toIndependentInputs.toJointData baseOps lift))).AllHold := by
      rw [← ProtocolDataRefinement.toProtocolData_toJointData_eq
        baseOps extensionOps lift liftLaws unifiedData]
      simpa [Statement.sourceProtocolData, unifiedData] using tableTruth
    have independentSemantic :=
      (ConcreteJointData.jointTableTruth_iff_semanticTruth
        baseOps baseZero noZeroDivisors extensionOps extensionLaws lift
        liftLaws.toZeroReflectingLift unifiedData.toIndependentInputs).mp
          independentTableTruth
    have sourceSemantic :
        (statement.sourceConnectedInputs witness).SemanticTruth
          baseOps extensionOps lift := by
      simpa [ConnectedInputs.SemanticTruth, unifiedData] using
        (unifiedData.toIndependentInputs_semanticTruth_iff
          baseOps extensionOps lift).mp independentSemantic
    refine ⟨?_, sourceSemantic⟩
    intro source
    have ambientOpening := (ambient source).1
    refine ⟨ambientOpening.1, ambientOpening.2.1, ?_⟩
    intro column
    change centeredMagnitude (witness.assignments source column) < params.b
    rw [freshBound]
    exact sourceSemantic.2.1 source column
  · exact Or.inr (Or.inl mixingRoot)
  · exact Or.inr (Or.inr badChallenge)
  · exfalso
    apply outputMismatch
    unfold ProtocolPolynomial.qAtPoint
    rw [projectedOutput_eq_messageAt_of_ambientOutputHolds
      baseLaws baseZero extensionOps lift openingMaps params statement constantLaw
      probe witness ambient]

theorem fixedWidthAcceptedProbe_extracts_source_or_badEvent
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    [DecidableEq Extension]
    {shape : Shape}
    {columns blockCount : Nat}
    {baseOps : InterpolationOps F}
    (baseLaws : InterpolationEvaluationLaws baseOps)
    (baseZero : NormResidualTable.BaseZeroAgreement baseOps)
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    (extensionOps : InterpolationOps Extension)
    (extensionLaws : InterpolationEvaluationLaws extensionOps)
    (extensionZeroLaws : InterpolationZeroLaws extensionOps)
    (lift : F -> Extension)
    (liftLaws : ProtocolDataRefinement.ProtocolLift
      baseOps extensionOps lift)
    (openingMaps : OpeningMaps Commitment PublicInput columns)
    (params : GlobalParams)
    (freshBound : params.b = 2)
    (statement : Statement Extension Commitment PublicInput shape
      columns blockCount baseOps)
    (constantLaw : ConstantTermLaw baseOps statement.matrixSource.kernel)
    (width : Nat)
    (degreeCovers :
      (statement.verifierInput lift).sumcheckDegreeBound <= width)
    (challengeSetSize : Nat)
    (probe : Probe Extension shape)
    (witness : OutputWitness shape columns)
    (ambient : AmbientOutputHolds extensionOps lift openingMaps params
      statement probe witness)
    (accepted :
      probe.FixedWidthAccepted extensionOps lift statement width) :
    SourceHolds extensionOps lift openingMaps params statement witness \/
      SignedCoefficientObject.MixingRoot extensionOps
        ((statement.sourceProtocolData lift witness).toJointData extensionOps)
        probe.coins.alpha probe.coins.gamma \/
      FixedWidthSumCheckFailure extensionOps lift statement width
        challengeSetSize probe witness := by
  let data := statement.sourceProtocolData lift witness
  have inputEqual : data.toVerifierInput = statement.verifierInput lift :=
    statement.sourceProtocolData_toVerifierInput lift witness
  have checked :
      ProtocolPolynomial.FixedWidth.check extensionOps width
          data.toVerifierInput
          probe.coins.alpha probe.coins.gamma probe.coins.roundPoint
          (statement.projectOutput probe.response.fullOutput)
          probe.response.rounds = true := by
    rw [inputEqual]
    exact accepted
  obtain ⟨certificate, decoded, chain⟩ :=
    (ProtocolPolynomial.FixedWidth.check_eq_true_iff extensionOps width
      data.toVerifierInput probe.coins.alpha probe.coins.gamma
      probe.coins.roundPoint
      (statement.projectOutput probe.response.fullOutput)
      probe.response.rounds).1 checked
  have dataDegreeCovers : data.toVerifierInput.sumcheckDegreeBound <= width := by
    rw [inputEqual]
    exact degreeCovers
  rcases
      ProtocolPolynomial.FixedWidth.accepted_implies_tableTruth_or_badEvent
        extensionOps extensionLaws extensionZeroLaws data
        probe.coins.alpha probe.coins.gamma width dataDegreeCovers
        challengeSetSize probe.coins.roundPoint
        (statement.projectOutput probe.response.fullOutput)
        certificate chain with
    tableTruth | mixingRoot | badChallenge | outputMismatch
  · left
    let unifiedData :=
      (statement.sourceConnectedInputs witness).toUnifiedInputs baseOps
    have independentTableTruth :
        (TableResidualData.toTableObligations extensionOps
          (SignedCoefficientObject.toTableResidualData extensionOps
            (unifiedData.toIndependentInputs.toJointData baseOps lift))).AllHold := by
      rw [← ProtocolDataRefinement.toProtocolData_toJointData_eq
        baseOps extensionOps lift liftLaws unifiedData]
      simpa [data, Statement.sourceProtocolData, unifiedData] using tableTruth
    have independentSemantic :=
      (ConcreteJointData.jointTableTruth_iff_semanticTruth
        baseOps baseZero noZeroDivisors extensionOps extensionLaws lift
        liftLaws.toZeroReflectingLift unifiedData.toIndependentInputs).mp
          independentTableTruth
    have sourceSemantic :
        (statement.sourceConnectedInputs witness).SemanticTruth
          baseOps extensionOps lift := by
      simpa [ConnectedInputs.SemanticTruth, unifiedData] using
        (unifiedData.toIndependentInputs_semanticTruth_iff
          baseOps extensionOps lift).mp independentSemantic
    refine ⟨?_, sourceSemantic⟩
    intro source
    have ambientOpening := (ambient source).1
    refine ⟨ambientOpening.1, ambientOpening.2.1, ?_⟩
    intro column
    change centeredMagnitude (witness.assignments source column) < params.b
    rw [freshBound]
    exact sourceSemantic.2.1 source column
  · exact Or.inr (Or.inl mixingRoot)
  · exact Or.inr (Or.inr ⟨certificate, decoded, badChallenge⟩)
  · exfalso
    apply outputMismatch
    unfold ProtocolPolynomial.qAtPoint
    rw [projectedOutput_eq_messageAt_of_ambientOutputHolds
      baseLaws baseZero extensionOps lift openingMaps params statement constantLaw
      probe witness ambient]

end NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.StrongReduction
