import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.FullOutputCoordinates
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ProtocolPolynomialFixedWidth
import Nightstream.SuperNeo.Folding.PiRLC.PaperCorrections

/-!
Operational public-coin core of the paper's strong `Pi_CCS` reduction.

Protocol: SuperNeo `Pi_CCS` (Section 7.3 and Appendix D.4).
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

Authority boundary: the statement owns matrices, prior claims, commitments,
and public inputs. The response owns neither a verifier input nor a public
output instance. The verifier constructs every output instance by copying
the statement commitment/public input and attaching its sampled point and the
response's one complete output family. Ambient membership is stated literally
with `PiRLC.PaperCorrections.CorrectedAmbientHolds`. It is not source validity.
Those ambient evaluation equations derive that the response is `honestAt`,
so the generic protocol-polynomial `OutputMismatch` branch is impossible.

| Protocol path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `pi_ccs.strong.statement` | public structure, prior claims, commitments, and public inputs | direct dataflow | `Statement` |
| `pi_ccs.strong.public_coins` | independent `alpha`, `gamma`, and all SumCheck round challenges | checked | `PublicCoins` |
| `pi_ccs.strong.response` | finite round messages plus one complete `y'` family | prover message | `Response` |
| `pi_ccs.strong.output` | output CE instances are verifier constructed | computed | `Statement.publicOutput` |
| `pi_ccs.strong.phi` | repeated outputs copy the same statement commitments | derived/definitional | `repeatedPublicOutputs_same_phi` |
| `pi_ccs.strong.ambient` | each output witness satisfies the corrected strict ambient relation | checked target relation | `AmbientOutputHolds` |
| `pi_ccs.strong.output_adapter` | output witness becomes one connected all-running source family | computed | `ambientConnectedInputs` / `ambientUnifiedInputs` |
| `pi_ccs.strong.output_exact` | ambient evaluation truth forces the complete response to be honest | derived | `fullOutput_eq_honestAt_of_ambientOutputHolds` |
| `pi_ccs.strong.extract` | acceptance gives source truth, a mixing root, or a SumCheck collision | derived | `acceptedProbe_extracts_source_or_badEvent` |
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongReduction

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.SumCheck
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open FullOutputCoordinates
open MatrixCoefficientSource
open PaperLinearAlgebra
open UnifiedSources

universe uExtension uCommitment uPublicInput

/-- One source's complete CE evaluation family. Keeping matrices and
coefficients typed avoids introducing an unrelated array-serialization order. -/
abbrev EvaluationFamily (Extension : Type uExtension) (shape : Shape) :=
  Fin shape.matrixCount -> Fin shape.coefficientCount -> Extension

/-- Public binding maps shared by the source and corrected-ambient target
relations. Their cryptographic properties are outside this deterministic
leaf. -/
structure OpeningMaps
    (Commitment : Type uCommitment)
    (PublicInput : Type uPublicInput)
    (columns : Nat) where
  commit : Assignment F columns -> Commitment
  projectPublicInput : Assignment F columns -> PublicInput

/-- Canonical paper relation operations over the connected matrix source.
The evaluation array has one typed element containing the complete
matrix/coefficient family; no coordinate is omitted or reordered. -/
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
      (MatrixSource F shape columns blockCount)
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
    CCSResidualTable.ConstraintSatisfied baseOps source.system assignment
  evaluationPointValid := fun _ _ => True
  evaluations := fun source assignment point => #[fun matrix coefficient =>
    (BooleanTable.tabulate fun vertex =>
      lift (matrixVectorAt baseOps
        (source.coefficientMatrix baseOps matrix coefficient)
        assignment vertex)).evaluate extensionOps point]

/-- The public input to one paper `Pi_CCS` execution. Source assignments are
deliberately absent. `M_1 = [I; 0]` is entrywise structure data, not an assumed
matrix-image equality. -/
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
  claimedCoefficient : CarriedCoordinate shape -> Extension
  matrixCountPositive : 0 < shape.matrixCount
  identityFirstEntry : forall
      (vertex : BooleanVertex shape.cubeVariables)
      (column : Fin columns),
    matrixSource.matrices ⟨0, matrixCountPositive⟩ vertex column =
      cubeLayout.paddedIdentityEntry baseOps.zero baseOps.one vertex column

/-- An extracted output witness is exactly one assignment for every source in
the statement's canonical `K+k` order. -/
structure OutputWitness (shape : Shape) (columns : Nat) where
  assignments : Fin shape.sourceCount -> Assignment F columns

namespace Statement

/-- Attach an extracted assignment vector to the public source statement.
This is the sole source-side `ConnectedInputs` constructor in this leaf. -/
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
  claimedCoefficient := statement.claimedCoefficient

/-- The public statement fixes the exact first-matrix identity proof for any
attached witness; the proof does not inspect that witness. -/
def identityFirstMatrix
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {shape : Shape}
    {columns blockCount : Nat}
    {baseOps : InterpolationOps F}
    (statement : Statement Extension Commitment PublicInput shape
      columns blockCount baseOps)
    (witness : OutputWitness shape columns) :
    IdentityFirstMatrix baseOps (statement.sourceConnectedInputs witness) where
  matrixCountPositive := statement.matrixCountPositive
  entry := statement.identityFirstEntry

/-- Verifier-visible protocol input derived solely from public statement
fields. No source assignment or semantic table is consulted. -/
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
  claimedCoefficient := statement.claimedCoefficient

/-- Rich source protocol data used only in the semantic reduction theorem. -/
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

/-- Attaching a witness changes hidden semantic tables but not the verifier's
public input. -/
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

/-- Verifier projection of one complete response. Only the first-matrix index
and constant kernel coordinate are used for assignment/fresh scalar fields. -/
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
    output.coordinate (freshSourceIndex source) matrix
      statement.matrixSource.kernel.constant
  sourceAssignment := fun source =>
    output.coordinate source ⟨0, statement.matrixCountPositive⟩
      statement.matrixSource.kernel.constant
  carriedImage := fun coordinate =>
    output.coordinate (runningSourceIndex coordinate.running)
      coordinate.matrix coordinate.coefficient

/-- The verifier projection is exactly the projection from the complete
output owner proved in `FullOutputCoordinates`. -/
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
      output.toOutputMessage (statement.identityFirstMatrix witness) := by
  rfl

end Statement

/-- All verifier randomness in one interactive paper probe. The prover does
not supply any of these values. -/
structure PublicCoins (Extension : Type uExtension) (shape : Shape) where
  alpha : CubePoint Extension shape.cubeVariables
  gamma : Extension
  roundPoint : CubePoint Extension shape.cubeVariables

/-- The prover's complete response after the public coins: one finite
SumCheck certificate and one complete `y'` family. -/
structure Response (Extension : Type uExtension) (shape : Shape) where
  rounds : SumCheck.Finite.Certificate Extension
  fullOutput : FullOutput Extension shape

/-- One deterministic probe of the public-coin interaction. -/
structure Probe (Extension : Type uExtension) (shape : Shape) where
  coins : PublicCoins Extension shape
  response : Response Extension shape

namespace Probe

/-- Operational acceptance with explicit public coins. This is intentionally
the paper-polynomial checker, not Fiat--Shamir or `ProtocolVerifier`. -/
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

/-- Operational paper acceptance at one verifier-owned common coefficient
width.  Unlike `Accepted`, this relation does not impose canonical trimming;
it rejects only width mismatch or a failed SumCheck equation. -/
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

/-- Exact fixed-width SumCheck collision exposed from the submitted raw
certificate.  The decoder receipt prevents an existential certificate from
being substituted after the fact. -/
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

/-- The verifier's complete public output product. -/
abbrev PublicOutput
    (Extension : Type uExtension)
    (Commitment : Type uCommitment)
    (PublicInput : Type uPublicInput)
    (shape : Shape)
    (columns blockCount : Nat) :=
  Fin shape.sourceCount ->
    CE.Instance
      (MatrixSource F shape columns blockCount)
      PublicInput
      (CubePoint Extension shape.cubeVariables)
      (EvaluationFamily Extension shape)
      Commitment

namespace Statement

/-- Construct the public output CE product. Commitments and public inputs are
copied from the statement; the point and evaluations come from this probe. -/
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
    constraintSystem := statement.matrixSource
    commitment := statement.commitments source
    publicInput := statement.publicInputs source
    point := probe.coins.roundPoint
    evaluations := #[fun matrix coefficient =>
      probe.response.fullOutput.coordinate source matrix coefficient]
    /- The protocol output is an instance of the honest target `CE(b)`.  The
    strong reduction's relaxed target `CE(⌊q/2⌋+1)` is a second relation over this
    same public instance and is stated by `CorrectedAmbientHolds`, which
    deliberately ignores this tag.  Marking the instance itself `.ambient`
    would make it unusable as the literal input of `Pi_RLC` and would conflate
    `R₂` with `R₂'` in Theorem 6. -/
    stage := .fresh
  }

end Statement

/-- The strong-reduction projection of a public output is its complete
commitment vector. -/
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

/-- Repeated public outputs for one statement have identical `phi` by
reduction, because the verifier copies commitments rather than accepting them
from either response. -/
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

/-- The corrected ambient target relation for the verifier-constructed output
product. This is literally the relation shared with `Pi_RLC`; no fresh norm
convention or `q / 2` alias is introduced here. -/
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
    Nightstream.SuperNeo.Folding.PiRLC.PaperCorrections.CorrectedAmbientHolds
      (paperRelationSemantics baseOps extensionOps lift openingMaps)
      params (statement.publicOutput probe source)
      (witness.assignments source)

/-- The source relation extracted by this bounded strong-reduction leaf:
every assignment opens the statement at the verifier-owned fresh bound, and
the single connected paper source family satisfies CCS, strict norm, and
prior carried-evaluation semantics. -/
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

/-- Re-index every output as a running CE source. This is the paper target's
`K+k` evaluation product, not a new protocol shape. -/
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

/-- Corrected-ambient output adapter into the same connected source
vocabulary used by the paper polynomial. Every original source is a running
evaluation source at the verifier's sampled output point. -/
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
  claimedCoefficient := fun coordinate =>
    probe.response.fullOutput.coordinate coordinate.running
      coordinate.matrix coordinate.coefficient

/-- Unified form of the corrected-ambient output adapter. Coefficient
matrices remain derived from the sole statement matrix source. -/
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
    (equal : forall source matrix coefficient,
      left.coordinate source matrix coefficient =
        right.coordinate source matrix coefficient) :
    left = right := by
  cases left with
  | mk leftCoordinate =>
      cases right with
      | mk rightCoordinate =>
          congr
          funext source matrix coefficient
          exact equal source matrix coefficient

/-- Corrected-ambient evaluation truth forces every coordinate in the
response's complete output family to be the honest evaluation of the extracted
assignment at the verifier's sampled point. -/
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
  intro source matrix coefficient
  have evaluationsEqual := (ambient source).2.2
  change
    #[fun currentMatrix currentCoefficient =>
      (BooleanTable.tabulate fun vertex =>
        lift (matrixVectorAt baseOps
          (statement.matrixSource.coefficientMatrix baseOps
            currentMatrix currentCoefficient)
          (witness.assignments source) vertex)).evaluate
            extensionOps probe.coins.roundPoint] =
      #[fun currentMatrix currentCoefficient =>
        probe.response.fullOutput.coordinate source
          currentMatrix currentCoefficient] at evaluationsEqual
  have familyEqual :
      (fun currentMatrix currentCoefficient =>
        (BooleanTable.tabulate fun vertex =>
          lift (matrixVectorAt baseOps
            (statement.matrixSource.coefficientMatrix baseOps
              currentMatrix currentCoefficient)
            (witness.assignments source) vertex)).evaluate
              extensionOps probe.coins.roundPoint) =
        (fun currentMatrix currentCoefficient =>
          probe.response.fullOutput.coordinate source
            currentMatrix currentCoefficient) := by
    have listEqual := congrArg Array.toList evaluationsEqual
    simpa using listEqual
  exact (congrFun (congrFun familyEqual matrix) coefficient).symm

/-- Consequently the verifier's projected response is exactly
`ProtocolPolynomial.messageAt`; the generic `OutputMismatch` event has no
remaining premise. -/
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
          (statement.identityFirstMatrix witness) :=
      statement.projectOutput_eq_toOutputMessage witness _
    _ = (FullOutput.honestAt baseOps extensionOps lift
          (statement.sourceConnectedInputs witness) probe.coins.roundPoint).toOutputMessage
        (statement.identityFirstMatrix witness) := by
      rw [fullOutputEqual]
    _ = ProtocolPolynomial.messageAt extensionOps
        (statement.sourceProtocolData lift witness) probe.coins.roundPoint :=
      FullOutput.honestAt_toOutputMessage_eq_messageAt
        baseOps baseLaws baseZero extensionOps lift
        (statement.sourceConnectedInputs witness) constantLaw
        (statement.identityFirstMatrix witness) probe.coins.roundPoint

/-- One accepted explicit public-coin probe plus a witness for the
verifier-constructed corrected-ambient output yields source membership, an
explicit alpha/gamma mixing-polynomial root, or a concrete SumCheck bad
challenge. There is no output-mismatch or generic refinement branch. -/
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

/-- Fixed-width counterpart of `acceptedProbe_extracts_source_or_badEvent`.

This is the paper-owned gate used by the causal interactive composition.  It
accepts the same exact-width messages as the frozen NIFS verifier and exposes
the same fixed-phase bad-challenge event; canonical variable-length encoding
is absent. -/
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

end Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongReduction
