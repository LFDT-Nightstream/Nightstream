import Nightstream.Implementation.NebulaV2.NIFS.Core.Poseidon2
import Nightstream.Implementation.R1CS.Canonical.KPiCcsOccurrence
import Nightstream.Implementation.R1CS.Canonical.SymbolicDuplex

/-!
Contract: exact symbolic Poseidon2 and arithmetic rows for V2 PiCCS.

Owns the selected V2 public-input absorption, Construction-3 statement and
challenge frames, twenty-five fixed-width SumCheck messages, complete PiCCS
output absorption, and the fixed-phase PiCCS arithmetic occurrence. The
projected output read by PiCCS is a view of the same coefficient-complete
output wires absorbed before PiRLC.

Does not own physical placement of public-input or proof fields, transcript
semantics, PiRLC, PiDEC, honest witness generation, cryptographic security,
Rust refinement, or a complete NIFS verifier result.

Emits constraints: through `SymbolicDuplex.rows` and `KPiCcsOccurrence.rows`.
-/

set_option autoImplicit false
set_option maxRecDepth 30000

namespace Nightstream.Implementation.NebulaV2.ProductPiCcsTranscriptRows

open Nightstream.Implementation.NebulaV2
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.UnifiedSources

def selectedShape :
    Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Shape :=
  ProductNifsCodec.shape
abbrev Carried := KMul.Carried
abbrev Round := KFixedPhaseSumCheck.Round 9

/-- A verifier-owned field constant on the constant-one wire. -/
def word (value : Nat) : LinComb := [(0, value % goldilocksP)]

def carriedFields (value : Carried) : List LinComb :=
  [value.low, value.high]

def constantKFields (value : K) : List LinComb :=
  [word value.c0.val, word value.c1.val]

def shapeFields
    (shape : Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Shape) :
    List LinComb :=
  [word shape.cubeVariables, word shape.freshCount,
    word shape.runningCount, word shape.matrixCount,
    word shape.coefficientCount]

def monomialFields
    {shape : Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Shape}
    (monomial : CCSResidualTable.Monomial K shape.matrixCount) : List LinComb :=
  constantKFields monomial.coefficient ++
    (canonicalFinIndices shape.matrixCount).map fun index =>
      word (monomial.exponents index)

def polynomialFields
    {shape : Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Shape}
    (polynomial : CCSResidualTable.ConstraintPolynomial K shape.matrixCount) :
    List LinComb :=
  word polynomial.degreeBound :: word polynomial.terms.length ::
    polynomial.terms.flatMap monomialFields

def pointFields
    {count : Nat} (point : Fin count -> Carried) : List LinComb :=
  (canonicalFinIndices count).flatMap fun index => carriedFields (point index)

def verifierChallengeFields
    (eventIndex challengeIndex challengeType : Nat)
    (coordinates : List Nat) : List LinComb :=
  ProductPoseidon2.construction3DomainFields.map word ++
    ProductPoseidon2.verifierChallengeLabelFields.map word ++
    [word eventIndex, word challengeIndex, word challengeType,
      word coordinates.length] ++ coordinates.map word

def proverMessageFields
    (eventIndex messageIndex messageType : Nat)
    (payload : List LinComb) : List LinComb :=
  ProductPoseidon2.proverMessageLabelFields.map word ++
    [word eventIndex, word messageIndex, word messageType,
      word payload.length] ++ payload

/-- Row-visible values for one exact V2 PiCCS execution.

`publicNifsFields` must contain exactly the selected public serialization.
Its physical placement and typed meaning are proved by a separate bridge.
All other prover values use typed fixed-size functions or fixed-width round
objects, so there is no length or optional-presence certificate surface. -/
structure Input where
  statementId : ProductPoseidon2.StatementId
  constraintPolynomial :
    CCSResidualTable.ConstraintPolynomial K selectedShape.matrixCount
  publicNifsFields : List LinComb
  publicNifsFields_length : publicNifsFields.length = 87655
  priorPoint : Fin selectedShape.cubeVariables -> Carried
  claimedCoefficient : CarriedCoordinate selectedShape -> Carried
  rounds : Fin selectedShape.cubeVariables -> Round
  fullOutput : Fin selectedShape.sourceCount -> Fin selectedShape.matrixCount ->
    Fin selectedShape.coefficientCount -> Carried
  current : Carried
  terminal : Carried
  transcriptBase : Nat

def statementFields (input : Input) : List LinComb :=
  [word 41] ++ shapeFields selectedShape ++
    polynomialFields input.constraintPolynomial ++
    [word selectedShape.cubeVariables] ++ pointFields input.priorPoint ++
    [word selectedShape.carriedEvaluationCount] ++
    (canonicalCarriedCoordinates selectedShape).flatMap fun coordinate =>
      carriedFields (input.claimedCoefficient coordinate)

def roundFields (roundIndex : Nat) (message : Round) :
    List LinComb :=
  proverMessageFields (3 + 2 * roundIndex) (1 + roundIndex) 45
    (message.coefficients.flatMap carriedFields)

def fullOutputPayload (input : Input) : List LinComb :=
  (canonicalFinIndices selectedShape.sourceCount).flatMap fun source =>
    (canonicalFinIndices selectedShape.matrixCount).flatMap fun matrix =>
      (canonicalFinIndices selectedShape.coefficientCount).flatMap fun coefficient =>
        carriedFields (input.fullOutput source matrix coefficient)

def fullOutputFields (input : Input) : List LinComb :=
  proverMessageFields 53 26 47 (fullOutputPayload input)

def initialLanes (statementId : ProductPoseidon2.StatementId) :
    Poseidon2Core.State :=
  fun lane => word ((ProductPoseidon2.initialStateForStatement statementId).lanes lane)

def initialBuilder (input : Input) : SymbolicDuplex.Builder :=
  SymbolicDuplex.start (initialLanes input.statementId)
    (ProductPoseidon2.initialStateForStatement input.statementId).absorbed

def absorbPublicInput (input : Input) : SymbolicDuplex.Builder :=
  SymbolicDuplex.absorbMany input.transcriptBase input.publicNifsFields
    (initialBuilder input)

def absorbStatement (input : Input) : SymbolicDuplex.Builder :=
  SymbolicDuplex.absorbMany input.transcriptBase (statementFields input)
    (absorbPublicInput input)

def squeezeVerifierChallenge
    (eventIndex challengeIndex challengeType : Nat)
    (coordinates : List Nat) (input : Input)
    (builder : SymbolicDuplex.Builder) : Carried × SymbolicDuplex.Builder :=
  SymbolicDuplex.squeezeK input.transcriptBase
    (SymbolicDuplex.absorbMany input.transcriptBase
      (verifierChallengeFields eventIndex challengeIndex challengeType
        coordinates) builder)

def deriveAlphaGo (input : Input) :
    Nat -> Nat -> SymbolicDuplex.Builder -> List Carried × SymbolicDuplex.Builder
  | _, 0, builder => ([], builder)
  | index, remaining + 1, builder =>
      let sampled := squeezeVerifierChallenge 1 1 42 [index] input builder
      let tail := deriveAlphaGo input (index + 1) remaining sampled.2
      (sampled.1 :: tail.1, tail.2)

def deriveAlpha (input : Input) : List Carried × SymbolicDuplex.Builder :=
  deriveAlphaGo input 0 selectedShape.cubeVariables (absorbStatement input)

theorem deriveAlphaGo_length (input : Input) (index count : Nat)
    (builder : SymbolicDuplex.Builder) :
    (deriveAlphaGo input index count builder).1.length = count := by
  induction count generalizing index builder with
  | zero => rfl
  | succ remaining inductionHypothesis =>
      simp only [deriveAlphaGo, List.length_cons]
      rw [inductionHypothesis]

@[simp] theorem deriveAlpha_length (input : Input) :
    (deriveAlpha input).1.length = selectedShape.cubeVariables := by
  exact deriveAlphaGo_length input 0 selectedShape.cubeVariables _

def deriveGamma (input : Input) : Carried × SymbolicDuplex.Builder :=
  squeezeVerifierChallenge 2 2 43 [] input (deriveAlpha input).2

structure RoundReplay where
  challenges : List Carried
  builder : SymbolicDuplex.Builder

def replayRoundsGo (input : Input) :
    List Round -> Nat -> SymbolicDuplex.Builder -> RoundReplay
  | [], _, builder => { challenges := [], builder }
  | round :: rest, index, builder =>
      let absorbed := SymbolicDuplex.absorbMany input.transcriptBase
        (roundFields index round) builder
      let sampled := squeezeVerifierChallenge
        (4 + 2 * index) (3 + index) 46 [] input absorbed
      let tail := replayRoundsGo input rest (index + 1) sampled.2
      { challenges := sampled.1 :: tail.challenges
        builder := tail.builder }

def replayRounds (input : Input) : RoundReplay :=
  replayRoundsGo input (List.ofFn input.rounds) 0 (deriveGamma input).2

theorem replayRoundsGo_length (input : Input) (rounds : List Round)
    (index : Nat) (builder : SymbolicDuplex.Builder) :
    (replayRoundsGo input rounds index builder).challenges.length =
      rounds.length := by
  induction rounds generalizing index builder with
  | nil => rfl
  | cons round rest inductionHypothesis =>
      simp only [replayRoundsGo, List.length_cons]
      rw [inductionHypothesis]

@[simp] theorem replayRounds_length (input : Input) :
    (replayRounds input).challenges.length = selectedShape.cubeVariables := by
  rw [replayRounds, replayRoundsGo_length, List.length_ofFn]

def afterFullOutput (input : Input) : SymbolicDuplex.Builder :=
  SymbolicDuplex.absorbMany input.transcriptBase (fullOutputFields input)
    (replayRounds input).builder

def alphaAt (input : Input)
    (index : Fin selectedShape.cubeVariables) : Carried :=
  (deriveAlpha input).1.get
    ⟨index.val, by rw [deriveAlpha_length]; exact index.isLt⟩

def pointAt (input : Input)
    (index : Fin selectedShape.cubeVariables) : Carried :=
  (replayRounds input).challenges.get
    ⟨index.val, by rw [replayRounds_length]; exact index.isLt⟩

def firstMatrix : Fin selectedShape.matrixCount := ⟨0, by decide⟩
def constantCoefficient : Fin selectedShape.coefficientCount :=
  ⟨0, by decide⟩

def projectedFresh (input : Input)
    (source : Fin selectedShape.freshCount)
    (matrix : Fin selectedShape.matrixCount) : Carried :=
  input.fullOutput (freshSourceIndex source) matrix constantCoefficient

def projectedAssignment (input : Input)
    (source : Fin selectedShape.sourceCount) : Carried :=
  input.fullOutput source firstMatrix constantCoefficient

def projectedCarried (input : Input)
    (coordinate : CarriedCoordinate selectedShape) : Carried :=
  input.fullOutput (runningSourceIndex coordinate.running)
    coordinate.matrix coordinate.coefficient

def occurrenceInput (input : Input) :
    KPiCcsOccurrence.Input selectedShape 9 where
  constraintPolynomial := input.constraintPolynomial
  gamma := (deriveGamma input).1
  alpha := alphaAt input
  point := pointAt input
  priorPoint := input.priorPoint
  claimedCoefficient := input.claimedCoefficient
  freshMatrixImage := projectedFresh input
  sourceAssignment := projectedAssignment input
  carriedImage := projectedCarried input
  current := input.current
  terminal := input.terminal
  rounds := List.ofFn input.rounds
  rounds_length := List.length_ofFn
  frameBase := input.transcriptBase +
    (afterFullOutput input).entries.length * SymbolicDuplex.stride

def rows (input : Input) : List Row :=
  SymbolicDuplex.rows input.transcriptBase ProductPoseidon2.constants
      (afterFullOutput input) ++
    KPiCcsOccurrence.rows (occurrenceInput input)

theorem rows_length (input : Input) :
    (rows input).length =
      (afterFullOutput input).entries.length * SymbolicDuplex.stride +
        (KPiCcsOccurrence.rows (occurrenceInput input)).length := by
  simp [rows, SymbolicDuplex.rows_length, SymbolicDuplex.stride]

theorem transcriptRows_satisfied (input : Input) (assignment : Nat -> Nat)
    (satisfied : Satisfies (rows input) assignment) :
    Satisfies
      (SymbolicDuplex.rows input.transcriptBase ProductPoseidon2.constants
        (afterFullOutput input)) assignment := by
  intro row member
  exact satisfied row (List.mem_append_left _ member)

theorem occurrenceRows_satisfied (input : Input) (assignment : Nat -> Nat)
    (satisfied : Satisfies (rows input) assignment) :
    Satisfies (KPiCcsOccurrence.rows (occurrenceInput input)) assignment := by
  intro row member
  exact satisfied row (List.mem_append_right _ member)

/-- The arithmetic part has no independent challenge columns. Its alpha,
gamma, and SumCheck point expressions are direct outputs of the exact
symbolic replay above. -/
theorem arithmetic_rows_sound (input : Input) (assignment : Nat -> Nat)
    (one : assignment 0 = 1)
    (satisfied : Satisfies (rows input) assignment) :
    SumCheck.Finite.FixedPhase.Chain
      ConcreteCarrier.extensionOps.toOps
      ((KPiCcsOccurrence.decodedVerifierInput (occurrenceInput input)
          assignment).initial ConcreteCarrier.extensionOps
        (KPiCcsOccurrence.decodedGamma (occurrenceInput input) assignment))
      (KPiCcsOccurrence.decodedCertificate (occurrenceInput input)
        assignment).rounds
      (KPiCcsOccurrence.decodedPoint (occurrenceInput input)
        assignment).coordinates
      (ProtocolPolynomial.terminalFromMessage ConcreteCarrier.extensionOps
        (KPiCcsOccurrence.decodedVerifierInput (occurrenceInput input)
          assignment)
        (KPiCcsOccurrence.decodedAlpha (occurrenceInput input) assignment)
        (KPiCcsOccurrence.decodedGamma (occurrenceInput input) assignment)
        (KPiCcsOccurrence.decodedPoint (occurrenceInput input) assignment)
        (KPiCcsOccurrence.decodedMessage (occurrenceInput input)
          assignment)) :=
  KPiCcsOccurrence.rows_sound (occurrenceInput input) assignment one
    (occurrenceRows_satisfied input assignment satisfied)

end Nightstream.Implementation.NebulaV2.ProductPiCcsTranscriptRows
