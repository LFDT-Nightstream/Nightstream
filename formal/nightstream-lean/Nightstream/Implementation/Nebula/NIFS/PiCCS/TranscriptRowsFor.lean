import Nightstream.Implementation.Nebula.NIFS.PiCCS.TranscriptRows

/-!
Contract: complete PiCCS transcript and arithmetic rows indexed by the exact
augmented-relation exponent.

The public frame width, alpha vector, SumCheck round family, transcript event
numbers, output event number, and arithmetic occurrence all use the same
`rowVariables`. This module contains the full PiCCS row program. It is not the
public-prefix-only bridge.

Does not own physical placement, typed key refinement, PiRLC, PiDEC,
cryptographic security, generated artifact containment, or Rust refinement.

Assurance tier: exponent-indexed row implementation.

Emits constraints: through `SymbolicDuplex.rows` and `KPiCcsOccurrence.rows`.
-/

set_option autoImplicit false
set_option maxRecDepth 30000

namespace Nightstream.Implementation.Nebula.ProductPiCcsTranscriptRowsFor

open Nightstream.Implementation.Nebula
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.UnifiedSources

abbrev Shape (rowVariables : Nat) := ProductNifsCodec.shapeFor rowVariables
abbrev Carried := KMul.Carried
abbrev Round := KFixedPhaseSumCheck.Round 9
abbrev LinComb := LinCombNormal.LinComb

def word := ProductPiCcsTranscriptRows.word
def carriedFields := ProductPiCcsTranscriptRows.carriedFields
def constantKFields := ProductPiCcsTranscriptRows.constantKFields
def shapeFields := ProductPiCcsTranscriptRows.shapeFields
def monomialFields := @ProductPiCcsTranscriptRows.monomialFields
def polynomialFields := @ProductPiCcsTranscriptRows.polynomialFields
def pointFields := @ProductPiCcsTranscriptRows.pointFields
def verifierChallengeFields :=
  ProductPiCcsTranscriptRows.verifierChallengeFields
def proverMessageFields := ProductPiCcsTranscriptRows.proverMessageFields

def publicFieldCount (rowVariables : Nat) : Nat :=
  17 + ProductNifsCodec.runningFieldCountFor rowVariables + 3888 + 540

/-- Row-visible values for one PiCCS execution at the selected exponent. -/
structure Input (rowVariables : Nat) where
  statementId : ProductPoseidon2.StatementId
  constraintPolynomial :
    CCSResidualTable.ConstraintPolynomial K (Shape rowVariables).matrixCount
  publicNifsFields : List LinComb
  publicNifsFields_length :
    publicNifsFields.length = publicFieldCount rowVariables
  priorPoint : Fin (Shape rowVariables).cubeVariables -> Carried
  claimedCoefficient : CarriedCoordinate (Shape rowVariables) -> Carried
  rounds : Fin (Shape rowVariables).cubeVariables -> Round
  fullOutput : Fin (Shape rowVariables).sourceCount ->
    Fin (Shape rowVariables).matrixCount ->
    Fin (Shape rowVariables).coefficientCount -> Carried
  current : Carried
  terminal : Carried
  transcriptBase : Nat

def statementFields {rowVariables : Nat}
    (input : Input rowVariables) : List LinComb :=
  [word 41] ++ shapeFields (Shape rowVariables) ++
    polynomialFields input.constraintPolynomial ++
    [word (Shape rowVariables).cubeVariables] ++ pointFields input.priorPoint ++
    [word (Shape rowVariables).carriedEvaluationCount] ++
    (canonicalCarriedCoordinates (Shape rowVariables)).flatMap fun coordinate =>
      carriedFields (input.claimedCoefficient coordinate)

def roundFields (roundIndex : Nat) (message : Round) : List LinComb :=
  proverMessageFields (3 + 2 * roundIndex) (1 + roundIndex) 45
    (message.coefficients.flatMap carriedFields)

def fullOutputPayload {rowVariables : Nat}
    (input : Input rowVariables) : List LinComb :=
  (canonicalFinIndices (Shape rowVariables).sourceCount).flatMap fun source =>
    (canonicalFinIndices (Shape rowVariables).matrixCount).flatMap fun matrix =>
      (canonicalFinIndices (Shape rowVariables).coefficientCount).flatMap
        fun coefficient => carriedFields (input.fullOutput source matrix coefficient)

def fullOutputFields {rowVariables : Nat}
    (input : Input rowVariables) : List LinComb :=
  proverMessageFields (3 + 2 * rowVariables) (1 + rowVariables) 47
    (fullOutputPayload input)

def initialLanes (statementId : ProductPoseidon2.StatementId) :
    Poseidon2Core.State :=
  fun lane => word
    ((ProductPoseidon2.initialStateForStatement statementId).lanes lane)

def initialBuilder {rowVariables : Nat}
    (input : Input rowVariables) : SymbolicDuplex.Builder :=
  SymbolicDuplex.start (initialLanes input.statementId)
    (ProductPoseidon2.initialStateForStatement input.statementId).absorbed

def absorbPublicInput {rowVariables : Nat}
    (input : Input rowVariables) : SymbolicDuplex.Builder :=
  SymbolicDuplex.absorbMany input.transcriptBase input.publicNifsFields
    (initialBuilder input)

def absorbStatement {rowVariables : Nat}
    (input : Input rowVariables) : SymbolicDuplex.Builder :=
  SymbolicDuplex.absorbMany input.transcriptBase (statementFields input)
    (absorbPublicInput input)

def squeezeVerifierChallenge {rowVariables : Nat}
    (eventIndex challengeIndex challengeType : Nat)
    (coordinates : List Nat) (input : Input rowVariables)
    (builder : SymbolicDuplex.Builder) : Carried × SymbolicDuplex.Builder :=
  SymbolicDuplex.squeezeK input.transcriptBase
    (SymbolicDuplex.absorbMany input.transcriptBase
      (verifierChallengeFields eventIndex challengeIndex challengeType
        coordinates) builder)

def deriveAlphaGo {rowVariables : Nat} (input : Input rowVariables) :
    Nat -> Nat -> SymbolicDuplex.Builder -> List Carried × SymbolicDuplex.Builder
  | _, 0, builder => ([], builder)
  | index, remaining + 1, builder =>
      let sampled := squeezeVerifierChallenge 1 1 42 [index] input builder
      let tail := deriveAlphaGo input (index + 1) remaining sampled.2
      (sampled.1 :: tail.1, tail.2)

def deriveAlpha {rowVariables : Nat}
    (input : Input rowVariables) : List Carried × SymbolicDuplex.Builder :=
  deriveAlphaGo input 0 (Shape rowVariables).cubeVariables
    (absorbStatement input)

theorem deriveAlphaGo_length {rowVariables : Nat}
    (input : Input rowVariables) (index count : Nat)
    (builder : SymbolicDuplex.Builder) :
    (deriveAlphaGo input index count builder).1.length = count := by
  induction count generalizing index builder with
  | zero => rfl
  | succ remaining inductionHypothesis =>
      simp only [deriveAlphaGo, List.length_cons]
      rw [inductionHypothesis]

@[simp] theorem deriveAlpha_length {rowVariables : Nat}
    (input : Input rowVariables) :
    (deriveAlpha input).1.length = (Shape rowVariables).cubeVariables := by
  exact deriveAlphaGo_length input 0 (Shape rowVariables).cubeVariables _

def deriveGamma {rowVariables : Nat}
    (input : Input rowVariables) : Carried × SymbolicDuplex.Builder :=
  squeezeVerifierChallenge 2 2 43 [] input (deriveAlpha input).2

structure RoundReplay where
  challenges : List Carried
  builder : SymbolicDuplex.Builder

def replayRoundsGo {rowVariables : Nat} (input : Input rowVariables) :
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

def replayRounds {rowVariables : Nat}
    (input : Input rowVariables) : RoundReplay :=
  replayRoundsGo input (List.ofFn input.rounds) 0 (deriveGamma input).2

theorem replayRoundsGo_length {rowVariables : Nat}
    (input : Input rowVariables) (rounds : List Round)
    (index : Nat) (builder : SymbolicDuplex.Builder) :
    (replayRoundsGo input rounds index builder).challenges.length =
      rounds.length := by
  induction rounds generalizing index builder with
  | nil => rfl
  | cons round rest inductionHypothesis =>
      simp only [replayRoundsGo, List.length_cons]
      rw [inductionHypothesis]

@[simp] theorem replayRounds_length {rowVariables : Nat}
    (input : Input rowVariables) :
    (replayRounds input).challenges.length =
      (Shape rowVariables).cubeVariables := by
  rw [replayRounds, replayRoundsGo_length, List.length_ofFn]

def afterFullOutput {rowVariables : Nat}
    (input : Input rowVariables) : SymbolicDuplex.Builder :=
  SymbolicDuplex.absorbMany input.transcriptBase (fullOutputFields input)
    (replayRounds input).builder

def alphaAt {rowVariables : Nat} (input : Input rowVariables)
    (index : Fin (Shape rowVariables).cubeVariables) : Carried :=
  (deriveAlpha input).1.get
    ⟨index.val, by rw [deriveAlpha_length]; exact index.isLt⟩

def pointAt {rowVariables : Nat} (input : Input rowVariables)
    (index : Fin (Shape rowVariables).cubeVariables) : Carried :=
  (replayRounds input).challenges.get
    ⟨index.val, by rw [replayRounds_length]; exact index.isLt⟩

def firstMatrix (rowVariables : Nat) :
    Fin (Shape rowVariables).matrixCount :=
  ⟨0, by simp [Shape, ProductNifsCodec.shapeFor]⟩

def constantCoefficient (rowVariables : Nat) :
    Fin (Shape rowVariables).coefficientCount :=
  ⟨0, by norm_num [Shape, ProductNifsCodec.shapeFor, ringDegree]⟩

def projectedFresh {rowVariables : Nat} (input : Input rowVariables)
    (source : Fin (Shape rowVariables).freshCount)
    (matrix : Fin (Shape rowVariables).matrixCount) : Carried :=
  input.fullOutput (freshSourceIndex source) matrix
    (constantCoefficient rowVariables)

def projectedAssignment {rowVariables : Nat} (input : Input rowVariables)
    (source : Fin (Shape rowVariables).sourceCount) : Carried :=
  input.fullOutput source (firstMatrix rowVariables)
    (constantCoefficient rowVariables)

def projectedCarried {rowVariables : Nat} (input : Input rowVariables)
    (coordinate : CarriedCoordinate (Shape rowVariables)) : Carried :=
  input.fullOutput (runningSourceIndex coordinate.running)
    coordinate.matrix coordinate.coefficient

def occurrenceInput {rowVariables : Nat} (input : Input rowVariables) :
    KPiCcsOccurrence.Input (Shape rowVariables) 9 where
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

def rows {rowVariables : Nat} (input : Input rowVariables) : List Row :=
  SymbolicDuplex.rows input.transcriptBase ProductPoseidon2.constants
      (afterFullOutput input) ++
    KPiCcsOccurrence.rows (occurrenceInput input)

theorem rows_length {rowVariables : Nat} (input : Input rowVariables) :
    (rows input).length =
      (afterFullOutput input).entries.length * SymbolicDuplex.stride +
        (KPiCcsOccurrence.rows (occurrenceInput input)).length := by
  simp [rows, SymbolicDuplex.rows_length, SymbolicDuplex.stride]

theorem transcriptRows_satisfied {rowVariables : Nat}
    (input : Input rowVariables) (assignment : Nat -> Nat)
    (satisfied : Satisfies (rows input) assignment) :
    Satisfies
      (SymbolicDuplex.rows input.transcriptBase ProductPoseidon2.constants
        (afterFullOutput input)) assignment := by
  intro row member
  exact satisfied row (List.mem_append_left _ member)

theorem occurrenceRows_satisfied {rowVariables : Nat}
    (input : Input rowVariables) (assignment : Nat -> Nat)
    (satisfied : Satisfies (rows input) assignment) :
    Satisfies (KPiCcsOccurrence.rows (occurrenceInput input)) assignment := by
  intro row member
  exact satisfied row (List.mem_append_right _ member)

/-- Arithmetic row soundness at the exact exponent. Every challenge expression
is an output of the transcript program above. -/
theorem arithmetic_rows_sound {rowVariables : Nat}
    (input : Input rowVariables) (assignment : Nat -> Nat)
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

end Nightstream.Implementation.Nebula.ProductPiCcsTranscriptRowsFor
