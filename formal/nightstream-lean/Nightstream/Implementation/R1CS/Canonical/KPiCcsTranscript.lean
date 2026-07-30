import Nightstream.Implementation.R1CS.Canonical.KPiCcsOccurrence
import Nightstream.Implementation.R1CS.Canonical.SymbolicDuplex

/-!
Contract: the concrete Lean-owned transcript occurrence for paper-joint PiCCS.

The paper fixes the causal schedule but leaves the random oracle abstract.  This
module selects the width-8 Poseidon2 duplex already used by the canonical
constraint layer and fixes one typed serialization:

* the complete verifier input is absorbed before any challenge;
* every alpha coordinate and gamma has a distinct typed label;
* every fixed-width round message is absorbed before its indexed challenge;
* the complete output message is absorbed after the final round.

The transcript outputs are used directly as the `alpha`, `gamma`, and
SumCheck-point expressions of `KPiCcsOccurrence`.  A certificate therefore has
no independent challenge columns that could drift from replay.

The numeric tags below are local encoding choices, not paper claims.  They are
compiled as coefficients on the constant-one wire and are part of the
Lean-owned program.  No Rust row or digest defines them.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Canonical.KPiCcsTranscript

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal
open Nightstream.Implementation.R1CS.Canonical.KMul

abbrev ConcreteK := Nightstream.SuperNeo.Concrete.K

/-! ## Typed serialization -/

/-- A verifier constant as a row-free base-field expression. -/
def word (value : Nat) : LinComb := [(0, value % goldilocksP)]

/-- Quadratic-extension limbs in canonical `(c0,c1)` order. -/
def carriedFields (value : Carried) : List LinComb :=
  [value.low, value.high]

/-- A setup constant in canonical `(c0,c1)` order. -/
def constantKFields (value : ConcreteK) : List LinComb :=
  [word value.c0.val, word value.c1.val]

def shapeFields (shape : Shape) : List LinComb :=
  [ word shape.cubeVariables,
    word shape.freshCount,
    word shape.runningCount,
    word shape.matrixCount,
    word shape.coefficientCount ]

def monomialFields
    {shape : Shape}
    (monomial : CCSResidualTable.Monomial ConcreteK shape.matrixCount) :
    List LinComb :=
  constantKFields monomial.coefficient ++
    (canonicalFinIndices shape.matrixCount).map fun index =>
      word (monomial.exponents index)

def polynomialFields
    {shape : Shape}
    (polynomial :
      CCSResidualTable.ConstraintPolynomial ConcreteK shape.matrixCount) :
    List LinComb :=
  word polynomial.degreeBound :: word polynomial.terms.length ::
    polynomial.terms.flatMap monomialFields

/-- Inputs that remain authoritative after challenge fields are removed. -/
structure Input (shape : Shape) (degree : Nat) where
  priorLanes : Poseidon2Core.State
  constraintPolynomial :
    CCSResidualTable.ConstraintPolynomial ConcreteK shape.matrixCount
  priorPoint : Fin shape.cubeVariables → Carried
  claimedCoefficient : CarriedCoordinate shape → Carried
  freshMatrixImage :
    Fin shape.freshCount → Fin shape.matrixCount → Carried
  sourceAssignment : Fin shape.sourceCount → Carried
  carriedImage : CarriedCoordinate shape → Carried
  current : Carried
  terminal : Carried
  rounds : List (KFixedPhaseSumCheck.Round degree)
  rounds_length : rounds.length = shape.cubeVariables
  transcriptBase : Nat

/-- Exact complete public statement serialization.  The sparse polynomial is
setup data, but including its syntax makes the chosen oracle instance explicit
rather than relying on an unnamed profile. -/
def statementFields
    {shape : Shape} {degree : Nat} (input : Input shape degree) :
    List LinComb :=
  [word 41] ++ shapeFields shape ++
    polynomialFields input.constraintPolynomial ++
    [word shape.cubeVariables] ++
    ((KPointEquality.indices shape.cubeVariables).flatMap fun index =>
      carriedFields (input.priorPoint index)) ++
    [word shape.carriedEvaluationCount] ++
    ((canonicalCarriedCoordinates shape).flatMap fun coordinate =>
      carriedFields (input.claimedCoefficient coordinate))

def roundFields
    {degree : Nat} (roundIndex : Nat)
    (round : KFixedPhaseSumCheck.Round degree) : List LinComb :=
  [word 45, word roundIndex, word round.coefficients.length] ++
    round.coefficients.flatMap carriedFields

def outputFields
    {shape : Shape} {degree : Nat} (input : Input shape degree) :
    List LinComb :=
  [word 47] ++
    ((canonicalFinIndices shape.freshCount).flatMap fun source =>
      (canonicalFinIndices shape.matrixCount).flatMap fun matrix =>
        carriedFields (input.freshMatrixImage source matrix)) ++
    ((canonicalFinIndices shape.sourceCount).flatMap fun source =>
      carriedFields (input.sourceAssignment source)) ++
    ((canonicalCarriedCoordinates shape).flatMap fun coordinate =>
      carriedFields (input.carriedImage coordinate))

/-! ## Causal replay -/

def squeezeLabel (base label : Nat) (builder : SymbolicDuplex.Builder) :
    Carried × SymbolicDuplex.Builder :=
  SymbolicDuplex.squeezeK base
    (SymbolicDuplex.absorb base (word label) builder)

def squeezeIndexedGo (base label index : Nat) :
    Nat → SymbolicDuplex.Builder → List Carried × SymbolicDuplex.Builder
  | 0, state => ([], state)
  | remaining + 1, state =>
      let tagged := SymbolicDuplex.absorbMany base
        [word label, word index] state
      let sampled := SymbolicDuplex.squeezeK base tagged
      let tail := squeezeIndexedGo base label (index + 1) remaining sampled.2
      (sampled.1 :: tail.1, tail.2)

def squeezeIndexed (base label count : Nat)
    (builder : SymbolicDuplex.Builder) :
    List Carried × SymbolicDuplex.Builder :=
  squeezeIndexedGo base label 0 count builder

theorem squeezeIndexed_length (base label count : Nat)
    (builder : SymbolicDuplex.Builder) :
    (squeezeIndexed base label count builder).1.length = count := by
  unfold squeezeIndexed
  generalize indexEq : 0 = index
  clear indexEq
  induction count generalizing index builder with
  | zero => rfl
  | succ count inductionHypothesis =>
      simp only [squeezeIndexedGo, List.length_cons]
      rw [inductionHypothesis]

structure PreSumcheck (shape : Shape) where
  alpha : List Carried
  alpha_length : alpha.length = shape.cubeVariables
  gamma : Carried
  builder : SymbolicDuplex.Builder

def derivePreSumcheck
    {shape : Shape} {degree : Nat} (input : Input shape degree) :
    PreSumcheck shape :=
  let initial := SymbolicDuplex.start input.priorLanes 0
  let statement :=
    SymbolicDuplex.absorbMany input.transcriptBase
      (statementFields input) initial
  let alpha :=
    squeezeIndexed input.transcriptBase 42 shape.cubeVariables statement
  let gamma := squeezeLabel input.transcriptBase 43 alpha.2
  { alpha := alpha.1
    alpha_length := squeezeIndexed_length _ _ _ _
    gamma := gamma.1
    builder := gamma.2 }

structure RoundReplay where
  challenges : List Carried
  builder : SymbolicDuplex.Builder

def replayRounds
    {degree : Nat} (base : Nat) :
    List (KFixedPhaseSumCheck.Round degree) → Nat →
      SymbolicDuplex.Builder → RoundReplay
  | [], _, builder => { challenges := [], builder }
  | round :: rest, index, builder =>
      let absorbed :=
        SymbolicDuplex.absorbMany base (roundFields index round) builder
      let sampled := squeezeIndexedGo base 46 index 1 absorbed
      let tail := replayRounds base rest (index + 1) sampled.2
      { challenges := sampled.1 ++ tail.challenges
        builder := tail.builder }

theorem replayRounds_challenges_length
    {degree : Nat} (base : Nat) :
    ∀ (rounds : List (KFixedPhaseSumCheck.Round degree)) index builder,
      (replayRounds base rounds index builder).challenges.length =
        rounds.length
  | [], _, _ => rfl
  | round :: rest, index, builder => by
      simp only [replayRounds, List.length_append,
        squeezeIndexed_length,
        replayRounds_challenges_length base rest (index + 1)]
      exact Nat.add_comm 1 rest.length

structure Replay (shape : Shape) where
  alpha : List Carried
  alpha_length : alpha.length = shape.cubeVariables
  gamma : Carried
  point : List Carried
  point_length : point.length = shape.cubeVariables
  beforeOutput : SymbolicDuplex.Builder
  afterOutput : SymbolicDuplex.Builder

def replay
    {shape : Shape} {degree : Nat} (input : Input shape degree) :
    Replay shape :=
  let pre := derivePreSumcheck input
  let rounds := replayRounds input.transcriptBase input.rounds 0 pre.builder
  let outgoing :=
    SymbolicDuplex.absorbMany input.transcriptBase
      (outputFields input) rounds.builder
  { alpha := pre.alpha
    alpha_length := pre.alpha_length
    gamma := pre.gamma
    point := rounds.challenges
    point_length := by
      rw [replayRounds_challenges_length, input.rounds_length]
    beforeOutput := rounds.builder
    afterOutput := outgoing }

def alphaAt
    {shape : Shape} (execution : Replay shape)
    (index : Fin shape.cubeVariables) : Carried :=
  execution.alpha.get
    ⟨index.val, by rw [execution.alpha_length]; exact index.isLt⟩

def pointAt
    {shape : Shape} (execution : Replay shape)
    (index : Fin shape.cubeVariables) : Carried :=
  execution.point.get
    ⟨index.val, by rw [execution.point_length]; exact index.isLt⟩

/-- The PiCCS arithmetic occurrence receives no free challenge expressions:
all three challenge families are exact projections of this replay. -/
def occurrenceInput
    {shape : Shape} {degree : Nat} (input : Input shape degree) :
    KPiCcsOccurrence.Input shape degree :=
  let execution := replay input
  { constraintPolynomial := input.constraintPolynomial
    gamma := execution.gamma
    alpha := alphaAt execution
    point := pointAt execution
    priorPoint := input.priorPoint
    claimedCoefficient := input.claimedCoefficient
    freshMatrixImage := input.freshMatrixImage
    sourceAssignment := input.sourceAssignment
    carriedImage := input.carriedImage
    current := input.current
    terminal := input.terminal
    rounds := input.rounds
    rounds_length := input.rounds_length
    frameBase :=
      input.transcriptBase +
        (replay input).afterOutput.entries.length * SymbolicDuplex.stride }

def rows
    {shape : Shape} {degree : Nat}
    (constants : Poseidon2Schedule.Constants) (input : Input shape degree) :
    List Row :=
  SymbolicDuplex.rows input.transcriptBase constants
      (replay input).afterOutput ++
    KPiCcsOccurrence.rows (occurrenceInput input)

theorem rows_length
    {shape : Shape} {degree : Nat}
    (constants : Poseidon2Schedule.Constants) (input : Input shape degree) :
    (rows constants input).length =
      (replay input).afterOutput.entries.length * 352 +
        ((3 * (shape.jointCoefficientCount - 1) + 2)
          + (shape.cubeVariables * (3 * degree + 2) + 2)
          + (KPiCcsTerminal.sparseRowsPerSource
                (KPiCcsOccurrence.terminalInput (occurrenceInput input))
                * shape.freshCount
              + 6 * shape.sourceCount
              + 2 * KPiCcsTerminal.pointEqualityRows shape.cubeVariables
              + 3 * (shape.freshCount + shape.sourceCount - 1)
              + 3 * (shape.jointCoefficientCount - 1)
              + 8)) := by
  unfold rows
  rw [List.length_append, SymbolicDuplex.rows_length,
    KPiCcsOccurrence.rows_length]

theorem occurrence_satisfied
    {shape : Shape} {degree : Nat}
    (constants : Poseidon2Schedule.Constants) (input : Input shape degree)
    (assignment : Nat → Nat)
    (satisfied : Satisfies (rows constants input) assignment) :
    Satisfies (KPiCcsOccurrence.rows (occurrenceInput input)) assignment :=
  fun row member => satisfied row (List.mem_append_right _ member)

/-- The combined transcript/arithmetic rows reach the unchanged paper
fixed-phase chain with transcript-derived challenge expressions. -/
theorem rows_sound
    {shape : Shape} {degree : Nat}
    (constants : Poseidon2Schedule.Constants) (input : Input shape degree)
    (assignment : Nat → Nat)
    (constantWire : assignment 0 = 1)
    (satisfied : Satisfies (rows constants input) assignment) :
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
  KPiCcsOccurrence.rows_sound (occurrenceInput input) assignment constantWire
    (occurrence_satisfied constants input assignment satisfied)

end Nightstream.Implementation.R1CS.Canonical.KPiCcsTranscript
