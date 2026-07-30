import Nightstream.Implementation.R1CS.Canonical.KPiCcsInitial
import Nightstream.Implementation.R1CS.Canonical.KPiCcsTerminal
import Nightstream.Implementation.R1CS.Canonical.KConcreteFixedPhaseBridge
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ProtocolPolynomialFixedWidth

/-!
Contract: one Lean-owned fixed-width `Pi_CCS` occurrence.

The occurrence shares one set of verifier inputs between:

* the verifier-computed shifted initial claim;
* the ghost-free fixed-phase SumCheck chain; and
* the exact message-derived paper terminal.

The SumCheck challenges are the terminal point coordinates themselves.  There
is therefore no point/challenge alignment premise, and neither endpoint is a
caller-supplied semantic conclusion.  This module owns the canonical row list,
its exact length, and deterministic reduction to the unchanged paper
fixed-width verifier boundary.

Transcript generation, physical placement, honest witnesses, and probability
bounds are separate boundaries.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Canonical.KPiCcsOccurrence

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.SumCheck.Finite
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal
open Nightstream.Implementation.R1CS.Canonical.KMul
open Nightstream.Implementation.R1CS.Canonical.KFixedPhaseSumCheck
open Nightstream.Implementation.R1CS.Canonical.KConcreteFixedPhaseBridge

abbrev ConcreteK := Nightstream.SuperNeo.Concrete.K

/-- The complete row-visible data for one fixed-width `Pi_CCS` occurrence.

All three phases are derived from these fields.  In particular there are no
independent initial-input, chain, and terminal-input records that could drift
apart. -/
structure Input (shape : Shape) (degree : Nat) where
  constraintPolynomial :
    CCSResidualTable.ConstraintPolynomial ConcreteK shape.matrixCount
  gamma : Carried
  alpha : Fin shape.cubeVariables → Carried
  point : Fin shape.cubeVariables → Carried
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
  frameBase : Nat

/-- The exact challenge list is the selected terminal point, in canonical
coordinate order. -/
def challenges
    {shape : Shape} {degree : Nat} (input : Input shape degree) :
    List Carried :=
  (KPointEquality.indices shape.cubeVariables).map input.point

theorem challenges_length
    {shape : Shape} {degree : Nat} (input : Input shape degree) :
    (challenges input).length = shape.cubeVariables := by
  unfold challenges
  rw [List.length_map, KPointEquality.indices_length]

def initialBase
    {shape : Shape} {degree : Nat} (input : Input shape degree) : Nat :=
  input.frameBase

def chainBase
    {shape : Shape} {degree : Nat} (input : Input shape degree) : Nat :=
  initialBase input + 3 * (shape.jointCoefficientCount - 1)

def terminalBase
    {shape : Shape} {degree : Nat} (input : Input shape degree) : Nat :=
  chainBase input + 3 * degree * input.rounds.length

def initialInput
    {shape : Shape} {degree : Nat} (input : Input shape degree) :
    KPiCcsInitial.Input shape where
  gamma := input.gamma
  claimedCoefficient := input.claimedCoefficient
  initial := input.current
  frameBase := initialBase input

def terminalInput
    {shape : Shape} {degree : Nat} (input : Input shape degree) :
    KPiCcsTerminal.Input shape where
  constraintPolynomial := input.constraintPolynomial
  gamma := input.gamma
  alpha := input.alpha
  point := input.point
  priorPoint := input.priorPoint
  claimedCoefficient := input.claimedCoefficient
  freshMatrixImage := input.freshMatrixImage
  sourceAssignment := input.sourceAssignment
  carriedImage := input.carriedImage
  terminal := input.terminal
  frameBase := terminalBase input

/-- The emitted order is initial binding, claimed SumCheck chain, then the
message-terminal computation. -/
def rows
    {shape : Shape} {degree : Nat} (input : Input shape degree) : List Row :=
  KPiCcsInitial.rows (initialInput input) ++
    KFixedPhaseSumCheck.chainRows input.current input.rounds
      (challenges input) input.terminal (chainBase input) ++
    KPiCcsTerminal.rows (terminalInput input)

theorem rows_length
    {shape : Shape} {degree : Nat} (input : Input shape degree) :
    (rows input).length =
      (3 * (shape.jointCoefficientCount - 1) + 2)
        + (shape.cubeVariables * (3 * degree + 2) + 2)
        + (KPiCcsTerminal.sparseRowsPerSource (terminalInput input)
              * shape.freshCount
            + 6 * shape.sourceCount
            + 2 * KPiCcsTerminal.pointEqualityRows shape.cubeVariables
            + 3 * (shape.freshCount + shape.sourceCount - 1)
            + 3 * (shape.jointCoefficientCount - 1)
            + 8) := by
  unfold rows
  rw [List.length_append, List.length_append,
    KPiCcsInitial.rows_length,
    KFixedPhaseSumCheck.chainRows_length _ _ _ _ _
      (input.rounds_length.trans (challenges_length input).symm),
    KPiCcsTerminal.rows_length]
  rw [input.rounds_length]

private theorem satisfies_first
    {left middle right : List Row} {assignment : Nat → Nat}
    (satisfied : Satisfies (left ++ middle ++ right) assignment) :
    Satisfies left assignment := by
  intro row member
  exact satisfied row
    (List.mem_append_left right (List.mem_append_left middle member))

private theorem satisfies_middle
    {left middle right : List Row} {assignment : Nat → Nat}
    (satisfied : Satisfies (left ++ middle ++ right) assignment) :
    Satisfies middle assignment := by
  intro row member
  exact satisfied row
    (List.mem_append_left right (List.mem_append_right left member))

private theorem satisfies_last
    {left middle right : List Row} {assignment : Nat → Nat}
    (satisfied : Satisfies (left ++ middle ++ right) assignment) :
    Satisfies right assignment := by
  intro row member
  exact satisfied row
    (List.mem_append_right (left ++ middle) member)

/-! ## Authoritative decoding -/

def decodedRound
    {degree : Nat} (round : KFixedPhaseSumCheck.Round degree)
    (assignment : Nat → Nat) :
    FixedPolynomial ConcreteK degree where
  coefficients :=
    round.coefficients.map fun coefficient =>
      ofProjection (decodeCarried assignment coefficient)
  coefficients_length := by
    rw [List.length_map, round.coefficients_length]

def decodedCertificate
    {shape : Shape} {degree : Nat} (input : Input shape degree)
    (assignment : Nat → Nat) :
    FixedPhase.Certificate ConcreteK degree where
  rounds := input.rounds.map fun round => decodedRound round assignment

def decodedPoint
    {shape : Shape} {degree : Nat} (input : Input shape degree)
    (assignment : Nat → Nat) :
    CubePoint ConcreteK shape.cubeVariables :=
  KPiCcsTerminal.decodedPoint (terminalInput input) assignment

def decodedGamma
    {shape : Shape} {degree : Nat} (input : Input shape degree)
    (assignment : Nat → Nat) : ConcreteK :=
  KPiCcsTerminal.decoded assignment input.gamma

def decodedVerifierInput
    {shape : Shape} {degree : Nat} (input : Input shape degree)
    (assignment : Nat → Nat) :
    ProtocolPolynomial.VerifierInput ConcreteK shape :=
  KPiCcsTerminal.decodedInput (terminalInput input) assignment

def decodedMessage
    {shape : Shape} {degree : Nat} (input : Input shape degree)
    (assignment : Nat → Nat) :
    ProtocolPolynomial.OutputMessage ConcreteK shape :=
  KPiCcsTerminal.decodedMessage (terminalInput input) assignment

def decodedAlpha
    {shape : Shape} {degree : Nat} (input : Input shape degree)
    (assignment : Nat → Nat) :
    CubePoint ConcreteK shape.cubeVariables :=
  KPiCcsTerminal.decodedAlpha (terminalInput input) assignment

private theorem fixedPolynomial_eq_of_coefficients_eq
    {Field : Type} {degree : Nat}
    {left right : FixedPolynomial Field degree}
    (equal : left.coefficients = right.coefficients) :
    left = right := by
  cases left with
  | mk leftCoefficients leftLength =>
      cases right with
      | mk rightCoefficients rightLength =>
          simp only at equal
          subst rightCoefficients
          rfl

private theorem map_decodedRound
    {degree : Nat} (round : KFixedPhaseSumCheck.Round degree)
    (assignment : Nat → Nat) :
    mapPolynomial (decodedRound round assignment) =
      round.polynomial assignment := by
  cases round with
  | mk coefficients coefficientsLength =>
      apply fixedPolynomial_eq_of_coefficients_eq
      simp only [decodedRound, mapPolynomial, Round.polynomial,
        List.map_map]
      apply List.map_congr_left
      intro coefficient _
      exact toProjection_ofProjection _

private theorem map_decodedRounds
    {shape : Shape} {degree : Nat} (input : Input shape degree)
    (assignment : Nat → Nat) :
    (decodedCertificate input assignment).rounds.map mapPolynomial =
      input.rounds.map fun round => round.polynomial assignment := by
  unfold decodedCertificate
  simp only [List.map_map, Function.comp_apply]
  apply List.map_congr_left
  intro round _
  exact map_decodedRound round assignment

private theorem map_decodedPoint
    {shape : Shape} {degree : Nat} (input : Input shape degree)
    (assignment : Nat → Nat) :
    (decodedPoint input assignment).coordinates.map toProjection =
      (challenges input).map (decodeCarried assignment) := by
  unfold decodedPoint KPiCcsTerminal.decodedPoint
    KPiCcsTerminal.alphaEqualityInput KPointEquality.decodedLeft
    KPointEquality.decoded challenges KPointEquality.indices
  simp only [List.map_map]
  apply List.map_congr_left
  intro index _
  exact toProjection_ofProjection _

private theorem decoded_initial_eq
    {shape : Shape} {degree : Nat} (input : Input shape degree)
    (assignment : Nat → Nat)
    (constantWire : assignment 0 = 1)
    (satisfied : Satisfies (rows input) assignment) :
    ofProjection (decodeCarried assignment input.current) =
      (decodedVerifierInput input assignment).initial
        ConcreteCarrier.extensionOps (decodedGamma input assignment) := by
  have initialSatisfied :
      Satisfies (KPiCcsInitial.rows (initialInput input)) assignment := by
    apply satisfies_first
    simpa [rows] using satisfied
  have bound :=
    KPiCcsInitial.rows_sound (initialInput input) assignment
      constantWire initialSatisfied
  simpa [initialInput, decodedVerifierInput, decodedGamma,
    KPiCcsTerminal.decodedInput, KPiCcsTerminal.decoded,
    ProtocolPolynomial.VerifierInput.initial,
    ProtocolPolynomial.VerifierInput.targetCoefficients,
    KPiCcsInitial.decodedInitial, KPiCcsInitial.decodedTarget,
    KPiCcsInitial.decodedGamma] using bound

private theorem decoded_terminal_eq
    {shape : Shape} {degree : Nat} (input : Input shape degree)
    (assignment : Nat → Nat)
    (constantWire : assignment 0 = 1)
    (satisfied : Satisfies (rows input) assignment) :
    ofProjection (decodeCarried assignment input.terminal) =
      ProtocolPolynomial.terminalFromMessage ConcreteCarrier.extensionOps
        (decodedVerifierInput input assignment)
        (decodedAlpha input assignment)
        (decodedGamma input assignment)
        (decodedPoint input assignment)
        (decodedMessage input assignment) := by
  have terminalSatisfied :
      Satisfies (KPiCcsTerminal.rows (terminalInput input)) assignment := by
    apply satisfies_last
    simpa [rows] using satisfied
  simpa [decodedVerifierInput, decodedAlpha, decodedGamma, decodedPoint,
    decodedMessage, terminalInput, KPiCcsTerminal.decoded] using
      KPiCcsTerminal.rows_sound (terminalInput input) assignment
        constantWire terminalSatisfied

/-- **Exact row-to-paper refinement.**

Satisfying the emitted rows reconstructs the unchanged fixed-phase paper
chain with a verifier-derived initial claim and a message-derived terminal.
There is no semantic-chain premise. -/
theorem rows_sound
    {shape : Shape} {degree : Nat} (input : Input shape degree)
    (assignment : Nat → Nat)
    (constantWire : assignment 0 = 1)
    (satisfied : Satisfies (rows input) assignment) :
    FixedPhase.Chain ConcreteCarrier.extensionOps.toOps
      ((decodedVerifierInput input assignment).initial
        ConcreteCarrier.extensionOps (decodedGamma input assignment))
      (decodedCertificate input assignment).rounds
      (decodedPoint input assignment).coordinates
      (ProtocolPolynomial.terminalFromMessage ConcreteCarrier.extensionOps
        (decodedVerifierInput input assignment)
        (decodedAlpha input assignment)
        (decodedGamma input assignment)
        (decodedPoint input assignment)
        (decodedMessage input assignment)) := by
  have chainSatisfied :
      Satisfies
        (KFixedPhaseSumCheck.chainRows input.current input.rounds
          (challenges input) input.terminal (chainBase input)) assignment := by
    apply satisfies_middle
    simpa [rows] using satisfied
  have rowChain :=
    KFixedPhaseSumCheck.chainRows_sound assignment constantWire
      input.current input.rounds (challenges input) input.terminal
      (chainBase input) chainSatisfied
  have projectedChain :
      FixedPhase.Chain sumCheckOps
        (toProjection (ofProjection
          (decodeCarried assignment input.current)))
        ((decodedCertificate input assignment).rounds.map mapPolynomial)
        ((decodedPoint input assignment).coordinates.map toProjection)
        (toProjection (ofProjection
          (decodeCarried assignment input.terminal))) := by
    rw [toProjection_ofProjection, toProjection_ofProjection,
      map_decodedRounds input assignment, map_decodedPoint input assignment]
    exact rowChain
  have paperChain :=
    chain_of_toProjection
      (ofProjection (decodeCarried assignment input.current))
      (ofProjection (decodeCarried assignment input.terminal))
      (decodedCertificate input assignment).rounds
      (decodedPoint input assignment).coordinates projectedChain
  rw [decoded_initial_eq input assignment constantWire satisfied,
    decoded_terminal_eq input assignment constantWire satisfied] at paperChain
  exact paperChain

/-- Deterministic paper reduction for the row occurrence.  The only data
premise is exact source binding between the rich semantic tables and the
authoritative verifier-visible columns.  It is not a generic validity or
source-failure escape. -/
theorem rows_imply_tableTruth_or_badEvent
    {shape : Shape} {degree : Nat} (input : Input shape degree)
    (assignment : Nat → Nat)
    (constantWire : assignment 0 = 1)
    (satisfied : Satisfies (rows input) assignment)
    (data : ProtocolPolynomial.Data ConcreteK shape)
    (sourceBinding :
      data.toVerifierInput = decodedVerifierInput input assignment)
    (degreeCovers :
      data.toVerifierInput.sumcheckDegreeBound ≤ degree)
    (challengeSetSize : Nat) :
    (TableResidualData.toTableObligations ConcreteCarrier.extensionOps
        (SignedCoefficientObject.toTableResidualData
          ConcreteCarrier.extensionOps
          (data.toJointData ConcreteCarrier.extensionOps))).AllHold ∨
      SignedCoefficientObject.MixingRoot ConcreteCarrier.extensionOps
        (data.toJointData ConcreteCarrier.extensionOps)
        (decodedAlpha input assignment) (decodedGamma input assignment) ∨
      ProtocolPolynomial.FixedWidth.SumCheckCollision
        ConcreteCarrier.extensionOps data
        (decodedAlpha input assignment) (decodedGamma input assignment)
        degree challengeSetSize (decodedPoint input assignment)
        (decodedCertificate input assignment) ∨
      ProtocolPolynomial.OutputMismatch ConcreteCarrier.extensionOps data
        (decodedAlpha input assignment) (decodedGamma input assignment)
        (decodedPoint input assignment) (decodedMessage input assignment) := by
  have chain := rows_sound input assignment constantWire satisfied
  rw [← sourceBinding] at chain
  exact
    ProtocolPolynomial.FixedWidth.accepted_implies_tableTruth_or_badEvent
      ConcreteCarrier.extensionOps ConcreteCarrier.extensionLaws
      ConcreteCarrier.extensionZeroLaws data
      (decodedAlpha input assignment) (decodedGamma input assignment)
      degree degreeCovers challengeSetSize
      (decodedPoint input assignment) (decodedMessage input assignment)
      (decodedCertificate input assignment) chain

end Nightstream.Implementation.R1CS.Canonical.KPiCcsOccurrence
