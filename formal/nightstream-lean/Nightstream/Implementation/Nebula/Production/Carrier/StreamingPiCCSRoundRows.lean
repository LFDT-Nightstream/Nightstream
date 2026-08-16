import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiCCSAuthority
import Nightstream.Implementation.R1CS.Canonical.KConcreteFixedPhaseBridge

/-!
Contract: the exact arithmetic rows for one production PiCCS SumCheck round.

Assurance tier: handwritten row soundness.

Owns the fixed 54-column phase layout, the 31-row degree-nine program, its
decode to the concrete paper carrier, and the implication from local row
satisfaction plus control placement to `RoundPhaseRelation`.

The same ten coefficient pairs and one challenge pair drive the incoming
claim equation, the outgoing Horner evaluation, and the authoritative
transcript placement. No intermediate Horner output pair is materialized.

Does not own Poseidon2 permutation rows, a generated artifact, a Rust
assignment, recursive orchestration, terminal integration, or collision
resistance.

Emits constraints: 31 R1CS rows. Two rows check the incoming claim, 27 rows
perform nine Karatsuba extension multiplications, and two rows bind the next
claim. The source layout has 54 columns, including the constant-one column.
-/

set_option autoImplicit false
set_option maxRecDepth 100000
set_option maxHeartbeats 2400000

namespace Nightstream.Implementation.Nebula.ProductionStreamingPiCcsRoundRows

open Nightstream.Implementation.Nebula.ProductionStreamingPiCcsAuthority
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.KConcreteFixedPhaseBridge
open Nightstream.Implementation.R1CS.Canonical.KFixedPhaseSumCheck
open Nightstream.Implementation.R1CS.Canonical.KMul
open Nightstream.Implementation.R1CS.ProjectionProgram
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.SumCheck.Finite

/-- Column starts selected by the phase artifact. Each extension value uses
two consecutive columns. -/
structure Layout where
  currentStart : Nat
  coefficientStart : Nat
  challengeStart : Nat
  nextStart : Nat
  auxiliaryStart : Nat
deriving DecidableEq, Repr

/-- One extension value stored in two consecutive source columns. -/
def carriedAt (start : Nat) : Carried where
  low := [(start, 1)]
  high := [(start + 1, 1)]

def current (layout : Layout) : Carried :=
  carriedAt layout.currentStart

def coefficient (layout : Layout) (index : Fin 10) : Carried :=
  carriedAt (layout.coefficientStart + 2 * index.val)

def coefficients (layout : Layout) : List Carried :=
  List.ofFn (coefficient layout)

@[simp] theorem coefficients_length (layout : Layout) :
    (coefficients layout).length = 10 := by
  simp [coefficients]

def round (layout : Layout) : Round 9 where
  coefficients := coefficients layout
  coefficients_length := by
    rw [coefficients_length]

def challenge (layout : Layout) : Carried :=
  carriedAt layout.challengeStart

def next (layout : Layout) : Carried :=
  carriedAt layout.nextStart

/-- The exact degree-nine one-round row program. -/
def rows (layout : Layout) : List R1CS.Row :=
  chainRows (current layout) [round layout] [challenge layout]
    (next layout) layout.auxiliaryStart

@[simp] theorem rows_length (layout : Layout) :
    (rows layout).length = 31 := by
  unfold rows
  rw [chainRows_length]
  · norm_num
  · rfl

abbrev ConcreteK := Nightstream.SuperNeo.Concrete.K

/-- Decode one row-visible extension value into the concrete paper carrier. -/
def decoded (assignment : Nat → Nat) (value : Carried) : ConcreteK :=
  ofProjection (decodeCarried assignment value)

def decodedCurrent (layout : Layout) (assignment : Nat → Nat) : ConcreteK :=
  decoded assignment (current layout)

def decodedChallenge (layout : Layout) (assignment : Nat → Nat) : ConcreteK :=
  decoded assignment (challenge layout)

def decodedNext (layout : Layout) (assignment : Nat → Nat) : ConcreteK :=
  decoded assignment (next layout)

/-- The exact fixed-width polynomial read from the ten coefficient pairs. -/
def decodedPolynomial (layout : Layout) (assignment : Nat → Nat) :
    FixedPolynomial ConcreteK 9 where
  coefficients := (coefficients layout).map (decoded assignment)
  coefficients_length := by
    rw [List.length_map, coefficients_length]

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

private theorem map_decodedPolynomial
    (layout : Layout) (assignment : Nat → Nat) :
    mapPolynomial (decodedPolynomial layout assignment) =
      (round layout).polynomial assignment := by
  apply fixedPolynomial_eq_of_coefficients_eq
  simp only [decodedPolynomial, mapPolynomial, Round.polynomial,
    List.map_map]
  apply List.map_congr_left
  intro value _
  exact toProjection_ofProjection _

/-- The direct row-carrier equations before the coordinate-preserving paper
carrier map is applied. -/
private theorem rows_imply_projection_round
    (layout : Layout)
    (assignment : Nat → Nat)
    (constantWire : assignment 0 = 1)
    (satisfied : Satisfies (rows layout) assignment) :
    decodeCarried assignment (current layout) =
        Nightstream.Implementation.R1CS.ProjectionProgram.K.add
          (FixedPolynomial.evaluate sumCheckOps
            ((round layout).polynomial assignment)
            Nightstream.Implementation.R1CS.ProjectionProgram.K.zero)
          (FixedPolynomial.evaluate sumCheckOps
            ((round layout).polynomial assignment)
            Nightstream.Implementation.R1CS.ProjectionProgram.K.one) /\
      decodeCarried assignment (next layout) =
        FixedPolynomial.evaluate sumCheckOps
          ((round layout).polynomial assignment)
          (decodeCarried assignment (challenge layout)) := by
  have rowChain := chainRows_sound assignment constantWire
    (current layout) [round layout] [challenge layout]
    (next layout) layout.auxiliaryStart satisfied
  change
    decodeCarried assignment (current layout) =
        Nightstream.Implementation.R1CS.ProjectionProgram.K.add
          (FixedPolynomial.evaluate sumCheckOps
            ((round layout).polynomial assignment)
            Nightstream.Implementation.R1CS.ProjectionProgram.K.zero)
          (FixedPolynomial.evaluate sumCheckOps
            ((round layout).polynomial assignment)
            Nightstream.Implementation.R1CS.ProjectionProgram.K.one) /\
      FixedPhase.Chain sumCheckOps
        (FixedPolynomial.evaluate sumCheckOps
          ((round layout).polynomial assignment)
          (decodeCarried assignment (challenge layout)))
        [] [] (decodeCarried assignment (next layout)) at rowChain
  exact ⟨rowChain.1, rowChain.2.symm⟩

/-- Local row satisfaction reconstructs both equations of one concrete paper
SumCheck round. -/
theorem rows_imply_concrete_round
    (layout : Layout)
    (assignment : Nat → Nat)
    (constantWire : assignment 0 = 1)
    (satisfied : Satisfies (rows layout) assignment) :
    decodedCurrent layout assignment =
        Nightstream.SuperNeo.Concrete.K.add
          ((decodedPolynomial layout assignment).evaluate
            ConcreteCarrier.extensionOps.toOps
            Nightstream.SuperNeo.Concrete.K.zero)
          ((decodedPolynomial layout assignment).evaluate
            ConcreteCarrier.extensionOps.toOps
            Nightstream.SuperNeo.Concrete.K.one) /\
      decodedNext layout assignment =
        (decodedPolynomial layout assignment).evaluate
          ConcreteCarrier.extensionOps.toOps
          (decodedChallenge layout assignment) := by
  have projection := rows_imply_projection_round layout assignment
    constantWire satisfied
  constructor
  · apply toProjection_injective
    rw [toProjection_add, ← evaluate_mapPolynomial,
      ← evaluate_mapPolynomial, map_decodedPolynomial,
      toProjection_zero, toProjection_one]
    simpa [decodedCurrent, decoded] using projection.1
  · apply toProjection_injective
    rw [← evaluate_mapPolynomial, map_decodedPolynomial]
    simpa [decodedNext, decodedChallenge, decoded] using projection.2

/-- Non-arithmetic placement owned by the surrounding phase. These equalities
bind the exact row assignment to the carried verifier state and to the
Poseidon2-derived challenge. The outgoing claim is not assumed; rows derive
it before the complete successor is reconstructed. -/
def ControlPlacement
    {rowVariables : Nat}
    (layout : Layout)
    (assignment : Nat → Nat)
    (roundIndex : Fin (ProductNifsCodec.shapeFor rowVariables).cubeVariables)
    (polynomial : ProductionRound)
    (before after : Continuation ConcreteK BindingState) : Prop :=
  before.cursor = roundIndex.val /\
    decodedCurrent layout assignment = before.current /\
    decodedPolynomial layout assignment = polynomial /\
    decodedChallenge layout assignment =
      (productionReplay before.transcriptState roundIndex polynomial).1 /\
    decodedNext layout assignment = after.current /\
    after.transcriptState =
      (productionReplay before.transcriptState roundIndex polynomial).2 /\
    after.point = before.point ++ [decodedChallenge layout assignment] /\
    after.cursor = before.cursor + 1

private theorem continuation_eq
    {Field State : Type}
    (left right : Continuation Field State)
    (state : left.transcriptState = right.transcriptState)
    (currentValue : left.current = right.current)
    (pointValue : left.point = right.point)
    (cursorValue : left.cursor = right.cursor) :
    left = right := by
  cases left
  cases right
  simp_all

/-- The arithmetic rows and control placement imply the complete fused
production round relation. -/
theorem rows_imply_roundPhaseRelation
    {rowVariables : Nat}
    (layout : Layout)
    (assignment : Nat → Nat)
    (constantWire : assignment 0 = 1)
    (satisfied : Satisfies (rows layout) assignment)
    (roundIndex : Fin (ProductNifsCodec.shapeFor rowVariables).cubeVariables)
    (polynomial : ProductionRound)
    (before after : Continuation ConcreteK BindingState)
    (placed : ControlPlacement layout assignment roundIndex polynomial
      before after) :
    RoundPhaseRelation (ProductPoseidon2.transcriptFor rowVariables)
      ConcreteCarrier.extensionOps.toOps roundIndex polynomial before after := by
  rcases placed with
    ⟨cursorExact, currentExact, polynomialExact, challengeExact, nextExact,
      stateExact, pointExact, nextCursorExact⟩
  rcases rows_imply_concrete_round layout assignment constantWire satisfied with
    ⟨initialExact, outgoingExact⟩
  refine ⟨cursorExact, ?_, ?_⟩
  · rw [← currentExact, ← polynomialExact]
    exact initialExact
  · apply continuation_eq
    · simpa [step, productionReplay] using stateExact
    · calc
        after.current = decodedNext layout assignment := nextExact.symm
        _ = (decodedPolynomial layout assignment).evaluate
            ConcreteCarrier.extensionOps.toOps
            (decodedChallenge layout assignment) := outgoingExact
        _ = polynomial.evaluate ConcreteCarrier.extensionOps.toOps
            (productionReplay before.transcriptState roundIndex polynomial).1 := by
              rw [polynomialExact, challengeExact]
        _ = (step (ProductPoseidon2.transcriptFor rowVariables)
            ConcreteCarrier.extensionOps.toOps roundIndex polynomial before).current := by
              rfl
    · calc
        after.point = before.point ++ [decodedChallenge layout assignment] :=
          pointExact
        _ = before.point ++
            [(productionReplay before.transcriptState roundIndex polynomial).1] := by
              rw [challengeExact]
        _ = (step (ProductPoseidon2.transcriptFor rowVariables)
            ConcreteCarrier.extensionOps.toOps roundIndex polynomial before).point := by
              rfl
    · simpa [step] using nextCursorExact

end Nightstream.Implementation.Nebula.ProductionStreamingPiCcsRoundRows
