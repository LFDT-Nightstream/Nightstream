import Nightstream.Implementation.R1CS.Canonical.KPiCcsInitial

/-!
Honest completeness for the Lean-owned joint `Pi_CCS` initial-claim rows.

The witness allocates only the Horner frames.  The final equality allocates
nothing, so it is discharged from the exact paper target relation after
proving that every authoritative source expression is preserved below the
allocation base.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Canonical.KPiCcsInitialHonest

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal
open Nightstream.Implementation.R1CS.Canonical.KHorner
open Nightstream.Implementation.R1CS.Canonical.KMul
open Nightstream.Implementation.R1CS.Canonical.KFixedPhaseSumCheck
open Nightstream.Implementation.R1CS.Canonical.KConcreteFixedPhaseBridge
open Nightstream.Implementation.R1CS.Canonical.KPiCcsInitial

private abbrev concreteOps := ConcreteCarrier.extensionOps

/-- The only columns written by the honest initial-claim witness are the
canonical Horner frames at and above `input.frameBase`. -/
def honestAssignment
    {shape : Shape} (input : Input shape) (assignment : Nat → Nat) :
    Nat → Nat :=
  KHornerHonest.hornerWitness assignment input.gamma input.frameBase
    (coefficients input) 0

theorem honestAssignment_off_block
    {shape : Shape} (input : Input shape) (assignment : Nat → Nat)
    (column : Nat) (below : column < input.frameBase) :
    honestAssignment input assignment column = assignment column :=
  KHornerHonest.hornerWitness_off_block assignment input.gamma
    input.frameBase (coefficients input) 0 column (by simpa using below)

private theorem decode_preserved
    {shape : Shape} (input : Input shape) (assignment : Nat → Nat)
    (value : Carried)
    (belowLow : KHornerHonest.BelowBase value.low input.frameBase)
    (belowHigh : KHornerHonest.BelowBase value.high input.frameBase) :
    decodeCarried (honestAssignment input assignment) value =
      decodeCarried assignment value := by
  apply KBridge.toPair_injective
  simp only [toPair_decodeCarried, carriedValue, Pair.mk.injEq]
  constructor
  · exact KMulHonest.lcEval_congr _ _ value.low
      (fun column mentioned =>
        honestAssignment_off_block input assignment column
          (belowLow column mentioned))
  · exact KMulHonest.lcEval_congr _ _ value.high
      (fun column mentioned =>
        honestAssignment_off_block input assignment column
          (belowHigh column mentioned))

private theorem zero_below (base : Nat) :
    KHornerHonest.BelowBase zeroCarried.low base ∧
      KHornerHonest.BelowBase zeroCarried.high base := by
  constructor <;> intro column mentioned <;>
    simp [zeroCarried, KHornerHonest.BelowBase, Mentions] at mentioned

private theorem coefficients_below
    {shape : Shape} (input : Input shape)
    (claimedBelow : ∀ coordinate,
      KHornerHonest.BelowBase (input.claimedCoefficient coordinate).low
          input.frameBase ∧
        KHornerHonest.BelowBase (input.claimedCoefficient coordinate).high
          input.frameBase) :
    ∀ coefficient ∈ coefficients input,
      KHornerHonest.BelowBase coefficient.low input.frameBase ∧
        KHornerHonest.BelowBase coefficient.high input.frameBase := by
  intro coefficient member
  unfold coefficients at member
  rcases List.mem_append.1 member with inZeros | inClaimed
  · have equal : coefficient = zeroCarried := by
      have replicated :
          shape.carriedEvaluationOffset ≠ 0 ∧ coefficient = zeroCarried := by
        simpa only [List.mem_replicate] using inZeros
      exact replicated.2
    subst coefficient
    exact zero_below input.frameBase
  · rcases List.mem_map.1 inClaimed with
      ⟨coordinate, _, rfl⟩
    exact claimedBelow coordinate

private theorem satisfies_append
    {left right : List Row} {assignment : Nat → Nat}
    (leftSatisfied : Satisfies left assignment)
    (rightSatisfied : Satisfies right assignment) :
    Satisfies (left ++ right) assignment := by
  intro row member
  exact (List.mem_append.1 member).elim
    (leftSatisfied row) (rightSatisfied row)

/-- **Honest completeness.**  An authoritative assignment whose decoded
initial value is the exact paper target extends to a satisfying assignment for
every emitted row.  All other hypotheses are physical placement facts. -/
theorem rows_honest
    {shape : Shape} (input : Input shape) (assignment : Nat → Nat)
    (basePositive : 0 < input.frameBase)
    (constantWire : assignment 0 = 1)
    (gammaBelow :
      KHornerHonest.BelowBase input.gamma.low input.frameBase ∧
        KHornerHonest.BelowBase input.gamma.high input.frameBase)
    (claimedBelow : ∀ coordinate,
      KHornerHonest.BelowBase (input.claimedCoefficient coordinate).low
          input.frameBase ∧
        KHornerHonest.BelowBase (input.claimedCoefficient coordinate).high
          input.frameBase)
    (initialBelow :
      KHornerHonest.BelowBase input.initial.low input.frameBase ∧
        KHornerHonest.BelowBase input.initial.high input.frameBase)
    (valid :
      decodedInitial input assignment =
        TargetPolynomial.evaluateShifted concreteOps.toOps
          (decodedTarget input assignment)
          (decodedGamma input assignment)) :
    Satisfies (rows input) (honestAssignment input assignment) := by
  let witness := honestAssignment input assignment
  have coefficientBelow := coefficients_below input claimedBelow
  have hornerSatisfied :
      Satisfies
        (hornerRows input.gamma (KFrames.frameAt input.frameBase)
          (coefficients input) 0) witness := by
    exact KHornerHonest.hornerWitness_satisfies assignment input.gamma
      input.frameBase gammaBelow.1 gammaBelow.2 (coefficients input) 0
      coefficientBelow
  have constantWireWitness : witness 0 = 1 := by
    rw [show witness 0 = assignment 0 by
      exact honestAssignment_off_block input assignment 0 basePositive]
    exact constantWire
  have gammaPreserved :
      decodedGamma input witness = decodedGamma input assignment := by
    unfold decodedGamma
    exact congrArg ofProjection
      (decode_preserved input assignment input.gamma gammaBelow.1 gammaBelow.2)
  have initialPreserved :
      decodedInitial input witness = decodedInitial input assignment := by
    unfold decodedInitial
    exact congrArg ofProjection
      (decode_preserved input assignment input.initial
        initialBelow.1 initialBelow.2)
  have targetPreserved :
      decodedTarget input witness = decodedTarget input assignment := by
    unfold decodedTarget
    congr 1
    funext coordinate
    exact congrArg ofProjection
      (decode_preserved input assignment
        (input.claimedCoefficient coordinate)
        (claimedBelow coordinate).1 (claimedBelow coordinate).2)
  have validWitness :
      decodedInitial input witness =
        TargetPolynomial.evaluateShifted concreteOps.toOps
          (decodedTarget input witness) (decodedGamma input witness) := by
    rw [initialPreserved, targetPreserved, gammaPreserved]
    exact valid
  have computed :=
    horner_rows_sound input witness hornerSatisfied
  have decodedEqual :
      decodeCarried witness (evaluated input) =
        decodeCarried witness input.initial := by
    apply KBridge.toPair_injective
    have concreteEqual :
        decodedEvaluated input witness = decodedInitial input witness := by
      exact computed.trans validWitness.symm
    have projectedEqual := congrArg toProjection concreteEqual
    unfold decodedEvaluated decodedInitial at projectedEqual
    simpa only [toProjection_ofProjection, toPair_decodeCarried] using
      congrArg KBridge.toPair projectedEqual
  have pairEqual := congrArg KBridge.toPair decodedEqual
  simp only [toPair_decodeCarried, carriedValue, Pair.mk.injEq] at pairEqual
  have equalitySatisfied :
      Satisfies (KEquality.rows (evaluated input) input.initial) witness :=
    KEquality.rows_complete witness (evaluated input) input.initial
      constantWireWitness pairEqual.1 pairEqual.2
  exact satisfies_append hornerSatisfied equalitySatisfied

end Nightstream.Implementation.R1CS.Canonical.KPiCcsInitialHonest
