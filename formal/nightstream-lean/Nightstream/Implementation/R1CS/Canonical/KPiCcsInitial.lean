import Nightstream.Implementation.R1CS.Canonical.KConcreteFixedPhaseBridge
import Nightstream.Implementation.R1CS.Canonical.KHornerHonest
import Nightstream.Implementation.R1CS.Canonical.KEquality
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.SignedCoefficientPolynomial

/-!
Lean-owned rows for the verifier-computed initial claim of joint `Pi_CCS`.

The paper initial is the shifted carried target

`Σ coordinate, gamma^(2K+k+I(coordinate)) * claimedCoefficient coordinate`.

`canonicalCarriedCoordinates_localGammaExponents` already proves that the
canonical coordinate traversal has local exponents `0, ..., k*t*d-1`.
Consequently the exact constant-first coefficient vector is:

```
[0; ...; 0]                    -- `2K+k` verifier-owned zeros
++ canonical carried coefficients
```

This module evaluates that vector with the existing Lean-owned Horner rows and
binds the result to the fixed-phase chain's initial column pair.  It consumes
no claimed exponent list and no Rust row layout.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Canonical.KPiCcsInitial

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.SumCheck.Finite
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal
open Nightstream.Implementation.R1CS.Canonical.KHorner
open Nightstream.Implementation.R1CS.Canonical.KMul
open Nightstream.Implementation.R1CS.Canonical.KFixedPhaseSumCheck
open Nightstream.Implementation.R1CS.Canonical.KConcreteFixedPhaseBridge

private abbrev ConcreteK := Nightstream.SuperNeo.Concrete.K
private abbrev RowK := Nightstream.Implementation.R1CS.ProjectionProgram.K
private abbrev concreteOps := ConcreteCarrier.extensionOps

/-- The exact row-level inputs of the initial-claim computation. -/
structure Input (shape : Shape) where
  gamma : Carried
  claimedCoefficient : CarriedCoordinate shape → Carried
  initial : Carried
  frameBase : Nat

/-- Constant-first dense target vector.  The leading values are expressions
for zero, not witness columns. -/
def coefficients
    {shape : Shape} (input : Input shape) : List Carried :=
  List.replicate shape.carriedEvaluationOffset zeroCarried ++
    (canonicalCarriedCoordinates shape).map input.claimedCoefficient

/-- The carried Horner output. -/
def evaluated
    {shape : Shape} (input : Input shape) : Carried :=
  hornerCarried input.gamma (KFrames.frameAt input.frameBase)
    (coefficients input) 0

/-- Horner evaluation followed by the two-coordinate binding to the chain's
initial value. -/
def rows
    {shape : Shape} (input : Input shape) : List Row :=
  hornerRows input.gamma (KFrames.frameAt input.frameBase)
      (coefficients input) 0 ++
    KEquality.rows (evaluated input) input.initial

theorem coefficients_length
    {shape : Shape} (input : Input shape) :
    (coefficients input).length = shape.jointCoefficientCount := by
  unfold coefficients Shape.jointCoefficientCount
  rw [List.length_append, List.length_replicate,
    List.length_map, canonicalCarriedCoordinates_length]
  unfold Shape.carriedEvaluationOffset Shape.sourceCount
  omega

/-- Exact row count, derived from the emitted list. -/
theorem rows_length
    {shape : Shape} (input : Input shape) :
    (rows input).length = 3 * (shape.jointCoefficientCount - 1) + 2 := by
  unfold rows
  rw [List.length_append, hornerRows_length, KEquality.rows_length,
    coefficients_length]

/-- Exact auxiliary allocation of the Horner chain. -/
def columns
    {shape : Shape} (input : Input shape) : List Nat :=
  KFrames.frameColumns input.frameBase ((coefficients input).length - 1)

theorem columns_length
    {shape : Shape} (input : Input shape) :
    (columns input).length = 3 * (shape.jointCoefficientCount - 1) := by
  unfold columns
  rw [KFrames.frameColumns_length, coefficients_length]

theorem columns_nodup
    {shape : Shape} (input : Input shape) :
    (columns input).Nodup :=
  KFrames.frameColumns_nodup _ _

/-- Decode the verifier-owned values directly from the row expressions. -/
def decodedTarget
    {shape : Shape} (input : Input shape) (assignment : Nat → Nat) :
    TargetPolynomial.CarriedTargetCoefficients ConcreteK shape where
  coefficient := fun coordinate =>
    ofProjection (decodeCarried assignment
      (input.claimedCoefficient coordinate))

def decodedGamma
    {shape : Shape} (input : Input shape) (assignment : Nat → Nat) :
    ConcreteK :=
  ofProjection (decodeCarried assignment input.gamma)

def decodedInitial
    {shape : Shape} (input : Input shape) (assignment : Nat → Nat) :
    ConcreteK :=
  ofProjection (decodeCarried assignment input.initial)

def decodedEvaluated
    {shape : Shape} (input : Input shape) (assignment : Nat → Nat) :
    ConcreteK :=
  ofProjection (decodeCarried assignment (evaluated input))

private def concreteShiftLaws :
    TargetPolynomial.ShiftLaws concreteOps.toOps where
  one_mul := ConcreteCarrier.extensionLaws.one_mul
  mul_assoc := ConcreteCarrier.extensionLaws.mul_assoc
  mul_zero := ConcreteCarrier.extensionLaws.mul_zero
  mul_add := ConcreteCarrier.extensionLaws.left_distrib

private theorem evaluate_zeros (gamma : ConcreteK) :
    ∀ count,
      Message.evaluateCoefficients concreteOps.toOps gamma
          (List.replicate count concreteOps.zero) =
        concreteOps.zero
  | 0 => rfl
  | count + 1 => by
      simp only [List.replicate_succ, Message.evaluateCoefficients]
      rw [evaluate_zeros gamma count,
        ConcreteCarrier.extensionLaws.mul_zero,
        ConcreteCarrier.extensionLaws.zero_add]

/-- The dense vector is definitionally selected by the paper exponent layout:
its concrete evaluation is exactly `VerifierInput.initial`. -/
theorem concrete_coefficients_evaluate
    {shape : Shape} (input : Input shape) (assignment : Nat → Nat) :
    Message.evaluateCoefficients concreteOps.toOps
        (decodedGamma input assignment)
        (List.replicate shape.carriedEvaluationOffset concreteOps.zero ++
          (canonicalCarriedCoordinates shape).map
            (decodedTarget input assignment).coefficient) =
      TargetPolynomial.evaluateShifted concreteOps.toOps
        (decodedTarget input assignment) (decodedGamma input assignment) := by
  rw [SignedCoefficientPolynomial.evaluate_append concreteOps
    ConcreteCarrier.extensionLaws, evaluate_zeros,
    ConcreteCarrier.extensionLaws.zero_add,
    SignedCoefficientPolynomial.evaluate_canonicalCarriedMap_eq_targetLocal
      concreteOps ConcreteCarrier.extensionLaws,
    TargetPolynomial.evaluateShifted_eq_shift_mul_evaluateLocal
      concreteOps.toOps concreteShiftLaws]
  simp only [List.length_replicate]

private theorem decoded_coefficients_map
    {shape : Shape} (input : Input shape) (assignment : Nat → Nat) :
    (coefficients input).map (decodeCarried assignment) =
      (List.replicate shape.carriedEvaluationOffset concreteOps.zero ++
        (canonicalCarriedCoordinates shape).map
          (decodedTarget input assignment).coefficient).map toProjection := by
  unfold coefficients decodedTarget
  have zeroEqual :
      decodeCarried assignment zeroCarried =
        toProjection concreteOps.zero := by
    rw [decodeCarried_zero]
    exact toProjection_zero.symm
  rw [List.map_append, List.map_append, List.map_replicate,
    List.map_replicate, zeroEqual]
  congr 1
  rw [List.map_map, List.map_map]
  apply List.map_congr_left
  intro coordinate _
  simp only [Function.comp_apply, toProjection_ofProjection]

/-- The Horner subprogram alone computes the verifier-owned shifted target.
This is factored from the final binding so honest completeness can use the same
semantic bridge without assuming the binding row it is about to prove. -/
theorem horner_rows_sound
    {shape : Shape} (input : Input shape) (assignment : Nat → Nat)
    (satisfied :
      Satisfies
        (hornerRows input.gamma (KFrames.frameAt input.frameBase)
          (coefficients input) 0) assignment) :
    decodedEvaluated input assignment =
      TargetPolynomial.evaluateShifted concreteOps.toOps
        (decodedTarget input assignment) (decodedGamma input assignment) := by
  have computed :=
    hornerRows_sound assignment input.gamma
      (KFrames.frameAt input.frameBase) (coefficients input) 0 satisfied
  have carriedCoefficients :
      (coefficients input).map (carriedValue assignment) =
        ((coefficients input).map (decodeCarried assignment)).map
          KBridge.toPair := by
    rw [List.map_map]
    apply List.map_congr_left
    intro coefficient _
    exact (toPair_decodeCarried assignment coefficient).symm
  have gammaProjected :
      decodeCarried assignment input.gamma =
        toProjection (decodedGamma input assignment) := by
    unfold decodedGamma
    rw [toProjection_ofProjection]
  apply toProjection_injective
  unfold decodedEvaluated
  rw [toProjection_ofProjection]
  apply KBridge.toPair_injective
  rw [toPair_decodeCarried, toPair_toProjection]
  unfold evaluated
  rw [computed, ← toPair_decodeCarried, carriedCoefficients,
    ← toPair_evaluateCoefficients, gammaProjected,
    decoded_coefficients_map, ← evaluateCoefficients_map,
    toPair_toProjection]
  exact congrArg KConcreteBridge.ofConcrete
    (concrete_coefficients_evaluate input assignment)

/-- **Soundness of the initial binding.**  Satisfying rows force the chain's
initial value to be the exact shifted target decoded from the same authoritative
coefficient and gamma expressions. -/
theorem rows_sound
    {shape : Shape} (input : Input shape) (assignment : Nat → Nat)
    (constantWire : assignment 0 = 1)
    (satisfied : Satisfies (rows input) assignment) :
    decodedInitial input assignment =
      TargetPolynomial.evaluateShifted concreteOps.toOps
        (decodedTarget input assignment) (decodedGamma input assignment) := by
  have hornerSatisfied :
      Satisfies
        (hornerRows input.gamma (KFrames.frameAt input.frameBase)
          (coefficients input) 0) assignment :=
    fun row member => satisfied row (List.mem_append_left _ member)
  have equalitySatisfied :
      Satisfies (KEquality.rows (evaluated input) input.initial) assignment :=
    fun row member => satisfied row (List.mem_append_right _ member)
  have bound :=
    KEquality.rows_sound assignment (evaluated input) input.initial
      constantWire equalitySatisfied
  have pairBound :
      carriedValue assignment (evaluated input) =
        carriedValue assignment input.initial := by
    simpa only [carriedValue, Pair.mk.injEq] using bound
  have decodedBound :
      decodedEvaluated input assignment = decodedInitial input assignment := by
    apply toProjection_injective
    unfold decodedEvaluated decodedInitial
    simp only [toProjection_ofProjection]
    apply KBridge.toPair_injective
    simpa only [toPair_decodeCarried] using pairBound
  calc
    decodedInitial input assignment =
        decodedEvaluated input assignment := decodedBound.symm
    _ = TargetPolynomial.evaluateShifted concreteOps.toOps
          (decodedTarget input assignment)
          (decodedGamma input assignment) :=
      horner_rows_sound input assignment hornerSatisfied

end Nightstream.Implementation.R1CS.Canonical.KPiCcsInitial
