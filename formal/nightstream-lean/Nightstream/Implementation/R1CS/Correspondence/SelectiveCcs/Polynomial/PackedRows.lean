import Mathlib.Data.ZMod.Basic
import Mathlib.Tactic.Ring
import Nightstream.Implementation.R1CS.Correspondence.SelectiveCcs.Polynomial.Rows
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConcreteCarrier.NoZeroDivisors

/-!
Contract: algebraic soundness of the two-coordinate selective domain row.

Owns: expansion of the exact ten centered terms at `G=E=1`, equivalence of
one packed row to two centered-unit residuals under the named projective-seven
premise, and the fixed-zero odd-tail specialization.

Does not own: emitted matrix coefficients, selector Booleanity, production
row schedules, Rust conformance, or the concrete proof of projective
nonresiduosity.

Emits constraints: no.

Assurance tier: model-level. The pair theorem is conditional on
`SevenProjectiveNonresidue`; a higher implementation theorem must discharge
that premise and bind these port images to generated Rust rows.
-/

namespace Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.PackedRows

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConcreteCarrier
open Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.Ports
open Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.Semantics
open Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.Components
open Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.Rows

private abbrev Modulus := Nightstream.SuperNeo.Concrete.goldilocksModulus

/-- Exact active-row expansion of the Rust ten-term centered component. -/
theorem centeredResidualAt_one_one (left right : F) :
    centeredResidualAt 1 1 left right =
      centeredUnitResidual left ^ 2 - 7 * centeredUnitResidual right ^ 2 := by
  have coreCube (value : F) :
      Lean.Grind.Fin.npow value 3 = value * value * value := by
    rw [show Lean.Grind.Fin.npow value 3 = value ^ 3 by rfl]
    simp only [pow_succ, pow_zero, one_mul]
  have unitExpansion (value : F) :
      centeredUnitResidual value = value * value * value - value := by
    change Lean.Grind.Fin.npow value 3 - value = _
    rw [coreCube]
  rw [unitExpansion left, unitExpansion right]
  let equiv : F ≃+* ZMod Modulus := ZMod.finEquiv Modulus
  have equivOne : equiv (1 : F) = (1 : ZMod Modulus) := by
    rfl
  have equivTwo : equiv (2 : F) = (2 : ZMod Modulus) := by
    rfl
  have equivSeven : equiv (7 : F) = (7 : ZMod Modulus) := by
    rfl
  have equivFourteen : equiv (14 : F) = (14 : ZMod Modulus) := by
    rfl
  apply equiv.injective
  simp only [centeredResidualAt, map_add, map_sub,
    map_neg, map_mul, map_pow, equivOne, equivTwo, equivSeven,
    equivFourteen]
  ring

theorem evaluate_centeredPairPoint_one (left right : F) :
    evaluate (centeredPairPoint 1 left right) =
      centeredUnitResidual left ^ 2 - 7 * centeredUnitResidual right ^ 2 := by
  rw [evaluate_centeredPairPoint]
  simpa [centeredResidual, centeredPairPoint, sparsePoint, Role.index] using
    centeredResidualAt_one_one left right

/-- Model-level pair soundness under the explicit irreducibility boundary. -/
theorem evaluate_centeredPairPoint_one_zero_iff
    (sevenNonresidue : SevenProjectiveNonresidue)
    (left right : F) :
    evaluate (centeredPairPoint 1 left right) = 0 ↔
      centeredUnitResidual left = 0 ∧ centeredUnitResidual right = 0 := by
  rw [evaluate_centeredPairPoint_one]
  constructor
  · intro packed
    apply sevenNonresidue _ _
    rw [mul_assoc]
    simpa [pow_two] using packed
  · rintro ⟨leftZero, rightZero⟩
    rw [leftZero, rightZero]
    rfl

/-- Model-level odd-tail soundness with a fixed-zero right coordinate. -/
theorem evaluate_centeredPairTailPoint_one_zero_iff
    (sevenNonresidue : SevenProjectiveNonresidue)
    (left : F) :
    evaluate (centeredPairPoint 1 left 0) = 0 ↔
      centeredUnitResidual left = 0 := by
  rw [evaluate_centeredPairPoint_one_zero_iff sevenNonresidue]
  have zeroResidual : centeredUnitResidual (0 : F) = 0 := by
    rfl
  simp [zeroResidual]

end Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.PackedRows
