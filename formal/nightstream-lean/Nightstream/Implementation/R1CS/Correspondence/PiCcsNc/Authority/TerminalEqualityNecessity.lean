import Nightstream.Implementation.R1CS.Correspondence.PiCcsNc.Terminal.Identity

/-!
Contract: exhibit a model-level counterexample showing that equality of the
Π_CCS NC terminal scalar does not establish carried `y_zcol` authority.

Owns: one zero authoritative assignment, one independently changed projection
sidecar, and the kernel-checked comparison of their terminal range factors.

Does not own: a production-verifier acceptance theorem, transcript or SumCheck
soundness, Rust/R1CS correspondence, protocol unsoundness, or row-removal
permission.

Emits constraints: no.

Authority boundary: the raw assignment remains independent authority. The
claimed sidecar differs at its only active lane, while both projected scalars
are roots of the non-injective `b = 2` range polynomial.

| Witness surface | Authoritative projection | Claimed projection | Terminal consequence | Authority consequence |
|---|---:|---:|---|---|
| one raw column and one packed lane | `0` | `1` | `rangeProductB2 0 = rangeProductB2 1 = 0` | terminal equality holds while `YZcolBound` is false |
-/

namespace Nightstream.Implementation.R1CS.PiCcsNc.Authority.TerminalEqualityNecessity

open Nightstream.Implementation.R1CS.PiCcsNc
open Nightstream.Implementation.R1CS.PiCcsNc.MixedPolynomial
open Nightstream.Implementation.R1CS.PiCcsNc.RangePolynomial
open Nightstream.Implementation.R1CS.PiCcsNc.Terminal
open Nightstream.SuperNeo.Concrete

/-- Smallest shape with one raw column and one packed lane. -/
def collisionShape : Shape := { ellM := 0, ellD := 0 }

/-- One authoritative coordinate whose direct projection is zero. -/
def zeroAssignments : List (List F) := [[0]]

/-- A claimed sidecar that changes only the sole active lane to one. -/
def claimedOneYZcol : YZcol := fun lane =>
  if lane = 0 then K.one else K.zero

/-- The one claimed output consumed by the terminal expression. -/
def claimedOneOutputs : List YZcol := [claimedOneYZcol]

/-- The honest zero sidecar for the authoritative zero assignment. -/
def authoritativeZeroOutputs : List YZcol := [zeroYZcol]

private theorem rangeProductB2_zero :
    rangeProductB2 K.zero = K.zero := by
  unfold rangeProductB2
  rw [mul_zero, zero_mul]

private theorem rangeProductB2_one :
    rangeProductB2 K.one = K.zero := by
  have embedOne : K.embed 1 = K.one := by rfl
  have oneSubOne : K.sub K.one K.one = K.zero := by rfl
  unfold rangeProductB2
  rw [embedOne, oneSubOne, mul_zero]

/-- The honest zero sidecar satisfies exact projection authority. -/
theorem authoritativeZeroOutputs_yZcolBound :
    YZcolBound collisionShape zeroAssignments []
      authoritativeZeroOutputs := by
  constructor
  · rfl
  · intro outputIndex laneIndex outputLt laneLt
    have outputZero : outputIndex = 0 := by
      simpa [zeroAssignments] using outputLt
    have laneZero : laneIndex = 0 := by
      simpa [collisionShape, Shape.laneDomain] using laneLt
    subst outputIndex
    subst laneIndex
    rfl

/-- The claimed sidecar is not the projection of the authoritative zero
assignment. -/
theorem claimedOneOutputs_not_yZcolBound :
    ¬ YZcolBound collisionShape zeroAssignments []
      claimedOneOutputs := by
  intro bound
  have lane := bound.lane 0 0 (by decide) (by decide)
  have oneEqZero : K.one = K.zero := by
    simpa [claimedOneOutputs, claimedOneYZcol, zeroAssignments,
      collisionShape, authoritativeYZcol, directDiagonal] using lane
  exact (by decide : K.one ≠ K.zero) oneEqZero

private theorem claimedOneOutputs_terminal_zero :
    terminalRhs collisionShape [] [] K.one
      claimedOneOutputs [] [] = K.zero := by
  simp [terminalRhs, collisionShape, claimedOneOutputs,
    Shape.laneDomain, claimedOneYZcol, dotChi, chi, eqPoint, sumRange,
    productRange, powK,
    rangeProductB2_one, mul_zero, mul_one, zero_add,
    add_zero]

private theorem authoritativeZeroOutputs_terminal_zero :
    terminalRhs collisionShape [] [] K.one
      authoritativeZeroOutputs [] [] = K.zero := by
  simp [terminalRhs, collisionShape, authoritativeZeroOutputs, zeroYZcol,
    Shape.laneDomain, dotChi, chi, eqPoint, sumRange, productRange, powK,
    rangeProductB2_zero, mul_zero, mul_one, zero_add,
    add_zero]

/-- The terminal scalar nevertheless equals the independently evaluated NC
polynomial because both zero and one vanish under `rangeProductB2`.

This theorem states only that terminal equality is insufficient to derive
`YZcolBound`. -/
theorem terminalEquality_without_yZcolBound :
    terminalRhs collisionShape [] [] K.one
        claimedOneOutputs [] [] =
      qNc collisionShape [] [] K.one zeroAssignments [] [] ∧
    ¬ YZcolBound collisionShape zeroAssignments []
      claimedOneOutputs := by
  constructor
  · calc
      terminalRhs collisionShape [] [] K.one
          claimedOneOutputs [] [] = K.zero :=
        claimedOneOutputs_terminal_zero
      _ = terminalRhs collisionShape [] [] K.one
          authoritativeZeroOutputs [] [] :=
        authoritativeZeroOutputs_terminal_zero.symm
      _ = qNc collisionShape [] [] K.one zeroAssignments [] [] :=
        terminalRhs_eq_qNc_of_yZcolBound
          authoritativeZeroOutputs_yZcolBound [] [] K.one []
  · exact claimedOneOutputs_not_yZcolBound

end Nightstream.Implementation.R1CS.PiCcsNc.Authority.TerminalEqualityNecessity
