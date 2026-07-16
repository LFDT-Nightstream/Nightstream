import Nightstream.Implementation.R1CS.Correspondence.PiCcsNc.Terminal.Identity

/-!
Contract: exhibit a concrete model-level witness that the carried `y_zcol`
authority premise cannot be erased from the Π_CCS NC terminal identity.

Owns: one production-shaped semantic assignment and the comparison between
its authoritative projection sidecar and an all-zero erased sidecar.

Does not own: production-row correspondence, any row-removal permission, or
the paper CE relation. This refutes only unsound sidecar erasure.

Emits constraints: no.

Authority boundary: the raw assignment is independent authority; changing only
the carried projection sidecar cannot change that assignment or its true NC
polynomial.

| Witness surface | Fixed authority | Changed sidecar | Consequence | Permits row removal? |
|---|---|---|---|---|
| one raw column and one packed lane | shape, assignment `[[2]]`, and evaluation points | authoritative projection replaced by all zero | `YZcolBound` fails and `terminalRhs != qNc` | no |
-/

namespace Nightstream.Implementation.R1CS.PiCcsNc.Authority.ProjectionNecessity

open Nightstream.Implementation.R1CS.PiCcsNc
open Nightstream.Implementation.R1CS.PiCcsNc.MixedPolynomial
open Nightstream.Implementation.R1CS.PiCcsNc.Terminal
open Nightstream.SuperNeo.Concrete

/-- Smallest shape with one raw column and one packed lane. -/
def counterexampleShape : Shape := { ellM := 0, ellD := 0 }

/-- A single authoritative raw coordinate outside the centered `b = 2`
zero set. -/
def counterexampleAssignments : List (List F) := [[2]]

/-- The honest carried output for the fixed shape, assignment, and point. -/
def authoritativeOutputs : List YZcol :=
  [authoritativeYZcol counterexampleShape [2] []]

/-- Unsound erasure changes only the carried projection sidecar. -/
def erasedOutputs : List YZcol := [zeroYZcol]

/-- The independently evaluated lane is nonzero for the raw coordinate `2`. -/
theorem authoritativeLane_nonzero :
    authoritativeYZcol counterexampleShape [2] [] 0 ≠ K.zero := by
  decide

/-- The honest sidecar satisfies the exact projection authority premise. -/
theorem authoritativeOutputs_yZcolBound :
    YZcolBound counterexampleShape counterexampleAssignments []
      authoritativeOutputs := by
  constructor
  · rfl
  · intro outputIndex laneIndex outputLt laneLt
    have outputLtOne : outputIndex < 1 := by
      simpa [counterexampleAssignments] using outputLt
    have outputZero : outputIndex = 0 := by omega
    have laneZero : laneIndex = 0 := by
      simpa [counterexampleShape, Shape.laneDomain] using laneLt
    subst outputIndex
    subst laneIndex
    rfl

/-- Replacing only the carried sidecar by zero destroys projection authority. -/
theorem erasedOutputs_not_yZcolBound :
    ¬ YZcolBound counterexampleShape counterexampleAssignments []
      erasedOutputs := by
  intro bound
  have lane := bound.lane 0 0 (by decide) (by decide)
  apply authoritativeLane_nonzero
  simpa [erasedOutputs, zeroYZcol] using lane.symm

/-- With the same shape and authoritative assignment, the erased sidecar makes
the terminal result diverge from the independently evaluated NC polynomial. -/
theorem erasedOutputs_terminalMismatch :
    TerminalMismatch counterexampleShape [] [] K.one
      counterexampleAssignments erasedOutputs [] [] := by
  unfold TerminalMismatch
  decide

end Nightstream.Implementation.R1CS.PiCcsNc.Authority.ProjectionNecessity
