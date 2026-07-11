import Nightstream.Implementation.R1CS.FPrimeFullHistoryPiCcsRecursiveAllocation
import Nightstream.Implementation.R1CS.FPrimeFullHistoryPiCcsRecursiveAuthority
import Nightstream.Implementation.R1CS.FPrimeFullHistoryPiCcsRecursiveOutputBinding
import Nightstream.Implementation.R1CS.FPrimeFullHistoryPiCcsTerminalAllocation
import Nightstream.Implementation.R1CS.FPrimeFullHistoryPiCcsTerminalOutputBinding
import Nightstream.Implementation.R1CS.FPrimeFullHistoryPiRlcRecursiveLinearFolds
import Nightstream.Implementation.R1CS.FPrimeFullHistoryPiRlcRecursiveShape
import Nightstream.Implementation.R1CS.FPrimeFullHistoryPiRlcTerminalLinearFolds
import Nightstream.Implementation.R1CS.FPrimeFullHistoryPiRlcTerminalShape

/-!
Contract: exact soundness and witness completeness for every affine PiCCS and
PiRLC phase in the supported full-history artifact.

Each conclusion states the zero, constant, or equality semantics decoded from
the production sparse rows.  No row count or digest is used as semantic
authority.
-/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryAffineSound

open Nightstream.Implementation.R1CS

namespace Recursive

structure RowsSatisfy (assignment : Nat → Nat) : Prop where
  piCcsAllocation : Satisfies
    FPrimeFullHistoryPiCcsRecursiveAllocation.rows assignment
  piCcsAuthority : Satisfies
    FPrimeFullHistoryPiCcsRecursiveAuthority.rows assignment
  piCcsOutputBinding : Satisfies
    FPrimeFullHistoryPiCcsRecursiveOutputBinding.rows assignment
  piRlcShape : Satisfies
    FPrimeFullHistoryPiRlcRecursiveShape.rows assignment
  piRlcLinearFolds : Satisfies
    FPrimeFullHistoryPiRlcRecursiveLinearFolds.rows assignment

structure Holds (assignment : Nat → Nat) : Prop where
  piCcsAllocation : ∀ pin ∈
      FPrimeFullHistoryPiCcsRecursiveAllocation.pins,
    AffinePins.Pin.Holds assignment pin
  piCcsAuthority : ∀ pin ∈
      FPrimeFullHistoryPiCcsRecursiveAuthority.pins,
    AffinePins.Pin.Holds assignment pin
  piCcsOutputBinding : ∀ pin ∈
      FPrimeFullHistoryPiCcsRecursiveOutputBinding.pins,
    AffinePins.Pin.Holds assignment pin
  piRlcShape : ∀ pin ∈ FPrimeFullHistoryPiRlcRecursiveShape.pins,
    AffinePins.Pin.Holds assignment pin
  piRlcLinearFolds : ∀ pin ∈
      FPrimeFullHistoryPiRlcRecursiveLinearFolds.pins,
    AffinePins.Pin.Holds assignment pin

theorem sound
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (rows : RowsSatisfy assignment) :
    Holds assignment := by
  exact {
    piCcsAllocation := AffinePins.rows_sound
      FPrimeFullHistoryPiCcsRecursiveAllocation.pins_canonical
      canonical one rows.piCcsAllocation
    piCcsAuthority := AffinePins.rows_sound
      FPrimeFullHistoryPiCcsRecursiveAuthority.pins_canonical
      canonical one rows.piCcsAuthority
    piCcsOutputBinding := AffinePins.rows_sound
      FPrimeFullHistoryPiCcsRecursiveOutputBinding.pins_canonical
      canonical one rows.piCcsOutputBinding
    piRlcShape := AffinePins.rows_sound
      FPrimeFullHistoryPiRlcRecursiveShape.pins_canonical
      canonical one rows.piRlcShape
    piRlcLinearFolds := AffinePins.rows_sound
      FPrimeFullHistoryPiRlcRecursiveLinearFolds.pins_canonical
      canonical one rows.piRlcLinearFolds
  }

theorem complete
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (holds : Holds assignment) :
    RowsSatisfy assignment := by
  exact {
    piCcsAllocation := AffinePins.rows_complete
      FPrimeFullHistoryPiCcsRecursiveAllocation.pins_canonical
      canonical one holds.piCcsAllocation
    piCcsAuthority := AffinePins.rows_complete
      FPrimeFullHistoryPiCcsRecursiveAuthority.pins_canonical
      canonical one holds.piCcsAuthority
    piCcsOutputBinding := AffinePins.rows_complete
      FPrimeFullHistoryPiCcsRecursiveOutputBinding.pins_canonical
      canonical one holds.piCcsOutputBinding
    piRlcShape := AffinePins.rows_complete
      FPrimeFullHistoryPiRlcRecursiveShape.pins_canonical
      canonical one holds.piRlcShape
    piRlcLinearFolds := AffinePins.rows_complete
      FPrimeFullHistoryPiRlcRecursiveLinearFolds.pins_canonical
      canonical one holds.piRlcLinearFolds
  }

end Recursive

namespace Terminal

structure RowsSatisfy (assignment : Nat → Nat) : Prop where
  piCcsAllocation : Satisfies
    FPrimeFullHistoryPiCcsTerminalAllocation.rows assignment
  piCcsOutputBinding : Satisfies
    FPrimeFullHistoryPiCcsTerminalOutputBinding.rows assignment
  piRlcShape : Satisfies
    FPrimeFullHistoryPiRlcTerminalShape.rows assignment
  piRlcLinearFolds : Satisfies
    FPrimeFullHistoryPiRlcTerminalLinearFolds.rows assignment

structure Holds (assignment : Nat → Nat) : Prop where
  piCcsAllocation : ∀ pin ∈
      FPrimeFullHistoryPiCcsTerminalAllocation.pins,
    AffinePins.Pin.Holds assignment pin
  piCcsOutputBinding : ∀ pin ∈
      FPrimeFullHistoryPiCcsTerminalOutputBinding.pins,
    AffinePins.Pin.Holds assignment pin
  piRlcShape : ∀ pin ∈ FPrimeFullHistoryPiRlcTerminalShape.pins,
    AffinePins.Pin.Holds assignment pin
  piRlcLinearFolds : ∀ pin ∈
      FPrimeFullHistoryPiRlcTerminalLinearFolds.pins,
    AffinePins.Pin.Holds assignment pin

theorem sound
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (rows : RowsSatisfy assignment) :
    Holds assignment := by
  exact {
    piCcsAllocation := AffinePins.rows_sound
      FPrimeFullHistoryPiCcsTerminalAllocation.pins_canonical
      canonical one rows.piCcsAllocation
    piCcsOutputBinding := AffinePins.rows_sound
      FPrimeFullHistoryPiCcsTerminalOutputBinding.pins_canonical
      canonical one rows.piCcsOutputBinding
    piRlcShape := AffinePins.rows_sound
      FPrimeFullHistoryPiRlcTerminalShape.pins_canonical
      canonical one rows.piRlcShape
    piRlcLinearFolds := AffinePins.rows_sound
      FPrimeFullHistoryPiRlcTerminalLinearFolds.pins_canonical
      canonical one rows.piRlcLinearFolds
  }

theorem complete
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (holds : Holds assignment) :
    RowsSatisfy assignment := by
  exact {
    piCcsAllocation := AffinePins.rows_complete
      FPrimeFullHistoryPiCcsTerminalAllocation.pins_canonical
      canonical one holds.piCcsAllocation
    piCcsOutputBinding := AffinePins.rows_complete
      FPrimeFullHistoryPiCcsTerminalOutputBinding.pins_canonical
      canonical one holds.piCcsOutputBinding
    piRlcShape := AffinePins.rows_complete
      FPrimeFullHistoryPiRlcTerminalShape.pins_canonical
      canonical one holds.piRlcShape
    piRlcLinearFolds := AffinePins.rows_complete
      FPrimeFullHistoryPiRlcTerminalLinearFolds.pins_canonical
      canonical one holds.piRlcLinearFolds
  }

end Terminal

end Nightstream.Implementation.R1CS.FPrimeFullHistoryAffineSound
