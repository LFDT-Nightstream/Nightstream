import Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.CurrentM4Cost
import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.PiCcsDomains

/-!
Contract: derive the current transcript-domain capacity for the exact
42-times-6 benchmark Step encoding.

Assurance tier: model-level.

Owns: the exact aligned and carrier widths, the live Phi81 block count,
coverage by the selected `25/19/6` domain, and minimality of the 19-variable
block cube for this encoding.

Does not own: an Ajtai verifier key, MSIS security for this domain, selection
of the reduced benchmark as a production application, or Rust equality.

Emits constraints: none.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.CurrentM4Domain

open Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.CurrentM4Cost
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc

/-- Exact logical width of the complete emitted Step encoding. -/
theorem alignedLogicalWidth_exact :
    dimensions.alignedLogicalWidth = 19_969_313 := by
  rfl

/-- The complete Phi81 carrier adds the final 49 zero-padding coordinates. -/
theorem carrierWidth_exact :
    shape.carrierWidth = 19_969_362 := by
  decide

/-- Exact number of live Phi81 blocks needed by the emitted Step columns. -/
theorem liveBlockCount_exact :
    Phi81ColumnLayout.blockCount shape.carrierWidth = 369_803 := by
  decide

/-- The active block cube has exactly `2^19` vertices. -/
theorem currentBlockCapacity_exact :
    PiCcsDomains.currentLeanProduction.nc.blockCount = 524_288 := by
  decide

/-- The active lane cube has exactly `2^6` vertices. -/
theorem currentLaneCapacity_exact :
    PiCcsDomains.currentLeanProduction.nc.laneCount = 64 := by
  decide

/-- The current block/lane domain covers every complete-carrier coordinate. -/
theorem currentNc_covers :
    PiCcsDomains.currentLeanProduction.nc.Covers shape := by
  constructor
  · rw [liveBlockCount_exact, currentBlockCapacity_exact]
    decide
  · rw [currentLaneCapacity_exact]
    decide

/-- The current FE compatibility domain also covers the complete carrier.

The FE implementation does not consume its flat-column axis, but the shared
domain record still satisfies the stronger legacy coverage predicate. -/
theorem currentFe_covers :
    PiCcsDomains.currentLeanProduction.fe.Covers shape := by
  constructor
  · rw [carrierWidth_exact]
    change 19_969_362 <= 2 ^ 25
    decide
  · change ringDegree <= 2 ^ 6
    decide

/-- Eighteen block variables cannot cover the exact live block count. -/
theorem eighteenBlockVariables_do_not_cover :
    2 ^ 18 < Phi81ColumnLayout.blockCount shape.carrierWidth := by
  rw [liveBlockCount_exact]
  decide

/-- Nineteen is the least binary block width that covers this encoding. -/
theorem blockVariables_minimal
    {variableCount : Nat}
    (covers :
      Phi81ColumnLayout.blockCount shape.carrierWidth <= 2 ^ variableCount) :
    19 <= variableCount := by
  rcases Nat.lt_or_ge variableCount 19 with smaller | enough
  · have variablesLe : variableCount <= 18 := by
      omega
    have powerLe : 2 ^ variableCount <= 2 ^ 18 :=
      Nat.pow_le_pow_of_le (by decide) variablesLe
    rw [liveBlockCount_exact] at covers
    have powerExact : 2 ^ 18 = 262_144 := by
      decide
    rw [powerExact] at powerLe
    omega
  · exact enough

/-- The current row cube has enough points for every emitted Step row. -/
theorem currentRowCube_covers :
    19_859_562 <= 2 ^ dimensions.rowVariables := by
  decide

/-- The selected row cube is nonempty without a caller-supplied premise. -/
theorem rowNonempty : 0 < shape.rowVariables := by
  decide

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.CurrentM4Domain
