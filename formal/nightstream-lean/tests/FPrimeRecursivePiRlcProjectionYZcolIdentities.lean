import Nightstream.Implementation.R1CS.Artifacts.FPrimeRecursive.PiRlcProjection.YZcolIdentities

/-!
Public regressions for the complete active PiRLC `y_zcol` identity artifacts.

| Test family | Obligation | Expected result |
|---|---|---|
| physical ownership | two contiguous identities, 3,616 newly owned rows | exact equality |
| shared ownership | beta and rho rows precede the first identity | no local recharge |
| generated shards | six input shards, two tail shards, and both checks match their reconstructed rows | accepted |
| trace shape | both degree-106 traces satisfy the projection layout contract | accepted |
-/

namespace NightstreamTests.FPrimeRecursivePiRlcProjectionYZcolIdentities

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcProjection.YZcolIdentities

example : StructureValid := structure_check

example : newLocalSourceRows.length = 3616 := new_local_row_count

example : traces.length = 2 := traces_count

example :
    ladderOwner.rowEnd ≤
        FPrimeRecursivePiRlcProjection.RhoEvaluations.generatedStageRowStart ∧
      FPrimeRecursivePiRlcProjection.RhoEvaluations.generatedStageRowEnd ≤
        limb0Owner.identityRowStart :=
  shared_rows_precede_identity_rows

example : Limb0InputRowsMatch := limb0_input_rows_match

example : Limb1InputRowsMatch := limb1_input_rows_match

example : TailRowsMatch := tail_rows_match

example : CheckRowsMatch := check_rows_match

example : limb0Trace.LayoutValid := limb0_layout

example : limb1Trace.LayoutValid := limb1_layout

end NightstreamTests.FPrimeRecursivePiRlcProjectionYZcolIdentities
