import Nightstream.Implementation.R1CS.Artifacts.FPrimeRecursive.PiRlcProjection.BetaLadder

/-!
Public regressions for the active shared PiRLC beta-ladder artifact.

| Test family | Obligation | Expected result |
|---|---|---|
| small schema | two powers reconstruct two base rows plus one K-mul | accepted |
| bad beta input | beta must predate the ladder allocation | rejected |
| fixed profile | 55 powers, 272 exact rows, exact `y_zcol` prefix | exact equality |
-/

namespace NightstreamTests.FPrimeRecursivePiRlcProjectionBetaLadder

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcProjection.BetaLadder

def smallOwner : PiRlcProjectionBetaLadderOwner where
  stagePath := "test.pi_rlc.beta_ladder"
  rowStart := 10
  rowEnd := 17
  allocatedStart := 100
  allocatedEnd := 107
  betaColumns := { c0 := 50, c1 := 51 }
  powerColumns := [{ c0 := 100, c1 := 101 }, { c0 := 105, c1 := 106 }]

example : smallOwner.Valid 2 := by
  decide

example : ¬ ({ smallOwner with
    betaColumns := { c0 := 100, c1 := 51 } }).Valid 2 := by
  decide

example :
    FPrimeRecursivePiRlcProjectionBetaLadderData.powerColumns.length = 55 :=
  power_count

example : ownedRowDefinitions.length = 272 := owned_row_count

example : SourceRowsMatch := source_rows_match

example :
    FPrimeRecursivePiRlcProjectionBetaLadderData.powerColumns.take
        FPrimeRecursiveYZcolProjectionData.activeLaneCount =
      FPrimeRecursiveYZcolProjectionData.sharedPowerColumns :=
  y_zcol_power_prefix

end NightstreamTests.FPrimeRecursivePiRlcProjectionBetaLadder
