import Nightstream.Implementation.R1CS.Artifacts.FPrimeRecursive.PiRlcProjection.RhoEvaluations

/-!
Public regressions for the active shared PiRLC rho-evaluation artifact.

| Test family | Obligation | Expected result |
|---|---|---|
| small schema | one two-coefficient evaluator reconstructs four rows | accepted |
| bad input order | coefficient inputs must predate the evaluator allocation | rejected |
| fixed profile | 15 leaves, 1,620 exact rows, three checked shards | exact equality |
-/

namespace NightstreamTests.FPrimeRecursivePiRlcProjectionRhoEvaluations

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcProjection.RhoEvaluations

def smallOwner : PiRlcRhoEvaluationOwner where
  stagePath := "test.pi_rlc.rho_evaluation"
  pairIndex := 0
  traceIndex := 0
  rowStart := 10
  rowEnd := 14
  allocatedStart := 100
  allocatedEnd := 104
  coefficientColumns := [40, 41]
  powerColumns := [{ c0 := 50, c1 := 51 }, { c0 := 52, c1 := 53 }]
  outputColumns := { c0 := 102, c1 := 103 }

example : smallOwner.Valid 2 := by
  decide

example : ¬ ({ smallOwner with
    coefficientColumns := [100, 41] }).Valid 2 := by
  decide

example : owners.length = 15 := owner_count

example : ownedRowDefinitions.length = 1620 := owned_row_count

example : sourceRows.length = 1620 := source_row_count

example : SourceRowsMatch := source_rows_match

end NightstreamTests.FPrimeRecursivePiRlcProjectionRhoEvaluations
