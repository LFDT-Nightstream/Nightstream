import Nightstream.Implementation.R1CS.Artifacts.FPrimeRecursive.YZcolProjection

/-!
Public regressions for the fixed-profile parent `y_zcol` output-evaluation
ownership tree.

Owns: small fail-closed schema mutations and stable-facade shape checks.

Does not own: artifact generation, Rust conformance, semantic parent authority,
transcript timing, projection soundness, bad-root bounds, or row removal.

Emits constraints: no.

| Test family | Mathematical obligation | Expected result |
|---|---|---|
| small schema | one two-coefficient evaluator has four exact equation rows | accepted |
| malformed output | an output outside the allocated tail is rejected | rejected |
| fixed profile | two ordered 54-lane leaves own exactly 216 rows | exact equality |
-/

namespace NightstreamTests.FPrimeRecursiveYZcolProjection

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.FPrimeRecursiveYZcolProjection

def smallOwner : YZcolOutputEvaluationOwner where
  stagePath := "test.y_zcol.output.limb0"
  identityIndex := 7
  limb := 0
  identityRowStart := 10
  identityRowEnd := 30
  evaluationRowStart := 20
  evaluationRowEnd := 24
  evaluationAllocatedStart := 100
  evaluationAllocatedEnd := 104
  parentCoefficientColumns := [40, 41]
  powerColumns := [{ c0 := 50, c1 := 51 }, { c0 := 52, c1 := 53 }]
  evaluationOutputColumns := { c0 := 102, c1 := 103 }

example : smallOwner.Valid 2 := by
  decide

example : ¬ ({ smallOwner with
    evaluationOutputColumns := { c0 := 101, c1 := 103 } }).Valid 2 := by
  decide

example :
    FPrimeRecursiveYZcolProjectionData.owners.length = 2 := owner_count

example :
    FPrimeRecursiveYZcolProjectionData.owners.map (·.limb) = [0, 1] :=
  owner_limb_order

example : ownedRowDefinitions.length = 216 := owned_row_count

example : SourceRowsMatch := source_rows_match

example :
    (FPrimeRecursiveYZcolProjectionData.sourceRows.map Prod.fst).Nodup :=
  source_rows_distinct

end NightstreamTests.FPrimeRecursiveYZcolProjection
