import Nightstream.SuperNeo.Relations

/-! Type-level regressions for verifier-owned SuperNeo relation parameters. -/

namespace NightstreamTests.SuperNeoRelations

open Nightstream.SuperNeo

def testParams : GlobalParams where
  q := 97
  b := 2
  k := 4
  maxFresh := 3
  expansionT := 1
  rlc_bound := by decide

example : NormStage.bound testParams .fresh = 2 := rfl

example : NormStage.bound testParams .combined = 16 := rfl

/-- The ambient bound comes from verifier-owned `GlobalParams.q`; a statement
cannot provide a different modulus through its `NormStage`. -/
example : NormStage.bound testParams .ambient = 49 := rfl

end NightstreamTests.SuperNeoRelations
