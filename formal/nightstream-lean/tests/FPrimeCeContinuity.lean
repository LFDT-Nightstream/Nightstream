import Nightstream.Implementation.R1CS.Correspondence.FPrime.FPrimeCeContinuitySound

namespace NightstreamTests.FPrimeCeContinuity

open Nightstream.Implementation.R1CS.FPrimeCeContinuity

set_option maxRecDepth 32768

example : pairRuns.length = 8 := by decide
example : columnPairs.length = 1297 := by decide
example : continuityRows.length = 1297 := by decide
example : rowCount - continuityRowCount = 6 := by decide

end NightstreamTests.FPrimeCeContinuity
