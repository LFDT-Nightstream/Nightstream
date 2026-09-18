import Nightstream.Implementation.R1CS.Correspondence.FPrimeBase.FPrimeBaseProgramSound

namespace NightstreamTests.FPrimeBaseProgram

open Nightstream.Implementation.R1CS.FPrimeBaseProgram
open Nightstream.Implementation.R1CS.FPrimeBaseProgramSound

example : rowCount = 12498 := by decide
example : definitionCount + checkCount = rowCount := by decide
example : xOutColumns.length = 4 := by decide

#check fPrimeBaseProgram_sound
#check fPrimeBaseProgram_xOut_unique
#check fPrimeBaseProgram_complete

end NightstreamTests.FPrimeBaseProgram
