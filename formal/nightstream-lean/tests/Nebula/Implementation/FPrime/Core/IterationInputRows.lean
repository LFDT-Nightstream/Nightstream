import Nightstream.Implementation.Nebula.FPrime.Core.IterationInputRows

/-! Regression surface for the authoritative base F-prime iteration row. -/

set_option autoImplicit false

namespace tests.NebulaFPrimeIterationInputRows

open Nightstream.Implementation.Nebula

#check FPrimeIterationInputRows.rows_length
#check FPrimeIterationInputRows.sound
#check FPrimeIterationInputRows.complete

end tests.NebulaFPrimeIterationInputRows
