import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ProductionXOutSponge23InputAlignment

/-!
Focused elaboration boundary for exact plain/stateless XOut source-vector
alignment with the selected fused 23-field Poseidon2 sponge recipe.
-/

namespace NightstreamTests.FPrimeProductionXOutSponge23InputAlignment

open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding
open ProductionXOutSponge23InputAlignment

#check Source.fields_eq_encodeStateXOutPreimage
#check Source.fields_length
#check Source.emptyTable_wellFormed_but_sourceWidth_seven
#check numericInputs_eq_sourceFields
#check semanticLane_eq_sourceLane
#check active_sound
#check active_complete

end NightstreamTests.FPrimeProductionXOutSponge23InputAlignment
