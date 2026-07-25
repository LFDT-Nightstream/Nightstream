import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ProductionHashCallBoundary

/-!
Focused elaboration boundary for the totalized production hash wrapper and
the impossibility of an always-present digest-only refinement.
-/

namespace NightstreamTests.FPrimeProductionHashCallBoundary

open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ProductionHashCallBoundary

#check absentCurrentPreimage_not_aligned
#check paperHash_absentCurrent
#check paperHash_eq_none_of_not_aligned
#check paperHash_eq_none_iff
#check absentCurrent_encoding_exact
#check paperHash_encoding_eq_absent_iff
#check alignedCurrent_encoding_exact
#check no_nonoptionalCoreRefines

end NightstreamTests.FPrimeProductionHashCallBoundary
