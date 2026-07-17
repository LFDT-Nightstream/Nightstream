import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.PiCcsDomains

/-! Focused regressions for the concrete NIFS shared Π_CCS dimensions. -/

namespace NightstreamTests.NifsConcretePhi81PiCcsDomains

open Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81

#check PiCcsDomains.production_fe
#check PiCcsDomains.production_nc

example : PiCcsDomains.production.fe = PiCcsDomain.domain :=
  PiCcsDomains.production_fe

example : PiCcsDomains.production.nc = PiCcsDomain.blockDomain :=
  PiCcsDomains.production_nc

end NightstreamTests.NifsConcretePhi81PiCcsDomains
