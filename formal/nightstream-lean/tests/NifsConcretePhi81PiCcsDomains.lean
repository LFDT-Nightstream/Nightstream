import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.PiCcsDomains

/-! Focused regressions for the concrete NIFS shared Π_CCS dimensions. -/

namespace NightstreamTests.NifsConcretePhi81PiCcsDomains

open Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81

#check PiCcsDomains.production_fe
#check PiCcsDomains.production_nc
#check PiCcsDomains.publicPrefix_fe
#check PiCcsDomains.publicPrefix_nc

example : PiCcsDomains.publicPrefix.fe = PiCcsDomain.domain :=
  PiCcsDomains.publicPrefix_fe

example : PiCcsDomains.publicPrefix.nc = PiCcsDomain.blockDomain :=
  PiCcsDomains.publicPrefix_nc

example : PiCcsDomains.production.fe.columnVariables = 24 := by
  rfl

example : PiCcsDomains.production.nc.blockVariables = 19 := by
  rfl

example : PiCcsDomains.production.nc.laneVariables = 6 := by
  rfl

example :
    PiCcsDomains.production.fe.columnVariables +
        PiCcsDomains.production.fe.laneVariables = 30 :=
  PiCcsDomains.fixedPointProduction_flatRoundCount

example :
    PiCcsDomains.production.nc.blockVariables +
        PiCcsDomains.production.nc.laneVariables = 25 :=
  PiCcsDomains.fixedPointProduction_blockRoundCount

end NightstreamTests.NifsConcretePhi81PiCcsDomains
