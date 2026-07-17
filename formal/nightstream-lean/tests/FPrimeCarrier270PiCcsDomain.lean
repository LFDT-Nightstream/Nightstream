import Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.PiCcsDomain

/-! Focused checks for the independently derived plain F-prime NC domain. -/

namespace tests.FPrimeCarrier270PiCcsDomain

open Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.PiCcsDomain

#check alignedWidth_eq_legacy_add_padding
#check plainShape_carrierWidth
#check domain_columnVariables
#check domain_laneVariables
#check domain_columnCount
#check domain_laneCount
#check domain_covers
#check columnVariables_minimal
#check laneVariables_minimal
#check domain_variableCount
#check blockDomain_blockVariables
#check blockDomain_laneVariables
#check blockDomain_blockCount
#check blockDomain_laneCount
#check blockDomain_covers
#check blockVariables_minimal
#check blockDomain_laneVariables_minimal
#check blockDomain_variableCount

example : domain.columnVariables = 9 := domain_columnVariables
example : domain.laneVariables = 6 := domain_laneVariables
example : domain.columnVariables + domain.laneVariables = 15 :=
  domain_variableCount
example : blockDomain.blockVariables = 3 := blockDomain_blockVariables
example : blockDomain.laneVariables = 6 := blockDomain_laneVariables
example : blockDomain.blockVariables + blockDomain.laneVariables = 9 :=
  blockDomain_variableCount

end tests.FPrimeCarrier270PiCcsDomain
