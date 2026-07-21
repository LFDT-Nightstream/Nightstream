import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.Accumulator

/-! Focused interface regression for the pending-family accumulator codec. -/

namespace Nightstream.Tests.FPrimeSelectiveFixedPointAccumulatorPendingFamilyCodec

open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.Accumulator.PendingFamilyCodec

#check encodeChild_injective
#check encodeChildren_injective
#check encodePending_injective
#check encodeCarrier_injective
#check semanticPayload_columnPoint_eq_of_encode_eq
#check encodeWithPrefix_injective
#check encodeProductionCarrier_injective
#check fixed_child_field_count
#check fixed_carrier_field_count
#check bounded_field_count
#check production_field_count
#check pendingFamily_field_saving

example (shape : Shape)
    (rowVariables : shape.rowVariables = 24)
    (publicWidth : shape.publicWidth = 270)
    (matrixCount : shape.matrixCount = 13) :
    productionHeader.length + carrierFieldCount shape 4 14 = 26711 :=
  bounded_field_count shape rowVariables publicWidth matrixCount

example (shape : Shape)
    (rowVariables : shape.rowVariables = 24)
    (publicWidth : shape.publicWidth = 270)
    (matrixCount : shape.matrixCount = 13) :
    productionHeader.length + carrierFieldCount shape 18 14 = 37295 :=
  production_field_count shape rowVariables publicWidth matrixCount

example : conservativeFamilyFieldCount 18 - 37295 = 5749 := by
  simpa using pendingFamily_field_saving 18

end Nightstream.Tests.FPrimeSelectiveFixedPointAccumulatorPendingFamilyCodec
