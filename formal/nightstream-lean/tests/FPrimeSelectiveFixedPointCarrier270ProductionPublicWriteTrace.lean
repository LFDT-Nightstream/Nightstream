import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.Carrier270
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.ActivePublicWriteExecution

/-! Focused interface regression for the active Carrier270 public-write trace. -/

namespace Nightstream.Tests.FPrimeSelectiveFixedPointCarrier270ProductionPublicWriteTrace

open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270

#check ProductionPublicWriteTrace.productionProfile_exact
#check ProductionPublicWriteTrace.generated_chunk0_exact
#check ProductionPublicWriteTrace.generated_chunk1_exact
#check ProductionPublicWriteTrace.productionTrace_certificate
#check ProductionPublicWriteTrace.production_projectPhysical270_execute_eq_projectPublicInput
#check PiCcsNc.DelayedProjection.CombinedNc.Materialized.ActivePublicWriteExecution.execution_activePublicWritesBound
#check PiCcsNc.DelayedProjection.CombinedNc.Materialized.ActivePublicWriteExecution.execution_normalizedPublicInput_eq_projectPublicInput

example : ProductionPublicWriteTrace.productionArm = 2 /\
    ProductionPublicWriteTrace.shardWidth = 135 /\
    PublicDecoder.alignedPublicWidth = 270 := by
  decide

example :
    PublicWriteTrace.PendingProductionExporterCertificate
      ProductionPublicWriteTrace.productionArm
      ProductionPublicWriteTrace.productionTrace :=
  ProductionPublicWriteTrace.productionTrace_certificate

end Nightstream.Tests.FPrimeSelectiveFixedPointCarrier270ProductionPublicWriteTrace
