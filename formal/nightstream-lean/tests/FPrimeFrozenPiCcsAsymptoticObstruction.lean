import Nightstream.Protocol.FPrime.Frozen.PiCcsAsymptoticObstruction

/-! Kernel checks for the exact frozen PiCCS asymptotic obstruction. -/

namespace Nightstream.Tests.FPrimeFrozenPiCcsAsymptoticObstruction

open Nightstream.Protocol.FPrime.Frozen.Obligations
open Nightstream.Protocol.FPrime.Frozen.PiCcsAsymptoticObstruction

example : PiCcsStrong (games True) :=
  (piCcsStrong_iff_runtime True).2 True.intro

example : ¬ EventuallySucceeds (sharedCoinRetry false) :=
  sharedCoinRetry_false_has_no_success

#check frozenTarget_without_samplerLink_countermodel
#check not_attemptedBridgeWithoutSamplerLink

end Nightstream.Tests.FPrimeFrozenPiCcsAsymptoticObstruction
