import Nightstream.Protocol.FPrime.Frozen.PiCcsAsymptoticObstruction
import tests.Axioms.Support

/-! Fail-closed guards for the exact frozen PiCCS asymptotic obstruction. -/

/-- info: 'Nightstream.Protocol.FPrime.Frozen.PiCcsAsymptoticObstruction.piCcsStrong_iff_runtime' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.PiCcsAsymptoticObstruction.piCcsStrong_iff_runtime

/-- info: 'Nightstream.Protocol.FPrime.Frozen.PiCcsAsymptoticObstruction.frozenTarget_without_samplerLink_countermodel' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.PiCcsAsymptoticObstruction.frozenTarget_without_samplerLink_countermodel

/-- info: 'Nightstream.Protocol.FPrime.Frozen.PiCcsAsymptoticObstruction.not_attemptedBridgeWithoutSamplerLink' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.PiCcsAsymptoticObstruction.not_attemptedBridgeWithoutSamplerLink
