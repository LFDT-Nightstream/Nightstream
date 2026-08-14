import Nightstream.Protocol.Nebula.ProductionBatchedDelayedReverse
import tests.Axioms.Support

set_option autoImplicit false

namespace tests.Axioms.NebulaProductionBatchedDelayedReverse

open Nightstream.Protocol.Nebula.ProductionBatchedDelayedReverse

/-- info: 'Nightstream.Protocol.Nebula.ProductionBatchedDelayedReverse.delayedRun_to_segmentChain' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms delayedRun_to_segmentChain

/-- info: 'Nightstream.Protocol.Nebula.ProductionBatchedDelayedReverse.segmentChain_iff_delayedRun' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms segmentChain_iff_delayedRun

end tests.Axioms.NebulaProductionBatchedDelayedReverse
