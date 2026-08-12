import Nightstream.Protocol.NebulaV2.ProductionBatchedDelayedReverse
import tests.Axioms.Support

set_option autoImplicit false

namespace tests.Axioms.NebulaV2ProductionBatchedDelayedReverse

open Nightstream.Protocol.NebulaV2.ProductionBatchedDelayedReverse

/-- info: 'Nightstream.Protocol.NebulaV2.ProductionBatchedDelayedReverse.delayedRun_to_segmentChain' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms delayedRun_to_segmentChain

/-- info: 'Nightstream.Protocol.NebulaV2.ProductionBatchedDelayedReverse.segmentChain_iff_delayedRun' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms segmentChain_iff_delayedRun

end tests.Axioms.NebulaV2ProductionBatchedDelayedReverse
