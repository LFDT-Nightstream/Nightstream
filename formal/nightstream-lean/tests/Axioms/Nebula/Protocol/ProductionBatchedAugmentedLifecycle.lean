import Nightstream.Protocol.Nebula.ProductionBatchedAugmentedLifecycle
import tests.Axioms.Support

set_option autoImplicit false

namespace tests.Axioms.NebulaProductionBatchedAugmentedLifecycle

open Nightstream.Protocol.Nebula.ProductionBatchedAugmentedLifecycle

/-- info: 'Nightstream.Protocol.Nebula.ProductionBatchedAugmentedLifecycle.DelayedRun.finish_segment' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms DelayedRun.finish_segment

/-- info: 'Nightstream.Protocol.Nebula.ProductionBatchedAugmentedLifecycle.DelayedRun.prepend_segment' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms DelayedRun.prepend_segment

/-- info: 'Nightstream.Protocol.Nebula.ProductionBatchedAugmentedLifecycle.SegmentChain.toDelayedRun' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms SegmentChain.toDelayedRun

/-- info: 'Nightstream.Protocol.Nebula.ProductionBatchedAugmentedLifecycle.CompleteRun.exact_delayed_lifecycle' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms CompleteRun.exact_delayed_lifecycle

/-- info: 'Nightstream.Protocol.Nebula.ProductionBatchedAugmentedLifecycle.CompleteRun.application_executes' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms CompleteRun.application_executes

/-- info: 'Nightstream.Protocol.Nebula.ProductionBatchedAugmentedLifecycle.CompleteRun.final_application_valid' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms CompleteRun.final_application_valid

end tests.Axioms.NebulaProductionBatchedAugmentedLifecycle
