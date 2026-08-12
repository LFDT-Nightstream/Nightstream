import Nightstream.Implementation.NebulaV2.FullClaimManifestTrace
import tests.Axioms.Support

/-! Fail-closed dependency guard for the paired full-claim manifest trace. -/

/-- info: 'Nightstream.Implementation.NebulaV2.FullClaimManifestTrace.TerminalNode.producerCarries' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.FullClaimManifestTrace.TerminalNode.producerCarries

/-- info: 'Nightstream.Implementation.NebulaV2.FullClaimManifestTrace.TerminalNode.trailingLink' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.FullClaimManifestTrace.TerminalNode.trailingLink

/-- info: 'Nightstream.Implementation.NebulaV2.FullClaimManifestTrace.Candidate.toDelayed' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.FullClaimManifestTrace.Candidate.toDelayed

/-- info: 'Nightstream.Implementation.NebulaV2.FullClaimManifestTrace.Candidate.toManifest' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.FullClaimManifestTrace.Candidate.toManifest

/-- info: 'Nightstream.Implementation.NebulaV2.FullClaimManifestTrace.Candidate.authoritySoundOrCollision' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.FullClaimManifestTrace.Candidate.authoritySoundOrCollision

/-- info: 'Nightstream.Implementation.NebulaV2.FullClaimManifestTrace.Candidate.exactProducerCount' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.FullClaimManifestTrace.Candidate.exactProducerCount

/-- info: 'Nightstream.Implementation.NebulaV2.FullClaimManifestTrace.ExactChain.exactConsumerCount' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.FullClaimManifestTrace.ExactChain.exactConsumerCount

/-- info: 'Nightstream.Implementation.NebulaV2.FullClaimManifestTrace.ExactChain.exactProducerCount' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.FullClaimManifestTrace.ExactChain.exactProducerCount

/-- info: 'Nightstream.Implementation.NebulaV2.FullClaimManifestTrace.ExactChain.completeDelayedSchedule' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.FullClaimManifestTrace.ExactChain.completeDelayedSchedule

/-- info: 'Nightstream.Implementation.NebulaV2.FullClaimManifestTrace.ExactChain.tailProducersAreRecursive' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.FullClaimManifestTrace.ExactChain.tailProducersAreRecursive

/-- info: 'Nightstream.Implementation.NebulaV2.FullClaimManifestTrace.ExactChain.exactBranchSchedule' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.FullClaimManifestTrace.ExactChain.exactBranchSchedule

/-- info: 'Nightstream.Implementation.NebulaV2.FullClaimManifestTrace.ExactChain.authoritySoundOrCollision' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.FullClaimManifestTrace.ExactChain.authoritySoundOrCollision
