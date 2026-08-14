import Nightstream.Implementation.Nebula.FPrime.Claim.ManifestTrace
import tests.Axioms.Support

/-! Fail-closed dependency guard for the paired full-claim manifest trace. -/

/-- info: 'Nightstream.Implementation.Nebula.FullClaimManifestTrace.TerminalNode.producerCarries' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.FullClaimManifestTrace.TerminalNode.producerCarries

/-- info: 'Nightstream.Implementation.Nebula.FullClaimManifestTrace.TerminalNode.trailingLink' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.FullClaimManifestTrace.TerminalNode.trailingLink

/-- info: 'Nightstream.Implementation.Nebula.FullClaimManifestTrace.Candidate.toDelayed' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.FullClaimManifestTrace.Candidate.toDelayed

/-- info: 'Nightstream.Implementation.Nebula.FullClaimManifestTrace.Candidate.toManifest' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.FullClaimManifestTrace.Candidate.toManifest

/-- info: 'Nightstream.Implementation.Nebula.FullClaimManifestTrace.Candidate.authoritySoundOrCollision' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.FullClaimManifestTrace.Candidate.authoritySoundOrCollision

/-- info: 'Nightstream.Implementation.Nebula.FullClaimManifestTrace.Candidate.exactProducerCount' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.FullClaimManifestTrace.Candidate.exactProducerCount

/-- info: 'Nightstream.Implementation.Nebula.FullClaimManifestTrace.ExactChain.exactConsumerCount' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.FullClaimManifestTrace.ExactChain.exactConsumerCount

/-- info: 'Nightstream.Implementation.Nebula.FullClaimManifestTrace.ExactChain.exactProducerCount' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.FullClaimManifestTrace.ExactChain.exactProducerCount

/-- info: 'Nightstream.Implementation.Nebula.FullClaimManifestTrace.ExactChain.completeDelayedSchedule' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.FullClaimManifestTrace.ExactChain.completeDelayedSchedule

/-- info: 'Nightstream.Implementation.Nebula.FullClaimManifestTrace.ExactChain.tailProducersAreRecursive' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.FullClaimManifestTrace.ExactChain.tailProducersAreRecursive

/-- info: 'Nightstream.Implementation.Nebula.FullClaimManifestTrace.ExactChain.exactBranchSchedule' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.FullClaimManifestTrace.ExactChain.exactBranchSchedule

/-- info: 'Nightstream.Implementation.Nebula.FullClaimManifestTrace.ExactChain.authoritySoundOrCollision' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.FullClaimManifestTrace.ExactChain.authoritySoundOrCollision
