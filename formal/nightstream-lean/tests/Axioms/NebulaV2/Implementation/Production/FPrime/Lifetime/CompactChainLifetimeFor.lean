import Nightstream.Implementation.NebulaV2.Production.FPrime.Lifetime.CompactChainLifetimeFor
import tests.Axioms.Support

/-! Fail-closed dependency audit for exact compact-chain lifetime extraction. -/

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionPaperCompactChainLifetimeFor.ClaimLifetime.Lifetime.compactExact' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionPaperCompactChainLifetimeFor.ClaimLifetime.Lifetime.compactExact

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionPaperCompactChainLifetimeFor.rootRun_of_consumes_to_closed' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionPaperCompactChainLifetimeFor.rootRun_of_consumes_to_closed

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionPaperCompactChainLifetimeFor.ClaimLifetime.ChainExact.ofRows' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionPaperCompactChainLifetimeFor.ClaimLifetime.ChainExact.ofRows

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionPaperCompactChainLifetimeFor.ClaimLifetime.SegmentExact.precommitExact' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionPaperCompactChainLifetimeFor.ClaimLifetime.SegmentExact.precommitExact

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionPaperCompactChainLifetimeFor.SemanticLifetime.LifetimeExtraction.compactChain' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionPaperCompactChainLifetimeFor.SemanticLifetime.LifetimeExtraction.compactChain

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionPaperCompactChainLifetimeFor.SemanticLifetime.LifetimeExtraction.precommitChain' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionPaperCompactChainLifetimeFor.SemanticLifetime.LifetimeExtraction.precommitChain

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionPaperCompactChainLifetimeFor.ClaimLifetime.LaneStepExact.after_eq_next' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionPaperCompactChainLifetimeFor.ClaimLifetime.LaneStepExact.after_eq_next

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionPaperCompactChainLifetimeFor.ClaimLifetime.VerifiedBundleSequence.chainRoot_framedSequence' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionPaperCompactChainLifetimeFor.ClaimLifetime.VerifiedBundleSequence.chainRoot_framedSequence

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionPaperCompactChainLifetimeFor.ClaimLifetime.VerifiedBundleSequence.run_eq_root' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionPaperCompactChainLifetimeFor.ClaimLifetime.VerifiedBundleSequence.run_eq_root

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionPaperCompactChainLifetimeFor.ClaimLifetime.SegmentPrecommitExact.knownPrecommit_correct' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionPaperCompactChainLifetimeFor.ClaimLifetime.SegmentPrecommitExact.knownPrecommit_correct
