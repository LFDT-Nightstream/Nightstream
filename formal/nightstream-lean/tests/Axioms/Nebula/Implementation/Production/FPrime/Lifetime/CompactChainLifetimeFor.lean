import Nightstream.Implementation.Nebula.Production.FPrime.Lifetime.CompactChainLifetimeFor
import tests.Axioms.Support

/-! Fail-closed dependency audit for exact compact-chain lifetime extraction. -/

/-- info: 'Nightstream.Implementation.Nebula.ProductionPaperCompactChainLifetimeFor.ClaimLifetime.Lifetime.compactExact' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionPaperCompactChainLifetimeFor.ClaimLifetime.Lifetime.compactExact

/-- info: 'Nightstream.Implementation.Nebula.ProductionPaperCompactChainLifetimeFor.rootRun_of_consumes_to_closed' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionPaperCompactChainLifetimeFor.rootRun_of_consumes_to_closed

/-- info: 'Nightstream.Implementation.Nebula.ProductionPaperCompactChainLifetimeFor.ClaimLifetime.ChainExact.ofRows' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionPaperCompactChainLifetimeFor.ClaimLifetime.ChainExact.ofRows

/-- info: 'Nightstream.Implementation.Nebula.ProductionPaperCompactChainLifetimeFor.ClaimLifetime.SegmentExact.precommitExact' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionPaperCompactChainLifetimeFor.ClaimLifetime.SegmentExact.precommitExact

/-- info: 'Nightstream.Implementation.Nebula.ProductionPaperCompactChainLifetimeFor.SemanticLifetime.LifetimeExtraction.compactChain' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionPaperCompactChainLifetimeFor.SemanticLifetime.LifetimeExtraction.compactChain

/-- info: 'Nightstream.Implementation.Nebula.ProductionPaperCompactChainLifetimeFor.SemanticLifetime.LifetimeExtraction.precommitChain' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionPaperCompactChainLifetimeFor.SemanticLifetime.LifetimeExtraction.precommitChain

/-- info: 'Nightstream.Implementation.Nebula.ProductionPaperCompactChainLifetimeFor.ClaimLifetime.LaneStepExact.after_eq_next' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionPaperCompactChainLifetimeFor.ClaimLifetime.LaneStepExact.after_eq_next

/-- info: 'Nightstream.Implementation.Nebula.ProductionPaperCompactChainLifetimeFor.ClaimLifetime.VerifiedBundleSequence.chainRoot_framedSequence' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionPaperCompactChainLifetimeFor.ClaimLifetime.VerifiedBundleSequence.chainRoot_framedSequence

/-- info: 'Nightstream.Implementation.Nebula.ProductionPaperCompactChainLifetimeFor.ClaimLifetime.VerifiedBundleSequence.run_eq_root' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionPaperCompactChainLifetimeFor.ClaimLifetime.VerifiedBundleSequence.run_eq_root

/-- info: 'Nightstream.Implementation.Nebula.ProductionPaperCompactChainLifetimeFor.ClaimLifetime.SegmentPrecommitExact.knownPrecommit_correct' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionPaperCompactChainLifetimeFor.ClaimLifetime.SegmentPrecommitExact.knownPrecommit_correct
