import Nightstream.Implementation.NebulaV2.ProductionPaperClaimOpeningLifetimeFor
import tests.Axioms.Support

/-! Fail-closed dependency audit for producer-derived same-witness commitment
authority over the complete delayed F-prime lifetime. -/

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionPaperClaimOpeningLifetimeFor.ClaimLifetime.ClaimOpening.bundleOpens' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionPaperClaimOpeningLifetimeFor.ClaimLifetime.ClaimOpening.bundleOpens

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionPaperClaimOpeningLifetimeFor.ClaimLifetime.ReceiptOpening.claimHolds' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionPaperClaimOpeningLifetimeFor.ClaimLifetime.ReceiptOpening.claimHolds

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionPaperClaimOpeningLifetimeFor.ClaimLifetime.ReceiptOpening.exactDecodedBranch' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionPaperClaimOpeningLifetimeFor.ClaimLifetime.ReceiptOpening.exactDecodedBranch

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionPaperClaimOpeningLifetimeFor.ClaimLifetime.Lifetime.exactOpenings' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionPaperClaimOpeningLifetimeFor.ClaimLifetime.Lifetime.exactOpenings

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionPaperClaimOpeningLifetimeFor.ClaimLifetime.Lifetime.everyConsumedBundleOpens' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionPaperClaimOpeningLifetimeFor.ClaimLifetime.Lifetime.everyConsumedBundleOpens

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionPaperClaimOpeningLifetimeFor.ClaimLifetime.Lifetime.everyConsumedClaimHolds' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionPaperClaimOpeningLifetimeFor.ClaimLifetime.Lifetime.everyConsumedClaimHolds

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionPaperClaimOpeningLifetimeFor.SemanticLifetime.LifetimeExtraction.precommitChainWithOpenings' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionPaperClaimOpeningLifetimeFor.SemanticLifetime.LifetimeExtraction.precommitChainWithOpenings
