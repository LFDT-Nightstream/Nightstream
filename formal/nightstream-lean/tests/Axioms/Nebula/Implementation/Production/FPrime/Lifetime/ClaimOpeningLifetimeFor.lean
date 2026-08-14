import Nightstream.Implementation.Nebula.Production.FPrime.Lifetime.ClaimOpeningLifetimeFor
import tests.Axioms.Support

/-! Fail-closed dependency audit for producer-derived same-witness commitment
authority over the complete delayed F-prime lifetime. -/

/-- info: 'Nightstream.Implementation.Nebula.ProductionPaperClaimOpeningLifetimeFor.ClaimLifetime.ClaimOpening.bundleOpens' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionPaperClaimOpeningLifetimeFor.ClaimLifetime.ClaimOpening.bundleOpens

/-- info: 'Nightstream.Implementation.Nebula.ProductionPaperClaimOpeningLifetimeFor.ClaimLifetime.ReceiptOpening.claimHolds' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionPaperClaimOpeningLifetimeFor.ClaimLifetime.ReceiptOpening.claimHolds

/-- info: 'Nightstream.Implementation.Nebula.ProductionPaperClaimOpeningLifetimeFor.ClaimLifetime.ReceiptOpening.exactDecodedBranch' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionPaperClaimOpeningLifetimeFor.ClaimLifetime.ReceiptOpening.exactDecodedBranch

/-- info: 'Nightstream.Implementation.Nebula.ProductionPaperClaimOpeningLifetimeFor.ClaimLifetime.Lifetime.exactOpenings' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionPaperClaimOpeningLifetimeFor.ClaimLifetime.Lifetime.exactOpenings

/-- info: 'Nightstream.Implementation.Nebula.ProductionPaperClaimOpeningLifetimeFor.ClaimLifetime.Lifetime.everyConsumedBundleOpens' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionPaperClaimOpeningLifetimeFor.ClaimLifetime.Lifetime.everyConsumedBundleOpens

/-- info: 'Nightstream.Implementation.Nebula.ProductionPaperClaimOpeningLifetimeFor.ClaimLifetime.Lifetime.everyConsumedClaimHolds' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionPaperClaimOpeningLifetimeFor.ClaimLifetime.Lifetime.everyConsumedClaimHolds

/-- info: 'Nightstream.Implementation.Nebula.ProductionPaperClaimOpeningLifetimeFor.SemanticLifetime.LifetimeExtraction.precommitChainWithOpenings' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionPaperClaimOpeningLifetimeFor.SemanticLifetime.LifetimeExtraction.precommitChainWithOpenings
