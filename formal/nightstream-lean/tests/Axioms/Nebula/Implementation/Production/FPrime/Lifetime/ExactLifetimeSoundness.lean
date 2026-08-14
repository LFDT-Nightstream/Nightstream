import Nightstream.Implementation.Nebula.Production.FPrime.Lifetime.ExactLifetimeSoundness
import tests.Axioms.Support

/-! Dependency audit for the exponent-indexed F-prime soundness bridge. -/

/-- info: 'Nightstream.Implementation.Nebula.ProductionPaperExactLifetimeSoundness.CertifiedExecution.baseChallengeAuthorityExact' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionPaperExactLifetimeSoundness.CertifiedExecution.baseChallengeAuthorityExact

/-- info: 'Nightstream.Implementation.Nebula.ProductionPaperExactLifetimeSoundness.CertifiedExecution.exactClaimSchedule' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionPaperExactLifetimeSoundness.CertifiedExecution.exactClaimSchedule

/-- info: 'Nightstream.Implementation.Nebula.ProductionPaperExactLifetimeSoundness.CertifiedExecution.fixedBranchSchedule' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionPaperExactLifetimeSoundness.CertifiedExecution.fixedBranchSchedule

/-- info: 'Nightstream.Implementation.Nebula.ProductionPaperExactLifetimeSoundness.CertifiedExecution.receiptsExact' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionPaperExactLifetimeSoundness.CertifiedExecution.receiptsExact

/-- info: 'Nightstream.Implementation.Nebula.ProductionPaperExactLifetimeSoundness.CertifiedExecution.consumerInvocationIndicesExact' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionPaperExactLifetimeSoundness.CertifiedExecution.consumerInvocationIndicesExact

/-- info: 'Nightstream.Implementation.Nebula.ProductionPaperExactLifetimeSoundness.CertifiedExecution.fullStateContinuityExact' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionPaperExactLifetimeSoundness.CertifiedExecution.fullStateContinuityExact

/-- info: 'Nightstream.Implementation.Nebula.ProductionPaperExactLifetimeSoundness.CertifiedExecution.exactClaimOpenings' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionPaperExactLifetimeSoundness.CertifiedExecution.exactClaimOpenings

/-- info: 'Nightstream.Implementation.Nebula.ProductionPaperExactLifetimeSoundness.CertifiedExecution.everyConsumedBundleOpens' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionPaperExactLifetimeSoundness.CertifiedExecution.everyConsumedBundleOpens

/-- info: 'Nightstream.Implementation.Nebula.ProductionPaperExactLifetimeSoundness.CertifiedExecution.everyConsumedClaimHolds' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionPaperExactLifetimeSoundness.CertifiedExecution.everyConsumedClaimHolds

/-- info: 'Nightstream.Implementation.Nebula.ProductionPaperExactLifetimeSoundness.exact_lifetime_implies_certificate_or_failure' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionPaperExactLifetimeSoundness.exact_lifetime_implies_certificate_or_failure

/-- info: 'Nightstream.Implementation.Nebula.ProductionPaperExactLifetimeSoundness.exact_lifetime_implies_execution_or_failure' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionPaperExactLifetimeSoundness.exact_lifetime_implies_execution_or_failure
