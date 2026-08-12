import Nightstream.Implementation.NebulaV2.ProductionPaperExactLifetimeSoundness
import tests.Axioms.Support

/-! Dependency audit for the exponent-indexed F-prime soundness bridge. -/

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionPaperExactLifetimeSoundness.CertifiedExecution.baseChallengeAuthorityExact' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionPaperExactLifetimeSoundness.CertifiedExecution.baseChallengeAuthorityExact

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionPaperExactLifetimeSoundness.CertifiedExecution.exactClaimSchedule' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionPaperExactLifetimeSoundness.CertifiedExecution.exactClaimSchedule

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionPaperExactLifetimeSoundness.CertifiedExecution.fixedBranchSchedule' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionPaperExactLifetimeSoundness.CertifiedExecution.fixedBranchSchedule

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionPaperExactLifetimeSoundness.CertifiedExecution.receiptsExact' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionPaperExactLifetimeSoundness.CertifiedExecution.receiptsExact

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionPaperExactLifetimeSoundness.CertifiedExecution.consumerInvocationIndicesExact' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionPaperExactLifetimeSoundness.CertifiedExecution.consumerInvocationIndicesExact

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionPaperExactLifetimeSoundness.CertifiedExecution.fullStateContinuityExact' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionPaperExactLifetimeSoundness.CertifiedExecution.fullStateContinuityExact

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionPaperExactLifetimeSoundness.CertifiedExecution.exactClaimOpenings' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionPaperExactLifetimeSoundness.CertifiedExecution.exactClaimOpenings

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionPaperExactLifetimeSoundness.CertifiedExecution.everyConsumedBundleOpens' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionPaperExactLifetimeSoundness.CertifiedExecution.everyConsumedBundleOpens

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionPaperExactLifetimeSoundness.CertifiedExecution.everyConsumedClaimHolds' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionPaperExactLifetimeSoundness.CertifiedExecution.everyConsumedClaimHolds

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionPaperExactLifetimeSoundness.exact_lifetime_implies_certificate_or_failure' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionPaperExactLifetimeSoundness.exact_lifetime_implies_certificate_or_failure

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionPaperExactLifetimeSoundness.exact_lifetime_implies_execution_or_failure' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionPaperExactLifetimeSoundness.exact_lifetime_implies_execution_or_failure
