import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.RewriteSourceSemantics
import tests.Axioms.Support

/-! Fail-closed dependency gate for model-level closed rewrite-chain agreement. -/

open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.RewriteSourceSemantics.ChainAgreement

/-- info: 'Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.RewriteSourceSemantics.ChainAgreement.exactChainMatch_implies_sourceValue_eq_compiler_of_contributionsEqual' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms exactChainMatch_implies_sourceValue_eq_compiler_of_contributionsEqual
