import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.RewriteSourceSemantics.Core
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.RewriteSourceSemantics.ChainAgreement

/-!
Public semantic boundary for combined-NC source/rewrite comparison.

Owns: the model-level source-definition semantics and closed-chain agreement
step consumed by the forthcoming generated batch and dependency schedule.

Does not own: generated certificate truth, selected-row satisfaction,
source-row satisfaction, production assignment authority, transcript order,
commitment binding, costs, or row removal.

Emits constraints: none; facade only.

Assurance tier: model-level.
-/
