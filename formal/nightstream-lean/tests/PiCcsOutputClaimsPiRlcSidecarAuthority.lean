import Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism.Authority.PiRlcSidecar

/-!
Focused regressions for the conditional packed `Pi_RLC` sidecar authority
theorem.

| Stage path | Property under test | Failure caught |
|---|---|---|
| `nifs.pi_rlc.verify.authority.packed_y_zcol.aggregate` | source claims enter one exact finite challenge fold | digest or alternate fold promoted to authority |
| `nifs.pi_rlc.verify.authority.packed_y_zcol.parent_projection` | the combined claim matches the proposed parent projection | semantic equality mistaken for opening evidence |
| `nifs.pi_rlc.verify.authority.packed_y_zcol.parent_assignment` | the parent is the canonical assignment fold | unrelated valid parent accepted |
| `nifs.pi_rlc.verify.authority.packed_y_zcol.exact` | source truth or collision is exactly canonical aggregate equality | exposed intermediates mistaken for irreducible obligations |
| `nifs.pi_rlc.verify.authority.packed_y_zcol.reduce` | disagreement remains a named mixing collision | unconditional sourcewise authority |
-/

namespace NightstreamTests.PiCcsOutputClaimsPiRlcSidecarAuthority

open Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism.Authority.PiRlcSidecar

#check SourceBound
#check AggregateEquation
#check ParentProjectionMatches
#check ParentAssignmentMatches
#check CanonicalAggregateEquality
#check MixingCollision
#check sourceBound_or_mixingCollision_iff_aggregateEquality
#check sourceBound_or_mixingCollision

end NightstreamTests.PiCcsOutputClaimsPiRlcSidecarAuthority
