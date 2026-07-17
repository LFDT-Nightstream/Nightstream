import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81

/-!
Focused compile-time regressions for the exact concrete Phi81 Π_RLC sampler
boundary.

| Stage path | Property under test | Failure caught |
|---|---|---|
| `nifs.concrete.pi_rlc.sampler.machine` | the abstract four-block production schedule is specialized to Phi81 RingF assembly | unrelated sampler semantics |
| `nifs.concrete.pi_rlc.sampler.binding` | one typed batch carries both execution and exact challenge equality | unbound or digest-only challenge authority |
| `nifs.concrete.pi_rlc.challenge.membership` | production-set membership is derived from batch replay | duplicated prover assertion |
| `nifs.concrete.pi_rlc.sampler.shortfall` | successful batch execution excludes every fixed-prefix shortfall | silent fallback acceptance |
| `nifs.concrete.pi_rlc.sampler.outcome` | finite execution yields a bound challenge vector or one named shortfall coordinate | hidden totality assumption |
| `nifs.concrete.pi_rlc.sampler.handoff` | the batch begins at the derived Π_CCS outgoing state | caller-supplied transcript fork |
-/

open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Sampler

#check Specification
#check Bound
#check Bound.challengeValid
#check Bound.excludesShortfall
#check exists_bound_or_exists_shortfall
#check CertificateBound
#check CertificateAccepted
#check certificateBound_challengesValid
#check certificateAccepted_challengesValid
