import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.TranscriptAuthority.BlockLane
import tests.Axioms.Support

/-!
Fail-closed dependency gate for canonical FE plus block×lane transcript
authority.

| Stage path | Guarded obligation | Emits constraints? |
|---|---|---|
| `nifs.pi_ccs.transcript.block_lane.coins.beta_a` | FE and canonical NC use the same `betaA` | no |
| `nifs.pi_ccs.transcript.block_lane.coins.gamma` | FE and canonical NC use the same `gamma` | no |
-/

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.TranscriptAuthority.BlockLane.Challenges.ncCoins_betaA_eq_feCoins_betaA' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.TranscriptAuthority.BlockLane.Challenges.ncCoins_betaA_eq_feCoins_betaA

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.TranscriptAuthority.BlockLane.Challenges.ncCoins_gamma_eq_feCoins_gamma' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.TranscriptAuthority.BlockLane.Challenges.ncCoins_gamma_eq_feCoins_gamma
