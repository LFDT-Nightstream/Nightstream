import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.TranscriptAuthority
import tests.Axioms.Support

/-!
Fail-closed dependency gate for the single-owner Split-NC transcript
authority.

| Stage path | Guarded obligation | Emits constraints? |
|---|---|---|
| `nifs.pi_ccs.transcript.coins.shared.beta_a` | FE and NC use the same `betaA` | no |
| `nifs.pi_ccs.transcript.coins.shared.gamma` | FE and NC use the same `gamma` | no |
-/

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.TranscriptAuthority.Challenges.ncCoins_betaA_eq_feCoins_betaA' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.TranscriptAuthority.Challenges.ncCoins_betaA_eq_feCoins_betaA

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.TranscriptAuthority.Challenges.ncCoins_gamma_eq_feCoins_gamma' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.TranscriptAuthority.Challenges.ncCoins_gamma_eq_feCoins_gamma
