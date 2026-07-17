import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.NifsPaper.PiRlc.Profiles

/-!
Focused compile-time regression for the terminal production rows-to-concrete
NIFS sampler certificate bridge.

Assurance tier: artifact-checked and conditional on explicit upstream
message/post-NC bindings; this is not a security-reduced authority theorem.

| Rust stage | Property under test | Failure caught |
|---|---|---|
| `nifs.pi_rlc.challenge` | production context functions are explicitly bound | arbitrary verifier context accepted as production |
| `nifs.pi_rlc.challenge` | certificate challenges are constructed from decoded sampler outputs | prover-selected challenge vector |
| `nifs.pi_rlc.challenge` | accepted rows construct the exact semantic certificate predicate | artifact-only acceptance not connected to NIFS |
| `nifs.pi_rlc.challenge` | constructed challenge field equals the exact batch columns | semantic and equation challenge vectors diverge |
-/

open Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.Terminal

#check Certificate.terminalProfile
#check Certificate.ContextBinding
#check Certificate.decodedChallenges
#check Certificate.withDecodedChallenges
#check Certificate.withDecodedChallenges_challenge_eq_columns
#check Certificate.accepted_refines_certificateAccepted
#check Certificate.accepted_refines_withDecodedChallenges
#check Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.Profiles.terminalSampler_refines_decodedBatchChallenges
