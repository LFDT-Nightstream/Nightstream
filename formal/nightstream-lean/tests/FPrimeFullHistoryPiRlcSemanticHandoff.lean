import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Sampler.Refinement.Terminal.SemanticHandoff

/-!
Focused compile-time regression for the terminal bounded sampler rooted at the
typed post-`Pi_CCS` semantic handoff.

| Stage path | Property under test | Failure caught |
|---|---|---|
| `nifs.pi_rlc.challenge.semantic_handoff` | the exact 15-challenge batch starts at `PiCcsOutputDigest.SemanticHandoff.run` | artifact initial state treated as authority |
-/

#check Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.Terminal.SemanticHandoff.accepted_refines_semanticHandoffBound
