import Nightstream.Implementation.R1CS.Correspondence.PiCcsOutputDigest.SemanticHandoff

/-!
Focused compile-time regressions for the typed Split-NC output projection and
the exact post-`Pi_CCS` handoff.

| Stage path | Property under test | Failure caught |
|---|---|---|
| `nifs.pi_ccs.output_digest.profile` | source and matrix dimensions are explicit proof data | silent truncation to the fixed digest shape |
| `nifs.pi_ccs.output_digest.projection.lossless` | the 15-message projection is injective | omitted active output coordinate |
| `nifs.pi_ccs.output_digest.semantic` | digest and transcript state are recomputed from typed inputs | prover-authoritative digest/state |
| `nifs.pi_rlc.output_digest_bind.r1cs` | accepted rows refine the pure handoff only through named authority premises | self-consistent artifact treated as semantic authority |
-/

open Nightstream.Implementation.R1CS.PiCcsOutputDigest

#check Projection.SplitNc.Profile
#check Projection.SplitNc.Profile.ofAlignment
#check Projection.SplitNc.projectOutputs
#check Projection.SplitNc.projectOutputs_injective
#check SemanticHandoff.serializedValues
#check SemanticHandoff.digestValue
#check SemanticHandoff.digest
#check SemanticHandoff.run
#check SemanticHandoff.MessageBound
#check SemanticHandoff.CatchupInputBound
#check SemanticHandoff.accepted_refines_run
