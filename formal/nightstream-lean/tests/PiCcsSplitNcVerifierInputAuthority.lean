import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.InputAuthority

/-!
Focused compile-time regressions for the Split-NC `Pi_CCS` input-authority
bridge.

| Stage path | Property under test | Failure caught |
|---|---|---|
| `nifs.pi_ccs.input.running.prior` | carried paper truth fixes the complete concrete prior-evaluation array | caller-chosen or partially bound running claims |
| `nifs.pi_ccs.input.partition` | fresh/running indices preserve their semantic partitions | total-only alignment or source permutation |
| `nifs.pi_ccs.input.opening` | commitment and public input are projections of the aligned authoritative assignment | digest-only or unrelated opening authority |
| `nifs.pi_ccs.input.membership` | paper truth plus explicit public bindings implies actual CCS/CE membership | semantic data disconnected from public NIFS inputs |
-/

open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.InputAuthority

#check relationEvaluations_eq_priorEvaluations_of_carriedTruth
#check BoundToSources.sourceCommitment
#check BoundToSources.sourcePublicInput
#check productAssignments_fresh
#check productAssignments_running
#check freshSource_holds
#check runningSource_holds
#check allSourcesHold

