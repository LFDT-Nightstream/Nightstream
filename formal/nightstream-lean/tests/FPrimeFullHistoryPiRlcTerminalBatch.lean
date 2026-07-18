import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Sampler.Refinement.Terminal.Batch

/-!
Focused kernel checks for the terminal Π_RLC implementation-to-batch bridge.

| Stage path | Property under test | Failure caught |
|---|---|---|
| `nifs.pi_rlc.challenge.batch.state` | all fifteen semantic states follow one connected transcript schedule | independently replayed scalar transcripts |
| `nifs.pi_rlc.challenge.batch.prefix` | each semantic source sees the exact 64 decoded candidates | candidate-order or source-state drift |
| `nifs.pi_rlc.challenge.batch.execution` | accepted rows construct a bounded least-cursor execution | silent rejection-sampler fallback |
| `nifs.pi_rlc.challenge.batch.output` | batch assembly equals the decoded Phi81 RingF challenge | coefficient or lane reordering |
| `nifs.pi_rlc.challenge.batch.binding` | all decoded challenges form one semantic sampler `Bound` | unbound carried challenge authority |
-/

open Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.Terminal.Batch

#check stateAfterFourBlocks_eq_block3
#check stateAt_refines
#check candidateStreamPrefix_eq_machineCandidates
#check sourcePrefix_eq_machineCandidates
#check execution_exists
#check RefinedBatch
#check accepted_refines_batch
#check RefinedBatch.challenge_eq_machineChallenge
#check accepted_refines_initialStateBound
