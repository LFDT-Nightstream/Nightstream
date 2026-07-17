import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.Refinement.Terminal.Nc.Rows
import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.Refinement.Terminal.Nc.Schedule
import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.Refinement.Terminal.Nc.Carrier
import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.Refinement.Terminal.RoundExecution
import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.Refinement.Terminal.Nc.Prologue
import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.Refinement.Terminal.Nc.FirstRound
import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.Refinement.Terminal.Nc.LaterRound
import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.Refinement.Terminal.Nc.FinalRound
import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.Refinement.Terminal.Nc.FinalState
import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.Refinement.Terminal.Nc.Replay

/-!
Phase-to-family tree for the terminal Split-NC SumCheck artifact.

Owns: only the parent boundary from the complete physical NC owner to the
selected final round and retained post-NC state.

Does not own: FE execution, NC polynomial soundness, the final algebra rows,
production-domain construction, Rust conformance, costs, necessity, or row
removal.

Emits constraints: no.

Authority boundary: physical owner classification is kept separate from
semantic execution. The final round is selected from the typed schedule, not
supplied by a prover or inferred from artifact column values.

| Stage path | Mathematical obligation | Child owner |
|---|---|---|
| `nifs.pi_ccs.nc_sumcheck.owner` | exact 81-piece owner address space | `Rows` |
| `nifs.pi_ccs.nc_sumcheck.phase_tree` | prologue, first round, and fourteen later rounds cover the owner | `Schedule` |
| `nifs.pi_ccs.nc_sumcheck.certificate` | every typed round maps to its exact assignment coefficient pairs | `Carrier` |
| `nifs.pi_ccs.nc_sumcheck.round.execution` | instantiate the shared terminal SumCheck serialization lemmas | `Terminal.RoundExecution` |
| `nifs.pi_ccs.nc_sumcheck.prologue` | exact verifier constants and two pre-round permutations | `Prologue` |
| `nifs.pi_ccs.nc_sumcheck.round.0` | distinct three-message-permutation first-round artifact | `FirstRound` |
| `nifs.pi_ccs.nc_sumcheck.round.1_14` | every uniform later round has one indexed affine artifact layout | `LaterRound` |
| `nifs.pi_ccs.nc_sumcheck.round.14` | exact artifact leaves refine the verifier-selected final typed round | `FinalRound` |
| `nifs.pi_ccs.nc.post_state` | accepted final permutation binds the retained catch-up state | `FinalState` |
| `nifs.pi_ccs.nc_sumcheck.replay` | one typed carrier replays the complete prologue and fifteen-round transcript | `Replay` |
-/
