import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.Refinement.Terminal.Nc.LaterRound.Artifact
import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.Refinement.Terminal.Nc.LaterRound.Execution
import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.Refinement.Terminal.Nc.LaterRound.Source
import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.Refinement.Terminal.Nc.LaterRound.Connectivity
import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.Refinement.Terminal.Nc.LaterRound.Replay

/-!
Family boundary for the fourteen uniform terminal-NC later rounds.

Owns: only the parent grouping for semantic rounds one through fourteen.

Does not own: round zero, semantic transcript replay, SumCheck algebra,
costs, necessity, or row removal.

Emits constraints: no.

Authority boundary: indexed artifact ownership is structural evidence only.
The typed transcript remains the semantic authority.

| Child path | Mathematical obligation | Emits constraints? | Lean owner |
|---|---|---|---|
| `nifs.pi_ccs.nc_sumcheck.round.1_14.artifact` | exact affine owner/call formulas and accepted constants | no | `Artifact` |
| `nifs.pi_ccs.nc_sumcheck.round.1_14.execution` | one typed round refines to the exact indexed call output | no | `Execution` |
| `nifs.pi_ccs.nc_sumcheck.round.1_14.source` | the complete typed carrier decoder supplies each message boundary | no | `Source` |
| `nifs.pi_ccs.nc_sumcheck.round.1_13.connectivity` | each complete squeeze output is the next round's incoming state | no | `Connectivity` |
| `nifs.pi_ccs.nc_sumcheck.round.1_14.replay` | one round-one boundary induces the complete typed later-round suffix | no | `Replay` |
-/
