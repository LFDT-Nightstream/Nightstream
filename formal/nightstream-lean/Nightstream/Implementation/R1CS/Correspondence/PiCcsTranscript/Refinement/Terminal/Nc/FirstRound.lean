import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.Refinement.Terminal.Nc.FirstRound.Artifact
import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.Refinement.Terminal.Nc.FirstRound.Execution
import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.Refinement.Terminal.Nc.FirstRound.Source
import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.Refinement.Terminal.Nc.FirstRound.Connectivity

/-!
Terminal-NC semantic round-zero proof group.

Owns: the physically distinct three-message-permutation artifact layout and
its semantic execution from the proved prologue state.

Does not own: prologue replay, typed coefficient authority, semantic
execution, later rounds, SumCheck algebra, costs, necessity, or row removal.

Emits constraints: no.

| Child path | Mathematical obligation | Emits constraints? | Lean owner |
|---|---|---|---|
| `nifs.pi_ccs.nc_sumcheck.round.0.artifact` | exact constants, four calls, and owner membership | no | `Artifact` |
| `nifs.pi_ccs.nc_sumcheck.round.0.execution` | prologue state and one message binding reach the exact successor | no | `Execution` |
| `nifs.pi_ccs.nc_sumcheck.round.0.source` | carrier coordinate zero supplies the ten message fields | no | `Source` |
| `nifs.pi_ccs.nc_sumcheck.round.0_to_1` | exact round-zero output supplies uniform round one | no | `Connectivity` |
-/
