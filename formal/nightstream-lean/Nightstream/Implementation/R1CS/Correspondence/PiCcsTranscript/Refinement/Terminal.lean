import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.Refinement.Terminal.Rows
import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.Refinement.Terminal.Schedule
import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.Refinement.Terminal.DigestRounds
import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.Refinement.Terminal.PinSchedule
import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.Refinement.Terminal.ScheduleRefinement
import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.Refinement.Terminal.PostNcBoundary
import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.Refinement.Terminal.RoundExecution
import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.Refinement.Terminal.Fe
import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.Refinement.Terminal.Nc

/-!
Protocol-to-phase tree for the terminal F-prime `Pi_CCS` transcript artifact.

Owns: only the stable parent boundary for the terminal owner partition,
binding-prefix refinement, explicit FE wire-format gap, terminal NC phase,
and post-NC handoff.

Does not own: paper or Split-NC algebra, outer input authority, Rust
conformance, aggregate F-prime acceptance, costs, necessity, or row removal.

Emits constraints: no.

Authority boundary: `Rows` and `Schedule` classify physical evidence;
`DigestRounds`, `PinSchedule`, and `ScheduleRefinement` give that evidence
semantic meaning; `Nc` continues the same refinement through the final NC
round. No child may use a generated digest as semantic authority.

| Phase path | Mathematical obligation | Child owner |
|---|---|---|
| `nifs.pi_ccs.terminal.owner` | exact terminal pieces have a total, non-overlapping address space | `Rows`, `Schedule` |
| `nifs.pi_ccs.terminal.binding` | accepted calls and pins refine the verifier-computed binding prefix | `DigestRounds`, `PinSchedule`, `ScheduleRefinement` |
| `nifs.pi_ccs.sumcheck.round.execution` | pure FE/NC transcript serialization and cursor lemmas | `RoundExecution` |
| `nifs.pi_ccs.fe_sumcheck` | classify the legacy FE owner and expose its width mismatch with the minimal typed language | `Fe` |
| `nifs.pi_ccs.nc_sumcheck` | terminal NC owner refines the typed NC execution | `Nc` |
| `nifs.pi_ccs.nc.post_state` | retain exactly the state consumed by catch-up | `PostNcBoundary` |
-/
