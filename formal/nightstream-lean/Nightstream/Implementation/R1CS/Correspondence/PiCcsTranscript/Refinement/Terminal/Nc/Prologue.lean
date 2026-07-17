import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.Refinement.Terminal.Nc.Prologue.Artifact
import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.Refinement.Terminal.Nc.Prologue.Execution

/-!
Terminal-NC prologue proof group.

Owns: the physical constants, two permutation calls, and semantic execution
preceding round zero.

Does not own: FE successor authority, semantic execution, round-zero replay,
SumCheck algebra, costs, necessity, or row removal.

Emits constraints: no.

| Child path | Mathematical obligation | Emits constraints? | Lean owner |
|---|---|---|---|
| `nifs.pi_ccs.nc_sumcheck.prologue.artifact` | exact constants, calls, and owner membership | no | `Artifact` |
| `nifs.pi_ccs.nc_sumcheck.prologue.execution` | FE successor plus accepted calls reaches the cursor-one round-tag state | no | `Execution` |
-/
