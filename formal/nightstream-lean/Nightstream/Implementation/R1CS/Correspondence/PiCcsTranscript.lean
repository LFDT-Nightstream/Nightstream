import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.Primitives
import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.Refinement
import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.Binding
import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.Challenges
import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.Coins
import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.SumCheck
import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.FeRefinement
import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.NcRefinement
import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.BlockLane.NcRefinement
import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.ExactMessages
import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.Exact
import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.Schedule

/-!
Public proof tree for the production-shaped `Pi_CCS` transcript.

Owns: only the stable import boundary from executable transcript semantics to
typed FE/NC carriers and artifact refinement.

Does not own: paper or Split-NC soundness, authority of outer inputs,
Fiat--Shamir security, Rust conformance, costs, necessity, or row removal.

Emits constraints: no.

Authority boundary: semantic children compute verifier state and challenges;
refinement children must independently connect those computations to accepted
artifact rows. This facade adds no acceptance predicate.

| Child path | Mathematical ownership | Assurance tier |
|---|---|---|
| `Primitives`, `Binding`, `Challenges`, `Coins` | typed sponge operations and verifier-owned pre-SumCheck challenge flow | executable implementation semantics |
| `SumCheck`, `Schedule` | FE then NC message replay and catch-up handoff | executable implementation semantics |
| `ExactMessages`, `Exact` | fixed-width typed carrier and lossless exact schedule | executable implementation semantics |
| `FeRefinement`, `NcRefinement` | typed phase adapters agree with the executable schedule | model-level implementation refinement |
| `BlockLane.NcRefinement` | exact block-then-lane NC certificates agree with the reusable Poseidon2 round machine | model-level executable Poseidon2 refinement |
| `Refinement` | exact generated-row and Poseidon2 correspondence | artifact-checked correspondence |
-/
