import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Sampler.CandidateOrder
import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Sampler.ChunkRows
import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Sampler.Chunk
import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Sampler.Chunk.Acceptance
import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Sampler.Chunk.Mod5
import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Sampler.Lane
import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Sampler.Refinement
import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Sampler.Selection
import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Sampler.SelectionRows

/-! Parent for the concrete Pi_RLC challenge sampler.

Owns: candidate/chunk/lane shape, packed Mod-5 leaves, selection constraints,
and row-to-output refinement.

Does not own: Poseidon2 transcript derivation, Pi_RLC ring identities, or the
full recursive verifier.

Emits constraints: no.

| Child | Mathematical obligation | Excluded boundary |
|---|---|---|
| candidate/chunk/lane | ordered candidate decoding and range equations | transcript generation |
| `nifs.pi_rlc.challenge.sampler.chunk.accept.packed` | model-level aggregate acceptance exactness and necessity | production placement and row removal |
| `Chunk.Mod5` | packed Mod-5 leaf semantics and isolated artifact | full-profile placement |
| `Selection` / `SelectionRows` | first-accepted control-flow equations | ring output semantics |
| `Refinement` | emitted sampler rows refine typed scalar/ring outputs | paper-level reduction |
-/
