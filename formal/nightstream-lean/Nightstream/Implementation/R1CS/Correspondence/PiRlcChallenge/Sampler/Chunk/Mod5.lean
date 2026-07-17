import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Sampler.Chunk.Mod5.PackedRows
import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Sampler.Chunk.Mod5.PackedArtifactSchema
import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Sampler.Chunk.Mod5.PackedArtifactRefinement
import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Sampler.Chunk.Mod5.PackedDecoderImageRefinement

/-!
Ownership boundary for the packed Mod-5 constraint family of one sampler
chunk.

Owns: composition of the independent packed-row semantics, artifact-to-column
schema, isolated-profile artifact refinement, and isolated decoder/image
refinement.

Does not own: production discharge of the coordinate aliases, full sampler
placement, physical active rows, selectors, the Goldilocks nonresidue
certificate, or row-removal authority.

Emits constraints: no.

| Child | Mathematical obligation | Excluded boundary |
|---|---|---|
| `PackedRows` | eight packed equations iff sixteen scalar equations, conditional on `SevenNonresidue` | production placement |
| `PackedArtifactSchema` | generated roles map to collision-free source/coordinate columns and direct evaluators | semantic acceptance |
| `PackedArtifactRefinement` | exact isolated artifact shape, source normalization, degrees, and role-point polynomials | physical high decoder and source-coordinate aliases |
| `PackedDecoderImageRefinement` | generated high decoder equals the independent derived-high formula under explicit low-coordinate aliases | recursive sparse outer image and physical placement |
-/
