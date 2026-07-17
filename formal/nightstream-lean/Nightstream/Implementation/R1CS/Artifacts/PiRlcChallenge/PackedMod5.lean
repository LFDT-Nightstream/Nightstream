import Nightstream.Implementation.R1CS.Artifacts.PiRlcChallenge.Generated.PackedMod5ArtifactData

/-!
Stable facade for the generated packed Mod-5 lowering artifact.

Owns: the public import boundary for one role-normalized production source
block, its projected decoder, eight emitted rows, and sparse polynomial terms.

Does not own: mathematical semantics, full-F' placement, selector composition,
or permission to remove source rows.

Emits constraints: no.

Authority boundary: the generated payload is exact production evidence, not
semantic authority. Handwritten correspondence proves what its equations mean.

| Child | Evidence | Semantic owner |
|---|---|---|
| `PackedMod5ArtifactData` | 20 source rows, 6 decoder definitions, 8 emitted rows, arity-56 polynomial | `Correspondence.PiRlcChallenge.Sampler.Chunk.Mod5` |
-/
