import Nightstream.Implementation.R1CS.Artifacts.Phi81.Generated.Phi81BarMatrixArtifact

/-!
Stable facade for the exhaustive production Phi81 bar-matrix export.

Owns: the public import boundary for generated runtime matrix data.

Does not own: mathematical bar semantics, Rust conformance by itself, matrix
packing, R1CS constraints, or row-removal authority.

Emits constraints: no.

Authority boundary: generated entries are evidence only. Handwritten
correspondence must prove them equal to an independent semantic definition.

| Child path | Mathematical obligation | Emits constraints? | Owner |
|---|---|---|---|
| `Generated.Phi81BarMatrixArtifact` | every runtime `54 by 54` bar entry | no | Rust drift artifact |
-/
