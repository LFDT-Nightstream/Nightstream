import Nightstream.Implementation.R1CS.Artifacts.PiRlcChallenge.Generated.AggregateAcceptanceArtifactData

/-!
Stable facade for the generated aggregate-acceptance leaf artifact.

Owns: the public import boundary for the production arity, occupied matrix
bindings, nine normalized rows, and exact sparse-polynomial specialization.

Does not own: mathematical semantics, full-F′ placement, selector composition,
source-bit decoding, or permission to remove rows.

Emits constraints: no.

Authority boundary: the generated payload is exact production evidence, not
semantic authority. Handwritten correspondence assigns meaning to its rows.

| Child | Evidence | Semantic owner |
|---|---|---|
| `AggregateAcceptanceArtifactData` | arity 56, 40 bindings, 9 rows, 25 terms | `Correspondence.PiRlcChallenge.Sampler.Chunk.Acceptance.Aggregate` |
-/
