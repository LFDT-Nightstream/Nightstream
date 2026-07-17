import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Sampler.Chunk.Acceptance.Aggregate.OuterImage.Semantics
import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Sampler.Chunk.Acceptance.Aggregate.OuterImage.ArtifactRefinement

/-!
Parent contract for the recursive aggregate-acceptance outer image.

Owns: the stable import surface and responsibility split between handwritten
outer-image semantics and kernel-checked fixed-profile artifact accounting.

Does not own: Rust extraction, complete R1CS satisfaction, complete Π_RLC or
NIFS soundness, global cost totals, or row-removal authority.

Emits constraints: no.

| Child | Mathematical ownership | Assurance tier |
|---|---|---|
| `OuterImage.Semantics` | decoder, source-definition, Boolean-owner and active-row equations | model-level, handwritten |
| `OuterImage.ArtifactRefinement` | exact 960-chunk leaf census and row reconciliation | artifact-checked |
-/
