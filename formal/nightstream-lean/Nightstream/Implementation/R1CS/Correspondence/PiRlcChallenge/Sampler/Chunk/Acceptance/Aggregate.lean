import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Sampler.Chunk.Acceptance.Aggregate.Exactness
import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Sampler.Chunk.Acceptance.Aggregate.Necessity
import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Sampler.Chunk.Acceptance.Aggregate.ArtifactRefinement
import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Sampler.Chunk.Acceptance.Aggregate.OuterImage

/-!
Parent contract for the aggregate acceptance subtree.

Owns: the stable import surface and responsibility map for source semantics,
aggregate exactness, independent necessity countermodels, and leaf-local
generated-artifact refinement.

Does not own: Rust trace decoding, fixed selectors, inactive rows, complete
R1CS satisfaction, global cost totals, or row-removal authority.

Emits constraints: no.

| Exact Rust stage subtree | Child | Mathematical ownership | Assurance tier |
|---|---|---|---|
| `nifs.pi_rlc.challenge.sampler.chunk.accept.packed` | `Aggregate.Semantics` | source bits, balanced tree, rejection and verifier meaning | model-level |
| `nifs.pi_rlc.challenge.sampler.chunk.accept.packed` | `Aggregate.Exactness` | paired bit equations, collision-free aggregate, root binding, unique extension | model-level |
| `nifs.pi_rlc.challenge.sampler.chunk.accept.packed` | `Aggregate.Necessity` | independent countermodels for each retained family | model-level |
| `nifs.pi_rlc.challenge.sampler.chunk.accept.packed` | `Aggregate.ArtifactEvaluation` | handwritten evaluator for generated role data | model-level evaluator |
| `nifs.pi_rlc.challenge.sampler.chunk.accept.packed` | `Aggregate.ArtifactRefinement` | exact arity-56 nine-row leaf against independent semantics | artifact-checked, leaf-local |
| `nifs.pi_rlc.challenge.sampler.chunk` | `Aggregate.OuterImage` | 960-chunk decoder, Boolean-owner and active-row placement tree | artifact-checked, conditional semantic bridge |
-/
