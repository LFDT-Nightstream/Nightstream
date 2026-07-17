import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Sampler.Chunk.Acceptance.Aggregate

/-!
Parent for model-level acceptance relations attached to one sampler chunk.

Owns: the stable chunk-acceptance import surface and child responsibility map.

Does not own: existing four-row production acceptance, generated artifacts,
Rust emission, production placement, fixed selectors, outer-image composition,
cost totals, or row-removal authority.

Emits constraints: no.

| Exact Rust stage subtree | Child | Mathematical obligation | Assurance tier |
|---|---|---|---|
| `nifs.pi_rlc.challenge.sampler.chunk.accept.packed` | `Aggregate` | balanced tree, paired output bitness, collision-free aggregate, root binding | model-level |
-/
