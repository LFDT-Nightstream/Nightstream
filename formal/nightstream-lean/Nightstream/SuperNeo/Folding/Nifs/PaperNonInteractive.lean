import Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.HonestCompleteness

/-!
Paper-authoritative non-interactive SuperNeo NIFS facade.

Owns: curated access to the typed one-message verifier, independent semantic
transition, five named failure classes, deterministic soundness, graph
completeness, and honest-source construction.

Does not own: HyperNova/F-prime integration, concrete transcript or
commitment primitives, event probabilities, Rust, R1CS, artifacts,
minimality, or costs.

Emits constraints: no.

| Child | Mathematical obligation | Excluded boundary |
|---|---|---|
| `Types` | typed message and verifier-computed protocol dataflow | no semantic theorem |
| `Verifier` | compact deterministic executable checker | no extraction claim |
| `Semantics` | independent transition and closed event family | no probability bound |
| `Soundness` | exact soundness/completeness correspondence | no concrete primitive refinement |
| `HonestCompleteness` | causal honest accepted construction | no Rust/R1CS refinement |
-/
