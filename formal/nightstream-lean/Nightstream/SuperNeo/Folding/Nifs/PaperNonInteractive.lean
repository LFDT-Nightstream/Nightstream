import Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.HonestCompleteness
import Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.CoordinateForkBridge
import Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.OracleSoundness
import Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.RandomOracleBoundary
import Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.RewindableContinuation
import Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.InteractiveCompositionBridge
import Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.CausalPrefixCoupling
import Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.CausalFirstSuccessBridge
import Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.CausalPostPrefixBridge
import Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.RewindableOracleSoundness
import Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.PostPrefixWorldSoundness
import Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.FullOracleSoundness

/-!
Paper-authoritative non-interactive SuperNeo NIFS facade.

Owns: curated access to the typed one-message verifier, independent semantic
transition, five named failure classes, deterministic soundness, graph
completeness, honest-source construction, and the exact typed random-oracle
input/handoff and coordinate-fork boundary, including the complete finite
correlated prefix/post-prefix experiment and its exact programming loss. The
old conditioned-first carrier remains imported only as a legacy regression
artifact; it is not current paper authority.

Does not own: HyperNova/F-prime integration, concrete transcript or
commitment primitives, the preceding-prefix distribution, interactive or
collision event bounds, Rust, R1CS, artifacts, minimality, or costs.

Emits constraints: no.

| Child | Mathematical obligation | Excluded boundary |
|---|---|---|
| `Types` | typed message and verifier-computed protocol dataflow | no semantic theorem |
| `Verifier` | compact deterministic executable checker | no extraction claim |
| `Semantics` | independent transition and closed event family | no probability bound |
| `Soundness` | exact soundness/completeness correspondence | no concrete primitive refinement |
| `HonestCompleteness` | causal honest accepted construction | no Rust/R1CS refinement |
| `RandomOracleBoundary` | full public-input absorption, exact replay/output/coordinate handoff, and closed collision events | no event probability or Poseidon2 refinement |
| `CoordinateForkBridge` | exact NIFS `Pi_RLC` batch, concrete six-event predicates, and accepted-fork ambient extraction | no oracle-programming probability theorem |
| `RewindableContinuation` | one owned PiCCS prefix/PiDEC continuation and definitional PiRLC oracle linkage | no continuation or fork distribution |
| `InteractiveCompositionBridge` | exact NIFS-to-interactive contexts, fixed-width PiCCS gate/residual-event identity, and causal-prefix batch/parent alignment | no ideal-oracle causal coupling, concrete transcript encoding, or PiDEC target witness |
| `CausalPrefixCoupling` | explicit prover-seed × verifier-coin support, exact causal replay/continuation, and pointwise fixed-witness PiCCS reduction | no ideal-oracle construction, collision bound, or corrected success-gated extraction bridge |
| `CausalFirstSuccessBridge` | legacy conditioned-first causal/target seed reindexing and fixed-first-witness regression | not the corrected success-gated extractor; no current extraction or runtime authority |
| `CausalPostPrefixBridge` | exact interactive/NIFS PiDEC execution and D.6 target-success identity in programmed worlds | no target-witness existence |
| `OracleSoundness` | exact eleven-event cover and conditional `nonInteractiveTotal` probability theorem | no event-bound or outcome-distribution instantiation |
| `RewindableOracleSoundness` | eleven-event theorem pulled back to owned continuations | no malicious-prover or random-oracle experiment construction |
| `PostPrefixOracleWorld` | exact PiRLC-vector oracle reprogramming at one fixed D.5 prefix | no preceding-prefix distribution |
| `PostPrefixForkExperiment` | finite uniform D.5 coordinate-fork experiment and exact programming loss | no collision or interactive-event bound |
| `PostPrefixWorldSoundness` | dependent-world eleven-event composition and concrete conditional soundness theorem | no preceding-prefix distribution or collision bounds |
| `PiCcsPrefixOracleWorld` | correlated prefix-oracle, public-input, and malicious-prover seed owner | no ideal-oracle support or event bound |
| `FullOracleExperiment` | two-level finite mixture, global cover, zero sampling loss, and D.5 programming bound | no interactive or collision bound |
| `FullOracleSoundness` | global eleven-event subtractive theorem over the complete owned outcome | one target-witness extraction, four interactive, and four collision bounds remain |
-/
