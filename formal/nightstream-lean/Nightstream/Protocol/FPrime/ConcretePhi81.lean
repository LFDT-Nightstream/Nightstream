import Nightstream.Protocol.FPrime.ConcretePhi81.Context
import Nightstream.Protocol.FPrime.ConcretePhi81.BootstrapContext
import Nightstream.Protocol.FPrime.ConcretePhi81.Outer
import Nightstream.Protocol.FPrime.ConcretePhi81.BaseSemantics
import Nightstream.Protocol.FPrime.ConcretePhi81.SelectedNifsSemantics
import Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics
import Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Construction2
import Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.HonestNifs
import Nightstream.Protocol.FPrime.ConcretePhi81.ActiveEvaluator
import Nightstream.Protocol.FPrime.ConcretePhi81.ActiveEvaluator.SemanticBoundary
import Nightstream.Protocol.FPrime.ConcretePhi81.ActiveEvaluator.FixedOneCanonical
import Nightstream.Protocol.FPrime.ConcretePhi81.Semantics

/-!
Canonical public surface for the independent Phi81 F-prime semantic spine.

Assurance tier: model-level; not Rust-conformant or security-reduced.

Owns: the curated imports and ownership map for the concrete Phi81 F-prime
semantic spine.

Does not own: physical selected-NIFS transcript/security refinement, reverse
paper refinement, Rust/R1CS refinement, constraint emission, cost accounting,
or row-removal authority.

Emits constraints: no.

| Child | Ownership | Current assurance boundary |
|---|---|---|
| `Context` | exact one-fresh plus fourteen-running NIFS context construction | semantic construction |
| `BootstrapContext` | exact one-fresh plus zero-running NIFS context with absent incoming parent | semantic construction; default-vector equivalence open |
| `Outer` | branch-neutral input, output, and complete running carriers | exact paper projection; parent caches erased explicitly |
| `BaseSemantics` | three checked base obligations and canonical rich default output | model-level completeness and forward paper soundness |
| `SelectedNifsSemantics` | public child-only NIFS edge with semantic internal witnesses | independent model-level relation |
| `ActiveSemantics` | six retained active obligations and canonical computed output | independent relation |
| `ActiveSemantics.Construction2` | explicit and canonical selected-NIFS refinement | model-level forward paper soundness; no implementation alias |
| `ActiveSemantics.HonestNifs` | shared honest paper/source premises and semantic result construction | honest completeness |
| `ActiveEvaluator` | fail-closed executable checker | exact physical acceptance only |
| `ActiveEvaluator.SemanticBoundary` | close execution to semantics and construct honest accepted outputs | conditional soundness plus honest completeness |
| `ActiveEvaluator.FixedOneCanonical` | payload-only carrier, exact physical checks, fail-closed output, and explicit semantic closure | exact physical execution; conditional semantic soundness |
| `Semantics` | disjoint base/recursive full relation | model-level forward Construction-2 soundness with the concrete selected edge |

Physical transcript/security refinement, reverse paper refinement, Rust/R1CS
refinement, and cost ownership remain separate until their theorem boundaries
close. The provisional zero-arity lifecycle, legacy callback and
certificate-shaped verifiers, diagnostic honest baselines, and necessity plans
require explicit imports and are intentionally not re-exported here.
-/
