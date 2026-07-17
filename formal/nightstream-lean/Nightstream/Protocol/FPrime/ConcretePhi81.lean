import Nightstream.Protocol.FPrime.ConcretePhi81.Context
import Nightstream.Protocol.FPrime.ConcretePhi81.BootstrapContext
import Nightstream.Protocol.FPrime.ConcretePhi81.ZeroArityLifecycle
import Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics
import Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.HonestNifs
import Nightstream.Protocol.FPrime.ConcretePhi81.ActiveEvaluator
import Nightstream.Protocol.FPrime.ConcretePhi81.ActiveEvaluator.SemanticBoundary
import Nightstream.Protocol.FPrime.ConcretePhi81.ActiveEvaluator.FixedOneCanonical

/-!
Canonical public surface for the independent fixed-active Phi81 F-prime
proof spine.

Assurance tier: model-level; not Rust-conformant or security-reduced.

Owns: the curated imports and ownership map for the concrete Phi81 F-prime
semantic spine.

Does not own: paper refinement, Rust/R1CS refinement, constraint emission,
cost accounting, or row-removal authority.

Emits constraints: no.

| Child | Ownership | Current assurance boundary |
|---|---|---|
| `Context` | exact one-fresh plus fourteen-running NIFS context construction | semantic construction |
| `BootstrapContext` | exact one-fresh plus zero-running NIFS context with absent incoming parent | semantic construction; default-vector equivalence open |
| `ZeroArityLifecycle` | typed base, contextual bootstrap, and steady phases over concrete model-level NIFS results | setup provenance, delayed authority, hashes, and paper refinement open |
| `ActiveSemantics` | six retained active obligations and canonical computed output | independent relation |
| `ActiveSemantics.HonestNifs` | shared honest paper/source premises and semantic result construction | honest completeness |
| `ActiveEvaluator` | fail-closed executable checker | exact physical acceptance only |
| `ActiveEvaluator.SemanticBoundary` | close execution to semantics and construct honest accepted outputs | conditional soundness plus honest completeness |
| `ActiveEvaluator.FixedOneCanonical` | payload-only carrier, exact physical checks, fail-closed output, and explicit semantic closure | exact physical execution; conditional semantic soundness |

Paper refinement, Rust/R1CS refinement, and cost ownership remain separate
until their theorem boundaries close. Legacy callback and certificate-shaped
verifiers, diagnostic honest baselines, and provisional necessity plans require
explicit imports and are intentionally not re-exported here.
-/
