import Nightstream.Protocol.FPrime.ConcretePhi81.ActiveEvaluator.FixedOneCanonical.Context
import Nightstream.Protocol.FPrime.ConcretePhi81.ActiveEvaluator.FixedOneCanonical.Physical
import Nightstream.Protocol.FPrime.ConcretePhi81.ActiveEvaluator.FixedOneCanonical.Evaluator
import Nightstream.Protocol.FPrime.ConcretePhi81.ActiveEvaluator.FixedOneCanonical.PaperBoundary
import Nightstream.Protocol.FPrime.ConcretePhi81.ActiveEvaluator.FixedOneCanonical.SemanticBoundary

/-!
Payload-minimal fixed-one F-prime verifier surface.

Owns: the canonical carrier, exact physical checking, fail-closed evaluation,
and the explicit semantic closure boundary.

Does not own: Poseidon2/Rust/R1CS refinement, physical rows, costs, bad-event
probability bounds, or row removal.

Emits constraints: no.

| Child | Mathematical ownership | Assurance boundary |
|---|---|---|
| `Context` | compute selection, counter, structures, stages, parent, and NIFS context | exact model construction |
| `Physical` | check positive iteration, full prior link, and the exact raw NIFS certificate | exact physical acceptance |
| `Evaluator` | compute the checked NIFS result and complete outer output | fail-closed exact execution |
| `PaperBoundary` | enforce exact paper PiDEC outputs and refine to Construction 2 or named failure | model-level soundness/completeness |
| `SemanticBoundary` | connect execution to independent semantics under explicit closure premises | conditional soundness; honest completeness |
-/
