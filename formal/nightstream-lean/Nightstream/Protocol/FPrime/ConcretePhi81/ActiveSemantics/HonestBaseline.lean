import Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.HonestBaseline.Sources
import Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.HonestBaseline.RunningAuthority
import Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.HonestBaseline.Context

/-!
Curated model-level honest baseline for the fixed-active Phi81 F-prime proof.

Owns: one independently proved source statement at the explicit
270-coordinate fixed-active carrier shape, one reusable construction of
checked incoming-parent authority, and one degenerate context that composes
them with an explicitly constructed centered-zero sampler batch.

Does not own: a complete outer F-prime invocation, transcript security, Rust
or artifact conformance, R1CS refinement, costs, or row removal.

Emits constraints: no.

Authority boundary: source truth and incoming-parent authority are distinct
branches. Neither is inferred from a verifier acceptance bit, digest, generated
artifact, or the other branch. `HonestBaseline.Context` composes them by
proving the exact public source and parent/child equalities for an explicit
zero-row, constant-transcript fixture, then constructs one physically accepted
model transition. That fixture is not conformance evidence.

| Child path | Mathematical obligation | Emits constraints? | Lean owner |
|---|---|---|---|
| `fprime.active.honest_baseline.sources` | explicit 270-coordinate source data satisfies the independent paper statement | no | `HonestBaseline.Sources` |
| `fprime.active.honest_baseline.running` | a valid combined opening and its canonical children satisfy strict incoming `PiDEC` authority | no | `HonestBaseline.RunningAuthority` |
| `fprime.active.honest_baseline.context` | an explicit degenerate context binds the source and running branches, constructs the bounded centered-zero sampler batch, and yields one accepted semantic NIFS transition | no | `HonestBaseline.Context` |

This facade is model-level. Rust and artifact conformance require separate
theorems under `Nightstream.Implementation`.
-/
