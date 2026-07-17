import Nightstream.Protocol.FPrime.ConcretePhi81.ZeroArityLifecycle.State
import Nightstream.Protocol.FPrime.ConcretePhi81.ZeroArityLifecycle.Context
import Nightstream.Protocol.FPrime.ConcretePhi81.ZeroArityLifecycle.Transition

/-!
Public surface for the concrete contextual zero-running F-prime lifecycle.

This component records the production-shaped `initial -> primed -> running`
schedule at the model level. It is not yet a proof that the schedule refines
HyperNova Construction 2: production context derivation depends on the newly
deposited claim, and the paper/default-vector simulation remains open.

Owns: the public boundary and immediate-child ownership map for the contextual
zero-running lifecycle.

Does not own: child equations, production transcript provenance, paper
refinement, executable acceptance, Rust/R1CS refinement, costs, or row
removal.

Emits constraints: no.

| Child | Owns | Excluded assurance boundary |
|---|---|---|
| `State` | typed claim, result payload, and three lifecycle phases | raw state carries no provenance evidence |
| `Context` | fixed assumed verifier configuration and bootstrap/active context construction | production transcript and public-input provenance remain open |
| `Transition` | three arms, outgoing premise, state validity, result provenance, and reachability | application semantics, paper refinement, executable checking, and Rust/R1CS refinement remain open |

No module in this component emits constraints or authorizes row removal.
-/
