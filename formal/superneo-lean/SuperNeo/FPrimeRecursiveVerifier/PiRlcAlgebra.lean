import SuperNeo.FPrimeRecursiveVerifier.PiRlcAlgebra.Authority.ChildYZcolElision
import SuperNeo.FPrimeRecursiveVerifier.PiRlcAlgebra.Authority.Consistency
import SuperNeo.FPrimeRecursiveVerifier.PiRlcAlgebra.Claims.Adv
import SuperNeo.FPrimeRecursiveVerifier.PiRlcAlgebra.Claims.X
import SuperNeo.FPrimeRecursiveVerifier.PiRlcAlgebra.Claims.YRing
import SuperNeo.FPrimeRecursiveVerifier.PiRlcAlgebra.Claims.YZcol
import SuperNeo.FPrimeRecursiveVerifier.PiRlcAlgebra.Refinement.ProductSum
import SuperNeo.FPrimeRecursiveVerifier.PiRlcAlgebra.Refinement.ProjectionBatching
import SuperNeo.FPrimeRecursiveVerifier.PiRlcAlgebra.Refinement.ProjectionBindingShapeArtifact
import SuperNeo.FPrimeRecursiveVerifier.PiRlcAlgebra.Refinement.ProjectionBindingSerialization
import SuperNeo.FPrimeRecursiveVerifier.PiRlcAlgebra.Refinement.ProjectionIdentityCertificate
import SuperNeo.FPrimeRecursiveVerifier.PiRlcAlgebra.Semantics.ProjectionBoundary
import SuperNeo.FPrimeRecursiveVerifier.PiRlcAlgebra.Semantics.ProjectionBindingShape

/-!
Owns: the public import root and three-level mathematical ownership map for
the fixed Pi_RLC algebra verifier.

Does not own: transcript challenge derivation, the production
exact-or-bad-root reduction, or Rust trace conformance.

Emits constraints: no. This module only composes semantic and refinement
theorems.

Authority boundary: claim equations bind parents only after input/rho
authority; `Authority.Consistency` makes its upstream common-value premise
explicit.

| Child path | Mathematical obligation | Emits constraints? | Rust owner | Lean owner |
|---|---|---|---|---|
| `Semantics.RingAction` | Exact Phi_81 multiplication | No | `ring_action.rs` | `Semantics/RingAction.lean` |
| `Semantics.Combination` | One direct 15-input parent equation | No | algebra parent | `Semantics/Combination.lean` |
| `Semantics.ProjectionBoundary` | Exact polynomial equality or a root of the nonzero difference | No | `ring_action.rs` final limb checks | `Semantics/ProjectionBoundary.lean` |
| `Semantics.ProjectionBindingShape` | Explicit 64-entry carriers, zero tails, and exact active-prefix pairing | No | `nifs/circuit/pi_rlc/{projection,padding}` | `Semantics/ProjectionBindingShape.lean` |
| `Claims.Commitment` | 18 commitment lanes | No | `commitment.rs` | `Claims/Commitment.lean` |
| `Claims.Adv` | Present-case `ops`/`is`/`fs` commitment coordinates | No | views: `nifs/circuit/pi_rlc/fold_wires.rs`; orchestration: `projection/identities.rs`; rows: `pi_rlc_circuit/commitment.rs` | `Claims/Adv.lean` |
| `Claims.X` | 5 active public-X columns plus full-width inactive zeroing | No | `x.rs` | `Claims/X.lean` |
| `Claims.YRing` | 3 padded K-vector rows | No | `padded_k.rs` | `Claims/YRing.lean` |
| `Claims.YZcol` | 1 padded K-vector claim | No | `padded_k.rs` | `Claims/YZcol.lean` |
| `Claims.Padding` | Shared active/tail boundary | No | `padded_k.rs` | `Claims/Padding.lean` |
| `Authority.Consistency` | Common `s_col` and fold-digest authority | No | `pi_rlc_circuit/consistency.rs`; call site `nifs/circuit/pi_rlc/consistency.rs` | `Authority/Consistency.lean` |
| `Authority.ChildYZcolElision` | Diagnostic no-read model for paired child/next-running `y_zcol` sidecars | No | does not justify production erasure; delayed authority refinement required | `Authority/ChildYZcolElision.lean` |
| `Refinement.ExactMaterialization` | Exact semantic product substitution | No | semantic comparison | `Refinement/ExactMaterialization.lean` |
| `Refinement.ProductSum` | Generic scalar substitution and carry algebra | No | candidate selective lowering | `Refinement/ProductSum.lean` |
| `Refinement.ProjectionBatching` | Exact 18-term evaluation chunks and two-limb Karatsuba substitution | No | `ring_action.rs` candidate compaction | `Refinement/ProjectionBatching.lean` |
| `Refinement.ProjectionBindingShapeArtifact` | Generated fixed-profile dimensions, role order, padded width, and 3,616-field production SIS input count | No | `f_prime_recursive_manifest/projection_binding_shape.rs` | `Refinement/ProjectionBindingShapeArtifact.lean` |
| `Refinement.ProjectionBindingSerialization` | Version-one field framing and exact 3,616-field plain preimage | No | `nifs/circuit/pi_rlc/projection/binding.rs` | `Refinement/ProjectionBindingSerialization.lean` |
| `Refinement.ProjectionIdentityCertificate` | Exact 31-role production schema, source topology, cost formulas, and concrete theorem parameters | No | `projection_identity_trace.rs`; generator regression | `Refinement/ProjectionIdentityCertificate.lean` |

Production ring-action checks are one-point projections. Their deterministic
guarantee is exact-or-bad-root, not unconditional coefficient equality. Rust
replays the production source-row trace, validates mixed-SSA non-escape and the
retained decoder, and drift-checks the generated compact plan. Lean separately
proves the corresponding abstract source-to-emitted equivalence and checks the
generated shape, schedule, and cost data. The concrete theorem connecting
actual source/emitted assignments and their column decoders to that abstract
equivalence remains open, so this is Rust-regression conformance plus a Lean
semantic model, not Lean-kernel closure of the selected row deletion. The
security bridge from Rust's two base-field limbs to the polynomial-coefficient
model required by the exact-or-bad-root argument also remains open.

`Claims.X` models the `padding.x` zero-tail predicate, but the concrete bridge
from Rust's row-major `D * m_in` matrices and verifier-derived active width is
still open; the ownership-map entry is not a conformance claim.

Spec: `specs/PiRlcAlgebra.spec.md`.
-/
