import Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism.PiDEC
import Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.Commitment
import Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.PublicInput
import Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.Radix

/-!
Curated status surface for the concrete typed Phi81 `PiDEC.Algebra` work.

Protocol: SuperNeo `Pi_DEC`.
Phase: independent semantic algebra required by the NIFS verifier.
Constraint family: none; this parent emits no rows.

Owns: the dependency and completion status of each concrete algebra field.

Does not own: Ajtai binding or MSIS security, a complete `PiDEC.Algebra` value,
child CE membership, Rust/R1CS refinement, constraint accounting, security
reductions, or permission to remove rows.

Emits constraints: no.

Authority boundary: every result is a model-level theorem over the typed Phi81
relation. Commitment recomposition consumes only the typed verifier key and
public child commitments; public-input recomposition consumes only public
child inputs. Assembly of the complete algebra remains a separate explicit
step so this facade does not silently advertise security or Rust conformance.

| Stage path | Algebra field | Mathematical owner | Status |
|---|---|---|---|
| `nifs.pi_dec.verify.radix.split` | `splitAssignment` | `PiDECAlgebra.Radix` | deterministic bounded signed-binary split with exact total fallback proved |
| `nifs.pi_dec.verify.radix.recompose` | `recomposeAssignment`, `split_recompose` | `PiDECAlgebra.Radix` | proved for the complete typed assignment |
| `nifs.pi_dec.verify.radix.split_norm` | `split_norm` | `PiDECAlgebra.Radix` | proved for strict production `B = 16384` |
| `nifs.pi_dec.verify.radix.recompose_norm` | `recompose_norm` | `PiDECAlgebra.Radix` | proved for fourteen strict-`2` children |
| `nifs.pi_dec.verify.commitment_hom` | `recomposeCommitment`, `commit_hom` | `PiDECAlgebra.Commitment` | proved for the typed Ajtai map; binding security open |
| `nifs.pi_dec.verify.public_input_hom` | `recomposePublicInput`, `publicInput_hom` | `PiDECAlgebra.PublicInput` | proved for the complete typed public carrier |
| `nifs.pi_dec.verify.evaluation_hom` | `recomposeEvaluations`, `evaluations_hom` | `EvaluationHomomorphism.PiDEC` | proved |
| `nifs.pi_dec.verify.algebra` | complete `PiDEC.Algebra` | not assembled in this facade | all mathematical fields available independently |
-/
