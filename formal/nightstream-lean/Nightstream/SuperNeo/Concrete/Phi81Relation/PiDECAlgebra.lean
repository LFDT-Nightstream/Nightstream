import Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.Algebra

/-!
Curated construction surface for the concrete typed Phi81 `PiDEC.Algebra`.

Protocol: SuperNeo `Pi_DEC`.
Phase: independent semantic algebra required by the NIFS verifier.
Constraint family: none; this parent emits no rows.

Owns: the dependency boundary and theorem owner of each concrete algebra field.

Does not own: Ajtai binding or MSIS security, child CE membership, PiCCS or
PiRLC acceptance, NIFS composition, Rust/R1CS refinement, constraint
accounting, security reductions, or permission to remove rows.

Emits constraints: no.

Authority boundary: every result is a model-level theorem over the typed Phi81
relation. Commitment recomposition consumes only the typed verifier key and
public child commitments; public-input recomposition consumes only public
child inputs. The complete algebra is assembled explicitly from these theorem
owners and does not advertise security or Rust conformance.

| Stage path | Algebra field | Mathematical owner | Exported guarantee |
|---|---|---|---|
| `nifs.pi_dec.verify.radix.split` | `splitAssignment` | `PiDECAlgebra.Radix` | deterministic bounded signed-binary split with exact total fallback |
| `nifs.pi_dec.verify.radix.recompose` | `recomposeAssignment`, `split_recompose` | `PiDECAlgebra.Radix` | exact recomposition of the complete typed assignment |
| `nifs.pi_dec.verify.radix.split_norm` | `split_norm` | `PiDECAlgebra.Radix` | strict production `B = 16384` child norm |
| `nifs.pi_dec.verify.radix.recompose_norm` | `recompose_norm` | `PiDECAlgebra.Radix` | strict recomposition norm for fourteen bound-`2` children |
| `nifs.pi_dec.verify.commitment_hom` | `recomposeCommitment`, `commit_hom` | `PiDECAlgebra.Commitment` | typed Ajtai homomorphism; binding security remains separate |
| `nifs.pi_dec.verify.public_input_hom` | `recomposePublicInput`, `publicInput_hom` | `PiDECAlgebra.PublicInput` | homomorphism for the complete typed public carrier |
| `nifs.pi_dec.verify.evaluation_hom` | `recomposeEvaluations`, `evaluations_hom` | `EvaluationHomomorphism.PiDEC` | exact evaluation homomorphism |
| `nifs.pi_dec.verify.algebra` | complete `PiDEC.Algebra` | `PiDECAlgebra.Algebra.concrete` | assembly from the independently proved fields |
-/
