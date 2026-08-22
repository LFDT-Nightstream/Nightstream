import NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.Algebra
import NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.Challenge
import NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.Commitment
import NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.Norm
import NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.PublicInput

/-! Provenance: copied from `formal/nightstream-lean/Nightstream/SuperNeo/Concrete/Phi81Relation/PiRLCAlgebra.lean`
at commit `fb7a8a99aefbb8ebb5474681ecf80f1b95a1b7a2`; namespaces renamed, otherwise unchanged. -/

/-!
Curated construction surface for the concrete typed Phi81 `PiRLC.Algebra`.

Protocol: SuperNeo `Pi_RLC`.
Phase: independent semantic algebra required by the NIFS verifier.
Constraint family: none; this parent emits no rows.

Owns: the theorem and dependency boundary of every concrete algebra field. Each
child computes its verifier operation independently and proves the
corresponding homomorphism or bound. `Algebra.concrete` assembles those closed
fields without adding an oracle or caller-supplied law.

Does not own: Fiat--Shamir transcript derivation, Rust/R1CS refinement,
constraint accounting, security reductions, or permission to remove rows.

Emits constraints: no.

Authority boundary: the constructed algebra is model-level. It does not turn
the typed Ajtai map into a binding commitment or identify the semantic key,
sampler, transcript, or carrier with production Rust/R1CS data.

| Stage path | Algebra field | Mathematical owner | Exported guarantee or excluded boundary |
|---|---|---|---|
| `nifs.pi_rlc.verify.challenge` | `challengeValid` | `PiRLCAlgebra.Challenge`; exact Phi81 production-set membership | exact semantic predicate; transcript refinement is excluded |
| `nifs.pi_rlc.verify.assignment_hom` | `combineAssignment` | `EvaluationHomomorphism.PiRLCFinite.combineAssignments` | exact assignment homomorphism |
| `nifs.pi_rlc.verify.commitment_hom` | `combineCommitment`, `commit_hom` | `PiRLCAlgebra.Commitment` | exact typed commitment homomorphism |
| `nifs.pi_rlc.verify.public_input_hom` | `combinePublicInput`, `publicInput_hom` | `PiRLCAlgebra.PublicInput` | exact complete-carrier homomorphism |
| `nifs.pi_rlc.verify.evaluation_hom` | `combineEvaluations`, `evaluations_hom` | `EvaluationHomomorphism.PiRLCFinite` | exact evaluation homomorphism |
| `nifs.pi_rlc.verify.norm_growth` | `norm_growth` | `PiRLCAlgebra.Norm`; centered Goldilocks + exact executable Phi81 support + finite production bound | exact finite production bound |
| `nifs.pi_rlc.verify.algebra` | complete `PiRLC.Algebra` | `PiRLCAlgebra.Algebra.concrete` | assembled semantic algebra; production refinement is excluded |
-/
