import Nightstream.SuperNeo.Concrete.Phi81Relation.PiRLCAlgebra
import tests.Axioms.Support

/-!
Fail-closed dependency gate for the typed Phi81 `PiRLC.Algebra` tree.

| Stage path | Guarded theorem |
|---|---|
| `nifs.pi_rlc.verify.challenge.honest` | semantic sampler embeddings satisfy exact membership |
| `nifs.pi_rlc.verify.challenge.pairwise_security` | pairwise security exposes the analytic theorem boundary |
| `nifs.pi_rlc.verify.commitment_hom.action` | one challenge action commutes with the typed commitment |
| `nifs.pi_rlc.verify.commitment_hom.finite` | exact assignment and public commitment folds agree |
| `nifs.pi_rlc.verify.commitment_hom.algebra` | exact commitment algebra-field theorem |
| `nifs.pi_rlc.verify.public_input_hom.finite` | public-only and complete-assignment folds agree |
| `nifs.pi_rlc.verify.public_input_hom.algebra` | exact public-input algebra-field theorem |
| `nifs.pi_rlc.verify.norm_growth.algebra` | exact executable Phi81/production norm-growth theorem |
| `nifs.pi_rlc.verify.algebra` | complete concrete algebra and its exported field-identification theorems |
-/

/-- info: 'Nightstream.SuperNeo.Concrete.Phi81Relation.PiRLCAlgebra.Challenge.embedScalar_valid' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.Phi81Relation.PiRLCAlgebra.Challenge.embedScalar_valid

/-- info: 'Nightstream.SuperNeo.Concrete.Phi81Relation.PiRLCAlgebra.Challenge.pairwiseSecure_of_lowNormInvertibility' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.Phi81Relation.PiRLCAlgebra.Challenge.pairwiseSecure_of_lowNormInvertibility

/-- info: 'Nightstream.SuperNeo.Concrete.Phi81Relation.PiRLCAlgebra.Commitment.commit_act' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.Phi81Relation.PiRLCAlgebra.Commitment.commit_act

/-- info: 'Nightstream.SuperNeo.Concrete.Phi81Relation.PiRLCAlgebra.Commitment.commit_combine' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.Phi81Relation.PiRLCAlgebra.Commitment.commit_combine

/-- info: 'Nightstream.SuperNeo.Concrete.Phi81Relation.PiRLCAlgebra.Commitment.relation_commit_hom' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.Phi81Relation.PiRLCAlgebra.Commitment.relation_commit_hom

/-- info: 'Nightstream.SuperNeo.Concrete.Phi81Relation.PiRLCAlgebra.PublicInput.projectPublicInput_combine' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.Phi81Relation.PiRLCAlgebra.PublicInput.projectPublicInput_combine

/-- info: 'Nightstream.SuperNeo.Concrete.Phi81Relation.PiRLCAlgebra.PublicInput.relation_publicInput_hom' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.Phi81Relation.PiRLCAlgebra.PublicInput.relation_publicInput_hom

/-- info: 'Nightstream.SuperNeo.Concrete.Phi81Relation.PiRLCAlgebra.Norm.Finite.relation_norm_growth' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.Phi81Relation.PiRLCAlgebra.Norm.Finite.relation_norm_growth

/-- info: 'Nightstream.SuperNeo.Concrete.Phi81Relation.PiRLCAlgebra.Algebra.concrete' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.Phi81Relation.PiRLCAlgebra.Algebra.concrete

/-- info: 'Nightstream.SuperNeo.Concrete.Phi81Relation.PiRLCAlgebra.Algebra.concrete_challengeValid' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.Phi81Relation.PiRLCAlgebra.Algebra.concrete_challengeValid

/-- info: 'Nightstream.SuperNeo.Concrete.Phi81Relation.PiRLCAlgebra.Algebra.concrete_combineAssignment' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.Phi81Relation.PiRLCAlgebra.Algebra.concrete_combineAssignment
