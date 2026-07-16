import Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra
import tests.Axioms.Support

/-!
Fail-closed dependency gate for typed Phi81 Π_DEC public recomposition.

| Stage path | Guarded theorem |
|---|---|
| `nifs.pi_dec.verify.commitment_hom.scale` | one base-field scale commutes with the typed Ajtai commitment |
| `nifs.pi_dec.verify.commitment_hom.finite` | finite assignment and public commitment folds agree |
| `nifs.pi_dec.verify.commitment_hom.algebra` | exact commitment algebra-field theorem |
| `nifs.pi_dec.verify.public_input_hom.finite` | finite assignment and public-input folds agree |
| `nifs.pi_dec.verify.public_input_hom.algebra` | exact public-input algebra-field theorem |
| `nifs.pi_dec.verify.algebra` | complete concrete algebra assembly |
-/

/-! Commitment proofs inherit finite typed Ajtai traversal; public-input
proofs require only extensionality and the field quotient. -/

/-- info: 'Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.Commitment.commit_scale' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.Commitment.commit_scale

/-- info: 'Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.Commitment.commit_combine' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.Commitment.commit_combine

/-- info: 'Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.Commitment.relation_commit_hom' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.Commitment.relation_commit_hom

/-- info: 'Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.PublicInput.projectPublicInput_combine' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.PublicInput.projectPublicInput_combine

/-- info: 'Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.PublicInput.relation_publicInput_hom' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.PublicInput.relation_publicInput_hom

/-- info: 'Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.Algebra.concrete' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.Algebra.concrete
