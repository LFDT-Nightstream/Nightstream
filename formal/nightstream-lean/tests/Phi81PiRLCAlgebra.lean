import Nightstream.SuperNeo.Concrete.Phi81Relation.PiRLCAlgebra

/-!
Focused theorem-surface checks for the typed Phi81 `PiRLC.Algebra` tree.

| Stage path | Regression |
|---|---|
| `nifs.pi_rlc.verify.challenge.membership` | algebra validity is exact production-set membership |
| `nifs.pi_rlc.verify.commitment_hom.key` | the key has exact verifier-row and complete-block domains |
| `nifs.pi_rlc.verify.commitment_hom.row` | each Ajtai row is the canonical finite key/assignment inner product |
| `nifs.pi_rlc.verify.commitment_hom.action` | one challenge action commutes with the typed commitment |
| `nifs.pi_rlc.verify.commitment_hom.finite` | public commitments use the identical finite challenge order |
| `nifs.pi_rlc.verify.commitment_hom.algebra` | the theorem has the exact commitment algebra-field shape |
| `nifs.pi_rlc.verify.public_input_hom.block` | the public carrier is exposed as exact 54-lane blocks |
| `nifs.pi_rlc.verify.public_input_hom.action` | the public-only action matches complete-assignment projection |
| `nifs.pi_rlc.verify.public_input_hom.finite` | the public-only fold uses the identical finite challenge order |
| `nifs.pi_rlc.verify.public_input_hom.algebra` | the theorem has the exact public-input algebra-field shape |
| `nifs.pi_rlc.verify.norm_growth.algebra` | the exact executable Phi81/production norm theorem closes the algebra field |
| `nifs.pi_rlc.verify.algebra` | the complete concrete algebra is assembled from the independently proved fields |
-/

namespace tests.Phi81PiRLCAlgebra

open Nightstream.SuperNeo.Concrete.Phi81Relation.PiRLCAlgebra

#check Challenge.challengeValid
#check Challenge.challengeValid_iff
#check Challenge.embedScalar_valid
#check Challenge.pairwiseSecure_of_lowNormInvertibility
#check Commitment.Key
#check Commitment.Value
#check Commitment.ajtaiRow
#check Commitment.commit
#check Commitment.combineCommitments
#check Commitment.commit_act
#check Commitment.commit_combine
#check Commitment.relation_commit_hom
#check PublicInput.publicBlock
#check PublicInput.publicAct
#check PublicInput.combinePublicInputs
#check PublicInput.projectPublicInput_act
#check PublicInput.projectPublicInput_combine
#check PublicInput.relation_publicInput_hom
#check Norm.relation_norm_growth
#check Algebra.concrete
#check Algebra.concrete_challengeValid
#check Algebra.concrete_combineAssignment

end tests.Phi81PiRLCAlgebra
