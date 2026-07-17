import Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism.PackedBlockAction

/-!
Compile-time surface checks for packed block-projection action compatibility.

| Stage path | Property under test | Failure caught |
|---|---|---|
| `nifs.pi_rlc.verify.authority.packed_y_zcol.action` | exact Phi81 action commutes with block projection | repeating the flat-column non-homomorphism in the replacement carrier |
-/

#check Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism.PackedBlockAction.blockRows_act
#check Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism.PackedBlockAction.packedYZcol_ringAction
