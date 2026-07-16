import Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism.Authority

/-!
Focused surface checks for the combined-parent `yZcol` authority bridge.

Owns: regression coverage that the public bridge takes one `PiDEC` attempt and
retains the explicit parent-opening binding-collision alternative.

Does not own: construction of the production algebra, the upstream `PiRLC`
source combination, collision hardness, NIFS closure, or row removal.

| Stage path | Regression |
|---|---|
| `nifs.pi_dec.verify.authority.parent_opening.public_shape` | the bridge is generic over a whole-ring public width; raw 257-field input is not admitted |
| `nifs.pi_dec.verify.authority.parent_opening.recomposition` | production assignment recomposition remains an explicit refinement premise |
| `nifs.pi_dec.verify.authority.parent_opening.binding` | valid CE openings plus acceptance yield equality or the named collision |
| `nifs.pi_dec.verify.authority.parent_opening.y_zcol` | the same one-parent dichotomy transports all 54 lanes |
| `nifs.pi_rlc.verify.authority.combined_assignment.y_zcol` | remains outside this module rather than being modeled as per-source `PiDEC` attempts |
-/

namespace tests.PiCcsOutputClaimsYZcolAuthority

open Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism.Authority

#check PiDECParentOpening.ConcretePiDEC.UsesProductionAssignmentRecomposition
#check PiDECParentOpening.ConcretePiDEC.toYZcolAssignment_recompose
#check PiDECParentOpening.parentAssignment_eq_recompose_or_bindingCollision
#check PiDECParentOpening.parentYZcol_transport_or_bindingCollision

end tests.PiCcsOutputClaimsYZcolAuthority
