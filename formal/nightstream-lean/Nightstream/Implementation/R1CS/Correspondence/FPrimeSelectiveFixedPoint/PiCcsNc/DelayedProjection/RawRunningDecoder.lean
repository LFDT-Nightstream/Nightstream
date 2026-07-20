import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder.ArtifactRefinement

/-!
Public facade for the generated bounded running-`X` public-prefix decoder.

Assurance tier: artifact-checked for the generated fixed profile.

Owns: the public import boundary for exact generated coordinate ownership and
typed 270-coordinate fixture decoding.

Does not own: combined-NC rows, protocol acceptance, transcript scheduling,
commitment binding, costs, or row-removal permission.

Emits constraints: none; facade only.

| Stable stage path | Obligation | Authority |
|---|---|---|
| `f_prime.pi_ccs_nc.delayed_projection.running_x_prefix_decoder` | Export the generated-column refinement contract | artifact-checked |

This facade exports the exact fixed-profile coordinate schema, generated
source/final column ownership, and construction of a bounded typed fixture
from `CeClaim.X` physical columns. Full packed-witness decoding, combined-NC rows, acceptance,
transcript scheduling, commitment authority, and row-removal permission remain
open.
-/
