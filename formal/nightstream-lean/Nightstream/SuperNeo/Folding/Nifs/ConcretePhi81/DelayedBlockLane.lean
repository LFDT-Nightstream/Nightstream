import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.PiCcsDomains
import Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism

/-!
Typed pending state for the production delayed packed-`yZcol` check.

Protocol: concrete Phi81 NIFS.
Phase: one output is carried to the next combined BlockLane NC check.
Constraint family: typed state only; this file emits no rows.

Assurance tier: model-level carrier.

Owns: exactly the old production BlockLane point and one complete 54-lane
packed parent vector.

Does not own: acceptance, continuity, transcript replay, raw-child authority,
terminal closure, Rust/R1CS refinement, hashing, costs, or row removal.

Emits constraints: none.

Authority boundary: constructing this value establishes no projection truth.
The vector becomes authoritative only after a successor or terminal check
compares it with actual raw child assignments. Padded lanes are absent from
the type and are verifier-computed zeros in the combined-NC adapter.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_ccs.nc.delayed.state` | carry the exact old block point and 54 active packed-parent lanes | direct dataflow | `ProductionDelayedBlockLane` |
-/

namespace Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism

/-- One pending packed parent produced by the prior fold. The fixed production
domain gives a 21-coordinate old block point; `RingK` gives exactly 54 active
coefficients. -/
structure ProductionDelayedBlockLane where
  oldBlock : CubePoint K PiCcsDomains.production.nc.blockVariables
  parentYZcol : RingK

namespace ProductionDelayedBlockLane

@[ext] theorem ext
    (left right : ProductionDelayedBlockLane)
    (oldBlock : left.oldBlock = right.oldBlock)
    (parentYZcol : left.parentYZcol = right.parentYZcol) :
    left = right := by
  cases left
  cases right
  simp_all

end ProductionDelayedBlockLane

end Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
