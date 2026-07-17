import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Sampling
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Parameters

/-!
Raw output-claim carriers shared by the independent `Pi_CCS` semantics and
the executable Split-NC verifier.

Owns: the verifier-derived row/column point pair and the prover-shaped
`yRing`/`yZcol` value product, with no semantic source binding.

Does not own: canonical claim construction, source assignments, matrices,
SumCheck, transcript derivation, Rust, R1CS, rows, costs, or row removal.

Emits constraints: no.

Authority boundary: these structures contain data only. Importing them cannot
establish that either point came from a verifier transcript or that either
claim family agrees with authoritative matrices and assignments.

| Stage path | Carrier | Mathematical obligation | Authority class |
|---|---|---|---|
| `nifs.pi_ccs.output.point` | `VerifierPoints` | carry the verifier-derived FE row and NC column points | computed upstream |
| `nifs.pi_ccs.output.y_ring` | `Claims.yRing` | carry every active row-point coefficient evaluation | untrusted payload |
| `nifs.pi_ccs.output.y_zcol` | `Claims.yZcol` | carry every active column-point assignment projection | untrusted payload |
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.OutputClaims

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc

/-- The two output points supplied by verifier conclusions. They remain
explicit inputs to canonical construction and are never accepted from a claim. -/
structure VerifierPoints
    (shape : SemanticShape) (domain : FlatNcDomain) where
  rPrime : CubePoint K shape.rowVariables
  sPrime : CubePoint K domain.columnVariables

/-- Complete active output-claim product in canonical source order: fresh
sources first, then running sources. `Claims` itself asserts no correctness. -/
structure Claims (shape : SemanticShape) where
  yRing : Fin shape.sourceCount -> Fin shape.matrixCount ->
    Fin ringDegree -> K
  yZcol : Fin shape.sourceCount -> Fin ringDegree -> K

/-- Two claim products are equal when every active output coordinate is equal. -/
@[ext] theorem Claims.ext
    {shape : SemanticShape}
    (left right : Claims shape)
    (yRing : forall source matrix lane,
      left.yRing source matrix lane = right.yRing source matrix lane)
    (yZcol : forall source lane,
      left.yZcol source lane = right.yZcol source lane) :
    left = right := by
  cases left with
  | mk leftYRing leftYZcol =>
      cases right with
      | mk rightYRing rightYZcol =>
          simp only [Claims.mk.injEq]
          constructor
          · funext source matrix lane
            exact yRing source matrix lane
          · funext source lane
            exact yZcol source lane

end Nightstream.SuperNeo.Folding.PiCCS.OutputClaims
