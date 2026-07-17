import Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.Types
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.CCSResidualTable

/-!
Verifier-visible carriers for the production-shaped Split-NC `Pi_CCS`
semantics.

Protocol: SuperNeo `Pi_CCS`, with FE and NC checked on separate Boolean
domains.
Phase: public semantic input and raw output-evaluation message.
Constraint family: typed authority boundaries only; this file emits no rows.

Owns: the exact public fields needed by the FE/NC semantic checks and the raw
`yRing`/`yZcol` output product whose evaluation points are deliberately
absent.

Does not own: commitments, public `x`, transcript context, challenges,
SumCheck messages, output-point derivation, FE/NC polynomials, Poseidon2,
Rust, R1CS, row removal, or constraint counts.

Emits constraints: no.

Authority boundary: `PublicInput` contains the verifier-owned constraint
polynomial, the prior row point, and the running coefficient claims. It cannot
read matrices or assignments. `OutputMessage` contains values only; the
verifier must derive `rPrime` and `sPrime` and prove that both output branches
are bound to the same authoritative sources.

| Stage path | Carrier | Mathematical obligation | Authority class |
|---|---|---|---|
| `nifs.pi_ccs.verify.input.structure` | `constraintPolynomial` | one sparse CCS polynomial shared by every fresh source | verifier-owned |
| `nifs.pi_ccs.verify.input.running` | `priorPoint`, `claimedYRing` | prior CE evaluation claims consumed by FE | public claim |
| `nifs.pi_ccs.verify.output.y_ring` | `OutputMessage.yRing` | new row-point coefficient evaluations | checked prover payload |
| `nifs.pi_ccs.verify.output.y_zcol` | `OutputMessage.yZcol` | new column-point assignment projection | checked prover payload; independent binding required |
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.OutputClaims

/-- Exact semantic input surface visible to the Split-NC executable checker.

The commitment and public-input forwarding fields of a complete `Pi_CCS`
instance are intentionally outside this polynomial subprotocol carrier. They
must be composed as separately named NIFS obligations rather than smuggled in
through hidden source data. -/
structure PublicInput (shape : SemanticShape) where
  constraintPolynomial :
    CCSResidualTable.ConstraintPolynomial F shape.matrixCount
  priorPoint : CubePoint K shape.rowVariables
  claimedYRing : Fin shape.runningCount -> Fin shape.matrixCount ->
    Fin ringDegree -> K

namespace PublicInput

/-- Equality of the three authoritative public families is equality of the
complete semantic verifier input. -/
@[ext] theorem ext
    {shape : SemanticShape}
    (left right : PublicInput shape)
    (constraintPolynomial :
      left.constraintPolynomial = right.constraintPolynomial)
    (priorPoint : left.priorPoint = right.priorPoint)
    (claimedYRing : left.claimedYRing = right.claimedYRing) :
    left = right := by
  cases left
  cases right
  simp_all

end PublicInput

/-- Raw output values sent after the two SumChecks. Neither output point is a
field of this message. -/
abbrev OutputMessage (shape : SemanticShape) := OutputClaims.Claims shape

end Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier
