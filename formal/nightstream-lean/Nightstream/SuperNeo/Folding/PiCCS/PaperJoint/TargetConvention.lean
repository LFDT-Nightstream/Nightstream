import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Coefficients

/-!
Carried-evaluation target convention for the corrected paper.

Owns: a typed distinction between the paper's local helper target and its
selected absolute shifted target.

Does not own: polynomial identities, CCS/norm table construction, SumCheck,
transcript semantics, Rust, R1CS, or production policy.

Emits constraints: no.

Authority boundary: the corrected paper selects the coherent absolute
convention `T_abs = C^(2K+k) * T_local`. This file records that selection. It
does not prove the joint polynomial identity or implementation conformance.

| Selected choice | Mathematical meaning | Proof owner |
|---|---|---|
| target exponent | `2K+k+I(i,j,l)` | equality with the declared shifted layout is proved here |
| residual orientation | `T_local - sum Eval_local` | the coherent-absolute signed identity is proved in `SignedJointIdentity` |
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

/-- The two exponent conventions must remain visibly distinct. -/
inductive CarriedTargetConvention where
  | literalLocal
  | coherentAbsolute
deriving Repr, DecidableEq

/-- Sign orientation for a future concrete carried-evaluation residual. -/
inductive CarriedResidualOrientation where
  | targetMinusEvaluation
  | evaluationMinusTarget
deriving Repr, DecidableEq

namespace CarriedTargetConvention

/-- Target exponent selected by a convention. -/
def exponent
    {shape : Shape}
    (convention : CarriedTargetConvention)
    (coordinate : CarriedCoordinate shape) : Nat :=
  match convention with
  | .literalLocal => coordinate.localGammaExponent
  | .coherentAbsolute => coordinate.gammaExponent

end CarriedTargetConvention

/-- Coherent absolute convention selected by the corrected paper. -/
def selectedCarriedTargetConvention : CarriedTargetConvention :=
  .coherentAbsolute

/-- Sign convention selected by the corrected paper. The coherent-absolute
signed identity itself is owned by `SignedJointIdentity`. -/
def selectedCarriedResidualOrientation : CarriedResidualOrientation :=
  .targetMinusEvaluation

/-- The selected target exponent is definitionally the absolute exponent in
this model's declared carried layout. The corresponding `Q` identity is owned
by `SignedJointIdentity`, not duplicated in this convention leaf. -/
theorem selectedTargetExponent_eq_declaredCarriedExponent
    {shape : Shape}
    (coordinate : CarriedCoordinate shape) :
    selectedCarriedTargetConvention.exponent coordinate =
      coordinate.gammaExponent := by
  rfl

/-- Negative exponent-level audit result: existence of a carried coordinate
forces `k > 0`, so the literal local exponent differs from this model's
declared shifted exponent. This theorem does not formalize `Q`, support sets,
or evaluation at a gamma value. -/
theorem literalTargetExponent_ne_declaredCarriedExponent
    {shape : Shape}
    (coordinate : CarriedCoordinate shape) :
    CarriedTargetConvention.literalLocal.exponent coordinate ≠
      coordinate.gammaExponent := by
  change coordinate.localGammaExponent ≠
    shape.carriedEvaluationOffset + coordinate.localGammaExponent
  have runningBound := coordinate.running.isLt
  rw [Shape.carriedEvaluationOffset_eq]
  omega

end Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
