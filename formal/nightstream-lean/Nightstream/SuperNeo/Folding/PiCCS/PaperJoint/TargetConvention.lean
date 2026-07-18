import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Coefficients

/-!
Carried-evaluation target convention and paper erratum boundary.

Owns: a typed distinction between the literal local target exponents in the
paper's displayed `T` and a candidate absolute shifted exponent layout.

Does not own: polynomial identities, CCS/norm table construction, a repair to
the paper, SumCheck, transcript semantics, Rust, R1CS, or production policy.

Emits constraints: no.

Authority boundary: the model records the coherent absolute convention
`T_abs = C^(2K+k) * T_local` as a candidate pending protocol review. This is
an explicit modeling convention, not a claim that the literal displayed
equations already agree or that the candidate is production-approved.

| Candidate choice | Mathematical meaning | Proof owner / policy boundary |
|---|---|---|
| target exponent | `2K+k+I(i,j,l)` | equality with the declared shifted layout is proved here; protocol approval is separate |
| residual orientation | `T_local - sum Eval_local` | the coherent-absolute signed identity is proved in `SignedJointIdentity`; protocol approval is separate |
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

/-- Coherent candidate convention pending protocol review. Defining it here
does not select or approve it for a production verifier. -/
def candidateCarriedTargetConvention : CarriedTargetConvention :=
  .coherentAbsolute

/-- Candidate sign convention pending protocol review. The coherent-absolute
signed identity itself is owned by `SignedJointIdentity`. -/
def candidateCarriedResidualOrientation : CarriedResidualOrientation :=
  .targetMinusEvaluation

/-- The candidate target exponent is definitionally the absolute exponent in
this model's declared carried layout. The corresponding `Q` identity is owned
by `SignedJointIdentity`, not duplicated in this convention leaf. -/
theorem candidateTargetExponent_eq_declaredCarriedExponent
    {shape : Shape}
    (coordinate : CarriedCoordinate shape) :
    candidateCarriedTargetConvention.exponent coordinate =
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
