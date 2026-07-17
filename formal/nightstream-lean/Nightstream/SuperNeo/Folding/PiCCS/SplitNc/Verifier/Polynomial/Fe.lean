import Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.Semantics
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Fe.Parameters
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Types.SourceProjection
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConstraintPolynomialLift
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.NumericBooleanDomain
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.SignedJointIdentity
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.SumCheckTruthPath
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Phi81CoefficientKernel

/-!
Independent production-shaped FE polynomial for Split-NC `Pi_CCS`.

Assurance tier: model-level; not Rust-conformant or security-reduced.

Protocol: SuperNeo `Pi_CCS`, with FE and NC checked separately.
Phase: FE initial claim and terminal polynomial before executable SumCheck.
Constraint family: fresh CCS compression and running CE compression only;
this file emits no rows.

Owns: the verifier challenge carrier; 54-lane zero-extended evaluation;
source-derived `yRing`; the lifted CCS polynomial; the fresh and carried
gamma blocks; one shared terminal formula; and its semantic/message
specializations.

Does not own: the FE point and physical degree parameters in
`Fe.Parameters`; the paper's one-joint polynomial; NC; Fiat--Shamir
derivation; SumCheck round checking or degree soundness; output `yZcol`;
proof that raw Rust/R1CS lanes 54 through 63 are zero; native/circuit
conformance; row emission; row removal; or constraint counts.

Emits constraints: no.

Authority boundary: semantic evaluation and verifier evaluation instantiate
the same `terminalFromYRing`. Hidden matrices and assignments enter only
through `sourceYRingAt`; the verifier path sees only `PublicInput`, derived
coins, a derived terminal point, and raw output values. Coins and points are
parameters in this algebraic layer, not certificate fields; transcript
refinement must derive them later.

This is production-shaped, not paper-exact. The paper specifies a single
row-domain polynomial containing FE and NC. The split product-domain formula,
its mixing schedule, and its relation to `Semantics.Fe.Truth` require their
own soundness and completeness theorem.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_ccs.fe.domain.row_lane` | SumCheck coordinates are `row ++ lane` | computed | `Fe.Parameters` |
| `nifs.pi_ccs.fe.domain.profile` | nonempty row/fresh domains and exact 64-lane cube | security boundary | `SupportedProfile` |
| `nifs.pi_ccs.fe.domain.lane.live` | 54 Phi81 lanes fit in the padded lane cube | derived from profile | `LaneCovers`, `liveLane` |
| `nifs.pi_ccs.fe.output.lane_mle` | `sum_rho chi_rho(a) * y[rho]`, with no authority in padded lanes | computed | `paddedLaneEvaluation` |
| `nifs.pi_ccs.fe.output.source` | every semantic `yRing` comes from the sole matrix/assignment source | computed | `sourceYRingAt` |
| `nifs.pi_ccs.fe.fresh.ccs` | `sum_i gamma^i f(Y[i,*,constant])` | checked | `freshTermFromYRing` |
| `nifs.pi_ccs.fe.running.exponent` | `K + h + jN`, shared by initial and terminal | computed | `carriedGammaExponent` |
| `nifs.pi_ccs.fe.running.eval` | `sum_j,h gamma^(K+h+jN) Y[K+h,j](a)` | checked | `carriedTermFromYRing` |
| `nifs.pi_ccs.fe.terminal` | equality-gated fresh block plus `gamma^N` shifted carried block | checked | `terminalFromYRing` |
| `nifs.pi_ccs.fe.initial` | claimed carried block at verifier challenge `alpha` | computed | `initial` |
| `nifs.pi_ccs.fe.degree.row` | `max(canonical equality-gated CCS degree, 2)` | computed; model-level proof closed | `Fe.Parameters`, `SumCheck.Fe.expectedRowRound_bounded` |
| `nifs.pi_ccs.fe.degree.lane` | padded-lane MLE times one lane selector | computed; model-level proof closed | `Fe.Parameters`, `SumCheck.Fe.expectedLaneRound_quadratic` |
| assurance | semantic/message anti-drift | source-bound message terminal equals semantic terminal | derived | `terminalFromMessage_eq_qAtPoint_of_yRing_eq` |
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Fe

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.OutputClaims
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources

/-- Active `yRing` values in canonical fresh-then-running source order. -/
abbrev YRingValues (shape : SemanticShape) :=
  Fin shape.sourceCount -> Fin shape.matrixCount -> Fin ringDegree -> K

/-- FE only needs Phi81 lanes to fit in its padded lane cube. NC column
coverage is a separate obligation and is intentionally not coupled here. -/
def LaneCovers (domain : FlatNcDomain) : Prop :=
  ringDegree <= domain.laneCount

/-- FE shape guards enforced by the current native/circuit protocol surface.

This is deliberately narrower than algebraic lane coverage: native rejects an
empty row or fresh batch and derives a 64-lane cube for the 54 active Phi81
coefficients. The valid one-fresh/no-running case remains supported; its
initial carried sum is canonically zero. A later Rust refinement must prove
its decoded dimensions instantiate this predicate. -/
structure SupportedProfile
    (shape : SemanticShape) (domain : FlatNcDomain) : Prop where
  row_nonempty : 0 < shape.rowVariables
  fresh_nonempty : 0 < shape.freshCount
  lane_variables : domain.laneVariables = 6

namespace SupportedProfile

/-- The exact 64-lane production profile covers all 54 active Phi81 lanes. -/
theorem laneCovers
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (profile : SupportedProfile shape domain) :
    LaneCovers domain := by
  simp [LaneCovers, FlatNcDomain.laneCount, profile.lane_variables, ringDegree]

end SupportedProfile

/-- Full flat-domain coverage implies the narrower FE lane obligation. -/
theorem laneCovers_of_flatCovers
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (covers : domain.Covers shape) :
    LaneCovers domain :=
  covers.2

/-- Embed one active Phi81 lane into the padded Boolean lane domain without
changing its numeric index. -/
def liveLane
    {domain : FlatNcDomain}
    (covers : LaneCovers domain)
    (lane : Fin ringDegree) : Fin domain.laneCount :=
  ⟨lane.val, Nat.lt_of_lt_of_le lane.isLt covers⟩

@[simp] theorem liveLane_val
    {domain : FlatNcDomain}
    (covers : LaneCovers domain)
    (lane : Fin ringDegree) :
    (liveLane covers lane).val = lane.val := by
  rfl

/-- Verifier challenges consumed by the FE polynomial. A later transcript
machine must derive every field of this record. -/
structure Coins (shape : SemanticShape) (domain : FlatNcDomain) where
  alpha : CubePoint K domain.laneVariables
  betaA : CubePoint K domain.laneVariables
  betaR : CubePoint K shape.rowVariables
  gamma : K

/-- Zero-extended MLE of 54 active Phi81 lanes. No padded lane value is an
input: lanes outside `ringDegree` contribute by absence, hence as zero. -/
def paddedLaneEvaluation
    {domain : FlatNcDomain}
    (covers : LaneCovers domain)
    (values : Fin ringDegree -> K)
    (point : CubePoint K domain.laneVariables) : K :=
  SignedJointIdentity.sumMap ConcreteCarrier.extensionOps
    (canonicalFinIndices ringDegree) fun lane =>
      K.mul
        (NumericBooleanDomain.tensorWeight ConcreteCarrier.extensionOps
          (liveLane covers lane) point)
        (values lane)

/-- All semantic `yRing` values at one verifier-owned row point, derived from
the sole matrix source and authoritative assignments. -/
def sourceYRingAt
    {shape : SemanticShape}
    (data : Data shape)
    (row : CubePoint K shape.rowVariables) : YRingValues shape :=
  fun source matrix lane =>
    yRingForAssignment data (data.assignment source) row matrix lane

/-- Structural lift of the public sparse CCS polynomial into the quadratic
extension. Exponents and term order are unchanged. -/
def liftedConstraintPolynomial
    {shape : SemanticShape}
    (input : PublicInput shape) :
    CCSResidualTable.ConstraintPolynomial K shape.matrixCount :=
  ConstraintPolynomialLift.liftConstraintPolynomial
    K.embed input.constraintPolynomial

/-- Fresh CCS contribution `sum_i gamma^i f(Y[i,*,constant])`. -/
def freshTermFromYRing
    {shape : SemanticShape}
    (input : PublicInput shape)
    (gamma : K)
    (yRing : YRingValues shape) : K :=
  SignedJointIdentity.sumMap ConcreteCarrier.extensionOps
    (canonicalFinIndices shape.freshCount) fun fresh =>
      SignedJointIdentity.gammaTerm ConcreteCarrier.extensionOps gamma
        fresh.val <|
        CCSResidualTable.evaluatePolynomial ConcreteCarrier.extensionOps
          (liftedConstraintPolynomial input) fun matrix =>
            yRing (Data.freshIndex fresh) matrix
              Phi81CoefficientKernel.constant

/-- One owner for the production carried gamma exponent. Both the verifier
initial claim and the terminal polynomial call this definition. -/
def carriedGammaExponent
    (shape : SemanticShape)
    (running : Fin shape.runningCount)
    (matrix : Fin shape.matrixCount) : Nat :=
  shape.freshCount + running.val + matrix.val * shape.sourceCount

/-- Unshifted running CE contribution
`sum_j,h gamma^(K+h+jN) Y[K+h,j](lanePoint)`. -/
def carriedTermFromYRing
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (covers : LaneCovers domain)
    (gamma : K)
    (lanePoint : CubePoint K domain.laneVariables)
    (yRing : YRingValues shape) : K :=
  SignedJointIdentity.sumMap ConcreteCarrier.extensionOps
    (canonicalFinIndices shape.matrixCount) fun matrix =>
      SignedJointIdentity.sumMap ConcreteCarrier.extensionOps
        (canonicalFinIndices shape.runningCount) fun running =>
          SignedJointIdentity.gammaTerm ConcreteCarrier.extensionOps gamma
            (carriedGammaExponent shape running matrix) <|
            paddedLaneEvaluation covers
              (yRing (Data.runningIndex running) matrix) lanePoint

/-- One anti-drift FE terminal formula shared by the semantic and raw-message
paths. The carried block receives the production outer `gamma^N` shift. -/
def terminalFromYRing
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (profile : SupportedProfile shape domain)
    (input : PublicInput shape)
    (coins : Coins shape domain)
    (point : Point shape domain)
    (yRing : YRingValues shape) : K :=
  K.add
    (K.mul
      (K.mul
        (SumCheckTruthPath.pointEquality ConcreteCarrier.extensionOps
          point.lane coins.betaA)
        (SumCheckTruthPath.pointEquality ConcreteCarrier.extensionOps
          point.row coins.betaR))
      (freshTermFromYRing input coins.gamma yRing))
    (K.mul
      (K.mul
        (SumCheckTruthPath.pointEquality ConcreteCarrier.extensionOps
          point.lane coins.alpha)
        (SumCheckTruthPath.pointEquality ConcreteCarrier.extensionOps
          point.row input.priorPoint))
      (SignedJointIdentity.gammaTerm ConcreteCarrier.extensionOps coins.gamma
        shape.sourceCount
        (carriedTermFromYRing profile.laneCovers coins.gamma point.lane yRing)))

/-- Verifier-owned initial claim from public running coefficient claims only. -/
def initial
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (profile : SupportedProfile shape domain)
    (input : PublicInput shape)
    (coins : Coins shape domain) : K :=
  SignedJointIdentity.gammaTerm ConcreteCarrier.extensionOps coins.gamma
    shape.sourceCount <|
    SignedJointIdentity.sumMap ConcreteCarrier.extensionOps
      (canonicalFinIndices shape.matrixCount) fun matrix =>
        SignedJointIdentity.sumMap ConcreteCarrier.extensionOps
          (canonicalFinIndices shape.runningCount) fun running =>
            SignedJointIdentity.gammaTerm ConcreteCarrier.extensionOps
              coins.gamma
              (carriedGammaExponent shape running matrix) <|
              paddedLaneEvaluation profile.laneCovers
                (input.claimedYRing running matrix) coins.alpha

/-- Semantic FE polynomial at an arbitrary typed product point. -/
def qAtPoint
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (profile : SupportedProfile shape domain)
    (data : Data shape)
    (coins : Coins shape domain)
    (point : Point shape domain) : K :=
  terminalFromYRing profile (PublicInput.ofSources data) coins point
    (sourceYRingAt data point.row)

/-- Raw-message FE terminal. `yZcol` is deliberately not consumed here. -/
def terminalFromMessage
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (profile : SupportedProfile shape domain)
    (input : PublicInput shape)
    (coins : Coins shape domain)
    (point : Point shape domain)
    (message : OutputMessage shape) : K :=
  terminalFromYRing profile input coins point message.yRing

/-- Wrong-arity coordinate lists fail closed. Exact-length lists are decoded
as row coordinates followed by lane coordinates. -/
def polynomial
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (profile : SupportedProfile shape domain)
    (data : Data shape)
    (coins : Coins shape domain)
    (coordinates : List K) : Option K :=
  if length : coordinates.length =
      shape.rowVariables + domain.laneVariables then
    some (qAtPoint profile data coins
      (Point.ofCoordinates coordinates length))
  else
    none

/-- The fail-closed list evaluator agrees with the typed FE polynomial on the
exact row-then-lane serialization. -/
theorem polynomial_coordinates_eq_qAtPoint
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (profile : SupportedProfile shape domain)
    (data : Data shape)
    (coins : Coins shape domain)
    (point : Point shape domain) :
    polynomial profile data coins point.coordinates =
      some (qAtPoint profile data coins point) := by
  unfold polynomial
  rw [dif_pos point.coordinates_length]
  rw [Point.ofCoordinates_coordinates]

/-- Any coordinate list with the wrong total round count is rejected by the
polynomial carrier rather than truncated or padded. -/
theorem polynomial_eq_none_of_length_ne
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (profile : SupportedProfile shape domain)
    (data : Data shape)
    (coins : Coins shape domain)
    (coordinates : List K)
    (different : coordinates.length ≠
      shape.rowVariables + domain.laneVariables) :
    polynomial profile data coins coordinates = none := by
  simp [polynomial, different]

/-- A message is output-mismatched when any active `yRing` value differs from
the source-derived value at the verifier-owned row point. -/
def OutputMismatch
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (data : Data shape)
    (point : Point shape domain)
    (message : OutputMessage shape) : Prop :=
  message.yRing ≠ sourceYRingAt data point.row

/-- Binding every active output value to the sole source family makes the raw
message terminal definitionally equal to the semantic terminal. -/
theorem terminalFromMessage_eq_qAtPoint_of_yRing_eq
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (profile : SupportedProfile shape domain)
    (data : Data shape)
    (coins : Coins shape domain)
    (point : Point shape domain)
    (message : OutputMessage shape)
    (bound : message.yRing = sourceYRingAt data point.row) :
    terminalFromMessage profile (PublicInput.ofSources data) coins point message =
      qAtPoint profile data coins point := by
  unfold terminalFromMessage qAtPoint
  rw [bound]

/-- The existing CE/extraction authority predicate is sufficient for FE
terminal binding; the independent `yZcol` branch is not required here. -/
theorem terminalFromMessage_eq_qAtPoint_of_yRingBoundToSources
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (profile : SupportedProfile shape domain)
    (data : Data shape)
    (coins : Coins shape domain)
    (point : Point shape domain)
    (points : VerifierPoints shape domain)
    (message : OutputMessage shape)
    (sameRow : points.rPrime = point.row)
    (bound : YRingBoundToSources data points message) :
    terminalFromMessage profile (PublicInput.ofSources data) coins point message =
      qAtPoint profile data coins point := by
  apply terminalFromMessage_eq_qAtPoint_of_yRing_eq
  funext source matrix lane
  calc
    message.yRing source matrix lane =
        canonicalYRing data points source matrix lane :=
      bound source matrix lane
    _ = sourceYRingAt data point.row source matrix lane := by
      simp [canonicalYRing, sourceYRingAt, sameRow]

end Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Fe
