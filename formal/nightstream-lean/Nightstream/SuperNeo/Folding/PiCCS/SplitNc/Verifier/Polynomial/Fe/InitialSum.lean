import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Fe.SourceRefinement
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.BooleanReproduction
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.FiniteSumAlgebra
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConstraintPolynomialLift.Evaluation
import Nightstream.SuperNeo.SumCheck.HypercubeTruth

/-!
Model-level initial-sum decomposition for the Split-NC FE polynomial.

Protocol: SuperNeo `Pi_CCS`.
Phase: FE Boolean sum before the first SumCheck message.
Constraint family: fresh CCS and running carried-evaluation residuals only;
this file emits no rows.

Owns: the explicit row-by-lane Boolean sum of the independent FE polynomial;
the total exact-round semantic evaluator and its structural equality to the
generic recursive completion sum; the fresh/carried branch decomposition;
independent fresh residual, carried claim, carried computed, and carried
residual mixes; the closed fresh CCS bridge; the row/lane reproduction leaves
used by the child carried bridge; and the signed mixed residual which an
honest FE statement makes zero.

Does not own: transcript challenges; executable SumCheck messages or round
degree proofs; finite matrix/running closure of the carried branch (owned by
`InitialSum.CarriedBridge`); NC; Rust; R1CS; row emission; row removal; or
constraint counts.

Emits constraints: no.

Authority boundary: `hypercubeSum` evaluates `qAtPoint`, whose matrices and
assignments come from `Sources.Data`. `freshResidualMix` and
`carriedResidualMix` are defined independently from CCS and carried-evaluation
residual semantics; neither is defined by FE polynomial acceptance.
`sumcheckPolynomial` maps the public fail-closed `Option` evaluator to a total
semantic function only after the exact round count makes its zero default
unreachable.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_ccs.fe.initial.boolean_sum` | sum `q(row,lane)` over the exact typed product cube | computed | `hypercubeSum` |
| `nifs.pi_ccs.fe.initial.sumcheck.total` | totalize only the unreachable wrong-arity branch | semantic adapter | `sumcheckPolynomial` |
| `nifs.pi_ccs.fe.initial.sumcheck.bridge` | recursive `sumCompletions` equals the typed row-by-lane cube | derived | `sumcheckHypercubeSum_eq_hypercubeSum` |
| `nifs.pi_ccs.fe.initial.branches` | separate the fresh and carried summands without changing either cube | derived | `hypercubeSum_eq_fresh_add_carried` |
| `nifs.pi_ccs.fe.initial.fresh.residual` | gamma-mix equality-weighted embedded CCS residuals | independent semantic residual | `freshResidualMix` |
| `nifs.pi_ccs.fe.initial.fresh.bridge` | fresh Boolean restriction equals the CCS residual mix | derived | `freshHypercubeContribution_eq_freshResidualMix` |
| `nifs.pi_ccs.fe.initial.carried.claimed` | gamma-mix public prior coefficient claims | public claim | `carriedClaimedMix` |
| `nifs.pi_ccs.fe.initial.carried.computed` | gamma-mix source-derived prior evaluations | computed | `carriedComputedMix` |
| `nifs.pi_ccs.fe.initial.carried.residual` | claimed-minus-computed mix | independent semantic residual | `carriedResidualMix` |
| `nifs.pi_ccs.fe.initial.signed` | `-fresh + gamma^N * carried` | derived | `mixedResidual` |
| `nifs.pi_ccs.fe.initial.carried.selector` | reproduce padded lane values and running source rows | derived | `paddedLaneEvaluation_selectorSum`, `paddedRunningSource_selectorSum` |
| assurance | independent FE truth makes the signed mix zero | model-level | `mixedResidual_eq_zero_of_truth` |
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Fe.InitialSum

set_option maxRecDepth 10000
set_option maxHeartbeats 1000000

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.SumCheck
open Nightstream.SuperNeo.Folding.PiCCS.OutputClaims
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Fe

private abbrev ops := ConcreteCarrier.extensionOps
private abbrev laws := ConcreteCarrier.extensionLaws

/-- Fresh branch of `qAtPoint`, kept separate solely for the exact sum
decomposition below. -/
def freshAtPoint
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (data : Data shape)
    (coins : Coins shape domain)
    (point : Point shape domain) : K :=
  K.mul
    (K.mul
      (SumCheckTruthPath.pointEquality ops point.lane coins.betaA)
      (SumCheckTruthPath.pointEquality ops point.row coins.betaR))
    (freshTermFromYRing (PublicInput.ofSources data) coins.gamma
      (sourceYRingAt data point.row))

/-- Carried branch of `qAtPoint`, including the production `gamma^N` shift. -/
def carriedAtPoint
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (profile : SupportedProfile shape domain)
    (data : Data shape)
    (coins : Coins shape domain)
    (point : Point shape domain) : K :=
  K.mul
    (K.mul
      (SumCheckTruthPath.pointEquality ops point.lane coins.alpha)
      (SumCheckTruthPath.pointEquality ops point.row
        (PublicInput.ofSources data).priorPoint))
    (SignedJointIdentity.gammaTerm ops coins.gamma shape.sourceCount
      (carriedTermFromYRing profile.laneCovers coins.gamma point.lane
        (sourceYRingAt data point.row)))

/-- Exact typed product-cube sum of the independent FE polynomial. The row
cube is the outer family and the lane cube is the inner family; both use the
sole canonical `BooleanVertex.all` enumeration. -/
def hypercubeSum
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (profile : SupportedProfile shape domain)
    (data : Data shape)
    (coins : Coins shape domain) : K :=
  FiniteSumAlgebra.sumMap ops (BooleanVertex.all shape.rowVariables) fun row =>
    FiniteSumAlgebra.sumMap ops (BooleanVertex.all domain.laneVariables) fun lane =>
      qAtPoint profile data coins {
        row := row.toCubePoint ops
        lane := lane.toCubePoint ops
      }

/-- Total semantic evaluator used only to instantiate generic SumCheck truth.
The public `polynomial` remains fail-closed as an `Option`; zero here is the
semantic value of the unreachable wrong-arity branch after the verifier has
checked the exact round count. -/
def sumcheckPolynomial
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (profile : SupportedProfile shape domain)
    (data : Data shape)
    (coins : Coins shape domain)
    (coordinates : List K) : K :=
  (polynomial profile data coins coordinates).getD K.zero

/-- On an exact typed point, the total SumCheck evaluator is the same
independent FE polynomial; the default branch is unreachable. -/
theorem sumcheckPolynomial_coordinates_eq_qAtPoint
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (profile : SupportedProfile shape domain)
    (data : Data shape)
    (coins : Coins shape domain)
    (point : Point shape domain) :
    sumcheckPolynomial profile data coins point.coordinates =
      qAtPoint profile data coins point := by
  unfold sumcheckPolynomial
  rw [polynomial_coordinates_eq_qAtPoint]
  rfl

/-- Generic recursive SumCheck initial sum for the exact FE round count. -/
def sumcheckHypercubeSum
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (profile : SupportedProfile shape domain)
    (data : Data shape)
    (coins : Coins shape domain) : K :=
  SumCheck.Finite.HypercubeTruth.sumCompletions ops.toOps
    (sumcheckPolynomial profile data coins) []
    (shape.rowVariables + domain.laneVariables)

/-- The generic recursive completion sum and the typed row-by-lane product
sum enumerate exactly the same FE points. The proof splits the round domain
at the row/lane boundary before invoking the sole canonical vertex
enumeration theorem on each side. -/
theorem sumcheckHypercubeSum_eq_hypercubeSum
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (profile : SupportedProfile shape domain)
    (data : Data shape)
    (coins : Coins shape domain) :
    sumcheckHypercubeSum profile data coins =
      hypercubeSum profile data coins := by
  unfold sumcheckHypercubeSum
  rw [SumCheck.Finite.HypercubeTruth.sumCompletions_add]
  rw [SumCheckTruthPath.sumCompletions_eq_vertexSum ops laws]
  unfold hypercubeSum FiniteSumAlgebra.sumMap
  simp only [List.nil_append]
  congr 1
  apply List.map_congr_left
  intro row _
  rw [SumCheckTruthPath.sumCompletions_eq_vertexSum ops laws]
  congr 1
  apply List.map_congr_left
  intro lane _
  exact sumcheckPolynomial_coordinates_eq_qAtPoint profile data coins {
    row := row.toCubePoint ops
    lane := lane.toCubePoint ops
  }

/-- The same product cube restricted to the fresh branch. -/
def freshHypercubeContribution
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (data : Data shape)
    (coins : Coins shape domain) : K :=
  FiniteSumAlgebra.sumMap ops (BooleanVertex.all shape.rowVariables) fun row =>
    FiniteSumAlgebra.sumMap ops (BooleanVertex.all domain.laneVariables) fun lane =>
      freshAtPoint data coins {
        row := row.toCubePoint ops
        lane := lane.toCubePoint ops
      }

/-- The same product cube restricted to the carried branch. -/
def carriedHypercubeContribution
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (profile : SupportedProfile shape domain)
    (data : Data shape)
    (coins : Coins shape domain) : K :=
  FiniteSumAlgebra.sumMap ops (BooleanVertex.all shape.rowVariables) fun row =>
    FiniteSumAlgebra.sumMap ops (BooleanVertex.all domain.laneVariables) fun lane =>
      carriedAtPoint profile data coins {
        row := row.toCubePoint ops
        lane := lane.toCubePoint ops
      }

/-- Pointwise, the FE polynomial is exactly its named fresh and carried
branches. -/
theorem qAtPoint_eq_fresh_add_carried
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (profile : SupportedProfile shape domain)
    (data : Data shape)
    (coins : Coins shape domain)
    (point : Point shape domain) :
    qAtPoint profile data coins point =
      K.add (freshAtPoint data coins point)
        (carriedAtPoint profile data coins point) := by
  rfl

/-- Finite-sum linearity separates the two protocol branches without
dropping, duplicating, or reordering a Boolean point. -/
theorem hypercubeSum_eq_fresh_add_carried
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (profile : SupportedProfile shape domain)
    (data : Data shape)
    (coins : Coins shape domain) :
    hypercubeSum profile data coins =
      K.add (freshHypercubeContribution data coins)
        (carriedHypercubeContribution profile data coins) := by
  unfold hypercubeSum freshHypercubeContribution
    carriedHypercubeContribution
  calc
    FiniteSumAlgebra.sumMap ops (BooleanVertex.all shape.rowVariables)
        (fun row =>
          FiniteSumAlgebra.sumMap ops (BooleanVertex.all domain.laneVariables)
            (fun lane => qAtPoint profile data coins {
              row := row.toCubePoint ops
              lane := lane.toCubePoint ops })) =
      FiniteSumAlgebra.sumMap ops (BooleanVertex.all shape.rowVariables)
        (fun row => K.add
          (FiniteSumAlgebra.sumMap ops (BooleanVertex.all domain.laneVariables)
            (fun lane => freshAtPoint data coins {
              row := row.toCubePoint ops
              lane := lane.toCubePoint ops }))
          (FiniteSumAlgebra.sumMap ops (BooleanVertex.all domain.laneVariables)
            (fun lane => carriedAtPoint profile data coins {
              row := row.toCubePoint ops
              lane := lane.toCubePoint ops }))) := by
        apply FiniteSumAlgebra.sumMap_congr
        intro row _
        calc
          _ = FiniteSumAlgebra.sumMap ops
              (BooleanVertex.all domain.laneVariables)
              (fun lane => K.add
                (freshAtPoint data coins {
                  row := row.toCubePoint ops
                  lane := lane.toCubePoint ops })
                (carriedAtPoint profile data coins {
                  row := row.toCubePoint ops
                  lane := lane.toCubePoint ops })) := by
                apply FiniteSumAlgebra.sumMap_congr
                intro lane _
                exact qAtPoint_eq_fresh_add_carried profile data coins _
          _ = _ := FiniteSumAlgebra.sumMap_add ops laws _ _ _
    _ = _ := FiniteSumAlgebra.sumMap_add ops laws _ _ _

/-- Independent fresh semantic residual compressed with the verifier's gamma
and row-equality challenges. -/
def freshResidualMix
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (data : Data shape)
    (coins : Coins shape domain) : K :=
  FiniteSumAlgebra.sumMap ops (canonicalFinIndices shape.freshCount) fun fresh =>
    SignedJointIdentity.gammaTerm ops coins.gamma fresh.val <|
      FiniteSumAlgebra.sumMap ops (BooleanVertex.all shape.rowVariables) fun row =>
        K.mul (row.equalityWeight ops coins.betaR) <|
          K.embed <| CCSResidualTable.residualAt ConcreteCarrier.baseOps
            data.freshBatch.system (data.freshBatch.assignments fresh) row

/-- At one Boolean row, the fresh constant Phi81 lane is exactly the lifted
original CCS matrix image. -/
theorem sourceYRingAt_fresh_constant_toCubePoint
    {shape : SemanticShape}
    (data : Data shape)
    (fresh : Fin shape.freshCount)
    (matrix : Fin shape.matrixCount)
    (row : BooleanVertex shape.rowVariables) :
    sourceYRingAt data (row.toCubePoint ops) (Data.freshIndex fresh) matrix
        Phi81CoefficientKernel.constant =
      K.embed (CCSResidualTable.matrixImagesAt ConcreteCarrier.baseOps
        data.freshBatch.system (data.freshBatch.assignments fresh) row matrix) := by
  rw [SourceRefinement.sourceYRingAt_fresh_constant_eq_completedMatrixImage]
  rw [SumCheckTruthPath.evaluate_toCubePoint_eq_valueAt ops laws]
  simp [SourceRefinement.completedMatrixImageTable]

/-- The fresh FE term restricted to one Boolean row is the gamma compression
of the independently defined lifted CCS residual at that row. -/
theorem freshTermFromYRing_toCubePoint_eq_residuals
    {shape : SemanticShape}
    (data : Data shape)
    (gamma : K)
    (row : BooleanVertex shape.rowVariables) :
    freshTermFromYRing (PublicInput.ofSources data) gamma
        (sourceYRingAt data (row.toCubePoint ops)) =
      FiniteSumAlgebra.sumMap ops
        (canonicalFinIndices shape.freshCount) fun fresh =>
          SignedJointIdentity.gammaTerm ops gamma fresh.val <|
            K.embed <| CCSResidualTable.residualAt ConcreteCarrier.baseOps
              data.freshBatch.system (data.freshBatch.assignments fresh) row := by
  unfold freshTermFromYRing
  apply FiniteSumAlgebra.sumMap_congr
  intro fresh _
  unfold SignedJointIdentity.gammaTerm
  apply congrArg
  change
    CCSResidualTable.evaluatePolynomial ops
        (ConstraintPolynomialLift.liftConstraintPolynomial K.embed
          data.constraintPolynomial)
        (fun matrix => sourceYRingAt data (row.toCubePoint ops)
          (Data.freshIndex fresh) matrix Phi81CoefficientKernel.constant) =
      K.embed (CCSResidualTable.residualAt ConcreteCarrier.baseOps
        data.freshBatch.system (data.freshBatch.assignments fresh) row)
  rw [show
    (fun matrix => sourceYRingAt data (row.toCubePoint ops)
      (Data.freshIndex fresh) matrix Phi81CoefficientKernel.constant) =
      fun matrix => K.embed (CCSResidualTable.matrixImagesAt
        ConcreteCarrier.baseOps data.freshBatch.system
        (data.freshBatch.assignments fresh) row matrix) by
          funext matrix
          exact sourceYRingAt_fresh_constant_toCubePoint
            data fresh matrix row]
  exact ConstraintPolynomialLift.Evaluation.evaluatePolynomial_lift
    ConcreteCarrier.baseOps ops K.embed
    ConcreteCarrier.constraintEvaluationLaws
    data.constraintPolynomial
    (CCSResidualTable.matrixImagesAt ConcreteCarrier.baseOps
      data.freshBatch.system (data.freshBatch.assignments fresh) row)

private theorem freshLaneSum
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (data : Data shape)
    (coins : Coins shape domain)
    (row : BooleanVertex shape.rowVariables) :
    FiniteSumAlgebra.sumMap ops (BooleanVertex.all domain.laneVariables)
        (fun lane => freshAtPoint data coins {
          row := row.toCubePoint ops
          lane := lane.toCubePoint ops }) =
      K.mul (row.equalityWeight ops coins.betaR)
        (freshTermFromYRing (PublicInput.ofSources data) coins.gamma
          (sourceYRingAt data (row.toCubePoint ops))) := by
  let rowWeight := row.equalityWeight ops coins.betaR
  let freshTerm := freshTermFromYRing (PublicInput.ofSources data) coins.gamma
    (sourceYRingAt data (row.toCubePoint ops))
  calc
    FiniteSumAlgebra.sumMap ops (BooleanVertex.all domain.laneVariables)
        (fun lane => freshAtPoint data coins {
          row := row.toCubePoint ops
          lane := lane.toCubePoint ops }) =
      FiniteSumAlgebra.sumMap ops (BooleanVertex.all domain.laneVariables)
        (fun lane => K.mul
          (K.mul (lane.equalityWeight ops coins.betaA) rowWeight)
          freshTerm) := by
            apply FiniteSumAlgebra.sumMap_congr
            intro lane _
            unfold freshAtPoint rowWeight freshTerm
            rw [SumCheckTruthPath.pointEquality_toCubePoint_eq_equalityWeight
                ops laws,
              SumCheckTruthPath.pointEquality_toCubePoint_eq_equalityWeight
                ops laws]
    _ = FiniteSumAlgebra.sumMap ops
        (BooleanVertex.all domain.laneVariables)
        (fun lane => K.mul (K.mul rowWeight freshTerm)
          (lane.equalityWeight ops coins.betaA)) := by
            apply FiniteSumAlgebra.sumMap_congr
            intro lane _
            calc
              K.mul (K.mul (lane.equalityWeight ops coins.betaA) rowWeight)
                  freshTerm =
                K.mul (lane.equalityWeight ops coins.betaA)
                  (K.mul rowWeight freshTerm) := laws.mul_assoc _ _ _
              _ = K.mul (K.mul rowWeight freshTerm)
                  (lane.equalityWeight ops coins.betaA) :=
                    laws.mul_comm _ _
    _ = K.mul (K.mul rowWeight freshTerm)
        (FiniteSumAlgebra.sumMap ops
          (BooleanVertex.all domain.laneVariables)
          (fun lane => lane.equalityWeight ops coins.betaA)) :=
            FiniteSumAlgebra.sumMap_mul_left ops laws _ _ _
    _ = K.mul (K.mul rowWeight freshTerm) ops.one := by
          rw [BooleanReproduction.equalityWeight_sum_eq_one ops laws]
    _ = K.mul rowWeight freshTerm := laws.mul_one _

/-- The complete fresh Boolean restriction is exactly the independent CCS
residual mix. Lane partition of unity removes `betaA`; finite Fubini then
places each gamma factor outside its row-weighted residual family. -/
theorem freshHypercubeContribution_eq_freshResidualMix
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (data : Data shape)
    (coins : Coins shape domain) :
    freshHypercubeContribution data coins = freshResidualMix data coins := by
  unfold freshHypercubeContribution freshResidualMix
  calc
    FiniteSumAlgebra.sumMap ops (BooleanVertex.all shape.rowVariables)
        (fun row => FiniteSumAlgebra.sumMap ops
          (BooleanVertex.all domain.laneVariables)
          (fun lane => freshAtPoint data coins {
            row := row.toCubePoint ops
            lane := lane.toCubePoint ops })) =
      FiniteSumAlgebra.sumMap ops (BooleanVertex.all shape.rowVariables)
        (fun row => K.mul (row.equalityWeight ops coins.betaR)
          (freshTermFromYRing (PublicInput.ofSources data) coins.gamma
            (sourceYRingAt data (row.toCubePoint ops)))) := by
              apply FiniteSumAlgebra.sumMap_congr
              intro row _
              exact freshLaneSum data coins row
    _ = FiniteSumAlgebra.sumMap ops (BooleanVertex.all shape.rowVariables)
        (fun row => K.mul (row.equalityWeight ops coins.betaR)
          (FiniteSumAlgebra.sumMap ops
            (canonicalFinIndices shape.freshCount) fun fresh =>
              SignedJointIdentity.gammaTerm ops coins.gamma fresh.val <|
                K.embed <| CCSResidualTable.residualAt
                  ConcreteCarrier.baseOps data.freshBatch.system
                  (data.freshBatch.assignments fresh) row)) := by
              apply FiniteSumAlgebra.sumMap_congr
              intro row _
              rw [freshTermFromYRing_toCubePoint_eq_residuals]
    _ = FiniteSumAlgebra.sumMap ops (BooleanVertex.all shape.rowVariables)
        (fun row => FiniteSumAlgebra.sumMap ops
          (canonicalFinIndices shape.freshCount) fun fresh =>
            K.mul (row.equalityWeight ops coins.betaR)
              (SignedJointIdentity.gammaTerm ops coins.gamma fresh.val <|
                K.embed <| CCSResidualTable.residualAt
                  ConcreteCarrier.baseOps data.freshBatch.system
                  (data.freshBatch.assignments fresh) row)) := by
              apply FiniteSumAlgebra.sumMap_congr
              intro row _
              exact (FiniteSumAlgebra.sumMap_mul_left ops laws
                (row.equalityWeight ops coins.betaR)
                (canonicalFinIndices shape.freshCount) _).symm
    _ = FiniteSumAlgebra.sumMap ops
        (canonicalFinIndices shape.freshCount) (fun fresh =>
          FiniteSumAlgebra.sumMap ops (BooleanVertex.all shape.rowVariables)
            (fun row => K.mul (row.equalityWeight ops coins.betaR)
              (SignedJointIdentity.gammaTerm ops coins.gamma fresh.val <|
                K.embed <| CCSResidualTable.residualAt
                  ConcreteCarrier.baseOps data.freshBatch.system
                  (data.freshBatch.assignments fresh) row))) :=
            FiniteSumAlgebra.sumMap_swap ops laws _ _ _
    _ = FiniteSumAlgebra.sumMap ops
        (canonicalFinIndices shape.freshCount) (fun fresh =>
          SignedJointIdentity.gammaTerm ops coins.gamma fresh.val <|
            FiniteSumAlgebra.sumMap ops
              (BooleanVertex.all shape.rowVariables) fun row =>
                K.mul (row.equalityWeight ops coins.betaR) <|
                  K.embed <| CCSResidualTable.residualAt
                    ConcreteCarrier.baseOps data.freshBatch.system
                    (data.freshBatch.assignments fresh) row) := by
              apply FiniteSumAlgebra.sumMap_congr
              intro fresh _
              unfold SignedJointIdentity.gammaTerm
              rw [← FiniteSumAlgebra.sumMap_mul_left ops laws]
              apply FiniteSumAlgebra.sumMap_congr
              intro row _
              calc
                K.mul (row.equalityWeight ops coins.betaR)
                    (K.mul
                      (TargetPolynomial.power ops.toOps coins.gamma fresh.val)
                      (K.embed (CCSResidualTable.residualAt
                        ConcreteCarrier.baseOps data.freshBatch.system
                        (data.freshBatch.assignments fresh) row))) =
                  K.mul
                    (K.mul (row.equalityWeight ops coins.betaR)
                      (TargetPolynomial.power ops.toOps coins.gamma fresh.val))
                    (K.embed (CCSResidualTable.residualAt
                      ConcreteCarrier.baseOps data.freshBatch.system
                      (data.freshBatch.assignments fresh) row)) :=
                        (laws.mul_assoc _ _ _).symm
                _ = K.mul
                    (K.mul
                      (TargetPolynomial.power ops.toOps coins.gamma fresh.val)
                      (row.equalityWeight ops coins.betaR))
                    (K.embed (CCSResidualTable.residualAt
                      ConcreteCarrier.baseOps data.freshBatch.system
                      (data.freshBatch.assignments fresh) row)) :=
                        congrArg
                          (fun factor => K.mul factor
                            (K.embed (CCSResidualTable.residualAt
                              ConcreteCarrier.baseOps data.freshBatch.system
                              (data.freshBatch.assignments fresh) row)))
                          (laws.mul_comm
                            (row.equalityWeight ops coins.betaR)
                            (TargetPolynomial.power ops.toOps coins.gamma
                              fresh.val))
                _ = K.mul
                    (TargetPolynomial.power ops.toOps coins.gamma fresh.val)
                    (K.mul (row.equalityWeight ops coins.betaR)
                      (K.embed (CCSResidualTable.residualAt
                        ConcreteCarrier.baseOps data.freshBatch.system
                        (data.freshBatch.assignments fresh) row))) :=
                          laws.mul_assoc _ _ _
    _ = _ := rfl

private theorem paddedLaneEvaluation_commuted
    {domain : FlatNcDomain}
    (covers : LaneCovers domain)
    (values : Fin ringDegree -> K)
    (point : CubePoint K domain.laneVariables) :
    paddedLaneEvaluation covers values point =
      FiniteSumAlgebra.sumMap ops (canonicalFinIndices ringDegree)
        (fun lane => K.mul (values lane)
          (NumericBooleanDomain.tensorWeight ops (liveLane covers lane)
            point)) := by
  unfold paddedLaneEvaluation
  apply FiniteSumAlgebra.sumMap_congr
  intro lane _
  exact laws.mul_comm _ _

/-- Equality weighting over every Boolean lane reproduces the padded 54-lane
MLE at the verifier's target point. No padded lane is introduced as an input. -/
theorem paddedLaneEvaluation_selectorSum
    {domain : FlatNcDomain}
    (covers : LaneCovers domain)
    (values : Fin ringDegree -> K)
    (point : CubePoint K domain.laneVariables) :
    FiniteSumAlgebra.sumMap ops (BooleanVertex.all domain.laneVariables)
        (fun sampled => K.mul (sampled.equalityWeight ops point)
          (paddedLaneEvaluation covers values (sampled.toCubePoint ops))) =
      paddedLaneEvaluation covers values point := by
  change BooleanReproduction.equalityWeighted ops point
      (fun sampled => paddedLaneEvaluation covers values
        (sampled.toCubePoint ops)) = _
  calc
    BooleanReproduction.equalityWeighted ops point (fun sampled =>
        paddedLaneEvaluation covers values (sampled.toCubePoint ops)) =
      BooleanReproduction.equalityWeighted ops point
        (fun sampled =>
          FiniteSumAlgebra.sumMap ops (canonicalFinIndices ringDegree)
            (fun lane => K.mul (values lane)
              (NumericBooleanDomain.tensorWeight ops (liveLane covers lane)
                (sampled.toCubePoint ops)))) := by
          apply congrArg (BooleanReproduction.equalityWeighted ops point)
          funext sampled
          exact paddedLaneEvaluation_commuted covers values _
    _ = FiniteSumAlgebra.sumMap ops (canonicalFinIndices ringDegree)
        (fun lane => K.mul (values lane)
          (BooleanReproduction.equalityWeighted ops point (fun sampled =>
            NumericBooleanDomain.tensorWeight ops (liveLane covers lane)
              (sampled.toCubePoint ops)))) :=
      BooleanReproduction.equalityWeighted_sumMap ops laws
        (canonicalFinIndices ringDegree) values
        (fun lane sampled => NumericBooleanDomain.tensorWeight ops
          (liveLane covers lane) (sampled.toCubePoint ops)) point
    _ = FiniteSumAlgebra.sumMap ops (canonicalFinIndices ringDegree)
        (fun lane => K.mul (values lane)
          (NumericBooleanDomain.tensorWeight ops (liveLane covers lane)
            point)) := by
            apply FiniteSumAlgebra.sumMap_congr
            intro lane _
            apply congrArg (K.mul (values lane))
            exact BooleanReproduction.equalityWeighted_tensorWeight_eq_tensorWeight
              ops laws (liveLane covers lane) point
    _ = _ := (paddedLaneEvaluation_commuted covers values point).symm

/-- Row equality weighting of one source-derived running coefficient image
reproduces its independently computed prior-point coefficient. -/
theorem sourceYRingAt_running_selectorSum
    {shape : SemanticShape}
    (data : Data shape)
    (running : Fin shape.runningCount)
    (matrix : Fin shape.matrixCount)
    (lane : Fin ringDegree) :
    FiniteSumAlgebra.sumMap ops (BooleanVertex.all shape.rowVariables)
        (fun row => K.mul (row.equalityWeight ops data.priorPoint)
          (sourceYRingAt data (row.toCubePoint ops)
            (Data.runningIndex running) matrix lane)) =
      CarriedEvaluationResidual.computedCoefficient
        ConcreteCarrier.baseOps ops K.embed data.carriedData {
          running := running
          matrix := matrix
          coefficient := lane
        } := by
  let table := yRingTableForAssignment data
    (data.assignment (Data.runningIndex running)) matrix lane
  calc
    FiniteSumAlgebra.sumMap ops (BooleanVertex.all shape.rowVariables)
        (fun row => K.mul (row.equalityWeight ops data.priorPoint)
          (sourceYRingAt data (row.toCubePoint ops)
            (Data.runningIndex running) matrix lane)) =
      FiniteSumAlgebra.sumMap ops (BooleanVertex.all shape.rowVariables)
        (fun row => K.mul (row.equalityWeight ops data.priorPoint)
          (table.valueAt row)) := by
            apply FiniteSumAlgebra.sumMap_congr
            intro row _
            apply congrArg (K.mul (row.equalityWeight ops data.priorPoint))
            change table.evaluate ops (row.toCubePoint ops) = table.valueAt row
            exact SumCheckTruthPath.evaluate_toCubePoint_eq_valueAt
              ops laws table row
    _ = table.equalityWeightedSum ops data.priorPoint := rfl
    _ = table.evaluate ops data.priorPoint :=
      (BooleanTable.evaluate_eq_equalityWeightedSum
        ops laws table data.priorPoint).symm
    _ = sourceYRingAt data data.priorPoint (Data.runningIndex running)
        matrix lane := rfl
    _ = _ := SourceRefinement.sourceYRingAt_running_eq_computedCoefficient
      data running matrix lane

/-- Row equality weighting commutes with padded lane materialization and
turns every running source image into its computed prior coefficient. -/
theorem paddedRunningSource_selectorSum
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (covers : LaneCovers domain)
    (data : Data shape)
    (running : Fin shape.runningCount)
    (matrix : Fin shape.matrixCount)
    (lanePoint : CubePoint K domain.laneVariables) :
    FiniteSumAlgebra.sumMap ops (BooleanVertex.all shape.rowVariables)
        (fun row => K.mul (row.equalityWeight ops data.priorPoint)
          (paddedLaneEvaluation covers
            (sourceYRingAt data (row.toCubePoint ops)
              (Data.runningIndex running) matrix) lanePoint)) =
      paddedLaneEvaluation covers
        (fun lane => CarriedEvaluationResidual.computedCoefficient
          ConcreteCarrier.baseOps ops K.embed data.carriedData {
            running := running
            matrix := matrix
            coefficient := lane
          }) lanePoint := by
  unfold paddedLaneEvaluation
  change BooleanReproduction.equalityWeighted ops data.priorPoint
      (fun row => FiniteSumAlgebra.sumMap ops
        (canonicalFinIndices ringDegree) (fun lane => K.mul
          (NumericBooleanDomain.tensorWeight ops (liveLane covers lane)
            lanePoint)
          (sourceYRingAt data (row.toCubePoint ops)
            (Data.runningIndex running) matrix lane))) = _
  calc
    BooleanReproduction.equalityWeighted ops data.priorPoint
        (fun row => FiniteSumAlgebra.sumMap ops
          (canonicalFinIndices ringDegree) (fun lane => K.mul
            (NumericBooleanDomain.tensorWeight ops (liveLane covers lane)
              lanePoint)
            (sourceYRingAt data (row.toCubePoint ops)
              (Data.runningIndex running) matrix lane))) =
      FiniteSumAlgebra.sumMap ops (canonicalFinIndices ringDegree)
        (fun lane => K.mul
          (NumericBooleanDomain.tensorWeight ops (liveLane covers lane)
            lanePoint)
          (BooleanReproduction.equalityWeighted ops data.priorPoint
            (fun row => sourceYRingAt data (row.toCubePoint ops)
              (Data.runningIndex running) matrix lane))) :=
        BooleanReproduction.equalityWeighted_sumMap ops laws
          (canonicalFinIndices ringDegree)
          (fun lane => NumericBooleanDomain.tensorWeight ops
            (liveLane covers lane) lanePoint)
          (fun lane row => sourceYRingAt data (row.toCubePoint ops)
            (Data.runningIndex running) matrix lane)
          data.priorPoint
    _ = FiniteSumAlgebra.sumMap ops (canonicalFinIndices ringDegree)
        (fun lane => K.mul
          (NumericBooleanDomain.tensorWeight ops (liveLane covers lane)
            lanePoint)
          (CarriedEvaluationResidual.computedCoefficient
            ConcreteCarrier.baseOps ops K.embed data.carriedData {
              running := running
              matrix := matrix
              coefficient := lane })) := by
            apply FiniteSumAlgebra.sumMap_congr
            intro lane _
            apply congrArg (K.mul
              (NumericBooleanDomain.tensorWeight ops
                (liveLane covers lane) lanePoint))
            unfold BooleanReproduction.equalityWeighted
            exact sourceYRingAt_running_selectorSum data running matrix lane
    _ = _ := rfl

/-- Unshifted public carried-claim mix. -/
def carriedClaimedMix
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (profile : SupportedProfile shape domain)
    (data : Data shape)
    (coins : Coins shape domain) : K :=
  FiniteSumAlgebra.sumMap ops (canonicalFinIndices shape.matrixCount) fun matrix =>
    FiniteSumAlgebra.sumMap ops (canonicalFinIndices shape.runningCount) fun running =>
      SignedJointIdentity.gammaTerm ops coins.gamma
        (carriedGammaExponent shape running matrix) <|
        paddedLaneEvaluation profile.laneCovers
          (fun lane => data.claimedCoefficient {
            running := running
            matrix := matrix
            coefficient := lane
          }) coins.alpha

/-- Unshifted source-derived carried-evaluation mix. -/
def carriedComputedMix
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (profile : SupportedProfile shape domain)
    (data : Data shape)
    (coins : Coins shape domain) : K :=
  FiniteSumAlgebra.sumMap ops (canonicalFinIndices shape.matrixCount) fun matrix =>
    FiniteSumAlgebra.sumMap ops (canonicalFinIndices shape.runningCount) fun running =>
      SignedJointIdentity.gammaTerm ops coins.gamma
        (carriedGammaExponent shape running matrix) <|
        paddedLaneEvaluation profile.laneCovers
          (fun lane => CarriedEvaluationResidual.computedCoefficient
            ConcreteCarrier.baseOps ops K.embed data.carriedData {
              running := running
              matrix := matrix
              coefficient := lane
            }) coins.alpha

/-- Unshifted carried claimed-minus-computed residual mix. -/
def carriedResidualMix
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (profile : SupportedProfile shape domain)
    (data : Data shape)
    (coins : Coins shape domain) : K :=
  FiniteSumAlgebra.sumMap ops (canonicalFinIndices shape.matrixCount) fun matrix =>
    FiniteSumAlgebra.sumMap ops (canonicalFinIndices shape.runningCount) fun running =>
      SignedJointIdentity.gammaTerm ops coins.gamma
        (carriedGammaExponent shape running matrix) <|
        paddedLaneEvaluation profile.laneCovers
          (fun lane => CarriedEvaluationResidual.residual
            ConcreteCarrier.baseOps ops K.embed data.carriedData {
              running := running
              matrix := matrix
              coefficient := lane
            }) coins.alpha

/-- Signed FE residual forced by `initial - sum Q`: fresh CCS is negative and
the carried claimed-minus-computed family is positive after `gamma^N`. -/
def mixedResidual
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (profile : SupportedProfile shape domain)
    (data : Data shape)
    (coins : Coins shape domain) : K :=
  K.add (ops.neg (freshResidualMix data coins)) <|
    SignedJointIdentity.gammaTerm ops coins.gamma shape.sourceCount
      (carriedResidualMix profile data coins)

/-- The verifier initial claim is precisely the shifted public carried-claim
mix. -/
theorem initial_eq_shifted_carriedClaimedMix
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (profile : SupportedProfile shape domain)
    (data : Data shape)
    (coins : Coins shape domain) :
    initial profile (PublicInput.ofSources data) coins =
      SignedJointIdentity.gammaTerm ops coins.gamma shape.sourceCount
        (carriedClaimedMix profile data coins) := by
  rfl

private theorem paddedLaneEvaluation_zero
    {domain : FlatNcDomain}
    (covers : LaneCovers domain)
    (point : CubePoint K domain.laneVariables) :
    paddedLaneEvaluation covers (fun _ => K.zero) point = K.zero := by
  unfold paddedLaneEvaluation
  calc
    FiniteSumAlgebra.sumMap ops (canonicalFinIndices ringDegree)
        (fun lane => K.mul
          (NumericBooleanDomain.tensorWeight ops (liveLane covers lane) point)
          K.zero) =
      FiniteSumAlgebra.sumMap ops (canonicalFinIndices ringDegree)
        (fun _ => K.zero) := by
          apply FiniteSumAlgebra.sumMap_congr
          intro lane _
          exact laws.mul_zero _
    _ = K.zero := FiniteSumAlgebra.sumMap_zero ops laws _

/-- Lane materialization is linear in the independent carried residual. -/
theorem paddedLaneEvaluation_residual_eq_sub
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (covers : LaneCovers domain)
    (data : Data shape)
    (running : Fin shape.runningCount)
    (matrix : Fin shape.matrixCount)
    (point : CubePoint K domain.laneVariables) :
    paddedLaneEvaluation covers
        (fun lane => CarriedEvaluationResidual.residual
          ConcreteCarrier.baseOps ops K.embed data.carriedData {
            running := running
            matrix := matrix
            coefficient := lane
          }) point =
      ops.sub
        (paddedLaneEvaluation covers
          (fun lane => data.claimedCoefficient {
            running := running
            matrix := matrix
            coefficient := lane
          }) point)
        (paddedLaneEvaluation covers
          (fun lane => CarriedEvaluationResidual.computedCoefficient
            ConcreteCarrier.baseOps ops K.embed data.carriedData {
              running := running
              matrix := matrix
              coefficient := lane
            }) point) := by
  unfold paddedLaneEvaluation CarriedEvaluationResidual.residual
  calc
    FiniteSumAlgebra.sumMap ops (canonicalFinIndices ringDegree)
        (fun lane => K.mul
          (NumericBooleanDomain.tensorWeight ops (liveLane covers lane) point)
          (ops.sub
            (data.claimedCoefficient {
              running := running
              matrix := matrix
              coefficient := lane })
            (CarriedEvaluationResidual.computedCoefficient
              ConcreteCarrier.baseOps ops K.embed data.carriedData {
                running := running
                matrix := matrix
                coefficient := lane }))) =
      FiniteSumAlgebra.sumMap ops (canonicalFinIndices ringDegree)
        (fun lane => ops.sub
          (K.mul
            (NumericBooleanDomain.tensorWeight ops (liveLane covers lane) point)
            (data.claimedCoefficient {
              running := running
              matrix := matrix
              coefficient := lane }))
          (K.mul
            (NumericBooleanDomain.tensorWeight ops (liveLane covers lane) point)
            (CarriedEvaluationResidual.computedCoefficient
              ConcreteCarrier.baseOps ops K.embed data.carriedData {
                running := running
                matrix := matrix
                coefficient := lane }))) := by
          apply FiniteSumAlgebra.sumMap_congr
          intro lane _
          exact FiniteSumAlgebra.mul_sub ops laws _ _ _
    _ = _ := FiniteSumAlgebra.sumMap_sub ops laws _ _ _

/-- Compressing carried residuals is exactly subtracting the compressed
source-derived evaluations from the compressed public claims. -/
theorem carriedResidualMix_eq_sub
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (profile : SupportedProfile shape domain)
    (data : Data shape)
    (coins : Coins shape domain) :
    carriedResidualMix profile data coins =
      ops.sub (carriedClaimedMix profile data coins)
        (carriedComputedMix profile data coins) := by
  unfold carriedResidualMix carriedClaimedMix carriedComputedMix
  calc
    FiniteSumAlgebra.sumMap ops (canonicalFinIndices shape.matrixCount)
        (fun matrix =>
          FiniteSumAlgebra.sumMap ops (canonicalFinIndices shape.runningCount)
            (fun running => SignedJointIdentity.gammaTerm ops coins.gamma
              (carriedGammaExponent shape running matrix)
              (paddedLaneEvaluation profile.laneCovers
                (fun lane => CarriedEvaluationResidual.residual
                  ConcreteCarrier.baseOps ops K.embed data.carriedData {
                    running := running
                    matrix := matrix
                    coefficient := lane }) coins.alpha))) =
      FiniteSumAlgebra.sumMap ops (canonicalFinIndices shape.matrixCount)
        (fun matrix =>
          FiniteSumAlgebra.sumMap ops (canonicalFinIndices shape.runningCount)
            (fun running => ops.sub
              (SignedJointIdentity.gammaTerm ops coins.gamma
                (carriedGammaExponent shape running matrix)
                (paddedLaneEvaluation profile.laneCovers
                  (fun lane => data.claimedCoefficient {
                    running := running
                    matrix := matrix
                    coefficient := lane }) coins.alpha))
              (SignedJointIdentity.gammaTerm ops coins.gamma
                (carriedGammaExponent shape running matrix)
                (paddedLaneEvaluation profile.laneCovers
                  (fun lane => CarriedEvaluationResidual.computedCoefficient
                    ConcreteCarrier.baseOps ops K.embed data.carriedData {
                      running := running
                      matrix := matrix
                      coefficient := lane }) coins.alpha)))) := by
          apply FiniteSumAlgebra.sumMap_congr
          intro matrix _
          apply FiniteSumAlgebra.sumMap_congr
          intro running _
          unfold SignedJointIdentity.gammaTerm
          rw [paddedLaneEvaluation_residual_eq_sub]
          exact FiniteSumAlgebra.mul_sub ops laws _ _ _
    _ = FiniteSumAlgebra.sumMap ops (canonicalFinIndices shape.matrixCount)
        (fun matrix => ops.sub
          (FiniteSumAlgebra.sumMap ops (canonicalFinIndices shape.runningCount)
            (fun running => SignedJointIdentity.gammaTerm ops coins.gamma
              (carriedGammaExponent shape running matrix)
              (paddedLaneEvaluation profile.laneCovers
                (fun lane => data.claimedCoefficient {
                  running := running
                  matrix := matrix
                  coefficient := lane }) coins.alpha)))
          (FiniteSumAlgebra.sumMap ops (canonicalFinIndices shape.runningCount)
            (fun running => SignedJointIdentity.gammaTerm ops coins.gamma
              (carriedGammaExponent shape running matrix)
              (paddedLaneEvaluation profile.laneCovers
                (fun lane => CarriedEvaluationResidual.computedCoefficient
                  ConcreteCarrier.baseOps ops K.embed data.carriedData {
                    running := running
                    matrix := matrix
                    coefficient := lane }) coins.alpha)))) := by
          apply FiniteSumAlgebra.sumMap_congr
          intro matrix _
          exact FiniteSumAlgebra.sumMap_sub ops laws _ _ _
    _ = _ := FiniteSumAlgebra.sumMap_sub ops laws _ _ _

private theorem shiftedSub_eq_signedResidual
    (fresh claimed computed shift : K) :
    ops.sub (ops.mul shift claimed)
        (ops.add fresh (ops.mul shift computed)) =
      ops.add (ops.neg fresh)
        (ops.mul shift (ops.sub claimed computed)) := by
  rw [FiniteSumAlgebra.mul_sub ops laws]
  unfold InterpolationOps.sub
  rw [laws.neg_add]
  calc
    ops.add (ops.mul shift claimed)
        (ops.add (ops.neg fresh) (ops.neg (ops.mul shift computed))) =
      ops.add
        (ops.add (ops.mul shift claimed) (ops.neg fresh))
        (ops.neg (ops.mul shift computed)) :=
          (laws.add_assoc _ _ _).symm
    _ = ops.add
        (ops.add (ops.neg fresh) (ops.mul shift claimed))
        (ops.neg (ops.mul shift computed)) := by
          rw [laws.add_comm (ops.mul shift claimed) (ops.neg fresh)]
    _ = ops.add (ops.neg fresh)
        (ops.add (ops.mul shift claimed)
          (ops.neg (ops.mul shift computed))) :=
            laws.add_assoc _ _ _

/-- Exact algebraic closure once the remaining carried Boolean-selector
bridge is supplied. The premise is deliberately the named missing model-level
theorem, not an opaque callback: carried Boolean restriction must equal the
shifted source-derived evaluation mix. The fresh bridge is closed above. -/
theorem initial_sub_hypercubeSum_eq_mixedResidual_of_carried_bridge
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (profile : SupportedProfile shape domain)
    (data : Data shape)
    (coins : Coins shape domain)
    (carriedBridge : carriedHypercubeContribution profile data coins =
      SignedJointIdentity.gammaTerm ops coins.gamma shape.sourceCount
        (carriedComputedMix profile data coins)) :
    ops.sub (initial profile (PublicInput.ofSources data) coins)
        (hypercubeSum profile data coins) =
      mixedResidual profile data coins := by
  rw [initial_eq_shifted_carriedClaimedMix,
    hypercubeSum_eq_fresh_add_carried,
    freshHypercubeContribution_eq_freshResidualMix, carriedBridge]
  unfold mixedResidual
  rw [carriedResidualMix_eq_sub]
  unfold SignedJointIdentity.gammaTerm
  change
    ops.sub
        (ops.mul
          (TargetPolynomial.power ops.toOps coins.gamma shape.sourceCount)
          (carriedClaimedMix profile data coins))
        (ops.add (freshResidualMix data coins)
          (ops.mul
            (TargetPolynomial.power ops.toOps coins.gamma shape.sourceCount)
            (carriedComputedMix profile data coins))) =
      ops.add (ops.neg (freshResidualMix data coins))
        (ops.mul
          (TargetPolynomial.power ops.toOps coins.gamma shape.sourceCount)
          (ops.sub (carriedClaimedMix profile data coins)
            (carriedComputedMix profile data coins)))
  exact shiftedSub_eq_signedResidual _ _ _ _

/-- Independent fresh CCS truth makes its complete compressed residual block
zero for every verifier challenge. -/
theorem freshResidualMix_eq_zero_of_freshTruth
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (data : Data shape)
    (coins : Coins shape domain)
    (truth : Semantics.Fe.FreshTruth data) :
    freshResidualMix data coins = K.zero := by
  unfold freshResidualMix
  calc
    FiniteSumAlgebra.sumMap ops (canonicalFinIndices shape.freshCount)
        (fun fresh => SignedJointIdentity.gammaTerm ops coins.gamma fresh.val
          (FiniteSumAlgebra.sumMap ops (BooleanVertex.all shape.rowVariables)
            (fun row => K.mul (row.equalityWeight ops coins.betaR)
              (K.embed (CCSResidualTable.residualAt ConcreteCarrier.baseOps
                data.freshBatch.system (data.freshBatch.assignments fresh)
                row))))) =
      FiniteSumAlgebra.sumMap ops (canonicalFinIndices shape.freshCount)
        (fun _ => K.zero) := by
          apply FiniteSumAlgebra.sumMap_congr
          intro fresh _
          have rowsZero :
              FiniteSumAlgebra.sumMap ops
                  (BooleanVertex.all shape.rowVariables)
                  (fun row => K.mul (row.equalityWeight ops coins.betaR)
                    (K.embed (CCSResidualTable.residualAt
                      ConcreteCarrier.baseOps data.freshBatch.system
                      (data.freshBatch.assignments fresh) row))) = K.zero := by
            calc
              _ = FiniteSumAlgebra.sumMap ops
                    (BooleanVertex.all shape.rowVariables)
                    (fun _ => K.zero) := by
                  apply FiniteSumAlgebra.sumMap_congr
                  intro row _
                  rw [truth fresh row]
                  exact laws.mul_zero _
              _ = K.zero := FiniteSumAlgebra.sumMap_zero ops laws _
          unfold SignedJointIdentity.gammaTerm
          rw [rowsZero]
          exact laws.mul_zero _
    _ = K.zero := FiniteSumAlgebra.sumMap_zero ops laws _

/-- Independent carried-evaluation truth makes its complete compressed
residual block zero for every verifier challenge. -/
theorem carriedResidualMix_eq_zero_of_carriedTruth
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (profile : SupportedProfile shape domain)
    (data : Data shape)
    (coins : Coins shape domain)
    (truth : Semantics.Fe.CarriedTruth data) :
    carriedResidualMix profile data coins = K.zero := by
  unfold carriedResidualMix
  calc
    FiniteSumAlgebra.sumMap ops (canonicalFinIndices shape.matrixCount)
        (fun matrix =>
          FiniteSumAlgebra.sumMap ops (canonicalFinIndices shape.runningCount)
            (fun running => SignedJointIdentity.gammaTerm ops coins.gamma
              (carriedGammaExponent shape running matrix)
              (paddedLaneEvaluation profile.laneCovers
                (fun lane => CarriedEvaluationResidual.residual
                  ConcreteCarrier.baseOps ops K.embed data.carriedData {
                    running := running
                    matrix := matrix
                    coefficient := lane }) coins.alpha))) =
      FiniteSumAlgebra.sumMap ops (canonicalFinIndices shape.matrixCount)
        (fun _ => K.zero) := by
          apply FiniteSumAlgebra.sumMap_congr
          intro matrix _
          calc
            FiniteSumAlgebra.sumMap ops
                (canonicalFinIndices shape.runningCount)
                (fun running => SignedJointIdentity.gammaTerm ops coins.gamma
                  (carriedGammaExponent shape running matrix)
                  (paddedLaneEvaluation profile.laneCovers
                    (fun lane => CarriedEvaluationResidual.residual
                      ConcreteCarrier.baseOps ops K.embed data.carriedData {
                        running := running
                        matrix := matrix
                        coefficient := lane }) coins.alpha)) =
              FiniteSumAlgebra.sumMap ops
                (canonicalFinIndices shape.runningCount) (fun _ => K.zero) := by
                  apply FiniteSumAlgebra.sumMap_congr
                  intro running _
                  have paddedZero :
                      paddedLaneEvaluation profile.laneCovers
                          (fun lane => CarriedEvaluationResidual.residual
                            ConcreteCarrier.baseOps ops K.embed data.carriedData {
                              running := running
                              matrix := matrix
                              coefficient := lane }) coins.alpha = K.zero := by
                    unfold paddedLaneEvaluation
                    calc
                      FiniteSumAlgebra.sumMap ops
                          (canonicalFinIndices ringDegree)
                          (fun lane => K.mul
                            (NumericBooleanDomain.tensorWeight ops
                              (liveLane profile.laneCovers lane) coins.alpha)
                            (CarriedEvaluationResidual.residual
                              ConcreteCarrier.baseOps ops K.embed
                              data.carriedData {
                                running := running
                                matrix := matrix
                                coefficient := lane })) =
                        FiniteSumAlgebra.sumMap ops
                          (canonicalFinIndices ringDegree) (fun _ => K.zero) := by
                            apply FiniteSumAlgebra.sumMap_congr
                            intro lane _
                            rw [(CarriedEvaluationResidual.residual_eq_zero_iff_evaluationClaimHolds
                                ConcreteCarrier.baseOps ops laws K.embed
                                data.carriedData {
                                  running := running
                                  matrix := matrix
                                  coefficient := lane }).mpr
                              (truth {
                                running := running
                                matrix := matrix
                                coefficient := lane })]
                            exact laws.mul_zero _
                      _ = K.zero := FiniteSumAlgebra.sumMap_zero ops laws _
                  unfold SignedJointIdentity.gammaTerm
                  rw [paddedZero]
                  exact laws.mul_zero _
            _ = K.zero := FiniteSumAlgebra.sumMap_zero ops laws _
    _ = K.zero := FiniteSumAlgebra.sumMap_zero ops laws _

/-- Independent FE truth makes the signed compressed residual zero. This is
model-level honest completeness of the residual side; it does not by itself
claim the still-open Boolean-sum bridge to executable SumCheck. -/
theorem mixedResidual_eq_zero_of_truth
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (profile : SupportedProfile shape domain)
    (data : Data shape)
    (coins : Coins shape domain)
    (truth : Semantics.Fe.Truth data) :
    mixedResidual profile data coins = K.zero := by
  rcases truth with ⟨freshTruth, carriedTruth⟩
  unfold mixedResidual
  rw [freshResidualMix_eq_zero_of_freshTruth data coins freshTruth,
    carriedResidualMix_eq_zero_of_carriedTruth profile data coins carriedTruth]
  unfold SignedJointIdentity.gammaTerm
  change ops.add (ops.neg ops.zero) (ops.mul _ ops.zero) = ops.zero
  rw [laws.mul_zero, FiniteSumAlgebra.neg_zero ops laws, laws.zero_add]

/-- Honest initial-sum completeness parameterized by the one explicit carried
selector bridge above. The child `InitialSum.CarriedBridge` supplies this
premise from source-derived selector reproduction without duplicating the
residual algebra here. -/
theorem initial_eq_hypercubeSum_of_truth_and_carried_bridge
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (profile : SupportedProfile shape domain)
    (data : Data shape)
    (coins : Coins shape domain)
    (truth : Semantics.Fe.Truth data)
    (carriedBridge : carriedHypercubeContribution profile data coins =
      SignedJointIdentity.gammaTerm ops coins.gamma shape.sourceCount
        (carriedComputedMix profile data coins)) :
    initial profile (PublicInput.ofSources data) coins =
      hypercubeSum profile data coins := by
  apply (FiniteSumAlgebra.sub_eq_zero_iff ops laws _ _).mp
  rw [initial_sub_hypercubeSum_eq_mixedResidual_of_carried_bridge
    profile data coins carriedBridge]
  exact mixedResidual_eq_zero_of_truth profile data coins truth

end Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Fe.InitialSum
