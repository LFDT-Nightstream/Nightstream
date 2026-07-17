import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Fe.InitialSum

/-!
Carried Boolean-selector closure for the Split-NC FE initial sum.

Protocol: SuperNeo `Pi_CCS`.
Phase: FE running-source Boolean restriction.
Constraint family: running carried-evaluation images only; this file emits no
rows.

Owns: finite matrix/running linearity over the lane and row selector lemmas
proved by `InitialSum`, and the final equality between the carried product-cube
contribution and the shifted source-derived carried mix.

Does not own: source-table construction; lane materialization; fresh CCS;
transcripts; SumCheck rounds; Rust; R1CS; or counts.

Emits constraints: no.

Authority boundary: every value is derived from `Sources.Data`. This file only
reassociates and reorders finite sums using `FiniteSumAlgebra` and the shared
Boolean reproduction theorem.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_ccs.fe.initial.carried.lane` | lane selector reproduces every padded running term | derived | `carriedTerm_selectorSum` |
| `nifs.pi_ccs.fe.initial.carried.row` | row selector converts source tables to prior computed coefficients | derived | `carriedTerm_runningSource_selectorSum` |
| `nifs.pi_ccs.fe.initial.carried.bridge` | carried product-cube sum equals `gamma^N` times computed mix | derived | `carriedHypercubeContribution_eq_shiftedComputedMix` |
| assurance | `initial - sum Q` equals the independent signed residual | model-level | `initial_sub_hypercubeSum_eq_mixedResidual` |
| assurance | FE truth implies exact generic SumCheck initial equality | model-level | `initial_eq_sumcheckHypercubeSum_of_truth` |
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Fe.InitialSum.CarriedBridge

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Fe
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Fe.InitialSum

private abbrev ops := ConcreteCarrier.extensionOps
private abbrev laws := ConcreteCarrier.extensionLaws

set_option maxRecDepth 10000
set_option maxHeartbeats 1000000

private theorem equalityWeighted_sumMap_unweighted
    {Index : Type}
    {variables : Nat}
    (indices : List Index)
    (values : Index -> BooleanVertex variables -> K)
    (point : CubePoint K variables) :
    BooleanReproduction.equalityWeighted ops point (fun vertex =>
        FiniteSumAlgebra.sumMap ops indices (fun index =>
          values index vertex)) =
      FiniteSumAlgebra.sumMap ops indices (fun index =>
        BooleanReproduction.equalityWeighted ops point (values index)) := by
  calc
    BooleanReproduction.equalityWeighted ops point (fun vertex =>
        FiniteSumAlgebra.sumMap ops indices (fun index =>
          values index vertex)) =
      BooleanReproduction.equalityWeighted ops point (fun vertex =>
        FiniteSumAlgebra.sumMap ops indices (fun index =>
          K.mul ops.one (values index vertex))) := by
            apply congrArg (BooleanReproduction.equalityWeighted ops point)
            funext vertex
            apply FiniteSumAlgebra.sumMap_congr
            intro index _
            exact (laws.one_mul _).symm
    _ = FiniteSumAlgebra.sumMap ops indices (fun index =>
        K.mul ops.one
          (BooleanReproduction.equalityWeighted ops point (values index))) :=
      BooleanReproduction.equalityWeighted_sumMap ops laws indices
        (fun _ => ops.one) values point
    _ = FiniteSumAlgebra.sumMap ops indices (fun index =>
        BooleanReproduction.equalityWeighted ops point (values index)) := by
          apply FiniteSumAlgebra.sumMap_congr
          intro index _
          exact laws.one_mul _

/-- Lane equality weighting commutes through the exact matrix/running gamma
mix and reproduces every padded lane MLE at `lanePoint`. -/
theorem carriedTerm_selectorSum
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (covers : LaneCovers domain)
    (gamma : K)
    (lanePoint : CubePoint K domain.laneVariables)
    (yRing : YRingValues shape) :
    BooleanReproduction.equalityWeighted ops lanePoint (fun sampled =>
        carriedTermFromYRing covers gamma (sampled.toCubePoint ops) yRing) =
      carriedTermFromYRing covers gamma lanePoint yRing := by
  unfold carriedTermFromYRing
  calc
    BooleanReproduction.equalityWeighted ops lanePoint (fun sampled =>
        FiniteSumAlgebra.sumMap ops (canonicalFinIndices shape.matrixCount)
          (fun matrix => FiniteSumAlgebra.sumMap ops
            (canonicalFinIndices shape.runningCount) (fun running =>
              SignedJointIdentity.gammaTerm ops gamma
                (carriedGammaExponent shape running matrix)
                (paddedLaneEvaluation covers
                  (yRing (Data.runningIndex running) matrix)
                  (sampled.toCubePoint ops))))) =
      FiniteSumAlgebra.sumMap ops (canonicalFinIndices shape.matrixCount)
        (fun matrix => BooleanReproduction.equalityWeighted ops lanePoint
          (fun sampled => FiniteSumAlgebra.sumMap ops
            (canonicalFinIndices shape.runningCount) (fun running =>
              SignedJointIdentity.gammaTerm ops gamma
                (carriedGammaExponent shape running matrix)
                (paddedLaneEvaluation covers
                  (yRing (Data.runningIndex running) matrix)
                  (sampled.toCubePoint ops))))) :=
        equalityWeighted_sumMap_unweighted _ _ _
    _ = FiniteSumAlgebra.sumMap ops (canonicalFinIndices shape.matrixCount)
        (fun matrix => FiniteSumAlgebra.sumMap ops
          (canonicalFinIndices shape.runningCount) (fun running =>
            SignedJointIdentity.gammaTerm ops gamma
              (carriedGammaExponent shape running matrix)
              (BooleanReproduction.equalityWeighted ops lanePoint
                (fun sampled => paddedLaneEvaluation covers
                  (yRing (Data.runningIndex running) matrix)
                  (sampled.toCubePoint ops))))) := by
          apply FiniteSumAlgebra.sumMap_congr
          intro matrix _
          unfold SignedJointIdentity.gammaTerm
          exact BooleanReproduction.equalityWeighted_sumMap ops laws
            (canonicalFinIndices shape.runningCount)
            (fun running => TargetPolynomial.power ops.toOps gamma
              (carriedGammaExponent shape running matrix))
            (fun running sampled => paddedLaneEvaluation covers
              (yRing (Data.runningIndex running) matrix)
              (sampled.toCubePoint ops)) lanePoint
    _ = FiniteSumAlgebra.sumMap ops (canonicalFinIndices shape.matrixCount)
        (fun matrix => FiniteSumAlgebra.sumMap ops
          (canonicalFinIndices shape.runningCount) (fun running =>
            SignedJointIdentity.gammaTerm ops gamma
              (carriedGammaExponent shape running matrix)
              (paddedLaneEvaluation covers
                (yRing (Data.runningIndex running) matrix) lanePoint))) := by
          apply FiniteSumAlgebra.sumMap_congr
          intro matrix _
          apply FiniteSumAlgebra.sumMap_congr
          intro running _
          apply congrArg (SignedJointIdentity.gammaTerm ops gamma
            (carriedGammaExponent shape running matrix))
          unfold BooleanReproduction.equalityWeighted
          exact paddedLaneEvaluation_selectorSum covers
            (yRing (Data.runningIndex running) matrix) lanePoint
    _ = _ := rfl

/-- Row equality weighting of the carried term converts every running source
table into the independently computed prior coefficient mix. -/
theorem carriedTerm_runningSource_selectorSum
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (profile : SupportedProfile shape domain)
    (data : Data shape)
    (coins : Coins shape domain) :
    BooleanReproduction.equalityWeighted ops data.priorPoint (fun row =>
        carriedTermFromYRing profile.laneCovers coins.gamma coins.alpha
          (sourceYRingAt data (row.toCubePoint ops))) =
      carriedComputedMix profile data coins := by
  unfold carriedTermFromYRing carriedComputedMix
  calc
    BooleanReproduction.equalityWeighted ops data.priorPoint (fun row =>
        FiniteSumAlgebra.sumMap ops (canonicalFinIndices shape.matrixCount)
          (fun matrix => FiniteSumAlgebra.sumMap ops
            (canonicalFinIndices shape.runningCount) (fun running =>
              SignedJointIdentity.gammaTerm ops coins.gamma
                (carriedGammaExponent shape running matrix)
                (paddedLaneEvaluation profile.laneCovers
                  (sourceYRingAt data (row.toCubePoint ops)
                    (Data.runningIndex running) matrix) coins.alpha)))) =
      FiniteSumAlgebra.sumMap ops (canonicalFinIndices shape.matrixCount)
        (fun matrix => BooleanReproduction.equalityWeighted ops data.priorPoint
          (fun row => FiniteSumAlgebra.sumMap ops
            (canonicalFinIndices shape.runningCount) (fun running =>
              SignedJointIdentity.gammaTerm ops coins.gamma
                (carriedGammaExponent shape running matrix)
                (paddedLaneEvaluation profile.laneCovers
                  (sourceYRingAt data (row.toCubePoint ops)
                    (Data.runningIndex running) matrix) coins.alpha)))) :=
        equalityWeighted_sumMap_unweighted _ _ _
    _ = FiniteSumAlgebra.sumMap ops (canonicalFinIndices shape.matrixCount)
        (fun matrix => FiniteSumAlgebra.sumMap ops
          (canonicalFinIndices shape.runningCount) (fun running =>
            SignedJointIdentity.gammaTerm ops coins.gamma
              (carriedGammaExponent shape running matrix)
              (BooleanReproduction.equalityWeighted ops data.priorPoint
                (fun row => paddedLaneEvaluation profile.laneCovers
                  (sourceYRingAt data (row.toCubePoint ops)
                    (Data.runningIndex running) matrix) coins.alpha)))) := by
          apply FiniteSumAlgebra.sumMap_congr
          intro matrix _
          unfold SignedJointIdentity.gammaTerm
          exact BooleanReproduction.equalityWeighted_sumMap ops laws
            (canonicalFinIndices shape.runningCount)
            (fun running => TargetPolynomial.power ops.toOps coins.gamma
              (carriedGammaExponent shape running matrix))
            (fun running row => paddedLaneEvaluation profile.laneCovers
              (sourceYRingAt data (row.toCubePoint ops)
                (Data.runningIndex running) matrix) coins.alpha)
            data.priorPoint
    _ = FiniteSumAlgebra.sumMap ops (canonicalFinIndices shape.matrixCount)
        (fun matrix => FiniteSumAlgebra.sumMap ops
          (canonicalFinIndices shape.runningCount) (fun running =>
            SignedJointIdentity.gammaTerm ops coins.gamma
              (carriedGammaExponent shape running matrix)
              (paddedLaneEvaluation profile.laneCovers
                (fun lane => CarriedEvaluationResidual.computedCoefficient
                  ConcreteCarrier.baseOps ops K.embed data.carriedData {
                    running := running
                    matrix := matrix
                    coefficient := lane }) coins.alpha))) := by
          apply FiniteSumAlgebra.sumMap_congr
          intro matrix _
          apply FiniteSumAlgebra.sumMap_congr
          intro running _
          apply congrArg (SignedJointIdentity.gammaTerm ops coins.gamma
            (carriedGammaExponent shape running matrix))
          unfold BooleanReproduction.equalityWeighted
          exact paddedRunningSource_selectorSum profile.laneCovers data
            running matrix coins.alpha
    _ = _ := rfl

private theorem carriedLaneSum
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (profile : SupportedProfile shape domain)
    (data : Data shape)
    (coins : Coins shape domain)
    (row : BooleanVertex shape.rowVariables) :
    FiniteSumAlgebra.sumMap ops (BooleanVertex.all domain.laneVariables)
        (fun lane => carriedAtPoint profile data coins {
          row := row.toCubePoint ops
          lane := lane.toCubePoint ops }) =
      K.mul (row.equalityWeight ops data.priorPoint)
        (SignedJointIdentity.gammaTerm ops coins.gamma shape.sourceCount
          (carriedTermFromYRing profile.laneCovers coins.gamma coins.alpha
            (sourceYRingAt data (row.toCubePoint ops)))) := by
  let rowWeight := row.equalityWeight ops data.priorPoint
  let shift := TargetPolynomial.power ops.toOps coins.gamma shape.sourceCount
  let yRing := sourceYRingAt data (row.toCubePoint ops)
  calc
    FiniteSumAlgebra.sumMap ops (BooleanVertex.all domain.laneVariables)
        (fun lane => carriedAtPoint profile data coins {
          row := row.toCubePoint ops
          lane := lane.toCubePoint ops }) =
      FiniteSumAlgebra.sumMap ops (BooleanVertex.all domain.laneVariables)
        (fun lane => K.mul rowWeight (K.mul shift
          (K.mul (lane.equalityWeight ops coins.alpha)
            (carriedTermFromYRing profile.laneCovers coins.gamma
              (lane.toCubePoint ops) yRing)))) := by
          apply FiniteSumAlgebra.sumMap_congr
          intro lane _
          unfold carriedAtPoint SignedJointIdentity.gammaTerm rowWeight shift yRing
          rw [SumCheckTruthPath.pointEquality_toCubePoint_eq_equalityWeight
                ops laws,
            SumCheckTruthPath.pointEquality_toCubePoint_eq_equalityWeight
              ops laws]
          calc
            K.mul
                (K.mul (lane.equalityWeight ops coins.alpha)
                  (row.equalityWeight ops data.priorPoint))
                (K.mul
                  (TargetPolynomial.power ops.toOps coins.gamma shape.sourceCount)
                  (carriedTermFromYRing profile.laneCovers coins.gamma
                    (lane.toCubePoint ops)
                    (sourceYRingAt data (row.toCubePoint ops)))) =
              K.mul (row.equalityWeight ops data.priorPoint)
                (K.mul
                  (lane.equalityWeight ops coins.alpha)
                  (K.mul
                    (TargetPolynomial.power ops.toOps coins.gamma
                      shape.sourceCount)
                    (carriedTermFromYRing profile.laneCovers coins.gamma
                      (lane.toCubePoint ops)
                      (sourceYRingAt data (row.toCubePoint ops))))) := by
                        calc
                          K.mul
                              (K.mul (lane.equalityWeight ops coins.alpha)
                                (row.equalityWeight ops data.priorPoint)) _ =
                            K.mul
                              (K.mul (row.equalityWeight ops data.priorPoint)
                                (lane.equalityWeight ops coins.alpha)) _ :=
                                  congrArg (fun factor => K.mul factor _)
                                    (laws.mul_comm _ _)
                          _ = K.mul
                              (row.equalityWeight ops data.priorPoint)
                              (K.mul (lane.equalityWeight ops coins.alpha) _) :=
                                laws.mul_assoc _ _ _
            _ = K.mul (row.equalityWeight ops data.priorPoint)
                (K.mul
                  (TargetPolynomial.power ops.toOps coins.gamma
                    shape.sourceCount)
                  (K.mul (lane.equalityWeight ops coins.alpha)
                    (carriedTermFromYRing profile.laneCovers coins.gamma
                      (lane.toCubePoint ops)
                      (sourceYRingAt data (row.toCubePoint ops))))) := by
                        apply congrArg (K.mul
                          (row.equalityWeight ops data.priorPoint))
                        calc
                          K.mul (lane.equalityWeight ops coins.alpha)
                              (K.mul
                                (TargetPolynomial.power ops.toOps coins.gamma
                                  shape.sourceCount) _) =
                            K.mul
                              (K.mul (lane.equalityWeight ops coins.alpha)
                                (TargetPolynomial.power ops.toOps coins.gamma
                                  shape.sourceCount)) _ :=
                                    (laws.mul_assoc _ _ _).symm
                          _ = K.mul
                              (K.mul
                                (TargetPolynomial.power ops.toOps coins.gamma
                                  shape.sourceCount)
                                (lane.equalityWeight ops coins.alpha)) _ :=
                                  congrArg (fun factor => K.mul factor _)
                                    (laws.mul_comm _ _)
                          _ = K.mul
                              (TargetPolynomial.power ops.toOps coins.gamma
                                shape.sourceCount)
                              (K.mul (lane.equalityWeight ops coins.alpha) _) :=
                                laws.mul_assoc _ _ _
    _ = K.mul rowWeight
        (FiniteSumAlgebra.sumMap ops
          (BooleanVertex.all domain.laneVariables) (fun lane =>
            K.mul shift (K.mul (lane.equalityWeight ops coins.alpha)
              (carriedTermFromYRing profile.laneCovers coins.gamma
                (lane.toCubePoint ops) yRing)))) :=
      FiniteSumAlgebra.sumMap_mul_left ops laws _ _ _
    _ = K.mul rowWeight (K.mul shift
        (BooleanReproduction.equalityWeighted ops coins.alpha (fun lane =>
          carriedTermFromYRing profile.laneCovers coins.gamma
            (lane.toCubePoint ops) yRing))) := by
          apply congrArg (K.mul rowWeight)
          unfold BooleanReproduction.equalityWeighted
          change FiniteSumAlgebra.sumMap ops
              (BooleanVertex.all domain.laneVariables) (fun lane =>
                ops.mul shift
                  (ops.mul (lane.equalityWeight ops coins.alpha)
                    (carriedTermFromYRing profile.laneCovers coins.gamma
                      (lane.toCubePoint ops) yRing))) =
            ops.mul shift
              (FiniteSumAlgebra.sumMap ops
                (BooleanVertex.all domain.laneVariables) (fun lane =>
                  ops.mul (lane.equalityWeight ops coins.alpha)
                    (carriedTermFromYRing profile.laneCovers coins.gamma
                      (lane.toCubePoint ops) yRing)))
          exact FiniteSumAlgebra.sumMap_mul_left ops laws _ _ _
    _ = K.mul rowWeight (K.mul shift
        (carriedTermFromYRing profile.laneCovers coins.gamma coins.alpha
          yRing)) := by
            apply congrArg (K.mul rowWeight)
            apply congrArg (K.mul shift)
            exact carriedTerm_selectorSum profile.laneCovers coins.gamma
              coins.alpha yRing
    _ = _ := rfl

/-- The carried product-cube contribution is exactly the production
`gamma^N` shift applied to the independently source-derived computed mix. -/
theorem carriedHypercubeContribution_eq_shiftedComputedMix
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (profile : SupportedProfile shape domain)
    (data : Data shape)
    (coins : Coins shape domain) :
    carriedHypercubeContribution profile data coins =
      SignedJointIdentity.gammaTerm ops coins.gamma shape.sourceCount
        (carriedComputedMix profile data coins) := by
  unfold carriedHypercubeContribution
  calc
    FiniteSumAlgebra.sumMap ops (BooleanVertex.all shape.rowVariables)
        (fun row => FiniteSumAlgebra.sumMap ops
          (BooleanVertex.all domain.laneVariables) (fun lane =>
            carriedAtPoint profile data coins {
              row := row.toCubePoint ops
              lane := lane.toCubePoint ops })) =
      FiniteSumAlgebra.sumMap ops (BooleanVertex.all shape.rowVariables)
        (fun row => K.mul (row.equalityWeight ops data.priorPoint)
          (SignedJointIdentity.gammaTerm ops coins.gamma shape.sourceCount
            (carriedTermFromYRing profile.laneCovers coins.gamma coins.alpha
              (sourceYRingAt data (row.toCubePoint ops))))) := by
                apply FiniteSumAlgebra.sumMap_congr
                intro row _
                exact carriedLaneSum profile data coins row
    _ = SignedJointIdentity.gammaTerm ops coins.gamma shape.sourceCount
        (BooleanReproduction.equalityWeighted ops data.priorPoint (fun row =>
          carriedTermFromYRing profile.laneCovers coins.gamma coins.alpha
            (sourceYRingAt data (row.toCubePoint ops)))) := by
              unfold SignedJointIdentity.gammaTerm
              unfold BooleanReproduction.equalityWeighted
              calc
                FiniteSumAlgebra.sumMap ops
                    (BooleanVertex.all shape.rowVariables) (fun row =>
                      K.mul (row.equalityWeight ops data.priorPoint)
                        (K.mul
                          (TargetPolynomial.power ops.toOps coins.gamma
                            shape.sourceCount)
                          (carriedTermFromYRing profile.laneCovers coins.gamma
                            coins.alpha
                            (sourceYRingAt data (row.toCubePoint ops))))) =
                  FiniteSumAlgebra.sumMap ops
                    (BooleanVertex.all shape.rowVariables) (fun row =>
                      K.mul
                        (TargetPolynomial.power ops.toOps coins.gamma
                          shape.sourceCount)
                        (K.mul (row.equalityWeight ops data.priorPoint)
                          (carriedTermFromYRing profile.laneCovers coins.gamma
                            coins.alpha
                            (sourceYRingAt data (row.toCubePoint ops))))) := by
                              apply FiniteSumAlgebra.sumMap_congr
                              intro row _
                              calc
                                K.mul (row.equalityWeight ops data.priorPoint)
                                    (K.mul
                                      (TargetPolynomial.power ops.toOps
                                        coins.gamma shape.sourceCount) _) =
                                  K.mul
                                    (K.mul
                                      (row.equalityWeight ops data.priorPoint)
                                      (TargetPolynomial.power ops.toOps
                                        coins.gamma shape.sourceCount)) _ :=
                                          (laws.mul_assoc _ _ _).symm
                                _ = K.mul
                                    (K.mul
                                      (TargetPolynomial.power ops.toOps
                                        coins.gamma shape.sourceCount)
                                      (row.equalityWeight ops
                                        data.priorPoint)) _ :=
                                          congrArg (fun factor => K.mul factor _)
                                            (laws.mul_comm _ _)
                                _ = K.mul
                                    (TargetPolynomial.power ops.toOps
                                      coins.gamma shape.sourceCount)
                                    (K.mul (row.equalityWeight ops
                                      data.priorPoint) _) :=
                                        laws.mul_assoc _ _ _
                _ = _ := FiniteSumAlgebra.sumMap_mul_left ops laws _ _ _
    _ = SignedJointIdentity.gammaTerm ops coins.gamma shape.sourceCount
        (carriedComputedMix profile data coins) := by
          apply congrArg (SignedJointIdentity.gammaTerm ops coins.gamma
            shape.sourceCount)
          exact carriedTerm_runningSource_selectorSum profile data coins

/-- Complete source-derived residual identity for the typed FE product cube. -/
theorem initial_sub_hypercubeSum_eq_mixedResidual
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (profile : SupportedProfile shape domain)
    (data : Data shape)
    (coins : Coins shape domain) :
    ops.sub (initial profile (PublicInput.ofSources data) coins)
        (hypercubeSum profile data coins) =
      mixedResidual profile data coins :=
  initial_sub_hypercubeSum_eq_mixedResidual_of_carried_bridge
    profile data coins
    (carriedHypercubeContribution_eq_shiftedComputedMix profile data coins)

/-- Independent FE truth makes the verifier initial claim equal the exact
typed Boolean sum of the production-shaped FE polynomial. -/
theorem initial_eq_hypercubeSum_of_truth
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (profile : SupportedProfile shape domain)
    (data : Data shape)
    (coins : Coins shape domain)
    (truth : Semantics.Fe.Truth data) :
    initial profile (PublicInput.ofSources data) coins =
      hypercubeSum profile data coins :=
  initial_eq_hypercubeSum_of_truth_and_carried_bridge
    profile data coins truth
    (carriedHypercubeContribution_eq_shiftedComputedMix profile data coins)

/-- The same honest completeness statement at the generic recursive
`sumCompletions` boundary consumed by SumCheck truth construction. -/
theorem initial_eq_sumcheckHypercubeSum_of_truth
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (profile : SupportedProfile shape domain)
    (data : Data shape)
    (coins : Coins shape domain)
    (truth : Semantics.Fe.Truth data) :
    initial profile (PublicInput.ofSources data) coins =
      sumcheckHypercubeSum profile data coins := by
  rw [sumcheckHypercubeSum_eq_hypercubeSum]
  exact initial_eq_hypercubeSum_of_truth profile data coins truth

end Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Fe.InitialSum.CarriedBridge
