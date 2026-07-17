import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane.Mixing
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.BooleanReproduction
import Nightstream.SuperNeo.SumCheck.HypercubeTruth

/-!
Initial Boolean sum for the canonical Split-NC block×lane polynomial.

Assurance tier: model-level.

Owns: the exact Boolean cube of the equality-gated polynomial, one
independently grouped source specialization, the paper-relative source mix,
their finite-sum equality, and honest zero-claim completeness.

Does not own: recursive SumCheck messages, off-cube degree, transcript
derivation, mixing-root soundness, packed `yZcol` terminal binding, Rust,
R1CS, costs, or row removal. It owns only the exact semantic adapter used by
generic SumCheck truth.

Emits constraints: no.

Authority boundary: the cube evaluates `Mixing.qAtPoint`, whose leaves are
derived from authoritative assignments. `claimedInitial` is definitionally
zero and is not certificate data. The source specialization below is the MLE
of Boolean leaf cubics; it is deliberately not identified with the cubic of
the source MLE at a non-Boolean point.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_ccs.nc.block_lane.initial.source.lane` | lane-specialize one source/block leaf-cubic row at `betaA` | independent semantic residual | `laneResidualAtBeta` |
| `nifs.pi_ccs.nc.block_lane.initial.source.block` | block-specialize the lane results at `betaBlock` | independent semantic residual | `sourceResidualAtBeta` |
| `nifs.pi_ccs.nc.block_lane.initial.gamma_mix` | compress source specializations with exactly `gamma^i` | computed | `mixedResidualAtBeta` |
| `nifs.pi_ccs.nc.block_lane.initial.polynomial_cube` | sum the actual equality-gated polynomial over block then lane | computed | `hypercubeSum` |
| `nifs.pi_ccs.nc.block_lane.initial.bridge` | the polynomial cube equals the independently grouped source mix | derived | `hypercubeSum_eq_mixedResidualAtBeta` |
| `nifs.pi_ccs.nc.block_lane.initial.sumcheck.total` | totalize only malformed coordinate lists to zero | semantic adapter | `sumcheckPolynomial` |
| `nifs.pi_ccs.nc.block_lane.initial.sumcheck.order` | generic recursive completions enumerate block then lane | derived | `sumcheckHypercubeSum_eq_hypercubeSum` |
| `nifs.pi_ccs.nc.block_lane.initial.claim` | the verifier-owned initial claim is zero | computed | `claimedInitial` |
| assurance | independent NC truth zeros every source specialization and the complete cube | derived | `sourceResidualAtBeta_eq_zero_of_truth`, `claimedInitial_eq_hypercubeSum_of_truth` |
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane.InitialSum

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane

private abbrev ops := ConcreteCarrier.extensionOps
private abbrev laws := ConcreteCarrier.extensionLaws

/-- Lane equality specialization of one source/block leaf-cubic row. -/
def laneResidualAtBeta
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Mixing.Coins domain)
    (source : Fin shape.sourceCount)
    (block : BooleanVertex domain.blockVariables) : K :=
  BooleanReproduction.equalityWeighted ops coins.betaA fun lane =>
    SourceProjection.rangeValueAt covers data source {
      block := block.toCubePoint ops
      lane := lane.toCubePoint ops }

/-- Block equality specialization of the lane-specialized source table. This
is an MLE of Boolean leaf cubics, not a cubic after MLE. -/
def sourceResidualAtBeta
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Mixing.Coins domain)
    (source : Fin shape.sourceCount) : K :=
  BooleanReproduction.equalityWeighted ops coins.betaBlock fun block =>
    laneResidualAtBeta covers data coins source block

/-- Paper-relative gamma compression of the independent source
specializations. -/
def mixedResidualAtBeta
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Mixing.Coins domain) : K :=
  FiniteSumAlgebra.sumMap ops
    (canonicalFinIndices shape.sourceCount) fun source =>
      SignedJointIdentity.gammaTerm ops coins.gamma
        (Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.Mixing.sourceExponent
          shape .paperNc source)
        (sourceResidualAtBeta covers data coins source)

/-- Exact block-then-lane Boolean cube of the actual NC polynomial. -/
def hypercubeSum
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Mixing.Coins domain) : K :=
  FiniteSumAlgebra.sumMap ops
    (BooleanVertex.all domain.blockVariables) fun block =>
      FiniteSumAlgebra.sumMap ops
        (BooleanVertex.all domain.laneVariables) fun lane =>
          Mixing.qAtPoint covers data coins {
            block := block.toCubePoint ops
            lane := lane.toCubePoint ops }

/-- Equality-weighted form of the same polynomial cube. Kept private so the
public surface exposes only the actual `qAtPoint` cube and its source-grouped
semantic form. -/
private def equalityWeightedMix
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Mixing.Coins domain) : K :=
  BooleanReproduction.equalityWeighted ops coins.betaBlock fun block =>
    BooleanReproduction.equalityWeighted ops coins.betaA fun lane =>
      Mixing.mixedRangeAt covers data coins {
        block := block.toCubePoint ops
        lane := lane.toCubePoint ops }

private theorem hypercubeSum_eq_equalityWeightedMix
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Mixing.Coins domain) :
    hypercubeSum covers data coins =
      equalityWeightedMix covers data coins := by
  unfold hypercubeSum equalityWeightedMix
    BooleanReproduction.equalityWeighted
  apply FiniteSumAlgebra.sumMap_congr
  intro block _
  rw [← FiniteSumAlgebra.sumMap_mul_left ops laws
    (block.equalityWeight ops coins.betaBlock)]
  apply FiniteSumAlgebra.sumMap_congr
  intro lane _
  unfold Mixing.qAtPoint
  rw [SumCheckTruthPath.pointEquality_toCubePoint_eq_equalityWeight ops laws]
  rw [SumCheckTruthPath.pointEquality_toCubePoint_eq_equalityWeight ops laws]
  exact laws.mul_assoc _ _ _

private theorem equalityWeightedMix_eq_mixedResidualAtBeta
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Mixing.Coins domain) :
    equalityWeightedMix covers data coins =
      mixedResidualAtBeta covers data coins := by
  let indices := canonicalFinIndices shape.sourceCount
  let weights : Fin shape.sourceCount → K := fun source =>
    TargetPolynomial.power ops.toOps coins.gamma
      (Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.Mixing.sourceExponent
        shape .paperNc source)
  let values : Fin shape.sourceCount →
      BooleanVertex domain.blockVariables →
      BooleanVertex domain.laneVariables → K := fun source block lane =>
    SourceProjection.rangeValueAt covers data source {
      block := block.toCubePoint ops
      lane := lane.toCubePoint ops }
  unfold equalityWeightedMix Mixing.mixedRangeAt mixedResidualAtBeta
    sourceResidualAtBeta laneResidualAtBeta SignedJointIdentity.gammaTerm
  change
    BooleanReproduction.equalityWeighted ops coins.betaBlock (fun block =>
      BooleanReproduction.equalityWeighted ops coins.betaA (fun lane =>
        FiniteSumAlgebra.sumMap ops indices fun source =>
          ops.mul (weights source) (values source block lane))) =
      FiniteSumAlgebra.sumMap ops indices fun source =>
        ops.mul (weights source)
          (BooleanReproduction.equalityWeighted ops coins.betaBlock fun block =>
            BooleanReproduction.equalityWeighted ops coins.betaA fun lane =>
              values source block lane)
  calc
    BooleanReproduction.equalityWeighted ops coins.betaBlock (fun block =>
        BooleanReproduction.equalityWeighted ops coins.betaA (fun lane =>
          FiniteSumAlgebra.sumMap ops indices fun source =>
            ops.mul (weights source) (values source block lane))) =
      BooleanReproduction.equalityWeighted ops coins.betaBlock (fun block =>
        FiniteSumAlgebra.sumMap ops indices fun source =>
          ops.mul (weights source)
            (BooleanReproduction.equalityWeighted ops coins.betaA fun lane =>
              values source block lane)) := by
        apply congrArg
        funext block
        exact BooleanReproduction.equalityWeighted_sumMap ops laws
          indices weights (fun source lane => values source block lane)
          coins.betaA
    _ = FiniteSumAlgebra.sumMap ops indices fun source =>
          ops.mul (weights source)
            (BooleanReproduction.equalityWeighted ops coins.betaBlock fun block =>
              BooleanReproduction.equalityWeighted ops coins.betaA fun lane =>
                values source block lane) :=
      BooleanReproduction.equalityWeighted_sumMap ops laws
        indices weights
        (fun source block =>
          BooleanReproduction.equalityWeighted ops coins.betaA fun lane =>
            values source block lane)
        coins.betaBlock

/-- The actual polynomial cube is exactly the independently grouped source
specialization mix. This is finite selector linearity, not a soundness
assumption and not an off-cube MLE/cubic identification. -/
theorem hypercubeSum_eq_mixedResidualAtBeta
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Mixing.Coins domain) :
    hypercubeSum covers data coins =
      mixedResidualAtBeta covers data coins := by
  rw [hypercubeSum_eq_equalityWeightedMix]
  exact equalityWeightedMix_eq_mixedResidualAtBeta covers data coins

/-- Total semantic evaluator used only to instantiate generic SumCheck truth.
The public polynomial remains fail-closed, and exact round-count checking
makes this default branch unreachable in an accepted verifier execution. -/
def sumcheckPolynomial
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Mixing.Coins domain)
    (coordinates : List K) : K :=
  (Mixing.polynomial covers data coins coordinates).getD K.zero

/-- On an exact typed point, totalization agrees with the source-derived NC
polynomial. -/
theorem sumcheckPolynomial_coordinates_eq_qAtPoint
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Mixing.Coins domain)
    (point : Point domain) :
    sumcheckPolynomial covers data coins point.coordinates =
      Mixing.qAtPoint covers data coins point := by
  unfold sumcheckPolynomial
  rw [Mixing.polynomial_coordinates_eq_qAtPoint]
  rfl

/-- Generic recursive SumCheck initial sum for the exact block×lane round
count. -/
def sumcheckHypercubeSum
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Mixing.Coins domain) : K :=
  SumCheck.Finite.HypercubeTruth.sumCompletions ops.toOps
    (sumcheckPolynomial covers data coins) []
    (domain.blockVariables + domain.laneVariables)

/-- Recursive completions and the typed product sum enumerate the same
block-then-lane Boolean points. This is the semantic coordinate-order bridge
for later SumCheck rounds. -/
theorem sumcheckHypercubeSum_eq_hypercubeSum
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Mixing.Coins domain) :
    sumcheckHypercubeSum covers data coins =
      hypercubeSum covers data coins := by
  unfold sumcheckHypercubeSum
  rw [SumCheck.Finite.HypercubeTruth.sumCompletions_add]
  rw [SumCheckTruthPath.sumCompletions_eq_vertexSum ops laws]
  unfold hypercubeSum FiniteSumAlgebra.sumMap
  simp only [List.nil_append]
  congr 1
  apply List.map_congr_left
  intro block _
  rw [SumCheckTruthPath.sumCompletions_eq_vertexSum ops laws]
  congr 1
  apply List.map_congr_left
  intro lane _
  exact sumcheckPolynomial_coordinates_eq_qAtPoint covers data coins {
    block := block.toCubePoint ops
    lane := lane.toCubePoint ops }

/-- Independent full-carrier NC truth zeros one source's complete
equality-weighted Boolean leaf-cubic table. -/
theorem sourceResidualAtBeta_eq_zero_of_truth
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Mixing.Coins domain)
    (truth : Semantics.Nc.Truth data)
    (source : Fin shape.sourceCount) :
    sourceResidualAtBeta covers data coins source = K.zero := by
  have residuals :=
    SourceProjection.booleanResidualsZero_of_truth covers data truth
  unfold sourceResidualAtBeta laneResidualAtBeta
    BooleanReproduction.equalityWeighted
  calc
    FiniteSumAlgebra.sumMap ops
        (BooleanVertex.all domain.blockVariables) _ =
      FiniteSumAlgebra.sumMap ops
        (BooleanVertex.all domain.blockVariables) (fun _ => ops.zero) := by
          apply FiniteSumAlgebra.sumMap_congr
          intro block _
          have innerZero :
              FiniteSumAlgebra.sumMap ops
                  (BooleanVertex.all domain.laneVariables) (fun lane =>
                    ops.mul (lane.equalityWeight ops coins.betaA)
                      (SourceProjection.rangeValueAt covers data source {
                        block := block.toCubePoint ops
                        lane := lane.toCubePoint ops })) = ops.zero := by
            calc
              FiniteSumAlgebra.sumMap ops
                  (BooleanVertex.all domain.laneVariables) _ =
                FiniteSumAlgebra.sumMap ops
                  (BooleanVertex.all domain.laneVariables)
                  (fun _ => ops.zero) := by
                    apply FiniteSumAlgebra.sumMap_congr
                    intro lane _
                    rw [show
                      SourceProjection.rangeValueAt covers data source {
                          block := block.toCubePoint ops
                          lane := lane.toCubePoint ops } = ops.zero by
                        simpa [booleanPoint] using
                          residuals source
                            (BlockNcDomain.blockIndex block)
                            (BlockNcDomain.laneIndex lane)]
                    exact laws.mul_zero _
              _ = ops.zero := FiniteSumAlgebra.sumMap_zero ops laws _
          change ops.mul (block.equalityWeight ops coins.betaBlock)
              (FiniteSumAlgebra.sumMap ops
                (BooleanVertex.all domain.laneVariables) (fun lane =>
                  ops.mul (lane.equalityWeight ops coins.betaA)
                    (SourceProjection.rangeValueAt covers data source {
                      block := block.toCubePoint ops
                      lane := lane.toCubePoint ops }))) = ops.zero
          rw [innerZero]
          exact laws.mul_zero _
    _ = ops.zero := FiniteSumAlgebra.sumMap_zero ops laws _

/-- Honest NC truth zeros the paper-relative source mixture. -/
theorem mixedResidualAtBeta_eq_zero_of_truth
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Mixing.Coins domain)
    (truth : Semantics.Nc.Truth data) :
    mixedResidualAtBeta covers data coins = K.zero := by
  unfold mixedResidualAtBeta
  calc
    FiniteSumAlgebra.sumMap ops
        (canonicalFinIndices shape.sourceCount) _ =
      FiniteSumAlgebra.sumMap ops
        (canonicalFinIndices shape.sourceCount) (fun _ => ops.zero) := by
          apply FiniteSumAlgebra.sumMap_congr
          intro source _
          unfold SignedJointIdentity.gammaTerm
          rw [sourceResidualAtBeta_eq_zero_of_truth
            covers data coins truth source]
          exact laws.mul_zero _
    _ = ops.zero := FiniteSumAlgebra.sumMap_zero ops laws _

/-- Honest NC truth zeros the actual equality-gated Boolean polynomial cube. -/
theorem hypercubeSum_eq_zero_of_truth
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Mixing.Coins domain)
    (truth : Semantics.Nc.Truth data) :
    hypercubeSum covers data coins = K.zero := by
  rw [hypercubeSum_eq_mixedResidualAtBeta]
  exact mixedResidualAtBeta_eq_zero_of_truth covers data coins truth

/-- The NC initial claim is verifier-owned constant data. -/
def claimedInitial : K := K.zero

/-- Honest full-carrier NC truth makes the verifier-owned zero claim equal
the exact canonical block×lane polynomial cube. -/
theorem claimedInitial_eq_hypercubeSum_of_truth
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Mixing.Coins domain)
    (truth : Semantics.Nc.Truth data) :
    claimedInitial = hypercubeSum covers data coins := by
  exact (hypercubeSum_eq_zero_of_truth covers data coins truth).symm

/-- Honest full-carrier NC truth also closes the generic recursive SumCheck
initial sum. -/
theorem claimedInitial_eq_sumcheckHypercubeSum_of_truth
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Mixing.Coins domain)
    (truth : Semantics.Nc.Truth data) :
    claimedInitial = sumcheckHypercubeSum covers data coins := by
  rw [sumcheckHypercubeSum_eq_hypercubeSum]
  exact claimedInitial_eq_hypercubeSum_of_truth covers data coins truth

end Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane.InitialSum
