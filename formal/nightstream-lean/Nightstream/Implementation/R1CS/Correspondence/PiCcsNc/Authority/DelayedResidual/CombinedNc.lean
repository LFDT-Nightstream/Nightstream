import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane.Degree.SumCheck
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane.Terminal
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane.OutputBridge

/-!
Contract: add a production-native delayed running-assignment projection to
the canonical block×lane NC polynomial without increasing its degree-four,
five-coefficient SumCheck interface.

Assurance tier: model-level.

Owns: a delayed term derived only from `Sources.Data.runningAssignments`, its
exact Boolean-cube sum and terminal formula, coordinate-slice degree closure,
Boolean-suffix degree closure, and the explicit `batchWeight = 0` degeneration
to the ordinary NC polynomial.

Does not own: transcript sampling or domain separation, SumCheck acceptance,
the carried-parent projection identity, one-fold state continuity, commitment
binding, Rust, generated rows, costs, or row-removal permission.

Emits constraints: no.

Authority boundary: child values enter only through the typed authoritative
running assignments and the canonical `SourceProjection` block×lane table.
No `CeClaim.y_zcol` sidecar is an input. This is the active block-domain
contract, not the legacy flat nine-column-variable `directDiagonal` model.
Consequently this leaf does not by itself close `PackedYZcolBoundAtBlock`; a
later accepted-protocol theorem must bind the carried parent vector to the
cube sum and discharge transcript/state premises.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `pi_ccs.nc.delayed_running.source` | read only the raw running assignments through the canonical block×lane source table | direct dataflow | `authoritativeRunningValueAt`, `authoritativeRunningValueAt_live` |
| `pi_ccs.nc.delayed_running.cube` | the delayed Boolean cube equals the independently weighted old-block projection | derived | `delayedHypercubeSum_eq_weightedProjection` |
| `pi_ccs.nc.delayed_running.combined_cube` | the combined Boolean cube is the ordinary production NC cube plus that delayed projection | derived | `combinedHypercubeSum_eq_ordinary_add_weightedProjection` |
| `pi_ccs.nc.delayed_running.terminal` | ordinary NC terminal plus the delayed raw-child terminal is the exact combined terminal | derived | `combinedAtPoint_eq_terminalFromMessage_of_bound` |
| `pi_ccs.nc.delayed_running.degree.block` | every block-coordinate slice remains degree at most four | derived | `combinedAtPoint_block_quartic` |
| `pi_ccs.nc.delayed_running.degree.lane` | every lane-coordinate slice remains degree at most four | derived | `combinedAtPoint_lane_quartic` |
| `pi_ccs.nc.delayed_running.degree.round` | every Boolean-suffix SumCheck round remains degree at most four and uses five slots | derived | `expectedRound_quartic`, `expectedRound_has_five_coefficients` |
| `pi_ccs.nc.delayed_running.zero_weight` | zero batch weight erases exactly the delayed summand | computed | `combinedAtPoint_eq_ordinary_of_batchWeight_eq_zero` |
-/

namespace Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual.CombinedNc

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.DegreeSupport
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.Degree
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane.Degree.Polynomial
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane.Degree.Source

private abbrev ops := ConcreteCarrier.extensionOps
private abbrev laws := ConcreteCarrier.extensionLaws
private abbrev Polynomial := Nightstream.SuperNeo.SumCheck.Finite.FixedPolynomial K

/-! ## Authoritative running-source polynomial -/

/-- Base-field weights used to combine the authoritative running assignments.
The production Π_DEC specialization supplies its fixed radix powers; this leaf
keeps the finite family generic and does not sample these weights. -/
abbrev RunningWeights (shape : SemanticShape) := Fin shape.runningCount → F

/-- Nested block×lane MLE of the weighted authoritative running assignments.
The source injection is definitional and `assignment_runningIndex` identifies
every leaf with `data.runningAssignments running`. -/
def authoritativeRunningValueAt
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (weights : RunningWeights shape)
    (point : Point domain) : K :=
  FiniteSumAlgebra.sumMap ops
    (canonicalFinIndices shape.runningCount) fun running =>
      K.mul (K.embed (weights running))
        (SourceProjection.sourceValueAt covers data
          (Data.runningIndex running) point)

/-- At each live Boolean block×lane leaf, the delayed source is visibly a
weighted expression over `data.runningAssignments`; no carried sidecar occurs
in the statement. -/
theorem authoritativeRunningValueAt_live
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (weights : RunningWeights shape)
    (block : Fin (Phi81ColumnLayout.blockCount shape.carrierWidth))
    (lane : Fin ringDegree) :
    authoritativeRunningValueAt covers data weights
        (booleanPoint
          (domain.carrierBlock covers block)
          (domain.phi81Lane covers lane)) =
      FiniteSumAlgebra.sumMap ops
        (canonicalFinIndices shape.runningCount) fun running =>
          K.mul (K.embed (weights running))
            (K.embed (Semantics.Nc.BlockLane.value
              (data.runningAssignments running) block lane)) := by
  unfold authoritativeRunningValueAt
  apply FiniteSumAlgebra.sumMap_congr
  intro running _
  apply congrArg (K.mul (K.embed (weights running)))
  rw [SourceProjection.sourceValueAt_live]
  rw [data.assignment_runningIndex]

/-- Canonical MLE of the lane monomials `producerBeta^lane`. Padded lanes are
included in the Boolean table; their authoritative source values are zero. -/
def betaPowerSelector
    {domain : BlockNcDomain}
    (producerBeta : K)
    (lanePoint : CubePoint K domain.laneVariables) : K :=
  (BooleanTable.tabulate fun lane =>
    TargetPolynomial.power ops.toOps producerBeta
      (BlockNcDomain.laneIndex lane).val).evaluate
      ops lanePoint

/-- Delayed running-assignment term at one active block×lane point. -/
def delayedAtPoint
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (weights : RunningWeights shape)
    (producerBeta batchWeight : K)
    (oldBlock : CubePoint K domain.blockVariables)
    (point : Point domain) : K :=
  K.mul batchWeight
    (K.mul
      (SumCheckTruthPath.pointEquality ops point.block oldBlock)
      (K.mul (betaPowerSelector producerBeta point.lane)
        (authoritativeRunningValueAt covers data weights point)))

/-- Ordinary semantic NC plus the production-native delayed term. -/
def combinedAtPoint
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Mixing.Coins domain)
    (weights : RunningWeights shape)
    (producerBeta batchWeight : K)
    (oldBlock : CubePoint K domain.blockVariables)
    (point : Point domain) : K :=
  K.add (Mixing.qAtPoint covers data coins point)
    (delayedAtPoint covers data weights producerBeta batchWeight
      oldBlock point)

/-! ## Exact Boolean-cube normalization -/

/-- Independently evaluated old-block projection of the authoritative running
assignments. This is the exact scalar the delayed cube carries. -/
def authoritativeRunningProjection
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (weights : RunningWeights shape)
    (producerBeta : K)
    (oldBlock : CubePoint K domain.blockVariables) : K :=
  FiniteSumAlgebra.sumMap ops
    (BooleanVertex.all domain.laneVariables) fun lane =>
      K.mul (TargetPolynomial.power ops.toOps producerBeta
          (BlockNcDomain.laneIndex lane).val)
        (authoritativeRunningValueAt covers data weights {
          block := oldBlock
          lane := lane.toCubePoint ops })

/-- Boolean-cube sum of the delayed term in production block-then-lane order. -/
def delayedHypercubeSum
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (weights : RunningWeights shape)
    (producerBeta batchWeight : K)
    (oldBlock : CubePoint K domain.blockVariables) : K :=
  FiniteSumAlgebra.sumMap ops
    (BooleanVertex.all domain.blockVariables) fun block =>
      FiniteSumAlgebra.sumMap ops
        (BooleanVertex.all domain.laneVariables) fun lane =>
          delayedAtPoint covers data weights producerBeta batchWeight
            oldBlock {
              block := block.toCubePoint ops
              lane := lane.toCubePoint ops }

/-- Boolean-cube sum of the exact combined production NC polynomial, in the
same block-then-lane order as `InitialSum.hypercubeSum`. -/
def combinedHypercubeSum
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Mixing.Coins domain)
    (weights : RunningWeights shape)
    (producerBeta batchWeight : K)
    (oldBlock : CubePoint K domain.blockVariables) : K :=
  FiniteSumAlgebra.sumMap ops
    (BooleanVertex.all domain.blockVariables) fun block =>
      FiniteSumAlgebra.sumMap ops
        (BooleanVertex.all domain.laneVariables) fun lane =>
          combinedAtPoint covers data coins weights producerBeta batchWeight
            oldBlock {
              block := block.toCubePoint ops
              lane := lane.toCubePoint ops }

private theorem sourceValueAt_booleanLane
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (source : Fin shape.sourceCount)
    (block : CubePoint K domain.blockVariables)
    (lane : BooleanVertex domain.laneVariables) :
    SourceProjection.sourceValueAt covers data source {
        block := block
        lane := lane.toCubePoint ops } =
      SourceProjection.blockValueAt covers data source block lane := by
  unfold SourceProjection.sourceValueAt SourceProjection.laneTableAtBlock
  rw [SumCheckTruthPath.evaluate_toCubePoint_eq_valueAt ops laws]
  rw [BooleanTable.valueAt_tabulate]

private theorem sourceValueAt_block_reproduce
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (source : Fin shape.sourceCount)
    (oldBlock : CubePoint K domain.blockVariables)
    (lane : BooleanVertex domain.laneVariables) :
    BooleanReproduction.equalityWeighted ops oldBlock (fun block =>
        SourceProjection.sourceValueAt covers data source {
          block := block.toCubePoint ops
          lane := lane.toCubePoint ops }) =
      SourceProjection.sourceValueAt covers data source {
        block := oldBlock
        lane := lane.toCubePoint ops } := by
  calc
    BooleanReproduction.equalityWeighted ops oldBlock (fun block =>
        SourceProjection.sourceValueAt covers data source {
          block := block.toCubePoint ops
          lane := lane.toCubePoint ops }) =
      BooleanReproduction.equalityWeighted ops oldBlock (fun block =>
        K.embed (SourceProjection.paddedValue covers data source
          (BlockNcDomain.blockIndex block)
          (BlockNcDomain.laneIndex lane))) := by
        apply congrArg
        funext block
        exact SourceProjection.sourceValueAt_toCubePoint_eq_embed_paddedValue
          covers data source block lane
    _ = (BooleanTable.tabulate (fun block =>
          K.embed (SourceProjection.paddedValue covers data source
            (BlockNcDomain.blockIndex block)
            (BlockNcDomain.laneIndex lane)))).evaluate ops oldBlock :=
      BooleanReproduction.equalityWeighted_tabulate_eq_evaluate
        ops laws oldBlock _
    _ = SourceProjection.sourceValueAt covers data source {
        block := oldBlock
        lane := lane.toCubePoint ops } := by
      rw [sourceValueAt_booleanLane covers data source oldBlock lane]
      rfl

private theorem authoritativeRunningValueAt_block_reproduce
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (weights : RunningWeights shape)
    (oldBlock : CubePoint K domain.blockVariables)
    (lane : BooleanVertex domain.laneVariables) :
    BooleanReproduction.equalityWeighted ops oldBlock (fun block =>
        authoritativeRunningValueAt covers data weights {
          block := block.toCubePoint ops
          lane := lane.toCubePoint ops }) =
      authoritativeRunningValueAt covers data weights {
        block := oldBlock
        lane := lane.toCubePoint ops } := by
  unfold authoritativeRunningValueAt
  calc
    BooleanReproduction.equalityWeighted ops oldBlock (fun block =>
        FiniteSumAlgebra.sumMap ops
          (canonicalFinIndices shape.runningCount) fun running =>
            K.mul (K.embed (weights running))
              (SourceProjection.sourceValueAt covers data
                (Data.runningIndex running) {
                  block := block.toCubePoint ops
                  lane := lane.toCubePoint ops })) =
      FiniteSumAlgebra.sumMap ops
        (canonicalFinIndices shape.runningCount) fun running =>
          K.mul (K.embed (weights running))
            (BooleanReproduction.equalityWeighted ops oldBlock fun block =>
              SourceProjection.sourceValueAt covers data
                (Data.runningIndex running) {
                  block := block.toCubePoint ops
                  lane := lane.toCubePoint ops }) :=
        BooleanReproduction.equalityWeighted_sumMap ops laws
          (canonicalFinIndices shape.runningCount)
          (fun running => K.embed (weights running))
          (fun running block =>
            SourceProjection.sourceValueAt covers data
              (Data.runningIndex running) {
                block := block.toCubePoint ops
                lane := lane.toCubePoint ops }) oldBlock
    _ = FiniteSumAlgebra.sumMap ops
        (canonicalFinIndices shape.runningCount) fun running =>
          K.mul (K.embed (weights running))
            (SourceProjection.sourceValueAt covers data
              (Data.runningIndex running) {
                block := oldBlock
                lane := lane.toCubePoint ops }) := by
      apply FiniteSumAlgebra.sumMap_congr
      intro running _
      apply congrArg (K.mul (K.embed (weights running)))
      exact sourceValueAt_block_reproduce covers data
        (Data.runningIndex running) oldBlock lane

/-- Exact cube normalization of the production-native delayed term. The
zero-weight case is deliberately not cancelled. -/
theorem delayedHypercubeSum_eq_weightedProjection
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (weights : RunningWeights shape)
    (producerBeta batchWeight : K)
    (oldBlock : CubePoint K domain.blockVariables) :
    delayedHypercubeSum covers data weights producerBeta batchWeight
        oldBlock =
      K.mul batchWeight
        (authoritativeRunningProjection covers data weights
          producerBeta oldBlock) := by
  let blocks := BooleanVertex.all domain.blockVariables
  let lanes := BooleanVertex.all domain.laneVariables
  unfold delayedHypercubeSum delayedAtPoint
  calc
    FiniteSumAlgebra.sumMap ops blocks (fun block =>
        FiniteSumAlgebra.sumMap ops lanes (fun lane =>
          K.mul batchWeight
            (K.mul
              (SumCheckTruthPath.pointEquality ops
                (block.toCubePoint ops) oldBlock)
              (K.mul (betaPowerSelector producerBeta (lane.toCubePoint ops))
                (authoritativeRunningValueAt covers data weights {
                  block := block.toCubePoint ops
                  lane := lane.toCubePoint ops }))))) =
      FiniteSumAlgebra.sumMap ops blocks (fun block =>
        FiniteSumAlgebra.sumMap ops lanes (fun lane =>
          K.mul batchWeight
            (K.mul (block.equalityWeight ops oldBlock)
              (K.mul (TargetPolynomial.power ops.toOps producerBeta
                  (BlockNcDomain.laneIndex lane).val)
                (authoritativeRunningValueAt covers data weights {
                  block := block.toCubePoint ops
                  lane := lane.toCubePoint ops }))))) := by
        apply FiniteSumAlgebra.sumMap_congr
        intro block _
        apply FiniteSumAlgebra.sumMap_congr
        intro lane _
        rw [SumCheckTruthPath.pointEquality_toCubePoint_eq_equalityWeight
          ops laws]
        unfold betaPowerSelector
        rw [SumCheckTruthPath.evaluate_toCubePoint_eq_valueAt ops laws]
        rw [BooleanTable.valueAt_tabulate]
    _ = K.mul batchWeight
        (FiniteSumAlgebra.sumMap ops blocks (fun block =>
          FiniteSumAlgebra.sumMap ops lanes (fun lane =>
            K.mul (block.equalityWeight ops oldBlock)
              (K.mul (TargetPolynomial.power ops.toOps producerBeta
                  (BlockNcDomain.laneIndex lane).val)
                (authoritativeRunningValueAt covers data weights {
                  block := block.toCubePoint ops
                  lane := lane.toCubePoint ops }))))) := by
        calc
          FiniteSumAlgebra.sumMap ops blocks (fun block =>
              FiniteSumAlgebra.sumMap ops lanes (fun lane =>
                K.mul batchWeight
                  (K.mul (block.equalityWeight ops oldBlock)
                    (K.mul (TargetPolynomial.power ops.toOps producerBeta
                        (BlockNcDomain.laneIndex lane).val)
                      (authoritativeRunningValueAt covers data weights {
                        block := block.toCubePoint ops
                        lane := lane.toCubePoint ops }))))) =
            FiniteSumAlgebra.sumMap ops blocks (fun block =>
              K.mul batchWeight
                (FiniteSumAlgebra.sumMap ops lanes (fun lane =>
                  K.mul (block.equalityWeight ops oldBlock)
                    (K.mul (TargetPolynomial.power ops.toOps producerBeta
                        (BlockNcDomain.laneIndex lane).val)
                      (authoritativeRunningValueAt covers data weights {
                        block := block.toCubePoint ops
                        lane := lane.toCubePoint ops }))))) := by
              apply FiniteSumAlgebra.sumMap_congr
              intro block _
              exact FiniteSumAlgebra.sumMap_mul_left
                ops laws batchWeight lanes _
          _ = K.mul batchWeight
              (FiniteSumAlgebra.sumMap ops blocks (fun block =>
                FiniteSumAlgebra.sumMap ops lanes (fun lane =>
                  K.mul (block.equalityWeight ops oldBlock)
                    (K.mul (TargetPolynomial.power ops.toOps producerBeta
                        (BlockNcDomain.laneIndex lane).val)
                      (authoritativeRunningValueAt covers data weights {
                        block := block.toCubePoint ops
                        lane := lane.toCubePoint ops }))))) :=
            FiniteSumAlgebra.sumMap_mul_left ops laws batchWeight blocks _
    _ = K.mul batchWeight
        (FiniteSumAlgebra.sumMap ops lanes (fun lane =>
          FiniteSumAlgebra.sumMap ops blocks (fun block =>
            K.mul (block.equalityWeight ops oldBlock)
              (K.mul (TargetPolynomial.power ops.toOps producerBeta
                  (BlockNcDomain.laneIndex lane).val)
                (authoritativeRunningValueAt covers data weights {
                  block := block.toCubePoint ops
                  lane := lane.toCubePoint ops }))))) := by
        apply congrArg (K.mul batchWeight)
        exact FiniteSumAlgebra.sumMap_swap ops laws blocks lanes _
    _ = K.mul batchWeight
        (FiniteSumAlgebra.sumMap ops lanes (fun lane =>
            K.mul (TargetPolynomial.power ops.toOps producerBeta
                (BlockNcDomain.laneIndex lane).val)
            (BooleanReproduction.equalityWeighted ops oldBlock (fun block =>
              authoritativeRunningValueAt covers data weights {
                block := block.toCubePoint ops
                lane := lane.toCubePoint ops })))) := by
        apply congrArg (K.mul batchWeight)
        apply FiniteSumAlgebra.sumMap_congr
        intro lane _
        calc
          FiniteSumAlgebra.sumMap ops blocks (fun block =>
              K.mul (block.equalityWeight ops oldBlock)
                (K.mul (TargetPolynomial.power ops.toOps producerBeta
                    (BlockNcDomain.laneIndex lane).val)
                  (authoritativeRunningValueAt covers data weights {
                    block := block.toCubePoint ops
                    lane := lane.toCubePoint ops }))) =
            FiniteSumAlgebra.sumMap ops blocks (fun block =>
              K.mul (TargetPolynomial.power ops.toOps producerBeta
                  (BlockNcDomain.laneIndex lane).val)
                (K.mul (block.equalityWeight ops oldBlock)
                  (authoritativeRunningValueAt covers data weights {
                    block := block.toCubePoint ops
                    lane := lane.toCubePoint ops }))) := by
              apply FiniteSumAlgebra.sumMap_congr
              intro block _
              let selector := block.equalityWeight ops oldBlock
              let power := TargetPolynomial.power ops.toOps producerBeta
                (BlockNcDomain.laneIndex lane).val
              let value := authoritativeRunningValueAt covers data weights {
                block := block.toCubePoint ops
                lane := lane.toCubePoint ops }
              change K.mul selector (K.mul power value) =
                K.mul power (K.mul selector value)
              calc
                K.mul selector (K.mul power value) =
                    K.mul (K.mul selector power) value :=
                  (laws.mul_assoc selector power value).symm
                _ = K.mul (K.mul power selector) value :=
                  congrArg (fun product : K => K.mul product value)
                    (laws.mul_comm selector power)
                _ = K.mul power (K.mul selector value) :=
                  laws.mul_assoc power selector value
          _ = K.mul (TargetPolynomial.power ops.toOps producerBeta
                (BlockNcDomain.laneIndex lane).val)
              (FiniteSumAlgebra.sumMap ops blocks (fun block =>
                K.mul (block.equalityWeight ops oldBlock)
                  (authoritativeRunningValueAt covers data weights {
                    block := block.toCubePoint ops
                    lane := lane.toCubePoint ops }))) :=
            FiniteSumAlgebra.sumMap_mul_left ops laws _ blocks _
          _ = K.mul (TargetPolynomial.power ops.toOps producerBeta
                (BlockNcDomain.laneIndex lane).val)
              (BooleanReproduction.equalityWeighted ops oldBlock (fun block =>
                authoritativeRunningValueAt covers data weights {
                  block := block.toCubePoint ops
                  lane := lane.toCubePoint ops })) := by
            rfl
    _ = K.mul batchWeight
        (authoritativeRunningProjection covers data weights
          producerBeta oldBlock) := by
        unfold authoritativeRunningProjection
        apply congrArg (K.mul batchWeight)
        apply FiniteSumAlgebra.sumMap_congr
        intro lane _
        rw [authoritativeRunningValueAt_block_reproduce
          covers data weights oldBlock lane]

/-- Exact initial-sum formula for the production-native combined polynomial.
No nonzero premise is used: when `batchWeight = 0`, the second summand is
literally zero rather than cancelled. -/
theorem combinedHypercubeSum_eq_ordinary_add_weightedProjection
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Mixing.Coins domain)
    (weights : RunningWeights shape)
    (producerBeta batchWeight : K)
    (oldBlock : CubePoint K domain.blockVariables) :
    combinedHypercubeSum covers data coins weights producerBeta batchWeight
        oldBlock =
      K.add (InitialSum.hypercubeSum covers data coins)
        (K.mul batchWeight
          (authoritativeRunningProjection covers data weights
            producerBeta oldBlock)) := by
  unfold combinedHypercubeSum combinedAtPoint
  calc
    FiniteSumAlgebra.sumMap ops
        (BooleanVertex.all domain.blockVariables) (fun block =>
          FiniteSumAlgebra.sumMap ops
            (BooleanVertex.all domain.laneVariables) (fun lane =>
              K.add
                (Mixing.qAtPoint covers data coins {
                  block := block.toCubePoint ops
                  lane := lane.toCubePoint ops })
                (delayedAtPoint covers data weights producerBeta batchWeight
                  oldBlock {
                    block := block.toCubePoint ops
                    lane := lane.toCubePoint ops }))) =
      FiniteSumAlgebra.sumMap ops
        (BooleanVertex.all domain.blockVariables) (fun block =>
          K.add
            (FiniteSumAlgebra.sumMap ops
              (BooleanVertex.all domain.laneVariables) (fun lane =>
                Mixing.qAtPoint covers data coins {
                  block := block.toCubePoint ops
                  lane := lane.toCubePoint ops }))
            (FiniteSumAlgebra.sumMap ops
              (BooleanVertex.all domain.laneVariables) (fun lane =>
                delayedAtPoint covers data weights producerBeta batchWeight
                  oldBlock {
                    block := block.toCubePoint ops
                    lane := lane.toCubePoint ops }))) := by
        apply FiniteSumAlgebra.sumMap_congr
        intro block _
        exact FiniteSumAlgebra.sumMap_add ops laws
          (BooleanVertex.all domain.laneVariables) _ _
    _ = K.add
        (FiniteSumAlgebra.sumMap ops
          (BooleanVertex.all domain.blockVariables) (fun block =>
            FiniteSumAlgebra.sumMap ops
              (BooleanVertex.all domain.laneVariables) (fun lane =>
                Mixing.qAtPoint covers data coins {
                  block := block.toCubePoint ops
                  lane := lane.toCubePoint ops })))
        (FiniteSumAlgebra.sumMap ops
          (BooleanVertex.all domain.blockVariables) (fun block =>
            FiniteSumAlgebra.sumMap ops
              (BooleanVertex.all domain.laneVariables) (fun lane =>
                delayedAtPoint covers data weights producerBeta batchWeight
                  oldBlock {
                    block := block.toCubePoint ops
                    lane := lane.toCubePoint ops }))) :=
      FiniteSumAlgebra.sumMap_add ops laws
        (BooleanVertex.all domain.blockVariables) _ _
    _ = K.add (InitialSum.hypercubeSum covers data coins)
        (delayedHypercubeSum covers data weights producerBeta batchWeight
          oldBlock) := by
      rfl
    _ = K.add (InitialSum.hypercubeSum covers data coins)
        (K.mul batchWeight
          (authoritativeRunningProjection covers data weights
            producerBeta oldBlock)) := by
      rw [delayedHypercubeSum_eq_weightedProjection]

/-! ## Terminal and zero-weight formulas -/

/-- Exact delayed terminal scalar at the final block×lane point. -/
def delayedTerminalRhs
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (weights : RunningWeights shape)
    (producerBeta batchWeight : K)
    (oldBlock : CubePoint K domain.blockVariables)
    (point : Point domain) : K :=
  K.mul batchWeight
    (K.mul
      (SumCheckTruthPath.pointEquality ops point.block oldBlock)
      (K.mul (betaPowerSelector producerBeta point.lane)
        (authoritativeRunningValueAt covers data weights point)))

theorem delayedAtPoint_eq_terminalRhs
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (weights : RunningWeights shape)
    (producerBeta batchWeight : K)
    (oldBlock : CubePoint K domain.blockVariables)
    (point : Point domain) :
    delayedAtPoint covers data weights producerBeta batchWeight
        oldBlock point =
      delayedTerminalRhs covers data weights producerBeta batchWeight
        oldBlock point := by
  rfl

/-- Exact combined verifier terminal formula. The ordinary source-binding
premise remains explicit and is never inferred from a terminal scalar. -/
def terminalFromMessage
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (message : Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.Claims shape)
    (coins : Mixing.Coins domain)
    (covers : domain.Covers shape)
    (data : Data shape)
    (weights : RunningWeights shape)
    (producerBeta batchWeight : K)
    (oldBlock : CubePoint K domain.blockVariables)
    (point : Point domain) : K :=
  K.add (Terminal.terminalFromMessage message coins point)
    (delayedTerminalRhs covers data weights producerBeta batchWeight
      oldBlock point)

theorem combinedAtPoint_eq_terminalFromMessage_of_bound
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Mixing.Coins domain)
    (weights : RunningWeights shape)
    (producerBeta batchWeight : K)
    (oldBlock : CubePoint K domain.blockVariables)
    (point : Point domain)
    (message : Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.Claims shape)
    (bound : Terminal.PackedYZcolBoundAtBlock
      covers data point.block message) :
    combinedAtPoint covers data coins weights producerBeta batchWeight
        oldBlock point =
      terminalFromMessage message coins covers data weights producerBeta
        batchWeight oldBlock point := by
  unfold combinedAtPoint terminalFromMessage
  rw [Terminal.terminal_eq_qAtPoint_of_bound
    covers data coins point message bound]
  rw [delayedAtPoint_eq_terminalRhs]

/-- Zero batch weight erases the delayed summand; no later theorem may cancel
this factor without retaining the corresponding root branch. -/
theorem delayedAtPoint_eq_zero_of_batchWeight_eq_zero
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (weights : RunningWeights shape)
    (producerBeta batchWeight : K)
    (oldBlock : CubePoint K domain.blockVariables)
    (point : Point domain)
    (batchWeightZero : batchWeight = K.zero) :
    delayedAtPoint covers data weights producerBeta batchWeight
        oldBlock point = K.zero := by
  subst batchWeight
  unfold delayedAtPoint
  exact calc
    K.mul K.zero
        (K.mul (SumCheckTruthPath.pointEquality ops point.block oldBlock)
          (K.mul (betaPowerSelector producerBeta point.lane)
            (authoritativeRunningValueAt covers data weights point))) =
      K.mul
        (K.mul (SumCheckTruthPath.pointEquality ops point.block oldBlock)
          (K.mul (betaPowerSelector producerBeta point.lane)
            (authoritativeRunningValueAt covers data weights point)))
        K.zero := laws.mul_comm _ _
    _ = K.zero := laws.mul_zero _

theorem combinedAtPoint_eq_ordinary_of_batchWeight_eq_zero
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Mixing.Coins domain)
    (weights : RunningWeights shape)
    (producerBeta batchWeight : K)
    (oldBlock : CubePoint K domain.blockVariables)
    (point : Point domain)
    (batchWeightZero : batchWeight = K.zero) :
    combinedAtPoint covers data coins weights producerBeta batchWeight
        oldBlock point =
      Mixing.qAtPoint covers data coins point := by
  unfold combinedAtPoint
  rw [delayedAtPoint_eq_zero_of_batchWeight_eq_zero
    covers data weights producerBeta batchWeight oldBlock point
    batchWeightZero]
  exact laws.add_zero _

/-! ## Degree closure -/

private theorem authoritativeRunningValueAt_block_affine
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (weights : RunningWeights shape)
    (lane : CubePoint K domain.laneVariables)
    (before after : List K)
    (length : before.length + 1 + after.length = domain.blockVariables) :
    Represents 1 (fun point =>
      authoritativeRunningValueAt covers data weights {
        block := cubeSlice before after length point
        lane := lane }) := by
  unfold authoritativeRunningValueAt
  apply polynomial_sum_exists
  intro running _
  exact sourceValueAt_block_affine covers data
    (Data.runningIndex running) lane before after length

private theorem authoritativeRunningValueAt_lane_affine
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (weights : RunningWeights shape)
    (block : CubePoint K domain.blockVariables)
    (before after : List K)
    (length : before.length + 1 + after.length = domain.laneVariables) :
    Represents 1 (fun point =>
      authoritativeRunningValueAt covers data weights {
        block := block
        lane := cubeSlice before after length point }) := by
  unfold authoritativeRunningValueAt
  apply polynomial_sum_exists
  intro running _
  exact sourceValueAt_lane_affine covers data
    (Data.runningIndex running) block before after length

private theorem blockSelector_affine
    {domain : BlockNcDomain}
    (oldBlock : CubePoint K domain.blockVariables)
    (before after : List K)
    (length : before.length + 1 + after.length = domain.blockVariables) :
    Represents 1 (fun point =>
      SumCheckTruthPath.pointEquality ops
        (cubeSlice before after length point) oldBlock) := by
  unfold SumCheckTruthPath.pointEquality
  exact pointEqualityCoordinates_affine before after
    oldBlock.coordinates (by rw [oldBlock.dimension]; exact length)

private theorem betaPowerSelector_lane_affine
    {domain : BlockNcDomain}
    (producerBeta : K)
    (before after : List K)
    (length : before.length + 1 + after.length = domain.laneVariables) :
    Represents 1 (fun point =>
      betaPowerSelector producerBeta
        (cubeSlice before after length point)) := by
  unfold betaPowerSelector BooleanTable.evaluate
  exact evaluateCoordinates_affine
    (BooleanTable.tabulate fun lane =>
      TargetPolynomial.power ops.toOps producerBeta
        (BlockNcDomain.laneIndex lane).val)
    before after length

/-- The production-native delayed term is quadratic in each block coordinate. -/
theorem delayedAtPoint_block_quadratic
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (weights : RunningWeights shape)
    (producerBeta batchWeight : K)
    (oldBlock : CubePoint K domain.blockVariables)
    (lane : CubePoint K domain.laneVariables)
    (before after : List K)
    (length : before.length + 1 + after.length = domain.blockVariables) :
    Represents 2 (fun point =>
      delayedAtPoint covers data weights producerBeta batchWeight oldBlock {
        block := cubeSlice before after length point
        lane := lane }) := by
  unfold delayedAtPoint
  have raw := authoritativeRunningValueAt_block_affine
    covers data weights lane before after length
  have scaledRaw := Represents.scale
    (betaPowerSelector producerBeta lane) raw
  exact Represents.scale batchWeight
    (Represents.mul
      (blockSelector_affine oldBlock before after length) scaledRaw)

/-- The production-native delayed term is quadratic in each lane coordinate. -/
theorem delayedAtPoint_lane_quadratic
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (weights : RunningWeights shape)
    (producerBeta batchWeight : K)
    (oldBlock : CubePoint K domain.blockVariables)
    (block : CubePoint K domain.blockVariables)
    (before after : List K)
    (length : before.length + 1 + after.length = domain.laneVariables) :
    Represents 2 (fun point =>
      delayedAtPoint covers data weights producerBeta batchWeight oldBlock {
        block := block
        lane := cubeSlice before after length point }) := by
  unfold delayedAtPoint
  have product := Represents.mul
    (betaPowerSelector_lane_affine producerBeta before after length)
    (authoritativeRunningValueAt_lane_affine
      covers data weights block before after length)
  exact Represents.scale batchWeight
    (Represents.scale
      (SumCheckTruthPath.pointEquality ops block oldBlock) product)

theorem combinedAtPoint_block_quartic
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Mixing.Coins domain)
    (weights : RunningWeights shape)
    (producerBeta batchWeight : K)
    (oldBlock : CubePoint K domain.blockVariables)
    (lane : CubePoint K domain.laneVariables)
    (before after : List K)
    (length : before.length + 1 + after.length = domain.blockVariables) :
    Represents ncSumcheckDegreeBound (fun point =>
      combinedAtPoint covers data coins weights producerBeta batchWeight
        oldBlock {
          block := cubeSlice before after length point
          lane := lane }) := by
  apply Represents.add
  · exact qAtPoint_block_quartic
      covers data coins lane before after length
  · exact Represents.widen
      (degree := 2) (target := ncSumcheckDegreeBound) (by decide)
      (delayedAtPoint_block_quadratic covers data weights producerBeta
        batchWeight oldBlock lane before after length)

theorem combinedAtPoint_lane_quartic
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Mixing.Coins domain)
    (weights : RunningWeights shape)
    (producerBeta batchWeight : K)
    (oldBlock : CubePoint K domain.blockVariables)
    (block : CubePoint K domain.blockVariables)
    (before after : List K)
    (length : before.length + 1 + after.length = domain.laneVariables) :
    Represents ncSumcheckDegreeBound (fun point =>
      combinedAtPoint covers data coins weights producerBeta batchWeight
        oldBlock {
          block := block
          lane := cubeSlice before after length point }) := by
  apply Represents.add
  · exact qAtPoint_lane_quartic
      covers data coins block before after length
  · exact Represents.widen
      (degree := 2) (target := ncSumcheckDegreeBound) (by decide)
      (delayedAtPoint_lane_quadratic covers data weights producerBeta
        batchWeight oldBlock block before after length)

/-! ## Flattened production SumCheck rounds -/

/-- Fail-closed block-then-lane evaluator for the combined polynomial. -/
def sumcheckPolynomial
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Mixing.Coins domain)
    (weights : RunningWeights shape)
    (producerBeta batchWeight : K)
    (oldBlock : CubePoint K domain.blockVariables)
    (coordinates : List K) : K :=
  if length : coordinates.length =
      domain.blockVariables + domain.laneVariables then
    combinedAtPoint covers data coins weights producerBeta batchWeight
      oldBlock (Point.ofCoordinates coordinates length)
  else
    K.zero

private theorem cubePoint_eq_of_coordinates_eq
    {variables : Nat}
    (left right : CubePoint K variables)
    (coordinates : left.coordinates = right.coordinates) :
    left = right := by
  cases left
  cases right
  simp_all

private theorem ofCoordinates_eq_blockSlice
    {domain : BlockNcDomain}
    (before after : List K)
    (beforeBlock : before.length < domain.blockVariables)
    (totalLength : before.length + 1 + after.length =
      domain.blockVariables + domain.laneVariables)
    (point : K) :
    let blockAfter := after.take
      (domain.blockVariables - before.length - 1)
    let laneCoordinates := after.drop
      (domain.blockVariables - before.length - 1)
    let blockLength : before.length + 1 + blockAfter.length =
        domain.blockVariables := by
      dsimp only [blockAfter]
      rw [List.length_take]
      omega
    let laneLength : laneCoordinates.length = domain.laneVariables := by
      dsimp only [laneCoordinates]
      rw [List.length_drop]
      omega
    Point.ofCoordinates (before ++ point :: after) (by
      simp only [List.length_append, List.length_cons]
      omega) = {
      block := cubeSlice before blockAfter blockLength point
      lane := { coordinates := laneCoordinates, dimension := laneLength } } := by
  dsimp only
  apply Point.ext
  · apply cubePoint_eq_of_coordinates_eq
    unfold Point.ofCoordinates cubeSlice
    simp only
    rw [List.take_append]
    rw [List.take_of_length_le (Nat.le_of_lt beforeBlock)]
    have remainingSucc :
        domain.blockVariables - before.length =
          (domain.blockVariables - before.length - 1) + 1 := by omega
    rw [remainingSucc]
    rfl
  · apply cubePoint_eq_of_coordinates_eq
    unfold Point.ofCoordinates
    simp only
    rw [List.drop_append]
    rw [List.drop_eq_nil_of_le (Nat.le_of_lt beforeBlock)]
    have remainingSucc :
        domain.blockVariables - before.length =
          (domain.blockVariables - before.length - 1) + 1 := by omega
    rw [remainingSucc]
    rfl

private theorem ofCoordinates_eq_laneSlice
    {domain : BlockNcDomain}
    (before after : List K)
    (blockBefore : domain.blockVariables ≤ before.length)
    (totalLength : before.length + 1 + after.length =
      domain.blockVariables + domain.laneVariables)
    (point : K) :
    let blockCoordinates := before.take domain.blockVariables
    let laneBefore := before.drop domain.blockVariables
    let blockLength : blockCoordinates.length = domain.blockVariables := by
      dsimp only [blockCoordinates]
      rw [List.length_take]
      omega
    let laneLength : laneBefore.length + 1 + after.length =
        domain.laneVariables := by
      dsimp only [laneBefore]
      rw [List.length_drop]
      omega
    Point.ofCoordinates (before ++ point :: after) (by
      simp only [List.length_append, List.length_cons]
      omega) = {
      block := { coordinates := blockCoordinates, dimension := blockLength }
      lane := cubeSlice laneBefore after laneLength point } := by
  dsimp only
  apply Point.ext
  · apply cubePoint_eq_of_coordinates_eq
    unfold Point.ofCoordinates
    simp only
    exact List.take_append_of_le_length blockBefore
  · apply cubePoint_eq_of_coordinates_eq
    unfold Point.ofCoordinates cubeSlice
    simp only
    exact List.drop_append_of_le_length blockBefore

theorem sumcheckPolynomial_slice_quartic
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Mixing.Coins domain)
    (weights : RunningWeights shape)
    (producerBeta batchWeight : K)
    (oldBlock : CubePoint K domain.blockVariables)
    (before after : List K)
    (length : before.length + 1 + after.length =
      domain.blockVariables + domain.laneVariables) :
    Represents ncSumcheckDegreeBound (fun point =>
      sumcheckPolynomial covers data coins weights producerBeta batchWeight
        oldBlock (before ++ point :: after)) := by
  by_cases beforeBlock : before.length < domain.blockVariables
  · let blockAfter := after.take
      (domain.blockVariables - before.length - 1)
    let laneCoordinates := after.drop
      (domain.blockVariables - before.length - 1)
    have blockLength : before.length + 1 + blockAfter.length =
        domain.blockVariables := by
      dsimp only [blockAfter]
      rw [List.length_take]
      omega
    have laneLength : laneCoordinates.length = domain.laneVariables := by
      dsimp only [laneCoordinates]
      rw [List.length_drop]
      omega
    let lane : CubePoint K domain.laneVariables := {
      coordinates := laneCoordinates
      dimension := laneLength }
    rcases combinedAtPoint_block_quartic covers data coins weights
      producerBeta batchWeight oldBlock lane before blockAfter blockLength with
      ⟨polynomial, represents⟩
    refine ⟨polynomial, ?_⟩
    intro point
    rw [represents]
    unfold sumcheckPolynomial
    dsimp only
    rw [dif_pos (by
      simp only [List.length_append, List.length_cons]
      omega)]
    rw [ofCoordinates_eq_blockSlice before after beforeBlock length]
  · have blockBefore : domain.blockVariables ≤ before.length :=
      Nat.le_of_not_gt beforeBlock
    let blockCoordinates := before.take domain.blockVariables
    let laneBefore := before.drop domain.blockVariables
    have blockLength : blockCoordinates.length = domain.blockVariables := by
      dsimp only [blockCoordinates]
      rw [List.length_take]
      omega
    have laneLength : laneBefore.length + 1 + after.length =
        domain.laneVariables := by
      dsimp only [laneBefore]
      rw [List.length_drop]
      omega
    let block : CubePoint K domain.blockVariables := {
      coordinates := blockCoordinates
      dimension := blockLength }
    rcases combinedAtPoint_lane_quartic covers data coins weights
      producerBeta batchWeight oldBlock block laneBefore after laneLength with
      ⟨polynomial, represents⟩
    refine ⟨polynomial, ?_⟩
    intro point
    rw [represents]
    unfold sumcheckPolynomial
    dsimp only
    rw [dif_pos (by
      simp only [List.length_append, List.length_cons]
      omega)]
    rw [ofCoordinates_eq_laneSlice before after blockBefore length]

theorem expectedRound_quartic
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Mixing.Coins domain)
    (weights : RunningWeights shape)
    (producerBeta batchWeight : K)
    (oldBlock : CubePoint K domain.blockVariables)
    (fixed : List K) (remaining : Nat)
    (length : fixed.length + 1 + remaining =
      domain.blockVariables + domain.laneVariables) :
    Represents ncSumcheckDegreeBound (fun point =>
      Nightstream.SuperNeo.SumCheck.Finite.HypercubeTruth.sumCompletions
        ops.toOps
        (sumcheckPolynomial covers data coins weights producerBeta
          batchWeight oldBlock)
        (fixed ++ [point]) remaining) := by
  apply sumCompletions_represents
  intro vertex
  have suffixLength :
      (SumCheckTruthPath.VertexEncoding.fieldCoordinates ops vertex).length =
        remaining :=
    SumCheckTruthPath.VertexEncoding.fieldCoordinates_length ops vertex
  simpa only [List.append_assoc, List.singleton_append] using
    sumcheckPolynomial_slice_quartic covers data coins weights producerBeta
      batchWeight oldBlock fixed
      (SumCheckTruthPath.VertexEncoding.fieldCoordinates ops vertex) (by
        rw [suffixLength]
        exact length)

theorem expectedRound_has_five_coefficients
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Mixing.Coins domain)
    (weights : RunningWeights shape)
    (producerBeta batchWeight : K)
    (oldBlock : CubePoint K domain.blockVariables)
    (fixed : List K) (remaining : Nat)
    (length : fixed.length + 1 + remaining =
      domain.blockVariables + domain.laneVariables) :
    ∃ message : Nightstream.SuperNeo.SumCheck.Finite.Message K,
      message.coefficients.length = ncMessageWidth ∧
      message.degreeUpperBound = ncSumcheckDegreeBound ∧
      ∀ point, message.evaluate ops.toOps point =
        Nightstream.SuperNeo.SumCheck.Finite.HypercubeTruth.sumCompletions
          ops.toOps
          (sumcheckPolynomial covers data coins weights producerBeta
            batchWeight oldBlock)
          (fixed ++ [point]) remaining := by
  simpa only [ncMessageWidth] using
    Represents.message_shape
      (expectedRound_quartic covers data coins weights producerBeta
        batchWeight oldBlock fixed remaining length)

end Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual.CombinedNc
