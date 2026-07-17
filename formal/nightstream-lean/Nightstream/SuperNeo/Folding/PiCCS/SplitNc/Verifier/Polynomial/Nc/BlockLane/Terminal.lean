import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane.Mixing
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane.OutputBridge
import Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.Types

/-!
Packed-output terminal for the canonical Split-NC block×lane polynomial.

Assurance tier: model-level.

Owns: semantic binding of every active `yZcol` lane at the final SumCheck
block point, zero extension across the padded lane domain, lane interpolation,
the strict cubic after interpolation, paper-relative source mixing, and exact
terminal equality with the independent polynomial.

Does not own: a commitment/opening protocol that establishes the binding
premise, transcript derivation of the final point, SumCheck checking, Rust,
R1CS, costs, or row removal.

Emits constraints: no.

Authority boundary: `Claims.yZcol` is untrusted. `PackedYZcolBoundAtBlock` is
an explicit semantic premise until a verifier-driven opening or proof derives
it from authoritative sources. The binding point is `point.block`, the final
NC SumCheck block coordinate, not the selector target `coins.betaBlock`.
Terminal scalar equality is only a consequence of full lane binding and is
never used to infer that binding.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_ccs.nc.block_lane.terminal.output.binding` | every active message lane equals the canonical packed projection at `blockPrime` | explicit proof boundary | `PackedYZcolBoundAtBlock` |
| `nifs.pi_ccs.nc.block_lane.terminal.output.live` | active padded leaves read the bound message | checked payload | `paddedYZcol_live` |
| `nifs.pi_ccs.nc.block_lane.terminal.output.padding` | every lane after 53 is computed zero | computed | `paddedYZcol_padding` |
| `nifs.pi_ccs.nc.block_lane.terminal.output.table` | the complete padded message table equals the source block MLE table | derived | `laneTable_eq_laneTableAtBlock_of_bound` |
| `nifs.pi_ccs.nc.block_lane.terminal.output.mle` | interpolate the bound lane table at `lanePrime` | computed then derived | `valueAt_eq_sourceValueAt_of_bound` |
| `nifs.pi_ccs.nc.block_lane.terminal.range` | apply the strict cubic only after interpolation | computed | `rangeAt` |
| `nifs.pi_ccs.nc.block_lane.terminal.mixing` | source `i` has exactly weight `gamma^i` | computed | `mixedRangeAt` |
| `nifs.pi_ccs.nc.block_lane.terminal.equality` | selectors and bound source mix reproduce semantic `qAtPoint` | derived | `terminal_eq_qAtPoint_of_bound` |
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane.Terminal

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.OutputClaims
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane
open Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism

private abbrev ops := ConcreteCarrier.extensionOps

/-- Full active-lane binding at one verifier-owned final block point. This is
a semantic premise, not a digest or message authority claim. -/
def PackedYZcolBoundAtBlock
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (blockPrime : CubePoint K domain.blockVariables)
    (message : Claims shape) : Prop :=
  ∀ source lane,
    message.yZcol source lane =
      PackedBlockAction.packedYZcol covers
        (data.assignment source) blockPrime lane

/-- Zero-extend the 54 active message lanes to the complete padded lane
domain. No padded value is supplied by the prover. -/
def paddedYZcol
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (message : Claims shape)
    (source : Fin shape.sourceCount)
    (lane : Fin domain.laneCount) : K :=
  if live : lane.val < ringDegree then
    message.yZcol source ⟨lane.val, live⟩
  else
    K.zero

/-- An active lane is read without reindexing. -/
@[simp] theorem paddedYZcol_live
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (message : Claims shape)
    (source : Fin shape.sourceCount)
    (lane : Fin ringDegree) :
    paddedYZcol (domain := domain) message source
        (domain.phi81Lane covers lane) =
      message.yZcol source lane := by
  simp [paddedYZcol, BlockNcDomain.phi81Lane]

/-- Every padded lane is computed zero. -/
theorem paddedYZcol_padding
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (message : Claims shape)
    (source : Fin shape.sourceCount)
    (lane : Fin domain.laneCount)
    (padding : ringDegree ≤ lane.val) :
    paddedYZcol message source lane = K.zero := by
  simp [paddedYZcol, Nat.not_lt.mpr padding]

/-- Complete padded lane table from one output-message source. -/
def laneTable
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (message : Claims shape)
    (source : Fin shape.sourceCount) :
    BooleanTable K domain.laneVariables :=
  BooleanTable.tabulate fun lane =>
    paddedYZcol message source (BlockNcDomain.laneIndex lane)

/-- Evaluate the padded message lane table at the final lane point. -/
def valueAt
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (message : Claims shape)
    (source : Fin shape.sourceCount)
    (lanePrime : CubePoint K domain.laneVariables) : K :=
  (laneTable (domain := domain) message source).evaluate ops lanePrime

/-- Apply strict-`b = 2` only after lane interpolation. -/
def rangeAt
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (message : Claims shape)
    (source : Fin shape.sourceCount)
    (lanePrime : CubePoint K domain.laneVariables) : K :=
  let value := valueAt (domain := domain) message source lanePrime
  K.mul (K.mul (K.add value (K.embed 1)) value)
    (K.sub value (K.embed 1))

/-- Paper-relative gamma compression of output-derived source cubics. -/
def mixedRangeAt
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (message : Claims shape)
    (coins : Mixing.Coins domain)
    (lanePrime : CubePoint K domain.laneVariables) : K :=
  FiniteSumAlgebra.sumMap ops
    (canonicalFinIndices shape.sourceCount) fun source =>
      SignedJointIdentity.gammaTerm ops coins.gamma
        (Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.Mixing.sourceExponent
          shape .paperNc source)
        (rangeAt (domain := domain) message source lanePrime)

/-- Message-derived terminal at the typed final block×lane point. -/
def terminalFromMessage
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (message : Claims shape)
    (coins : Mixing.Coins domain)
    (point : Point domain) : K :=
  K.mul
    (K.mul
      (SumCheckTruthPath.pointEquality ops point.block coins.betaBlock)
      (SumCheckTruthPath.pointEquality ops point.lane coins.betaA))
    (mixedRangeAt message coins point.lane)

/-- Full active-lane binding plus computed padding makes each padded message
leaf equal the source-derived block MLE leaf. -/
theorem paddedYZcol_eq_blockValueAt_of_bound
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (blockPrime : CubePoint K domain.blockVariables)
    (message : Claims shape)
    (bound : PackedYZcolBoundAtBlock covers data blockPrime message)
    (source : Fin shape.sourceCount)
    (lane : Fin domain.laneCount) :
    paddedYZcol message source lane =
      SourceProjection.blockValueAt covers data source blockPrime
        (BlockNcDomain.laneVertex lane) := by
  by_cases live : lane.val < ringDegree
  · let active : Fin ringDegree := ⟨lane.val, live⟩
    have activeLane : domain.phi81Lane covers active = lane := by
      apply Fin.ext
      rfl
    rw [show paddedYZcol message source lane =
        message.yZcol source active by
      simp [paddedYZcol, live, active]]
    rw [bound source active]
    simpa [activeLane] using
      OutputBridge.packedYZcol_lane_eq_blockValueAt
        covers data source blockPrime active
  · have padding : ringDegree ≤ lane.val := Nat.le_of_not_gt live
    rw [paddedYZcol_padding message source lane padding]
    rw [SourceProjection.blockValueAt_lane_padding
      covers data source blockPrime lane padding]

/-- The complete padded message table equals the source table at the same
final block point. -/
theorem laneTable_eq_laneTableAtBlock_of_bound
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (blockPrime : CubePoint K domain.blockVariables)
    (message : Claims shape)
    (bound : PackedYZcolBoundAtBlock covers data blockPrime message)
    (source : Fin shape.sourceCount) :
    laneTable (domain := domain) message source =
      SourceProjection.laneTableAtBlock covers data source blockPrime := by
  unfold laneTable SourceProjection.laneTableAtBlock
  apply congrArg BooleanTable.tabulate
  funext lane
  simpa using paddedYZcol_eq_blockValueAt_of_bound
    covers data blockPrime message bound source
      (BlockNcDomain.laneIndex lane)

/-- Bound output interpolation equals the independent nested source MLE
before applying the cubic. -/
theorem valueAt_eq_sourceValueAt_of_bound
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (point : Point domain)
    (message : Claims shape)
    (bound : PackedYZcolBoundAtBlock covers data point.block message)
    (source : Fin shape.sourceCount) :
    valueAt (domain := domain) message source point.lane =
      SourceProjection.sourceValueAt covers data source point := by
  unfold valueAt SourceProjection.sourceValueAt
  rw [laneTable_eq_laneTableAtBlock_of_bound
    covers data point.block message bound source]

/-- The strict cubic therefore agrees source-by-source. -/
theorem rangeAt_eq_rangeValueAt_of_bound
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (point : Point domain)
    (message : Claims shape)
    (bound : PackedYZcolBoundAtBlock covers data point.block message)
    (source : Fin shape.sourceCount) :
    rangeAt (domain := domain) message source point.lane =
      SourceProjection.rangeValueAt covers data source point := by
  unfold rangeAt SourceProjection.rangeValueAt
  rw [valueAt_eq_sourceValueAt_of_bound
    covers data point message bound source]

/-- Full source binding makes output mixing equal the independent semantic
mix at the same final point. -/
theorem mixedRangeAt_eq_semantic_of_bound
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Mixing.Coins domain)
    (point : Point domain)
    (message : Claims shape)
    (bound : PackedYZcolBoundAtBlock covers data point.block message) :
    mixedRangeAt message coins point.lane =
      Mixing.mixedRangeAt covers data coins point := by
  unfold mixedRangeAt Mixing.mixedRangeAt
  apply FiniteSumAlgebra.sumMap_congr
  intro source _
  rw [rangeAt_eq_rangeValueAt_of_bound
    covers data point message bound source]

/-- Exact one-way terminal theorem: full active-lane source binding implies
message terminal equality with the independent NC polynomial. -/
theorem terminal_eq_qAtPoint_of_bound
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Mixing.Coins domain)
    (point : Point domain)
    (message : Claims shape)
    (bound : PackedYZcolBoundAtBlock covers data point.block message) :
    terminalFromMessage message coins point =
      Mixing.qAtPoint covers data coins point := by
  unfold terminalFromMessage Mixing.qAtPoint
  rw [mixedRangeAt_eq_semantic_of_bound
    covers data coins point message bound]

end Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane.Terminal
