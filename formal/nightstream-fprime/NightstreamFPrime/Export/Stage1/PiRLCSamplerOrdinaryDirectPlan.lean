import NightstreamFPrime.Export.Stage1.DirectPiDECPrefixPlan
import NightstreamFPrime.Export.Stage1.PiRLCSamplerOrdinaryRetainedGeometry

/-!
Owns the executable source resolver and direct 14-matrix plan for the
canonical PiRLC sampler ordinary rows.

The resolver recognizes only the exact Poseidon2 endpoints, digest-lane
logical and fresh intervals, and final First54 selector outputs proved by the
source module. This module does not close PiRLC conformance.
-/

namespace NightstreamFPrime.Export.Stage1.PiRLCSamplerOrdinaryDirectPlan

open NightstreamFPrime.Circuit
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.PiRLC.v1_1
open NightstreamFPrime.Gadgets.Sampling
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra
open PiRLCSamplerOrdinaryRetainedBlocks

/-- One exact retained source used by the sampler ordinary rows. -/
inductive Location where
  | poseidon (descriptor : Lane) (sourceLane : Fin 4)
  | logical (descriptor : Lane) (position : Fin logicalCountPerLane)
  | fresh (descriptor : Lane) (position : Fin freshCountPerLane)
  | selector (source : Fin sourceCount)

namespace Location

def sourceColumn : Location → Nat
  | .poseidon descriptor sourceLane =>
      PiRLCSamplerOrdinaryDirectSource.poseidonSource descriptor.source.val
        descriptor.round.val sourceLane
  | .logical descriptor position => logicalSource descriptor position
  | .fresh descriptor position => freshSource descriptor position
  | .selector source =>
      PiRLCSamplerOrdinaryDirectSource.selectorSource source.val

theorem sourceSupport (location : Location) :
    PiRLCSamplerOrdinaryDirectSource.Source location.sourceColumn := by
  cases location with
  | poseidon descriptor sourceLane =>
      exact PiRLCSamplerOrdinaryDirectSource.Source.poseidon
        descriptor.source.val descriptor.round.val sourceLane
        descriptor.source.isLt descriptor.round.isLt
  | logical descriptor position =>
      exact PiRLCSamplerOrdinaryDirectSource.Source.logical
        descriptor.source.val descriptor.round.val descriptor.lane.val
        position.val descriptor.source.isLt descriptor.round.isLt
        descriptor.lane.isLt position.isLt
  | fresh descriptor position =>
      exact PiRLCSamplerOrdinaryDirectSource.Source.fresh
        descriptor.source.val descriptor.round.val descriptor.lane.val
        position.val descriptor.source.isLt descriptor.round.isLt
        descriptor.lane.isLt position.isLt
  | selector source =>
      exact PiRLCSamplerOrdinaryDirectSource.Source.selector source.val
        source.isLt

end Location

private def logicalSourceIndex (column : Nat) : Nat :=
  (column - PiRLCStarts.samplerLogicalStart) / 15504

private def logicalSourceOffset (column : Nat) : Nat :=
  (column - PiRLCStarts.samplerLogicalStart) % 15504

private def logicalWindowOffset (column : Nat) : Nat :=
  logicalSourceOffset column - 592

private def logicalRoundIndex (column : Nat) : Nat :=
  logicalWindowOffset column / 992

private def logicalRoundOffset (column : Nat) : Nat :=
  logicalWindowOffset column % 992

private def logicalLaneIndex (column : Nat) : Nat :=
  logicalRoundOffset column / 100

private def logicalPosition (column : Nat) : Nat :=
  logicalRoundOffset column % 100

private def freshSourceIndex (column : Nat) : Nat :=
  (column - PiRLCStarts.samplerFreshStart) / 43743

private def freshSourceOffset (column : Nat) : Nat :=
  (column - PiRLCStarts.samplerFreshStart) % 43743

private def freshRoundIndex (column : Nat) : Nat :=
  freshSourceOffset column / 1212

private def freshRoundOffset (column : Nat) : Nat :=
  freshSourceOffset column % 1212

private def freshLaneIndex (column : Nat) : Nat :=
  freshRoundOffset column / 303

private def freshPosition (column : Nat) : Nat :=
  freshRoundOffset column % 303

private def logicalCandidate (column : Nat) : Location :=
  .logical
    { source := ⟨logicalSourceIndex column % sourceCount,
        Nat.mod_lt _ (by decide)⟩
      round := ⟨logicalRoundIndex column % roundCount,
        Nat.mod_lt _ (by decide)⟩
      lane := ⟨logicalLaneIndex column % laneCount,
        Nat.mod_lt _ (by decide)⟩ }
    ⟨logicalPosition column, Nat.mod_lt _ (by decide)⟩

private def freshCandidate (column : Nat) : Location :=
  .fresh
    { source := ⟨freshSourceIndex column % sourceCount,
        Nat.mod_lt _ (by decide)⟩
      round := ⟨freshRoundIndex column % roundCount,
        Nat.mod_lt _ (by decide)⟩
      lane := ⟨freshLaneIndex column % laneCount,
        Nat.mod_lt _ (by decide)⟩ }
    ⟨freshPosition column, Nat.mod_lt _ (by decide)⟩

private def poseidonEntryCandidate (column : Nat) : Location :=
  .poseidon
    { source := ⟨logicalSourceIndex column % sourceCount,
        Nat.mod_lt _ (by decide)⟩
      round := ⟨0, by decide⟩
      lane := ⟨0, by decide⟩ }
    ⟨(logicalSourceOffset column - 584) % 4,
      Nat.mod_lt _ (by decide)⟩

private def poseidonWindowCandidate (column : Nat) : Location :=
  let previousOffset := logicalSourceOffset column - 1576
  let previous := previousOffset / 992
  .poseidon
    { source := ⟨logicalSourceIndex column % sourceCount,
        Nat.mod_lt _ (by decide)⟩
      round := ⟨(previous + 1) % roundCount, Nat.mod_lt _ (by decide)⟩
      lane := ⟨0, by decide⟩ }
    ⟨(previousOffset % 992) % 4, Nat.mod_lt _ (by decide)⟩

/-- Entry outputs occupy offsets 584 through 587. Window outputs start at
offset 1576, so this branch is exact and leaves no candidate precedence. -/
private def poseidonCandidate (column : Nat) : Location :=
  if logicalSourceOffset column < 592 then
    poseidonEntryCandidate column
  else
    poseidonWindowCandidate column

private def selectorCandidate (column : Nat) : Location :=
  .selector ⟨logicalSourceIndex column % sourceCount,
    Nat.mod_lt _ (by decide)⟩

private def poseidonOwned (column : Nat) : Prop :=
  let offset := logicalSourceOffset column
  584 ≤ offset ∧ (offset - 584) / 992 < roundCount ∧
    (offset - 584) % 992 < 4

private def logicalOwned (column : Nat) : Prop :=
  let offset := logicalSourceOffset column
  592 ≤ offset ∧ (offset - 592) / 992 < roundCount ∧
    (offset - 592) % 992 < 400

private instance poseidonOwnedDecidable (column : Nat) :
    Decidable (poseidonOwned column) := by
  unfold poseidonOwned
  infer_instance

private instance logicalOwnedDecidable (column : Nat) :
    Decidable (logicalOwned column) := by
  unfold logicalOwned
  infer_instance

private def exactCandidate (column : Nat) (candidate : Location) :
    Option Location :=
  if candidate.sourceColumn = column then
    some candidate
  else
    none

/-- Constant-time, fail-closed source classifier. Final exact equality checks
prevent a quotient or remainder from accepting a gap between owned ranges. -/
def classifySource (column : Nat) : Option Location :=
  if PiRLCStarts.samplerFreshStart ≤ column then
    exactCandidate column (freshCandidate column)
  else if poseidonOwned column then
    exactCandidate column (poseidonCandidate column)
  else if logicalOwned column then
    exactCandidate column (logicalCandidate column)
  else
    exactCandidate column (selectorCandidate column)

private theorem exactCandidate_self (location : Location) :
    exactCandidate location.sourceColumn location =
      some location := by
  simp [exactCandidate]

private theorem exactCandidate_sound {column : Nat} {candidate location : Location}
    (found : exactCandidate column candidate = some location) :
    location.sourceColumn = column := by
  unfold exactCandidate at found
  split at found
  · rename_i owns
    have same : candidate = location := Option.some.inj found
    rw [← same]
    exact owns
  · contradiction

/-- A successful classification always owns the exact requested source
column. The equality test is authority; quotient and remainder candidates
alone never authorize a source. -/
theorem classifySource_sound {column : Nat} {location : Location}
    (found : classifySource column = some location) :
    location.sourceColumn = column := by
  unfold classifySource at found
  split at found
  · exact exactCandidate_sound found
  · split at found
    · exact exactCandidate_sound found
    · split at found
      · exact exactCandidate_sound found
      · exact exactCandidate_sound found

private theorem stride_div (outer offset stride : Nat) (stridePositive : 0 < stride)
    (offsetLt : offset < stride) :
    (outer * stride + offset) / stride = outer := by
  rw [Nat.mul_comm outer stride, Nat.mul_add_div stridePositive]
  rw [Nat.div_eq_of_lt offsetLt, Nat.add_zero]

private theorem stride_mod (outer offset stride : Nat) (offsetLt : offset < stride) :
    (outer * stride + offset) % stride = offset := by
  exact Nat.mul_add_mod_of_lt offsetLt

private theorem logicalSourceIndex_at (source offset : Nat)
    (offsetLt : offset < 15504) :
    logicalSourceIndex
        (PiRLCStarts.samplerLogicalStart + source * 15504 + offset) =
      source := by
  unfold logicalSourceIndex
  rw [show
      PiRLCStarts.samplerLogicalStart + source * 15504 + offset -
          PiRLCStarts.samplerLogicalStart = source * 15504 + offset by
    omega]
  exact stride_div source offset 15504 (by decide) offsetLt

private theorem logicalSourceOffset_at (source offset : Nat)
    (offsetLt : offset < 15504) :
    logicalSourceOffset
        (PiRLCStarts.samplerLogicalStart + source * 15504 + offset) =
      offset := by
  unfold logicalSourceOffset
  rw [show
      PiRLCStarts.samplerLogicalStart + source * 15504 + offset -
          PiRLCStarts.samplerLogicalStart = source * 15504 + offset by
    omega]
  exact stride_mod source offset 15504 offsetLt

private theorem freshSourceIndex_at (source offset : Nat)
    (offsetLt : offset < 43743) :
    freshSourceIndex
        (PiRLCStarts.samplerFreshStart + source * 43743 + offset) =
      source := by
  unfold freshSourceIndex
  rw [show
      PiRLCStarts.samplerFreshStart + source * 43743 + offset -
          PiRLCStarts.samplerFreshStart = source * 43743 + offset by
    omega]
  exact stride_div source offset 43743 (by decide) offsetLt

private theorem freshSourceOffset_at (source offset : Nat)
    (offsetLt : offset < 43743) :
    freshSourceOffset
        (PiRLCStarts.samplerFreshStart + source * 43743 + offset) =
      offset := by
  unfold freshSourceOffset
  rw [show
      PiRLCStarts.samplerFreshStart + source * 43743 + offset -
          PiRLCStarts.samplerFreshStart = source * 43743 + offset by
    omega]
  exact stride_mod source offset 43743 offsetLt

private theorem logicalCandidate_source
    (source round lane position : Nat)
    (sourceLt : source < sourceCount) (roundLt : round < roundCount)
    (laneLt : lane < laneCount) (positionLt : position < logicalCountPerLane) :
    logicalCandidate
        (PiRLCStarts.digestLaneLogicalStart source round lane + position) =
      .logical
        { source := ⟨source, sourceLt⟩
          round := ⟨round, roundLt⟩
          lane := ⟨lane, laneLt⟩ }
        ⟨position, positionLt⟩ := by
  let sourceOffset := 592 + round * 992 + lane * 100 + position
  let roundOffset := lane * 100 + position
  have roundLt8 : round < 8 := by
    simpa [roundCount] using roundLt
  have laneLt4 : lane < 4 := by
    simpa [laneCount] using laneLt
  have positionLt100 : position < 100 := by
    simpa [logicalCountPerLane] using positionLt
  have sourceOffsetLt : sourceOffset < 15504 := by
    dsimp [sourceOffset]
    omega
  have roundOffsetLt : roundOffset < 992 := by
    dsimp [roundOffset]
    omega
  have inputEq :
      PiRLCStarts.digestLaneLogicalStart source round lane + position =
        PiRLCStarts.samplerLogicalStart + source * 15504 + sourceOffset := by
    dsimp [sourceOffset]
    simp [PiRLCStarts.digestLaneLogicalStart,
      PiRLCStarts.windowLogicalStart,
      PiRLCStarts.samplerSourceLogicalStart]
    omega
  let column :=
    PiRLCStarts.digestLaneLogicalStart source round lane + position
  have sourceIndexEq : logicalSourceIndex column = source := by
    rw [show column = PiRLCStarts.samplerLogicalStart + source * 15504 +
        sourceOffset by exact inputEq]
    exact logicalSourceIndex_at source sourceOffset sourceOffsetLt
  have sourceOffsetEq : logicalSourceOffset column = sourceOffset := by
    rw [show column = PiRLCStarts.samplerLogicalStart + source * 15504 +
        sourceOffset by exact inputEq]
    exact logicalSourceOffset_at source sourceOffset sourceOffsetLt
  have windowOffsetEq :
      logicalWindowOffset column = round * 992 + roundOffset := by
    unfold logicalWindowOffset
    rw [sourceOffsetEq]
    dsimp [sourceOffset, roundOffset]
    omega
  have roundIndexEq : logicalRoundIndex column = round := by
    unfold logicalRoundIndex
    rw [windowOffsetEq]
    exact stride_div round roundOffset 992 (by decide) roundOffsetLt
  have roundOffsetEq : logicalRoundOffset column = roundOffset := by
    unfold logicalRoundOffset
    rw [windowOffsetEq]
    exact stride_mod round roundOffset 992 roundOffsetLt
  have laneIndexEq : logicalLaneIndex column = lane := by
    unfold logicalLaneIndex
    rw [roundOffsetEq]
    exact stride_div lane position 100 (by decide) positionLt100
  have positionEq : logicalPosition column = position := by
    unfold logicalPosition
    rw [roundOffsetEq]
    exact stride_mod lane position 100 positionLt100
  have sourceFinEq :
      (⟨logicalSourceIndex column % sourceCount,
        Nat.mod_lt _ (by decide)⟩ : Fin sourceCount) = ⟨source, sourceLt⟩ := by
    apply Fin.ext
    exact (congrArg (fun value => value % sourceCount) sourceIndexEq).trans
      (Nat.mod_eq_of_lt sourceLt)
  have roundFinEq :
      (⟨logicalRoundIndex column % roundCount,
        Nat.mod_lt _ (by decide)⟩ : Fin roundCount) = ⟨round, roundLt⟩ := by
    apply Fin.ext
    exact (congrArg (fun value => value % roundCount) roundIndexEq).trans
      (Nat.mod_eq_of_lt roundLt)
  have laneFinEq :
      (⟨logicalLaneIndex column % laneCount,
        Nat.mod_lt _ (by decide)⟩ : Fin laneCount) = ⟨lane, laneLt⟩ := by
    apply Fin.ext
    exact (congrArg (fun value => value % laneCount) laneIndexEq).trans
      (Nat.mod_eq_of_lt laneLt)
  have positionFinEq :
      (⟨logicalPosition column, Nat.mod_lt _ (by decide)⟩ :
          Fin logicalCountPerLane) = ⟨position, positionLt⟩ := by
    apply Fin.ext
    exact positionEq
  change logicalCandidate column = _
  unfold logicalCandidate
  apply congrArg₂ Location.logical
  · simpa only [sourceFinEq, roundFinEq, laneFinEq]
  · exact positionFinEq

private theorem freshCandidate_source
    (source round lane position : Nat)
    (sourceLt : source < sourceCount) (roundLt : round < roundCount)
    (laneLt : lane < laneCount) (positionLt : position < freshCountPerLane) :
    freshCandidate
        (PiRLCStarts.digestLaneFreshStart source round lane + position) =
      .fresh
        { source := ⟨source, sourceLt⟩
          round := ⟨round, roundLt⟩
          lane := ⟨lane, laneLt⟩ }
        ⟨position, positionLt⟩ := by
  let sourceOffset := round * 1212 + lane * 303 + position
  let roundOffset := lane * 303 + position
  have roundLt8 : round < 8 := by
    simpa [roundCount] using roundLt
  have laneLt4 : lane < 4 := by
    simpa [laneCount] using laneLt
  have positionLt303 : position < 303 := by
    simpa [freshCountPerLane] using positionLt
  have sourceOffsetLt : sourceOffset < 43743 := by
    dsimp [sourceOffset]
    omega
  have roundOffsetLt : roundOffset < 1212 := by
    have laneSuccLe : lane + 1 ≤ 4 := Nat.succ_le_of_lt laneLt4
    calc
      roundOffset = lane * 303 + position := by rfl
      _ < lane * 303 + 303 := Nat.add_lt_add_left positionLt303 _
      _ = (lane + 1) * 303 := by omega
      _ ≤ 4 * 303 := Nat.mul_le_mul_right 303 laneSuccLe
      _ = 1212 := by norm_num
  have sourceOffsetDecomp : sourceOffset = round * 1212 + roundOffset := by
    dsimp [sourceOffset, roundOffset]
    omega
  have inputEq :
      PiRLCStarts.digestLaneFreshStart source round lane + position =
        PiRLCStarts.samplerFreshStart + source * 43743 + sourceOffset := by
    dsimp [sourceOffset]
    simp [PiRLCStarts.digestLaneFreshStart, PiRLCStarts.windowFreshStart,
      PiRLCStarts.samplerSourceFreshStart]
    omega
  let column := PiRLCStarts.digestLaneFreshStart source round lane + position
  have sourceIndexEq : freshSourceIndex column = source := by
    rw [show column = PiRLCStarts.samplerFreshStart + source * 43743 +
        sourceOffset by exact inputEq]
    exact freshSourceIndex_at source sourceOffset sourceOffsetLt
  have sourceOffsetEq : freshSourceOffset column = sourceOffset := by
    rw [show column = PiRLCStarts.samplerFreshStart + source * 43743 +
        sourceOffset by exact inputEq]
    exact freshSourceOffset_at source sourceOffset sourceOffsetLt
  have roundIndexEq : freshRoundIndex column = round := by
    unfold freshRoundIndex
    rw [sourceOffsetEq, sourceOffsetDecomp]
    exact stride_div round roundOffset 1212 (by decide) roundOffsetLt
  have roundOffsetEq : freshRoundOffset column = roundOffset := by
    unfold freshRoundOffset
    rw [sourceOffsetEq, sourceOffsetDecomp]
    exact stride_mod round roundOffset 1212 roundOffsetLt
  have laneIndexEq : freshLaneIndex column = lane := by
    unfold freshLaneIndex
    rw [roundOffsetEq]
    exact stride_div lane position 303 (by decide) positionLt303
  have positionEq : freshPosition column = position := by
    unfold freshPosition
    rw [roundOffsetEq]
    exact stride_mod lane position 303 positionLt303
  have sourceFinEq :
      (⟨freshSourceIndex column % sourceCount,
        Nat.mod_lt _ (by decide)⟩ : Fin sourceCount) = ⟨source, sourceLt⟩ := by
    apply Fin.ext
    exact (congrArg (fun value => value % sourceCount) sourceIndexEq).trans
      (Nat.mod_eq_of_lt sourceLt)
  have roundFinEq :
      (⟨freshRoundIndex column % roundCount,
        Nat.mod_lt _ (by decide)⟩ : Fin roundCount) = ⟨round, roundLt⟩ := by
    apply Fin.ext
    exact (congrArg (fun value => value % roundCount) roundIndexEq).trans
      (Nat.mod_eq_of_lt roundLt)
  have laneFinEq :
      (⟨freshLaneIndex column % laneCount,
        Nat.mod_lt _ (by decide)⟩ : Fin laneCount) = ⟨lane, laneLt⟩ := by
    apply Fin.ext
    exact (congrArg (fun value => value % laneCount) laneIndexEq).trans
      (Nat.mod_eq_of_lt laneLt)
  have positionFinEq :
      (⟨freshPosition column, Nat.mod_lt _ (by decide)⟩ :
          Fin freshCountPerLane) = ⟨position, positionLt⟩ := by
    apply Fin.ext
    exact positionEq
  change freshCandidate column = _
  unfold freshCandidate
  apply congrArg₂ Location.fresh
  · simpa only [sourceFinEq, roundFinEq, laneFinEq]
  · exact positionFinEq

private theorem poseidonEntryCandidate_source (source : Nat) (lane : Fin 4)
    (sourceLt : source < sourceCount) :
    poseidonEntryCandidate
        (PiRLCSamplerOrdinaryDirectSource.poseidonSource source 0 lane) =
      .poseidon
        { source := ⟨source, sourceLt⟩
          round := ⟨0, by decide⟩
          lane := ⟨0, by decide⟩ }
        lane := by
  let sourceOffset := 584 + lane.val
  have sourceOffsetLt : sourceOffset < 15504 := by
    dsimp [sourceOffset]
    have laneLt := lane.isLt
    omega
  have inputEq :
      PiRLCSamplerOrdinaryDirectSource.poseidonSource source 0 lane =
        PiRLCStarts.samplerLogicalStart + source * 15504 + sourceOffset := by
    simp [sourceOffset, PiRLCSamplerOrdinaryDirectSource.poseidonSource,
      PiRLCStarts.samplerSourceLogicalStart]
    omega
  let column :=
    PiRLCSamplerOrdinaryDirectSource.poseidonSource source 0 lane
  have sourceIndexEq : logicalSourceIndex column = source := by
    rw [show column = PiRLCStarts.samplerLogicalStart + source * 15504 +
        sourceOffset by exact inputEq]
    exact logicalSourceIndex_at source sourceOffset sourceOffsetLt
  have sourceOffsetEq : logicalSourceOffset column = sourceOffset := by
    rw [show column = PiRLCStarts.samplerLogicalStart + source * 15504 +
        sourceOffset by exact inputEq]
    exact logicalSourceOffset_at source sourceOffset sourceOffsetLt
  have laneEq : (logicalSourceOffset column - 584) % 4 = lane.val := by
    rw [sourceOffsetEq]
    dsimp [sourceOffset]
    have laneLt := lane.isLt
    rw [show 584 + lane.val - 584 = lane.val by omega]
    exact Nat.mod_eq_of_lt laneLt
  have sourceFinEq :
      (⟨logicalSourceIndex column % sourceCount,
        Nat.mod_lt _ (by decide)⟩ : Fin sourceCount) = ⟨source, sourceLt⟩ := by
    apply Fin.ext
    exact (congrArg (fun value => value % sourceCount) sourceIndexEq).trans
      (Nat.mod_eq_of_lt sourceLt)
  have laneFinEq :
      (⟨(logicalSourceOffset column - 584) % 4,
        Nat.mod_lt _ (by decide)⟩ : Fin 4) = lane := by
    apply Fin.ext
    exact laneEq
  change poseidonEntryCandidate column = _
  unfold poseidonEntryCandidate
  apply congrArg₂ Location.poseidon
  · simpa only [sourceFinEq]
  · exact laneFinEq

private theorem poseidonWindowCandidate_source (source previous : Nat)
    (lane : Fin 4) (sourceLt : source < sourceCount)
    (roundLt : previous + 1 < roundCount) :
    poseidonWindowCandidate
        (PiRLCSamplerOrdinaryDirectSource.poseidonSource
          source (previous + 1) lane) =
      .poseidon
        { source := ⟨source, sourceLt⟩
          round := ⟨previous + 1, roundLt⟩
          lane := ⟨0, by decide⟩ }
        lane := by
  let sourceOffset := 1576 + previous * 992 + lane.val
  let previousOffset := previous * 992 + lane.val
  have roundLt8 : previous + 1 < 8 := by
    simpa [roundCount] using roundLt
  have sourceOffsetLt : sourceOffset < 15504 := by
    dsimp [sourceOffset]
    have laneLt := lane.isLt
    omega
  have laneLt992 : lane.val < 992 := lt_trans lane.isLt (by decide)
  have inputEq :
      PiRLCSamplerOrdinaryDirectSource.poseidonSource
          source (previous + 1) lane =
        PiRLCStarts.samplerLogicalStart + source * 15504 + sourceOffset := by
    dsimp [sourceOffset]
    simp [PiRLCSamplerOrdinaryDirectSource.poseidonSource,
      PiRLCStarts.samplerSourceLogicalStart, Sampler.windowOffset,
      Sampler.windowBase, SamplerChain.sourceOffset,
      DigestWindow.permutationOffset, Sampler.logicalPrivateCount,
      Sampler.entryPrivateCount, DigestWindow.logicalPrivateCount,
      DigestLane.logicalPrivateCount]
    omega
  let column := PiRLCSamplerOrdinaryDirectSource.poseidonSource
    source (previous + 1) lane
  have sourceIndexEq : logicalSourceIndex column = source := by
    rw [show column = PiRLCStarts.samplerLogicalStart + source * 15504 +
        sourceOffset by exact inputEq]
    exact logicalSourceIndex_at source sourceOffset sourceOffsetLt
  have sourceOffsetEq : logicalSourceOffset column = sourceOffset := by
    rw [show column = PiRLCStarts.samplerLogicalStart + source * 15504 +
        sourceOffset by exact inputEq]
    exact logicalSourceOffset_at source sourceOffset sourceOffsetLt
  have previousOffsetEq :
      logicalSourceOffset column - 1576 = previousOffset := by
    rw [sourceOffsetEq]
    dsimp [sourceOffset, previousOffset]
    omega
  have previousEq :
      (logicalSourceOffset column - 1576) / 992 = previous := by
    rw [previousOffsetEq]
    exact stride_div previous lane.val 992 (by decide) laneLt992
  have laneEq :
      ((logicalSourceOffset column - 1576) % 992) % 4 = lane.val := by
    rw [previousOffsetEq, stride_mod previous lane.val 992 laneLt992]
    exact Nat.mod_eq_of_lt lane.isLt
  have sourceFinEq :
      (⟨logicalSourceIndex column % sourceCount,
        Nat.mod_lt _ (by decide)⟩ : Fin sourceCount) = ⟨source, sourceLt⟩ := by
    apply Fin.ext
    exact (congrArg (fun value => value % sourceCount) sourceIndexEq).trans
      (Nat.mod_eq_of_lt sourceLt)
  have roundFinEq :
      (⟨((logicalSourceOffset column - 1576) / 992 + 1) % roundCount,
        Nat.mod_lt _ (by decide)⟩ : Fin roundCount) =
        ⟨previous + 1, roundLt⟩ := by
    apply Fin.ext
    exact (congrArg (fun value => (value + 1) % roundCount)
      previousEq).trans (Nat.mod_eq_of_lt roundLt)
  have laneFinEq :
      (⟨((logicalSourceOffset column - 1576) % 992) % 4,
        Nat.mod_lt _ (by decide)⟩ : Fin 4) = lane := by
    apply Fin.ext
    exact laneEq
  change poseidonWindowCandidate column = _
  unfold poseidonWindowCandidate
  dsimp only
  apply congrArg₂ Location.poseidon
  · simpa only [sourceFinEq, roundFinEq]
  · exact laneFinEq

private theorem logicalSourceOffset_poseidonEntry (source : Nat)
    (lane : Fin 4) :
    logicalSourceOffset
        (PiRLCSamplerOrdinaryDirectSource.poseidonSource source 0 lane) =
      584 + lane.val := by
  have inputEq :
      PiRLCSamplerOrdinaryDirectSource.poseidonSource source 0 lane =
        PiRLCStarts.samplerLogicalStart + source * 15504 +
          (584 + lane.val) := by
    simp [PiRLCSamplerOrdinaryDirectSource.poseidonSource,
      PiRLCStarts.samplerSourceLogicalStart]
    omega
  rw [inputEq]
  exact logicalSourceOffset_at source (584 + lane.val) (by
    have laneLt := lane.isLt
    omega)

private theorem logicalSourceOffset_poseidonWindow (source previous : Nat)
    (lane : Fin 4) (roundLt : previous + 1 < roundCount) :
    logicalSourceOffset
        (PiRLCSamplerOrdinaryDirectSource.poseidonSource source
          (previous + 1) lane) =
      1576 + previous * 992 + lane.val := by
  let offset := 1576 + previous * 992 + lane.val
  have roundLt8 : previous + 1 < 8 := by
    simpa [roundCount] using roundLt
  have offsetLt : offset < 15504 := by
    dsimp [offset]
    have laneLt := lane.isLt
    omega
  have inputEq :
      PiRLCSamplerOrdinaryDirectSource.poseidonSource source
          (previous + 1) lane =
        PiRLCStarts.samplerLogicalStart + source * 15504 + offset := by
    dsimp [offset]
    simp [PiRLCSamplerOrdinaryDirectSource.poseidonSource,
      PiRLCStarts.samplerSourceLogicalStart, Sampler.windowOffset,
      Sampler.windowBase, SamplerChain.sourceOffset,
      DigestWindow.permutationOffset, Sampler.logicalPrivateCount,
      Sampler.entryPrivateCount, DigestWindow.logicalPrivateCount,
      DigestLane.logicalPrivateCount]
    omega
  rw [inputEq]
  exact logicalSourceOffset_at source offset offsetLt

private theorem poseidonCandidate_entry (source : Nat) (lane : Fin 4)
    (sourceLt : source < sourceCount) :
    poseidonCandidate
        (PiRLCSamplerOrdinaryDirectSource.poseidonSource source 0 lane) =
      .poseidon
        { source := ⟨source, sourceLt⟩
          round := ⟨0, by decide⟩
          lane := ⟨0, by decide⟩ }
        lane := by
  unfold poseidonCandidate
  rw [if_pos (by
    rw [logicalSourceOffset_poseidonEntry]
    have laneLt := lane.isLt
    omega)]
  exact poseidonEntryCandidate_source source lane sourceLt

private theorem poseidonCandidate_window (source previous : Nat)
    (lane : Fin 4) (sourceLt : source < sourceCount)
    (roundLt : previous + 1 < roundCount) :
    poseidonCandidate
        (PiRLCSamplerOrdinaryDirectSource.poseidonSource source
          (previous + 1) lane) =
      .poseidon
        { source := ⟨source, sourceLt⟩
          round := ⟨previous + 1, roundLt⟩
          lane := ⟨0, by decide⟩ }
        lane := by
  unfold poseidonCandidate
  rw [if_neg (by
    rw [logicalSourceOffset_poseidonWindow source previous lane roundLt]
    omega)]
  exact poseidonWindowCandidate_source source previous lane sourceLt roundLt

private theorem logicalSourceOffset_logical (descriptor : Lane)
    (position : Fin logicalCountPerLane) :
    logicalSourceOffset (logicalSource descriptor position) =
      592 + descriptor.round.val * 992 + descriptor.lane.val * 100 +
        position.val := by
  let offset := 592 + descriptor.round.val * 992 +
    descriptor.lane.val * 100 + position.val
  have offsetLt : offset < 15504 := by
    have roundLt := descriptor.round.isLt
    have laneLt := descriptor.lane.isLt
    have positionLt := position.isLt
    norm_num [roundCount, laneCount, logicalCountPerLane,
      PiRLCSamplerOrdinaryRows.digestRoundCount] at roundLt laneLt positionLt
    dsimp [offset]
    omega
  have inputEq : logicalSource descriptor position =
      PiRLCStarts.samplerLogicalStart + descriptor.source.val * 15504 +
        offset := by
    dsimp [offset]
    simp [logicalSource, PiRLCStarts.digestLaneLogicalStart,
      PiRLCStarts.windowLogicalStart, PiRLCStarts.samplerSourceLogicalStart]
    omega
  calc
    logicalSourceOffset (logicalSource descriptor position) =
        logicalSourceOffset
          (PiRLCStarts.samplerLogicalStart +
            descriptor.source.val * 15504 + offset) :=
      congrArg logicalSourceOffset inputEq
    _ = offset :=
      logicalSourceOffset_at descriptor.source.val offset offsetLt
    _ = 592 + descriptor.round.val * 992 +
        descriptor.lane.val * 100 + position.val := rfl

private theorem poseidonOwned_entry (source : Nat) (lane : Fin 4) :
    poseidonOwned
      (PiRLCSamplerOrdinaryDirectSource.poseidonSource source 0 lane) := by
  unfold poseidonOwned
  dsimp only
  rw [logicalSourceOffset_poseidonEntry]
  have laneLt := lane.isLt
  norm_num [roundCount, PiRLCSamplerOrdinaryRows.digestRoundCount]
  omega

private theorem poseidonOwned_window (source previous : Nat) (lane : Fin 4)
    (roundLt : previous + 1 < roundCount) :
    poseidonOwned
      (PiRLCSamplerOrdinaryDirectSource.poseidonSource source
        (previous + 1) lane) := by
  unfold poseidonOwned
  dsimp only
  rw [logicalSourceOffset_poseidonWindow source previous lane roundLt]
  have laneLt := lane.isLt
  have shifted : 1576 + previous * 992 + lane.val - 584 =
      (previous + 1) * 992 + lane.val := by omega
  rw [shifted, stride_div (previous + 1) lane.val 992 (by decide)
    (lt_trans laneLt (by decide)),
    stride_mod (previous + 1) lane.val 992
      (lt_trans laneLt (by decide))]
  exact ⟨by omega, roundLt, laneLt⟩

private theorem poseidonOwned_not_logical (descriptor : Lane)
    (position : Fin logicalCountPerLane) :
    ¬ poseidonOwned (logicalSource descriptor position) := by
  unfold poseidonOwned
  dsimp only
  rw [logicalSourceOffset_logical]
  let inner := 8 + descriptor.lane.val * 100 + position.val
  have laneLt := descriptor.lane.isLt
  have positionLt := position.isLt
  norm_num [laneCount, logicalCountPerLane] at laneLt positionLt
  have innerLt : inner < 992 := by
    dsimp [inner]
    omega
  have shifted :
      592 + descriptor.round.val * 992 + descriptor.lane.val * 100 +
          position.val - 584 =
        descriptor.round.val * 992 + inner := by
    dsimp [inner]
    omega
  rw [shifted, stride_mod descriptor.round.val inner 992 innerLt]
  dsimp [inner]
  omega

private theorem logicalOwned_logical (descriptor : Lane)
    (position : Fin logicalCountPerLane) :
    logicalOwned (logicalSource descriptor position) := by
  unfold logicalOwned
  dsimp only
  rw [logicalSourceOffset_logical]
  let inner := descriptor.lane.val * 100 + position.val
  have roundLt := descriptor.round.isLt
  have laneLt := descriptor.lane.isLt
  have positionLt := position.isLt
  norm_num [roundCount, laneCount, logicalCountPerLane,
    PiRLCSamplerOrdinaryRows.digestRoundCount] at roundLt laneLt positionLt
  have innerLt : inner < 992 := by
    dsimp [inner]
    omega
  have shifted :
      592 + descriptor.round.val * 992 + descriptor.lane.val * 100 +
          position.val - 592 =
        descriptor.round.val * 992 + inner := by
    dsimp [inner]
    omega
  rw [shifted, stride_div descriptor.round.val inner 992 (by decide) innerLt,
    stride_mod descriptor.round.val inner 992 innerLt]
  dsimp [inner]
  exact ⟨by omega, roundLt, by omega⟩

private theorem selectorSourceOffset (source : Nat) :
    logicalSourceOffset
      (PiRLCSamplerOrdinaryDirectSource.selectorSource source) = 15449 := by
  have inputEq : PiRLCSamplerOrdinaryDirectSource.selectorSource source =
      PiRLCStarts.samplerLogicalStart + source * 15504 + 15449 := by
    simp [PiRLCSamplerOrdinaryDirectSource.selectorSource,
      PiRLCStarts.selectorLogicalStart, PiRLCStarts.samplerSourceLogicalStart,
      First54.positionOffset, First54.candidateCount,
      First54.roundPrivateCount, First54.fullSlot, First54Step.fullSlot,
      First54Step.slotCount, First54ValueStep.outputCount]
  rw [inputEq]
  exact logicalSourceOffset_at source 15449 (by decide)

private theorem selector_not_poseidonOwned (source : Nat) :
    ¬ poseidonOwned
      (PiRLCSamplerOrdinaryDirectSource.selectorSource source) := by
  unfold poseidonOwned
  dsimp only
  rw [selectorSourceOffset]
  norm_num [roundCount]

private theorem selector_not_logicalOwned (source : Nat) :
    ¬ logicalOwned
      (PiRLCSamplerOrdinaryDirectSource.selectorSource source) := by
  unfold logicalOwned
  dsimp only
  rw [selectorSourceOffset]
  norm_num [roundCount]

private theorem poseidonEntry_before_fresh (source : Fin sourceCount)
    (lane : Fin 4) :
    PiRLCSamplerOrdinaryDirectSource.poseidonSource source.val 0 lane <
      PiRLCStarts.samplerFreshStart := by
  have sourceLt := source.isLt
  have laneLt := lane.isLt
  norm_num [sourceCount, PiRLCSamplerOrdinaryRows.sourceCount] at sourceLt
  unfold PiRLCStarts.samplerFreshStart
  rw [PiRLCStarts.phaseFreshStart_eq]
  norm_num [PiRLCSamplerOrdinaryDirectSource.poseidonSource,
    PiRLCStarts.samplerSourceLogicalStart, PiRLCStarts.samplerLogicalStart,
    PiRLCStarts.phaseLogicalStart, PiRLCInputs.phaseOffset,
    NightstreamFPrime.Lifecycle.PiRLC.v1_1.Formal.samplerOffset]
  omega

private theorem poseidonWindow_before_fresh (source : Fin sourceCount)
    (previous : Nat) (roundLt : previous + 1 < roundCount) (lane : Fin 4) :
    PiRLCSamplerOrdinaryDirectSource.poseidonSource source.val
        (previous + 1) lane < PiRLCStarts.samplerFreshStart := by
  have sourceLt := source.isLt
  have laneLt := lane.isLt
  have roundLt8 : previous + 1 < 8 := by simpa [roundCount] using roundLt
  norm_num [sourceCount, PiRLCSamplerOrdinaryRows.sourceCount] at sourceLt
  unfold PiRLCStarts.samplerFreshStart
  rw [PiRLCStarts.phaseFreshStart_eq]
  norm_num [PiRLCSamplerOrdinaryDirectSource.poseidonSource,
    PiRLCStarts.samplerSourceLogicalStart, PiRLCStarts.samplerLogicalStart,
    PiRLCStarts.phaseLogicalStart, PiRLCInputs.phaseOffset,
    Sampler.windowOffset, Sampler.windowBase, SamplerChain.sourceOffset,
    DigestWindow.permutationOffset, Sampler.logicalPrivateCount,
    Sampler.entryPrivateCount, DigestWindow.logicalPrivateCount,
    DigestLane.logicalPrivateCount,
    NightstreamFPrime.Lifecycle.PiRLC.v1_1.Formal.samplerOffset]
  omega

private theorem logical_before_fresh (descriptor : Lane)
    (position : Fin logicalCountPerLane) :
    logicalSource descriptor position < PiRLCStarts.samplerFreshStart := by
  have sourceLt := descriptor.source.isLt
  have roundLt := descriptor.round.isLt
  have laneLt := descriptor.lane.isLt
  have positionLt := position.isLt
  norm_num [sourceCount, roundCount, laneCount, logicalCountPerLane,
    PiRLCSamplerOrdinaryRows.sourceCount,
    PiRLCSamplerOrdinaryRows.digestRoundCount] at sourceLt roundLt laneLt positionLt
  unfold PiRLCStarts.samplerFreshStart
  rw [PiRLCStarts.phaseFreshStart_eq]
  norm_num [logicalSource, PiRLCStarts.digestLaneLogicalStart,
    PiRLCStarts.windowLogicalStart, PiRLCStarts.samplerSourceLogicalStart,
    PiRLCStarts.samplerLogicalStart, PiRLCStarts.phaseLogicalStart,
    PiRLCInputs.phaseOffset,
    NightstreamFPrime.Lifecycle.PiRLC.v1_1.Formal.samplerOffset]
  omega

private theorem selector_before_fresh (source : Fin sourceCount) :
    PiRLCSamplerOrdinaryDirectSource.selectorSource source.val <
      PiRLCStarts.samplerFreshStart := by
  have sourceLt := source.isLt
  norm_num [sourceCount, PiRLCSamplerOrdinaryRows.sourceCount] at sourceLt
  unfold PiRLCStarts.samplerFreshStart
  rw [PiRLCStarts.phaseFreshStart_eq]
  norm_num [PiRLCSamplerOrdinaryDirectSource.selectorSource,
    PiRLCStarts.selectorLogicalStart, PiRLCStarts.samplerSourceLogicalStart,
    PiRLCStarts.samplerLogicalStart, PiRLCStarts.phaseLogicalStart,
    PiRLCInputs.phaseOffset, First54.positionOffset,
    First54.candidateCount, First54.roundPrivateCount,
    First54.fullSlot, First54Step.fullSlot, First54Step.slotCount,
    First54ValueStep.outputCount,
    NightstreamFPrime.Lifecycle.PiRLC.v1_1.Formal.samplerOffset]
  omega

private theorem fresh_at_or_after (descriptor : Lane)
    (position : Fin freshCountPerLane) :
    PiRLCStarts.samplerFreshStart ≤ freshSource descriptor position := by
  simp [freshSource, PiRLCStarts.digestLaneFreshStart,
    PiRLCStarts.windowFreshStart, PiRLCStarts.samplerSourceFreshStart]
  omega

private theorem selectorCandidate_source (source : Nat)
    (sourceLt : source < sourceCount) :
    selectorCandidate
        (PiRLCSamplerOrdinaryDirectSource.selectorSource source) =
      .selector ⟨source, sourceLt⟩ := by
  have inputEq :
      PiRLCSamplerOrdinaryDirectSource.selectorSource source =
        PiRLCStarts.samplerLogicalStart + source * 15504 + 15449 := by
    simp [PiRLCSamplerOrdinaryDirectSource.selectorSource,
      PiRLCStarts.selectorLogicalStart,
      PiRLCStarts.samplerSourceLogicalStart, First54.positionOffset,
      First54.candidateCount, First54.roundPrivateCount, First54.fullSlot,
      First54Step.fullSlot, First54Step.slotCount,
      First54ValueStep.outputCount]
  let column := PiRLCSamplerOrdinaryDirectSource.selectorSource source
  have sourceIndexEq : logicalSourceIndex column = source := by
    rw [show column = PiRLCStarts.samplerLogicalStart + source * 15504 +
        15449 by exact inputEq]
    exact logicalSourceIndex_at source 15449 (by decide)
  have sourceFinEq :
      (⟨logicalSourceIndex column % sourceCount,
        Nat.mod_lt _ (by decide)⟩ : Fin sourceCount) = ⟨source, sourceLt⟩ := by
    apply Fin.ext
    exact (congrArg (fun value => value % sourceCount) sourceIndexEq).trans
      (Nat.mod_eq_of_lt sourceLt)
  change selectorCandidate column = _
  unfold selectorCandidate
  exact congrArg Location.selector sourceFinEq

def samplerStateColumn (source : Fin sourceCount)
    (step : Fin PiRLCSamplerPoseidonPlan.invocationsPerSource)
    (lane : Fin Spec.Poseidon2.width) : Nat :=
  PiRLCStarts.samplerLogicalStart + source.val * 15504 + 584 +
    step.val * 992 + lane.val

/-- State outputs not read by the ordinary sampler rows are rejected by the
ordinary classifier. These are lanes four through seven and the final digest
window output. -/
theorem classifySource_samplerState_missing
    (source : Fin sourceCount)
    (step : Fin PiRLCSamplerPoseidonPlan.invocationsPerSource)
    (lane : Fin Spec.Poseidon2.width)
    (missing : step.val = 8 ∨ 4 ≤ lane.val) :
    classifySource (samplerStateColumn source step lane) = none := by
  let inner := 584 + step.val * 992 + lane.val
  have sourceLt := source.isLt
  change source.val < 17 at sourceLt
  have stepLt := step.isLt
  change step.val < 9 at stepLt
  have laneLt := lane.isLt
  change lane.val < 8 at laneLt
  have innerLt : inner < 15504 := by
    dsimp [inner]
    omega
  have columnEq : samplerStateColumn source step lane =
      PiRLCStarts.samplerLogicalStart + source.val * 15504 + inner := by
    unfold samplerStateColumn
    dsimp [inner]
    omega
  have offsetEq : logicalSourceOffset (samplerStateColumn source step lane) =
      inner := by
    rw [columnEq]
    exact logicalSourceOffset_at source.val inner innerLt
  have indexEq : logicalSourceIndex (samplerStateColumn source step lane) =
      source.val := by
    rw [columnEq]
    exact logicalSourceIndex_at source.val inner innerLt
  have beforeFresh : samplerStateColumn source step lane <
      PiRLCStarts.samplerFreshStart := by
    unfold samplerStateColumn PiRLCStarts.samplerFreshStart
    rw [PiRLCStarts.phaseFreshStart_eq]
    norm_num [PiRLCStarts.samplerLogicalStart, PiRLCStarts.phaseLogicalStart,
      PiRLCInputs.phaseOffset,
      NightstreamFPrime.Lifecycle.PiRLC.v1_1.Formal.samplerOffset]
    omega
  have notPoseidon : ¬ poseidonOwned (samplerStateColumn source step lane) := by
    intro owned
    unfold poseidonOwned at owned
    dsimp only at owned
    rw [offsetEq] at owned
    change 584 ≤ inner ∧ (inner - 584) / 992 < 8 ∧
      (inner - 584) % 992 < 4 at owned
    have shifted : inner - 584 = step.val * 992 + lane.val := by
      dsimp [inner]
      omega
    rw [shifted, stride_div step.val lane.val 992 (by decide) (by omega),
      stride_mod step.val lane.val 992 (by omega)] at owned
    rcases missing with final | high
    · omega
    · omega
  have notLogical : ¬ logicalOwned (samplerStateColumn source step lane) := by
    intro owned
    unfold logicalOwned at owned
    dsimp only at owned
    rw [offsetEq] at owned
    change 592 ≤ inner ∧ (inner - 592) / 992 < 8 ∧
      (inner - 592) % 992 < 400 at owned
    by_cases first : step.val = 0
    · dsimp [inner] at owned
      omega
    · obtain ⟨previous, stepEq⟩ := Nat.exists_eq_succ_of_ne_zero first
      let tail := 984 + lane.val
      have tailLt : tail < 992 := by
        dsimp [tail]
        omega
      have shifted : inner - 592 = previous * 992 + tail := by
        dsimp [inner, tail]
        omega
      rw [shifted, stride_mod previous tail 992 tailLt] at owned
      dsimp [tail] at owned
      omega
  have candidateEq : selectorCandidate (samplerStateColumn source step lane) =
      .selector source := by
    unfold selectorCandidate
    apply congrArg Location.selector
    apply Fin.ext
    change logicalSourceIndex (samplerStateColumn source step lane) %
      sourceCount = source.val
    rw [indexEq, Nat.mod_eq_of_lt source.isLt]
  have notSelector : (Location.selector source).sourceColumn ≠
      samplerStateColumn source step lane := by
    intro same
    norm_num [Location.sourceColumn, samplerStateColumn,
      PiRLCSamplerOrdinaryDirectSource.selectorSource,
      PiRLCStarts.selectorLogicalStart,
      PiRLCStarts.samplerSourceLogicalStart, First54.positionOffset,
      First54.candidateCount, First54.roundPrivateCount, First54.fullSlot,
      First54Step.fullSlot, First54Step.slotCount,
      First54ValueStep.outputCount] at same
    omega
  unfold classifySource
  rw [if_neg (Nat.not_le_of_lt beforeFresh), if_neg notPoseidon,
    if_neg notLogical, candidateEq]
  simp [exactCandidate, notSelector]

theorem classifySource_complete {column : Nat}
    (support : PiRLCSamplerOrdinaryDirectSource.Source column) :
    (classifySource column).isSome := by
  cases support with
  | poseidon source round lane sourceLt roundLt =>
      cases round with
    | zero =>
        unfold classifySource
        rw [if_neg (Nat.not_le_of_lt
          (poseidonEntry_before_fresh ⟨source, sourceLt⟩ lane))]
        rw [if_pos (poseidonOwned_entry source lane)]
        rw [poseidonCandidate_entry source lane sourceLt]
        simp [exactCandidate, Location.sourceColumn]
    | succ previous =>
        unfold classifySource
        rw [if_neg (Nat.not_le_of_lt
          (poseidonWindow_before_fresh ⟨source, sourceLt⟩ previous roundLt
            lane))]
        rw [if_pos (poseidonOwned_window source previous lane roundLt)]
        rw [poseidonCandidate_window source previous lane sourceLt roundLt]
        simp [exactCandidate, Location.sourceColumn]
  | logical source round lane position sourceLt roundLt laneLt positionLt =>
      let descriptor : Lane :=
        ⟨⟨source, sourceLt⟩, ⟨round, roundLt⟩, ⟨lane, laneLt⟩⟩
      let boundedPosition : Fin logicalCountPerLane := ⟨position, positionLt⟩
      have before := logical_before_fresh descriptor boundedPosition
      have noPoseidon := poseidonOwned_not_logical descriptor boundedPosition
      have ownsLogical := logicalOwned_logical descriptor boundedPosition
      change PiRLCStarts.digestLaneLogicalStart source round lane + position <
        PiRLCStarts.samplerFreshStart at before
      change ¬ poseidonOwned
        (PiRLCStarts.digestLaneLogicalStart source round lane + position) at noPoseidon
      change logicalOwned
        (PiRLCStarts.digestLaneLogicalStart source round lane + position) at ownsLogical
      unfold classifySource
      rw [if_neg (Nat.not_le_of_lt before)]
      rw [if_neg noPoseidon]
      rw [if_pos ownsLogical]
      rw [logicalCandidate_source source round lane position sourceLt roundLt
        laneLt positionLt]
      simp [exactCandidate, Location.sourceColumn, logicalSource]
  | fresh source round lane position sourceLt roundLt laneLt positionLt =>
      let descriptor : Lane :=
        ⟨⟨source, sourceLt⟩, ⟨round, roundLt⟩, ⟨lane, laneLt⟩⟩
      let boundedPosition : Fin freshCountPerLane := ⟨position, positionLt⟩
      have after := fresh_at_or_after descriptor boundedPosition
      change PiRLCStarts.samplerFreshStart ≤
        PiRLCStarts.digestLaneFreshStart source round lane + position at after
      unfold classifySource
      rw [if_pos after]
      rw [freshCandidate_source source round lane position sourceLt roundLt
        laneLt positionLt]
      simp [exactCandidate, Location.sourceColumn, freshSource]
  | selector source sourceLt =>
      unfold classifySource
      rw [if_neg (Nat.not_le_of_lt
        (selector_before_fresh ⟨source, sourceLt⟩))]
      rw [if_neg (selector_not_poseidonOwned source)]
      rw [if_neg (selector_not_logicalOwned source)]
      rw [selectorCandidate_source source sourceLt]
      simp [exactCandidate, Location.sourceColumn]

/-- Exact source-classifier result for a sampler entry-permutation output. -/
theorem classifySource_poseidonEntry (source : Fin sourceCount)
    (lane : Fin 4) :
    classifySource
        (PiRLCSamplerOrdinaryDirectSource.poseidonSource source.val 0 lane) =
      some
        (.poseidon
          { source := source
            round := ⟨0, by decide⟩
            lane := ⟨0, by decide⟩ }
          lane) := by
  let location : Location :=
    .poseidon
      { source := source
        round := ⟨0, by decide⟩
        lane := ⟨0, by decide⟩ }
      lane
  have candidateEq : poseidonCandidate
      (PiRLCSamplerOrdinaryDirectSource.poseidonSource source.val 0 lane) =
      location := by
    simpa [location] using poseidonCandidate_entry source.val lane source.isLt
  have owns : location.sourceColumn =
      PiRLCSamplerOrdinaryDirectSource.poseidonSource source.val 0 lane := by
    rfl
  unfold classifySource
  rw [if_neg (Nat.not_le_of_lt (poseidonEntry_before_fresh source lane))]
  rw [if_pos (poseidonOwned_entry source.val lane)]
  rw [candidateEq]
  rw [show exactCandidate
      (PiRLCSamplerOrdinaryDirectSource.poseidonSource source.val 0 lane)
      location = some location by
    unfold exactCandidate
    rw [if_pos owns]]

/-- Exact source-classifier result for a sampler window-permutation output. -/
theorem classifySource_poseidonWindow (source : Fin sourceCount)
    (previous : Nat) (roundLt : previous + 1 < roundCount) (lane : Fin 4) :
    classifySource
        (PiRLCSamplerOrdinaryDirectSource.poseidonSource source.val
          (previous + 1) lane) =
      some
        (.poseidon
          { source := source
            round := ⟨previous + 1, roundLt⟩
            lane := ⟨0, by decide⟩ }
          lane) := by
  let location : Location :=
    .poseidon
      { source := source
        round := ⟨previous + 1, roundLt⟩
        lane := ⟨0, by decide⟩ }
      lane
  have candidateEq : poseidonCandidate
      (PiRLCSamplerOrdinaryDirectSource.poseidonSource source.val
        (previous + 1) lane) = location := by
    simpa [location] using poseidonCandidate_window source.val previous lane
      source.isLt roundLt
  have owns : location.sourceColumn =
      PiRLCSamplerOrdinaryDirectSource.poseidonSource source.val
        (previous + 1) lane := by
    rfl
  unfold classifySource
  rw [if_neg (Nat.not_le_of_lt
    (poseidonWindow_before_fresh source previous roundLt lane))]
  rw [if_pos (poseidonOwned_window source.val previous lane roundLt)]
  rw [candidateEq]
  rw [show exactCandidate
      (PiRLCSamplerOrdinaryDirectSource.poseidonSource source.val
        (previous + 1) lane) location = some location by
    unfold exactCandidate
    rw [if_pos owns]]

theorem classifySource_logical (descriptor : Lane)
    (position : Fin logicalCountPerLane) :
    classifySource (logicalSource descriptor position) =
      some (.logical descriptor position) := by
  have before := logical_before_fresh descriptor position
  have noPoseidon := poseidonOwned_not_logical descriptor position
  have ownsLogical := logicalOwned_logical descriptor position
  rcases descriptor with ⟨source, round, lane⟩
  unfold classifySource
  rw [if_neg (Nat.not_le_of_lt before), if_neg noPoseidon,
    if_pos ownsLogical]
  simp only [logicalSource]
  rw [logicalCandidate_source source.val round.val lane.val position.val
    source.isLt round.isLt lane.isLt position.isLt]
  simp [exactCandidate, Location.sourceColumn, logicalSource]

theorem classifySource_fresh (descriptor : Lane)
    (position : Fin freshCountPerLane) :
    classifySource (freshSource descriptor position) =
      some (.fresh descriptor position) := by
  have after := fresh_at_or_after descriptor position
  rcases descriptor with ⟨source, round, lane⟩
  unfold classifySource
  rw [if_pos after]
  simp only [freshSource]
  rw [freshCandidate_source source.val round.val lane.val position.val
    source.isLt round.isLt lane.isLt position.isLt]
  simp [exactCandidate, Location.sourceColumn, freshSource]

theorem classifySource_selector (source : Fin sourceCount) :
    classifySource
        (PiRLCSamplerOrdinaryDirectSource.selectorSource source.val) =
      some (.selector source) := by
  unfold classifySource
  rw [if_neg (Nat.not_le_of_lt (selector_before_fresh source)),
    if_neg (selector_not_poseidonOwned source.val),
    if_neg (selector_not_logicalOwned source.val)]
  rw [selectorCandidate_source source.val source.isLt]
  simp [exactCandidate, Location.sourceColumn]

def piDecGeometry {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program
      logicalWidth) :
    PiDECRetainedGeometry.Geometry program logicalWidth :=
  PiRLCSamplerOrdinaryRetainedGeometry.prefixGeometry geometry

def poseidonGeometry {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program
      logicalWidth) :
    PiCCSPoseidonPlan.Geometry program logicalWidth :=
  DirectPiDECPrefixPlan.poseidonGeometry (piDecGeometry geometry)

def piRlcGeometry {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program
      logicalWidth) :
    PiRLCRetainedGeometry.Geometry program logicalWidth :=
  DirectPrefixPlan.prefixGeometry (poseidonGeometry geometry)

namespace Location

def poseidonInvocation (descriptor : Lane) :
    Fin PiRLCSamplerPoseidonPlan.invocationCount :=
  PiRLCSamplerPoseidonPlan.invocation descriptor.source
    ⟨descriptor.round.val, lt_trans descriptor.round.isLt (by decide)⟩

/-- Existing retained form selected for one exact sampler ordinary source. -/
def form {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program
      logicalWidth) : Location → SparseForm logicalWidth
  | .poseidon descriptor sourceLane =>
      (PiRLCSamplerPoseidonPlan.interface (poseidonGeometry geometry)).output
        (poseidonInvocation descriptor) (DigestWindow.rateLane sourceLane)
  | .logical descriptor position =>
      (logicalBlock program).form
        (PiRLCSamplerOrdinaryRetainedGeometry.logicalStart program)
        (PiRLCSamplerOrdinaryRetainedGeometry.logicalFits geometry)
        (logicalSlot descriptor position)
  | .fresh descriptor position =>
      (freshBlock program).form
        (PiRLCSamplerOrdinaryRetainedGeometry.freshStart program)
        (PiRLCSamplerOrdinaryRetainedGeometry.freshFits geometry)
        (freshSlot descriptor position)
  | .selector source =>
      (PiRLCFirst54RetainedBlocks.positionBlock program).form
        (PiRLCRetainedGeometry.positionStart program)
        (PiRLCRetainedGeometry.positionFits (piRlcGeometry geometry))
        (PiRLCFirst54DirectSchedule.positionIndex
          (PiRLCFirst54DirectPlan.finalPositionDescriptor source))

private theorem poseidonSource_lt (descriptor : Lane) (sourceLane : Fin 4) :
    (Location.poseidon descriptor sourceLane).sourceColumn <
      Spartan.SourceColumnCount := by
  rcases descriptor with ⟨source, round, lane⟩
  have sourceLt := source.isLt
  have roundLt := round.isLt
  have sourceLaneLt := sourceLane.isLt
  change source.val < 17 at sourceLt
  change round.val < 8 at roundLt
  rw [Spartan.sourceColumnCount_eq]
  cases roundValue : round.val with
  | zero =>
      simp [Location.sourceColumn,
        PiRLCSamplerOrdinaryDirectSource.poseidonSource, roundValue,
        PiRLCStarts.samplerSourceLogicalStart,
        PiRLCStarts.samplerLogicalStart, PiRLCStarts.phaseLogicalStart,
        PiRLCInputs.phaseOffset,
        NightstreamFPrime.Lifecycle.PiRLC.v1_1.Formal.samplerOffset]
      omega
  | succ previous =>
      simp [Location.sourceColumn,
        PiRLCSamplerOrdinaryDirectSource.poseidonSource, roundValue,
        Sampler.windowOffset, Sampler.windowBase, SamplerChain.sourceOffset,
        DigestWindow.permutationOffset, Sampler.logicalPrivateCount,
        Sampler.entryPrivateCount, DigestWindow.logicalPrivateCount,
        DigestLane.logicalPrivateCount,
        PiRLCStarts.samplerLogicalStart, PiRLCStarts.phaseLogicalStart,
        PiRLCInputs.phaseOffset,
        NightstreamFPrime.Lifecycle.PiRLC.v1_1.Formal.samplerOffset]
      omega

private theorem selectorSource_lt (source : Fin sourceCount) :
    (Location.selector source).sourceColumn < Spartan.SourceColumnCount := by
  have sourceLt := source.isLt
  change source.val < 17 at sourceLt
  rw [Spartan.sourceColumnCount_eq]
  simp [Location.sourceColumn,
    PiRLCSamplerOrdinaryDirectSource.selectorSource,
    PiRLCStarts.selectorLogicalStart, PiRLCStarts.samplerSourceLogicalStart,
    PiRLCStarts.samplerLogicalStart, PiRLCStarts.phaseLogicalStart,
    PiRLCInputs.phaseOffset,
    NightstreamFPrime.Lifecycle.PiRLC.v1_1.Formal.samplerOffset,
    First54.positionOffset, First54.candidateCount,
    First54.roundPrivateCount, First54Step.slotCount,
    First54ValueStep.outputCount, First54.fullSlot, First54Step.fullSlot]
  omega

theorem sourceColumn_lt (location : Location) :
    location.sourceColumn < Spartan.SourceColumnCount := by
  cases location with
  | poseidon descriptor sourceLane =>
      exact poseidonSource_lt descriptor sourceLane
  | logical descriptor position =>
      exact logicalSource_lt descriptor position
  | fresh descriptor position =>
      exact freshSource_lt descriptor position
  | selector source =>
      exact selectorSource_lt source

end Location

def classifyTarget (column : Nat) : Option Location :=
  match Spartan.spartanToSource column with
  | none => none
  | some source => classifySource source

theorem classifyTarget_complete {column : Nat}
    (support : PiRLCSamplerOrdinaryDirectSource.Target column) :
    ∃ decoded, classifyTarget column = some decoded := by
  rcases support with ⟨source, sourceSupport, rfl⟩
  have complete := classifySource_complete sourceSupport
  cases found : classifySource source with
  | none =>
      rw [found] at complete
      contradiction
  | some location =>
      have sourceBound : source < Spartan.SourceColumnCount := by
        have locationBound := location.sourceColumn_lt
        have owns := classifySource_sound found
        exact Eq.mp (congrArg (fun value => value < Spartan.SourceColumnCount)
          owns) locationBound
      have inverse := Spartan.spartanToSource_sourceToSpartan source sourceBound
      refine ⟨location, ?_⟩
      calc
        classifyTarget (Spartan.sourceToSpartan source) =
            classifySource source := by
          unfold classifyTarget
          rw [inverse]
        _ = some location := found

def resolvedForm {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program
      logicalWidth) (column : Nat) : SparseForm logicalWidth :=
  match classifyTarget column with
  | none => .empty
  | some location => location.form geometry

/-- Assignment-derived source environment for the exact canonical sampler
ordinary rows. Supported columns select a retained form; unsupported columns
are irrelevant to those rows and evaluate to zero. -/
def resolvedEnv {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program
      logicalWidth) (assignment : Assignment F logicalWidth) : Env :=
  fun column => (resolvedForm geometry column).eval assignment

def sourceMap {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program
      logicalWidth) :
    SourceCompiler.SourceMap Spartan.spartanColumnCount logicalWidth where
  form := fun column => resolvedForm geometry column.val

@[simp] theorem sourceMap_form_eval
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program
      logicalWidth) (assignment : Assignment F logicalWidth)
    (column : Fin Spartan.spartanColumnCount) :
    ((sourceMap geometry).form column).eval assignment =
      resolvedEnv geometry assignment column.val := by
  rfl

private theorem preservesCombination
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program
      logicalWidth) (assignment : Assignment F logicalWidth)
    (combination : R1CS.LinearCombination)
    (bounded : SourceCompiler.CombinationBounded Spartan.spartanColumnCount
      combination) :
    OrdinarySourcePlan.SourceMap.PreservesCombination (sourceMap geometry)
      assignment (resolvedEnv geometry assignment) combination bounded := by
  intro term member
  exact sourceMap_form_eval geometry assignment ⟨term.1, bounded term member⟩

variable {relationLogicalWidth : Nat}
  {relationPublicFits : ringDegree * publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth relationLogicalWidth}

def inputs
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (_relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program
      logicalWidth) :
    (PiRLCSamplerOrdinaryDirectSource.program
      (logicalWidth := relationLogicalWidth)
      (publicFits := relationPublicFits)).Inputs logicalWidth where
  oneColumn := PiRLCSamplerOrdinaryRetainedGeometry.oneColumn geometry
  sourceMap := fun _ => sourceMap geometry

theorem inputs_preserve
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program
      logicalWidth) (assignment : Assignment F logicalWidth) :
    ∀ index, OrdinarySourcePlan.SourceMap.PreservesRow
      ((inputs relation geometry).sourceMap index) assignment
      (resolvedEnv geometry assignment)
      ((PiRLCSamplerOrdinaryDirectSource.program
        (logicalWidth := relationLogicalWidth)
        (publicFits := relationPublicFits)).row index)
      ((PiRLCSamplerOrdinaryDirectSource.program
        (logicalWidth := relationLogicalWidth)
        (publicFits := relationPublicFits)).bounded index) := by
  intro index
  exact ⟨
    preservesCombination geometry assignment _ _ ,
    preservesCombination geometry assignment _ _ ,
    preservesCombination geometry assignment _ _ ⟩

/-- Exact row-local preservation for one indexed canonical sampler row. -/
theorem programRow_preserve
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program
      logicalWidth) (assignment : Assignment F logicalWidth)
    (index : Fin 220881) :
    OrdinarySourcePlan.SourceMap.PreservesRow (sourceMap geometry) assignment
      (resolvedEnv geometry assignment)
      (PiRLCSamplerOrdinaryDirectSource.programRow
        (logicalWidth := relationLogicalWidth)
        (publicFits := relationPublicFits) index)
      (PiRLCSamplerOrdinaryDirectSource.programRow_bounded
        (logicalWidth := relationLogicalWidth)
        (publicFits := relationPublicFits) index) := by
  exact ⟨
    preservesCombination geometry assignment _ _,
    preservesCombination geometry assignment _ _,
    preservesCombination geometry assignment _ _ ⟩

/-- Exact sparse forms for one canonical Lean-lowered sampler row. -/
def rowForms
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (_relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program
      logicalWidth) (index : Fin 220881) : OrdinaryRow.Forms logicalWidth :=
  SourceCompiler.compileRow (sourceMap geometry)
    (PiRLCSamplerOrdinaryRetainedGeometry.oneColumn geometry)
    (PiRLCSamplerOrdinaryDirectSource.programRow
      (logicalWidth := relationLogicalWidth)
      (publicFits := relationPublicFits) index)
    (PiRLCSamplerOrdinaryDirectSource.programRow_bounded
      (logicalWidth := relationLogicalWidth)
      (publicFits := relationPublicFits) index)

/-- Canonical direct 14-matrix rows for every sampler ordinary constraint. -/
def plan
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (_relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program
      logicalWidth) : ProductionRelation.Plan logicalWidth :=
  OrdinaryRow.planOfForms (by norm_num [Lifecycle.cubeVariables])
    (rowForms _relation geometry)

end NightstreamFPrime.Export.Stage1.PiRLCSamplerOrdinaryDirectPlan
