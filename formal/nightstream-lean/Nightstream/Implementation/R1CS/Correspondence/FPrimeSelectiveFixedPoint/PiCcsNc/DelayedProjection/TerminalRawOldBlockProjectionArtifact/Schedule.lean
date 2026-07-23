import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.TerminalRawOldBlockProjectionArtifact.Shape

/-!
Symbolic compact-prefix schedule for the fixed production projection.

The proof is indexed by a tensor round and parent.  It never materializes the
211,797 terminal tensor entries or the 262,143 multiplication traces.

Owns: symbolic agreement of every generated eighteen-round compact-prefix
tensor trace with the canonical low/high recurrence, output schedule, terminal
tensor coordinate map, and explicit common-factor operands.

Does not own: row ordinals, physical ownership, row coefficients, assignment
values, satisfaction, semantic acceptance, costs, or row-removal authority.

Emits constraints: no.

| Stable stage path | Mathematical obligation | Authority class |
|---|---|---|
| `f_prime.pi_ccs_nc.delayed.projection_schedule.round` | generated parent counts and live-high counts match the compact-prefix recurrence | derived |
| `f_prime.pi_ccs_nc.delayed.projection_schedule.trace` | every multiplication trace has the canonical input, output, and subtraction terms | derived |
| `f_prime.pi_ccs_nc.delayed.projection_schedule.terminal` | the prefix tensor output enumerates exactly the logical block coordinates | derived |
| `f_prime.pi_ccs_nc.delayed.projection_schedule.final_scale` | each generated lane sum is multiplied by `1 - oldBlock[18]` | derived / direct dataflow |
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.TerminalRawOldBlockProjectionArtifact

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.ProjectionProgram
open Nightstream.Implementation.R1CS.TerminalRawOldBlockProjectionCompiler

private abbrev generatedCount (round : Nat) : Nat :=
  Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Execution.RawOldBlockProjectionRowAt.tensorRoundMulCount round

private abbrev generatedHigh (round : Nat) : Nat :=
  Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Execution.RawOldBlockProjectionRowAt.tensorRoundHighCount round

private abbrev generatedTermsAt (round index : Nat) : KTerms :=
  Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Execution.RawOldBlockProjectionRowAt.tensorTermsAt round index

private abbrev generatedTrace (round parent : Nat) : KMulTrace :=
  Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Execution.RawOldBlockProjectionRowAt.tensorTrace round parent

private abbrev generatedOutput (round parent : Nat) : KColumns :=
  Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Execution.RawOldBlockProjectionRowAt.tensorOutputColumns round parent

private abbrev generatedSubtractOutput (terms : KTerms)
    (output : KColumns) : KTerms :=
  Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Execution.RawOldBlockProjectionRowAt.subtractOutput terms output

private def canonicalParents (round : Nat) : List KTerms :=
  (List.range (generatedCount round)).map (generatedTermsAt round)

@[simp] private theorem canonicalParents_length (round : Nat) :
    (canonicalParents round).length = generatedCount round := by
  simp [canonicalParents]

@[simp] private theorem canonicalParents_get (round : Nat)
    (index : Fin (canonicalParents round).length) :
    (canonicalParents round).get index = generatedTermsAt round index.val := by
  simp [canonicalParents]

private theorem generatedCount_positive (round : Nat) :
    0 < generatedCount round := by
  change 0 < Nat.min 211797 (2 ^ round)
  exact Nat.lt_min.mpr ⟨by decide, Nat.pow_pos (by decide)⟩

private theorem highLive_iff_generatedHigh
    (round parent : Nat) (parentInRange : parent < generatedCount round) :
    highLive 211797 round parent = true <->
      parent < generatedHigh round := by
  have parentPower : parent < 2 ^ round :=
    Nat.lt_of_lt_of_le parentInRange (Nat.min_le_right _ _)
  constructor
  · intro live
    have sumLt : parent + 2 ^ round < 211797 := by
      simpa [highLive] using live
    have remainderLt : parent < 211797 - 2 ^ round :=
      Nat.lt_sub_of_add_lt sumLt
    exact Nat.lt_min.mpr ⟨remainderLt, parentPower⟩
  · intro high
    have remainderLt : parent < 211797 - 2 ^ round :=
      (Nat.lt_min.mp high).1
    have sumLt : parent + 2 ^ round < 211797 :=
      Nat.add_lt_of_lt_sub remainderLt
    simpa [highLive] using sumLt

private theorem generatedCount_succ (round : Nat) :
    generatedCount (round + 1) =
      generatedCount round + generatedHigh round := by
  change Nat.min 211797 (2 ^ (round + 1)) =
    Nat.min 211797 (2 ^ round) +
      Nat.min (211797 - 2 ^ round) (2 ^ round)
  rw [Nat.pow_succ]
  exact natMin_doublePrefix 211797 (2 ^ round)

private theorem generatedHigh_le_count (round : Nat) :
    generatedHigh round ≤ generatedCount round := by
  apply Nat.le_min_of_le_of_le
  · exact Nat.le_trans (Nat.min_le_left _ _) (Nat.sub_le _ _)
  · exact Nat.min_le_right _ _

private theorem generatedTermsAt_succ_low
    (round parent : Nat) (parentInRange : parent < generatedCount round) :
    generatedTermsAt (round + 1) parent =
      if parent < generatedHigh round then
        generatedSubtractOutput (generatedTermsAt round parent)
          (generatedOutput round parent)
      else
        KTerms.ofColumns (generatedOutput round parent) := by
  simp [generatedTermsAt, generatedCount, generatedHigh,
    generatedSubtractOutput, generatedOutput, parentInRange,
    Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Execution.RawOldBlockProjectionRowAt.tensorTermsAt]

private theorem generatedTermsAt_succ_high
    (round parent : Nat) (parentInRange : parent < generatedHigh round) :
    generatedTermsAt (round + 1) (generatedCount round + parent) =
      KTerms.ofColumns (generatedOutput round parent) := by
  have countLe : generatedCount round ≤ generatedCount round + parent :=
    Nat.le_add_right _ _
  simp [generatedTermsAt, generatedCount, generatedHigh,
    generatedOutput, Nat.not_lt_of_ge countLe, parentInRange,
    Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Execution.RawOldBlockProjectionRowAt.tensorTermsAt]

private theorem canonicalParents_getD (round parent : Nat)
    (parentInRange : parent < generatedCount round) :
    (canonicalParents round).getD parent tensorRoot =
      generatedTermsAt round parent := by
  simp [canonicalParents, List.getD, parentInRange]

private theorem productionTensorTraceAt
    (round parent : Nat) (parentInRange : parent < generatedCount round) :
    tensorTraceAt (productionTensorLevel round) parent =
      generatedTrace round parent := by
  simp [tensorTraceAt, productionTensorLevel, parentInRange]

@[simp] private theorem generatedTrace_output (round parent : Nat) :
    (generatedTrace round parent).output = generatedOutput round parent := by
  rfl

private theorem productionLowTerms
    (round : Nat) :
    lowTerms 211797 round (canonicalParents round)
        (productionTensorLevel round) =
      (List.range (generatedCount round)).map
        (generatedTermsAt (round + 1)) := by
  unfold lowTerms
  rw [canonicalParents_length]
  apply List.map_congr_left
  intro parent parentMember
  have parentInRange : parent < generatedCount round :=
    List.mem_range.mp parentMember
  have highEquivalence :=
    highLive_iff_generatedHigh round parent parentInRange
  rw [canonicalParents_getD round parent parentInRange,
    productionTensorTraceAt round parent parentInRange]
  dsimp only
  rw [generatedTrace_output,
    generatedTermsAt_succ_low round parent parentInRange]
  by_cases live : highLive 211797 round parent = true
  · have high : parent < generatedHigh round := highEquivalence.mp live
    simp [live, high, generatedSubtractOutput, generatedOutput,
      Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Execution.RawOldBlockProjectionRowAt.subtractOutput]
  · have notHigh : ¬parent < generatedHigh round := by
      intro high
      exact live (highEquivalence.mpr high)
    simp [live, notHigh, generatedTrace, generatedOutput]

private theorem filterMap_congr_on
    {Input Output : Type} (indices : List Input)
    (left right : Input -> Option Output)
    (equal : forall index, index ∈ indices -> left index = right index) :
    indices.filterMap left = indices.filterMap right := by
  induction indices with
  | nil => rfl
  | cons index tail inductionHypothesis =>
      simp only [List.filterMap_cons]
      rw [equal index (by simp)]
      rw [inductionHypothesis (fun current member =>
        equal current (by simp [member]))]

private theorem filterMap_range_lt
    {Value : Type} (value : Nat -> Value) (count cutoff : Nat) :
    (List.range count).filterMap
        (fun index => if index < cutoff then some (value index) else none) =
      (List.range (Nat.min count cutoff)).map value := by
  induction count with
  | zero => simp
  | succ count inductionHypothesis =>
      rw [List.range_succ, List.filterMap_append, inductionHypothesis]
      by_cases live : count < cutoff
      · have minimum : Nat.min (count + 1) cutoff = count + 1 :=
          Nat.min_eq_left (by omega)
        have priorMinimum : Nat.min count cutoff = count :=
          Nat.min_eq_left (by omega)
        simp [live, minimum, priorMinimum, List.range_succ]
      · have minimum : Nat.min (count + 1) cutoff = cutoff :=
          Nat.min_eq_right (by omega)
        have priorMinimum : Nat.min count cutoff = cutoff :=
          Nat.min_eq_right (by omega)
        simp [live, minimum, priorMinimum]

private theorem productionHighTerms
    (round : Nat) :
    highTerms 211797 round (canonicalParents round)
        (productionTensorLevel round) =
      (List.range (generatedHigh round)).map fun parent =>
        generatedTermsAt (round + 1) (generatedCount round + parent) := by
  unfold highTerms
  rw [canonicalParents_length]
  calc
    (List.range (generatedCount round)).filterMap (fun parent =>
        if highLive 211797 round parent then
          some (KTerms.ofColumns
            (tensorTraceAt (productionTensorLevel round) parent).output)
        else none) =
      (List.range (generatedCount round)).filterMap (fun parent =>
        if parent < generatedHigh round then
          some (KTerms.ofColumns (generatedOutput round parent))
        else none) := by
          apply filterMap_congr_on
          intro parent parentMember
          have parentInRange : parent < generatedCount round :=
            List.mem_range.mp parentMember
          have highEquivalence :=
            highLive_iff_generatedHigh round parent parentInRange
          rw [productionTensorTraceAt round parent parentInRange,
            generatedTrace_output]
          by_cases live : highLive 211797 round parent = true
          · have high := highEquivalence.mp live
            simp [live, high, generatedTrace, generatedOutput]
          · have notHigh : ¬parent < generatedHigh round := by
              intro high
              exact live (highEquivalence.mpr high)
            simp [live, notHigh]
    _ = (List.range (Nat.min (generatedCount round)
          (generatedHigh round))).map fun parent =>
          KTerms.ofColumns (generatedOutput round parent) :=
      filterMap_range_lt
        (fun parent => KTerms.ofColumns (generatedOutput round parent))
        (generatedCount round) (generatedHigh round)
    _ = (List.range (generatedHigh round)).map fun parent =>
          KTerms.ofColumns (generatedOutput round parent) := by
      have minimum : Nat.min (generatedCount round)
          (generatedHigh round) = generatedHigh round :=
        Nat.min_eq_right (generatedHigh_le_count round)
      rw [minimum]
    _ = (List.range (generatedHigh round)).map fun parent =>
          generatedTermsAt (round + 1)
            (generatedCount round + parent) := by
      apply List.map_congr_left
      intro parent parentMember
      exact (generatedTermsAt_succ_high round parent
        (List.mem_range.mp parentMember)).symm

private theorem productionNextTensorTerms (round : Nat) :
    nextTensorTerms 211797 round (canonicalParents round)
        (productionTensorLevel round) =
      canonicalParents (round + 1) := by
  rw [nextTensorTerms, productionLowTerms, productionHighTerms]
  unfold canonicalParents
  rw [generatedCount_succ, List.range_add, List.map_append]
  rw [List.map_map]
  congr 1

private theorem canonicalParents_zero :
    canonicalParents 0 =
      [TerminalRawOldBlockProjectionCompiler.tensorRoot] := by
  rfl

private theorem generatedPointTerms_eq
    (round : Nat) (roundInRange : round < productionLayout.blockVariables) :
    Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Execution.RawOldBlockProjectionRowAt.pointTerms round =
      pointTerms (productionLayout.oldBlock ⟨round, roundInRange⟩) := by
  rfl

private theorem generatedOneMinusPointTerms_eq
    (round : Nat) (roundInRange : round < productionLayout.blockVariables) :
    Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Execution.RawOldBlockProjectionRowAt.oneMinusPointTerms round =
      oneMinusPointTerms
        (productionLayout.oldBlock ⟨round, roundInRange⟩) := by
  rfl

private theorem productionTensorLevel_valid
    (round : Nat) (roundInRange : round < productionLayout.blockVariables) :
    TensorLevelValid productionLayout round (canonicalParents round)
      (productionTensorLevel round) := by
  refine
    { roundWithin := roundInRange
      parentWidth := ?_
      multiplicationWidth := ?_
      operands := ?_ }
  · rw [canonicalParents_length, productionBlockCount]
    rfl
  · simpa [productionTensorLevel]
  · intro parent
    have parentInRange : parent.val < generatedCount round := by
      simpa using parent.isLt
    have highEquivalence :=
      highLive_iff_generatedHigh round parent.val parentInRange
    constructor
    · change (generatedTrace round parent.val).left =
          (canonicalParents round).get parent
      rw [canonicalParents_get]
      rfl
    · by_cases live : highLive 211797 round parent.val = true
      · have high : parent.val < generatedHigh round :=
          highEquivalence.mp live
        change
          (if parent.val < generatedHigh round then
            Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Execution.RawOldBlockProjectionRowAt.pointTerms round
          else
            Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Execution.RawOldBlockProjectionRowAt.oneMinusPointTerms round) =
          if highLive (blockCount productionLayout) round parent.val = true then
            pointTerms (productionLayout.oldBlock ⟨round, roundInRange⟩)
          else
            oneMinusPointTerms
              (productionLayout.oldBlock ⟨round, roundInRange⟩)
        rw [if_pos high, generatedPointTerms_eq round roundInRange]
        simp [live, productionBlockCount]
      · have notHigh : ¬ parent.val < generatedHigh round := by
          intro high
          exact live (highEquivalence.mpr high)
        change
          (if parent.val < generatedHigh round then
            Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Execution.RawOldBlockProjectionRowAt.pointTerms round
          else
            Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Execution.RawOldBlockProjectionRowAt.oneMinusPointTerms round) =
          if highLive (blockCount productionLayout) round parent.val = true then
            pointTerms (productionLayout.oldBlock ⟨round, roundInRange⟩)
          else
            oneMinusPointTerms
              (productionLayout.oldBlock ⟨round, roundInRange⟩)
        rw [if_neg notHigh,
          generatedOneMinusPointTerms_eq round roundInRange]
        simp [live, productionBlockCount]

private theorem productionScheduleFrom :
    forall (remaining round : Nat), round + remaining = 18 ->
      TensorScheduleValidFrom productionLayout round
        (canonicalParents round)
        ((List.range' round remaining).map productionTensorLevel)
  | 0, round, reachesEnd => by
      have roundEq : round = 18 := by omega
      subst round
      constructor
      · rfl
      · rw [canonicalParents_length, productionBlockCount]
        rfl
  | remaining + 1, round, reachesEnd => by
      have roundInRange : round < productionLayout.blockVariables := by
        change round < 18
        omega
      rw [List.range'_succ, List.map_cons]
      constructor
      · exact productionTensorLevel_valid round roundInRange
      · rw [productionBlockCount,
          productionNextTensorTerms]
        exact productionScheduleFrom remaining (round + 1) (by omega)

/-- The generated eighteen levels are exactly the compiler's compact-prefix
schedule.  This proof recurses over eighteen level descriptors; it never
constructs the 211,797 terminal tensor terms. -/
theorem productionTensorSchedule : TensorScheduleValid productionLayout := by
  unfold TensorScheduleValid
  rw [← canonicalParents_zero]
  simpa [productionTensorLevels] using
    productionScheduleFrom 18 0 (by decide)

private theorem productionTensorTermsFrom :
    forall (remaining round : Nat), round + remaining = 18 ->
      tensorTermsFrom 211797 round (canonicalParents round)
          ((List.range' round remaining).map productionTensorLevel) =
        canonicalParents 18
  | 0, round, reachesEnd => by
      have roundEq : round = 18 := by omega
      subst round
      rfl
  | remaining + 1, round, reachesEnd => by
      rw [List.range'_succ, List.map_cons, tensorTermsFrom,
        productionNextTensorTerms]
      exact productionTensorTermsFrom remaining (round + 1) (by omega)

theorem productionTensorTerms_eq_generated :
    tensorTerms (blockCount productionLayout)
      productionLayout.tensorLevels =
      (List.range 211797).map (generatedTermsAt 18) := by
  rw [productionBlockCount]
  unfold tensorTerms
  rw [← canonicalParents_zero]
  change
    tensorTermsFrom 211797 0 (canonicalParents 0)
      productionTensorLevels = _
  rw [show tensorTermsFrom 211797 0 (canonicalParents 0)
      productionTensorLevels = canonicalParents 18 by
    simpa [productionTensorLevels] using
      productionTensorTermsFrom 18 0 (by decide)]
  rfl

theorem productionCoordinateChiTerms
    (coordinate : Fin productionLayout.logicalWidth) :
    coordinateChiTerms productionLayout coordinate =
      generatedTermsAt 18 (coordinate.val / 54) := by
  have coordinateLt : coordinate.val < 11437038 := by
    change coordinate.val < 11437038
    exact coordinate.isLt
  have blockInRange : coordinate.val / 54 < 211797 := by
    exact (Nat.div_lt_iff_lt_mul (by decide : 0 < 54)).2 coordinateLt
  unfold coordinateChiTerms coordinateBlock
  change
    (tensorTerms (blockCount productionLayout)
      productionLayout.tensorLevels).getD
        (coordinate.val / 54) tensorRoot =
      generatedTermsAt 18 (coordinate.val / 54)
  rw [productionTensorTerms_eq_generated]
  simp [List.getD, blockInRange]

theorem productionFinalTensorWidth :
    (tensorTerms (blockCount productionLayout)
      productionLayout.tensorLevels).length =
        blockCount productionLayout := by
  rw [productionTensorTerms_eq_generated]
  rw [productionBlockCount]
  simp

private def productionLaneCoordinateNat
    (lane : Fin productionLayout.activeLanes) (block : Nat) :
    Fin productionLayout.logicalWidth :=
  if within : block < 211797 then
    ⟨block * 54 + lane.val, by
      have laneLt : lane.val < 54 := by
        have := lane.isLt
        change lane.val < 54 at this
        exact this
      change block * 54 + lane.val < 11437038
      omega⟩
  else
    ⟨0, by decide⟩

private theorem filterMap_eq_map_of_mem
    {Input Output : Type} (indices : List Input)
    (source : Input -> Option Output) (value : Input -> Output)
    (equal : forall index, index ∈ indices ->
      source index = some (value index)) :
    indices.filterMap source = indices.map value := by
  induction indices with
  | nil => rfl
  | cons index tail inductionHypothesis =>
      simp only [List.filterMap_cons, List.map_cons]
      rw [equal index (by simp)]
      rw [inductionHypothesis (fun current member =>
        equal current (by simp [member]))]

private theorem productionLaneCoordinates_eq_range
    (lane : Fin productionLayout.activeLanes) :
    laneCoordinates productionLayout lane =
      (List.range 211797).map (productionLaneCoordinateNat lane) := by
  unfold laneCoordinates
  rw [productionBlockCount]
  apply filterMap_eq_map_of_mem
  intro block blockMember
  have blockLt : block < 211797 := List.mem_range.mp blockMember
  have laneLt : lane.val < 54 := by
    have := lane.isLt
    change lane.val < 54 at this
    exact this
  have coordinateLt : block * 54 + lane.val < 11437038 := by omega
  change
    (if inRange : block * 54 + lane.val < 11437038 then
      some (⟨block * 54 + lane.val, inRange⟩ :
        Fin productionLayout.logicalWidth)
    else none) =
      some (productionLaneCoordinateNat lane block)
  simp [productionLaneCoordinateNat, blockLt, coordinateLt]

private theorem productionLaneProductColumns
    (lane : Fin productionLayout.activeLanes) (block : Nat)
    (blockLt : block < 211797) :
    productColumns productionLayout (productionLaneCoordinateNat lane block) =
      { c0 :=
          Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Execution.RawOldBlockProjectionRowAt.productColumn
            lane.val block 0
        c1 :=
          Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Execution.RawOldBlockProjectionRowAt.productColumn
            lane.val block 1 } := by
  have laneLt : lane.val < 54 := by
    have := lane.isLt
    change lane.val < 54 at this
    exact this
  have modEq : (block * 54 + lane.val) % 54 = lane.val := by
    simpa [Nat.mod_eq_of_lt laneLt] using
      Nat.mul_add_mod_self_right block 54 lane.val
  have divEq : (block * 54 + lane.val) / 54 = block := by
    rw [Nat.mul_comm block 54,
      Nat.mul_add_div (by decide : 0 < 54),
      Nat.div_eq_of_lt laneLt, Nat.add_zero]
  unfold productColumns
  simp [productionLaneCoordinateNat, blockLt, productionActiveLanes,
    productionBlockCount, productionProductFirst,
    Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Execution.RawOldBlockProjectionRowAt.productColumn,
    Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Execution.RawOldBlockProjectionRowAt.witnessOffset,
    Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Execution.RawOldBlockProjectionPlan.blockCount,
    modEq, divEq]

theorem productionLaneProductTermsC0
    (lane : Fin productionLayout.activeLanes) :
    Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Execution.RawOldBlockProjectionRowAt.laneTerms
        lane.val 0 =
      TerminalRawOldBlockProjectionCompiler.laneTerms productionLayout
        KColumns.c0 lane := by
  unfold
    Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Execution.RawOldBlockProjectionRowAt.laneTerms
    TerminalRawOldBlockProjectionCompiler.laneTerms
  rw [productionLaneCoordinates_eq_range, List.map_map]
  apply List.map_congr_left
  intro block blockMember
  have blockLt : block < 211797 := List.mem_range.mp blockMember
  change
    (Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Execution.RawOldBlockProjectionRowAt.productColumn
        lane.val block 0, 1) =
      ((productColumns productionLayout
        (productionLaneCoordinateNat lane block)).c0, 1)
  rw [productionLaneProductColumns lane block blockLt]

theorem productionLaneProductTermsC1
    (lane : Fin productionLayout.activeLanes) :
    Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Execution.RawOldBlockProjectionRowAt.laneTerms
        lane.val 1 =
      TerminalRawOldBlockProjectionCompiler.laneTerms productionLayout
        KColumns.c1 lane := by
  unfold
    Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Execution.RawOldBlockProjectionRowAt.laneTerms
    TerminalRawOldBlockProjectionCompiler.laneTerms
  rw [productionLaneCoordinates_eq_range, List.map_map]
  apply List.map_congr_left
  intro block blockMember
  have blockLt : block < 211797 := List.mem_range.mp blockMember
  change
    (Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Execution.RawOldBlockProjectionRowAt.productColumn
        lane.val block 1, 1) =
      ((productColumns productionLayout
        (productionLaneCoordinateNat lane block)).c1, 1)
  rw [productionLaneProductColumns lane block blockLt]

/-- Exact generated operands for the one common-factor multiplication in a
lane.  The right operand is tied directly to verifier-owned old-block column
18; no digest or sidecar participates in this association. -/
theorem productionFinalScaleOperands
    (lane : Fin productionFactoredLayout.base.activeLanes) :
    (productionFactoredLayout.scale lane).left =
        Nightstream.Implementation.R1CS.TerminalRawOldBlockProjectionFinalScaleCompiler.laneProductTerms
          productionFactoredLayout lane /\
      (productionFactoredLayout.scale lane).right =
        Nightstream.Implementation.R1CS.TerminalRawOldBlockProjectionFinalScaleCompiler.finalFactorTerms
          productionFactoredLayout := by
  constructor
  · change
      (⟨Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Execution.RawOldBlockProjectionRowAt.laneTerms
            lane.val 0,
          Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Execution.RawOldBlockProjectionRowAt.laneTerms
            lane.val 1⟩ : KTerms) =
        (⟨TerminalRawOldBlockProjectionCompiler.laneTerms
            productionLayout KColumns.c0 lane,
          TerminalRawOldBlockProjectionCompiler.laneTerms
            productionLayout KColumns.c1 lane⟩ : KTerms)
    rw [productionLaneProductTermsC0, productionLaneProductTermsC1]
  · rfl

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.TerminalRawOldBlockProjectionArtifact
