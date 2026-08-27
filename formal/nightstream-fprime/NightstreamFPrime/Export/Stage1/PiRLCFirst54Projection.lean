import NightstreamFPrime.Export.Stage1.Data
import NightstreamFPrime.Export.Stage1.PiRLCFirst54Invocations
import NightstreamFPrime.Export.Stage1.PiRLCSamplerOrdinaryRows
import NightstreamFPrime.Layout.PiRLC.v1_1.Leaves.First54
import NightstreamFPrime.Layout.R1CS.Segments

/-!
Owns structural projection from one exact production First54 selector lowering
to each position and value constraint at the canonical package fresh start.
The heavy selector owner supplies only order and cost theorems.
-/

namespace NightstreamFPrime.Export.Stage1.PiRLCFirst54Projection

open NightstreamFPrime.Circuit
open NightstreamFPrime.Gadgets.Sampling
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.PiRLC.v1_1.Leaves
open NightstreamFPrime.Layout.Stage1

def sourceInterface (source : Nat) :=
  PiRLCSamplerOrdinaryRows.sourceInterface
    (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits) source

private theorem flatConstraints_roundOpsPrefix_eq_range
    (source count : Nat) :
    flatConstraints
        (NightstreamFPrime.Gadgets.Sampling.First54.roundOpsPrefix
          (First54.selectorInterface (sourceInterface source) source
            (PiRLCStarts.samplerSourceLogicalStart source))
          (PiRLCStarts.selectorLogicalStart source) count) =
      ((List.range count).map fun round =>
        First54.roundConstraints (sourceInterface source) source
          (PiRLCStarts.samplerSourceLogicalStart source)
          (PiRLCStarts.selectorLogicalStart source) round).flatten := by
  induction count with
  | zero => rfl
  | succ count inductionHypothesis =>
      rw [First54.roundOpsPrefix_succ, flatConstraints_append,
        inductionHypothesis, First54.flatConstraints_roundOps]
      simp [List.range_succ, List.map_append, List.flatten_append]

private theorem roundsConstraints_eq_ofFn (source : Nat) :
    flatConstraints
        (NightstreamFPrime.Gadgets.Sampling.First54.roundOpsPrefix
          (First54.selectorInterface (sourceInterface source) source
            (PiRLCStarts.samplerSourceLogicalStart source))
          (PiRLCStarts.selectorLogicalStart source)
          NightstreamFPrime.Gadgets.Sampling.First54.candidateCount) =
      (List.ofFn fun round :
        Fin NightstreamFPrime.Gadgets.Sampling.First54.candidateCount =>
        First54.roundConstraints (sourceInterface source) source
          (PiRLCStarts.samplerSourceLogicalStart source)
          (PiRLCStarts.selectorLogicalStart source) round.val).flatten := by
  rw [flatConstraints_roundOpsPrefix_eq_range]
  apply congrArg List.flatten
  rw [List.ofFn_eq_map, ← List.map_coe_finRange_eq_range, List.map_map]
  simp [Function.comp_def]

private theorem roundFreshCounts_eq (source : Nat) :
    (List.ofFn fun round :
      Fin NightstreamFPrime.Gadgets.Sampling.First54.candidateCount =>
      R1CS.totalFreshCount
        (First54.roundConstraints (sourceInterface source) source
          (PiRLCStarts.samplerSourceLogicalStart source)
          (PiRLCStarts.selectorLogicalStart source) round.val)) =
      216 :: List.ofFn (fun _ : Fin 63 => 537) := by
  let cost : Fin 64 → Nat := fun round =>
    R1CS.totalFreshCount
      (First54.roundConstraints (sourceInterface source) source
        (PiRLCStarts.samplerSourceLogicalStart source)
        (PiRLCStarts.selectorLogicalStart source) round.val)
  change List.ofFn cost = 216 :: List.ofFn (fun _ : Fin 63 => 537)
  calc
    List.ofFn cost = cost 0 :: List.ofFn (fun round : Fin 63 =>
        cost round.succ) := List.ofFn_succ
    _ = _ := by
      apply congrArg₂ List.cons
      · exact First54.roundZero_totalFreshCount _ _ _ _
      · apply congrArg List.ofFn
        funext round
        exact First54.roundSucc_totalFreshCount _ _ _ _ round.val

private theorem sum_take_ofFn_const {count : Nat} (value : Nat)
    (index : Fin count) :
    ((List.ofFn fun _ : Fin count => value).take index.val).sum =
      index.val * value := by
  simp [index.isLt.le]

private theorem roundFreshPrefix (source : Nat)
    (round : Fin NightstreamFPrime.Gadgets.Sampling.First54.candidateCount) :
    ((List.ofFn fun current :
      Fin NightstreamFPrime.Gadgets.Sampling.First54.candidateCount =>
      R1CS.totalFreshCount
        (First54.roundConstraints (sourceInterface source) source
          (PiRLCStarts.samplerSourceLogicalStart source)
          (PiRLCStarts.selectorLogicalStart source) current.val)).take
            round.val).sum =
      PiRLCFirst54Invocations.roundFreshPrefix round.val := by
  rw [roundFreshCounts_eq]
  refine Fin.cases ?_ (fun previous => ?_) round
  · rfl
  · simp only [Fin.val_succ, List.take_succ_cons, List.sum_cons]
    rw [sum_take_ofFn_const 537 previous]
    simp [PiRLCFirst54Invocations.roundFreshPrefix]

private theorem positionFreshPrefix (source : Nat)
    (round : Fin NightstreamFPrime.Gadgets.Sampling.First54.candidateCount)
    (slot : Fin First54Step.slotCount) :
    ((List.ofFn fun current : Fin First54Step.slotCount =>
      R1CS.constraintFreshCount
        (First54.positionConstraint (sourceInterface source) source
          (PiRLCStarts.samplerSourceLogicalStart source)
          (PiRLCStarts.selectorLogicalStart source) round.val current)).take
            slot.val).sum =
      PiRLCFirst54Invocations.positionFreshPrefix round.val slot.val := by
  refine Fin.cases ?_ (fun previous => ?_) round
  · change
      ((List.ofFn fun current : Fin First54Step.slotCount =>
        R1CS.constraintFreshCount
          (First54.positionConstraint (sourceInterface source) source
            (PiRLCStarts.samplerSourceLogicalStart source)
            (PiRLCStarts.selectorLogicalStart source) 0 current)).take
              slot.val).sum =
        PiRLCFirst54Invocations.positionFreshPrefix 0 slot.val
    have countsEq :
        (List.ofFn fun current : Fin First54Step.slotCount =>
          R1CS.constraintFreshCount
            (First54.positionConstraint (sourceInterface source) source
              (PiRLCStarts.samplerSourceLogicalStart source)
              (PiRLCStarts.selectorLogicalStart source) 0 current)) =
          List.ofFn (fun _ : Fin First54Step.slotCount => 0) := by
      apply congrArg List.ofFn
      funext current
      exact (First54.positionZero_cost _ _ _ _ current).1
    rw [countsEq, sum_take_ofFn_const 0 slot]
    simp [PiRLCFirst54Invocations.positionFreshPrefix]
  · change
      ((List.ofFn fun current : Fin First54Step.slotCount =>
        R1CS.constraintFreshCount
          (First54.positionConstraint (sourceInterface source) source
            (PiRLCStarts.samplerSourceLogicalStart source)
            (PiRLCStarts.selectorLogicalStart source) (previous.val + 1)
            current)).take slot.val).sum =
        PiRLCFirst54Invocations.positionFreshPrefix (previous.val + 1)
          slot.val
    have countsEq :
        (List.ofFn fun current : Fin First54Step.slotCount =>
          R1CS.constraintFreshCount
            (First54.positionConstraint (sourceInterface source) source
              (PiRLCStarts.samplerSourceLogicalStart source)
              (PiRLCStarts.selectorLogicalStart source) (previous.val + 1)
              current)) =
          List.ofFn First54.runningPositionFresh := by
      apply congrArg List.ofFn
      funext current
      exact (First54.positionSucc_cost _ _ _ _ previous.val current).1
    rw [countsEq]
    fin_cases slot <;> rfl

private theorem valueFreshPrefix (source : Nat)
    (round : Fin NightstreamFPrime.Gadgets.Sampling.First54.candidateCount)
    (slot : Fin First54ValueStep.outputCount) :
    ((List.ofFn fun current : Fin First54ValueStep.outputCount =>
      R1CS.constraintFreshCount
        (First54.valueConstraint (sourceInterface source) source
          (PiRLCStarts.samplerSourceLogicalStart source)
          (PiRLCStarts.selectorLogicalStart source) round.val current)).take
            slot.val).sum =
      4 * slot.val := by
  have countsEq :
      (List.ofFn fun current : Fin First54ValueStep.outputCount =>
        R1CS.constraintFreshCount
          (First54.valueConstraint (sourceInterface source) source
            (PiRLCStarts.samplerSourceLogicalStart source)
            (PiRLCStarts.selectorLogicalStart source) round.val current)) =
        List.ofFn (fun _ : Fin First54ValueStep.outputCount => 4) := by
    apply congrArg List.ofFn
    funext current
    refine Fin.cases ?_ (fun previous => ?_) round
    · exact (First54.valueZero_cost _ _ _ _ current).1
    · exact (First54.valueSucc_cost _ _ _ _ previous.val current).1
  rw [countsEq, sum_take_ofFn_const 4 slot]
  omega

private theorem positionTotalFreshCount (source : Nat)
    (round : Fin NightstreamFPrime.Gadgets.Sampling.First54.candidateCount) :
    R1CS.totalFreshCount
        (First54.positionConstraints (sourceInterface source) source
          (PiRLCStarts.samplerSourceLogicalStart source)
          (PiRLCStarts.selectorLogicalStart source) round.val) =
      PiRLCFirst54Invocations.positionFreshCount round.val := by
  refine Fin.cases ?_ (fun previous => ?_) round
  · simpa [PiRLCFirst54Invocations.positionFreshCount] using
      First54.positionZero_totalFreshCount (sourceInterface source) source
        (PiRLCStarts.samplerSourceLogicalStart source)
        (PiRLCStarts.selectorLogicalStart source)
  · simpa [PiRLCFirst54Invocations.positionFreshCount] using
      First54.positionSucc_totalFreshCount (sourceInterface source) source
        (PiRLCStarts.samplerSourceLogicalStart source)
        (PiRLCStarts.selectorLogicalStart source) previous.val

private theorem selectorRows_imply_roundRows (source : Nat) (env : Env)
    (rows : R1CS.RowsHold env
      (R1CS.lowerConstraints
        (First54.logicalConstraints (sourceInterface source) source
          (PiRLCStarts.samplerSourceLogicalStart source)
          (PiRLCStarts.selectorLogicalStart source))
        (PiRLCStarts.selectorFreshStart source)).rows)
    (round : Fin NightstreamFPrime.Gadgets.Sampling.First54.candidateCount) :
    R1CS.RowsHold env
      (R1CS.lowerConstraints
        (First54.roundConstraints (sourceInterface source) source
          (PiRLCStarts.samplerSourceLogicalStart source)
          (PiRLCStarts.selectorLogicalStart source) round.val)
        (PiRLCStarts.selectorFreshStart source +
          PiRLCFirst54Invocations.roundFreshPrefix round.val)).rows := by
  rw [First54.logicalConstraints_eq_rounds_append_final,
    R1CS.lowerConstraints_append_rows] at rows
  have roundPrefixRows := (R1CS.rowsHold_append env _ _).mp rows |>.1
  rw [roundsConstraints_eq_ofFn] at roundPrefixRows
  have segments := (R1CS.rowsHold_flatten_iff _ _ _).mp roundPrefixRows
  have selected := R1CS.segmentsHold_ofFn_get env
    (fun current :
      Fin NightstreamFPrime.Gadgets.Sampling.First54.candidateCount =>
      First54.roundConstraints (sourceInterface source) source
        (PiRLCStarts.samplerSourceLogicalStart source)
        (PiRLCStarts.selectorLogicalStart source) current.val)
    (PiRLCStarts.selectorFreshStart source) segments round
  rw [roundFreshPrefix source round] at selected
  exact selected

/-- Exact selector rows project to one position constraint at the canonical
package fresh start. -/
theorem selectorRows_imply_positionRows (source : Nat) (env : Env)
    (rows : R1CS.RowsHold env
      (R1CS.lowerConstraints
        (First54.logicalConstraints (sourceInterface source) source
          (PiRLCStarts.samplerSourceLogicalStart source)
          (PiRLCStarts.selectorLogicalStart source))
        (PiRLCStarts.selectorFreshStart source)).rows)
    (round : Fin NightstreamFPrime.Gadgets.Sampling.First54.candidateCount)
    (slot : Fin First54Step.slotCount) :
    R1CS.RowsHold env
      (R1CS.lowerConstraint
        (First54.positionConstraint (sourceInterface source) source
          (PiRLCStarts.samplerSourceLogicalStart source)
          (PiRLCStarts.selectorLogicalStart source) round.val slot)
        (PiRLCStarts.selectorFreshStart source +
          PiRLCFirst54Invocations.roundFreshPrefix round.val +
          PiRLCFirst54Invocations.positionFreshPrefix round.val slot.val)).rows := by
  have roundRows := selectorRows_imply_roundRows source env rows round
  unfold First54.roundConstraints at roundRows
  rw [R1CS.lowerConstraints_append_rows] at roundRows
  have positionRows := (R1CS.rowsHold_append env _ _).mp roundRows |>.1
  rw [First54.positionConstraints_eq] at positionRows
  have selected := R1CS.rowsHold_lowerConstraints_ofFn_get env _ _
    positionRows slot
  rw [positionFreshPrefix source round slot] at selected
  exact selected

/-- Exact selector rows project to one value constraint at the canonical
package fresh start. -/
theorem selectorRows_imply_valueRows (source : Nat) (env : Env)
    (rows : R1CS.RowsHold env
      (R1CS.lowerConstraints
        (First54.logicalConstraints (sourceInterface source) source
          (PiRLCStarts.samplerSourceLogicalStart source)
          (PiRLCStarts.selectorLogicalStart source))
        (PiRLCStarts.selectorFreshStart source)).rows)
    (round : Fin NightstreamFPrime.Gadgets.Sampling.First54.candidateCount)
    (slot : Fin First54ValueStep.outputCount) :
    R1CS.RowsHold env
      (R1CS.lowerConstraint
        (First54.valueConstraint (sourceInterface source) source
          (PiRLCStarts.samplerSourceLogicalStart source)
          (PiRLCStarts.selectorLogicalStart source) round.val slot)
        (PiRLCStarts.selectorFreshStart source +
          PiRLCFirst54Invocations.roundFreshPrefix round.val +
          PiRLCFirst54Invocations.valueFreshPrefix round.val slot.val)).rows := by
  have roundRows := selectorRows_imply_roundRows source env rows round
  unfold First54.roundConstraints at roundRows
  rw [R1CS.lowerConstraints_append_rows] at roundRows
  have valueRows := (R1CS.rowsHold_append env _ _).mp roundRows |>.2
  rw [First54.valueConstraints_eq] at valueRows
  have selected := R1CS.rowsHold_lowerConstraints_ofFn_get env _ _ valueRows
    slot
  rw [valueFreshPrefix source round slot] at selected
  rw [positionTotalFreshCount source round] at selected
  simpa [PiRLCFirst54Invocations.valueFreshPrefix, Nat.add_assoc]
    using selected

end NightstreamFPrime.Export.Stage1.PiRLCFirst54Projection
