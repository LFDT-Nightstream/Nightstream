import NightstreamFPrime.Export.Stage1.PiRLCFirst54Invocations
import NightstreamFPrime.Export.Stage1.PiRLCSamplerOrdinaryRows
import NightstreamFPrime.Layout.PiRLC.v1_1.Leaves.First54

/-!
Owns conformance between the compact PiRLC `First54` export and the exact
production selector constraints.

The source maps below name logical columns before the Stage 1 Spartan column
permutation. Later theorems prove that each compact invocation expands to the
Spartan image of the corresponding Lean-lowered constraint.
-/

namespace NightstreamFPrime.Export.Stage1.PiRLCFirst54Conformance

open NightstreamFPrime.Circuit
open NightstreamFPrime.Export.Package
open NightstreamFPrime.Gadgets.Range
open NightstreamFPrime.Gadgets.Sampling
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.PiRLC.v1_1
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Export.Stage1.PiRLCFirst54Invocations

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}

def sourceInterface (source : Nat) : Sampler.Interface :=
  PiRLCSamplerOrdinaryRows.sourceInterface
    (logicalWidth := logicalWidth) (publicFits := publicFits) source

def selectorInterface (source : Nat) : First54.Interface :=
  Sampler.selectorInterface
    (sourceInterface (logicalWidth := logicalWidth)
      (publicFits := publicFits) source)
    source (PiRLCStarts.samplerSourceLogicalStart source)

def candidateIndexOf (candidate : Nat) : Fin First54.candidateCount :=
  First54.candidateIndex candidate

theorem candidateRound_val (candidate : Nat) :
    (Sampler.candidateRound (candidateIndexOf candidate)).val =
      candidate % First54.candidateCount / 8 := by
  rfl

theorem candidatePosition_val (candidate : Nat) :
    (Sampler.candidatePosition (candidateIndexOf candidate)).val =
      candidate % First54.candidateCount % 8 := by
  rfl

theorem decoderLogicalStart_eq (source candidate : Nat)
    (candidateLt : candidate < First54.candidateCount) :
    decoderLogicalStart source candidate =
      DigestLane.decoderOffset
        (DigestWindow.laneOffset
          (Sampler.windowOffset (PiRLCStarts.samplerSourceLogicalStart source)
            (Sampler.candidateRound (candidateIndexOf candidate)).val)
          (DigestWindow.laneOf
            (Sampler.candidatePosition (candidateIndexOf candidate))))
        (DigestWindow.partOf
          (Sampler.candidatePosition (candidateIndexOf candidate))) := by
  have candidateLt64 : candidate < 64 := by
    simpa [First54.candidateCount] using candidateLt
  have candidateMod64 : candidate % 64 = candidate :=
    Nat.mod_eq_of_lt candidateLt64
  simp only [decoderLogicalStart, PiRLCStarts.digestLaneLogicalStart,
    PiRLCStarts.windowLogicalStart, PiRLCStarts.samplerSourceLogicalStart,
    Sampler.windowOffset, Sampler.windowBase, Sampler.entryPrivateCount,
    DigestWindow.logicalPrivateCount, DigestWindow.laneOffset,
    DigestWindow.laneOf, DigestWindow.partOf, DigestLane.decoderOffset,
    candidateRound_val, candidatePosition_val, First54.candidateCount,
    CanonicalU64.auxiliaryCount, Candidate16Five.auxiliaryCount,
    DigestLane.logicalPrivateCount, candidateDigestRound, candidateLane,
    candidatePart, candidateMod64]
  omega

theorem rejectSourceColumn_eq (source candidate : Nat)
    (candidateLt : candidate < First54.candidateCount) :
    Expr.var (rejectSourceColumn source candidate) =
      DigestWindow.reject
        (Sampler.windowOffset (PiRLCStarts.samplerSourceLogicalStart source)
          (Sampler.candidateRound (candidateIndexOf candidate)).val)
        (Sampler.candidatePosition (candidateIndexOf candidate)) := by
  rw [rejectSourceColumn, decoderLogicalStart_eq source candidate candidateLt]
  rfl

theorem remainderSourceColumn_eq (source candidate : Nat)
    (candidateLt : candidate < First54.candidateCount) :
    Expr.var (remainderSourceColumn source candidate) =
      DigestWindow.remainder
        (Sampler.windowOffset (PiRLCStarts.samplerSourceLogicalStart source)
          (Sampler.candidateRound (candidateIndexOf candidate)).val)
        (Sampler.candidatePosition (candidateIndexOf candidate)) := by
  rw [remainderSourceColumn, decoderLogicalStart_eq source candidate candidateLt]
  rfl

def firstPositionSourceInput (source : Nat)
    (slot : Fin First54Step.slotCount) (input : Nat) : Nat :=
  if input = 0 then rejectSourceColumn source 0
  else if input = 1 then positionSourceStart source 0 + slot.val
  else 0

def laterPositionSourceInput (source round : Nat)
    (slot : Fin First54Step.slotCount) (input : Nat) : Nat :=
  if input = 0 then rejectSourceColumn source round
  else if input < 1 + First54Step.slotCount then
    previousPositionSourceStart source round + (input - 1)
  else if input = PiRLCFirst54Templates.laterPositionOutputInput then
    positionSourceStart source round + slot.val
  else 0

def firstValueSourceInput (source : Nat)
    (slot : Fin First54ValueStep.outputCount) (input : Nat) : Nat :=
  if input = 0 then rejectSourceColumn source 0
  else if input = 1 then remainderSourceColumn source 0
  else if input = PiRLCFirst54Templates.firstValueOutputInput then
    valueSourceStart source 0 + slot.val
  else 0

def laterValueSourceInput (source round : Nat)
    (slot : Fin First54ValueStep.outputCount) (input : Nat) : Nat :=
  if input = 0 then rejectSourceColumn source round
  else if input = 1 then remainderSourceColumn source round
  else if input < PiRLCFirst54Templates.laterValuePriorOutputStart then
    previousPositionSourceStart source round +
      (input - PiRLCFirst54Templates.laterValuePriorPositionStart)
  else if input < PiRLCFirst54Templates.laterValueOutputInput then
    previousValueSourceStart source round +
      (input - PiRLCFirst54Templates.laterValuePriorOutputStart)
  else if input = PiRLCFirst54Templates.laterValueOutputInput then
    valueSourceStart source round + slot.val
  else 0

theorem selectorLogicalStart_local (source : Nat) :
    Spartan.piCcsPhaseOffset ≤ PiRLCStarts.selectorLogicalStart source := by
  have phaseLocal :
      Spartan.piCcsPhaseOffset ≤ PiRLCStarts.phaseLogicalStart := by
    rw [PiRLCStarts.phaseLogicalStart_eq]
    norm_num [Spartan.piCcsPhaseOffset]
  have selectorAfterPhase :
      PiRLCStarts.phaseLogicalStart ≤
        PiRLCStarts.selectorLogicalStart source := by
    unfold PiRLCStarts.selectorLogicalStart
      PiRLCStarts.samplerSourceLogicalStart PiRLCStarts.samplerLogicalStart
      Formal.samplerOffset
    omega
  exact phaseLocal.trans selectorAfterPhase

theorem previousPositionSourceStart_local (source round : Nat) :
    Spartan.piCcsPhaseOffset ≤ previousPositionSourceStart source round := by
  have selectorLocal := selectorLogicalStart_local source
  unfold previousPositionSourceStart positionSourceStart First54.positionOffset
  omega

theorem previousValueSourceStart_local (source round : Nat) :
    Spartan.piCcsPhaseOffset ≤ previousValueSourceStart source round := by
  have selectorLocal := selectorLogicalStart_local source
  unfold previousValueSourceStart valueSourceStart First54.valueOffset
    First54.positionOffset
  omega

theorem finalColumn_previousPosition_add (source round offset : Nat) :
    finalColumn (previousPositionSourceStart source round) + offset =
      finalColumn (previousPositionSourceStart source round + offset) := by
  symm
  exact Spartan.sourceToSpartan_add_of_piCcsLocal _ _
    (previousPositionSourceStart_local source round)

theorem finalColumn_previousValue_add (source round offset : Nat) :
    finalColumn (previousValueSourceStart source round) + offset =
      finalColumn (previousValueSourceStart source round + offset) := by
  symm
  exact Spartan.sourceToSpartan_add_of_piCcsLocal _ _
    (previousValueSourceStart_local source round)

theorem selectorFreshStart_local (source : Nat) :
    Spartan.piCcsPhaseOffset ≤ PiRLCStarts.selectorFreshStart source := by
  have phaseLocal :
      Spartan.piCcsPhaseOffset ≤ PiRLCStarts.phaseLogicalStart := by
    rw [PiRLCStarts.phaseLogicalStart_eq]
    norm_num [Spartan.piCcsPhaseOffset]
  have selectorAfterPhase :
      PiRLCStarts.phaseLogicalStart ≤ PiRLCStarts.selectorFreshStart source := by
    unfold PiRLCStarts.selectorFreshStart PiRLCStarts.samplerSourceFreshStart
      PiRLCStarts.samplerFreshStart PiRLCStarts.phaseFreshStart
    omega
  exact phaseLocal.trans selectorAfterPhase

theorem positionInvocation_localBound (source round slot : Nat) :
    PiRLCFirst54Templates.laterPositionInputCount ≤
      (positionInvocation source round slot).localStart := by
  have sourceLocal : Spartan.piCcsPhaseOffset ≤
      PiRLCStarts.selectorFreshStart source + roundFreshPrefix round +
        positionFreshPrefix round slot := by
    have selectorLocal := selectorFreshStart_local source
    omega
  have mappedLocal := Spartan.piCcsLocalStart_le_sourceToSpartan _ sourceLocal
  change PiRLCFirst54Templates.laterPositionInputCount ≤
    finalColumn (PiRLCStarts.selectorFreshStart source +
      roundFreshPrefix round + positionFreshPrefix round slot)
  unfold finalColumn
  exact (by
    calc
      PiRLCFirst54Templates.laterPositionInputCount ≤
          Spartan.piCcsLocalStart := by
            norm_num [PiRLCFirst54Templates.laterPositionInputCount,
              Spartan.piCcsLocalStart]
      _ ≤ Spartan.sourceToSpartan
          (PiRLCStarts.selectorFreshStart source + roundFreshPrefix round +
            positionFreshPrefix round slot) := mappedLocal)

theorem valueInvocation_localBound (source round slot : Nat) :
    PiRLCFirst54Templates.laterValueInputCount ≤
      (valueInvocation source round slot).localStart := by
  have sourceLocal : Spartan.piCcsPhaseOffset ≤
      PiRLCStarts.selectorFreshStart source + roundFreshPrefix round +
        valueFreshPrefix round slot := by
    have selectorLocal := selectorFreshStart_local source
    omega
  have mappedLocal := Spartan.piCcsLocalStart_le_sourceToSpartan _ sourceLocal
  change PiRLCFirst54Templates.laterValueInputCount ≤
    finalColumn (PiRLCStarts.selectorFreshStart source +
      roundFreshPrefix round + valueFreshPrefix round slot)
  unfold finalColumn
  exact (by
    calc
      PiRLCFirst54Templates.laterValueInputCount ≤
          Spartan.piCcsLocalStart := by
            norm_num [PiRLCFirst54Templates.laterValueInputCount,
              Spartan.piCcsLocalStart]
      _ ≤ Spartan.sourceToSpartan
          (PiRLCStarts.selectorFreshStart source + roundFreshPrefix round +
            valueFreshPrefix round slot) := mappedLocal)

theorem firstPositionInputColumn_eq (source : Nat)
    (slot : Fin First54Step.slotCount) (input : Nat)
    (inputLt : input < PiRLCFirst54Templates.firstPositionInputCount) :
    compactInputColumn (firstPositionInputRanges source 0 slot.val) input =
      finalColumn (firstPositionSourceInput source slot input) := by
  have bound : input < 2 := by
    simpa [PiRLCFirst54Templates.firstPositionInputCount] using inputLt
  interval_cases input <;>
    simp [compactInputColumn,
      firstPositionInputRanges, firstPositionSourceInput]

set_option maxRecDepth 100000 in -- fixed-size: 57 position inputs
theorem laterPositionInputColumn_eq (source round : Nat)
    (slot : Fin First54Step.slotCount) (input : Nat)
    (inputLt : input < PiRLCFirst54Templates.laterPositionInputCount) :
    compactInputColumn
        (laterPositionInputRanges source round slot.val) input =
      finalColumn (laterPositionSourceInput source round slot input) := by
  have bound : input < 57 := by
    simpa [PiRLCFirst54Templates.laterPositionInputCount] using inputLt
  interval_cases input <;>
    simp [compactInputColumn,
      laterPositionInputRanges, laterPositionSourceInput,
      PiRLCFirst54Templates.laterPositionOutputInput,
      First54Step.slotCount, finalColumn_previousPosition_add]

theorem firstValueInputColumn_eq (source : Nat)
    (slot : Fin First54ValueStep.outputCount) (input : Nat)
    (inputLt : input < PiRLCFirst54Templates.firstValueInputCount) :
    compactInputColumn (firstValueInputRanges source 0 slot.val) input =
      finalColumn (firstValueSourceInput source slot input) := by
  have bound : input < 3 := by
    simpa [PiRLCFirst54Templates.firstValueInputCount] using inputLt
  interval_cases input <;>
    simp [compactInputColumn, firstValueInputRanges,
      firstValueSourceInput, PiRLCFirst54Templates.firstValueOutputInput]

set_option maxRecDepth 100000 in -- fixed-size: 112 value inputs
theorem laterValueInputColumn_eq (source round : Nat)
    (slot : Fin First54ValueStep.outputCount) (input : Nat)
    (inputLt : input < PiRLCFirst54Templates.laterValueInputCount) :
    compactInputColumn (laterValueInputRanges source round slot.val) input =
      finalColumn (laterValueSourceInput source round slot input) := by
  have bound : input < 112 := by
    simpa [PiRLCFirst54Templates.laterValueInputCount] using inputLt
  interval_cases input <;>
    simp [compactInputColumn, laterValueInputRanges,
      laterValueSourceInput,
      PiRLCFirst54Templates.laterValuePriorPositionStart,
      PiRLCFirst54Templates.laterValuePriorOutputStart,
      PiRLCFirst54Templates.laterValueOutputInput,
      First54Step.slotCount, First54ValueStep.outputCount,
      finalColumn_previousPosition_add, finalColumn_previousValue_add]

def compactEvalEnv (inputCount localStart : Nat)
    (ranges : List CompactInputRange) (env : Env) : Env :=
  fun input => env
    (CompactRows.relocate inputCount (localStart - inputCount)
      (compactInputColumn ranges) input)

theorem firstPositionCompactEvalEnv_eq (source localStart : Nat)
    (slot : Fin First54Step.slotCount) (input : Nat)
    (inputLt : input < PiRLCFirst54Templates.firstPositionInputCount)
    (env : Env) :
    compactEvalEnv PiRLCFirst54Templates.firstPositionInputCount localStart
        (firstPositionInputRanges source 0 slot.val) env input =
      Spartan.pullback env (firstPositionSourceInput source slot input) := by
  unfold compactEvalEnv
  rw [CompactRows.relocate_input _ _ _ _ inputLt,
    firstPositionInputColumn_eq source slot input inputLt]
  rfl

theorem laterPositionCompactEvalEnv_eq (source round localStart : Nat)
    (slot : Fin First54Step.slotCount) (input : Nat)
    (inputLt : input < PiRLCFirst54Templates.laterPositionInputCount)
    (env : Env) :
    compactEvalEnv PiRLCFirst54Templates.laterPositionInputCount localStart
        (laterPositionInputRanges source round slot.val) env input =
      Spartan.pullback env
        (laterPositionSourceInput source round slot input) := by
  unfold compactEvalEnv
  rw [CompactRows.relocate_input _ _ _ _ inputLt,
    laterPositionInputColumn_eq source round slot input inputLt]
  rfl

theorem firstValueCompactEvalEnv_eq (source localStart : Nat)
    (slot : Fin First54ValueStep.outputCount) (input : Nat)
    (inputLt : input < PiRLCFirst54Templates.firstValueInputCount)
    (env : Env) :
    compactEvalEnv PiRLCFirst54Templates.firstValueInputCount localStart
        (firstValueInputRanges source 0 slot.val) env input =
      Spartan.pullback env (firstValueSourceInput source slot input) := by
  unfold compactEvalEnv
  rw [CompactRows.relocate_input _ _ _ _ inputLt,
    firstValueInputColumn_eq source slot input inputLt]
  rfl

theorem laterValueCompactEvalEnv_eq (source round localStart : Nat)
    (slot : Fin First54ValueStep.outputCount) (input : Nat)
    (inputLt : input < PiRLCFirst54Templates.laterValueInputCount)
    (env : Env) :
    compactEvalEnv PiRLCFirst54Templates.laterValueInputCount localStart
        (laterValueInputRanges source round slot.val) env input =
      Spartan.pullback env
        (laterValueSourceInput source round slot input) := by
  unfold compactEvalEnv
  rw [CompactRows.relocate_input _ _ _ _ inputLt,
    laterValueInputColumn_eq source round slot input inputLt]
  rfl

theorem positionRecipe_eval_congr
    (leftInterface rightInterface : First54Step.Interface)
    (leftOffset rightOffset : Nat) (leftEnv rightEnv : Env)
    (slot : Fin First54Step.slotCount)
    (accepted : (leftInterface.accepted leftOffset).eval leftEnv =
      (rightInterface.accepted rightOffset).eval rightEnv)
    (prior : ∀ current,
      (leftInterface.prior leftOffset current).eval leftEnv =
        (rightInterface.prior rightOffset current).eval rightEnv) :
    (First54Step.recipe leftInterface leftOffset slot).eval leftEnv =
      (First54Step.recipe rightInterface rightOffset slot).eval rightEnv := by
  have one : (1 : Expr).eval leftEnv = (1 : Expr).eval rightEnv := rfl
  unfold First54Step.recipe
  by_cases first : slot.val = 0
  · simp only [dif_pos first, Expr.eval_hmul, Expr.eval_sub]
    rw [accepted, prior slot, one]
  · by_cases full : slot.val = First54Step.fullSlot
    · simp only [dif_neg first, dif_pos full, Expr.eval_hadd,
        Expr.eval_hmul]
      rw [accepted, prior slot,
        prior (First54Step.previousSlot slot (by omega))]
    · simp only [dif_neg first, dif_neg full, Expr.eval_hadd,
        Expr.eval_hmul, Expr.eval_sub]
      rw [accepted, prior slot,
        prior (First54Step.previousSlot slot (by omega)), one]

theorem valueRecipe_eval_congr
    (leftInterface rightInterface : First54ValueStep.Interface)
    (leftOffset rightOffset : Nat) (leftEnv rightEnv : Env)
    (slot : Fin First54ValueStep.outputCount)
    (accepted : (leftInterface.accepted leftOffset).eval leftEnv =
      (rightInterface.accepted rightOffset).eval rightEnv)
    (symbol : (leftInterface.symbol leftOffset).eval leftEnv =
      (rightInterface.symbol rightOffset).eval rightEnv)
    (priorPosition : ∀ current,
      (leftInterface.priorPosition leftOffset current).eval leftEnv =
        (rightInterface.priorPosition rightOffset current).eval rightEnv)
    (priorOutput : ∀ current,
      (leftInterface.priorOutput leftOffset current).eval leftEnv =
        (rightInterface.priorOutput rightOffset current).eval rightEnv) :
    (First54ValueStep.recipe leftInterface leftOffset slot).eval leftEnv =
      (First54ValueStep.recipe rightInterface rightOffset slot).eval
        rightEnv := by
  unfold First54ValueStep.recipe
  simp only [Expr.eval_hadd, Expr.eval_hmul]
  rw [accepted, symbol, priorPosition, priorOutput]

theorem positionConstraints_imply_spec (interface : First54Step.Interface)
    (offset : Nat) (env : Env)
    (constraints : ∀ slot,
      (First54Step.output offset slot -
        First54Step.recipe interface offset slot).eval env = 0) :
    First54Step.SpecHolds interface offset env := by
  intro slot
  have constraint := constraints slot
  rw [Expr.eval_sub] at constraint
  have outputRecipe := sub_eq_zero.mp constraint
  rw [outputRecipe]
  have evalOne : (1 : Expr).eval env = (1 : F) := rfl
  unfold First54Step.recipe First54Step.update
  by_cases first : slot.val = 0
  · simp only [dif_pos first, Expr.eval_hmul, Expr.eval_sub, evalOne]
  · by_cases full : slot.val = First54Step.fullSlot
    · simp only [dif_neg first, dif_pos full, Expr.eval_hadd,
        Expr.eval_hmul]
    · simp only [dif_neg first, dif_neg full, Expr.eval_hadd,
        Expr.eval_hmul, Expr.eval_sub, evalOne]

theorem valueConstraints_imply_spec (interface : First54ValueStep.Interface)
    (offset : Nat) (env : Env)
    (constraints : ∀ slot,
      (First54ValueStep.output offset slot -
        First54ValueStep.recipe interface offset slot).eval env = 0) :
    First54ValueStep.SpecHolds interface offset env := by
  intro slot
  have constraint := constraints slot
  rw [Expr.eval_sub] at constraint
  have outputRecipe := sub_eq_zero.mp constraint
  rw [outputRecipe]
  rfl

def exactPositionInterface (source round : Nat) :
    First54Step.Interface :=
  First54.positionInterface
    (selectorInterface (logicalWidth := logicalWidth)
      (publicFits := publicFits) source)
    (PiRLCStarts.selectorLogicalStart source) round

def exactValueInterface (source round : Nat) :
    First54ValueStep.Interface :=
  First54.valueInterface
    (selectorInterface (logicalWidth := logicalWidth)
      (publicFits := publicFits) source)
    (PiRLCStarts.selectorLogicalStart source) round

theorem exactPositionAccepted_eq (source round : Nat)
    (roundLt : round < First54.candidateCount) :
    (exactPositionInterface (logicalWidth := logicalWidth)
        (publicFits := publicFits) source round).accepted
        (positionSourceStart source round) =
      PiRLCFirst54Templates.acceptedFromReject
        (rejectSourceColumn source round) := by
  unfold exactPositionInterface First54.positionInterface selectorInterface
    Sampler.selectorInterface PiRLCFirst54Templates.acceptedFromReject
  rw [rejectSourceColumn_eq source round roundLt]
  rfl

theorem exactValueAccepted_eq (source round : Nat)
    (roundLt : round < First54.candidateCount) :
    (exactValueInterface (logicalWidth := logicalWidth)
        (publicFits := publicFits) source round).accepted
        (valueSourceStart source round) =
      PiRLCFirst54Templates.acceptedFromReject
        (rejectSourceColumn source round) := by
  unfold exactValueInterface First54.valueInterface selectorInterface
    Sampler.selectorInterface PiRLCFirst54Templates.acceptedFromReject
  rw [rejectSourceColumn_eq source round roundLt]
  rfl

theorem exactValueSymbol_eq (source round : Nat)
    (roundLt : round < First54.candidateCount) :
    (exactValueInterface (logicalWidth := logicalWidth)
        (publicFits := publicFits) source round).symbol
        (valueSourceStart source round) =
      Expr.var (remainderSourceColumn source round) := by
  unfold exactValueInterface First54.valueInterface selectorInterface
    Sampler.selectorInterface
  rw [remainderSourceColumn_eq source round roundLt]
  rfl

theorem exactPositionPrior_zero_eq (source : Nat)
    (slot : Fin First54Step.slotCount) :
    (exactPositionInterface (logicalWidth := logicalWidth)
        (publicFits := publicFits) source 0).prior
        (positionSourceStart source 0) slot =
      First54.initialPosition slot := by
  rfl

theorem exactPositionPrior_later_eq (source round : Nat)
    (roundPos : 0 < round) (slot : Fin First54Step.slotCount) :
    (exactPositionInterface (logicalWidth := logicalWidth)
        (publicFits := publicFits) source round).prior
        (positionSourceStart source round) slot =
      Expr.var (previousPositionSourceStart source round + slot.val) := by
  cases round with
  | zero => omega
  | succ previous => rfl

theorem exactValuePriorPosition_zero_eq (source : Nat)
    (slot : Fin First54Step.slotCount) :
    (exactValueInterface (logicalWidth := logicalWidth)
        (publicFits := publicFits) source 0).priorPosition
        (valueSourceStart source 0) slot =
      First54.initialPosition slot := by
  rfl

theorem exactValuePriorOutput_zero_eq (source : Nat)
    (slot : Fin First54ValueStep.outputCount) :
    (exactValueInterface (logicalWidth := logicalWidth)
        (publicFits := publicFits) source 0).priorOutput
        (valueSourceStart source 0) slot = 0 := by
  rfl

theorem exactValuePriorPosition_later_eq (source round : Nat)
    (roundPos : 0 < round) (slot : Fin First54Step.slotCount) :
    (exactValueInterface (logicalWidth := logicalWidth)
        (publicFits := publicFits) source round).priorPosition
        (valueSourceStart source round) slot =
      Expr.var (previousPositionSourceStart source round + slot.val) := by
  cases round with
  | zero => omega
  | succ previous => rfl

theorem exactValuePriorOutput_later_eq (source round : Nat)
    (roundPos : 0 < round) (slot : Fin First54ValueStep.outputCount) :
    (exactValueInterface (logicalWidth := logicalWidth)
        (publicFits := publicFits) source round).priorOutput
        (valueSourceStart source round) slot =
      Expr.var (previousValueSourceStart source round + slot.val) := by
  cases round with
  | zero => omega
  | succ previous => rfl

def exactPositionConstraint (source round : Nat)
    (slot : Fin First54Step.slotCount) : Expr :=
  let offset := positionSourceStart source round
  Expr.var (offset + slot.val) -
    First54Step.recipe (exactPositionInterface
      (logicalWidth := logicalWidth) (publicFits := publicFits) source round)
      offset slot

def exactValueConstraint (source round : Nat)
    (slot : Fin First54ValueStep.outputCount) : Expr :=
  let offset := valueSourceStart source round
  Expr.var (offset + slot.val) -
    First54ValueStep.recipe (exactValueInterface
      (logicalWidth := logicalWidth) (publicFits := publicFits) source round)
      offset slot

theorem firstPositionConstraint_eval_eq (source localStart : Nat)
    (slot : Fin First54Step.slotCount) (env : Env) :
    (Expr.var PiRLCFirst54Templates.firstPositionOutputInput -
        PiRLCFirst54Templates.firstPositionRecipe slot).eval
        (compactEvalEnv PiRLCFirst54Templates.firstPositionInputCount
          localStart (firstPositionInputRanges source 0 slot.val) env) =
      (exactPositionConstraint (logicalWidth := logicalWidth)
        (publicFits := publicFits) source 0 slot).eval
        (Spartan.pullback env) := by
  let compact : Env :=
    compactEvalEnv PiRLCFirst54Templates.firstPositionInputCount
      localStart (firstPositionInputRanges source 0 slot.val) env
  let exact : Env := Spartan.pullback env
  change
    (Expr.var PiRLCFirst54Templates.firstPositionOutputInput -
      First54Step.recipe PiRLCFirst54Templates.firstPositionInterface 0
        slot).eval compact =
    (Expr.var (positionSourceStart source 0 + slot.val) -
      First54Step.recipe
        (exactPositionInterface (logicalWidth := logicalWidth)
          (publicFits := publicFits) source 0)
        (positionSourceStart source 0) slot).eval exact
  have rejectValue : compact 0 = exact (rejectSourceColumn source 0) := by
    simpa [compact, exact, firstPositionSourceInput] using
      firstPositionCompactEvalEnv_eq source localStart slot 0
        (by norm_num [PiRLCFirst54Templates.firstPositionInputCount]) env
  have outputValue :
      compact PiRLCFirst54Templates.firstPositionOutputInput =
        exact (positionSourceStart source 0 + slot.val) := by
    simpa [compact, exact, firstPositionSourceInput,
      PiRLCFirst54Templates.firstPositionOutputInput] using
      firstPositionCompactEvalEnv_eq source localStart slot
        PiRLCFirst54Templates.firstPositionOutputInput
        (by norm_num [PiRLCFirst54Templates.firstPositionOutputInput,
          PiRLCFirst54Templates.firstPositionInputCount]) env
  have acceptedValue :
      (PiRLCFirst54Templates.firstPositionInterface.accepted 0).eval compact =
        ((exactPositionInterface (logicalWidth := logicalWidth)
          (publicFits := publicFits) source 0).accepted
            (positionSourceStart source 0)).eval exact := by
    rw [exactPositionAccepted_eq source 0
      (by norm_num [First54.candidateCount])]
    change (1 - Expr.var 0).eval compact =
      (1 - Expr.var (rejectSourceColumn source 0)).eval exact
    simp only [Expr.eval_sub, Expr.eval_var]
    rw [rejectValue]
    rfl
  have priorValue : ∀ current,
      (PiRLCFirst54Templates.firstPositionInterface.prior 0 current).eval
          compact =
        ((exactPositionInterface (logicalWidth := logicalWidth)
          (publicFits := publicFits) source 0).prior
            (positionSourceStart source 0) current).eval exact := by
    intro current
    rw [exactPositionPrior_zero_eq]
    change (First54.initialPosition current).eval compact =
      (First54.initialPosition current).eval exact
    unfold First54.initialPosition
    split <;> rfl
  have recipeValue := positionRecipe_eval_congr
    PiRLCFirst54Templates.firstPositionInterface
    (exactPositionInterface (logicalWidth := logicalWidth)
      (publicFits := publicFits) source 0)
    0 (positionSourceStart source 0) compact exact slot acceptedValue priorValue
  simp only [Expr.eval_sub, Expr.eval_var]
  rw [outputValue, recipeValue]

theorem laterPositionConstraint_eval_eq (source round localStart : Nat)
    (roundPos : 0 < round) (roundLt : round < First54.candidateCount)
    (slot : Fin First54Step.slotCount) (env : Env) :
    (Expr.var PiRLCFirst54Templates.laterPositionOutputInput -
        PiRLCFirst54Templates.laterPositionRecipe slot).eval
        (compactEvalEnv PiRLCFirst54Templates.laterPositionInputCount
          localStart (laterPositionInputRanges source round slot.val) env) =
      (exactPositionConstraint (logicalWidth := logicalWidth)
        (publicFits := publicFits) source round slot).eval
        (Spartan.pullback env) := by
  let compact : Env :=
    compactEvalEnv PiRLCFirst54Templates.laterPositionInputCount
      localStart (laterPositionInputRanges source round slot.val) env
  let exact : Env := Spartan.pullback env
  change
    (Expr.var PiRLCFirst54Templates.laterPositionOutputInput -
      First54Step.recipe PiRLCFirst54Templates.laterPositionInterface 0
        slot).eval compact =
    (Expr.var (positionSourceStart source round + slot.val) -
      First54Step.recipe
        (exactPositionInterface (logicalWidth := logicalWidth)
          (publicFits := publicFits) source round)
        (positionSourceStart source round) slot).eval exact
  have rejectValue : compact 0 = exact (rejectSourceColumn source round) := by
    simpa [compact, exact, laterPositionSourceInput] using
      laterPositionCompactEvalEnv_eq source round localStart slot 0
        (by norm_num [PiRLCFirst54Templates.laterPositionInputCount]) env
  have outputValue :
      compact PiRLCFirst54Templates.laterPositionOutputInput =
        exact (positionSourceStart source round + slot.val) := by
    simpa [compact, exact, laterPositionSourceInput,
      PiRLCFirst54Templates.laterPositionOutputInput,
      First54Step.slotCount] using
      laterPositionCompactEvalEnv_eq source round localStart slot
        PiRLCFirst54Templates.laterPositionOutputInput
        (by norm_num [PiRLCFirst54Templates.laterPositionOutputInput,
          PiRLCFirst54Templates.laterPositionInputCount]) env
  have acceptedValue :
      (PiRLCFirst54Templates.laterPositionInterface.accepted 0).eval compact =
        ((exactPositionInterface (logicalWidth := logicalWidth)
          (publicFits := publicFits) source round).accepted
            (positionSourceStart source round)).eval exact := by
    rw [exactPositionAccepted_eq source round roundLt]
    change (1 - Expr.var 0).eval compact =
      (1 - Expr.var (rejectSourceColumn source round)).eval exact
    simp only [Expr.eval_sub, Expr.eval_var]
    rw [rejectValue]
    rfl
  have priorValue : ∀ current,
      (PiRLCFirst54Templates.laterPositionInterface.prior 0 current).eval
          compact =
        ((exactPositionInterface (logicalWidth := logicalWidth)
          (publicFits := publicFits) source round).prior
            (positionSourceStart source round) current).eval exact := by
    intro current
    have currentBound := current.isLt
    have inputBound :
        1 + current.val < PiRLCFirst54Templates.laterPositionInputCount := by
      norm_num [PiRLCFirst54Templates.laterPositionInputCount,
        First54Step.slotCount] at currentBound ⊢
      omega
    have priorRange : 1 + current.val < 56 := by
      norm_num [First54Step.slotCount] at currentBound
      omega
    have mapped := laterPositionCompactEvalEnv_eq source round localStart
      slot (1 + current.val) inputBound env
    have priorMapped : compact (1 + current.val) =
        exact (previousPositionSourceStart source round + current.val) := by
      simpa [compact, exact, laterPositionSourceInput,
        First54Step.slotCount, priorRange] using mapped
    rw [exactPositionPrior_later_eq source round roundPos current]
    change compact (PiRLCFirst54Templates.laterPositionPriorStart +
        current.val) =
      exact (previousPositionSourceStart source round + current.val)
    simpa [PiRLCFirst54Templates.laterPositionPriorStart] using priorMapped
  have recipeValue := positionRecipe_eval_congr
    PiRLCFirst54Templates.laterPositionInterface
    (exactPositionInterface (logicalWidth := logicalWidth)
      (publicFits := publicFits) source round)
    0 (positionSourceStart source round) compact exact slot acceptedValue
      priorValue
  simp only [Expr.eval_sub, Expr.eval_var]
  rw [outputValue, recipeValue]

theorem firstValueConstraint_eval_eq (source localStart : Nat)
    (slot : Fin First54ValueStep.outputCount) (env : Env) :
    (Expr.var PiRLCFirst54Templates.firstValueOutputInput -
        PiRLCFirst54Templates.firstValueRecipe slot).eval
        (compactEvalEnv PiRLCFirst54Templates.firstValueInputCount
          localStart (firstValueInputRanges source 0 slot.val) env) =
      (exactValueConstraint (logicalWidth := logicalWidth)
        (publicFits := publicFits) source 0 slot).eval
        (Spartan.pullback env) := by
  let compact : Env :=
    compactEvalEnv PiRLCFirst54Templates.firstValueInputCount
      localStart (firstValueInputRanges source 0 slot.val) env
  let exact : Env := Spartan.pullback env
  change
    (Expr.var PiRLCFirst54Templates.firstValueOutputInput -
      First54ValueStep.recipe PiRLCFirst54Templates.firstValueInterface 0
        slot).eval compact =
    (Expr.var (valueSourceStart source 0 + slot.val) -
      First54ValueStep.recipe
        (exactValueInterface (logicalWidth := logicalWidth)
          (publicFits := publicFits) source 0)
        (valueSourceStart source 0) slot).eval exact
  have rejectValue : compact 0 = exact (rejectSourceColumn source 0) := by
    simpa [compact, exact, firstValueSourceInput] using
      firstValueCompactEvalEnv_eq source localStart slot 0
        (by norm_num [PiRLCFirst54Templates.firstValueInputCount]) env
  have symbolInputValue : compact 1 =
      exact (remainderSourceColumn source 0) := by
    simpa [compact, exact, firstValueSourceInput] using
      firstValueCompactEvalEnv_eq source localStart slot 1
        (by norm_num [PiRLCFirst54Templates.firstValueInputCount]) env
  have outputValue :
      compact PiRLCFirst54Templates.firstValueOutputInput =
        exact (valueSourceStart source 0 + slot.val) := by
    simpa [compact, exact, firstValueSourceInput,
      PiRLCFirst54Templates.firstValueOutputInput] using
      firstValueCompactEvalEnv_eq source localStart slot
        PiRLCFirst54Templates.firstValueOutputInput
        (by norm_num [PiRLCFirst54Templates.firstValueOutputInput,
          PiRLCFirst54Templates.firstValueInputCount]) env
  have acceptedValue :
      (PiRLCFirst54Templates.firstValueInterface.accepted 0).eval compact =
        ((exactValueInterface (logicalWidth := logicalWidth)
          (publicFits := publicFits) source 0).accepted
            (valueSourceStart source 0)).eval exact := by
    rw [exactValueAccepted_eq source 0
      (by norm_num [First54.candidateCount])]
    change (1 - Expr.var 0).eval compact =
      (1 - Expr.var (rejectSourceColumn source 0)).eval exact
    simp only [Expr.eval_sub, Expr.eval_var]
    rw [rejectValue]
    rfl
  have symbolValue :
      (PiRLCFirst54Templates.firstValueInterface.symbol 0).eval compact =
        ((exactValueInterface (logicalWidth := logicalWidth)
          (publicFits := publicFits) source 0).symbol
            (valueSourceStart source 0)).eval exact := by
    rw [exactValueSymbol_eq source 0
      (by norm_num [First54.candidateCount])]
    exact symbolInputValue
  have priorPositionValue : ∀ current,
      (PiRLCFirst54Templates.firstValueInterface.priorPosition 0 current).eval
          compact =
        ((exactValueInterface (logicalWidth := logicalWidth)
          (publicFits := publicFits) source 0).priorPosition
            (valueSourceStart source 0) current).eval exact := by
    intro current
    rw [exactValuePriorPosition_zero_eq]
    change (First54.initialPosition current).eval compact =
      (First54.initialPosition current).eval exact
    unfold First54.initialPosition
    split <;> rfl
  have priorOutputValue : ∀ current,
      (PiRLCFirst54Templates.firstValueInterface.priorOutput 0 current).eval
          compact =
        ((exactValueInterface (logicalWidth := logicalWidth)
          (publicFits := publicFits) source 0).priorOutput
            (valueSourceStart source 0) current).eval exact := by
    intro current
    rw [exactValuePriorOutput_zero_eq]
    rfl
  have recipeValue := valueRecipe_eval_congr
    PiRLCFirst54Templates.firstValueInterface
    (exactValueInterface (logicalWidth := logicalWidth)
      (publicFits := publicFits) source 0)
    0 (valueSourceStart source 0) compact exact slot acceptedValue symbolValue
      priorPositionValue priorOutputValue
  simp only [Expr.eval_sub, Expr.eval_var]
  rw [outputValue, recipeValue]

theorem laterValueConstraint_eval_eq (source round localStart : Nat)
    (roundPos : 0 < round) (roundLt : round < First54.candidateCount)
    (slot : Fin First54ValueStep.outputCount) (env : Env) :
    (Expr.var PiRLCFirst54Templates.laterValueOutputInput -
        PiRLCFirst54Templates.laterValueRecipe slot).eval
        (compactEvalEnv PiRLCFirst54Templates.laterValueInputCount
          localStart (laterValueInputRanges source round slot.val) env) =
      (exactValueConstraint (logicalWidth := logicalWidth)
        (publicFits := publicFits) source round slot).eval
        (Spartan.pullback env) := by
  let compact : Env :=
    compactEvalEnv PiRLCFirst54Templates.laterValueInputCount
      localStart (laterValueInputRanges source round slot.val) env
  let exact : Env := Spartan.pullback env
  change
    (Expr.var PiRLCFirst54Templates.laterValueOutputInput -
      First54ValueStep.recipe PiRLCFirst54Templates.laterValueInterface 0
        slot).eval compact =
    (Expr.var (valueSourceStart source round + slot.val) -
      First54ValueStep.recipe
        (exactValueInterface (logicalWidth := logicalWidth)
          (publicFits := publicFits) source round)
        (valueSourceStart source round) slot).eval exact
  have rejectValue : compact 0 = exact (rejectSourceColumn source round) := by
    simpa [compact, exact, laterValueSourceInput] using
      laterValueCompactEvalEnv_eq source round localStart slot 0
        (by norm_num [PiRLCFirst54Templates.laterValueInputCount]) env
  have symbolInputValue : compact 1 =
      exact (remainderSourceColumn source round) := by
    simpa [compact, exact, laterValueSourceInput] using
      laterValueCompactEvalEnv_eq source round localStart slot 1
        (by norm_num [PiRLCFirst54Templates.laterValueInputCount]) env
  have outputValue :
      compact PiRLCFirst54Templates.laterValueOutputInput =
        exact (valueSourceStart source round + slot.val) := by
    simpa [compact, exact, laterValueSourceInput,
      PiRLCFirst54Templates.laterValuePriorPositionStart,
      PiRLCFirst54Templates.laterValuePriorOutputStart,
      PiRLCFirst54Templates.laterValueOutputInput] using
      laterValueCompactEvalEnv_eq source round localStart slot
        PiRLCFirst54Templates.laterValueOutputInput
        (by norm_num [PiRLCFirst54Templates.laterValueOutputInput,
          PiRLCFirst54Templates.laterValueInputCount]) env
  have acceptedValue :
      (PiRLCFirst54Templates.laterValueInterface.accepted 0).eval compact =
        ((exactValueInterface (logicalWidth := logicalWidth)
          (publicFits := publicFits) source round).accepted
            (valueSourceStart source round)).eval exact := by
    rw [exactValueAccepted_eq source round roundLt]
    change (1 - Expr.var 0).eval compact =
      (1 - Expr.var (rejectSourceColumn source round)).eval exact
    simp only [Expr.eval_sub, Expr.eval_var]
    rw [rejectValue]
    rfl
  have symbolValue :
      (PiRLCFirst54Templates.laterValueInterface.symbol 0).eval compact =
        ((exactValueInterface (logicalWidth := logicalWidth)
          (publicFits := publicFits) source round).symbol
            (valueSourceStart source round)).eval exact := by
    rw [exactValueSymbol_eq source round roundLt]
    exact symbolInputValue
  have priorPositionValue : ∀ current,
      (PiRLCFirst54Templates.laterValueInterface.priorPosition 0 current).eval
          compact =
        ((exactValueInterface (logicalWidth := logicalWidth)
          (publicFits := publicFits) source round).priorPosition
            (valueSourceStart source round) current).eval exact := by
    intro current
    have currentBound := current.isLt
    have inputBound :
        2 + current.val < PiRLCFirst54Templates.laterValueInputCount := by
      norm_num [PiRLCFirst54Templates.laterValueInputCount,
        First54Step.slotCount] at currentBound ⊢
      omega
    have priorRange : 2 + current.val < 57 := by
      norm_num [First54Step.slotCount] at currentBound
      omega
    have notSymbol : 2 + current.val ≠ 1 := by omega
    have mapped := laterValueCompactEvalEnv_eq source round localStart slot
      (2 + current.val) inputBound env
    have priorMapped : compact (2 + current.val) =
        exact (previousPositionSourceStart source round + current.val) := by
      simpa [compact, exact, laterValueSourceInput,
        PiRLCFirst54Templates.laterValuePriorPositionStart,
        PiRLCFirst54Templates.laterValuePriorOutputStart, priorRange,
        notSymbol] using mapped
    rw [exactValuePriorPosition_later_eq source round roundPos current]
    change compact (PiRLCFirst54Templates.laterValuePriorPositionStart +
        current.val) =
      exact (previousPositionSourceStart source round + current.val)
    simpa [PiRLCFirst54Templates.laterValuePriorPositionStart] using
      priorMapped
  have priorOutputValue : ∀ current,
      (PiRLCFirst54Templates.laterValueInterface.priorOutput 0 current).eval
          compact =
        ((exactValueInterface (logicalWidth := logicalWidth)
          (publicFits := publicFits) source round).priorOutput
            (valueSourceStart source round) current).eval exact := by
    intro current
    have currentBound := current.isLt
    have inputBound :
        57 + current.val < PiRLCFirst54Templates.laterValueInputCount := by
      norm_num [PiRLCFirst54Templates.laterValueInputCount,
        First54ValueStep.outputCount] at currentBound ⊢
      omega
    have outputRange : 57 + current.val < 111 := by
      norm_num [First54ValueStep.outputCount] at currentBound
      omega
    have notSymbol : 57 + current.val ≠ 1 := by omega
    have mapped := laterValueCompactEvalEnv_eq source round localStart slot
      (57 + current.val) inputBound env
    have priorMapped : compact (57 + current.val) =
        exact (previousValueSourceStart source round + current.val) := by
      simpa [compact, exact, laterValueSourceInput,
        PiRLCFirst54Templates.laterValuePriorPositionStart,
        PiRLCFirst54Templates.laterValuePriorOutputStart,
        PiRLCFirst54Templates.laterValueOutputInput, outputRange,
        notSymbol] using mapped
    rw [exactValuePriorOutput_later_eq source round roundPos current]
    change compact (PiRLCFirst54Templates.laterValuePriorOutputStart +
        current.val) =
      exact (previousValueSourceStart source round + current.val)
    simpa [PiRLCFirst54Templates.laterValuePriorOutputStart] using priorMapped
  have recipeValue := valueRecipe_eval_congr
    PiRLCFirst54Templates.laterValueInterface
    (exactValueInterface (logicalWidth := logicalWidth)
      (publicFits := publicFits) source round)
    0 (valueSourceStart source round) compact exact slot acceptedValue
      symbolValue priorPositionValue priorOutputValue
  simp only [Expr.eval_sub, Expr.eval_var]
  rw [outputValue, recipeValue]

theorem positionInvocation_zero_implies_constraint
    (package : CircuitPackage)
    (templates : package.compactRowTemplates = packageTemplates)
    (source : Nat) (slot : Fin First54Step.slotCount) (env : Env)
    (holds : CompactRowInvocationHolds package
      (positionInvocation source 0 slot.val) env) :
    (exactPositionConstraint (logicalWidth := logicalWidth)
      (publicFits := publicFits) source 0 slot).eval
        (Spartan.pullback env) = 0 := by
  have packageRows := holds
  unfold CompactRowInvocationHolds at packageRows
  rw [templates, positionInvocation_zero_template source slot] at packageRows
  have instantiatedRows : R1CS.RowsHold env
      (CompactRows.instantiateRows
        (compactInputColumn
          (positionInvocation source 0 slot.val).inputRanges)
        (positionInvocation source 0 slot.val).localStart
        (PiRLCFirst54Templates.firstPositionTemplate slot)) := by
    rw [CompactRows.instantiateRows_eq_package]
    exact packageRows
  have localBound : PiRLCFirst54Templates.firstPositionInputCount ≤
      (positionInvocation source 0 slot.val).localStart := by
    have maximum := positionInvocation_localBound source 0 slot.val
    norm_num [PiRLCFirst54Templates.firstPositionInputCount,
      PiRLCFirst54Templates.laterPositionInputCount] at maximum ⊢
    omega
  have normalized :=
    CompactRows.compactConstraintTemplate_rows_imply_eval_zero
      PiRLCFirst54Templates.firstPositionInputCount
      PiRLCFirst54Templates.firstPositionOutputInput
      (positionInvocation source 0 slot.val).localStart
      (compactInputColumn
        (positionInvocation source 0 slot.val).inputRanges)
      (PiRLCFirst54Templates.firstPositionRecipe slot) env localBound
      (by
        simpa [PiRLCFirst54Templates.firstPositionTemplate] using
          instantiatedRows)
  change
    (Expr.var PiRLCFirst54Templates.firstPositionOutputInput -
      PiRLCFirst54Templates.firstPositionRecipe slot).eval
      (compactEvalEnv PiRLCFirst54Templates.firstPositionInputCount
        (positionInvocation source 0 slot.val).localStart
        (positionInvocation source 0 slot.val).inputRanges env) = 0 at normalized
  simpa [positionInvocation] using
    (firstPositionConstraint_eval_eq
      (logicalWidth := logicalWidth) (publicFits := publicFits) source
      (positionInvocation source 0 slot.val).localStart slot env).symm.trans
        normalized

theorem positionInvocation_succ_implies_constraint
    (package : CircuitPackage)
    (templates : package.compactRowTemplates = packageTemplates)
    (source round : Nat) (roundLt : round + 1 < First54.candidateCount)
    (slot : Fin First54Step.slotCount) (env : Env)
    (holds : CompactRowInvocationHolds package
      (positionInvocation source (round + 1) slot.val) env) :
    (exactPositionConstraint (logicalWidth := logicalWidth)
      (publicFits := publicFits) source (round + 1) slot).eval
        (Spartan.pullback env) = 0 := by
  have packageRows := holds
  unfold CompactRowInvocationHolds at packageRows
  rw [templates, positionInvocation_succ_template source round slot]
    at packageRows
  have instantiatedRows : R1CS.RowsHold env
      (CompactRows.instantiateRows
        (compactInputColumn
          (positionInvocation source (round + 1) slot.val).inputRanges)
        (positionInvocation source (round + 1) slot.val).localStart
        (PiRLCFirst54Templates.laterPositionTemplate slot)) := by
    rw [CompactRows.instantiateRows_eq_package]
    exact packageRows
  have localBound : PiRLCFirst54Templates.laterPositionInputCount ≤
      (positionInvocation source (round + 1) slot.val).localStart :=
    positionInvocation_localBound source (round + 1) slot.val
  have normalized :=
    CompactRows.compactConstraintTemplate_rows_imply_eval_zero
      PiRLCFirst54Templates.laterPositionInputCount
      PiRLCFirst54Templates.laterPositionOutputInput
      (positionInvocation source (round + 1) slot.val).localStart
      (compactInputColumn
        (positionInvocation source (round + 1) slot.val).inputRanges)
      (PiRLCFirst54Templates.laterPositionRecipe slot) env localBound
      (by
        simpa [PiRLCFirst54Templates.laterPositionTemplate] using
          instantiatedRows)
  change
    (Expr.var PiRLCFirst54Templates.laterPositionOutputInput -
      PiRLCFirst54Templates.laterPositionRecipe slot).eval
      (compactEvalEnv PiRLCFirst54Templates.laterPositionInputCount
        (positionInvocation source (round + 1) slot.val).localStart
        (positionInvocation source (round + 1) slot.val).inputRanges env) = 0
    at normalized
  simpa [positionInvocation] using
    (laterPositionConstraint_eval_eq
      (logicalWidth := logicalWidth) (publicFits := publicFits) source
      (round + 1)
      (positionInvocation source (round + 1) slot.val).localStart
      (by omega) roundLt slot env).symm.trans normalized

theorem valueInvocation_zero_implies_constraint
    (package : CircuitPackage)
    (templates : package.compactRowTemplates = packageTemplates)
    (source : Nat) (slot : Fin First54ValueStep.outputCount) (env : Env)
    (holds : CompactRowInvocationHolds package
      (valueInvocation source 0 slot.val) env) :
    (exactValueConstraint (logicalWidth := logicalWidth)
      (publicFits := publicFits) source 0 slot).eval
        (Spartan.pullback env) = 0 := by
  have packageRows := holds
  unfold CompactRowInvocationHolds at packageRows
  rw [templates, valueInvocation_zero_template source slot] at packageRows
  have instantiatedRows : R1CS.RowsHold env
      (CompactRows.instantiateRows
        (compactInputColumn (valueInvocation source 0 slot.val).inputRanges)
        (valueInvocation source 0 slot.val).localStart
        (PiRLCFirst54Templates.firstValueTemplate slot)) := by
    rw [CompactRows.instantiateRows_eq_package]
    exact packageRows
  have localBound : PiRLCFirst54Templates.firstValueInputCount ≤
      (valueInvocation source 0 slot.val).localStart := by
    have maximum := valueInvocation_localBound source 0 slot.val
    norm_num [PiRLCFirst54Templates.firstValueInputCount,
      PiRLCFirst54Templates.laterValueInputCount] at maximum ⊢
    omega
  have normalized :=
    CompactRows.compactConstraintTemplate_rows_imply_eval_zero
      PiRLCFirst54Templates.firstValueInputCount
      PiRLCFirst54Templates.firstValueOutputInput
      (valueInvocation source 0 slot.val).localStart
      (compactInputColumn (valueInvocation source 0 slot.val).inputRanges)
      (PiRLCFirst54Templates.firstValueRecipe slot) env localBound
      (by
        simpa [PiRLCFirst54Templates.firstValueTemplate] using
          instantiatedRows)
  change
    (Expr.var PiRLCFirst54Templates.firstValueOutputInput -
      PiRLCFirst54Templates.firstValueRecipe slot).eval
      (compactEvalEnv PiRLCFirst54Templates.firstValueInputCount
        (valueInvocation source 0 slot.val).localStart
        (valueInvocation source 0 slot.val).inputRanges env) = 0 at normalized
  simpa [valueInvocation] using
    (firstValueConstraint_eval_eq
      (logicalWidth := logicalWidth) (publicFits := publicFits) source
      (valueInvocation source 0 slot.val).localStart slot env).symm.trans
        normalized

theorem valueInvocation_succ_implies_constraint
    (package : CircuitPackage)
    (templates : package.compactRowTemplates = packageTemplates)
    (source round : Nat) (roundLt : round + 1 < First54.candidateCount)
    (slot : Fin First54ValueStep.outputCount) (env : Env)
    (holds : CompactRowInvocationHolds package
      (valueInvocation source (round + 1) slot.val) env) :
    (exactValueConstraint (logicalWidth := logicalWidth)
      (publicFits := publicFits) source (round + 1) slot).eval
        (Spartan.pullback env) = 0 := by
  have packageRows := holds
  unfold CompactRowInvocationHolds at packageRows
  rw [templates, valueInvocation_succ_template source round slot] at packageRows
  have instantiatedRows : R1CS.RowsHold env
      (CompactRows.instantiateRows
        (compactInputColumn
          (valueInvocation source (round + 1) slot.val).inputRanges)
        (valueInvocation source (round + 1) slot.val).localStart
        (PiRLCFirst54Templates.laterValueTemplate slot)) := by
    rw [CompactRows.instantiateRows_eq_package]
    exact packageRows
  have localBound : PiRLCFirst54Templates.laterValueInputCount ≤
      (valueInvocation source (round + 1) slot.val).localStart :=
    valueInvocation_localBound source (round + 1) slot.val
  have normalized :=
    CompactRows.compactConstraintTemplate_rows_imply_eval_zero
      PiRLCFirst54Templates.laterValueInputCount
      PiRLCFirst54Templates.laterValueOutputInput
      (valueInvocation source (round + 1) slot.val).localStart
      (compactInputColumn
        (valueInvocation source (round + 1) slot.val).inputRanges)
      (PiRLCFirst54Templates.laterValueRecipe slot) env localBound
      (by
        simpa [PiRLCFirst54Templates.laterValueTemplate] using
          instantiatedRows)
  change
    (Expr.var PiRLCFirst54Templates.laterValueOutputInput -
      PiRLCFirst54Templates.laterValueRecipe slot).eval
      (compactEvalEnv PiRLCFirst54Templates.laterValueInputCount
        (valueInvocation source (round + 1) slot.val).localStart
        (valueInvocation source (round + 1) slot.val).inputRanges env) = 0
    at normalized
  simpa [valueInvocation] using
    (laterValueConstraint_eval_eq
      (logicalWidth := logicalWidth) (publicFits := publicFits) source
      (round + 1) (valueInvocation source (round + 1) slot.val).localStart
      (by omega) roundLt slot env).symm.trans normalized

/-- The exact compact First54 invocations and the ordinary final assertion
imply the authoritative First54 relation for one bounded scalar source. -/
theorem packageInvocations_imply_spec
    (package : CircuitPackage)
    (templates : package.compactRowTemplates = packageTemplates)
    (source : Nat) (sourceLt : source < sourceCount) (env : Env)
    (compactHolds : ∀ invocation ∈ invocations,
      CompactRowInvocationHolds package invocation env)
    (ordinaryHolds : R1CS.RowsHold env
      ((PiRLCSamplerOrdinaryRows.rows
        (logicalWidth := logicalWidth) (publicFits := publicFits)).map
          Rows.CompiledRow.toR1CS)) :
    First54.SpecHolds
      (selectorInterface (logicalWidth := logicalWidth)
        (publicFits := publicFits) source)
      (PiRLCStarts.selectorLogicalStart source)
      (Spartan.pullback env) := by
  refine ⟨?_, ?_, ?_⟩
  · intro round
    change First54Step.SpecHolds
      (exactPositionInterface (logicalWidth := logicalWidth)
        (publicFits := publicFits) source round.val)
      (positionSourceStart source round.val) (Spartan.pullback env)
    apply positionConstraints_imply_spec
    intro slot
    change
      (exactPositionConstraint (logicalWidth := logicalWidth)
        (publicFits := publicFits) source round.val slot).eval
          (Spartan.pullback env) = 0
    by_cases first : round.val = 0
    · rw [first]
      apply positionInvocation_zero_implies_constraint package templates
      apply compactHolds
      apply positionInvocation_mem
      · exact sourceLt
      · norm_num [roundCount, First54.candidateCount]
    · obtain ⟨previous, previousEq⟩ := Nat.exists_eq_succ_of_ne_zero first
      rw [previousEq]
      apply positionInvocation_succ_implies_constraint package templates
        source previous
      · simpa [previousEq] using round.isLt
      · apply compactHolds
        apply positionInvocation_mem
        · exact sourceLt
        · simpa [roundCount, previousEq] using round.isLt
  · intro round
    change First54ValueStep.SpecHolds
      (exactValueInterface (logicalWidth := logicalWidth)
        (publicFits := publicFits) source round.val)
      (valueSourceStart source round.val) (Spartan.pullback env)
    apply valueConstraints_imply_spec
    intro slot
    change
      (exactValueConstraint (logicalWidth := logicalWidth)
        (publicFits := publicFits) source round.val slot).eval
          (Spartan.pullback env) = 0
    by_cases first : round.val = 0
    · rw [first]
      apply valueInvocation_zero_implies_constraint package templates
      apply compactHolds
      apply valueInvocation_mem
      · exact sourceLt
      · norm_num [roundCount, First54.candidateCount]
    · obtain ⟨previous, previousEq⟩ := Nat.exists_eq_succ_of_ne_zero first
      rw [previousEq]
      apply valueInvocation_succ_implies_constraint package templates source
        previous
      · simpa [previousEq] using round.isLt
      · apply compactHolds
        apply valueInvocation_mem
        · exact sourceLt
        · simpa [roundCount, previousEq] using round.isLt
  · apply PiRLCSamplerOrdinaryRows.rows_imply_selectorFull source
    · simpa [sourceCount, PiRLCSamplerOrdinaryRows.sourceCount] using sourceLt
    · exact ordinaryHolds

end NightstreamFPrime.Export.Stage1.PiRLCFirst54Conformance
