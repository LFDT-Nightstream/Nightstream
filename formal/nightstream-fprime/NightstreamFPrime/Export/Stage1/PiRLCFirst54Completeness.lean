import NightstreamFPrime.Export.Stage1.PiRLCFirst54Conformance
import NightstreamFPrime.Export.Stage1.PiRLCFirst54Projection
import NightstreamFPrime.Export.Stage1.PiRLCSamplerCompleteness

/-!
Owns the constructive bridge from the exact First54 selector lowering to all
canonical compact First54 package invocations. The proof stays symbolic in
the source, round, and slot indices.
-/

namespace NightstreamFPrime.Export.Stage1.PiRLCFirst54Completeness

open NightstreamFPrime.Circuit
open NightstreamFPrime.Export.Package
open NightstreamFPrime.Gadgets.Sampling
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.PiRLC.v1_1
open NightstreamFPrime.Layout.PiRLC.v1_1.Leaves
open NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Export.Stage1.PiRLCFirst54Invocations
open NightstreamFPrime.Export.Stage1.PiRLCFirst54Conformance

private theorem rename_positionRecipe
    (column : Nat → Nat)
    (left right : First54Step.Interface) (leftOffset rightOffset : Nat)
    (slot : Fin First54Step.slotCount)
    (accepted : CompactRows.renameExpr column (left.accepted leftOffset) =
      right.accepted rightOffset)
    (prior : ∀ current,
      CompactRows.renameExpr column (left.prior leftOffset current) =
        right.prior rightOffset current) :
    CompactRows.renameExpr column
        (First54Step.recipe left leftOffset slot) =
      First54Step.recipe right rightOffset slot := by
  unfold First54Step.recipe
  by_cases first : slot.val = 0
  · rw [dif_pos first]
    rw [dif_pos first]
    simp only [CompactRows.renameExpr, CompactRows.renameExpr_sub, accepted,
      prior]
    rfl
  · by_cases full : slot.val = First54Step.fullSlot
    · rw [dif_neg first, dif_pos full]
      rw [dif_neg first, dif_pos full]
      simp only [CompactRows.renameExpr, accepted, prior]
      rfl
    · rw [dif_neg first, dif_neg full]
      rw [dif_neg first, dif_neg full]
      simp only [CompactRows.renameExpr, CompactRows.renameExpr_sub, accepted,
        prior]
      rfl

private theorem rename_valueRecipe
    (column : Nat → Nat)
    (left right : First54ValueStep.Interface) (leftOffset rightOffset : Nat)
    (slot : Fin First54ValueStep.outputCount)
    (accepted : CompactRows.renameExpr column (left.accepted leftOffset) =
      right.accepted rightOffset)
    (symbol : CompactRows.renameExpr column (left.symbol leftOffset) =
      right.symbol rightOffset)
    (priorPosition : ∀ current,
      CompactRows.renameExpr column (left.priorPosition leftOffset current) =
        right.priorPosition rightOffset current)
    (priorOutput : ∀ current,
      CompactRows.renameExpr column (left.priorOutput leftOffset current) =
        right.priorOutput rightOffset current) :
    CompactRows.renameExpr column
        (First54ValueStep.recipe left leftOffset slot) =
      First54ValueStep.recipe right rightOffset slot := by
  unfold First54ValueStep.recipe
  simp only [CompactRows.renameExpr, accepted, symbol, priorPosition,
    priorOutput]
  rfl

private theorem firstPositionExpression_eq (source : Nat)
    (slot : Fin First54Step.slotCount) :
    CompactRows.renameExpr (firstPositionSourceInput source slot)
        (Expr.var PiRLCFirst54Templates.firstPositionOutputInput -
          PiRLCFirst54Templates.firstPositionRecipe slot) =
      exactPositionConstraint (logicalWidth := Data.logicalWidth)
        (publicFits := Data.publicFits) source 0 slot := by
  have accepted :
      CompactRows.renameExpr (firstPositionSourceInput source slot)
          (PiRLCFirst54Templates.firstPositionInterface.accepted 0) =
        (exactPositionInterface (logicalWidth := Data.logicalWidth)
          (publicFits := Data.publicFits) source 0).accepted
            (positionSourceStart source 0) := by
    rw [exactPositionAccepted_eq source 0
      (by norm_num [First54.candidateCount])]
    rfl
  have prior : ∀ current,
      CompactRows.renameExpr (firstPositionSourceInput source slot)
          (PiRLCFirst54Templates.firstPositionInterface.prior 0 current) =
        (exactPositionInterface (logicalWidth := Data.logicalWidth)
          (publicFits := Data.publicFits) source 0).prior
            (positionSourceStart source 0) current := by
    intro current
    rw [exactPositionPrior_zero_eq]
    unfold PiRLCFirst54Templates.firstPositionInterface First54.initialPosition
    by_cases first : current.val = 0
    · simp only [if_pos first, CompactRows.renameExpr]
      rfl
    · simp only [if_neg first, CompactRows.renameExpr]
      rfl
  have recipe := rename_positionRecipe
    (firstPositionSourceInput source slot)
    PiRLCFirst54Templates.firstPositionInterface
    (exactPositionInterface (logicalWidth := Data.logicalWidth)
      (publicFits := Data.publicFits) source 0)
    0 (positionSourceStart source 0) slot accepted prior
  unfold PiRLCFirst54Templates.firstPositionRecipe exactPositionConstraint
  rw [CompactRows.renameExpr_sub, recipe]
  change Expr.var (firstPositionSourceInput source slot
      PiRLCFirst54Templates.firstPositionOutputInput) - _ = _
  simp [firstPositionSourceInput,
    PiRLCFirst54Templates.firstPositionOutputInput]

private theorem laterPositionExpression_eq (source round : Nat)
    (roundPos : 0 < round) (roundLt : round < First54.candidateCount)
    (slot : Fin First54Step.slotCount) :
    CompactRows.renameExpr (laterPositionSourceInput source round slot)
        (Expr.var PiRLCFirst54Templates.laterPositionOutputInput -
          PiRLCFirst54Templates.laterPositionRecipe slot) =
      exactPositionConstraint (logicalWidth := Data.logicalWidth)
        (publicFits := Data.publicFits) source round slot := by
  have accepted :
      CompactRows.renameExpr (laterPositionSourceInput source round slot)
          (PiRLCFirst54Templates.laterPositionInterface.accepted 0) =
        (exactPositionInterface (logicalWidth := Data.logicalWidth)
          (publicFits := Data.publicFits) source round).accepted
            (positionSourceStart source round) := by
    rw [exactPositionAccepted_eq source round roundLt]
    rfl
  have prior : ∀ current,
      CompactRows.renameExpr (laterPositionSourceInput source round slot)
          (PiRLCFirst54Templates.laterPositionInterface.prior 0 current) =
        (exactPositionInterface (logicalWidth := Data.logicalWidth)
          (publicFits := Data.publicFits) source round).prior
            (positionSourceStart source round) current := by
    intro current
    rw [exactPositionPrior_later_eq source round roundPos current]
    have currentLt : 1 + current.val < 1 + First54Step.slotCount := by
      omega
    change Expr.var (laterPositionSourceInput source round slot
      (1 + current.val)) = _
    congr 1
    unfold laterPositionSourceInput
    rw [if_neg (by omega), if_pos currentLt]
    omega
  have recipe := rename_positionRecipe
    (laterPositionSourceInput source round slot)
    PiRLCFirst54Templates.laterPositionInterface
    (exactPositionInterface (logicalWidth := Data.logicalWidth)
      (publicFits := Data.publicFits) source round)
    0 (positionSourceStart source round) slot accepted prior
  unfold PiRLCFirst54Templates.laterPositionRecipe exactPositionConstraint
  rw [CompactRows.renameExpr_sub, recipe]
  change Expr.var (laterPositionSourceInput source round slot
      PiRLCFirst54Templates.laterPositionOutputInput) - _ = _
  simp [laterPositionSourceInput,
    PiRLCFirst54Templates.laterPositionOutputInput, First54Step.slotCount]

private theorem firstValueExpression_eq (source : Nat)
    (slot : Fin First54ValueStep.outputCount) :
    CompactRows.renameExpr (firstValueSourceInput source slot)
        (Expr.var PiRLCFirst54Templates.firstValueOutputInput -
          PiRLCFirst54Templates.firstValueRecipe slot) =
      exactValueConstraint (logicalWidth := Data.logicalWidth)
        (publicFits := Data.publicFits) source 0 slot := by
  have accepted :
      CompactRows.renameExpr (firstValueSourceInput source slot)
          (PiRLCFirst54Templates.firstValueInterface.accepted 0) =
        (exactValueInterface (logicalWidth := Data.logicalWidth)
          (publicFits := Data.publicFits) source 0).accepted
            (valueSourceStart source 0) := by
    rw [exactValueAccepted_eq source 0
      (by norm_num [First54.candidateCount])]
    rfl
  have symbol :
      CompactRows.renameExpr (firstValueSourceInput source slot)
          (PiRLCFirst54Templates.firstValueInterface.symbol 0) =
        (exactValueInterface (logicalWidth := Data.logicalWidth)
          (publicFits := Data.publicFits) source 0).symbol
            (valueSourceStart source 0) := by
    rw [exactValueSymbol_eq source 0
      (by norm_num [First54.candidateCount])]
    rfl
  have priorPosition : ∀ current,
      CompactRows.renameExpr (firstValueSourceInput source slot)
          (PiRLCFirst54Templates.firstValueInterface.priorPosition 0 current) =
        (exactValueInterface (logicalWidth := Data.logicalWidth)
          (publicFits := Data.publicFits) source 0).priorPosition
            (valueSourceStart source 0) current := by
    intro current
    rw [exactValuePriorPosition_zero_eq]
    unfold PiRLCFirst54Templates.firstValueInterface First54.initialPosition
    by_cases first : current.val = 0
    · simp only [if_pos first, CompactRows.renameExpr]
      rfl
    · simp only [if_neg first, CompactRows.renameExpr]
      rfl
  have priorOutput : ∀ current,
      CompactRows.renameExpr (firstValueSourceInput source slot)
          (PiRLCFirst54Templates.firstValueInterface.priorOutput 0 current) =
        (exactValueInterface (logicalWidth := Data.logicalWidth)
          (publicFits := Data.publicFits) source 0).priorOutput
            (valueSourceStart source 0) current := by
    intro current
    rw [exactValuePriorOutput_zero_eq]
    rfl
  have recipe := rename_valueRecipe
    (firstValueSourceInput source slot)
    PiRLCFirst54Templates.firstValueInterface
    (exactValueInterface (logicalWidth := Data.logicalWidth)
      (publicFits := Data.publicFits) source 0)
    0 (valueSourceStart source 0) slot accepted symbol priorPosition priorOutput
  unfold PiRLCFirst54Templates.firstValueRecipe exactValueConstraint
  rw [CompactRows.renameExpr_sub, recipe]
  change Expr.var (firstValueSourceInput source slot
      PiRLCFirst54Templates.firstValueOutputInput) - _ = _
  simp [firstValueSourceInput, PiRLCFirst54Templates.firstValueOutputInput]

private theorem laterValueExpression_eq (source round : Nat)
    (roundPos : 0 < round) (roundLt : round < First54.candidateCount)
    (slot : Fin First54ValueStep.outputCount) :
    CompactRows.renameExpr (laterValueSourceInput source round slot)
        (Expr.var PiRLCFirst54Templates.laterValueOutputInput -
          PiRLCFirst54Templates.laterValueRecipe slot) =
      exactValueConstraint (logicalWidth := Data.logicalWidth)
        (publicFits := Data.publicFits) source round slot := by
  have accepted :
      CompactRows.renameExpr (laterValueSourceInput source round slot)
          (PiRLCFirst54Templates.laterValueInterface.accepted 0) =
        (exactValueInterface (logicalWidth := Data.logicalWidth)
          (publicFits := Data.publicFits) source round).accepted
            (valueSourceStart source round) := by
    rw [exactValueAccepted_eq source round roundLt]
    rfl
  have symbol :
      CompactRows.renameExpr (laterValueSourceInput source round slot)
          (PiRLCFirst54Templates.laterValueInterface.symbol 0) =
        (exactValueInterface (logicalWidth := Data.logicalWidth)
          (publicFits := Data.publicFits) source round).symbol
            (valueSourceStart source round) := by
    rw [exactValueSymbol_eq source round roundLt]
    rfl
  have priorPosition : ∀ current,
      CompactRows.renameExpr (laterValueSourceInput source round slot)
          (PiRLCFirst54Templates.laterValueInterface.priorPosition 0 current) =
        (exactValueInterface (logicalWidth := Data.logicalWidth)
          (publicFits := Data.publicFits) source round).priorPosition
            (valueSourceStart source round) current := by
    intro current
    rw [exactValuePriorPosition_later_eq source round roundPos current]
    have currentLt :
        PiRLCFirst54Templates.laterValuePriorPositionStart + current.val <
          PiRLCFirst54Templates.laterValuePriorOutputStart := by
      have bounded := current.isLt
      norm_num [PiRLCFirst54Templates.laterValuePriorPositionStart,
        PiRLCFirst54Templates.laterValuePriorOutputStart,
        First54Step.slotCount] at bounded ⊢
      omega
    change Expr.var (laterValueSourceInput source round slot
      (PiRLCFirst54Templates.laterValuePriorPositionStart + current.val)) = _
    congr 1
    unfold laterValueSourceInput
    rw [if_neg (by
      norm_num [PiRLCFirst54Templates.laterValuePriorPositionStart]),
      if_neg (by
        norm_num [PiRLCFirst54Templates.laterValuePriorPositionStart]
        omega),
      if_pos currentLt]
    omega
  have priorOutput : ∀ current,
      CompactRows.renameExpr (laterValueSourceInput source round slot)
          (PiRLCFirst54Templates.laterValueInterface.priorOutput 0 current) =
        (exactValueInterface (logicalWidth := Data.logicalWidth)
          (publicFits := Data.publicFits) source round).priorOutput
            (valueSourceStart source round) current := by
    intro current
    rw [exactValuePriorOutput_later_eq source round roundPos current]
    have currentLt :
        PiRLCFirst54Templates.laterValuePriorOutputStart + current.val <
          PiRLCFirst54Templates.laterValueOutputInput := by
      have bounded := current.isLt
      norm_num [PiRLCFirst54Templates.laterValuePriorOutputStart,
        PiRLCFirst54Templates.laterValueOutputInput,
        First54ValueStep.outputCount] at bounded ⊢
      omega
    have notPosition : ¬
        PiRLCFirst54Templates.laterValuePriorOutputStart + current.val <
          PiRLCFirst54Templates.laterValuePriorOutputStart := by omega
    change Expr.var (laterValueSourceInput source round slot
      (PiRLCFirst54Templates.laterValuePriorOutputStart + current.val)) = _
    congr 1
    unfold laterValueSourceInput
    rw [if_neg (by
      norm_num [PiRLCFirst54Templates.laterValuePriorOutputStart]),
      if_neg (by
        norm_num [PiRLCFirst54Templates.laterValuePriorOutputStart]
        omega),
      if_neg notPosition, if_pos currentLt]
    omega
  have recipe := rename_valueRecipe
    (laterValueSourceInput source round slot)
    PiRLCFirst54Templates.laterValueInterface
    (exactValueInterface (logicalWidth := Data.logicalWidth)
      (publicFits := Data.publicFits) source round)
    0 (valueSourceStart source round) slot accepted symbol priorPosition
      priorOutput
  unfold PiRLCFirst54Templates.laterValueRecipe exactValueConstraint
  rw [CompactRows.renameExpr_sub, recipe]
  change Expr.var (laterValueSourceInput source round slot
      PiRLCFirst54Templates.laterValueOutputInput) - _ = _
  simp [laterValueSourceInput,
    PiRLCFirst54Templates.laterValuePriorOutputStart,
    PiRLCFirst54Templates.laterValueOutputInput]

private theorem compactConstraintTemplate_rows_eq_compactTemplate_rows
    (inputCount outputInput : Nat) (outputRecipe : Expr)
    (positive : 0 < R1CS.constraintFreshCount
      (Expr.var outputInput - outputRecipe)) :
    (CompactRows.compactConstraintTemplate inputCount outputInput
        outputRecipe).rows =
      (CompactRows.compactTemplate inputCount outputInput outputRecipe).rows := by
  unfold CompactRows.compactConstraintTemplate CompactRows.compactTemplate
  change (R1CS.lowerConstraint (Expr.var outputInput - outputRecipe)
      inputCount).rows.map (CompactRows.abstractRow inputCount) =
    (R1CS.lowerGenericConstraint (Expr.var outputInput - outputRecipe)
      inputCount).rows.map (CompactRows.abstractRow inputCount)
  rw [R1CS.lowerConstraint_eq_lowerGenericConstraint_of_fresh_pos
    (Expr.var outputInput - outputRecipe) inputCount positive]

private theorem positiveCompactRows
    (env : Env) (inputCount outputInput sourceFresh finalFresh : Nat)
    (actualInput sourceInput : Nat → Nat) (outputRecipe sourceExpression : Expr)
    (sourceBound : inputCount ≤ sourceFresh)
    (finalBound : inputCount ≤ finalFresh)
    (scope : (Expr.var outputInput - outputRecipe).VarsBelow inputCount)
    (compactPositive : 0 < R1CS.constraintFreshCount
      (Expr.var outputInput - outputRecipe))
    (sourcePositive : 0 < R1CS.constraintFreshCount sourceExpression)
    (expressionEq : CompactRows.renameExpr sourceInput
      (Expr.var outputInput - outputRecipe) = sourceExpression)
    (sourceLocal : Spartan.piCcsPhaseOffset ≤ sourceFresh)
    (finalFreshEq : finalFresh = Spartan.sourceToSpartan sourceFresh)
    (inputsEq : ∀ input, input < inputCount →
      actualInput input = Spartan.sourceToSpartan (sourceInput input))
    (sourceRows : R1CS.RowsHold (Spartan.pullback env)
      (R1CS.lowerConstraint sourceExpression sourceFresh).rows) :
    R1CS.RowsHold env
      (CompactRows.instantiateRows actualInput finalFresh
        (CompactRows.compactConstraintTemplate inputCount outputInput
          outputRecipe)) := by
  have templateRows := compactConstraintTemplate_rows_eq_compactTemplate_rows
    inputCount outputInput outputRecipe compactPositive
  unfold CompactRows.instantiateRows
  rw [templateRows]
  change R1CS.RowsHold env
    (CompactRows.instantiateRows actualInput finalFresh
      (CompactRows.compactTemplate inputCount outputInput outputRecipe))
  rw [CompactRows.instantiate_compactTemplate_congr_inputs inputCount
    outputInput finalFresh actualInput
    (fun input => Spartan.sourceToSpartan (sourceInput input)) outputRecipe
    inputsEq]
  have expanded := CompactRows.instantiate_compactTemplate_remap inputCount
    outputInput sourceFresh finalFresh sourceInput Spartan.sourceToSpartan
    outputRecipe sourceBound finalBound scope (by
      intro offset
      rw [finalFreshEq]
      exact Spartan.sourceToSpartan_add_of_piCcsLocal sourceFresh offset
        sourceLocal)
  rw [expanded, expressionEq]
  have remapped := (Spartan.remapRows_hold env
    (R1CS.lowerConstraint sourceExpression sourceFresh).rows).mpr sourceRows
  rw [PiRLCCombinationInvocations.spartanRemapRows_eq] at remapped
  rw [R1CS.lowerConstraint_eq_lowerGenericConstraint_of_fresh_pos
    sourceExpression sourceFresh sourcePositive] at remapped
  exact remapped

private theorem zeroCompactRows
    (env : Env) (inputCount outputInput localStart : Nat)
    (actualInput : Nat → Nat) (outputRecipe : Expr)
    (localBound : inputCount ≤ localStart)
    (freshZero : R1CS.constraintFreshCount
      (Expr.var outputInput - outputRecipe) = 0)
    (evalZero : (Expr.var outputInput - outputRecipe).eval
      (fun index => env (CompactRows.relocate inputCount
        (localStart - inputCount) actualInput index)) = 0) :
    R1CS.RowsHold env
      (CompactRows.instantiateRows actualInput localStart
        (CompactRows.compactConstraintTemplate inputCount outputInput
          outputRecipe)) := by
  have normalized := R1CS.lowerConstraint_complete_of_freshCount_zero
    (fun index => env (CompactRows.relocate inputCount
      (localStart - inputCount) actualInput index))
    (Expr.var outputInput - outputRecipe) inputCount freshZero evalZero
  have startEq : inputCount + (localStart - inputCount) = localStart := by
    omega
  have expanded := CompactRows.instantiate_compactConstraintTemplate
    inputCount outputInput (localStart - inputCount) actualInput outputRecipe
  rw [startEq] at expanded
  rw [expanded]
  intro row member
  rcases List.mem_map.mp member with ⟨sourceRow, sourceMember, rfl⟩
  apply (CompactRows.renameRow_holds
    (CompactRows.relocate inputCount (localStart - inputCount) actualInput)
    sourceRow env).mpr
  exact normalized sourceRow sourceMember

private theorem remappedPacket_implies_positionSourceRows
    (env : Env) (packets : PiRLCPackageCompleteness.RemappedPacketRowsHold env)
    (source : Fin sourceCount) (round : Fin First54.candidateCount)
    (slot : Fin First54Step.slotCount) :
    R1CS.RowsHold (Spartan.pullback env)
      (R1CS.lowerConstraint
        (exactPositionConstraint (logicalWidth := Data.logicalWidth)
          (publicFits := Data.publicFits) source.val round.val slot)
        (PiRLCStarts.selectorFreshStart source.val +
          roundFreshPrefix round.val +
          positionFreshPrefix round.val slot.val)).rows := by
  have selectorRows :=
    PiRLCSamplerCompleteness.remappedPacket_implies_selectorSourceRows env
      packets source
  have projected := PiRLCFirst54Projection.selectorRows_imply_positionRows
    source.val (Spartan.pullback env) selectorRows round slot
  simpa [PiRLCFirst54Projection.sourceInterface,
    PiRLCFirst54Conformance.sourceInterface, exactPositionConstraint,
    exactPositionInterface, positionSourceStart, First54.positionConstraint]
    using projected

private theorem remappedPacket_implies_valueSourceRows
    (env : Env) (packets : PiRLCPackageCompleteness.RemappedPacketRowsHold env)
    (source : Fin sourceCount) (round : Fin First54.candidateCount)
    (slot : Fin First54ValueStep.outputCount) :
    R1CS.RowsHold (Spartan.pullback env)
      (R1CS.lowerConstraint
        (exactValueConstraint (logicalWidth := Data.logicalWidth)
          (publicFits := Data.publicFits) source.val round.val slot)
        (PiRLCStarts.selectorFreshStart source.val +
          roundFreshPrefix round.val +
          valueFreshPrefix round.val slot.val)).rows := by
  have selectorRows :=
    PiRLCSamplerCompleteness.remappedPacket_implies_selectorSourceRows env
      packets source
  have projected := PiRLCFirst54Projection.selectorRows_imply_valueRows
    source.val (Spartan.pullback env) selectorRows round slot
  simpa [PiRLCFirst54Projection.sourceInterface,
    PiRLCFirst54Conformance.sourceInterface, exactValueConstraint,
    exactValueInterface, valueSourceStart, First54.valueConstraint]
    using projected

private theorem exactPositionZeroFresh (source : Nat)
    (slot : Fin First54Step.slotCount) :
    R1CS.constraintFreshCount
        (exactPositionConstraint (logicalWidth := Data.logicalWidth)
          (publicFits := Data.publicFits) source 0 slot) = 0 := by
  have cost := (First54.positionZero_cost
    (PiRLCFirst54Projection.sourceInterface source) source
    (PiRLCStarts.samplerSourceLogicalStart source)
    (PiRLCStarts.selectorLogicalStart source) slot).1
  simpa [PiRLCFirst54Projection.sourceInterface,
    PiRLCFirst54Conformance.sourceInterface, exactPositionConstraint,
    exactPositionInterface, positionSourceStart, First54.positionConstraint]
    using cost

private theorem exactPositionSuccFresh (source round : Nat)
    (slot : Fin First54Step.slotCount) :
    R1CS.constraintFreshCount
        (exactPositionConstraint (logicalWidth := Data.logicalWidth)
          (publicFits := Data.publicFits) source (round + 1) slot) =
      First54.runningPositionFresh slot := by
  have cost := (First54.positionSucc_cost
    (PiRLCFirst54Projection.sourceInterface source) source
    (PiRLCStarts.samplerSourceLogicalStart source)
    (PiRLCStarts.selectorLogicalStart source) round slot).1
  simpa [PiRLCFirst54Projection.sourceInterface,
    PiRLCFirst54Conformance.sourceInterface, exactPositionConstraint,
    exactPositionInterface, positionSourceStart, First54.positionConstraint]
    using cost

private theorem exactValueZeroFresh (source : Nat)
    (slot : Fin First54ValueStep.outputCount) :
    R1CS.constraintFreshCount
        (exactValueConstraint (logicalWidth := Data.logicalWidth)
          (publicFits := Data.publicFits) source 0 slot) = 4 := by
  have cost := (First54.valueZero_cost
    (PiRLCFirst54Projection.sourceInterface source) source
    (PiRLCStarts.samplerSourceLogicalStart source)
    (PiRLCStarts.selectorLogicalStart source) slot).1
  simpa [PiRLCFirst54Projection.sourceInterface,
    PiRLCFirst54Conformance.sourceInterface, exactValueConstraint,
    exactValueInterface, valueSourceStart, First54.valueConstraint]
    using cost

private theorem exactValueSuccFresh (source round : Nat)
    (slot : Fin First54ValueStep.outputCount) :
    R1CS.constraintFreshCount
        (exactValueConstraint (logicalWidth := Data.logicalWidth)
          (publicFits := Data.publicFits) source (round + 1) slot) = 4 := by
  have cost := (First54.valueSucc_cost
    (PiRLCFirst54Projection.sourceInterface source) source
    (PiRLCStarts.samplerSourceLogicalStart source)
    (PiRLCStarts.selectorLogicalStart source) round slot).1
  simpa [PiRLCFirst54Projection.sourceInterface,
    PiRLCFirst54Conformance.sourceInterface, exactValueConstraint,
    exactValueInterface, valueSourceStart, First54.valueConstraint]
    using cost

private theorem remappedPacket_implies_positionZeroRows
    (env : Env) (packets : PiRLCPackageCompleteness.RemappedPacketRowsHold env)
    (source : Fin sourceCount) (slot : Fin First54Step.slotCount) :
    R1CS.RowsHold env
      (CompactRows.instantiateRows
        (compactInputColumn
          (positionInvocation source.val 0 slot.val).inputRanges)
        (positionInvocation source.val 0 slot.val).localStart
        (PiRLCFirst54Templates.firstPositionTemplate slot)) := by
  have sourceRows := remappedPacket_implies_positionSourceRows env packets
    source ⟨0, by norm_num [First54.candidateCount]⟩ slot
  have sourceEval := R1CS.lowerConstraint_sound (Spartan.pullback env)
    (exactPositionConstraint (logicalWidth := Data.logicalWidth)
      (publicFits := Data.publicFits) source.val 0 slot)
    (PiRLCStarts.selectorFreshStart source.val) (by
      simpa [roundFreshPrefix, positionFreshPrefix] using sourceRows)
  have normalizedEval :=
    (firstPositionConstraint_eval_eq
      (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits)
      source.val (positionInvocation source.val 0 slot.val).localStart slot env)
      |>.trans sourceEval
  have localBound : PiRLCFirst54Templates.firstPositionInputCount ≤
      (positionInvocation source.val 0 slot.val).localStart := by
    exact Nat.le_trans (by
      norm_num [PiRLCFirst54Templates.firstPositionInputCount,
        PiRLCFirst54Templates.laterPositionInputCount])
      (positionInvocation_localBound source.val 0 slot.val)
  have built := zeroCompactRows env
    PiRLCFirst54Templates.firstPositionInputCount
    PiRLCFirst54Templates.firstPositionOutputInput
    (positionInvocation source.val 0 slot.val).localStart
    (compactInputColumn (positionInvocation source.val 0 slot.val).inputRanges)
    (PiRLCFirst54Templates.firstPositionRecipe slot) localBound
    (PiRLCFirst54Templates.firstPosition_constraintFreshCount slot) (by
      simpa [compactEvalEnv, positionInvocation] using normalizedEval)
  simpa [PiRLCFirst54Templates.firstPositionTemplate] using built

private theorem remappedPacket_implies_positionSuccRows
    (env : Env) (packets : PiRLCPackageCompleteness.RemappedPacketRowsHold env)
    (source : Fin sourceCount) (round : Nat)
    (roundLt : round + 1 < First54.candidateCount)
    (slot : Fin First54Step.slotCount) :
    R1CS.RowsHold env
      (CompactRows.instantiateRows
        (compactInputColumn
          (positionInvocation source.val (round + 1) slot.val).inputRanges)
        (positionInvocation source.val (round + 1) slot.val).localStart
        (PiRLCFirst54Templates.laterPositionTemplate slot)) := by
  let boundedRound : Fin First54.candidateCount := ⟨round + 1, roundLt⟩
  have sourceRows := remappedPacket_implies_positionSourceRows env packets
    source boundedRound slot
  by_cases slotZero : slot.val = 0
  · have sourceEval := R1CS.lowerConstraint_sound (Spartan.pullback env)
      (exactPositionConstraint (logicalWidth := Data.logicalWidth)
        (publicFits := Data.publicFits) source.val (round + 1) slot)
      (PiRLCStarts.selectorFreshStart source.val +
        roundFreshPrefix (round + 1) +
        positionFreshPrefix (round + 1) slot.val) sourceRows
    have normalizedEval :=
      (laterPositionConstraint_eval_eq
        (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits)
        source.val (round + 1)
        (positionInvocation source.val (round + 1) slot.val).localStart
        (by omega) roundLt slot env) |>.trans sourceEval
    have localBound := positionInvocation_localBound source.val (round + 1)
      slot.val
    have compactZero : R1CS.constraintFreshCount
        (Expr.var PiRLCFirst54Templates.laterPositionOutputInput -
          PiRLCFirst54Templates.laterPositionRecipe slot) = 0 := by
      rw [PiRLCFirst54Templates.laterPosition_constraintFreshCount]
      simp [slotZero]
    have built := zeroCompactRows env
      PiRLCFirst54Templates.laterPositionInputCount
      PiRLCFirst54Templates.laterPositionOutputInput
      (positionInvocation source.val (round + 1) slot.val).localStart
      (compactInputColumn
        (positionInvocation source.val (round + 1) slot.val).inputRanges)
      (PiRLCFirst54Templates.laterPositionRecipe slot) localBound compactZero
      (by simpa [compactEvalEnv, positionInvocation] using normalizedEval)
    simpa [PiRLCFirst54Templates.laterPositionTemplate] using built
  · have compactPositive : 0 < R1CS.constraintFreshCount
        (Expr.var PiRLCFirst54Templates.laterPositionOutputInput -
          PiRLCFirst54Templates.laterPositionRecipe slot) := by
      rw [PiRLCFirst54Templates.laterPosition_constraintFreshCount]
      by_cases full : slot.val = 54 <;> simp [slotZero, full]
    have sourcePositive : 0 < R1CS.constraintFreshCount
        (exactPositionConstraint (logicalWidth := Data.logicalWidth)
          (publicFits := Data.publicFits) source.val (round + 1) slot) := by
      rw [exactPositionSuccFresh, First54.runningPositionFresh]
      by_cases full : slot.val = 54 <;> simp [slotZero, full]
    have sourceLocal : Spartan.piCcsPhaseOffset ≤
        PiRLCStarts.selectorFreshStart source.val +
          roundFreshPrefix (round + 1) +
          positionFreshPrefix (round + 1) slot.val := by
      have base := selectorFreshStart_local source.val
      omega
    have sourceBound : PiRLCFirst54Templates.laterPositionInputCount ≤
        PiRLCStarts.selectorFreshStart source.val +
          roundFreshPrefix (round + 1) +
          positionFreshPrefix (round + 1) slot.val := by
      calc
        PiRLCFirst54Templates.laterPositionInputCount ≤
            Spartan.piCcsPhaseOffset := by
          norm_num [PiRLCFirst54Templates.laterPositionInputCount,
            Spartan.piCcsPhaseOffset]
        _ ≤ _ := sourceLocal
    have built := positiveCompactRows env
      PiRLCFirst54Templates.laterPositionInputCount
      PiRLCFirst54Templates.laterPositionOutputInput
      (PiRLCStarts.selectorFreshStart source.val +
        roundFreshPrefix (round + 1) +
        positionFreshPrefix (round + 1) slot.val)
      (positionInvocation source.val (round + 1) slot.val).localStart
      (compactInputColumn
        (positionInvocation source.val (round + 1) slot.val).inputRanges)
      (laterPositionSourceInput source.val (round + 1) slot)
      (PiRLCFirst54Templates.laterPositionRecipe slot)
      (exactPositionConstraint (logicalWidth := Data.logicalWidth)
        (publicFits := Data.publicFits) source.val (round + 1) slot)
      sourceBound
      (positionInvocation_localBound source.val (round + 1) slot.val)
      (PiRLCFirst54Templates.laterPosition_constraint_varsBelow slot)
      compactPositive sourcePositive
      (laterPositionExpression_eq source.val (round + 1) (by omega)
        roundLt slot)
      sourceLocal (by rfl) (by
        intro input inputLt
        simpa [positionInvocation, finalColumn] using
          laterPositionInputColumn_eq source.val (round + 1) slot input
            inputLt)
      sourceRows
    simpa [PiRLCFirst54Templates.laterPositionTemplate] using built

private theorem remappedPacket_implies_valueZeroRows
    (env : Env) (packets : PiRLCPackageCompleteness.RemappedPacketRowsHold env)
    (source : Fin sourceCount) (slot : Fin First54ValueStep.outputCount) :
    R1CS.RowsHold env
      (CompactRows.instantiateRows
        (compactInputColumn
          (valueInvocation source.val 0 slot.val).inputRanges)
        (valueInvocation source.val 0 slot.val).localStart
        (PiRLCFirst54Templates.firstValueTemplate slot)) := by
  have sourceRows := remappedPacket_implies_valueSourceRows env packets source
    ⟨0, by norm_num [First54.candidateCount]⟩ slot
  have compactPositive : 0 < R1CS.constraintFreshCount
      (Expr.var PiRLCFirst54Templates.firstValueOutputInput -
        PiRLCFirst54Templates.firstValueRecipe slot) := by
    rw [PiRLCFirst54Templates.firstValue_constraintFreshCount]
    omega
  have sourcePositive : 0 < R1CS.constraintFreshCount
      (exactValueConstraint (logicalWidth := Data.logicalWidth)
        (publicFits := Data.publicFits) source.val 0 slot) := by
    rw [exactValueZeroFresh]
    omega
  have sourceLocal : Spartan.piCcsPhaseOffset ≤
      PiRLCStarts.selectorFreshStart source.val + roundFreshPrefix 0 +
        valueFreshPrefix 0 slot.val := by
    have base := selectorFreshStart_local source.val
    omega
  have sourceBound : PiRLCFirst54Templates.firstValueInputCount ≤
      PiRLCStarts.selectorFreshStart source.val + roundFreshPrefix 0 +
        valueFreshPrefix 0 slot.val := by
    calc
      PiRLCFirst54Templates.firstValueInputCount ≤
          Spartan.piCcsPhaseOffset := by
        norm_num [PiRLCFirst54Templates.firstValueInputCount,
          Spartan.piCcsPhaseOffset]
      _ ≤ _ := sourceLocal
  have finalBound : PiRLCFirst54Templates.firstValueInputCount ≤
      (valueInvocation source.val 0 slot.val).localStart := by
    exact Nat.le_trans (by
      norm_num [PiRLCFirst54Templates.firstValueInputCount,
        PiRLCFirst54Templates.laterValueInputCount])
      (valueInvocation_localBound source.val 0 slot.val)
  have built := positiveCompactRows env
    PiRLCFirst54Templates.firstValueInputCount
    PiRLCFirst54Templates.firstValueOutputInput
    (PiRLCStarts.selectorFreshStart source.val + roundFreshPrefix 0 +
      valueFreshPrefix 0 slot.val)
    (valueInvocation source.val 0 slot.val).localStart
    (compactInputColumn (valueInvocation source.val 0 slot.val).inputRanges)
    (firstValueSourceInput source.val slot)
    (PiRLCFirst54Templates.firstValueRecipe slot)
    (exactValueConstraint (logicalWidth := Data.logicalWidth)
      (publicFits := Data.publicFits) source.val 0 slot)
    sourceBound finalBound
    (PiRLCFirst54Templates.firstValue_constraint_varsBelow slot)
    compactPositive sourcePositive (firstValueExpression_eq source.val slot)
    sourceLocal (by rfl) (by
      intro input inputLt
      simpa [valueInvocation, finalColumn] using
        firstValueInputColumn_eq source.val slot input inputLt)
    sourceRows
  simpa [PiRLCFirst54Templates.firstValueTemplate] using built

private theorem remappedPacket_implies_valueSuccRows
    (env : Env) (packets : PiRLCPackageCompleteness.RemappedPacketRowsHold env)
    (source : Fin sourceCount) (round : Nat)
    (roundLt : round + 1 < First54.candidateCount)
    (slot : Fin First54ValueStep.outputCount) :
    R1CS.RowsHold env
      (CompactRows.instantiateRows
        (compactInputColumn
          (valueInvocation source.val (round + 1) slot.val).inputRanges)
        (valueInvocation source.val (round + 1) slot.val).localStart
        (PiRLCFirst54Templates.laterValueTemplate slot)) := by
  let boundedRound : Fin First54.candidateCount := ⟨round + 1, roundLt⟩
  have sourceRows := remappedPacket_implies_valueSourceRows env packets source
    boundedRound slot
  have compactPositive : 0 < R1CS.constraintFreshCount
      (Expr.var PiRLCFirst54Templates.laterValueOutputInput -
        PiRLCFirst54Templates.laterValueRecipe slot) := by
    rw [PiRLCFirst54Templates.laterValue_constraintFreshCount]
    omega
  have sourcePositive : 0 < R1CS.constraintFreshCount
      (exactValueConstraint (logicalWidth := Data.logicalWidth)
        (publicFits := Data.publicFits) source.val (round + 1) slot) := by
    rw [exactValueSuccFresh]
    omega
  have sourceLocal : Spartan.piCcsPhaseOffset ≤
      PiRLCStarts.selectorFreshStart source.val +
        roundFreshPrefix (round + 1) +
        valueFreshPrefix (round + 1) slot.val := by
    have base := selectorFreshStart_local source.val
    omega
  have sourceBound : PiRLCFirst54Templates.laterValueInputCount ≤
      PiRLCStarts.selectorFreshStart source.val +
        roundFreshPrefix (round + 1) +
        valueFreshPrefix (round + 1) slot.val := by
    calc
      PiRLCFirst54Templates.laterValueInputCount ≤
          Spartan.piCcsPhaseOffset := by
        norm_num [PiRLCFirst54Templates.laterValueInputCount,
          Spartan.piCcsPhaseOffset]
      _ ≤ _ := sourceLocal
  have built := positiveCompactRows env
    PiRLCFirst54Templates.laterValueInputCount
    PiRLCFirst54Templates.laterValueOutputInput
    (PiRLCStarts.selectorFreshStart source.val +
      roundFreshPrefix (round + 1) +
      valueFreshPrefix (round + 1) slot.val)
    (valueInvocation source.val (round + 1) slot.val).localStart
    (compactInputColumn
      (valueInvocation source.val (round + 1) slot.val).inputRanges)
    (laterValueSourceInput source.val (round + 1) slot)
    (PiRLCFirst54Templates.laterValueRecipe slot)
    (exactValueConstraint (logicalWidth := Data.logicalWidth)
      (publicFits := Data.publicFits) source.val (round + 1) slot)
    sourceBound
    (valueInvocation_localBound source.val (round + 1) slot.val)
    (PiRLCFirst54Templates.laterValue_constraint_varsBelow slot)
    compactPositive sourcePositive
    (laterValueExpression_eq source.val (round + 1) (by omega) roundLt slot)
    sourceLocal (by rfl) (by
      intro input inputLt
      simpa [valueInvocation, finalColumn] using
        laterValueInputColumn_eq source.val (round + 1) slot input inputLt)
    sourceRows
  simpa [PiRLCFirst54Templates.laterValueTemplate] using built

private theorem remappedPacket_implies_positionZeroInvocation
    (env : Env) (packets : PiRLCPackageCompleteness.RemappedPacketRowsHold env)
    (source : Fin sourceCount) (slot : Fin First54Step.slotCount) :
    CompactRowInvocationHolds (Data.circuitPackage ())
      (positionInvocation source.val 0 slot.val) env := by
  have rows := remappedPacket_implies_positionZeroRows env packets source slot
  unfold CompactRowInvocationHolds
  rw [Data.circuitPackage_compactRowTemplates, Data.compactRowTemplates_eq]
  change match packageTemplates[
      (positionInvocation source.val 0 slot.val).templateIndex]? with
    | none => False
    | some template => R1CS.RowsHold env
        (template.rows.map
          (instantiateCompactRow (positionInvocation source.val 0 slot.val)))
  rw [positionInvocation_zero_template source.val slot]
  dsimp only
  rw [← CompactRows.instantiateRows_eq_package]
  exact rows

private theorem remappedPacket_implies_positionSuccInvocation
    (env : Env) (packets : PiRLCPackageCompleteness.RemappedPacketRowsHold env)
    (source : Fin sourceCount) (round : Nat)
    (roundLt : round + 1 < First54.candidateCount)
    (slot : Fin First54Step.slotCount) :
    CompactRowInvocationHolds (Data.circuitPackage ())
      (positionInvocation source.val (round + 1) slot.val) env := by
  have rows := remappedPacket_implies_positionSuccRows env packets source round
    roundLt slot
  unfold CompactRowInvocationHolds
  rw [Data.circuitPackage_compactRowTemplates, Data.compactRowTemplates_eq]
  change match packageTemplates[
      (positionInvocation source.val (round + 1) slot.val).templateIndex]? with
    | none => False
    | some template => R1CS.RowsHold env
        (template.rows.map (instantiateCompactRow
          (positionInvocation source.val (round + 1) slot.val)))
  rw [positionInvocation_succ_template source.val round slot]
  dsimp only
  rw [← CompactRows.instantiateRows_eq_package]
  exact rows

private theorem remappedPacket_implies_valueZeroInvocation
    (env : Env) (packets : PiRLCPackageCompleteness.RemappedPacketRowsHold env)
    (source : Fin sourceCount) (slot : Fin First54ValueStep.outputCount) :
    CompactRowInvocationHolds (Data.circuitPackage ())
      (valueInvocation source.val 0 slot.val) env := by
  have rows := remappedPacket_implies_valueZeroRows env packets source slot
  unfold CompactRowInvocationHolds
  rw [Data.circuitPackage_compactRowTemplates, Data.compactRowTemplates_eq]
  change match packageTemplates[
      (valueInvocation source.val 0 slot.val).templateIndex]? with
    | none => False
    | some template => R1CS.RowsHold env
        (template.rows.map
          (instantiateCompactRow (valueInvocation source.val 0 slot.val)))
  rw [valueInvocation_zero_template source.val slot]
  dsimp only
  rw [← CompactRows.instantiateRows_eq_package]
  exact rows

private theorem remappedPacket_implies_valueSuccInvocation
    (env : Env) (packets : PiRLCPackageCompleteness.RemappedPacketRowsHold env)
    (source : Fin sourceCount) (round : Nat)
    (roundLt : round + 1 < First54.candidateCount)
    (slot : Fin First54ValueStep.outputCount) :
    CompactRowInvocationHolds (Data.circuitPackage ())
      (valueInvocation source.val (round + 1) slot.val) env := by
  have rows := remappedPacket_implies_valueSuccRows env packets source round
    roundLt slot
  unfold CompactRowInvocationHolds
  rw [Data.circuitPackage_compactRowTemplates, Data.compactRowTemplates_eq]
  change match packageTemplates[
      (valueInvocation source.val (round + 1) slot.val).templateIndex]? with
    | none => False
    | some template => R1CS.RowsHold env
        (template.rows.map (instantiateCompactRow
          (valueInvocation source.val (round + 1) slot.val)))
  rw [valueInvocation_succ_template source.val round slot]
  dsimp only
  rw [← CompactRows.instantiateRows_eq_package]
  exact rows

/-- The exact remapped PiRLC sampler packet constructs every canonical
compact First54 invocation in the circuit package. -/
theorem remappedPacket_implies_first54Invocations
    (env : Env)
    (packets : PiRLCPackageCompleteness.RemappedPacketRowsHold env) :
    ∀ selected ∈ invocations,
      CompactRowInvocationHolds (Data.circuitPackage ()) selected env := by
  intro selected member
  unfold invocations at member
  rcases List.mem_flatMap.mp member with
    ⟨sourceValue, sourceMember, sourceInvocation⟩
  have sourceLt : sourceValue < sourceCount := List.mem_range.mp sourceMember
  let source : Fin sourceCount := ⟨sourceValue, sourceLt⟩
  unfold sourceInvocations at sourceInvocation
  rcases List.mem_flatMap.mp sourceInvocation with
    ⟨roundValue, roundMember, roundInvocation⟩
  have roundLt : roundValue < roundCount := List.mem_range.mp roundMember
  simp only [roundInvocations, List.mem_append] at roundInvocation
  rcases roundInvocation with positionMember | valueMember
  · unfold positionInvocations at positionMember
    rcases List.mem_map.mp positionMember with ⟨slot, _slotMember, rfl⟩
    by_cases roundZero : roundValue = 0
    · subst roundValue
      exact remappedPacket_implies_positionZeroInvocation env packets source slot
    · obtain ⟨previous, rfl⟩ := Nat.exists_eq_succ_of_ne_zero roundZero
      apply remappedPacket_implies_positionSuccInvocation env packets source
        previous
      simpa [roundCount] using roundLt
  · unfold valueInvocations at valueMember
    rcases List.mem_map.mp valueMember with ⟨slot, _slotMember, rfl⟩
    by_cases roundZero : roundValue = 0
    · subst roundValue
      exact remappedPacket_implies_valueZeroInvocation env packets source slot
    · obtain ⟨previous, rfl⟩ := Nat.exists_eq_succ_of_ne_zero roundZero
      apply remappedPacket_implies_valueSuccInvocation env packets source previous
      simpa [roundCount] using roundLt

end NightstreamFPrime.Export.Stage1.PiRLCFirst54Completeness
