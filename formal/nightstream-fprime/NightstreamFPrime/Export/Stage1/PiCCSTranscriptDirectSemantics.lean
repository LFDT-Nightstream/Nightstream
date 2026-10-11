import NightstreamFPrime.Export.Stage1.PiCCSPoseidonPreservation

/-!
Owns the exact transcript-family views of the one direct PiCCS Poseidon2
schedule. Statement, challenge, round, and output views are fixed slices of
the global indexed schedule and retain the lifecycle action-list authority.

This module does not assemble the non-transcript PiCCS leaves.
-/

namespace NightstreamFPrime.Export.Stage1.PiCCSTranscriptDirectSemantics

open NightstreamFPrime.Export.Stage1
open NightstreamFPrime.Gadgets.Poseidon2.Duplex
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra

def statementOffset : Nat := 0
def challengeOffset : Nat := 123
def roundOffset : Nat := 127
def outputOffset : Nat := 183

def statementCount : Nat := 123
def challengeCount : Nat := 4
def roundCount : Nat := 56
def outputCount : Nat := 765

def statementFits : statementOffset + statementCount ≤
    PiCCSActionPayloadBlock.invocationCount := by
  norm_num [statementOffset, statementCount,
    PiCCSActionPayloadBlock.invocationCount]

def challengeFits : challengeOffset + challengeCount ≤
    PiCCSActionPayloadBlock.invocationCount := by
  norm_num [challengeOffset, challengeCount,
    PiCCSActionPayloadBlock.invocationCount]

def roundFits : roundOffset + roundCount ≤
    PiCCSActionPayloadBlock.invocationCount := by
  norm_num [roundOffset, roundCount,
    PiCCSActionPayloadBlock.invocationCount]

def outputFits : outputOffset + outputCount ≤
    PiCCSActionPayloadBlock.invocationCount := by
  norm_num [outputOffset, outputCount,
    PiCCSActionPayloadBlock.invocationCount]

def statementOffsetBound : statementOffset <
    PiCCSActionPayloadBlock.invocationCount := by
  norm_num [statementOffset, PiCCSActionPayloadBlock.invocationCount]

def challengeOffsetBound : challengeOffset <
    PiCCSActionPayloadBlock.invocationCount := by
  norm_num [challengeOffset, PiCCSActionPayloadBlock.invocationCount]

def roundOffsetBound : roundOffset <
    PiCCSActionPayloadBlock.invocationCount := by
  norm_num [roundOffset, PiCCSActionPayloadBlock.invocationCount]

def outputOffsetBound : outputOffset <
    PiCCSActionPayloadBlock.invocationCount := by
  norm_num [outputOffset, PiCCSActionPayloadBlock.invocationCount]

def statementKindAt : Fin statementCount → PoseidonActionSchedule.Kind :=
  fun current => PiCCSActionPayloadBlock.kindAt <|
    PoseidonActionSemantics.sliceIndex statementOffset statementCount
      statementFits current

def challengeKindAt : Fin challengeCount → PoseidonActionSchedule.Kind :=
  fun current => PiCCSActionPayloadBlock.kindAt <|
    PoseidonActionSemantics.sliceIndex challengeOffset challengeCount
      challengeFits current

def roundKindAt : Fin roundCount → PoseidonActionSchedule.Kind :=
  fun current => PiCCSActionPayloadBlock.kindAt <|
    PoseidonActionSemantics.sliceIndex roundOffset roundCount roundFits current

def outputKindAt : Fin outputCount → PoseidonActionSchedule.Kind :=
  fun current => PiCCSActionPayloadBlock.kindAt <|
    PoseidonActionSemantics.sliceIndex outputOffset outputCount outputFits current

theorem statementKindAt_eq (current : Fin statementCount) :
    statementKindAt current = PiCCSActionPayloadBlock.statementKindAt current := by
  have currentBound := current.isLt
  unfold statementCount at currentBound
  unfold statementKindAt PoseidonActionSemantics.sliceIndex
    statementOffset statementCount PiCCSActionPayloadBlock.kindAt
    PiCCSActionPayloadBlock.invocationCount
  rw [show
      (⟨0 + current.val, by omega⟩ : Fin 948) =
        Fin.castAdd (4 + (56 + 765)) current by
    apply Fin.ext
    change 0 + current.val = current.val
    omega]
  simp [statementCount, challengeCount, roundCount, outputCount]

theorem challengeKindAt_eq (current : Fin challengeCount) :
    challengeKindAt current = PiCCSActionPayloadBlock.challengeKindAt current := by
  have currentBound := current.isLt
  unfold challengeCount at currentBound
  unfold challengeKindAt PoseidonActionSemantics.sliceIndex challengeOffset
    challengeCount PiCCSActionPayloadBlock.kindAt
    PiCCSActionPayloadBlock.invocationCount
  rw [show
      (⟨123 + current.val, by omega⟩ : Fin 948) =
        Fin.natAdd 123 (Fin.castAdd (56 + 765) current) by
    apply Fin.ext
    change 123 + current.val = 123 + current.val
    rfl]
  simp [statementCount, challengeCount, roundCount, outputCount]

theorem roundKindAt_eq (current : Fin roundCount) :
    roundKindAt current = PiCCSActionPayloadBlock.roundKindAt current := by
  have currentBound := current.isLt
  unfold roundCount at currentBound
  unfold roundKindAt PoseidonActionSemantics.sliceIndex roundOffset roundCount
    PiCCSActionPayloadBlock.kindAt PiCCSActionPayloadBlock.invocationCount
  rw [show
      (⟨127 + current.val, by omega⟩ : Fin 948) =
        Fin.natAdd 123
          (Fin.natAdd 4 (Fin.castAdd 765 current)) by
    apply Fin.ext
    change 127 + current.val = 123 + (4 + current.val)
    omega]
  simp [statementCount, challengeCount, roundCount, outputCount]

theorem outputKindAt_eq (current : Fin outputCount) :
    outputKindAt current = PiCCSActionPayloadBlock.outputKindAt current := by
  have currentBound := current.isLt
  unfold outputCount at currentBound
  unfold outputKindAt PoseidonActionSemantics.sliceIndex outputOffset outputCount
    PiCCSActionPayloadBlock.kindAt PiCCSActionPayloadBlock.invocationCount
  rw [show
      (⟨183 + current.val, by omega⟩ : Fin 948) =
        Fin.natAdd 123 (Fin.natAdd 4 (Fin.natAdd 56 current)) by
    apply Fin.ext
    change 183 + current.val = 123 + (4 + (56 + current.val))
    omega]
  simp [statementCount, challengeCount, roundCount, outputCount]

theorem statementKindAt_materializes :
    List.ofFn statementKindAt =
      PoseidonActionSchedule.kinds PiCCSActionPayloadBlock.statementActions := by
  calc
    List.ofFn statementKindAt =
        List.ofFn PiCCSActionPayloadBlock.statementKindAt := by
      apply congrArg List.ofFn
      funext current
      exact statementKindAt_eq current
    _ = _ := PiCCSActionPayloadBlock.statementKindAt_materializes

theorem challengeKindAt_materializes :
    List.ofFn challengeKindAt =
      PoseidonActionSchedule.kinds PiCCSActionPayloadBlock.challengeActions := by
  calc
    List.ofFn challengeKindAt =
        List.ofFn PiCCSActionPayloadBlock.challengeKindAt := by
      apply congrArg List.ofFn
      funext current
      exact challengeKindAt_eq current
    _ = _ := PiCCSActionPayloadBlock.challengeKindAt_materializes

theorem roundKindAt_materializes :
    List.ofFn roundKindAt =
      PoseidonActionSchedule.kinds PiCCSActionPayloadBlock.roundActions := by
  calc
    List.ofFn roundKindAt =
        List.ofFn PiCCSActionPayloadBlock.roundKindAt := by
      apply congrArg List.ofFn
      funext current
      exact roundKindAt_eq current
    _ = _ := PiCCSActionPayloadBlock.roundKindAt_materializes

theorem outputKindAt_materializes :
    List.ofFn outputKindAt =
      PoseidonActionSchedule.kinds PiCCSActionPayloadBlock.outputActions := by
  calc
    List.ofFn outputKindAt =
        List.ofFn PiCCSActionPayloadBlock.outputKindAt := by
      apply congrArg List.ofFn
      funext current
      exact outputKindAt_eq current
    _ = _ := PiCCSActionPayloadBlock.outputKindAt_materializes

def statementLast : Fin PiCCSActionPayloadBlock.invocationCount := ⟨122, by
  norm_num [PiCCSActionPayloadBlock.invocationCount]⟩

def challengeLast : Fin PiCCSActionPayloadBlock.invocationCount := ⟨126, by
  norm_num [PiCCSActionPayloadBlock.invocationCount]⟩

def roundLast : Fin PiCCSActionPayloadBlock.invocationCount := ⟨182, by
  norm_num [PiCCSActionPayloadBlock.invocationCount]⟩

def outputLast : Fin PiCCSActionPayloadBlock.invocationCount := ⟨947, by
  norm_num [PiCCSActionPayloadBlock.invocationCount]⟩

/-- The value state after `index` invocations of one transcript slice. -/
def sliceState {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (geometry : PiCCSPoseidonPlan.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth) (offset count : Nat)
    (fits : offset + count ≤ PiCCSActionPayloadBlock.invocationCount)
    (offsetBound : offset < PiCCSActionPayloadBlock.invocationCount) :
    Nat → Spec.Poseidon2.State :=
  PoseidonActionSemantics.stateAfter
    (PoseidonActionSemantics.sliceInitial Spec.Poseidon2.zeroState
      (PiCCSPoseidonPreservation.valueState geometry assignment) offset
      offsetBound)
    (PoseidonActionSemantics.sliceOutput
      (PiCCSPoseidonPreservation.valueState geometry assignment) offset count
      fits)

/-- Every challenge and round read sees the value state at its position. -/
structure Reads {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (geometry : PiCCSPoseidonPlan.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (env : Circuit.Env) : Prop where
  challenge : PoseidonActionSemantics.ReadsAt env
    (sliceState geometry assignment challengeOffset challengeCount
      challengeFits challengeOffsetBound) 0
    PiCCSActionPayloadBlock.challengeActions
  rounds : PoseidonActionSemantics.ReadsAt env
    (sliceState geometry assignment roundOffset roundCount roundFits
      roundOffsetBound) 0 PiCCSActionPayloadBlock.roundActions

structure Traces {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (geometry : PiCCSPoseidonPlan.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (env : Circuit.Env) : Prop where
  statement : Formal.TraceHolds Spec.Poseidon2.zeroState
    (PiCCSActionPayloadBlock.statementActions.map
      (Formal.Action.eval env))
    (PiCCSPoseidonPreservation.valueState geometry assignment statementLast)
  challenge : Formal.TraceHolds
    (PiCCSPoseidonPreservation.valueState geometry assignment statementLast)
    (PiCCSActionPayloadBlock.challengeActions.map
      (Formal.Action.eval env))
    (PiCCSPoseidonPreservation.valueState geometry assignment challengeLast)
  rounds : Formal.TraceHolds
    (PiCCSPoseidonPreservation.valueState geometry assignment challengeLast)
    (PiCCSActionPayloadBlock.roundActions.map
      (Formal.Action.eval env))
    (PiCCSPoseidonPreservation.valueState geometry assignment roundLast)
  output : Formal.TraceHolds
    (PiCCSPoseidonPreservation.valueState geometry assignment roundLast)
    (PiCCSActionPayloadBlock.outputActions.map
      (Formal.Action.eval env))
    (PiCCSPoseidonPreservation.valueState geometry assignment outputLast)

theorem indexedSemantics_implies_traces
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiCCSPoseidonPlan.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (env : Circuit.Env)
    (semantics : PoseidonActionSemantics.IndexedSemantics env
      Spec.Poseidon2.zeroState PiCCSActionPayloadBlock.kindAt
      (PiCCSPoseidonPreservation.valueState geometry assignment))
    (reads : Reads geometry assignment env) :
    Traces geometry assignment env := by
  let globalOutput := PiCCSPoseidonPreservation.valueState geometry assignment
  have global := semantics
  have statementSemantics := global.slice statementOffset statementCount
    statementFits statementOffsetBound
  have challengeSemantics := global.slice challengeOffset challengeCount
    challengeFits challengeOffsetBound
  have roundSemantics := global.slice roundOffset roundCount roundFits
    roundOffsetBound
  have outputSemantics := global.slice outputOffset outputCount outputFits
    outputOffsetBound
  have statementTrace := PoseidonActionSemantics.indexed_traceHolds
    statementCount env
    (PoseidonActionSemantics.sliceInitial Spec.Poseidon2.zeroState globalOutput
      statementOffset statementOffsetBound)
    statementKindAt
    (PoseidonActionSemantics.sliceOutput globalOutput statementOffset
      statementCount statementFits)
    PiCCSActionPayloadBlock.statementActions statementKindAt_materializes
    statementSemantics (by
      simp only [PiCCSActionPayloadBlock.statementActions,
        PiCCSInvocations.statementActions, Lifecycle.PiCCS.v1_2.StatementAbsorption.actions,
        Lifecycle.PiCCS.v1_2.StatementAbsorption.publicInputActions,
        PoseidonActionSemantics.ReadsAt])
  have challengeTrace := PoseidonActionSemantics.indexed_traceHolds
    challengeCount env
    (PoseidonActionSemantics.sliceInitial Spec.Poseidon2.zeroState globalOutput
      challengeOffset challengeOffsetBound)
    challengeKindAt
    (PoseidonActionSemantics.sliceOutput globalOutput challengeOffset
      challengeCount challengeFits)
    PiCCSActionPayloadBlock.challengeActions challengeKindAt_materializes
    challengeSemantics reads.challenge
  have roundTrace := PoseidonActionSemantics.indexed_traceHolds
    roundCount env
    (PoseidonActionSemantics.sliceInitial Spec.Poseidon2.zeroState globalOutput
      roundOffset roundOffsetBound)
    roundKindAt
    (PoseidonActionSemantics.sliceOutput globalOutput roundOffset roundCount
      roundFits)
    PiCCSActionPayloadBlock.roundActions roundKindAt_materializes roundSemantics
    reads.rounds
  have outputTrace := PoseidonActionSemantics.indexed_traceHolds
    outputCount env
    (PoseidonActionSemantics.sliceInitial Spec.Poseidon2.zeroState globalOutput
      outputOffset outputOffsetBound)
    outputKindAt
    (PoseidonActionSemantics.sliceOutput globalOutput outputOffset outputCount
      outputFits)
    PiCCSActionPayloadBlock.outputActions outputKindAt_materializes
    outputSemantics (by
      simp only [PiCCSActionPayloadBlock.outputActions,
        PiCCSInvocations.outputActions, Lifecycle.PiCCS.v1_2.OutputBinding.actions,
        PoseidonActionSemantics.ReadsAt])
  refine ⟨?_, ?_, ?_, ?_⟩
  · simpa [globalOutput, statementCount, statementOffset, statementLast,
      PoseidonActionSemantics.sliceInitial,
      PoseidonActionSemantics.sliceOutput,
      PoseidonActionSemantics.sliceIndex, PoseidonActionSemantics.stateAfter]
      using statementTrace
  · simpa [globalOutput, challengeCount, challengeOffset, statementLast,
      challengeLast, PoseidonActionSemantics.sliceInitial,
      PoseidonActionSemantics.sliceOutput,
      PoseidonActionSemantics.sliceIndex, PoseidonActionSemantics.stateAfter]
      using challengeTrace
  · simpa [globalOutput, roundCount, roundOffset, challengeLast,
      roundLast, PoseidonActionSemantics.sliceInitial,
      PoseidonActionSemantics.sliceOutput,
      PoseidonActionSemantics.sliceIndex, PoseidonActionSemantics.stateAfter]
      using roundTrace
  · simpa [globalOutput, outputCount, outputOffset, roundLast, outputLast,
      PoseidonActionSemantics.sliceInitial,
      PoseidonActionSemantics.sliceOutput,
      PoseidonActionSemantics.sliceIndex, PoseidonActionSemantics.stateAfter]
      using outputTrace

end NightstreamFPrime.Export.Stage1.PiCCSTranscriptDirectSemantics
