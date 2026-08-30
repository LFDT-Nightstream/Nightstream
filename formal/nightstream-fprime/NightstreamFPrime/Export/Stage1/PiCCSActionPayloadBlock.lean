import NightstreamFPrime.Export.Stage1.PiCCSInvocations
import NightstreamFPrime.Export.Stage1.PiRLCPoseidonGeometry
import NightstreamFPrime.Export.Stage1.PoseidonActionSchedule
import NightstreamFPrime.Layout.ProductionRelation.FieldSuffixBlock

/-!
Owns the retained four-lane payload for every PiCCS Poseidon2 invocation.
The exact action lists remain the transcript authority. This module only
selects their absorb chunks in invocation-major order and gives squeeze
invocations a zero payload.

Original PiCCS expressions are evaluated through the per-application column
pullback. The prover does not select an application or an action schedule.
-/

namespace NightstreamFPrime.Export.Stage1.PiCCSActionPayloadBlock

open NightstreamFPrime.Circuit
open NightstreamFPrime.Export.Stage1.Invocations
open NightstreamFPrime.Gadgets.Poseidon2
open NightstreamFPrime.Gadgets.Poseidon2.Duplex
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Lifecycle.PiCCS.v1_1
open NightstreamFPrime.Spec

def statementActions : List Formal.Action :=
  PiCCSInvocations.statementActions Data.logicalWidth Data.publicFits

def challengeActions : List Formal.Action :=
  ChallengeDerivation.actions
    (PiCCSInvocations.challengeInterface Data.logicalWidth Data.publicFits)
    PiCCSInvocations.challengeWitnessStart

def roundActions : List Formal.Action :=
  RoundTranscript.actions
    (PiCCSInvocations.roundInterface Data.logicalWidth Data.publicFits)
    PiCCSInvocations.roundWitnessStart

def outputActions : List Formal.Action :=
  PiCCSInvocations.outputActions Data.logicalWidth Data.publicFits

theorem challengeInvocationCount_eq :
    Invocations.invocationCount challengeActions = 87 := by
  have same := Invocations.invocationCount_eq_of_shapes challengeActions
    (PiCCSInvocations.challengeActions Data.logicalWidth Data.publicFits) (by
      unfold challengeActions
      exact (PiCCSInvocations.challengeActions_shape_matches
        Data.logicalWidth Data.publicFits).symm)
  exact same.trans (PiCCSInvocations.challengeInvocationCount_eq
    Data.logicalWidth Data.publicFits)

theorem roundInvocationCount_eq :
    Invocations.invocationCount roundActions = 252 := by
  have same := Invocations.invocationCount_eq_of_shapes roundActions
    (PiCCSInvocations.roundActions Data.logicalWidth Data.publicFits) (by
      unfold roundActions
      exact (PiCCSInvocations.roundActions_shape_matches
        Data.logicalWidth Data.publicFits).symm)
  exact same.trans (PiCCSInvocations.roundInvocationCount_eq
    Data.logicalWidth Data.publicFits)

def statementKindAt : Fin 325 → PoseidonActionSchedule.Kind :=
  fun index => PoseidonActionSchedule.kindAt statementActions <|
    Fin.cast (PiCCSInvocations.statementInvocationCount_eq
      Data.logicalWidth Data.publicFits).symm index

def challengeKindAt : Fin 87 → PoseidonActionSchedule.Kind :=
  fun index => PoseidonActionSchedule.kindAt challengeActions <|
    Fin.cast challengeInvocationCount_eq.symm index

def roundKindAt : Fin 252 → PoseidonActionSchedule.Kind :=
  fun index => PoseidonActionSchedule.kindAt roundActions <|
    Fin.cast roundInvocationCount_eq.symm index

def outputKindAt : Fin 6886 → PoseidonActionSchedule.Kind :=
  fun index => PoseidonActionSchedule.kindAt outputActions <|
    Fin.cast (PiCCSInvocations.outputInvocationCount_eq
      Data.logicalWidth Data.publicFits).symm index

theorem statementKindAt_materializes :
    List.ofFn statementKindAt = PoseidonActionSchedule.kinds statementActions := by
  calc
    List.ofFn statementKindAt =
        List.ofFn (PoseidonActionSchedule.kindAt statementActions) :=
      (List.ofFn_congr
        (PiCCSInvocations.statementInvocationCount_eq
          Data.logicalWidth Data.publicFits)
        (PoseidonActionSchedule.kindAt statementActions)).symm
    _ = PoseidonActionSchedule.kinds statementActions :=
      PoseidonActionSchedule.kindAt_materializes statementActions

theorem challengeKindAt_materializes :
    List.ofFn challengeKindAt = PoseidonActionSchedule.kinds challengeActions := by
  calc
    List.ofFn challengeKindAt =
        List.ofFn (PoseidonActionSchedule.kindAt challengeActions) :=
      (List.ofFn_congr
        challengeInvocationCount_eq
        (PoseidonActionSchedule.kindAt challengeActions)).symm
    _ = PoseidonActionSchedule.kinds challengeActions :=
      PoseidonActionSchedule.kindAt_materializes challengeActions

theorem roundKindAt_materializes :
    List.ofFn roundKindAt = PoseidonActionSchedule.kinds roundActions := by
  calc
    List.ofFn roundKindAt =
        List.ofFn (PoseidonActionSchedule.kindAt roundActions) :=
      (List.ofFn_congr
        roundInvocationCount_eq
        (PoseidonActionSchedule.kindAt roundActions)).symm
    _ = PoseidonActionSchedule.kinds roundActions :=
      PoseidonActionSchedule.kindAt_materializes roundActions

theorem outputKindAt_materializes :
    List.ofFn outputKindAt = PoseidonActionSchedule.kinds outputActions := by
  calc
    List.ofFn outputKindAt =
        List.ofFn (PoseidonActionSchedule.kindAt outputActions) :=
      (List.ofFn_congr
        (PiCCSInvocations.outputInvocationCount_eq
          Data.logicalWidth Data.publicFits)
        (PoseidonActionSchedule.kindAt outputActions)).symm
    _ = PoseidonActionSchedule.kinds outputActions :=
      PoseidonActionSchedule.kindAt_materializes outputActions

def invocationCount : Nat := 7550

@[simp] theorem invocationCount_eq : invocationCount = 7550 := by
  rfl

def payloadCount : Nat := invocationCount * Spec.Poseidon2.rate

@[simp] theorem payloadCount_eq : payloadCount = 30200 := by
  rw [payloadCount, invocationCount_eq]
  rfl

def kindAt : Fin invocationCount → PoseidonActionSchedule.Kind :=
  Fin.append statementKindAt <|
    Fin.append challengeKindAt <| Fin.append roundKindAt outputKindAt

/-- The random-access schedule materializes to the canonical action expansion. -/
theorem kindAt_materializes :
    List.ofFn kindAt =
      PoseidonActionSchedule.kinds statementActions ++
        (PoseidonActionSchedule.kinds challengeActions ++
          (PoseidonActionSchedule.kinds roundActions ++
            PoseidonActionSchedule.kinds outputActions)) := by
  calc
    List.ofFn kindAt =
        List.ofFn statementKindAt ++
          (List.ofFn challengeKindAt ++
            (List.ofFn roundKindAt ++ List.ofFn outputKindAt)) := by
      unfold kindAt invocationCount
      rw [List.ofFn_fin_append, List.ofFn_fin_append,
        List.ofFn_fin_append]
    _ = _ := by
      rw [statementKindAt_materializes, challengeKindAt_materializes,
        roundKindAt_materializes, outputKindAt_materializes]

theorem kindAt_wellFormed (invocation : Fin invocationCount) :
    (kindAt invocation).WellFormed := by
  have member : kindAt invocation ∈ List.ofFn kindAt :=
    (List.mem_ofFn).2 ⟨invocation, rfl⟩
  rw [kindAt_materializes] at member
  simp only [List.mem_append] at member
  rcases member with statement | challenge | round | output
  · exact PoseidonActionSchedule.kinds_wellFormed statementActions _ statement
  · exact PoseidonActionSchedule.kinds_wellFormed challengeActions _ challenge
  · exact PoseidonActionSchedule.kinds_wellFormed roundActions _ round
  · exact PoseidonActionSchedule.kinds_wellFormed outputActions _ output

def selectedBlock (invocation : Fin invocationCount) : List Expr :=
  match kindAt invocation with
  | .absorb block => block
  | .squeezeFirst expected => [expected.c0, expected.c1]
  | .squeezeSecond => []

/-- One exact rate-lane payload. `getD` is the canonical absorb padding rule. -/
def payloadExpr (invocation : Fin invocationCount)
    (lane : Fin Spec.Poseidon2.rate) : Expr :=
  (selectedBlock invocation).getD lane.val 0

def payloadExpression (index : Fin payloadCount) : Expr :=
  let decoded : Fin invocationCount × Fin Spec.Poseidon2.rate :=
    Fin.decodeProd index
  payloadExpr decoded.1 decoded.2

@[simp] theorem payloadExpression_encode
    (invocation : Fin invocationCount) (lane : Fin Spec.Poseidon2.rate) :
    payloadExpression (Fin.encodeProd (invocation, lane)) =
      payloadExpr invocation lane := by
  simp [payloadExpression]

def prefixSourceWidth (program : Lifecycle.Stage1.Application.Program) : Nat :=
  PiRLCRetainedGeometry.sourceWidth program

def sourceWidth (program : Lifecycle.Stage1.Application.Program) : Nat :=
  FieldSuffixBlock.sourceWidth (prefixSourceWidth program) payloadCount

def prefixColumn (program : Lifecycle.Stage1.Application.Program)
    (column : Fin (prefixSourceWidth program)) : Fin (sourceWidth program) :=
  FieldSuffixBlock.baseColumn (prefixSourceWidth program) payloadCount column

def payloadColumn (program : Lifecycle.Stage1.Application.Program)
    (index : Fin payloadCount) : Fin (sourceWidth program) :=
  FieldSuffixBlock.derivedColumn (prefixSourceWidth program) payloadCount index

/-- Original PiCCS expressions read the exact shifted per-application package
columns from the retained prefix. -/
def packageEnv (program : Lifecycle.Stage1.Application.Program)
    (prefixAssignment : Fin (prefixSourceWidth program) → F) : Env :=
  NightstreamFPrime.Layout.Stage1.Spartan.pullback <|
    PerApplicationPackage.baseEnv program
      (SourceCompiler.sourceEnv prefixAssignment)

def payloadValue (program : Lifecycle.Stage1.Application.Program)
    (prefixAssignment : Fin (prefixSourceWidth program) → F)
    (index : Fin payloadCount) : F :=
  (payloadExpression index).eval (packageEnv program prefixAssignment)

def sourceAssignment (program : Lifecycle.Stage1.Application.Program)
    (prefixAssignment : Fin (prefixSourceWidth program) → F) :
    Fin (sourceWidth program) → F :=
  FieldSuffixBlock.sourceAssignment (prefixSourceWidth program) payloadCount
    prefixAssignment (payloadValue program prefixAssignment)

@[simp] theorem sourceAssignment_prefix
    (program : Lifecycle.Stage1.Application.Program)
    (prefixAssignment : Fin (prefixSourceWidth program) → F)
    (column : Fin (prefixSourceWidth program)) :
    sourceAssignment program prefixAssignment (prefixColumn program column) =
      prefixAssignment column := by
  exact FieldSuffixBlock.sourceAssignment_base _ _ _ _ column

@[simp] theorem sourceAssignment_payload
    (program : Lifecycle.Stage1.Application.Program)
    (prefixAssignment : Fin (prefixSourceWidth program) → F)
    (index : Fin payloadCount) :
    sourceAssignment program prefixAssignment (payloadColumn program index) =
      payloadValue program prefixAssignment index := by
  exact FieldSuffixBlock.sourceAssignment_derived _ _ _ _ index

def block (program : Lifecycle.Stage1.Application.Program) :
    LowNormBlock.Block (sourceWidth program) :=
  FieldSuffixBlock.block (prefixSourceWidth program) payloadCount

@[simp] theorem block_slotCount
    (program : Lifecycle.Stage1.Application.Program) :
    (block program).slotCount = 30200 := by
  rw [block, FieldSuffixBlock.block_slotCount, payloadCount_eq]

@[simp] theorem block_coordinateCount
    (program : Lifecycle.Stage1.Application.Program) :
    (block program).coordinateCount = 1238200 := by
  change payloadCount * 41 = 1238200
  rw [payloadCount_eq]

theorem block_sourceAssignment
    (program : Lifecycle.Stage1.Application.Program)
    (prefixAssignment : Fin (prefixSourceWidth program) → F)
    (index : Fin payloadCount) :
    sourceAssignment program prefixAssignment ((block program).source index) =
      payloadValue program prefixAssignment index := by
  exact FieldSuffixBlock.block_sourceAssignment _ _ _ _ index

def payloadStart (program : Lifecycle.Stage1.Application.Program) : Nat :=
  PiRLCPoseidonGeometry.pilotLogicalWidth program

def logicalWidth (program : Lifecycle.Stage1.Application.Program) : Nat :=
  payloadStart program + (block program).coordinateCount

@[simp] theorem logicalWidth_eq
    (program : Lifecycle.Stage1.Application.Program) :
    logicalWidth program = 185542820 := by
  rw [logicalWidth, payloadStart, PiRLCPoseidonGeometry.pilotLogicalWidth_eq,
    block_coordinateCount]

theorem logicalWidth_le_cube
    (program : Lifecycle.Stage1.Application.Program) :
    logicalWidth program ≤ 2 ^ NightstreamFPrime.Lifecycle.cubeVariables := by
  rw [logicalWidth_eq]
  norm_num [NightstreamFPrime.Lifecycle.cubeVariables]

end NightstreamFPrime.Export.Stage1.PiCCSActionPayloadBlock
