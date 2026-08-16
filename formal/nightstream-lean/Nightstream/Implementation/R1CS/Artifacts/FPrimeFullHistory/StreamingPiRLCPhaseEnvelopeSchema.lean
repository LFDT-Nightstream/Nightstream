import Nightstream.Implementation.R1CS.Core.Poseidon2Sponge

/-!
Contract: compact exact-row schema for the PiRLC carry-phase semantic
envelope.

Owns the two local-digest alias blocks, one shared 2,169-bit delayed Nebula
payload, two fixed-width Poseidon2 preimages, and the exact repeated sponge
row recipe. A Rust drift test compares every represented row to the emitted
source relation.

Does not own the local digest semantics, lifecycle source links, selective-CCS
lowering, or Poseidon2 collision resistance.

Emits constraints: no. It describes Rust-emitted constraints.
-/

set_option autoImplicit false
set_option maxRecDepth 524288

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCPhaseEnvelope.Artifact

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.Poseidon2Call
open Nightstream.Implementation.R1CS.Poseidon2Sponge

def digestFields : Nat := 4
def payloadFields : Nat := 2169
def domainFields : Nat := 10
def hashConstantFields : Nat := domainFields + 1
def hashInputFields : Nat := hashConstantFields + digestFields + payloadFields
def absorbRounds : Nat := hashInputFields / 4
def permutationRows : Nat := 600
def absorbRoundRows : Nat := 4 + permutationRows
def hashTraceRows : Nat := 1 + absorbRounds * absorbRoundRows + 1 + permutationRows
def hashTotalRows : Nat := hashConstantFields + hashTraceRows
def aliasAndPayloadRows : Nat := digestFields + payloadFields + digestFields
def phaseRows : Nat := aliasAndPayloadRows + 2 * hashTotalRows

inductive StateSide where
  | before
  | after
deriving DecidableEq, Repr

structure RawArm where
  sourceIdentity : String
  sourceRowsSha256 : String
  bodyRows : Nat
  bodyColumns : Nat
  phaseRowStart : Nat
  phaseRowEnd : Nat
  phaseColumnStart : Nat
  phaseColumnEnd : Nat
  beforeLocalSourceColumns : List Nat
  afterLocalSourceColumns : List Nat
  beforeLocalAliasColumns : List Nat
  afterLocalAliasColumns : List Nat
  payloadStartColumn : Nat
  beforeHashConstantStartColumn : Nat
  afterHashConstantStartColumn : Nat
  beforeSemanticDigestColumns : List Nat
  afterSemanticDigestColumns : List Nat
  beforeXOutSemanticColumns : List Nat
  afterXOutSemanticColumns : List Nat
deriving DecidableEq, Repr

def RawArm.localSourceColumns (arm : RawArm) : StateSide → List Nat
  | .before => arm.beforeLocalSourceColumns
  | .after => arm.afterLocalSourceColumns

def RawArm.localAliasColumns (arm : RawArm) : StateSide → List Nat
  | .before => arm.beforeLocalAliasColumns
  | .after => arm.afterLocalAliasColumns

def RawArm.hashConstantStartColumn (arm : RawArm) : StateSide → Nat
  | .before => arm.beforeHashConstantStartColumn
  | .after => arm.afterHashConstantStartColumn

def RawArm.semanticDigestColumns (arm : RawArm) : StateSide → List Nat
  | .before => arm.beforeSemanticDigestColumns
  | .after => arm.afterSemanticDigestColumns

def RawArm.xOutSemanticColumns (arm : RawArm) : StateSide → List Nat
  | .before => arm.beforeXOutSemanticColumns
  | .after => arm.afterXOutSemanticColumns

def RawArm.payloadColumns (arm : RawArm) : List Nat :=
  List.range' arm.payloadStartColumn payloadFields

structure HashRecipe where
  constantValues : List Nat
  constantStartColumn : Nat
  localColumns : List Nat
  payloadColumns : List Nat
  outputColumns : List Nat
deriving DecidableEq, Repr

def HashRecipe.constantColumns (recipe : HashRecipe) : List Nat :=
  List.range' recipe.constantStartColumn recipe.constantValues.length

def HashRecipe.inputColumns (recipe : HashRecipe) : List Nat :=
  recipe.constantColumns ++ recipe.localColumns ++ recipe.payloadColumns

def HashRecipe.zeroColumn (recipe : HashRecipe) : Nat :=
  recipe.constantStartColumn + recipe.constantValues.length

def HashRecipe.roundColumnStart (recipe : HashRecipe) (round : Nat) : Nat :=
  recipe.zeroColumn + 1 + round * absorbRoundRows

def HashRecipe.definitionCount (round : Nat) : Nat :=
  if round < absorbRounds then 4 else 1

def HashRecipe.callFirstAllocatedColumn
    (recipe : HashRecipe) (round : Nat) : Nat :=
  recipe.roundColumnStart round + definitionCount round

def HashRecipe.callOutputColumns
    (recipe : HashRecipe) (round : Nat) : List Nat :=
  List.range' (recipe.callFirstAllocatedColumn round + 592) 8

def HashRecipe.stateBeforeColumns
    (recipe : HashRecipe) (round : Nat) : List Nat :=
  if round = 0 then List.replicate 8 recipe.zeroColumn
  else recipe.callOutputColumns (round - 1)

def HashRecipe.chunkColumns
    (recipe : HashRecipe) (round : Nat) : List Nat :=
  (recipe.inputColumns.drop (4 * round)).take 4

def HashRecipe.callInputColumns
    (recipe : HashRecipe) (round : Nat) : List Nat :=
  if round < absorbRounds then
    List.range' (recipe.roundColumnStart round) 4 ++
      (recipe.stateBeforeColumns round).drop 4
  else
    recipe.roundColumnStart round ::
      (recipe.stateBeforeColumns round).drop 1

def HashRecipe.call (recipe : HashRecipe) (round : Nat) : Call where
  rowStart := HashRecipe.definitionCount round
  rowEnd := HashRecipe.definitionCount round + permutationRows
  inputColumns := recipe.callInputColumns round
  firstAllocatedColumn := recipe.callFirstAllocatedColumn round

def HashRecipe.absorbRound (recipe : HashRecipe) (round : Nat) : Round where
  kind := .absorb (recipe.chunkColumns round)
  stateBeforeColumns := recipe.stateBeforeColumns round
  permutationInputColumns := recipe.callInputColumns round
  permutationOutputColumns := recipe.callOutputColumns round
  definingRows := List.range 4
  call := recipe.call round

def HashRecipe.padRound (recipe : HashRecipe) : Round where
  kind := .pad
  stateBeforeColumns := recipe.stateBeforeColumns absorbRounds
  permutationInputColumns := recipe.callInputColumns absorbRounds
  permutationOutputColumns := recipe.callOutputColumns absorbRounds
  definingRows := [0]
  call := recipe.call absorbRounds

def HashRecipe.rounds (recipe : HashRecipe) : List Round :=
  (List.range absorbRounds).map recipe.absorbRound ++ [recipe.padRound]

def HashRecipe.trace (recipe : HashRecipe) : Trace where
  inputColumns := recipe.inputColumns
  zeroColumn := recipe.zeroColumn
  zeroRow := 0
  rounds := recipe.rounds
  outputColumns := recipe.outputColumns

def RawArm.hashRecipe
    (arm : RawArm) (constantValues : List Nat) (side : StateSide) :
    HashRecipe where
  constantValues := constantValues
  constantStartColumn := arm.hashConstantStartColumn side
  localColumns := arm.localAliasColumns side
  payloadColumns := arm.payloadColumns
  outputColumns := arm.semanticDigestColumns side

def constantRows (recipe : HashRecipe) : List Row :=
  (recipe.constantColumns.zip recipe.constantValues).map fun entry =>
    builderLinearRow entry.1 [(0, entry.2)]

def aliasRows (sources aliases : List Nat) : List Row :=
  (aliases.zip sources).map fun entry =>
    builderLinearRow entry.1 [(entry.2, 1)]

def payloadRows (arm : RawArm) : List Row :=
  arm.payloadColumns.map bitRow

def RawArm.phasePieces (arm : RawArm)
    (constantValues : List Nat) : List (List Row) :=
  let before := arm.hashRecipe constantValues .before
  let after := arm.hashRecipe constantValues .after
  [aliasRows arm.beforeLocalSourceColumns arm.beforeLocalAliasColumns,
   payloadRows arm,
   aliasRows arm.afterLocalSourceColumns arm.afterLocalAliasColumns,
   constantRows before,
   before.trace.rows,
   constantRows after,
   after.trace.rows]

def RawArm.phaseProgram (arm : RawArm) (constantValues : List Nat) : List Row :=
  (arm.phasePieces constantValues).flatten

def RawArm.Satisfied
    (arm : RawArm) (constantValues : List Nat) (assignment : Nat → Nat) :
    Prop :=
  Satisfies (arm.phaseProgram constantValues) assignment

instance (arm : RawArm) (constantValues : List Nat)
    (assignment : Nat → Nat) :
    Decidable (arm.Satisfied constantValues assignment) := by
  unfold RawArm.Satisfied
  infer_instance

def columnsValid (columnCount expectedLength : Nat)
    (columns : List Nat) : Prop :=
  columns.length = expectedLength ∧ columns.Nodup ∧
    ∀ column ∈ columns, column < columnCount

instance (columnCount expectedLength : Nat) (columns : List Nat) :
    Decidable (columnsValid columnCount expectedLength columns) := by
  unfold columnsValid
  infer_instance

def HashRecipe.Valid (recipe : HashRecipe) (columnCount : Nat) : Prop :=
  recipe.constantValues.length = hashConstantFields ∧
    (∀ value ∈ recipe.constantValues, 0 < value ∧ value < goldilocksP) ∧
    columnsValid columnCount hashConstantFields recipe.constantColumns ∧
    columnsValid columnCount digestFields recipe.localColumns ∧
    columnsValid columnCount payloadFields recipe.payloadColumns ∧
    columnsValid columnCount digestFields recipe.outputColumns ∧
    recipe.inputColumns.length = hashInputFields ∧
    recipe.trace.OwnedValid

instance (recipe : HashRecipe) (columnCount : Nat) :
    Decidable (recipe.Valid columnCount) := by
  unfold HashRecipe.Valid
  infer_instance

def RawArm.Valid (arm : RawArm) (constantValues : List Nat) : Prop :=
  arm.sourceIdentity.length > 0 ∧ arm.sourceRowsSha256.length = 64 ∧
    arm.phaseRowEnd = arm.phaseRowStart + phaseRows ∧
    arm.phaseColumnEnd = arm.phaseColumnStart + phaseRows ∧
    arm.phaseRowEnd ≤ arm.bodyRows ∧ arm.phaseColumnEnd ≤ arm.bodyColumns ∧
    columnsValid arm.bodyColumns digestFields arm.beforeLocalSourceColumns ∧
    columnsValid arm.bodyColumns digestFields arm.afterLocalSourceColumns ∧
    (∀ column ∈ arm.beforeLocalSourceColumns, column < arm.phaseColumnStart) ∧
    (∀ column ∈ arm.afterLocalSourceColumns, column < arm.phaseColumnStart) ∧
    arm.beforeLocalAliasColumns =
      List.range' arm.phaseColumnStart digestFields ∧
    arm.payloadStartColumn = arm.phaseColumnStart + digestFields ∧
    arm.afterLocalAliasColumns =
      List.range' (arm.payloadStartColumn + payloadFields) digestFields ∧
    arm.beforeHashConstantStartColumn =
      arm.payloadStartColumn + payloadFields + digestFields ∧
    arm.afterHashConstantStartColumn =
      arm.beforeHashConstantStartColumn + hashTotalRows ∧
    arm.phaseColumnEnd = arm.afterHashConstantStartColumn + hashTotalRows ∧
    arm.beforeXOutSemanticColumns = arm.beforeSemanticDigestColumns ∧
    arm.afterXOutSemanticColumns = arm.afterSemanticDigestColumns ∧
    (arm.hashRecipe constantValues .before).Valid arm.bodyColumns ∧
    (arm.hashRecipe constantValues .after).Valid arm.bodyColumns

instance (arm : RawArm) (constantValues : List Nat) :
    Decidable (arm.Valid constantValues) := by
  unfold RawArm.Valid
  infer_instance

structure RawArtifact where
  schemaVersion : Nat
  profileId : String
  constantValues : List Nat
  even : RawArm
  odd : RawArm
deriving DecidableEq, Repr

def RawArtifact.Valid (artifact : RawArtifact) : Prop :=
  artifact.schemaVersion = 1 ∧
    artifact.profileId =
      "nebula-f-prime-streaming-pi-rlc-phase-envelope-v1" ∧
    artifact.constantValues =
      [57, 30521782141150574, 31069335676202596,
       27422324158721583, 30796712690673199, 27414614995316581,
       29396737889036653, 30792317818729313, 33266151269363297,
       49, 2169] ∧
    hashInputFields = 2184 ∧ absorbRounds = 546 ∧
    hashTraceRows = 330386 ∧ hashTotalRows = 330397 ∧ phaseRows = 662971 ∧
    artifact.even.sourceIdentity = "rust:pi-rlc-family-even/body-v3" ∧
    artifact.even.bodyRows = 1232857 ∧
    artifact.even.bodyColumns = 1233086 ∧
    artifact.even.phaseRowStart = 558380 ∧
    artifact.even.phaseRowEnd = 1221351 ∧
    artifact.odd.sourceIdentity = "rust:pi-rlc-family-odd/body-v3" ∧
    artifact.odd.bodyRows = 1234057 ∧
    artifact.odd.bodyColumns = 1234286 ∧
    artifact.odd.phaseRowStart = 559580 ∧
    artifact.odd.phaseRowEnd = 1222551 ∧
    artifact.even.Valid artifact.constantValues ∧
    artifact.odd.Valid artifact.constantValues

instance (artifact : RawArtifact) : Decidable artifact.Valid := by
  unfold RawArtifact.Valid
  infer_instance

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCPhaseEnvelope.Artifact
