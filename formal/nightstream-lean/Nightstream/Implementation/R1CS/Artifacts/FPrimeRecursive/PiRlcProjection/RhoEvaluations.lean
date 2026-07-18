import Nightstream.Implementation.R1CS.Artifacts.FPrimeRecursive.PiRlcProjection.BetaLadder
import Nightstream.Implementation.R1CS.Artifacts.FPrimeRecursive.PiRlcProjection.Generated.RhoEvaluationsData0
import Nightstream.Implementation.R1CS.Artifacts.FPrimeRecursive.PiRlcProjection.Generated.RhoEvaluationsData1
import Nightstream.Implementation.R1CS.Artifacts.FPrimeRecursive.PiRlcProjection.Generated.RhoEvaluationsData2

/-!
Stable artifact facade for the active shared PiRLC rho evaluations.

Owns: one checked 15-leaf owner tree, three exact 540-row shards, exact
row/column adjacency, and linkage to the 54-power prefix of the active beta
ladder.

Does not own: transcript derivation or semantic authority of rho, beta
transcript authority, projection-identity soundness, whole-matrix embedding,
encoded lowering, costs beyond the source-R1CS rows, or row removal.

Emits constraints: no.

Assurance tier: artifact-checked local ownership after the Rust drift test.
This is not whole-verifier Rust conformance.

| Child path | Mathematical obligation | Exact owner | Consumers |
|---|---|---:|---|
| `projection_shared.rho_evaluations` | evaluate 15 exact 54-coefficient rho polynomials at one beta | 1,620 source-R1CS rows | both returned-parent `y_zcol` identities |
| row shards 0/1/2 | preserve five ordered evaluator leaves per shard | 540 rows each | stable facade only |
-/

namespace Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcProjection.RhoEvaluations

open Nightstream.Implementation.R1CS

def generatedStagePath : String :=
  FPrimeRecursivePiRlcProjectionRhoEvaluationsData.Shard0.stagePath
def generatedStageRowStart : Nat :=
  FPrimeRecursivePiRlcProjectionRhoEvaluationsData.Shard0.stageRowStart
def generatedStageRowEnd : Nat :=
  FPrimeRecursivePiRlcProjectionRhoEvaluationsData.Shard0.stageRowEnd
def generatedStageAllocatedStart : Nat :=
  FPrimeRecursivePiRlcProjectionRhoEvaluationsData.Shard0.stageAllocatedStart
def generatedStageAllocatedEnd : Nat :=
  FPrimeRecursivePiRlcProjectionRhoEvaluationsData.Shard0.stageAllocatedEnd
def generatedConsumerIdentityIndices : List Nat :=
  FPrimeRecursivePiRlcProjectionRhoEvaluationsData.Shard0.consumerIdentityIndices
def ladderPowerColumns : List ProjectionProgram.KColumns :=
  FPrimeRecursivePiRlcProjectionBetaLadderData.powerColumns

def expectedStagePath : String :=
  "nifs.pi_rlc.verify.projection_shared.rho_evaluations"

def coefficientCount : Nat := 54

def evaluationCount : Nat := 15

def owners0 : List PiRlcRhoEvaluationOwner :=
  FPrimeRecursivePiRlcProjectionRhoEvaluationsData.Shard0.owners
def owners1 : List PiRlcRhoEvaluationOwner :=
  FPrimeRecursivePiRlcProjectionRhoEvaluationsData.Shard1.owners
def owners2 : List PiRlcRhoEvaluationOwner :=
  FPrimeRecursivePiRlcProjectionRhoEvaluationsData.Shard2.owners

def owners : List PiRlcRhoEvaluationOwner := owners0 ++ owners1 ++ owners2

def sourceRows0 : List (Nat × Row) :=
  FPrimeRecursivePiRlcProjectionRhoEvaluationsData.Shard0.sourceRows
def sourceRows1 : List (Nat × Row) :=
  FPrimeRecursivePiRlcProjectionRhoEvaluationsData.Shard1.sourceRows
def sourceRows2 : List (Nat × Row) :=
  FPrimeRecursivePiRlcProjectionRhoEvaluationsData.Shard2.sourceRows

def sourceRows : List (Nat × Row) :=
  sourceRows0 ++ sourceRows1 ++ sourceRows2

def rowDefinitionsFor (chunk : List PiRlcRhoEvaluationOwner) :
    List (Nat × Program.Definition) :=
  chunk.flatMap PiRlcRhoEvaluationOwner.rowDefinitions

def rowDefinitions0 : List (Nat × Program.Definition) :=
  rowDefinitionsFor owners0
def rowDefinitions1 : List (Nat × Program.Definition) :=
  rowDefinitionsFor owners1
def rowDefinitions2 : List (Nat × Program.Definition) :=
  rowDefinitionsFor owners2

def ownedRowDefinitions : List (Nat × Program.Definition) :=
  rowDefinitions0 ++ rowDefinitions1 ++ rowDefinitions2

def ownedSourceRows : List Row := sourceRows.map Prod.snd

def ShardRowsMatch (rows : List (Nat × Row))
    (definitions : List (Nat × Program.Definition)) : Prop :=
  ActiveIndexedRows.indexedRowsMatch rows definitions = true

instance (rows : List (Nat × Row))
    (definitions : List (Nat × Program.Definition)) :
    Decidable (ShardRowsMatch rows definitions) := by
  unfold ShardRowsMatch
  infer_instance

def SourceRowsMatch : Prop :=
  ShardRowsMatch sourceRows0 rowDefinitions0 ∧
  ShardRowsMatch sourceRows1 rowDefinitions1 ∧
  ShardRowsMatch sourceRows2 rowDefinitions2

instance : Decidable SourceRowsMatch := by
  unfold SourceRowsMatch
  infer_instance

def OwnersValid : Prop :=
  ∀ owner ∈ owners, owner.Valid coefficientCount

instance : Decidable OwnersValid := by
  unfold OwnersValid
  infer_instance

def OwnersUseLadderPrefix : Prop :=
  ∀ owner ∈ owners,
    owner.powerColumns = ladderPowerColumns.take coefficientCount

instance : Decidable OwnersUseLadderPrefix := by
  unfold OwnersUseLadderPrefix
  infer_instance

def StructureValid : Prop :=
  generatedStagePath = expectedStagePath ∧
  generatedStageRowEnd - generatedStageRowStart = 1620 ∧
  generatedStageAllocatedEnd - generatedStageAllocatedStart = 1620 ∧
  generatedConsumerIdentityIndices = [29, 30] ∧
  owners.length = evaluationCount ∧
  owners.map (·.stagePath) =
    List.replicate evaluationCount expectedStagePath ∧
  owners.map (·.pairIndex) = List.range evaluationCount ∧
  owners.map (·.traceIndex) = List.range evaluationCount ∧
  (owners.getD 0 default).rowStart = generatedStageRowStart ∧
  (owners.getD (evaluationCount - 1) default).rowEnd =
    generatedStageRowEnd ∧
  (owners.getD 0 default).allocatedStart =
    generatedStageAllocatedStart ∧
  (owners.getD (evaluationCount - 1) default).allocatedEnd =
    generatedStageAllocatedEnd ∧
  PiRlcRhoEvaluationOwner.OrderedContiguous owners ∧
  OwnersValid ∧
  OwnersUseLadderPrefix ∧
  sourceRows0.length = 540 ∧
  sourceRows1.length = 540 ∧
  sourceRows2.length = 540 ∧
  sourceRows.length = 1620 ∧
  ownedRowDefinitions.length = 1620

instance : Decidable StructureValid := by
  unfold StructureValid
  infer_instance

def DataValid : Prop := StructureValid ∧ SourceRowsMatch

instance : Decidable DataValid := by
  unfold DataValid
  infer_instance

theorem structure_check : StructureValid := by
  set_option maxRecDepth 100000 in
  set_option maxHeartbeats 1000000 in
    decide

theorem source_rows_match : SourceRowsMatch := by
  constructor
  · set_option maxRecDepth 100000 in
    set_option maxHeartbeats 1000000 in
      decide
  constructor
  · set_option maxRecDepth 100000 in
    set_option maxHeartbeats 1000000 in
      decide
  · set_option maxRecDepth 100000 in
    set_option maxHeartbeats 1000000 in
      decide

theorem data_check : DataValid := ⟨structure_check, source_rows_match⟩

theorem owner_count : owners.length = 15 := by
  decide

theorem owners_valid : OwnersValid := by
  set_option maxRecDepth 100000 in
    decide

theorem owner_valid {owner : PiRlcRhoEvaluationOwner}
    (member : owner ∈ owners) : owner.Valid coefficientCount := by
  exact owners_valid owner member

theorem owners_use_ladder_prefix : OwnersUseLadderPrefix := by
  set_option maxRecDepth 100000 in
    decide

theorem owner_power_prefix {owner : PiRlcRhoEvaluationOwner}
    (member : owner ∈ owners) :
    owner.powerColumns = ladderPowerColumns.take coefficientCount := by
  exact owners_use_ladder_prefix owner member

theorem owned_row_count : ownedRowDefinitions.length = 1620 := by
  set_option maxRecDepth 100000 in
    decide

theorem source_row_count : sourceRows.length = 1620 := by
  set_option maxRecDepth 100000 in
    decide

end Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcProjection.RhoEvaluations
