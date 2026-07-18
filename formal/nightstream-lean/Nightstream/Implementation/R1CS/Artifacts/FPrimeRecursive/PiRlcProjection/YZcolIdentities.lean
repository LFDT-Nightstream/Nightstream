import Nightstream.Implementation.R1CS.Artifacts.FPrimeRecursive.PiRlcProjection.BetaLadder
import Nightstream.Implementation.R1CS.Artifacts.FPrimeRecursive.PiRlcProjection.RhoEvaluations
import Nightstream.Implementation.R1CS.Artifacts.FPrimeRecursive.PiRlcProjection.Generated.YZcolIdentityLimb0Inputs0Data
import Nightstream.Implementation.R1CS.Artifacts.FPrimeRecursive.PiRlcProjection.Generated.YZcolIdentityLimb0Inputs1Data
import Nightstream.Implementation.R1CS.Artifacts.FPrimeRecursive.PiRlcProjection.Generated.YZcolIdentityLimb0Inputs2Data
import Nightstream.Implementation.R1CS.Artifacts.FPrimeRecursive.PiRlcProjection.Generated.YZcolIdentityLimb0TailData
import Nightstream.Implementation.R1CS.Artifacts.FPrimeRecursive.PiRlcProjection.Generated.YZcolIdentityLimb1Inputs0Data
import Nightstream.Implementation.R1CS.Artifacts.FPrimeRecursive.PiRlcProjection.Generated.YZcolIdentityLimb1Inputs1Data
import Nightstream.Implementation.R1CS.Artifacts.FPrimeRecursive.PiRlcProjection.Generated.YZcolIdentityLimb1Inputs2Data
import Nightstream.Implementation.R1CS.Artifacts.FPrimeRecursive.PiRlcProjection.Generated.YZcolIdentityLimb1TailData

/-!
Stable artifact facade for both complete active PiRLC `y_zcol` identities.

Owns: two checked 1,916-row identity schedules, 3,616 newly exported local
source rows, exact linkage to separately owned beta/rho/output traces, and
index-preserving reconstruction of every new definition and final check.

Does not own: satisfaction, transcript or semantic column authority, shared
beta/rho rows, the existing 216 output-evaluation rows, padding, encoded
lowering, bad-root probability, whole-matrix embedding, or row removal.

Emits constraints: no.

Assurance tier: artifact-checked local ownership after the focused Rust drift
test. The large closed-list distinctness checks use `native_decide`, so their
theorems inherit the native compiler trust boundary; focused trust-surface
guards record that dependency exactly. This is not whole-verifier Rust
conformance.

| Protocol → phase → leaf | Rows per limb | Artifact owner |
|---|---:|---|
| `identities.y_zcol.evaluations.inputs` | 1,620 | three 540-row shards |
| `identities.y_zcol.k_products.rho_times_input` | 75 | tail shard |
| `identities.y_zcol.evaluations.output` | 108 | existing output owner |
| `identities.y_zcol.evaluations.quotient` | 106 | tail shard |
| `identities.y_zcol.k_products.quotient_times_phi` | 5 | tail shard |
| `identities.y_zcol.final_limb_checks` | 2 | tail shard |
-/

namespace Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcProjection.YZcolIdentities

open Nightstream.Implementation.R1CS

def pairCount : Nat := 15
def coefficientCount : Nat := 54

def ladderOwner : PiRlcProjectionBetaLadderOwner :=
  FPrimeRecursivePiRlcProjection.BetaLadder.owner

def rhoOwners : List PiRlcRhoEvaluationOwner :=
  FPrimeRecursivePiRlcProjection.RhoEvaluations.owners

def outputLimb0Owner : YZcolOutputEvaluationOwner :=
  FPrimeRecursiveYZcolProjection.limb0Owner

def outputLimb1Owner : YZcolOutputEvaluationOwner :=
  FPrimeRecursiveYZcolProjection.limb1Owner

def limb0Owner : PiRlcYZcolIdentityOwner :=
  FPrimeRecursivePiRlcProjectionYZcolIdentitiesData.Limb0.owner
def limb1Owner : PiRlcYZcolIdentityOwner :=
  FPrimeRecursivePiRlcProjectionYZcolIdentitiesData.Limb1.owner

def limb0Trace : ProjectionProgram.ProjectionTrace :=
  limb0Owner.trace ladderOwner.ladderTrace rhoOwners outputLimb0Owner

def limb1Trace : ProjectionProgram.ProjectionTrace :=
  limb1Owner.trace ladderOwner.ladderTrace rhoOwners outputLimb1Owner

def traces : List ProjectionProgram.ProjectionTrace :=
  [limb0Trace, limb1Trace]

def limb0InputSourceRows0 : List (Nat × Row) :=
  FPrimeRecursivePiRlcProjectionYZcolIdentitiesData.Limb0.inputSourceRows0
def limb0InputSourceRows1 : List (Nat × Row) :=
  FPrimeRecursivePiRlcProjectionYZcolIdentitiesData.Limb0.inputSourceRows1
def limb0InputSourceRows2 : List (Nat × Row) :=
  FPrimeRecursivePiRlcProjectionYZcolIdentitiesData.Limb0.inputSourceRows2
def limb1InputSourceRows0 : List (Nat × Row) :=
  FPrimeRecursivePiRlcProjectionYZcolIdentitiesData.Limb1.inputSourceRows0
def limb1InputSourceRows1 : List (Nat × Row) :=
  FPrimeRecursivePiRlcProjectionYZcolIdentitiesData.Limb1.inputSourceRows1
def limb1InputSourceRows2 : List (Nat × Row) :=
  FPrimeRecursivePiRlcProjectionYZcolIdentitiesData.Limb1.inputSourceRows2

def limb0TailDefinitionSourceRows : List (Nat × Row) :=
  FPrimeRecursivePiRlcProjectionYZcolIdentitiesData.Limb0.tailDefinitionSourceRows
def limb1TailDefinitionSourceRows : List (Nat × Row) :=
  FPrimeRecursivePiRlcProjectionYZcolIdentitiesData.Limb1.tailDefinitionSourceRows
def limb0CheckSourceRows : List (Nat × Row) :=
  FPrimeRecursivePiRlcProjectionYZcolIdentitiesData.Limb0.checkSourceRows
def limb1CheckSourceRows : List (Nat × Row) :=
  FPrimeRecursivePiRlcProjectionYZcolIdentitiesData.Limb1.checkSourceRows

def inputRowDefinitionsFor (owner : PiRlcYZcolIdentityOwner)
    (start count : Nat) : List (Nat × Program.Definition) :=
  ((owner.pairs.drop start).take count).flatMap fun pair =>
    pair.inputRowDefinitions ladderOwner.powerColumns

def limb0InputRowDefinitions0 : List (Nat × Program.Definition) :=
  inputRowDefinitionsFor limb0Owner 0 5
def limb0InputRowDefinitions1 : List (Nat × Program.Definition) :=
  inputRowDefinitionsFor limb0Owner 5 5
def limb0InputRowDefinitions2 : List (Nat × Program.Definition) :=
  inputRowDefinitionsFor limb0Owner 10 5
def limb1InputRowDefinitions0 : List (Nat × Program.Definition) :=
  inputRowDefinitionsFor limb1Owner 0 5
def limb1InputRowDefinitions1 : List (Nat × Program.Definition) :=
  inputRowDefinitionsFor limb1Owner 5 5
def limb1InputRowDefinitions2 : List (Nat × Program.Definition) :=
  inputRowDefinitionsFor limb1Owner 10 5

def limb0TailDefinitionRowDefinitions : List (Nat × Program.Definition) :=
  limb0Owner.tailDefinitionRowDefinitions ladderOwner.powerColumns rhoOwners
def limb1TailDefinitionRowDefinitions : List (Nat × Program.Definition) :=
  limb1Owner.tailDefinitionRowDefinitions ladderOwner.powerColumns rhoOwners

def limb0CheckRows : List (Nat × Row) :=
  limb0Owner.checkRows limb0Trace
def limb1CheckRows : List (Nat × Row) :=
  limb1Owner.checkRows limb1Trace

def limb0NewLocalSourceRows : List (Nat × Row) :=
  limb0InputSourceRows0 ++ limb0InputSourceRows1 ++
    limb0InputSourceRows2 ++ limb0TailDefinitionSourceRows ++
    limb0CheckSourceRows

def limb1NewLocalSourceRows : List (Nat × Row) :=
  limb1InputSourceRows0 ++ limb1InputSourceRows1 ++
    limb1InputSourceRows2 ++ limb1TailDefinitionSourceRows ++
    limb1CheckSourceRows

def newLocalSourceRows : List (Nat × Row) :=
  limb0NewLocalSourceRows ++ limb1NewLocalSourceRows

def limb0NewLocalDefinitionRowDefinitions :
    List (Nat × Program.Definition) :=
  limb0InputRowDefinitions0 ++ limb0InputRowDefinitions1 ++
    limb0InputRowDefinitions2 ++ limb0TailDefinitionRowDefinitions

def limb1NewLocalDefinitionRowDefinitions :
    List (Nat × Program.Definition) :=
  limb1InputRowDefinitions0 ++ limb1InputRowDefinitions1 ++
    limb1InputRowDefinitions2 ++ limb1TailDefinitionRowDefinitions

def RowsMatchDefinitions (rows : List (Nat × Row))
    (definitions : List (Nat × Program.Definition)) : Prop :=
  ActiveIndexedRows.indexedRowsMatch rows definitions = true

instance (rows : List (Nat × Row))
    (definitions : List (Nat × Program.Definition)) :
    Decidable (RowsMatchDefinitions rows definitions) := by
  unfold RowsMatchDefinitions
  infer_instance

def Limb0InputRowsMatch : Prop :=
  RowsMatchDefinitions limb0InputSourceRows0 limb0InputRowDefinitions0 ∧
  RowsMatchDefinitions limb0InputSourceRows1 limb0InputRowDefinitions1 ∧
  RowsMatchDefinitions limb0InputSourceRows2 limb0InputRowDefinitions2

def Limb1InputRowsMatch : Prop :=
  RowsMatchDefinitions limb1InputSourceRows0 limb1InputRowDefinitions0 ∧
  RowsMatchDefinitions limb1InputSourceRows1 limb1InputRowDefinitions1 ∧
  RowsMatchDefinitions limb1InputSourceRows2 limb1InputRowDefinitions2

def TailRowsMatch : Prop :=
  RowsMatchDefinitions limb0TailDefinitionSourceRows
      limb0TailDefinitionRowDefinitions ∧
    RowsMatchDefinitions limb1TailDefinitionSourceRows
      limb1TailDefinitionRowDefinitions

def CheckRowsMatch : Prop :=
  ActiveIndexedRows.indexedRowsMatchRows limb0CheckSourceRows
      limb0CheckRows = true ∧
    ActiveIndexedRows.indexedRowsMatchRows limb1CheckSourceRows
      limb1CheckRows = true

instance : Decidable Limb0InputRowsMatch := by
  unfold Limb0InputRowsMatch
  infer_instance
instance : Decidable Limb1InputRowsMatch := by
  unfold Limb1InputRowsMatch
  infer_instance
instance : Decidable TailRowsMatch := by
  unfold TailRowsMatch
  infer_instance
instance : Decidable CheckRowsMatch := by
  unfold CheckRowsMatch
  infer_instance

def OutputOwnerLinked (owner : PiRlcYZcolIdentityOwner)
    (output : YZcolOutputEvaluationOwner) : Prop :=
  owner.identityIndex = output.identityIndex ∧
  owner.limb = output.limb ∧
  owner.identityRowStart = output.identityRowStart ∧
  owner.identityRowEnd = output.identityRowEnd ∧
  owner.outputStagePath = output.stagePath ∧
  owner.outputColumns = output.parentCoefficientColumns ∧
  owner.outputEvaluationRowStart = output.evaluationRowStart ∧
  owner.outputEvaluationRowEnd = output.evaluationRowEnd ∧
  owner.outputEvaluationAllocatedStart = output.evaluationAllocatedStart ∧
  owner.outputEvaluationAllocatedEnd = output.evaluationAllocatedEnd ∧
  owner.outputEvaluationOutput = output.evaluationOutputColumns

instance (owner : PiRlcYZcolIdentityOwner)
    (output : YZcolOutputEvaluationOwner) :
    Decidable (OutputOwnerLinked owner output) := by
  unfold OutputOwnerLinked
  infer_instance

def GeometryValid : Prop :=
  limb0Owner.Valid pairCount coefficientCount ∧
  limb1Owner.Valid pairCount coefficientCount ∧
  limb0Owner.identityIndex = 29 ∧ limb1Owner.identityIndex = 30 ∧
  limb0Owner.identityRowEnd = limb1Owner.identityRowStart ∧
  limb0Owner.identityAllocatedEnd = limb1Owner.identityAllocatedStart ∧
  limb0Owner.inputStagePath =
    "nifs.pi_rlc.verify.identities.y_zcol.evaluations.inputs.limb0" ∧
  limb1Owner.inputStagePath =
    "nifs.pi_rlc.verify.identities.y_zcol.evaluations.inputs.limb1" ∧
  limb0Owner.productStagePath =
    "nifs.pi_rlc.verify.identities.y_zcol.k_products.rho_times_input.limb0" ∧
  limb1Owner.productStagePath =
    "nifs.pi_rlc.verify.identities.y_zcol.k_products.rho_times_input.limb1" ∧
  limb0Owner.quotientStagePath =
    "nifs.pi_rlc.verify.identities.y_zcol.evaluations.quotient.limb0" ∧
  limb1Owner.quotientStagePath =
    "nifs.pi_rlc.verify.identities.y_zcol.evaluations.quotient.limb1" ∧
  limb0Owner.quotientPhiStagePath =
    "nifs.pi_rlc.verify.identities.y_zcol.k_products.quotient_times_phi.limb0" ∧
  limb1Owner.quotientPhiStagePath =
    "nifs.pi_rlc.verify.identities.y_zcol.k_products.quotient_times_phi.limb1" ∧
  limb0Owner.finalChecksStagePath =
    "nifs.pi_rlc.verify.identities.y_zcol.final_limb_checks.limb0" ∧
  limb1Owner.finalChecksStagePath =
    "nifs.pi_rlc.verify.identities.y_zcol.final_limb_checks.limb1" ∧
  OutputOwnerLinked limb0Owner outputLimb0Owner ∧
  OutputOwnerLinked limb1Owner outputLimb1Owner ∧
  FPrimeRecursivePiRlcProjection.RhoEvaluations.generatedConsumerIdentityIndices =
    [limb0Owner.identityIndex, limb1Owner.identityIndex] ∧
  ladderOwner.rowEnd ≤
    FPrimeRecursivePiRlcProjection.RhoEvaluations.generatedStageRowStart ∧
  FPrimeRecursivePiRlcProjection.RhoEvaluations.generatedStageRowEnd ≤
    limb0Owner.identityRowStart

def TraceLayoutsValid : Prop :=
  limb0Trace.LayoutValid ∧ limb1Trace.LayoutValid

def RowCountsValid : Prop :=
  limb0InputSourceRows0.length = 540 ∧
  limb0InputSourceRows1.length = 540 ∧
  limb0InputSourceRows2.length = 540 ∧
  limb1InputSourceRows0.length = 540 ∧
  limb1InputSourceRows1.length = 540 ∧
  limb1InputSourceRows2.length = 540 ∧
  limb0TailDefinitionSourceRows.length = 186 ∧
  limb1TailDefinitionSourceRows.length = 186 ∧
  limb0CheckSourceRows.length = 2 ∧ limb1CheckSourceRows.length = 2 ∧
  limb0NewLocalSourceRows.length = 1808 ∧
  limb1NewLocalSourceRows.length = 1808 ∧
  newLocalSourceRows.length = 3616

def LocalRowsDistinct : Prop :=
  (newLocalSourceRows.map Prod.fst).Nodup

def LocalOutputRowsDistinct : Prop :=
  ((newLocalSourceRows ++ FPrimeRecursiveYZcolProjectionData.sourceRows).map
    Prod.fst).Nodup

def StructureValid : Prop :=
  GeometryValid ∧ TraceLayoutsValid ∧ RowCountsValid ∧
    LocalRowsDistinct ∧ LocalOutputRowsDistinct

instance : Decidable GeometryValid := by
  unfold GeometryValid
  infer_instance
instance : Decidable TraceLayoutsValid := by
  unfold TraceLayoutsValid
  infer_instance
instance : Decidable RowCountsValid := by
  unfold RowCountsValid
  infer_instance
instance : Decidable LocalRowsDistinct := by
  unfold LocalRowsDistinct
  infer_instance
instance : Decidable LocalOutputRowsDistinct := by
  unfold LocalOutputRowsDistinct
  infer_instance

instance : Decidable StructureValid := by
  unfold StructureValid
  infer_instance

def DataValid : Prop :=
  StructureValid ∧ Limb0InputRowsMatch ∧ Limb1InputRowsMatch ∧
    TailRowsMatch ∧ CheckRowsMatch

instance : Decidable DataValid := by
  unfold DataValid
  infer_instance

theorem geometry_check : GeometryValid := by
  set_option maxRecDepth 100000 in
  set_option maxHeartbeats 2000000 in
    decide

theorem trace_layouts_check : TraceLayoutsValid := by
  set_option maxRecDepth 100000 in
  set_option maxHeartbeats 2000000 in
    decide

theorem row_counts_check : RowCountsValid := by
  set_option maxRecDepth 100000 in decide

theorem local_rows_distinct : LocalRowsDistinct := by
  native_decide

theorem local_output_rows_distinct : LocalOutputRowsDistinct := by
  native_decide

theorem structure_check : StructureValid :=
  ⟨geometry_check, trace_layouts_check, row_counts_check,
    local_rows_distinct, local_output_rows_distinct⟩

theorem limb0_input_rows_match : Limb0InputRowsMatch := by
  constructor
  · set_option maxRecDepth 100000 in decide
  · constructor <;> set_option maxRecDepth 100000 in decide

theorem limb1_input_rows_match : Limb1InputRowsMatch := by
  constructor
  · set_option maxRecDepth 100000 in decide
  · constructor <;> set_option maxRecDepth 100000 in decide

theorem tail_rows_match : TailRowsMatch := by
  constructor <;> set_option maxRecDepth 100000 in decide

theorem check_rows_match : CheckRowsMatch := by
  constructor <;> set_option maxRecDepth 100000 in decide

theorem data_check : DataValid :=
  ⟨structure_check, limb0_input_rows_match, limb1_input_rows_match,
    tail_rows_match, check_rows_match⟩

theorem limb0_layout : limb0Trace.LayoutValid := by
  set_option maxRecDepth 100000 in decide

theorem limb1_layout : limb1Trace.LayoutValid := by
  set_option maxRecDepth 100000 in decide

theorem traces_count : traces.length = 2 := by decide

theorem shared_rows_precede_identity_rows :
    ladderOwner.rowEnd ≤
        FPrimeRecursivePiRlcProjection.RhoEvaluations.generatedStageRowStart ∧
      FPrimeRecursivePiRlcProjection.RhoEvaluations.generatedStageRowEnd ≤
        limb0Owner.identityRowStart := by
  set_option maxRecDepth 100000 in decide

theorem new_local_row_count : newLocalSourceRows.length = 3616 := by
  set_option maxRecDepth 100000 in decide

end Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcProjection.YZcolIdentities
