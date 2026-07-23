import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Execution

/-!
Artifact census for the fixed production combined-NC execution export.

Assurance tier: artifact-checked data for the single generated profile.

Owns: exact generated header pins; bounded shard cardinalities and ordered
coverage; the child-major 14-by-64 old-block lane schedule; the 270 public
write schedule; the 25-round message shape and claim chain; and the 1,289
generated K-binding slot schedule.

Does not own: Rust dataflow, `WitnessMat` authority, field decoding, R1CS row
satisfaction, transcript soundness, commitment binding, protocol acceptance,
costs, or permission to remove rows.  In particular, no theorem here is a
`Refines` theorem.

Emits constraints: none; proof-only bounded artifact checks.

| Stable stage path | Mathematical obligation | Authority class |
|---|---|---|
| `f_prime.pi_ccs_nc.delayed.combined.execution.header` | pin the fixed execution profile and boundary dimensions | checked artifact |
| `f_prime.pi_ccs_nc.delayed.combined.execution.public` | check two ordered 135-record public-write shards | checked artifact |
| `f_prime.pi_ccs_nc.delayed.combined.execution.raw_old_block` | check four ordered 224-record child/lane shards | checked artifact |
| `f_prime.pi_ccs_nc.delayed.combined.execution.rounds` | check 25 five-coefficient messages and their recorded claim chain | checked artifact |
| `f_prime.pi_ccs_nc.delayed.combined.execution.bindings` | check six bounded K-binding shards against the complete slot schedule | checked artifact |
-/

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.ExecutionAudit

open Generated.Execution

private def rawKZero : RawK := { c0 := 0, c1 := 0 }

private def recordsMatchFrom {α : Type} (offset : Nat)
    (predicate : Nat → α → Bool) (records : List α) : Bool :=
  ((List.range records.length).zip records).all fun pair =>
    predicate (offset + pair.1) pair.2

/-! ## Fixed header -/

/-- Exact fixed-profile facts exposed by the generated execution header. -/
def HeaderExact : Prop :=
  let header := Generated.Execution.header
  header.schemaVersion = 1 ∧
  header.branch = 2 ∧
  header.proofVariant = 1 ∧
  header.outputSources = 15 ∧
  header.outputMatrices = 13 ∧
  header.outputActiveLanes = 54 ∧
  header.freshCount = 1 ∧
  header.runningCount = 14 ∧
  header.logicalColumns = 11437038 ∧
  header.packedRows = 54 ∧
  header.packedColumns = 211797 ∧
  header.packedRows * header.packedColumns = header.logicalColumns ∧
  header.sourceRows = 11308137 ∧
  header.sourceColumns = 10997363 ∧
  header.finalRows = 14944219 ∧
  header.finalColumns = 11437038 ∧
  header.publicWriteCount = 270 ∧
  header.selectorColumns = [270, 271, 272] ∧
  header.selectorValues = [0, 0, 1] ∧
  header.oldBlock.length = 19 ∧
  header.parentYZcol.length = 54 ∧
  header.radix = { c0 := 2, c1 := 0 } ∧
  header.betaBlock.length = 19 ∧
  header.betaLane.length = 6 ∧
  header.blockPoint.length = 19 ∧
  header.lanePoint.length = 6 ∧
  header.terminalFinal = header.terminalRhs ∧
  header.fieldDecodingTag = 0 ∧
  header.rawAuthorityTag = 0

/-- `native_decide` input: one proof-free `RawExecutionHeader` containing
exactly 129 inline `RawK` values: 19 old-block, 54 parent, three scalar
challenges, 19+6 transcript betas, 19+6 transcript points, and three terminal
values. No generated row or proof-bearing structure is traversed. -/
theorem header_exact : HeaderExact := by
  unfold HeaderExact
  native_decide

/-! ## Public writes -/

private def someBuilderBelow (bound : Nat) : Option Nat → Bool
  | some column => decide (0 < column ∧ column < bound)
  | none => false

private def publicWriteMatches (index : Nat) (record : RawPublicWrite) : Bool :=
  record.schemaVersion == 1 &&
  record.logicalColumn == index &&
  record.packedRow == index % 54 &&
  record.packedColumn == index / 54 &&
  record.normalizedColumn == index &&
  if index = 0 then
    record.sourceKind == 0 &&
    record.builderColumn == some 0 &&
    record.value == 1
  else if index ≤ 256 then
    record.sourceKind == 1 &&
    someBuilderBelow Generated.Execution.header.sourceColumns record.builderColumn
  else
    record.sourceKind == 2 &&
    record.builderColumn == none &&
    record.value == 0

private def optionNatPairLt (pair : Option Nat × Option Nat) : Bool :=
  match pair with
  | (some left, some right) => decide (left < right)
  | _ => false

private def builderColumnsStrict (records : List RawPublicWrite) : Bool :=
  let columns := records.map (fun record => record.builderColumn)
  (columns.zip columns.tail).all optionNatPairLt

/-- `native_decide` input: exactly 135 proof-free `RawPublicWrite` records
(logical indices 0 through 134), each containing only scalar fields and one
`Option Nat`. No other shard is traversed. -/
theorem public_write_chunk0_exact :
    PublicWrites.Chunk0.values.length = 135 ∧
    recordsMatchFrom 0 publicWriteMatches PublicWrites.Chunk0.values = true ∧
    builderColumnsStrict PublicWrites.Chunk0.values = true := by
  native_decide

/-- `native_decide` input: exactly 135 proof-free `RawPublicWrite` records
(logical indices 135 through 269). The monotonic subcheck reads only its first
122 records, corresponding to builder-backed indices 135 through 256. -/
theorem public_write_chunk1_exact :
    PublicWrites.Chunk1.values.length = 135 ∧
    recordsMatchFrom 135 publicWriteMatches PublicWrites.Chunk1.values = true ∧
    builderColumnsStrict (PublicWrites.Chunk1.values.take 122) = true := by
  native_decide

theorem public_writes_exact_chunk_coverage :
    Generated.Execution.publicWrites =
      PublicWrites.Chunk0.values ++
      PublicWrites.Chunk1.values := by
  rfl

theorem public_writes_count_exact :
    Generated.Execution.publicWrites.length = 270 := by
  rw [public_writes_exact_chunk_coverage, List.length_append,
    public_write_chunk0_exact.1, public_write_chunk1_exact.1]

/-! ## Raw old-block lanes -/

private def rawOldBlockLaneMatches (index : Nat)
    (record : RawOldBlockLane) : Bool :=
  let lane := index % 64
  record.schemaVersion == 1 &&
  record.child == index / 64 &&
  record.lane == lane &&
  record.padding == decide (54 ≤ lane) &&
  (!record.padding || record.value == rawKZero)

/-- `native_decide` input: exactly 224 proof-free `RawOldBlockLane` records,
each with one inline two-`Nat` `RawK` payload. This is global lane indices
0 through 223; no other lane shard is traversed. -/
theorem raw_old_block_chunk0_exact :
    RawOldBlockLanes.Chunk0.values.length = 224 ∧
    recordsMatchFrom 0 rawOldBlockLaneMatches
      RawOldBlockLanes.Chunk0.values = true := by
  native_decide

/-- `native_decide` input: exactly 224 proof-free `RawOldBlockLane` records,
each with one inline two-`Nat` `RawK` payload. This is global lane indices
224 through 447; no other lane shard is traversed. -/
theorem raw_old_block_chunk1_exact :
    RawOldBlockLanes.Chunk1.values.length = 224 ∧
    recordsMatchFrom 224 rawOldBlockLaneMatches
      RawOldBlockLanes.Chunk1.values = true := by
  native_decide

/-- `native_decide` input: exactly 224 proof-free `RawOldBlockLane` records,
each with one inline two-`Nat` `RawK` payload. This is global lane indices
448 through 671; no other lane shard is traversed. -/
theorem raw_old_block_chunk2_exact :
    RawOldBlockLanes.Chunk2.values.length = 224 ∧
    recordsMatchFrom 448 rawOldBlockLaneMatches
      RawOldBlockLanes.Chunk2.values = true := by
  native_decide

/-- `native_decide` input: exactly 224 proof-free `RawOldBlockLane` records,
each with one inline two-`Nat` `RawK` payload. This is global lane indices
672 through 895; no other lane shard is traversed. -/
theorem raw_old_block_chunk3_exact :
    RawOldBlockLanes.Chunk3.values.length = 224 ∧
    recordsMatchFrom 672 rawOldBlockLaneMatches
      RawOldBlockLanes.Chunk3.values = true := by
  native_decide

theorem raw_old_block_lanes_exact_chunk_coverage :
    Generated.Execution.rawOldBlockLanes =
      RawOldBlockLanes.Chunk0.values ++
      RawOldBlockLanes.Chunk1.values ++
      RawOldBlockLanes.Chunk2.values ++
      RawOldBlockLanes.Chunk3.values := by
  rfl

theorem raw_old_block_lane_count_exact :
    Generated.Execution.rawOldBlockLanes.length = 14 * 64 := by
  rw [raw_old_block_lanes_exact_chunk_coverage,
    List.length_append, List.length_append, List.length_append,
    raw_old_block_chunk0_exact.1, raw_old_block_chunk1_exact.1,
    raw_old_block_chunk2_exact.1, raw_old_block_chunk3_exact.1]

/-- The four bounded checks cover every global index `child * 64 + lane`.
They pin exactly 54 non-padding lanes followed by ten zero padding lanes for
each of fourteen children. This is an artifact statement, not a dataflow
statement about the source of the active values. -/
theorem raw_old_block_schedule_exact :
    recordsMatchFrom 0 rawOldBlockLaneMatches
        RawOldBlockLanes.Chunk0.values = true ∧
      recordsMatchFrom 224 rawOldBlockLaneMatches
        RawOldBlockLanes.Chunk1.values = true ∧
      recordsMatchFrom 448 rawOldBlockLaneMatches
        RawOldBlockLanes.Chunk2.values = true ∧
      recordsMatchFrom 672 rawOldBlockLaneMatches
        RawOldBlockLanes.Chunk3.values = true :=
  ⟨raw_old_block_chunk0_exact.2,
    raw_old_block_chunk1_exact.2,
    raw_old_block_chunk2_exact.2,
    raw_old_block_chunk3_exact.2⟩

/-! ## SumCheck rounds -/

private def adjacentClaimsMatch (rounds : List RawCombinedNcRound) : Bool :=
  (rounds.zip rounds.tail).all fun pair =>
    pair.1.claimOut == pair.2.claimIn

/-- Exact fixed-profile shape and ordering of the exported round family. -/
def RoundsExact : Prop :=
  let rounds := Generated.Execution.rounds
  rounds.length = 25 ∧
  rounds.map (fun round => round.index) = List.range 25 ∧
  rounds.all (fun round => decide (round.coefficients.length = 5)) = true ∧
  rounds.map (fun round => round.challenge) =
    Generated.Execution.header.blockPoint ++ Generated.Execution.header.lanePoint ∧
  rounds.head?.map (fun round => round.claimIn) =
    some Generated.Execution.header.terminalInitial ∧
  adjacentClaimsMatch rounds = true ∧
  rounds.reverse.head?.map (fun round => round.claimOut) =
    some Generated.Execution.header.terminalFinal ∧
  Generated.Execution.header.terminalFinal = Generated.Execution.header.terminalRhs

/-- `native_decide` input: exactly 25 proof-free `RawCombinedNcRound`
records. Each contains five coefficient `RawK`s plus one challenge and two
claims, for exactly 200 inline `RawK` payloads. No row artifact is traversed. -/
theorem rounds_exact : RoundsExact := by
  unfold RoundsExact
  native_decide

/-! ## Generated K bindings -/

private structure ExpectedSlot where
  kind : Nat
  index0 : Nat
  index1 : Nat

private def expectedBindingSlot (index : Nat) : ExpectedSlot :=
  if index = 0 then
    ⟨0, 0, 0⟩
  else if index < 7 then
    ⟨1, index - 1, 0⟩
  else if index < 26 then
    ⟨2, index - 7, 0⟩
  else if index = 26 then
    ⟨3, 0, 0⟩
  else if index = 27 then
    ⟨4, 0, 0⟩
  else if index < 47 then
    ⟨5, index - 28, 0⟩
  else if index < 101 then
    ⟨6, index - 47, 0⟩
  else if index < 1061 then
    let outputLane := index - 101
    ⟨7, outputLane / 64, outputLane % 64⟩
  else if index < 1080 then
    ⟨8, index - 1061, 0⟩
  else if index < 1086 then
    ⟨9, index - 1080, 0⟩
  else if index = 1086 then
    ⟨10, 0, 0⟩
  else if index = 1087 then
    ⟨11, 0, 0⟩
  else if index = 1088 then
    ⟨12, 0, 0⟩
  else
    let roundSlot := index - 1089
    let round := roundSlot / 8
    let component := roundSlot % 8
    if component < 5 then
      ⟨13, round, component⟩
    else if component = 5 then
      ⟨14, round, 0⟩
    else if component = 6 then
      ⟨15, round, 0⟩
    else
      ⟨16, round, 0⟩

private def generatedKBindingMatches (index : Nat)
    (record : RawGeneratedKBinding) : Bool :=
  let expected := expectedBindingSlot index
  record.schemaVersion == 1 &&
  record.slotKind == expected.kind &&
  record.slotIndex0 == expected.index0 &&
  record.slotIndex1 == expected.index1 &&
  decide (record.builderC0 ≠ record.builderC1) &&
  decide (record.normalizedC0 ≠ record.normalizedC1) &&
  decide (record.builderC0 < Generated.Execution.header.sourceColumns) &&
  decide (record.builderC1 < Generated.Execution.header.sourceColumns) &&
  decide (record.normalizedC0 < Generated.Execution.header.sourceColumns) &&
  decide (record.normalizedC1 < Generated.Execution.header.sourceColumns)

/-- `native_decide` input: exactly 224 proof-free
`RawGeneratedKBinding` records (global slots 0 through 223), each with scalar
columns and one inline two-`Nat` `RawK` payload. -/
theorem generated_k_binding_chunk0_exact :
    GeneratedKBindings.Chunk0.values.length = 224 ∧
    recordsMatchFrom 0 generatedKBindingMatches
      GeneratedKBindings.Chunk0.values = true := by
  native_decide

/-- `native_decide` input: exactly 224 proof-free
`RawGeneratedKBinding` records (global slots 224 through 447), each with
scalar columns and one inline two-`Nat` `RawK` payload. -/
theorem generated_k_binding_chunk1_exact :
    GeneratedKBindings.Chunk1.values.length = 224 ∧
    recordsMatchFrom 224 generatedKBindingMatches
      GeneratedKBindings.Chunk1.values = true := by
  native_decide

/-- `native_decide` input: exactly 224 proof-free
`RawGeneratedKBinding` records (global slots 448 through 671), each with
scalar columns and one inline two-`Nat` `RawK` payload. -/
theorem generated_k_binding_chunk2_exact :
    GeneratedKBindings.Chunk2.values.length = 224 ∧
    recordsMatchFrom 448 generatedKBindingMatches
      GeneratedKBindings.Chunk2.values = true := by
  native_decide

/-- `native_decide` input: exactly 224 proof-free
`RawGeneratedKBinding` records (global slots 672 through 895), each with
scalar columns and one inline two-`Nat` `RawK` payload. -/
theorem generated_k_binding_chunk3_exact :
    GeneratedKBindings.Chunk3.values.length = 224 ∧
    recordsMatchFrom 672 generatedKBindingMatches
      GeneratedKBindings.Chunk3.values = true := by
  native_decide

/-- `native_decide` input: exactly 224 proof-free
`RawGeneratedKBinding` records (global slots 896 through 1119), each with
scalar columns and one inline two-`Nat` `RawK` payload. -/
theorem generated_k_binding_chunk4_exact :
    GeneratedKBindings.Chunk4.values.length = 224 ∧
    recordsMatchFrom 896 generatedKBindingMatches
      GeneratedKBindings.Chunk4.values = true := by
  native_decide

/-- `native_decide` input: exactly 169 proof-free
`RawGeneratedKBinding` records (global slots 1120 through 1288), each with
scalar columns and one inline two-`Nat` `RawK` payload. -/
theorem generated_k_binding_chunk5_exact :
    GeneratedKBindings.Chunk5.values.length = 169 ∧
    recordsMatchFrom 1120 generatedKBindingMatches
      GeneratedKBindings.Chunk5.values = true := by
  native_decide

theorem generated_k_bindings_exact_chunk_coverage :
    Generated.Execution.generatedKBindings =
      GeneratedKBindings.Chunk0.values ++
      GeneratedKBindings.Chunk1.values ++
      GeneratedKBindings.Chunk2.values ++
      GeneratedKBindings.Chunk3.values ++
      GeneratedKBindings.Chunk4.values ++
      GeneratedKBindings.Chunk5.values := by
  rfl

theorem generated_k_binding_count_exact :
    Generated.Execution.generatedKBindings.length = 1289 := by
  rw [generated_k_bindings_exact_chunk_coverage,
    List.length_append, List.length_append, List.length_append,
    List.length_append, List.length_append,
    generated_k_binding_chunk0_exact.1,
    generated_k_binding_chunk1_exact.1,
    generated_k_binding_chunk2_exact.1,
    generated_k_binding_chunk3_exact.1,
    generated_k_binding_chunk4_exact.1,
    generated_k_binding_chunk5_exact.1]

/-- Complete bounded slot schedule, split exactly as emitted. -/
theorem generated_k_binding_schedule_exact :
    recordsMatchFrom 0 generatedKBindingMatches
        GeneratedKBindings.Chunk0.values = true ∧
      recordsMatchFrom 224 generatedKBindingMatches
        GeneratedKBindings.Chunk1.values = true ∧
      recordsMatchFrom 448 generatedKBindingMatches
        GeneratedKBindings.Chunk2.values = true ∧
      recordsMatchFrom 672 generatedKBindingMatches
        GeneratedKBindings.Chunk3.values = true ∧
      recordsMatchFrom 896 generatedKBindingMatches
        GeneratedKBindings.Chunk4.values = true ∧
      recordsMatchFrom 1120 generatedKBindingMatches
        GeneratedKBindings.Chunk5.values = true :=
  ⟨generated_k_binding_chunk0_exact.2,
    generated_k_binding_chunk1_exact.2,
    generated_k_binding_chunk2_exact.2,
    generated_k_binding_chunk3_exact.2,
    generated_k_binding_chunk4_exact.2,
    generated_k_binding_chunk5_exact.2⟩

end Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.ExecutionAudit
