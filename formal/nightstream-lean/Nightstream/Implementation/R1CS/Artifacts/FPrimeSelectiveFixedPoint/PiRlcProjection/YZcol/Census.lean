import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Generated.Metadata
import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Generated.StagePaths

/-!
Computational structural census for the bounded tiny-lifecycle PiRLC
cross-branch source fixture.

Owns: reflected finite checks for the fixture shape, the exact 14-leaf Rust
stage census, shared and per-limb schedules, row ownership and bounds,
definition-output/allocated-column identity, fresh-column order and bounds,
and separately recorded producer vectors.

Does not own: row satisfaction, selective-lowering refinement, serializer
semantics, producer/consumer authority, global producer-column disjointness,
security bounds, final cost, or permission to remove rows.

Emits constraints: no.

| Census branch | Mathematical obligation | Fixed fixture value |
|---|---|---:|
| shared definitions | beta ladder plus 15 rho evaluations | 1,892 |
| low/high definitions | 15 input/product pairs plus tail | 1,914 each |
| low/high checks | final projection identities | 2 each |
| rows | ordered, in-bounds, exactly classified | 5,724 |
| columns | definition output equals advertised fresh column | `1,892 + 2×1,914 = 5,720` |
| source stage leaves | unique stable paths with exact definition/check ownership | 14 |
| producers | 30 separately recorded 54-lane vectors | no global disjointness claim |

All large checks are Boolean programs reflected back into propositions.
Shared, low-limb, and high-limb physical data are checked separately. No
`native_decide`, classical proposition oracle, or monolithic `StructureValid`
decision is used.
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Census

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol

private abbrev artifact : Artifact := Generated.Metadata.artifact
private abbrev lowLimb : LimbOwner := Generated.Metadata.limb0
private abbrev highLimb : LimbOwner := Generated.Metadata.limb1
private abbrev sourceStageLeaves : List SourceStageLeaf :=
  artifact.sourceStageLeaves Generated.StagePaths.paths

/-! ## Generic finite reflection helpers -/

private def rowBlockFitsCheck (rows : RowBlock) (length : Nat) : Bool :=
  decide (rows.start ≤ rows.stop) && decide (rows.count = length)

private theorem rowBlockFitsCheck_eq_true_iff
    (rows : RowBlock) (length : Nat) :
    rowBlockFitsCheck rows length = true ↔ rows.Fits length := by
  simp [rowBlockFitsCheck, RowBlock.Fits, decide_eq_true_eq]

private def evaluationValidCheck
    (owner : EvaluationOwner) (coefficientCount : Nat) : Bool :=
  decide (owner.coefficients.length = coefficientCount) &&
    decide owner.trace.LayoutValid &&
    rowBlockFitsCheck owner.rows owner.trace.definitions.length

private theorem evaluationValidCheck_eq_true_iff
    (owner : EvaluationOwner) (coefficientCount : Nat) :
    evaluationValidCheck owner coefficientCount = true ↔
      owner.Valid coefficientCount := by
  simp [evaluationValidCheck, EvaluationOwner.Valid,
    rowBlockFitsCheck_eq_true_iff, decide_eq_true_eq, and_assoc]

private def kProductValidCheck (owner : KProductOwner) : Bool :=
  decide owner.trace.SumLayoutValid &&
    rowBlockFitsCheck owner.rows owner.trace.definitions.length

private theorem kProductValidCheck_eq_true_iff (owner : KProductOwner) :
    kProductValidCheck owner = true ↔ owner.Valid := by
  simp [kProductValidCheck, KProductOwner.Valid,
    rowBlockFitsCheck_eq_true_iff, decide_eq_true_eq]

private def pairValidCheck
    (owner : PairOwner) (rhoEvaluation : EvaluationOwner)
    (laneCount : Nat) : Bool :=
  evaluationValidCheck owner.inputEvaluation laneCount &&
    kProductValidCheck owner.rhoProduct &&
    decide (owner.rhoProduct.left =
      ProjectionProgram.KTerms.ofColumns rhoEvaluation.output) &&
    decide (owner.rhoProduct.right =
      ProjectionProgram.KTerms.ofColumns owner.inputEvaluation.output)

private theorem pairValidCheck_eq_true_iff
    (owner : PairOwner) (rhoEvaluation : EvaluationOwner)
    (laneCount : Nat) :
    pairValidCheck owner rhoEvaluation laneCount = true ↔
      owner.Valid rhoEvaluation laneCount := by
  simp [pairValidCheck, PairOwner.Valid,
    evaluationValidCheck_eq_true_iff, kProductValidCheck_eq_true_iff,
    decide_eq_true_eq, and_assoc]

private def sharedValidCheck (owner : SharedOwner) (scope : Scope) : Bool :=
  decide (owner.powers.length = scope.powerCount) &&
    decide (owner.ladderProducts.length + 1 = scope.powerCount) &&
    decide (owner.rhoEvaluations.length = scope.sourceCount) &&
    rowBlockFitsCheck owner.betaLadderRows owner.ladderTrace.definitions.length &&
    decide (owner.ladderDefinitionIndices = owner.betaLadderRows.indices) &&
    decide (owner.definitionIndices =
      List.range' owner.betaLadderRows.start owner.indexedDefinitions.length) &&
    decide owner.ladderTrace.LayoutValid &&
    owner.ladderProducts.all kProductValidCheck &&
    owner.rhoEvaluations.all fun evaluation =>
      evaluationValidCheck evaluation scope.laneCount &&
        decide (evaluation.powers = owner.powers.take scope.laneCount)

private theorem sharedValidCheck_eq_true_iff
    (owner : SharedOwner) (scope : Scope) :
    sharedValidCheck owner scope = true ↔ owner.Valid scope := by
  simp [sharedValidCheck, SharedOwner.Valid, List.all_eq_true,
    rowBlockFitsCheck_eq_true_iff, kProductValidCheck_eq_true_iff,
    evaluationValidCheck_eq_true_iff, decide_eq_true_eq, and_assoc]

private def scheduleValidCheck
    (owner : LimbOwner) (shared : SharedOwner) : Bool :=
  match owner.pairs with
  | [] => false
  | first :: _ => decide (owner.ownedIndices shared =
      List.range' first.inputEvaluation.rows.start
        (owner.ownedIndices shared).length)

private theorem scheduleValidCheck_eq_true_iff
    (owner : LimbOwner) (shared : SharedOwner) :
    scheduleValidCheck owner shared = true ↔ owner.ScheduleValid shared := by
  unfold scheduleValidCheck LimbOwner.ScheduleValid
  split <;> simp_all [decide_eq_true_eq]

private def limbValidCheck
    (owner : LimbOwner) (scope : Scope) (shared : SharedOwner) : Bool :=
  decide (owner.limb < 2) &&
    decide (owner.pairs.length = scope.sourceCount) &&
    decide (owner.pairs.map PairOwner.sourceIndex =
      List.range scope.sourceCount) &&
    ((List.zip owner.pairs shared.rhoEvaluations).all fun pair =>
      pairValidCheck pair.1 pair.2 scope.laneCount) &&
    evaluationValidCheck owner.parentEvaluation scope.laneCount &&
    decide (owner.parentEvaluation.powers =
      shared.powers.take scope.laneCount) &&
    evaluationValidCheck owner.quotientEvaluation scope.quotientCount &&
    decide (owner.quotientEvaluation.powers =
      shared.powers.take scope.quotientCount) &&
    kProductValidCheck owner.quotientPhiProduct &&
    decide (owner.quotientPhiProduct.left =
      ProjectionProgram.KTerms.ofColumns owner.quotientEvaluation.output) &&
    decide (owner.quotientPhiProduct.right =
      ProjectionProgram.phiTerms shared.powers) &&
    rowBlockFitsCheck owner.finalRows (owner.trace shared).checks.length &&
    scheduleValidCheck owner shared &&
    decide (owner.maxDegree = scope.maxDegree) &&
    decide (owner.trace shared).LayoutValid

private theorem limbValidCheck_eq_true_iff
    (owner : LimbOwner) (scope : Scope) (shared : SharedOwner) :
    limbValidCheck owner scope shared = true ↔ owner.Valid scope shared := by
  simp [limbValidCheck, LimbOwner.Valid, List.all_eq_true,
    pairValidCheck_eq_true_iff, evaluationValidCheck_eq_true_iff,
    kProductValidCheck_eq_true_iff, rowBlockFitsCheck_eq_true_iff,
    scheduleValidCheck_eq_true_iff, decide_eq_true_eq, and_assoc]

private def producerVectorShapeCheck
    (owner : ProducerVector) (laneCount : Nat) : Bool :=
  decide (owner.entries.length = laneCount) &&
    decide owner.serializerFieldIndices.Nodup &&
    decide owner.sourceColumns.Nodup

private theorem producerVectorShapeCheck_eq_true_iff
    (owner : ProducerVector) (laneCount : Nat) :
    producerVectorShapeCheck owner laneCount = true ↔
      owner.HasShape laneCount := by
  simp [producerVectorShapeCheck, ProducerVector.HasShape,
    decide_eq_true_eq, and_assoc]

private def producerShapeCheck (candidate : Artifact) : Bool :=
  decide (candidate.producerKeys =
      Artifact.expectedProducerKeys candidate.scope) &&
    (candidate.producers.all fun producer =>
      producerVectorShapeCheck producer candidate.scope.laneCount) &&
    (candidate.producerFieldIndices.all fun index =>
      decide (index < candidate.scope.serializerFieldCount)) &&
    candidate.producerSourceColumns.all fun column =>
      decide (column ≠ 0 ∧ column < candidate.scope.sourceArmColumnCount)

private theorem producerShapeCheck_eq_true_iff (candidate : Artifact) :
    producerShapeCheck candidate = true ↔ candidate.ProducerShapeValid := by
  simp [producerShapeCheck, Artifact.ProducerShapeValid, List.all_eq_true,
    producerVectorShapeCheck_eq_true_iff, decide_eq_true_eq, and_assoc]

private def strictlyIncreasingCheck : List Nat -> Bool
  | [] => true
  | [_] => true
  | first :: second :: rest =>
      decide (first < second) && strictlyIncreasingCheck (second :: rest)

private theorem strictlyIncreasingCheck_eq_true_iff (values : List Nat) :
    strictlyIncreasingCheck values = true ↔ StrictlyIncreasing values := by
  induction values with
  | nil => simp [strictlyIncreasingCheck, StrictlyIncreasing]
  | cons head tail inductionHypothesis =>
      cases tail with
      | nil => simp [strictlyIncreasingCheck, StrictlyIncreasing]
      | cons next rest =>
          simp [strictlyIncreasingCheck, StrictlyIncreasing,
            inductionHypothesis, decide_eq_true_eq]

private def sparseTermsBoundedCheck
    (terms : List (Nat × Nat)) (width : Nat) : Bool :=
  terms.all fun term => decide (term.1 < width)

private theorem sparseTermsBoundedCheck_eq_true_iff
    (terms : List (Nat × Nat)) (width : Nat) :
    sparseTermsBoundedCheck terms width = true ↔
      Artifact.SparseTermsBounded terms width := by
  simp [sparseTermsBoundedCheck, Artifact.SparseTermsBounded,
    List.all_eq_true, decide_eq_true_eq]

private def rowBoundedCheck (row : Row) (width : Nat) : Bool :=
  sparseTermsBoundedCheck row.a width &&
    sparseTermsBoundedCheck row.b width &&
    sparseTermsBoundedCheck row.c width

private theorem rowBoundedCheck_eq_true_iff (row : Row) (width : Nat) :
    rowBoundedCheck row width = true ↔ Artifact.RowBounded row width := by
  simp [rowBoundedCheck, Artifact.RowBounded,
    sparseTermsBoundedCheck_eq_true_iff, and_assoc]

private def entriesBoundedCheck
    (rows : List (Nat × Row)) (rowCount columnCount : Nat) : Bool :=
  rows.all fun entry =>
    decide (entry.1 < rowCount) && rowBoundedCheck entry.2 columnCount

private theorem entriesBoundedCheck_eq_true_iff
    (rows : List (Nat × Row)) (rowCount columnCount : Nat) :
    entriesBoundedCheck rows rowCount columnCount = true ↔
      ∀ entry ∈ rows,
        entry.1 < rowCount ∧ Artifact.RowBounded entry.2 columnCount := by
  simp [entriesBoundedCheck, List.all_eq_true,
    rowBoundedCheck_eq_true_iff, decide_eq_true_eq]

private def columnsBoundedCheck
    (columns : List Nat) (columnCount : Nat) : Bool :=
  columns.all fun column => decide (column ≠ 0 ∧ column < columnCount)

private theorem columnsBoundedCheck_eq_true_iff
    (columns : List Nat) (columnCount : Nat) :
    columnsBoundedCheck columns columnCount = true ↔
      ∀ column ∈ columns, column ≠ 0 ∧ column < columnCount := by
  simp [columnsBoundedCheck, List.all_eq_true, decide_eq_true_eq]

/-! ## Shape and phase-local schedule checks -/

theorem scope : artifact.scope.IsTinyFixture := by
  unfold Scope.IsTinyFixture
  decide

private theorem shared_valid_check :
    sharedValidCheck artifact.shared artifact.scope = true := by
  set_option maxRecDepth 100000 in
    decide

theorem shared : artifact.shared.Valid artifact.scope :=
  (sharedValidCheck_eq_true_iff artifact.shared artifact.scope).mp
    shared_valid_check

private theorem low_limb_valid_check :
    limbValidCheck lowLimb artifact.scope artifact.shared = true := by
  set_option maxRecDepth 100000 in
    decide

private theorem high_limb_valid_check :
    limbValidCheck highLimb artifact.scope artifact.shared = true := by
  set_option maxRecDepth 100000 in
    decide

theorem lowLimbValid : lowLimb.Valid artifact.scope artifact.shared :=
  (limbValidCheck_eq_true_iff lowLimb artifact.scope artifact.shared).mp
    low_limb_valid_check

theorem highLimbValid : highLimb.Valid artifact.scope artifact.shared :=
  (limbValidCheck_eq_true_iff highLimb artifact.scope artifact.shared).mp
    high_limb_valid_check

private theorem limb_list : artifact.limbs = [lowLimb, highLimb] := by
  rfl

theorem limbCount : artifact.limbs.length = 2 := by
  rw [limb_list]
  decide

theorem limbOrder : artifact.limbs.map LimbOwner.limb = [0, 1] := by
  set_option maxRecDepth 100000 in
    decide

theorem limbs : ∀ limb ∈ artifact.limbs,
    limb.Valid artifact.scope artifact.shared := by
  intro limb member
  rw [limb_list] at member
  simp at member
  rcases member with rfl | rfl
  · exact lowLimbValid
  · exact highLimbValid

/-! ## Separately recorded producer vectors -/

private theorem producer_shape_check : producerShapeCheck artifact = true := by
  set_option maxRecDepth 100000 in
    decide

/-- Each recorded vector has its own shape and bounds. This deliberately does
not claim global disjointness between producer vectors. -/
theorem producers : artifact.ProducerShapeValid :=
  (producerShapeCheck_eq_true_iff artifact).mp producer_shape_check

/-! ## Modular physical-row census -/

private def sharedRows : List (Nat × Row) := artifact.sourceRows.take 1892
private def rowsAfterShared : List (Nat × Row) := artifact.sourceRows.drop 1892
private def lowRows : List (Nat × Row) := rowsAfterShared.take 1916
private def highRows : List (Nat × Row) := rowsAfterShared.drop 1916

private def definitionRows (rows : List (Nat × Row)) : List (Nat × Row) :=
  rows.filter fun row => !(artifact.checkIndices.contains row.1)

private def checkRows (rows : List (Nat × Row)) : List (Nat × Row) :=
  rows.filter fun row => artifact.checkIndices.contains row.1

set_option maxRecDepth 100000 in
private theorem source_rows_partition :
    artifact.sourceRows = sharedRows ++ lowRows ++ highRows := by
  rw [List.append_assoc]
  unfold sharedRows lowRows highRows rowsAfterShared
  rw [List.take_append_drop, List.take_append_drop]

theorem sharedDefinitionCount : artifact.shared.indexedDefinitions.length = 1892 := by
  set_option maxRecDepth 100000 in
    decide

theorem lowDefinitionCount : lowLimb.indexedDefinitions.length = 1914 := by
  set_option maxRecDepth 100000 in
    decide

theorem highDefinitionCount : highLimb.indexedDefinitions.length = 1914 := by
  set_option maxRecDepth 100000 in
    decide

theorem lowCheckCount : (lowLimb.indexedChecks artifact.shared).length = 2 := by
  set_option maxRecDepth 100000 in
    decide

theorem highCheckCount : (highLimb.indexedChecks artifact.shared).length = 2 := by
  set_option maxRecDepth 100000 in
    decide

theorem rowCountReconciles : 1892 + (1914 + 2) + (1914 + 2) = 5724 := by
  decide

private theorem shared_rows_length : sharedRows.length = 1892 := by
  set_option maxRecDepth 100000 in
    decide

private theorem low_rows_length : lowRows.length = 1916 := by
  set_option maxRecDepth 100000 in
    decide

private theorem high_rows_length : highRows.length = 1916 := by
  set_option maxRecDepth 100000 in
    decide

private theorem shared_definition_indices :
    (definitionRows sharedRows).map Prod.fst =
      artifact.shared.definitionIndices := by
  set_option maxRecDepth 100000 in
    decide

private theorem low_definition_indices :
    (definitionRows lowRows).map Prod.fst =
      lowLimb.indexedDefinitions.map Prod.fst := by
  set_option maxRecDepth 100000 in
    decide

private theorem high_definition_indices :
    (definitionRows highRows).map Prod.fst =
      highLimb.indexedDefinitions.map Prod.fst := by
  set_option maxRecDepth 100000 in
    decide

private theorem shared_check_indices :
    (checkRows sharedRows).map Prod.fst = [] := by
  set_option maxRecDepth 100000 in
    decide

private theorem low_check_indices :
    (checkRows lowRows).map Prod.fst =
      (lowLimb.indexedChecks artifact.shared).map Prod.fst := by
  set_option maxRecDepth 100000 in
    decide

private theorem high_check_indices :
    (checkRows highRows).map Prod.fst =
      (highLimb.indexedChecks artifact.shared).map Prod.fst := by
  set_option maxRecDepth 100000 in
    decide

private theorem definition_schedule_partition :
    artifact.definitionIndices =
      artifact.shared.definitionIndices ++
      lowLimb.indexedDefinitions.map Prod.fst ++
      highLimb.indexedDefinitions.map Prod.fst := by
  set_option maxRecDepth 100000 in
    decide

private theorem check_schedule_partition :
    artifact.checkIndices =
      (lowLimb.indexedChecks artifact.shared).map Prod.fst ++
      (highLimb.indexedChecks artifact.shared).map Prod.fst := by
  set_option maxRecDepth 100000 in
    decide

private theorem definition_source_indices :
    artifact.definitionSourceRows.map Prod.fst = artifact.definitionIndices := by
  calc
    artifact.definitionSourceRows.map Prod.fst =
        (definitionRows sharedRows).map Prod.fst ++
        (definitionRows lowRows).map Prod.fst ++
        (definitionRows highRows).map Prod.fst := by
      simp [Artifact.definitionSourceRows, definitionRows,
        source_rows_partition, List.filter_append]
    _ = artifact.shared.definitionIndices ++
        lowLimb.indexedDefinitions.map Prod.fst ++
        highLimb.indexedDefinitions.map Prod.fst := by
      rw [shared_definition_indices, low_definition_indices,
        high_definition_indices]
    _ = artifact.definitionIndices := definition_schedule_partition.symm

private theorem check_source_indices :
    artifact.checkSourceRows.map Prod.fst = artifact.checkIndices := by
  calc
    artifact.checkSourceRows.map Prod.fst =
        (checkRows sharedRows).map Prod.fst ++
        (checkRows lowRows).map Prod.fst ++
        (checkRows highRows).map Prod.fst := by
      simp [Artifact.checkSourceRows, checkRows,
        source_rows_partition, List.filter_append]
    _ = (lowLimb.indexedChecks artifact.shared).map Prod.fst ++
        (highLimb.indexedChecks artifact.shared).map Prod.fst := by
      rw [shared_check_indices, low_check_indices, high_check_indices,
        List.nil_append]
    _ = artifact.checkIndices := check_schedule_partition.symm

private theorem shared_rows_bounded_check :
    entriesBoundedCheck sharedRows artifact.scope.sourceArmRowCount
      artifact.scope.sourceArmColumnCount = true := by
  set_option maxRecDepth 100000 in
    decide

private theorem low_rows_bounded_check :
    entriesBoundedCheck lowRows artifact.scope.sourceArmRowCount
      artifact.scope.sourceArmColumnCount = true := by
  set_option maxRecDepth 100000 in
    decide

private theorem high_rows_bounded_check :
    entriesBoundedCheck highRows artifact.scope.sourceArmRowCount
      artifact.scope.sourceArmColumnCount = true := by
  set_option maxRecDepth 100000 in
    decide

set_option maxRecDepth 100000 in
private theorem all_rows_bounded : ∀ entry ∈ artifact.sourceRows,
    entry.1 < artifact.scope.sourceArmRowCount ∧
      Artifact.RowBounded entry.2 artifact.scope.sourceArmColumnCount := by
  have sharedBounded := (entriesBoundedCheck_eq_true_iff sharedRows
    artifact.scope.sourceArmRowCount artifact.scope.sourceArmColumnCount).mp
      shared_rows_bounded_check
  have lowBounded := (entriesBoundedCheck_eq_true_iff lowRows
    artifact.scope.sourceArmRowCount artifact.scope.sourceArmColumnCount).mp
      low_rows_bounded_check
  have highBounded := (entriesBoundedCheck_eq_true_iff highRows
    artifact.scope.sourceArmRowCount artifact.scope.sourceArmColumnCount).mp
      high_rows_bounded_check
  intro entry member
  have partitioned : entry ∈ sharedRows ++ lowRows ++ highRows := by
    rw [← source_rows_partition]
    exact member
  rcases List.mem_append.mp partitioned with inFirstTwo | inHigh
  · rcases List.mem_append.mp inFirstTwo with inShared | inLow
    · exact sharedBounded entry inShared
    · exact lowBounded entry inLow
  · exact highBounded entry inHigh

private theorem row_indices_ordered_check :
    strictlyIncreasingCheck (artifact.sourceRows.map Prod.fst) = true := by
  set_option maxRecDepth 100000 in
    decide

theorem rowIndicesOrdered :
    StrictlyIncreasing (artifact.sourceRows.map Prod.fst) :=
  (strictlyIncreasingCheck_eq_true_iff _).mp row_indices_ordered_check

theorem definitionCount : artifact.definitionIndices.length = activeDefinitionCount := by
  set_option maxRecDepth 100000 in
    decide

theorem checkCount : artifact.checkIndices.length = activeCheckCount := by
  set_option maxRecDepth 100000 in
    decide

theorem rowCount : artifact.sourceRows.length = activeRowCount := by
  rw [source_rows_partition, List.length_append, List.length_append,
    shared_rows_length, low_rows_length, high_rows_length]
  decide

theorem rows : artifact.RowsOwned :=
  ⟨rowIndicesOrdered, definitionCount, checkCount, rowCount,
    all_rows_bounded, definition_source_indices, check_source_indices⟩

/-! ## Exact protocol → phase → family → source-constraint census -/

private def sourceStageCounts (leaf : SourceStageLeaf) : Nat × Nat × Nat :=
  (leaf.definitionCount, leaf.checkCount, leaf.freshColumnCount)

/-- The physical stage vocabulary comes from the Rust constants used by the
emitter. Every leaf is unique; shared beta/rho work appears once. -/
theorem sourceStagePathsUnique :
    (sourceStageLeaves.map SourceStageLeaf.stagePath).Nodup := by
  decide

theorem sourceStageLeafCount : sourceStageLeaves.length = 14 := by
  decide

/-- Exact leaf-level source-R1CS census. Entries are
`(definition rows, assertion rows, fresh definition columns)`. -/
theorem sourceStageLeafCounts :
    sourceStageLeaves.map sourceStageCounts =
      [ (272, 0, 272),
        (1620, 0, 1620),
        (1620, 0, 1620),
        (75, 0, 75),
        (108, 0, 108),
        (106, 0, 106),
        (5, 0, 5),
        (0, 2, 0),
        (1620, 0, 1620),
        (75, 0, 75),
        (108, 0, 108),
        (106, 0, 106),
        (5, 0, 5),
        (0, 2, 0) ] := by
  set_option maxRecDepth 100000 in
    decide

theorem sourceStageDefinitionCount :
    (sourceStageLeaves.map SourceStageLeaf.definitionCount).sum =
      activeDefinitionCount := by
  set_option maxRecDepth 100000 in
    decide

theorem sourceStageCheckCount :
    (sourceStageLeaves.map SourceStageLeaf.checkCount).sum =
      activeCheckCount := by
  set_option maxRecDepth 100000 in
    decide

theorem sourceStageRowCount :
    (sourceStageLeaves.map SourceStageLeaf.rowCount).sum = activeRowCount := by
  set_option maxRecDepth 100000 in
    decide

theorem sourceStageFreshColumnCount :
    (sourceStageLeaves.map SourceStageLeaf.freshColumnCount).sum =
      activeDefinitionCount := by
  set_option maxRecDepth 100000 in
    decide

/-! ## Definition-output and allocated-column census -/

private def definitionOutputs
    (definitions : List (Nat × Definition)) : List Nat :=
  definitions.map fun entry => entry.2.output

private theorem definitionOutputs_append
    (left right : List (Nat × Definition)) :
    definitionOutputs (left ++ right) =
      definitionOutputs left ++ definitionOutputs right := by
  simp [definitionOutputs]

theorem sharedColumnCount : artifact.shared.allocatedColumns.length = 1892 := by
  set_option maxRecDepth 100000 in
    decide

theorem lowColumnCount : lowLimb.allocatedColumns.length = 1914 := by
  set_option maxRecDepth 100000 in
    decide

theorem highColumnCount : highLimb.allocatedColumns.length = 1914 := by
  set_option maxRecDepth 100000 in
    decide

theorem columnCountReconciles : 1892 + 2 * 1914 = 5720 := by
  decide

private theorem shared_outputs_owned :
    definitionOutputs artifact.shared.indexedDefinitions =
      artifact.shared.allocatedColumns := by
  set_option maxRecDepth 100000 in
    decide

private theorem low_outputs_owned :
    definitionOutputs lowLimb.indexedDefinitions =
      lowLimb.allocatedColumns := by
  set_option maxRecDepth 100000 in
    decide

private theorem high_outputs_owned :
    definitionOutputs highLimb.indexedDefinitions =
      highLimb.allocatedColumns := by
  set_option maxRecDepth 100000 in
    decide

private theorem indexed_definitions_partition :
    artifact.indexedDefinitions =
      artifact.shared.indexedDefinitions ++ lowLimb.indexedDefinitions ++
      highLimb.indexedDefinitions := by
  set_option maxRecDepth 100000 in
    decide

private theorem allocated_columns_partition :
    artifact.allocatedColumns =
      artifact.shared.allocatedColumns ++ lowLimb.allocatedColumns ++
      highLimb.allocatedColumns := by
  set_option maxRecDepth 100000 in
    decide

/-- Every scheduled definition's actual `Definition.output` is exactly the
advertised allocated column in the same order. -/
theorem definitionOutputs_eq_allocatedColumns :
    definitionOutputs artifact.indexedDefinitions = artifact.allocatedColumns := by
  rw [indexed_definitions_partition]
  simp only [definitionOutputs_append]
  rw [shared_outputs_owned, low_outputs_owned, high_outputs_owned]
  exact allocated_columns_partition.symm

private theorem shared_columns_bounded_check :
    columnsBoundedCheck artifact.shared.allocatedColumns
      artifact.scope.sourceArmColumnCount = true := by
  set_option maxRecDepth 100000 in
    decide

private theorem low_columns_bounded_check :
    columnsBoundedCheck lowLimb.allocatedColumns
      artifact.scope.sourceArmColumnCount = true := by
  set_option maxRecDepth 100000 in
    decide

private theorem high_columns_bounded_check :
    columnsBoundedCheck highLimb.allocatedColumns
      artifact.scope.sourceArmColumnCount = true := by
  set_option maxRecDepth 100000 in
    decide

set_option maxRecDepth 100000 in
private theorem all_columns_bounded : ∀ column ∈ artifact.allocatedColumns,
    column ≠ 0 ∧ column < artifact.scope.sourceArmColumnCount := by
  have sharedBounded := (columnsBoundedCheck_eq_true_iff
    artifact.shared.allocatedColumns artifact.scope.sourceArmColumnCount).mp
      shared_columns_bounded_check
  have lowBounded := (columnsBoundedCheck_eq_true_iff
    lowLimb.allocatedColumns artifact.scope.sourceArmColumnCount).mp
      low_columns_bounded_check
  have highBounded := (columnsBoundedCheck_eq_true_iff
    highLimb.allocatedColumns artifact.scope.sourceArmColumnCount).mp
      high_columns_bounded_check
  intro column member
  have partitioned : column ∈ artifact.shared.allocatedColumns ++
      lowLimb.allocatedColumns ++ highLimb.allocatedColumns := by
    rw [← allocated_columns_partition]
    exact member
  rcases List.mem_append.mp partitioned with inFirstTwo | inHigh
  · rcases List.mem_append.mp inFirstTwo with inShared | inLow
    · exact sharedBounded column inShared
    · exact lowBounded column inLow
  · exact highBounded column inHigh

private theorem columns_ordered_check :
    strictlyIncreasingCheck artifact.allocatedColumns = true := by
  set_option maxRecDepth 100000 in
    decide

theorem columnsOrdered : StrictlyIncreasing artifact.allocatedColumns :=
  (strictlyIncreasingCheck_eq_true_iff _).mp columns_ordered_check

theorem allocatedColumnCount :
    artifact.allocatedColumns.length = activeDefinitionCount := by
  rw [allocated_columns_partition, List.length_append, List.length_append,
    sharedColumnCount, lowColumnCount, highColumnCount]
  decide

theorem columns : artifact.ColumnsOwned :=
  ⟨allocatedColumnCount, columnsOrdered, all_columns_bounded⟩

/-- Complete source-arm structural census for the selected fixture. This is an
artifact-checked ownership result, not a row-satisfaction or protocol-authority
theorem. -/
theorem structureValid : artifact.StructureValid :=
  ⟨scope, shared, limbCount, limbOrder, limbs, producers, rows, columns⟩

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Census
