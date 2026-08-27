import NightstreamFPrime.Export.Stage1.PiRLCCombinationConformance
import NightstreamFPrime.Export.Stage1.PiRLCCombinationProjection
import NightstreamFPrime.Export.Stage1.PiRLCPackageCompleteness
import NightstreamFPrime.Layout.R1CS.Segments

/-!
Owns the constructive direction from exact PiRLC combination-family physical
rows to the canonical compact invocations. The proof exposes only the fixed
source, block, lane, and cell schedule; it never materializes a production
family row list.
-/

namespace NightstreamFPrime.Export.Stage1.PiRLCCombinationCompleteness

open NightstreamFPrime.Circuit
open NightstreamFPrime.Export.Package
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.PiRLC.v1_1
open NightstreamFPrime.Spec
open NightstreamFPrime.Export.Stage1.PiRLCCombinationInvocations
open NightstreamFPrime.Export.Stage1.PiRLCCombinationConformance
open NightstreamFPrime.Export.Stage1.PiRLCCombinationProjection

def laneConstraints (logicalStart blockCount cellCount valueStride source : Nat)
    (valueSourceStart : Nat → Nat → Nat → Nat) (block : Fin blockCount)
    (lane : Fin ringDegree) : List Expr :=
  List.ofFn fun cell : Fin cellCount =>
    sourceConstraint logicalStart blockCount cellCount valueStride source
      block.val cell.val valueSourceStart lane

def blockConstraints (logicalStart blockCount cellCount valueStride source : Nat)
    (valueSourceStart : Nat → Nat → Nat → Nat) (block : Fin blockCount) :
    List Expr :=
  (List.ofFn fun lane : Fin ringDegree =>
    laneConstraints logicalStart blockCount cellCount valueStride source
      valueSourceStart block lane).flatten

def blockConstraintLists (logicalStart blockCount cellCount valueStride source : Nat)
    (valueSourceStart : Nat → Nat → Nat → Nat) : List (List Expr) :=
  List.ofFn fun block : Fin blockCount =>
    blockConstraints logicalStart blockCount cellCount valueStride source
      valueSourceStart block

private theorem sourceConstraint_coordinates_eq
    (logicalStart blockCount cellCount valueStride source : Nat)
    (valueSourceStart : Nat → Nat → Nat → Nat)
    (index : Fin (CombinationStep.privateCount blockCount cellCount))
    (block : Fin blockCount) (lane : Fin ringDegree) (cell : Fin cellCount)
    (value : index.val = logicalIndex cellCount block.val lane.val cell.val) :
    sourceConstraint logicalStart blockCount cellCount valueStride source
        (CombinationStep.coordinates index).1.val
        (CombinationStep.coordinates index).2.2.val valueSourceStart
        (CombinationStep.coordinates index).2.1 =
      sourceConstraint logicalStart blockCount cellCount valueStride source
        block.val cell.val valueSourceStart lane := by
  rw [coordinates_eq_of_val index block lane cell value]

private theorem totalFreshCount_flatten (lists : List (List Expr)) :
    R1CS.totalFreshCount lists.flatten =
      (lists.map R1CS.totalFreshCount).sum := by
  induction lists with
  | nil => rfl
  | cons first rest inductionHypothesis =>
      simp only [List.flatten_cons, R1CS.totalFreshCount_append,
        List.map_cons, List.sum_cons, inductionHypothesis]

private theorem sum_map_mul_left (factor : Nat) (values : List Nat) :
    (values.map fun value => factor * value).sum = factor * values.sum := by
  induction values with
  | nil => simp
  | cons value rest inductionHypothesis =>
      simp [inductionHypothesis, Nat.mul_add]

private theorem laneFreshCost_eq (lane : Fin ringDegree) :
    laneFreshCost lane.val =
      NightstreamFPrime.Layout.PiRLC.v1_1.CombinationStep.laneFreshCount lane := by
  simp [laneFreshCost, laneFreshCosts, lane.isLt]

private theorem laneFreshCosts_ofFn :
    List.ofFn (fun lane : Fin ringDegree => laneFreshCost lane.val) =
      laneFreshCosts := by
  unfold laneFreshCosts
  apply congrArg List.ofFn
  funext lane
  exact laneFreshCost_eq lane

private theorem laneConstraints_totalFreshCount
    (logicalStart blockCount cellCount valueStride source : Nat)
    (valueSourceStart : Nat → Nat → Nat → Nat)
    (freshCount : ∀ (block : Fin blockCount) (lane : Fin ringDegree)
      (cell : Fin cellCount),
      R1CS.constraintFreshCount
          (sourceConstraint logicalStart blockCount cellCount valueStride source
            block.val cell.val valueSourceStart lane) =
        laneFreshCost lane.val)
    (block : Fin blockCount) (lane : Fin ringDegree) :
    R1CS.totalFreshCount
        (laneConstraints logicalStart blockCount cellCount valueStride source
          valueSourceStart block lane) =
      cellCount * laneFreshCost lane.val := by
  unfold R1CS.totalFreshCount laneConstraints
  rw [List.map_ofFn]
  change (List.ofFn fun cell : Fin cellCount =>
    R1CS.constraintFreshCount
      (sourceConstraint logicalStart blockCount cellCount valueStride source
        block.val cell.val valueSourceStart lane)).sum = _
  have costsEq :
      (List.ofFn fun cell : Fin cellCount =>
        R1CS.constraintFreshCount
          (sourceConstraint logicalStart blockCount cellCount valueStride source
            block.val cell.val valueSourceStart lane)) =
        List.ofFn (fun _ : Fin cellCount => laneFreshCost lane.val) := by
    apply congrArg List.ofFn
    funext cell
    exact freshCount block lane cell
  rw [costsEq]
  rw [List.ofFn_const, List.sum_const_nat]

private theorem blockConstraints_totalFreshCount
    (logicalStart blockCount cellCount valueStride source : Nat)
    (valueSourceStart : Nat → Nat → Nat → Nat)
    (freshCount : ∀ (block : Fin blockCount) (lane : Fin ringDegree)
      (cell : Fin cellCount),
      R1CS.constraintFreshCount
          (sourceConstraint logicalStart blockCount cellCount valueStride source
            block.val cell.val valueSourceStart lane) =
        laneFreshCost lane.val)
    (block : Fin blockCount) :
    R1CS.totalFreshCount
        (blockConstraints logicalStart blockCount cellCount valueStride source
          valueSourceStart block) =
      cellCount * 8100 := by
  unfold blockConstraints
  rw [totalFreshCount_flatten, List.map_ofFn]
  change (List.ofFn fun lane : Fin ringDegree =>
    R1CS.totalFreshCount
      (laneConstraints logicalStart blockCount cellCount valueStride source
        valueSourceStart block lane)).sum = _
  have laneCountsEq :
      (List.ofFn fun lane : Fin ringDegree =>
        R1CS.totalFreshCount
          (laneConstraints logicalStart blockCount cellCount valueStride source
            valueSourceStart block lane)) =
        List.ofFn fun lane : Fin ringDegree =>
          cellCount * laneFreshCost lane.val := by
    apply congrArg List.ofFn
    funext lane
    exact laneConstraints_totalFreshCount logicalStart blockCount cellCount
      valueStride source valueSourceStart freshCount block lane
  rw [laneCountsEq]
  calc
    (List.ofFn fun lane : Fin ringDegree =>
        cellCount * laneFreshCost lane.val).sum =
      ((List.ofFn fun lane : Fin ringDegree => laneFreshCost lane.val).map
        fun cost => cellCount * cost).sum := by
          simp [List.map_ofFn, Function.comp_def]
    _ = cellCount *
        (List.ofFn fun lane : Fin ringDegree => laneFreshCost lane.val).sum :=
      sum_map_mul_left cellCount _
    _ = cellCount * laneFreshCosts.sum := by rw [laneFreshCosts_ofFn]
    _ = cellCount * 8100 := by rw [laneFreshCosts_sum]

private theorem sum_take_ofFn_const {count : Nat} (value : Nat)
    (index : Fin count) :
    ((List.ofFn fun _ : Fin count => value).take index.val).sum =
      index.val * value := by
  simp [List.sum_const_nat, index.isLt.le]

private theorem weightedLaneFreshPrefix (cellCount : Nat)
    (lane : Fin ringDegree) :
    ((List.ofFn fun current : Fin ringDegree =>
      cellCount * laneFreshCost current.val).take lane.val).sum =
        cellCount * laneFreshPrefix lane.val := by
  have weightedEq :
      (List.ofFn fun current : Fin ringDegree =>
        cellCount * laneFreshCost current.val) =
        (List.ofFn fun current : Fin ringDegree =>
          laneFreshCost current.val).map (fun cost => cellCount * cost) := by
    simp [List.map_ofFn, Function.comp_def]
  rw [weightedEq]
  calc
    (((List.ofFn fun current : Fin ringDegree =>
        laneFreshCost current.val).map
          (fun cost => cellCount * cost)).take lane.val).sum =
      (((List.ofFn fun current : Fin ringDegree =>
        laneFreshCost current.val).take lane.val).map
          (fun cost => cellCount * cost)).sum := by
            simp only [List.map_take]
    _ = cellCount *
        ((List.ofFn fun current : Fin ringDegree =>
          laneFreshCost current.val).take lane.val).sum :=
      sum_map_mul_left cellCount _
    _ = cellCount * laneFreshPrefix lane.val := by
      rw [laneFreshCosts_ofFn]
      rfl

private theorem blockFreshPrefix
    (logicalStart blockCount cellCount valueStride source : Nat)
    (valueSourceStart : Nat → Nat → Nat → Nat)
    (freshCount : ∀ (block : Fin blockCount) (lane : Fin ringDegree)
      (cell : Fin cellCount),
      R1CS.constraintFreshCount
          (sourceConstraint logicalStart blockCount cellCount valueStride source
            block.val cell.val valueSourceStart lane) =
        laneFreshCost lane.val)
    (block : Fin blockCount) :
    ((List.ofFn fun current : Fin blockCount =>
      R1CS.totalFreshCount
        (blockConstraints logicalStart blockCount cellCount valueStride source
          valueSourceStart current)).take block.val).sum =
      block.val * (cellCount * 8100) := by
  have countsEq :
      (List.ofFn fun current : Fin blockCount =>
        R1CS.totalFreshCount
          (blockConstraints logicalStart blockCount cellCount valueStride source
            valueSourceStart current)) =
        List.ofFn (fun _ : Fin blockCount => cellCount * 8100) := by
    apply congrArg List.ofFn
    funext current
    exact blockConstraints_totalFreshCount logicalStart blockCount cellCount
      valueStride source valueSourceStart freshCount current
  rw [countsEq]
  exact sum_take_ofFn_const (cellCount * 8100) block

private theorem laneFreshPrefix_total
    (logicalStart blockCount cellCount valueStride source : Nat)
    (valueSourceStart : Nat → Nat → Nat → Nat)
    (freshCount : ∀ (block : Fin blockCount) (lane : Fin ringDegree)
      (cell : Fin cellCount),
      R1CS.constraintFreshCount
          (sourceConstraint logicalStart blockCount cellCount valueStride source
            block.val cell.val valueSourceStart lane) =
        laneFreshCost lane.val)
    (block : Fin blockCount) (lane : Fin ringDegree) :
    ((List.ofFn fun current : Fin ringDegree =>
      R1CS.totalFreshCount
        (laneConstraints logicalStart blockCount cellCount valueStride source
          valueSourceStart block current)).take lane.val).sum =
      cellCount * laneFreshPrefix lane.val := by
  have countsEq :
      (List.ofFn fun current : Fin ringDegree =>
        R1CS.totalFreshCount
          (laneConstraints logicalStart blockCount cellCount valueStride source
            valueSourceStart block current)) =
        List.ofFn fun current : Fin ringDegree =>
          cellCount * laneFreshCost current.val := by
    apply congrArg List.ofFn
    funext current
    exact laneConstraints_totalFreshCount logicalStart blockCount cellCount
      valueStride source valueSourceStart freshCount block current
  rw [countsEq]
  exact weightedLaneFreshPrefix cellCount lane

private theorem cellFreshPrefix
    (logicalStart blockCount cellCount valueStride source : Nat)
    (valueSourceStart : Nat → Nat → Nat → Nat)
    (freshCount : ∀ (block : Fin blockCount) (lane : Fin ringDegree)
      (cell : Fin cellCount),
      R1CS.constraintFreshCount
          (sourceConstraint logicalStart blockCount cellCount valueStride source
            block.val cell.val valueSourceStart lane) =
        laneFreshCost lane.val)
    (block : Fin blockCount) (lane : Fin ringDegree) (cell : Fin cellCount) :
    ((List.ofFn fun current : Fin cellCount =>
      R1CS.constraintFreshCount
        (sourceConstraint logicalStart blockCount cellCount valueStride source
          block.val current.val valueSourceStart lane)).take cell.val).sum =
      cell.val * laneFreshCost lane.val := by
  have countsEq :
      (List.ofFn fun current : Fin cellCount =>
        R1CS.constraintFreshCount
          (sourceConstraint logicalStart blockCount cellCount valueStride source
            block.val current.val valueSourceStart lane)) =
        List.ofFn (fun _ : Fin cellCount => laneFreshCost lane.val) := by
    apply congrArg List.ofFn
    funext current
    exact freshCount block lane current
  rw [countsEq]
  exact sum_take_ofFn_const (laneFreshCost lane.val) cell

/-- The compiler's source list has exactly the canonical block-major,
lane-major, cell-major nesting used by the compact invocation schedule. -/
theorem sourceConstraints_eq_blockConstraintLists
    (logicalStart blockCount cellCount valueStride source : Nat)
    (valueSourceStart : Nat → Nat → Nat → Nat) :
    sourceConstraints logicalStart blockCount cellCount valueStride source
        valueSourceStart =
      (blockConstraintLists logicalStart blockCount cellCount valueStride source
        valueSourceStart).flatten := by
  unfold sourceConstraints blockConstraintLists blockConstraints laneConstraints
  unfold CombinationStep.privateCount
  rw [List.ofFn_mul]
  apply congrArg List.flatten
  apply congrArg List.ofFn
  funext block
  rw [List.ofFn_mul]
  apply congrArg List.flatten
  apply congrArg List.ofFn
  funext lane
  apply congrArg List.ofFn
  funext cell
  dsimp only
  apply sourceConstraint_coordinates_eq
  simp [logicalIndex]
  ring

private theorem familyConstraints_eq_sourceConstraintLists
    (logicalStart blockCount cellCount valueStride : Nat)
    (valueSourceStart : Nat → Nat → Nat → Nat) :
    familyConstraints logicalStart blockCount cellCount valueStride
        valueSourceStart =
      (List.ofFn fun source : Fin sourceCount =>
        sourceConstraints logicalStart blockCount cellCount valueStride
          source.val valueSourceStart).flatten := by
  unfold familyConstraints
  apply congrArg List.flatten
  rw [List.ofFn_eq_map, ← List.map_coe_finRange_eq_range,
    List.map_map]
  simp [Function.comp_def]

private theorem sourceConstraints_totalFreshCount
    {blockCount cellCount : Nat} [NeZero cellCount]
    (logicalStart valueStride source : Nat)
    (valueSourceStart : Nat → Nat → Nat → Nat)
    (freshCount : ∀ (block : Fin blockCount) (lane : Fin ringDegree)
      (cell : Fin cellCount),
      R1CS.constraintFreshCount
          (sourceConstraint logicalStart blockCount cellCount valueStride source
            block.val cell.val valueSourceStart lane) =
        laneFreshCost lane.val) :
    R1CS.totalFreshCount
        (sourceConstraints logicalStart blockCount cellCount valueStride source
          valueSourceStart) =
      sourceFreshCount blockCount cellCount := by
  rw [sourceConstraints_eq_blockConstraintLists, totalFreshCount_flatten]
  unfold blockConstraintLists
  rw [List.map_ofFn]
  change (List.ofFn fun block : Fin blockCount =>
    R1CS.totalFreshCount
      (blockConstraints logicalStart blockCount cellCount valueStride source
        valueSourceStart block)).sum = _
  have countsEq :
      (List.ofFn fun block : Fin blockCount =>
        R1CS.totalFreshCount
          (blockConstraints logicalStart blockCount cellCount valueStride source
            valueSourceStart block)) =
        List.ofFn (fun _ : Fin blockCount => cellCount * 8100) := by
    apply congrArg List.ofFn
    funext block
    exact blockConstraints_totalFreshCount logicalStart blockCount cellCount
      valueStride source valueSourceStart freshCount block
  rw [countsEq, List.ofFn_const, List.sum_const_nat]
  simp [sourceFreshCount]
  ring

private theorem sourceFreshPrefix_total
    {blockCount cellCount : Nat} [NeZero cellCount]
    (logicalStart valueStride : Nat)
    (valueSourceStart : Nat → Nat → Nat → Nat)
    (freshCount : ∀ (source : Fin sourceCount) (block : Fin blockCount)
      (lane : Fin ringDegree) (cell : Fin cellCount),
      R1CS.constraintFreshCount
          (sourceConstraint logicalStart blockCount cellCount valueStride
            source.val block.val cell.val valueSourceStart lane) =
        laneFreshCost lane.val)
    (source : Fin sourceCount) :
    ((List.ofFn fun current : Fin sourceCount =>
      R1CS.totalFreshCount
        (sourceConstraints logicalStart blockCount cellCount valueStride
          current.val valueSourceStart)).take source.val).sum =
      source.val * sourceFreshCount blockCount cellCount := by
  have countsEq :
      (List.ofFn fun current : Fin sourceCount =>
        R1CS.totalFreshCount
          (sourceConstraints logicalStart blockCount cellCount valueStride
            current.val valueSourceStart)) =
        List.ofFn (fun _ : Fin sourceCount =>
          sourceFreshCount blockCount cellCount) := by
    apply congrArg List.ofFn
    funext current
    exact sourceConstraints_totalFreshCount logicalStart valueStride current.val
      valueSourceStart (freshCount current)
  rw [countsEq]
  exact sum_take_ofFn_const (sourceFreshCount blockCount cellCount) source

/-- Exact source-family rows project to the canonical compact invocation at
one block, lane, and cell. All three fresh-column prefixes come from the
Lean source list; the compact invocation contributes no independent index
or allocation rule. -/
theorem sourcePhysicalRows_imply_invocationRows
    {blockCount cellCount : Nat} [NeZero cellCount]
    (logicalStart rowStart freshStart valueStride source : Nat)
    (valueSourceStart : Nat → Nat → Nat → Nat)
    (freshStartLocal : Spartan.piCcsPhaseOffset ≤ freshStart)
    (freshCount : ∀ (block : Fin blockCount) (lane : Fin ringDegree)
      (cell : Fin cellCount),
      R1CS.constraintFreshCount
          (sourceConstraint logicalStart blockCount cellCount valueStride source
            block.val cell.val valueSourceStart lane) =
        laneFreshCost lane.val)
    (valueAffine : ∀ (block : Fin blockCount) (cell : Fin cellCount)
      (offset : Nat), offset < ringDegree →
      Spartan.sourceToSpartan
          (valueSourceStart source block.val cell.val + offset * valueStride) =
        Spartan.sourceToSpartan
            (valueSourceStart source block.val cell.val) +
          offset * valueStride)
    (env : Env)
    (rows : R1CS.RowsHold (Spartan.pullback env)
      (R1CS.lowerConstraints
        (sourceConstraints logicalStart blockCount cellCount valueStride source
          valueSourceStart)
        (freshStart + source * sourceFreshCount blockCount cellCount)).rows)
    (block : Fin blockCount) (lane : Fin ringDegree) (cell : Fin cellCount) :
    R1CS.RowsHold env
      (CompactRows.instantiateRows
        (CompactRows.inputColumnOfRanges
          (invocation logicalStart rowStart freshStart blockCount cellCount
            valueStride source block.val lane.val cell.val
              valueSourceStart).inputRanges)
        (invocation logicalStart rowStart freshStart blockCount cellCount
          valueStride source block.val lane.val cell.val
            valueSourceStart).localStart
        (PiRLCCombinationTemplates.template (firstSource source) lane)) := by
  rw [sourceConstraints_eq_blockConstraintLists] at rows
  have blockSegments : R1CS.SegmentsHold (Spartan.pullback env)
      (List.ofFn fun current : Fin blockCount =>
        blockConstraints logicalStart blockCount cellCount valueStride source
          valueSourceStart current)
      (freshStart + source * sourceFreshCount blockCount cellCount) := by
    apply (R1CS.rowsHold_flatten_iff _ _ _).mp
    simpa [blockConstraintLists] using rows
  have blockRows := R1CS.segmentsHold_ofFn_get (Spartan.pullback env)
    (fun current : Fin blockCount =>
      blockConstraints logicalStart blockCount cellCount valueStride source
        valueSourceStart current)
    (freshStart + source * sourceFreshCount blockCount cellCount)
    blockSegments block
  rw [blockFreshPrefix logicalStart blockCount cellCount valueStride source
    valueSourceStart freshCount block] at blockRows
  have laneSegments : R1CS.SegmentsHold (Spartan.pullback env)
      (List.ofFn fun current : Fin ringDegree =>
        laneConstraints logicalStart blockCount cellCount valueStride source
          valueSourceStart block current)
      ((freshStart + source * sourceFreshCount blockCount cellCount) +
        block.val * (cellCount * 8100)) := by
    apply (R1CS.rowsHold_flatten_iff _ _ _).mp
    simpa [blockConstraints] using blockRows
  have laneRows := R1CS.segmentsHold_ofFn_get (Spartan.pullback env)
    (fun current : Fin ringDegree =>
      laneConstraints logicalStart blockCount cellCount valueStride source
        valueSourceStart block current)
    ((freshStart + source * sourceFreshCount blockCount cellCount) +
      block.val * (cellCount * 8100)) laneSegments lane
  rw [laneFreshPrefix_total logicalStart blockCount cellCount valueStride source
    valueSourceStart freshCount block lane] at laneRows
  have cellRows := R1CS.rowsHold_lowerConstraints_ofFn_get
    (Spartan.pullback env)
    (fun current : Fin cellCount =>
      sourceConstraint logicalStart blockCount cellCount valueStride source
        block.val current.val valueSourceStart lane)
    (((freshStart + source * sourceFreshCount blockCount cellCount) +
      block.val * (cellCount * 8100)) +
        cellCount * laneFreshPrefix lane.val)
    (by simpa [laneConstraints] using laneRows) cell
  rw [cellFreshPrefix logicalStart blockCount cellCount valueStride source
    valueSourceStart freshCount block lane cell] at cellRows
  have startEq :
      ((((freshStart + source * sourceFreshCount blockCount cellCount) +
          block.val * (cellCount * 8100)) +
        cellCount * laneFreshPrefix lane.val) +
          cell.val * laneFreshCost lane.val) =
        invocationFreshSource freshStart blockCount cellCount source block.val
          lane.val cell.val := by
    simp [invocationFreshSource, sourceFreshCount, coordinateFreshPrefix]
    ring
  rw [startEq] at cellRows
  have costPositive : 0 < laneFreshCost lane.val := by
    rw [laneFreshCost_eq lane]
    unfold NightstreamFPrime.Layout.PiRLC.v1_1.CombinationStep.laneFreshCount
    omega
  have constraintPositive :
      0 < R1CS.constraintFreshCount
        (sourceConstraint logicalStart blockCount cellCount valueStride source
          block.val cell.val valueSourceStart lane) := by
    rw [freshCount block lane cell]
    exact costPositive
  rw [R1CS.lowerConstraint_eq_lowerGenericConstraint_of_fresh_pos _ _
    constraintPositive] at cellRows
  exact PiRLCCombinationConformance.sourceRows_imply_invocationRows
    logicalStart rowStart freshStart blockCount cellCount valueStride source
      block.val cell.val lane valueSourceStart
      (invocationFreshSource_local _ _ _ _ _ _ _ freshStartLocal)
      (valueAffine block cell) env cellRows

/-- One exact combination-family row packet constructs all of its canonical
compact invocations in source-major order. -/
theorem familyPhysicalRows_imply_invocationRows
    {blockCount cellCount : Nat} [NeZero cellCount]
    (logicalStart rowStart freshStart valueStride : Nat)
    (valueSourceStart : Nat → Nat → Nat → Nat)
    (freshStartLocal : Spartan.piCcsPhaseOffset ≤ freshStart)
    (freshCount : ∀ (source : Fin sourceCount) (block : Fin blockCount)
      (lane : Fin ringDegree) (cell : Fin cellCount),
      R1CS.constraintFreshCount
          (sourceConstraint logicalStart blockCount cellCount valueStride
            source.val block.val cell.val valueSourceStart lane) =
        laneFreshCost lane.val)
    (valueAffine : ∀ (source : Fin sourceCount) (block : Fin blockCount)
      (cell : Fin cellCount) (offset : Nat), offset < ringDegree →
      Spartan.sourceToSpartan
          (valueSourceStart source.val block.val cell.val +
            offset * valueStride) =
        Spartan.sourceToSpartan
            (valueSourceStart source.val block.val cell.val) +
          offset * valueStride)
    (env : Env)
    (rows : R1CS.RowsHold (Spartan.pullback env)
      (R1CS.lowerConstraints
        (familyConstraints logicalStart blockCount cellCount valueStride
          valueSourceStart) freshStart).rows) :
    FamilyInvocationRowsHold logicalStart rowStart freshStart blockCount
      cellCount valueStride valueSourceStart env := by
  rw [familyConstraints_eq_sourceConstraintLists] at rows
  have sourceSegments : R1CS.SegmentsHold (Spartan.pullback env)
      (List.ofFn fun current : Fin sourceCount =>
        sourceConstraints logicalStart blockCount cellCount valueStride
          current.val valueSourceStart) freshStart :=
    (R1CS.rowsHold_flatten_iff _ _ _).mp rows
  intro source index
  dsimp only
  have sourceRows := R1CS.segmentsHold_ofFn_get (Spartan.pullback env)
    (fun current : Fin sourceCount =>
      sourceConstraints logicalStart blockCount cellCount valueStride
        current.val valueSourceStart)
    freshStart sourceSegments source
  rw [sourceFreshPrefix_total logicalStart valueStride valueSourceStart
    freshCount source] at sourceRows
  exact sourcePhysicalRows_imply_invocationRows logicalStart rowStart freshStart
    valueStride source.val valueSourceStart freshStartLocal (freshCount source)
    (valueAffine source) env sourceRows (CombinationStep.coordinates index).1
      (CombinationStep.coordinates index).2.1
      (CombinationStep.coordinates index).2.2

private theorem sourceConstraint_freshCount_of_production
    {blockCount cellCount : Nat} [NeZero cellCount]
    (interface : CombinationFamily.Interface blockCount cellCount)
    (logicalStart valueStride : Nat)
    (valueSourceStart : Nat → Nat → Nat → Nat)
    (inputs :
      NightstreamFPrime.Layout.PiRLC.v1_1.CombinationFamily.ProductionInputs
        interface logicalStart)
    (source : Fin sourceCount) (block : Fin blockCount)
    (lane : Fin ringDegree) (cell : Fin cellCount)
    (constraintEq :
      sourceConstraint logicalStart blockCount cellCount valueStride source.val
          block.val cell.val valueSourceStart lane =
        CombinationStep.output
            (CombinationFamily.stepOffset logicalStart source.val blockCount
              cellCount)
            (CombinationStep.indexOf block lane cell) -
          CombinationStep.recipe
            (CombinationFamily.stepInterface interface logicalStart source.val)
            (CombinationFamily.stepOffset logicalStart source.val blockCount
              cellCount)
            (CombinationStep.indexOf block lane cell)) :
    R1CS.constraintFreshCount
        (sourceConstraint logicalStart blockCount cellCount valueStride source.val
          block.val cell.val valueSourceStart lane) =
      laneFreshCost lane.val := by
  have sourceLt : source.val < CombinationFamily.sourceCount := by
    rw [CombinationFamily.sourceCount_eq]
    exact source.isLt
  exact PiRLCCombinationConformance.sourceConstraint_freshCount_eq interface
    logicalStart valueStride source.val sourceLt valueSourceStart inputs block
      lane cell constraintEq

private theorem commitmentSourceFreshCount
    (source : Fin sourceCount) (block : Fin 18) (lane : Fin ringDegree)
    (cell : Fin 1) :
    R1CS.constraintFreshCount
        (sourceConstraint PiRLCStarts.commitmentLogicalStart 18 1 1 source.val
          block.val cell.val commitmentValueSourceStart lane) =
      laneFreshCost lane.val := by
  let canonicalSource : Fin CombinationFamily.sourceCount :=
    ⟨source.val, by
      rw [CombinationFamily.sourceCount_eq]
      exact source.isLt⟩
  have inputs := PiRLCInputs.commitmentProductionInputs
    (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits)
  exact sourceConstraint_freshCount_of_production
    (productionCommitmentFamilyInterface (logicalWidth := Data.logicalWidth)
      (publicFits := Data.publicFits))
    PiRLCStarts.commitmentLogicalStart 1 commitmentValueSourceStart
    (by simpa [productionCommitmentFamilyInterface, productionSharedInterface,
      PiRLCStarts.commitmentLogicalStart, PiRLCStarts.phaseLogicalStart] using
      inputs)
    source block lane cell (by simpa [canonicalSource] using
      (commitmentSourceConstraint_eq_stepAssertion
        (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits)
        canonicalSource block lane cell))

private theorem publicInputSourceFreshCount
    (source : Fin sourceCount) (block : Fin 1) (lane : Fin ringDegree)
    (cell : Fin 1) :
    R1CS.constraintFreshCount
        (sourceConstraint PiRLCStarts.publicInputLogicalStart 1 1 1 source.val
          block.val cell.val publicInputValueSourceStart lane) =
      laneFreshCost lane.val := by
  let canonicalSource : Fin CombinationFamily.sourceCount :=
    ⟨source.val, by
      rw [CombinationFamily.sourceCount_eq]
      exact source.isLt⟩
  have inputs := PiRLCInputs.publicInputProductionInputs
    (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits)
  exact sourceConstraint_freshCount_of_production
    (productionPublicInputFamilyInterface (logicalWidth := Data.logicalWidth)
      (publicFits := Data.publicFits))
    PiRLCStarts.publicInputLogicalStart 1 publicInputValueSourceStart
    (by simpa [productionPublicInputFamilyInterface, productionSharedInterface,
      PiRLCStarts.publicInputLogicalStart, PiRLCStarts.phaseLogicalStart] using
      inputs)
    source block lane cell (by simpa [canonicalSource] using
      (publicInputSourceConstraint_eq_stepAssertion
        (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits)
        canonicalSource block lane cell))

private theorem evalKSourceFreshCount
    (source : Fin sourceCount) (block : Fin 1) (lane : Fin ringDegree)
    (cell : Fin 2) :
    R1CS.constraintFreshCount
        (sourceConstraint PiRLCStarts.evalKLogicalStart 1 2 2 source.val
          block.val cell.val evalKValueSourceStart lane) =
      laneFreshCost lane.val := by
  let canonicalSource : Fin CombinationFamily.sourceCount :=
    ⟨source.val, by
      rw [CombinationFamily.sourceCount_eq]
      exact source.isLt⟩
  have inputs := PiRLCInputs.evalKProductionInputs
    (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits)
  exact sourceConstraint_freshCount_of_production
    (productionEvalKFamilyInterface (logicalWidth := Data.logicalWidth)
      (publicFits := Data.publicFits))
    PiRLCStarts.evalKLogicalStart 2 evalKValueSourceStart
    (by simpa [productionEvalKFamilyInterface, productionSharedInterface,
      PiRLCStarts.evalKLogicalStart, PiRLCStarts.phaseLogicalStart] using
      inputs)
    source block lane cell (by simpa [canonicalSource] using
      (evalKSourceConstraint_eq_stepAssertion
        (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits)
        canonicalSource block lane cell))

private theorem evalASourceFreshCount
    (source : Fin sourceCount) (block : Fin 14) (lane : Fin ringDegree)
    (cell : Fin 2) :
    R1CS.constraintFreshCount
        (sourceConstraint PiRLCStarts.evalALogicalStart 14 2 2 source.val
          block.val cell.val evalAValueSourceStart lane) =
      laneFreshCost lane.val := by
  let canonicalSource : Fin CombinationFamily.sourceCount :=
    ⟨source.val, by
      rw [CombinationFamily.sourceCount_eq]
      exact source.isLt⟩
  have inputs := PiRLCInputs.evalAProductionInputs
    (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits)
  exact sourceConstraint_freshCount_of_production
    (productionEvalAFamilyInterface (logicalWidth := Data.logicalWidth)
      (publicFits := Data.publicFits))
    PiRLCStarts.evalALogicalStart 2 evalAValueSourceStart
    (by simpa [productionEvalAFamilyInterface, productionSharedInterface,
      PiRLCStarts.evalALogicalStart, PiRLCStarts.phaseLogicalStart] using
      inputs)
    source block lane cell (by simpa [canonicalSource] using
      (evalASourceConstraint_eq_stepAssertion
        (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits)
        canonicalSource block lane cell))

/-- The four exact production combination packets, expressed as their full
canonical compact invocation families. -/
structure ProductionFamilyInvocationRowsHold (env : Env) : Prop where
  commitment : FamilyInvocationRowsHold PiRLCStarts.commitmentLogicalStart
    PiRLCStarts.commitmentRowStart PiRLCStarts.commitmentFreshStart 18 1 1
    commitmentValueSourceStart env
  publicInput : FamilyInvocationRowsHold PiRLCStarts.publicInputLogicalStart
    PiRLCStarts.publicInputRowStart PiRLCStarts.publicInputFreshStart 1 1 1
    publicInputValueSourceStart env
  evalK : FamilyInvocationRowsHold PiRLCStarts.evalKLogicalStart
    PiRLCStarts.evalKRowStart PiRLCStarts.evalKFreshStart 1 2 2
    evalKValueSourceStart env
  evalA : FamilyInvocationRowsHold PiRLCStarts.evalALogicalStart
    PiRLCStarts.evalARowStart PiRLCStarts.evalAFreshStart 14 2 2
    evalAValueSourceStart env

/-- The remapped child packets construct every exact production combination
invocation. The proof reverses only the one fixed Spartan column map. -/
theorem remappedPackets_imply_familyInvocationRows
    (env : Env)
    (packets : PiRLCPackageCompleteness.RemappedPacketRowsHold env) :
    ProductionFamilyInvocationRowsHold env := by
  have commitmentRows := (Spartan.remapRows_hold env _).mp packets.commitment
  have publicInputRows :=
    (Spartan.remapRows_hold env _).mp packets.publicInput
  have evalKRows := (Spartan.remapRows_hold env _).mp packets.evalK
  have evalARows := (Spartan.remapRows_hold env _).mp packets.evalA
  change R1CS.RowsHold (Spartan.pullback env)
    (R1CS.lowerConstraints
      (NightstreamFPrime.Layout.PiRLC.v1_1.CombinationFamily.logicalConstraints
        (productionCommitmentFamilyInterface (logicalWidth := Data.logicalWidth)
          (publicFits := Data.publicFits)) PiRLCStarts.commitmentLogicalStart)
      PiRLCStarts.commitmentFreshStart).rows at commitmentRows
  change R1CS.RowsHold (Spartan.pullback env)
    (R1CS.lowerConstraints
      (NightstreamFPrime.Layout.PiRLC.v1_1.CombinationFamily.logicalConstraints
        (productionPublicInputFamilyInterface
          (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits))
        PiRLCStarts.publicInputLogicalStart)
      PiRLCStarts.publicInputFreshStart).rows at publicInputRows
  change R1CS.RowsHold (Spartan.pullback env)
    (R1CS.lowerConstraints
      (NightstreamFPrime.Layout.PiRLC.v1_1.CombinationFamily.logicalConstraints
        (productionEvalKFamilyInterface (logicalWidth := Data.logicalWidth)
          (publicFits := Data.publicFits)) PiRLCStarts.evalKLogicalStart)
      PiRLCStarts.evalKFreshStart).rows at evalKRows
  change R1CS.RowsHold (Spartan.pullback env)
    (R1CS.lowerConstraints
      (NightstreamFPrime.Layout.PiRLC.v1_1.CombinationFamily.logicalConstraints
        (productionEvalAFamilyInterface (logicalWidth := Data.logicalWidth)
          (publicFits := Data.publicFits)) PiRLCStarts.evalALogicalStart)
      PiRLCStarts.evalAFreshStart).rows at evalARows
  rw [← commitmentFamilyConstraints_eq_parent] at commitmentRows
  rw [← publicInputFamilyConstraints_eq_parent] at publicInputRows
  rw [← evalKFamilyConstraints_eq_parent] at evalKRows
  rw [← evalAFamilyConstraints_eq_parent] at evalARows
  refine ⟨?_, ?_, ?_, ?_⟩
  · refine familyPhysicalRows_imply_invocationRows
      (blockCount := 18) (cellCount := 1)
      PiRLCStarts.commitmentLogicalStart PiRLCStarts.commitmentRowStart
      PiRLCStarts.commitmentFreshStart 1 commitmentValueSourceStart
      commitmentFreshStart_local commitmentSourceFreshCount ?_ env
      commitmentRows
    intro source block cell offset offsetLt
    simpa using commitmentValueSource_affine source.val block.val cell.val offset
      source.isLt block.isLt offsetLt
  · refine familyPhysicalRows_imply_invocationRows
      (blockCount := 1) (cellCount := 1)
      PiRLCStarts.publicInputLogicalStart PiRLCStarts.publicInputRowStart
      PiRLCStarts.publicInputFreshStart 1 publicInputValueSourceStart
      publicInputFreshStart_local publicInputSourceFreshCount ?_ env
      publicInputRows
    intro source block cell offset offsetLt
    simpa using publicInputValueSource_affine source.val block.val cell.val offset
      source.isLt offsetLt
  · refine familyPhysicalRows_imply_invocationRows
      (blockCount := 1) (cellCount := 2)
      PiRLCStarts.evalKLogicalStart PiRLCStarts.evalKRowStart
      PiRLCStarts.evalKFreshStart 2 evalKValueSourceStart
      evalKFreshStart_local evalKSourceFreshCount ?_ env evalKRows
    intro source block cell offset offsetLt
    exact evalKValueSource_affine source.val block.val cell.val offset source.isLt
      cell.isLt offsetLt
  · refine familyPhysicalRows_imply_invocationRows
      (blockCount := 14) (cellCount := 2)
      PiRLCStarts.evalALogicalStart PiRLCStarts.evalARowStart
      PiRLCStarts.evalAFreshStart 2 evalAValueSourceStart
      evalAFreshStart_local evalASourceFreshCount ?_ env evalARows
    intro source block cell offset offsetLt
    exact evalAValueSource_affine source.val block.val cell.val offset source.isLt
      block.isLt cell.isLt offsetLt

private theorem familyRows_imply_packageInvocations
    {blockCount cellCount : Nat} [NeZero cellCount]
    (logicalStart rowStart freshStart valueStride : Nat)
    (valueSourceStart : Nat → Nat → Nat → Nat) (env : Env)
    (rows : FamilyInvocationRowsHold logicalStart rowStart freshStart blockCount
      cellCount valueStride valueSourceStart env)
    (selected : CompactRowInvocation)
    (member : selected ∈ familyInvocations logicalStart rowStart freshStart
      blockCount cellCount valueStride valueSourceStart) :
    CompactRowInvocationHolds (Data.circuitPackage ()) selected env := by
  unfold familyInvocations at member
  rcases List.mem_flatMap.mp member with
    ⟨sourceValue, sourceMember, indexedMember⟩
  have sourceLt : sourceValue < sourceCount := List.mem_range.mp sourceMember
  rcases List.mem_ofFn.mp indexedMember with ⟨index, rfl⟩
  let source : Fin sourceCount := ⟨sourceValue, sourceLt⟩
  let coordinates := CombinationStep.coordinates index
  have instantiated := rows source index
  have selection :=
    NightstreamFPrime.Export.Stage1.Package.circuitPackage_piRlcCombinationTemplateSelection
      source coordinates.2.1
  unfold CompactRowInvocationHolds
  have selectedIndex :
      (invocation logicalStart rowStart freshStart blockCount cellCount
        valueStride sourceValue coordinates.1.val coordinates.2.1.val
          coordinates.2.2.val valueSourceStart).templateIndex =
        PiRLCCombinationTemplates.templateIndex sourceValue
          coordinates.2.1.val := by
    rfl
  rw [selectedIndex, selection]
  dsimp only
  rw [← CompactRows.instantiateRows_eq_package]
  exact instantiated

/-- The remapped production packets satisfy every canonical compact
combination invocation in the package-facing form. -/
theorem remappedPackets_imply_packageCombinationInvocations
    (env : Env)
    (packets : PiRLCPackageCompleteness.RemappedPacketRowsHold env) :
    ∀ selected ∈ PiRLCCombinationInvocations.invocations,
      CompactRowInvocationHolds (Data.circuitPackage ()) selected env := by
  have families := remappedPackets_imply_familyInvocationRows env packets
  intro selected member
  simp only [PiRLCCombinationInvocations.invocations,
    List.mem_append] at member
  rcases member with firstThree | evalAMember
  rcases firstThree with firstTwo | evalKMember
  rcases firstTwo with commitmentMember | publicInputMember
  · exact familyRows_imply_packageInvocations
      PiRLCStarts.commitmentLogicalStart PiRLCStarts.commitmentRowStart
      PiRLCStarts.commitmentFreshStart 1 commitmentValueSourceStart env
      families.commitment selected commitmentMember
  · exact familyRows_imply_packageInvocations
      PiRLCStarts.publicInputLogicalStart PiRLCStarts.publicInputRowStart
      PiRLCStarts.publicInputFreshStart 1 publicInputValueSourceStart env
      families.publicInput selected publicInputMember
  · exact familyRows_imply_packageInvocations
      PiRLCStarts.evalKLogicalStart PiRLCStarts.evalKRowStart
      PiRLCStarts.evalKFreshStart 2 evalKValueSourceStart env families.evalK
      selected evalKMember
  · exact familyRows_imply_packageInvocations
      PiRLCStarts.evalALogicalStart PiRLCStarts.evalARowStart
      PiRLCStarts.evalAFreshStart 2 evalAValueSourceStart env families.evalA
      selected evalAMember

end NightstreamFPrime.Export.Stage1.PiRLCCombinationCompleteness
