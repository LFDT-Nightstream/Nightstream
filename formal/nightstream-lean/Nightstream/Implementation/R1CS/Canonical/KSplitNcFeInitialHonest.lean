import Nightstream.Implementation.R1CS.Canonical.KBooleanMleSequentialHonest
import Nightstream.Implementation.R1CS.Canonical.KCompositeRowSupport
import Nightstream.Implementation.R1CS.Canonical.KSplitNcFeInitial

/-!
Contract: constructive completeness for the selected Split-NC FE-initial
arithmetic.

Owns: the exact MLE-batch traversal, the following dense gamma fold, prefix
preservation, and the final two-row binding once the enclosing frozen
FE-initial equation supplies that equality.

The calculation witness itself assumes no endpoint value.  The final binding
is kept as a named boundary so the selected NIFS completeness theorem must
derive it from authoritative protocol data rather than hide it in an
acceptance record.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Canonical.KSplitNcFeInitialHonest

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal
open Nightstream.Implementation.R1CS.Canonical.KMul
open Nightstream.Implementation.R1CS.Canonical.KHornerHonest
open Nightstream.Implementation.R1CS.Canonical.KHornerSupport
open Nightstream.Implementation.R1CS.Canonical.KCompositeRowSupport
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc

private theorem flatMap_congr_local
    {α β : Type} (items : List α) (left right : α → List β)
    (each : ∀ item ∈ items, left item = right item) :
    items.flatMap left = items.flatMap right := by
  induction items with
  | nil => rfl
  | cons item rest inductionHypothesis =>
      rw [List.flatMap_cons, List.flatMap_cons, each item (by simp)]
      congr 1
      exact inductionHypothesis
        (fun value member => each value (by simp [member]))

private theorem range_flatMap_chunks
    (start chunk blockCount : Nat) :
    (List.range blockCount).flatMap (fun block =>
        List.range' (start + chunk * block) chunk) =
      List.range' start (blockCount * chunk) := by
  induction blockCount with
  | zero => simp
  | succ blockCount inductionHypothesis =>
      rw [List.range_succ, List.flatMap_append]
      simp only [List.flatMap_cons, List.flatMap_nil, List.append_nil]
      rw [inductionHypothesis]
      have blockStart :
          start + chunk * blockCount = start + blockCount * chunk := by
        rw [Nat.mul_comm chunk blockCount]
      have totalWidth :
          (blockCount + 1) * chunk = blockCount * chunk + chunk := by
        rw [Nat.add_mul]
        simp
      rw [blockStart, totalWidth]
      exact List.range'_append_1

private theorem running_positions
    {shape : SemanticShape} (matrix : Fin shape.matrixCount) :
    (canonicalFinIndices shape.runningCount).map (fun running =>
        KSplitNcFeInitial.ordinal (matrix, running)) =
      List.range' (shape.runningCount * matrix.val)
        shape.runningCount := by
  rw [List.range'_eq_map_range, ← canonicalFinIndices_values,
    List.map_map]
  apply List.map_congr_left
  intro running _
  unfold KSplitNcFeInitial.ordinal
  simp only [Function.comp_apply]
  rw [Nat.mul_comm matrix.val shape.runningCount]

/-- The nested matrix/running traversal is exactly ordinal order. -/
theorem indices_positions (shape : SemanticShape) :
    (KSplitNcFeInitial.indices shape).map KSplitNcFeInitial.ordinal =
      List.range' 0 (KSplitNcFeInitial.indices shape).length := by
  rw [KSplitNcFeInitial.indices_length]
  unfold KSplitNcFeInitial.indices
  rw [List.map_flatMap]
  calc
    _ = (canonicalFinIndices shape.matrixCount).flatMap fun matrix =>
          List.range' (shape.runningCount * matrix.val)
            shape.runningCount := by
      apply flatMap_congr_local
      intro matrix _
      rw [List.map_map]
      exact running_positions matrix
    _ = ((canonicalFinIndices shape.matrixCount).map
          (fun matrix => matrix.val)).flatMap fun matrix =>
          List.range' (0 + shape.runningCount * matrix)
            shape.runningCount := by
      simpa only [Nat.zero_add] using
        (List.flatMap_map
          (fun matrix : Fin shape.matrixCount => matrix.val)
          (fun matrix =>
            List.range' (0 + shape.runningCount * matrix)
              shape.runningCount)
          (canonicalFinIndices shape.matrixCount)).symm
    _ = (List.range shape.matrixCount).flatMap fun matrix =>
          List.range' (0 + shape.runningCount * matrix)
            shape.runningCount := by
      rw [canonicalFinIndices_values]
    _ = List.range' 0
          (shape.matrixCount * shape.runningCount) :=
      range_flatMap_chunks 0 shape.runningCount shape.matrixCount

private theorem mleRows_eq_rowsFrom
    {shape : SemanticShape} {domain : FlatNcDomain}
    (input : KSplitNcFeInitial.Input shape domain) :
    KSplitNcFeInitial.mleRows input =
      KBooleanMleSequentialHonest.rowsFrom
        (KSplitNcFeInitial.indices shape)
        (KSplitNcFeInitial.table input)
        (fun _ => KSplitNcFeInitial.alphaCoordinates input)
        input.frameBase 0 := by
  unfold KSplitNcFeInitial.mleRows
  symm
  simpa [KBooleanMleSequentialHonest.blockWidth,
    KSplitNcFeInitial.mleBase, KSplitNcFeInitial.rowsPerMle] using
    KBooleanMleSequentialHonest.rowsFrom_eq_flatMap
      (KSplitNcFeInitial.indices shape)
      (KSplitNcFeInitial.table input)
      (fun _ => KSplitNcFeInitial.alphaCoordinates input)
      KSplitNcFeInitial.ordinal input.frameBase 0
      (indices_positions shape)

private theorem tables_below
    {shape : SemanticShape} {domain : FlatNcDomain}
    (input : KSplitNcFeInitial.Input shape domain)
    (claimsBelow :
      ∀ running matrix lane,
        CarriedBelow
          (input.claimedYRing running matrix lane) input.frameBase) :
    ∀ index,
      KBooleanMleSupport.TableBelowBase
        (KSplitNcFeInitial.table input index) input.frameBase := by
  intro index
  unfold KSplitNcFeInitial.table
  apply paddedTable_below
  intro lane
  exact claimsBelow index.2 index.1 lane

private theorem coordinates_below
    {shape : SemanticShape} {domain : FlatNcDomain}
    (input : KSplitNcFeInitial.Input shape domain)
    (alphaBelow :
      ∀ coordinate, CarriedBelow (input.alpha coordinate) input.frameBase) :
    KBooleanMleSupport.CoordinatesBelowBase
      (KSplitNcFeInitial.alphaCoordinates input) input.frameBase := by
  unfold KSplitNcFeInitial.alphaCoordinates
  exact coordinates_below_ofFn input.alpha alphaBelow

def afterMle
    {shape : SemanticShape} {domain : FlatNcDomain}
    (input : KSplitNcFeInitial.Input shape domain)
    (assignment : Nat → Nat) : Nat → Nat :=
  KBooleanMleSequentialHonest.witnessFrom assignment
    (KSplitNcFeInitial.indices shape)
    (KSplitNcFeInitial.table input)
    (fun _ => KSplitNcFeInitial.alphaCoordinates input)
    input.frameBase 0

def witness
    {shape : SemanticShape} {domain : FlatNcDomain}
    (input : KSplitNcFeInitial.Input shape domain)
    (assignment : Nat → Nat) : Nat → Nat :=
  KHornerHonest.hornerWitness (afterMle input assignment)
    input.gamma (KSplitNcFeInitial.hornerBase input)
    (KSplitNcFeInitial.coefficients input) 0

private theorem ordinal_lt
    {shape : SemanticShape}
    (index : KSplitNcFeInitial.Index shape) :
    KSplitNcFeInitial.ordinal index <
      shape.matrixCount * shape.runningCount := by
  unfold KSplitNcFeInitial.ordinal
  calc
    index.1.val * shape.runningCount + index.2.val <
        index.1.val * shape.runningCount + shape.runningCount :=
      Nat.add_lt_add_left index.2.isLt _
    _ = (index.1.val + 1) * shape.runningCount := by
      rw [Nat.add_mul, Nat.one_mul]
    _ ≤ shape.matrixCount * shape.runningCount :=
      Nat.mul_le_mul_right shape.runningCount
        (Nat.succ_le_iff.mpr index.1.isLt)

private theorem mleOutput_below_hornerBase
    {shape : SemanticShape} {domain : FlatNcDomain}
    (input : KSplitNcFeInitial.Input shape domain)
    (alphaBelow :
      ∀ coordinate, CarriedBelow (input.alpha coordinate) input.frameBase)
    (claimsBelow :
      ∀ running matrix lane,
        CarriedBelow
          (input.claimedYRing running matrix lane) input.frameBase)
    (index : KSplitNcFeInitial.Index shape) :
    CarriedBelow (KSplitNcFeInitial.mleOutput input index)
      (KSplitNcFeInitial.hornerBase input) := by
  apply boolean_output_below
  · exact KBooleanMleSequentialHonest.tableBelow_mono
      (KSplitNcFeInitial.table input index)
      (tables_below input claimsBelow index)
      (by unfold KSplitNcFeInitial.mleBase; omega)
  · exact KBooleanMleSequentialHonest.coordinatesBelow_mono
      (KSplitNcFeInitial.alphaCoordinates input)
      (coordinates_below input alphaBelow)
      (by unfold KSplitNcFeInitial.mleBase; omega)
  · unfold KSplitNcFeInitial.mleBase
      KSplitNcFeInitial.hornerBase
      KSplitNcFeInitial.rowsPerMle
    have bound :
        KSplitNcFeInitial.ordinal index + 1 ≤
          shape.matrixCount * shape.runningCount :=
      Nat.succ_le_iff.mpr (ordinal_lt index)
    rw [Nat.add_assoc]
    apply Nat.add_le_add_left
    calc
      3 * KBooleanMle.frameCount domain.laneVariables *
            KSplitNcFeInitial.ordinal index +
          3 * KBooleanMle.frameCount domain.laneVariables =
        (3 * KBooleanMle.frameCount domain.laneVariables) *
          (KSplitNcFeInitial.ordinal index + 1) := by
            rw [Nat.mul_add, Nat.mul_one]
      _ ≤
          (3 * KBooleanMle.frameCount domain.laneVariables) *
            (shape.matrixCount * shape.runningCount) :=
        Nat.mul_le_mul_left _ bound

private theorem coefficients_below_hornerBase
    {shape : SemanticShape} {domain : FlatNcDomain}
    (input : KSplitNcFeInitial.Input shape domain)
    (alphaBelow :
      ∀ coordinate, CarriedBelow (input.alpha coordinate) input.frameBase)
    (claimsBelow :
      ∀ running matrix lane,
        CarriedBelow
          (input.claimedYRing running matrix lane) input.frameBase) :
    ∀ coefficient ∈ KSplitNcFeInitial.coefficients input,
      CarriedBelow coefficient (KSplitNcFeInitial.hornerBase input) := by
  intro coefficient member
  rcases List.mem_append.1 member with inInitialZeros | inMatrices
  · have same : coefficient = KLinear.zeroCarried :=
      List.eq_of_mem_replicate inInitialZeros
    subst coefficient
    exact zero_below _
  · rcases List.mem_flatMap.1 inMatrices with
      ⟨matrix, _, inBlock⟩
    rcases List.mem_append.1 inBlock with inFreshZeros | inRunning
    · have same : coefficient = KLinear.zeroCarried :=
        List.eq_of_mem_replicate inFreshZeros
      subst coefficient
      exact zero_below _
    · rcases List.mem_map.1 inRunning with ⟨running, _, rfl⟩
      exact mleOutput_below_hornerBase input alphaBelow claimsBelow
        (matrix, running)

private theorem mleRows_honest
    {shape : SemanticShape} {domain : FlatNcDomain}
    (input : KSplitNcFeInitial.Input shape domain)
    (assignment : Nat → Nat)
    (positive : 0 < input.frameBase)
    (alphaBelow :
      ∀ coordinate, CarriedBelow (input.alpha coordinate) input.frameBase)
    (claimsBelow :
      ∀ running matrix lane,
        CarriedBelow
          (input.claimedYRing running matrix lane) input.frameBase) :
    Satisfies (KSplitNcFeInitial.mleRows input)
      (afterMle input assignment) := by
  rw [mleRows_eq_rowsFrom]
  exact KBooleanMleSequentialHonest.rowsFrom_honest assignment
    (KSplitNcFeInitial.table input)
    (fun _ => KSplitNcFeInitial.alphaCoordinates input)
    positive
    (tables_below input claimsBelow)
    (fun _ => coordinates_below input alphaBelow)
    (KSplitNcFeInitial.indices shape) 0

private theorem mleRows_below_hornerBase
    {shape : SemanticShape} {domain : FlatNcDomain}
    (input : KSplitNcFeInitial.Input shape domain)
    (alphaBelow :
      ∀ coordinate, CarriedBelow (input.alpha coordinate) input.frameBase)
    (claimsBelow :
      ∀ running matrix lane,
        CarriedBelow
          (input.claimedYRing running matrix lane) input.frameBase) :
    RowsBelow (KSplitNcFeInitial.mleRows input)
      (KSplitNcFeInitial.hornerBase input) := by
  rw [mleRows_eq_rowsFrom]
  intro row member column mentioned
  have bounded :=
    KBooleanMleSequentialHonest.rowsFrom_below_end
      (KSplitNcFeInitial.table input)
      (fun _ => KSplitNcFeInitial.alphaCoordinates input)
      (tables_below input claimsBelow)
      (fun _ => coordinates_below input alphaBelow)
      (KSplitNcFeInitial.indices shape) 0 row member column mentioned
  simpa [KBooleanMleSequentialHonest.blockWidth,
    KSplitNcFeInitial.indices_length, KSplitNcFeInitial.hornerBase,
    KSplitNcFeInitial.rowsPerMle] using bounded

private theorem computedRows_honest
    {shape : SemanticShape} {domain : FlatNcDomain}
    (input : KSplitNcFeInitial.Input shape domain)
    (assignment : Nat → Nat)
    (positive : 0 < input.frameBase)
    (gammaBelow : CarriedBelow input.gamma input.frameBase)
    (alphaBelow :
      ∀ coordinate, CarriedBelow (input.alpha coordinate) input.frameBase)
    (claimsBelow :
      ∀ running matrix lane,
        CarriedBelow
          (input.claimedYRing running matrix lane) input.frameBase) :
    Satisfies
      (KSplitNcFeInitial.mleRows input ++
        KSplitNcFeInitial.hornerRows input)
      (witness input assignment) := by
  have mleSatisfied :=
    mleRows_honest input assignment positive alphaBelow claimsBelow
  have mlePreserved :
      Satisfies (KSplitNcFeInitial.mleRows input)
        (witness input assignment) := by
    apply satisfies_extend _ (afterMle input assignment)
      (witness input assignment)
    · intro row member column mentioned
      symm
      apply KHornerHonest.hornerWitness_off_block
      exact mleRows_below_hornerBase input alphaBelow claimsBelow
        row member column mentioned
    · exact mleSatisfied
  have hornerSatisfied :
      Satisfies (KSplitNcFeInitial.hornerRows input)
        (witness input assignment) := by
    exact KHornerHonest.hornerWitness_satisfies
      (afterMle input assignment) input.gamma
      (KSplitNcFeInitial.hornerBase input)
      (carried_mono gammaBelow (by
        unfold KSplitNcFeInitial.hornerBase
        omega)).1
      (carried_mono gammaBelow (by
        unfold KSplitNcFeInitial.hornerBase
        omega)).2
      (KSplitNcFeInitial.coefficients input) 0
      (fun coefficient member =>
        coefficients_below_hornerBase input alphaBelow claimsBelow
          coefficient member)
  intro row member
  exact (List.mem_append.1 member).elim
    (mlePreserved row) (hornerSatisfied row)

theorem witness_off_block
    {shape : SemanticShape} {domain : FlatNcDomain}
    (input : KSplitNcFeInitial.Input shape domain)
    (assignment : Nat → Nat)
    (column : Nat) (below : column < input.frameBase) :
    witness input assignment column = assignment column := by
  unfold witness
  rw [KHornerHonest.hornerWitness_off_block
      (afterMle input assignment) input.gamma
      (KSplitNcFeInitial.hornerBase input)
      (KSplitNcFeInitial.coefficients input) 0 column
      (by unfold KSplitNcFeInitial.hornerBase; omega)]
  exact KBooleanMleSequentialHonest.witnessFrom_off_before assignment
    (KSplitNcFeInitial.table input)
    (fun _ => KSplitNcFeInitial.alphaCoordinates input)
    (KSplitNcFeInitial.indices shape) input.frameBase 0 column
    (by simpa using below)

/-- The canonical two-column materialization immediately following the
calculation block.  The rows bind to these columns but do not allocate any
additional multiplication frame. -/
def canonicalTargetBase
    {shape : SemanticShape} {domain : FlatNcDomain}
    (input : KSplitNcFeInitial.Input shape domain) : Nat :=
  input.frameBase + KSplitNcFeInitial.allocationWidth input

def canonicalTarget
    {shape : SemanticShape} {domain : FlatNcDomain}
    (input : KSplitNcFeInitial.Input shape domain) : Carried where
  low := [(canonicalTargetBase input, 1)]
  high := [(canonicalTargetBase input + 1, 1)]

/-- Complete the calculation witness by materializing its two field
coordinates in the canonical target columns. -/
def materializedWitness
    {shape : SemanticShape} {domain : FlatNcDomain}
    (input : KSplitNcFeInitial.Input shape domain)
    (assignment : Nat → Nat) : Nat → Nat :=
  let computed := witness input assignment
  fun column =>
    if column = canonicalTargetBase input then
      lcEval computed (KSplitNcFeInitial.evaluated input).low
    else if column = canonicalTargetBase input + 1 then
      lcEval computed (KSplitNcFeInitial.evaluated input).high
    else
      computed column

theorem materializedWitness_off_target
    {shape : SemanticShape} {domain : FlatNcDomain}
    (input : KSplitNcFeInitial.Input shape domain)
    (assignment : Nat → Nat)
    (column : Nat) (below : column < canonicalTargetBase input) :
    materializedWitness input assignment column =
      witness input assignment column := by
  unfold materializedWitness
  rw [if_neg (by omega), if_neg (by omega)]

theorem materializedWitness_off_block
    {shape : SemanticShape} {domain : FlatNcDomain}
    (input : KSplitNcFeInitial.Input shape domain)
    (assignment : Nat → Nat)
    (column : Nat) (below : column < input.frameBase) :
    materializedWitness input assignment column = assignment column := by
  rw [materializedWitness_off_target input assignment column (by
      unfold canonicalTargetBase
      omega),
    witness_off_block input assignment column below]

private theorem frames_end_at_target
    {shape : SemanticShape} {domain : FlatNcDomain}
    (input : KSplitNcFeInitial.Input shape domain) :
    KSplitNcFeInitial.hornerBase input +
        3 * ((KSplitNcFeInitial.coefficients input).length - 1) ≤
      canonicalTargetBase input := by
  rw [KSplitNcFeInitial.coefficients_length]
  unfold canonicalTargetBase KSplitNcFeInitial.hornerBase
    KSplitNcFeInitial.allocationWidth
  omega

private theorem evaluated_below_target
    {shape : SemanticShape} {domain : FlatNcDomain}
    (input : KSplitNcFeInitial.Input shape domain)
    (alphaBelow :
      ∀ coordinate, CarriedBelow (input.alpha coordinate) input.frameBase)
    (claimsBelow :
      ∀ running matrix lane,
        CarriedBelow
          (input.claimedYRing running matrix lane) input.frameBase) :
    CarriedBelow (KSplitNcFeInitial.evaluated input)
      (canonicalTargetBase input) := by
  unfold KSplitNcFeInitial.evaluated
  apply horner_output_below
  · intro coefficient member
    exact carried_mono
      (coefficients_below_hornerBase input alphaBelow claimsBelow
        coefficient member)
      (by
        unfold canonicalTargetBase KSplitNcFeInitial.hornerBase
          KSplitNcFeInitial.allocationWidth
        omega)
  · simpa using frames_end_at_target input

private theorem computedRows_below_target
    {shape : SemanticShape} {domain : FlatNcDomain}
    (input : KSplitNcFeInitial.Input shape domain)
    (gammaBelow : CarriedBelow input.gamma input.frameBase)
    (alphaBelow :
      ∀ coordinate, CarriedBelow (input.alpha coordinate) input.frameBase)
    (claimsBelow :
      ∀ running matrix lane,
        CarriedBelow
          (input.claimedYRing running matrix lane) input.frameBase) :
    RowsBelow
      (KSplitNcFeInitial.mleRows input ++
        KSplitNcFeInitial.hornerRows input)
      (canonicalTargetBase input) := by
  have hornerBaseOrdered :
      KSplitNcFeInitial.hornerBase input ≤ canonicalTargetBase input := by
    unfold canonicalTargetBase KSplitNcFeInitial.hornerBase
      KSplitNcFeInitial.allocationWidth
    omega
  have mleBelow :
      RowsBelow (KSplitNcFeInitial.mleRows input)
        (canonicalTargetBase input) := by
    intro row member column mentioned
    exact Nat.lt_of_lt_of_le
      (mleRows_below_hornerBase input alphaBelow claimsBelow
        row member column mentioned)
      hornerBaseOrdered
  have hornerBelow :
      RowsBelow (KSplitNcFeInitial.hornerRows input)
        (canonicalTargetBase input) := by
    unfold KSplitNcFeInitial.hornerRows
    apply horner_rows_below
    · exact carried_mono gammaBelow
        (Nat.le_trans
          (Nat.le_add_right input.frameBase
            (KSplitNcFeInitial.allocationWidth input))
          (Nat.le_refl _))
    · intro coefficient member
      exact carried_mono
        (coefficients_below_hornerBase input alphaBelow claimsBelow
          coefficient member)
        hornerBaseOrdered
    · simpa using frames_end_at_target input
  intro row member column mentioned
  exact (List.mem_append.1 member).elim
    (fun inMle => mleBelow row inMle column mentioned)
    (fun inHorner => hornerBelow row inHorner column mentioned)

private theorem materialized_preserves_evaluated
    {shape : SemanticShape} {domain : FlatNcDomain}
    (input : KSplitNcFeInitial.Input shape domain)
    (assignment : Nat → Nat)
    (alphaBelow :
      ∀ coordinate, CarriedBelow (input.alpha coordinate) input.frameBase)
    (claimsBelow :
      ∀ running matrix lane,
        CarriedBelow
          (input.claimedYRing running matrix lane) input.frameBase) :
    lcEval (materializedWitness input assignment)
        (KSplitNcFeInitial.evaluated input).low =
        lcEval (witness input assignment)
          (KSplitNcFeInitial.evaluated input).low ∧
      lcEval (materializedWitness input assignment)
        (KSplitNcFeInitial.evaluated input).high =
        lcEval (witness input assignment)
          (KSplitNcFeInitial.evaluated input).high := by
  have below := evaluated_below_target input alphaBelow claimsBelow
  constructor
  · exact KMulHonest.lcEval_congr _ _
      (KSplitNcFeInitial.evaluated input).low
      (fun column mentioned =>
        materializedWitness_off_target input assignment column
          (below.1 column mentioned))
  · exact KMulHonest.lcEval_congr _ _
      (KSplitNcFeInitial.evaluated input).high
      (fun column mentioned =>
        materializedWitness_off_target input assignment column
          (below.2 column mentioned))

private theorem target_values
    {shape : SemanticShape} {domain : FlatNcDomain}
    (input : KSplitNcFeInitial.Input shape domain)
    (assignment : Nat → Nat) :
    lcEval (materializedWitness input assignment)
        (canonicalTarget input).low =
        lcEval (witness input assignment)
          (KSplitNcFeInitial.evaluated input).low ∧
      lcEval (materializedWitness input assignment)
        (canonicalTarget input).high =
        lcEval (witness input assignment)
          (KSplitNcFeInitial.evaluated input).high := by
  constructor
  · unfold canonicalTarget
    rw [KMul.lcEval_singleton_col]
    unfold materializedWitness
    rw [if_pos rfl, Nat.mod_eq_of_lt]
    unfold lcEval
    exact Nat.mod_lt _ (by decide)
  · unfold canonicalTarget
    rw [KMul.lcEval_singleton_col]
    unfold materializedWitness
    rw [if_neg (by omega), if_pos rfl, Nat.mod_eq_of_lt]
    unfold lcEval
    exact Nat.mod_lt _ (by decide)

/-- The remaining two-row boundary after the calculation witness.  A selected
protocol completeness theorem must construct this from its frozen FE-initial
equation. -/
structure Binding
    {shape : SemanticShape} {domain : FlatNcDomain}
    (input : KSplitNcFeInitial.Input shape domain)
    (assignment : Nat → Nat) : Prop where
  low :
    lcEval (witness input assignment)
        (KSplitNcFeInitial.evaluated input).low =
      lcEval (witness input assignment) input.initial.low
  high :
    lcEval (witness input assignment)
        (KSplitNcFeInitial.evaluated input).high =
      lcEval (witness input assignment) input.initial.high

theorem rows_honest_of_binding
    {shape : SemanticShape} {domain : FlatNcDomain}
    (input : KSplitNcFeInitial.Input shape domain)
    (assignment : Nat → Nat)
    (positive : 0 < input.frameBase)
    (constantWire : assignment 0 = 1)
    (gammaBelow : CarriedBelow input.gamma input.frameBase)
    (alphaBelow :
      ∀ coordinate, CarriedBelow (input.alpha coordinate) input.frameBase)
    (claimsBelow :
      ∀ running matrix lane,
        CarriedBelow
          (input.claimedYRing running matrix lane) input.frameBase)
    (binding : Binding input assignment) :
    Satisfies (KSplitNcFeInitial.rows input) (witness input assignment) := by
  have computed :=
    computedRows_honest input assignment positive gammaBelow alphaBelow
      claimsBelow
  have one :
      witness input assignment 0 = 1 := by
    rw [witness_off_block input assignment 0 positive]
    exact constantWire
  have equality :=
    KEquality.rows_complete (witness input assignment)
      (KSplitNcFeInitial.evaluated input) input.initial one
      binding.low binding.high
  intro row member
  unfold KSplitNcFeInitial.rows at member
  rcases List.mem_append.1 member with inComputed | inEquality
  · exact computed row inComputed
  · exact equality row inEquality

/-- Honest completeness when the enclosing allocator uses the canonical
two-column target.  Unlike `rows_honest_of_binding`, no endpoint equation is
supplied: the witness materializes the computed value itself. -/
theorem rows_honest_canonicalTarget
    {shape : SemanticShape} {domain : FlatNcDomain}
    (input : KSplitNcFeInitial.Input shape domain)
    (assignment : Nat → Nat)
    (positive : 0 < input.frameBase)
    (constantWire : assignment 0 = 1)
    (gammaBelow : CarriedBelow input.gamma input.frameBase)
    (alphaBelow :
      ∀ coordinate, CarriedBelow (input.alpha coordinate) input.frameBase)
    (claimsBelow :
      ∀ running matrix lane,
        CarriedBelow
          (input.claimedYRing running matrix lane) input.frameBase)
    (target : input.initial = canonicalTarget input) :
    Satisfies (KSplitNcFeInitial.rows input)
      (materializedWitness input assignment) := by
  have computed :=
    computedRows_honest input assignment positive gammaBelow alphaBelow
      claimsBelow
  have computedPreserved :
      Satisfies
        (KSplitNcFeInitial.mleRows input ++
          KSplitNcFeInitial.hornerRows input)
        (materializedWitness input assignment) := by
    apply satisfies_extend _ (witness input assignment)
      (materializedWitness input assignment)
    · intro row member column mentioned
      exact
        (materializedWitness_off_target input assignment column
          (computedRows_below_target input gammaBelow alphaBelow claimsBelow
            row member column mentioned)).symm
    · exact computed
  have one :
      materializedWitness input assignment 0 = 1 := by
    rw [materializedWitness_off_target input assignment 0 (by
        unfold canonicalTargetBase
        omega),
      witness_off_block input assignment 0 positive]
    exact constantWire
  have preserved :=
    materialized_preserves_evaluated input assignment alphaBelow claimsBelow
  have stored := target_values input assignment
  have equality :
      Satisfies
        (KEquality.rows
          (KSplitNcFeInitial.evaluated input) input.initial)
        (materializedWitness input assignment) := by
    rw [target]
    exact KEquality.rows_complete
      (materializedWitness input assignment)
      (KSplitNcFeInitial.evaluated input) (canonicalTarget input) one
      (preserved.1.trans stored.1.symm)
      (preserved.2.trans stored.2.symm)
  intro row member
  unfold KSplitNcFeInitial.rows at member
  exact (List.mem_append.1 member).elim
    (computedPreserved row) (equality row)

end Nightstream.Implementation.R1CS.Canonical.KSplitNcFeInitialHonest
