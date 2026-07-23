import Nightstream.Implementation.R1CS.Correspondence.Gadgets.TerminalRawOldBlockProjectionCompiler
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.NumericBooleanDomain

/-!
Semantic bridge for the compact-prefix tensor emitted by the direct terminal
raw-old-block projection compiler.

This leaf is deliberately independent of generated profile data.  It proves
that the all-lows-then-live-highs schedule is the prefix of the ordinary
little-endian Boolean tensor, including a partially populated last layer.
The artifact-facing compiler remains responsible for physical rows and
columns.
-/

namespace Nightstream.Implementation.R1CS.TerminalRawOldBlockProjectionTensorPrefix

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.ProjectionProgram
open Nightstream.Implementation.R1CS.TerminalRawOldBlockProjectionCompiler
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

/-- Exact extension-field operations used by the compact tensor compiler.
Downstream row refinements use this definition directly instead of rebuilding
a merely fieldwise-equal operation record. -/
def projectionTensorOps : InterpolationOps K where
  zero := K.zero
  one := K.one
  add := K.add
  mul := K.mul
  neg := fun value => K.sub K.zero value

def oldBlockValues (layout : Layout) (assignment : Nat -> Nat) : List K :=
  List.ofFn fun bit : Fin layout.blockVariables =>
    (layout.oldBlock bit).value assignment

def oldBlockPoint (layout : Layout) (assignment : Nat -> Nat) :
    CubePoint K layout.blockVariables where
  coordinates := oldBlockValues layout assignment
  dimension := by simp [oldBlockValues]

/-- Append one most-significant Boolean-tensor coordinate. -/
def appendPoint {variables : Nat}
    (point : CubePoint K variables) (final : K) :
    CubePoint K (variables + 1) where
  coordinates := point.coordinates ++ [final]
  dimension := by simp [point.dimension]

private def prefixWeight (coordinates : List K) (round index : Nat) : K :=
  (List.range round).foldl
    (fun accumulated bit =>
      K.mul accumulated
        (if Nat.testBit index bit then
          coordinates.getD bit K.zero
        else
          K.sub K.one (coordinates.getD bit K.zero)))
    K.one

private theorem foldl_congr_on
    {Element State : Type} (elements : List Element)
    (left right : State -> Element -> State) (initial : State)
    (equal : forall state element, element ∈ elements ->
      left state element = right state element) :
    elements.foldl left initial = elements.foldl right initial := by
  induction elements generalizing initial with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      rw [List.foldl_cons, List.foldl_cons]
      rw [equal initial head (by simp)]
      apply inductionHypothesis
      intro state element member
      exact equal state element (by simp [member])

private theorem filterMap_congr_on
    {Element Value : Type} (elements : List Element)
    (left right : Element -> Option Value)
    (equal : forall element, element ∈ elements -> left element = right element) :
    elements.filterMap left = elements.filterMap right := by
  induction elements with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      simp only [List.filterMap_cons]
      rw [equal head (by simp)]
      rw [inductionHypothesis (fun element member =>
        equal element (by simp [member]))]

private theorem testBit_add_twoPow_low
    {width mask bit : Nat}
    (maskLt : mask < 2 ^ width) (bitLt : bit < width) :
    Nat.testBit (mask + 2 ^ width) bit = Nat.testBit mask bit := by
  have modulo : (mask + 2 ^ width) % 2 ^ width = mask := by
    calc
      (mask + 2 ^ width) % 2 ^ width = mask % 2 ^ width := by
        simpa using Nat.add_mul_mod_self_left mask (2 ^ width) 1
      _ = mask := Nat.mod_eq_of_lt maskLt
  have projected := Nat.testBit_mod_two_pow
    (mask + 2 ^ width) width bit
  rw [modulo] at projected
  simp [bitLt] at projected
  exact projected.symm

private theorem testBit_add_twoPow_self
    {width mask : Nat} (maskLt : mask < 2 ^ width) :
    Nat.testBit (mask + 2 ^ width) width = true := by
  unfold Nat.testBit
  rw [Nat.shiftRight_eq_div_pow]
  have powerPositive : 0 < 2 ^ width := Nat.two_pow_pos width
  have quotient : (mask + 2 ^ width) / 2 ^ width = 1 := by
    calc
      (mask + 2 ^ width) / 2 ^ width = mask / 2 ^ width + 1 := by
        simpa using Nat.add_mul_div_right mask 1 powerPositive
      _ = 1 := by rw [Nat.div_eq_of_lt maskLt]
  rw [quotient]
  decide

private theorem testBit_eq_false_of_lt_twoPow
    {width mask : Nat} (maskLt : mask < 2 ^ width) :
    Nat.testBit mask width = false := by
  unfold Nat.testBit
  rw [Nat.shiftRight_eq_div_pow, Nat.div_eq_of_lt maskLt]
  decide

private theorem prefixWeight_succ_low
    (coordinates : List K) {round index : Nat}
    (indexLt : index < 2 ^ round) :
    prefixWeight coordinates (round + 1) index =
      K.mul (prefixWeight coordinates round index)
        (K.sub K.one (coordinates.getD round K.zero)) := by
  rw [prefixWeight, List.range_succ, List.foldl_append]
  simp only [List.foldl_cons, List.foldl_nil]
  rw [testBit_eq_false_of_lt_twoPow indexLt]
  rfl

/-- Radix-weighted projection using only the first `round` Boolean-tensor
coordinates.  This is the narrow public interface to the pointwise prefix
weight: callers supply an explicit finite index set and its values, while the
compact tensor implementation remains private to this leaf. -/
def weightedPrefixProjection (coordinates : List K) (round : Nat)
    (indices : List Nat) (value : Nat -> K) : K :=
  indices.foldr
    (fun index suffix =>
      K.add (K.mul (value index) (prefixWeight coordinates round index))
        suffix)
    K.zero

/-- If every projected index is in the all-low half of the next Boolean
tensor round, its new coordinate is common to every summand and can be
factored out once after the weighted fold. -/
theorem weightedPrefixProjection_succ_allLow
    (coordinates : List K) (round : Nat) (indices : List Nat)
    (value : Nat -> K)
    (allLow : forall index, index ∈ indices -> index < 2 ^ round) :
    weightedPrefixProjection coordinates (round + 1) indices value =
      K.mul (weightedPrefixProjection coordinates round indices value)
        (K.sub K.one (coordinates.getD round K.zero)) := by
  revert allLow
  induction indices with
  | nil =>
      intro _
      simp [weightedPrefixProjection]
  | cons index tail inductionHypothesis =>
      intro allLow
      have indexLow : index < 2 ^ round := allLow index (by simp)
      have tailLow : forall current, current ∈ tail ->
          current < 2 ^ round := by
        intro current member
        exact allLow current (by simp [member])
      change
        K.add
            (K.mul (value index)
              (prefixWeight coordinates (round + 1) index))
            (weightedPrefixProjection coordinates (round + 1) tail value) =
          K.mul
            (K.add
              (K.mul (value index)
                (prefixWeight coordinates round index))
              (weightedPrefixProjection coordinates round tail value))
            (K.sub K.one (coordinates.getD round K.zero))
      rw [prefixWeight_succ_low coordinates indexLow,
        inductionHypothesis tailLow, K.add_mul, K.mul_assoc]

private theorem prefixWeight_add_twoPow_prefix
    (coordinates : List K) {round index : Nat}
    (indexLt : index < 2 ^ round) :
    prefixWeight coordinates round (index + 2 ^ round) =
      prefixWeight coordinates round index := by
  unfold prefixWeight
  apply foldl_congr_on
  intro accumulated bit bitMember
  have bitLt : bit < round := List.mem_range.mp bitMember
  rw [testBit_add_twoPow_low indexLt bitLt]

private theorem prefixWeight_succ_high
    (coordinates : List K) {round index : Nat}
    (indexLt : index < 2 ^ round) :
    prefixWeight coordinates (round + 1) (index + 2 ^ round) =
      K.mul (prefixWeight coordinates round index)
        (coordinates.getD round K.zero) := by
  rw [prefixWeight, List.range_succ, List.foldl_append]
  simp only [List.foldl_cons, List.foldl_nil]
  rw [testBit_add_twoPow_self indexLt]
  change K.mul
      (prefixWeight coordinates round (index + 2 ^ round))
      (coordinates.getD round K.zero) = _
  rw [prefixWeight_add_twoPow_prefix coordinates indexLt]

private theorem filterMap_range_lt
    {Value : Type} (value : Nat -> Value) (count cutoff : Nat) :
    (List.range count).filterMap
        (fun index => if index < cutoff then some (value index) else none) =
      (List.range (Nat.min count cutoff)).map value := by
  induction count with
  | zero => simp
  | succ count inductionHypothesis =>
      rw [List.range_succ, List.filterMap_append, inductionHypothesis]
      by_cases live : count < cutoff
      · have minimum : Nat.min (count + 1) cutoff = count + 1 :=
          Nat.min_eq_left (by omega)
        have priorMinimum : Nat.min count cutoff = count :=
          Nat.min_eq_left (by omega)
        simp [live, minimum, priorMinimum, List.range_succ]
      · have minimum : Nat.min (count + 1) cutoff = cutoff :=
          Nat.min_eq_right (by omega)
        have priorMinimum : Nat.min count cutoff = cutoff :=
          Nat.min_eq_right (by omega)
        simp [live, minimum, priorMinimum]

private theorem filterMap_range_highLive
    {Value : Type} (value : Nat -> Value) (count blocks round : Nat) :
    (List.range count).filterMap
        (fun parent =>
          if highLive blocks round parent then some (value parent) else none) =
      (List.range (Nat.min count (blocks - 2 ^ round))).map value := by
  calc
    _ = (List.range count).filterMap
        (fun parent =>
          if parent < blocks - 2 ^ round then some (value parent) else none) := by
      apply filterMap_congr_on
      intro parent member
      by_cases live : parent + 2 ^ round < blocks
      · have cutoff : parent < blocks - 2 ^ round :=
          Nat.lt_sub_of_add_lt live
        simp [highLive, live, cutoff]
      · have cutoff : ¬parent < blocks - 2 ^ round :=
          fun parentLt => live (Nat.add_lt_of_lt_sub parentLt)
        simp [highLive, live, cutoff]
    _ = _ := filterMap_range_lt value count (blocks - 2 ^ round)

private def expectedPrefix
    (blocks : Nat) (coordinates : List K) (round : Nat) : List K :=
  (List.range (Nat.min blocks (2 ^ round))).map
    (prefixWeight coordinates round)

private def semanticNext
    (blocks round : Nat) (parents : List K) (coordinate : K) : List K :=
  parents.map (fun parent =>
      K.mul parent (K.sub K.one coordinate)) ++
    (List.range parents.length).filterMap fun parent =>
      if highLive blocks round parent then
        some (K.mul (parents.getD parent K.zero) coordinate)
      else
        none

private theorem map_range_getD
    {Value Result : Type} (values : List Value) (fallback : Value)
    (mapValue : Value -> Result) :
    (List.range values.length).map
        (fun index => mapValue (values.getD index fallback)) =
      values.map mapValue := by
  apply List.ext_get
  · simp
  · intro index leftLt rightLt
    have inRange : index < values.length := by simpa using leftLt
    rw [List.get_eq_getElem, List.get_eq_getElem]
    simp only [List.getElem_map, List.getElem_range]
    exact congrArg mapValue (List.getElem_eq_getD fallback).symm

private theorem nextTensorValues_eq_semanticNext
    (blocks round : Nat) (parents : List K) (coordinate : K) :
    nextTensorValues blocks round coordinate parents =
      semanticNext blocks round parents coordinate := by
  unfold nextTensorValues semanticNext
  congr 1
  · exact map_range_getD parents K.one
      (fun parent => K.mul parent (K.sub K.one coordinate))
  · apply filterMap_congr_on
    intro index member
    have inRange : index < parents.length := List.mem_range.mp member
    by_cases live : highLive blocks round index
    · simp only [live, if_true]
      have oneEq : parents.getD index K.one =
          parents.get ⟨index, inRange⟩ :=
        (List.getElem_eq_getD K.one).symm
      have zeroEq : parents.getD index K.zero =
          parents.get ⟨index, inRange⟩ :=
        (List.getElem_eq_getD K.zero).symm
      rw [oneEq, zeroEq]
    · simp [live]

private theorem semanticNext_expectedPrefix
    (blocks : Nat) (coordinates : List K) (round : Nat) :
    semanticNext blocks round (expectedPrefix blocks coordinates round)
        (coordinates.getD round K.zero) =
      expectedPrefix blocks coordinates (round + 1) := by
  unfold semanticNext expectedPrefix
  let power := 2 ^ round
  by_cases capped : blocks <= power
  · have lowWidth : Nat.min blocks power = blocks := Nat.min_eq_left capped
    have nextWidth : Nat.min blocks (2 ^ (round + 1)) = blocks := by
      apply Nat.min_eq_left
      rw [Nat.pow_succ]
      omega
    have noHigh : blocks - power = 0 := Nat.sub_eq_zero_of_le capped
    rw [filterMap_range_highLive, List.length_map, List.length_range,
      lowWidth, noHigh]
    simp only [Nat.min_zero, List.range_zero, List.map_nil, List.append_nil]
    rw [nextWidth]
    rw [List.map_map]
    apply List.map_congr_left
    intro index member
    have indexLtBlocks : index < blocks := List.mem_range.mp member
    simp only [Function.comp_apply]
    exact (prefixWeight_succ_low coordinates (Nat.lt_of_lt_of_le
      indexLtBlocks capped)).symm
  · have powerLt : power < blocks := Nat.lt_of_not_ge capped
    have lowWidth : Nat.min blocks power = power := Nat.min_eq_right
      (Nat.le_of_lt powerLt)
    have expectedLength :
        ((List.range power).map (prefixWeight coordinates round)).length =
          power := by simp
    rw [lowWidth, filterMap_range_highLive, expectedLength]
    have remainingBound : power + Nat.min power (blocks - power) =
        Nat.min blocks (2 ^ (round + 1)) := by
      rw [Nat.pow_succ]
      by_cases fitsNext : blocks <= power * 2
      · have remainingFits : blocks - power <= power := by omega
        calc
          power + Nat.min power (blocks - power) =
              power + (blocks - power) := by
            exact congrArg (fun remainder => power + remainder)
              (Nat.min_eq_right remainingFits)
          _ = blocks := Nat.add_sub_of_le (Nat.le_of_lt powerLt)
          _ = Nat.min blocks (2 ^ round * 2) := by
            symm
            apply Nat.min_eq_left
            simpa [power] using fitsNext
      · have powerFits : power <= blocks - power := by omega
        have nextPowerFits : 2 ^ round * 2 <= blocks := by
          change power * 2 <= blocks
          omega
        calc
          power + Nat.min power (blocks - power) = power + power := by
            exact congrArg (fun remainder => power + remainder)
              (Nat.min_eq_left powerFits)
          _ = power * 2 := by omega
          _ = 2 ^ round * 2 := rfl
          _ = Nat.min blocks (2 ^ round * 2) := by
            symm
            exact Nat.min_eq_right nextPowerFits
    rw [← remainingBound, List.range_add]
    simp only [List.map_append]
    congr 1
    · rw [List.map_map]
      apply List.map_congr_left
      intro index member
      simp only [Function.comp_apply]
      exact (prefixWeight_succ_low coordinates
        (List.mem_range.mp member)).symm
    · rw [List.map_map]
      apply List.map_congr_left
      intro index member
      have indexLt : index < power := Nat.lt_of_lt_of_le
        (List.mem_range.mp member) (Nat.min_le_left _ _)
      simp only [Function.comp_apply]
      have indexInRange : index < (List.range power).length := by simpa
      rw [List.getD_eq_getElem?_getD,
        List.getElem?_eq_getElem (by simpa using indexInRange),
        List.getElem_map, List.getElem_range]
      simp only [Option.getD_some]
      simpa [power, Nat.add_comm] using
        (prefixWeight_succ_high coordinates indexLt).symm

private theorem tensorValuesFrom_eq_expectedPrefix
    (layout : Layout) (assignment : Nat -> Nat)
    (positiveBlocks : 0 < blockCount layout) :
    forall (round : Nat) (parents : List K) (levels : List TensorLevel),
      parents = expectedPrefix (blockCount layout)
          (oldBlockValues layout assignment) round ->
      round + levels.length = layout.blockVariables ->
      tensorValuesFrom layout assignment round parents levels =
        expectedPrefix (blockCount layout)
          (oldBlockValues layout assignment) layout.blockVariables := by
  intro round parents levels parentValues remaining
  cases levels with
  | nil =>
      simp only [List.length_nil, Nat.add_zero] at remaining
      subst round
      simpa [tensorValuesFrom] using parentValues
  | cons level tail =>
      have roundWithin : round < layout.blockVariables := by
        simp only [List.length_cons] at remaining
        omega
      have pointValue : oldBlockValue layout assignment round =
          (oldBlockValues layout assignment).getD round K.zero := by
        simp [oldBlockValue, oldBlockValues, roundWithin, List.getD]
      rw [tensorValuesFrom, parentValues, pointValue,
        nextTensorValues_eq_semanticNext,
        semanticNext_expectedPrefix]
      apply tensorValuesFrom_eq_expectedPrefix layout assignment positiveBlocks
      · rfl
      · simp only [List.length_cons] at remaining
        omega

theorem tensorValues_eq_expectedPrefix
    (layout : Layout) (assignment : Nat -> Nat)
    (positiveBlocks : 0 < blockCount layout)
    (levelCount : layout.tensorLevels.length = layout.blockVariables) :
    tensorValues layout assignment =
      expectedPrefix (blockCount layout)
        (oldBlockValues layout assignment) layout.blockVariables := by
  unfold tensorValues
  apply tensorValuesFrom_eq_expectedPrefix layout assignment positiveBlocks
  · unfold expectedPrefix prefixWeight
    simp [Nat.min_eq_right positiveBlocks]
  · simpa using levelCount

private theorem tensorOps_sub (left right : K) :
    InterpolationOps.sub projectionTensorOps left right = K.sub left right := by
  rcases left with ⟨left0, left1⟩
  rcases right with ⟨right0, right1⟩
  simp only [InterpolationOps.sub, projectionTensorOps, K.add, K.sub, K.zero,
    K.mk.injEq, Fin.sub_eq_add_neg]
  constructor <;> simp [K.add]

theorem prefixWeight_eq_testBitWeight
    (layout : Layout) (assignment : Nat -> Nat)
    (block : Fin (2 ^ layout.blockVariables)) :
    prefixWeight (oldBlockValues layout assignment)
        layout.blockVariables block.val =
      NumericBooleanDomain.testBitWeight projectionTensorOps
        (oldBlockPoint layout assignment) block := by
  unfold prefixWeight NumericBooleanDomain.testBitWeight
  let factor : Nat -> K := fun bit =>
    if Nat.testBit block.val bit then
      (oldBlockValues layout assignment).getD bit K.zero
    else
      K.sub K.one
        ((oldBlockValues layout assignment).getD bit K.zero)
  calc
    _ = ((List.range layout.blockVariables).map factor).foldl K.mul K.one := by
      rw [List.foldl_map]
    _ = ((canonicalFinIndices layout.blockVariables).map
          (fun bit => factor bit.val)).foldl K.mul K.one := by
      congr 1
      simpa only [List.map_map, Function.comp_apply] using
        congrArg (List.map factor)
          (canonicalFinIndices_values layout.blockVariables).symm
    _ = (canonicalFinIndices layout.blockVariables).foldl
          (fun accumulated bit =>
            K.mul accumulated (factor bit.val)) K.one := by
      rw [List.foldl_map]
    _ = _ := by
      apply foldl_congr_on
      intro accumulated bit _
      by_cases bitValue : Nat.testBit block.val bit
      · simp [factor, bitValue, oldBlockPoint, projectionTensorOps]
      · simp only [factor, bitValue, if_false, oldBlockPoint,
          projectionTensorOps,
          InterpolationOps.sub]
        congr 1
        exact (tensorOps_sub K.one
          ((oldBlockValues layout assignment).getD bit K.zero)).symm

private theorem prefixWeight_appendPoint_prefix
    {variables : Nat} (point : CubePoint K variables) (final : K)
    (index : Nat) :
    prefixWeight (point.coordinates ++ [final]) variables index =
      prefixWeight point.coordinates variables index := by
  unfold prefixWeight
  apply foldl_congr_on
  intro accumulated bit member
  have bitWithin : bit < variables := List.mem_range.mp member
  have coordinateWithin : bit < point.coordinates.length := by
    simpa [point.dimension] using bitWithin
  have lookup :
      (point.coordinates ++ [final]).getD bit K.zero =
        point.coordinates.getD bit K.zero := by
    simp only [List.getD_eq_getElem?_getD]
    rw [List.getElem?_append_left coordinateWithin]
  rw [lookup]

private theorem prefixWeight_eq_testBitWeight_point
    {variables : Nat} (point : CubePoint K variables)
    (block : Fin (2 ^ variables)) :
    prefixWeight point.coordinates variables block.val =
      NumericBooleanDomain.testBitWeight projectionTensorOps point block := by
  unfold prefixWeight NumericBooleanDomain.testBitWeight
  let factor : Nat -> K := fun bit =>
    if Nat.testBit block.val bit then
      point.coordinates.getD bit K.zero
    else
      K.sub K.one (point.coordinates.getD bit K.zero)
  calc
    _ = ((List.range variables).map factor).foldl K.mul K.one := by
      rw [List.foldl_map]
    _ = ((canonicalFinIndices variables).map
          (fun bit => factor bit.val)).foldl K.mul K.one := by
      congr 1
      simpa only [List.map_map, Function.comp_apply] using
        congrArg (List.map factor)
          (canonicalFinIndices_values variables).symm
    _ = (canonicalFinIndices variables).foldl
          (fun accumulated bit => K.mul accumulated (factor bit.val)) K.one := by
      rw [List.foldl_map]
    _ = _ := by
      apply foldl_congr_on
      intro accumulated bit _
      by_cases bitValue : Nat.testBit block.val bit
      · simp [factor, bitValue, projectionTensorOps]
      · simp only [factor, bitValue, if_false, projectionTensorOps,
          InterpolationOps.sub]
        congr 1
        exact (tensorOps_sub K.one
          (point.coordinates.getD bit K.zero)).symm

/-- Appending one most-significant coordinate multiplies every all-low prefix
weight by the common `(1 - final)` factor.  The proof lives with the compact
tensor schedule so consumers never depend on the private numeric parity
implementation. -/
theorem testBitWeight_appendPoint_low
    {variables : Nat} (point : CubePoint K variables) (final : K)
    (index : Fin (2 ^ (variables + 1)))
    (low : index.val < 2 ^ variables) :
    NumericBooleanDomain.testBitWeight projectionTensorOps
        (appendPoint point final) index =
      K.mul
        (NumericBooleanDomain.testBitWeight projectionTensorOps point
          ⟨index.val, low⟩)
        (K.sub K.one final) := by
  rw [← prefixWeight_eq_testBitWeight_point (appendPoint point final) index,
    ← prefixWeight_eq_testBitWeight_point point ⟨index.val, low⟩]
  change prefixWeight (point.coordinates ++ [final]) (variables + 1)
      index.val =
    K.mul (prefixWeight point.coordinates variables index.val)
      (K.sub K.one final)
  rw [prefixWeight_succ_low (point.coordinates ++ [final]) low,
    prefixWeight_appendPoint_prefix point final index.val]
  have finalLookup :
      (point.coordinates ++ [final]).getD variables K.zero = final := by
    simp [List.getD_eq_getElem?_getD, point.dimension]
  rw [finalLookup]

private theorem expectedPrefix_getD
    (blocks : Nat) (coordinates : List K) (round index : Nat)
    (indexWithin : index < blocks) (blocksFit : blocks <= 2 ^ round) :
    (expectedPrefix blocks coordinates round).getD index K.one =
      prefixWeight coordinates round index := by
  unfold expectedPrefix
  simp only [Nat.min_eq_left blocksFit]
  have inRange : index <
      ((List.range blocks).map (prefixWeight coordinates round)).length := by
    simpa using indexWithin
  rw [List.getD_eq_getElem?_getD,
    List.getElem?_eq_getElem inRange, List.getElem_map,
    List.getElem_range]
  simp only [Option.getD_some]

/-- First half of the coordinate bridge: physical tensor rows determine the
exact entry of the semantic compact-prefix list. -/
theorem coordinateChiTerms_value_eq_expectedPrefixGetD
    {layout : Layout} (valid : ShapeValid layout)
    {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : RowsSatisfied layout assignment)
    (positiveBlocks : 0 < blockCount layout)
    (coordinate : Fin layout.logicalWidth) :
    (coordinateChiTerms layout coordinate).value assignment =
      (expectedPrefix (blockCount layout)
        (oldBlockValues layout assignment) layout.blockVariables).getD
          (coordinateBlock layout coordinate) K.one := by
  calc
    (coordinateChiTerms layout coordinate).value assignment =
        (tensorValues layout assignment).getD
          (coordinateBlock layout coordinate) K.one :=
      coordinateChiTerms_value_eq_tensorValue valid canonical one satisfies
        coordinate
    _ = (expectedPrefix (blockCount layout)
          (oldBlockValues layout assignment) layout.blockVariables).getD
            (coordinateBlock layout coordinate) K.one :=
      congrArg
        (fun values => values.getD (coordinateBlock layout coordinate) K.one)
        (tensorValues_eq_expectedPrefix layout assignment positiveBlocks
          valid.levelCount)

/-- Second half of the coordinate bridge: an in-domain compact-prefix entry
is the ordinary little-endian Boolean tensor weight. -/
theorem expectedPrefixGetD_eq_testBitWeight
    (layout : Layout) (assignment : Nat -> Nat)
    (block : Fin (2 ^ layout.blockVariables))
    (blockWithin : block.val < blockCount layout)
    (blocksFit : blockCount layout <= 2 ^ layout.blockVariables) :
    (expectedPrefix (blockCount layout)
        (oldBlockValues layout assignment) layout.blockVariables).getD
          block.val K.one =
      NumericBooleanDomain.testBitWeight projectionTensorOps
        (oldBlockPoint layout assignment) block := by
  calc
    (expectedPrefix (blockCount layout)
        (oldBlockValues layout assignment) layout.blockVariables).getD
          block.val K.one =
      prefixWeight (oldBlockValues layout assignment)
        layout.blockVariables block.val :=
      expectedPrefix_getD (blockCount layout) (oldBlockValues layout assignment)
        layout.blockVariables block.val blockWithin blocksFit
    _ = NumericBooleanDomain.testBitWeight projectionTensorOps
        (oldBlockPoint layout assignment) block :=
      prefixWeight_eq_testBitWeight layout assignment block

/-- Exact compact-prefix bridge for any in-domain logical coordinate.  The
two arithmetic premises describe only the layout domain; the production
artifact discharges them from `211797 <= 2^19` and the coordinate decoder.
No assignment value or semantic projection is assumed. -/
theorem coordinateChiTerms_value_eq_testBitWeight
    {layout : Layout} (valid : ShapeValid layout)
    {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : RowsSatisfied layout assignment)
    (coordinate : Fin layout.logicalWidth)
    (coordinateWithin :
      coordinateBlock layout coordinate < blockCount layout)
    (blocksFit : blockCount layout <= 2 ^ layout.blockVariables) :
    (coordinateChiTerms layout coordinate).value assignment =
      NumericBooleanDomain.testBitWeight projectionTensorOps
        (oldBlockPoint layout assignment)
        ⟨coordinateBlock layout coordinate,
          Nat.lt_of_lt_of_le coordinateWithin blocksFit⟩ := by
  have positiveBlocks : 0 < blockCount layout :=
    Nat.lt_of_le_of_lt (Nat.zero_le _) coordinateWithin
  let blockFin : Fin (2 ^ layout.blockVariables) :=
    ⟨coordinateBlock layout coordinate,
      Nat.lt_of_lt_of_le coordinateWithin blocksFit⟩
  change (coordinateChiTerms layout coordinate).value assignment =
    NumericBooleanDomain.testBitWeight projectionTensorOps
      (oldBlockPoint layout assignment) blockFin
  calc
    (coordinateChiTerms layout coordinate).value assignment =
        (expectedPrefix (blockCount layout)
          (oldBlockValues layout assignment) layout.blockVariables).getD
            blockFin.val K.one := by
      change (coordinateChiTerms layout coordinate).value assignment =
        (expectedPrefix (blockCount layout)
          (oldBlockValues layout assignment) layout.blockVariables).getD
            (coordinateBlock layout coordinate) K.one
      exact coordinateChiTerms_value_eq_expectedPrefixGetD valid canonical one
        satisfies positiveBlocks coordinate
    _ = NumericBooleanDomain.testBitWeight projectionTensorOps
        (oldBlockPoint layout assignment) blockFin :=
      expectedPrefixGetD_eq_testBitWeight layout assignment blockFin
        (by simpa [blockFin] using coordinateWithin) blocksFit


end Nightstream.Implementation.R1CS.TerminalRawOldBlockProjectionTensorPrefix
