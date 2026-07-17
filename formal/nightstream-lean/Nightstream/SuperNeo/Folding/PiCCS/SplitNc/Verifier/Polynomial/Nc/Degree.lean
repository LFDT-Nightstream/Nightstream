import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.Degree.Parameters
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.DegreeSupport
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.InitialSum

/-!
Model-level degree contract for the independent strict-`b = 2` Split-NC
polynomial.

Owns: protocol-specific affine, cubic, and quartic constructions over the
shared fixed-width SumCheck polynomial carrier; coordinate-slice bounds for
the nested source MLE, strict range cubic, challenge mixing, equality gates,
and the complete NC polynomial; and closure under Boolean suffix sums.

Does not own: the shared coefficient-list implementation, prover-message
canonicalization or padding, transcript replay, Rust/R1CS refinement, emitted
rows, or a claim that every honest round has exact degree four. Production
currently accepts padded/trailing-zero message shapes that are deliberately
not identified here with `Message.Canonical`.

Emits constraints: no.

Authority boundary: coefficients are derived from independent source tables
and verifier challenges. No prover-supplied degree or polynomial callback is
accepted.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_ccs.nc.degree.source` | each nested source-MLE coordinate slice is affine | derived | `sourceValueAt_column_affine`, `sourceValueAt_lane_affine` |
| `nifs.pi_ccs.nc.degree.range` | strict-`b = 2` maps an affine slice to a cubic | derived | `rangeValueAt_column_cubic`, `rangeValueAt_lane_cubic` |
| `nifs.pi_ccs.nc.degree.mixing` | gamma compression preserves the cubic ceiling | derived | `mixedRangeAt_column_cubic`, `mixedRangeAt_lane_cubic` |
| `nifs.pi_ccs.nc.degree.selector` | one equality coordinate is affine | derived | `pointEqualityCoordinates_affine` |
| `nifs.pi_ccs.nc.degree.polynomial` | equality-gated NC slices have five coefficients | derived | `qAtPoint_column_quartic`, `qAtPoint_lane_quartic` |
| `nifs.pi_ccs.nc.degree.sumcheck` | Boolean suffix summation preserves five coefficients | derived | `sumCompletions_quartic` |
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.Degree

set_option maxRecDepth 10000
set_option maxHeartbeats 1000000

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.SumCheck
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.Mixing
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.DegreeSupport

/-- A function has an exact fixed-width five-coefficient representation. -/
abbrev RepresentsAtMostFour (function : K → K) : Prop :=
  DegreeSupport.Represents ncSumcheckDegreeBound function

/-- Every degree-four representation projects to a five-coefficient raw
verifier message whose computed degree upper bound is four. -/
theorem representsAtMostFour_message_shape
    {function : K → K}
    (represented : RepresentsAtMostFour function) :
    ∃ message : SumCheck.Finite.Message K,
      message.coefficients.length = ncMessageWidth ∧
      message.degreeUpperBound = ncSumcheckDegreeBound ∧
      ∀ point, message.evaluate ops.toOps point = function point := by
  exact DegreeSupport.Represents.message_shape represented

/-- Every column coordinate of one source's inner MLE is affine. -/
theorem columnValueAt_affine
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (source : Fin shape.sourceCount)
    (lane : BooleanVertex domain.laneVariables)
    (before after : List K)
    (length : before.length + 1 + after.length = domain.columnVariables) :
    ∃ polynomial : Polynomial 1, ∀ point,
      polynomial.evaluate ops.toOps point =
        SourceProjection.columnValueAt covers data source
          (cubeSlice before after length point) lane := by
  unfold SourceProjection.columnValueAt BooleanTable.evaluate
  exact evaluateCoordinates_affine
    (SourceProjection.columnTable covers data source lane)
    before after length

/-- Every column coordinate of the complete nested source MLE is affine. -/
theorem sourceValueAt_column_affine
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (source : Fin shape.sourceCount)
    (lane : CubePoint K domain.laneVariables)
    (before after : List K)
    (length : before.length + 1 + after.length = domain.columnVariables) :
    ∃ polynomial : Polynomial 1, ∀ point,
      polynomial.evaluate ops.toOps point =
        SourceProjection.sourceValueAt covers data source {
          column := cubeSlice before after length point
          lane := lane } := by
  have represented := polynomial_sum_exists
    (BooleanVertex.all domain.laneVariables)
    (fun vertex => vertex.equalityWeight ops lane)
    (fun vertex point => SourceProjection.columnValueAt covers data source
      (cubeSlice before after length point) vertex)
    (by
      intro vertex _
      exact columnValueAt_affine covers data source vertex
        before after length)
  rcases represented with ⟨polynomial, represents⟩
  refine ⟨polynomial, ?_⟩
  intro point
  rw [represents]
  unfold SourceProjection.sourceValueAt SourceProjection.laneTableAtColumn
  rw [← BooleanReproduction.equalityWeighted_tabulate_eq_evaluate
    ops laws]
  rfl

/-- Every lane coordinate of the complete nested source MLE is affine. -/
theorem sourceValueAt_lane_affine
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (source : Fin shape.sourceCount)
    (column : CubePoint K domain.columnVariables)
    (before after : List K)
    (length : before.length + 1 + after.length = domain.laneVariables) :
    ∃ polynomial : Polynomial 1, ∀ point,
      polynomial.evaluate ops.toOps point =
        SourceProjection.sourceValueAt covers data source {
          column := column
          lane := cubeSlice before after length point } := by
  unfold SourceProjection.sourceValueAt BooleanTable.evaluate
  exact evaluateCoordinates_affine
    (SourceProjection.laneTableAtColumn covers data source column)
    before after length

/-- Strict-`b = 2` turns an affine column slice into a cubic. -/
theorem rangeValueAt_column_cubic
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (source : Fin shape.sourceCount)
    (lane : CubePoint K domain.laneVariables)
    (before after : List K)
    (length : before.length + 1 + after.length = domain.columnVariables) :
    ∃ polynomial : Polynomial 3, ∀ point,
      polynomial.evaluate ops.toOps point =
        SourceProjection.rangeValueAt covers data source {
          column := cubeSlice before after length point
          lane := lane } := by
  rcases sourceValueAt_column_affine covers data source lane
    before after length with ⟨sourcePolynomial, sourceRepresents⟩
  refine ⟨strictRangeOfAffine sourcePolynomial, ?_⟩
  intro point
  unfold SourceProjection.rangeValueAt
  rw [evaluate_strictRangeOfAffine, sourceRepresents]
  rw [ConcreteCarrier.derived_sub_eq_concrete_sub]
  let value := SourceProjection.sourceValueAt covers data source {
    column := cubeSlice before after length point
    lane := lane }
  change K.mul (K.mul (K.add value (K.embed 1)) value)
    (K.sub value (K.embed 1)) = _
  rfl

/-- Strict-`b = 2` turns an affine lane slice into a cubic. -/
theorem rangeValueAt_lane_cubic
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (source : Fin shape.sourceCount)
    (column : CubePoint K domain.columnVariables)
    (before after : List K)
    (length : before.length + 1 + after.length = domain.laneVariables) :
    ∃ polynomial : Polynomial 3, ∀ point,
      polynomial.evaluate ops.toOps point =
        SourceProjection.rangeValueAt covers data source {
          column := column
          lane := cubeSlice before after length point } := by
  rcases sourceValueAt_lane_affine covers data source column
    before after length with ⟨sourcePolynomial, sourceRepresents⟩
  refine ⟨strictRangeOfAffine sourcePolynomial, ?_⟩
  intro point
  unfold SourceProjection.rangeValueAt
  rw [evaluate_strictRangeOfAffine, sourceRepresents]
  rw [ConcreteCarrier.derived_sub_eq_concrete_sub]
  let value := SourceProjection.sourceValueAt covers data source {
    column := column
    lane := cubeSlice before after length point }
  change K.mul (K.mul (K.add value (K.embed 1)) value)
    (K.sub value (K.embed 1)) = _
  rfl

/-- Gamma compression preserves the cubic column ceiling. -/
theorem mixedRangeAt_column_cubic
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (convention : GammaConvention)
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Coins domain)
    (lane : CubePoint K domain.laneVariables)
    (before after : List K)
    (length : before.length + 1 + after.length = domain.columnVariables) :
    ∃ polynomial : Polynomial 3, ∀ point,
      polynomial.evaluate ops.toOps point =
        mixedRangeAt convention covers data coins {
          column := cubeSlice before after length point
          lane := lane } := by
  unfold mixedRangeAt
  apply polynomial_sum_exists
  intro source _
  exact rangeValueAt_column_cubic covers data source lane
    before after length

/-- Gamma compression preserves the cubic lane ceiling. -/
theorem mixedRangeAt_lane_cubic
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (convention : GammaConvention)
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Coins domain)
    (column : CubePoint K domain.columnVariables)
    (before after : List K)
    (length : before.length + 1 + after.length = domain.laneVariables) :
    ∃ polynomial : Polynomial 3, ∀ point,
      polynomial.evaluate ops.toOps point =
        mixedRangeAt convention covers data coins {
          column := column
          lane := cubeSlice before after length point } := by
  unfold mixedRangeAt
  apply polynomial_sum_exists
  intro source _
  exact rangeValueAt_lane_cubic covers data source column
    before after length

/-- The column equality selector is affine in each column coordinate. -/
private theorem pointEquality_column_affine
    {domain : FlatNcDomain}
    (coins : Coins domain)
    (before after : List K)
    (length : before.length + 1 + after.length = domain.columnVariables) :
    ∃ polynomial : Polynomial 1, ∀ point,
      polynomial.evaluate ops.toOps point =
        SumCheckTruthPath.pointEquality ops
          (cubeSlice before after length point) coins.betaM := by
  unfold SumCheckTruthPath.pointEquality
  apply pointEqualityCoordinates_affine
  rw [coins.betaM.dimension]
  exact length

/-- The lane equality selector is affine in each lane coordinate. -/
private theorem pointEquality_lane_affine
    {domain : FlatNcDomain}
    (coins : Coins domain)
    (before after : List K)
    (length : before.length + 1 + after.length = domain.laneVariables) :
    ∃ polynomial : Polynomial 1, ∀ point,
      polynomial.evaluate ops.toOps point =
        SumCheckTruthPath.pointEquality ops
          (cubeSlice before after length point) coins.betaA := by
  unfold SumCheckTruthPath.pointEquality
  apply pointEqualityCoordinates_affine
  rw [coins.betaA.dimension]
  exact length

/-- Each column-coordinate slice of the complete equality-gated NC
polynomial has exactly five derived coefficient slots. -/
theorem qAtPoint_column_quartic
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (convention : GammaConvention)
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Coins domain)
    (lane : CubePoint K domain.laneVariables)
    (before after : List K)
    (length : before.length + 1 + after.length = domain.columnVariables) :
    ∃ polynomial : Polynomial ncSumcheckDegreeBound, ∀ point,
      polynomial.evaluate ops.toOps point =
        qAtPoint convention covers data coins {
          column := cubeSlice before after length point
          lane := lane } := by
  rcases pointEquality_column_affine coins before after length with
    ⟨selectorPolynomial, selectorRepresents⟩
  rcases mixedRangeAt_column_cubic convention covers data coins lane
    before after length with ⟨rangePolynomial, rangeRepresents⟩
  let laneSelector := SumCheckTruthPath.pointEquality ops lane coins.betaA
  refine ⟨SumCheck.Finite.FixedPolynomial.scale ops.toOps laneSelector
    (SumCheck.Finite.FixedPolynomial.mul ops.toOps
      selectorPolynomial rangePolynomial), ?_⟩
  intro point
  rw [SumCheck.Finite.FixedPolynomial.evaluate_scale ops.toOps polynomialLaws]
  calc
    ops.mul laneSelector
        ((SumCheck.Finite.FixedPolynomial.mul ops.toOps
          selectorPolynomial rangePolynomial).evaluate ops.toOps point) =
      ops.mul laneSelector
        (ops.mul (selectorPolynomial.evaluate ops.toOps point)
          (rangePolynomial.evaluate ops.toOps point)) :=
      congrArg (ops.mul laneSelector)
        (SumCheck.Finite.FixedPolynomial.evaluate_mul ops.toOps polynomialLaws
          selectorPolynomial rangePolynomial point)
    _ = qAtPoint convention covers data coins {
          column := cubeSlice before after length point
          lane := lane } := by
      rw [selectorRepresents, rangeRepresents]
      unfold qAtPoint
      dsimp only [laneSelector]
      let columnSelector := SumCheckTruthPath.pointEquality ops
        (cubeSlice before after length point) coins.betaM
      let rangeValue := mixedRangeAt convention covers data coins {
        column := cubeSlice before after length point
        lane := lane }
      change ops.mul laneSelector (ops.mul columnSelector rangeValue) =
        ops.mul (ops.mul columnSelector laneSelector) rangeValue
      calc
        ops.mul laneSelector (ops.mul columnSelector rangeValue) =
            ops.mul (ops.mul laneSelector columnSelector) rangeValue :=
          (laws.mul_assoc _ _ _).symm
        _ = ops.mul (ops.mul columnSelector laneSelector) rangeValue := by
          rw [laws.mul_comm laneSelector columnSelector]

/-- Each lane-coordinate slice of the complete equality-gated NC polynomial
has exactly five derived coefficient slots. -/
theorem qAtPoint_lane_quartic
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (convention : GammaConvention)
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Coins domain)
    (column : CubePoint K domain.columnVariables)
    (before after : List K)
    (length : before.length + 1 + after.length = domain.laneVariables) :
    ∃ polynomial : Polynomial ncSumcheckDegreeBound, ∀ point,
      polynomial.evaluate ops.toOps point =
        qAtPoint convention covers data coins {
          column := column
          lane := cubeSlice before after length point } := by
  rcases pointEquality_lane_affine coins before after length with
    ⟨selectorPolynomial, selectorRepresents⟩
  rcases mixedRangeAt_lane_cubic convention covers data coins column
    before after length with ⟨rangePolynomial, rangeRepresents⟩
  let columnSelector :=
    SumCheckTruthPath.pointEquality ops column coins.betaM
  refine ⟨SumCheck.Finite.FixedPolynomial.scale ops.toOps columnSelector
    (SumCheck.Finite.FixedPolynomial.mul ops.toOps
      selectorPolynomial rangePolynomial), ?_⟩
  intro point
  rw [SumCheck.Finite.FixedPolynomial.evaluate_scale ops.toOps polynomialLaws]
  calc
    ops.mul columnSelector
        ((SumCheck.Finite.FixedPolynomial.mul ops.toOps
          selectorPolynomial rangePolynomial).evaluate ops.toOps point) =
      ops.mul columnSelector
        (ops.mul (selectorPolynomial.evaluate ops.toOps point)
          (rangePolynomial.evaluate ops.toOps point)) :=
      congrArg (ops.mul columnSelector)
        (SumCheck.Finite.FixedPolynomial.evaluate_mul ops.toOps polynomialLaws
          selectorPolynomial rangePolynomial point)
    _ = qAtPoint convention covers data coins {
          column := column
          lane := cubeSlice before after length point } := by
      rw [selectorRepresents, rangeRepresents]
      unfold qAtPoint
      dsimp only [columnSelector]
      let laneSelector := SumCheckTruthPath.pointEquality ops
        (cubeSlice before after length point) coins.betaA
      let rangeValue := mixedRangeAt convention covers data coins {
        column := column
        lane := cubeSlice before after length point }
      change ops.mul columnSelector (ops.mul laneSelector rangeValue) =
        ops.mul (ops.mul columnSelector laneSelector) rangeValue
      exact (laws.mul_assoc _ _ _).symm

/-- The typed column slice therefore projects to a raw five-coefficient
message. This is semantic representability, not a claim that production's
padded wire message is `Message.Canonical`. -/
theorem qAtPoint_column_has_five_coefficients
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (convention : GammaConvention)
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Coins domain)
    (lane : CubePoint K domain.laneVariables)
    (before after : List K)
    (length : before.length + 1 + after.length = domain.columnVariables) :
    ∃ message : SumCheck.Finite.Message K,
      message.coefficients.length = ncMessageWidth ∧
      message.degreeUpperBound = ncSumcheckDegreeBound ∧
      ∀ point, message.evaluate ops.toOps point =
        qAtPoint convention covers data coins {
          column := cubeSlice before after length point
          lane := lane } := by
  apply representsAtMostFour_message_shape
  exact qAtPoint_column_quartic convention covers data coins lane
    before after length

/-- The typed lane slice has the same raw five-coefficient representation. -/
theorem qAtPoint_lane_has_five_coefficients
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (convention : GammaConvention)
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Coins domain)
    (column : CubePoint K domain.columnVariables)
    (before after : List K)
    (length : before.length + 1 + after.length = domain.laneVariables) :
    ∃ message : SumCheck.Finite.Message K,
      message.coefficients.length = ncMessageWidth ∧
      message.degreeUpperBound = ncSumcheckDegreeBound ∧
      ∀ point, message.evaluate ops.toOps point =
        qAtPoint convention covers data coins {
          column := column
          lane := cubeSlice before after length point } := by
  apply representsAtMostFour_message_shape
  exact qAtPoint_lane_quartic convention covers data coins column
    before after length

private theorem cubePoint_eq_of_coordinates_eq
    {variables : Nat}
    (left right : CubePoint K variables)
    (coordinates : left.coordinates = right.coordinates) :
    left = right := by
  cases left
  cases right
  simp_all

private theorem ofCoordinates_eq_columnSlice
    {domain : FlatNcDomain}
    (before after : List K)
    (beforeColumn : before.length < domain.columnVariables)
    (totalLength : before.length + 1 + after.length =
      domain.columnVariables + domain.laneVariables)
    (point : K) :
    let columnAfter :=
      after.take (domain.columnVariables - before.length - 1)
    let laneCoordinates :=
      after.drop (domain.columnVariables - before.length - 1)
    let columnLength : before.length + 1 + columnAfter.length =
        domain.columnVariables := by
      dsimp only [columnAfter]
      rw [List.length_take]
      omega
    let laneLength : laneCoordinates.length = domain.laneVariables := by
      dsimp only [laneCoordinates]
      rw [List.length_drop]
      omega
    Point.ofCoordinates (before ++ point :: after) (by simp; omega) = {
      column := cubeSlice before columnAfter columnLength point
      lane := { coordinates := laneCoordinates, dimension := laneLength } } := by
  dsimp only
  apply Point.ext
  · apply cubePoint_eq_of_coordinates_eq
    unfold Point.ofCoordinates cubeSlice
    simp only
    rw [List.take_append]
    rw [List.take_of_length_le (Nat.le_of_lt beforeColumn)]
    have remainingSucc :
        domain.columnVariables - before.length =
          (domain.columnVariables - before.length - 1) + 1 := by
      omega
    rw [remainingSucc]
    rfl
  · apply cubePoint_eq_of_coordinates_eq
    unfold Point.ofCoordinates
    simp only
    rw [List.drop_append]
    rw [List.drop_eq_nil_of_le (Nat.le_of_lt beforeColumn)]
    have remainingSucc :
        domain.columnVariables - before.length =
          (domain.columnVariables - before.length - 1) + 1 := by
      omega
    rw [remainingSucc]
    rfl

private theorem ofCoordinates_eq_laneSlice
    {domain : FlatNcDomain}
    (before after : List K)
    (columnBefore : domain.columnVariables ≤ before.length)
    (totalLength : before.length + 1 + after.length =
      domain.columnVariables + domain.laneVariables)
    (point : K) :
    let columnCoordinates := before.take domain.columnVariables
    let laneBefore := before.drop domain.columnVariables
    let columnLength : columnCoordinates.length =
        domain.columnVariables := by
      dsimp only [columnCoordinates]
      rw [List.length_take]
      omega
    let laneLength : laneBefore.length + 1 + after.length =
        domain.laneVariables := by
      dsimp only [laneBefore]
      rw [List.length_drop]
      omega
    Point.ofCoordinates (before ++ point :: after) (by simp; omega) = {
      column := { coordinates := columnCoordinates, dimension := columnLength }
      lane := cubeSlice laneBefore after laneLength point } := by
  dsimp only
  apply Point.ext
  · apply cubePoint_eq_of_coordinates_eq
    unfold Point.ofCoordinates
    simp only
    exact List.take_append_of_le_length columnBefore
  · apply cubePoint_eq_of_coordinates_eq
    unfold Point.ofCoordinates cubeSlice
    simp only
    exact List.drop_append_of_le_length columnBefore

private theorem sumcheckPolynomial_eq_qAtPoint_of_length
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (convention : GammaConvention)
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Coins domain)
    (coordinates : List K)
    (length : coordinates.length =
      domain.columnVariables + domain.laneVariables) :
    InitialSum.sumcheckPolynomial convention covers data coins coordinates =
      qAtPoint convention covers data coins
        (Point.ofCoordinates coordinates length) := by
  unfold InitialSum.sumcheckPolynomial Mixing.polynomial Point.decode
  rw [dif_pos length]
  rfl

/-- Every exact-arity coordinate slice of the totalized NC SumCheck
polynomial is quartic. This closes the column/lane decoder split without
trusting a caller-selected typed point. -/
theorem sumcheckPolynomial_slice_quartic
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (convention : GammaConvention)
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Coins domain)
    (before after : List K)
    (length : before.length + 1 + after.length =
      domain.columnVariables + domain.laneVariables) :
    RepresentsAtMostFour fun point =>
      InitialSum.sumcheckPolynomial convention covers data coins
        (before ++ point :: after) := by
  by_cases beforeColumn : before.length < domain.columnVariables
  · let columnAfter :=
      after.take (domain.columnVariables - before.length - 1)
    let laneCoordinates :=
      after.drop (domain.columnVariables - before.length - 1)
    have columnLength : before.length + 1 + columnAfter.length =
        domain.columnVariables := by
      dsimp only [columnAfter]
      rw [List.length_take]
      omega
    have laneLength : laneCoordinates.length = domain.laneVariables := by
      dsimp only [laneCoordinates]
      rw [List.length_drop]
      omega
    let lane : CubePoint K domain.laneVariables := {
      coordinates := laneCoordinates
      dimension := laneLength }
    rcases qAtPoint_column_quartic convention covers data coins lane
      before columnAfter columnLength with ⟨slice, sliceRepresents⟩
    refine ⟨slice, ?_⟩
    intro point
    change slice.evaluate ops.toOps point =
      InitialSum.sumcheckPolynomial convention covers data coins
        (before ++ point :: after)
    rw [sumcheckPolynomial_eq_qAtPoint_of_length
      convention covers data coins (before ++ point :: after)
        (by
          simp only [List.length_append, List.length_cons]
          omega)]
    rw [ofCoordinates_eq_columnSlice before after beforeColumn length]
    exact sliceRepresents point
  · have columnBefore : domain.columnVariables ≤ before.length :=
      Nat.le_of_not_gt beforeColumn
    let columnCoordinates := before.take domain.columnVariables
    let laneBefore := before.drop domain.columnVariables
    have columnLength : columnCoordinates.length =
        domain.columnVariables := by
      dsimp only [columnCoordinates]
      rw [List.length_take]
      omega
    have laneLength : laneBefore.length + 1 + after.length =
        domain.laneVariables := by
      dsimp only [laneBefore]
      rw [List.length_drop]
      omega
    let column : CubePoint K domain.columnVariables := {
      coordinates := columnCoordinates
      dimension := columnLength }
    rcases qAtPoint_lane_quartic convention covers data coins column
      laneBefore after laneLength with ⟨slice, sliceRepresents⟩
    refine ⟨slice, ?_⟩
    intro point
    change slice.evaluate ops.toOps point =
      InitialSum.sumcheckPolynomial convention covers data coins
        (before ++ point :: after)
    rw [sumcheckPolynomial_eq_qAtPoint_of_length
      convention covers data coins (before ++ point :: after)
        (by
          simp only [List.length_append, List.length_cons]
          omega)]
    rw [ofCoordinates_eq_laneSlice before after columnBefore length]
    exact sliceRepresents point

/-- Finite Boolean suffix summation preserves a five-coefficient slice.
The premise is pointwise and source-derived: each complete Boolean suffix
must already expose one quartic slice of the same total polynomial. -/
theorem sumCompletions_quartic
    (polynomial : List K → K)
    (fixed : List K)
    (remaining : Nat)
    (represented : ∀ vertex : BooleanVertex remaining,
      ∃ slice : Polynomial ncSumcheckDegreeBound, ∀ point,
        slice.evaluate ops.toOps point =
          polynomial
            ((fixed ++ [point]) ++
              SumCheckTruthPath.VertexEncoding.fieldCoordinates ops vertex)) :
    RepresentsAtMostFour fun point =>
      SumCheck.Finite.HypercubeTruth.sumCompletions ops.toOps polynomial
        (fixed ++ [point]) remaining := by
  exact DegreeSupport.sumCompletions_represents
    polynomial fixed remaining represented

/-- Every honest NC SumCheck round polynomial has an exact quartic
representation. `fixed` is the verifier challenge prefix and `remaining` is
the number of Boolean coordinates summed after the exposed round variable. -/
theorem expectedRound_quartic
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (convention : GammaConvention)
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Coins domain)
    (fixed : List K)
    (remaining : Nat)
    (length : fixed.length + 1 + remaining =
      domain.columnVariables + domain.laneVariables) :
    RepresentsAtMostFour fun point =>
      SumCheck.Finite.HypercubeTruth.sumCompletions ops.toOps
        (InitialSum.sumcheckPolynomial convention covers data coins)
        (fixed ++ [point]) remaining := by
  apply sumCompletions_quartic
  intro vertex
  have suffixLength :
      (SumCheckTruthPath.VertexEncoding.fieldCoordinates ops vertex).length =
        remaining :=
    SumCheckTruthPath.VertexEncoding.fieldCoordinates_length ops vertex
  rcases sumcheckPolynomial_slice_quartic convention covers data coins fixed
    (SumCheckTruthPath.VertexEncoding.fieldCoordinates ops vertex) (by
      rw [suffixLength]
      exact length) with ⟨slice, sliceRepresents⟩
  refine ⟨slice, ?_⟩
  intro point
  simpa only [List.append_assoc, List.singleton_append] using
    sliceRepresents point

/-- Consequently every honest NC round has a raw five-coefficient message
whose verifier-computed upper bound is four. This theorem intentionally says
nothing about trimming or production's padded wire encoding. -/
theorem expectedRound_has_five_coefficients
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (convention : GammaConvention)
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Coins domain)
    (fixed : List K)
    (remaining : Nat)
    (length : fixed.length + 1 + remaining =
      domain.columnVariables + domain.laneVariables) :
    ∃ message : SumCheck.Finite.Message K,
      message.coefficients.length = ncMessageWidth ∧
      message.degreeUpperBound = ncSumcheckDegreeBound ∧
      ∀ point, message.evaluate ops.toOps point =
        SumCheck.Finite.HypercubeTruth.sumCompletions ops.toOps
          (InitialSum.sumcheckPolynomial convention covers data coins)
          (fixed ++ [point]) remaining := by
  apply representsAtMostFour_message_shape
  exact expectedRound_quartic convention covers data coins
    fixed remaining length

end Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.Degree
