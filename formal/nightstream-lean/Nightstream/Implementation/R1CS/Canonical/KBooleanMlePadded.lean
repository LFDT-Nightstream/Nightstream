import Nightstream.Implementation.R1CS.Canonical.KBooleanMleSemantics
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Fe

/-!
Contract: the canonical 54-live-lane/Boolean-cube padding bridge.

The physical MLE consumes a complete Boolean table.  The selected FE relation
instead defines its lane MLE as a sum over the 54 authoritative Phi81 lanes.
This module proves those are the same computation when the remaining Boolean
vertices are derived zero leaves.  The proof reindexes the semantic
low/high vertex order through the authoritative little-endian numeric bridge;
it does not assume an external serialization or Rust layout.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Canonical.KBooleanMlePadded

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial

abbrev Ops := ConcreteCarrier.extensionOps

/-- Numeric image of the semantic low/high Boolean-vertex enumeration. -/
def numericEnumeration (variables : Nat) : List (Fin (2 ^ variables)) :=
  (BooleanVertex.all variables).map fun vertex =>
    ⟨NumericBooleanDomain.index vertex,
      NumericBooleanDomain.index_lt_twoPow vertex⟩

private theorem numericVertex_injective
    (variables : Nat) :
    Function.Injective
      (fun vertex : BooleanVertex variables =>
        (⟨NumericBooleanDomain.index vertex,
          NumericBooleanDomain.index_lt_twoPow vertex⟩ :
          Fin (2 ^ variables))) := by
  intro left right equal
  have equalFin :
      (⟨NumericBooleanDomain.index left,
          NumericBooleanDomain.index_lt_twoPow left⟩ :
        Fin (2 ^ variables)) =
      ⟨NumericBooleanDomain.index right,
        NumericBooleanDomain.index_lt_twoPow right⟩ := equal
  calc
    left =
        NumericBooleanDomain.vertex variables
          ⟨NumericBooleanDomain.index left,
            NumericBooleanDomain.index_lt_twoPow left⟩ :=
      (NumericBooleanDomain.vertex_index left).symm
    _ = NumericBooleanDomain.vertex variables
          ⟨NumericBooleanDomain.index right,
            NumericBooleanDomain.index_lt_twoPow right⟩ := by
      exact congrArg (NumericBooleanDomain.vertex variables) equalFin
    _ = right := NumericBooleanDomain.vertex_index right

theorem numericEnumeration_nodup (variables : Nat) :
    (numericEnumeration variables).Nodup := by
  unfold numericEnumeration
  exact
    LinCombNormal.nodup_map (BooleanVertex.all variables) _
      (numericVertex_injective variables)
      (BooleanVertex.all_nodup variables)

private theorem perm_of_nodup_members
    {Value : Type}
    [BEq Value] [LawfulBEq Value]
    {left right : List Value}
    (leftNodup : left.Nodup)
    (rightNodup : right.Nodup)
    (sameMembers : ∀ value, value ∈ left ↔ value ∈ right) :
    left.Perm right := by
  rw [List.perm_iff_count]
  intro value
  rw [leftNodup.count, rightNodup.count]
  by_cases member : value ∈ left
  · have rightMember := (sameMembers value).mp member
    simp [member, rightMember]
  · have rightMember : value ∉ right :=
      fun present => member ((sameMembers value).mpr present)
    simp [member, rightMember]

/-- The numeric image is a permutation of every bounded numeric index. -/
theorem numericEnumeration_perm (variables : Nat) :
    (numericEnumeration variables).Perm
      (canonicalFinIndices (2 ^ variables)) := by
  apply perm_of_nodup_members
  · exact numericEnumeration_nodup variables
  · exact canonicalFinIndices_nodup _
  · intro index
    constructor
    · intro _
      simp [canonicalFinIndices]
    · intro _
      unfold numericEnumeration
      apply List.mem_map.mpr
      let vertex := NumericBooleanDomain.vertex variables index
      refine ⟨vertex, BooleanVertex.mem_all vertex, ?_⟩
      apply Fin.ext
      exact NumericBooleanDomain.index_vertex variables index

private theorem finiteSum_append
    {Field : Type}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (left right : List Field) :
    BooleanTable.finiteSum ops (left ++ right) =
      ops.add (BooleanTable.finiteSum ops left)
        (BooleanTable.finiteSum ops right) := by
  induction left with
  | nil => simp [BooleanTable.finiteSum, laws.zero_add]
  | cons value values inductionHypothesis =>
      simp only [List.cons_append, BooleanTable.finiteSum,
        inductionHypothesis]
      exact (laws.add_assoc _ _ _).symm

private theorem finiteSum_perm
    {Field : Type}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    {left right : List Field}
    (permutation : left.Perm right) :
    BooleanTable.finiteSum ops left =
      BooleanTable.finiteSum ops right := by
  induction permutation with
  | nil => rfl
  | cons value permutation inductionHypothesis =>
      simp only [BooleanTable.finiteSum]
      rw [inductionHypothesis]
  | swap left right rest =>
      simp only [BooleanTable.finiteSum]
      calc
        ops.add right
            (ops.add left (BooleanTable.finiteSum ops rest)) =
          ops.add (ops.add right left)
            (BooleanTable.finiteSum ops rest) :=
          (laws.add_assoc _ _ _).symm
        _ = ops.add (ops.add left right)
            (BooleanTable.finiteSum ops rest) := by
          rw [laws.add_comm right left]
        _ = ops.add left
            (ops.add right (BooleanTable.finiteSum ops rest)) :=
          laws.add_assoc _ _ _
  | trans first second firstInduction secondInduction =>
      exact firstInduction.trans secondInduction

private theorem finiteSum_all_zero
    {Field Index : Type}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (indices : List Index)
    (value : Index → Field)
    (zero : ∀ index ∈ indices, value index = ops.zero) :
    BooleanTable.finiteSum ops (indices.map value) = ops.zero := by
  induction indices with
  | nil => rfl
  | cons index indices inductionHypothesis =>
      simp only [List.map_cons, BooleanTable.finiteSum]
      rw [zero index (by simp),
        inductionHypothesis (fun prior member =>
          zero prior (List.mem_cons_of_mem index member)),
        laws.add_zero]

/-- Zero-extend exactly the 54 authoritative lanes to the full Boolean cube. -/
def semanticTable
    {variables : Nat}
    (values : Fin ringDegree → K) :
    BooleanTable K variables :=
  BooleanTable.tabulate fun vertex =>
    if bounded : NumericBooleanDomain.index vertex < ringDegree then
      values ⟨NumericBooleanDomain.index vertex, bounded⟩
    else
      K.zero

private def paddedTerm
    {variables : Nat}
    (covers : ringDegree ≤ 2 ^ variables)
    (values : Fin ringDegree → K)
    (point : CubePoint K variables)
    (index : Nat) : K :=
  if bounded : index < ringDegree then
    K.mul
      (NumericBooleanDomain.tensorWeight Ops
        ⟨index, Nat.lt_of_lt_of_le bounded covers⟩ point)
      (values ⟨index, bounded⟩)
  else
    K.zero

private theorem paddedTerm_of_fin
    {variables : Nat}
    (covers : ringDegree ≤ 2 ^ variables)
    (values : Fin ringDegree → K)
    (point : CubePoint K variables)
    (index : Fin (2 ^ variables)) :
    K.mul
        (NumericBooleanDomain.tensorWeight Ops index point)
        (if bounded : index.val < ringDegree then
          values ⟨index.val, bounded⟩
        else
          K.zero) =
      paddedTerm covers values point index.val := by
  unfold paddedTerm
  by_cases bounded : index.val < ringDegree
  · rw [dif_pos bounded, dif_pos bounded]
  · rw [dif_neg bounded, dif_neg bounded]
    change Ops.mul _ Ops.zero = Ops.zero
    exact ConcreteCarrier.extensionLaws.mul_zero _

private theorem paddedTerm_live
    {variables : Nat}
    (covers : ringDegree ≤ 2 ^ variables)
    (values : Fin ringDegree → K)
    (point : CubePoint K variables)
    (lane : Fin ringDegree) :
    paddedTerm covers values point lane.val =
      K.mul
        (NumericBooleanDomain.tensorWeight Ops
          ⟨lane.val, Nat.lt_of_lt_of_le lane.isLt covers⟩ point)
        (values lane) := by
  unfold paddedTerm
  rw [dif_pos lane.isLt]

private theorem paddedFiniteSum
    {variables : Nat}
    (covers : ringDegree ≤ 2 ^ variables)
    (values : Fin ringDegree → K)
    (point : CubePoint K variables) :
    BooleanTable.finiteSum Ops
        ((List.range (2 ^ variables)).map
          (paddedTerm covers values point)) =
      FiniteSumAlgebra.sumMap Ops (canonicalFinIndices ringDegree) fun lane =>
        K.mul
          (NumericBooleanDomain.tensorWeight Ops
            ⟨lane.val, Nat.lt_of_lt_of_le lane.isLt covers⟩ point)
          (values lane) := by
  have split : 2 ^ variables =
      ringDegree + (2 ^ variables - ringDegree) := by omega
  have rangeSplit :
      List.range (2 ^ variables) =
        List.range ringDegree ++
          (List.range (2 ^ variables - ringDegree)).map
            (fun offset => ringDegree + offset) := by
    calc
      List.range (2 ^ variables) =
          List.range
            (ringDegree + (2 ^ variables - ringDegree)) :=
        congrArg List.range split
      _ = _ := List.range_add
  rw [rangeSplit, List.map_append,
    finiteSum_append Ops ConcreteCarrier.extensionLaws]
  have live :
      BooleanTable.finiteSum Ops
          ((List.range ringDegree).map (paddedTerm covers values point)) =
        FiniteSumAlgebra.sumMap Ops (canonicalFinIndices ringDegree) fun lane =>
          K.mul
            (NumericBooleanDomain.tensorWeight Ops
              ⟨lane.val, Nat.lt_of_lt_of_le lane.isLt covers⟩ point)
            (values lane) := by
    unfold FiniteSumAlgebra.sumMap
    rw [← canonicalFinIndices_values ringDegree, List.map_map]
    congr 1
  rw [live]
  have tail :
      BooleanTable.finiteSum Ops
          (((List.range (2 ^ variables - ringDegree)).map
              (fun offset => ringDegree + offset)).map
            (paddedTerm covers values point)) =
        K.zero := by
    rw [List.map_map]
    apply finiteSum_all_zero Ops ConcreteCarrier.extensionLaws
    intro offset member
    simp only [Function.comp_apply]
    unfold paddedTerm
    rw [dif_neg (by omega)]
    rfl
  rw [tail]
  change Ops.add _ Ops.zero = _
  exact ConcreteCarrier.extensionLaws.add_zero _

/-- The full padded Boolean table evaluates to the unchanged FE lane formula. -/
theorem semanticTable_evaluate
    {domain : FlatNcDomain}
    (covers : Fe.LaneCovers domain)
    (values : Fin ringDegree → K)
    (point : CubePoint K domain.laneVariables) :
    BooleanTable.evaluate Ops (semanticTable values) point =
      Fe.paddedLaneEvaluation covers values point := by
  rw [BooleanTable.evaluate_eq_equalityWeightedSum
    Ops ConcreteCarrier.extensionLaws]
  unfold BooleanTable.equalityWeightedSum semanticTable
  simp only [BooleanTable.valueAt_tabulate]
  change
    BooleanTable.finiteSum Ops
        ((BooleanVertex.all domain.laneVariables).map fun vertex =>
          K.mul (BooleanVertex.equalityWeight Ops vertex point)
            (if bounded : NumericBooleanDomain.index vertex < ringDegree then
              values ⟨NumericBooleanDomain.index vertex, bounded⟩
            else K.zero)) =
      _
  let term : Fin (2 ^ domain.laneVariables) → K :=
    fun index =>
      K.mul (NumericBooleanDomain.tensorWeight Ops index point)
        (if bounded : index.val < ringDegree then
          values ⟨index.val, bounded⟩
        else K.zero)
  have reindexed :
      ((BooleanVertex.all domain.laneVariables).map fun vertex =>
          K.mul (BooleanVertex.equalityWeight Ops vertex point)
            (if bounded : NumericBooleanDomain.index vertex < ringDegree then
              values ⟨NumericBooleanDomain.index vertex, bounded⟩
            else K.zero)) =
        (numericEnumeration domain.laneVariables).map term := by
    unfold numericEnumeration
    rw [List.map_map]
    apply List.map_congr_left
    intro vertex _
    unfold term
    simp only [Function.comp_apply]
    rw [NumericBooleanDomain.tensorWeight_eq_equalityWeight,
      NumericBooleanDomain.vertex_index]
  rw [reindexed]
  have permuted :=
    (numericEnumeration_perm domain.laneVariables).map term
  rw [finiteSum_perm Ops ConcreteCarrier.extensionLaws permuted]
  have covers' : ringDegree ≤ 2 ^ domain.laneVariables := by
    simpa [FlatNcDomain.laneCount] using covers
  have finMap :
      (canonicalFinIndices (2 ^ domain.laneVariables)).map term =
        (List.range (2 ^ domain.laneVariables)).map
          (paddedTerm covers' values point) := by
    rw [← canonicalFinIndices_values (2 ^ domain.laneVariables),
      List.map_map]
    apply List.map_congr_left
    intro index _
    exact paddedTerm_of_fin covers' values point index
  rw [finMap, paddedFiniteSum covers' values point]
  rfl

end Nightstream.Implementation.R1CS.Canonical.KBooleanMlePadded
