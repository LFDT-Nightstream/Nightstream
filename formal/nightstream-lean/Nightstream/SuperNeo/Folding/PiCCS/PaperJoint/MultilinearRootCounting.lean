import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.FiniteRootCounting
import Nightstream.SuperNeo.InteractiveReduction.CoordinateForking.FiniteWords

/-!
Finite root counting for the canonical Boolean-table multilinear extension.

Assurance tier: model-level.

Owns: a direct finite-product proof that a nonzero `ell`-variable table MLE
vanishes on at most `ell * |S|^(ell-1)` points of a duplicate-free Cartesian
support. The induction follows the verifier's coordinate order and invokes the
existing degree-one univariate root theorem at each head coordinate.

Does not own: a protocol residual object, gamma mixing, SumCheck, Fiat--Shamir,
production challenge derivation, Rust, R1CS, artifacts, or costs.

Emits constraints: no.

| Object | Ownership | Exact bound |
|---|---|---|
| scalar support | verifier-owned duplicate-free list | `|S|` |
| point support | recursive Cartesian enumeration | `|S|^ell` |
| table polynomial | canonical recursive MLE | `ell * |S|^(ell-1)` roots |
-/

set_option autoImplicit false

namespace Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.MultilinearRootCounting

open Nightstream.SuperNeo.InteractiveReduction.CoordinateForking
open Nightstream.SuperNeo.InteractiveReduction.CoordinateForking.FiniteWords
open Nightstream.SuperNeo.SumCheck.Finite

universe uField uHead uTail

private theorem fixedPolynomialLaws
    {Field : Type uField}
    {ops : InterpolationOps Field}
    (laws : InterpolationEvaluationLaws ops) :
    FixedPolynomial.Laws ops.toOps := {
  add_assoc := laws.add_assoc
  add_comm := laws.add_comm
  zero_add := laws.zero_add
  add_zero := laws.add_zero
  mul_assoc := laws.mul_assoc
  mul_comm := laws.mul_comm
  mul_zero := laws.mul_zero
  left_distrib := laws.left_distrib
  right_distrib := laws.right_distrib
}

private theorem zero_mul
    {Field : Type uField}
    {ops : InterpolationOps Field}
    (laws : InterpolationEvaluationLaws ops)
    (value : Field) :
    ops.mul ops.zero value = ops.zero := by
  rw [laws.mul_comm, laws.mul_zero]

private theorem add_sub_self_right
    {Field : Type uField}
    {ops : InterpolationOps Field}
    (laws : InterpolationEvaluationLaws ops)
    (low high : Field) :
    ops.add low (ops.sub high low) = high := by
  unfold InterpolationOps.sub
  calc
    ops.add low (ops.add high (ops.neg low)) =
        ops.add (ops.add low high) (ops.neg low) :=
      (laws.add_assoc _ _ _).symm
    _ = ops.add (ops.add high low) (ops.neg low) := by
      rw [laws.add_comm low high]
    _ = ops.add high (ops.add low (ops.neg low)) :=
      laws.add_assoc _ _ _
    _ = ops.add high ops.zero := by rw [laws.add_neg]
    _ = high := laws.add_zero high

private theorem nat_sum_map_add
    {Element : Type uHead}
    (values : List Element)
    (left right : Element -> Nat) :
    (values.map fun value => left value + right value).sum =
      (values.map left).sum + (values.map right).sum := by
  induction values with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      simp only [List.map_cons, List.sum_cons, inductionHypothesis]
      omega

private theorem nat_sum_swap
    {Head : Type uHead}
    {Tail : Type uTail}
    (heads : List Head)
    (tails : List Tail)
    (value : Head -> Tail -> Nat) :
    (heads.map fun head =>
        (tails.map fun tail => value head tail).sum).sum =
      (tails.map fun tail =>
        (heads.map fun head => value head tail).sum).sum := by
  induction heads with
  | nil => simp [List.map_const', List.sum_replicate_nat]
  | cons head heads inductionHypothesis =>
      simp only [List.map_cons, List.sum_cons, inductionHypothesis]
      exact (nat_sum_map_add tails
        (fun tail => value head tail)
        (fun tail => (heads.map fun prior => value prior tail).sum)).symm

private theorem countP_eq_indicator_sum
    {Element : Type uHead}
    (values : List Element)
    (event : Element -> Bool) :
    values.countP event =
      (values.map fun value => if event value then 1 else 0).sum := by
  induction values with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      rw [List.countP_cons, List.map_cons, List.sum_cons, inductionHypothesis]
      exact Nat.add_comm _ _

/-- Finite Fubini identity for Boolean counts on a Cartesian list. -/
private theorem sum_countP_swap
    {Head : Type uHead}
    {Tail : Type uTail}
    (heads : List Head)
    (tails : List Tail)
    (event : Head -> Tail -> Bool) :
    (heads.map fun head => tails.countP (event head)).sum =
      (tails.map fun tail => heads.countP (fun head => event head tail)).sum := by
  simpa only [countP_eq_indicator_sum] using
    nat_sum_swap heads tails fun head tail =>
      if event head tail then 1 else 0

/-- The recursive word enumeration may be counted tail-first without changing
the exact event multiplicity. -/
private theorem vectors_succ_countP_eq_tail_sum
    {Field : Type uField}
    (alphabet : List Field)
    (variables : Nat)
    (event : (Fin (variables + 1) -> Field) -> Bool) :
    (vectors alphabet (variables + 1)).countP event =
      ((vectors alphabet variables).map fun tail =>
        alphabet.countP (fun head => event (prepend head tail))).sum := by
  rw [vectors, List.countP_flatMap]
  calc
    (alphabet.map fun head =>
        ((vectors alphabet variables).map
          (prepend head)).countP event).sum =
        (alphabet.map fun head =>
          (vectors alphabet variables).countP
            (fun tail => event (prepend head tail))).sum := by
          apply congrArg List.sum
          apply List.map_congr_left
          intro head _member
          rw [List.countP_map]
          rfl
    _ = _ :=
      sum_countP_swap alphabet (vectors alphabet variables)
        (fun head tail => event (prepend head tail))

/-- Sum a fiber bound in which every bad tail permits the whole head support,
while every good tail permits at most one head. The extra `tails.length`
charges one possible root uniformly and keeps the arithmetic branch-free. -/
private theorem sum_fibers_le_bad_mul_length_add_length
    {Head : Type uHead}
    {Tail : Type uTail}
    (heads : List Head)
    (tails : List Tail)
    (bad : Tail -> Bool)
    (event : Head -> Tail -> Bool)
    (goodBound :
      forall tail, tail ∈ tails -> bad tail = false ->
        heads.countP (fun head => event head tail) <= 1) :
    (tails.map fun tail =>
        heads.countP (fun head => event head tail)).sum <=
      tails.countP bad * heads.length + tails.length := by
  induction tails with
  | nil => simp
  | cons tail tails inductionHypothesis =>
      have tailGoodBound :
          forall prior, prior ∈ tails -> bad prior = false ->
            heads.countP (fun head => event head prior) <= 1 := by
        intro prior member priorGood
        exact goodBound prior (by simp [member]) priorGood
      have remainder := inductionHypothesis tailGoodBound
      cases tailBad : bad tail with
      | false =>
          have current := goodBound tail (by simp) tailBad
          have combined := Nat.add_le_add current remainder
          simpa [tailBad, Nat.add_mul, Nat.add_assoc, Nat.add_comm,
            Nat.add_left_comm] using combined
      | true =>
          have current :
              heads.countP (fun head => event head tail) <= heads.length :=
            List.countP_le_length
          have withSlack :
              heads.countP (fun head => event head tail) <=
                heads.length + 1 :=
            Nat.le_trans current (Nat.le_add_right _ _)
          have combined := Nat.add_le_add withSlack remainder
          simpa [tailBad, Nat.add_mul, Nat.add_assoc, Nat.add_comm,
            Nat.add_left_comm] using combined

private theorem branch_allEntriesZero_iff
    {Field : Type uField}
    (ops : InterpolationOps Field)
    {variables : Nat}
    (low high : BooleanTable Field variables) :
    (BooleanTable.branch low high).AllEntriesZero ops ↔
      low.AllEntriesZero ops /\ high.AllEntriesZero ops := by
  constructor
  · intro allZero
    constructor
    · intro value member
      exact allZero value (by
        simp only [BooleanTable.entries, List.mem_append]
        exact Or.inl member)
    · intro value member
      exact allZero value (by
        simp only [BooleanTable.entries, List.mem_append]
        exact Or.inr member)
  · rintro ⟨lowZero, highZero⟩ value member
    simp only [BooleanTable.entries, List.mem_append] at member
    cases member with
    | inl lowMember => exact lowZero value lowMember
    | inr highMember => exact highZero value highMember

private theorem affine_nonzero_of_low
    {Field : Type uField}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    {low high : Field}
    (lowNonzero : low ≠ ops.zero) :
    (fun point =>
        (FixedPolynomial.affine low (ops.sub high low)).evaluate
          ops.toOps point) ≠
      fun _ => ops.zero := by
  intro identicallyZero
  have atZero := congrFun identicallyZero ops.zero
  rw [FixedPolynomial.evaluate_affine ops.toOps
    (fixedPolynomialLaws laws), zero_mul laws, laws.add_zero] at atZero
  exact lowNonzero atZero

private theorem affine_nonzero_of_high
    {Field : Type uField}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    {low high : Field}
    (highNonzero : high ≠ ops.zero) :
    (fun point =>
        (FixedPolynomial.affine low (ops.sub high low)).evaluate
          ops.toOps point) ≠
      fun _ => ops.zero := by
  intro identicallyZero
  have atOne := congrFun identicallyZero ops.one
  rw [FixedPolynomial.evaluate_affine ops.toOps
    (fixedPolynomialLaws laws), laws.one_mul,
    add_sub_self_right laws] at atOne
  exact highNonzero atOne

/-- Exact finite count form of multilinear Schwartz--Zippel for the
verifier-owned recursive Cartesian support. -/
theorem zeros_count_le
    {Field : Type uField}
    [DecidableEq Field]
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (noZeroDivisors : FiniteRootCounting.NoZeroDivisors ops)
    {variables : Nat}
    (table : BooleanTable Field variables)
    (alphabet : List Field)
    (alphabetNodup : alphabet.Nodup)
    (nonzero : Not (table.AllEntriesZero ops)) :
    (vectors alphabet variables).countP (fun word =>
        decide
          (table.evaluateCoordinates ops (List.ofFn word) = ops.zero)) <=
      variables * alphabet.length ^ variables.pred := by
  induction table with
  | leaf value =>
      have valueNonzero : value ≠ ops.zero := by
        intro valueZero
        apply nonzero
        simp [BooleanTable.AllEntriesZero, BooleanTable.entries, valueZero]
      simp [vectors, BooleanTable.evaluateCoordinates, valueNonzero]
  | @branch variables low high lowInduction highInduction =>
      have oneVariableRootBound
          (lowValue highValue : Field)
          (endpointNonzero :
            lowValue ≠ ops.zero \/ highValue ≠ ops.zero) :
          alphabet.countP (fun point =>
              decide
                (ops.add lowValue
                    (ops.mul point (ops.sub highValue lowValue)) =
                  ops.zero)) <= 1 := by
        let polynomial :=
          FixedPolynomial.affine lowValue (ops.sub highValue lowValue)
        have polynomialNonzero :
            (fun point => polynomial.evaluate ops.toOps point) ≠
              fun _ => ops.zero := by
          cases endpointNonzero with
          | inl lowNonzero =>
              exact affine_nonzero_of_low ops laws lowNonzero
          | inr highNonzero =>
              exact affine_nonzero_of_high ops laws highNonzero
        simpa [polynomial, FixedPolynomial.evaluate_affine ops.toOps
          (fixedPolynomialLaws laws)] using
          FiniteRootCounting.roots_count_le_degree ops laws noZeroDivisors
            1 polynomial alphabet alphabetNodup polynomialNonzero
      have branchNonzero :
          Not (low.AllEntriesZero ops) \/
            Not (high.AllEntriesZero ops) := by
        by_cases lowZero : low.AllEntriesZero ops
        · by_cases highZero : high.AllEntriesZero ops
          · exact False.elim
              (nonzero
                ((branch_allEntriesZero_iff ops low high).2
                  ⟨lowZero, highZero⟩))
          · exact Or.inr highZero
        · exact Or.inl lowZero
      rw [vectors_succ_countP_eq_tail_sum]
      cases branchNonzero with
      | inl lowNonzero =>
          let tails := vectors alphabet variables
          let lowBad : (Fin variables -> Field) -> Bool := fun tail =>
            decide
              (low.evaluateCoordinates ops (List.ofFn tail) = ops.zero)
          have goodFiber :
              forall tail, tail ∈ tails -> lowBad tail = false ->
                alphabet.countP (fun head =>
                  decide
                    ((BooleanTable.branch low high).evaluateCoordinates ops
                        (List.ofFn (prepend head tail)) = ops.zero)) <= 1 := by
            intro tail _member tailGood
            have lowValueNonzero :
                low.evaluateCoordinates ops (List.ofFn tail) ≠ ops.zero := by
              simpa [lowBad] using tailGood
            simpa [BooleanTable.evaluateCoordinates, List.ofFn_succ,
              lowBad] using
              oneVariableRootBound
                (low.evaluateCoordinates ops (List.ofFn tail))
                (high.evaluateCoordinates ops (List.ofFn tail))
                (Or.inl lowValueNonzero)
          have fiberBound :=
            sum_fibers_le_bad_mul_length_add_length alphabet tails lowBad
              (fun head tail =>
                decide
                  ((BooleanTable.branch low high).evaluateCoordinates ops
                      (List.ofFn (prepend head tail)) = ops.zero))
              goodFiber
          have badBound :
              tails.countP lowBad <=
                variables * alphabet.length ^ variables.pred := by
            simpa [tails, lowBad] using
              lowInduction lowNonzero
          have combined := Nat.add_le_add_right
            (Nat.mul_le_mul_right alphabet.length badBound) tails.length
          refine Nat.le_trans fiberBound (Nat.le_trans combined ?_)
          rw [vectors_length]
          cases variables with
          | zero => simp
          | succ prior =>
              apply Nat.le_of_eq
              simp only [Nat.pred_succ, Nat.pow_succ, Nat.add_mul]
              ac_rfl
      | inr highNonzero =>
          let tails := vectors alphabet variables
          let highBad : (Fin variables -> Field) -> Bool := fun tail =>
            decide
              (high.evaluateCoordinates ops (List.ofFn tail) = ops.zero)
          have goodFiber :
              forall tail, tail ∈ tails -> highBad tail = false ->
                alphabet.countP (fun head =>
                  decide
                    ((BooleanTable.branch low high).evaluateCoordinates ops
                        (List.ofFn (prepend head tail)) = ops.zero)) <= 1 := by
            intro tail _member tailGood
            have highValueNonzero :
                high.evaluateCoordinates ops (List.ofFn tail) ≠ ops.zero := by
              simpa [highBad] using tailGood
            simpa [BooleanTable.evaluateCoordinates, List.ofFn_succ,
              highBad] using
              oneVariableRootBound
                (low.evaluateCoordinates ops (List.ofFn tail))
                (high.evaluateCoordinates ops (List.ofFn tail))
                (Or.inr highValueNonzero)
          have fiberBound :=
            sum_fibers_le_bad_mul_length_add_length alphabet tails highBad
              (fun head tail =>
                decide
                  ((BooleanTable.branch low high).evaluateCoordinates ops
                      (List.ofFn (prepend head tail)) = ops.zero))
              goodFiber
          have badBound :
              tails.countP highBad <=
                variables * alphabet.length ^ variables.pred := by
            simpa [tails, highBad] using
              highInduction highNonzero
          have combined := Nat.add_le_add_right
            (Nat.mul_le_mul_right alphabet.length badBound) tails.length
          refine Nat.le_trans fiberBound (Nat.le_trans combined ?_)
          rw [vectors_length]
          cases variables with
          | zero => simp
          | succ prior =>
              apply Nat.le_of_eq
              simp only [Nat.pred_succ, Nat.pow_succ, Nat.add_mul]
              ac_rfl

end Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.MultilinearRootCounting
