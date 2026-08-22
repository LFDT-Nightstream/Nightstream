import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.BooleanHypercubeSum

/-! Provenance: copied from `formal/nightstream-lean/Nightstream/SuperNeo/Folding/PiCCS/PaperJoint/FiniteSumAlgebra.lean`
at commit `fb7a8a99aefbb8ebb5474681ecf80f1b95a1b7a2`; namespaces renamed, otherwise unchanged. -/

/-!
Algebraic laws for explicit finite sums used by `Pi_CCS` compression.

Protocol: shared `Pi_CCS` polynomial infrastructure.
Phase: finite-family and product-domain summation.
Constraint family: none; this file emits no rows.

Owns: one right-associated finite indexed sum and the congruence, zero,
additive, subtractive, scalar-distribution, and Fubini laws needed to audit
mixed residual identities.

Does not own: any protocol index family, gamma exponents, Boolean selectors,
residual formulas, SumCheck, Rust, R1CS, or constraint counts.

Emits constraints: no.

Authority boundary: index lists and value functions are explicit. These
lemmas rearrange an existing finite computation under stated algebraic laws;
they do not invent or omit indices.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `pi_ccs.algebra.sum.map` | right-associated sum over an explicit index list | computed | `sumMap` |
| `pi_ccs.algebra.sum.congr` | pointwise equality preserves the sum | derived | `sumMap_congr` |
| `pi_ccs.algebra.sum.linear` | zero/add/neg/sub and left scaling commute with sum | derived | `sumMap_zero`, `sumMap_add`, `sumMap_neg`, `sumMap_sub`, `sumMap_mul_left` |
| `pi_ccs.algebra.sub.zero` | `left - right = 0` exactly when `left = right` | derived | `sub_eq_zero_iff` |
| `pi_ccs.algebra.sum.product` | finite product sums may swap traversal order | derived | `sumMap_swap` |
-/

namespace NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.FiniteSumAlgebra

universe uField uIndex uLeft uRight

open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

/-- Right-associated finite sum over a caller-independent index list. -/
def sumMap
    {Field : Type uField}
    {Index : Type uIndex}
    (ops : InterpolationOps Field)
    (indices : List Index)
    (value : Index -> Field) : Field :=
  BooleanTable.finiteSum ops (indices.map value)

theorem sumMap_congr
    {Field : Type uField}
    {Index : Type uIndex}
    (ops : InterpolationOps Field)
    (indices : List Index)
    (left right : Index -> Field)
    (equal : forall index, index ∈ indices -> left index = right index) :
    sumMap ops indices left = sumMap ops indices right := by
  unfold sumMap
  congr 1
  exact List.map_congr_left equal

theorem neg_zero
    {Field : Type uField}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops) :
    ops.neg ops.zero = ops.zero := by
  have inverse := laws.add_neg ops.zero
  simpa only [laws.zero_add] using inverse

theorem mul_neg
    {Field : Type uField}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (left right : Field) :
    ops.mul left (ops.neg right) = ops.neg (ops.mul left right) := by
  rw [laws.mul_comm left (ops.neg right), laws.neg_mul,
    laws.mul_comm right left]

theorem mul_sub
    {Field : Type uField}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (left middle right : Field) :
    ops.mul left (ops.sub middle right) =
      ops.sub (ops.mul left middle) (ops.mul left right) := by
  unfold InterpolationOps.sub
  rw [laws.left_distrib, mul_neg ops laws]

/-- Subtraction vanishes exactly when its two explicit operands agree. -/
theorem sub_eq_zero_iff
    {Field : Type uField}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (left right : Field) :
    ops.sub left right = ops.zero <-> left = right := by
  constructor
  · intro differenceZero
    have negAdd : ops.add (ops.neg right) right = ops.zero := by
      rw [laws.add_comm]
      exact laws.add_neg right
    calc
      left = ops.add left ops.zero := (laws.add_zero left).symm
      _ = ops.add left (ops.add (ops.neg right) right) := by rw [negAdd]
      _ = ops.add (ops.add left (ops.neg right)) right :=
        (laws.add_assoc _ _ _).symm
      _ = ops.add ops.zero right := by
        change ops.add (ops.sub left right) right = _
        rw [differenceZero]
      _ = right := laws.zero_add right
  · intro equal
    subst left
    exact laws.add_neg right

theorem sumMap_zero
    {Field : Type uField}
    {Index : Type uIndex}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops) :
    forall indices : List Index,
      sumMap ops indices (fun _ => ops.zero) = ops.zero
  | [] => rfl
  | _ :: indices => by
      change ops.add ops.zero
        (sumMap ops indices (fun _ => ops.zero)) = ops.zero
      rw [sumMap_zero ops laws indices, laws.zero_add]

theorem sumMap_add
    {Field : Type uField}
    {Index : Type uIndex}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (indices : List Index)
    (left right : Index -> Field) :
    sumMap ops indices (fun index => ops.add (left index) (right index)) =
      ops.add (sumMap ops indices left) (sumMap ops indices right) := by
  induction indices with
  | nil => simp [sumMap, BooleanTable.finiteSum, laws.zero_add]
  | cons index indices inductionHypothesis =>
      change ops.add (ops.add (left index) (right index))
          (sumMap ops indices fun prior =>
            ops.add (left prior) (right prior)) =
        ops.add
          (ops.add (left index) (sumMap ops indices left))
          (ops.add (right index) (sumMap ops indices right))
      rw [inductionHypothesis]
      calc
        ops.add (ops.add (left index) (right index))
            (ops.add
              (sumMap ops indices left)
              (sumMap ops indices right)) =
          ops.add (left index)
            (ops.add (right index)
              (ops.add
                (sumMap ops indices left)
                (sumMap ops indices right))) :=
            laws.add_assoc _ _ _
        _ = ops.add (left index)
            (ops.add
              (sumMap ops indices left)
              (ops.add (right index)
                (sumMap ops indices right))) := by
          congr 1
          calc
            ops.add (right index)
                (ops.add
                  (sumMap ops indices left)
                  (sumMap ops indices right)) =
              ops.add
                (ops.add (right index)
                  (sumMap ops indices left))
                (sumMap ops indices right) :=
                  (laws.add_assoc _ _ _).symm
            _ = ops.add
                (ops.add
                  (sumMap ops indices left)
                  (right index))
                (sumMap ops indices right) := by
                  rw [laws.add_comm (right index)]
            _ = ops.add
                (sumMap ops indices left)
                (ops.add (right index)
                  (sumMap ops indices right)) :=
                    laws.add_assoc _ _ _
        _ = ops.add
            (ops.add (left index)
              (sumMap ops indices left))
            (ops.add (right index)
              (sumMap ops indices right)) :=
                (laws.add_assoc _ _ _).symm

theorem sumMap_mul_left
    {Field : Type uField}
    {Index : Type uIndex}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (factor : Field)
    (indices : List Index)
    (value : Index -> Field) :
    sumMap ops indices (fun index => ops.mul factor (value index)) =
      ops.mul factor (sumMap ops indices value) := by
  induction indices with
  | nil =>
      simp [sumMap, BooleanTable.finiteSum, laws.mul_zero]
  | cons index indices inductionHypothesis =>
      change ops.add (ops.mul factor (value index))
          (sumMap ops indices fun prior => ops.mul factor (value prior)) =
        ops.mul factor (ops.add (value index) (sumMap ops indices value))
      rw [inductionHypothesis]
      exact (laws.left_distrib factor _ _).symm

theorem sumMap_neg
    {Field : Type uField}
    {Index : Type uIndex}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (indices : List Index)
    (value : Index -> Field) :
    sumMap ops indices (fun index => ops.neg (value index)) =
      ops.neg (sumMap ops indices value) := by
  induction indices with
  | nil =>
      simp [sumMap, BooleanTable.finiteSum, neg_zero ops laws]
  | cons index indices inductionHypothesis =>
      change ops.add (ops.neg (value index))
          (sumMap ops indices fun prior => ops.neg (value prior)) =
        ops.neg (ops.add (value index) (sumMap ops indices value))
      rw [inductionHypothesis, laws.neg_add]

theorem sumMap_sub
    {Field : Type uField}
    {Index : Type uIndex}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (indices : List Index)
    (left right : Index -> Field) :
    sumMap ops indices (fun index => ops.sub (left index) (right index)) =
      ops.sub (sumMap ops indices left) (sumMap ops indices right) := by
  unfold InterpolationOps.sub
  rw [sumMap_add ops laws, sumMap_neg ops laws]

theorem sumMap_swap
    {Field : Type uField}
    {Left : Type uLeft}
    {Right : Type uRight}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (leftIndices : List Left)
    (rightIndices : List Right)
    (value : Left -> Right -> Field) :
    sumMap ops leftIndices (fun left =>
        sumMap ops rightIndices (value left)) =
      sumMap ops rightIndices (fun right =>
        sumMap ops leftIndices (fun left => value left right)) := by
  induction leftIndices with
  | nil =>
      change ops.zero = sumMap ops rightIndices (fun _ => ops.zero)
      exact (sumMap_zero ops laws rightIndices).symm
  | cons left leftIndices inductionHypothesis =>
      change ops.add
          (sumMap ops rightIndices (value left))
          (sumMap ops leftIndices fun prior =>
            sumMap ops rightIndices (value prior)) = _
      rw [inductionHypothesis]
      rw [← sumMap_add ops laws rightIndices
        (value left)
        (fun right => sumMap ops leftIndices fun prior => value prior right)]
      apply sumMap_congr
      intro right _
      rfl

end NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.FiniteSumAlgebra
