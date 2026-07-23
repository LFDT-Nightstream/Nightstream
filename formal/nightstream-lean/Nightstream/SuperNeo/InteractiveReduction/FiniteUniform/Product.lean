import Nightstream.SuperNeo.InteractiveReduction.FiniteUniform

/-!
Exact Cartesian products for finite-uniform seed supports.

Owns: lexicographic product enumeration, duplicate freedom, nonemptiness,
membership, cardinality, event counts, and the two uniform Boolean marginals.

Does not own: a protocol, challenge interpretation, rejection conditioning,
root bounds, Fiat--Shamir, Rust, R1CS, or costs.
-/

set_option autoImplicit false

namespace Nightstream.SuperNeo.InteractiveReduction.FiniteUniform

universe uLeft uRight

variable {Left : Type uLeft}
variable {Right : Type uRight}

namespace Support

/-- Lexicographic Cartesian product. The left seed is the outer loop and the
right seed is the inner loop. -/
def product (left : Support Left) (right : Support Right) :
    Support (Left × Right) where
  values := left.values.flatMap fun leftSeed =>
    right.values.map fun rightSeed => (leftSeed, rightSeed)
  nodup := by
    change List.Pairwise (fun first second : Left × Right => first ≠ second)
      (left.values.flatMap fun leftSeed =>
        right.values.map fun rightSeed => (leftSeed, rightSeed))
    rw [List.pairwise_flatMap]
    constructor
    · intro leftSeed _
      exact right.nodup.map (fun rightSeed => (leftSeed, rightSeed)) (by
        intro first second distinct equal
        exact distinct (Prod.mk.inj equal).2)
    · exact left.nodup.imp (by
        intro firstLeft secondLeft distinct
        intro firstPair firstMember secondPair secondMember equal
        rcases List.mem_map.mp firstMember with ⟨firstRight, _, rfl⟩
        rcases List.mem_map.mp secondMember with ⟨secondRight, _, rfl⟩
        exact distinct (Prod.mk.inj equal).1)
  nonempty := by
    rcases List.exists_mem_of_ne_nil left.values left.nonempty with
      ⟨leftSeed, leftMember⟩
    rcases List.exists_mem_of_ne_nil right.values right.nonempty with
      ⟨rightSeed, rightMember⟩
    intro empty
    have pairMember :
        (leftSeed, rightSeed) ∈ left.values.flatMap (fun leftValue =>
          right.values.map fun rightValue => (leftValue, rightValue)) := by
      apply List.mem_flatMap.mpr
      exact ⟨leftSeed, leftMember,
        List.mem_map.mpr ⟨rightSeed, rightMember, rfl⟩⟩
    rw [empty] at pairMember
    exact List.not_mem_nil pairMember

@[simp] theorem product_values
    (left : Support Left)
    (right : Support Right) :
    (left.product right).values = left.values.flatMap fun leftSeed =>
      right.values.map fun rightSeed => (leftSeed, rightSeed) :=
  rfl

@[simp] theorem mem_product_iff
    (left : Support Left)
    (right : Support Right)
    (seed : Left × Right) :
    seed ∈ (left.product right).values ↔
      seed.1 ∈ left.values ∧ seed.2 ∈ right.values := by
  constructor
  · intro member
    rcases List.mem_flatMap.mp member with
      ⟨leftSeed, leftMember, pairMember⟩
    rcases List.mem_map.mp pairMember with
      ⟨rightSeed, rightMember, pairEqual⟩
    cases pairEqual
    exact ⟨leftMember, rightMember⟩
  · intro member
    apply List.mem_flatMap.mpr
    exact ⟨seed.1, member.1,
      List.mem_map.mpr ⟨seed.2, member.2, rfl⟩⟩

private theorem product_values_length
    (leftValues : List Left)
    (rightValues : List Right) :
    (leftValues.flatMap fun leftSeed =>
      rightValues.map fun rightSeed => (leftSeed, rightSeed)).length =
      leftValues.length * rightValues.length := by
  induction leftValues with
  | nil => simp
  | cons _ leftValues inductionHypothesis =>
      simp only [List.flatMap_cons, List.length_append, List.length_map,
        List.length_cons, inductionHypothesis]
      simp [Nat.add_mul, Nat.add_comm]

@[simp] theorem product_cardinality
    (left : Support Left)
    (right : Support Right) :
    (left.product right).cardinality =
      left.cardinality * right.cardinality := by
  unfold Support.cardinality product
  exact product_values_length left.values right.values

private theorem product_countP_first
    (leftValues : List Left)
    (rightValues : List Right)
    (event : Left -> Bool) :
    (leftValues.flatMap fun leftSeed =>
      rightValues.map fun rightSeed => (leftSeed, rightSeed)).countP
        (fun seed => event seed.1) =
      leftValues.countP event * rightValues.length := by
  induction leftValues with
  | nil => simp
  | cons leftSeed leftValues inductionHypothesis =>
      cases eventAtSeed : event leftSeed with
      | false =>
          have mappedCount :
              (rightValues.map fun rightSeed =>
                (leftSeed, rightSeed)).countP
                  (fun seed => event seed.1) = 0 := by
            simp [eventAtSeed]
          rw [List.flatMap_cons, List.countP_append, mappedCount,
            inductionHypothesis]
          simp [eventAtSeed]
      | true =>
          have mappedCount :
              (rightValues.map fun rightSeed =>
                (leftSeed, rightSeed)).countP
                  (fun seed => event seed.1) = rightValues.length := by
            simp [eventAtSeed]
          rw [List.flatMap_cons, List.countP_append, mappedCount,
            inductionHypothesis]
          simp [eventAtSeed, Nat.add_mul, Nat.add_comm]

private theorem product_countP_second
    (leftValues : List Left)
    (rightValues : List Right)
    (event : Right -> Bool) :
    (leftValues.flatMap fun leftSeed =>
      rightValues.map fun rightSeed => (leftSeed, rightSeed)).countP
        (fun seed => event seed.2) =
      leftValues.length * rightValues.countP event := by
  induction leftValues with
  | nil => simp
  | cons leftSeed leftValues inductionHypothesis =>
      have mappedCount :
          (rightValues.map fun rightSeed =>
            (leftSeed, rightSeed)).countP
              (fun seed => event seed.2) = rightValues.countP event := by
        rw [List.countP_map]
        change rightValues.countP (fun rightSeed => event rightSeed) = _
        rfl
      rw [List.flatMap_cons, List.countP_append, mappedCount,
        inductionHypothesis]
      simp [Nat.add_mul, Nat.add_comm]

@[simp] theorem product_uniform_countBool_first
    (left : Support Left)
    (right : Support Right)
    (event : Left -> Bool) :
    ((left.product right).uniform).countBool (fun seed => event seed.1) =
      left.uniform.countBool event * right.cardinality := by
  exact product_countP_first left.values right.values event

@[simp] theorem product_uniform_countBool_second
    (left : Support Left)
    (right : Support Right)
    (event : Right -> Bool) :
    ((left.product right).uniform).countBool (fun seed => event seed.2) =
      left.cardinality * right.uniform.countBool event := by
  exact product_countP_second left.values right.values event

private theorem rat_natCast_cardinality_ne_zero
    {Seed : Type uLeft}
    (support : Support Seed) :
    (support.cardinality : Rat) ≠ 0 := by
  exact Rat.ne_of_gt (Rat.natCast_pos.mpr support.cardinality_pos)

/-- The first marginal of the uniform product is exactly the uniform left
support. -/
theorem product_uniform_probabilityBool_first
    (left : Support Left)
    (right : Support Right)
    (event : Left -> Bool) :
    ((left.product right).uniform).probabilityBool (fun seed => event seed.1) =
      left.uniform.probabilityBool event := by
  change
    (((left.product right).uniform.countBool
          (fun seed => event seed.1) : Nat) : Rat) /
        ((left.product right).cardinality : Rat) =
      ((left.uniform.countBool event : Nat) : Rat) /
        (left.cardinality : Rat)
  rw [product_uniform_countBool_first, product_cardinality, Rat.natCast_mul,
    Rat.natCast_mul, Rat.div_def, Rat.inv_mul_rev]
  have rightNonzero := rat_natCast_cardinality_ne_zero right
  calc
    ((left.uniform.countBool event : Rat) * (right.cardinality : Rat)) *
          ((right.cardinality : Rat)⁻¹ * (left.cardinality : Rat)⁻¹) =
        (left.uniform.countBool event : Rat) *
          (((right.cardinality : Rat) * (right.cardinality : Rat)⁻¹) *
            (left.cardinality : Rat)⁻¹) := by
      rw [Rat.mul_assoc (left.uniform.countBool event : Rat)
        (right.cardinality : Rat)
        ((right.cardinality : Rat)⁻¹ * (left.cardinality : Rat)⁻¹)]
      rw [← Rat.mul_assoc (right.cardinality : Rat)
        (right.cardinality : Rat)⁻¹ (left.cardinality : Rat)⁻¹]
    _ = (left.uniform.countBool event : Rat) *
          (left.cardinality : Rat)⁻¹ := by
      rw [Rat.mul_inv_cancel _ rightNonzero, Rat.one_mul]

/-- The second marginal of the uniform product is exactly the uniform right
support. -/
theorem product_uniform_probabilityBool_second
    (left : Support Left)
    (right : Support Right)
    (event : Right -> Bool) :
    ((left.product right).uniform).probabilityBool (fun seed => event seed.2) =
      right.uniform.probabilityBool event := by
  change
    (((left.product right).uniform.countBool
          (fun seed => event seed.2) : Nat) : Rat) /
        ((left.product right).cardinality : Rat) =
      ((right.uniform.countBool event : Nat) : Rat) /
        (right.cardinality : Rat)
  rw [product_uniform_countBool_second, product_cardinality, Rat.natCast_mul,
    Rat.natCast_mul, Rat.div_def, Rat.inv_mul_rev]
  have leftNonzero := rat_natCast_cardinality_ne_zero left
  calc
    ((left.cardinality : Rat) * (right.uniform.countBool event : Rat)) *
          ((right.cardinality : Rat)⁻¹ * (left.cardinality : Rat)⁻¹) =
        (right.uniform.countBool event : Rat) *
          ((right.cardinality : Rat)⁻¹ *
            ((left.cardinality : Rat) * (left.cardinality : Rat)⁻¹)) := by
      calc
        ((left.cardinality : Rat) * (right.uniform.countBool event : Rat)) *
              ((right.cardinality : Rat)⁻¹ *
                (left.cardinality : Rat)⁻¹) =
            ((right.uniform.countBool event : Rat) *
              (left.cardinality : Rat)) *
                ((right.cardinality : Rat)⁻¹ *
                  (left.cardinality : Rat)⁻¹) := by
              rw [Rat.mul_comm (left.cardinality : Rat)]
        _ = (right.uniform.countBool event : Rat) *
              ((left.cardinality : Rat) *
                ((right.cardinality : Rat)⁻¹ *
                  (left.cardinality : Rat)⁻¹)) := by
              rw [Rat.mul_assoc]
        _ = (right.uniform.countBool event : Rat) *
              ((right.cardinality : Rat)⁻¹ *
                ((left.cardinality : Rat) *
                  (left.cardinality : Rat)⁻¹)) := by
              congr 1
              rw [← Rat.mul_assoc, Rat.mul_comm (left.cardinality : Rat),
                Rat.mul_assoc]
    _ = (right.uniform.countBool event : Rat) *
          (right.cardinality : Rat)⁻¹ := by
      rw [Rat.mul_inv_cancel _ leftNonzero, Rat.mul_one]

end Support

end Nightstream.SuperNeo.InteractiveReduction.FiniteUniform
