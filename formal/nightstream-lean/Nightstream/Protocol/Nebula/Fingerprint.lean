import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConcreteCarrier.Algebra

/-!
Contract: Lean-owned public-coin fingerprint semantics for Nebula memory.

Assurance tier: model-level.

Owns the packed timestamp/address map, the selected two-challenge
fingerprint over the production Goldilocks extension, list products, and the
exact bad-event boundary between multiset equality and a fingerprint
collision.

Does not own challenge derivation, memory-transition semantics, circuit rows,
commitments, Rust layouts, or a probability bound for the collision event.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Protocol.Nebula.Fingerprint

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConcreteCarrier

/-- Width of every Nebula timestamp. -/
def timestampBits : Nat := 44

/-- Radix that separates the timestamp from the global cell index. -/
def timestampRadix : Nat := 2 ^ timestampBits

/-- Verifier-derived challenges for one committed memory segment. -/
structure Challenges where
  gamma1 : K
  gamma2 : K
deriving DecidableEq, Repr

/-- One memory-multiset element: timestamp, global cell index, and value. -/
structure MemTuple where
  timestamp : Nat
  globalIndex : Nat
  value : Nat
deriving DecidableEq, Repr

/-- Integer packing before reduction into the base field. -/
def packedNat (entry : MemTuple) : Nat :=
  entry.timestamp + timestampRadix * entry.globalIndex

/-- Canonical base-field image of the packed timestamp and cell index. -/
def packed (entry : MemTuple) : F :=
  ⟨packedNat entry % goldilocksModulus,
    Nat.mod_lt _ (by decide)⟩

/-- Canonical base-field image of a memory value. -/
def valueField (entry : MemTuple) : F :=
  ⟨entry.value % goldilocksModulus,
    Nat.mod_lt _ (by decide)⟩

/-- The selected compact fingerprint
`gamma2 - (packed(timestamp,index) + gamma1 * value)`.

The sign does not affect multiset equality.  Packing is a separate injective
map under the explicit bounds below. -/
def fingerprint (challenges : Challenges) (entry : MemTuple) : K :=
  K.sub challenges.gamma2
    (K.add (K.embed (packed entry))
      (K.mul challenges.gamma1 (K.embed (valueField entry))))

/-- Product of entry fingerprints.  The empty product is one. -/
def product (challenges : Challenges) : List MemTuple → K
  | [] => K.one
  | entry :: rest =>
      K.mul (fingerprint challenges entry) (product challenges rest)

private theorem k_mul_assoc (left middle right : K) :
    K.mul (K.mul left middle) right = K.mul left (K.mul middle right) :=
  extensionLaws.mul_assoc left middle right

private theorem k_mul_comm (left right : K) :
    K.mul left right = K.mul right left :=
  extensionLaws.mul_comm left right

private theorem k_one_mul (value : K) : K.mul K.one value = value :=
  extensionLaws.one_mul value

private theorem k_mul_one (value : K) : K.mul value K.one = value :=
  extensionLaws.mul_one value

theorem product_append
    (challenges : Challenges) (left right : List MemTuple) :
    product challenges (left ++ right) =
      K.mul (product challenges left) (product challenges right) := by
  induction left with
  | nil =>
      simp only [List.nil_append, product]
      exact (k_one_mul _).symm
  | cons head tail inductionHypothesis =>
      simp only [List.cons_append, product, inductionHypothesis]
      exact (k_mul_assoc _ _ _).symm

theorem product_perm
    (challenges : Challenges) {left right : List MemTuple}
    (permutation : left.Perm right) :
    product challenges left = product challenges right := by
  induction permutation with
  | nil => rfl
  | cons head permutation inductionHypothesis =>
      simp only [product, inductionHypothesis]
  | swap left right rest =>
      simp only [product]
      calc
        K.mul (fingerprint challenges right)
            (K.mul (fingerprint challenges left)
              (product challenges rest)) =
          K.mul
            (K.mul (fingerprint challenges right)
              (fingerprint challenges left))
            (product challenges rest) :=
              (k_mul_assoc _ _ _).symm
        _ = K.mul
            (K.mul (fingerprint challenges left)
              (fingerprint challenges right))
            (product challenges rest) := by
              rw [k_mul_comm (fingerprint challenges right)]
        _ = K.mul (fingerprint challenges left)
            (K.mul (fingerprint challenges right)
              (product challenges rest)) :=
              k_mul_assoc _ _ _
  | trans first second firstHypothesis secondHypothesis =>
      exact firstHypothesis.trans secondHypothesis

/-- Bounds under which timestamp/address packing is injective before field
reduction. -/
def PackingValid (entry : MemTuple) : Prop :=
  entry.timestamp < timestampRadix ∧
    packedNat entry < goldilocksModulus ∧
    entry.value < goldilocksModulus

theorem packedNat_injective
    {left right : MemTuple}
    (leftTimestamp : left.timestamp < timestampRadix)
    (rightTimestamp : right.timestamp < timestampRadix)
    (equal : packedNat left = packedNat right) :
    left.timestamp = right.timestamp ∧
      left.globalIndex = right.globalIndex := by
  unfold packedNat at equal
  simp only [timestampRadix, timestampBits] at leftTimestamp rightTimestamp equal
  omega

theorem packed_injective
    {left right : MemTuple}
    (leftValid : PackingValid left)
    (rightValid : PackingValid right)
    (equal : packed left = packed right) :
    left.timestamp = right.timestamp ∧
      left.globalIndex = right.globalIndex := by
  have equalValues := congrArg Fin.val equal
  change packedNat left % goldilocksModulus =
    packedNat right % goldilocksModulus at equalValues
  rw [Nat.mod_eq_of_lt leftValid.2.1,
    Nat.mod_eq_of_lt rightValid.2.1] at equalValues
  exact packedNat_injective leftValid.1 rightValid.1 equalValues

/-- Named security event: two different multisets have the same product at
the verifier-selected challenge.  This definition does not assign a
probability to the event. -/
def Collision (challenges : Challenges) : Prop :=
  ∃ left right : List MemTuple,
    ¬ left.Perm right ∧
      product challenges left = product challenges right

theorem exact_or_collision_of_equal_product
    (challenges : Challenges)
    (left right : List MemTuple)
    (equal : product challenges left = product challenges right) :
    left.Perm right ∨ Collision challenges := by
  by_cases exact : left.Perm right
  · exact Or.inl exact
  · exact Or.inr ⟨left, right, exact, equal⟩

end Nightstream.Protocol.Nebula.Fingerprint
