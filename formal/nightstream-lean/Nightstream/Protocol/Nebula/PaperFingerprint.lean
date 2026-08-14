import Nightstream.Protocol.Nebula.Types
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConcreteCarrier.Algebra

/-!
Contract: literal Nebula Corollary 8 fingerprint semantics.

Assurance tier: paper-comparison model.

Owns the paper formula
`a + gamma1 * v + gamma1^2 * t - gamma2`, its commutative list product,
and its exact collision boundary.

This formula is not the packed two-coordinate production fingerprint in
`Fingerprint`. The two modules model different encodings of the same memory
record. This module does not authorize the production transcript, circuit,
or Rust implementation.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Protocol.Nebula.PaperFingerprint

open Nightstream.Protocol.Nebula
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConcreteCarrier

/-- Challenges in the literal paper formula. They are separate from the
production polynomial-coordinate challenge type. -/
structure Challenges where
  gamma1 : K
  gamma2 : K
deriving DecidableEq, Repr

/-- Canonical base-field image of the paper address coordinate. -/
def addressField (entry : MemTuple) : F :=
  ⟨entry.globalIndex % goldilocksModulus,
    Nat.mod_lt _ (by decide)⟩

/-- Canonical base-field image of the paper timestamp coordinate. -/
def timestampField (entry : MemTuple) : F :=
  ⟨entry.timestamp % goldilocksModulus,
    Nat.mod_lt _ (by decide)⟩

/-- Canonical base-field image of the memory value coordinate. -/
def valueField (entry : MemTuple) : F :=
  ⟨entry.value % goldilocksModulus,
    Nat.mod_lt _ (by decide)⟩

/-- Literal Nebula Corollary 8 fingerprint
`a + gamma1 * v + gamma1^2 * t - gamma2`. -/
def fingerprint (challenges : Challenges) (entry : MemTuple) : K :=
  K.sub
    (K.add
      (K.add
        (K.embed (addressField entry))
        (K.mul challenges.gamma1 (K.embed (valueField entry))))
      (K.mul
        (K.mul challenges.gamma1 challenges.gamma1)
        (K.embed (timestampField entry))))
    challenges.gamma2

/-- Product of paper fingerprints. The empty product is one. -/
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

/-- The literal paper product is independent of list order. -/
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

/-- Named paper bad event. This definition assigns no probability. -/
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

end Nightstream.Protocol.Nebula.PaperFingerprint
