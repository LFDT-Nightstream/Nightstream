import Nightstream.Protocol.Nebula.Memory

/-!
Contract: literal Nebula Corollary 8 fingerprint semantics.

Assurance tier: model-level.

Owns the paper formula
`a + gamma1 * v + gamma1^2 * t - gamma2`, its commutative list product,
the exact bad-event boundary, and honest completeness from the fingerprint-
independent multiset identity in `Memory.executes_perm`.

This module is separate from `Fingerprint`, which owns Nightstream's packed
production variant. It does not own commitment binding, challenge derivation,
collision probability, Layer-2 finalization, circuit rows, or Rust.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Protocol.Nebula.PaperFingerprint

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConcreteCarrier
open Nightstream.Protocol.Nebula.Fingerprint
open Nightstream.Protocol.Nebula.Memory

/-- Canonical base-field image of the paper address coordinate. -/
def addressField (entry : MemTuple) : F :=
  ⟨entry.globalIndex % goldilocksModulus,
    Nat.mod_lt _ (by decide)⟩

/-- Canonical base-field image of the paper timestamp coordinate. -/
def timestampField (entry : MemTuple) : F :=
  ⟨entry.timestamp % goldilocksModulus,
    Nat.mod_lt _ (by decide)⟩

/-- Literal Nebula Corollary 8 fingerprint
`a + gamma1 * v + gamma1^2 * t - gamma2`. -/
def fingerprint (challenges : Challenges) (entry : MemTuple) : K :=
  K.sub
    (K.add
      (K.add
        (K.embed (addressField entry))
        (K.mul challenges.gamma1
          (K.embed (Fingerprint.valueField entry))))
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

/-- Paper product order: `[read, write, initial, final]`. -/
def products
    (challenges : Challenges)
    (initial : List MemTuple) (accesses : List Access)
    (final : List MemTuple) : Fin 4 → K
  | ⟨0, _⟩ => product challenges (readTuples accesses)
  | ⟨1, _⟩ => product challenges (writeTuples accesses)
  | ⟨2, _⟩ => product challenges initial
  | _ => product challenges final

/-- Honest execution satisfies Corollary 8's product equation for every
challenge. This is the probability-one completeness direction only. -/
theorem executes_balanced
    (challenges : Challenges)
    {initial final : List MemTuple}
    {timestampIn timestampOut : Nat}
    {accesses : List Access}
    (execution : Executes initial timestampIn accesses final timestampOut) :
    Balanced (products challenges initial accesses final) := by
  have exactProduct :=
    product_perm challenges (Memory.executes_perm execution)
  rw [product_append, product_append] at exactProduct
  exact exactProduct

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
