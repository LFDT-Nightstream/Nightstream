import Nightstream.Implementation.R1CS.Canonical.KPolyEval
import Nightstream.Implementation.R1CS.Canonical.KBridge
import Nightstream.SuperNeo.Concrete.Algebra

/-!
Contract: the semantic layer's `K` is the canonical track's `Pair`.

Owns: the coordinate map from `SuperNeo.Concrete.K` to `KHorner.Pair` and the
proofs that it carries extension addition and multiplication.

Does not own: agreement of `polyMul` with `rawMulCoeffK`, which needs this plus
an indexing argument; or any row program.

## The third `K`

Cycle 255 mapped three `K`-like algebras: `KHorner.Pair` in the row layer,
`ProjectionProgram.K` in the R1CS projection layer, and `SuperNeo.Concrete.K` in
the semantic layer where the frozen combine lives. `KBridge` connected the first
two. This connects the third.

`goldilocksModulus` and `goldilocksP` are the same literal, so the two `K`
structures are isomorphic — but they are distinct Lean types, and a conversion
has to be written rather than assumed. That distinction is the whole reason this
module exists.

## One parenthesization difference

`Concrete.K.mul` writes `7 * a.c1 * b.c1`, left-associated;
`ProjectionProgram.K.mul` and `mulPair` write `7 * (a.c1 * b.c1)`. Same value,
different term, so the proof has to reassociate rather than close by `rfl`.
Worth noting because it is exactly the kind of difference that looks like
nothing and blocks a `simp`.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Canonical.KConcreteBridge

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.KHorner
open Nightstream.SuperNeo.Concrete

/-- The two moduli are the same literal.  Separate `def`s, so this is needed
explicitly wherever the two layers meet. -/
theorem moduli_eq : goldilocksModulus = goldilocksP := rfl

/-- The coordinates of a semantic extension element, as naturals. -/
def ofConcrete (value : K) : Pair where
  low := value.c0.val
  high := value.c1.val

/-- Coordinate translation is injective, so a row-layer equality can be
transported back to the semantic concrete extension carrier. -/
theorem ofConcrete_injective
    {left right : K} (equal : ofConcrete left = ofConcrete right) :
    left = right := by
  cases left with
  | mk leftLow leftHigh =>
      cases right with
      | mk rightLow rightHigh =>
          simp only [ofConcrete, Pair.mk.injEq] at equal
          simp only [K.mk.injEq]
          exact ⟨Fin.ext equal.1, Fin.ext equal.2⟩

theorem ofConcrete_zero : ofConcrete K.zero = ⟨0, 0⟩ := rfl

theorem ofConcrete_add (x y : K) :
    ofConcrete (K.add x y) = addPair (ofConcrete x) (ofConcrete y) := by
  unfold ofConcrete K.add addPair
  simp only [Fin.val_add, moduli_eq]

/-- `Fin` subtraction is the complement-and-reduce that `subPair` performs, so
the bridge carries it.  Needed because `ringKMul`'s reduction subtracts. -/
theorem ofConcrete_sub (x y : K) :
    ofConcrete (K.sub x y)
      = KPairLaws.subPair (ofConcrete x) (ofConcrete y) := by
  have component : ∀ a b : F,
      (a - b).val
        = (a.val + (goldilocksP - b.val % goldilocksP)) % goldilocksP := by
    intro a b
    have residue : b.val % goldilocksP = b.val :=
      Nat.mod_eq_of_lt (by rw [← moduli_eq]; exact b.isLt)
    rw [residue, Nat.add_comm a.val (goldilocksP - b.val)]
    rfl
  unfold ofConcrete K.sub KPairLaws.subPair
  simp only [component]

theorem ofConcrete_mul (x y : K) :
    ofConcrete (K.mul x y) = mulPair (ofConcrete x) (ofConcrete y) := by
  have seven : (7 : Fin goldilocksModulus).val = 7 := rfl
  unfold ofConcrete K.mul mulPair
  simp only [Fin.val_add, Fin.val_mul, seven, moduli_eq, Pair.mk.injEq]
  refine ⟨?_, ?_⟩
  · have strip :
        (x.c0.val * y.c0.val % goldilocksP
            + 7 * x.c1.val % goldilocksP * y.c1.val % goldilocksP) % goldilocksP
          = (x.c0.val * y.c0.val + 7 * x.c1.val * y.c1.val) % goldilocksP := by
      refine KPairLaws.add_congr _ _ _ _ (KPairLaws.mod_self_congr _) ?_
      rw [Nat.mod_mod]
      exact KPairLaws.mul_congr _ _ _ _ (KPairLaws.mod_self_congr _) rfl
    rw [strip, Nat.mul_assoc]
  · rw [← Nat.add_mod]

/-- **The two bridges agree.**  Going through the semantic `K` and through the
projection `K` lands on the same `Pair`, so nothing depends on which route a
value took. -/
theorem ofConcrete_agrees_with_toPair
    (semantic : K) (projection : ProjectionProgram.K)
    (sameLow : semantic.c0.val = projection.c0.val)
    (sameHigh : semantic.c1.val = projection.c1.val) :
    ofConcrete semantic = KBridge.toPair projection := by
  unfold ofConcrete KBridge.toPair
  simp only [Pair.mk.injEq]
  exact ⟨sameLow, sameHigh⟩

end Nightstream.Implementation.R1CS.Canonical.KConcreteBridge
