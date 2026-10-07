import Mathlib.Algebra.QuadraticAlgebra.Defs
import Mathlib.Algebra.Ring.TransferInstance
import NightstreamFPrime.Spec.GoldilocksExtension
import NightstreamFPrime.Spec.FieldTower
import NightstreamFPrime.Spec.Nebula.RetryAveraging

/-! Owns security note Lemma 3 on the trunk challenge field
`K = F_q[U]/(U² − 7)`: part 1, `ε_test ≤ 2·m / q²`, and part 2 for one segment.
The ring structure on `K` is a scoped instance block, so a consumer opens
`NightstreamFPrime.Spec.Nebula.GoldilocksFingerprint` to state Lemma 3 at
`K`. -/

namespace NightstreamFPrime.Spec.Nebula.GoldilocksFingerprint

/-- `K` as Mathlib's quadratic algebra over `ZMod q`. -/
def carrierEquiv : K ≃ QuadraticAlgebra (ZMod goldilocksModulus) 7 0 where
  toFun value := ⟨value.c0, value.c1⟩
  invFun value := ⟨value.re, value.im⟩
  left_inv _ := rfl
  right_inv _ := rfl

scoped instance : CommRing K := carrierEquiv.commRing

private theorem zero_eq : (0 : K) = K.zero := rfl

private theorem mul_eq (left right : K) : left * right = K.mul left right := by
  change K.mk _ (left.c0 * right.c1 + left.c1 * right.c0 + 0 * left.c1 * right.c1) = _
  simp only [Fin.zero_mul, Fin.add_zero]
  rfl

scoped instance : Nontrivial K := ⟨⟨K.zero, K.one, by
  intro same
  have : (0 : F) = 1 := congrArg K.c0 same
  exact (by decide : (0 : F) ≠ 1) this⟩⟩

scoped instance : NoZeroDivisors K where
  eq_zero_or_eq_zero_of_mul_eq_zero := by
    intro left right productZero
    exact GoldilocksExtension.extensionNoZeroDivisors left right
      (by simpa only [mul_eq, zero_eq] using productZero)

scoped instance : IsDomain K := NoZeroDivisors.to_isDomain K

/-- `K` is finite: one element per pair of coordinates. -/
noncomputable scoped instance kFintype : Fintype K := Fintype.ofEquiv (F × F) {
  toFun value := ⟨value.1, value.2⟩
  invFun value := (value.c0, value.c1)
  left_inv _ := rfl
  right_inv _ := rfl }

scoped instance : CharP K goldilocksModulus := by
  haveI : CharP (QuadraticAlgebra (ZMod goldilocksModulus) 7 0) goldilocksModulus :=
    charP_of_injective_algebraMap QuadraticAlgebra.algebraMap_injective _
  exact charP_of_injective_ringHom (f := carrierEquiv.ringEquiv.symm.toRingHom)
    carrierEquiv.ringEquiv.symm.injective _

/-- `K` has `q²` elements. -/
theorem card_K : Fintype.card K = goldilocksModulus ^ 2 := by
  rw [← Nat.card_eq_fintype_card]
  exact FieldTower.extension_cardinality

/-- Lemma 3 part 1 on `K`. -/
theorem goldilocks_badChallenges_frequency {A B : Multiset Tuple} {m : ℕ} (different : A ≠ B)
    (small : ∀ τ ∈ A + B, τ.Small) (sizeA : Multiset.card A ≤ m)
    (sizeB : Multiset.card B ≤ m) :
    ((BadChallenges (E := K) A B).card : ℚ≥0) / (Fintype.card (K × K) : ℚ≥0) ≤
      2 * m / (goldilocksModulus ^ 2 : ℕ) := by
  have bound := badChallenges_frequency (E := K) different small sizeA sizeB
  rwa [card_K] at bound

open Classical in
/-- Lemma 3 part 2 on `K` for one segment of the translated game, with
`ε_test = 2·m_mem / q²`: `Pr[Err] ≤ ε_test + Pr[Err and the retry disagrees]`,
and every disagreeing pair gives a collision among the two calls' chain
inputs. -/
theorem goldilocks_segment_fingerprint_bound {Digest : Type} {ctx : Context K Digest}
    (valid : ctx.plan.Valid) (inp : EtaInput Digest) {Coins : Type} [Fintype Coins]
    (play : (K × K) × Coins → Option (List StepRecords))
    (closes : ∀ c z, play c = some z → (inp.view z).ClosesAt ctx c.1) :
    ((errors play fun z => ¬ ((inp.view z).multisets ctx.plan).Balanced).card : ℚ≥0) /
        Fintype.card ((K × K) × Coins) ≤
      2 * (ctx.plan.maxTuples : ℚ≥0) / (goldilocksModulus ^ 2 : ℕ) +
        ((disagreements play fun z => ¬ ((inp.view z).multisets ctx.plan).Balanced).card :
            ℚ≥0) /
          ((Fintype.card ((K × K) × Coins) : ℚ≥0) * (successes play).card) ∧
      ∀ q ∈ disagreements play (fun z => ¬ ((inp.view z).multisets ctx.plan).Balanced),
        ∃ z z', play q.1 = some z ∧ play q.2 = some z' ∧
          CollisionIn ctx.hash ((inp.view z).chainInputs ctx) ((inp.view z').chainInputs ctx) := by
  have bound := segment_fingerprint_bound valid inp play closes
  rwa [card_K] at bound

end NightstreamFPrime.Spec.Nebula.GoldilocksFingerprint
