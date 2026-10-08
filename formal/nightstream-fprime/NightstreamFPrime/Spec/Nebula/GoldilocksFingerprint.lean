import NightstreamFPrime.Spec.GoldilocksExtensionRing
import NightstreamFPrime.Spec.Nebula.RetryAveraging

/-! Owns security note Lemma 3 on the trunk challenge field
`K = F_q[U]/(U² − 7)`: part 1, `ε_test ≤ 2·m / q²`, and part 2 for one segment.
The ring structure on `K` is the scoped instance block of
`GoldilocksExtensionRing`. -/

namespace NightstreamFPrime.Spec.Nebula.GoldilocksFingerprint

open scoped NightstreamFPrime.Spec.GoldilocksExtensionRing
open NightstreamFPrime.Spec.GoldilocksExtensionRing (card_K)

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
    (valid : ctx.plan.Valid) (inp : EtaInput Digest) (k : ℕ) {Coins : Type} [Fintype Coins]
    (play : (K × K) × Coins → Option (List StepRecords))
    (closes : ∀ c z, play c = some z → (inp.view k z).ClosesAt ctx c.1) :
    ((errors play fun z => ¬ ((inp.view k z).multisets ctx.plan).Balanced).card : ℚ≥0) /
        Fintype.card ((K × K) × Coins) ≤
      2 * (ctx.plan.maxTuples : ℚ≥0) / (goldilocksModulus ^ 2 : ℕ) +
        ((disagreements play fun z => ¬ ((inp.view k z).multisets ctx.plan).Balanced).card :
            ℚ≥0) /
          ((Fintype.card ((K × K) × Coins) : ℚ≥0) * (successes play).card) ∧
      ∀ q ∈ disagreements play (fun z => ¬ ((inp.view k z).multisets ctx.plan).Balanced),
        ∃ z z', play q.1 = some z ∧ play q.2 = some z' ∧
          CollisionIn ctx.hash ((inp.view k z).chainInputs ctx) ((inp.view k z').chainInputs ctx) := by
  have bound := segment_fingerprint_bound valid inp k play closes
  rwa [card_K] at bound

end NightstreamFPrime.Spec.Nebula.GoldilocksFingerprint
