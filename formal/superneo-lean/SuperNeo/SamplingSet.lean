import SuperNeo.Norm
import SuperNeo.Parameters

/-!
Sampling-set norm boundary layer (Theorem 9 style).

This module provides a theorem-native contract for bounding sampling inputs by a
shared norm cap. It is intentionally lightweight and does not encode probability
semantics; that lives in the protocol/security layers.
-/

namespace SuperNeo

/-- Sampling-set carrier over ring elements. -/
abbrev SamplingCarrier := Coeffs → Prop

/-- Difference set `C - C` used in relaxed-binding analyses. -/
def samplingDiffSet (C : SamplingCarrier) : SamplingCarrier :=
  fun δ => ∃ c1 c2 : Coeffs, C c1 ∧ C c2 ∧ δ = vecAdd c1 (vecScale (-1) c2)

/--
Canonical ring-shaped carrier with an explicit norm cap.

This is a theorem-facing helper used to derive strong-sampling contracts from
operation-level norm-bound bundles.
-/
def ringNormCarrier (K : Nat) : SamplingCarrier :=
  fun c => hasRingDegreeShape c ∧ normInfCoeffs c ≤ K

/--
Strong-sampling expansion-factor contract (Definition 17 / Theorem 9 style):
every `δ ∈ C-C` scales any bounded vector by at most `4*T*B` in `‖·‖∞`.
-/
def strongSamplingExpansionProp
  (C : SamplingCarrier)
  (T : Nat) : Prop :=
  ∀ δ : Coeffs, samplingDiffSet C δ →
    ∀ z : Coeffs, ∀ B : Nat,
      normInfCoeffs z ≤ B →
      normInfCoeffs (mulRqPhi δ z) ≤ 4 * T * B

/-- Both challenge-set and sample-set blocks are bounded by `B`. -/
def samplingNormBoundProp
  (cset samples : Array Coeffs)
  (B : Nat) : Prop :=
  (∀ i : Fin cset.size, normInfCoeffs cset[i] ≤ B) ∧
  (∀ j : Fin samples.size, normInfCoeffs samples[j] ≤ B)

/-- Existential form used by downstream protocol composition. -/
def samplingExpansionProp
  (cset samples : Array Coeffs) : Prop :=
  ∃ B : Nat, samplingNormBoundProp cset samples B

theorem samplingNormBoundProp_left
  {cset samples : Array Coeffs}
  {B : Nat}
  (h : samplingNormBoundProp cset samples B) :
  ∀ i : Fin cset.size, normInfCoeffs cset[i] ≤ B := h.1

theorem samplingNormBoundProp_right
  {cset samples : Array Coeffs}
  {B : Nat}
  (h : samplingNormBoundProp cset samples B) :
  ∀ j : Fin samples.size, normInfCoeffs samples[j] ≤ B := h.2

theorem samplingNormBoundProp_mono
  {cset samples : Array Coeffs}
  {B B' : Nat}
  (h : samplingNormBoundProp cset samples B)
  (hBB' : B ≤ B') :
  samplingNormBoundProp cset samples B' := by
  constructor
  · intro i
    exact Nat.le_trans (h.1 i) hBB'
  · intro j
    exact Nat.le_trans (h.2 j) hBB'

/-- Constructor from explicit per-entry bounds. -/
theorem samplingExpansionProp_of_bounds
  {cset samples : Array Coeffs}
  {B : Nat}
  (hCset : ∀ i : Fin cset.size, normInfCoeffs cset[i] ≤ B)
  (hSamples : ∀ j : Fin samples.size, normInfCoeffs samples[j] ≤ B) :
  samplingExpansionProp cset samples := by
  exact ⟨B, hCset, hSamples⟩

theorem samplingExpansionProp_mono
  {cset samples : Array Coeffs}
  {B B' : Nat}
  (h : samplingNormBoundProp cset samples B)
  (hBB' : B ≤ B') :
  samplingExpansionProp cset samples := by
  exact ⟨B', samplingNormBoundProp_mono h hBB'⟩

theorem samplingExpansionProp_empty :
  samplingExpansionProp (#[] : Array Coeffs) (#[] : Array Coeffs) := by
  refine ⟨0, ?_⟩
  constructor
  · intro i
    exact False.elim (Nat.not_lt_zero _ i.2)
  · intro j
    exact False.elim (Nat.not_lt_zero _ j.2)

theorem strongSamplingExpansionProp_mono
  {C : SamplingCarrier}
  {T T' : Nat}
  (h : strongSamplingExpansionProp C T)
  (hTT' : T ≤ T') :
  strongSamplingExpansionProp C T' := by
  intro δ hδ z B hB
  have hδz : normInfCoeffs (mulRqPhi δ z) ≤ 4 * T * B := h δ hδ z B hB
  have hBound : 4 * T * B ≤ 4 * T' * B := by
    exact Nat.mul_le_mul_right B (Nat.mul_le_mul_left 4 hTT')
  exact Nat.le_trans hδz hBound

theorem expansionFactor_of_strongSampling
  {C : SamplingCarrier}
  {T : Nat}
  (h : strongSamplingExpansionProp C T)
  {δ z : Coeffs}
  (hδ : samplingDiffSet C δ)
  {B : Nat}
  (hB : normInfCoeffs z ≤ B) :
  normInfCoeffs (mulRqPhi δ z) ≤ 4 * T * B :=
  h δ hδ z B hB

theorem strongSamplingExpansionProp_of_ringNormCarrier
  {K T D : Nat}
  (hSub : coeffSubNormBoundFromOperands K K D)
  (hMul : ∀ B : Nat, mulRqPhiNormBoundFromOperands D B (4 * T * B)) :
  strongSamplingExpansionProp (ringNormCarrier K) T := by
  intro δ hδ z B hB
  rcases hδ with ⟨c1, c2, hc1, hc2, rfl⟩
  rcases hc1 with ⟨hc1Shape, hc1Norm⟩
  rcases hc2 with ⟨hc2Shape, hc2Norm⟩
  have hShapeEq : c1.size = c2.size := hc1Shape.trans hc2Shape.symm
  have hDeltaNorm : normInfCoeffs (vecAdd c1 (vecScale (-1) c2)) ≤ D := by
    exact hSub c1 c2 hShapeEq hc1Norm hc2Norm
  exact hMul B (vecAdd c1 (vecScale (-1) c2)) z hDeltaNorm hB

abbrev paperCarrier : SamplingCarrier :=
  ringNormCarrier 2

/-- Any paper-carrier difference remains ring-shaped and has infinity norm at most `4`. -/
theorem samplingDiffSet_paperCarrier_hasRingDegreeShape_and_norm_le_four
  {δ : Coeffs}
  (hδ : samplingDiffSet paperCarrier δ) :
  hasRingDegreeShape δ ∧ normInfCoeffs δ ≤ 4 := by
  rcases hδ with ⟨c1, c2, hc1, hc2, rfl⟩
  rcases hc1 with ⟨hc1Shape, hc1Norm⟩
  rcases hc2 with ⟨hc2Shape, hc2Norm⟩
  have hSizeEq : c1.size = c2.size := by
    calc
      c1.size = d := hc1Shape
      _ = c2.size := hc2Shape.symm
  have hScaledEq : c1.size = (vecScale (-1) c2).size := by
    simpa [vecScale_size] using hSizeEq
  constructor
  · unfold hasRingDegreeShape
    exact (vecAdd_size_of_eq hScaledEq).trans hc1Shape
  · exact coeffSubNormBoundFromOperands_two_two_four c1 c2 hSizeEq hc1Norm hc2Norm

theorem strongSamplingExpansionProp_of_paperCarrier
  {T D : Nat}
  (hSub : coeffSubNormBoundFromOperands 2 2 D)
  (hMul : ∀ B : Nat, mulRqPhiNormBoundFromOperands D B (4 * T * B)) :
  strongSamplingExpansionProp paperCarrier T := by
  simpa [paperCarrier] using
    (strongSamplingExpansionProp_of_ringNormCarrier (K := 2) (T := T) (D := D) hSub hMul)

theorem strongSamplingExpansionProp_paperCarrier_of_three_d_le
  {T : Nat}
  (hTd : 3 * d ≤ T) :
  strongSamplingExpansionProp paperCarrier T := by
  apply strongSamplingExpansionProp_of_paperCarrier (T := T) (D := 4)
  · exact coeffSubNormBoundFromOperands_two_two_four
  · intro B
    exact mulRqPhiNormBoundFromOperands_four_of_three_d_le (T := T) (B := B) hTd

theorem strongSamplingExpansionProp_paperCarrier_concrete :
  strongSamplingExpansionProp paperCarrier Parameters.Goldilocks.T := by
  have hTd : 3 * d ≤ Parameters.Goldilocks.T := by
    rw [SuperNeo.d_eq_54, Parameters.Goldilocks.T_eq_216]
    decide
  simpa using
    (strongSamplingExpansionProp_paperCarrier_of_three_d_le
      (T := Parameters.Goldilocks.T) hTd)

/-! Compatibility check surface retained for protocol-level glue. -/

/--
Executable compatibility check for sampling expansion obligations.

This compact scaffold reflects the theorem-facing proposition directly.
-/
noncomputable def samplingSetBoundCheck
  (cset samples : Array Coeffs) : Bool := by
  classical
  exact decide (samplingExpansionProp cset samples)

theorem samplingSetBoundCheck_sound
  {cset samples : Array Coeffs}
  (hOk : samplingSetBoundCheck cset samples = true) :
  samplingExpansionProp cset samples := by
  classical
  unfold samplingSetBoundCheck at hOk
  exact decide_eq_true_eq.mp hOk

theorem samplingSetBoundCheck_complete
  {cset samples : Array Coeffs}
  (hProp : samplingExpansionProp cset samples) :
  samplingSetBoundCheck cset samples = true := by
  classical
  unfold samplingSetBoundCheck
  exact decide_eq_true hProp

theorem samplingSetBoundCheck_iff
  {cset samples : Array Coeffs} :
  samplingSetBoundCheck cset samples = true ↔ samplingExpansionProp cset samples := by
  constructor
  · exact samplingSetBoundCheck_sound
  · exact samplingSetBoundCheck_complete


end SuperNeo
