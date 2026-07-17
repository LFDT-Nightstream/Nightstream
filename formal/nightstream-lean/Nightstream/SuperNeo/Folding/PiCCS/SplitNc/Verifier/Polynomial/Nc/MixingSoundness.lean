import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.InitialSum
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConcreteCarrier.NoZeroDivisors

/-!
Deterministic bad-event decomposition for paper-relative Split-NC mixing.

Protocol: SuperNeo `Pi_CCS`, split NC branch.
Phase: semantic soundness of the equality selectors and source gamma mixture.
Constraint family: no emitted rows; this file classifies why a zero mixed
residual may fail to imply the independent full-carrier norm relation.

Owns: the selector-root event, the paper-relative gamma-root event, the extra
Split-V1 gamma-zero event, zero mixtures from pointwise-zero source
specializations, and exact truth-or-root decompositions for both schedules.

Does not own: probability bounds, challenge sampling, transcript derivation,
SumCheck messages, the concrete Goldilocks primality/nonresidue certificates,
`yZcol`, terminal/output binding, Rust, R1CS, row emission, row removal, or
constraint counts.

Emits constraints: no.

Authority boundary: `SelectorRoot` says a nonzero full Boolean residual table
vanished under both equality-selector specializations. `GammaRoot` says at
least one specialized source residual was nonzero but their paper-relative
gamma mixture vanished. These are deterministic propositions, not claims
about sampling probability.

Split-V1 is one common gamma factor times the paper-relative mixture. This
file derives the required `K` cancellation from explicit base-field and
seven-nonresidue premises rather than assuming a verifier conclusion.
`gamma = 0` remains a distinct unconditional deterministic root.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_ccs.nc.soundness.selector_root` | Boolean relation nonzero, yet every source specialization is zero | security boundary | `SelectorRoot` |
| `nifs.pi_ccs.nc.soundness.gamma_root` | some source specialization is nonzero, yet paper-relative gamma mixing is zero | security boundary | `GammaRoot` |
| `nifs.pi_ccs.nc.soundness.all_source_zero` | all source specializations zero implies every convention's mix is zero | derived | `mixedResidualAtBeta_eq_zero_of_all_source_specializations_zero` |
| `nifs.pi_ccs.nc.soundness.paper_nc` | zero paper-relative mixture iff truth or one named deterministic root | derived | `paperNc_mixedResidualAtBeta_eq_zero_iff_truth_or_selectorRoot_or_gammaRoot` |
| `nifs.pi_ccs.nc.soundness.split_v1_gamma_zero` | Split-V1's extra common gamma is zero | security boundary | `SplitV1GammaZero` |
| `nifs.pi_ccs.nc.soundness.split_v1` | zero Split-V1 mixture iff truth or one of three named roots | derived | `splitV1_mixedResidualAtBeta_eq_zero_iff_truth_or_selectorRoot_or_gammaRoot_or_gammaZero` |
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.MixingSoundness

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.Mixing
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.InitialSum

private abbrev ops := ConcreteCarrier.extensionOps
private abbrev laws := ConcreteCarrier.extensionLaws

/-- Equality selectors missed a nonzero Boolean residual relation: the full
source-derived Boolean table is not identically zero, but every source's
specialization at `betaM` and `betaA` is zero. -/
structure SelectorRoot
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Coins domain) : Prop where
  relationNonzero :
    ¬ SourceProjection.BooleanResidualsZero covers data
  everySourceSpecializationZero :
    ∀ source, sourceResidualAtBeta covers data coins source = K.zero

/-- Gamma compression canceled at least one nonzero source specialization
under the paper-relative exponent schedule. -/
structure GammaRoot
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Coins domain) : Prop where
  someSourceSpecializationNonzero :
    ∃ source, sourceResidualAtBeta covers data coins source ≠ K.zero
  paperNcMixtureZero :
    mixedResidualAtBeta .paperNc covers data coins = K.zero

/-- The production Split-V1 exponent schedule adds one common gamma factor.
When that factor is zero, every source mixture vanishes independently of the
relation. -/
structure SplitV1GammaZero
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (_covers : domain.Covers shape)
    (_data : Data shape)
    (coins : Coins domain) : Prop where
  gammaZero : coins.gamma = K.zero

/-- If every independently derived source specialization is zero, gamma
compression is zero under every explicitly named exponent convention. -/
theorem mixedResidualAtBeta_eq_zero_of_all_source_specializations_zero
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (convention : GammaConvention)
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Coins domain)
    (allZero :
      ∀ source, sourceResidualAtBeta covers data coins source = K.zero) :
    mixedResidualAtBeta convention covers data coins = K.zero := by
  unfold mixedResidualAtBeta
  calc
    FiniteSumAlgebra.sumMap ops
        (canonicalFinIndices shape.sourceCount) _ =
      FiniteSumAlgebra.sumMap ops
        (canonicalFinIndices shape.sourceCount) (fun _ => ops.zero) := by
          apply FiniteSumAlgebra.sumMap_congr
          intro source _
          unfold SignedJointIdentity.gammaTerm
          rw [allZero source]
          exact laws.mul_zero _
    _ = ops.zero := FiniteSumAlgebra.sumMap_zero ops laws _

/-- Exact deterministic decomposition of a zero paper-relative NC source
mixture. No probability, transcript, or SumCheck acceptance claim is present:
an invalid relation must have been hidden either by the equality selectors or
by gamma compression. -/
theorem paperNc_mixedResidualAtBeta_eq_zero_iff_truth_or_selectorRoot_or_gammaRoot
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Coins domain) :
    mixedResidualAtBeta .paperNc covers data coins = K.zero ↔
      Semantics.Nc.Truth data ∨
        SelectorRoot covers data coins ∨ GammaRoot covers data coins := by
  classical
  constructor
  · intro mixtureZero
    by_cases truth : Semantics.Nc.Truth data
    · exact Or.inl truth
    · have relationNonzero :
          ¬ SourceProjection.BooleanResidualsZero covers data := by
        intro relationZero
        exact truth
          ((SourceProjection.booleanResidualsZero_iff_truth
            noZeroDivisors covers data).mp relationZero)
      by_cases allZero :
          ∀ source, sourceResidualAtBeta covers data coins source = K.zero
      · exact Or.inr <| Or.inl {
          relationNonzero := relationNonzero
          everySourceSpecializationZero := allZero }
      · apply Or.inr
        apply Or.inr
        refine {
          paperNcMixtureZero := mixtureZero
          someSourceSpecializationNonzero := ?_ }
        exact Classical.not_forall.mp allZero
  · rintro (truth | selectorRoot | gammaRoot)
    · exact mixedResidualAtBeta_eq_zero_of_truth
        .paperNc covers data coins truth
    · exact mixedResidualAtBeta_eq_zero_of_all_source_specializations_zero
        .paperNc covers data coins
        selectorRoot.everySourceSpecializationZero
    · exact gammaRoot.paperNcMixtureZero

/-- The independently grouped Split-V1 source sum is exactly one common gamma
factor times the paper-relative source sum. -/
theorem splitV1_mixedResidualAtBeta_eq_gamma_mul_paperNc
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Coins domain) :
    mixedResidualAtBeta .splitV1 covers data coins =
      K.mul coins.gamma
        (mixedResidualAtBeta .paperNc covers data coins) := by
  unfold mixedResidualAtBeta
  exact splitV1Sum_eq_gamma_mul_paperNcSum coins.gamma _

/-- Exact deterministic Split-V1 decomposition. Besides semantic NC truth,
an accepting zero can arise from the equality selectors, paper-relative gamma
compression, or the extra common Split-V1 gamma factor. The theorem is
model-level: the two number-theoretic premises are explicit, and no transcript
probability or production refinement is claimed. -/
theorem splitV1_mixedResidualAtBeta_eq_zero_iff_truth_or_selectorRoot_or_gammaRoot_or_gammaZero
    (baseNoZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    (sevenNonresidue : ConcreteCarrier.SevenProjectiveNonresidue)
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Coins domain) :
    mixedResidualAtBeta .splitV1 covers data coins = K.zero ↔
      Semantics.Nc.Truth data ∨
        SelectorRoot covers data coins ∨
          GammaRoot covers data coins ∨
            SplitV1GammaZero covers data coins := by
  constructor
  · intro splitZero
    have productZero :
        K.mul coins.gamma
          (mixedResidualAtBeta .paperNc covers data coins) = K.zero := by
      rw [← splitV1_mixedResidualAtBeta_eq_gamma_mul_paperNc]
      exact splitZero
    have extensionNoZeroDivisors :=
      ConcreteCarrier.extensionNoZeroDivisors_of_base_and_seven
        baseNoZeroDivisors sevenNonresidue
    rcases extensionNoZeroDivisors _ _ productZero with
      gammaZero | paperZero
    · exact Or.inr <| Or.inr <| Or.inr { gammaZero := gammaZero }
    · rcases
          (paperNc_mixedResidualAtBeta_eq_zero_iff_truth_or_selectorRoot_or_gammaRoot
            baseNoZeroDivisors covers data coins).mp paperZero with
        truth | selectorRoot | gammaRoot
      · exact Or.inl truth
      · exact Or.inr <| Or.inl selectorRoot
      · exact Or.inr <| Or.inr <| Or.inl gammaRoot
  · rintro (truth | selectorRoot | gammaRoot | gammaZero)
    · exact mixedResidualAtBeta_eq_zero_of_truth
        .splitV1 covers data coins truth
    · exact mixedResidualAtBeta_eq_zero_of_all_source_specializations_zero
        .splitV1 covers data coins
        selectorRoot.everySourceSpecializationZero
    · rw [splitV1_mixedResidualAtBeta_eq_gamma_mul_paperNc,
        gammaRoot.paperNcMixtureZero]
      exact ConcreteCarrier.extensionLaws.mul_zero _
    · rw [splitV1_mixedResidualAtBeta_eq_gamma_mul_paperNc,
        gammaZero.gammaZero]
      change ops.mul ops.zero _ = ops.zero
      rw [laws.mul_comm, laws.mul_zero]

end Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.MixingSoundness
