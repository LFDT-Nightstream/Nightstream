import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane.InitialSum
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConcreteCarrier.NoZeroDivisors
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.SignedCoefficientPolynomial

/-!
Deterministic bad-event decomposition for canonical block×lane NC mixing.

Assurance tier: model-level.

Owns: separate lane-selector, block-selector, and gamma-polynomial root events,
plus the exact decomposition of a zero paper-relative source mixture.

Does not own: challenge sampling, probability bounds, SumCheck messages,
transcript derivation, packed-output terminal binding, Rust, R1CS, costs, or
row removal.

Emits constraints: no.

Authority boundary: selector roots record failure at the precise selector
stage where a nonzero table vanished. `GammaPolynomialRoot` records a root of
the explicit constant-first source polynomial. These are deterministic
security events, not trusted conclusions or probability bounds.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_ccs.nc.block_lane.soundness.lane_selector_root` | nonzero Boolean relation, but every lane specialization is zero | security boundary | `LaneSelectorRoot` |
| `nifs.pi_ccs.nc.block_lane.soundness.block_selector_root` | a lane specialization survives, but every block specialization is zero | security boundary | `BlockSelectorRoot` |
| `nifs.pi_ccs.nc.block_lane.soundness.gamma_polynomial` | source specializations are the coefficients of the explicit gamma polynomial | computed | `gammaPolynomial` |
| `nifs.pi_ccs.nc.block_lane.soundness.gamma_root` | a nonzero source coefficient cancels at gamma | security boundary | `GammaPolynomialRoot` |
| `nifs.pi_ccs.nc.block_lane.soundness.decompose` | zero source mix iff truth or one named root | derived | `mixedResidualAtBeta_eq_zero_iff_truth_or_laneSelectorRoot_or_blockSelectorRoot_or_gammaPolynomialRoot` |
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane.MixingSoundness

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane

private abbrev ops := ConcreteCarrier.extensionOps
private abbrev laws := ConcreteCarrier.extensionLaws

/-- Lane equality specialization hid a nonzero complete Boolean residual
table. -/
structure LaneSelectorRoot
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Mixing.Coins domain) : Prop where
  relationNonzero :
    ¬ SourceProjection.BooleanResidualsZero covers data
  everyLaneSpecializationZero :
    ∀ source block,
      InitialSum.laneResidualAtBeta covers data coins source block = K.zero

/-- A lane specialization survived, but block equality specialization hid
every source table. -/
structure BlockSelectorRoot
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Mixing.Coins domain) : Prop where
  someLaneSpecializationNonzero :
    ∃ source block,
      InitialSum.laneResidualAtBeta covers data coins source block ≠ K.zero
  everySourceSpecializationZero :
    ∀ source,
      InitialSum.sourceResidualAtBeta covers data coins source = K.zero

/-- The verifier-visible constant-first polynomial whose coefficients are the
independently block-and-lane-specialized source residuals. -/
def gammaPolynomial
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Mixing.Coins domain) :
    Nightstream.SuperNeo.SumCheck.Finite.Message K where
  coefficients :=
    (canonicalFinIndices shape.sourceCount).map fun source =>
      InitialSum.sourceResidualAtBeta covers data coins source

/-- The gamma polynomial's degree bound is derived from its coefficient-list
shape rather than supplied by a prover. -/
theorem gammaPolynomial_degreeUpperBound
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Mixing.Coins domain) :
    (gammaPolynomial covers data coins).degreeUpperBound =
      shape.sourceCount - 1 := by
  simp [gammaPolynomial,
    Nightstream.SuperNeo.SumCheck.Finite.Message.degreeUpperBound,
    canonicalFinIndices_length]

/-- Evaluation of the explicit gamma polynomial is exactly the named
paper-relative source mixture. -/
theorem gammaPolynomial_evaluate_eq_mixedResidualAtBeta
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Mixing.Coins domain) :
    (gammaPolynomial covers data coins).evaluate ops.toOps coins.gamma =
      InitialSum.mixedResidualAtBeta covers data coins := by
  unfold gammaPolynomial
    Nightstream.SuperNeo.SumCheck.Finite.Message.evaluate
    InitialSum.mixedResidualAtBeta
  simpa [Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.Mixing.sourceExponent]
    using SignedCoefficientPolynomial.evaluate_canonicalFinMap_eq_gammaSum
      ops laws coins.gamma shape.sourceCount
        (InitialSum.sourceResidualAtBeta covers data coins)

/-- At least one source coefficient survived both selector stages, but the
explicit gamma polynomial vanishes at the verifier's gamma. -/
structure GammaPolynomialRoot
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Mixing.Coins domain) : Prop where
  someCoefficientNonzero :
    ∃ source,
      InitialSum.sourceResidualAtBeta covers data coins source ≠ K.zero
  polynomialRoot :
    (gammaPolynomial covers data coins).evaluate ops.toOps coins.gamma = K.zero

/-- If every lane-specialized Boolean row is zero, then every block-specialized
source residual is zero. -/
theorem sourceResidualAtBeta_eq_zero_of_all_lane_specializations_zero
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Mixing.Coins domain)
    (allZero : ∀ source block,
      InitialSum.laneResidualAtBeta covers data coins source block = K.zero)
    (source : Fin shape.sourceCount) :
    InitialSum.sourceResidualAtBeta covers data coins source = K.zero := by
  unfold InitialSum.sourceResidualAtBeta
    BooleanReproduction.equalityWeighted
  calc
    FiniteSumAlgebra.sumMap ops
        (BooleanVertex.all domain.blockVariables) _ =
      FiniteSumAlgebra.sumMap ops
        (BooleanVertex.all domain.blockVariables) (fun _ => ops.zero) := by
          apply FiniteSumAlgebra.sumMap_congr
          intro block _
          change ops.mul
            (BooleanVertex.equalityWeight ops block coins.betaBlock)
            (InitialSum.laneResidualAtBeta
              covers data coins source block) = ops.zero
          rw [allZero source block]
          exact laws.mul_zero _
    _ = ops.zero := FiniteSumAlgebra.sumMap_zero ops laws _

/-- If every independently derived source specialization is zero, their
paper-relative gamma mixture is zero. -/
theorem mixedResidualAtBeta_eq_zero_of_all_source_specializations_zero
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Mixing.Coins domain)
    (allZero : ∀ source,
      InitialSum.sourceResidualAtBeta covers data coins source = K.zero) :
    InitialSum.mixedResidualAtBeta covers data coins = K.zero := by
  unfold InitialSum.mixedResidualAtBeta
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

/-- Exact deterministic decomposition of a zero canonical NC source mix.
No SumCheck acceptance, transcript, sampling, or probability claim appears in
this theorem. -/
theorem mixedResidualAtBeta_eq_zero_iff_truth_or_laneSelectorRoot_or_blockSelectorRoot_or_gammaPolynomialRoot
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Mixing.Coins domain) :
    InitialSum.mixedResidualAtBeta covers data coins = K.zero ↔
      Semantics.Nc.Truth data ∨
        LaneSelectorRoot covers data coins ∨
        BlockSelectorRoot covers data coins ∨
        GammaPolynomialRoot covers data coins := by
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
      by_cases allLaneZero : ∀ source block,
          InitialSum.laneResidualAtBeta covers data coins source block = K.zero
      · exact Or.inr <| Or.inl {
          relationNonzero := relationNonzero
          everyLaneSpecializationZero := allLaneZero }
      · obtain ⟨source, notAllBlocksZero⟩ :=
          Classical.not_forall.mp allLaneZero
        obtain ⟨block, laneNonzero⟩ :=
          Classical.not_forall.mp notAllBlocksZero
        by_cases allSourceZero : ∀ source,
            InitialSum.sourceResidualAtBeta covers data coins source = K.zero
        · exact Or.inr <| Or.inr <| Or.inl {
            someLaneSpecializationNonzero := ⟨source, block, laneNonzero⟩
            everySourceSpecializationZero := allSourceZero }
        · apply Or.inr
          apply Or.inr
          apply Or.inr
          refine {
            someCoefficientNonzero := Classical.not_forall.mp allSourceZero
            polynomialRoot := ?_ }
          rw [gammaPolynomial_evaluate_eq_mixedResidualAtBeta]
          exact mixtureZero
  · rintro (truth | laneRoot | blockRoot | gammaRoot)
    · exact InitialSum.mixedResidualAtBeta_eq_zero_of_truth
        covers data coins truth
    · apply mixedResidualAtBeta_eq_zero_of_all_source_specializations_zero
        covers data coins
      intro source
      exact sourceResidualAtBeta_eq_zero_of_all_lane_specializations_zero
        covers data coins laneRoot.everyLaneSpecializationZero source
    · exact mixedResidualAtBeta_eq_zero_of_all_source_specializations_zero
        covers data coins blockRoot.everySourceSpecializationZero
    · rw [← gammaPolynomial_evaluate_eq_mixedResidualAtBeta]
      exact gammaRoot.polynomialRoot

end Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane.MixingSoundness
