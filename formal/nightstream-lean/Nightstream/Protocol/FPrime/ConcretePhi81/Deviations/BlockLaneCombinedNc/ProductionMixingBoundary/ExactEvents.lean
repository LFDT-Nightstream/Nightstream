import Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc.ProductionRefinement
import Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc.CausalSoundness

/-!
Exact production event inventory at the Split-NC mixing boundary.

Assurance tier: model-level interface evidence.

Owns: exhaustive elimination theorems over the actual production
`FeFailure`/`NcFailure` families. FE is refined through the exact nested
`SumCheck.Fe.BadEvent` constructors, and NC remains right-nested in the frozen
constructor order.

Does not own: a replacement event family, a mixing-soundness contract,
challenge sampling, a probability theorem beyond the existing
`splitCollision_probability_le`, Fiat--Shamir, Rust/R1CS, encoding, or rows.

Emits constraints: no.

| Boundary | Owned equation | Excluded boundary |
|---|---|---|
| FE failures | exact nested mixing-root / round-collision elimination | replacement event family |
| NC failures | frozen right-nested constructor order | reassociated loss accounting |
-/

set_option autoImplicit false

namespace Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc.ProductionMixingBoundary

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier

universe uState

variable
  {shape : SemanticShape}
  {State : Type uState}
  {publicRingColumns verifierRows : Nat}
  {publicFits : ringDegree * publicRingColumns <= shape.carrierWidth}

/-- The actual production FE family has exactly the nested mixing-root and
round-collision cases. No generic mixing event replaces either constructor. -/
theorem feFailure_exact_cases
    {input : ProductionRefinement.AuthoritativeInput
      (shape := shape) (State := State)
      (publicRingColumns := publicRingColumns)
      (verifierRows := verifierRows)
      (publicFits := publicFits)}
    {certificate : ProductionRefinement.Certificate input}
    (failure : ProductionRefinement.FeFailure input certificate) :
    (∃ bound root,
        failure = .sumcheck bound (.mixingRoot root)) ∨
      (∃ bound round collision,
        failure = .sumcheck bound (.roundCollision round collision)) := by
  cases failure with
  | sumcheck bound bad =>
      cases bad with
      | mixingRoot root =>
          exact Or.inl ⟨bound, root, rfl⟩
      | roundCollision round collision =>
          exact Or.inr ⟨bound, round, collision, rfl⟩

/-- The actual production NC family, preserved in the frozen loss order:
lane selector, block selector, shared-gamma polynomial, delayed residual
weight, then SumCheck round collision. The right-nested disjunction is not
reassociated. -/
theorem ncFailure_exact_cases
    {input : ProductionRefinement.AuthoritativeInput
      (shape := shape) (State := State)
      (publicRingColumns := publicRingColumns)
      (verifierRows := verifierRows)
      (publicFits := publicFits)}
    {certificate : ProductionRefinement.Certificate input}
    (failure : ProductionRefinement.NcFailure input certificate) :
    (∃ root, failure = .laneSelectorRoot root) ∨
      (∃ root, failure = .blockSelectorRoot root) ∨
      (∃ root, failure = .gammaPolynomialRoot root) ∨
      (∃ pending pendingEq root,
        failure = .residualWeightRoot pending pendingEq root) ∨
      (∃ round collision,
        failure = .roundCollision round collision) := by
  cases failure with
  | laneSelectorRoot root =>
      exact Or.inl ⟨root, rfl⟩
  | blockSelectorRoot root =>
      exact Or.inr (Or.inl ⟨root, rfl⟩)
  | gammaPolynomialRoot root =>
      exact Or.inr (Or.inr (Or.inl ⟨root, rfl⟩))
  | residualWeightRoot pending pendingEq root =>
      exact Or.inr (Or.inr (Or.inr (Or.inl ⟨pending, pendingEq, root, rfl⟩)))
  | roundCollision round collision =>
      exact Or.inr (Or.inr (Or.inr (Or.inr ⟨round, collision, rfl⟩)))

end Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc.ProductionMixingBoundary
