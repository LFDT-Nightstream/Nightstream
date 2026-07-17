import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Fe.InitialSum.CarriedBridge

/-!
Deterministic compression boundary for the Split-NC FE residual family.

Owns: the exact event in which a nonzero uncompressed FE residual family is
hidden by the verifier's FE selectors and gamma mixture, and the exhaustive
truth-or-compression-root decomposition of a zero mixed residual.

Does not own: SumCheck messages, transcript sampling, root counting,
probability bounds, output-message authority, Rust, R1CS, rows, costs, or row
removal.

Emits constraints: no.

Authority boundary: semantic invalidity is stated using the independent
uncompressed residual family. `MixingRoot` is not defined as disagreement
with Rust or with an accepted circuit; it records that a nonzero semantic
residual family maps to zero under the explicit FE compression.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_ccs.fe.soundness.uncompressed` | the independent FE residual family is nonzero | semantic source | `MixingRoot.residualsNonzero` |
| `nifs.pi_ccs.fe.soundness.compression_root` | selectors/gamma map that nonzero family to zero | security boundary | `MixingRoot.compressedZero` |
| `nifs.pi_ccs.fe.soundness.decomposition` | zero FE mix iff truth or the named compression root | derived | `mixedResidual_eq_zero_iff_truth_or_mixingRoot` |
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Fe.MixingSoundness

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Fe
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Fe.InitialSum

/-- A verifier-selected FE compression hid a nonzero independent residual
family. This is the deterministic event to be reduced to selector/gamma root
bounds after the concrete transcript schedule is instantiated. -/
structure MixingRoot
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (profile : SupportedProfile shape domain)
    (data : Data shape)
    (coins : Coins shape domain) : Prop where
  residualsNonzero : ¬ Semantics.Fe.ResidualsZero data
  compressedZero : mixedResidual profile data coins = K.zero

/-- A zero FE compression has exactly two deterministic explanations:
independent FE truth, or a nonzero residual family hidden by the sampled
compression. No probability claim is made here. -/
theorem mixedResidual_eq_zero_iff_truth_or_mixingRoot
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (profile : SupportedProfile shape domain)
    (data : Data shape)
    (coins : Coins shape domain) :
    mixedResidual profile data coins = K.zero ↔
      Semantics.Fe.Truth data ∨ MixingRoot profile data coins := by
  constructor
  · intro compressedZero
    by_cases truth : Semantics.Fe.Truth data
    · exact Or.inl truth
    · apply Or.inr
      exact {
        residualsNonzero := by
          intro residualsZero
          exact truth ((Semantics.Fe.residualsZero_iff_truth data).mp residualsZero)
        compressedZero := compressedZero }
  · rintro (truth | root)
    · exact mixedResidual_eq_zero_of_truth profile data coins truth
    · exact root.compressedZero

end Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Fe.MixingSoundness
