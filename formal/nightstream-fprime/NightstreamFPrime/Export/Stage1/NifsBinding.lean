import NightstreamFPrime.Export.Stage1.SetupBinding
import NightstreamFPrime.Export.Stage1.PiDECInputCheck
import NightstreamFPrime.Lifecycle.Nifs.InteractiveAgreement

/-! The actual NIFS binding event at the selected relation and fixed setup. -/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.NifsBinding

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.Nifs
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra
open NightstreamFPrime.Lifecycle
open Poseidon2HashChainV1Setup

/-- An actual observation binding event supplies a nonzero integer kernel
vector for the selected fixed setup, with strict norm below `8TB`.
`PiDECInputCheck.relation_eq_selected` identifies the relation with the
application fixed point. The collision's commitment and norm fields are
transported to the existing setup reduction; no matrix entry is evaluated.

Compose the returned vector with
`Poseidon2HashChainV1Setup.productionShortKernel_to_approvedMsis` to reach the
larger fixed instance in
`docs/reviews/nightstream-fprime-requirements/PUBLIC_SEED_MSIS_ASSUMPTION.md`.
The reduction appends zeros after the selected carrier and keeps its strict
norm. The assumption has no numerical success bound. This theorem identifies the search problem;
the existing executable reduction and its work premises remain separate. -/
theorem bindingEvent_to_shortKernel
    {Context State : Type}
    (running : Context → Lifecycle.Running
      (logicalWidth := PiDECInputCheck.logicalWidth) (publicFits := PiDECInputCheck.publicFits))
    (fresh : Context → Lifecycle.Fresh
      (logicalWidth := PiDECInputCheck.logicalWidth) (publicFits := PiDECInputCheck.publicFits))
    (program : PiRLC.PaperForkExtractionWork.Primitives RingF
      (PaperAlgebra.Assignment
        (logicalWidth := PiDECInputCheck.logicalWidth) (publicFits := PiDECInputCheck.publicFits)))
    (context : Context)
    (left right : PaperCompositionAgreement.Observation State
      (Lifecycle.Nifs.InteractiveComposition.Endpoint PiDECInputCheck.relation productionAjtaiKey)
      productionShape)
    (event : Lifecycle.Nifs.InteractiveAgreement.BindingEvent
      PiDECInputCheck.relation productionAjtaiKey running fresh program context left right) :
    Nonempty (Binding.ShortKernelVector productionAjtaiKey productionGlobalParams.msisNormBound) := by
  cases left with
  | none => exact False.elim event
  | some left =>
    cases right with
    | none => exact False.elim event
    | some right =>
      obtain ⟨_, _, _, _, _, coordinate, ⟨collision⟩⟩ := event
      exact ⟨productionRelaxedBindingCollision_to_shortKernel
        (((ProductionKey.key PiDECInputCheck.relation productionAjtaiKey).statement
          (running context) (fresh context)).commitments coordinate) {
          delta₁ := collision.delta₁
          delta₂ := collision.delta₂
          opening₁ := collision.opening₁
          opening₂ := collision.opening₂
          delta₁Valid := collision.delta₁Valid
          delta₂Valid := collision.delta₂Valid
          firstEquation := collision.firstEquation
          secondEquation := collision.secondEquation
          firstNorm := collision.firstNorm
          secondNorm := collision.secondNorm
          crossDifferent := collision.crossDifferent }⟩

end NightstreamFPrime.Export.Stage1.NifsBinding
