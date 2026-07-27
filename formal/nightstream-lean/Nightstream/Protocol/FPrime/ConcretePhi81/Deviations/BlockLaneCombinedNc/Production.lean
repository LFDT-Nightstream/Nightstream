import Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc.ProductionRefinement.SemanticComposition
import Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc.ProductionRefinement.OutputAuthority
import Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc.ProductionRefinement.ResidualAlignment
import Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc.CausalSoundness
import Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc.ProductionMixingBoundary.IdealInteractiveSoundness

/-!
Public production-refinement surface for the complete typed Split-NC prefix.
Separated from the polynomial facade to keep the delayed paper-step import
graph acyclic.

Owns: the acyclic public export surface for deterministic and causal
production Split-NC refinement.

Does not own: protocol semantics, theorem proofs, Fiat--Shamir, concrete
primitive instantiation, implementation refinement, constraints, costs, or
rows.

Emits constraints: none.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `pi_ccs.production.deterministic` | expose the typed deterministic refinement without changing it | exported | `ProductionRefinement` |
| `pi_ccs.production.probability` | expose the ideal-interactive collision theorem without changing it | exported | `CausalSoundness` |
| `pi_ccs.production.mixing` | expose the selected ordered mixing-plus-collision theorem | exported | `ProductionMixingBoundary.IdealInteractiveSoundness` |
-/

namespace Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc.Production

export ProductionRefinement
  (AuthoritativeInput ProductionVerifierAccepts AcceptedOutput
    FeFailure NcFailure TranscriptFailure BindingFailure
    RegisteredDeviationObligation
    accepted_implies_paper_or_named_failure
    accepted_implies_paper_or_algebraic_failure
    not_transcriptFailure not_bindingFailure
    blockLaneCombinedNc_refines_paperNc everyCoordinate_has_exact_owner
    delayedProjection_refines_rawRecomposition honest_complete
    honest_complete_with_output
    accepted_output_suitable_for_piRlc)

export Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc.CausalSoundness
  (ncRoundCount rawRoundRepresentable Strategy ncCertificate
    ncCertificate_toSumCheck FeRoundCollision NcRoundCollision SplitCollision
    splitCollision_implies_detects splitCollision_probability_le)

export ProductionRefinement.SemanticComposition
  (fold_extraction_or_named_failure)

export ProductionRefinement.OutputAuthority
  (accepted_implies_paper_and_authority_or_named_failure)

export ProductionRefinement.ResidualAlignment
  (LiteralResidualSlotAlignment
    not_literalResidualSlotAlignment
    semanticResidualsZero_iff_paperHolds
    accepted_implies_paper_or_residual_failure)

namespace IdealInteractiveMixing

export Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc.ProductionMixingBoundary.IdealInteractiveCarrier
  (PreSeed Seed support input input_supportAligned
    derivePreSumcheck_shared_gamma derivePreSumcheck_delayed)

export Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc.ProductionMixingBoundary.IdealInteractiveExecution
  (Suffix certificate certificate_fe_coordinates certificate_nc_coordinates)

export Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc.ProductionMixingBoundary.IdealInteractiveSoundness
  (NamedFailure namedFailureEvent feMixingBudget ncMixingBudget
    splitCollisionBudget totalBudget
    algebraicFailureEvent_eq_namedFailureEvent
    namedFailure_probability_le
    namedFailure_probability_le_of_productionField)

end IdealInteractiveMixing

end Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc.Production
