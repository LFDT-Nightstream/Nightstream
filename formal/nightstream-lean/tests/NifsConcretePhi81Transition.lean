import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81

/-!
Focused compile-time regressions for the exact concrete Phi81 NIFS
composition.

| Stage path | Property under test | Failure caught |
|---|---|---|
| `nifs.concrete.derive` | `Pi_CCS` outputs and the `Pi_RLC` parent are absent from the raw certificate | duplicate or prover-authoritative phase boundary |
| `nifs.concrete.pi_ccs.coins` | one statement-bound schedule derives shared FE/NC coin authority | duplicated or caller-supplied coin records |
| `nifs.concrete.input` | semantic source authority is separate from physical acceptance | hidden private-witness premise in the verifier |
| `nifs.semantic.fold` | parent and children are computed from paper sources, one row point, and valid challenges without a certificate | restating implementation acceptance as semantics |
| `nifs.concrete.refinement` | certificate-indexed evidence refines the independent fold | leaving physical/semantic equivalence implicit |
| `nifs.concrete.pi_rlc` | sampler replay plus the sole retained structure family derives complete Π_RLC acceptance | duplicated challenge-membership or computed-parent check |
| `nifs.concrete.pi_dec` | canonical children reduce generic Π_DEC acceptance to three recomposition equations | duplicated inherited child fields or checks |
| `nifs.concrete.soundness` | physical acceptance yields semantic truth or a named FE/NC bad event | unconditional transcript soundness |
| `nifs.concrete.completeness` | honest sources construct physical messages, combined parent, and valid children | abstract existence without a concrete phase witness |
| `nifs.concrete.completeness.outcome` | honest sources yield a transition or one exact bounded-sampler shortfall | hidden sampler-totality assumption |
| `nifs.concrete.running_authority.bootstrap` | zero-running bootstrap accepts only with no parent authority | treating an unvalidated parent as bootstrap authority |
| `nifs.fixed_active.structure` | selected running sources and returned children share the sole fresh-source structure | duplicate outer structure checks |
-/

open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81

namespace Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive

#check arity
#check arity_total
#check FoldResult
#check resultOf
#check ResultTransition
#check ResultTransition.children_transition
#check ResultTransition.parentOpening
#check ResultTransition.childOpening
#check ResultTransition.runningStructure_eq_fresh
#check ResultTransition.childStructure_eq_fresh
#check ResultTransition.parent_eq_of_children_eq
#check transition_iff_exists_resultTransition
#check Evaluator.Checker
#check Evaluator.run
#check Evaluator.run_eq_some_iff_accepted
#check Evaluator.run_sound
#check Evaluator.run_complete
#check Evaluator.run_complete_or_samplerShortfall

end Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive

#check Context
#check StatementInput
#check Context.piCcsStatement
#check Context.piCcsStatement_sources
#check Context.piCcsStatement_polynomial
#check Context.piCcsStatement_runningParent
#check Context.piCcsPreSumcheck
#check Context.system
#check Context.system_eq_firstFresh
#check Context.ncCoins_betaA_eq_feCoins_betaA
#check Context.ncCoins_gamma_eq_feCoins_gamma
#check Certificate
#check PiDecChildPayload
#check Execution
#check derive
#check outputChildren
#check derive_piRlcInitialState
#check PublicInputBound
#check InputBound
#check SemanticInput
#check SemanticFold.Witness
#check SemanticFold.ChallengesValid
#check SemanticFold.outputs
#check SemanticFold.parentOf
#check SemanticFold.childrenOf
#check SemanticFold.Realization
#check SemanticFold.Holds
#check SemanticFold.complete
#check SemanticFold.Holds.outputsHold
#check SemanticFold.Holds.parentOpening
#check SemanticFold.Holds.childOpening
#check SemanticFold.Holds.piDecAccepted
#check SemanticFold.ObligationPlan.Phase
#check SemanticFold.ObligationPlan.Family
#check SemanticFold.ObligationPlan.Leaf
#check SemanticFold.ObligationPlan.phase
#check SemanticFold.ObligationPlan.family
#check SemanticFold.ObligationPlan.authority
#check SemanticFold.ObligationPlan.path
#check SemanticFold.ObligationPlan.Candidate
#check SemanticFold.ObligationPlan.accepts_iff_target
#check SemanticFold.ObligationPlan.exact
#check SemanticFold.ObligationPlan.Necessity.Realization
#check SemanticFold.ObligationPlan.Necessity.Realization.parentNecessary
#check SemanticFold.ObligationPlan.Necessity.Realization.challengeNecessary
#check SemanticFold.ObligationPlan.Necessity.Realization.childrenNecessary
#check Nightstream.SuperNeo.Concrete.Phi81StrongSet.outsideChallenge_not_member
#check
  Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.OutputRefinement.materializedOutputs_eq_honestOutputs_of_yRingEq
#check OutputBound
#check ChildOpenings
#check RunningAuthority.children
#check RunningAuthority.attempt
#check RunningAuthority.Bound
#check RunningAuthority.Accepted
#check RunningAuthority.Bound.childStructure_eq_parent
#check RunningAuthority.Bound.childPoint_eq_parent
#check RunningAuthority.Bound.inputPoint_eq_parent
#check RunningAuthority.Bound.children_sharePoint
#check RunningAuthority.Accepted.bootstrap
#check RunningAuthority.Accepted.active
#check RunningAuthority.Accepted.parentAbsent_of_bootstrap
#check RunningAuthority.Accepted.iff_parentAbsent_of_bootstrap
#check RunningAuthority.Accepted.iff_nonemptyBound_of_active
#check RunningAuthority.Accepted.rejects_parent_in_bootstrap
#check RunningAuthority.Accepted.children_sharePoint
#check Accepted
#check CertificateRefinement
#check Transition
#check TailAccepted.piRlcAccepted
#check DerivedPiRlc.SourceStructuresBound
#check DerivedPiRlc.equations_of_sourceStructures
#check DerivedPiRlc.sourceStructures_of_equations
#check DerivedPiRlc.equations_iff_sourceStructures
#check Sampler.Checker.candidatePrefix
#check Sampler.Checker.sampleChallenge?
#check Sampler.Checker.check
#check Sampler.Checker.check_eq_true_iff_bound
#check Sampler.Checker.certificateCheck
#check Sampler.Checker.certificateCheck_eq_true_iff_accepted
#check DerivedPiDec.RecompositionEquations
#check DerivedPiDec.accepted_of_recomposition
#check DerivedPiDec.recomposition_of_accepted
#check DerivedPiDec.accepted_iff_recomposition
#check CarrierEquality.ringFEqual
#check CarrierEquality.ringFEqual_eq_true_iff
#check CarrierEquality.commitmentEqual
#check CarrierEquality.commitmentEqual_eq_true_iff
#check CarrierEquality.publicInputEqual
#check CarrierEquality.publicInputEqual_eq_true_iff
#check CarrierEquality.pointEqual
#check CarrierEquality.pointEqual_eq_true_iff
#check CarrierEquality.evaluationsEqual
#check CarrierEquality.evaluationsEqual_eq_true_iff
#check DerivedPiDec.Checker.check
#check DerivedPiDec.Checker.check_eq_true_iff_recomposition
#check FixedActive.Canonical.FreshPayload.materialize
#check FixedActive.Canonical.RunningPayload.materialize
#check FixedActive.Canonical.ParentPayload.materialize
#check FixedActive.Canonical.Input.materialize
#check FixedActive.Canonical.Context.materialize
#check FixedActive.Canonical.Context.materialize_system
#check FixedActive.Canonical.Context.sourceStructures
#check FixedActive.Canonical.RunningAuthority.Equations
#check FixedActive.Canonical.RunningAuthority.accepted_of_equations
#check FixedActive.Canonical.RunningAuthority.equations_of_accepted
#check FixedActive.Canonical.RunningAuthority.accepted_iff_equations
#check FixedActive.Canonical.RunningAuthority.check
#check FixedActive.Canonical.RunningAuthority.check_eq_true_iff_equations
#check FixedActive.Canonical.RunningAuthority.check_eq_true_iff_accepted
#check FixedActive.Canonical.Checker.check
#check FixedActive.Canonical.Checker.check_eq_true_iff_accepted
#check FixedActive.Canonical.Checker.evaluatorChecker
#check accepted_implies_paper_or_outputUnbound_or_badEvent
#check accepted_implies_refinement_or_outputUnbound_or_badEvent
#check accepted_implies_transition_or_outputUnbound_or_badEvent
#check CertificateRefinement.piRlcParentOpening
#check CertificateRefinement.packedYZcolBound
#check CertificateRefinement.semanticWitness
#check CertificateRefinement.toSemanticRealization
#check CertificateRefinement.toSemanticFold
#check Result.resultOf_refines
#check Result.resultTransition_iff_exists_obligationPlan
#check HonestSamplerShortfall
#check complete_of_paperObligations
#check complete_or_samplerShortfall
