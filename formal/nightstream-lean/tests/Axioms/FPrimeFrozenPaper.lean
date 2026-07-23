import Nightstream.Protocol.FPrime.Frozen
import tests.Axioms.Support

/-!
Fail-closed trusted-dependency gate for the frozen paper-authoritative facade.

This file guards proved headline equations and formula obstructions. The
still-open target propositions are definitions and therefore are not presented
as established security theorems here.
-/

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperCorrections.carriedTargetExponent_eq_absolute' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.PaperCorrections.carriedTargetExponent_eq_absolute

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperCorrections.literalTargetExponent_ne_frozen' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.PaperCorrections.literalTargetExponent_ne_frozen

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperCorrections.literalSection73NormIndices_ne_strictCentered_at_two' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.PaperCorrections.literalSection73NormIndices_ne_strictCentered_at_two

/-- info: 'Nightstream.HyperNova.NonInteractiveMultiFold.accepts_iff_verify' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.HyperNova.NonInteractiveMultiFold.accepts_iff_verify

/-- info: 'Nightstream.HyperNova.Construction2.Paper.holds_iff_transition' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.HyperNova.Construction2.Paper.holds_iff_transition

/-- info: 'Nightstream.HyperNova.Construction2.Paper.terminalHolds_iff_transition' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.HyperNova.Construction2.Paper.terminalHolds_iff_transition

/-- info: 'Nightstream.Protocol.FPrime.Frozen.HyperNova.nifsV_accepts_iff' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.HyperNova.nifsV_accepts_iff

/-- info: 'Nightstream.Protocol.FPrime.Frozen.HyperNova.fprime_accepts_iff_transition' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.HyperNova.fprime_accepts_iff_transition

/-- info: 'Nightstream.Protocol.FPrime.Frozen.HyperNova.canonicalFprime_accepts_iff_transition' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.HyperNova.canonicalFprime_accepts_iff_transition

/-- info: 'Nightstream.Protocol.FPrime.Frozen.HyperNova.canonicalFprime_accepts_implies_semanticTransition_or_selectedNifsBadEvent' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.HyperNova.canonicalFprime_accepts_implies_semanticTransition_or_selectedNifsBadEvent

/-- info: 'Nightstream.Protocol.FPrime.Frozen.HyperNova.canonicalFprime_semanticTransition_implies_exists_nifsProof_accepts' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.HyperNova.canonicalFprime_semanticTransition_implies_exists_nifsProof_accepts

/-- info: 'Nightstream.Protocol.FPrime.Frozen.HyperNova.terminal_accepts_iff_transition' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.HyperNova.terminal_accepts_iff_transition

/-- info: 'Nightstream.Protocol.FPrime.Frozen.HyperNova.canonicalTerminal_accepts_iff_transition' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.HyperNova.canonicalTerminal_accepts_iff_transition

/-- info: 'Nightstream.Protocol.FPrime.Frozen.HyperNova.canonicalTerminal_exact_without_nifs' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.HyperNova.canonicalTerminal_exact_without_nifs

/-- info: 'Nightstream.Protocol.FPrime.Frozen.SuperNeo.piDec_reductionOfKnowledge' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.SuperNeo.piDec_reductionOfKnowledge

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.PiRlcComposition.PiDec.finiteReductionOfKnowledge' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.SuperNeo.finiteReductionOfKnowledge

/-- info: 'Nightstream.Protocol.FPrime.Frozen.SuperNeo.piRlc_literalAmbientBound_obstruction' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.SuperNeo.piRlc_literalAmbientBound_obstruction

/-- info: 'Nightstream.Protocol.FPrime.Frozen.SuperNeo.piRlc_correctedAmbientBound_covers' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.SuperNeo.piRlc_correctedAmbientBound_covers

/-- info: 'Nightstream.Protocol.FPrime.CanonicalVerifier.PaperNonInteractiveNifs.nifsSoundAndCompleteModulo' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.SuperNeo.nifsSoundAndCompleteModulo

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.sourceValid_exists_verifiedTransition' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.SuperNeo.sourceValid_exists_verifiedTransition

/-- info: 'Nightstream.Protocol.FPrime.CanonicalVerifier.PaperNonInteractiveNifs.canonicalFprime_accepts_implies_paperTransition_or_nifsBadEvent' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.HyperNova.canonicalFprime_accepts_implies_paperTransition_or_nifsBadEvent

/-- info: 'Nightstream.Protocol.FPrime.CanonicalVerifier.PaperNonInteractiveNifs.canonicalFprime_paperTransition_implies_exists_nifsProof_accepts' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.HyperNova.canonicalFprime_paperTransition_implies_exists_nifsProof_accepts

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.Key.statement_sumcheckDegreeBound_le' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.Key.statement_sumcheckDegreeBound_le

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.piCcsCheck_eq_true_iff' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.piCcsCheck_eq_true_iff

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.piCcsRoundChain_of_check' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.piCcsRoundChain_of_check
