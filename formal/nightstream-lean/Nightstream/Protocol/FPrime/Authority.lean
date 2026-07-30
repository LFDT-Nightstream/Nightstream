import Nightstream.Protocol.FPrime.CanonicalVerifier.PaperNonInteractiveNifs

/-!
Narrow public authority facade for the F-prime protocol.

Owns: selection of the unchanged HyperNova Construction-2 step and terminal
relations, the canonical executable checkers, and the selected SuperNeo
non-interactive NIFS relation.

Does not own: paper comparison facades, ConcretePhi81 implementation models,
production deviations, obstruction theorems, security reductions, Rust,
R1CS, artifacts, or costs.

This file adds no semantic definition. The exports below give stable public
names to relations and checkers that remain owned by their source modules.
-/

namespace Nightstream.Protocol.FPrime.Authority

namespace Step

export Nightstream.HyperNova.Construction2.Paper
  (InRange selectedIndex oneBased HashPreimage Setup Machine Input Output
    priorHashPreimage nextHashPreimage ApplicationHolds OutputHolds
    BaseHolds RecursiveHolds Transition holds_iff_base_or_recursive
    holds_iff_transition)

export Nightstream.Protocol.FPrime.CanonicalVerifier
  (replaceSelected outputFor eval Accepts accepts_implies_transition
    transition_implies_accepts accepts_iff_transition)

end Step

namespace Nifs

export Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive
  (Key Running Fresh Proof verify Transition BadEvent)

export Nightstream.Protocol.FPrime.CanonicalVerifier.PaperNonInteractiveNifs
  (nifsVerifier nifsSoundAndCompleteModulo construction2Setup
    canonicalFprime_accepts_implies_paperTransition_or_nifsBadEvent
    canonicalFprime_paperTransition_implies_exists_nifsProof_accepts)

end Nifs

namespace Terminal

export Nightstream.HyperNova.Construction2.Paper
  (TerminalProof TerminalStatement TerminalRelations TerminalTransition
    terminalHolds_iff_transition)

export Nightstream.Protocol.FPrime.CanonicalTerminalVerifier
  (RelationChecks allRunningAccepted eval eval_eq_true_iff_transition)

end Terminal

end Nightstream.Protocol.FPrime.Authority
