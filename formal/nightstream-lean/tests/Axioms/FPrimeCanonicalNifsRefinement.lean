import Nightstream.Protocol.FPrime.CanonicalVerifier.NifsRefinement
import tests.Axioms.Support

/-!
Fail-closed dependency guard for Construction-2 refinement through an
independent NIFS transition.
-/

open Nightstream.Protocol.FPrime.CanonicalVerifier.NifsRefinement

/-- info: 'Nightstream.Protocol.FPrime.CanonicalVerifier.NifsRefinement.accepts_implies_semanticTransition_or_selectedNifsBadEvent' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms accepts_implies_semanticTransition_or_selectedNifsBadEvent

/-- info: 'Nightstream.Protocol.FPrime.CanonicalVerifier.NifsRefinement.semanticTransition_implies_exists_nifsProof_accepts' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms semanticTransition_implies_exists_nifsProof_accepts

/-- info: 'Nightstream.Protocol.FPrime.CanonicalVerifier.NifsRefinement.terminal_exact_without_nifs' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms terminal_exact_without_nifs
