import Nightstream.Protocol.FPrime.CanonicalVerifier.NifsRefinement

/-!
Focused interface regression for Construction-2 refinement through an
independent NIFS transition.
-/

open Nightstream.Protocol.FPrime.CanonicalVerifier.NifsRefinement

#check withNifsProof
#check SemanticTransition
#check SelectedNifsBadEvent
#check accepts_implies_semanticTransition_or_selectedNifsBadEvent
#check semanticTransition_implies_exists_nifsProof_accepts
#check terminal_exact_without_nifs
