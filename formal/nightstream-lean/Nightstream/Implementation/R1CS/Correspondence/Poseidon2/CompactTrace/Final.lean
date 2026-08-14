import Nightstream.Implementation.R1CS.Correspondence.Poseidon2.CompactTrace.Schema

set_option autoImplicit false
set_option maxRecDepth 65536
set_option maxHeartbeats 5000000

namespace Nightstream.Implementation.R1CS.Poseidon2CompactTraceRefinement

open Nightstream.Implementation.R1CS.Canonical.Poseidon2Core
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Schedule
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Layout

/-- Exact eight-element comparison of the post-round external layer. -/
theorem compact_final_exact : ∀ lane : Fin width,
    traceTerms (LinearSubstitution.terms expansion (finalState canonicalLayout lane)) =
      traceTerms (traceFinalForm lane) := by
  native_decide

end Nightstream.Implementation.R1CS.Poseidon2CompactTraceRefinement
