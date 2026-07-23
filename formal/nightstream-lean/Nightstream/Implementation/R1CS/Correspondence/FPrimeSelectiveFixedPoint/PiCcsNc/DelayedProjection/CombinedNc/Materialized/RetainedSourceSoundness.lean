import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.RetainedSourceArtifact
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.SelectiveArtifactPairs.Artifact

/-!
Selective-row soundness for the 52 directly retained production combined-NC
source checks.

Owns: the one-step composition from literal generated emitted-row
satisfaction and the steady-selector equation to satisfaction of the exact
50 round-head and two terminal source rows under the reconstructed compiler
assignment.

Does not own: eliminated source equations, selector enforcement,
source-program execution, transcript order, parent or raw-child authority,
commitment binding, costs, or row removal.

Emits constraints: none.

Assurance tier: artifact-checked for the fixed generated production profile.
-/

/-!
| Stable stage path | Obligation | Authority class |
|---|---|---|
| `f_prime.pi_ccs_nc.delayed.combined.retained_source_soundness` | Derive retained source obligations from satisfaction of their exact decoded rows. | derived |

-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.RetainedSourceSoundness

open Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized

/-- Every directly retained physical source check follows from the literal
generated emitted rows. The source assignment in the conclusion is exactly
`SourceAssignment.compilerAssignment assignment`, as fixed by
`GeneratedRetainedSourceRowsSatisfy`; it is not caller supplied. -/
theorem generatedEmittedRowsSatisfy_implies_retainedSourceRowsSatisfy
    {assignment : Nat → Nat}
    (satisfies :
      SelectiveArtifactPairs.Artifact.GeneratedEmittedRowsSatisfy assignment)
    (selectorOne : assignment Metadata.steadySelectorColumn = 1) :
    RetainedSourceArtifact.GeneratedRetainedSourceRowsSatisfy assignment := by
  have compilerObligations :=
    SelectiveArtifactPairs.Artifact.generatedEmittedRowsSatisfy_implies_allCompilerObligations
      satisfies selectorOne
  exact
    RetainedSourceArtifact.allRetainedObligationsHold_implies_sourceRowsSatisfy
      compilerObligations.2

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.RetainedSourceSoundness
