import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.SourceDisposition.Physical
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.SourceDisposition.InputBoundary
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.SourceDisposition.Pivots
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.SourceDisposition.Ownership
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.SourceDisposition.RetainedChecks

/-!
Import-only facade for exact ownership of the fixed production combined-NC
source program.

The child leaves separately own physical-definition membership, compiler-input
inclusion, rewrite-terminal pivots, exhaustive definition classification, and
retained-check lockstep. This facade adds no theorem or executable certificate.

Owns: the stable import boundary for the exact source-disposition partition.
Does not own: generated certificate truth, source satisfaction, assignment authority, or row removal.
Emits constraints: none.

| Stable stage path | Obligation | Authority class |
|---|---|---|
| `f_prime.pi_ccs_nc.delayed.combined.source_disposition` | Export the exhaustive physical/input/pivot/retained source-row partition. | checked artifact interface |
-/
