import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.SelectiveArtifactPairs.Chunks

/-!
Proof-free pair certificates for the 52 retained records and rewrite shards
0--5.  Each `native_decide` invocation sees one list of at most 64 raw pair
records; decoded structures and proof fields are absent.
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.SelectiveArtifactPairs.CertificateShard0

open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.SelectiveArtifactPairs
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.SelectiveArtifactPairs.Chunks

set_option maxRecDepth 100000 in
theorem retained : RetainedPairsCertificate retainedPairs := by native_decide

set_option maxRecDepth 100000 in
theorem rewrite0 : RewritePairsCertificate rewritePairs0 := by native_decide
set_option maxRecDepth 100000 in
theorem rewrite1 : RewritePairsCertificate rewritePairs1 := by native_decide
set_option maxRecDepth 100000 in
theorem rewrite2 : RewritePairsCertificate rewritePairs2 := by native_decide
set_option maxRecDepth 100000 in
theorem rewrite3 : RewritePairsCertificate rewritePairs3 := by native_decide
set_option maxRecDepth 100000 in
theorem rewrite4 : RewritePairsCertificate rewritePairs4 := by native_decide
set_option maxRecDepth 100000 in
theorem rewrite5 : RewritePairsCertificate rewritePairs5 := by native_decide

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.SelectiveArtifactPairs.CertificateShard0
