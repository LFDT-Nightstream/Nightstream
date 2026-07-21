import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.SelectiveArtifactPairs.Chunks

/-!
Proof-free pair certificates for rewrite shards 12--17.  Each invocation sees
exactly 64 raw pair records and no decoded or proof-carrying collection.
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.SelectiveArtifactPairs.CertificateShard2

open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.SelectiveArtifactPairs
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.SelectiveArtifactPairs.Chunks

set_option maxRecDepth 100000 in
theorem rewrite12 : RewritePairsCertificate rewritePairs12 := by native_decide
set_option maxRecDepth 100000 in
theorem rewrite13 : RewritePairsCertificate rewritePairs13 := by native_decide
set_option maxRecDepth 100000 in
theorem rewrite14 : RewritePairsCertificate rewritePairs14 := by native_decide
set_option maxRecDepth 100000 in
theorem rewrite15 : RewritePairsCertificate rewritePairs15 := by native_decide
set_option maxRecDepth 100000 in
theorem rewrite16 : RewritePairsCertificate rewritePairs16 := by native_decide
set_option maxRecDepth 100000 in
theorem rewrite17 : RewritePairsCertificate rewritePairs17 := by native_decide

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.SelectiveArtifactPairs.CertificateShard2
