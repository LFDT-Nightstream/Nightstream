import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.SelectiveArtifactPairs.Chunks

/-!
Proof-free pair certificates for rewrite shards 6--11.  Each invocation sees
exactly 64 raw pair records and no decoded or proof-carrying collection.
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.SelectiveArtifactPairs.CertificateShard1

open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.SelectiveArtifactPairs
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.SelectiveArtifactPairs.Chunks

set_option maxRecDepth 100000 in
theorem rewrite6 : RewritePairsCertificate rewritePairs6 := by native_decide
set_option maxRecDepth 100000 in
theorem rewrite7 : RewritePairsCertificate rewritePairs7 := by native_decide
set_option maxRecDepth 100000 in
theorem rewrite8 : RewritePairsCertificate rewritePairs8 := by native_decide
set_option maxRecDepth 100000 in
theorem rewrite9 : RewritePairsCertificate rewritePairs9 := by native_decide
set_option maxRecDepth 100000 in
theorem rewrite10 : RewritePairsCertificate rewritePairs10 := by native_decide
set_option maxRecDepth 100000 in
theorem rewrite11 : RewritePairsCertificate rewritePairs11 := by native_decide

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.SelectiveArtifactPairs.CertificateShard1
