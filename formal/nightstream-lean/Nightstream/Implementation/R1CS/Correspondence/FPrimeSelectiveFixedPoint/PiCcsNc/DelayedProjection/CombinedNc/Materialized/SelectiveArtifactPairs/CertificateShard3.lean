import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.SelectiveArtifactPairs.Chunks

/-!
Proof-free pair certificates for rewrite shards 18--23.  Shards 18--22 contain
64 raw records; shard 23 contains the exact 21-record remainder.  No decoded
or proof-carrying collection is an executable input.
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.SelectiveArtifactPairs.CertificateShard3

open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.SelectiveArtifactPairs
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.SelectiveArtifactPairs.Chunks

set_option maxRecDepth 100000 in
theorem rewrite18 : RewritePairsCertificate rewritePairs18 := by native_decide
set_option maxRecDepth 100000 in
theorem rewrite19 : RewritePairsCertificate rewritePairs19 := by native_decide
set_option maxRecDepth 100000 in
theorem rewrite20 : RewritePairsCertificate rewritePairs20 := by native_decide
set_option maxRecDepth 100000 in
theorem rewrite21 : RewritePairsCertificate rewritePairs21 := by native_decide
set_option maxRecDepth 100000 in
theorem rewrite22 : RewritePairsCertificate rewritePairs22 := by native_decide
set_option maxRecDepth 100000 in
theorem rewrite23 : RewritePairsCertificate rewritePairs23 := by native_decide

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.SelectiveArtifactPairs.CertificateShard3
