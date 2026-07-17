import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.NifsPaper.PiRlc.ClaimShape
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.NifsPaper.PiRlc.RecursiveCarrierArtifact
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.NifsPaper.PiRlc.TerminalCarrierArtifact
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.RelationProfile

/-!
Artifact-specific CE-claim layout alignment with the selective relation.

Assurance tier: model-level schema plus artifact-checked specializations.

Owns: fail-closed specializations proving that the current recursive and
terminal three-row artifacts do not satisfy the low-level claim-shape contract
for the active thirteen-matrix selective relation.

Does not own: construction of a production relation profile, point values,
`Pi_CCS` point authority, evaluation contents, transcript authority, CE
membership, Rust conformance, R1CS rows, costs, or row removal.

Emits constraints: no.

Authority boundary: point and evaluation lengths are independent obligations.
Neither a digest nor successful decoding can substitute for them. The current
artifacts are rejected on evaluation count before any value-level theorem is
used.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.claim.shape.recursive` | current recursive parent is not an active selective CE claim | artifact-checked | `recursiveArtifact_not_selectiveAligned` |
| `nifs.claim.shape.terminal` | current terminal parent is not an active selective CE claim | artifact-checked | `terminalArtifact_not_selectiveAligned` |
-/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.ClaimShapeAlignment

open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.Implementation.R1CS.PiDecStrictCompiler
open Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.ClaimShape

/-- The current recursive artifact has three physical evaluation rows and
therefore cannot represent the active thirteen-matrix selective relation. -/
theorem recursiveArtifact_not_selectiveAligned
    {rows columns : Nat}
    (profile :
      Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.RelationProfile.Profile
        rows columns) :
    Not
      (Holds
        (Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.RelationProfile.Profile.shape
          profile)
        Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.RecursiveCarrierArtifact.parentClaim) :=
  ClaimShape.not_aligned_of_threeRows
    Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.RecursiveCarrierArtifact.parentEvaluationCount
    (Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.RelationProfile.Profile.shape_matrixCount_ne_three
      profile)

/-- The current terminal artifact has the same incompatible three-row shape. -/
theorem terminalArtifact_not_selectiveAligned
    {rows columns : Nat}
    (profile :
      Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.RelationProfile.Profile
        rows columns) :
    Not
      (Holds
        (Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.RelationProfile.Profile.shape
          profile)
        Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.TerminalCarrierArtifact.parentClaim) :=
  ClaimShape.not_aligned_of_threeRows
    Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.TerminalCarrierArtifact.parentEvaluationCount
    (Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.RelationProfile.Profile.shape_matrixCount_ne_three
      profile)

end Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.ClaimShapeAlignment
