import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.NifsPaper.PiRlc.ClaimEvaluationCarrier
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.RelationProfile

/-!
Active selective-relation specialization of the generic claim carrier.

Assurance tier: model-level composition.

Owns: the fact that a claim aligned to the independently specified active
selective relation decodes to exactly thirteen matrix evaluations.

Does not own: a production profile inhabitant, a generated claim alignment,
evaluation values, transcript authority, Rust conformance, R1CS rows, costs,
or row removal.

Emits constraints: no.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.claim.evaluations.selective_count` | the active selective carrier decodes exactly thirteen evaluations | derived | `decode_size` |
-/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.SelectiveCarrier

open Nightstream.Implementation.R1CS.PiDecStrictCompiler
open Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.ClaimShape
open Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.ClaimEvaluationCarrier

/-- The independent selective port vocabulary, rather than a generated
header, fixes the physical CE evaluation count. -/
theorem decode_size
    {rows columns : Nat}
    (profile :
      Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.RelationProfile.Profile
        rows columns)
    (claim : ClaimLayout)
    (alignment : Holds
      (Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.RelationProfile.Profile.shape
        profile)
      claim)
    (assignment : Nat -> Nat) :
    (ClaimEvaluationCarrier.decode assignment
      (ClaimEvaluationCarrier.fromClaim alignment)).size = 13 := by
  rw [ClaimEvaluationCarrier.decode_size]
  exact
    Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.RelationProfile.Profile.shape_matrixCount_eq_13
      profile

end Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.SelectiveCarrier
