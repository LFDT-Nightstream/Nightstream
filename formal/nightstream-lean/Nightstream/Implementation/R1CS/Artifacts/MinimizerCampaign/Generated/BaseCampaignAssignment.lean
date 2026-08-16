import Nightstream.Assurance.CompactSourceArtifact
import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignmentData0

/-!
GENERATED FILE - do not edit by hand.

Shared accepted background assignment, decoded once. Removal
counterexamples apply per-column overrides to these values.
-/

namespace Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment

open Nightstream.Assurance.CompactSourceArtifact

def payload : String := Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignmentData0.part

theorem decode_succeeds : (decodeAssignment payload).isSome := by
  native_decide

def values : Array Nat := (decodeAssignment payload).get decode_succeeds

end Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment
