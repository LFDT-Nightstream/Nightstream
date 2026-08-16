import Nightstream.Assurance.CompactSourceArtifact
import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignmentData0

/-!
GENERATED FILE - do not edit by hand.

Shared accepted background assignment, decoded once. Removal
counterexamples override single columns of these values. The two
leaves below are the only facts that force the decode.
-/

namespace Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment

open Nightstream.Assurance.CompactSourceArtifact

def payload : String :=
  Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignmentData0.part

def values : Array Nat := (decodeAssignment payload).getD #[]

theorem values_size : values.size = 38626 := by
  native_decide

theorem values_one : values.getD 0 0 = 1 := by
  native_decide

end Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment
