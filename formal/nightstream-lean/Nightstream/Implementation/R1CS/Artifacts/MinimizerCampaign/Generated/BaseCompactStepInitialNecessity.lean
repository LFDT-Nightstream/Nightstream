import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifact
import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment
set_option maxHeartbeats 2000000
set_option maxRecDepth 65536

namespace Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactStepInitialNecessity

open Nightstream.Assurance.ConstraintMinimization
open Nightstream.Assurance.CompactSourceArtifact
open Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifact

def reviewedPlan : List String := ["fprime.base.finalize.application","fprime.base.step.advance","fprime.base.step.initial","fprime.base.step.output","fprime.base.step.prelude","fprime.base.step.source"]

def overrides : List (Nat × Nat) := [(3811, 1055183102398969390)]

theorem overrides_apply :
    (applyOverrides Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values overrides).isSome := by
  native_decide

def removalCounterexampleValues : List Field :=
  ((applyOverrides Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values overrides).get overrides_apply).toList.map
    (fun value => (value : Field))

def removalCounterexample : RemovalCounterexample where
  removedFamily := "fprime.base.step.initial"
  values := removalCounterexampleValues

theorem removalCounterexample_valid :
    removalCounterexample.Valid sourceArtifact reviewedPlan := by
  native_decide

theorem necessary :
    NecessaryForSoundness (FamilyHolds sourceArtifact)
      (Target sourceArtifact) reviewedPlan "fprime.base.step.initial" :=
  removalCounterexample.necessary_of_full_valid
    sourceArtifact sourceArtifact reviewedPlan
    sourceArtifact_coversFullRelation sourceArtifact_exactValidation
    removalCounterexample_valid

theorem necessaryNormalized :
    NecessaryForSoundness
      (NormalizedFamilyHolds sourceArtifact)
      (NormalizedTarget sourceArtifact) reviewedPlan "fprime.base.step.initial" :=
  removalCounterexample.necessary_normalized_of_full_valid
    sourceArtifact sourceArtifact reviewedPlan
    sourceArtifact_coversFullRelation sourceArtifact_exactValidation
    removalCounterexample_valid

end Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactStepInitialNecessity
