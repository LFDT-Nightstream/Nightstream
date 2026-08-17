import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifact
import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeaves

/-!
GENERATED FILE - do not edit by hand.
-/

namespace Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactStepInitialNecessity

open Nightstream.Assurance.CompactSourceArtifact
open Nightstream.Assurance.ConstraintMinimization
open Nightstream.SuperNeo.CheckPlan
open Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifact

set_option maxHeartbeats 2000000
set_option maxRecDepth 65536

def column : Nat := 3811

def value : Nat := 7452973988482309350

def removedFamily : String := "fprime.base.step.initial"

def violatedRow : IndexedRow :=
  ⟨14623, "fprime.base.step.initial", ⟨[(3811, 18446744069414584320), (3815, 1)], [(0, 1)], []⟩⟩

theorem violated_mem : violatedRow ∈ rowsChunk wire 57 := by
  native_decide

theorem violation :
    ¬ Algebraic.Holds
      (overrideAt Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeaves.background column (value : Field))
      violatedRow.row := by
  native_decide

theorem pair_member : (column, removedFamily) ∈ Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeaves.overridePairs := by
  native_decide

theorem column_inRange : column < Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values.size := by
  rw [Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values_size]
  decide

theorem constant_one :
    overrideAt Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeaves.background column (value : Field)
      wire.constantOneColumn = 1 := by
  have distinct : wire.constantOneColumn ≠ column := by decide
  show overrideAt _ _ _ wire.constantOneColumn = 1
  unfold overrideAt
  rw [if_neg distinct]
  show Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeaves.background wire.constantOneColumn = 1
  have zero : wire.constantOneColumn = 0 := by decide
  rw [zero]
  show (((Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values.getD 0 0 : Nat)) : Field) = 1
  rw [Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values_one]
  norm_num

def removalCounterexample : RemovalCounterexample :=
  mkCounterexample Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values column value removedFamily

theorem removalCounterexample_valid :
    removalCounterexample.Valid sourceArtifact reviewedPlan :=
  mkCounterexample_valid wire Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values
    Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeaves.overridePairs column value removedFamily reviewedPlan
    reviewedPlan_subset
    (by rw [Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values_size]; decide)
    column_inRange constant_one pair_member Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeaves.guardsAll
    Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeaves.holdsAll violatedRow 57
    ⟨by rw [chunkCount_eq]; decide, violated_mem⟩ violation

theorem necessary :
    NecessaryForSoundness (FamilyHolds sourceArtifact)
      (Target sourceArtifact) reviewedPlan removedFamily :=
  removalCounterexample.necessary_of_full_valid
    sourceArtifact sourceArtifact reviewedPlan
    sourceArtifact_coversFullRelation sourceArtifact_exactValidation
    removalCounterexample_valid

theorem necessaryNormalized :
    NecessaryForSoundness
      (NormalizedFamilyHolds sourceArtifact)
      (NormalizedTarget sourceArtifact) reviewedPlan removedFamily :=
  removalCounterexample.necessary_normalized_of_full_valid
    sourceArtifact sourceArtifact reviewedPlan
    sourceArtifact_coversFullRelation sourceArtifact_exactValidation
    removalCounterexample_valid

end Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactStepInitialNecessity
