import Nightstream.Assurance.CompactSourceArtifact
import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact
import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactData0
import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactData1
import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactData2
import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactData3
import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactData4
import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactData5
import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactData6
import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactData7
import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactData8
import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactData9

/-!
GENERATED FILE - do not edit by hand.

Assembly of the complete string-payload source artifact. The
emitter replayed every payload row against the independent sparse
recovery before rendering; `expand` re-derives the artifact
natively and fails closed on any malformation.
-/

namespace Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifact

open Nightstream.Assurance.CompactSourceArtifact
open Nightstream.Assurance.ConstraintMinimization
open Nightstream.Implementation.R1CS

set_option maxHeartbeats 2000000
set_option maxRecDepth 65536

def families : List FamilyRanges :=
  [⟨"fprime.base.finalize.application", [(0, 4416)]⟩,
   ⟨"fprime.base.step.advance", [(14631, 15157)]⟩,
   ⟨"fprime.base.step.initial", [(14621, 14631)]⟩,
   ⟨"fprime.base.step.output", [(15157, 39949)]⟩,
   ⟨"fprime.base.step.prelude", [(4416, 14158)]⟩,
   ⟨"fprime.base.step.source", [(14158, 14621)]⟩]

def matrixA : MatrixWire where
  rowCounts := Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactData1.part
  columns := Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactData2.part
  valueIndexes := Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactData3.part
  seededBlocks := []

def matrixB : MatrixWire where
  rowCounts := Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactData4.part
  columns := Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactData5.part
  valueIndexes := Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactData6.part
  seededBlocks := []

def matrixC : MatrixWire where
  rowCounts := Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactData7.part
  columns := Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactData8.part
  valueIndexes := Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactData9.part
  seededBlocks := []

def wire : Wire where
  schema := "nightstream/r1cs-redundancy-problem/v3"
  profile := "campaign-base-classification-v1"
  scope := "branch"
  diagnosticDigest := "sha256:54bec6fa7de4ec475e2fd43a1c015bfede809d2d1370b67677ea66dbda6839e7"
  fieldModulus := "18446744069414584321"
  totalRows := 39949
  columnCount := 38626
  constantOneColumn := 0
  publicInputCount := 2426
  completeFamilies := ["fprime.base.finalize.application", "fprime.base.step.advance", "fprime.base.step.initial", "fprime.base.step.output", "fprime.base.step.prelude", "fprime.base.step.source"]
  valueTable := Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactData0.part
  families := families
  a := matrixA
  b := matrixB
  c := matrixC

theorem expand_succeeds : (expand wire).isSome := by native_decide

def sourceArtifact : Artifact := (expand wire).get expand_succeeds

theorem sourceArtifact_coversFullRelation :
    sourceArtifact.CoversFullRelation := by native_decide

theorem sourceArtifact_exactValidation :
    Artifact.ExactValidation sourceArtifact sourceArtifact = true := by
  native_decide

theorem sourceArtifact_matches_committed :
    sourceArtifact = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifact := by native_decide

end Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifact
