import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingTerminalProfileSchema

/-! Generated compact ownership profile for the exact Rust streaming terminal reference slice.

The complete Rust rows remain authoritative. This file records their exact source-to-final placement.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalProfile

open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingTerminalProfile.Artifact

def finalStageRuns : List FinalRun := [
    { family := "Retained", rows := { start := 5008421, stop := 5010598 } },
    { family := "Poseidon2", rows := { start := 5096241, stop := 5190325 } },
  ]

def sourceStageBindings : List SourceStageBinding := [
    { occurrence := 2, path := "fprime.recursive.step.prelude", sourceRows := { start := 0, stop := 26131 }, sourceFieldRuns := [{ start := 642, stop := 650 }, { start := 672, stop := 676 }, { start := 722, stop := 726 }], finalRuns := [
    { family := "Retained", rows := { start := 4879501, stop := 4880230 } },
    { family := "Poseidon2", rows := { start := 5010599, stop := 5014211 } },
  ] },
    { occurrence := 11872, path := "fprime.recursive.step.nebula.private_delayed_input", sourceRows := { start := 19322558, stop := 19325532 }, sourceFieldRuns := [{ start := 19231123, stop := 19231124 }], finalRuns := [
    { family := "Retained", rows := { start := 5003576, stop := 5006513 } },
  ] },
    { occurrence := 11873, path := "fprime.recursive.step.nebula", sourceRows := { start := 19325532, stop := 19436810 }, sourceFieldRuns := [{ start := 19341627, stop := 19341672 }], finalRuns := [
    { family := "Retained", rows := { start := 5006513, stop := 5007175 } },
    { family := "Poseidon2", rows := { start := 5078005, stop := 5087551 } },
    { family := "ShiftedTernaryCanonical", rows := { start := 8261176, stop := 8268547 } },
    { family := "ProductSum", rows := { start := 10306212, stop := 10306216 } },
  ] },
    { occurrence := 11879, path := "fprime.recursive.step.accumulator.output_authority.aggregate", sourceRows := { start := 30634723, stop := 30638361 }, sourceFieldRuns := [{ start := 30362937, stop := 30362941 }, { start := 30362945, stop := 30362949 }], finalRuns := [
    { family := "Retained", rows := { start := 5007540, stop := 5007544 } },
    { family := "Poseidon2", rows := { start := 5090475, stop := 5090991 } },
  ] },
    { occurrence := 11881, path := "fprime.recursive.step.output", sourceRows := { start := 30638883, stop := 30664206 }, sourceFieldRuns := [{ start := 30382553, stop := 30382565 }], finalRuns := [
    { family := "Retained", rows := { start := 5007876, stop := 5008413 } },
    { family := "Poseidon2", rows := { start := 5090991, stop := 5094521 } },
  ] },
    { occurrence := 11883, path := "fprime.recursive.finalize.semantic_links", sourceRows := { start := 30676324, stop := 31339295 }, sourceFieldRuns := [{ start := 30400385, stop := 30402558 }], finalRuns := [
    { family := "Retained", rows := { start := 5008421, stop := 5010598 } },
    { family := "Poseidon2", rows := { start := 5096241, stop := 5190325 } },
  ] },
  ]

def rawArtifact : RawArtifact :=
  { schemaVersion := 1,
    profileId := "nightstream/goldilocks/streaming-terminal-slice/v1", lifecycleScope := "recursive-terminal-arm-435",
    sourceArtifactIdentity := "rust:nightstream/streaming-lifecycle-recursive/source-rows/v1",
    finalArtifactIdentity := "rust:nightstream/streaming-selective-ccs/final-rows/v1",
    acceptedWorkItems := 436, terminalArm := 435,
    lifecycleGroup := 1, phaseKind := 22,
    scheduleSelectorColumn := 28038855, lifecycleSelectorColumn := 649,
    phaseSelectorColumn := 28033366,
    sourceRows := 31339296, sourceColumns := 31063352, sourcePublicColumns := 641,
    finalRows := 10332059, finalColumns := 28038960, finalPublicColumns := 648,
    columnLayout :=
      { publicColumns := { start := 0, stop := 648 }, lifecyclePrivate := { start := 648, stop := 28033344 }, phasePrivate := { start := 28033344, stop := 28038420 },
        scheduleSelectorRuns := [{ start := 28038420, stop := 28038856 }], scheduledRingPadding := { start := 28038856, stop := 28038906 },
        overlayPrivate := { start := 28038906, stop := 28038959 }, overlaySelectorRuns := [{ start := 28038906, stop := 28038908 }],
        finalRingPadding := { start := 28038959, stop := 28038960 } },
    sourceStageOccurrence := 11883, sourceStagePath := "fprime.recursive.finalize.semantic_links",
    sourceStageRows := { start := 30676324, stop := 31339295 }, finalStageRuns := finalStageRuns,
    afterXOutSourceFields := [30382557, 642, 643, 644, 645, 646, 647, 648, 649, 30382558, 30382559, 30382560, 30382561, 30382562, 30382563, 722, 723, 724, 725, 30362945, 30362946, 30362947, 30362948, 30362937, 30362938, 30362939, 30362940, 30382564, 30382553, 30382554, 30382555, 30382556],
    afterNebulaLaneSourceFields := [672, 673, 674, 675, 19341627, 19341628, 19341629, 19231123, 19341630, 19341631, 19341632, 19341633, 19341634, 19341635, 19341636, 19341637, 19341638, 19341639, 19341640, 19341641, 19341642, 19341643, 19341644, 19341645, 19341646, 19341647, 19341648, 19341649, 19341650, 19341651, 19341652, 19341653, 19341654, 19341655, 19341656, 19341657, 19341658, 19341659, 19341660, 19341661, 19341662, 19341663, 19341664, 19341665, 19341666, 19341667, 19341668, 19341669, 19341670, 19341671],
    afterLocalStateDigest :=
      { sourceFields := { start := 30400385, stop := 30400389 }, sourceDomain := "goldilocks", finalLinkRows := { start := 10329342, stop := 10329346 } },
    afterDelayedPayload :=
      { sourceFields := { start := 30400389, stop := 30402558 }, sourceDomain := "boolean", finalLinkRows := { start := 10329346, stop := 10331515 } },
    sourceStageBindings := sourceStageBindings }

theorem finalStageRunsWithin :
    FinalRunsWithin rawArtifact.finalRows finalStageRuns := by
  unfold finalStageRuns
  exact FinalRunsWithin.cons (by decide) (FinalRunsWithin.cons (by decide) (FinalRunsWithin.nil))

theorem sourceStageBindingsWithin :
    SourceStageBindingsWithin rawArtifact.sourceRows rawArtifact.finalRows
      sourceStageBindings := by
  unfold sourceStageBindings
  exact SourceStageBindingsWithin.cons (by decide) (FinalRunsWithin.cons (by decide) (FinalRunsWithin.cons (by decide) (FinalRunsWithin.nil))) (SourceStageBindingsWithin.cons (by decide) (FinalRunsWithin.cons (by decide) (FinalRunsWithin.nil)) (SourceStageBindingsWithin.cons (by decide) (FinalRunsWithin.cons (by decide) (FinalRunsWithin.cons (by decide) (FinalRunsWithin.cons (by decide) (FinalRunsWithin.cons (by decide) (FinalRunsWithin.nil))))) (SourceStageBindingsWithin.cons (by decide) (FinalRunsWithin.cons (by decide) (FinalRunsWithin.cons (by decide) (FinalRunsWithin.nil))) (SourceStageBindingsWithin.cons (by decide) (FinalRunsWithin.cons (by decide) (FinalRunsWithin.cons (by decide) (FinalRunsWithin.nil))) (SourceStageBindingsWithin.cons (by decide) (FinalRunsWithin.cons (by decide) (FinalRunsWithin.cons (by decide) (FinalRunsWithin.nil))) (SourceStageBindingsWithin.nil))))))

end Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalProfile
