import Nightstream.ChunkLayout
import Nightstream.BridgeTypes
import Nightstream.Rv64IM.Trace.MainLaneTraceBoundary
import SuperNeo.FoldingProtocol.PiDECInterface

/-!
Owns the theorem-facing chunked root execution proof surface for RV64IM. This
file packages the canonical chunk layout induced by the explicit fold schedule
with per-chunk SuperNeo backend statements; it does not own Twist/Shout
selection or the Nightstream bridge.
-/

namespace Nightstream.Rv64IM

open Nightstream

structure RootChunkBackendProofPackage where
  chunkIndex : Nat
  chunk : Nightstream.ChunkRange
  rowLabels : List Nat
  protocolTarget : SuperNeo.ProtocolTargetContext
  rowLabelsExact :
    rowLabels = List.range' chunk.start (Nightstream.ChunkRange.width chunk)
  piCCSStrong : Nightstream.SuperNeoPiCCSStrongStatement protocolTarget
  piRLCWeak : Nightstream.SuperNeoPiRLCWeakStatement protocolTarget

structure ChunkedRootProofPackage (Row PreparedStep : Type _) where
  mainLane : MainLaneTraceBoundaryProofPackage Row PreparedStep
  chunkProofs : List RootChunkBackendProofPackage
  chunkProofCount : chunkProofs.length = mainLane.chunks.length
  chunkProofAligned :
    ∀ j, j < mainLane.chunks.length →
      ∃ pkg,
        chunkProofs[j]? = some pkg ∧
          pkg.chunkIndex = j ∧
          mainLane.chunks[j]? = some pkg.chunk

theorem chunkedRootProof_scheduleValid
  {Row PreparedStep : Type _}
  (pkg : ChunkedRootProofPackage Row PreparedStep) :
  FoldSchedule.Valid pkg.mainLane.schedule :=
  mainLaneTraceBoundary_scheduleValid pkg.mainLane

theorem chunkedRootProof_chunksLayout
  {Row PreparedStep : Type _}
  (pkg : ChunkedRootProofPackage Row PreparedStep) :
  pkg.mainLane.chunks = ChunkLayout.layout pkg.mainLane.schedule pkg.mainLane.semanticRows :=
  mainLaneTraceBoundary_chunksLayout pkg.mainLane

theorem chunkedRootProof_chunkCount
  {Row PreparedStep : Type _}
  (pkg : ChunkedRootProofPackage Row PreparedStep) :
  pkg.chunkProofs.length =
    FoldSchedule.chunkCount pkg.mainLane.schedule pkg.mainLane.semanticRows := by
  calc
    pkg.chunkProofs.length = pkg.mainLane.chunks.length := pkg.chunkProofCount
    _ = FoldSchedule.chunkCount pkg.mainLane.schedule pkg.mainLane.semanticRows :=
      mainLaneTraceBoundary_chunksLength pkg.mainLane

theorem backendPackageAtIndex_of_chunkedRootProof
  {Row PreparedStep : Type _}
  {pkg : ChunkedRootProofPackage Row PreparedStep}
  {j : Nat}
  (hJ : j < pkg.mainLane.chunks.length) :
  ∃ backendPkg,
    pkg.chunkProofs[j]? = some backendPkg ∧
      backendPkg.chunkIndex = j ∧
      pkg.mainLane.chunks[j]? = some backendPkg.chunk :=
  pkg.chunkProofAligned j hJ

theorem rowLabelsExact_atIndex_of_chunkedRootProof
  {Row PreparedStep : Type _}
  {pkg : ChunkedRootProofPackage Row PreparedStep}
  {j : Nat}
  (hJ : j < pkg.mainLane.chunks.length) :
  ∃ backendPkg,
    pkg.chunkProofs[j]? = some backendPkg ∧
      backendPkg.rowLabels =
        List.range' backendPkg.chunk.start (Nightstream.ChunkRange.width backendPkg.chunk) := by
  rcases backendPackageAtIndex_of_chunkedRootProof (pkg := pkg) hJ with
    ⟨backendPkg, hPkg, _, _⟩
  exact ⟨backendPkg, hPkg, backendPkg.rowLabelsExact⟩

theorem piCCS_atIndex_of_chunkedRootProof
  {Row PreparedStep : Type _}
  {pkg : ChunkedRootProofPackage Row PreparedStep}
  {j : Nat}
  (hJ : j < pkg.mainLane.chunks.length) :
  ∃ backendPkg,
    pkg.chunkProofs[j]? = some backendPkg ∧
      Nightstream.SuperNeoPiCCSStrongStatement backendPkg.protocolTarget := by
  rcases backendPackageAtIndex_of_chunkedRootProof (pkg := pkg) hJ with
    ⟨backendPkg, hPkg, _, _⟩
  exact ⟨backendPkg, hPkg, backendPkg.piCCSStrong⟩

theorem piRLC_atIndex_of_chunkedRootProof
  {Row PreparedStep : Type _}
  {pkg : ChunkedRootProofPackage Row PreparedStep}
  {j : Nat}
  (hJ : j < pkg.mainLane.chunks.length) :
  ∃ backendPkg,
    pkg.chunkProofs[j]? = some backendPkg ∧
      Nightstream.SuperNeoPiRLCWeakStatement backendPkg.protocolTarget := by
  rcases backendPackageAtIndex_of_chunkedRootProof (pkg := pkg) hJ with
    ⟨backendPkg, hPkg, _, _⟩
  exact ⟨backendPkg, hPkg, backendPkg.piRLCWeak⟩

theorem piDEC_atIndex_of_chunkedRootProof
  {Row PreparedStep : Type _}
  {pkg : ChunkedRootProofPackage Row PreparedStep}
  {j : Nat}
  (hJ : j < pkg.mainLane.chunks.length) :
  ∃ backendPkg,
    pkg.chunkProofs[j]? = some backendPkg ∧
      Nightstream.SuperNeoPiDECKnowledgeStatement backendPkg.protocolTarget := by
  rcases piRLC_atIndex_of_chunkedRootProof (pkg := pkg) hJ with
    ⟨backendPkg, hPkg, hPiRLC⟩
  exact
    ⟨backendPkg, hPkg, SuperNeo.PiDECInterface.piDEC_of_weak hPiRLC⟩

theorem chunkedRootProof_wholeTrace_singleChunk
  {Row PreparedStep : Type _}
  (pkg : ChunkedRootProofPackage Row PreparedStep)
  (hSchedule : pkg.mainLane.schedule = .wholeTrace) :
  pkg.chunkProofs.length = 1 := by
  calc
    pkg.chunkProofs.length =
      FoldSchedule.chunkCount pkg.mainLane.schedule pkg.mainLane.semanticRows :=
      chunkedRootProof_chunkCount pkg
    _ = FoldSchedule.chunkCount .wholeTrace pkg.mainLane.semanticRows := by
      simpa [hSchedule]
    _ = 1 := FoldSchedule.chunkCount_wholeTrace pkg.mainLane.semanticRows

theorem owningChunkIndex_lt_chunkProofCount_of_rowIndex
  {Row PreparedStep : Type _}
  {pkg : ChunkedRootProofPackage Row PreparedStep}
  {rowIndex : Nat}
  (hRow : rowIndex < pkg.mainLane.semanticRows) :
  Nightstream.ChunkLayout.chunkIndexOf pkg.mainLane.schedule rowIndex <
    pkg.chunkProofs.length := by
  have hLayout :
      Nightstream.ChunkLayout.chunkIndexOf pkg.mainLane.schedule rowIndex <
        FoldSchedule.chunkCount pkg.mainLane.schedule pkg.mainLane.semanticRows :=
    Nightstream.ChunkLayout.chunkIndexOf_lt_chunkCount_of_lt_preparedStepCount
      (chunkedRootProof_scheduleValid pkg)
      hRow
  simpa [chunkedRootProof_chunkCount pkg] using hLayout

theorem backendPackageAtOwningChunkIndex_of_rowIndex
  {Row PreparedStep : Type _}
  {pkg : ChunkedRootProofPackage Row PreparedStep}
  {rowIndex : Nat}
  (hRow : rowIndex < pkg.mainLane.semanticRows) :
  ∃ backendPkg,
    pkg.chunkProofs[Nightstream.ChunkLayout.chunkIndexOf pkg.mainLane.schedule rowIndex]? =
        some backendPkg ∧
      backendPkg.chunkIndex =
        Nightstream.ChunkLayout.chunkIndexOf pkg.mainLane.schedule rowIndex ∧
      pkg.mainLane.chunks[
          Nightstream.ChunkLayout.chunkIndexOf pkg.mainLane.schedule rowIndex]? =
        some backendPkg.chunk := by
  let chunkIndex := Nightstream.ChunkLayout.chunkIndexOf pkg.mainLane.schedule rowIndex
  have hChunk :
      chunkIndex < pkg.mainLane.chunks.length := by
    have hLayout :
        chunkIndex <
          FoldSchedule.chunkCount pkg.mainLane.schedule pkg.mainLane.semanticRows :=
      Nightstream.ChunkLayout.chunkIndexOf_lt_chunkCount_of_lt_preparedStepCount
        (chunkedRootProof_scheduleValid pkg)
        hRow
    simpa [mainLaneTraceBoundary_chunksLength pkg.mainLane] using hLayout
  simpa [chunkIndex] using backendPackageAtIndex_of_chunkedRootProof (pkg := pkg) hChunk

end Nightstream.Rv64IM
