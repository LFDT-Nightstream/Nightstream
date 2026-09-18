import Nightstream.Implementation.R1CS.Core.SeededPhi81

/-! Generated placement schema for production full-history SeededPhi81 blocks. -/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistorySeededPhi81

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.SeededPhi81

def totalRows : Nat := 4076614
def totalColumns : Nat := 3298653
def rejectionFuel : Nat := 16

def rowEnd (block : SeededPhi81.Block) : Nat :=
  block.rowStart + SeededPhi81.dimension * block.kappa

def MetadataValid (block : SeededPhi81.Block) : Prop :=
  0 < block.wordWidth ∧ 0 < block.kappa ∧
  0 < block.schedule.chunkSize ∧
  block.superneoTransformedColumns = false ∧
  block.messageCols =
    (block.wordStarts.length * block.wordWidth + SeededPhi81.dimension - 1) /
      SeededPhi81.dimension ∧
  block.outputColumns.length = SeededPhi81.dimension * block.kappa ∧
  block.schedule.seedsByOutput.length = block.kappa ∧
  ∀ seeds ∈ block.schedule.seedsByOutput,
    seeds.length =
      (block.messageCols + block.schedule.chunkSize - 1) /
        block.schedule.chunkSize

def RowsMapped (block : SeededPhi81.Block) : Prop :=
  rowEnd block ≤ totalRows ∧
  block.outputColumns.length = SeededPhi81.dimension * block.kappa

instance (block : SeededPhi81.Block) : Decidable (MetadataValid block) := by
  unfold MetadataValid
  infer_instance

instance (block : SeededPhi81.Block) : Decidable (RowsMapped block) := by
  unfold RowsMapped rowEnd
  infer_instance

def CertifiedBlock :=
  { block : SeededPhi81.Block //
    block.Valid ∧ MetadataValid block ∧ RowsMapped block ∧
    block.superneoTransformedColumns = false }

end Nightstream.Implementation.R1CS.FPrimeFullHistorySeededPhi81
