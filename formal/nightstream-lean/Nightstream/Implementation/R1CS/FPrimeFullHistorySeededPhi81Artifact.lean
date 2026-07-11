import Nightstream.Implementation.R1CS.FPrimeFullHistorySeededPhi81Block0
import Nightstream.Implementation.R1CS.FPrimeFullHistorySeededPhi81Block1
import Nightstream.Implementation.R1CS.FPrimeFullHistorySeededPhi81Block2
import Nightstream.Implementation.R1CS.FPrimeFullHistorySeededPhi81Block3
import Nightstream.Implementation.R1CS.FPrimeFullHistorySeededPhi81Block4
import Nightstream.Implementation.R1CS.FPrimeFullHistorySeededPhi81Block5
import Nightstream.Implementation.R1CS.FPrimeFullHistorySeededPhi81Block6
import Nightstream.Implementation.R1CS.FPrimeFullHistorySeededPhi81Block7
import Nightstream.Implementation.R1CS.FPrimeFullHistorySeededPhi81Block8
import Nightstream.Implementation.R1CS.FPrimeFullHistorySeededPhi81Block9
import Nightstream.Implementation.R1CS.FPrimeFullHistorySeededPhi81Block10
import Nightstream.Implementation.R1CS.FPrimeFullHistorySeededPhi81Block11
import Nightstream.Implementation.R1CS.FPrimeFullHistorySeededPhi81Block12
import Nightstream.Implementation.R1CS.FPrimeFullHistorySeededPhi81Block13
import Nightstream.Implementation.R1CS.FPrimeFullHistorySeededPhi81Block14
import Nightstream.Implementation.R1CS.FPrimeFullHistorySeededPhi81Block15
import Nightstream.Implementation.R1CS.FPrimeFullHistorySeededPhi81Block16
import Nightstream.Implementation.R1CS.FPrimeFullHistorySeededPhi81Block17

/-! Generated certified index of every production full-history SeededPhi81 block. -/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistorySeededPhi81

def certifiedBlocks : List CertifiedBlock :=
  [certifiedBlock0, certifiedBlock1, certifiedBlock2, certifiedBlock3, certifiedBlock4, certifiedBlock5, certifiedBlock6, certifiedBlock7, certifiedBlock8, certifiedBlock9, certifiedBlock10, certifiedBlock11, certifiedBlock12, certifiedBlock13, certifiedBlock14, certifiedBlock15, certifiedBlock16, certifiedBlock17]

def blocks : List SeededPhi81.Block := certifiedBlocks.map Subtype.val

theorem blocks_valid {block : SeededPhi81.Block} (member : block ∈ blocks) :
    block.Valid := by
  rw [blocks] at member
  rcases List.mem_map.mp member with ⟨certified, _, rfl⟩
  exact certified.property.1

theorem metadata_valid {block : SeededPhi81.Block} (member : block ∈ blocks) :
    MetadataValid block := by
  rw [blocks] at member
  rcases List.mem_map.mp member with ⟨certified, _, rfl⟩
  exact certified.property.2.1

theorem rows_mapped {block : SeededPhi81.Block} (member : block ∈ blocks) :
    RowsMapped block := by
  rw [blocks] at member
  rcases List.mem_map.mp member with ⟨certified, _, rfl⟩
  exact certified.property.2.2.1

theorem transformed_columns_status {block : SeededPhi81.Block}
    (member : block ∈ blocks) : block.superneoTransformedColumns = false := by
  rw [blocks] at member
  rcases List.mem_map.mp member with ⟨certified, _, rfl⟩
  exact certified.property.2.2.2

end Nightstream.Implementation.R1CS.FPrimeFullHistorySeededPhi81
