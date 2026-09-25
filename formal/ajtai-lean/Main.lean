import Ajtai

open Ajtai.EstimatorModel
open Nightstream.Implementation.R1CS.CenteredTernaryBorrowChunk

def main : IO Unit := do
  IO.println s!"maximum_rank_two_ring_columns={computedMaxRingColumns}"
  IO.println s!"maximum_rank_two_source_fields={computedMaxSourceFields}"
  IO.println s!"minimum_accepted_beta={minimumAcceptedBeta}"
  IO.println s!"chunks={chunkEquations.length}"
  IO.println s!"chunk_borrows={chunkBorrowCount}"
  IO.println s!"covered_digits={
    (List.range chunkCount).foldl
      (fun count chunk => count + (chunkEntries chunk).length) 0}"
  IO.println s!"rows_23033={rowsForRankTwoCommitments 23033 1}"
  IO.println s!"coordinates_23033={
    coordinatesForRankTwoCommitments 23033 1}"
  IO.println s!"digest_rows_23033={rowsForRankTwoDigestChains 23033 1}"
  IO.println s!"digest_coordinates_23033={
    coordinatesForRankTwoDigestChains 23033 1}"
