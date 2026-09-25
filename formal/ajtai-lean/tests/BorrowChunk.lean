import Ajtai.BorrowChunk

namespace AjtaiTests.BorrowChunk

open Nightstream.Implementation.R1CS.CenteredTernaryBorrowChunk

example : chunkEquations.length = 21 :=
  chunkEquations_length

example : chunkBorrowCount = 20 :=
  chunkBorrowCount_eq

example : scheduledDigitIndices =
    List.range
      Nightstream.Implementation.R1CS.ShiftedTernaryCompiler.digitCount :=
  scheduledDigitIndices_eq_range

example :
    (List.range chunkCount).map normalizedChunkBound =
      [3, 0, 3, 3, 3, 0, 1, 3, 1, 2, 4, 3, 2, 1, 3, 0, 0, 0, 3, 4, 1] :=
  normalizedChunkBounds

example {chunk : Nat} (chunkLt : chunk < chunkCount) :
    normalizedChunkBound chunk < 5 :=
  normalizedChunkBound_lt_five chunkLt

example :
    ∀ equation ∈ chunkEquations, equation.degree ≤ 5 :=
  chunkEquations_degree_le_five

example : maximumChunkDegree = 5 :=
  maximumChunkDegree_eq_five

example : openingCoordinateCount = 61 :=
  openingCoordinateCount_eq

example : rowsForRankTwoCommitments 23033 1 = 483801 :=
  activeProfile_oneCommitment_rows

example : coordinatesForRankTwoCommitments 23033 1 = 1409441 :=
  activeProfile_oneCommitment_coordinates

example : rowsForRankTwoDigestChains 23033 1 = 486123 :=
  activeProfile_oneDigestChain_rows

example : coordinatesForRankTwoDigestChains 23033 1 = 1413815 :=
  activeProfile_oneDigestChain_coordinates

example :
    ∀ equation ∈ chunkEquations,
      uniformSelectorGatedDegree equation ≤ 7 :=
  uniformSelectorGatedDegrees_le_seven

example
    (bounds trits : Fin 2 → Nat) (input output : Nat) :
    output = chunkTwo bounds trits input ↔
      ∃ middle,
        middle = scalarStep (bounds 0) (trits 0) input ∧
        output = scalarStep (bounds 1) (trits 1) middle :=
  chunkTwo_iff_scalarWitness bounds trits input output

example
    {boundZero boundOne tritZero tritOne borrow : Nat}
    (boundZeroLt : boundZero < 3)
    (boundOneLt : boundOne < 3)
    (tritZeroLt : tritZero < 3)
    (tritOneLt : tritOne < 3)
    (borrowLe : borrow ≤ 1) :
    1 - scalarTwoValues boundZero boundOne tritZero tritOne borrow =
      scalarTwoValues
        (2 - boundZero) (2 - boundOne)
        (2 - tritZero) (2 - tritOne) (1 - borrow) :=
  scalarTwoValues_complement
    boundZeroLt boundOneLt tritZeroLt tritOneLt borrowLe

example
    {assignment : Nat → Nat}
    (norm :
      Nightstream.Implementation.R1CS.CenteredTernaryNormDischarged.DigitNormBoundTwo
        assignment)
    (holds : ChunkScheduleHolds assignment) :
    Nightstream.Implementation.R1CS.ShiftedTernarySound.lowValue
        (assignmentTritMod assignment)
        Nightstream.Implementation.R1CS.ShiftedTernaryCompiler.digitCount <
      Nightstream.Implementation.R1CS.goldilocksP :=
  chunkSchedule_encoded_lt_modulus norm holds

end AjtaiTests.BorrowChunk
