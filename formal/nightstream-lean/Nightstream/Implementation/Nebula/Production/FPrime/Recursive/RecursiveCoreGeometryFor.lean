import Nightstream.Implementation.Nebula.NIFS.PiCCS.TranscriptGeometryFor
import Nightstream.Implementation.Nebula.Production.Memory.CheckedBatchRows
import Nightstream.Implementation.Nebula.Production.Memory.SegmentContinuationRows
import Nightstream.Implementation.Nebula.Production.FPrime.Lifetime.PriorStateAuthorityRowsFor
import Nightstream.Implementation.Nebula.Production.FPrime.Recursive.RecursiveSuccessorRowsFor
import Nightstream.Implementation.Nebula.NIFS.PiDEC.Rows
import Nightstream.Implementation.Nebula.NIFS.PiRLC.AlgebraRows
import Nightstream.Implementation.Nebula.NIFS.PiRLC.CandidateClassificationRows
import Nightstream.Implementation.Nebula.NIFS.PiRLC.FirstAcceptedBatchRows
import Nightstream.Implementation.Nebula.NIFS.PiRLC.TranscriptRows
import Nightstream.Implementation.Nebula.Production.NIFS.Core.NifsOutputRowsFor
import Nightstream.Implementation.Nebula.Production.Carrier.FieldNativeCompactChainRowsFor

/-!
Contract: exponent-indexed row census for the specified recursive core.

The count includes the complete product PiCCS occurrence, all PiRLC sampler
sections, PiRLC algebra, PiDEC, prior-state authority, the delayed and current
checked memory batches, the complete NIFS output carrier, the field-native
compact-chain block, the mandatory segment continuation, and the recursive
successor. The PiCCS count comes from emitted rows at the same `rowVariables`
used by the NIFS shape.

This is a lower bound for the final generated F-prime relation. It excludes
the application compiler and generated placement glue. Therefore it can rule
out an exponent, but it cannot select one.

Assurance tier: exponent-indexed row implementation.
-/

set_option autoImplicit false
set_option maxRecDepth 10000

namespace Nightstream.Implementation.Nebula.ProductionRecursiveCoreGeometryFor

open Nightstream.Protocol.Nebula.ProductionProfileCandidates

def productNifsRows (rowVariables : Nat) : Nat :=
  ProductPiCcsTranscriptGeometryFor.rowCount rowVariables +
    ProductPiRlcTranscriptRows.aggregateRowCount +
    ProductPiRlcCandidateClassificationRows.aggregateRowCount +
    ProductPiRlcFirstAcceptedBatchRows.aggregateRowCount +
    4817340 + 5400 + (rowVariables * 2 + 23760)

def knownCoreRows (candidate : Id) (rowVariables : Nat) : Nat :=
  productNifsRows rowVariables +
    ProductionPaperPriorStateAuthorityRowsFor.rowCount candidate rowVariables +
    2 * ProductionMemoryCheckedBatchRows.rowCount candidate +
    ProductionFieldNativeCompactChainRowsFor.rowCount +
    ProductionMemorySegmentContinuationRows.rowCount +
    ProductionRecursiveSuccessorRowsFor.rowCount rowVariables + 20 + 8

theorem productNifsRows_25 : productNifsRows 25 = 18905026 := by
  rw [productNifsRows,
    ProductPiCcsTranscriptGeometryFor.rowCount_25,
    ProductPiRlcTranscriptRows.aggregateRowCount_eq,
    ProductPiRlcCandidateClassificationRows.aggregateRowCount_eq,
    ProductPiRlcFirstAcceptedBatchRows.aggregateRowCount_eq]

theorem productNifsRows_26 : productNifsRows 26 = 18920205 := by
  rw [productNifsRows,
    ProductPiCcsTranscriptGeometryFor.rowCount_26,
    ProductPiRlcTranscriptRows.aggregateRowCount_eq,
    ProductPiRlcCandidateClassificationRows.aggregateRowCount_eq,
    ProductPiRlcFirstAcceptedBatchRows.aggregateRowCount_eq]

theorem recursiveSuccessorRows_25 :
    ProductionRecursiveSuccessorRowsFor.rowCount 25 = 7349486 := by
  rw [ProductionRecursiveSuccessorRowsFor.rowCount,
    ProductionSuccessorStateBindingRowsFor.successorPermutationCount_25,
    ProductionPreCarryDigestRowsFor.permutationCount_25]

theorem recursiveSuccessorRows_26 :
    ProductionRecursiveSuccessorRowsFor.rowCount 26 = 7349838 := by
  rw [ProductionRecursiveSuccessorRowsFor.rowCount,
    ProductionSuccessorStateBindingRowsFor.successorPermutationCount_26,
    ProductionPreCarryDigestRowsFor.permutationCount_26]

theorem priorStateRows_26_table :
    ProductionPaperPriorStateAuthorityRowsFor.rowCount .e1 26 = 7364831 /\
      ProductionPaperPriorStateAuthorityRowsFor.rowCount .e4 26 = 7402280 /\
      ProductionPaperPriorStateAuthorityRowsFor.rowCount .e8 26 = 7452412 /\
      ProductionPaperPriorStateAuthorityRowsFor.rowCount .e16 26 = 7552676 := by
  rcases ProductionMemoryBatchCcsLinkRows.candidate_row_count_table with
    ⟨link1, link4, link8, link16⟩
  constructor
  · rw [ProductionPaperPriorStateAuthorityRowsFor.rowCount,
      ProductionSuccessorStateBindingRowsFor.successorPermutationCount_26,
      link1]
  constructor
  · rw [ProductionPaperPriorStateAuthorityRowsFor.rowCount,
      ProductionSuccessorStateBindingRowsFor.successorPermutationCount_26,
      link4]
  constructor
  · rw [ProductionPaperPriorStateAuthorityRowsFor.rowCount,
      ProductionSuccessorStateBindingRowsFor.successorPermutationCount_26,
      link8]
  · rw [ProductionPaperPriorStateAuthorityRowsFor.rowCount,
      ProductionSuccessorStateBindingRowsFor.successorPermutationCount_26,
      link16]

theorem priorStateRows_25_table :
    ProductionPaperPriorStateAuthorityRowsFor.rowCount .e1 25 = 7364831 /\
      ProductionPaperPriorStateAuthorityRowsFor.rowCount .e4 25 = 7402280 /\
      ProductionPaperPriorStateAuthorityRowsFor.rowCount .e8 25 = 7452412 /\
      ProductionPaperPriorStateAuthorityRowsFor.rowCount .e16 25 = 7552676 := by
  rcases ProductionMemoryBatchCcsLinkRows.candidate_row_count_table with
    ⟨link1, link4, link8, link16⟩
  constructor
  · rw [ProductionPaperPriorStateAuthorityRowsFor.rowCount,
      ProductionSuccessorStateBindingRowsFor.successorPermutationCount_25,
      link1]
  constructor
  · rw [ProductionPaperPriorStateAuthorityRowsFor.rowCount,
      ProductionSuccessorStateBindingRowsFor.successorPermutationCount_25,
      link4]
  constructor
  · rw [ProductionPaperPriorStateAuthorityRowsFor.rowCount,
      ProductionSuccessorStateBindingRowsFor.successorPermutationCount_25,
      link8]
  · rw [ProductionPaperPriorStateAuthorityRowsFor.rowCount,
      ProductionSuccessorStateBindingRowsFor.successorPermutationCount_25,
      link16]

/-- Exact mandatory core table at the historical fixed-25 exponent. The
complete output carrier is included. These counts are useful only as lower
bounds because every candidate already exceeds the 25-variable cube. -/
theorem knownCoreRows_25_table :
    knownCoreRows .e1 25 = 34154167 /\
      knownCoreRows .e4 25 = 34355362 /\
      knownCoreRows .e8 25 = 34623822 /\
      knownCoreRows .e16 25 = 35160742 := by
  rcases priorStateRows_25_table with ⟨prior1, prior4, prior8, prior16⟩
  rcases ProductionMemoryCheckedBatchRows.candidate_row_count_table with
    ⟨memory1, memory4, memory8, memory16⟩
  constructor
  · rw [knownCoreRows, productNifsRows_25, prior1, memory1,
      ProductionFieldNativeCompactChainRowsFor.rowCount,
      ProductionMemorySegmentContinuationRows.rowCount,
      recursiveSuccessorRows_25]
  constructor
  · rw [knownCoreRows, productNifsRows_25, prior4, memory4,
      ProductionFieldNativeCompactChainRowsFor.rowCount,
      ProductionMemorySegmentContinuationRows.rowCount,
      recursiveSuccessorRows_25]
  constructor
  · rw [knownCoreRows, productNifsRows_25, prior8, memory8,
      ProductionFieldNativeCompactChainRowsFor.rowCount,
      ProductionMemorySegmentContinuationRows.rowCount,
      recursiveSuccessorRows_25]
  · rw [knownCoreRows, productNifsRows_25, prior16, memory16,
      ProductionFieldNativeCompactChainRowsFor.rowCount,
      ProductionMemorySegmentContinuationRows.rowCount,
      recursiveSuccessorRows_25]

/-- Exact core table at the first exponent not already ruled out. This does
not prove that the omitted generated sections fit exponent 26. -/
theorem knownCoreRows_26_table :
    knownCoreRows .e1 26 = 34169698 /\
      knownCoreRows .e4 26 = 34370893 /\
      knownCoreRows .e8 26 = 34639353 /\
      knownCoreRows .e16 26 = 35176273 := by
  rcases priorStateRows_26_table with ⟨prior1, prior4, prior8, prior16⟩
  rcases ProductionMemoryCheckedBatchRows.candidate_row_count_table with
    ⟨memory1, memory4, memory8, memory16⟩
  constructor
  · rw [knownCoreRows, productNifsRows_26, prior1, memory1,
      ProductionFieldNativeCompactChainRowsFor.rowCount,
      ProductionMemorySegmentContinuationRows.rowCount,
      recursiveSuccessorRows_26]
  constructor
  · rw [knownCoreRows, productNifsRows_26, prior4, memory4,
      ProductionFieldNativeCompactChainRowsFor.rowCount,
      ProductionMemorySegmentContinuationRows.rowCount,
      recursiveSuccessorRows_26]
  constructor
  · rw [knownCoreRows, productNifsRows_26, prior8, memory8,
      ProductionFieldNativeCompactChainRowsFor.rowCount,
      ProductionMemorySegmentContinuationRows.rowCount,
      recursiveSuccessorRows_26]
  · rw [knownCoreRows, productNifsRows_26, prior16, memory16,
      ProductionFieldNativeCompactChainRowsFor.rowCount,
      ProductionMemorySegmentContinuationRows.rowCount,
      recursiveSuccessorRows_26]

theorem knownCoreRows_26_exceed_25 (candidate : Id) :
    2 ^ 25 < knownCoreRows candidate 26 := by
  cases candidate with
  | e1 => rw [knownCoreRows_26_table.1]; decide
  | e4 => rw [knownCoreRows_26_table.2.1]; decide
  | e8 => rw [knownCoreRows_26_table.2.2.1]; decide
  | e16 => rw [knownCoreRows_26_table.2.2.2]; decide

theorem knownCoreRows_26_fit_26 (candidate : Id) :
    knownCoreRows candidate 26 <= 2 ^ 26 := by
  cases candidate with
  | e1 => rw [knownCoreRows_26_table.1]; decide
  | e4 => rw [knownCoreRows_26_table.2.1]; decide
  | e8 => rw [knownCoreRows_26_table.2.2.1]; decide
  | e16 => rw [knownCoreRows_26_table.2.2.2]; decide

theorem knownCoreRows_26_power_interval (candidate : Id) :
    2 ^ 25 < knownCoreRows candidate 26 /\
      knownCoreRows candidate 26 <= 2 ^ 26 :=
  ⟨knownCoreRows_26_exceed_25 candidate,
    knownCoreRows_26_fit_26 candidate⟩

end Nightstream.Implementation.Nebula.ProductionRecursiveCoreGeometryFor
