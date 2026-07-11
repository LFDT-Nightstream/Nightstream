import Nightstream.Implementation.R1CS.FPrimeRecursiveManifestData

/-!
Contract: kernel-checked coverage of one exact production recursive F' program.

The Rust drift harness constructs the real plain/stateless steady recursive
step and byte-compares `FPrimeRecursiveManifestData.lean` against metadata
rendered from its exact sparse rows. The proofs below establish that those
generated named ranges cover the entire program and nested NIFS block without
gaps or overlaps.

Range identity is not circuit soundness. No hash or row count is authority for
a protocol proposition; separate compiler lemmas must prove that satisfaction
of each exact range implies its semantic owner fact.
-/

namespace Nightstream.Implementation.R1CS.FPrimeRecursiveManifest

theorem topLevel_covers_program : covers 0 totalRows topLevelFamilies = true := by
  decide

theorem nifs_covers_block : covers 20038 2592246 nifsFamilies = true := by
  decide

theorem topLevel_row_count :
    (topLevelFamilies.map RowRange.rowCount).sum = totalRows := by
  decide

theorem nifs_row_count :
    (nifsFamilies.map RowRange.rowCount).sum = 2572208 := by
  decide

theorem nifs_matches_top_level :
    (topLevelFamilies[2]!).rowStart = (nifsFamilies[0]!).rowStart ∧
    (topLevelFamilies[2]!).rowEnd = (nifsFamilies[3]!).rowEnd := by
  decide

/-- The exact steady-state profile has one shared beta/rho block and 31
projection identities, all owned by PiRLC. -/
theorem projection_census_shape :
    schemaVersion = 2 ∧
    projectionIdentityRanges.length = projectionIdentityCount ∧
    projectionPairCounts.length = projectionIdentityCount ∧
    projectionIdentityCount = 31 := by
  decide

theorem projection_shared_row_count :
    projectionShared.rowCount = 1892 := by
  decide

theorem projection_identity_row_counts :
    (projectionIdentityRanges.map RowRange.rowCount).sum =
      projectionIdentityRows ∧
    projectionIdentityRows = 59396 ∧
    ∀ range ∈ projectionIdentityRanges, range.rowCount = 1916 := by
  decide

theorem projection_pair_census :
    ∀ count ∈ projectionPairCounts, count = 15 := by
  decide

theorem projection_ranges_owned_by_piRlc :
    (nifsFamilies[1]!).rowStart ≤ projectionShared.rowStart ∧
    projectionShared.rowEnd ≤ (nifsFamilies[1]!).rowEnd ∧
    (∀ range ∈ projectionIdentityRanges,
      (nifsFamilies[1]!).rowStart ≤ range.rowStart ∧
      range.rowEnd ≤ (nifsFamilies[1]!).rowEnd) := by
  decide

end Nightstream.Implementation.R1CS.FPrimeRecursiveManifest
