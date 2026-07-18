import Nightstream.Assurance.FPrimeRecursiveCircuit

namespace NightstreamTests.FPrimeRecursiveManifest

open Nightstream.Implementation.R1CS.FPrimeRecursiveManifest

example : covers 0 totalRows topLevelFamilies = true :=
  topLevel_covers_program

example : covers nifsRowStart nifsRowEnd nifsFamilies = true :=
  nifs_covers_block

example : (topLevelFamilies.map RowRange.rowCount).sum = totalRows :=
  topLevel_row_count

example : (nifsFamilies.map RowRange.rowCount).sum = nifsRowCount :=
  nifs_row_count

example : projectionIdentityRanges.length = 31 := by
  decide

example : projectionShared.rowCount = 1892 :=
  projection_shared_row_count

example : (projectionIdentityRanges.map RowRange.rowCount).sum = 59396 := by
  exact projection_identity_row_counts.1

example : ∀ count ∈ projectionPairCounts, count = 15 :=
  projection_pair_census

def forgedGap : List RowRange :=
  match topLevelFamilies with
  | prelude :: transcript :: rest =>
      prelude :: { transcript with rowStart := transcript.rowStart + 1 } :: rest
  | _ => []

def forgedOverlap : List RowRange :=
  match nifsFamilies with
  | piCcs :: piRlc :: rest =>
      piCcs :: { piRlc with rowStart := piRlc.rowStart - 1 } :: rest
  | _ => []

example : covers 0 totalRows forgedGap = false := by
  decide

example : covers nifsRowStart nifsRowEnd forgedOverlap = false := by
  decide

#check Nightstream.Assurance.FPrimeConcreteNifs.recursive_rows_sound
#check Nightstream.Assurance.FPrimeConcreteNifs.recursive_verify_sound_or_badRoot

end NightstreamTests.FPrimeRecursiveManifest
