import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.NifsPaper.AlignedCompiler.CommitmentShape

/-! Compile and dependency checks for aligned Ajtai setup shape. -/

namespace tests.FPrimeFullHistoryAlignedCommitmentShape

open Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedCompiler.CommitmentShape

#check setupColumns_eq_blockCount
#check packAssignment_length
#check alignedKeyShape_matches_packing
#check setupColumns_exactRingWidth
#check paperDimensions_setupColumns
#check fixedPublicSetupColumns
#check sameShape_setupFacts
#check sameShape_legacyKey_commitment_changes
#check sameShape_legacyKey_commitment_not_preserved

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedCompiler.CommitmentShape.setupColumns_eq_blockCount' does not depend on any axioms -/
#guard_msgs in
#print axioms setupColumns_eq_blockCount

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedCompiler.CommitmentShape.packAssignment_length' depends on axioms: [propext] -/
#guard_msgs in
#print axioms packAssignment_length

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedCompiler.CommitmentShape.alignedKeyShape_matches_packing' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#print axioms alignedKeyShape_matches_packing

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedCompiler.CommitmentShape.setupColumns_exactRingWidth' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#print axioms setupColumns_exactRingWidth

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedCompiler.CommitmentShape.paperDimensions_setupColumns' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#print axioms paperDimensions_setupColumns

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedCompiler.CommitmentShape.fixedPublicSetupColumns' does not depend on any axioms -/
#guard_msgs in
#print axioms fixedPublicSetupColumns

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedCompiler.CommitmentShape.sameShape_setupFacts' depends on axioms: [propext] -/
#guard_msgs in
#print axioms sameShape_setupFacts

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedCompiler.CommitmentShape.sameShape_legacyKey_commitment_changes' depends on axioms: [propext] -/
#guard_msgs in
#print axioms sameShape_legacyKey_commitment_changes

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedCompiler.CommitmentShape.sameShape_legacyKey_commitment_not_preserved' depends on axioms: [propext] -/
#guard_msgs in
#print axioms sameShape_legacyKey_commitment_not_preserved

end tests.FPrimeFullHistoryAlignedCommitmentShape
