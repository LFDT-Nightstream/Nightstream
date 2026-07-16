import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.NifsPaper.AlignedCeRelation

/-! Kernel theorem and dependency checks for aligned CE and norm transport. -/

namespace tests.FPrimeFullHistoryAlignedCeRelation

open Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedCeRelation

#check normBounded_insertPublicPadding_iff
#check evaluationPointValid_align_iff
#check matrixEvaluations_align
#check legacyKey_commitment_changes
#check legacyKey_commitment_not_preserved
#check alignedCanonicalCE_holds

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedCeRelation.normBounded_insertPublicPadding_iff' depends on axioms: [propext] -/
#guard_msgs in
#print axioms normBounded_insertPublicPadding_iff

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedCeRelation.matrixEvaluations_align' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#print axioms matrixEvaluations_align

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedCeRelation.legacyKey_commitment_not_preserved' depends on axioms: [propext] -/
#guard_msgs in
#print axioms legacyKey_commitment_not_preserved

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedCeRelation.alignedCanonicalCE_holds' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#print axioms alignedCanonicalCE_holds

end tests.FPrimeFullHistoryAlignedCeRelation
