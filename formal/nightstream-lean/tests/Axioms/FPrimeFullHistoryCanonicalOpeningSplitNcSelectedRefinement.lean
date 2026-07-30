import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.CanonicalOpeningSplitNc.SelectedVerifierRefinement
import tests.Axioms.Support

/-! Fail-closed dependency gate for the selected physical opening refinement. -/

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.CanonicalOpeningSplitNc.SelectedPhysicalRefinement.coordinateOffset_injective' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.CanonicalOpeningSplitNc.SelectedPhysicalRefinement.coordinateOffset_injective

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.CanonicalOpeningSplitNc.SelectedPhysicalRefinement.requiredLocalColumns_used' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.CanonicalOpeningSplitNc.SelectedPhysicalRefinement.requiredLocalColumns_used

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.CanonicalOpeningSplitNc.SelectedPhysicalRefinement.sourceRowColumns_owned' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.CanonicalOpeningSplitNc.SelectedPhysicalRefinement.sourceRowColumns_owned

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.CanonicalOpeningSplitNc.SelectedPhysicalRefinement.selected_digit_column_used' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.CanonicalOpeningSplitNc.SelectedPhysicalRefinement.selected_digit_column_used

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.CanonicalOpeningSplitNc.SelectedPhysicalRefinement.selected_borrow_column_used' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.CanonicalOpeningSplitNc.SelectedPhysicalRefinement.selected_borrow_column_used

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.CanonicalOpeningSplitNc.SelectedPhysicalRefinement.emittedRows_columns_owned' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.CanonicalOpeningSplitNc.SelectedPhysicalRefinement.emittedRows_columns_owned

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.CanonicalOpeningSplitNc.SelectedPhysicalRefinement.selectedSplitNc_covers_opening' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.CanonicalOpeningSplitNc.SelectedPhysicalRefinement.selectedSplitNc_covers_opening

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.CanonicalOpeningSplitNc.SelectedPhysicalRefinement.selectedPhysicalRows_encoded_lt_modulus' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.CanonicalOpeningSplitNc.SelectedPhysicalRefinement.selectedPhysicalRows_encoded_lt_modulus

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.CanonicalOpeningSplitNc.SelectedVerifierRefinement.ncTruth_or_securityEvent_of_selectedVerifierRows' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.CanonicalOpeningSplitNc.SelectedVerifierRefinement.ncTruth_or_securityEvent_of_selectedVerifierRows

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.CanonicalOpeningSplitNc.SelectedVerifierRefinement.selectedVerifierAndPhysicalRows_encoded_lt_modulus_or_securityEvent' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.CanonicalOpeningSplitNc.SelectedVerifierRefinement.selectedVerifierAndPhysicalRows_encoded_lt_modulus_or_securityEvent
