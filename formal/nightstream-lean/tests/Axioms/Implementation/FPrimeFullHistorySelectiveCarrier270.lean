import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.NifsPaper.AlignedCompiler.ProductionCarrier
import tests.Axioms.Support

/-! Fail-closed dependency gate for the exact selective 270-carrier slice. -/

open Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedCompiler.ProductionCarrier

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedCompiler.ProductionCarrier.generated_layout_refines_model' depends on axioms: [Lean.trustCompiler] -/
#guard_msgs in
#audit_axioms generated_layout_refines_model

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedCompiler.ProductionCarrier.generated_padding_rows_shape' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms generated_padding_rows_shape

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedCompiler.ProductionCarrier.generated_padding_row_iff_zeroPin' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms generated_padding_row_iff_zeroPin

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedCompiler.ProductionCarrier.generated_padding_row_canonical_complete' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms generated_padding_row_canonical_complete
