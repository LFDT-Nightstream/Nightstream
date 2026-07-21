import Nightstream.Implementation.R1CS.Correspondence.Gadgets.PiDecStrictProductionCompiler.PaperBridge
import tests.Axioms.Support

/-! Fail-closed dependency gate for the reduced production strict-`PiDEC`
compiler and its typed paper bridge. -/

/-- info: 'Nightstream.Implementation.R1CS.PiDecStrictProductionCompiler.sound_noAdv' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiDecStrictProductionCompiler.sound_noAdv

/-- info: 'Nightstream.Implementation.R1CS.PiDecStrictProductionCompiler.complete_noAdv' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiDecStrictProductionCompiler.complete_noAdv

/-- info: 'Nightstream.Implementation.R1CS.PiDecStrictProductionCompiler.combined_source_saving' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiDecStrictProductionCompiler.combined_source_saving

/-- info: 'Nightstream.Implementation.R1CS.PiDecStrictProductionCompiler.PaperBridge.commitmentEquation' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiDecStrictProductionCompiler.PaperBridge.commitmentEquation

/-- info: 'Nightstream.Implementation.R1CS.PiDecStrictProductionCompiler.PaperBridge.evaluationEquation' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiDecStrictProductionCompiler.PaperBridge.evaluationEquation

/-- info: 'Nightstream.Implementation.R1CS.PiDecStrictProductionCompiler.PaperBridge.accepted_refines_typed' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiDecStrictProductionCompiler.PaperBridge.accepted_refines_typed

/-- info: 'Nightstream.Implementation.R1CS.PiDecStrictProductionCompiler.PaperBridge.accepted_refines_paper' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiDecStrictProductionCompiler.PaperBridge.accepted_refines_paper

/-- info: 'Nightstream.Implementation.R1CS.PiDecStrictProductionCompiler.PaperBridge.active_source_rows_saved_3500' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiDecStrictProductionCompiler.PaperBridge.active_source_rows_saved_3500
