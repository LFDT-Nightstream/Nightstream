import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.Refinement.Terminal.RoundExecution
import tests.Axioms.Support

/-! Fail-closed dependency gate for shared terminal FE/NC round serialization. -/

/-- info: 'Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.Terminal.RoundExecution.fieldAt_eq_wordField' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.Terminal.RoundExecution.fieldAt_eq_wordField

/-- info: 'Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.Terminal.RoundExecution.absorbAll_four_of_cursorZero' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.Terminal.RoundExecution.absorbAll_four_of_cursorZero

/-- info: 'Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.Terminal.RoundExecution.absorbAll_three_of_cursorOne' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.Terminal.RoundExecution.absorbAll_three_of_cursorOne

/-- info: 'Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.Terminal.RoundExecution.appendRaw_ten_of_cursorZero' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.Terminal.RoundExecution.appendRaw_ten_of_cursorZero

/-- info: 'Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.Terminal.RoundExecution.appendRaw_singletons_of_cursorZero' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.Terminal.RoundExecution.appendRaw_singletons_of_cursorZero

/-- info: 'Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.Terminal.RoundExecution.appendRaw_pair_then_singleton_of_cursorZero' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.Terminal.RoundExecution.appendRaw_pair_then_singleton_of_cursorZero

/-- info: 'Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.Terminal.RoundExecution.appendRaw_ten_of_cursorOne' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.Terminal.RoundExecution.appendRaw_ten_of_cursorOne
