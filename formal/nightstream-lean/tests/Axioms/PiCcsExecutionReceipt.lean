import Nightstream.Implementation.Rust.PiCcsExecution
import tests.Axioms.Support

open Nightstream.Implementation.Rust.PiCcsExecution

/-- info: 'Nightstream.Implementation.Rust.PiCcsExecution.CachedDuplex.referencePermutationCached_eq_reference' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms CachedDuplex.referencePermutationCached_eq_reference

/-- info: 'Nightstream.Implementation.Rust.PiCcsExecution.CachedDuplex.absorbList_eq_reference' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms CachedDuplex.absorbList_eq_reference

/-- info: 'Nightstream.Implementation.Rust.PiCcsExecution.absorbFields_eq_reference' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms absorbFields_eq_reference

/-- info: 'Nightstream.Implementation.Rust.PiCcsExecution.squeezeK_eq_reference' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms squeezeK_eq_reference

/-- info: 'Nightstream.Implementation.Rust.PiCcsExecution.zeroAwareInitial_eq_paperInitial' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms zeroAwareInitial_eq_paperInitial

/-- info: 'Nightstream.Implementation.Rust.PiCcsExecution.zeroAwareTerminalFromMessage_eq_paper' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms zeroAwareTerminalFromMessage_eq_paper

/-- info: 'Nightstream.Implementation.Rust.PiCcsExecution.checkReceipt_sound' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms checkReceipt_sound
