import Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalSamplerProgramConservation
import Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalSamplerSound
import tests.Axioms.Support

/-!
Fail-closed axiom guard for the complete fixed-active canonical `Pi_RLC`
sampler program.

Every expected report below was measured from the kernel dependency graph.
`Classical.choice` enters through finite-list ownership and membership proofs;
none of these theorems assumes a protocol conclusion.
-/

namespace NightstreamTests.Axioms.CanonicalPiRlcCanonicalSamplerProgram

open NightstreamTests.Axioms
open Nightstream.Implementation.R1CS.Canonical

/-- info: 'Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalSelector.embedCoefficient_val_eq_shift' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms PiRlcCanonicalSelector.embedCoefficient_val_eq_shift

/-- info: 'Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalSelectorSound.position_refines' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms PiRlcCanonicalSelectorSound.position_refines

/-- info: 'Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalSamplerSound.outputs_eq_embeddedFirstAccepted' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms PiRlcCanonicalSamplerSound.outputs_eq_embeddedFirstAccepted

/-- info: 'Nightstream.Implementation.R1CS.Canonical.SymbolicDuplexPhysical.ownership_is_positional' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms SymbolicDuplexPhysical.ownership_is_positional

/-- info: 'Nightstream.Implementation.R1CS.Canonical.SymbolicDuplexPhysical.rows_conservation' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms SymbolicDuplexPhysical.rows_conservation

/-- info: 'Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalSymbolicMachineHonest.fixedRows_honest' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms PiRlcCanonicalSymbolicMachineHonest.fixedRows_honest

/-- info: 'Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalSymbolicMachineHonest.fixedRows_ownership' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms PiRlcCanonicalSymbolicMachineHonest.fixedRows_ownership

/-- info: 'Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalSymbolicMachineHonest.fixedRows_conservation' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms PiRlcCanonicalSymbolicMachineHonest.fixedRows_conservation

/-- info: 'Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalSamplerHonest.suffixRows_complete' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms PiRlcCanonicalSamplerHonest.suffixRows_complete

/-- info: 'Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalU64Placement.laneInput_member_temporaryColumns' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms PiRlcCanonicalU64Placement.laneInput_member_temporaryColumns

/-- info: 'Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalCandidateConservation.rows_conservation' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms PiRlcCanonicalCandidateConservation.rows_conservation

/-- info: 'Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalSelectorConservation.rows_conservation' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms PiRlcCanonicalSelectorConservation.rows_conservation

/-- info: 'Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalSamplerProgram.rows_length' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms PiRlcCanonicalSamplerProgram.rows_length

/-- info: 'Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalSamplerProgram.allocation_length' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms PiRlcCanonicalSamplerProgram.allocation_length

/-- info: 'Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalSamplerProgram.allocation_nodup' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms PiRlcCanonicalSamplerProgram.allocation_nodup

/-- info: 'Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalSamplerProgram.ownership_is_positional' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms PiRlcCanonicalSamplerProgram.ownership_is_positional

/-- info: 'Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalSamplerProgram.rows_complete' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms PiRlcCanonicalSamplerProgram.rows_complete

/-- info: 'Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalSamplerProgramConservation.suffixRows_conservation' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms PiRlcCanonicalSamplerProgramConservation.suffixRows_conservation

/-- info: 'Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalSamplerProgramConservation.rows_conservation' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms PiRlcCanonicalSamplerProgramConservation.rows_conservation

end NightstreamTests.Axioms.CanonicalPiRlcCanonicalSamplerProgram
