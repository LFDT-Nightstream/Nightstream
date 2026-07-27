import Nightstream.Implementation.R1CS.Canonical.Poseidon2Support
import tests.Axioms.Support

/-!
Fail-closed dependency gate for the Poseidon2 support recurrence and its
exact partial-block characterization.
No theorem may acquire `Lean.trustCompiler`.
-/

namespace NightstreamTests.Axioms.CanonicalPoseidon2Support

open NightstreamTests.Axioms
open Nightstream.Implementation.R1CS.Canonical

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Support.mentions_applyMatrix' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Support.mentions_applyMatrix

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Support.mentions_addConstant' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Support.mentions_addConstant

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Support.partialSupportList_length' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Support.partialSupportList_length

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Support.partialState_mentions_subset' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Support.partialState_mentions_subset

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Support.partialSupport_bound' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Support.partialSupport_bound

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Support.partialState_mentions_fresh' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Support.partialState_mentions_fresh

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Support.partialState_zero_mentions_output' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Support.partialState_zero_mentions_output

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Support.terminalState_zero_mentions_subset' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Support.terminalState_zero_mentions_subset

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Support.terminalState_succ_mentions' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Support.terminalState_succ_mentions

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Support.scheduleOf_partial' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Support.scheduleOf_partial

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Support.partialSboxInput_mentions_bound' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Support.partialSboxInput_mentions_bound

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Support.partialState_mentions_superset' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Support.partialState_mentions_superset

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Support.partialSupportList_index' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Support.partialSupportList_index

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Support.sboxOutput_injective' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Support.sboxOutput_injective

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Support.partialSupportList_nodup' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Support.partialSupportList_nodup

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Support.partialState_normalize_length' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Support.partialState_normalize_length

/-! Support with a general entry state. -/

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Support.initialStateFrom_zero_mentions' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Support.initialStateFrom_zero_mentions

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Support.initialStateFrom_succ_mentions' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Support.initialStateFrom_succ_mentions

/-! Normalized length for an arbitrary entry state. -/

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Support.normalize_length_applyMatrix_witness' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Support.normalize_length_applyMatrix_witness

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Support.initialStateFrom_zero_normalize_length' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Support.initialStateFrom_zero_normalize_length

end NightstreamTests.Axioms.CanonicalPoseidon2Support
