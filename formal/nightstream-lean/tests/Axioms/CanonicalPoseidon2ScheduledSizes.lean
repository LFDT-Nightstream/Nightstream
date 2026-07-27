import Nightstream.Implementation.R1CS.Canonical.Poseidon2ScheduledSizes
import tests.Axioms.Support

/-!
Fail-closed dependency gate for the exact scheduled sizes and the
structural term total.
No theorem may acquire `Lean.trustCompiler`.
-/

namespace NightstreamTests.Axioms.CanonicalPoseidon2ScheduledSizes

open NightstreamTests.Axioms
open Nightstream.Implementation.R1CS.Canonical

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2ScheduledSizes.canonicalProgram_termCount_eq' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2ScheduledSizes.canonicalProgram_termCount_eq

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2ScheduledSizes.normalizedCanonicalProgram_termCount_bound' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2ScheduledSizes.normalizedCanonicalProgram_termCount_bound

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2ScheduledSizes.scheduledSizes_sum' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2ScheduledSizes.scheduledSizes_sum

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2ScheduledSizes.scheduledSizes_pointwise' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2ScheduledSizes.scheduledSizes_pointwise

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2ScheduledSizes.scheduledSize_sum' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Poseidon2ScheduledSizes.scheduledSize_sum

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2ScheduledSizes.normalize_addConstant_length' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Poseidon2ScheduledSizes.normalize_addConstant_length

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2ScheduledSizes.initialState_normalize_length' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2ScheduledSizes.initialState_normalize_length

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2ScheduledSizes.terminalState_succ_normalize_length' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2ScheduledSizes.terminalState_succ_normalize_length

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2ScheduledSizes.partialState_not_mentions_zero' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2ScheduledSizes.partialState_not_mentions_zero

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2ScheduledSizes.initialState_not_mentions_zero' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2ScheduledSizes.initialState_not_mentions_zero

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2ScheduledSizes.terminalState_succ_not_mentions_zero' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2ScheduledSizes.terminalState_succ_not_mentions_zero

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2ScheduledSizes.applyMatrix_singletons_not_zero' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2ScheduledSizes.applyMatrix_singletons_not_zero

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2ScheduledSizes.sboxOutput_ne_zero' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2ScheduledSizes.sboxOutput_ne_zero

/-! Structural no-cancellation in the full-round states. -/

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2ScheduledSizes.initialState_fieldNormalize_length' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2ScheduledSizes.initialState_fieldNormalize_length

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2ScheduledSizes.terminalState_succ_fieldNormalize_length' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2ScheduledSizes.terminalState_succ_fieldNormalize_length

end NightstreamTests.Axioms.CanonicalPoseidon2ScheduledSizes
