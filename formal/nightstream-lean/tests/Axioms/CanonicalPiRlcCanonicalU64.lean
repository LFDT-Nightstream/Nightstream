import Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalU64
import tests.Axioms.Support

/-!
Fail-closed axiom guards for the Lean-owned PiRLC digest-lane decomposition.
-/

namespace NightstreamTests.Axioms.CanonicalPiRlcCanonicalU64

open NightstreamTests.Axioms
open Nightstream.Implementation.R1CS.Canonical

/-- info: 'Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalU64.rows_length' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms PiRlcCanonicalU64.rows_length

/-- info: 'Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalU64.allocation_nodup' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms PiRlcCanonicalU64.allocation_nodup

/-- info: 'Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalU64.lane_allocation_mem' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms PiRlcCanonicalU64.lane_allocation_mem

/-- info: 'Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalU64.lane_refines' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms PiRlcCanonicalU64.lane_refines

/-- info: 'Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalU64.lane_bits_eq_digest' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms PiRlcCanonicalU64.lane_bits_eq_digest

end NightstreamTests.Axioms.CanonicalPiRlcCanonicalU64
