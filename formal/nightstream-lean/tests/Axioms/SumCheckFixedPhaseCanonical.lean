import Nightstream.SuperNeo.SumCheck.FixedPhase.Canonical
import tests.Axioms.Support

/-! Fail-closed dependency gate for the fixed-to-raw canonical bridge. -/

open Nightstream.SuperNeo.SumCheck.Finite.FixedPhase.Canonical

/-- info: 'Nightstream.SuperNeo.SumCheck.Finite.FixedPhase.Canonical.accepted_toFinite' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms accepted_toFinite
