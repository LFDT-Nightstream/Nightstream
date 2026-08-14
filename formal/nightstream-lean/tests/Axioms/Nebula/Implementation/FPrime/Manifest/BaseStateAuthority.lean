import Nightstream.Implementation.Nebula.FPrime.Manifest.BaseStateAuthority
import tests.Axioms.Support

/-! Axiom audit for row-derived Nebula V2 base state authority. -/

open Nightstream.Implementation.Nebula

/-- info: 'Nightstream.Implementation.Nebula.BaseManifestStateAuthority.Call.initialAccepted' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms BaseManifestStateAuthority.Call.initialAccepted

/-- info: 'Nightstream.Implementation.Nebula.BaseManifestStateAuthority.Call.outgoingAccepted' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms BaseManifestStateAuthority.Call.outgoingAccepted

#print axioms Nightstream.Implementation.Nebula.BaseManifestStateAuthority.Call.initialExact
#print axioms Nightstream.Implementation.Nebula.BaseManifestStateAuthority.Call.opensExactInitialCarry
#print axioms Nightstream.Implementation.Nebula.BaseManifestStateAuthority.Call.outgoingAuthority_digest_eq_columns
