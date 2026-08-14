import Nightstream.Implementation.Nebula.FPrime.Manifest.TerminalStateAuthority
import tests.Axioms.Support

open Nightstream.Implementation.Nebula

/-- info: 'Nightstream.Implementation.Nebula.TerminalManifestNifsCall.Call.priorStateCarryPlaced' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms TerminalManifestNifsCall.Call.priorStateCarryPlaced

/-- info: 'Nightstream.Implementation.Nebula.TerminalManifestStateAuthority.incomingAuthority_digest_eq_columns' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms TerminalManifestStateAuthority.incomingAuthority_digest_eq_columns

/-- info: 'Nightstream.Implementation.Nebula.TerminalManifestStateAuthority.boundaryFromPrevious' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms TerminalManifestStateAuthority.boundaryFromPrevious
